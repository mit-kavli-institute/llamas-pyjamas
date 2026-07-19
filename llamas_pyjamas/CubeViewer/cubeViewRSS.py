"""
SpectralScene implementation for row-stacked spectra (RSS) — the fibre view.

One RSS file holds one channel (the channel is in the primary header's ``CHANNEL`` key, not in
any column), so covering the full spectrum means loading the blue, green and red files
together. :meth:`RSSScene.open` takes any one of them and finds its siblings by name.

Two facts about the RSS format shape this module:

* ``WAVE`` is a per-fibre 2-D extension matching ``FLUX``, not a spectral WCS. Every fibre has
  its own irregular wavelength array, so plotting a fibre needs no resampling at all — and a
  wavelength cut is a per-fibre mask, not a column slice.
* Rows are LIVE-indexed (dead fibres are absent), so a row index is **not** a fibre id. Fibre
  identity is ``(BENCHSIDE, FIBER_ID)`` from the ``FIBERMAP`` table, and only that pair may be
  handed to ``FiberMap_LUT``. Mixing the two is precisely the bug class behind the 3B/4A and
  white-light dead-fibre regressions, so this module never indexes geometry by row.

Images are rendered as hexagonal tiles (:func:`~llamas_pyjamas.Image.WhiteLightModule.hex_tile_image`).
That is not cosmetic: each pixel takes exactly one fibre's raw flux, so a DS9 click inverts to
exactly one fibre. The interpolated white-light grid blends neighbouring fibres into each pixel
and cannot answer "which fibre did I click?" — it is available via ``hex_tiles=False`` for
display, but picking is then approximate.

Classes
-------
RSSChannel   One channel's arrays plus its (benchside, fibre) index
RSSScene     SpectralScene over one or more channels

Functions
---------
channel_siblings  Map a channel RSS path to all available channel paths
"""

import logging
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from llamas_pyjamas.CubeViewer.cubeViewScene import (
    CHANNEL_ORDER,
    Spectrum,
    SpectralScene,
    linear_wcs,
)
from llamas_pyjamas.File.llamasRSS import skysub_extname
from llamas_pyjamas.Utils.wcsLlamas import (
    ARCSEC_PER_FIBRE,
    celestial_wcs,
    pointing_from_header,
)
from llamas_pyjamas.Image.WhiteLightModule import (
    HEX_PITCH,
    FiberMap_LUT,
    hex_header_keys,
    hex_tile_image,
    whitelight_grid,
)

logger = logging.getLogger(__name__)

#: Matches the channel token in e.g. ``..._extract_RSS_green_FF_SKYSUB.fits``.
CHANNEL_RE = re.compile(r'(_RSS_)(blue|green|red)(?=[_.])')

DEFAULT_PIX_PER_UNIT = 10

#: A point inside a pointy-top hexagon lies at most one circumradius from its centre.
HEX_CIRCUMRADIUS = HEX_PITCH / np.sqrt(3.0)

#: Slack for distance comparisons against the lattice.
#:
#: An aperture of "N fibre spacings" lands exactly on a ring of neighbours, and the fibre map's
#: coordinates are not exact: HEX_PITCH derives as 0.9999996 rather than 1.0, and a ring that
#: should sit at 1.0 actually spreads over 0.999999-1.000002. Without slack a radius of one
#: pitch admits whichever neighbours happen to round below it -- 5 of 7 on real data, which
#: looks plausible enough to ship unnoticed.
#:
#: 1e-3 is chosen from the geometry rather than by taste: it is ~500x the observed coordinate
#: jitter (~2e-6), and ~700x smaller than the gap to the next ring (1.0 -> 1.732, i.e. 0.73
#: pitch), so it cannot pull in a ring that does not belong. hex_tile_image needs only 1e-6
#: because it compares against exact pixel-centre offsets; nearest-neighbour distances
#: accumulate error from both coordinates and need more room.
LATTICE_TOL = 1e-3 * HEX_PITCH

FibreKey = Tuple[str, int]


def channel_siblings(path: str) -> Dict[str, str]:
    """Find every channel RSS belonging to the same exposure and processing stage.

    Substitutes the channel token in the filename and keeps the files that exist, so
    ``..._RSS_green_FF.fits`` finds ``..._RSS_blue_FF.fits`` and ``..._RSS_red_FF.fits`` but
    never the un-flat-fielded or ``_SKYSUB`` variants — the suffix is part of the stem and is
    preserved.

    Returns
    -------
    dict
        channel -> path, always including `path` itself. Falls back to ``{channel: path}``
        when the name does not follow the convention.
    """
    directory, name = os.path.split(os.path.abspath(path))
    match = CHANNEL_RE.search(name)
    if match is None:
        logger.warning("Filename %r does not match the _RSS_{channel} convention; "
                       "loading it alone", name)
        return {}

    found: Dict[str, str] = {}
    for channel in CHANNEL_ORDER:
        sibling = os.path.join(directory, CHANNEL_RE.sub(rf'\g<1>{channel}', name))
        if os.path.exists(sibling):
            found[channel] = sibling
    return found


class RSSChannel:
    """One channel of an RSS exposure, with its fibre index and geometry.

    Attributes
    ----------
    channel : str
        'blue', 'green' or 'red'.
    flux, wave : ndarray
        Shape (nfibre, nwave). `wave` is per-fibre, in Angstrom.
    mask : ndarray or None
        Shape (nfibre, nwave), non-zero where bad.
    keys : list of (str, int)
        Row -> (benchside, fibre id).
    index : dict
        (benchside, fibre id) -> row.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        with fits.open(path, memmap=False) as hdul:
            header = hdul[0].header
            self.channel = str(header.get('CHANNEL', '')).strip().lower()
            if self.channel not in CHANNEL_ORDER:
                raise ValueError(f"{path}: primary CHANNEL={self.channel!r} is not one of "
                                 f"{CHANNEL_ORDER}")
            # Kept for the celestial WCS (header pointing); the hdul is closed on exit.
            self.primary_header = header.copy()
            self.flux = np.asarray(hdul[skysub_extname(hdul)].data, dtype=float)
            self.wave = np.asarray(hdul['WAVE'].data, dtype=float)
            self.mask = (np.asarray(hdul['MASK'].data) if 'MASK' in hdul else None)
            # Flux-calibrated plane, present only after apply_fluxcal has run on this RSS.
            self.flam = (np.asarray(hdul['FLAM'].data, dtype=float) if 'FLAM' in hdul else None)

            fibermap = hdul['FIBERMAP'].data
            benchsides = [str(b).strip() for b in fibermap['BENCHSIDE']]
            fibre_ids = [int(f) for f in fibermap['FIBER_ID']]

        if self.flux.shape != self.wave.shape:
            raise ValueError(f"{path}: FLUX {self.flux.shape} and WAVE {self.wave.shape} "
                             f"disagree")
        if len(benchsides) != self.flux.shape[0]:
            raise ValueError(f"{path}: FIBERMAP has {len(benchsides)} rows but FLUX has "
                             f"{self.flux.shape[0]}")

        self.keys: List[FibreKey] = list(zip(benchsides, fibre_ids))
        self.index: Dict[FibreKey, int] = {key: row for row, key in enumerate(self.keys)}
        logger.info("Loaded %s RSS: %d fibres x %d samples from %s",
                    self.channel, self.flux.shape[0], self.flux.shape[1],
                    os.path.basename(path))

    def wavelength_range(self) -> Tuple[float, float]:
        """Min and max finite wavelength across all fibres, in Angstrom."""
        finite = self.wave[np.isfinite(self.wave)]
        if finite.size == 0:
            return (np.nan, np.nan)
        return float(finite.min()), float(finite.max())

    def collapse_fibres(self, wave_min: float, wave_max: float) -> np.ndarray:
        """Sum each fibre's flux over a wavelength window.

        The window is applied per fibre against that fibre's own wavelength array, which is
        what the 2-D WAVE extension requires — a shared column slice would be wrong, since
        fibres are not on a common grid.

        Returns
        -------
        ndarray
            One value per row; NaN where a fibre has no samples in the window.
        """
        inside = (self.wave >= wave_min) & (self.wave <= wave_max)
        if self.mask is not None:
            inside &= (self.mask == 0)
        selected = np.where(inside, self.flux, np.nan)
        with np.errstate(invalid='ignore'):
            totals = np.nansum(selected, axis=1)
        # nansum gives 0.0 for an all-NaN row; that is a fibre with no data in the window,
        # which must stay distinguishable from a genuine zero flux.
        totals[~np.any(inside, axis=1)] = np.nan
        return totals

    def spectrum(self, key: FibreKey) -> Optional[Spectrum]:
        """This channel's spectrum for a fibre, or None if the fibre is absent here."""
        row = self.index.get(key)
        if row is None:
            return None
        benchside, fibre_id = key
        return Spectrum(
            wave=self.wave[row],
            flux=self.flux[row],
            channel=self.channel,
            label=f'{benchside} fibre {fibre_id}',
            mask=(self.mask[row] != 0) if self.mask is not None else None,
            flam=(self.flam[row] if self.flam is not None else None),
        )


class RSSScene(SpectralScene):
    """A :class:`~llamas_pyjamas.CubeViewer.cubeViewScene.SpectralScene` over RSS fibres.

    Parameters
    ----------
    paths : dict
        channel -> RSS path. Missing channels are simply absent.
    pix_per_unit : int
        Output pixels per fibre-map unit for the hex render.
    hex_tiles : bool
        Render exact hexagonal tiles (default). False uses the interpolated white-light grid,
        which looks smoother but makes picking approximate.
    """

    def __init__(self, paths: Dict[str, str], pix_per_unit: int = DEFAULT_PIX_PER_UNIT,
                 hex_tiles: bool = True) -> None:
        if not paths:
            raise ValueError('RSSScene needs at least one channel path')

        self._channels: Dict[str, RSSChannel] = {}
        for channel in CHANNEL_ORDER:
            if channel in paths:
                self._channels[channel] = RSSChannel(paths[channel])
        self.channels = tuple(self._channels)
        self.pix_per_unit = int(pix_per_unit)
        self.hex_tiles = bool(hex_tiles)
        #: Fibre lattice spacing, so callers can express apertures in fibres, not pixels.
        self.pitch = HEX_PITCH

        # Fibre geometry, from the union of channels. Position comes from (benchside, fibre
        # id) via the LUT -- never from the row index.
        keys: List[FibreKey] = []
        seen = set()
        for channel in self._channels.values():
            for key in channel.keys:
                if key not in seen:
                    seen.add(key)
                    keys.append(key)

        positions, placed = [], []
        for key in keys:
            x, y = FiberMap_LUT(key[0], key[1])
            if x < 0 and y < 0:            # LUT miss sentinel
                logger.warning('No fibre-map position for %s fibre %d; excluded', *key)
                continue
            positions.append((x, y))
            placed.append(key)

        self.keys: List[FibreKey] = placed
        self.positions = np.asarray(positions, dtype=float)
        if len(self.positions) == 0:
            raise ValueError('No fibres could be placed on the fibre map')
        self._tree = cKDTree(self.positions)
        #: True if any channel carries a flux-calibrated (FLAM) plane.
        self.has_flam = any(c.flam is not None for c in self._channels.values())

        # Header pointing for the celestial WCS (same across channels; take the first).
        first = next(iter(self._channels.values()))
        _hdr = getattr(first, 'primary_header', None)
        self.ra, self.dec, self.pa = pointing_from_header(_hdr)
        #: Object name from the header, for the DS9 frame / display.
        self.object = str(_hdr.get('OBJECT', '')) if _hdr is not None else ''
        #: A refined fibre-map WCS (from a prior registration, or set live by the interactive
        #: tool). When present it supersedes the rough header pointing for the displayed image.
        self.refined_wcs = self._load_refined_wcs(first.path)
        logger.info('RSSScene pointing: RA=%s DEC=%s PA=%s',
                    self.ra, self.dec, self.pa)
        logger.info('RSSScene: channels=%s, %d placed fibres, flux-calibrated=%s',
                    ','.join(self.channels), len(self.keys), self.has_flam)

    @staticmethod
    def _load_refined_wcs(path: str):
        """Reconstruct the refined fibre-map WCS from a registered file's FIBERWCS table, or None.

        The stored per-fibre RA/DEC were produced by the refined WCS, so re-fitting them against
        the fibre X/Y recovers it exactly -- letting a previously-registered file display with its
        star-tied grid instead of the rough header pointing. Best-effort: any problem -> None (fall
        back to the header WCS), so an unregistered or odd file still opens."""
        try:
            with fits.open(path, memmap=False) as hdul:
                if 'FIBERWCS' not in hdul:
                    return None
                hdr = hdul['FIBERWCS'].header
                if not bool(hdr.get('WCSREFIN', False)):
                    return None
                tab = hdul['FIBERWCS'].data
                x = np.asarray(tab['X_FIBERMAP'], float)
                y = np.asarray(tab['Y_FIBERMAP'], float)
                ra = np.asarray(tab['RA'], float)
                dec = np.asarray(tab['DEC'], float)
            good = np.isfinite(x) & np.isfinite(y) & np.isfinite(ra) & np.isfinite(dec) & (ra >= 0)
            if good.sum() < 3:
                return None
            from astropy.coordinates import SkyCoord
            from llamas_pyjamas.Utils.wcsLlamas import fit_wcs_from_stars
            pixels = list(zip(x[good], y[good]))
            sky = SkyCoord(ra[good] * u.deg, dec[good] * u.deg)
            return fit_wcs_from_stars(pixels, sky)
        except Exception as exc:                       # noqa: BLE001 - display fallback
            logger.debug('No refined WCS reconstructed from %s: %s', os.path.basename(path), exc)
            return None

    @classmethod
    def open(cls, path: str, **kwargs) -> 'RSSScene':
        """Open an RSS and every channel sibling alongside it."""
        paths = channel_siblings(path)
        if not paths:
            with fits.open(path) as hdul:
                channel = str(hdul[0].header.get('CHANNEL', '')).strip().lower()
            paths = {channel: path}
        return cls(paths, **kwargs)

    def wavelength_range(self, channel: Optional[str] = None) -> Tuple[float, float]:
        if channel is not None:
            return self._channels[channel].wavelength_range()
        ranges = [c.wavelength_range() for c in self._channels.values()]
        finite = [r for r in ranges if np.isfinite(r[0])]
        if not finite:
            return (np.nan, np.nan)
        return min(r[0] for r in finite), max(r[1] for r in finite)

    def _fibre_totals(self, wave_min: float, wave_max: float,
                      wanted: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int]]:
        """Per-fibre flux summed over the window, aligned to ``self.keys``/``self.positions``,
        plus the per-channel contributing-fibre counts. Shared by :meth:`collapse` (which tiles
        it into an image) and :meth:`fibre_flux` (which hands it to the centroider)."""
        totals = np.full(len(self.keys), np.nan, dtype=float)
        contributions: Dict[str, int] = {}
        row_of = {key: i for i, key in enumerate(self.keys)}
        for name in wanted:
            channel = self._channels.get(name)
            if channel is None:
                continue
            low, high = channel.wavelength_range()
            if not (np.isfinite(low) and low <= wave_max and high >= wave_min):
                logger.debug('%s does not overlap %.1f-%.1f A; skipped',
                             name, wave_min, wave_max)
                continue
            values = channel.collapse_fibres(wave_min, wave_max)
            used = 0
            for row, key in enumerate(channel.keys):
                target = row_of.get(key)
                if target is None or not np.isfinite(values[row]):
                    continue
                totals[target] = (values[row] if np.isnan(totals[target])
                                  else totals[target] + values[row])
                used += 1
            contributions[name] = used
        return totals, contributions

    def fibre_flux(self, wave_min: float, wave_max: float,
                   channels: Optional[Sequence[str]] = None) -> np.ndarray:
        """Per-fibre flux over a window, aligned to ``self.positions`` (never rendered to an
        image). This is the flux the interactive registration tool feeds to ``fibre_centroid`` so
        a clicked star is centroided on the real fibres, matching the automated path exactly."""
        wanted = tuple(channels) if channels else self.channels
        totals, _ = self._fibre_totals(wave_min, wave_max, wanted)
        return totals

    def collapse(self, wave_min: float, wave_max: float,
                 channels: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, WCS, Dict]:
        wanted = tuple(channels) if channels else self.channels

        totals, contributions = self._fibre_totals(wave_min, wave_max, wanted)
        if not contributions:
            raise ValueError(f'No channel covers {wave_min:.1f}-{wave_max:.1f} A '
                             f'(scene spans {self.wavelength_range()})')

        x, y = self.positions[:, 0], self.positions[:, 1]
        if self.hex_tiles:
            image, keys_dict = hex_tile_image(x, y, totals, pix_per_unit=self.pix_per_unit)
            meta = dict(keys_dict)
            step = 1.0 / float(self.pix_per_unit)
            wcs = linear_wcs(crpix=[1.0, 1.0], crval=[0.0, 0.0], cdelt=[step, step])
        else:
            from scipy.interpolate import LinearNDInterpolator
            grid_x, grid_y = whitelight_grid()
            good = np.isfinite(totals)
            interp = LinearNDInterpolator(self.positions[good], totals[good])
            image = interp(grid_x, grid_y)
            step = float(grid_x[0, 1] - grid_x[0, 0])
            meta = {'HEXTILE': (False, 'Interpolated white light')}
            wcs = linear_wcs(crpix=[1.0, 1.0], crval=[0.0, 0.0], cdelt=[step, step])

        # WCS priority: a refined (star-tied) solution if we have one, else the rough header
        # pointing, else the linear fibre-map map. Both render paths map pixel p -> fibre-map
        # (p-1)*step (CRPIX=1 at fibre-map 0), so a fibre-map WCS converts to the image with step.
        if self.refined_wcs is not None:
            from llamas_pyjamas.Utils.register import _image_wcs_from_fibremap
            wcs = _image_wcs_from_fibremap(self.refined_wcs, step)
            meta['WCSFRAME'] = ('sky', 'Refined star-tied WCS (registration)')
        elif self.ra is not None and self.dec is not None:
            cx = 0.5 * (x.min() + x.max())
            cy = 0.5 * (y.min() + y.max())
            crpix = (cx / step + 1.0, cy / step + 1.0)
            wcs = celestial_wcs(self.ra, self.dec, crpix,
                                arcsec_per_pixel=ARCSEC_PER_FIBRE * step, pa_deg=self.pa)
            meta['WCSFRAME'] = ('sky', 'Celestial WCS from header pointing (initial guess)')
        else:
            meta['WCSFRAME'] = ('fibremap', 'No header pointing; linear fibre-map WCS')

        meta['contributions'] = contributions
        meta['WAVEMIN'] = (float(wave_min), 'Collapse window start (Angstrom)')
        meta['WAVEMAX'] = (float(wave_max), 'Collapse window end (Angstrom)')
        return image, wcs, meta

    def element_at(self, x_pix: float, y_pix: float) -> Optional[FibreKey]:
        """The fibre whose hexagon contains an image pixel, or None.

        Exact in the hex render: each pixel belongs to exactly one fibre's Voronoi cell, and a
        point inside a pointy-top hexagon is at most one circumradius from its centre, so the
        nearest-neighbour query plus that bound is the hexagon test. Off-field pixels and dead
        fibres' empty tiles both return None.
        """
        world = self._to_world(x_pix, y_pix)
        distance, row = self._tree.query(world, k=1)
        if not np.isfinite(distance) or distance > HEX_CIRCUMRADIUS:
            return None
        return self.keys[int(row)]

    def spectra_at(self, x_pix: float, y_pix: float) -> List[Spectrum]:
        key = self.element_at(x_pix, y_pix)
        return [] if key is None else self.spectra_of([key])

    def spectra_of(self, elements: Sequence[FibreKey]) -> List[Spectrum]:
        spectra: List[Spectrum] = []
        for key in elements:
            for name in self.channels:
                spectrum = self._channels[name].spectrum(key)
                if spectrum is not None:
                    spectra.append(spectrum)
        return spectra

    def elements_within(self, x_pix: float, y_pix: float,
                        radius_pix: float) -> List[FibreKey]:
        """Fibres whose centres fall within `radius_pix` of an image pixel.

        A radius of zero degenerates to the single fibre under the pixel, so the aperture and
        single-pick paths are the same code.
        """
        if radius_pix <= 0:
            key = self.element_at(x_pix, y_pix)
            return [] if key is None else [key]

        world = self._to_world(x_pix, y_pix)
        # The lattice pitch derived from the LUT is 0.9999996, not 1.0 (finite precision in
        # the fibre map), while the true nearest-neighbour distances are exactly 1.0. An
        # aperture of "one fibre spacing" is therefore 0.9999996 and lands *just inside* the
        # neighbour ring, admitting only whichever neighbours happen to round below it -- 3
        # of 6 on real data. hex_tile_image hit the same trap and solved it the same way.
        radius_world = radius_pix * self._step() + LATTICE_TOL
        rows = self._tree.query_ball_point(world, radius_world)
        # Nearest first, so the aperture's reference grid is the central fibre's -- the
        # sensible choice when members must be interpolated onto one of them.
        rows.sort(key=lambda r: float(np.hypot(*(self.positions[r] - np.asarray(world)))))
        return [self.keys[r] for r in rows]

    def _step(self) -> float:
        """Fibre-map units per image pixel, for the active render."""
        if self.hex_tiles:
            return 1.0 / float(self.pix_per_unit)
        grid_x, _ = whitelight_grid()
        return float(grid_x[0, 1] - grid_x[0, 0])

    def _to_world(self, x_pix: float, y_pix: float) -> Tuple[float, float]:
        """Image pixel -> fibre-map coordinates. hex_header_keys pins CRPIX=1, CRVAL=0."""
        step = self._step()
        return ((x_pix - 1.0) * step, (y_pix - 1.0) * step)

    def _hexagon(self, centre: np.ndarray, step: float) -> str:
        """One pointy-top hexagon as a DS9 image-coordinate polygon."""
        angles = np.deg2rad(90.0 + 60.0 * np.arange(6))
        vertices = [
            f'{(centre[0] + HEX_CIRCUMRADIUS * np.cos(a)) / step + 1.0:.3f},'
            f'{(centre[1] + HEX_CIRCUMRADIUS * np.sin(a)) / step + 1.0:.3f}'
            for a in angles
        ]
        return f'polygon({",".join(vertices)})'

    def marker_region(self, x_pix: float, y_pix: float) -> str:
        """A DS9 polygon tracing the selected fibre's hexagon, in image coordinates."""
        key = self.element_at(x_pix, y_pix)
        return '' if key is None else self.region_for([key])

    def region_for(self, elements: Sequence[FibreKey]) -> str:
        """DS9 polygons tracing every fibre's hexagon.

        One region per line: DS9 accepts a multi-line region payload, so an aperture is drawn
        as the union of its members' hexagons rather than as an approximating circle. The
        fibres that actually went into the sum are then exactly what is outlined.
        """
        if not elements:
            return ''
        step = self._step()
        index = {key: row for row, key in enumerate(self.keys)}
        lines = ['image']
        for key in elements:
            row = index.get(key)
            if row is None:
                continue
            lines.append(f'{self._hexagon(self.positions[row], step)} # color=cyan width=2')
        return '\n'.join(lines) if len(lines) > 1 else ''
