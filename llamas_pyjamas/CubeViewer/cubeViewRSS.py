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
from astropy.io import fits
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from llamas_pyjamas.CubeViewer.cubeViewScene import (
    CHANNEL_ORDER,
    Spectrum,
    SpectralScene,
    linear_wcs,
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
            self.flux = np.asarray(hdul['FLUX'].data, dtype=float)
            self.wave = np.asarray(hdul['WAVE'].data, dtype=float)
            self.mask = (np.asarray(hdul['MASK'].data) if 'MASK' in hdul else None)

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
        logger.info('RSSScene: channels=%s, %d placed fibres',
                    ','.join(self.channels), len(self.keys))

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

    def collapse(self, wave_min: float, wave_max: float,
                 channels: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, WCS, Dict]:
        wanted = tuple(channels) if channels else self.channels

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

        meta['contributions'] = contributions
        meta['WAVEMIN'] = (float(wave_min), 'Collapse window start (Angstrom)')
        meta['WAVEMAX'] = (float(wave_max), 'Collapse window end (Angstrom)')
        return image, wcs, meta

    def _key_at(self, x_pix: float, y_pix: float) -> Optional[FibreKey]:
        """The fibre whose hexagon contains an image pixel, or None."""
        step = 1.0 / float(self.pix_per_unit) if self.hex_tiles else None
        if step is None:
            grid_x, _ = whitelight_grid()
            step = float(grid_x[0, 1] - grid_x[0, 0])
        # hex_header_keys pins CRPIX=1 (1-indexed), CRVAL=0, CDELT=step.
        world = ((x_pix - 1.0) * step, (y_pix - 1.0) * step)

        distance, row = self._tree.query(world, k=1)
        if not np.isfinite(distance) or distance > HEX_CIRCUMRADIUS:
            return None
        return self.keys[int(row)]

    def spectra_at(self, x_pix: float, y_pix: float) -> List[Spectrum]:
        key = self._key_at(x_pix, y_pix)
        if key is None:
            return []
        spectra = [self._channels[name].spectrum(key) for name in self.channels]
        return [s for s in spectra if s is not None]

    def marker_region(self, x_pix: float, y_pix: float) -> str:
        """A DS9 polygon tracing the selected fibre's hexagon, in image coordinates."""
        key = self._key_at(x_pix, y_pix)
        if key is None:
            return ''
        centre = self.positions[self.keys.index(key)]
        step = 1.0 / float(self.pix_per_unit)
        # Pointy-top hexagon: vertices every 60 deg starting at 90 deg.
        angles = np.deg2rad(90.0 + 60.0 * np.arange(6))
        vertices = []
        for angle in angles:
            wx = centre[0] + HEX_CIRCUMRADIUS * np.cos(angle)
            wy = centre[1] + HEX_CIRCUMRADIUS * np.sin(angle)
            vertices.append(f'{wx / step + 1.0:.3f},{wy / step + 1.0:.3f}')
        return f'image; polygon({",".join(vertices)}) # color=cyan width=2'
