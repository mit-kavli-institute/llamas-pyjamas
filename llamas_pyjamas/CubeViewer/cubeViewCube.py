"""
SpectralScene over combined datacubes — the cube view in CubeViewer.

Wraps one or more :class:`llamas_pyjamas.Combine.cube.CoaddCube` (per channel, on a SHARED spatial
grid) as a :class:`~llamas_pyjamas.CubeViewer.cubeViewScene.SpectralScene`, so the existing DS9
crosshair picking and spectrum panel work on a stacked cube unchanged: collapse a wavelength window
into a white-light image, click a spaxel, plot its full blue+green+red spectrum. The spatial element
is a spaxel ``(iy, ix)`` (shared across channels) and the WCS is a real RA---TAN/DEC--TAN.

Built for the "Combine" menu (stack a field's dithers -> cubes -> inspect) and for opening a saved
cube FITS (single channel).

Classes
-------
CoaddCubeScene   SpectralScene over per-channel combined cubes sharing a spatial grid
"""

import logging
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from llamas_pyjamas.Combine.coadd import COVERAGE_FRAC_MIN, low_coverage_mask
from llamas_pyjamas.CubeViewer.cubeViewScene import CHANNEL_ORDER, Spectrum, SpectralScene

logger = logging.getLogger(__name__)


class CoaddCubeScene(SpectralScene):
    """A scene over per-channel combined cubes (a spaxel is the same sky position in every channel).

    Accepts a single :class:`CoaddCube` or a ``{channel: CoaddCube}`` mapping; the cubes must share
    the same spatial grid (as produced by :func:`Combine.cube.combine_field_cubes`)."""

    def __init__(self, cubes, super_rss=None) -> None:
        if not isinstance(cubes, dict):
            cubes = {str(cubes.meta.get('CHANNEL', 'green')): cubes}
        #: the SuperRSS the cubes were built from, if available -> enables optimal extraction
        self.super_rss = super_rss
        self._cubes: Dict[str, object] = {c: cubes[c] for c in CHANNEL_ORDER if c in cubes}
        if not self._cubes:
            raise ValueError('CoaddCubeScene needs at least one cube')
        self.channels = tuple(self._cubes)
        self.cube = self._cubes[self.channels[0]]         # reference (shared grid/wcs/shape)
        self.object = str(self.cube.meta.get('FIELD', ''))
        self.has_flam = any('erg' in c.bunit for c in self._cubes.values())
        self.hex_tiles = None                             # no hex toggle for a cube
        self._pixscale = float(self.cube.meta.get('PIXSCALE', 0.5))
        self.pitch = 0.75                                 # arcsec -> radius UI reads in fibre-spacings
        self._ny, self._nx = self.cube.data.shape[1:]
        self._coverage = np.max([c.coverage for c in self._cubes.values()], axis=0)
        self._nexp = np.max([getattr(c, 'nexp', c.coverage) for c in self._cubes.values()], axis=0)
        #: white-light spaxels shallower than this fraction of peak NEXP are excluded from the
        #: collapsed image (biased at partial-coverage boundaries); the cube itself is untouched.
        self.coverage_frac = COVERAGE_FRAC_MIN
        self.keys: List[Tuple[int, int]] = list(zip(*np.nonzero(self._coverage > 0)))

    def _step(self) -> float:
        return self._pixscale

    # ---- SpectralScene ----
    def wavelength_range(self, channel: Optional[str] = None) -> Tuple[float, float]:
        chans = [channel] if channel else self.channels
        lo = min(float(np.nanmin(self._cubes[c].wave)) for c in chans)
        hi = max(float(np.nanmax(self._cubes[c].wave)) for c in chans)
        return lo, hi

    def collapse(self, wave_min: float, wave_max: float,
                 channels: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, object, Dict]:
        imgs, contrib = [], {}
        for c in (channels or self.channels):
            if c not in self._cubes:
                continue
            cube = self._cubes[c]
            sl = (cube.wave >= wave_min) & (cube.wave <= wave_max)
            if not sl.any():
                continue
            with np.errstate(invalid='ignore'):
                im = np.nanmean(np.where(np.isfinite(cube.data[sl]), cube.data[sl], np.nan), axis=0)
            imgs.append(im)
            contrib[c] = int(np.isfinite(im).sum())
        if not imgs:
            img = np.full((self._ny, self._nx), np.nan)
        else:
            img = np.nanmean(np.stack(imgs, axis=0), axis=0)
            # default-exclude biased partial-coverage spaxels (kept in the cube; only masked here)
            excl = low_coverage_mask(self._nexp, self.coverage_frac)
            n_excl = int((excl & np.isfinite(img)).sum())
            img = np.where(excl, np.nan, img)
            if n_excl:
                contrib['_low_coverage_excluded'] = n_excl
        meta = {'contributions': contrib, 'WAVEMIN': (float(wave_min), 'A'),
                'WAVEMAX': (float(wave_max), 'A'), 'WCSFRAME': ('sky', 'Combined cube white light'),
                'COVFRAC': (float(self.coverage_frac), 'min NEXP frac shown in white light')}
        return img, self.cube.wcs.celestial, meta

    def _spaxel(self, x_pix: float, y_pix: float) -> Optional[Tuple[int, int]]:
        ix, iy = int(round(x_pix)) - 1, int(round(y_pix)) - 1   # DS9 1-based -> array index
        if 0 <= iy < self._ny and 0 <= ix < self._nx and self._coverage[iy, ix] > 0:
            return (iy, ix)
        return None

    def element_at(self, x_pix: float, y_pix: float) -> Optional[Hashable]:
        return self._spaxel(x_pix, y_pix)

    def _spectrum(self, channel: str, iy: int, ix: int) -> Optional[Spectrum]:
        cube = self._cubes[channel]
        if cube.coverage[iy, ix] <= 0:
            return None
        val = np.asarray(cube.data[:, iy, ix], dtype=float)
        calibrated = 'erg' in cube.bunit                  # cube built from FLAM -> no counts plane
        return Spectrum(wave=cube.wave, flux=val, channel=channel, label=f'spaxel ({ix},{iy})',
                        mask=~np.isfinite(val), flam=(val if calibrated else None),
                        has_counts=not calibrated)

    def spectra_at(self, x_pix: float, y_pix: float) -> List[Spectrum]:
        e = self._spaxel(x_pix, y_pix)
        return [] if e is None else self.spectra_of([e])

    def spectra_of(self, elements: Sequence[Hashable]) -> List[Spectrum]:
        out = []
        for (iy, ix) in elements:
            for c in self.channels:
                sp = self._spectrum(c, iy, ix)
                if sp is not None:
                    out.append(sp)
        return out

    def elements_within(self, x_pix: float, y_pix: float,
                        radius_pix: float) -> List[Hashable]:
        if radius_pix <= 0:
            e = self._spaxel(x_pix, y_pix)
            return [] if e is None else [e]
        cy, cx = y_pix - 1, x_pix - 1
        r = int(np.ceil(radius_pix))
        out = []
        for iy in range(max(0, int(cy) - r), min(self._ny, int(cy) + r + 1)):
            for ix in range(max(0, int(cx) - r), min(self._nx, int(cx) + r + 1)):
                if (iy - cy) ** 2 + (ix - cx) ** 2 <= radius_pix ** 2 and self._coverage[iy, ix] > 0:
                    out.append((iy, ix))
        out.sort(key=lambda e: (e[0] - cy) ** 2 + (e[1] - cx) ** 2)
        return out

    def marker_region(self, x_pix: float, y_pix: float) -> str:
        e = self._spaxel(x_pix, y_pix)
        return '' if e is None else self.region_for([e])

    def region_for(self, elements: Sequence[Hashable]) -> str:
        if not elements:
            return ''
        lines = ['image'] + [f'box({ix + 1},{iy + 1},1,1,0) # color=cyan width=2'
                             for (iy, ix) in elements]
        return '\n'.join(lines) if len(lines) > 1 else ''

    def anchor_to_gaia(self, ra, dec, *, radius_arcsec=3.0, source_arcsec=5.0):
        """Compute the absolute-flux scale from the in-field Gaia source near the extraction: fit +
        extract the point source, then compare to its Gaia XP SED. Returns the fluxanchor result
        dict (scale, scatter, source_id, ...) or None. Needs gaiaxpy + network."""
        from llamas_pyjamas.Combine.fluxanchor import flux_anchor
        spec, fit = self._optimal_raw(ra, dec, radius_arcsec=radius_arcsec)
        return flux_anchor(spec, fit.ra, fit.dec, radius_arcsec=source_arcsec)

    def apply_flux_scale(self, scale):
        """Apply an absolute-flux scale to every channel cube in place (data*=scale, var*=scale^2)."""
        for cube in self._cubes.values():
            cube.data = cube.data * scale
            cube.var = cube.var * scale ** 2
            cube.meta['FLUXANCH'] = float(scale)

    def narrowband(self, line_wave, half_width=8.0, **kwargs):
        """Continuum-subtracted narrowband image at ``line_wave`` from whichever channel covers it."""
        from llamas_pyjamas.Combine.cube import narrowband_image
        for c in self.channels:
            cube = self._cubes[c]
            if cube.wave[0] <= line_wave <= cube.wave[-1]:
                return narrowband_image(cube, line_wave, half_width=half_width, **kwargs)
        raise ValueError(f'{line_wave} A not covered by any channel ({", ".join(self.channels)})')

    def _optimal_raw(self, ra, dec, *, radius_arcsec=3.0, **kwargs):
        """Raw optimal extraction: ``({channel: (wave, flux, var)}, ProfileFit)``. Fibre-space from
        the super-RSS when available (best), else cube-space from the spaxels."""
        if self.super_rss is not None:
            from llamas_pyjamas.Combine.spectrum import optimal_spectrum
            return optimal_spectrum(self.super_rss, ra, dec, radius_arcsec=radius_arcsec, **kwargs)
        return self._optimal_from_cubes(ra, dec, radius_arcsec)

    def optimal_spectrum(self, ra, dec, *, radius_arcsec=3.0, **kwargs):
        """PSF/ivar-weighted point-source spectrum at (ra,dec): a 2-D Gaussian is fit to the source
        and used as the extraction profile. Uses fibre-space extraction from the super-RSS when
        available (best), else cube-space (from the spaxels). Returns ``(spectra, ProfileFit)``."""
        spec, fit = self._optimal_raw(ra, dec, radius_arcsec=radius_arcsec, **kwargs)
        tag = 'fit' if fit.fitted else 'assumed'
        src = 'fibre' if self.super_rss is not None else 'cube'
        dar = f' DAR {fit.dar_shift:.1f}"' if fit.dar_shift > 0.3 else ''
        label = f'optimal ({fit.ra:.4f},{fit.dec:+.4f}) FWHM {fit.fwhm:.1f}"{dar} [{tag}/{src}]'
        cal = 'erg' in self.cube.bunit
        spectra = [Spectrum(wave=w, flux=f, channel=c, label=label, mask=~np.isfinite(f),
                            flam=(f if cal else None), has_counts=not cal)
                   for c, (w, f, v) in spec.items()]
        return spectra, fit

    def _optimal_from_cubes(self, ra, dec, radius_arcsec):
        """Cube-space optimal extraction (no super-RSS): fit a 2-D Gaussian to the white-light image
        and Horne-combine the cube spaxels within the aperture."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        from llamas_pyjamas.Combine.spectrum import fit_gaussian_image, ProfileFit
        lo, hi = self.wavelength_range()
        img, wcs2, _ = self.collapse(lo, hi)
        px, py = wcs2.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        cx, cy = float(px), float(py)
        rpix = radius_arcsec / self._pixscale
        g = fit_gaussian_image(img, cx, cy, rpix)
        if g is not None:
            cx, cy = float(g.x_mean.value), float(g.y_mean.value)
            sky = wcs2.pixel_to_world(cx, cy)
            fit = ProfileFit(float(sky.ra.deg), float(sky.dec.deg),
                             abs(float(g.x_stddev.value)) * self._pixscale,
                             abs(float(g.y_stddev.value)) * self._pixscale,
                             float(np.rad2deg(g.theta.value)), True)

            def prof(ix, iy):
                return float(g(ix, iy) / g.amplitude.value)
        else:
            sig = 1.2 / self._pixscale
            fit = ProfileFit(ra, dec, 1.2, 1.2, 0.0, False)

            def prof(ix, iy):
                return float(np.exp(-0.5 * ((ix - cx) ** 2 + (iy - cy) ** 2) / sig ** 2))

        # analytic PSF normalisation (fraction per spaxel) so F_hat is the TOTAL flux, matching the
        # fibre-space path; spaxel Gaussian sigmas (pixels) -> 1/(2*pi*sx*sy). Cube SB -> flux per
        # spaxel via the pixel solid angle (pixscale^2) when the cube is in surface brightness.
        sgx = abs(float(g.x_stddev.value)) if g is not None else 1.2 / self._pixscale
        sgy = abs(float(g.y_stddev.value)) if g is not None else sgx
        pnorm = 1.0 / (2.0 * np.pi * sgx * sgy)
        x0, x1 = max(0, int(cx - rpix)), min(self._nx - 1, int(cx + rpix))
        y0, y1 = max(0, int(cy - rpix)), min(self._ny - 1, int(cy + rpix))
        spec = {}
        for c in self.channels:
            cube = self._cubes[c]
            to_flux = self._pixscale ** 2 if '/arcsec2' in cube.bunit else 1.0   # SB -> flux/spaxel
            num = np.zeros(cube.wave.size)
            den = np.zeros(cube.wave.size)
            for iy in range(y0, y1 + 1):
                for ix in range(x0, x1 + 1):
                    if (ix - cx) ** 2 + (iy - cy) ** 2 > rpix ** 2 or cube.coverage[iy, ix] <= 0:
                        continue
                    P = prof(ix, iy) * pnorm
                    S = np.asarray(cube.data[:, iy, ix], float) * to_flux
                    V = np.asarray(cube.var[:, iy, ix], float) * to_flux ** 2
                    wgt = np.where(np.isfinite(V) & (V > 0) & np.isfinite(S), 1.0 / V, 0.0)
                    num += P * np.where(np.isfinite(S), S, 0.0) * wgt
                    den += P * P * wgt
            with np.errstate(invalid='ignore', divide='ignore'):
                spec[c] = (cube.wave, np.where(den > 0, num / den, np.nan),
                           np.where(den > 0, 1.0 / den, np.nan))
        return spec, fit

    def profile_ellipse_region(self, fit, radius_arcsec=3.0, tag='cubeview-ext'):
        """DS9 region for an extraction: the fitted centroid (x), the fitted PSF as 1-sigma and
        2-sigma ellipses (shape + tilt), and the extraction APERTURE as a dashed circle at
        ``radius_arcsec`` -- so both the fitted profile and the region actually summed are shown.
        The PSF ellipses can look small against the cube white-light because the cube is smoothed by
        the co-add kernel; the fit is the true (sharper) PSF."""
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        px, py = self.cube.wcs.celestial.world_to_pixel(SkyCoord(fit.ra * u.deg, fit.dec * u.deg))
        x, y = float(px) + 1, float(py) + 1               # DS9 1-based
        s = self._pixscale
        ang = -fit.theta_deg                              # tangent (E-left) -> image angle (mod 180)
        ax, ay = fit.sigma_x / s, fit.sigma_y / s
        lines = ['image', f'point({x:.2f},{y:.2f}) # point=x 12 color=cyan width=2 tag={{{tag}}}']
        for k in (1, 2):
            lines.append(f'ellipse({x:.2f},{y:.2f},{k * ax:.2f},{k * ay:.2f},{ang:.1f}) '
                         f'# color=cyan width=2 tag={{{tag}}}')
        lines.append(f'circle({x:.2f},{y:.2f},{radius_arcsec / s:.2f}) '
                     f'# color=green dash=1 width=1 tag={{{tag}}}')   # extraction aperture
        # DAR: the source position at blue/green/red, as coloured 1-sigma ellipses on the track
        if getattr(fit, 'dar', None) is not None:
            lo, hi = self.wavelength_range()
            for frac, col in ((0.12, 'blue'), (0.5, 'green'), (0.88, 'red')):
                lam = lo + frac * (hi - lo)
                cra, cdec = fit.dar.center_sky(lam)
                cpx, cpy = self.cube.wcs.celestial.world_to_pixel(SkyCoord(cra * u.deg, cdec * u.deg))
                lines.append(f'ellipse({cpx + 1:.2f},{cpy + 1:.2f},{ax:.2f},{ay:.2f},{ang:.1f}) '
                             f'# color={col} width=2 tag={{{tag}}}')
        return '\n'.join(lines)

    @staticmethod
    def _read_cube(path):
        """Read one cube FITS written by :meth:`CoaddCube.write` -> CoaddCube."""
        from astropy.io import fits
        from astropy.wcs import WCS
        from llamas_pyjamas.Combine.cube import CoaddCube
        with fits.open(path) as h:
            data = np.asarray(h[0].data, dtype=float)
            hdr = h[0].header
            var = np.asarray(h['VAR'].data, float) if 'VAR' in h else np.full_like(data, np.nan)
            cov = (np.asarray(h['COVERAGE'].data) if 'COVERAGE' in h
                   else np.ones(data.shape[1:], int))
            nexp = np.asarray(h['NEXP'].data) if 'NEXP' in h else np.zeros(data.shape[1:], int)
            wave = (np.asarray(h['WAVELENGTH'].data['WAVELENGTH'], float) if 'WAVELENGTH' in h
                    else np.arange(data.shape[0], dtype=float))
            n = int(hdr.get('NRSSFILE', 0))
            paths = [hdr[f'RSSFIL{i}'] for i in range(1, n + 1) if f'RSSFIL{i}' in hdr]
            meta = {'FIELD': hdr.get('FIELD', ''), 'CHANNEL': hdr.get('CHANNEL', 'green'),
                    'PIXSCALE': hdr.get('PIXSCALE', 0.5), 'exposure_paths': paths}
            return CoaddCube(data=data, var=var, wave=wave, coverage=cov, nexp=nexp,
                             wcs=WCS(hdr), bunit=hdr.get('BUNIT', ''), meta=meta)

    @classmethod
    def from_fits(cls, path: str) -> 'CoaddCubeScene':
        """Open a combined cube. Cubes are written one file per channel (``*_cube_{blue,green,red}
        .fits``); this loads every channel sibling it can find so a spaxel shows the full spectrum.
        Optimal extraction on an opened cube uses the cube spaxels (no super-RSS)."""
        import os
        import re
        d, name = os.path.split(path)
        cubes = {}
        m = re.search(r'_cube_(blue|green|red)', name)
        if m:
            for c in CHANNEL_ORDER:
                sib = os.path.join(d, re.sub(r'_cube_(blue|green|red)', f'_cube_{c}', name))
                if os.path.exists(sib):
                    cubes[c] = cls._read_cube(sib)
        if not cubes:                                     # not the standard naming: load it alone
            cube = cls._read_cube(path)
            cubes[str(cube.meta.get('CHANNEL', 'green'))] = cube
        return cls(cubes, super_rss=cls._rebuild_super_rss(next(iter(cubes.values()))))

    @staticmethod
    def _rebuild_super_rss(ref_cube):
        """Rebuild the super-RSS from a cube's recorded provenance (contributing RSS paths), so an
        opened cube gets full fibre-space extraction + DAR. Returns None if the paths are missing
        (extraction then falls back to cube-space)."""
        import os
        paths = ref_cube.meta.get('exposure_paths') or []
        if not paths or not all(os.path.exists(p) for p in paths):
            if paths:
                logger.info('super-RSS not rebuilt: some source RSS files are missing')
            return None
        try:
            from llamas_pyjamas.Combine.superRSS import build_super_rss
            from llamas_pyjamas.Combine.transparency import transparency_scales
            sr = build_super_rss(paths)
            try:                                          # match the combine's photometric scaling
                sr.apply_scales(transparency_scales(sr))
            except Exception as exc:                      # noqa: BLE001
                logger.debug('transparency scaling skipped on rebuild: %s', exc)
            return sr
        except Exception as exc:                          # noqa: BLE001
            logger.warning('super-RSS rebuild failed (%s); cube-space extraction only', exc)
            return None
