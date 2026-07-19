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

from llamas_pyjamas.CubeViewer.cubeViewScene import CHANNEL_ORDER, Spectrum, SpectralScene

logger = logging.getLogger(__name__)


class CoaddCubeScene(SpectralScene):
    """A scene over per-channel combined cubes (a spaxel is the same sky position in every channel).

    Accepts a single :class:`CoaddCube` or a ``{channel: CoaddCube}`` mapping; the cubes must share
    the same spatial grid (as produced by :func:`Combine.cube.combine_field_cubes`)."""

    def __init__(self, cubes) -> None:
        if not isinstance(cubes, dict):
            cubes = {str(cubes.meta.get('CHANNEL', 'green')): cubes}
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
        meta = {'contributions': contrib, 'WAVEMIN': (float(wave_min), 'A'),
                'WAVEMAX': (float(wave_max), 'A'), 'WCSFRAME': ('sky', 'Combined cube white light')}
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

    @classmethod
    def from_fits(cls, path: str) -> 'CoaddCubeScene':
        """Open a single-channel cube FITS written by :meth:`CoaddCube.write`."""
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
            wcs = WCS(hdr)
            meta = {'FIELD': hdr.get('FIELD', ''), 'CHANNEL': hdr.get('CHANNEL', 'green'),
                    'PIXSCALE': hdr.get('PIXSCALE', 0.5)}
            bunit = hdr.get('BUNIT', '')
        cube = CoaddCube(data=data, var=var, wave=wave, coverage=cov, nexp=nexp, wcs=wcs,
                         bunit=bunit, meta=meta)
        return cls(cube)
