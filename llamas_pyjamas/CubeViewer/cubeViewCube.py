"""
SpectralScene over a combined datacube — the cube view in CubeViewer.

Wraps a :class:`llamas_pyjamas.Combine.cube.CoaddCube` (or a cube FITS written by it) as a
:class:`~llamas_pyjamas.CubeViewer.cubeViewScene.SpectralScene`, so the existing DS9 crosshair
picking and spectrum panel work on a stacked cube unchanged: collapse a wavelength window into a
white-light image, click a spaxel, plot its spectrum. Unlike the RSS scene (spatial element = a
fibre) the element here is a spaxel ``(iy, ix)`` and the WCS is a real RA---TAN/DEC--TAN.

Built for the "Combine" menu (stack a field's dithers -> cube -> inspect) and for opening a saved
cube FITS.

Classes
-------
CoaddCubeScene   SpectralScene over a CoaddCube
"""

import logging
from typing import Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np

from llamas_pyjamas.CubeViewer.cubeViewScene import Spectrum, SpectralScene

logger = logging.getLogger(__name__)


class CoaddCubeScene(SpectralScene):
    """A scene backed by a combined (RA, DEC, wave) cube. Spatial element = spaxel ``(iy, ix)``."""

    def __init__(self, cube) -> None:
        self.cube = cube
        self.channel = str(cube.meta.get('CHANNEL', 'green'))
        self.channels = (self.channel,)
        self.object = str(cube.meta.get('FIELD', ''))
        self._calibrated = 'erg' in cube.bunit
        self.has_flam = self._calibrated
        self.hex_tiles = None                       # tell the window there is no hex toggle
        self._pixscale = float(cube.meta.get('PIXSCALE', 0.5))
        self.pitch = 0.75                           # arcsec, so the radius UI reads in fibre-spacings
        ny, nx = cube.data.shape[1:]
        self._ny, self._nx = ny, nx
        valid = cube.coverage > 0
        self.keys: List[Tuple[int, int]] = list(zip(*np.nonzero(valid)))

    # radius UI: image pixels per fibre-spacing
    def _step(self) -> float:
        return self._pixscale

    # ---- SpectralScene ----
    def wavelength_range(self, channel: Optional[str] = None) -> Tuple[float, float]:
        w = self.cube.wave
        return float(np.nanmin(w)), float(np.nanmax(w))

    def collapse(self, wave_min: float, wave_max: float,
                 channels: Optional[Sequence[str]] = None) -> Tuple[np.ndarray, object, Dict]:
        sl = (self.cube.wave >= wave_min) & (self.cube.wave <= wave_max)
        if not sl.any():
            sl = np.ones_like(self.cube.wave, dtype=bool)
        with np.errstate(invalid='ignore'):
            img = np.nanmean(np.where(np.isfinite(self.cube.data[sl]), self.cube.data[sl], np.nan),
                             axis=0)
        meta = {'contributions': {self.channel: int(np.isfinite(img).sum())},
                'WAVEMIN': (float(wave_min), 'A'), 'WAVEMAX': (float(wave_max), 'A'),
                'WCSFRAME': ('sky', 'Combined cube white light')}
        return img, self.cube.wcs.celestial, meta

    def _spaxel(self, x_pix: float, y_pix: float) -> Optional[Tuple[int, int]]:
        ix, iy = int(round(x_pix)) - 1, int(round(y_pix)) - 1   # DS9 1-based -> array index
        if 0 <= iy < self._ny and 0 <= ix < self._nx and self.cube.coverage[iy, ix] > 0:
            return (iy, ix)
        return None

    def element_at(self, x_pix: float, y_pix: float) -> Optional[Hashable]:
        return self._spaxel(x_pix, y_pix)

    def _spectrum(self, iy: int, ix: int, label: str = '') -> Spectrum:
        val = np.asarray(self.cube.data[:, iy, ix], dtype=float)
        mask = ~np.isfinite(val)
        return Spectrum(wave=self.cube.wave, flux=val, channel=self.channel,
                        label=label or f'spaxel ({ix},{iy})', mask=mask,
                        flam=(val if self._calibrated else None))

    def spectra_at(self, x_pix: float, y_pix: float) -> List[Spectrum]:
        e = self._spaxel(x_pix, y_pix)
        return [] if e is None else [self._spectrum(*e)]

    def spectra_of(self, elements: Sequence[Hashable]) -> List[Spectrum]:
        return [self._spectrum(iy, ix) for (iy, ix) in elements]

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
                if (iy - cy) ** 2 + (ix - cx) ** 2 <= radius_pix ** 2 and self.cube.coverage[iy, ix] > 0:
                    out.append((iy, ix))
        out.sort(key=lambda e: (e[0] - cy) ** 2 + (e[1] - cx) ** 2)   # nearest first
        return out

    def marker_region(self, x_pix: float, y_pix: float) -> str:
        e = self._spaxel(x_pix, y_pix)
        return '' if e is None else self.region_for([e])

    def region_for(self, elements: Sequence[Hashable]) -> str:
        if not elements:
            return ''
        lines = ['image']
        for (iy, ix) in elements:
            lines.append(f'box({ix + 1},{iy + 1},1,1,0) # color=cyan width=2')
        return '\n'.join(lines) if len(lines) > 1 else ''

    @classmethod
    def from_fits(cls, path: str) -> 'CoaddCubeScene':
        """Open a cube FITS written by :meth:`CoaddCube.write`."""
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
