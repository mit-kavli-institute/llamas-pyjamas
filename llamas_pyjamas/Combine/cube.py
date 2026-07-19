"""
Cube co-add: resample a field's fibres into an (RA, DEC, wavelength) datacube (Phase 4, step 3).

The general container the image and spectrum views are slices of. Each fibre keeps its native
wavelength solution until here; the cube is where the ONE spectral resample happens (onto a common
linear grid) alongside the spatial co-add.

Done efficiently with a single sparse spatial-kernel matrix. The spatial kernel (which output
spaxels a fibre touches, with what weight) is wavelength-independent, so it is built once as a
sparse matrix S (n_spaxel x n_fibre) and applied to every wavelength plane by sparse-dense matrix
products:

    num[p, l] = Σ_i S[p,i] w[i,l] x[i,l]  =  S · (w·x)
    den[p, l] = Σ_i S[p,i] w[i,l]         =  S · w
    var[p, l] = Σ_i S[p,i]^2 w[i,l]^2 V[i,l] / den^2  =  S² · (w²·V) / den²

so the whole cube is a few (n_spaxel x n_fibre) · (n_fibre x n_wave) products rather than a Python
loop over planes. With ``weighting='ivar'`` (w = 1/V) this is the inverse-variance combine plane by
plane; ``units='sb'`` gives an intensity-conserving surface-brightness cube.

Facility-general: channel, wavelength grid (dwave / range), pixel scale, kernel, weighting, units,
min-coverage are all parameters. A masked or uncovered (fibre, wavelength) is no-data (weight 0),
so broken fibres and gaps simply lower the coverage.

Classes
-------
CoaddCube    Datacube + variance + coverage/exposure-depth maps + 3-D WCS; writes a FITS cube

Functions
---------
combine_cube   SuperRSS + channel -> CoaddCube
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.sparse import coo_matrix

from llamas_pyjamas.Combine.coadd import _kernel, make_output_grid, _coverage_bool

logger = logging.getLogger(__name__)


@dataclass
class CoaddCube:
    """A combined datacube (nwave, ny, nx) with variance, 2-D depth maps, wavelength axis + WCS."""
    data: np.ndarray               #: (nw, ny, nx) combined value (SB or flux)
    var: np.ndarray                #: (nw, ny, nx) variance
    wave: np.ndarray               #: (nw,) Angstrom
    coverage: np.ndarray           #: (ny, nx) number of fibres contributing (spatial)
    nexp: np.ndarray               #: (ny, nx) distinct exposures contributing (spatial)
    wcs: WCS                       #: 3-D (RA, DEC, WAVE)
    bunit: str
    meta: Dict = field(default_factory=dict)

    def white_light(self):
        """Median over wavelength -> a 2-D image (quick sanity view)."""
        return np.nanmedian(self.data, axis=0)

    def write(self, path):
        """Write a FITS cube: SCI primary (3-D WCS) + VAR, COVERAGE, NEXP, and a WAVELENGTH table.
        CRVAL3/CDELT3 carry the linear wave axis, but astropy SI-normalises it on write, so the
        WAVELENGTH bin-table is the authoritative Angstrom axis (as in the pipeline cubes)."""
        hdr = self.wcs.to_header()
        hdr['BUNIT'] = self.bunit
        for k, v in self.meta.items():
            try:
                hdr[k] = v
            except (ValueError, TypeError):
                hdr[k] = str(v)
        hdus = [fits.PrimaryHDU(self.data.astype(np.float32), header=hdr),
                fits.ImageHDU(self.var.astype(np.float32), header=self.wcs.to_header(), name='VAR'),
                fits.ImageHDU(self.coverage.astype(np.int32), name='COVERAGE'),
                fits.ImageHDU(self.nexp.astype(np.int32), name='NEXP'),
                fits.BinTableHDU.from_columns(
                    [fits.Column(name='WAVELENGTH', format='D', unit='Angstrom', array=self.wave)],
                    name='WAVELENGTH')]
        fits.HDUList(hdus).writeto(path, overwrite=True)
        return path


def _wave_grid(st, dwave, wave_range):
    finite = st.wave[np.isfinite(st.wave)]
    lo, hi = (wave_range if wave_range is not None
              else (float(np.percentile(finite, 0.5)), float(np.percentile(finite, 99.5))))
    if dwave is None:
        dwave = float(np.nanmedian(np.abs(np.diff(st.wave, axis=1))))
    nw = int(round((hi - lo) / dwave)) + 1
    return lo + np.arange(nw) * dwave, dwave


def _resample_fibres(st, wl, units):
    """Resample every fibre onto the common wave grid ``wl``. Returns (x, V, bad) each (N, nw):
    x = flux (or SB), V = variance, bad = no-data (outside the fibre's coverage or masked). The one
    spectral resample -- linear interp, so mildly non-flux-conserving (documented)."""
    n, nw = st.n_fibres, wl.size
    x = np.zeros((n, nw), np.float32)
    V = np.full((n, nw), np.inf, np.float64)
    for i in range(n):
        m = (~st.mask[i]) & np.isfinite(st.wave[i]) & np.isfinite(st.flux[i]) & np.isfinite(st.var[i])
        if m.sum() < 2:
            continue
        wv = st.wave[i][m]
        order = np.argsort(wv)
        wv = wv[order]
        fi = np.interp(wl, wv, st.flux[i][m][order], left=np.nan, right=np.nan)
        vi = np.interp(wl, wv, st.var[i][m][order], left=np.inf, right=np.inf)
        cov = np.isfinite(fi)
        x[i, cov] = fi[cov]
        V[i, cov] = vi[cov]
    bad = ~np.isfinite(x) | ~np.isfinite(V) | (V <= 0)
    x[bad] = 0.0
    V[bad] = np.inf
    if units == 'sb':
        a = st.solid_angle[:, None]
        x = (x / a).astype(np.float32)
        V = V / a ** 2
    return x, V, bad


def combine_cube(super_rss, channel='green', *, dwave=None, wave_range=None, pixscale=0.5,
                 kernel='gaussian', kernel_fwhm=0.9, weighting='ivar', units='sb',
                 min_coverage=1, center=None) -> CoaddCube:
    """Resample one channel of a :class:`SuperRSS` into an (RA, DEC, wave) cube.

    ``units='sb'`` (default) is intensity-conserving surface brightness; ``'flux'`` co-adds flux.
    ``weighting``: ivar (default) | uniform | exptime. The wave grid defaults to the channel's
    range at its median native dispersion. Spaxels below ``min_coverage`` fibres are NaN."""
    if channel not in super_rss.channels:
        raise ValueError(f'channel {channel!r} not in super-RSS ({list(super_rss.channels)})')
    st = super_rss.channels[channel]
    wl, dwave = _wave_grid(st, dwave, wave_range)
    nw = wl.size

    x, V, bad = _resample_fibres(st, wl, units)
    if weighting == 'ivar':
        w = np.where(bad, 0.0, 1.0 / V)
    elif weighting == 'uniform':
        w = np.where(bad, 0.0, 1.0)
    elif weighting == 'exptime':
        et = np.array([e.exptime for e in super_rss.exposures], float)[st.exposure][:, None]
        w = np.where(bad, 0.0, et)
    else:
        raise ValueError(f'unknown weighting {weighting!r}')

    # spatial kernel matrix S (n_spaxel x n_fibre), wavelength-independent
    r, kern = _kernel(kernel, kernel_fwhm, pixscale)
    wcs2, ny, nx, px, py = make_output_grid(st.ra, st.dec, pixscale, r, center)
    r2 = r * r
    rows, cols, vals = [], [], []
    for i in range(st.n_fibres):
        xi, yi = px[i], py[i]
        x0, x1 = max(0, int(np.floor(xi - r))), min(nx - 1, int(np.ceil(xi + r)))
        y0, y1 = max(0, int(np.floor(yi - r))), min(ny - 1, int(np.ceil(yi + r)))
        if x0 > x1 or y0 > y1:
            continue
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        d2 = (xs[None, :] - xi) ** 2 + (ys[:, None] - yi) ** 2
        inside = d2 <= r2
        if not inside.any():
            continue
        yy, xx = np.nonzero(inside)
        rows.append((y0 + yy) * nx + (x0 + xx))
        cols.append(np.full(yy.size, i))
        vals.append(kern(np.sqrt(d2[inside])).astype(np.float32))
    npix = ny * nx
    S = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(npix, st.n_fibres)).tocsr()
    S2 = S.multiply(S)

    # float64 throughout: with FLAM ~1e-16, ivar weights ~1e34 and den^2 ~1e68 overflow float32.
    Vsafe = np.where(bad, 1.0, V)                         # avoid 0*inf in w^2 V
    wx = w * x
    w2V = np.where(bad, 0.0, w * w * Vsafe)
    with np.errstate(invalid='ignore', divide='ignore'):
        den = S.dot(w)                                    # (npix, nw)
        data = S.dot(wx) / den
        var = S2.dot(w2V) / den ** 2

    coverage = np.asarray((S > 0).sum(axis=1)).ravel().astype(np.int64)
    below = coverage < min_coverage
    data[below, :] = np.nan
    var[below, :] = np.nan

    nexp = np.zeros(npix, dtype=np.int32)
    for e in np.unique(st.exposure):
        sel = st.exposure == e
        nexp += _coverage_bool(px[sel], py[sel], r, ny, nx).astype(np.int32).ravel()
    nexp[below] = 0

    # to (nw, ny, nx)
    data = np.moveaxis(data.reshape(ny, nx, nw), 2, 0)
    var = np.moveaxis(var.reshape(ny, nx, nw), 2, 0)

    wcs = _wcs3d(wcs2, wl[0], dwave)
    unit = (super_rss.bunit + '/arcsec2') if units == 'sb' else super_rss.bunit
    meta = dict(FIELD=super_rss.field, CHANNEL=channel, WGHT=weighting, UNITS=units,
                KERNFWHM=kernel_fwhm, PIXSCALE=pixscale, DWAVE=dwave,
                NEXPTOT=super_rss.n_exposures)
    return CoaddCube(data=data, var=var, wave=wl, coverage=coverage.reshape(ny, nx),
                     nexp=nexp.reshape(ny, nx), wcs=wcs, bunit=unit, meta=meta)


def _wcs3d(wcs2, wl0, dwave):
    """Extend a 2-D spatial WCS with a linear WAVE axis (Angstrom)."""
    w = WCS(naxis=3)
    w.wcs.ctype = [wcs2.wcs.ctype[0], wcs2.wcs.ctype[1], 'WAVE']
    w.wcs.crval = [wcs2.wcs.crval[0], wcs2.wcs.crval[1], wl0]
    w.wcs.crpix = [wcs2.wcs.crpix[0], wcs2.wcs.crpix[1], 1.0]
    w.wcs.cdelt = [wcs2.wcs.cdelt[0], wcs2.wcs.cdelt[1], dwave]
    w.wcs.cunit = ['deg', 'deg', 'Angstrom']
    return w
