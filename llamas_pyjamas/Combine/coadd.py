"""
Co-add engine: resample a field's fibres onto an output sky grid (Phase 4, step 2).

Reads the per-fibre scalar table from :meth:`SuperRSS.collapse_band` and lays every fibre down on a
common RA/DEC grid with a spatial kernel, producing a combined image plus the maps that make the
varying depth explicit: a variance map, a fibre-coverage map, and an exposure-depth map.

The combine is a kernel-weighted mean per output pixel:

    value[p] = Σ_i W_i x_i / Σ_i W_i          W_i = weight_i · kernel(|p − fibre_i|)
    var[p]   = Σ_i W_i² var_i / (Σ_i W_i)²

With ``weighting='ivar'`` (weight_i = 1/var_i) this is the inverse-variance optimal combine, so a
deeply-covered pixel (many fibres / many exposures) reaches high S/N while a thinly-covered edge
pixel stays noisy but is kept — the deep-centre / shallow-halo behaviour dithered LLAMAS data has,
without trimming. Working in surface brightness (default) the weighted mean is intensity-conserving,
which is what diffuse emission needs.

Facility-general: kernel (gaussian/tophat + FWHM), weighting (ivar/uniform/exptime), units
(surface-brightness/flux), pixel scale and grid centre are all parameters.

Classes
-------
CoaddImage   The combined image + var/coverage/exposure-depth maps + WCS; writes a DS9 FITS

Functions
---------
make_output_grid   Build the output TAN WCS + shape and project fibres onto it
coadd_image        Kernel-weighted co-add of a FibreTable -> CoaddImage
combine_image      Convenience: SuperRSS + wavelength window -> CoaddImage
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)

# Default short-wavelength floor for WHITE-LIGHT collapses. Below this the blue detector sensitivity
# falls to ~0, so the flux calibration diverges and those planes are read-noise + artifact dominated
# (see Sky/diagnosis: the blue-edge over-subtraction spike). The CUBE keeps all wavelengths; this only
# floors the DEFAULT white-light window, and callers warn rather than discard if a user goes bluer.
WHITELIGHT_BLUE_MIN_A = 3600.0


def whitelight_floor(wave_min, wave_max):
    """Apply the white-light blue floor to a default window; returns (floored_min, below_floor).

    ``below_floor`` is True when the requested min was below the floor (so callers can warn). Only
    lowers when it wouldn't invert the window (max still above the floor)."""
    below = wave_min < WHITELIGHT_BLUE_MIN_A
    if below and wave_max > WHITELIGHT_BLUE_MIN_A:
        return WHITELIGHT_BLUE_MIN_A, True
    return wave_min, below


# Default coverage-depth floor for WHITE-LIGHT / collapsed views, as a fraction of the field's peak
# NEXP. Spaxels shallower than this are built from only a few dithers' edge-of-IFU fibres, so the
# co-add value is BIASED and its variance is under-estimated (Sky/diagnosis/boundary_diag: J0958 NaN-
# stripe boundaries reach ~90x sqrt(VAR); the bias clears by ~0.8*max). Default-EXCLUDED from images/
# stats but NEVER discarded from the cube -- the coverage/NEXP maps stay so the mask is reversible.
COVERAGE_FRAC_MIN = 0.7


def low_coverage_mask(nexp, frac=COVERAGE_FRAC_MIN):
    """Boolean map (True = exclude) of spaxels shallower than ``frac`` x peak NEXP. frac<=0 => none."""
    nexp = np.asarray(nexp, float)
    peak = np.nanmax(nexp) if np.isfinite(nexp).any() else 0.0
    if frac <= 0 or peak <= 0:
        return np.zeros(nexp.shape, bool)
    return nexp < frac * peak


def _tan_wcs(ra0, dec0, crpix, pixscale_arcsec):
    """A plain north-up / east-left TAN WCS (no instrument rotation) for the output grid."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [float(ra0), float(dec0)]
    w.wcs.crpix = [float(crpix[0]), float(crpix[1])]
    w.wcs.cdelt = [-pixscale_arcsec / 3600.0, pixscale_arcsec / 3600.0]   # E-left, N-up
    w.wcs.cunit = ['deg', 'deg']
    return w


def make_output_grid(ra, dec, pixscale, pad_pix, center=None):
    """Output TAN WCS + (ny, nx) sized to hold all fibres with ``pad_pix`` margin, plus the fibres'
    pixel coordinates. Returns (wcs, ny, nx, px, py)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra0 = float(np.median(ra)) if center is None else float(center[0])
    dec0 = float(np.median(dec)) if center is None else float(center[1])
    sc = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)

    w0 = _tan_wcs(ra0, dec0, (1.0, 1.0), pixscale)
    px, py = w0.world_to_pixel(sc)
    xmin, ymin = np.floor(px.min()), np.floor(py.min())
    xmax, ymax = np.ceil(px.max()), np.ceil(py.max())
    pad = int(np.ceil(pad_pix)) + 2
    nx = int(xmax - xmin) + 2 * pad + 1
    ny = int(ymax - ymin) + 2 * pad + 1
    wcs = _tan_wcs(ra0, dec0, (1.0 - xmin + pad, 1.0 - ymin + pad), pixscale)
    px, py = wcs.world_to_pixel(sc)
    return wcs, ny, nx, np.asarray(px, float), np.asarray(py, float)


def _kernel(name, fwhm_arcsec, pixscale):
    """Return (radius_pix, kernel_fn(dist_pix)->weight). Gaussian cut at 3 sigma, tophat at FWHM/2."""
    fwhm_pix = fwhm_arcsec / pixscale
    if name == 'tophat':
        r = 0.5 * fwhm_pix
        return max(r, 0.5), lambda d: np.ones_like(d)
    if name == 'gaussian':
        sig = fwhm_pix / 2.3548200450309493
        return max(3.0 * sig, 0.75), lambda d: np.exp(-0.5 * (d / sig) ** 2)
    raise ValueError(f'unknown kernel {name!r} (use gaussian|tophat)')


def _weights(weighting, var, ft, exptime):
    """Per-fibre base weight (before the spatial kernel)."""
    if weighting == 'ivar':
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(np.isfinite(var) & (var > 0), 1.0 / var, 0.0)
    if weighting == 'uniform':
        return np.ones_like(var)
    if weighting == 'exptime':
        if exptime is None:
            logger.warning("weighting='exptime' but no exptime given; using uniform")
            return np.ones_like(var)
        return np.asarray(exptime, float)[ft.exposure]
    raise ValueError(f'unknown weighting {weighting!r} (use ivar|uniform|exptime)')


def _coverage_bool(px, py, r, ny, nx):
    """Boolean map of pixels within ``r`` of any of the given fibre positions."""
    cov = np.zeros((ny, nx), dtype=bool)
    r2 = r * r
    for xi, yi in zip(px, py):
        x0, x1 = max(0, int(np.floor(xi - r))), min(nx - 1, int(np.ceil(xi + r)))
        y0, y1 = max(0, int(np.floor(yi - r))), min(ny - 1, int(np.ceil(yi + r)))
        if x0 > x1 or y0 > y1:
            continue
        xs = np.arange(x0, x1 + 1)
        ys = np.arange(y0, y1 + 1)
        d2 = (xs[None, :] - xi) ** 2 + (ys[:, None] - yi) ** 2
        cov[y0:y1 + 1, x0:x1 + 1] |= d2 <= r2
    return cov


@dataclass
class CoaddImage:
    """A combined image with its depth maps and WCS."""
    data: np.ndarray               #: (ny, nx) combined value (SB or flux)
    var: np.ndarray                #: (ny, nx) variance
    coverage: np.ndarray           #: (ny, nx) number of fibres contributing
    nexp: np.ndarray               #: (ny, nx) number of distinct exposures contributing
    wcs: WCS
    bunit: str
    meta: Dict = field(default_factory=dict)

    def snr(self):
        with np.errstate(invalid='ignore', divide='ignore'):
            return self.data / np.sqrt(self.var)

    def write(self, path):
        """Write a multi-extension FITS (SCI/VAR/SNR/COVERAGE/NEXP) with the WCS, for DS9."""
        hdr = self.wcs.to_header()
        hdr['BUNIT'] = self.bunit
        for k, v in self.meta.items():
            try:
                hdr[k] = v
            except (ValueError, TypeError):
                hdr[k] = str(v)
        hdus = [fits.PrimaryHDU(self.data.astype(np.float32), header=hdr)]
        for name, arr, bunit in (('VAR', self.var, f'({self.bunit})^2'),
                                 ('SNR', self.snr(), ''),
                                 ('COVERAGE', self.coverage.astype(np.int32), 'fibres'),
                                 ('NEXP', self.nexp.astype(np.int32), 'exposures')):
            h = fits.ImageHDU(arr.astype(np.float32) if arr.dtype.kind == 'f' else arr,
                              header=self.wcs.to_header(), name=name)
            if bunit:
                h.header['BUNIT'] = bunit
            hdus.append(h)
        fits.HDUList(hdus).writeto(path, overwrite=True)
        return path


def coadd_image(ft, *, pixscale=0.5, kernel='gaussian', kernel_fwhm=0.9, weighting='ivar',
                units='sb', min_coverage=1, center=None, exptime=None, bunit='') -> CoaddImage:
    """Kernel-weighted co-add of a :class:`~llamas_pyjamas.Combine.superRSS.FibreTable`.

    ``units='sb'`` co-adds surface brightness (value/fibre-area; intensity-conserving mean) for
    diffuse emission; ``'flux'`` co-adds the flux values directly. ``weighting``: ivar (default) |
    uniform | exptime. Pixels below ``min_coverage`` fibres are set NaN.
    """
    if units == 'sb':
        x, var = ft.surface_brightness()
        unit = (bunit or 'value') + '/arcsec2'
    elif units == 'flux':
        x, var = np.asarray(ft.value, float), np.asarray(ft.var, float)
        unit = bunit or 'value'
    else:
        raise ValueError(f'unknown units {units!r} (use sb|flux)')

    w = _weights(weighting, var, ft, exptime)
    good = (np.isfinite(x) & np.isfinite(var) & (var > 0) & np.isfinite(w) & (w > 0)
            & np.isfinite(ft.ra) & np.isfinite(ft.dec))
    if not good.any():
        raise ValueError('no valid fibres to co-add')
    x, var, w = x[good], var[good], w[good]
    ra, dec, expo = ft.ra[good], ft.dec[good], ft.exposure[good]

    r, kern = _kernel(kernel, kernel_fwhm, pixscale)
    wcs, ny, nx, px, py = make_output_grid(ra, dec, pixscale, r, center)

    npix = ny * nx
    sumW = np.zeros(npix)
    sumWX = np.zeros(npix)
    sumW2V = np.zeros(npix)
    cov = np.zeros(npix, dtype=np.int64)
    r2 = r * r
    for i in range(len(px)):
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
        yy, xx = np.nonzero(inside)                       # indices within the local box
        flat = (y0 + yy) * nx + (x0 + xx)                 # -> flat index into the (ny,nx) grid
        k = kern(np.sqrt(d2[inside]))
        W = k * w[i]
        np.add.at(sumW, flat, W)
        np.add.at(sumWX, flat, W * x[i])
        np.add.at(sumW2V, flat, W * W * var[i])
        np.add.at(cov, flat, 1)

    with np.errstate(invalid='ignore', divide='ignore'):
        data = (sumWX / sumW).reshape(ny, nx)
        vmap = (sumW2V / sumW ** 2).reshape(ny, nx)
    coverage = cov.reshape(ny, nx)
    below = coverage < min_coverage
    data[below] = np.nan
    vmap[below] = np.nan

    nexp = np.zeros((ny, nx), dtype=np.int32)
    for e in np.unique(expo):
        sel = expo == e
        nexp += _coverage_bool(px[sel], py[sel], r, ny, nx).astype(np.int32)
    nexp[below] = 0

    meta = dict(WGHT=weighting, KERNEL=kernel, KERNFWHM=kernel_fwhm, PIXSCALE=pixscale,
                UNITS=units, NEXPTOT=int(len(np.unique(expo))), NFIBRES=int(len(px)))
    return CoaddImage(data=data, var=vmap, coverage=coverage, nexp=nexp, wcs=wcs,
                      bunit=unit, meta=meta)


def combine_image(super_rss, wave_min, wave_max, *, channels=None, **kwargs) -> CoaddImage:
    """Convenience: collapse a SuperRSS over a wavelength window and co-add it into an image.

    ``bunit`` defaults to the SuperRSS plane's unit; everything else is passed to
    :func:`coadd_image`. This is the broadband / narrowband SB image view."""
    ft = super_rss.collapse_band(wave_min, wave_max, channels=channels)
    kwargs.setdefault('bunit', super_rss.bunit)
    img = coadd_image(ft, **kwargs)
    img.meta['FIELD'] = super_rss.field
    img.meta['WAVEMIN'] = float(wave_min)
    img.meta['WAVEMAX'] = float(wave_max)
    return img
