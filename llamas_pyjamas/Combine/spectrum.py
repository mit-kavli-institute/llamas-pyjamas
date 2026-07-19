"""
Optimal point-source spectrum from a field's stacked dithers (Phase 4).

For a point source (a quasar here) the deepest spectrum is NOT a straight aperture sum of spaxels
-- it is the PSF- and inverse-variance-weighted combination of every fibre that saw the source,
across all dithers (Horne 1986 optimal extraction, in fibre space):

    F(l) = Σ_i P_i S_i(l)/V_i(l) / Σ_i P_i^2/V_i(l),     Var F(l) = 1 / Σ_i P_i^2/V_i(l)

with S_i the fibre's (native, then resampled) flux, V_i its variance, and P_i the source PSF
sampled at fibre i's offset from the source centroid (achromatic Gaussian; normalised over the
aperture). Fibres near the core and in the sharpest exposures dominate; a hazy fibre far from the
core contributes little. The absolute scale carries the usual aperture loss (fix later by anchoring
to Gaia); the spectral SHAPE and S/N are what this optimises.

Functions
---------
estimate_psf_fwhm   Flux-weighted-moment seeing estimate at a position
optimal_spectrum    PSF/ivar-weighted 1D spectrum per channel at a sky position
"""

import logging
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from llamas_pyjamas.Combine.superRSS import ChannelStack

logger = logging.getLogger(__name__)


def _sep_arcsec(ra, dec, ra0, dec0):
    return SkyCoord(ra * u.deg, dec * u.deg).separation(SkyCoord(ra0 * u.deg, dec0 * u.deg)).arcsec


def _full_band(super_rss, channels):
    lo, hi = np.inf, -np.inf
    for c in (channels or list(super_rss.channels)):
        st = super_rss.channels.get(c)
        w = st.wave[np.isfinite(st.wave)] if st is not None else np.array([])
        if w.size:
            lo, hi = min(lo, float(w.min())), max(hi, float(w.max()))
    return lo, hi


def _subselect(st: ChannelStack, sel) -> ChannelStack:
    return ChannelStack(channel=st.channel, ra=st.ra[sel], dec=st.dec[sel], wave=st.wave[sel],
                        flux=st.flux[sel], var=st.var[sel], mask=st.mask[sel],
                        solid_angle=st.solid_angle[sel], exposure=st.exposure[sel])


def estimate_psf_fwhm(super_rss, ra, dec, *, channels=None, radius_arcsec=3.0, default=1.2):
    """Seeing FWHM (arcsec) at a position, from the background-subtracted flux-weighted second
    moment of the fibres within ``radius_arcsec``. Falls back to ``default`` if ill-conditioned."""
    lo, hi = _full_band(super_rss, channels)
    ft = super_rss.collapse_band(lo, hi, channels=channels)
    if len(ft) == 0:
        return default
    sep = _sep_arcsec(ft.ra, ft.dec, ra, dec)
    ann = (sep > radius_arcsec) & (sep < 2 * radius_arcsec)
    bg = float(np.nanmedian(ft.value[ann])) if ann.any() else 0.0
    sel = sep < radius_arcsec
    w = np.clip(ft.value[sel] - bg, 0.0, None)
    if w.sum() <= 0:
        return default
    sigma = np.sqrt(np.sum(w * sep[sel] ** 2) / np.sum(w))
    fwhm = 2.35482 * float(sigma)
    return fwhm if np.isfinite(fwhm) and fwhm > 0.3 else default


def optimal_spectrum(super_rss, ra, dec, *, radius_arcsec=3.0, fwhm=None, channels=None,
                     dwave=None) -> Tuple[Dict[str, tuple], float]:
    """PSF/inverse-variance-weighted 1D spectrum at ``(ra, dec)``, per channel.

    Returns ``({channel: (wave, flux, var)}, fwhm)``. Flux is in the super-RSS plane's units
    (FLAM if flux-calibrated). ``fwhm`` (arcsec) is estimated from the data if not given."""
    from llamas_pyjamas.Combine.cube import _wave_grid, _resample_fibres
    chans = [c for c in ('blue', 'green', 'red')
             if c in super_rss.channels and (channels is None or c in channels)]
    if fwhm is None:
        fwhm = estimate_psf_fwhm(super_rss, ra, dec, channels=chans, radius_arcsec=radius_arcsec)
    sigma = fwhm / 2.35482

    out: Dict[str, tuple] = {}
    for c in chans:
        st = super_rss.channels[c]
        sep = _sep_arcsec(st.ra, st.dec, ra, dec)
        sel = sep < radius_arcsec
        if sel.sum() == 0:
            continue
        sub = _subselect(st, sel)
        wl, _dw = _wave_grid(sub, dwave, None)
        x, V, bad = _resample_fibres(sub, wl, units='flux')      # (nsel, nw) flux + variance
        P = np.exp(-0.5 * (sep[sel] / sigma) ** 2)
        P = P / P.sum() if P.sum() > 0 else P                    # normalise over the aperture
        Pc = P[:, None]
        wgt = np.where(bad, 0.0, 1.0 / V)
        num = np.nansum(Pc * x * wgt, axis=0)
        den = np.nansum(Pc ** 2 * wgt, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            flux = np.where(den > 0, num / den, np.nan)
            var = np.where(den > 0, 1.0 / den, np.nan)
        out[c] = (wl, flux, var)
    return out, float(fwhm)
