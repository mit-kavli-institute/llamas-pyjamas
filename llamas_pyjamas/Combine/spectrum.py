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
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from llamas_pyjamas.Combine.superRSS import ChannelStack

logger = logging.getLogger(__name__)


@dataclass
class ProfileFit:
    """The fitted source profile used as the extraction weight, in sky terms so a viewer can draw
    it (1-sigma / 2-sigma ellipses)."""
    ra: float                 #: fitted centroid RA (deg)
    dec: float                #: fitted centroid Dec (deg)
    sigma_x: float            #: arcsec (major/x axis)
    sigma_y: float            #: arcsec
    theta_deg: float          #: position angle of the x axis (deg)
    fitted: bool              #: True = 2-D Gaussian fit; False = fallback (moment / assumed)

    @property
    def fwhm(self) -> float:
        return 2.35482 * float(np.sqrt(abs(self.sigma_x * self.sigma_y)))


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


def fit_source_profile(super_rss, ra, dec, *, radius_arcsec=3.0, channels=None):
    """Fit a 2-D Gaussian to the source's broadband spatial profile (fibre space, background-
    subtracted). Returns ``(model, ProfileFit)`` -- the astropy Gaussian2D in the (ra,dec)-centred
    arcsec frame plus a sky-terms summary -- or ``None`` if it can't fit (too few fibres / no
    signal), so the caller can fall back."""
    from astropy.modeling import models, fitting
    lo, hi = _full_band(super_rss, channels)
    ft = super_rss.collapse_band(lo, hi, channels=channels)
    if len(ft) == 0:
        return None
    cosd = np.cos(np.deg2rad(dec))
    dx = (ft.ra - ra) * cosd * 3600.0
    dy = (ft.dec - dec) * 3600.0
    sep = np.hypot(dx, dy)
    sel = sep < radius_arcsec
    ann = (sep >= radius_arcsec) & (sep < 2 * radius_arcsec)
    bg = float(np.nanmedian(ft.value[ann])) if ann.any() else 0.0
    z = ft.value[sel] - bg
    good = np.isfinite(z) & np.isfinite(dx[sel]) & np.isfinite(dy[sel])
    if good.sum() < 6 or not np.any(z[good] > 0):
        return None
    g0 = models.Gaussian2D(amplitude=float(np.nanmax(z[good])), x_mean=0.0, y_mean=0.0,
                           x_stddev=0.6, y_stddev=0.6)
    g0.x_mean.bounds = g0.y_mean.bounds = (-radius_arcsec, radius_arcsec)
    g0.x_stddev.bounds = g0.y_stddev.bounds = (0.2, radius_arcsec)
    try:
        fitter = fitting.TRFLSQFitter()
    except AttributeError:                                        # older astropy
        fitter = fitting.LevMarLSQFitter()
    try:
        g = fitter(g0, dx[sel][good], dy[sel][good], z[good], maxiter=300)
    except Exception as exc:                                      # noqa: BLE001
        logger.warning('profile fit failed (%s); falling back', exc)
        return None
    ra_fit = ra + float(g.x_mean.value) / (cosd * 3600.0)
    dec_fit = dec + float(g.y_mean.value) / 3600.0
    fit = ProfileFit(ra=ra_fit, dec=dec_fit, sigma_x=abs(float(g.x_stddev.value)),
                     sigma_y=abs(float(g.y_stddev.value)),
                     theta_deg=float(np.rad2deg(g.theta.value)), fitted=True)
    return g, fit


def optimal_spectrum(super_rss, ra, dec, *, radius_arcsec=3.0, fwhm=None, channels=None,
                     dwave=None, fit_profile=True) -> Tuple[Dict[str, tuple], ProfileFit]:
    """PSF/inverse-variance-weighted 1D spectrum at ``(ra, dec)``, per channel.

    ``fit_profile`` (default) fits a 2-D Gaussian to the source (centroid + widths + PA) and uses
    that fitted profile -- and its refined centroid -- as the extraction weight P; if the fit fails
    it falls back to an assumed circular Gaussian (``fwhm``, or a moment estimate) at the input
    position. Returns ``({channel: (wave, flux, var)}, ProfileFit)``. Flux is in the plane's units
    (FLAM if flux-calibrated)."""
    from llamas_pyjamas.Combine.cube import _wave_grid, _resample_fibres
    chans = [c for c in ('blue', 'green', 'red')
             if c in super_rss.channels and (channels is None or c in channels)]

    model = fit_source_profile(super_rss, ra, dec, radius_arcsec=radius_arcsec,
                               channels=chans) if fit_profile else None
    if model is not None:
        g, fit = model
        cxm, cym = float(g.x_mean.value), float(g.y_mean.value)   # aperture centre (arcsec frame)

        def profile(dxa, dya):                                    # normalised shape (amp=1)
            return g(dxa, dya) / g.amplitude.value
    else:
        fw = fwhm or estimate_psf_fwhm(super_rss, ra, dec, channels=chans,
                                       radius_arcsec=radius_arcsec)
        sig = fw / 2.35482
        cxm = cym = 0.0
        fit = ProfileFit(ra=ra, dec=dec, sigma_x=sig, sigma_y=sig, theta_deg=0.0, fitted=False)

        def profile(dxa, dya):
            return np.exp(-0.5 * (dxa ** 2 + dya ** 2) / sig ** 2)

    cosd = np.cos(np.deg2rad(dec))
    out: Dict[str, tuple] = {}
    for c in chans:
        st = super_rss.channels[c]
        dx = (st.ra - ra) * cosd * 3600.0
        dy = (st.dec - dec) * 3600.0
        sel = np.hypot(dx - cxm, dy - cym) < radius_arcsec       # aperture around the (fitted) centre
        if sel.sum() == 0:
            continue
        sub = _subselect(st, sel)
        wl, _dw = _wave_grid(sub, dwave, None)
        x, V, bad = _resample_fibres(sub, wl, units='flux')      # (nsel, nw) flux + variance
        P = profile(dx[sel], dy[sel])
        P = P / P.sum() if P.sum() > 0 else P                    # normalise over the aperture
        Pc = P[:, None]
        wgt = np.where(bad, 0.0, 1.0 / V)
        num = np.nansum(Pc * x * wgt, axis=0)
        den = np.nansum(Pc ** 2 * wgt, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            flux = np.where(den > 0, num / den, np.nan)
            var = np.where(den > 0, 1.0 / den, np.nan)
        out[c] = (wl, flux, var)
    return out, fit
