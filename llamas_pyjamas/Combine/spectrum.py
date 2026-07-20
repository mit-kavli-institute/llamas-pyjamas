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


def _refractivity(lam_A):
    """Air refractivity (n-1) vs wavelength (Angstrom), standard dispersion formula. The DAR offset
    is proportional to this, so centroid-vs-refractivity is linear -- with the correct nonlinear-in-
    wavelength (blue-steepening) shape built in."""
    lam_um = np.asarray(lam_A, float) / 1e4
    s = 1.0 / lam_um ** 2
    return 1e-6 * (64.328 + 29498.1 / (146.0 - s) + 255.4 / (41.0 - s))


@dataclass
class DARModel:
    """Differential-refraction centroid track: tangent-plane offset (arcsec) vs wavelength, LINEAR
    in refractivity Delta-n (physics-shaped, so robust to faint/low-blue signal)."""
    ax: float
    bx: float
    ay: float
    by: float
    lam0: float               #: reference wavelength (Angstrom)
    ra0: float = 0.0          #: sky reference the offsets are measured from (deg)
    dec0: float = 0.0

    def center(self, lam):
        """Tangent-plane centroid offset (arcsec) from (ra0, dec0) at wavelength ``lam``."""
        dn = _refractivity(lam) - _refractivity(self.lam0)
        return self.ax * dn + self.bx, self.ay * dn + self.by

    def center_sky(self, lam):
        """Centroid (RA, DEC) in degrees at wavelength ``lam``."""
        cx, cy = self.center(lam)
        cosd = np.cos(np.deg2rad(self.dec0))
        return self.ra0 + cx / (cosd * 3600.0), self.dec0 + cy / 3600.0

    def shift(self, lo, hi):
        (x0, y0), (x1, y1) = self.center(lo), self.center(hi)
        return float(np.hypot(x1 - x0, y1 - y0))


@dataclass
class ProfileFit:
    """The fitted source profile used as the extraction weight, in sky terms so a viewer can draw
    it (1-sigma / 2-sigma ellipses)."""
    ra: float                 #: reference (broadband) centroid RA (deg)
    dec: float                #: reference centroid Dec (deg)
    sigma_x: float            #: arcsec (major/x axis)
    sigma_y: float            #: arcsec
    theta_deg: float          #: position angle of the x axis (deg)
    fitted: bool              #: True = 2-D Gaussian fit; False = fallback (moment / assumed)
    dar_shift: float = 0.0    #: total centroid shift across the band (arcsec), 0 if DAR not tracked
    dar: Optional['DARModel'] = None   #: the fitted DAR track (for the R/G/B overlay), or None

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
    # Rescale to O(1) before the least-squares fit. Flux-calibrated data is ~1e-14 (FLAM); at that
    # magnitude the LSQ fitter's convergence tolerance is met immediately and it returns the INITIAL
    # guess unchanged (sigma=0.6" -> a spurious constant 1.41" FWHM, and a wrong extraction profile).
    # Amplitude scaling leaves the centroid, widths and PA unchanged, so this is safe; we restore the
    # physical amplitude afterwards for callers that might read it.
    scale = float(np.nanmax(z[good]))
    if not np.isfinite(scale) or scale <= 0:
        return None
    g0 = models.Gaussian2D(amplitude=1.0, x_mean=0.0, y_mean=0.0, x_stddev=0.6, y_stddev=0.6)
    g0.x_mean.bounds = g0.y_mean.bounds = (-radius_arcsec, radius_arcsec)
    g0.x_stddev.bounds = g0.y_stddev.bounds = (0.2, radius_arcsec)
    try:
        fitter = fitting.TRFLSQFitter()
    except AttributeError:                                        # older astropy
        fitter = fitting.LevMarLSQFitter()
    try:
        g = fitter(g0, dx[sel][good], dy[sel][good], z[good] / scale, maxiter=300)
    except Exception as exc:                                      # noqa: BLE001
        logger.warning('profile fit failed (%s); falling back', exc)
        return None
    g.amplitude.value = float(g.amplitude.value) * scale          # back to physical units
    ra_fit = ra + float(g.x_mean.value) / (cosd * 3600.0)
    dec_fit = dec + float(g.y_mean.value) / 3600.0
    fit = ProfileFit(ra=ra_fit, dec=dec_fit, sigma_x=abs(float(g.x_stddev.value)),
                     sigma_y=abs(float(g.y_stddev.value)),
                     theta_deg=float(np.rad2deg(g.theta.value)), fitted=True)
    return g, fit


def fit_gaussian_image(img, x0, y0, radius_pix):
    """Fit a 2-D Gaussian to a white-light image around pixel ``(x0, y0)`` (background-subtracted).
    Returns the astropy Gaussian2D model in pixel coords, or None. Used for cube-space extraction
    (when there is no super-RSS)."""
    from astropy.modeling import models, fitting
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(xx - x0, yy - y0)
    sel = (r < radius_pix) & np.isfinite(img)
    ann = (r >= radius_pix) & (r < 2 * radius_pix) & np.isfinite(img)
    if sel.sum() < 6:
        return None
    bg = float(np.median(img[ann])) if ann.any() else 0.0
    z = img[sel] - bg
    if not np.any(z > 0):
        return None
    # Rescale to O(1) so the LSQ fit converges on flux-calibrated (~1e-14) data instead of returning
    # the initial guess (see fit_source_profile). Widths/centroid are amplitude-invariant.
    scale = float(np.nanmax(z))
    if not np.isfinite(scale) or scale <= 0:
        return None
    g0 = models.Gaussian2D(1.0, x0, y0, radius_pix / 3, radius_pix / 3)
    g0.x_mean.bounds = (x0 - radius_pix, x0 + radius_pix)
    g0.y_mean.bounds = (y0 - radius_pix, y0 + radius_pix)
    g0.x_stddev.bounds = g0.y_stddev.bounds = (0.5, radius_pix)
    try:
        fitter = fitting.TRFLSQFitter()
    except AttributeError:
        fitter = fitting.LevMarLSQFitter()
    try:
        g = fitter(g0, xx[sel], yy[sel], z / scale, maxiter=300)
    except Exception as exc:                                      # noqa: BLE001
        logger.warning('image profile fit failed (%s)', exc)
        return None
    g.amplitude.value = float(g.amplitude.value) * scale          # back to physical units
    return g


def measure_dar(super_rss, ra, dec, *, radius_arcsec=3.0, channels=None, nbins=12, min_frac=0.1):
    """Track the source centroid vs wavelength (differential atmospheric refraction / ADC residual).

    Bins the wavelength range, takes the background-subtracted flux-weighted centroid of the fibres
    in each bin, and fits the centroid LINEARLY against air refractivity Delta-n (tangent-plane
    arcsec, relative to ``(ra, dec)``) -- a 2-parameter, physics-shaped fit that captures the blue
    steepening without overfitting. Low-S/N bins (amplitude below ``min_frac`` of the brightest) are
    rejected so faint / low-blue-signal sources don't drive it. Returns a :class:`DARModel` or None.
    """
    chans = [c for c in ('blue', 'green', 'red')
             if c in super_rss.channels and (channels is None or c in channels)]
    lo, hi = _full_band(super_rss, chans)
    edges = np.linspace(lo, hi, nbins + 1)
    cosd = np.cos(np.deg2rad(dec))
    wc, x0s, y0s, amps = [], [], [], []
    for i in range(nbins):
        ft = super_rss.collapse_band(edges[i], edges[i + 1], channels=chans)
        if len(ft) == 0:
            continue
        dx = (ft.ra - ra) * cosd * 3600.0
        dy = (ft.dec - dec) * 3600.0
        sep = np.hypot(dx, dy)
        sel = sep < radius_arcsec
        ann = (sep >= radius_arcsec) & (sep < 2 * radius_arcsec)
        bg = float(np.nanmedian(ft.value[ann])) if ann.any() else 0.0
        w = np.clip(ft.value[sel] - bg, 0.0, None)
        if sel.sum() < 6 or not np.isfinite(w).any() or w.sum() <= 0:
            continue
        wc.append(0.5 * (edges[i] + edges[i + 1]))
        x0s.append(float(np.sum(w * dx[sel]) / w.sum()))
        y0s.append(float(np.sum(w * dy[sel]) / w.sum()))
        amps.append(float(w.sum()))
    if len(wc) < 2:
        return None
    wc, x0s, y0s, amps = map(np.asarray, (wc, x0s, y0s, amps))
    keep = amps > min_frac * amps.max()                  # drop low-S/N bins (avoid overfitting)
    if keep.sum() < 2:
        keep = np.ones_like(amps, dtype=bool)
    lam0 = float(np.average(wc[keep], weights=amps[keep]))
    dn = _refractivity(wc) - _refractivity(lam0)
    ax, bx = np.polyfit(dn[keep], x0s[keep], 1, w=amps[keep])   # linear in refractivity
    ay, by = np.polyfit(dn[keep], y0s[keep], 1, w=amps[keep])
    return DARModel(float(ax), float(bx), float(ay), float(by), lam0, ra0=ra, dec0=dec)


def optimal_spectrum(super_rss, ra, dec, *, radius_arcsec=3.0, fwhm=None, channels=None,
                     dwave=None, fit_profile=True, dar=True) -> Tuple[Dict[str, tuple], ProfileFit]:
    """PSF/inverse-variance-weighted 1D spectrum at ``(ra, dec)``, per channel.

    ``fit_profile`` (default) fits a 2-D Gaussian to the source (centroid + widths + PA) for the
    extraction weight P; falls back to an assumed circular Gaussian if it can't. ``dar`` (default)
    tracks the centroid vs wavelength (differential atmospheric refraction / ADC residual) and
    centres P per-wavelength, so blue/red flux is not lost when the source walks with wavelength.
    Returns ``({channel: (wave, flux, var)}, ProfileFit)``; flux is in the plane's units."""
    from llamas_pyjamas.Combine.cube import _wave_grid, _resample_fibres
    chans = [c for c in ('blue', 'green', 'red')
             if c in super_rss.channels and (channels is None or c in channels)]

    model = fit_source_profile(super_rss, ra, dec, radius_arcsec=radius_arcsec,
                               channels=chans) if fit_profile else None
    if model is not None:
        g, fit = model
        cx0, cy0 = float(g.x_mean.value), float(g.y_mean.value)   # reference centre (arcsec frame)
        sigx, sigy = fit.sigma_x, fit.sigma_y
        th = np.deg2rad(fit.theta_deg)
    else:
        fw = fwhm or estimate_psf_fwhm(super_rss, ra, dec, channels=chans,
                                       radius_arcsec=radius_arcsec)
        sigx = sigy = fw / 2.35482
        cx0 = cy0 = 0.0
        th = 0.0
        fit = ProfileFit(ra=ra, dec=dec, sigma_x=sigx, sigma_y=sigy, theta_deg=0.0, fitted=False)

    # centroid track vs wavelength (DAR); constant reference centre if disabled/unavailable
    track = measure_dar(super_rss, ra, dec, radius_arcsec=radius_arcsec, channels=chans) if dar \
        else None
    if track is not None:
        lo, hi = _full_band(super_rss, chans)
        fit.dar = track
        fit.dar_shift = track.shift(lo, hi)

    ct, stheta = np.cos(th), np.sin(th)
    cosd = np.cos(np.deg2rad(dec))
    out: Dict[str, tuple] = {}
    for c in chans:
        st = super_rss.channels[c]
        dx = (st.ra - ra) * cosd * 3600.0
        dy = (st.dec - dec) * 3600.0
        sel = np.hypot(dx - cx0, dy - cy0) < radius_arcsec       # aperture around the reference centre
        if sel.sum() == 0:
            continue
        sub = _subselect(st, sel)
        wl, _dw = _wave_grid(sub, dwave, None)
        x, V, bad = _resample_fibres(sub, wl, units='flux')      # (nsel, nw) flux + variance
        if track is not None:
            cxg, cyg = track.center(wl)                           # (nw,) per-wavelength centre
        else:
            cxg, cyg = cx0, cy0                                   # constant reference centre
        ddx = dx[sel][:, None] - np.atleast_1d(cxg)[None, :]     # (nsel, nw) or broadcast scalar
        ddy = dy[sel][:, None] - np.atleast_1d(cyg)[None, :]
        xr = ddx * ct + ddy * stheta                             # rotate into the profile frame
        yr = -ddx * stheta + ddy * ct
        # P = fraction of the source's TOTAL flux falling in each fibre = normalised-PSF * fibre
        # area. Analytic (2*pi*sigx*sigy) normalisation, NOT sum-over-fibres: the latter divides by
        # the number of sampling fibre-instances, so N dithers would inflate F_hat by ~N. This makes
        # F_hat the true total flux (any exposure count) -> directly comparable to Gaia.
        area = sub.solid_angle[:, None]                          # arcsec^2 per fibre
        P = np.exp(-0.5 * ((xr / sigx) ** 2 + (yr / sigy) ** 2)) * area / (2.0 * np.pi * sigx * sigy)
        wgt = np.where(bad, 0.0, 1.0 / V)
        num = np.nansum(P * x * wgt, axis=0)
        den = np.nansum(P * P * wgt, axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            flux = np.where(den > 0, num / den, np.nan)
            var = np.where(den > 0, 1.0 / den, np.nan)
        out[c] = (wl, flux, var)
    return out, fit
