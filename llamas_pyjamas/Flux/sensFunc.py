"""
Sensitivity functions from spectrophotometric standard stars.

A sensitivity function converts an observed count rate into physical flux. For a standard whose
true flux F_ref(lambda) is known, the instrument response is

    S(lambda) = F_ref(lambda) / ( C_obs(lambda) / t_exp )

where C_obs is the extracted counts summed over the star's aperture. Applying S to a science
count rate then yields calibrated flux. This module builds S from a standard's extracted
spectrum and fits a smooth model to it; it does not apply it (that is Phase III) and it does
not correct for atmospheric extinction (also Phase III — the standard and science are generally
at different airmass).

The fit must follow the *instrument*, not the *star*, so two families of wavelength are masked
before fitting: telluric absorption bands (atmosphere, not instrument) and the standard's own
strong intrinsic lines. LLAMAS optical standards are hot subdwarfs and white dwarfs (GD108 sdB,
Feige110 DOp) with broad Balmer and He absorption that would otherwise drag the fit down. The
raw ratio confirms the need: on GD108 the blue ratio swings ~12x, almost all of it Balmer.

The fit is done in log space (S spans orders of magnitude and is far smoother logarithmically)
with pypeit's robust iterative b-spline (the same `iterfit` the flat-field and sky steps use),
so outliers left by imperfect masking are rejected rather than followed.

Classes
-------
SensChannel   The fitted sensitivity for one channel
SensFunc      A standard's sensitivity across channels, with FITS save/load

Functions
---------
default_masks       Telluric bands + hot-star line windows to exclude from the fit
build_good_mask     Boolean keep-mask for a wavelength array given exclusion regions
sensitivity_ratio   Raw S(lambda) = F_ref / (counts/s) on the observed grid
fit_sensitivity     Robust log-space b-spline fit of a raw ratio
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

#: Telluric absorption bands (Angstrom) — atmosphere, not instrument, so excluded from the fit.
#: The A and B O2 bands and the main H2O complexes across the optical/near-IR.
TELLURIC_BANDS: Tuple[Tuple[float, float], ...] = (
    (6270.0, 6330.0),    # O2 gamma
    (6860.0, 6970.0),    # O2 B-band
    (7150.0, 7350.0),    # H2O
    (7590.0, 7700.0),    # O2 A-band
    (8100.0, 8400.0),    # H2O
    (8900.0, 9900.0),    # H2O (strong red)
)

#: Strong intrinsic lines of hot subdwarf/white-dwarf standards, as (centre, half-width) in
#: Angstrom. These stars have pressure-broadened Balmer and He lines tens of A wide, so the
#: windows are generous — the goal is to fit the continuum response between them, not the lines.
STELLAR_LINES: Tuple[Tuple[float, float], ...] = (
    (6562.8, 45.0),   # H-alpha
    (4861.3, 45.0),   # H-beta
    (4340.5, 40.0),   # H-gamma
    (4101.7, 35.0),   # H-delta
    (3970.1, 30.0),   # H-epsilon
    (3889.1, 25.0),   # H-8
    (3835.4, 20.0),   # H-9
    (4685.7, 20.0),   # He II
    (4471.5, 15.0),   # He I
    (5875.6, 15.0),   # He I
    (6678.2, 15.0),   # He I
)

DEFAULT_NORD = 3          # cubic b-spline
DEFAULT_SIGMA = 3.0       # rejection threshold (both sides) in the iterative fit


def default_masks(include_telluric: bool = True,
                  include_stellar: bool = True) -> List[Tuple[float, float]]:
    """Default wavelength regions to exclude from a sensitivity fit.

    Returns a list of ``(low, high)`` bands combining telluric absorption and the broad lines
    of hot standards. Either family can be dropped; the interactive fitter adds or removes
    regions on top of this.
    """
    regions: List[Tuple[float, float]] = []
    if include_telluric:
        regions.extend(TELLURIC_BANDS)
    if include_stellar:
        regions.extend((c - hw, c + hw) for c, hw in STELLAR_LINES)
    return regions


def build_good_mask(wave: np.ndarray,
                    regions: Sequence[Tuple[float, float]]) -> np.ndarray:
    """Boolean mask, True where `wave` is outside every exclusion region (i.e. usable)."""
    good = np.ones(np.shape(wave), dtype=bool)
    for low, high in regions:
        good &= ~((wave >= low) & (wave <= high))
    return good


def sensitivity_ratio(obs_wave: np.ndarray, obs_flux: np.ndarray, exptime: float,
                      ref_wave: np.ndarray, ref_flux: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw sensitivity S = F_ref / (obs_flux / exptime), on the observed wavelength grid.

    Parameters
    ----------
    obs_wave, obs_flux : ndarray
        The standard's extracted spectrum (aperture-summed), native sampling.
    exptime : float
        Exposure time (s).
    ref_wave, ref_flux : ndarray
        Published reference spectrum (erg/s/cm^2/A).

    Returns
    -------
    wave, sens, valid : ndarray
        `sens` is S on `wave`; `valid` is True where S is finite and positive (observed rate
        above zero and inside the reference's wavelength coverage). `sens` is NaN where invalid.
    """
    if exptime <= 0:
        raise ValueError(f'exptime must be positive, got {exptime}')

    rate = np.asarray(obs_flux, dtype=float) / float(exptime)
    ref_on_obs = np.interp(obs_wave, ref_wave, ref_flux, left=np.nan, right=np.nan)
    with np.errstate(divide='ignore', invalid='ignore'):
        sens = ref_on_obs / rate
    valid = np.isfinite(sens) & (sens > 0) & (rate > 0)
    sens = np.where(valid, sens, np.nan)
    return np.asarray(obs_wave, dtype=float), sens, valid


def _auto_bkspace(wave: np.ndarray) -> float:
    """A breakpoint spacing that fits the response, not the noise: ~1/20 of the span."""
    span = float(np.nanmax(wave) - np.nanmin(wave))
    return max(span / 20.0, 50.0)


def fit_sensitivity(wave: np.ndarray, sens: np.ndarray, good: np.ndarray,
                    bkspace: Optional[float] = None, nord: int = DEFAULT_NORD,
                    sigma: float = DEFAULT_SIGMA, maxiter: int = 5
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Robust b-spline fit of a raw sensitivity ratio, in log space.

    Parameters
    ----------
    wave, sens : ndarray
        The raw ratio from :func:`sensitivity_ratio`.
    good : ndarray of bool
        Points to fit — typically ``valid & build_good_mask(...)`` so masked bands and invalid
        points are excluded.
    bkspace : float, optional
        Breakpoint spacing (A); a span-derived default is used if None.
    nord, sigma, maxiter :
        B-spline order, rejection threshold, and iteration count.

    Returns
    -------
    fit, used : ndarray
        `fit` is S evaluated on the full `wave` grid (finite everywhere within the fit domain,
        NaN outside). `used` is the boolean mask of points that survived rejection.

    Notes
    -----
    Fitting log10(S) keeps the huge dynamic range from dominating the least-squares and matches
    how sensitivity/zeropoint curves behave physically (smooth in magnitude). Uses pypeit's
    `iterfit`, the same solver the flat and sky steps use.
    """
    from pypeit.core.fitting import iterfit

    fit_pts = good & np.isfinite(sens) & (sens > 0)
    if fit_pts.sum() < max(2 * nord, 10):
        raise ValueError(f'too few usable points to fit ({int(fit_pts.sum())})')

    x = np.asarray(wave, dtype=float)[fit_pts]
    y = np.log10(np.asarray(sens, dtype=float)[fit_pts])
    order = np.argsort(x)
    x, y = x[order], y[order]

    if bkspace is None:
        bkspace = _auto_bkspace(x)

    sset, outmask = iterfit(x, y, maxiter=maxiter, nord=nord,
                            kwargs_bspline={'bkspace': bkspace},
                            upper=sigma, lower=sigma)

    fit = np.full(np.shape(wave), np.nan, dtype=float)
    domain = (wave >= x.min()) & (wave <= x.max())
    fit[domain] = 10.0 ** sset.value(np.asarray(wave, dtype=float)[domain])[0]

    used = np.zeros(np.shape(wave), dtype=bool)
    idx = np.where(fit_pts)[0][order]
    used[idx] = outmask.astype(bool)
    return fit, used


@dataclass
class SensChannel:
    """The fitted sensitivity for one channel.

    Attributes
    ----------
    channel : str
    wave : ndarray
        Wavelength grid (A) of the sampled fit.
    sens : ndarray
        Fitted S(lambda) on `wave`.
    raw : ndarray
        Raw ratio on `wave` (NaN where invalid), for inspection.
    good : ndarray of bool
        Points that entered the fit (after masking and rejection).
    """
    channel: str
    wave: np.ndarray
    sens: np.ndarray
    raw: np.ndarray
    good: np.ndarray

    def value(self, wave: np.ndarray) -> np.ndarray:
        """Evaluate S at arbitrary wavelengths by interpolating the sampled fit."""
        return np.interp(wave, self.wave, self.sens, left=np.nan, right=np.nan)


@dataclass
class SensFunc:
    """A standard star's sensitivity function across one or more channels.

    Attributes
    ----------
    channels : dict
        channel -> SensChannel.
    meta : dict
        Provenance (standard name, exptime, airmass, aperture size, source files, ...).
    """
    channels: Dict[str, SensChannel] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    def value(self, wave: np.ndarray, channel: str) -> np.ndarray:
        return self.channels[channel].value(wave)

    def save(self, path: str) -> str:
        """Write the sensitivity function to a FITS file.

        Layout: a provenance-only primary header, then one binary table per channel
        (``SENS_<CHANNEL>``) holding the sampled fit, the raw ratio and the fit mask. The
        sampled curve is stored densely and applied by interpolation, so the file is portable
        and does not depend on serialising b-spline internals.
        """
        primary = fits.PrimaryHDU()
        primary.header['HISTORY'] = 'LLAMAS sensitivity function'
        for key, val in self.meta.items():
            card = str(key).upper()[:8]
            try:
                primary.header[card] = val
            except (ValueError, TypeError):
                primary.header[card] = str(val)

        hdus = [primary]
        for name, ch in self.channels.items():
            cols = [
                fits.Column(name='WAVE', format='D', unit='Angstrom', array=ch.wave),
                fits.Column(name='SENS', format='D', array=ch.sens),
                fits.Column(name='RAW', format='D', array=ch.raw),
                fits.Column(name='GOOD', format='L', array=ch.good),
            ]
            hdu = fits.BinTableHDU.from_columns(cols, name=f'SENS_{name.upper()}')
            hdu.header['CHANNEL'] = name
            hdus.append(hdu)

        fits.HDUList(hdus).writeto(path, overwrite=True)
        logger.info('Wrote sensitivity function (%s) to %s',
                    ','.join(self.channels), path)
        return path

    @classmethod
    def load(cls, path: str) -> 'SensFunc':
        channels: Dict[str, SensChannel] = {}
        meta: Dict[str, object] = {}
        with fits.open(path) as hdul:
            for card in hdul[0].header:
                if card not in ('SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND', 'HISTORY'):
                    meta[card.lower()] = hdul[0].header[card]
            for hdu in hdul[1:]:
                name = hdu.header.get('CHANNEL', hdu.name.replace('SENS_', '').lower())
                d = hdu.data
                channels[name] = SensChannel(
                    channel=name,
                    wave=np.asarray(d['WAVE'], dtype=float),
                    sens=np.asarray(d['SENS'], dtype=float),
                    raw=np.asarray(d['RAW'], dtype=float),
                    good=np.asarray(d['GOOD'], dtype=bool),
                )
        return cls(channels=channels, meta=meta)


def build_sensfunc(spectra_by_channel: Dict[str, Tuple[np.ndarray, np.ndarray]],
                   exptime: float, ref_wave: np.ndarray, ref_flux: np.ndarray,
                   regions: Optional[Sequence[Tuple[float, float]]] = None,
                   bkspace: Optional[float] = None, nord: int = DEFAULT_NORD,
                   sigma: float = DEFAULT_SIGMA, airmass: Optional[float] = None,
                   meta: Optional[Dict] = None) -> SensFunc:
    """Build a :class:`SensFunc` from a standard's aperture spectra and reference flux.

    Parameters
    ----------
    spectra_by_channel : dict
        channel -> (wave, flux) aperture-summed observed spectrum.
    exptime : float
        Exposure time (s).
    ref_wave, ref_flux : ndarray
        Reference spectrum (erg/s/cm^2/A), e.g. from ``Standard.load_spectrum()``.
    regions : sequence of (low, high), optional
        Wavelength bands to exclude; :func:`default_masks` if None.
    bkspace, nord, sigma :
        Fit controls (see :func:`fit_sensitivity`).
    meta : dict, optional
        Provenance recorded in the output.

    This is the "let it rip" auto path; the interactive fitter calls the same pieces with
    user-edited `regions` and controls.
    """
    if regions is None:
        regions = default_masks()

    channels: Dict[str, SensChannel] = {}
    for name, (obs_wave, obs_flux) in spectra_by_channel.items():
        wave, sens, valid = sensitivity_ratio(obs_wave, obs_flux, exptime, ref_wave, ref_flux)
        good_in = valid & build_good_mask(wave, regions)
        if good_in.sum() < max(2 * nord, 10):
            logger.warning('channel %s: too few usable points (%d), skipped',
                           name, int(good_in.sum()))
            continue
        fit, used = fit_sensitivity(wave, sens, good_in, bkspace=bkspace, nord=nord, sigma=sigma)
        channels[name] = SensChannel(channel=name, wave=wave, sens=fit, raw=sens, good=used)

    if not channels:
        raise ValueError('no channel produced a sensitivity fit')

    # airmass is stored so the apply step can do the differential extinction correction
    # (X_science - X_standard); without it, only same-airmass application is exact.
    full_meta = {'exptime': exptime}
    if airmass is not None:
        full_meta['airmass'] = float(airmass)
    full_meta.update(meta or {})
    return SensFunc(channels=channels, meta=full_meta)
