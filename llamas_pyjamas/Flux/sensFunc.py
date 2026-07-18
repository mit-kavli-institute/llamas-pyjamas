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
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

from llamas_pyjamas.config import LUT_DIR

logger = logging.getLogger(__name__)

#: Bundled b-spline refine regions. The VPH grating blaze features are a fixed instrument
#: property (no moving parts), so the extra knots that capture them are established once and
#: shipped here; each night only re-solves the throughput on this fixed knot structure.
BREAKPOINTS_PATH = os.path.join(LUT_DIR, 'sensfunc_breakpoints.dat')

#: Inside a refine region, knots are placed this many times denser than the base spacing.
DEFAULT_REFINE_FACTOR = 4

_REFINE_CACHE = None


def load_refine_regions(path: str = BREAKPOINTS_PATH, use_cache: bool = True):
    """Bundled refine regions as a list of ``(low, high)`` wavelength bands (empty if none).

    These are instrument-level (the blaze never moves), so the same set is used for every
    night. Missing file is not an error — it just means no localised refinement.
    """
    global _REFINE_CACHE
    if use_cache and _REFINE_CACHE is not None and path == BREAKPOINTS_PATH:
        return list(_REFINE_CACHE)
    regions: List[Tuple[float, float]] = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    regions.append((float(parts[0]), float(parts[1])))
    if path == BREAKPOINTS_PATH:
        _REFINE_CACHE = list(regions)
    return regions


def save_refine_regions(regions: Sequence[Tuple[float, float]],
                        path: str = BREAKPOINTS_PATH) -> str:
    """Write refine regions to disk (default: the bundled instrument file) and refresh cache."""
    global _REFINE_CACHE
    with open(path, 'w') as fh:
        fh.write('# Sensitivity-function b-spline refine regions.\n')
        fh.write('# Finer knots for the fixed VPH-grating blaze features -- an instrument\n')
        fh.write('# property (no moving parts): establish once, reuse every night.\n')
        fh.write(f'# refine_factor = {DEFAULT_REFINE_FACTOR}\n')
        fh.write('# {:>9s} {:>9s}\n'.format('lo_wave', 'hi_wave'))
        for lo, hi in sorted(regions):
            fh.write(f'{lo:11.3f} {hi:11.3f}\n')
    if path == BREAKPOINTS_PATH:
        _REFINE_CACHE = [tuple(r) for r in regions]
    logger.info('Wrote %d refine region(s) to %s', len(regions), path)
    return path


def build_breakpoints(wmin: float, wmax: float, base_bkspace: float,
                      refine_regions: Sequence[Tuple[float, float]],
                      refine_factor: int = DEFAULT_REFINE_FACTOR) -> Optional[np.ndarray]:
    """Interior b-spline knots: uniform at `base_bkspace`, denser inside `refine_regions`.

    Returns the sorted interior knot vector for pypeit's ``iterfit(bkpt=...)``, or None if no
    refinement is requested (caller then falls back to uniform ``bkspace``).
    """
    if not refine_regions:
        return None
    knots = list(np.arange(wmin + base_bkspace, wmax, base_bkspace))
    fine = base_bkspace / max(1, refine_factor)
    for lo, hi in refine_regions:
        lo2, hi2 = max(lo, wmin), min(hi, wmax)
        if hi2 > lo2:
            knots.extend(np.arange(lo2, hi2, fine))
    if not knots:
        return None
    return np.unique(np.round(np.sort(np.asarray(knots, dtype=float)), 3))

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


def default_masks(include_telluric: bool = False,
                  include_stellar: bool = True) -> List[Tuple[float, float]]:
    """Default wavelength regions to hard-exclude from a sensitivity fit.

    Returns a list of ``(low, high)`` bands. Stellar lines (broad Balmer/He of the hot
    standards) are included by default — they are deep dips on a bright continuum, so S/N
    weighting will not down-weight them and they must be hard-masked. Tellurics are **off** by
    default: hard-masking a terminal telluric band truncates the sensfunc's red coverage, and
    the S/N weighting already down-weights absorbed (low-count) regions, so they are handled
    without a hard mask. Pass ``include_telluric=True`` to hard-mask them anyway.
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
                    weight: Optional[np.ndarray] = None, bkspace: Optional[float] = None,
                    nord: int = DEFAULT_NORD, sigma: float = DEFAULT_SIGMA, maxiter: int = 5,
                    refine_regions: Optional[Sequence[Tuple[float, float]]] = None,
                    refine_factor: int = DEFAULT_REFINE_FACTOR
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Robust, S/N-weighted b-spline fit of a raw sensitivity ratio, in log space.

    Parameters
    ----------
    wave, sens : ndarray
        The raw ratio from :func:`sensitivity_ratio`.
    good : ndarray of bool
        Points to fit — typically ``valid & build_good_mask(...) & above throughput floor``.
    weight : ndarray, optional
        Per-point weight (inverse variance) for the fit. Passing the observed counts here
        down-weights the low-throughput channel edges — the dominant source of edge
        instability, since there ``S = F_ref/(counts/s)`` blows up on almost no signal. Equal
        weights if None.
    bkspace : float, optional
        Breakpoint spacing (A); a span-derived default is used if None.
    nord, sigma, maxiter :
        B-spline order, rejection threshold, and iteration count.

    Returns
    -------
    fit, used : ndarray
        `fit` is S evaluated across the full span of the fit points on the `wave` grid (NaN
        outside that span). `used` is the boolean mask of points that survived rejection.

    Notes
    -----
    Fitting log10(S) keeps the huge dynamic range from dominating the least-squares and matches
    how sensitivity/zeropoint curves behave physically (smooth in magnitude). The counts-based
    weighting means the well-exposed middle of each channel drives the shape and the noisy
    dichroic-rolloff edges follow rather than lead. Uses pypeit's `iterfit`, the same solver the
    flat and sky steps use.
    """
    from pypeit.core.fitting import iterfit

    fit_pts = good & np.isfinite(sens) & (sens > 0)
    if fit_pts.sum() < max(2 * nord, 10):
        raise ValueError(f'too few usable points to fit ({int(fit_pts.sum())})')

    x = np.asarray(wave, dtype=float)[fit_pts]
    y = np.log10(np.asarray(sens, dtype=float)[fit_pts])
    order = np.argsort(x)
    x, y = x[order], y[order]

    invvar = None
    if weight is not None:
        w = np.asarray(weight, dtype=float)[fit_pts][order]
        invvar = np.clip(w, 0.0, None)
        if not np.any(invvar > 0):
            invvar = None

    if bkspace is None:
        bkspace = _auto_bkspace(x)

    kwargs = dict(maxiter=maxiter, nord=nord, upper=sigma, lower=sigma)
    if invvar is not None:
        kwargs['invvar'] = invvar
    # Localised refinement: explicit non-uniform interior knots (base spacing + denser inside
    # the fixed blaze regions). Otherwise uniform spacing.
    bkpt = build_breakpoints(float(x.min()), float(x.max()), bkspace,
                             refine_regions or [], refine_factor)
    if bkpt is not None and bkpt.size:
        kwargs['bkpt'] = bkpt
    else:
        kwargs['kwargs_bspline'] = {'bkspace': bkspace}
    sset, outmask = iterfit(x, y, **kwargs)

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


#: Points fainter than this fraction of a channel's peak counts are the dichroic-rolloff
#: edges where S = F_ref/(counts/s) is dominated by noise. Dropped from the fit by default.
DEFAULT_THROUGHPUT_FLOOR = 0.05


def fit_channel_sens(obs_wave: np.ndarray, obs_flux: np.ndarray, exptime: float,
                     ref_wave: np.ndarray, ref_flux: np.ndarray,
                     regions: Sequence[Tuple[float, float]],
                     bkspace: Optional[float] = None, nord: int = DEFAULT_NORD,
                     sigma: float = DEFAULT_SIGMA,
                     throughput_floor: float = DEFAULT_THROUGHPUT_FLOOR,
                     weighted: bool = True,
                     refine_regions: Optional[Sequence[Tuple[float, float]]] = None):
    """One channel's sensitivity: raw ratio, throughput floor, S/N-weighted b-spline fit.

    Shared by :func:`build_sensfunc` and the interactive fitter so the auto and live paths
    are identical. `refine_regions` add denser knots for the fixed blaze features. Returns
    ``(wave, raw, fit, used)`` or None if too few usable points.
    """
    wave, sens, valid = sensitivity_ratio(obs_wave, obs_flux, exptime, ref_wave, ref_flux)
    counts = np.asarray(obs_flux, dtype=float)

    good_in = valid & build_good_mask(wave, regions)
    if throughput_floor > 0:
        finite = counts[np.isfinite(counts)]
        if finite.size:
            good_in = good_in & (counts >= throughput_floor * np.nanmax(finite))
    if good_in.sum() < max(2 * nord, 10):
        return None

    fit, used = fit_sensitivity(wave, sens, good_in,
                                weight=(counts if weighted else None),
                                bkspace=bkspace, nord=nord, sigma=sigma,
                                refine_regions=refine_regions)
    return wave, sens, fit, used


def build_sensfunc(spectra_by_channel: Dict[str, Tuple[np.ndarray, np.ndarray]],
                   exptime: float, ref_wave: np.ndarray, ref_flux: np.ndarray,
                   regions: Optional[Sequence[Tuple[float, float]]] = None,
                   bkspace: Optional[float] = None, nord: int = DEFAULT_NORD,
                   sigma: float = DEFAULT_SIGMA, airmass: Optional[float] = None,
                   throughput_floor: float = DEFAULT_THROUGHPUT_FLOOR,
                   weighted: bool = True,
                   refine_regions: Optional[Sequence[Tuple[float, float]]] = None,
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
        Wavelength bands to exclude; :func:`default_masks` (stellar lines) if None. Tellurics
        are handled by the S/N weighting rather than hard-masked, so the fit keeps its red
        coverage instead of truncating at the first telluric band.
    bkspace, nord, sigma :
        Fit controls (see :func:`fit_sensitivity`).
    throughput_floor : float
        Fraction of a channel's peak counts below which points are dropped — the low-signal
        dichroic-rolloff edges. 0 disables.
    weighted : bool
        S/N-weight the fit by the observed counts (recommended; stabilises the edges).
    meta : dict, optional
        Provenance recorded in the output.

    This is the "let it rip" auto path; the interactive fitter calls the same pieces with
    user-edited `regions` and controls.
    """
    if regions is None:
        regions = default_masks()
    if refine_regions is None:
        refine_regions = load_refine_regions()      # bundled instrument blaze knots

    channels: Dict[str, SensChannel] = {}
    for name, (obs_wave, obs_flux) in spectra_by_channel.items():
        result = fit_channel_sens(obs_wave, obs_flux, exptime, ref_wave, ref_flux, regions,
                                  bkspace=bkspace, nord=nord, sigma=sigma,
                                  throughput_floor=throughput_floor, weighted=weighted,
                                  refine_regions=refine_regions)
        if result is None:
            logger.warning('channel %s: too few usable points, skipped', name)
            continue
        wave, sens, fit, used = result
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
