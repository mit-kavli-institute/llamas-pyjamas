"""Unbiased totals and integrals over data with masked or missing samples.

A masked sample must never count as zero in a total. ``np.nansum`` (or ``nan_to_num(..., 0)``
followed by a sum) does exactly that, so a total over a region where some samples are masked —
a fibre's zero-filled channel end, one aperture fibre's wavelength range ending early, a cosmic
ray — comes out low there. In LLAMAS the per-fibre wavelength ranges differ by ~100 A at the
channel ends, so such partial totals are everywhere; fed into a sensitivity function they make it
over-correct, and fed into per-exposure photometry they scale exposures up.

Three rules, chosen by what the quantity is:

* :func:`complete_sum` — a TOTAL over members (fibres in an aperture, spaxels, reference-source
  fibres). Valid only where every member is valid; never renormalised, because a point source's
  flux is not spread evenly over the members, so rescaling by the valid weight is biased too.
* :func:`band_integral` — an INTEGRAL over a wavelength window per spectrum. ``mean(valid) * N``
  within the window clipped to the spectra's common coverage, NaN below a minimum valid fraction.
* :func:`renormalised_mean` — a MEAN over members: ``sum_valid(w x) / sum_valid(w)``.

numpy only, so any module can import it.
"""

from collections import namedtuple

import numpy as np

#: Result of :func:`band_integral`, per row: the integral, its variance (``None`` without ``var``),
#: the valid fraction n/N and the number of valid samples n.
BandIntegral = namedtuple('BandIntegral', 'value var frac n')

#: Default minimum fraction of a window's samples that must be valid for a band integral.
MIN_VALID_FRAC = 0.5


def common_coverage(wave, good, lo_pct=2.0, hi_pct=98.0):
    """Robust common wavelength range of a set of spectra.

    ``(start, end)`` where ``start`` is the ``hi_pct`` percentile of the spectra's first good
    wavelengths and ``end`` the ``lo_pct`` percentile of their last good wavelengths, so a few
    stray spectra cannot shrink the range. Spectra with no good samples are ignored; with none at
    all the result is ``(nan, nan)``.
    """
    wave = np.atleast_2d(np.asarray(wave, dtype=float))
    good = np.atleast_2d(np.asarray(good, dtype=bool)) & np.isfinite(wave)
    has = good.any(axis=1)
    if not has.any():
        return float('nan'), float('nan')
    w_lo = np.where(good, wave, np.inf).min(axis=1)[has]
    w_hi = np.where(good, wave, -np.inf).max(axis=1)[has]
    return float(np.percentile(w_lo, hi_pct)), float(np.percentile(w_hi, lo_pct))


def band_integral(flux, wave, bad, lo, hi, *, var=None, common=None, min_frac=MIN_VALID_FRAC):
    """Integrate each spectrum (row) over ``[lo, hi]`` as ``mean(valid samples) * N``.

    ``N`` counts the window's samples (finite wavelength inside ``[lo, hi]``, intersected with
    ``common`` when given — pass :func:`common_coverage` so the falling-throughput channel ends
    are not extrapolated into); ``n`` counts the valid ones among them (not ``bad``, finite flux).
    The result is the sum-equivalent of the full window, unbiased by the masked samples.

    Returns a :data:`BandIntegral` ``(value, var, frac, n)`` of per-row arrays: ``var`` is
    ``(N/n)^2 * sum(var_valid)`` (``None`` if ``var`` is not given), ``frac = n/N`` and ``n`` the
    valid count. Rows with ``N == 0`` or ``frac < min_frac`` are NaN.
    """
    flux = np.atleast_2d(np.asarray(flux, dtype=float))
    wave = np.broadcast_to(np.atleast_2d(np.asarray(wave, dtype=float)), flux.shape)
    bad = np.broadcast_to(np.atleast_2d(np.asarray(bad, dtype=bool)), flux.shape)
    lo2, hi2 = float(lo), float(hi)
    if common is not None and np.all(np.isfinite(common)):
        lo2, hi2 = max(lo2, float(common[0])), min(hi2, float(common[1]))
    with np.errstate(invalid='ignore'):
        inwin = np.isfinite(wave) & (wave >= lo2) & (wave <= hi2)
    valid = inwin & ~bad & np.isfinite(flux)
    big_n = inwin.sum(axis=1).astype(float)
    n = valid.sum(axis=1).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.where(big_n > 0, n / big_n, 0.0)
        ok = (big_n > 0) & (n > 0) & (frac >= min_frac)
        scale = np.where(ok, big_n / np.where(n > 0, n, 1.0), np.nan)
        value = np.where(valid, flux, 0.0).sum(axis=1) * scale
        variance = None
        if var is not None:
            var = np.broadcast_to(np.atleast_2d(np.asarray(var, dtype=float)), flux.shape)
            vsum = np.where(valid & np.isfinite(var), var, 0.0).sum(axis=1)
            variance = vsum * scale ** 2
    return BandIntegral(value, variance, frac, n.astype(int))


def complete_sum(values, bad, weights=None):
    """Weighted total over members (axis 0), valid only where EVERY member is valid.

    ``values`` and ``bad`` are ``(n_member, n_sample)``; a non-finite value counts as bad.
    Returns ``(total, mask)`` where ``mask`` is True (and ``total`` NaN) wherever any member is
    bad. Never renormalised by the valid weight — see the module docstring.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    bad = np.atleast_2d(np.asarray(bad, dtype=bool)) | ~np.isfinite(values)
    w = np.ones(values.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    mask = bad.any(axis=0)
    total = np.where(bad, 0.0, values * w[:, None]).sum(axis=0)
    return np.where(mask, np.nan, total), mask


def renormalised_mean(values, bad, weights=None):
    """Weighted mean over members (axis 0) using only valid members at each sample.

    Returns ``(mean, mask)``; ``mask`` is True (and ``mean`` NaN) where no member is valid.
    """
    values = np.atleast_2d(np.asarray(values, dtype=float))
    bad = np.atleast_2d(np.asarray(bad, dtype=bool)) | ~np.isfinite(values)
    w = np.ones(values.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    wv = np.where(bad, 0.0, w[:, None] * np.ones_like(values))
    den = wv.sum(axis=0)
    num = np.where(bad, 0.0, values * wv).sum(axis=0)
    mask = den <= 0
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(mask, np.nan, num / np.where(mask, 1.0, den))
    return mean, mask


def resample_bad(ref_wave, wave, bad):
    """Badness of samples at ``wave`` carried onto ``ref_wave`` (linear-interpolation footprint).

    A reference sample is bad when either neighbouring source sample is bad, or when it falls
    outside the source's finite wavelength range. Use alongside ``np.interp`` of the values.
    """
    wave = np.asarray(wave, dtype=float)
    bad = np.asarray(bad, dtype=bool) | ~np.isfinite(wave)
    fin = np.isfinite(wave)
    if not fin.any():
        return np.ones(np.shape(ref_wave), dtype=bool)
    b = np.interp(ref_wave, wave[fin], bad[fin].astype(float), left=1.0, right=1.0)
    return b > 0


def fibre_band_flux(flux, wave, bad, lo, hi, *, min_frac=MIN_VALID_FRAC):
    """Per-fibre band flux over ``[lo, hi]``: :func:`band_integral` within the fibres' common
    coverage. The one implementation behind the viewer's fibre totals, registration's source
    detection and the RSS white light, so they agree. NaN where a fibre has too little data."""
    flux = np.atleast_2d(np.asarray(flux, dtype=float))
    bad = np.atleast_2d(np.asarray(bad, dtype=bool))
    common = common_coverage(wave, ~bad & np.isfinite(flux))
    return band_integral(flux, wave, bad, lo, hi, common=common, min_frac=min_frac).value
