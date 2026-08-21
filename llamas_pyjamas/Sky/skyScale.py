"""
llamas_pyjamas.Sky.skyScale
===========================
Per-fibre OH-line scaling (Kelson-style scaled sky subtraction), applied in the
fibre-flat-corrected FLUX frame.

The base model already subtracted a sky estimate (FF ``FLUX`` =
``(COUNTS - SKY)/C_i``).  Fibre-to-fibre LSF and throughput differences leave a
small *residual* OH-line component in each fibre that is proportional to the
sky line shape.  We measure that residual coefficient ``alpha`` per fibre by
least-squares against the (continuum-subtracted) base ``SKY`` template inside OH
line windows, then remove ``alpha * line(SKY)`` — touching only the line pixels,
never the continuum.

Public API
----------
scale_sky_per_fiber(flux, sky, config, sky_mask=None)
    -> (scale, correction, flux_scaled)
"""

import logging
import numpy as np
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)


def _continuum(spec_1d: np.ndarray, window: int) -> np.ndarray:
    """Rolling-median continuum estimate."""
    size = max(3, 2 * int(window) + 1)
    return median_filter(np.nan_to_num(spec_1d, nan=0.0, posinf=0.0, neginf=0.0),
                         size=size, mode="nearest")


def _line_mask(line_resid: np.ndarray, sigdetect: float,
               peak_floor_frac: float = 0.02) -> np.ndarray:
    """Boolean mask of OH-line pixels in a continuum-subtracted spectrum.

    Threshold = ``max(sigdetect * robust_sigma, peak_floor_frac * peak)``.  The
    MAD term dominates for noisy real spectra; the peak-fraction floor keeps the
    detector well-posed for a near-noise-free sky *model* (where MAD -> 0).
    """
    peak = np.nanmax(line_resid)
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(line_resid, dtype=bool)
    med = np.nanmedian(line_resid)
    mad = np.nanmedian(np.abs(line_resid - med))
    sigma = 1.4826 * mad
    thresh = max(sigdetect * sigma, peak_floor_frac * peak)
    return line_resid > thresh


def _count_lines(mask: np.ndarray) -> int:
    """Number of contiguous True runs (distinct OH lines)."""
    if not mask.any():
        return 0
    return int(np.sum(np.diff(mask.astype(int)) == 1) + (1 if mask[0] else 0))


def _fit_line_model(s_line, s1, s2, d, good, order, config):
    """Fit ``d ~ alpha*S + beta*S' + gamma*S''`` over ``good`` line pixels.

    ``S`` is the line-only sky template and ``S'``/``S''`` its pixel-space
    derivatives.  ``alpha`` is the classic amplitude residual (clipped via the
    effective scale ``1+alpha``); ``beta``/``gamma`` absorb, respectively, an
    along-fibre wavelength-shift (antisymmetric "P-Cygni" residual) and an
    LSF-width mismatch (symmetric residual).  The derivative terms are damped
    (column-relative ridge) and only kept when they reduce the line-pixel
    residual sum-of-squares by at least ``config.scale_deriv_gate`` — so a fibre
    whose base model is already good keeps the plain amplitude fit.

    Returns ``(eff_scale, correction_full, used_deriv)`` where ``correction_full``
    is evaluated over the whole spectrum (``S``, ``S'``, ``S''`` all ~0 off-line,
    so the correction stays line-localised).  ``eff_scale`` is ``1+alpha``.
    """
    S = s_line[good]
    d_good = d[good]
    denom = float(np.dot(S, S))
    if denom <= 0:
        return None

    # --- amplitude-only baseline (classic scaled sky) ---
    alpha0 = float(np.dot(d_good, S) / denom)
    eff0 = float(np.clip(1.0 + alpha0, config.scale_min, config.scale_max))
    alpha0 = eff0 - 1.0
    ssr0 = float(np.sum((d_good - alpha0 * S) ** 2))

    if order < 1:
        return eff0, alpha0 * s_line, False

    # --- derivative-augmented fit ---
    cols = [S, s1[good]]
    if order >= 2:
        cols.append(s2[good])
    A = np.column_stack(cols)                      # (Ngood, k)
    M = A.T @ A
    b = A.T @ d_good
    ridge = max(0.0, float(config.scale_deriv_ridge))
    for k in range(1, A.shape[1]):                 # damp derivative cols only
        M[k, k] += ridge * M[k, k]
    try:
        coef = np.linalg.solve(M, b)
    except np.linalg.LinAlgError:
        coef = np.linalg.lstsq(A, d_good, rcond=None)[0]

    # Clip the amplitude component just like the baseline; keep deriv coeffs.
    eff = float(np.clip(1.0 + coef[0], config.scale_min, config.scale_max))
    coef = coef.copy()
    coef[0] = eff - 1.0

    ssr_full = float(np.sum((d_good - A @ coef) ** 2))
    improve = (ssr0 - ssr_full) / ssr0 if ssr0 > 0 else 0.0
    if improve < config.scale_deriv_gate:
        return eff0, alpha0 * s_line, False        # deriv terms not worth it

    corr = coef[0] * s_line + coef[1] * s1
    if order >= 2:
        corr = corr + coef[2] * s2
    return eff, corr, True


def scale_sky_per_fiber(flux: np.ndarray, sky: np.ndarray, config,
                        sky_mask: np.ndarray = None, color: str = None):
    """Refine the per-fibre OH-line residual and remove it from ``flux``.

    Parameters
    ----------
    flux : np.ndarray
        ``(n_fiber, n_wave)`` fibre-flat-corrected, base-sky-subtracted flux
        (the FF ``FLUX`` extension).
    sky : np.ndarray
        ``(n_fiber, n_wave)`` base sky model (FF ``SKY`` extension) used as the
        per-fibre OH line-shape template.
    config : SkySubtractConfig
    sky_mask : np.ndarray, optional
        Unused for fitting (each fibre is scaled against its own template) but
        accepted for interface symmetry / future use.
    color : str, optional
        Camera colour ('blue'/'green'/'red').  When it is listed in
        ``config.scale_deriv_skip_colors`` the derivative augmentation is
        disabled for this file and the classic amplitude-only fit is used.

    Returns
    -------
    scale : np.ndarray
        ``(n_fiber,)`` effective OH-line scale (``1 + alpha``), clipped to
        ``[scale_min, scale_max]``.  ``1.0`` where the fibre was left unchanged.
    correction : np.ndarray
        ``(n_fiber, n_wave)`` line-only correction removed from ``flux``.
    flux_scaled : np.ndarray
        ``flux - correction``.
    """
    n_fiber, n_wave = flux.shape
    win = config.scale_window_pix
    scale = np.ones(n_fiber, dtype=float)
    correction = np.zeros((n_fiber, n_wave), dtype=np.float32)

    skip = str(color).lower() in getattr(config, "scale_deriv_skip_colors", [])
    order = 0 if skip else int(getattr(config, "scale_deriv_order", 0))

    n_scaled = 0
    n_deriv = 0
    for f in range(n_fiber):
        s_full = sky[f]
        if not np.any(np.isfinite(s_full)) or np.nansum(np.abs(s_full)) == 0:
            continue  # no model for this fibre -> leave flux as-is

        s_cont = _continuum(s_full, win)
        s_line = np.nan_to_num(s_full, nan=0.0) - s_cont      # line-only template
        lines = _line_mask(s_line, config.oh_sigdetect)
        if _count_lines(lines) < config.min_oh_lines:
            continue

        f_full = flux[f]
        f_cont = _continuum(f_full, win)
        d = np.nan_to_num(f_full, nan=0.0) - f_cont           # line-only data
        good = lines & np.isfinite(flux[f])

        # Pixel-space derivatives of the line template (shift/width basis).
        s1 = np.gradient(s_line) if order >= 1 else None
        s2 = np.gradient(s1) if order >= 2 else None

        result = _fit_line_model(s_line, s1, s2, d, good, order, config)
        if result is None:
            continue
        eff, corr, used_deriv = result

        scale[f] = eff
        correction[f] = corr.astype(np.float32)
        n_scaled += 1
        n_deriv += int(used_deriv)

    flux_scaled = flux - correction
    logger.info("skyScale: refined OH residual on %d/%d fibres "
                "(median scale=%.3f, deriv order=%d applied on %d fibres)",
                n_scaled, n_fiber, float(np.median(scale)), order, n_deriv)
    return scale, correction, flux_scaled
