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


def scale_sky_per_fiber(flux: np.ndarray, sky: np.ndarray, config,
                        sky_mask: np.ndarray = None):
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

    n_scaled = 0
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

        denom = float(np.sum(s_line[good] ** 2))
        if denom <= 0:
            continue
        alpha = float(np.sum(d[good] * s_line[good]) / denom)

        # Effective scale = 1 + alpha, clipped; recover clipped alpha.
        eff = np.clip(1.0 + alpha, config.scale_min, config.scale_max)
        alpha = eff - 1.0
        scale[f] = eff

        corr = (alpha * s_line).astype(np.float32)
        correction[f] = corr
        n_scaled += 1

    flux_scaled = flux - correction
    logger.info("skyScale: refined OH residual on %d/%d fibres "
                "(median scale=%.3f)", n_scaled, n_fiber, float(np.median(scale)))
    return scale, correction, flux_scaled
