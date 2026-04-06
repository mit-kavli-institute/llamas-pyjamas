"""2D scattered light correction for LLAMAS IFU detectors.

Uses inter-fibre gap pixels (TraceLlamas.fiberimg == -1) as control points
for a 2D polynomial surface fit that is subtracted from the detector image
before spectral extraction.  This is the standard approach used in MUSE,
KCWI, and LRIS IFU pipelines.
"""

import numpy as np
import logging
from typing import Optional, Tuple
from pypeit.core import fitting as pypeit_fitting

logger = logging.getLogger(__name__)

_DEFAULT_ORDER = (3, 3)   # Cubic in both dispersion and cross-dispersion directions
_MIN_GAP_PIXELS = 500     # Minimum gap pixels required to attempt a fit
_SIGMA_CLIP = 4.0         # Clip threshold applied to model residuals (not raw gap data)
_MAX_ITERS = 3            # Number of iterative fitting passes for CR rejection


def subtract_scattered_light(
    data: np.ndarray,
    tracer,
    order: Tuple[int, int] = _DEFAULT_ORDER,
    image_type: str = 'science',
    min_gap_pixels: int = _MIN_GAP_PIXELS,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Estimate and subtract a 2D polynomial scattered light model.

    Uses inter-fibre gap pixels (``tracer.fiberimg == -1``) as control points.
    Cosmic ray rejection is performed **iteratively against the fitted model**,
    not against the raw gap pixel distribution.  The latter approach is
    incorrect because the spatial gradient of scattered light dominates the
    global standard deviation, making edge-of-detector cosmic rays invisible
    to a global clip.

    Proactive boundary padding is applied before fitting: if the gap pixel
    coverage does not reach within ``pad_margin`` pixels of the detector edges
    in the cross-dispersion direction, the nearest gap row is projected onto
    the edge.  This prevents cubic Runge oscillations in the extrapolated
    region (typically the fibre-free top/bottom ~20–50 rows).

    Parameters
    ----------
    data : np.ndarray
        Bias-subtracted 2D detector image, shape ``(naxis2, naxis1)``.
    tracer : TraceLlamas
        Loaded trace object with a valid ``fiberimg`` attribute.
    order : (int, int)
        Polynomial order ``(x_order, y_order)``.  Default ``(3, 3)``.
    image_type : str
        Passed to ``build_interfibre_mask``.  Use ``'lamp_flat'`` for flat
        frames (triggers 1-pixel fibre-mask erosion, widening the gap region
        away from fibre wings).  Use ``'science'`` for science and arc frames.
    min_gap_pixels : int
        Minimum number of gap pixels required to attempt a fit.  If fewer are
        available (before or after clipping), the function returns the original
        data unchanged.

    Returns
    -------
    corrected : np.ndarray
        Scattered-light-subtracted detector image.
    model : np.ndarray or None
        The 2D polynomial model that was subtracted.  ``None`` if the
        correction was skipped (logged as a warning).
    """
    from llamas_pyjamas.Bias.biasChecking import build_interfibre_mask

    if tracer is None or getattr(tracer, 'fiberimg', None) is None:
        logger.warning("subtract_scattered_light: no valid tracer.fiberimg — skipping")
        return data, None

    # --- identify gap pixels ---
    gap_mask = build_interfibre_mask(tracer, data.shape, image_type=image_type)
    n_gap = int(gap_mask.sum())

    if n_gap < min_gap_pixels:
        logger.warning(
            f"subtract_scattered_light: only {n_gap} gap pixels (< {min_gap_pixels}) — skipping"
        )
        return data, None

    # --- coordinate grids ---
    # Use np.meshgrid (C-optimised) rather than np.outer; float32 saves memory
    # in Ray worker processes (each worker handles one 2048×2048 detector).
    naxis2, naxis1 = data.shape
    x_vec = np.arange(naxis1, dtype=np.float32)
    y_vec = np.arange(naxis2, dtype=np.float32)
    ximg, yimg = np.meshgrid(x_vec, y_vec)

    xgap = ximg[gap_mask]
    ygap = yimg[gap_mask]
    dgap = data[gap_mask].astype(np.float64)

    # --- initial outlier clip ---
    # Remove only extreme outliers via the 99.5th percentile.  DO NOT use a
    # global median ± k*std here: the spatial gradient dominates std and would
    # fail to flag cosmic rays near the detector edges.
    upper_limit = np.nanpercentile(dgap, 99.5)
    good = dgap < upper_limit

    # --- proactive boundary padding ---
    # A cubic polynomial extrapolating over a large unconstrained region (the
    # ~20–50 fibre-free rows at the detector top/bottom) is mathematically
    # guaranteed to produce Runge oscillations.  Proactively replicate the
    # nearest gap-row values onto the detector edges so the polynomial is
    # constrained to flatten there rather than curl.
    pad_margin = 10  # rows: pad if gap coverage misses the edge by more than this
    if float(ygap.min()) > pad_margin:
        bottom_mask = ygap < (ygap.min() + 5)
        xgap = np.concatenate([xgap, xgap[bottom_mask]])
        ygap = np.concatenate([ygap, np.zeros(int(bottom_mask.sum()), dtype=np.float32)])
        dgap = np.concatenate([dgap, dgap[bottom_mask]])
        good = np.concatenate([good, good[bottom_mask]])
    if float(ygap.max()) < naxis2 - pad_margin:
        top_mask = ygap > (ygap.max() - 5)
        xgap = np.concatenate([xgap, xgap[top_mask]])
        ygap = np.concatenate([ygap,
                               np.full(int(top_mask.sum()), naxis2 - 1, dtype=np.float32)])
        dgap = np.concatenate([dgap, dgap[top_mask]])
        good = np.concatenate([good, good[top_mask]])

    # --- iterative 2D polynomial fit with residual-based CR rejection ---
    model = None
    for iteration in range(_MAX_ITERS):
        if int(good.sum()) < min_gap_pixels:
            logger.warning(
                f"subtract_scattered_light: too few gap pixels after iter {iteration} — skipping"
            )
            return data, None

        try:
            c, _minx, _maxx, _miny, _maxy = pypeit_fitting.polyfit2d_general(
                xgap[good], ygap[good], dgap[good], order
            )
            if iteration < _MAX_ITERS - 1:
                # Intermediate iterations: evaluate only on the gap pixel
                # positions (fast) to compute residuals for the next clip pass.
                gap_model_vals = pypeit_fitting.evaluate_fit(
                    c, 'polynomial2d', xgap, x2=ygap
                )
                residuals = dgap - gap_model_vals
                std_resid = np.nanstd(residuals[good])
                good = np.abs(residuals) < (_SIGMA_CLIP * std_resid)
            else:
                # Final iteration: evaluate on the full detector grid.
                model = pypeit_fitting.evaluate_fit(c, 'polynomial2d', ximg, x2=yimg)

        except Exception as exc:
            logger.error(
                f"subtract_scattered_light: fit failed at iter {iteration} ({exc}) — skipping"
            )
            return data, None

    corrected = data.astype(float) - model
    logger.info(
        f"subtract_scattered_light: order={order}, image_type={image_type}, "
        f"gap_pixels_used={int(good.sum())}/{n_gap}, "
        f"model_range=[{model.min():.1f}, {model.max():.1f}] DN"
    )
    return corrected, model
