"""
llamas_pyjamas.Bias.biasChecking
=================================
Diagnostic module for bias quality assessment in the LLAMAS IFU pipeline.

Key components
--------------
build_interfibre_mask  -- build boolean gap mask from a TraceLlamas object's
                          fiberimg attribute (pixels where fiberimg == -1).
_check_single_detector -- compute per-detector bias residual statistics.
run_bias_checks        -- orchestrate checks over all 24 science detectors.
check_calibration_biases -- self-consistency check on a bias FITS file.

The mask cache (_MASK_CACHE) avoids recomputing the inter-fibre mask for
repeated calls with the same tracer object.
"""

import logging
import numpy as np
import scipy.ndimage
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Module-level in-memory cache: key = (id(tracer), image_type)
# Value: boolean 2-D numpy array (True = inter-fibre gap pixel)
_MASK_CACHE: dict = {}

# Image types where scattered light in fibre wings warrants wider gaps
_FLAT_IMAGE_TYPES = frozenset({'dome_flat', 'lamp_flat', 'trace_flat'})


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DetectorBiasStats:
    """Per-detector bias diagnostics produced by _check_single_detector."""
    bench: str
    side: str
    color: str
    n_interfibre_pixels: int          # number of gap pixels used
    interfibre_bias_median: float     # median of bias data in gap pixels
    interfibre_science_median: float  # median of science data in gap pixels
    residual_median: float            # median of (science - bias) in gap pixels
    residual_std: float               # std of (science - bias) in gap pixels
    test_region_median: float         # median of (science - bias) in rows 30-50
                                      # always computed regardless of mask
    image_type: str                   # 'science', 'dome_flat', etc.
    mask_source: str                  # 'interfibre', 'rows_30_50', or 'none'
    warning_flags: List[str] = field(default_factory=list)


@dataclass
class BiasCheckThresholds:
    """Configurable pass/fail thresholds for bias quality checks."""
    max_residual_median: float = 5.0    # DN — abs(residual_median) must be below this
    max_residual_std: float   = 20.0   # DN — residual_std must be below this
    min_interfibre_pixels: int = 100   # minimum gap pixels for reliable stats


@dataclass
class BiasCheckReport:
    """Aggregated report across all checked detectors."""
    detector_stats: List[DetectorBiasStats] = field(default_factory=list)
    thresholds: BiasCheckThresholds         = field(default_factory=BiasCheckThresholds)
    n_failed: int                           = 0
    failed_detectors: List[str]             = field(default_factory=list)
    summary: str                            = ''


# ---------------------------------------------------------------------------
# Inter-fibre mask builder
# ---------------------------------------------------------------------------

def build_interfibre_mask(tracer,
                           image_shape: Tuple[int, int],
                           image_type: str = 'science') -> np.ndarray:
    """
    Build a boolean mask selecting inter-fibre gap pixels.

    Uses the ``fiberimg`` attribute of a loaded ``TraceLlamas`` object.
    ``fiberimg`` is a 2-D integer array (shape naxis2 × naxis1) where pixels
    belonging to a fibre are set to that fibre's index and inter-fibre gap
    pixels are set to -1 (see traceLlamasMaster.py:769).

    For flat-field image types (dome_flat, lamp_flat, trace_flat), one pixel
    of binary erosion is applied to the fibre mask before inverting, widening
    the gap region to avoid contamination from scattered light in fibre wings.

    The result is cached in ``_MASK_CACHE`` keyed on ``(id(tracer), image_type)``
    to avoid recomputing for repeated calls with the same tracer.

    Parameters
    ----------
    tracer : TraceLlamas or None
        A loaded TraceLlamas object with a ``fiberimg`` attribute.
        If None or lacking ``fiberimg``, falls back to a rows 30-50 mask.
    image_shape : (int, int)
        (n_rows, n_cols) of the detector image, used for the fallback mask.
    image_type : str
        One of 'science', 'dome_flat', 'lamp_flat', 'trace_flat'.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``image_shape`` where True = inter-fibre gap.
    """
    # Fallback: no tracer or no fiberimg
    if tracer is None or not hasattr(tracer, 'fiberimg') or tracer.fiberimg is None:
        logger.warning(
            "build_interfibre_mask: tracer unavailable or lacks fiberimg; "
            "falling back to rows 30-50 mask"
        )
        mask = np.zeros(image_shape, dtype=bool)
        if image_shape[0] > 50:
            mask[30:50, :] = True
        return mask

    cache_key = (id(tracer), image_type)
    if cache_key in _MASK_CACHE:
        logger.debug(f"build_interfibre_mask: cache hit for {cache_key}")
        return _MASK_CACHE[cache_key]

    # Base gap mask: pixels not assigned to any fibre
    gap_mask = (tracer.fiberimg == -1)

    # For flat fields: erode the fibre mask by 1 pixel to widen gaps
    if image_type in _FLAT_IMAGE_TYPES:
        fibre_mask = ~gap_mask
        struct = np.ones((3, 1), dtype=bool)  # vertical structuring element
        eroded_fibre = scipy.ndimage.binary_erosion(fibre_mask, structure=struct)
        gap_mask = ~eroded_fibre
        logger.debug(
            f"build_interfibre_mask: applied 1px erosion for image_type='{image_type}'"
        )

    _MASK_CACHE[cache_key] = gap_mask
    logger.debug(
        f"build_interfibre_mask: built mask for {cache_key}, "
        f"{gap_mask.sum()} gap pixels ({100*gap_mask.mean():.1f}%)"
    )
    return gap_mask


# ---------------------------------------------------------------------------
# Single-detector check
# ---------------------------------------------------------------------------

def _check_single_detector(frame_ext_data: np.ndarray,
                            bias_ext_data: np.ndarray,
                            tracer,
                            image_type: str,
                            bench: str,
                            side: str,
                            color: str,
                            thresholds: BiasCheckThresholds = None) -> DetectorBiasStats:
    """
    Compute bias residual statistics for a single detector extension.

    Parameters
    ----------
    frame_ext_data : numpy.ndarray
        Raw 2-D science or flat frame data for this detector.
    bias_ext_data : numpy.ndarray
        Bias frame data for this detector (same shape).
    tracer : TraceLlamas or None
        Loaded trace object for this detector (provides fiberimg).
    image_type : str
        Frame type string (e.g. 'science', 'dome_flat').
    bench, side, color : str
        Camera identifiers for this detector.
    thresholds : BiasCheckThresholds, optional
        Pass/fail thresholds. Uses defaults if None.

    Returns
    -------
    DetectorBiasStats
    """
    if thresholds is None:
        thresholds = BiasCheckThresholds()

    warnings = []
    shape = frame_ext_data.shape

    # Effective residual threshold — relaxed for flats (scattered light)
    eff_max_median = (thresholds.max_residual_median * 1.5
                      if image_type in _FLAT_IMAGE_TYPES
                      else thresholds.max_residual_median)

    # --- Build inter-fibre mask ---
    gap_mask = build_interfibre_mask(tracer, shape, image_type)
    mask_source = 'rows_30_50' if (tracer is None or not hasattr(tracer, 'fiberimg')
                                   or tracer.fiberimg is None) else 'interfibre'

    n_gap = int(gap_mask.sum())

    if n_gap < thresholds.min_interfibre_pixels:
        logger.warning(
            f"_check_single_detector {bench}{side} {color}: only {n_gap} gap pixels "
            f"(< {thresholds.min_interfibre_pixels}); stats may be unreliable"
        )
        warnings.append(f"low_gap_pixels:{n_gap}")
        mask_source = 'none'

    # --- Gap-region statistics ---
    if n_gap > 0:
        sci_gap  = frame_ext_data[gap_mask].astype(float)
        bias_gap = bias_ext_data[gap_mask].astype(float)
        interfibre_sci_med  = float(np.nanmedian(sci_gap))
        interfibre_bias_med = float(np.nanmedian(bias_gap))
        residual            = sci_gap - bias_gap
        res_median          = float(np.nanmedian(residual))
        res_std             = float(np.nanstd(residual))
    else:
        interfibre_sci_med  = np.nan
        interfibre_bias_med = np.nan
        res_median          = np.nan
        res_std             = np.nan

    # --- Test-region check: always rows 30-50, independent of mask ---
    try:
        residual_img = frame_ext_data.astype(float) - bias_ext_data.astype(float)
        test_med = float(np.nanmedian(residual_img[30:51, :]))
    except Exception:
        test_med = np.nan

    # --- Flag threshold violations ---
    if not np.isnan(res_median) and abs(res_median) > eff_max_median:
        warnings.append(f"high_residual_median:{res_median:.1f}DN")
    if not np.isnan(res_std) and res_std > thresholds.max_residual_std:
        warnings.append(f"high_residual_std:{res_std:.1f}DN")

    return DetectorBiasStats(
        bench=bench,
        side=side,
        color=color,
        n_interfibre_pixels=n_gap,
        interfibre_bias_median=interfibre_bias_med,
        interfibre_science_median=interfibre_sci_med,
        residual_median=res_median,
        residual_std=res_std,
        test_region_median=test_med,
        image_type=image_type,
        mask_source=mask_source,
        warning_flags=warnings,
    )


# ---------------------------------------------------------------------------
# Orchestration helpers
# ---------------------------------------------------------------------------

def _match_bias_hdu(bias_hdul, bench: str, side: str, color: str):
    """Return the bias HDU matching (bench, side, color), or None."""
    target_bench = str(bench)
    target_side  = side.upper()
    target_color = color.lower()

    for hdu in bias_hdul[1:]:
        hdr = hdu.header
        if 'COLOR' in hdr and 'BENCH' in hdr and 'SIDE' in hdr:
            if (str(hdr['BENCH']) == target_bench and
                    str(hdr['SIDE']).upper() == target_side and
                    str(hdr['COLOR']).lower() == target_color):
                return hdu
        elif 'CAM_NAME' in hdr:
            cam = hdr['CAM_NAME']
            parts = cam.split('_')
            if len(parts) >= 2:
                c_color = parts[1].lower()
                c_bench = parts[0][0]
                c_side  = parts[0][1].upper()
                if c_color == target_color and c_bench == target_bench and c_side == target_side:
                    return hdu
    return None


def run_bias_checks(science_hdul,
                    bias_hdul,
                    tracers_by_cam: Dict[Tuple[str, str, str], object],
                    image_type: str = 'science',
                    thresholds: BiasCheckThresholds = None) -> BiasCheckReport:
    """
    Run bias quality checks across all science detector extensions.

    Parameters
    ----------
    science_hdul : astropy.io.fits.HDUList
        Science (or flat) FITS HDUList, extensions 1-24.
    bias_hdul : astropy.io.fits.HDUList
        Bias FITS HDUList, extensions 1-24.
    tracers_by_cam : dict
        Keys are (bench, side, color) tuples; values are loaded TraceLlamas
        objects. Pass an empty dict if no tracers are available.
    image_type : str
        Frame type string passed through to _check_single_detector.
    thresholds : BiasCheckThresholds, optional

    Returns
    -------
    BiasCheckReport
    """
    if thresholds is None:
        thresholds = BiasCheckThresholds()

    report = BiasCheckReport(thresholds=thresholds)

    for hdu in science_hdul[1:]:
        hdr = hdu.header
        if hdu.data is None:
            continue

        # Extract camera identifiers
        if 'COLOR' in hdr and 'BENCH' in hdr and 'SIDE' in hdr:
            color = str(hdr['COLOR']).lower()
            bench = str(hdr['BENCH'])
            side  = str(hdr['SIDE']).upper()
        elif 'CAM_NAME' in hdr:
            cam   = hdr['CAM_NAME']
            parts = cam.split('_')
            color = parts[1].lower() if len(parts) >= 2 else 'unknown'
            bench = parts[0][0] if parts else '?'
            side  = parts[0][1].upper() if parts and len(parts[0]) >= 2 else '?'
        else:
            logger.warning("run_bias_checks: extension missing camera header keywords; skipping")
            continue

        bias_hdu = _match_bias_hdu(bias_hdul, bench, side, color)
        if bias_hdu is None:
            logger.warning(f"run_bias_checks: no bias extension for {bench}{side} {color}; skipping")
            continue

        tracer = tracers_by_cam.get((bench, side, color), None)

        stats = _check_single_detector(
            frame_ext_data=hdu.data,
            bias_ext_data=bias_hdu.data,
            tracer=tracer,
            image_type=image_type,
            bench=bench,
            side=side,
            color=color,
            thresholds=thresholds,
        )
        report.detector_stats.append(stats)

        if stats.warning_flags:
            cam_id = f"{bench}{side}_{color}"
            report.n_failed += 1
            report.failed_detectors.append(cam_id)

    n_total = len(report.detector_stats)
    report.summary = (
        f"Bias check: {n_total} detectors checked, "
        f"{report.n_failed} failed thresholds "
        f"({', '.join(report.failed_detectors) or 'none'})"
    )
    logger.info(report.summary)
    return report


def check_calibration_biases(bias_hdul,
                              tracers_by_cam: Dict[Tuple[str, str, str], object],
                              thresholds: BiasCheckThresholds = None) -> BiasCheckReport:
    """
    Self-consistency check: compare each bias extension against the median
    of all 24 bias extensions.

    Parameters
    ----------
    bias_hdul : astropy.io.fits.HDUList
        Bias FITS HDUList with 24 camera extensions.
    tracers_by_cam : dict
        Keys are (bench, side, color); values are TraceLlamas objects.
    thresholds : BiasCheckThresholds, optional

    Returns
    -------
    BiasCheckReport
    """
    if thresholds is None:
        thresholds = BiasCheckThresholds()

    # Build a reference "median bias" from all valid extensions
    valid_data = [hdu.data.astype(float) for hdu in bias_hdul[1:]
                  if hdu.data is not None]
    if not valid_data:
        logger.error("check_calibration_biases: no valid bias extensions found")
        return BiasCheckReport(thresholds=thresholds,
                               summary="ERROR: no valid bias extensions")

    reference = np.nanmedian(np.stack(valid_data, axis=0), axis=0)

    # Build a synthetic HDUList where every extension is the reference
    from astropy.io import fits as _fits
    ref_hdul = _fits.HDUList([bias_hdul[0]])
    for hdu in bias_hdul[1:]:
        ref_ext = _fits.ImageHDU(data=reference, header=hdu.header)
        ref_hdul.append(ref_ext)

    return run_bias_checks(
        science_hdul=bias_hdul,
        bias_hdul=ref_hdul,
        tracers_by_cam=tracers_by_cam,
        image_type='science',
        thresholds=thresholds,
    )
