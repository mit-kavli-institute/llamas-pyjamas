"""
llamas_pyjamas.Bias
===================
Bias calibration package for LLAMAS IFU pipeline.

Public API
----------
BiasNotFoundError        -- raised when a bias FITS file cannot be read/located.
BiasReadModeError        -- raised when bias read-mode does not match science frame.
generate_fallback_bias_hdu -- returns a valid fits.ImageHDU even when no bias
                              file is available.
camera_of_header         -- (bench, side, color) of an extension header.
match_camera_hdu         -- the extension of a calibration file for one camera,
                            matched by header (the master bias and master dark
                            both go through it).
verify_camera_hdu        -- re-confirms a calibration extension's camera just
                            before it is subtracted.

The BiasLlamas class (master bias stacking) is in llamasBias.py and is not
re-exported here to avoid circular imports with astropy during Ray worker init.
"""

import logging
import numpy as np
from astropy.io import fits

from llamas_pyjamas.constants import idx_lookup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class BiasNotFoundError(FileNotFoundError):
    """
    Raised when a master-bias FITS file is absent, unreadable, or returns
    (None, None) from process_fits_by_color.

    Attributes
    ----------
    path : str
        The file path that could not be opened.
    """
    def __init__(self, path: str, message: str = None):
        self.path = path
        if message is None:
            message = (
                f"Master bias not found or could not be read: '{path}'. "
                f"Check that the file exists and is a valid 24-extension LLAMAS bias FITS."
            )
        super().__init__(message)


class BiasReadModeError(ValueError):
    """
    Raised when the read-mode encoded in the bias FITS primary header does
    not match the requested read-mode for the current science frame.

    Attributes
    ----------
    requested_mode : str
    bias_mode      : str
    bias_path      : str
    """
    def __init__(self, requested_mode: str, bias_mode: str, bias_path: str):
        self.requested_mode = requested_mode
        self.bias_mode = bias_mode
        self.bias_path = bias_path
        super().__init__(
            f"Read-mode mismatch: science frame needs '{requested_mode}' "
            f"but bias file '{bias_path}' has READ-MDE='{bias_mode}'."
        )


class BiasCameraMissingError(BiasNotFoundError):
    """
    Raised when a master-bias file is readable but has no extension for the
    requested camera (e.g. a master built while that camera was down).

    Subclasses BiasNotFoundError so every caller that already falls back to
    ``generate_fallback_bias_hdu`` for a missing bias does so here too.

    Attributes
    ----------
    path : str
    bench, side, color : str
    """
    def __init__(self, path: str, bench: str, side: str, color: str):
        self.bench, self.side, self.color = str(bench), str(side), str(color)
        super().__init__(
            path,
            f"Master bias '{path}' has no extension for {bench}{side} {color}; "
            f"the file is missing this camera."
        )


# ---------------------------------------------------------------------------
# Camera matching for per-camera calibrations (master bias, master dark)
# ---------------------------------------------------------------------------

def camera_of_header(header):
    """(bench, side, color) of an extension header, from COLOR/BENCH/SIDE or else CAM_NAME.

    Missing parts come back as ''. bench is a str, side upper case, color lower case.
    """
    if 'COLOR' in header and 'BENCH' in header and 'SIDE' in header:
        return (str(header['BENCH']).strip(), str(header['SIDE']).strip().upper(),
                str(header['COLOR']).strip().lower())
    if 'CAM_NAME' in header:
        parts = str(header['CAM_NAME']).split('_')
        bench = parts[0][0] if parts and parts[0] else ''
        side = parts[0][1].upper() if parts and len(parts[0]) >= 2 else ''
        color = parts[1].lower() if len(parts) >= 2 else ''
        return bench, side, color
    return '', '', ''


def _names_camera(header):
    return ('COLOR' in header and 'BENCH' in header and 'SIDE' in header) or 'CAM_NAME' in header


def match_camera_hdu(hdus, bench, side, color, source=''):
    """The extension of a calibration file that belongs to one camera.

    Matching order (the master bias has always been matched this way):
    1. COLOR/BENCH/SIDE header keywords, then CAM_NAME;
    2. if every extension names its camera and none is this one, the camera is
       absent: return None;
    3. otherwise the standard extension index, accepted only if that extension's
       COLOR, BENCH and SIDE all name this camera; anything else returns None.

    Raises ValueError when the index lookup itself fails. Callers decide what an
    absent camera means (the master bias raises BiasCameraMissingError).
    """
    target = (str(bench), str(side).upper(), str(color).lower())
    for hdu in hdus[1:]:
        if _names_camera(hdu.header) and camera_of_header(hdu.header) == target:
            logger.info(f"{source}: matched {target[0]}{target[1]} {target[2]} by header")
            return hdu

    if len(hdus) > 1 and all(_names_camera(h.header) for h in hdus[1:]):
        return None

    idx = idx_lookup.get((target[2], target[0], target[1]))
    print(f"Bias index: {idx} for {bench}/{side}/{color}")
    if idx is None or idx >= len(hdus):
        raise ValueError(f"Index lookup failed for {bench}/{side}/{color} in file {source}")
    logger.warning(f"{source}: {target[0]}{target[1]} {target[2]} matched by index only "
                   f"(header match failed)")
    hdr = hdus[idx].header
    found = (str(hdr['BENCH']) if 'BENCH' in hdr else None,
             str(hdr['SIDE']).upper() if 'SIDE' in hdr else None,
             str(hdr['COLOR']).lower() if 'COLOR' in hdr else None)
    if found != target:
        logger.warning(f"{source}: index lookup returned the wrong camera for "
                       f"{target[0]}{target[1]} {target[2]}: got {found} at index {idx}")
        return None
    return hdus[idx]


def verify_camera_hdu(header, bench, side, color):
    """True unless the extension header names a different camera.

    The last check before a calibration array is subtracted: compares
    BENCH/SIDE/COLOR (or CAM_NAME) with the frame's camera. A header with no
    camera keywords cannot contradict, and passes.
    """
    found = camera_of_header(header)
    if not all(found):
        return True
    return found == (str(bench), str(side).upper(), str(color).lower())


# ---------------------------------------------------------------------------
# Fallback HDU generator
# ---------------------------------------------------------------------------

def generate_fallback_bias_hdu(frame_data: np.ndarray, tracer=None) -> fits.ImageHDU:
    """
    Construct a minimal, valid fits.ImageHDU filled with a constant bias
    level estimated from the raw frame. Never raises — always returns an HDU
    that downstream code can subtract from ``frame_data``.

    The bias level is chosen by cross-validating two candidate estimates:

    1. **Inter-fibre estimate** (preferred when tracer available):
       Median of ``frame_data`` in the inter-fibre gap pixels identified by
       ``tracer.fiberimg == -1`` via ``biasChecking.build_interfibre_mask``.
    2. **Test-region estimate**: Median of rows 30–50 of ``frame_data``.

    Cross-validation rule (``cross_check_threshold = 5 DN``):
    * If the two estimates agree within 5 DN → use the inter-fibre estimate
      (more spatially representative).
    * If they diverge → use the lower estimate: contamination (sky, dark
      current, source) can only push a pedestal estimate up.

    When no tracer is available (or it lacks ``fiberimg``), only the
    test-region estimate is used.

    The returned HDU has the following header keywords:
    * ``BIASSRC``  -- which estimate was selected.
    * ``BIASWARN`` -- ``True`` to flag that no master bias HDU was available.
    * ``BIASLVL``  -- the constant value subtracted (DN).
    * ``BIASIFF``  -- inter-fibre estimate (DN), or NaN if unavailable.
    * ``BIASTST``  -- test-region estimate (DN), or NaN if unavailable.

    Parameters
    ----------
    frame_data : numpy.ndarray
        The 2-D raw science/flat frame. Must be convertible to float.
    tracer : TraceLlamas or None, optional
        A loaded TraceLlamas object with a ``fiberimg`` attribute.
        When provided, enables the inter-fibre gap estimate.

    Returns
    -------
    astropy.io.fits.ImageHDU
        ImageHDU whose ``.data`` is a 2-D float32 array of constant value
        equal to the estimated bias level, with shape matching
        ``frame_data.shape``.
    """
    _CROSS_CHECK_THRESHOLD = 5.0  # DN — max tolerated gap between the two estimates

    data  = np.asarray(frame_data, dtype=np.float64)
    shape = data.shape

    # --- Inter-fibre estimate ---
    est_interfibre = float('nan')
    try:
        from llamas_pyjamas.Bias.biasChecking import build_interfibre_mask
        gap_mask = build_interfibre_mask(tracer, shape, image_type='science')
        n_gap = int(gap_mask.sum())
        if n_gap >= 100:
            est_interfibre = float(np.nanmedian(data[gap_mask]))
            logger.info(
                f"generate_fallback_bias_hdu: inter-fibre estimate="
                f"{est_interfibre:.2f} DN from {n_gap} gap pixels"
            )
        else:
            logger.warning(
                f"generate_fallback_bias_hdu: only {n_gap} inter-fibre gap pixels "
                f"(< 100); skipping inter-fibre estimate"
            )
    except Exception as exc:
        logger.warning(f"generate_fallback_bias_hdu: inter-fibre estimate failed ({exc})")

    # --- Test-region estimate (rows 30–50) ---
    est_test_region = float('nan')
    try:
        if shape[0] > 50:
            region = data[30:50, :]
            if region.size > 0:
                est_test_region = float(np.nanmedian(region))
                logger.info(
                    f"generate_fallback_bias_hdu: test-region estimate="
                    f"{est_test_region:.2f} DN from rows 30-50"
                )
        else:
            logger.warning(
                "generate_fallback_bias_hdu: frame has fewer than 50 rows; "
                "test-region estimate unavailable"
            )
    except Exception as exc:
        logger.warning(
            f"generate_fallback_bias_hdu: test-region estimate failed ({exc})"
        )

    # --- Cross-validate and select ---
    both_valid = not (np.isnan(est_interfibre) or np.isnan(est_test_region))
    if both_valid:
        divergence = abs(est_interfibre - est_test_region)
        if divergence <= _CROSS_CHECK_THRESHOLD:
            # Estimates agree — prefer inter-fibre (more spatially accurate)
            bias_level = est_interfibre
            bias_src   = 'interfibre_mask'
            logger.info(
                f"generate_fallback_bias_hdu: estimates agree "
                f"(|{est_interfibre:.2f} - {est_test_region:.2f}| = {divergence:.2f} DN ≤ "
                f"{_CROSS_CHECK_THRESHOLD} DN); using inter-fibre estimate"
            )
        else:
            # Estimates diverge — prefer the lower (least contaminated) value.
            if est_interfibre <= est_test_region:
                bias_level = est_interfibre
                bias_src   = 'interfibre_mask'
            else:
                bias_level = est_test_region
                bias_src   = 'rows_30_50_median'
            logger.warning(
                f"generate_fallback_bias_hdu: estimates diverge "
                f"({divergence:.2f} DN > {_CROSS_CHECK_THRESHOLD} DN); "
                f"using lower estimate {bias_src} (level={bias_level:.2f} DN) "
                f"[inter-fibre={est_interfibre:.2f}, test-region={est_test_region:.2f}]"
            )
    elif not np.isnan(est_interfibre):
        bias_level = est_interfibre
        bias_src   = 'interfibre_mask'
    elif not np.isnan(est_test_region):
        bias_level = est_test_region
        bias_src   = 'rows_30_50_median'
    else:
        bias_level = 0.0
        bias_src   = 'zero_fallback'
        logger.warning("generate_fallback_bias_hdu: all estimates failed; using zero bias level")

    bias_array = np.full(shape, bias_level, dtype=np.float32)

    hdr = fits.Header()
    hdr['BIASSRC']  = (bias_src,          'Source used for fallback bias level')
    hdr['BIASWARN'] = (True,              'True = real master bias was unavailable')
    hdr['BIASLVL']  = (float(bias_level), 'Constant bias level subtracted (DN)')
    hdr['BIASIFF']  = (float(est_interfibre)  if not np.isnan(est_interfibre)  else float('nan'),
                       'Inter-fibre gap bias estimate (DN)')
    hdr['BIASTST']  = (float(est_test_region) if not np.isnan(est_test_region) else float('nan'),
                       'Test-region rows 30-50 bias estimate (DN)')

    logger.warning(
        f"generate_fallback_bias_hdu: returning fallback bias "
        f"level={bias_level:.2f} DN (BIASSRC={bias_src})"
    )

    return fits.ImageHDU(data=bias_array, header=hdr)
