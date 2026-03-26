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

The BiasLlamas class (master bias stacking) is in llamasBias.py and is not
re-exported here to avoid circular imports with astropy during Ray worker init.
"""

import logging
import numpy as np
from astropy.io import fits

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
    * If they diverge → use whichever is closer to the raw frame's
      inter-fibre median (i.e. whichever self-validates better).

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
            # Estimates diverge — prefer the lower value.
            # A CCD bias pedestal can only be over-estimated (contaminated by sky,
            # dark current, or source signal); it cannot be under-estimated.
            # The lower of the two estimates has the least contamination.
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
