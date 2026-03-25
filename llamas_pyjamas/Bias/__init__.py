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

def generate_fallback_bias_hdu(frame_data: np.ndarray) -> fits.ImageHDU:
    """
    Construct a minimal, valid fits.ImageHDU filled with a constant bias
    level estimated from rows 30-50 of the raw frame. Never raises — always
    returns an HDU that downstream code can subtract from ``frame_data``.

    The scalar bias level is determined by the following hierarchy:

    1. Rows 30-50 median of ``frame_data`` (if frame has >50 rows).
    2. 0.0 (if the frame has fewer than 50 rows or the median fails).

    For inter-fibre gap based bias estimation (more accurate but requires
    a loaded tracer object), use ``biasChecking.build_interfibre_mask``
    directly in diagnostic contexts where a tracer is available.

    The returned HDU has three extra header keywords:
    * ``BIASSRC``  -- string describing which fallback path was used.
    * ``BIASWARN`` -- set to ``True`` to flag that no real bias was available.
    * ``BIASLVL``  -- the constant value subtracted (DN).

    Parameters
    ----------
    frame_data : numpy.ndarray
        The 2-D raw science/flat frame. Must be convertible to float.

    Returns
    -------
    astropy.io.fits.ImageHDU
        ImageHDU whose ``.data`` is a 2-D float32 array of constant value
        equal to the estimated bias level, with shape matching
        ``frame_data.shape``.
    """
    data = np.asarray(frame_data, dtype=np.float64)
    shape = data.shape

    bias_level = 0.0
    bias_src   = 'zero_fallback'

    # Attempt: rows 30-50 median
    try:
        if shape[0] > 50:
            region = data[30:50, :]
            if region.size > 0:
                bias_level = float(np.nanmedian(region))
                bias_src   = 'rows_30_50_median'
                logger.info(
                    f"generate_fallback_bias_hdu: level={bias_level:.2f} DN from rows 30-50"
                )
        else:
            logger.warning(
                "generate_fallback_bias_hdu: frame has fewer than 50 rows; using zero bias level"
            )
    except Exception as exc:
        logger.warning(
            f"generate_fallback_bias_hdu: rows 30-50 median failed ({exc}); using zero"
        )

    bias_array = np.full(shape, bias_level, dtype=np.float32)

    hdr = fits.Header()
    hdr['BIASSRC']  = (bias_src,          'Source used for fallback bias level')
    hdr['BIASWARN'] = (True,              'True = real master bias was unavailable')
    hdr['BIASLVL']  = (float(bias_level), 'Constant bias level subtracted (DN)')

    logger.warning(
        f"generate_fallback_bias_hdu: returning fallback bias "
        f"level={bias_level:.2f} DN (BIASSRC={bias_src})"
    )

    return fits.ImageHDU(data=bias_array, header=hdr)
