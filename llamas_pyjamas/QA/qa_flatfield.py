import logging
import numpy as np
from scipy import ndimage, stats
from sklearn.decomposition import PCA
import warnings

_logger = logging.getLogger(__name__)


def qa_rss_residual_uniformity(rss_data, variance_threshold=0.02):
    """
    Evaluates the 1D residual uniformity of flat-fielded Row-Stacked Spectra (RSS).

    Parameters:
    -----------
    rss_data : numpy.ndarray
        2D array of shape (n_fibres, n_wavelengths). This should be an
        independent flat-fielded frame (e.g., twilight flat).
    variance_threshold : float
        Maximum allowed fractional RMS (default 2%).

    Algorithm:
    ----------
    1. Mask any NaN or zero values (dead fibres).
    2. Calculate the median spectrum across all valid fibres (axis=0).
    3. Divide the entire rss_data array by this median spectrum to create a residuals array.
    4. Calculate the standard deviation (RMS) of the residuals for each fibre.
    5. Calculate the global median of these RMS values.

    Returns:
    --------
    dict: {'global_rms': float, 'passed': bool, 'failed_fibres': list of indices}
          'passed' is True if global_rms < variance_threshold.
    """
    data = rss_data.copy().astype(np.float64)

    # Mask dead fibres (NaN or all-zero rows)
    valid_mask = np.isfinite(data) & (data != 0)
    data[~valid_mask] = np.nan

    # Median spectrum across fibres (axis=0)
    median_spectrum = np.nanmedian(data, axis=0)

    # Avoid division by zero
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        residuals = data / median_spectrum[np.newaxis, :]

    # Per-fibre RMS
    per_fibre_rms = np.nanstd(residuals, axis=1)

    global_rms = float(np.nanmedian(per_fibre_rms))
    failed_fibres = list(np.where(per_fibre_rms > variance_threshold)[0])

    return {
        'global_rms': global_rms,
        'passed': global_rms < variance_threshold,
        'failed_fibres': failed_fibres,
    }


def qa_rss_chromatic_stability(rss_data, correlation_threshold=0.3):
    """
    Checks if flat fielding introduces artificial wavelength-dependent throughput changes.

    Parameters:
    -----------
    rss_data : numpy.ndarray
        2D array of shape (n_fibres, n_wavelengths).
    correlation_threshold : float
        Maximum allowed Pearson r between blue and red residuals.

    Algorithm:
    ----------
    1. Divide the wavelength axis into three equal chunks (Blue, Green, Red).
    2. Calculate the median flux per fibre in the Blue chunk and the Red chunk.
    3. Normalize both the Blue and Red fibre arrays by their respective global medians.
    4. Calculate the Pearson correlation coefficient (r) between the normalized
       Blue array and normalized Red array across all fibres.

    Returns:
    --------
    dict: {'pearson_r': float, 'p_value': float, 'passed': bool}
          'passed' is True if abs(pearson_r) < correlation_threshold.
    """
    n_fibres, n_wavelengths = rss_data.shape
    chunk = n_wavelengths // 3

    blue_median = np.nanmedian(rss_data[:, :chunk], axis=1)
    red_median  = np.nanmedian(rss_data[:, 2 * chunk:], axis=1)

    blue_norm = blue_median / np.nanmedian(blue_median)
    red_norm  = red_median  / np.nanmedian(red_median)

    # Remove any NaN rows before computing correlation
    valid = np.isfinite(blue_norm) & np.isfinite(red_norm)
    r, p = stats.pearsonr(blue_norm[valid], red_norm[valid])

    return {
        'pearson_r': float(r),
        'p_value': float(p),
        'passed': abs(r) < correlation_threshold,
    }


def qa_rss_pca_diagnostics(rss_data, n_components=3):
    """
    Uses PCA to detect systematic flat-fielding residuals across fibres.

    Parameters:
    -----------
    rss_data : numpy.ndarray
        2D array of shape (n_fibres, n_wavelengths).

    Algorithm:
    ----------
    1. Clean data: Replace NaNs/Infs with 0, subtract the mean spectrum.
    2. Fit sklearn.decomposition.PCA(n_components=n_components) to the data.
    3. Extract the explained variance ratio for the first component (PC1).

    Returns:
    --------
    dict: {'explained_variance_ratios': list, 'pc1_vector': numpy.ndarray, 'passed': bool}
          'passed' is True if the explained variance of PC1 is < 0.10 (10%).
          If PC1 explains >10%, a dominant systematic error exists.
    """
    data = rss_data.copy().astype(np.float64)
    data[~np.isfinite(data)] = 0.0

    # Subtract mean spectrum (mean across fibres per wavelength)
    data -= np.mean(data, axis=0)

    pca = PCA(n_components=n_components)
    pca.fit(data)

    evr = pca.explained_variance_ratio_.tolist()
    pc1_vector = pca.components_[0]

    return {
        'explained_variance_ratios': evr,
        'pc1_vector': pc1_vector,
        'passed': evr[0] < 0.10,
    }


def qa_cube_high_frequency_noise(cube_data, filter_size=3, noise_threshold=0.05):
    """
    Detects "checkerboard" spatial artefacts in a 3D flat-fielded cube.

    Parameters:
    -----------
    cube_data : numpy.ndarray
        3D array of shape (n_wavelengths, spatial_y, spatial_x).

    Algorithm:
    ----------
    1. Collapse the cube along the wavelength axis (axis=0) using a median to
       create a 2D broadband image.
    2. Apply a median filter (scipy.ndimage.median_filter) with `filter_size`
       to this 2D image to create a "smooth" background map.
    3. Subtract the smooth map from the original 2D image, then divide by the
       smooth map to get fractional high-frequency residuals.
    4. Calculate the RMS (standard deviation) of these residuals.

    Returns:
    --------
    dict: {'high_freq_rms': float, 'passed': bool}
          'passed' is True if high_freq_rms < noise_threshold.
    """
    broadband = np.nanmedian(cube_data, axis=0)

    smooth = ndimage.median_filter(broadband, size=filter_size)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        residuals = (broadband - smooth) / smooth

    high_freq_rms = float(np.nanstd(residuals))

    return {
        'high_freq_rms': high_freq_rms,
        'passed': high_freq_rms < noise_threshold,
    }


def qa_cube_spatial_autocorrelation(cube_data):
    """
    Calculates Moran's I on the broadband image to check if adjacent spaxels
    have flat-fielding discontinuities.

    Parameters:
    -----------
    cube_data : numpy.ndarray
        3D array of shape (n_wavelengths, spatial_y, spatial_x).

    Algorithm:
    ----------
    1. Collapse the cube along the wavelength axis using a median into a 2D image.
    2. Flatten the 2D image into a 1D array (remove NaNs).
    3. Compute Moran's I spatial autocorrelation index for immediate
       neighbors (up/down/left/right).
       *Formula: I = (N / W) * sum(w_ij * (x_i - mean)*(x_j - mean)) / sum((x_i - mean)^2)*
       *(For Claude Code: implement a simplified shifting array matrix approach
       to calculate correlation with 1-pixel shifted arrays).*

    Returns:
    --------
    dict: {'morans_i': float, 'passed': bool}
          'passed' is True if -0.1 < morans_i < 0.1 (indicating roughly random
          distribution of residuals, barring interpolation smoothing).
    """
    broadband = np.nanmedian(cube_data, axis=0)

    mean_val = np.nanmean(broadband)
    z = broadband - mean_val

    ny, nx = z.shape

    # Cross-products with 1-pixel shifted neighbors (up/down/left/right)
    cross  = np.nansum(z[:-1, :] * z[1:, :])   # vertical: row i vs row i+1
    cross += np.nansum(z[1:, :]  * z[:-1, :])   # vertical: row i+1 vs row i
    cross += np.nansum(z[:, :-1] * z[:, 1:])    # horizontal: col j vs col j+1
    cross += np.nansum(z[:, 1:]  * z[:, :-1])   # horizontal: col j+1 vs col j

    N = float(np.sum(~np.isnan(broadband)))
    # Total number of directed neighbor pairs (each pair counted twice per direction)
    W = float(2 * (ny - 1) * nx + 2 * ny * (nx - 1))

    denom = np.nansum(z ** 2)
    if denom == 0 or W == 0:
        morans_i = 0.0
    else:
        morans_i = float((N / W) * cross / denom)

    return {
        'morans_i': morans_i,
        'passed': -0.1 < morans_i < 0.1,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Twilight fibre-flat QA functions
# ──────────────────────────────────────────────────────────────────────────────

def qa_twilight_correction_uniformity(twilight_extractions, corrections_file,
                                       variance_threshold=0.02):
    """Validate that the fibre-flat correction renders the twilight uniform.

    Divides each twilight fibre's spectrum by its C_i(lambda) correction and
    measures the residual fibre-to-fibre scatter.  A well-behaved correction
    should leave the corrected twilight with an RMS < 2% across fibres.

    Parameters
    ----------
    twilight_extractions : dict
        Calibrated twilight extraction dict (keys: 'extractions', 'metadata').
    corrections_file : str
        Path to ``fibre_flat_corrections.fits``.
    variance_threshold : float, optional
        Maximum allowed global median RMS (default 0.02 = 2%).

    Returns
    -------
    dict
        ``{'global_rms': float, 'passed': bool, 'per_fibre_rms': ndarray}``
    """
    from astropy.io import fits

    twi_list = twilight_extractions['extractions']
    twi_meta = twilight_extractions['metadata']

    with fits.open(corrections_file) as corr_hdul:
        corr_index = {hdu.name: hdu for hdu in corr_hdul
                      if hdu.name not in ('PRIMARY',)}

    per_fibre_rms = []

    for ext, meta in zip(twi_list, twi_meta):
        ext_name = (f"{meta['channel']}_{meta['bench']}_{meta['side']}"
                    ).upper()
        # Corrections FITS stores extension names like RED_1_A
        if ext_name not in corr_index:
            continue

        corr_tbl   = corr_index[ext_name].data
        fiber_ids  = corr_tbl['FIBER_ID']
        correction = corr_tbl['CORRECTION']   # shape (n_fibres, n_pix)

        twi_counts = ext.counts.astype(np.float64)
        dead_set   = set(getattr(ext, 'dead_fibers', []) or [])

        for row_idx, fid in enumerate(fiber_ids):
            if fid >= twi_counts.shape[0] or fid in dead_set:
                continue
            tc  = twi_counts[fid]
            c_i = correction[row_idx]
            valid = (np.isfinite(tc) & np.isfinite(c_i)
                     & (tc > 0) & (c_i > 0))
            if np.sum(valid) < 20:
                continue
            corrected_ratio = tc[valid] / c_i[valid]
            # Normalise per fibre so spectral shape does not inflate RMS
            med = np.nanmedian(corrected_ratio)
            if med > 0:
                per_fibre_rms.append(float(np.nanstd(corrected_ratio / med)))

    if not per_fibre_rms:
        _logger.warning("qa_twilight_correction_uniformity: no valid fibres found")
        return {'global_rms': np.nan, 'passed': False,
                'per_fibre_rms': np.array([])}

    rms_arr    = np.array(per_fibre_rms)
    global_rms = float(np.nanmedian(rms_arr))
    passed     = global_rms < variance_threshold

    if not passed:
        _logger.warning(
            f"qa_twilight_correction_uniformity: global RMS = {global_rms:.3f} "
            f"exceeds threshold {variance_threshold:.3f} — "
            "fibre flat may be striped or under-corrected")

    return {'global_rms': global_rms, 'passed': passed, 'per_fibre_rms': rms_arr}


def qa_dichroic_continuity(corrections_file, jump_threshold=0.05):
    """Check for flux discontinuities at dichroic boundaries.

    For each fibre, stitches the blue, green, and red C_i(lambda) correction
    curves end-to-end and measures the fractional flux jump where the colour
    channels meet.  Jumps > 5% indicate inconsistent gradient removal or T_i
    normalisation between channels.

    Parameters
    ----------
    corrections_file : str
        Path to ``fibre_flat_corrections.fits``.
    jump_threshold : float, optional
        Maximum allowed fractional jump at a dichroic boundary (default 0.05).

    Returns
    -------
    dict
        ``{'max_jump_bg': float, 'max_jump_gr': float,
           'n_bad_bg': int, 'n_bad_gr': int, 'passed': bool}``
    """
    from astropy.io import fits

    # Organise extensions by benchside
    bench_data = {}   # {benchside: {'blue': ..., 'green': ..., 'red': ...}}
    with fits.open(corrections_file) as hdul:
        for hdu in hdul:
            if hdu.name in ('PRIMARY', '') or not hasattr(hdu, 'data'):
                continue
            hdr = hdu.header
            channel = hdr.get('CHANNEL', '').lower()
            bench   = str(hdr.get('BENCH', ''))
            side    = str(hdr.get('SIDE', ''))
            if not channel or not bench or not side:
                continue
            bs_key = f"{bench}{side}"
            bench_data.setdefault(bs_key, {})[channel] = hdu.data

    max_jump_bg = 0.0
    max_jump_gr = 0.0
    n_bad_bg    = 0
    n_bad_gr    = 0

    for bs_key, channels in bench_data.items():
        blue_tbl  = channels.get('blue')
        green_tbl = channels.get('green')
        red_tbl   = channels.get('red')

        if blue_tbl is None or green_tbl is None or green_tbl is None:
            continue

        # Build fiber_id → row dicts
        def fid_map(tbl):
            return {int(fid): i
                    for i, fid in enumerate(tbl['FIBER_ID'])}

        b_map = fid_map(blue_tbl)
        g_map = fid_map(green_tbl)
        r_map = (fid_map(red_tbl) if red_tbl is not None else {})

        common_bg = set(b_map) & set(g_map)
        common_gr = set(g_map) & set(r_map)

        for fid in common_bg:
            b_corr = blue_tbl['CORRECTION'][b_map[fid]]
            g_corr = green_tbl['CORRECTION'][g_map[fid]]
            # Value at blue red-end and green blue-end
            b_finite = b_corr[np.isfinite(b_corr)]
            g_finite = g_corr[np.isfinite(g_corr)]
            if len(b_finite) == 0 or len(g_finite) == 0:
                continue
            jump = abs(float(b_finite[-1]) - float(g_finite[0])) / max(
                abs(float(g_finite[0])), 1e-6)
            max_jump_bg = max(max_jump_bg, jump)
            if jump > jump_threshold:
                n_bad_bg += 1

        for fid in common_gr:
            g_corr = green_tbl['CORRECTION'][g_map[fid]]
            r_corr = red_tbl['CORRECTION'][r_map[fid]]
            g_finite = g_corr[np.isfinite(g_corr)]
            r_finite = r_corr[np.isfinite(r_corr)]
            if len(g_finite) == 0 or len(r_finite) == 0:
                continue
            jump = abs(float(g_finite[-1]) - float(r_finite[0])) / max(
                abs(float(r_finite[0])), 1e-6)
            max_jump_gr = max(max_jump_gr, jump)
            if jump > jump_threshold:
                n_bad_gr += 1

    passed = (max_jump_bg <= jump_threshold and max_jump_gr <= jump_threshold)
    if not passed:
        _logger.warning(
            f"qa_dichroic_continuity: max B-G jump={max_jump_bg:.3f}, "
            f"max G-R jump={max_jump_gr:.3f} (threshold={jump_threshold:.3f}). "
            "Check gradient removal consistency across colour channels.")

    return {
        'max_jump_bg': max_jump_bg,
        'max_jump_gr': max_jump_gr,
        'n_bad_bg':    n_bad_bg,
        'n_bad_gr':    n_bad_gr,
        'passed':      passed,
    }


def qa_fraunhofer_residuals(smooth_ratio_arr, wave_arr,
                             line_to_cont_threshold=2.0):
    """Check for Fraunhofer line residuals in twilight smooth-ratio curves.

    If the Savitzky-Golay smoothing window is too small it fits the absorption
    lines rather than smoothing over them, leaving sharp spikes in
    smooth_ratio_i(lambda) at Halpha (6563 A) and Na D (5890 A).  This
    function compares the RMS inside a 20-A window around each line to the
    RMS in adjacent continuum windows.

    Parameters
    ----------
    smooth_ratio_arr : 2D ndarray, shape (n_fibres, n_wave)
        Smoothed ratio curves from ``compute_fibre_flat_twilight`` Pass 1.
    wave_arr : 2D ndarray, shape (n_fibres, n_wave)
        Corresponding wavelength arrays.
    line_to_cont_threshold : float, optional
        Maximum allowed ratio of in-line RMS to continuum RMS (default 2.0).

    Returns
    -------
    dict
        ``{'halpha_line_to_cont': float, 'nad_line_to_cont': float,
           'passed': bool}``
        If a line falls outside the wavelength coverage the corresponding
        value is ``np.nan`` and does not affect ``passed``.
    """
    LINES = {'halpha': 6563.0, 'nad': 5890.0}
    HALF_LINE_WIN  = 10.0   # +/- 10 A around line centre
    CONT_OFFSET    = 50.0   # continuum window starts 50 A from line centre
    HALF_CONT_WIN  = 20.0   # +/- 20 A around continuum centre

    results = {}

    for line_name, line_centre in LINES.items():
        line_rms_vals = []
        cont_rms_vals = []

        for i in range(smooth_ratio_arr.shape[0]):
            wave = wave_arr[i]
            ratio = smooth_ratio_arr[i]

            valid = np.isfinite(wave) & np.isfinite(ratio)
            if np.sum(valid) < 10:
                continue

            w = wave[valid]
            r = ratio[valid]

            # Check line falls in coverage
            if line_centre < w.min() or line_centre > w.max():
                continue

            # In-line window
            in_line = np.abs(w - line_centre) < HALF_LINE_WIN
            # Continuum window (blue side)
            cont_centre = line_centre - CONT_OFFSET
            in_cont = np.abs(w - cont_centre) < HALF_CONT_WIN

            if np.sum(in_line) < 3 or np.sum(in_cont) < 3:
                continue

            line_rms_vals.append(float(np.std(r[in_line])))
            cont_rms_vals.append(float(np.std(r[in_cont])))

        if line_rms_vals and cont_rms_vals:
            median_line = float(np.median(line_rms_vals))
            median_cont = float(np.median(cont_rms_vals))
            ratio_val = (median_line / median_cont
                         if median_cont > 0 else np.nan)
        else:
            ratio_val = np.nan

        results[f'{line_name}_line_to_cont'] = ratio_val

    halpha_ok = (np.isnan(results.get('halpha_line_to_cont', np.nan))
                 or results['halpha_line_to_cont'] <= line_to_cont_threshold)
    nad_ok    = (np.isnan(results.get('nad_line_to_cont', np.nan))
                 or results['nad_line_to_cont'] <= line_to_cont_threshold)
    passed    = halpha_ok and nad_ok

    if not passed:
        for key, val in results.items():
            if not np.isnan(val) and val > line_to_cont_threshold:
                _logger.warning(
                    f"qa_fraunhofer_residuals: {key} = {val:.2f} "
                    f"(threshold {line_to_cont_threshold:.1f}) — "
                    "consider increasing savgol_window in "
                    "compute_fibre_flat_twilight()")

    results['passed'] = passed
    return results
