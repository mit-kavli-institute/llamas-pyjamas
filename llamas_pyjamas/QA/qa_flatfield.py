import numpy as np
from scipy import ndimage, stats
from sklearn.decomposition import PCA
import warnings


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
