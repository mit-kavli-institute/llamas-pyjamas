"""Fibre-to-fibre flat fielding for LLAMAS.

Computes and applies relative throughput corrections per fibre using either:
- **Twilight + Lamp** (Branch A): spatial throughput from a twilight flat,
  wavelength-dependent shape from lamp smooth models.
- **Lamp-only** (Branch B): direct ratio of each fibre's smooth model to the
  per-benchside median reference.

The correction ``C_i(λ)`` is stored in a multi-extension FITS file and
applied to science RSS files by dividing the FLUX and ERROR arrays.
"""

import os
import logging
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from datetime import datetime
from numpy.polynomial.polynomial import polyvander2d
from llamas_pyjamas.Image.WhiteLightModule import FiberMap_LUT
logger = logging.getLogger(__name__)

# Bounds for the correction factor — prevents catastrophic edge divisions
CORRECTION_CLAMP = (0.2, 5.0)

# Minimum reference flux to apply correction (below this → C_i = 1.0)
MIN_REF_FLUX = 100.0


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def _load_smooth_models(smooth_models_file):
    """Load the smooth models FITS into a dict keyed by extension name.

    Returns
    -------
    models : dict
        ``{ext_name: {'fiber_ids': array, 'smooth': 2D, 'wave': 2D,
        'channel': str, 'bench': str, 'side': str}}``
    """
    models = {}
    with fits.open(smooth_models_file) as hdul:
        for ext in hdul[1:]:
            ext_name = ext.header['EXTNAME']
            models[ext_name] = {
                'fiber_ids': ext.data['FIBER_ID'].copy(),
                'smooth': ext.data['SMOOTH'].copy(),
                'wave': ext.data['WAVE'].copy(),
                'channel': ext.header['CHANNEL'],
                'bench': ext.header['BENCH'],
                'side': ext.header['SIDE'],
            }
    logger.info(f"Loaded smooth models from {smooth_models_file}: "
                f"{len(models)} extensions")
    return models


def _compute_benchside_reference(smooth_arr, wave_arr):
    """Compute the median reference spectrum for one benchside.

    Resamples all fibres onto a common wavelength grid, takes the
    per-wavelength median, then returns both the common grid and the
    reference spectrum.

    Parameters
    ----------
    smooth_arr : ndarray, shape (n_fibres, n_pix)
        Smooth model per fibre.
    wave_arr : ndarray, shape (n_fibres, n_pix)
        Wavelength grid per fibre.

    Returns
    -------
    common_wave : ndarray, shape (n_common,)
        Common wavelength grid.
    reference : ndarray, shape (n_common,)
        Median reference spectrum on the common grid.
    """
    # Determine common grid from the union of all fibre wavelength ranges
    valid_waves = wave_arr[np.isfinite(wave_arr) & (wave_arr > 0)]
    if len(valid_waves) == 0:
        logger.warning("No valid wavelength data for benchside reference")
        return np.array([]), np.array([])

    w_min = np.min(valid_waves)
    w_max = np.max(valid_waves)

    # Median wavelength step across all fibres
    steps = []
    for i in range(wave_arr.shape[0]):
        w = wave_arr[i]
        good = np.isfinite(w) & (w > 0)
        if np.sum(good) > 10:
            dw = np.median(np.abs(np.diff(w[good])))
            if dw > 0:
                steps.append(dw)
    if not steps:
        return np.array([]), np.array([])

    dw_common = np.median(steps)
    n_common = int(np.ceil((w_max - w_min) / dw_common)) + 1
    common_wave = np.linspace(w_min, w_max, n_common)

    # Resample each fibre onto the common grid
    resampled = np.full((smooth_arr.shape[0], n_common), np.nan)
    for i in range(smooth_arr.shape[0]):
        w = wave_arr[i]
        s = smooth_arr[i]
        good = np.isfinite(w) & np.isfinite(s) & (w > 0) & (s > 0)
        if np.sum(good) > 10:
            resampled[i] = np.interp(common_wave, w[good], s[good],
                                     left=np.nan, right=np.nan)

    reference = np.nanmedian(resampled, axis=0)
    return common_wave, reference


def _write_corrections_fits(corrections, output_path, method, header_extra=None):
    """Write fibre flat corrections to a multi-extension FITS file.

    For each benchside, writes a BinTableHDU with FIBER_ID, CORRECTION,
    and WAVE columns, followed by an ImageHDU with the reference spectrum.

    Parameters
    ----------
    corrections : dict
        ``{ext_name: {'fiber_ids': array, 'correction': 2D, 'wave': 2D,
        'channel': str, 'bench': str, 'side': str, 'n_ref': int,
        'common_wave': 1D, 'reference': 1D}}``
    output_path : str
        Output FITS path.
    method : str
        'twilight' or 'lamp_only'.
    header_extra : dict, optional
        Extra keywords for the primary header.
    """
    primary = fits.PrimaryHDU()
    primary.header['DATE'] = (datetime.now().isoformat(), 'File creation date')
    primary.header['METHOD'] = (method, 'Fibre flat method')
    primary.header['HISTORY'] = f'Fibre-to-fibre flat corrections ({method})'
    if header_extra:
        for k, v in header_extra.items():
            primary.header[k] = v
    hdul = fits.HDUList([primary])

    for ext_name in sorted(corrections.keys()):
        data = corrections[ext_name]
        fiber_ids = data['fiber_ids']
        corr = data['correction']
        wave = data['wave']
        naxis1 = corr.shape[1] if corr.ndim == 2 else 0

        if len(fiber_ids) == 0:
            continue

        # BinTable: per-fibre corrections on native grid
        cols = [
            fits.Column(name='FIBER_ID', format='J', array=fiber_ids),
            fits.Column(name='CORRECTION', format=f'{naxis1}D', array=corr),
            fits.Column(name='WAVE', format=f'{naxis1}D', array=wave),
        ]
        tbl = fits.BinTableHDU.from_columns(cols)
        tbl.header['EXTNAME'] = ext_name
        tbl.header['CHANNEL'] = data['channel']
        tbl.header['BENCH'] = data['bench']
        tbl.header['SIDE'] = data['side']
        tbl.header['NREF'] = (data['n_ref'], 'Fibres used for median reference')
        tbl.header['METHOD'] = method
        hdul.append(tbl)

        # ImageHDU: benchside reference spectrum on common grid
        common_wave = data.get('common_wave', np.array([]))
        reference = data.get('reference', np.array([]))
        if len(reference) > 0:
            ref_hdu = fits.ImageHDU(data=reference.astype(np.float64))
            ref_hdu.header['EXTNAME'] = f'{ext_name}_REF'
            ref_hdu.header['CRVAL1'] = (common_wave[0], 'Start wavelength (A)')
            dw = common_wave[1] - common_wave[0] if len(common_wave) > 1 else 1.0
            ref_hdu.header['CDELT1'] = (dw, 'Wavelength step (A)')
            ref_hdu.header['CRPIX1'] = (1, 'Reference pixel')
            ref_hdu.header['CTYPE1'] = 'WAVE'
            ref_hdu.header['CUNIT1'] = 'Angstrom'
            hdul.append(ref_hdu)

    hdul.writeto(output_path, overwrite=True)
    logger.info(f"Fibre flat corrections written: {output_path} "
                f"({len(hdul) - 1} extensions, method={method})")
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Branch B: Lamp-only fallback
# ──────────────────────────────────────────────────────────────────────────────

def compute_fibre_flat_lamp_only(smooth_models_file, output_dir):
    """Compute fibre-to-fibre flat from lamp smooth models only (fallback).

    For each benchside, computes the per-benchside median reference
    ``S̄_bs(λ)`` and forms ``C_i(λ) = smooth_i(λ) / S̄_bs(λ)``.

    Parameters
    ----------
    smooth_models_file : str
        Path to ``flat_smooth_models.fits``.
    output_dir : str
        Directory for the output corrections file.

    Returns
    -------
    corrections_file : str
        Path to ``fibre_flat_corrections.fits``.
    """
    logger.warning(
        "No twilight flat provided. Falling back to lamp-only fibre-to-fibre "
        "flat. Science cubes will contain artificial spatial illumination "
        "gradients from the lamp."
    )

    models = _load_smooth_models(smooth_models_file)
    corrections = {}

    # Group by channel for cross-bench diagnostics
    channel_refs = {}  # {channel: [(ext_name, common_wave, reference), ...]}

    for ext_name, data in models.items():
        channel = data['channel']
        smooth_arr = data['smooth']
        wave_arr = data['wave']
        fiber_ids = data['fiber_ids']

        if len(fiber_ids) == 0:
            logger.warning(f"No fibres for {ext_name}, skipping")
            continue

        # Compute per-benchside median reference
        common_wave, reference = _compute_benchside_reference(smooth_arr, wave_arr)
        if len(reference) == 0:
            logger.warning(f"Could not compute reference for {ext_name}, skipping")
            continue

        # Interpolate reference back onto each fibre's native grid
        # and compute C_i = smooth_i / S̄_bs (single interpolation)
        n_fibres, n_pix = smooth_arr.shape
        corr_arr = np.ones_like(smooth_arr)

        n_alive = 0
        for i in range(n_fibres):
            w = wave_arr[i]
            s = smooth_arr[i]
            good = np.isfinite(w) & (w > 0)
            if np.sum(good) < 10:
                continue
            n_alive += 1

            # Interpolate reference onto this fibre's native grid
            ref_native = np.interp(w, common_wave, reference,
                                   left=np.nan, right=np.nan)

            # Compute correction where reference is reliable
            valid = (np.isfinite(ref_native) & (ref_native > MIN_REF_FLUX)
                     & np.isfinite(s) & (s > 0))
            corr_arr[i, valid] = s[valid] / ref_native[valid]
            corr_arr[i, ~valid] = 1.0

        # Clamp to safe range
        corr_arr = np.clip(corr_arr, CORRECTION_CLAMP[0], CORRECTION_CLAMP[1])

        corrections[ext_name] = {
            'fiber_ids': fiber_ids,
            'correction': corr_arr,
            'wave': wave_arr,
            'channel': channel,
            'bench': data['bench'],
            'side': data['side'],
            'n_ref': n_alive,
            'common_wave': common_wave,
            'reference': reference,
        }

        # Collect for cross-bench diagnostics
        if channel not in channel_refs:
            channel_refs[channel] = []
        channel_refs[channel].append((ext_name, common_wave, reference))

        logger.info(f"  {ext_name}: C_i median={np.nanmedian(corr_arr):.4f} "
                    f"std={np.nanstd(corr_arr):.4f} "
                    f"n_ref={n_alive}")

    # Cross-bench diagnostic checks
    _cross_bench_diagnostic(channel_refs)

    # Write corrections FITS
    output_path = os.path.join(output_dir, 'fibre_flat_corrections.fits')
    _write_corrections_fits(corrections, output_path, method='lamp_only',
                            header_extra={'SMTHFILE': os.path.basename(
                                smooth_models_file)})
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Branch A: Twilight + Lamp
# ──────────────────────────────────────────────────────────────────────────────

def reduce_twilight_flat(twilight_file, p2p_map_file, trace_dir, arc_soln,
                         bias_file, output_dir):
    """Reduce a twilight flat: bias/P2P correct, extract, wavelength calibrate.

    Reuses existing pipeline infrastructure for each step.

    Parameters
    ----------
    twilight_file : str
        Path to raw twilight flat FITS (MEF).
    p2p_map_file : str
        Path to pixel-to-pixel flat map (``pixel_maps.fits``).
    trace_dir : str
        Directory containing trace pickle files.
    arc_soln : object or None
        Arc solution dict, or None to use reference arc.
    bias_file : str or None
        Path to bias file.
    output_dir : str
        Output directory for intermediates.

    Returns
    -------
    extraction_dict : dict
        Calibrated extraction dict with keys ``'extractions'``,
        ``'metadata'``, ``'primary_header'``.
    """
    from llamas_pyjamas.reduce import (apply_flat_field_correction,
                                       run_extraction, correct_wavelengths)

    logger.info(f"Reducing twilight flat: {twilight_file}")
    twi_output_dir = os.path.join(output_dir, 'twilight')
    os.makedirs(twi_output_dir, exist_ok=True)

    # Step 1: Apply P2P pixel map to twilight frame
    logger.info("Twilight: applying P2P pixel map correction")
    corrected_file, _ = apply_flat_field_correction(
        twilight_file, [p2p_map_file], twi_output_dir)
    if corrected_file is None:
        raise RuntimeError("Failed to apply P2P correction to twilight flat")
    logger.info(f"Twilight P2P-corrected: {corrected_file}")

    # Step 2: Extract using lamp traces
    logger.info("Twilight: extracting 1D spectra using lamp traces")
    extraction_file = run_extraction(
        corrected_file, twi_output_dir,
        use_bias=bias_file, trace_dir=trace_dir)
    logger.info(f"Twilight extraction: {extraction_file}")

    # Step 3: Wavelength calibration
    logger.info("Twilight: applying wavelength calibration")
    extraction_path = os.path.join(twi_output_dir, extraction_file)
    extraction_dict, primary_hdr = correct_wavelengths(
        extraction_path, soln=arc_soln)
    logger.info("Twilight wavelength calibration complete")

    return extraction_dict


def _remove_twilight_gradient(integrals_dict, channel, poly_order=2):
    """Remove large-scale spatial gradient from twilight fibre integrals.

    The twilight sky has a spatial gradient (Rayleigh scattering, strongest
    in blue).  Because LLAMAS routes different IFU spatial blocks to
    different benches, this gradient appears as bench-to-bench offsets in
    the raw integrals.

    This function reconstructs the integrals as a 2D IFU image, fits a
    low-order polynomial surface, and divides by the model so that only
    true fibre-to-fibre throughput variations remain.

    Parameters
    ----------
    integrals_dict : dict
        ``{(benchside_str, fiber_id): raw_integral}`` — all fibres across
        all benchsides for one colour channel.
    channel : str
        Channel name (for logging).
    poly_order : int, optional
        Polynomial order for the 2D surface fit (default 2).

    Returns
    -------
    corrected : dict
        Same structure as *integrals_dict* with gradient removed.
    diagnostics : dict or None
        Diagnostic data for saving/plotting, or None if gradient
        removal was skipped.
    """
    if not integrals_dict:
        logger.warning(f"Twilight gradient removal for {channel}: "
                       f"no integrals provided, skipping")
        return integrals_dict, None

    # ── 1. Map fibres to IFU positions ──
    keys = []
    x_list = []
    y_list = []
    flux_list = []
    for (bs, fid), integral in integrals_dict.items():
        x, y = FiberMap_LUT(bs, fid)
        if x == -1 and y == -1:
            logger.debug(f"  {channel}: fibre ({bs}, {fid}) not in LUT, "
                         f"skipping for gradient fit")
            continue
        keys.append((bs, fid))
        x_list.append(float(x))
        y_list.append(float(y))
        flux_list.append(float(integral))

    n_total = len(keys)
    if n_total < 10:
        logger.warning(f"Twilight gradient removal for {channel}: "
                       f"too few fibres ({n_total}) for gradient fit, "
                       f"skipping gradient removal")
        return integrals_dict, None

    x_arr = np.array(x_list)
    y_arr = np.array(y_list)
    flux_arr = np.array(flux_list)

    # ── 2. Sigma-clip outliers ──
    clipped = sigma_clip(flux_arr, sigma=3, maxiters=2)
    good = ~clipped.mask
    n_clipped = int(np.sum(clipped.mask))

    if np.sum(good) < 10:
        logger.warning(f"Twilight gradient removal for {channel}: "
                       f"too few fibres ({np.sum(good)}) after sigma-clip, "
                       f"skipping gradient removal")
        return integrals_dict, None

    # ── 3. Normalise coordinates to [-1, 1] for numerical stability ──
    x_min, x_max = x_arr[good].min(), x_arr[good].max()
    y_min, y_max = y_arr[good].min(), y_arr[good].max()
    x_span = x_max - x_min if x_max > x_min else 1.0
    y_span = y_max - y_min if y_max > y_min else 1.0
    x_norm = 2.0 * (x_arr - x_min) / x_span - 1.0
    y_norm = 2.0 * (y_arr - y_min) / y_span - 1.0

    # ── 4. Fit 2D polynomial surface ──
    # polyvander2d builds the Vandermonde matrix for terms up to
    # x^poly_order * y^poly_order.  For order=2: 1, x, y, x², xy, y²,
    # plus cross terms x²y, xy², x²y² which we trim via column selection
    # to keep only terms with total degree <= poly_order.
    vander_full = polyvander2d(x_norm[good], y_norm[good],
                               [poly_order, poly_order])
    # Select columns where total degree <= poly_order
    col_mask = []
    for ix in range(poly_order + 1):
        for iy in range(poly_order + 1):
            if ix + iy <= poly_order:
                col_mask.append(ix * (poly_order + 1) + iy)
    vander = vander_full[:, col_mask]

    coeffs, residuals, rank, sv = np.linalg.lstsq(vander, flux_arr[good],
                                                    rcond=None)

    # ── 5. Evaluate model at ALL fibre positions ──
    vander_all_full = polyvander2d(x_norm, y_norm,
                                    [poly_order, poly_order])
    vander_all = vander_all_full[:, col_mask]
    model = vander_all @ coeffs

    # ── 6. Divide integrals by model ──
    corrected = {}
    for i, key in enumerate(keys):
        if model[i] > 0:
            corrected[key] = flux_arr[i] / model[i]
        else:
            logger.warning(f"  {channel}: gradient model <= 0 at fibre "
                           f"{key}, keeping raw integral")
            corrected[key] = flux_arr[i]

    # Re-add any fibres that were skipped (not in LUT)
    for key, val in integrals_dict.items():
        if key not in corrected:
            corrected[key] = val

    # ── 7. Log diagnostics ──
    coeff_labels = []
    for ix in range(poly_order + 1):
        for iy in range(poly_order + 1):
            if ix + iy <= poly_order:
                if ix == 0 and iy == 0:
                    coeff_labels.append('const')
                elif ix == 1 and iy == 0:
                    coeff_labels.append('x')
                elif ix == 0 and iy == 1:
                    coeff_labels.append('y')
                elif ix == 2 and iy == 0:
                    coeff_labels.append('x2')
                elif ix == 1 and iy == 1:
                    coeff_labels.append('xy')
                elif ix == 0 and iy == 2:
                    coeff_labels.append('y2')
                else:
                    coeff_labels.append(f'x{ix}y{iy}')

    # Normalise coefficients for display (relative to constant term)
    norm_coeffs = coeffs / coeffs[0] if coeffs[0] != 0 else coeffs
    coeff_str = ', '.join(f'{lbl}={c:+.4f}' for lbl, c in
                          zip(coeff_labels, norm_coeffs))

    model_good = model[good]
    ptp = (model_good.max() - model_good.min()) / np.mean(model_good) * 100

    scatter_before = np.std(flux_arr) / np.median(flux_arr) * 100
    corr_values = np.array([corrected[k] for k in keys])
    scatter_after = np.std(corr_values) / np.median(corr_values) * 100

    logger.info(f"Twilight gradient removal for {channel} channel:")
    logger.info(f"  Fitted surface (order {poly_order}): {coeff_str}")
    logger.info(f"  Gradient peak-to-peak: {ptp:.1f}% of mean")
    logger.info(f"  Sigma-clipped {n_clipped}/{n_total} fibres")
    logger.info(f"  Scatter before: {scatter_before:.1f}%, "
                f"after: {scatter_after:.1f}%")

    # Normalise corrected integrals so their median = median of originals
    # (preserves absolute scale for downstream normalisation)
    orig_median = np.median(flux_arr)
    corr_median = np.median(corr_values)
    if corr_median > 0:
        scale = orig_median / corr_median
        corrected = {k: v * scale for k, v in corrected.items()}
        corr_values = corr_values * scale

    diagnostics = {
        'x': x_arr,
        'y': y_arr,
        'raw': flux_arr,
        'model': model,
        'corrected': corr_values,
        'keys': keys,
        'coeffs': coeffs,
        'coeff_labels': coeff_labels,
        'poly_order': poly_order,
        'good': good,
        'ptp_pct': ptp,
        'scatter_before': scatter_before,
        'scatter_after': scatter_after,
        'n_clipped': n_clipped,
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
    }

    return corrected, diagnostics


def _save_gradient_model(gradient_diagnostics, output_dir):
    """Save the twilight gradient model to a multi-extension FITS file.

    Parameters
    ----------
    gradient_diagnostics : dict
        ``{channel: diagnostics_dict}`` from ``_remove_twilight_gradient()``.
    output_dir : str
        Directory for the output file.
    """
    output_path = os.path.join(output_dir, 'twilight_gradient_model.fits')

    primary = fits.PrimaryHDU()
    primary.header['DATE'] = (datetime.now().isoformat(), 'File creation date')
    primary.header['NCHAN'] = (len(gradient_diagnostics), 'Number of channels')
    primary.header['HISTORY'] = 'Twilight gradient model for fibre flat fielding'
    hdul = fits.HDUList([primary])

    for channel, diag in gradient_diagnostics.items():
        poly_order = diag['poly_order']
        coeffs = diag['coeffs']
        coeff_labels = diag['coeff_labels']

        # Normalise coefficients for display
        norm_coeffs = coeffs / coeffs[0] if coeffs[0] != 0 else coeffs

        # ── Coefficient table ──
        max_label_len = max(len(lbl) for lbl in coeff_labels)
        cols = [
            fits.Column(name='LABEL', format=f'{max_label_len}A',
                        array=np.array(coeff_labels)),
            fits.Column(name='VALUE', format='D', array=coeffs),
            fits.Column(name='NORM_VALUE', format='D', array=norm_coeffs),
        ]
        coeff_tbl = fits.BinTableHDU.from_columns(cols)
        coeff_tbl.header['EXTNAME'] = f'{channel.upper()}_COEFF'
        coeff_tbl.header['CHANNEL'] = channel
        coeff_tbl.header['POLYORD'] = (poly_order, 'Polynomial order')
        coeff_tbl.header['PTP_PCT'] = (diag['ptp_pct'],
                                       'Gradient peak-to-peak %')
        coeff_tbl.header['SCATBEF'] = (diag['scatter_before'],
                                       'Scatter before gradient removal %')
        coeff_tbl.header['SCATAFT'] = (diag['scatter_after'],
                                       'Scatter after gradient removal %')
        coeff_tbl.header['NCLIP'] = (diag['n_clipped'],
                                     'Number of sigma-clipped fibres')
        coeff_tbl.header['NFIBRES'] = (len(diag['x']),
                                       'Total fibres in fit')
        hdul.append(coeff_tbl)

        # ── Per-fibre data table ──
        keys = diag['keys']
        benchsides = np.array([k[0] for k in keys])
        fiber_ids = np.array([k[1] for k in keys], dtype=np.int32)
        max_bs_len = max(len(bs) for bs in benchsides) if len(benchsides) > 0 else 3

        cols = [
            fits.Column(name='BENCHSIDE', format=f'{max_bs_len}A',
                        array=benchsides),
            fits.Column(name='FIBER_ID', format='J', array=fiber_ids),
            fits.Column(name='X', format='D', array=diag['x']),
            fits.Column(name='Y', format='D', array=diag['y']),
            fits.Column(name='RAW', format='D', array=diag['raw']),
            fits.Column(name='MODEL', format='D', array=diag['model']),
            fits.Column(name='CORRECTED', format='D', array=diag['corrected']),
            fits.Column(name='USED_IN_FIT', format='L',
                        array=diag['good'].astype(bool)),
        ]
        fibre_tbl = fits.BinTableHDU.from_columns(cols)
        fibre_tbl.header['EXTNAME'] = f'{channel.upper()}_FIBRES'
        fibre_tbl.header['CHANNEL'] = channel
        hdul.append(fibre_tbl)

    hdul.writeto(output_path, overwrite=True)
    logger.info(f"Twilight gradient model written: {output_path} "
                f"({len(hdul) - 1} extensions)")


def _plot_fibre_flat_diagnostic(gradient_diagnostics, t_i_all, models,
                                channel_groups, output_dir):
    """Generate a multi-panel IFU hex-map diagnostic plot.

    Layout: one row per channel, 4 columns:
    1. Raw twilight integrals
    2. Fitted gradient surface
    3. Corrected integrals
    4. T_i per fibre

    Parameters
    ----------
    gradient_diagnostics : dict
        ``{channel: diagnostics_dict}`` from ``_remove_twilight_gradient()``.
    t_i_all : dict
        ``{ext_name: {fid: t_i_value}}`` from Pass 2.
    models : dict
        Smooth models dict (for fiber_ids and bench/side metadata).
    channel_groups : dict
        ``{channel: [ext_name, ...]}``
    output_dir : str
        Directory for the output plot.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import RegularPolygon
    from matplotlib.colors import Normalize

    channel_order = ['blue', 'green', 'red']
    channels = [ch for ch in channel_order if ch in gradient_diagnostics]
    if not channels:
        logger.warning("No gradient diagnostics to plot")
        return

    n_rows = len(channels)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    col_titles = ['Raw Twilight Integrals', 'Fitted Gradient Surface',
                  'Corrected Integrals', 'T_i (Spatial Throughput)']

    for row_idx, channel in enumerate(channels):
        diag = gradient_diagnostics[channel]
        x = diag['x']
        y = diag['y']
        raw = diag['raw']
        model_vals = diag['model']
        corrected = diag['corrected']
        good = diag['good']

        # Compute hex size from fibre spacing
        x_sorted = np.sort(np.unique(x))
        if len(x_sorted) > 1:
            hex_size = np.nanmedian(np.diff(x_sorted)) / 1.5
        else:
            hex_size = 0.5

        # Build T_i array at same positions
        keys = diag['keys']
        ti_vals = np.ones(len(keys))
        for i, (bs, fid) in enumerate(keys):
            # Find which ext_name this benchside belongs to
            for ext_name in channel_groups.get(channel, []):
                if (f"{models[ext_name]['bench']}{models[ext_name]['side']}"
                        == bs):
                    ti_vals[i] = t_i_all.get(ext_name, {}).get(fid, 1.0)
                    break

        # Data for each column
        datasets = [raw, model_vals, corrected, ti_vals]

        # Shared scale for columns 1 and 3 (raw & corrected)
        shared_vmin = min(np.nanmin(raw), np.nanmin(corrected))
        shared_vmax = max(np.nanmax(raw), np.nanmax(corrected))

        for col_idx, data in enumerate(datasets):
            ax = axes[row_idx, col_idx]

            if col_idx in (0, 2):
                # Sequential colormap, shared scale
                cmap = matplotlib.colormaps['viridis']
                vmin, vmax = shared_vmin, shared_vmax
            elif col_idx == 1:
                # Diverging around mean for gradient surface
                cmap = matplotlib.colormaps['RdBu_r']
                mean_val = np.nanmean(data)
                dev = max(abs(data.max() - mean_val),
                          abs(data.min() - mean_val))
                vmin, vmax = mean_val - dev, mean_val + dev
            else:
                # T_i: diverging around 1.0
                cmap = matplotlib.colormaps['RdBu_r']
                dev = max(abs(data.max() - 1.0), abs(data.min() - 1.0), 0.1)
                vmin, vmax = 1.0 - dev, 1.0 + dev

            norm = Normalize(vmin=vmin, vmax=vmax)

            for i in range(len(x)):
                if np.isnan(data[i]):
                    continue
                color = cmap(norm(data[i]))
                hex_patch = RegularPolygon(
                    (x[i], y[i]), numVertices=6, radius=hex_size,
                    orientation=np.pi / 6,
                    facecolor=color, edgecolor='none', linewidth=0.2)
                ax.add_patch(hex_patch)

            # Mark sigma-clipped fibres in column 1
            if col_idx == 0:
                clipped_mask = ~good
                if np.any(clipped_mask):
                    ax.scatter(x[clipped_mask], y[clipped_mask],
                               marker='x', c='red', s=15, linewidths=0.8,
                               zorder=5)

            ax.set_xlim(x.min() - 2 * hex_size, x.max() + 2 * hex_size)
            ax.set_ylim(y.min() - 2 * hex_size, y.max() + 2 * hex_size)
            ax.set_aspect('equal')

            # Colorbar
            sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
            if col_idx == 3:
                cbar.set_label('T_i')

            # Titles
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], fontsize=11)

            # Row label
            if col_idx == 0:
                ax.set_ylabel(f'{channel.upper()}\n'
                              f'scatter: {diag["scatter_before"]:.1f}%'
                              f' -> {diag["scatter_after"]:.1f}%',
                              fontsize=10)

            # Annotation for gradient column
            if col_idx == 1:
                coeff_summary = ', '.join(
                    f'{lbl}: {c / diag["coeffs"][0]:+.3f}'
                    for lbl, c in zip(diag['coeff_labels'][1:],
                                      diag['coeffs'][1:])
                    if lbl in ('x', 'y'))
                ax.text(0.02, 0.02,
                        f'PtP: {diag["ptp_pct"]:.1f}%\n{coeff_summary}',
                        transform=ax.transAxes, fontsize=8,
                        verticalalignment='bottom',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='white', alpha=0.8))

            ax.tick_params(labelsize=7)

    fig.suptitle('Fibre Flat Diagnostic: Twilight Gradient Removal',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()

    output_path = os.path.join(output_dir, 'fibre_flat_diagnostic.png')
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Fibre flat diagnostic plot saved: {output_path}")


def compute_fibre_flat_twilight(twilight_extractions, smooth_models_file,
                                output_dir, integration_range=None):
    """Compute fibre-to-fibre flat using twilight spatial + lamp wavelength shape.

    For each benchside:
    1. Integrate twilight flux per fibre to get spatial throughput ``T_i``.
    2. Compute wavelength shape ``W_i(λ) = smooth_i / S̄_bs``,
       normalised to mean 1.0.
    3. Combine: ``C_i(λ) = T_i × W_i(λ)``.

    Parameters
    ----------
    twilight_extractions : dict
        Calibrated twilight extractions (from ``reduce_twilight_flat``).
    smooth_models_file : str
        Path to ``flat_smooth_models.fits``.
    output_dir : str
        Directory for the output corrections file.
    integration_range : dict, optional
        Per-channel wavelength ranges for T_i integration:
        ``{'red': (w_min, w_max), ...}``.  Defaults to central 80%.

    Returns
    -------
    corrections_file : str
        Path to ``fibre_flat_corrections.fits``.
    """
    logger.info("Computing fibre flat: Twilight + Lamp method")

    models = _load_smooth_models(smooth_models_file)
    twi_extractions = twilight_extractions['extractions']
    twi_metadata = twilight_extractions['metadata']

    # Build twilight lookup: (channel, bench, side) → ExtractLlamas
    twi_lookup = {}
    for ext, meta in zip(twi_extractions, twi_metadata):
        key = f"{meta['channel']}_{meta['bench']}_{meta['side']}"
        twi_lookup[key] = ext

    # ── Group models by channel for cross-benchside T_i computation ──
    channel_groups = {}  # {channel: [ext_name, ...]}
    for ext_name, data in models.items():
        channel_groups.setdefault(data['channel'], []).append(ext_name)

    # ══════════════════════════════════════════════════════════════════
    # Pass 1: Gather raw twilight integrals across ALL benchsides
    #         per channel, then remove the spatial gradient.
    # ══════════════════════════════════════════════════════════════════
    channel_integrals = {}  # {channel: {(benchside_str, fid): integral}}

    for channel, ext_names in channel_groups.items():
        all_integrals = {}

        for ext_name in ext_names:
            data = models[ext_name]
            bench = data['bench']
            side = data['side']
            fiber_ids = data['fiber_ids']
            wave_arr = data['wave']
            benchside_str = f"{bench}{side}"

            if len(fiber_ids) == 0:
                continue

            twi_ext = twi_lookup.get(ext_name)
            if twi_ext is None:
                logger.warning(f"No twilight extraction for {ext_name}, "
                               f"using T_i=1.0 (lamp-only for this benchside)")
                continue

            twi_counts = twi_ext.counts
            twi_wave = twi_ext.wave
            dead_set = set(getattr(twi_ext, 'dead_fibers', []) or [])
            total_twi = twi_counts.shape[0]

            # Handle wave/counts shape mismatch (dead fibre expansion)
            n_wave_rows = twi_wave.shape[0] if twi_wave is not None else 0
            wave_expanded = (n_wave_rows == total_twi)
            if not wave_expanded and n_wave_rows > 0:
                fid_to_wave_row = {}
                wave_row = 0
                for phys_id in range(total_twi):
                    if phys_id in dead_set:
                        continue
                    if wave_row < n_wave_rows:
                        fid_to_wave_row[phys_id] = wave_row
                        wave_row += 1
                logger.debug(f"  {ext_name}: wave/counts shape mismatch "
                             f"(wave={n_wave_rows}, counts={total_twi}), "
                             f"mapped {len(fid_to_wave_row)} alive fibres")
            else:
                fid_to_wave_row = None

            # Determine integration window
            if integration_range and channel in integration_range:
                w_lo, w_hi = integration_range[channel]
            else:
                all_valid_w = wave_arr[np.isfinite(wave_arr) & (wave_arr > 0)]
                if len(all_valid_w) > 0:
                    w_range = np.max(all_valid_w) - np.min(all_valid_w)
                    w_lo = np.min(all_valid_w) + 0.1 * w_range
                    w_hi = np.max(all_valid_w) - 0.1 * w_range
                else:
                    w_lo, w_hi = 0, np.inf

            # Integrate twilight per fibre
            for fid in fiber_ids:
                if fid >= total_twi or fid in dead_set:
                    continue

                if fid_to_wave_row is not None:
                    wave_idx = fid_to_wave_row.get(fid)
                    if wave_idx is None:
                        continue
                else:
                    wave_idx = fid

                tw = twi_wave[wave_idx] if twi_wave is not None else None
                tc = twi_counts[fid]

                if tw is not None and np.any(np.isfinite(tw)):
                    mask = (np.isfinite(tw) & (tw >= w_lo)
                            & (tw <= w_hi) & (tc > 0))
                    if np.sum(mask) > 10:
                        all_integrals[(benchside_str, fid)] = np.nansum(
                            tc[mask])

        channel_integrals[channel] = all_integrals
        logger.info(f"  {channel}: gathered {len(all_integrals)} twilight "
                    f"integrals across "
                    f"{sum(1 for en in ext_names if twi_lookup.get(en) is not None)} "
                    f"benchsides")

    # ── Pass 1.5: Remove twilight spatial gradient for ALL channels ──
    gradient_diagnostics = {}  # {channel: diagnostics_dict}
    for channel, integrals in channel_integrals.items():
        if len(integrals) > 0:
            corrected, diag = _remove_twilight_gradient(integrals, channel,
                                                         poly_order=2)
            channel_integrals[channel] = corrected
            if diag is not None:
                gradient_diagnostics[channel] = diag

    # ══════════════════════════════════════════════════════════════════
    # Pass 2: Compute W_i (per benchside) and T_i (from gradient-
    #         corrected integrals, normalised per channel), then combine.
    # ══════════════════════════════════════════════════════════════════
    corrections = {}
    channel_refs = {}
    t_i_all = {}  # {ext_name: {fid: t_i_value}} for diagnostics

    for ext_name, data in models.items():
        channel = data['channel']
        bench = data['bench']
        side = data['side']
        smooth_arr = data['smooth']
        wave_arr = data['wave']
        fiber_ids = data['fiber_ids']

        if len(fiber_ids) == 0:
            logger.warning(f"No fibres for {ext_name}, skipping")
            continue

        # ── Compute per-benchside reference S̄_bs ──
        common_wave, reference = _compute_benchside_reference(
            smooth_arr, wave_arr)
        if len(reference) == 0:
            logger.warning(f"Could not compute reference for {ext_name}")
            continue

        n_fibres, n_pix = smooth_arr.shape
        corr_arr = np.ones_like(smooth_arr)

        # ── Compute W_i(λ) from lamp (unchanged) ──
        w_arr = np.ones_like(smooth_arr)
        n_alive = 0
        for i in range(n_fibres):
            w = wave_arr[i]
            s = smooth_arr[i]
            good = np.isfinite(w) & (w > 0)
            if np.sum(good) < 10:
                continue
            n_alive += 1

            ref_native = np.interp(w, common_wave, reference,
                                   left=np.nan, right=np.nan)
            valid = (np.isfinite(ref_native) & (ref_native > MIN_REF_FLUX)
                     & np.isfinite(s) & (s > 0))

            wi = np.ones_like(s)
            wi[valid] = s[valid] / ref_native[valid]
            wi[~valid] = 1.0

            valid_wi = wi[valid]
            if len(valid_wi) > 0:
                wi_mean = np.nanmean(valid_wi)
                if wi_mean > 0:
                    wi /= wi_mean

            w_arr[i] = wi

        # ── Compute T_i from gradient-corrected integrals ──
        benchside_str = f"{bench}{side}"
        channel_ints = channel_integrals.get(channel, {})
        this_bs_integrals = {fid: v for (bs, fid), v in channel_ints.items()
                             if bs == benchside_str}
        all_channel_values = list(channel_ints.values())

        if len(this_bs_integrals) > 0 and len(all_channel_values) > 0:
            median_integral = np.median(all_channel_values)
            if median_integral > 0:
                t_i = {fid: val / median_integral
                       for fid, val in this_bs_integrals.items()}
            else:
                t_i = {fid: 1.0 for fid in this_bs_integrals}
            logger.info(f"  {ext_name}: T_i range "
                        f"[{min(t_i.values()):.3f}, {max(t_i.values()):.3f}] "
                        f"from {len(t_i)} fibres")
        else:
            logger.warning(f"  {ext_name}: no valid twilight integrals, "
                           f"using T_i=1.0")
            t_i = {}

        t_i_all[ext_name] = t_i

        # ── Combine C_i = T_i × W_i ──
        fid_to_row = {fid: idx for idx, fid in enumerate(fiber_ids)}
        for fid in fiber_ids:
            idx = fid_to_row[fid]
            ti = t_i.get(fid, 1.0)
            corr_arr[idx] = ti * w_arr[idx]

        # Clamp
        corr_arr = np.clip(corr_arr, CORRECTION_CLAMP[0], CORRECTION_CLAMP[1])

        corrections[ext_name] = {
            'fiber_ids': fiber_ids,
            'correction': corr_arr,
            'wave': wave_arr,
            'channel': channel,
            'bench': bench,
            'side': side,
            'n_ref': n_alive,
            'common_wave': common_wave,
            'reference': reference,
        }

        if channel not in channel_refs:
            channel_refs[channel] = []
        channel_refs[channel].append((ext_name, common_wave, reference))

        logger.info(f"  {ext_name}: C_i median={np.nanmedian(corr_arr):.4f} "
                    f"std={np.nanstd(corr_arr):.4f}")

    # ── Pass 2.5: Cross-bench T_i diagnostics per channel ──
    for channel, ext_names in channel_groups.items():
        logger.info(f"Cross-bench T_i diagnostic for {channel} channel:")
        logger.info(f"  {'Benchside':<12} {'N_fib':>6} {'median(T_i)':>12} "
                    f"{'std(T_i)':>10} {'min':>8} {'max':>8}")
        all_ti_values = []
        for ext_name in ext_names:
            if ext_name not in t_i_all:
                continue
            bs_str = f"{models[ext_name]['bench']}{models[ext_name]['side']}"
            ti_dict = t_i_all[ext_name]
            ti_vals = [ti_dict.get(fid, 1.0)
                       for fid in models[ext_name]['fiber_ids']
                       if fid in ti_dict]
            if ti_vals:
                all_ti_values.extend(ti_vals)
                logger.info(
                    f"  {bs_str:<12} {len(ti_vals):>6} "
                    f"{np.median(ti_vals):>12.4f} "
                    f"{np.std(ti_vals):>10.4f} "
                    f"{np.min(ti_vals):>8.4f} "
                    f"{np.max(ti_vals):>8.4f}")
            else:
                logger.info(f"  {bs_str:<12}      0  (no twilight data)")

        if all_ti_values:
            logger.info(
                f"  {'CHANNEL':<12} {len(all_ti_values):>6} "
                f"{np.median(all_ti_values):>12.4f} "
                f"{np.std(all_ti_values):>10.4f} "
                f"{np.min(all_ti_values):>8.4f} "
                f"{np.max(all_ti_values):>8.4f}")

            # Flag benchsides deviating > 5% from 1.0
            for ext_name in ext_names:
                if ext_name not in t_i_all:
                    continue
                bs_str = (f"{models[ext_name]['bench']}"
                          f"{models[ext_name]['side']}")
                ti_dict = t_i_all[ext_name]
                ti_vals = [ti_dict.get(fid, 1.0)
                           for fid in models[ext_name]['fiber_ids']
                           if fid in ti_dict]
                if ti_vals and abs(np.median(ti_vals) - 1.0) > 0.05:
                    logger.warning(
                        f"  *** {bs_str} median T_i = "
                        f"{np.median(ti_vals):.4f} — "
                        f"deviates >5% from 1.0 ***")

    _cross_bench_diagnostic(channel_refs)

    # ── Save gradient model and diagnostic plot ──
    if gradient_diagnostics:
        _save_gradient_model(gradient_diagnostics, output_dir)
        _plot_fibre_flat_diagnostic(gradient_diagnostics, t_i_all, models,
                                    channel_groups, output_dir)

    output_path = os.path.join(output_dir, 'fibre_flat_corrections.fits')
    _write_corrections_fits(
        corrections, output_path, method='twilight',
        header_extra={'SMTHFILE': os.path.basename(smooth_models_file)})
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3: Application to science RSS
# ──────────────────────────────────────────────────────────────────────────────

def apply_fibre_flat_to_rss(rss_file, corrections_file, output_file=None):
    """Apply fibre-to-fibre flat correction to a science RSS file.

    Divides FLUX and ERROR by the correction factor ``C_i(λ)`` for each
    fibre, producing a new RSS file with ``_FF.fits`` suffix.

    Parameters
    ----------
    rss_file : str
        Path to input science RSS FITS file.
    corrections_file : str
        Path to ``fibre_flat_corrections.fits``.
    output_file : str, optional
        Output path.  Defaults to replacing ``.fits`` with ``_FF.fits``.

    Returns
    -------
    output_file : str
        Path to the corrected RSS file.
    """
    if output_file is None:
        base, ext = os.path.splitext(rss_file)
        output_file = f"{base}_FF{ext}"

    logger.info(f"Applying fibre flat: {os.path.basename(rss_file)} → "
                f"{os.path.basename(output_file)}")

    # Load corrections indexed by (ext_name) → {fiber_id → row_index}
    corr_lookup = {}  # ext_name → {'fiber_ids': array, 'correction': 2D, 'wave': 2D}
    with fits.open(corrections_file) as corr_hdul:
        corr_method = corr_hdul[0].header.get('METHOD', 'unknown')
        for ext in corr_hdul[1:]:
            if ext.header.get('EXTNAME', '').endswith('_REF'):
                continue  # Skip reference spectrum extensions
            ext_name = ext.header['EXTNAME']
            corr_lookup[ext_name] = {
                'fiber_ids': ext.data['FIBER_ID'].copy(),
                'correction': ext.data['CORRECTION'].copy(),
                'wave': ext.data['WAVE'].copy(),
            }

    with fits.open(rss_file) as hdul:
        channel = hdul[0].header.get('CHANNEL', 'unknown').lower()

        flux = hdul['FLUX'].data.copy()
        error = hdul['ERROR'].data.copy()
        fibermap = hdul['FIBERMAP'].data

        # Read WAVE for grid-mismatch fallback
        wave_data = hdul['WAVE'].data

        benchsides = np.array([
            bs.decode().strip() if isinstance(bs, bytes)
            else str(bs).strip()
            for bs in fibermap['BENCHSIDE']
        ])
        fiber_ids = fibermap['FIBER_ID']

        n_applied = 0
        n_skipped = 0

        for row_idx in range(flux.shape[0]):
            bs = benchsides[row_idx]
            fid = int(fiber_ids[row_idx])

            # Parse benchside "1A" → bench='1', side='A'
            bench = bs[:-1]
            side_ch = bs[-1]
            ext_name = f"{channel}_{bench}_{side_ch}"

            if ext_name not in corr_lookup:
                logger.debug(f"Row {row_idx}: no corrections for {ext_name}")
                n_skipped += 1
                continue

            corr_data = corr_lookup[ext_name]
            fid_arr = corr_data['fiber_ids']

            # Find the row in corrections matching this FIBER_ID
            match_mask = fid_arr == fid
            if not np.any(match_mask):
                logger.debug(f"Row {row_idx}: FIBER_ID {fid} not in "
                             f"corrections for {ext_name}")
                n_skipped += 1
                continue

            match_idx = np.where(match_mask)[0][0]
            c_i = corr_data['correction'][match_idx]

            # Check if wavelength grids match (same arc solution → should match)
            if len(c_i) != flux.shape[1]:
                # Fallback: interpolate correction onto science grid
                c_wave = corr_data['wave'][match_idx]
                sci_wave = wave_data[row_idx]
                c_i = np.interp(sci_wave, c_wave, c_i, left=1.0, right=1.0)

            # Apply correction: divide flux and error
            safe_c = np.where(np.isfinite(c_i) & (c_i > 0), c_i, 1.0)
            flux[row_idx] /= safe_c
            error[row_idx] /= safe_c
            n_applied += 1

        logger.info(f"Applied fibre flat to {n_applied}/{flux.shape[0]} fibres "
                    f"({n_skipped} skipped)")

        # Write output RSS with corrected flux/error
        out_hdul = fits.HDUList()

        # Copy primary with extra keywords
        primary = hdul[0].copy()
        primary.header['FIBRFLAT'] = (True, 'Fibre-to-fibre flat applied')
        primary.header['FFLATFIL'] = (os.path.basename(corrections_file),
                                      'Fibre flat corrections file')
        primary.header['FFMETHD'] = (corr_method, 'Fibre flat method')
        primary.header['HISTORY'] = (f'Fibre flat applied: {n_applied} fibres '
                                     f'corrected ({corr_method})')
        out_hdul.append(primary)

        # FLUX — corrected
        flux_hdu = fits.ImageHDU(flux, header=hdul['FLUX'].header.copy())
        flux_hdu.header['HISTORY'] = 'Fibre-to-fibre flat applied'
        out_hdul.append(flux_hdu)

        # ERROR — corrected
        error_hdu = fits.ImageHDU(error, header=hdul['ERROR'].header.copy())
        error_hdu.header['HISTORY'] = 'Fibre-to-fibre flat applied'
        out_hdul.append(error_hdu)

        # Copy remaining extensions unchanged (MASK, WAVE, FWHM, FIBERMAP)
        for ext_name_copy in ['MASK', 'WAVE', 'FWHM', 'FIBERMAP']:
            try:
                out_hdul.append(hdul[ext_name_copy].copy())
            except KeyError:
                pass

        out_hdul.writeto(output_file, overwrite=True)
        logger.info(f"FF RSS written: {output_file}")

    return output_file


# ──────────────────────────────────────────────────────────────────────────────
# Cross-bench diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def _cross_bench_diagnostic(channel_refs):
    """Compare reference spectra across benchsides within each channel.

    Logs RMS deviation between each benchside's reference and the
    channel-wide median.  Warns if any benchside deviates by > 5%.

    Parameters
    ----------
    channel_refs : dict
        ``{channel: [(ext_name, common_wave, reference), ...]}``
    """
    for channel, ref_list in channel_refs.items():
        if len(ref_list) < 2:
            continue

        # Resample all references onto the widest common grid
        all_waves = [w for _, w, _ in ref_list if len(w) > 0]
        if not all_waves:
            continue
        w_min = min(w.min() for w in all_waves)
        w_max = max(w.max() for w in all_waves)
        n_pts = max(len(w) for w in all_waves)
        grid = np.linspace(w_min, w_max, n_pts)

        resampled = []
        names = []
        for name, w, r in ref_list:
            if len(w) == 0 or len(r) == 0:
                continue
            rs = np.interp(grid, w, r, left=np.nan, right=np.nan)
            resampled.append(rs)
            names.append(name)

        if len(resampled) < 2:
            continue

        stack = np.array(resampled)
        channel_median = np.nanmedian(stack, axis=0)

        logger.info(f"Cross-bench diagnostic for {channel} channel "
                    f"({len(names)} benchsides):")
        for i, name in enumerate(names):
            valid = np.isfinite(stack[i]) & np.isfinite(channel_median) \
                    & (channel_median > MIN_REF_FLUX)
            if np.sum(valid) == 0:
                continue
            ratio = stack[i, valid] / channel_median[valid]
            rms = np.std(ratio) * 100
            bias = (np.median(ratio) - 1.0) * 100
            flag = " *** CHECK ***" if rms > 5.0 or abs(bias) > 10.0 else ""
            logger.info(f"  {name}: median offset={bias:+.2f}%  "
                        f"RMS={rms:.2f}%{flag}")
