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
from datetime import datetime
from llamas_pyjamas.Utils.utils import setup_logger

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

    corrections = {}
    channel_refs = {}

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

        # Find matching twilight extraction
        twi_ext = twi_lookup.get(ext_name)
        if twi_ext is None:
            logger.warning(f"No twilight extraction for {ext_name}, "
                           f"falling back to lamp-only for this benchside")
            # Fall through to lamp-only for this benchside
            twi_ext = None

        # ── Step 1: Compute per-benchside reference S̄_bs ──
        common_wave, reference = _compute_benchside_reference(
            smooth_arr, wave_arr)
        if len(reference) == 0:
            logger.warning(f"Could not compute reference for {ext_name}")
            continue

        n_fibres, n_pix = smooth_arr.shape
        corr_arr = np.ones_like(smooth_arr)

        # ── Step 2: Compute W_i(λ) from lamp ──
        w_arr = np.ones_like(smooth_arr)  # wavelength shape
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

            # Normalise W_i to mean=1.0 (isolate shape, remove amplitude)
            valid_wi = wi[valid]
            if len(valid_wi) > 0:
                wi_mean = np.nanmean(valid_wi)
                if wi_mean > 0:
                    wi /= wi_mean

            w_arr[i] = wi

        # ── Step 3: Compute T_i from twilight ──
        if twi_ext is not None:
            twi_counts = twi_ext.counts
            twi_wave = twi_ext.wave
            dead_set = set(getattr(twi_ext, 'dead_fibers', []) or [])
            total_twi = twi_counts.shape[0]

            # Handle wave/counts shape mismatch: counts is expanded for dead
            # fibres (array_index == physical FIBER_ID) but wave is NOT
            # expanded (sequential alive-fibre rows only).
            n_wave_rows = twi_wave.shape[0] if twi_wave is not None else 0
            wave_expanded = (n_wave_rows == total_twi)
            if not wave_expanded and n_wave_rows > 0:
                # Build mapping: physical FIBER_ID → wave row index
                # Wave rows are sequential alive fibres (skipping dead IDs)
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
                fid_to_wave_row = None  # direct indexing OK

            # Determine integration window
            if integration_range and channel in integration_range:
                w_lo, w_hi = integration_range[channel]
            else:
                # Default: central 80% of wavelength range
                all_valid_w = wave_arr[np.isfinite(wave_arr) & (wave_arr > 0)]
                if len(all_valid_w) > 0:
                    w_range = np.max(all_valid_w) - np.min(all_valid_w)
                    w_lo = np.min(all_valid_w) + 0.1 * w_range
                    w_hi = np.max(all_valid_w) - 0.1 * w_range
                else:
                    w_lo, w_hi = 0, np.inf

            # Integrate twilight per fibre
            integrals = {}
            for fid in fiber_ids:
                if fid >= total_twi or fid in dead_set:
                    continue

                # Look up the correct wave row for this FIBER_ID
                if fid_to_wave_row is not None:
                    wave_idx = fid_to_wave_row.get(fid)
                    if wave_idx is None:
                        continue
                else:
                    wave_idx = fid

                tw = twi_wave[wave_idx] if twi_wave is not None else None
                tc = twi_counts[fid]

                if tw is not None and np.any(np.isfinite(tw)):
                    mask = np.isfinite(tw) & (tw >= w_lo) & (tw <= w_hi) & (tc > 0)
                    if np.sum(mask) > 10:
                        integrals[fid] = np.nansum(tc[mask])

            if len(integrals) > 0:
                median_integral = np.median(list(integrals.values()))
                if median_integral > 0:
                    t_i = {fid: val / median_integral
                           for fid, val in integrals.items()}
                else:
                    t_i = {fid: 1.0 for fid in integrals}
                logger.info(f"  {ext_name}: T_i range "
                            f"[{min(t_i.values()):.3f}, {max(t_i.values()):.3f}] "
                            f"from {len(t_i)} fibres")
            else:
                logger.warning(f"  {ext_name}: no valid twilight integrals, "
                               f"using T_i=1.0")
                t_i = {}
        else:
            t_i = {}

        # ── Step 4: Combine C_i = T_i × W_i ──
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

    _cross_bench_diagnostic(channel_refs)

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
