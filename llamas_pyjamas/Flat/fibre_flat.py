"""Fibre-to-fibre flat fielding for LLAMAS RSS spectra.

This module computes and applies wavelength-dependent fibre-to-fibre throughput
corrections to extracted RSS FITS files, following MUSE/WEAVE practice:

    correction[i, lambda] = flat_ref(lambda) / flat_i(lambda)

where the reference is fibre #150 from bench 4A (the pipeline-wide reference,
matching Arc/arcLlamas.py:108 and Arc/arcLlamasMulti.py:562).

Usage
-----
Standalone::

    from llamas_pyjamas.Flat.fibre_flat import run_fibre_flat
    results = run_fibre_flat(
        science_rss_path='science_RSS_red.fits',
        flat_rss_path='flat_RSS_red.fits',
        output_dir='/output/',
    )
    # Produces: science_RSS.fits (original copy) and science_RSSFF.fits (corrected)

Object-oriented::

    from llamas_pyjamas.Flat.fibre_flat import FibreFlatField
    ff = FibreFlatField('flat_RSS_red.fits')
    ff.compute()
    rss_path, rssff_path = ff.apply('science_RSS_red.fits', output_dir='/output/')
"""

import os
import logging
from datetime import datetime

import numpy as np
from scipy.ndimage import median_filter
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from llamas_pyjamas.Utils.utils import setup_logger


# ---------------------------------------------------------------------------
# Pipeline-wide reference fibre (matches Arc/arcLlamas.py:108-112 and
# Arc/arcLlamasMulti.py:562-573).  Bench 4A, fibre index 150.
# ---------------------------------------------------------------------------
REFERENCE_FIBRE    = 150
REFERENCE_BENCH    = '4'
REFERENCE_SIDE     = 'A'
REFERENCE_BENCHSIDE = '4A'   # Value stored in FIBERMAP BENCHSIDE column


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_reference_row(fibermap_table, bench='4', side='A', fibre_id=150):
    """Return the RSS row index of the pipeline reference fibre.

    Searches the FIBERMAP binary table for the row matching
    *benchside* (bench+side concatenated) and *fibre_id*.

    Parameters
    ----------
    fibermap_table : astropy BinTableHDU data
        The FIBERMAP extension data from an RSS FITS file.
    bench : str
        Bench identifier (default '4').
    side : str
        Side identifier (default 'A').
    fibre_id : int
        Fibre index within the bench/side (default 150).

    Returns
    -------
    int
        Zero-based row index into the RSS flux array.

    Raises
    ------
    ValueError
        If the reference fibre is not found in the FIBERMAP.
    """
    benchside = f"{bench}{side}"
    benchside_col = np.array(fibermap_table['BENCHSIDE'], dtype=str)
    fiber_id_col  = np.array(fibermap_table['FIBER_ID'],  dtype=int)

    mask = (benchside_col == benchside) & (fiber_id_col == fibre_id)
    indices = np.where(mask)[0]
    if len(indices) == 0:
        raise ValueError(
            f"Reference fibre {benchside}:{fibre_id} not found in FIBERMAP. "
            f"Available bench-sides: {np.unique(benchside_col).tolist()}"
        )
    return int(indices[0])


def _resolve_reference_row(fibermap_table, reference_fibre):
    """Resolve the reference row from a user-supplied override or pipeline default.

    Parameters
    ----------
    fibermap_table : astropy BinTableHDU data
    reference_fibre : None, int, or (bench, side, fibre_id) tuple
        - None  → use pipeline default (bench 4A, fibre 150)
        - int   → treat as a direct RSS row index
        - tuple → (bench, side, fibre_id) passed to get_reference_row()

    Returns
    -------
    int
        RSS row index of the reference fibre.
    """
    if reference_fibre is None:
        return get_reference_row(fibermap_table, REFERENCE_BENCH, REFERENCE_SIDE, REFERENCE_FIBRE)
    if isinstance(reference_fibre, int):
        return reference_fibre
    if isinstance(reference_fibre, (tuple, list)) and len(reference_fibre) == 3:
        bench, side, fid = reference_fibre
        return get_reference_row(fibermap_table, str(bench), str(side), int(fid))
    raise TypeError(
        f"reference_fibre must be None, an int, or a (bench, side, fibre_id) tuple; got {type(reference_fibre)}"
    )


def clean_output_filename(science_filepath, suffix, output_dir=None):
    """Generate a cleaned output filename for RSS products.

    Strips any existing ``_RSS`` or ``_RSSFF`` suffix from the input basename
    before appending the requested *suffix*.

    Parameters
    ----------
    science_filepath : str
        Path to the input science RSS FITS file.
    suffix : str
        Output suffix, e.g. ``'_RSS'`` or ``'_RSSFF'``.
    output_dir : str, optional
        Output directory.  Defaults to the directory of *science_filepath*.

    Returns
    -------
    str
        Full output path.

    Examples
    --------
    >>> clean_output_filename('obs_RSS_red.fits', '_RSSFF')
    'obs_RSSFF_red.fits'
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(science_filepath))

    basename = os.path.splitext(os.path.basename(science_filepath))[0]

    # Strip any existing _RSSFF or _RSS suffix to avoid double-appending
    for strip in ('_RSSFF', '_RSS'):
        if basename.endswith(strip):
            basename = basename[:-len(strip)]
            break

    return os.path.join(output_dir, f"{basename}{suffix}.fits")


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------

def load_rss(filepath, logger=None):
    """Load a single-channel RSS FITS file into memory.

    Parameters
    ----------
    filepath : str
        Path to the RSS FITS file.
    logger : logging.Logger, optional

    Returns
    -------
    dict with keys:
        flux, error, mask, wave, fwhm  — 2D float32/int16 arrays (n_fibres × n_wave)
        fibermap                        — astropy BinTableHDU data
        header                          — primary header
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info(f"Loading RSS file: {filepath}")
    with fits.open(filepath) as hdul:
        flux    = hdul['FLUX'].data.astype(np.float64)
        error   = hdul['ERROR'].data.astype(np.float64)
        mask    = hdul['MASK'].data.astype(np.int16)
        wave    = hdul['WAVE'].data.astype(np.float64)
        fwhm    = hdul['FWHM'].data.astype(np.float64)
        fibermap = hdul['FIBERMAP'].data
        header   = hdul[0].header.copy()

    n_fibres, n_wave = flux.shape
    logger.info(f"  Shape: {n_fibres} fibres × {n_wave} wavelength pixels")

    # Warn about pathological fibres
    all_zero = np.all(flux == 0, axis=1)
    all_nan  = np.all(np.isnan(flux), axis=1)
    if np.any(all_zero):
        logger.warning(f"  {np.sum(all_zero)} all-zero fibre(s) found")
    if np.any(all_nan):
        logger.warning(f"  {np.sum(all_nan)} all-NaN fibre(s) found")

    return {
        'flux': flux,
        'error': error,
        'mask': mask,
        'wave': wave,
        'fwhm': fwhm,
        'fibermap': fibermap,
        'header': header,
    }


def compute_fibre_flat(flat_flux, reference_row_idx, mask=None,
                       smooth_kernel=15, clip_sigma=3.0, min_throughput=0.1,
                       logger=None):
    """Compute wavelength-dependent fibre-to-fibre flat-field correction.

    For each fibre *i*::

        correction[i, lambda] = smoothed_ref[lambda] / smoothed_flat[i, lambda]

    where *smoothed_ref* is the flat spectrum of the reference fibre (bench 4A,
    fibre 150) after optional median filtering.

    Parameters
    ----------
    flat_flux : 2D ndarray, shape (n_fibres, n_wave)
        Extracted flat-field spectra (FLUX extension of flat RSS).
    reference_row_idx : int
        RSS row index of the reference fibre (from :func:`get_reference_row`).
    mask : 2D bool array, optional
        True where input flat data is bad.
    smooth_kernel : int or None
        Median filter kernel width along wavelength axis (pixels).
        Set to None to disable smoothing.
    clip_sigma : float
        Sigma-clipping threshold for outlier rejection across fibres.
    min_throughput : float
        Fibres with median flux below this fraction of the reference median
        are flagged as dead.
    logger : logging.Logger, optional

    Returns
    -------
    correction : 2D ndarray, shape (n_fibres, n_wave)
        Multiplicative correction.  ``correction[reference_row_idx, :] == 1.0``
        by construction.  Dead-fibre rows are NaN.
    dead_fibre_mask : 1D bool array, shape (n_fibres,)
        True for fibres flagged as dead/unreliable.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    n_fibres, n_wave = flat_flux.shape
    logger.info(f"Computing fibre-flat correction: {n_fibres} fibres, reference row {reference_row_idx}")

    # --- Step 1: Optional smoothing along wavelength axis ---
    if smooth_kernel is not None and smooth_kernel > 1:
        smoothed = median_filter(flat_flux, size=(1, smooth_kernel), mode='reflect')
        logger.debug(f"Applied median filter, kernel={smooth_kernel} pixels")
    else:
        smoothed = flat_flux.copy()

    # --- Step 2: Dead fibre detection (before ratio) ---
    ref_median = np.nanmedian(flat_flux[reference_row_idx, :])
    fibre_medians = np.nanmedian(flat_flux, axis=1)
    dead_fibre_mask = fibre_medians < min_throughput * ref_median
    n_dead = np.sum(dead_fibre_mask)
    if n_dead > 0:
        logger.warning(f"Flagged {n_dead} dead fibre(s) (throughput < {min_throughput:.0%} of reference)")

    # --- Step 3: Compute ratios ---
    ref_spectrum = smoothed[reference_row_idx, :].copy()
    correction = np.ones((n_fibres, n_wave), dtype=np.float64)

    bad_pixel_count = 0
    for i in range(n_fibres):
        if dead_fibre_mask[i]:
            correction[i, :] = np.nan
            continue
        denom = smoothed[i, :]
        bad = (denom <= 0) | np.isnan(denom)
        if mask is not None:
            bad |= mask[i, :]
        good = ~bad
        if np.sum(good) == 0:
            correction[i, :] = np.nan
            dead_fibre_mask[i] = True
            logger.warning(f"Fibre {i}: no valid pixels, flagging as dead")
            continue
        correction[i, good] = ref_spectrum[good] / denom[good]
        correction[i, bad] = 1.0   # neutral correction for bad pixels
        bad_pixel_count += np.sum(bad)

    if bad_pixel_count > 0:
        logger.debug(f"Set {bad_pixel_count} bad-pixel correction values to 1.0")

    # --- Step 4: Sigma-clip outliers along fibre axis per wavelength column ---
    valid_corr = correction.copy()
    valid_corr[dead_fibre_mask] = np.nan
    col_median = np.nanmedian(valid_corr, axis=0)            # shape (n_wave,)
    col_mad    = np.nanmedian(np.abs(valid_corr - col_median[np.newaxis, :]), axis=0)
    col_mad    = np.where(col_mad == 0, 1e-6, col_mad)       # avoid divide-by-zero
    outlier    = np.abs(valid_corr - col_median[np.newaxis, :]) > clip_sigma * col_mad
    # Only clip alive fibres
    clip_mask  = outlier & ~dead_fibre_mask[:, np.newaxis]
    n_clipped  = int(np.sum(clip_mask))
    if n_clipped > 0:
        logger.debug(f"Replaced {n_clipped} sigma-clipped correction pixels with column median")
        # Replace clipped positions with the per-column median (broadcast safely)
        correction = np.where(clip_mask, np.broadcast_to(col_median, correction.shape), correction)

    # --- Step 5: Force reference row to exactly 1.0 ---
    correction[reference_row_idx, :] = 1.0

    # Summary statistics
    valid_vals = correction[~dead_fibre_mask]
    valid_vals = valid_vals[~np.isnan(valid_vals)]
    if len(valid_vals) > 0:
        logger.info(
            f"Correction range: [{np.nanmin(valid_vals):.4f}, {np.nanmax(valid_vals):.4f}], "
            f"median={np.nanmedian(valid_vals):.4f}"
        )

    return correction, dead_fibre_mask


def apply_fibre_flat(science_flux, science_error, science_mask, correction,
                     dead_fibre_mask=None):
    """Apply fibre-to-fibre flat-field correction to science spectra.

    Parameters
    ----------
    science_flux : 2D ndarray (n_fibres, n_wave)
    science_error : 2D ndarray (n_fibres, n_wave)  — sigma values (NOT IVAR)
    science_mask : 2D int16 array (n_fibres, n_wave) — 0=good
    correction : 2D ndarray (n_fibres, n_wave)
        Multiplicative correction from :func:`compute_fibre_flat`.
    dead_fibre_mask : 1D bool array (n_fibres,), optional

    Returns
    -------
    corr_flux : 2D ndarray
    corr_error : 2D ndarray
    corr_mask : 2D int16 array
    """
    corr_flux  = science_flux  * correction
    corr_error = science_error * np.abs(correction)    # σ_out = σ_in × |correction|
    corr_mask  = science_mask.copy()

    # Flag pixels where correction is NaN
    nan_corr = np.isnan(correction)
    corr_mask[nan_corr] |= 4   # bit flag 4 = flat-field correction unavailable

    # NaN-out dead fibres
    if dead_fibre_mask is not None:
        corr_flux[dead_fibre_mask, :]  = np.nan
        corr_error[dead_fibre_mask, :] = np.nan
        corr_mask[dead_fibre_mask, :]  |= 8   # bit flag 8 = dead fibre

    return corr_flux.astype(np.float32), corr_error.astype(np.float32), corr_mask


# ---------------------------------------------------------------------------
# FITS I/O
# ---------------------------------------------------------------------------

def _write_rss_fits(template_hdul, out_path, flux=None, error=None, mask=None,
                    extra_header_kw=None):
    """Write an RSS FITS file from a template HDUList.

    Copies all extensions; optionally replaces FLUX/ERROR/MASK data and
    adds extra header keywords to the primary HDU.

    Parameters
    ----------
    template_hdul : astropy HDUList (open, read-only is fine)
    out_path : str
    flux, error, mask : ndarray or None
        If provided, replace the corresponding extension data.
    extra_header_kw : dict, optional
        Header keyword/value/comment triples to add to extension 0.
        Format: {keyword: (value, comment)} or {keyword: value}.
    """
    new_hdul = fits.HDUList()
    for ext in template_hdul:
        new_hdul.append(ext.copy())

    if flux  is not None: new_hdul['FLUX'].data  = flux.astype(np.float32)
    if error is not None: new_hdul['ERROR'].data = error.astype(np.float32)
    if mask  is not None: new_hdul['MASK'].data  = mask.astype(np.int16)

    if extra_header_kw:
        hdr = new_hdul[0].header
        for kw, val in extra_header_kw.items():
            if isinstance(val, tuple):
                hdr[kw] = val   # (value, comment)
            else:
                hdr[kw] = val

    new_hdul.writeto(out_path, overwrite=True)


# ---------------------------------------------------------------------------
# QA plots
# ---------------------------------------------------------------------------

def plot_fibre_flat_qa(correction, dead_mask, wavelength, output_path, channel=''):
    """Generate a 4-panel QA plot of the fibre-flat correction.

    Parameters
    ----------
    correction : 2D ndarray (n_fibres, n_wave)
    dead_mask  : 1D bool array (n_fibres,)
    wavelength : 1D ndarray (n_wave,) — representative wavelength array
    output_path : str  — save path (.png)
    channel : str      — channel name for plot titles
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Fibre-to-Fibre Flat Correction QA  ({channel})', fontsize=13)

    n_fibres, n_wave = correction.shape
    disp_corr = correction.copy()
    disp_corr[dead_mask] = np.nan

    # Panel 1: 2D correction image
    ax = axes[0, 0]
    vmin, vmax = 0.8, 1.2
    im = ax.imshow(disp_corr, aspect='auto', origin='lower',
                   vmin=vmin, vmax=vmax, cmap='RdBu_r',
                   extent=[wavelength[0], wavelength[-1], 0, n_fibres])
    plt.colorbar(im, ax=ax, label='Correction factor')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Fibre row')
    ax.set_title('Correction array (fibres × wavelength)')

    # Panel 2: Histogram at mid-channel wavelength
    ax = axes[0, 1]
    mid = n_wave // 2
    mid_vals = disp_corr[:, mid]
    mid_vals = mid_vals[~np.isnan(mid_vals)]
    ax.hist(mid_vals, bins=40, color='steelblue', edgecolor='white', linewidth=0.4)
    ax.axvline(np.nanmedian(mid_vals), color='red',    ls='--', lw=1.5, label='Median')
    ax.axvline(1.0,                    color='black',  ls=':',  lw=1.0, label='Unity')
    ax.set_xlabel('Correction factor')
    ax.set_ylabel('Count')
    ax.set_title(f'Correction histogram at λ={wavelength[mid]:.0f} Å')
    ax.legend(fontsize=8)

    # Panel 3: Sample correction spectra (every 20th fibre + reference row)
    ax = axes[1, 0]
    ref_rows = list(range(0, n_fibres, max(1, n_fibres // 20)))
    for r in ref_rows:
        ax.plot(wavelength, disp_corr[r, :], lw=0.6, alpha=0.7)
    ax.axhline(1.0, color='black', ls=':', lw=0.8)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Correction factor')
    ax.set_title('Sample fibre correction spectra (every ~20th fibre)')
    ax.set_ylim(0.5, 1.5)

    # Panel 4: Per-fibre median correction
    ax = axes[1, 1]
    per_fibre_median = np.nanmedian(disp_corr, axis=1)
    ax.plot(per_fibre_median, np.arange(n_fibres), lw=0.8, color='steelblue')
    ax.axvline(1.0, color='black', ls=':', lw=0.8)
    dead_idx = np.where(dead_mask)[0]
    if len(dead_idx):
        ax.scatter(np.ones(len(dead_idx)), dead_idx, color='red', s=10,
                   zorder=5, label=f'Dead ({len(dead_idx)})')
        ax.legend(fontsize=8)
    ax.set_xlabel('Median correction')
    ax.set_ylabel('Fibre row')
    ax.set_title('Per-fibre median correction')

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def plot_before_after_qa(science_flux, corrected_flux, wavelength, output_path, channel=''):
    """Generate a 2-panel before/after comparison QA plot.

    Parameters
    ----------
    science_flux : 2D ndarray (n_fibres, n_wave)
    corrected_flux : 2D ndarray (n_fibres, n_wave)
    wavelength : 1D ndarray
    output_path : str
    channel : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'Fibre-to-Fibre Flat: Before vs After  ({channel})', fontsize=12)

    # Panel 1: Median spectrum
    ax = axes[0]
    med_before = np.nanmedian(science_flux,    axis=0)
    med_after  = np.nanmedian(corrected_flux,  axis=0)
    ax.plot(wavelength, med_before, lw=0.8, label='Before', alpha=0.8)
    ax.plot(wavelength, med_after,  lw=0.8, label='After',  alpha=0.8)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Median flux (ADU)')
    ax.set_title('Median spectrum across fibres')
    ax.legend(fontsize=9)

    # Panel 2: Fibre-to-fibre scatter (std across fibres)
    ax = axes[1]
    std_before = np.nanstd(science_flux,   axis=0)
    std_after  = np.nanstd(corrected_flux, axis=0)
    ax.plot(wavelength, std_before, lw=0.8, label='Before', alpha=0.8)
    ax.plot(wavelength, std_after,  lw=0.8, label='After',  alpha=0.8)
    ax.set_xlabel('Wavelength (Å)')
    ax.set_ylabel('Std across fibres (ADU)')
    ax.set_title('Fibre-to-fibre scatter')
    ax.legend(fontsize=9)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_fibre_flat(science_rss_path, flat_rss_path,
                   output_dir=None, smooth_kernel=15,
                   clip_sigma=3.0, min_throughput=0.1,
                   reference_fibre=None, generate_qa=True):
    """Run the full fibre-to-fibre flat-field correction pipeline.

    Loads a flat RSS file, computes the per-fibre correction, and applies it
    to one or more science RSS files.  Writes both a straight copy
    (``*_RSS.fits``) and a corrected version (``*_RSSFF.fits``).

    Parameters
    ----------
    science_rss_path : str or list of str
        Path(s) to science RSS FITS file(s).
    flat_rss_path : str
        Path to the flat-field RSS FITS file (same channel as science).
    output_dir : str, optional
        Output directory.  Defaults to same directory as each science file.
    smooth_kernel : int or None
        Median filter kernel width (pixels along wavelength).  Default 15.
    clip_sigma : float
        Sigma-clipping threshold for outlier rejection.  Default 3.0.
    min_throughput : float
        Dead-fibre threshold (fraction of reference median).  Default 0.1.
    reference_fibre : None, int, or (bench, side, fibre_id) tuple
        Override for the reference fibre.  None → bench 4A, fibre 150.
    generate_qa : bool
        Generate QA PNG plots.  Default True.

    Returns
    -------
    list of (str, str)
        List of ``(rss_path, rssff_path)`` tuples, one per science file.
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logger(__name__, f'fibre_flat_{timestamp}.log')
    logger.info("=" * 60)
    logger.info("FIBRE-TO-FIBRE FLAT FIELDING")
    logger.info("=" * 60)

    # Normalise to list
    if isinstance(science_rss_path, str):
        science_rss_path = [science_rss_path]

    # --- Load flat RSS ---
    logger.info(f"Loading flat RSS: {flat_rss_path}")
    flat_data = load_rss(flat_rss_path, logger=logger)
    fibermap   = flat_data['fibermap']

    # --- Resolve reference row ---
    ref_row = _resolve_reference_row(fibermap, reference_fibre)
    fibermap_benchsides = np.array(fibermap['BENCHSIDE'], dtype=str)
    fibermap_fiber_ids  = np.array(fibermap['FIBER_ID'],  dtype=int)
    ref_benchside = fibermap_benchsides[ref_row]
    ref_fiber_id  = fibermap_fiber_ids[ref_row]
    logger.info(f"Reference fibre: row {ref_row} (bench-side {ref_benchside}, fibre {ref_fiber_id})")

    # --- Representative wavelength array (row 0 of flat, fallback to index) ---
    wave_arr = flat_data['wave']
    ref_wave = wave_arr[ref_row, :]
    if np.all(np.isnan(ref_wave)):
        valid_rows = ~np.all(np.isnan(wave_arr), axis=1)
        ref_wave = wave_arr[valid_rows][0] if np.any(valid_rows) else np.arange(wave_arr.shape[1])

    # --- Compute correction ---
    correction, dead_mask = compute_fibre_flat(
        flat_data['flux'],
        reference_row_idx=ref_row,
        mask=(flat_data['mask'] > 0),
        smooth_kernel=smooth_kernel,
        clip_sigma=clip_sigma,
        min_throughput=min_throughput,
        logger=logger,
    )

    # Determine channel name for logging/QA
    channel = str(flat_data['header'].get('CHANNEL', '')).upper()

    results = []

    for sci_path in science_rss_path:
        logger.info(f"\nProcessing science file: {sci_path}")

        # Determine output directory for this file
        file_out_dir = output_dir if output_dir is not None else os.path.dirname(os.path.abspath(sci_path))
        os.makedirs(file_out_dir, exist_ok=True)

        rss_path   = clean_output_filename(sci_path, '_RSS',   file_out_dir)
        rssff_path = clean_output_filename(sci_path, '_RSSFF', file_out_dir)

        # Load science RSS
        sci_data = load_rss(sci_path, logger=logger)

        # Validate shape consistency
        if sci_data['flux'].shape != flat_data['flux'].shape:
            logger.error(
                f"Shape mismatch: science {sci_data['flux'].shape} vs flat {flat_data['flux'].shape}. "
                "Skipping this file."
            )
            continue

        # Apply correction
        corr_flux, corr_error, corr_mask = apply_fibre_flat(
            sci_data['flux'],
            sci_data['error'],
            sci_data['mask'],
            correction,
            dead_fibre_mask=dead_mask,
        )

        # --- Write _RSS.fits (straight copy of input) ---
        with fits.open(sci_path) as hdul:
            _write_rss_fits(hdul, rss_path)
        logger.info(f"Written original RSS copy: {rss_path}")

        # --- Write _RSSFF.fits (corrected) ---
        extra_kw = {
            'FFREF':  (int(ref_fiber_id),                 'Reference fibre index within bench-side'),
            'FFBENCH': (str(ref_benchside),               'Reference bench-side for flat-field'),
            'FFFILE': (os.path.basename(flat_rss_path),   'Flat RSS file used'),
            'FFMETH': ('fibre_ratio',                     'Fibre flat correction method'),
        }

        with fits.open(sci_path) as hdul:
            _write_rss_fits(
                hdul, rssff_path,
                flux=corr_flux, error=corr_error, mask=corr_mask,
                extra_header_kw=extra_kw,
            )
        # Add HISTORY entries
        with fits.open(rssff_path, mode='update') as hdul:
            hdul[0].header.add_history(
                f'Fibre-to-fibre flat applied from {os.path.basename(flat_rss_path)}'
            )
            hdul[0].header.add_history(
                f'Reference: bench-side {ref_benchside}, fibre {ref_fiber_id}'
            )
            if smooth_kernel:
                hdul[0].header.add_history(f'Flat smoothing kernel: {smooth_kernel} px')
        logger.info(f"Written flat-fielded RSS: {rssff_path}")

        # --- QA plots ---
        if generate_qa:
            qa_dir = os.path.join(file_out_dir, 'QA')
            os.makedirs(qa_dir, exist_ok=True)
            sci_base = os.path.splitext(os.path.basename(sci_path))[0]
            for strip in ('_RSSFF', '_RSS'):
                if sci_base.endswith(strip):
                    sci_base = sci_base[:-len(strip)]
                    break

            qa_flat_path = os.path.join(qa_dir, f'{sci_base}_fibre_flat_correction_{channel.lower()}.png')
            plot_fibre_flat_qa(correction, dead_mask, ref_wave, qa_flat_path, channel=channel)
            logger.info(f"QA plot: {qa_flat_path}")

            qa_ba_path = os.path.join(qa_dir, f'{sci_base}_fibre_flat_before_after_{channel.lower()}.png')
            plot_before_after_qa(sci_data['flux'], corr_flux, ref_wave, qa_ba_path, channel=channel)
            logger.info(f"QA plot: {qa_ba_path}")

        results.append((rss_path, rssff_path))

    logger.info(f"\nFibre-flat complete. Processed {len(results)} science file(s).")
    return results


# ---------------------------------------------------------------------------
# Object-oriented interface
# ---------------------------------------------------------------------------

class FibreFlatField:
    """Fibre-to-fibre flat-field correction, OO interface.

    Mirrors the style of :class:`~llamas_pyjamas.Flat.flatPypeit.PypeItFlatField`.

    Parameters
    ----------
    flat_rss_path : str
        Path to the flat-field RSS FITS file.
    reference_fibre : None, int, or (bench, side, fibre_id) tuple
        Override for the reference fibre.  None → bench 4A, fibre 150.
    smooth_kernel : int or None
        Median filter width (pixels).  Default 15.
    clip_sigma : float
        Sigma-clipping threshold.  Default 3.0.
    min_throughput : float
        Dead-fibre threshold.  Default 0.1.
    generate_qa : bool
        Write QA plots on :meth:`apply`.  Default True.
    """

    def __init__(self, flat_rss_path, reference_fibre=None, smooth_kernel=15,
                 clip_sigma=3.0, min_throughput=0.1, generate_qa=True):
        self.flat_rss_path   = flat_rss_path
        self.reference_fibre = reference_fibre
        self.smooth_kernel   = smooth_kernel
        self.clip_sigma      = clip_sigma
        self.min_throughput  = min_throughput
        self.generate_qa     = generate_qa

        self.correction      = None
        self.dead_mask       = None
        self.reference_row   = None
        self._flat_data      = None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = setup_logger(__name__, f'FibreFlatField_{timestamp}.log')

    def compute(self):
        """Load the flat RSS and compute the correction array.

        Populates ``self.correction``, ``self.dead_mask``, and
        ``self.reference_row``.
        """
        self._flat_data = load_rss(self.flat_rss_path, logger=self.logger)
        self.reference_row = _resolve_reference_row(
            self._flat_data['fibermap'], self.reference_fibre
        )
        self.correction, self.dead_mask = compute_fibre_flat(
            self._flat_data['flux'],
            reference_row_idx=self.reference_row,
            mask=(self._flat_data['mask'] > 0),
            smooth_kernel=self.smooth_kernel,
            clip_sigma=self.clip_sigma,
            min_throughput=self.min_throughput,
            logger=self.logger,
        )

    def apply(self, science_rss_path, output_dir=None):
        """Apply the precomputed correction to a science RSS file.

        Calls :meth:`compute` automatically if not yet done.

        Parameters
        ----------
        science_rss_path : str
        output_dir : str, optional

        Returns
        -------
        tuple of (str, str)
            ``(rss_path, rssff_path)``
        """
        if self.correction is None:
            self.compute()

        results = run_fibre_flat(
            science_rss_path=science_rss_path,
            flat_rss_path=self.flat_rss_path,
            output_dir=output_dir,
            smooth_kernel=self.smooth_kernel,
            clip_sigma=self.clip_sigma,
            min_throughput=self.min_throughput,
            reference_fibre=self.reference_fibre,
            generate_qa=self.generate_qa,
        )
        return results[0] if results else (None, None)
