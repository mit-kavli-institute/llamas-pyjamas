"""Fibre-to-fibre flat fielding for LLAMAS RSS spectra.

This module computes and applies wavelength-dependent fibre-to-fibre throughput
corrections to extracted RSS FITS files.  The correction is computed as:

    correction[i, lambda] = synthetic_reference[lambda] / flat_i[lambda]

where the synthetic reference is the median of the N fibres closest to the
centre of each detector bench-side (default N=50), selected using spatial
positions from LUT/LLAMAS_FiberMap_rev04.dat.  This avoids edge-illumination
effects and provides a stable, representative reference spectrum.

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
from scipy.signal import savgol_filter
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from llamas_pyjamas.Utils.utils import setup_logger


# ---------------------------------------------------------------------------
# Pipeline-wide reference fibre constants — kept for get_reference_row() and
# _resolve_reference_row() which are still used externally.
# ---------------------------------------------------------------------------
REFERENCE_FIBRE    = 150
REFERENCE_BENCH    = '4'
REFERENCE_SIDE     = 'A'
REFERENCE_BENCHSIDE = '4A'   # Value stored in FIBERMAP BENCHSIDE column

# ---------------------------------------------------------------------------
# Default parameters for the synthetic-reference flat algorithm
# ---------------------------------------------------------------------------
FF_SAVGOL_WINDOW    = 51   # Savitzky-Golay smoothing window (pixels along wavelength)
FF_SAVGOL_POLYORDER = 3    # Savitzky-Golay polynomial order
FF_N_CENTRAL_FIBRES = 50   # Fibres per bench-side closest to detector centre

# Path to the LLAMAS spatial fibre map (bench, fiber, xpos, ypos)
_FIBERMAP_LUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'LUT', 'LLAMAS_FiberMap_rev04.dat'
)


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _load_fibermap_lut(lut_path=None):
    """Load the LLAMAS spatial fibre map from LLAMAS_FiberMap_rev04.dat.

    Returns a dict keyed by bench-side string (e.g. ``'1A'``) mapping to a
    list of ``(fiber_id, xpos)`` tuples.

    Parameters
    ----------
    lut_path : str, optional
        Override path to the .dat file.  Defaults to ``_FIBERMAP_LUT_PATH``.

    Returns
    -------
    dict
        ``{benchside: [(fiber_id, xpos), ...]}``
    """
    path = lut_path or _FIBERMAP_LUT_PATH
    lut = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip blank lines, comments, and the header row
            if not line or line.startswith('#'):
                continue
            if line.startswith('|') and 'bench' in line.lower():
                continue
            # Format: | bench | fiber | xindex | yindex | xpos | ypos |
            parts = [p.strip() for p in line.strip('|').split('|')]
            if len(parts) < 5:
                continue
            try:
                bench_str = parts[0].strip()   # e.g. '1A'
                fiber_id  = int(parts[1])
                xpos      = float(parts[4])
                lut.setdefault(bench_str, []).append((fiber_id, xpos))
            except (ValueError, IndexError):
                continue
    return lut


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


def compute_fibre_flat(flat_flux, flat_err, flat_mask, fibermap,
                       savgol_window=FF_SAVGOL_WINDOW,
                       savgol_polyorder=FF_SAVGOL_POLYORDER,
                       clip_sigma=3.0, min_throughput=0.1,
                       n_central_fibres=FF_N_CENTRAL_FIBRES,
                       logger=None):
    """Compute wavelength-dependent fibre-to-fibre flat-field correction.

    Builds a synthetic reference spectrum from the *n_central_fibres* fibres
    closest to the detector centre on each bench-side (selected via
    ``LUT/LLAMAS_FiberMap_rev04.dat``), then computes a per-fibre correction::

        correction[i, lambda] = synthetic_reference[lambda] / flat_flux[i, lambda]

    The raw ratio is smoothed along the wavelength axis with a Savitzky-Golay
    filter to remove noise while preserving smooth throughput gradients.

    Parameters
    ----------
    flat_flux : 2D ndarray, shape (n_fibres, n_wave)
        Extracted flat-field spectra (FLUX extension of flat RSS).
    flat_err : 2D ndarray, shape (n_fibres, n_wave)
        Corresponding error spectra (currently unused in computation but
        accepted for API consistency and future use).
    flat_mask : 2D int16 array, shape (n_fibres, n_wave)
        Pixel mask (0 = good, non-zero = bad).
    fibermap : astropy BinTableHDU data
        FIBERMAP extension data with BENCHSIDE and FIBER_ID columns.
    savgol_window : int
        Savitzky-Golay smoothing window in pixels.  Must be odd.
    savgol_polyorder : int
        Savitzky-Golay polynomial order (< savgol_window).
    clip_sigma : float
        MAD sigma-clipping threshold for column-wise outlier rejection.
    min_throughput : float
        Fibres with median flux below this fraction of the global maximum
        median are flagged as dead.
    n_central_fibres : int
        Number of fibres per bench-side closest to the detector xpos centre
        used to build the synthetic reference.
    logger : logging.Logger, optional

    Returns
    -------
    correction : 2D ndarray, shape (n_fibres, n_wave)
        Multiplicative correction.  No NaN or inf values — dead-fibre rows
        are set to 1.0 (flagging is handled by :func:`apply_fibre_flat`).
    dead_fibre_mask : 1D bool array, shape (n_fibres,)
        True for fibres flagged as dead/unreliable.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    n_fibres, n_wave = flat_flux.shape
    logger.info(f"Computing fibre-flat correction (synthetic reference): "
                f"{n_fibres} fibres × {n_wave} wavelength pixels")

    # ---- Combine input mask with non-positive / NaN pixels ----
    bad_flat = (flat_mask > 0) | (flat_flux <= 0) | np.isnan(flat_flux)
    masked_flux = np.where(bad_flat, np.nan, flat_flux.astype(np.float64))

    # ---- Step A: Identify dead fibres ----
    fibre_medians = np.nanmedian(masked_flux, axis=1)       # shape (n_fibres,)
    max_median    = np.nanmax(fibre_medians)
    dead_fibre_mask = fibre_medians < min_throughput * max_median
    n_dead = int(np.sum(dead_fibre_mask))
    if n_dead > 0:
        logger.warning(f"  Flagged {n_dead} dead fibre(s) "
                       f"(median flux < {min_throughput:.0%} of max_median={max_median:.2f})")

    # ---- Step A: Build synthetic reference from central fibres ----
    # Load spatial fibermap LUT: {benchside: [(fiber_id, xpos), ...]}
    try:
        spatial_lut = _load_fibermap_lut()
    except Exception as exc:
        logger.warning(f"  Could not load spatial fibermap LUT: {exc}. "
                       "Falling back to all healthy fibres as reference.")
        spatial_lut = {}

    benchside_col  = np.array(fibermap['BENCHSIDE'], dtype=str)
    fiber_id_col   = np.array(fibermap['FIBER_ID'],  dtype=int)
    unique_benchsides = np.unique(benchside_col)

    central_rows = np.zeros(n_fibres, dtype=bool)

    for bs in unique_benchsides:
        bs_rss_rows = np.where(benchside_col == bs)[0]   # rows in the RSS for this bench-side
        if bs in spatial_lut and len(spatial_lut[bs]) > 0:
            fib_xpos = np.array(spatial_lut[bs])          # shape (M, 2): fiber_id, xpos
            xpos_vals = fib_xpos[:, 1]
            xpos_centre = np.median(xpos_vals)
            dist        = np.abs(xpos_vals - xpos_centre)
            sorted_idx  = np.argsort(dist)
            n_sel       = min(n_central_fibres, len(sorted_idx))
            central_fib_ids = set(int(fib_xpos[sorted_idx[k], 0]) for k in range(n_sel))
            # Map selected fiber IDs back to RSS rows for this bench-side
            for row in bs_rss_rows:
                if int(fiber_id_col[row]) in central_fib_ids:
                    central_rows[row] = True
            logger.debug(f"  Bench-side {bs}: selected {n_sel} central fibres "
                         f"(xpos_centre={xpos_centre:.1f})")
        else:
            # LUT missing for this bench-side — use all alive fibres on this bench-side
            for row in bs_rss_rows:
                central_rows[row] = True
            logger.warning(f"  Bench-side {bs}: not found in spatial LUT — "
                           "using all alive fibres as reference for this bench-side")

    # Restrict central rows to alive fibres with adequate signal
    central_alive = central_rows & ~dead_fibre_mask
    n_central_alive = int(np.sum(central_alive))
    logger.info(f"  Central healthy fibres contributing to synthetic reference: {n_central_alive}")

    if n_central_alive == 0:
        raise ValueError(
            "No central healthy fibres available to build synthetic reference. "
            "Check flat RSS data or lower min_throughput."
        )
    if n_central_alive < n_central_fibres // 2:
        logger.warning(f"  Only {n_central_alive} central fibres available "
                       f"(requested {n_central_fibres}) — reference may be noisy")

    # Synthetic reference: median over central alive fibres per wavelength
    synthetic_reference = np.nanmedian(masked_flux[central_alive, :], axis=0)  # (n_wave,)

    # Interpolate any remaining bad pixels in the reference
    ref_bad = (synthetic_reference <= 0) | np.isnan(synthetic_reference)
    if np.any(ref_bad):
        good_idx = np.where(~ref_bad)[0]
        bad_idx  = np.where(ref_bad)[0]
        if len(good_idx) > 0:
            synthetic_reference[bad_idx] = np.interp(
                bad_idx, good_idx, synthetic_reference[good_idx]
            )
            logger.debug(f"  Interpolated {len(bad_idx)} bad pixels in synthetic reference")
        else:
            raise ValueError("Synthetic reference is entirely bad/zero — cannot proceed.")

    # ---- Step C: Raw ratio per fibre ----
    correction = np.ones((n_fibres, n_wave), dtype=np.float64)

    for i in range(n_fibres):
        if dead_fibre_mask[i]:
            correction[i, :] = np.nan
            continue
        denom = masked_flux[i, :]
        bad   = np.isnan(denom) | (denom <= 0)
        good  = ~bad
        if np.sum(good) == 0:
            correction[i, :] = np.nan
            dead_fibre_mask[i] = True
            logger.warning(f"  Fibre {i}: no valid pixels after masking — flagging dead")
            continue
        correction[i, good] = synthetic_reference[good] / denom[good]
        correction[i, bad]  = np.nan   # will be clamped in Step E

    # ---- Step D: Savitzky-Golay smoothing of ratio per fibre ----
    # Ensure window is odd, within bounds, and larger than polyorder + 1
    actual_window = min(savgol_window, n_wave)
    if actual_window % 2 == 0:
        actual_window -= 1
    actual_window = max(actual_window, savgol_polyorder + 2)
    if actual_window % 2 == 0:
        actual_window += 1

    smoothed_correction = correction.copy()

    for i in range(n_fibres):
        if dead_fibre_mask[i]:
            smoothed_correction[i, :] = np.nan
            continue
        row      = correction[i, :]
        nan_mask = np.isnan(row)
        if np.all(nan_mask):
            smoothed_correction[i, :] = np.nan
            dead_fibre_mask[i] = True
            continue
        # Fill NaN gaps before smoothing (same pattern as flux_calibration.py:147-153)
        row_filled = row.copy()
        if np.any(nan_mask):
            valid_idx   = np.where(~nan_mask)[0]
            invalid_idx = np.where(nan_mask)[0]
            row_filled[invalid_idx] = np.interp(invalid_idx, valid_idx, row[valid_idx])
        row_smooth = savgol_filter(row_filled, actual_window, savgol_polyorder)
        row_smooth[nan_mask] = np.nan   # restore NaN at originally bad positions
        smoothed_correction[i, :] = row_smooth

    # ---- Step E: Clamp NaN/inf → 1.0 (neutral, un-flat-fielded) ----
    final_correction = smoothed_correction.copy()
    unflat = np.isnan(final_correction) | np.isinf(final_correction)
    final_correction[unflat] = 1.0
    # Dead fibre rows also set to 1.0; apply_fibre_flat handles NaN output via dead_fibre_mask
    final_correction[dead_fibre_mask] = 1.0

    # ---- Step F: Sigma-clip outliers per wavelength column (alive fibres only) ----
    alive      = ~dead_fibre_mask
    valid_corr = final_correction.copy()
    valid_corr[~alive] = np.nan
    col_median = np.nanmedian(valid_corr, axis=0)
    col_mad    = np.nanmedian(np.abs(valid_corr - col_median[np.newaxis, :]), axis=0)
    col_mad    = np.where(col_mad == 0, 1e-6, col_mad)
    outlier    = np.abs(valid_corr - col_median[np.newaxis, :]) > clip_sigma * col_mad
    clip_mask  = outlier & alive[:, np.newaxis]
    n_clipped  = int(np.sum(clip_mask))
    if n_clipped > 0:
        logger.debug(f"  Replaced {n_clipped} sigma-clipped correction values with column median")
        final_correction = np.where(
            clip_mask, np.broadcast_to(col_median, final_correction.shape), final_correction
        )

    # Summary statistics
    alive_vals = final_correction[alive]
    logger.info(
        f"  Correction range (alive fibres): "
        f"[{np.nanmin(alive_vals):.4f}, {np.nanmax(alive_vals):.4f}], "
        f"median={np.nanmedian(alive_vals):.4f}"
    )

    return final_correction, dead_fibre_mask


def apply_fibre_flat(sci_flux, sci_err, sci_mask, correction,
                     dead_fibre_mask=None):
    """Apply fibre-to-fibre flat-field correction to science spectra.

    Parameters
    ----------
    sci_flux : 2D ndarray (n_fibres, n_wave)
    sci_err : 2D ndarray (n_fibres, n_wave)  — sigma values (NOT IVAR)
    sci_mask : 2D int16 array (n_fibres, n_wave) — 0 = good
    correction : 2D ndarray (n_fibres, n_wave)
        Multiplicative correction from :func:`compute_fibre_flat`.
        Contains no NaN or inf (dead-fibre rows are 1.0 there).
    dead_fibre_mask : 1D bool array (n_fibres,), optional
        True for dead fibres (rows that should be NaN'd out).

    Returns
    -------
    corr_flux : 2D float32 ndarray
    corr_err : 2D float32 ndarray
    corr_mask : 2D int16 array

    Mask bit conventions
    --------------------
    Bit 8 (value 8)  : dead fibre — entire row flagged
    Bit 4 (value 4)  : alive-fibre pixels where correction was clamped to 1.0
                       (i.e., the ratio was undefined — un-flat-fielded pixels)
    """
    corr_flux = sci_flux  * correction
    corr_err  = sci_err   * np.abs(correction)    # σ_out = σ_in × |correction|
    corr_mask = sci_mask.copy()

    # Bit 8: dead fibre rows
    if dead_fibre_mask is not None:
        corr_mask[dead_fibre_mask, :] |= 8
        corr_flux[dead_fibre_mask, :]  = np.nan
        corr_err[dead_fibre_mask, :]   = np.nan

    # Bit 4: alive-fibre pixels with correction == 1.0 exactly
    # These are positions clamped from NaN in compute_fibre_flat (un-flat-fielded).
    # Floating-point division never produces exactly 1.0 for alive pixels, so
    # this reliably identifies only the clamped values.
    un_flat = correction == 1.0
    if dead_fibre_mask is not None:
        un_flat[dead_fibre_mask, :] = False   # already flagged with bit 8
    corr_mask[un_flat] |= 4

    return corr_flux.astype(np.float32), corr_err.astype(np.float32), corr_mask


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
# FIBER_ID alignment helper
# ---------------------------------------------------------------------------

def _align_correction_to_science(correction, dead_mask,
                                  flat_fibermap, sci_fibermap, logger):
    """Reorder correction rows to match the science RSS FIBER_ID ordering.

    Both RSS files cover the same channel and should contain the same set of
    FIBER_ID values, but may be stored in different row orders (e.g., if the
    flat and science were extracted from separate runs).

    Parameters
    ----------
    correction : 2D ndarray (n_flat_fibres, n_wave)
    dead_mask : 1D bool array (n_flat_fibres,)
    flat_fibermap : astropy BinTableHDU data
    sci_fibermap : astropy BinTableHDU data
    logger : logging.Logger

    Returns
    -------
    aligned_correction : 2D ndarray (n_sci_fibres, n_wave)
    aligned_dead_mask : 1D bool array (n_sci_fibres,)
    """
    flat_fiber_ids = np.array(flat_fibermap['FIBER_ID'], dtype=int)
    sci_fiber_ids  = np.array(sci_fibermap['FIBER_ID'],  dtype=int)

    _log = logger if logger is not None else type('_NullLog', (), {
        'debug': staticmethod(lambda *a, **k: None),
        'info':  staticmethod(lambda *a, **k: None),
        'warning': staticmethod(lambda *a, **k: None),
    })()

    # Fast path: identical ordering
    if np.array_equal(flat_fiber_ids, sci_fiber_ids):
        _log.debug("  FIBER_ID ordering is identical — no alignment needed")
        return correction, dead_mask

    n_sci  = len(sci_fiber_ids)
    n_wave = correction.shape[1]

    # Build lookup: flat_fiber_id → flat_row_index (warn on duplicates)
    flat_id_to_row = {}
    for idx, fid in enumerate(flat_fiber_ids):
        if fid in flat_id_to_row:
            _log.warning(f"  Duplicate FIBER_ID {fid} in flat fibermap — using last occurrence")
        flat_id_to_row[int(fid)] = idx

    aligned_correction = np.ones((n_sci, n_wave), dtype=np.float64)
    aligned_dead_mask  = np.ones(n_sci, dtype=bool)   # default: dead (unmatched)

    unmatched = []
    for sci_row, fid in enumerate(sci_fiber_ids):
        flat_row = flat_id_to_row.get(int(fid))
        if flat_row is None:
            unmatched.append(int(fid))
            aligned_correction[sci_row, :] = 1.0
            aligned_dead_mask[sci_row]     = True
        else:
            aligned_correction[sci_row, :] = correction[flat_row, :]
            aligned_dead_mask[sci_row]     = dead_mask[flat_row]

    if unmatched:
        _log.warning(
            f"  {len(unmatched)} science fibre(s) not found in flat FIBER_ID list — "
            f"correction set to 1.0. Missing IDs: {unmatched[:10]}"
            + (" ..." if len(unmatched) > 10 else "")
        )
    else:
        _log.info(f"  FIBER_ID alignment complete: {n_sci} fibres matched")

    return aligned_correction, aligned_dead_mask


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_fibre_flat(science_rss_path, flat_rss_path,
                   output_dir=None,
                   savgol_window=FF_SAVGOL_WINDOW,
                   savgol_polyorder=FF_SAVGOL_POLYORDER,
                   smooth_kernel=None,       # deprecated alias for savgol_window
                   clip_sigma=3.0, min_throughput=0.1,
                   n_central_fibres=FF_N_CENTRAL_FIBRES,
                   reference_fibre=None,     # deprecated, accepted but unused
                   generate_qa=True):
    """Run the full fibre-to-fibre flat-field correction pipeline.

    Loads a flat RSS file, computes the per-fibre correction using a synthetic
    reference built from the central fibres of each detector bench-side, and
    applies it to one or more science RSS files.  Writes both a straight copy
    (``*_RSS.fits``) and a corrected version (``*_RSSFF.fits``).

    Parameters
    ----------
    science_rss_path : str or list of str
        Path(s) to science RSS FITS file(s).
    flat_rss_path : str
        Path to the flat-field RSS FITS file (same channel as science).
    output_dir : str, optional
        Output directory.  Defaults to same directory as each science file.
    savgol_window : int
        Savitzky-Golay smoothing window in pixels.  Default ``FF_SAVGOL_WINDOW``.
    savgol_polyorder : int
        Savitzky-Golay polynomial order.  Default ``FF_SAVGOL_POLYORDER``.
    smooth_kernel : int or None
        Deprecated alias for ``savgol_window``.  Accepted for backward
        compatibility with ``reduce.py``.
    clip_sigma : float
        MAD sigma-clipping threshold for outlier rejection.  Default 3.0.
    min_throughput : float
        Dead-fibre threshold (fraction of global maximum median).  Default 0.1.
    n_central_fibres : int
        Number of fibres per bench-side closest to detector centre used for
        the synthetic reference.  Default ``FF_N_CENTRAL_FIBRES``.
    reference_fibre : ignored
        Deprecated parameter kept for backward compatibility.  Has no effect.
    generate_qa : bool
        Generate QA PNG plots.  Default True.

    Returns
    -------
    list of (str, str)
        List of ``(rss_path, rssff_path)`` tuples, one per science file.
    """
    # Honour legacy smooth_kernel kwarg from reduce.py
    if smooth_kernel is not None:
        savgol_window = int(smooth_kernel)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logger(__name__, f'fibre_flat_{timestamp}.log')
    logger.info("=" * 60)
    logger.info("FIBRE-TO-FIBRE FLAT FIELDING")
    logger.info("=" * 60)
    logger.info(f"Savitzky-Golay window={savgol_window}, polyorder={savgol_polyorder}, "
                f"n_central_fibres={n_central_fibres}")

    # Normalise to list
    if isinstance(science_rss_path, str):
        science_rss_path = [science_rss_path]

    # --- Load flat RSS ---
    logger.info(f"Loading flat RSS: {flat_rss_path}")
    flat_data    = load_rss(flat_rss_path, logger=logger)
    flat_fibermap = flat_data['fibermap']

    # --- Representative wavelength array (first valid row) ---
    wave_arr = flat_data['wave']
    valid_rows = ~np.all(np.isnan(wave_arr), axis=1)
    ref_wave = (wave_arr[np.where(valid_rows)[0][0], :]
                if np.any(valid_rows) else np.arange(wave_arr.shape[1], dtype=np.float64))

    # --- Compute correction from flat data ---
    correction, dead_mask = compute_fibre_flat(
        flat_data['flux'],
        flat_data['error'],
        flat_data['mask'],
        flat_fibermap,
        savgol_window=savgol_window,
        savgol_polyorder=savgol_polyorder,
        clip_sigma=clip_sigma,
        min_throughput=min_throughput,
        n_central_fibres=n_central_fibres,
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

        # Validate wavelength axis consistency (fibre count may differ — alignment handles it)
        if sci_data['flux'].shape[1] != flat_data['flux'].shape[1]:
            logger.error(
                f"Wavelength axis mismatch: science has {sci_data['flux'].shape[1]} pixels, "
                f"flat has {flat_data['flux'].shape[1]} pixels. Skipping this file."
            )
            continue

        # --- FIBER_ID alignment: reorder correction to match science fibre order ---
        correction_aligned, dead_aligned = _align_correction_to_science(
            correction, dead_mask,
            flat_fibermap, sci_data['fibermap'],
            logger
        )

        # Apply correction
        corr_flux, corr_error, corr_mask = apply_fibre_flat(
            sci_data['flux'],
            sci_data['error'],
            sci_data['mask'],
            correction_aligned,
            dead_fibre_mask=dead_aligned,
        )

        # --- Write _RSS.fits (straight copy of input) ---
        with fits.open(sci_path) as hdul:
            _write_rss_fits(hdul, rss_path)
        logger.info(f"Written original RSS copy: {rss_path}")

        # --- Write _RSSFF.fits (corrected) ---
        extra_kw = {
            'FFMETH':  ('synthetic_ref',                   'Fibre flat method: synthetic central reference'),
            'FFFILE':  (os.path.basename(flat_rss_path),   'Flat RSS file used'),
            'FFSAVGW': (savgol_window,                     'Savitzky-Golay smoothing window (pixels)'),
            'FFSAVGP': (savgol_polyorder,                  'Savitzky-Golay polynomial order'),
            'FFNCENT': (n_central_fibres,                  'Central fibres per bench-side used for reference'),
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
                f'Method: synthetic reference from {n_central_fibres} central fibres per bench-side'
            )
            hdul[0].header.add_history(
                f'SavGol smooth: window={savgol_window}px, polyorder={savgol_polyorder}'
            )
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
            plot_fibre_flat_qa(correction_aligned, dead_aligned, ref_wave, qa_flat_path, channel=channel)
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
    savgol_window : int
        Savitzky-Golay filter window length (pixels, must be odd).  Default 51.
    savgol_polyorder : int
        Savitzky-Golay polynomial order.  Default 3.
    n_central_fibres : int
        Number of fibres closest to each detector centre used to build the
        synthetic reference spectrum.  Default 50.
    clip_sigma : float
        Sigma-clipping threshold.  Default 3.0.
    min_throughput : float
        Dead-fibre threshold.  Default 0.1.
    generate_qa : bool
        Write QA plots on :meth:`apply`.  Default True.
    reference_fibre : ignored
        Deprecated — accepted for backward compatibility but not used.
    smooth_kernel : ignored
        Deprecated alias for ``savgol_window`` — accepted for backward
        compatibility but not used (the value is ignored when the OO
        interface is used directly; use ``savgol_window`` instead).
    """

    def __init__(self, flat_rss_path,
                 savgol_window=FF_SAVGOL_WINDOW,
                 savgol_polyorder=FF_SAVGOL_POLYORDER,
                 n_central_fibres=FF_N_CENTRAL_FIBRES,
                 clip_sigma=3.0, min_throughput=0.1, generate_qa=True,
                 # deprecated kwargs kept for backward compatibility
                 reference_fibre=None, smooth_kernel=None):
        self.flat_rss_path    = flat_rss_path
        self.savgol_window    = int(smooth_kernel) if smooth_kernel is not None else savgol_window
        self.savgol_polyorder = savgol_polyorder
        self.n_central_fibres = n_central_fibres
        self.clip_sigma       = clip_sigma
        self.min_throughput   = min_throughput
        self.generate_qa      = generate_qa

        self.correction  = None
        self.dead_mask   = None
        self._flat_data  = None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger = setup_logger(__name__, f'FibreFlatField_{timestamp}.log')

    def compute(self):
        """Load the flat RSS and compute the correction array.

        Populates ``self.correction`` and ``self.dead_mask``.
        """
        self._flat_data = load_rss(self.flat_rss_path, logger=self.logger)
        self.correction, self.dead_mask = compute_fibre_flat(
            self._flat_data['flux'],
            self._flat_data['error'],
            self._flat_data['mask'],
            self._flat_data['fibermap'],
            savgol_window=self.savgol_window,
            savgol_polyorder=self.savgol_polyorder,
            n_central_fibres=self.n_central_fibres,
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
            savgol_window=self.savgol_window,
            savgol_polyorder=self.savgol_polyorder,
            n_central_fibres=self.n_central_fibres,
            clip_sigma=self.clip_sigma,
            min_throughput=self.min_throughput,
            generate_qa=self.generate_qa,
        )
        return results[0] if results else (None, None)
