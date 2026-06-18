import os
from collections import Counter
import logging
import glob
from llamas_pyjamas.Image.WhiteLightModule import WhiteLightFits
import numpy as np
from astropy.io import fits
from llamas_pyjamas.config import CALIB_DIR, OUTPUT_DIR, LUT_DIR
from llamas_pyjamas.constants import idx_lookup
from llamas_pyjamas.Flat.flatProcessing import produce_flat_extractions
from llamas_pyjamas.Utils.utils import concat_extractions, is_wavelength_solution_useable
from llamas_pyjamas.Arc.arcLlamasMulti import arcTransfer
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
import matplotlib.pyplot as plt
from pypeit.core.fitting import iterfit
import pickle
from datetime import datetime
from scipy.ndimage import median_filter, gaussian_filter


# Default per-channel signal thresholds for the simple pixel flat method
CHANNEL_SIGNAL_THRESHOLDS = {'red': 5000, 'green': 8000, 'blue': 5000}




logger = logging.getLogger(__name__)

def create_master_flat(file_list, target_color, output_dir=OUTPUT_DIR):
    """
    Stacks multiple flats into a 24-extension FITS file compatible with LLAMAS.
    
    Parameters:
    -----------
    file_list : list
        Paths to the raw LLAMAS FITS files.
    target_color : str
        'red', 'green', or 'blue'.
    output_dir : str
        Where to save the master flat.
    """
    if not file_list:
        raise ValueError(f"No files provided for {target_color} stacking.")

    target_color = target_color.lower()
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f"master_flat_{target_color}.fits")
    
    logger.info(f"Creating 24-extension Master Flat for {target_color.upper()}")

    # 1. Use the first file as a structural template
    with fits.open(file_list[0]) as template:
        master_hdul = fits.HDUList([fits.PrimaryHDU(header=template[0].header)])
        
        # 2. Iterate through all 24 extensions to maintain index integrity
        for ext_idx in range(1, len(template)):
            ext = template[ext_idx]
            ext_color = ext.header.get('COLOR', '').lower()
            
            if ext_color == target_color:
                # This extension matches our target color; perform the median stack
                logger.debug(f"Stacking extensions at index {ext_idx} ({ext.name})")
                
                data_stack = []
                for fname in file_list:
                    with fits.open(fname) as hdul:
                        data_stack.append(hdul[ext_idx].data)
                
                # Median stack to reject cosmic rays and increase SNR
                master_data = np.median(data_stack, axis=0).astype(np.float32)
                new_ext = fits.ImageHDU(data=master_data, header=ext.header, name=ext.name)
            else:
                # For other colors, we keep the extension/header but zero the data.
                # This ensures the file is still 24 extensions long for reduce_flat.
                new_ext = fits.ImageHDU(data=np.zeros_like(ext.data), header=ext.header, name=ext.name)
            
            master_hdul.append(new_ext)

    master_hdul.writeto(output_filename, overwrite=True)
    logger.info(f"Master {target_color} flat saved to: {output_filename}")
    return output_filename


def sanitize_extraction_dict_for_pickling(extraction_dict):
    """Sanitize extraction dictionary to remove problematic references for pickling.
    
    This function removes the 'trace' attribute from ExtractLlamas objects to prevent
    pickling issues caused by duplicate module imports (particularly TraceRay classes).
    
    Args:
        extraction_dict (dict): Dictionary containing 'extractions' and 'metadata' keys
        
    Returns:
        dict: Sanitized dictionary with trace references removed from extraction objects
    """
    import copy
    
    logger.info("Sanitizing extraction dictionary for pickling")
    
    # Make a deep copy to avoid modifying the original
    sanitized_dict = copy.deepcopy(extraction_dict)
    
    # Remove trace references from each extraction object
    extractions = sanitized_dict.get('extractions', [])
    sanitized_count = 0
    
    for i, extraction in enumerate(extractions):
        if hasattr(extraction, 'trace'):
            logger.debug(f"Removing trace reference from extraction {i}")
            extraction.trace = None
            sanitized_count += 1
            
        # Also remove any other potentially problematic attributes
        for attr_name in ['LUT', 'dead_fibers']:
            if hasattr(extraction, attr_name):
                logger.debug(f"Removing {attr_name} from extraction {i}")
                setattr(extraction, attr_name, None)
    
    logger.info(f"Sanitized {sanitized_count} extraction objects with trace references")
    return sanitized_dict



def sort_and_write_pixel_maps(pixel_maps, output_path, header_info=None):
    """
    Sort pixel maps in canonical order and write to a single FITS file.
    
    The ordering follows: bench (1-4) -> side (A, B) -> channel (red, green, blue)
    
    Args:
        pixel_maps: Dictionary of pixel maps keyed by extension name
        output_path: Path to output FITS file
        header_info: Optional dictionary with header keywords
    """
    
    # Parse extension names and assign sort indices
    ext_with_idx = []
    
    for ext_name, pixel_map in pixel_maps.items():
        # Parse the extension name
        parts = _parse_extension_name(ext_name)
        
        if parts is None:
            logger.warning(f"Could not parse extension name: {ext_name}")
            continue
        
        channel, bench, side = parts
        
        # Get sort index from lookup
        sort_key = (channel, bench, side)
        if sort_key not in idx_lookup:
            logger.warning(f"Extension {ext_name} ({sort_key}) not in standard ordering")
            continue
        
        idx = idx_lookup[sort_key]
        ext_with_idx.append((idx, ext_name, pixel_map, channel, bench, side))

    # Sort by index
    ext_with_idx.sort(key=lambda x: x[0])
    
    logger.info(f"Sorted {len(ext_with_idx)} pixel maps in canonical order")
    
    # Verify we have all 24 extensions
    if len(ext_with_idx) != 24:
        logger.warning(f"Expected 24 extensions, found {len(ext_with_idx)}")
    
    # Create primary HDU with header
    primary_hdu = fits.PrimaryHDU()
    
    if header_info:
        for key, value in header_info.items():
            primary_hdu.header[key] = value
    
    primary_hdu.header['NAXIS'] = 0
    primary_hdu.header['NEXTEND'] = len(ext_with_idx)
    primary_hdu.header['COMMENT'] = 'LLAMAS flat field pixel maps'
    
    # Create list to hold all HDUs
    hdu_list = [primary_hdu]
    
    # Add each pixel map as an extension (already sorted)
    for idx, ext_name, pixel_map, channel, bench, side in ext_with_idx:
        img_hdu = fits.ImageHDU(data=pixel_map, name=ext_name)
        img_hdu.header['EXTNAME'] = ext_name
        img_hdu.header['EXTVER'] = idx
        img_hdu.header['CHANNEL'] = channel.upper()
        img_hdu.header['BENCH'] = str(bench)
        img_hdu.header['SIDE'] = side.upper()
        img_hdu.header['COMMENT'] = f'Flat field pixel map for {ext_name}'

        hdu_list.append(img_hdu)
    
    # Write to file
    hdulist = fits.HDUList(hdu_list)
    hdulist.writeto(output_path, overwrite=True)
    logger.info(f"Wrote {len(ext_with_idx)} pixel maps to {output_path}")


def generate_pixel_flat_extension(extraction_obj, channel=None, filter_size=51,
                                  signal_threshold=None, clip_range=(0.90, 1.10),
                                  return_smooth_models=False):
    """Fiber-by-fiber 1D pixel sensitivity map for a single extension.

    Smooths extracted counts in wavelength (xshift) space using a median +
    Gaussian filter to capture the lamp envelope, then divides to isolate
    pixel-to-pixel sensitivity variations.  The 1D correction is projected
    back onto the 2D detector footprint via the trace fiberimg.

    Parameters
    ----------
    extraction_obj : ExtractLlamas
        Extraction object for this extension.  Must have ``.trace`` with
        ``.fiberimg``, ``.nfibers``, ``.naxis1``, ``.naxis2``, and
        ``.counts`` and ``.xshift`` arrays (populated by arc transfer).
    channel : str, optional
        Color channel ('red', 'green', 'blue').  Used to select default
        ``signal_threshold`` when not explicitly provided.
    filter_size : int
        Median filter kernel size (pixels) for lamp-envelope smoothing.
    signal_threshold : float, optional
        Minimum smooth-model ADU to apply correction.  Pixels below this
        are set to 1.0 (no correction).  If ``None``, uses
        ``CHANNEL_SIGNAL_THRESHOLDS`` based on *channel*.
    clip_range : tuple of float
        (min, max) clipping range for the output sensitivity map.
    return_smooth_models : bool
        If True, also return the smooth lamp-envelope models per fibre
        (keyed by array index) for use in fibre-to-fibre flat fielding.

    Returns
    -------
    sensitivity_map : ndarray, shape (naxis2, naxis1)
        2D pixel sensitivity map clipped to *clip_range*.
    smooth_models : dict, optional
        Only returned when ``return_smooth_models=True``.  Dictionary
        ``{fib_idx: smooth_1d_array}`` containing the smooth lamp-envelope
        model for each fibre.
    """
    trace_obj = extraction_obj.trace
    naxis1, naxis2 = trace_obj.naxis1, trace_obj.naxis2

    # Resolve threshold
    if signal_threshold is None:
        ch = (channel or getattr(extraction_obj, 'channel', 'green')).lower()
        signal_threshold = CHANNEL_SIGNAL_THRESHOLDS.get(ch, 8000)

    # Check whether xshift is populated (arc transfer has been applied)
    xshift_available = (extraction_obj.xshift is not None
                        and np.count_nonzero(extraction_obj.xshift) > 0)

    logger.info(f"Generating pixel flat: nfibers={trace_obj.nfibers}, "
                f"threshold={signal_threshold}, filter_size={filter_size}, "
                f"wavelength_space={xshift_available}")

    sensitivity_map = np.ones((naxis2, naxis1), dtype=np.float32)
    smooth_models = {} if return_smooth_models else None

    for fib_idx in range(trace_obj.nfibers):
        raw_1d = extraction_obj.counts[fib_idx, :]

        if xshift_available:
            # --- Smooth in wavelength (xshift) space ---
            xshift = extraction_obj.xshift[fib_idx, :]

            # Build a regular grid in xshift space
            xshift_regular = np.linspace(xshift.min(), xshift.max(), len(xshift))

            # Interpolate counts onto the regular wavelength grid
            counts_regular = np.interp(xshift_regular, xshift,
                                       raw_1d.astype(np.float64))

            # Smooth on the regular grid
            smooth_regular = median_filter(counts_regular, size=filter_size)
            smooth_regular = gaussian_filter(smooth_regular, sigma=2.0)

            # Map smoothed model back to original xshift positions
            smooth_1d = np.interp(xshift, xshift_regular, smooth_regular)
        else:
            # Fallback: smooth in pixel space (notebook behaviour)
            smooth_1d = median_filter(raw_1d.astype(np.float64), size=filter_size)
            smooth_1d = gaussian_filter(smooth_1d, sigma=2.0)

        if return_smooth_models:
            smooth_models[fib_idx] = smooth_1d.copy()

        # 1D sensitivity ratio where signal is reliable
        valid = smooth_1d > signal_threshold
        ratio_1d = np.ones_like(raw_1d, dtype=np.float32)
        ratio_1d[valid] = raw_1d[valid] / smooth_1d[valid]

        # Project 1D correction onto the 2D fiber footprint
        fiber_mask = (trace_obj.fiberimg == fib_idx)
        sensitivity_map[fiber_mask] = ratio_1d[np.where(fiber_mask)[1]]

    # Un-flip to match raw FITS coordinates.
    # process_fits_by_color applies channel-specific flips before extraction, so the
    # sensitivity_map is in flipped space.  The flat is applied to the raw (un-flipped)
    # science FITS, so we must invert those flips here.
    ch = (channel or getattr(extraction_obj, 'channel', 'green')).lower()
    if ch == 'green':
        sensitivity_map = np.fliplr(sensitivity_map)
    elif ch == 'blue':
        sensitivity_map = np.flipud(np.fliplr(sensitivity_map))
    # red: no flip applied by process_fits_by_color, nothing to undo

    # Clip to remove outliers
    sensitivity_map = np.clip(sensitivity_map, clip_range[0], clip_range[1])

    if return_smooth_models:
        return sensitivity_map, smooth_models
    return sensitivity_map


def _parse_extension_name(ext_name):
    """
    Parse extension name into (channel, bench, side) components.
    
    Handles formats like:
        - 'red_1_A'
        - 'red1A' 
        - 'RED_1_A'
    
    Args:
        ext_name: Extension name string
    
    Returns:
        tuple: (channel, bench, side) or None if parsing fails
    """
    ext_name = ext_name.lower()
    
    # Try underscore-separated format first
    if '_' in ext_name:
        parts = ext_name.split('_')
        if len(parts) == 3:
            channel, bench, side = parts
            return (channel, bench, side.upper())
    
    # Try compact format (e.g., 'red1a')
    import re
    match = re.match(r'(red|green|blue)(\d)([ab])', ext_name)
    if match:
        channel, bench, side = match.groups()
        return (channel, bench, side.upper())
    
    # If neither format works, return None
    return None


def _find_matching_trace(trace_files, channel, bench, side):
    """Find the trace file matching a given channel/bench/side combination.

    Parameters
    ----------
    trace_files : list of str
        Paths to available trace pickle files.
    channel : str
        Color channel (e.g. 'red', 'green', 'blue').
    bench : str
        Bench number (e.g. '1', '2', '3', '4').
    side : str
        Side identifier (e.g. 'A', 'B').

    Returns
    -------
    str or None
        Path to matching trace file, or None if not found.
    """
    target = f"{channel}{bench}{side}".lower()
    for trace_file in trace_files:
        with open(trace_file, 'rb') as tf:
            trace_obj = pickle.load(tf)
            trace_key = f"{trace_obj.channel}{trace_obj.bench}{trace_obj.side}".lower()
            if trace_key == target:
                return trace_file
    return None


def fit_spectrum_to_xshift(extraction, fiber_index, maxiter=6, bkspace=None, nord=4,
                          adaptive_breakpoints=True, min_breakpoints=50):
    """
    Improved spectrum fitting with adaptive breakpoints for complex spectral features.
    
    Parameters:
    -----------
    extraction : object
        Extraction object containing xshift and counts data
    fiber_index : int
        Index of the fiber to fit
    maxiter : int
        Maximum iterations for iterative fitting
    bkspace : float or None
        Breakpoint spacing. If None, will be calculated adaptively
    nord : int
        Order of B-spline (3=cubic, 4=quartic)
    adaptive_breakpoints : bool
        Whether to use adaptive breakpoint spacing based on data complexity
    min_breakpoints : int
        Minimum number of breakpoints to use
    """
    
    # Get the xshift and counts data for a specific fiber
    xshift = extraction.xshift[fiber_index, :]
    counts = extraction.counts[fiber_index, :]

    # Clean the data
    mask = np.isfinite(counts)
    xshift_clean = xshift[mask]
    counts_clean = counts[mask]
    
    logger.info(f"Removed NaNs: {len(xshift) - len(xshift_clean)} points removed")
    logger.debug(f"Fitting fiber {fiber_index} with {len(xshift_clean)} valid points")
    
    if len(xshift_clean) < 10:
        raise ValueError(f"Not enough valid data points ({len(xshift_clean)}) for fitting")
    
    # Calculate adaptive breakpoint spacing
    if bkspace is None or adaptive_breakpoints:
        # Method 1: Base spacing on data density and range
        xrange = xshift_clean.max() - xshift_clean.min()
        n_points = len(xshift_clean)
        
        # Calculate variance in the data to detect complexity
        counts_smooth = np.convolve(counts_clean, np.ones(5)/5, mode='same')  # Simple smoothing
        complexity = np.std(counts_clean - counts_smooth) / np.mean(counts_clean)
        
        # Adaptive breakpoint calculation
        if complexity > 0.1:  # High complexity (sharp features)
            target_breakpoints = max(min_breakpoints, n_points // 20)  # 1 breakpoint per 20 points
            bkspace = xrange / target_breakpoints
        else:  # Lower complexity
            target_breakpoints = max(min_breakpoints // 2, n_points // 50)  # 1 breakpoint per 50 points
            bkspace = xrange / target_breakpoints
            
        logger.info(f"Adaptive bkspace: {bkspace:.3f}, target breakpoints: {target_breakpoints}")
    
    # Try multiple fitting strategies
    fitting_strategies = [
        {'bkspace': bkspace, 'nord': nord, 'name': 'adaptive'},
        {'bkspace': bkspace * 0.5, 'nord': 3, 'name': 'fine_cubic'},  # Finer spacing, cubic
        {'bkspace': bkspace * 0.3, 'nord': 4, 'name': 'very_fine'},   # Very fine spacing
        {'bkspace': bkspace * 2.0, 'nord': 5, 'name': 'coarse_high_order'}  # Coarser but higher order
    ]
    
    for i, strategy in enumerate(fitting_strategies):
        try:
            logger.debug(f"Trying strategy {i+1}: {strategy['name']} (bkspace={strategy['bkspace']:.3f})")
            
            sset, outmask = iterfit(
                xshift_clean, counts_clean, 
                maxiter=maxiter,
                nord=strategy['nord'],
                kwargs_bspline={'bkspace': strategy['bkspace']},
                upper= 5.0, 
                lower =5.0  # Add outlier rejection
            )
            
            # Create evaluation grid
            xmodel = np.linspace(xshift_clean.min(), xshift_clean.max(), len(xshift))  # was 2* for Higher resolution
            y_fit = sset.value(xmodel)[0]
            
            # Evaluate fit quality
            y_fit_original = sset.value(xshift_clean)[0] 
            residuals = counts_clean - y_fit_original
            rms_residual = np.sqrt(np.mean(residuals**2))
            relative_rms = rms_residual / np.mean(counts_clean)
            
            logger.info(f"Strategy '{strategy['name']}' successful: RMS residual = {relative_rms:.4f}")
            
            # If fit quality is good, use this strategy
            if relative_rms < 0.5:  # Accept if RMS is less than 50% of mean signal
                return {
                    'xshift_clean': xshift_clean,
                    'counts_clean': counts_clean,
                    'xmodel': xmodel,
                    'y_fit': y_fit,
                    'y_fit_original_grid': y_fit_original,
                    'bspline_model': sset,
                    'outmask': outmask,
                    'residuals': residuals,
                    'rms_residual': rms_residual,
                    'relative_rms': relative_rms,
                    'strategy_used': strategy['name'],
                    'final_bkspace': strategy['bkspace'],
                    'final_nord': strategy['nord']
                }
            
        except Exception as e:
            logger.warning(f"Strategy '{strategy['name']}' failed: {str(e)}")
            continue
    
    # If all strategies failed, try a fallback approach
    logger.warning("All standard strategies failed, trying fallback approach")
    try:
        # Manual breakpoint specification as last resort
        n_manual_breaks = max(20, len(xshift_clean) // 100)
        xmin, xmax = xshift_clean.min(), xshift_clean.max()
        manual_bkpt = np.linspace(xmin + 0.01*(xmax-xmin), xmax - 0.01*(xmax-xmin), n_manual_breaks)
        
        sset, outmask = iterfit(
            xshift_clean, counts_clean,
            maxiter=maxiter,
            nord=3,  # Use cubic for stability
            bkpt=manual_bkpt
        )
        
        xmodel = np.linspace(xshift_clean.min(), xshift_clean.max(), len(xshift)) # why was this twice the length?
        y_fit = sset.value(xmodel)[0]
        y_fit_original = sset.value(xshift_clean)[0]
        residuals = counts_clean - y_fit_original
        rms_residual = np.sqrt(np.mean(residuals**2))
        relative_rms = rms_residual / np.mean(counts_clean)
        
        logger.info(f"Fallback approach successful: RMS residual = {relative_rms:.4f}")
        
        return {
            'xshift_clean': xshift_clean,
            'counts_clean': counts_clean,
            'xmodel': xmodel,
            'y_fit': y_fit,
            'y_fit_original_grid': y_fit_original,
            'bspline_model': sset,
            'outmask': outmask,
            'residuals': residuals,
            'rms_residual': rms_residual,
            'relative_rms': relative_rms,
            'strategy_used': 'ultra_fine_manual_breakpoints',
            'final_bkspace': None,
            'final_nord': 3
        }
        
    except Exception as e:
        logger.error(f"All fitting approaches failed for fiber {fiber_index}: {str(e)}")
        raise


def process_flat_field_complete(red_flat_file, green_flat_file, blue_flat_file,
                               arc_calib_file=None, use_bias=None, output_dir=OUTPUT_DIR,
                               trace_dir=CALIB_DIR, verbose=False, pixel_qe_mode=True):
    """Process complete flat field workflow with wavelength calibration and pixel mapping.

    This function implements the complete flat field processing workflow:
    1. Extract individual color flat fields using produce_flat_extractions
    2. Combine all extractions into a single .pkl file
    3. Apply wavelength solution from arc calibration
    4. Fit B-splines to xshift vs counts for each fiber
    5. Generate per-pixel flat field correction images
    6. (Optional) Generate true 2D pixel QE maps

    Args:
        red_flat_file (str): Path to red flat field FITS file
        green_flat_file (str): Path to green flat field FITS file
        blue_flat_file (str): Path to blue flat field FITS file
        arc_calib_file (str, optional): Path to arc calibration file.
            Defaults to 'LLAMAS_reference_arc.pkl' in trace_dir.
        use_bias (str, optional): Path to bias file. Defaults to None.
        output_dir (str, optional): Output directory. Defaults to OUTPUT_DIR.
        trace_dir (str, optional): Trace directory. Defaults to CALIB_DIR.
        verbose (bool, optional): Enable verbose console output. Defaults to False.
        pixel_qe_mode (bool, optional): If True, generate true 2D pixel-level QE maps
            that capture intra-fibre pixel-to-pixel variations. If False, use the legacy
            fibre-averaged normalization. Defaults to True.

    Returns:
        dict: Dictionary containing processing results and output file paths
    """
    logger.info("Starting complete flat field processing workflow")
    logger.info(f"Input files:")
    logger.info(f"  Red: {red_flat_file}")
    logger.info(f"  Green: {green_flat_file}")
    logger.info(f"  Blue: {blue_flat_file}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Trace directory: {trace_dir}")

    # Step 1: Produce individual flat extractions
    logger.info("Step 1: Producing individual flat field extractions")
    produce_flat_extractions(
        red_flat_file,
        green_flat_file,
        blue_flat_file,
        tracedir=trace_dir,
        outpath=output_dir,
        verbose=verbose,
        use_bias=use_bias
    )
    
    # Step 2: Combine all extractions into a single file
    logger.info("Step 2: Combining individual extractions into single file")
    red_extraction = os.path.join(output_dir, 'red_extractions_flat.pkl')
    green_extraction = os.path.join(output_dir, 'green_extractions_flat.pkl')
    blue_extraction = os.path.join(output_dir, 'blue_extractions_flat.pkl')
    
    # Check that all extraction files exist
    extraction_files = [red_extraction, green_extraction, blue_extraction]
    for file_path in extraction_files:
        if not os.path.exists(file_path):
            logger.error(f"Extraction file not found: {file_path}")
            raise FileNotFoundError(f"Missing extraction file: {file_path}")
    
    combined_flat_file = os.path.join(output_dir, 'combined_flat_extractions.pkl')
    concat_extractions(extraction_files, combined_flat_file)
    logger.info(f"Combined extractions saved to {combined_flat_file}")
    
    # Step 3: Apply wavelength solution from arc calibration
    logger.info("Step 3: Applying wavelength solution from arc calibration")
    if arc_calib_file is None:
        arc_calib_file = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')
    
    if not os.path.exists(arc_calib_file):
        try:
            logger.error(f"Arc calibration file not found: {arc_calib_file}, using default")
            # raise FileNotFoundError(f"Missing arc calibration file: {arc_calib_file}")
            arc_calib_file = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')
        except Exception as e:
            logger.critical(f"CRITICAL ERROR: Arc calibration file not found: {str(e)}")
            raise

    # Load the arc calibration
    logger.info(f"Loading arc calibration from {arc_calib_file}")
    arc_dict = ExtractLlamas.loadExtraction(arc_calib_file)

    # if not is_wavelength_solution_useable(arc_dict):
    #     logger.critical(f"CRITICAL ERROR: Arc calibration file {arc_calib_file} is not useable.")
    #     raise ValueError(f"Arc calibration file {arc_calib_file} is not useable.")
    
    # Load the combined flat extractions
    logger.info(f"Loading combined flat extractions from {combined_flat_file}")
    flat_dict = ExtractLlamas.loadExtraction(combined_flat_file)
    

    
    # Apply wavelength solution transfer
    logger.info("Transferring wavelength calibration to flat field extractions")
    flat_dict_calibrated = arcTransfer(flat_dict, arc_dict, enable_validation=True, verbose=True)
    logger.info("Wavelength calibration transferred to flat field extractions")

    # Validate transfer success
    transfer_failures = []
    for ext_idx in range(len(flat_dict_calibrated['extractions'])):
        ext = flat_dict_calibrated['extractions'][ext_idx]
        xshift_valid = np.count_nonzero(ext.xshift) > 0
        wave_valid = np.any(ext.wave > 0)
        if not (xshift_valid and wave_valid):
            meta = flat_dict_calibrated['metadata'][ext_idx]
            transfer_failures.append(f"{meta['channel']}{meta['bench']}{meta['side']}")
            logger.error(f"Extension {ext_idx} wavelength transfer validation failed!")

    if transfer_failures:
        raise ValueError(f"Wavelength transfer failed for extensions: {transfer_failures}")

    logger.info("✓ All extensions passed wavelength transfer validation")

    # Save the calibrated flat extractions (sanitized to avoid pickling issues)
    calibrated_flat_file = os.path.join(output_dir, 'combined_flat_extractions_calibrated.pkl')
    sanitized_flat_dict = sanitize_extraction_dict_for_pickling(flat_dict_calibrated) #why is this here?
    with open(calibrated_flat_file, 'wb') as f:
        pickle.dump(sanitized_flat_dict, f)
    logger.info(f"Calibrated flat extractions saved to {calibrated_flat_file}")
    
    # Step 4: Fit B-splines and generate pixel maps
    logger.info("Step 4: Fitting B-splines and generating pixel maps")

    # Initialize the Thresholding class for the remaining processing
    # NOTE: Thresholding expects the combined flat file, not individual color files
    threshold_processor = Thresholding(
        calibrated_flat_file,
        use_bias=use_bias, output_dir=output_dir, trace_dir=trace_dir
    )
    
    # Calculate fits for all extensions in the calibrated file
    fit_results = threshold_processor.calculate_fits_all_extensions(calibrated_flat_file)
    
    # Step 5: Generate the 2D maps
    logger.info("Step 5: Generating 2D Pixel-to-Pixel Sensitivity Maps")
    
    # This calls generate_thresholds, which now uses our updated _generate_single_pixel_map
    pixel_map_results = threshold_processor.generate_thresholds()
    
    # Ensure the results dictionary points to the newly created FITS file
    results = {
        'combined_flat_file': combined_flat_file,
        'calibrated_flat_file': calibrated_flat_file,
        'fit_results': fit_results,
        'pixel_map_file': threshold_processor.map_filename, # This is 'pixel_maps.fits'
        'processing_status': 'completed'
    }

    return results


def _write_smooth_models_fits(all_smooth_data, output_path, filter_size=51,
                              gaussian_sigma=2.0):
    """Write lamp smooth models to a multi-extension FITS file.

    Creates one BinTableHDU per benchside with columns FIBER_ID, SMOOTH,
    and WAVE.  Dead fibres are excluded.

    Parameters
    ----------
    all_smooth_data : dict
        ``{ext_name: {'fiber_ids': array, 'smooth': 2D array,
        'wave': 2D array, 'channel': str, 'bench': str, 'side': str}}``
    output_path : str
        Output FITS file path.
    filter_size : int
        Median filter kernel size used (recorded in header).
    gaussian_sigma : float
        Gaussian sigma used (recorded in header).
    """
    primary = fits.PrimaryHDU()
    primary.header['FILTSZ'] = (filter_size, 'Median filter kernel size')
    primary.header['GSIGMA'] = (gaussian_sigma, 'Gaussian smoothing sigma')
    primary.header['DATE'] = (datetime.now().isoformat(), 'File creation date')
    primary.header['HISTORY'] = 'Smooth lamp-envelope models from P2P flat step'
    hdul = fits.HDUList([primary])

    for ext_name in sorted(all_smooth_data.keys()):
        data = all_smooth_data[ext_name]
        fiber_ids = data['fiber_ids']
        smooth = data['smooth']
        wave = data['wave']
        naxis1 = smooth.shape[1] if smooth.ndim == 2 else 0

        if len(fiber_ids) == 0:
            logger.warning(f"No alive fibres for {ext_name}, skipping")
            continue

        cols = [
            fits.Column(name='FIBER_ID', format='J', array=fiber_ids),
            fits.Column(name='SMOOTH', format=f'{naxis1}D', array=smooth),
            fits.Column(name='WAVE', format=f'{naxis1}D', array=wave),
        ]
        tbl = fits.BinTableHDU.from_columns(cols)
        tbl.header['EXTNAME'] = ext_name
        tbl.header['CHANNEL'] = data['channel']
        tbl.header['BENCH'] = data['bench']
        tbl.header['SIDE'] = data['side']
        tbl.header['NALIVE'] = (len(fiber_ids), 'Number of alive fibres')
        hdul.append(tbl)

    hdul.writeto(output_path, overwrite=True)
    logger.info(f"Smooth models FITS written: {output_path} "
                f"({len(hdul) - 1} extensions)")


def process_pixel_flat_simple(red_flat_file, green_flat_file, blue_flat_file,
                              arc_calib_file=None, use_bias=None,
                              output_dir=OUTPUT_DIR, trace_dir=CALIB_DIR,
                              verbose=False, filter_size=12,
                              signal_thresholds=None, clip_range=(0.90, 1.10)):
    """Generate pixel-to-pixel sensitivity maps using the simple median+Gaussian method.

    Workflow:
    1. Extract flat spectra via ``produce_flat_extractions`` (shared with B-spline method)
    2. Concatenate into combined 24-extension file (shared)
    3. Apply wavelength solution via ``arcTransfer`` (mandatory — smoothing is
       done in xshift/wavelength space)
    4. For each extension, generate sensitivity map via ``generate_pixel_flat_extension``
    5. Write 24-extension FITS via ``sort_and_write_pixel_maps``

    Parameters
    ----------
    red_flat_file, green_flat_file, blue_flat_file : str
        Paths to color-channel flat field FITS files.
    arc_calib_file : str, optional
        Path to arc calibration pickle.  Defaults to
        ``LUT/LLAMAS_reference_arc.pkl``.
    use_bias : str, optional
        Path to bias file for bias subtraction during extraction.
    output_dir : str
        Output directory for extraction intermediates and final pixel maps.
    trace_dir : str
        Directory containing trace pickle files.
    verbose : bool
        Enable verbose logging.
    filter_size : int
        Median filter kernel size for lamp-envelope smoothing.
    signal_thresholds : dict, optional
        Per-channel thresholds ``{'red': N, 'green': N, 'blue': N}``.
        Defaults to ``CHANNEL_SIGNAL_THRESHOLDS``.
    clip_range : tuple of float
        (min, max) clipping for the sensitivity map.

    Returns
    -------
    dict
        ``{'combined_flat_file', 'calibrated_flat_file', 'pixel_map_file',
        'processing_status'}``
    """
    logger.info("Starting SIMPLE pixel flat processing workflow")

    if signal_thresholds is None:
        signal_thresholds = dict(CHANNEL_SIGNAL_THRESHOLDS)

    # ── Step 1: Extract flat spectra (shared with B-spline method) ──
    logger.info("Step 1: Producing flat field extractions")
    produce_flat_extractions(
        red_flat_file, green_flat_file, blue_flat_file,
        tracedir=trace_dir, outpath=output_dir,
        verbose=verbose, use_bias=use_bias,
    )

    # ── Step 2: Concatenate extractions (shared with B-spline method) ──
    logger.info("Step 2: Concatenating extractions")
    red_pkl = os.path.join(output_dir, 'red_extractions_flat.pkl')
    green_pkl = os.path.join(output_dir, 'green_extractions_flat.pkl')
    blue_pkl = os.path.join(output_dir, 'blue_extractions_flat.pkl')

    for pkl in [red_pkl, green_pkl, blue_pkl]:
        if not os.path.exists(pkl):
            raise FileNotFoundError(f"Missing extraction file: {pkl}")

    combined_flat_file = os.path.join(output_dir, 'combined_flat_extractions.pkl')
    concat_extractions([red_pkl, green_pkl, blue_pkl], combined_flat_file)
    logger.info(f"Combined extractions saved to {combined_flat_file}")

    # Remove per-color intermediate pkls now that they are merged
    for _pkl in [red_pkl, green_pkl, blue_pkl]:
        try:
            os.remove(_pkl)
            logger.info(f"Removed intermediate file: {os.path.basename(_pkl)}")
        except OSError:
            pass

    # ── Step 3: Apply wavelength solution (mandatory) ──
    logger.info("Step 3: Applying wavelength solution from arc calibration")

    if arc_calib_file is None:
        arc_calib_file = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')
    if not os.path.exists(arc_calib_file):
        raise FileNotFoundError(f"Arc calibration file not found: {arc_calib_file}")

    logger.info(f"Loading arc calibration from {arc_calib_file}")
    arc_dict = ExtractLlamas.loadExtraction(arc_calib_file)

    logger.info(f"Loading combined flat extractions from {combined_flat_file}")
    flat_dict = ExtractLlamas.loadExtraction(combined_flat_file)

    logger.info("Transferring wavelength calibration to flat field extractions")
    flat_dict_calibrated = arcTransfer(flat_dict, arc_dict,
                                       enable_validation=True, verbose=verbose)

    # Validate transfer success
    transfer_failures = []
    for ext_idx in range(len(flat_dict_calibrated['extractions'])):
        ext = flat_dict_calibrated['extractions'][ext_idx]
        xshift_valid = np.count_nonzero(ext.xshift) > 0
        wave_valid = np.any(ext.wave > 0)
        if not (xshift_valid and wave_valid):
            meta = flat_dict_calibrated['metadata'][ext_idx]
            transfer_failures.append(
                f"{meta['channel']}{meta['bench']}{meta['side']}")
            logger.error(
                f"Extension {ext_idx} wavelength transfer validation failed!")

    if transfer_failures:
        raise ValueError(
            f"Wavelength transfer failed for extensions: {transfer_failures}")
    logger.info("All extensions passed wavelength transfer validation")

    # Save calibrated extractions
    calibrated_flat_file = os.path.join(
        output_dir, 'combined_flat_extractions_calibrated.pkl')
    sanitized_flat_dict = sanitize_extraction_dict_for_pickling(
        flat_dict_calibrated)
    with open(calibrated_flat_file, 'wb') as f:
        pickle.dump(sanitized_flat_dict, f)
    logger.info(f"Calibrated flat extractions saved to {calibrated_flat_file}")

    # Raw combined pkl is superseded by the calibrated version; remove it
    try:
        os.remove(combined_flat_file)
        logger.info(f"Removed intermediate file: {os.path.basename(combined_flat_file)}")
    except OSError:
        pass

    # ── Step 4: Generate pixel maps using wavelength-calibrated extractions ──
    logger.info("Step 4: Generating pixel sensitivity maps (simple method)")

    # Use the in-memory calibrated extractions (traces still attached)
    extractions = flat_dict_calibrated['extractions']
    metadata = flat_dict_calibrated['metadata']

    # Load trace files for cases where trace is missing
    trace_files = glob.glob(os.path.join(trace_dir, 'LLAMAS*traces.pkl'))

    pixel_maps = {}  # keyed by underscore-separated ext_name
    all_smooth_data = {}  # keyed by ext_name, for fibre flat

    for ext_idx, (ext_obj, meta) in enumerate(zip(extractions, metadata)):
        channel = meta['channel']
        bench = str(meta['bench'])
        side = meta['side']
        ext_name = f"{channel}_{bench}_{side}"

        logger.info(f"Processing extension {ext_idx}: {ext_name}")

        # Ensure trace is available
        if not hasattr(ext_obj, 'trace') or ext_obj.trace is None:
            matching_trace = _find_matching_trace(
                trace_files, channel, bench, side)
            if matching_trace is None:
                logger.warning(f"No trace file for {ext_name}, skipping")
                continue
            with open(matching_trace, 'rb') as tf:
                ext_obj.trace = pickle.load(tf)

        threshold = signal_thresholds.get(channel, 8000)
        pixel_map, smooth_models = generate_pixel_flat_extension(
            ext_obj, channel=channel,
            filter_size=filter_size,
            signal_threshold=threshold,
            clip_range=clip_range,
            return_smooth_models=True,
        )
        pixel_maps[ext_name] = pixel_map

        # Build FIBER_ID list keyed by physical ID, excluding dead fibres
        dead_set = set(getattr(ext_obj, 'dead_fibers', []) or [])
        total_fibers = ext_obj.counts.shape[0]
        alive_ids = [i for i in range(total_fibers) if i not in dead_set]

        # Collect smooth models and wavelength arrays for alive fibres only
        alive_smooth = []
        alive_wave = []
        for fid in alive_ids:
            if fid in smooth_models:
                alive_smooth.append(smooth_models[fid])
                alive_wave.append(ext_obj.wave[fid, :] if ext_obj.wave is not None
                                  else np.full(ext_obj.counts.shape[1], np.nan))

        all_smooth_data[ext_name] = {
            'fiber_ids': np.array(alive_ids, dtype=np.int32),
            'smooth': np.array(alive_smooth, dtype=np.float64),
            'wave': np.array(alive_wave, dtype=np.float64),
            'channel': channel,
            'bench': bench,
            'side': side,
        }

        logger.info(f"  {ext_name}: shape={pixel_map.shape} "
                     f"median={np.nanmedian(pixel_map):.4f} "
                     f"std={np.nanstd(pixel_map):.4f} "
                     f"smooth_models={len(alive_ids)} alive fibres")

    # ── Step 5: Write pixel maps FITS ──
    logger.info("Step 5: Writing pixel maps FITS")
    pixel_map_path = os.path.join(output_dir, 'pixel_maps.fits')
    sort_and_write_pixel_maps(
        pixel_maps, pixel_map_path,
        header_info={'FLATMTHD': 'simple_median_gaussian'})

    # ── Step 6: Write smooth models FITS for fibre-to-fibre flat ──
    logger.info("Step 6: Writing smooth models FITS for fibre flat")
    smooth_models_path = os.path.join(output_dir, 'flat_smooth_models.fits')
    _write_smooth_models_fits(all_smooth_data, smooth_models_path,
                              filter_size=filter_size)

    logger.info(f"Pixel flat processing complete: {pixel_map_path}")
    logger.info(f"Smooth models saved to: {smooth_models_path}")

    return {
        'combined_flat_file': combined_flat_file,
        'calibrated_flat_file': calibrated_flat_file,
        'pixel_map_file': pixel_map_path,
        'smooth_models_file': smooth_models_path,
        'processing_status': 'completed',
    }


def apply_flat_field_correction(science_file, pixel_map_file, output_dir=None):
    """
    Apply flat field correction by dividing science frame by pixel map.
    
    This function validates that the science file and pixel map file have the same
    number of extensions in the same order, then performs the division to create
    a flat-fielded science frame.
    
    The science frame extensions are identified by COLOR, BENCH, and SIDE header keywords.
    The pixel map extensions are identified by EXTNAME header keyword.
    These must match in order.
    
    Parameters
    ----------
    science_file : str
        Path to the science FITS file to be corrected
    pixel_map_file : str
        Path to the pixel map FITS file (flat field correction map)
    output_dir : str, optional
        Directory to save the corrected file. If None, saves in same directory
        as science_file.
    
    Returns
    -------
    str
        Path to the output flat-fielded FITS file
        
    Raises
    ------
    ValueError
        If the files have mismatched extensions or ordering
    FileNotFoundError
        If input files don't exist
    """
    from astropy.io import fits
    import os
    import numpy as np
    # Validate input files exist
    if not os.path.exists(science_file):
        raise FileNotFoundError(f"Science file not found: {science_file}")
    if not os.path.exists(pixel_map_file):
        raise FileNotFoundError(f"Pixel map file not found: {pixel_map_file}")
    logger.info(f"Applying flat field correction to {science_file}")
    logger.info(f"Using pixel map: {pixel_map_file}")
    # Open both FITS files
    with fits.open(science_file) as science_hdul, fits.open(pixel_map_file) as pixmap_hdul:
        # Get image extensions (skip primary HDU which is typically header-only)
        science_exts = [hdu for hdu in science_hdul if isinstance(hdu, fits.ImageHDU)]
        pixmap_exts = [hdu for hdu in pixmap_hdul if isinstance(hdu, fits.ImageHDU)]
        # Check number of extensions match
        n_science = len(science_exts)
        n_pixmap = len(pixmap_exts)
        if n_science != n_pixmap:
            # Print all extension names for debugging
            science_names = []
            for i, hdu in enumerate(science_exts):
                color = hdu.header.get('COLOR', 'UNKNOWN')
                bench = hdu.header.get('BENCH', '?')
                side = hdu.header.get('SIDE', '?')
                science_names.append(f"{color}{bench}{side}")
            pixmap_names = [hdu.header.get('EXTNAME', f'EXT{i}') for i, hdu in enumerate(pixmap_exts)]
            print(f"\nExtension count mismatch!")
            print(f"Science file has {n_science} extensions: {science_names}")
            print(f"Pixel map has {n_pixmap} extensions: {pixmap_names}")
            raise ValueError(
                f"Extension count mismatch: science file has {n_science} extensions, "
                f"pixel map has {n_pixmap} extensions"
            )
        logger.info(f"Found {n_science} image extensions in both files")
        # Validate extension names and ordering
        for i, (sci_hdu, pix_hdu) in enumerate(zip(science_exts, pixmap_exts)):
            # Get science frame identifier from COLOR, BENCH, SIDE headers
            color = sci_hdu.header.get('COLOR', 'UNKNOWN')
            bench = sci_hdu.header.get('BENCH', '?')
            side = sci_hdu.header.get('SIDE', '?')
            sci_name = f"{color}{bench}{side}"
            # Get pixel map identifier from EXTNAME
            pix_name = pix_hdu.header.get('EXTNAME', f'EXT{i}')
            # Compare (case-insensitive)
            if sci_name.lower() != pix_name.lower():
                # Print full comparison of all extensions
                print(f"\nExtension ordering mismatch at position {i}!")
                print(f"Expected: {sci_name} (from science COLOR={color}, BENCH={bench}, SIDE={side})")
                print(f"Got:      {pix_name} (from pixel map EXTNAME)")
                print(f"\nFull extension comparison:")
                print(f"{'Index':<8} {'Science File':<20} {'Pixel Map File':<20} {'Match':<10}")
                print("-" * 60)
                for j, (s_hdu, p_hdu) in enumerate(zip(science_exts, pixmap_exts)):
                    s_color = s_hdu.header.get('COLOR', 'UNKNOWN')
                    s_bench = s_hdu.header.get('BENCH', '?')
                    s_side = s_hdu.header.get('SIDE', '?')
                    s_name = f"{s_color}{s_bench}{s_side}"
                    p_name = p_hdu.header.get('EXTNAME', f'EXT{j}')
                    match = "✓" if s_name.lower() == p_name.lower() else "✗"
                    print(f"{j:<8} {s_name:<20} {p_name:<20} {match:<10}")
                raise ValueError(
                    f"Extension name mismatch at position {i}: "
                    f"science has '{sci_name}', pixel map has '{pix_name}'. "
                    f"Extensions must be in the same order."
                )
            logger.debug(f"Extension {i}: {sci_name} - validated")
        logger.info("All extension names and ordering validated successfully")
        # Create output HDU list starting with primary header
        output_hdul = fits.HDUList()
        # Copy primary header from science file
        primary_hdu = fits.PrimaryHDU(header=science_hdul[0].header.copy())
        primary_hdu.header['FLATCORR'] = (True, 'Flat field correction applied')
        primary_hdu.header['FLATFILE'] = (os.path.basename(pixel_map_file), 'Flat field file used')
        output_hdul.append(primary_hdu)
        # Process each extension
        for i, (sci_hdu, pix_hdu) in enumerate(zip(science_exts, pixmap_exts)):
            color = sci_hdu.header.get('COLOR', 'UNKNOWN')
            bench = sci_hdu.header.get('BENCH', '?')
            side = sci_hdu.header.get('SIDE', '?')
            ext_name = f"{color}{bench}{side}"
            logger.info(f"Processing extension {i}: {ext_name}")
            # Get the data
            science_data = sci_hdu.data
            pixmap_data = pix_hdu.data
            # Check dimensions match
            if science_data.shape != pixmap_data.shape:
                raise ValueError(
                    f"Shape mismatch in extension {ext_name}: "
                    f"science {science_data.shape} vs pixel map {pixmap_data.shape}"
                )
            # Perform flat field correction (divide science by pixel map)
            # Handle potential division by zero or invalid values
            with np.errstate(divide='ignore', invalid='ignore'):
                corrected_data = science_data / pixmap_data
                # Replace inf and nan with original values (or zeros)
                # You may want to adjust this behavior
                corrected_data = np.where(np.isfinite(corrected_data),
                                         corrected_data,
                                         science_data)
            # Create output HDU with corrected data
            output_hdu = fits.ImageHDU(data=corrected_data.astype(np.float32),
                                      header=sci_hdu.header.copy())
            output_hdu.header['FLATCORR'] = (True, 'Flat field correction applied')
            output_hdul.append(output_hdu)
            logger.debug(f"Completed flat field correction for {ext_name}")
        # Generate output filename
        science_basename = os.path.basename(science_file)
        science_name, science_ext = os.path.splitext(science_basename)
        output_filename = f"{science_name}_flat_corrected{science_ext}"
        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(science_file)
        output_path = os.path.join(output_dir, output_filename)
        # Write the corrected file
        output_hdul.writeto(output_path, overwrite=True)
        logger.info(f"Flat-fielded science frame saved to: {output_path}")
    return output_path



    
class Thresholding():

    def __init__(self, combined_flat_file, use_bias=None, output_dir=OUTPUT_DIR, trace_dir=CALIB_DIR) -> None:
        """Initialize Thresholding class with combined flat extractions file.

        Parameters
        ----------
        combined_flat_file : str
            Path to combined_flat_extractions.pkl or combined_flat_extractions_calibrated.pkl
            containing 24 ExtractLlamas objects (3 colors × 4 benches × 2 sides) with metadata.
        use_bias : str, optional
            Path to bias file if needed for processing.
        output_dir : str, optional
            Directory for output files.
        trace_dir : str, optional
            Directory containing trace files for fiber mapping.
        """
        self.combined_flat_file = combined_flat_file
        self.use_bias = use_bias
        self.output_dir = output_dir
        self.trace_dir = trace_dir

        logger.info(f"Initializing Thresholding with combined flat file:")
        logger.info(f"  Combined flat: {combined_flat_file}")
        logger.info(f"  Use bias: {use_bias}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Trace directory: {trace_dir}")

        return None
    
    def calculate_fits_all_extensions(self, extraction_file):
        """Calculate pixel thresholds for flat fielding.

        This method calculates the pixel thresholds based on the flat field data
        and returns the threshold values.

        Returns:
            list: List of threshold values for each pixel.
        """
        logger.info(f"Calculating fits for all extensions in {extraction_file}")
        
        # Load the extraction data
        try:
            with open(extraction_file, 'rb') as f:
                extraction_data = pickle.load(f)
            
            logger.info(f"Successfully loaded extraction data")
        except Exception as e:
            logger.error(f"Failed to load extraction data: {str(e)}")
            raise
        
        extract_objs = extraction_data['extractions']
        metadata = extraction_data['metadata']
        
        logger.info(f"Found {len(extract_objs)} extraction objects")
        for i, meta in enumerate(metadata):
            logger.info(f"Extension {i}: {meta.get('bench', 'UNKNOWN')}{meta.get('side', 'UNKNOWN')} {meta.get('channel', 'UNKNOWN')}")
        
        # Dictionary to store results
        results = {}
        
        for ext_idx, item in enumerate(extract_objs):
            
            ext_metadata = metadata[ext_idx]
            benchside = f"{ext_metadata['bench']}{ext_metadata['side']}"
            channel = ext_metadata['channel']
            
            # ---------------------------------------------------------
            # DYNAMIC KNOT SPACING LOGIC
            # ---------------------------------------------------------
            channel_upper = channel.upper()
            if 'RED' in channel_upper:
                # Red channel: use smooth spacing — fringing should NOT be modeled in the flat
                # (fringe correction belongs in a separate fringe-frame step, not the pixel map)
                current_bkspace = 30.0
            elif 'GREEN' in channel_upper:
                # Green is smooth; standard spacing
                current_bkspace = 30.0 
            elif 'BLUE' in channel_upper:
                # Blue is noisier; wide spacing to prevent overfitting to photon noise
                current_bkspace = 50.0 
            else:
                current_bkspace = 30.0 # Default
            # ---------------------------------------------------------

            logger.info(f"Processing extension {ext_idx}: {channel} {benchside} (bkspace={current_bkspace})")
            
            # Create a key for this combination
            ext_key = f"{channel}{benchside}"
            results[ext_key] = {}
            
            nfibers = item.counts.shape[0]
            logger.info(f"Processing {nfibers} fibers for {ext_key}")
            
            for fiber_idx in range(nfibers):
                try:
                    logger.debug(f"Processing fiber {fiber_idx}")
                    
                    # Use fit_spectrum_to_xshift for this fiber with dynamic bkspace
                    # bkspace controls the distance between B-spline knots in pixel units
                    fiber_fit = fit_spectrum_to_xshift(
                        item, 
                        fiber_idx, 
                        bkspace=current_bkspace
                    )
                    
                    # Get the bspline model from the fit result
                    bspline_model = fiber_fit['bspline_model']
                    
                    # Get the fitted values at the original x coordinates
                    y_predicted = fiber_fit['y_fit']
                    
                    # Calculate residuals (actual - predicted)
                    residuals = fiber_fit['residuals']
                    rms_residual = fiber_fit['rms_residual']
                    relative_rms = fiber_fit['relative_rms']

                    # Store results for this fiber
                    results[ext_key][fiber_idx] = {
                        'xshift': item.xshift[fiber_idx, :],
                        'xshift_clean': fiber_fit['xshift_clean'],
                        'counts_clean': fiber_fit['counts_clean'],
                        'y_predicted': y_predicted,
                        'residuals': residuals,
                        'rms_residual': rms_residual,
                        'relative_rms': relative_rms,
                        'xmodel': fiber_fit['xmodel'],
                        'y_fit': fiber_fit['y_fit'],
                        'bspline_model': bspline_model
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing fiber {fiber_idx}: {str(e)}")
                    continue
            
            logger.info(f"Completed processing {len(results[ext_key])} fibers for {ext_key}")
        
        # Save the results
        output_file = os.path.splitext(os.path.basename(extraction_file))[0] + '_fits.pkl'
        output_path = os.path.join(self.output_dir, output_file)
        
        try:
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Saved fitting results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")

        self.fit_results = results
        
        return results
    

    def _generate_single_pixel_map(self, ext_name, extraction_obj, trace_obj, ext_results):
        """
        Creates a 2D map for a single extension where:
        Pixel Value = Observed Counts / Smooth B-Spline Model.
        Includes a histogram sanity check plot.
        """
        logger.info(f"Generating 2D Sensitivity Map and Sanity Plot for {ext_name}")

        # 1. Initialize the 2D map and data containers
        # We start with 1.0 (no correction)
        pixel_map = np.ones_like(trace_obj.fiberimg, dtype=np.float32)
        fiber_img = trace_obj.fiberimg
        prof_img = trace_obj.profimg    # Profile weight image — used to mask trace edges
        PROF_MIN = 0.05                 # Minimum profile weight; below this, leave correction = 1.0
        all_ratios_for_plot = []

        # 2. Iterate through each fiber to calculate the pixel-to-pixel ratio
        for fiber_idx, fit_data in ext_results.items():
            if 'y_predicted' not in fit_data:
                continue

            # Get the smooth B-spline model and the raw extracted counts
            smooth_model = fit_data['y_predicted']
            actual_counts = extraction_obj.counts[fiber_idx]

            # Calculate Ratio = Actual / Smooth
            # If Actual > Smooth, the pixel is more sensitive than average ('hot')
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_1d = np.divide(actual_counts, smooth_model)
                # Clean up NaNs or Infs (edges/low signal regions)
                ratio_1d[~np.isfinite(ratio_1d)] = 1.0
            # Clip to physically meaningful range — variations >±50% from unity are artefacts
            ratio_1d = np.clip(ratio_1d, 0.5, 2.0)

            # Collect values for the histogram (excluding the 1.0 fillers later)
            all_ratios_for_plot.extend(ratio_1d.tolist())

            # 3. Project the 1D ratio back to the 2D detector grid
            # Find pixels on the CCD assigned to this fiber index
            mask = (fiber_img == fiber_idx)
            rows, cols = np.where(mask)

            # Map the ratio to the coordinates.
            # Note: c (column) corresponds to the spectral index in the 1D array.
            # Only apply the correction where the profile weight is substantial; edge pixels
            # (low profimg weight) are left at 1.0 to avoid blow-up from trace wing artefacts.
            for r, c in zip(rows, cols):
                if c < len(ratio_1d):
                    if prof_img[r, c] >= PROF_MIN:
                        pixel_map[r, c] = ratio_1d[c]
                    # else: pixel_map[r, c] stays 1.0 (no correction at trace edges)

        # 4. Generate the Sanity Check Plot
        if all_ratios_for_plot:
            # Filter: Ignore the '1.0' fillers and extreme outliers for a clean histogram
            plot_data = np.array(all_ratios_for_plot)
            plot_data = plot_data[(plot_data != 1.0) & (plot_data > 0.5) & (plot_data < 1.5)]

            if len(plot_data) > 0:
                plt.figure(figsize=(10, 6))
                plt.hist(plot_data, bins=100, color='#2ab0ff', edgecolor='black', alpha=0.7)

                # Statistics
                mu, std = np.mean(plot_data), np.std(plot_data)

                # Formatting the plot
                plt.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal (1.0)')
                plt.title(f"Sensitivity Distribution: {ext_name}\nMean: {mu:.4f} | Std Dev: {std:.4f}")
                plt.xlabel("Pixel Ratio (Actual / Model)")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(axis='y', alpha=0.3)

                # Save to output directory
                if not os.path.exists(self.output_dir):
                    os.makedirs(self.output_dir)

                plot_path = os.path.join(self.output_dir, f"map_check_{ext_name}.png")
                plt.savefig(plot_path)
                plt.close()
                logger.info(f"Sanity plot saved to: {plot_path}")
            else:
                logger.warning(f"No valid ratio data found for plot in {ext_name}")

        return pixel_map, {}
    
    def plot_pixel_distribution(self, all_ratios, ext_name):
        """
        Plots a histogram of the pixel-to-pixel ratios to ensure they center at 1.0.
        """
        plt.figure(figsize=(10, 6))
        
        # Flatten the list of ratios and remove the '1.0' fillers (pixels not in fibers)
        data = np.array(all_ratios)
        data = data[data != 1.0] # Only look at active fiber pixels
        
        # Filter out extreme outliers for a cleaner plot (e.g., keep 0.5 to 1.5)
        data = data[(data > 0.5) & (data < 1.5)]
    
        plt.hist(data, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(1.0, color='red', linestyle='dashed', linewidth=2, label='Ideal Center (1.0)')
        
        mu, std = np.mean(data), np.std(data)
        plt.title(f"Pixel-to-Pixel Variation Distribution: {ext_name}\nMean: {mu:.4f}, Std: {std:.4f}")
        plt.xlabel("Sensitivity Ratio (Actual / Model)")
        plt.ylabel("Pixel Count")
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
    
        # Save the plot to the output directory
        plot_path = os.path.join(self.output_dir, f"sanity_check_{ext_name}.png")
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Sanity check plot saved to {plot_path}")

    def generate_thresholds(self):
        """Generate thresholds for flat fielding based on science data."""

        pixel_maps = {}
        bad_pixels = {}

        #grabbing the tracefiles

        trace_files = glob.glob(os.path.join(self.trace_dir, 'LLAMAS*traces.pkl'))

        # 1. grab each extension from the combined extraction file
        #2. for each extension, find the matching trace file, and the matching fit result
        with open(self.combined_flat_file, 'rb') as f:
            extraction_data = pickle.load(f)
        extract_objs = extraction_data['extractions']
        metadata = extraction_data['metadata']

        for ext_idx, item in enumerate(extract_objs):
            ext_metadata = metadata[ext_idx]
            benchside = f"{ext_metadata['bench']}{ext_metadata['side']}"
            channel = ext_metadata['channel']
            ext_name = f"{channel}{benchside}"

            logger.info(f"Generating pixel map for extension {ext_idx}: {ext_name}")

            # Find matching trace file
            matching_trace_file = None
            for trace_file in trace_files:
                with open(trace_file, 'rb') as tf:
                    trace_obj = pickle.load(tf)
                    trace_key = f"{trace_obj.channel}{trace_obj.bench}{trace_obj.side}"
                    if trace_key.lower() == ext_name.lower():
                        matching_trace_file = trace_file
                        break

            if matching_trace_file is None:
                logger.warning(f"No matching trace file found for extension {ext_name}, skipping")
                continue

            # Load the trace object
            with open(matching_trace_file, 'rb') as tf:
                trace_obj = pickle.load(tf)

            # Get the fit results for this extension
            if ext_name not in self.fit_results:
                logger.warning(f"No fit results found for extension {ext_name}, skipping")
                continue

            ext_fit_results = self.fit_results[ext_name]

            # Generate the pixel map
            pixel_map, bad_pixel_info = self._generate_single_pixel_map(
                ext_name, item, trace_obj, ext_fit_results
            )

            pixel_maps[ext_name] = pixel_map
            bad_pixels[ext_name] = bad_pixel_info

            # Optionally save individual pixel maps
            # output_filename = os.path.join(output_dir, f'pixel_map_{ext_name}.fits')
            # hdu = fits.PrimaryHDU(data=pixel_map.astype(np.float32))
            # hdu.writeto(output_filename, overwrite=True)
            # logger.info(f"Saved pixel map for {ext_name} to {output_filename}")
        self.map_filename = os.path.join(self.output_dir, 'pixel_maps.fits')
        sort_and_write_pixel_maps(pixel_maps, self.map_filename)

        return pixel_maps
