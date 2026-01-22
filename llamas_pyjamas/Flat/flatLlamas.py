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
<<<<<<< HEAD
from llamas_pyjamas.Utils.utils import concat_extractions
from llamas_pyjamas.Arc.arcLlamasMulti import arcTransfer
=======
from llamas_pyjamas.Utils.utils import concat_extractions, is_wavelength_solution_useable
from llamas_pyjamas.Arc.arcLlamas import arcTransfer
>>>>>>> ce70df31f7d94c3f2abe4d2b0d98f1c4cd12533c
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

from pypeit.core.fitting import iterfit
import pickle
from datetime import datetime



# Set up logging
def setup_logger(verbose=False):
    """Setup logger with configurable console verbosity."""
    log_dir = os.path.join(OUTPUT_DIR, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'flatLlamas_{timestamp}.log')

    # Configure the logger
    logger = logging.getLogger('flatLlamas')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create handlers
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    # Set levels
    file_handler.setLevel(logging.DEBUG)
    console_level = logging.INFO if verbose else logging.WARNING
    console_handler.setLevel(console_level)

    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # Add formatters to handlers
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    if verbose:
        logger.info(f"Verbose logging enabled. Log file: {log_file}")
    
    return logger

# Initialize with default settings
logger = setup_logger(verbose=False)


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
        ext_with_idx.append((idx, ext_name, pixel_map))
    
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
    for idx, ext_name, pixel_map in ext_with_idx:
        img_hdu = fits.ImageHDU(data=pixel_map, name=ext_name)
        img_hdu.header['EXTNAME'] = ext_name
        img_hdu.header['EXTVER'] = idx
        img_hdu.header['COMMENT'] = f'Flat field pixel map for {ext_name}'
        
        hdu_list.append(img_hdu)
    
    # Write to file
    hdulist = fits.HDUList(hdu_list)
    hdulist.writeto(output_path, overwrite=True)
    logger.info(f"Wrote {len(ext_with_idx)} pixel maps to {output_path}")


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
                               trace_dir=CALIB_DIR, verbose=False):
    """Process complete flat field workflow with wavelength calibration and pixel mapping.

    This function implements the complete flat field processing workflow:
    1. Extract individual color flat fields using produce_flat_extractions
    2. Combine all extractions into a single .pkl file
    3. Apply wavelength solution from arc calibration
    4. Fit B-splines to xshift vs counts for each fiber
    5. Generate per-pixel flat field correction images

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

    Returns:
        dict: Dictionary containing processing results and output file paths
    """
    # Setup logger with appropriate verbosity level
    global logger
    logger = setup_logger(verbose=verbose)
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
        verbose=verbose
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

    if not is_wavelength_solution_useable(arc_dict):
        logger.critical(f"CRITICAL ERROR: Arc calibration file {arc_calib_file} is not useable.")
        raise ValueError(f"Arc calibration file {arc_calib_file} is not useable.")
    
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
    
    # Step 5: Generate pixel maps for each channel/bench combination
    logger.info("Step 5: Generating pixel maps for each channel/bench combination")
    
    pixel_map_results = threshold_processor.generate_all_pixel_maps() #generate_complete_pixel_maps()
    

<<<<<<< HEAD
    # Step 6: Create normalized flat field FITS file using notebook method
    logger.info("Step 6: Creating normalized flat field FITS file using B-spline division method")

    try:
        logger.info("Starting normalized flat field creation...")
        normalized_flat_field_file = threshold_processor.generate_normalized_flat_from_bspline_fits(
            flat_dict_calibrated=flat_dict_calibrated,
            fit_results=fit_results,
            trace_dir=trace_dir
        )
        logger.info(f"✓ Successfully created normalized flat field: {os.path.basename(normalized_flat_field_file)}")

    except Exception as e:
        logger.error(f"CRITICAL ERROR: Failed to create normalized flat field: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        normalized_flat_field_file = None

        # Also raise the exception to ensure calling code knows about the failure
        raise
=======
>>>>>>> ce70df31f7d94c3f2abe4d2b0d98f1c4cd12533c
    
    results = {
        'combined_flat_file': combined_flat_file,
        'calibrated_flat_file': calibrated_flat_file,
        'fit_results': fit_results,
        'pixel_map_results': pixel_map_results,
        'processing_status': 'completed'
    }
    
    logger.info("Complete flat field processing workflow finished successfully")
    return results



        
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
            
            # Assuming item has a 'counts' attribute which is a 2D array
            
            ext_metadata = metadata[ext_idx]
            benchside = f"{ext_metadata['bench']}{ext_metadata['side']}"
            channel = ext_metadata['channel']
            
            logger.info(f"Processing extension {ext_idx}: {channel} {benchside}")
            
            # Create a key for this combination
            ext_key = f"{channel}{benchside}"
            results[ext_key] = {}
            
            nfibers = item.counts.shape[0]
            logger.info(f"Processing {nfibers} fibers for {ext_key}")
            
            for fiber_idx in range(nfibers):
                try:
                    logger.debug(f"Processing fiber {fiber_idx}")
                    
                    # Use fit_spectrum_to_xshift for this fiber
                    fiber_fit = fit_spectrum_to_xshift(item, fiber_idx)
                    
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
                        'xshift': item.xshift[fiber_idx, :],  # Original xshift array for direct pixel mapping
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
        Generate a single pixel map for one extension using B-spline fits and trace object.

        FIXED: Uses interpolation to map 2D image columns to extraction wavelengths.
        Previously, direct indexing assumed col_indices could index xshift array,
        which caused incorrect wavelength mapping.

        Args:
            fiber_fits (dict): Dictionary of B-spline fits for each fiber
            trace_obj: Trace object containing fiberimg and other trace information

        Returns:
            np.ndarray: 2D pixel map with flat field values
        """
        logger.debug(f"Generating pixel map for {trace_obj.channel} {trace_obj.bench}{trace_obj.side}")

        # Get the fiber image from the trace object
        fib_img = trace_obj.fiberimg

        mask_nonneg1 = fib_img != -1
        n_nonneg1 = np.count_nonzero(mask_nonneg1)
        vals = fib_img[mask_nonneg1]
    
        
        # Create an empty array matching the shape of the fiber image
        pixel_map = np.ones_like(fib_img, dtype=np.float32)

        # Dictionary to store bad pixel information
        bad_pixels = {
            'coords': [],  # List of (row, col) tuples
            'fiber_idx': [],  # Which fiber the bad pixel belongs to
            'spectral_idx': [],  # Position in the 1D spectrum
            'reason': []  # Why it's bad (e.g., 'nan_in_normalized', 'inf_in_normalized', etc.)
            }

        # Checking the number of fibres matched the fit results
        key_len = (ext_results.keys())
        _fibs = np.unique(vals, return_counts=True)
        fib_len = len(_fibs)
        logger.debug(f"Extension {ext_name}: Found {fib_len} unique fibers in trace image")
        if key_len != fib_len:
            logger.warning(f"Extension {ext_name}: Mismatch in number of fibers between trace image ({fib_len}) and fit results ({key_len})")   


        for fiber_idx in ext_results.keys():

            y_predicted = ext_results[fiber_idx]['y_predicted']
            fibre_counts = extraction_obj.counts[fiber_idx]
            try:
                normalised_flat = fibre_counts / y_predicted
            
                fibre_mask = fib_img == fiber_idx
                fibre_rows, fibre_cols = np.where(fibre_mask)

                unique_cols = np.unique(fibre_cols)

                for spectral_idx, col in enumerate(unique_cols):
                    rows_in_col = fibre_rows[fibre_cols == col]

                    if spectral_idx < len(normalised_flat):
                        norm_value = normalised_flat[spectral_idx]

                        if np.isnan(norm_value):
                            pixel_map[rows_in_col, col] = 1.0
                            for row in rows_in_col:
                                bad_pixels['coords'].append((row, col))
                                bad_pixels['fiber_idx'].append(fiber_idx)
                                bad_pixels['spectral_idx'].append(spectral_idx)
                                bad_pixels['reason'].append('nan_in_normalized')
                            
                        elif np.isinf(norm_value):
                            pixel_map[rows_in_col, col] = 1.0

                            for row in rows_in_col:
                                bad_pixels['coords'].append((row, col))
                                bad_pixels['fiber_idx'].append(fiber_idx)
                                bad_pixels['spectral_idx'].append(spectral_idx)
                                bad_pixels['reason'].append('inf_in_normalized')
                            
                        else:
                            pixel_map[rows_in_col, col] = norm_value
                        
            except Exception as e:
                logger.error(f"Error normalizing fiber {fiber_idx}: {str(e)}")
                continue

            # Print summary of bad pixels
            print(f"Total bad pixels found: {len(bad_pixels['coords'])}")
            print(f"Fibers affected: {len(set(bad_pixels['fiber_idx']))}")

            # Get counts by reason
            
            reason_counts = Counter(bad_pixels['reason'])
            print("Bad pixel breakdown:")
            for reason, count in reason_counts.items():
                print(f"  {reason}: {count}")

<<<<<<< HEAD
                # Get the column indices of the fiber pixels in this row (2D image columns)
                col_indices = np.where(row_pixels)[0]

                # FIX: Interpolate xshift values at actual image column positions
                # This maps: 2D image columns → extraction positions → wavelengths
                xshift_at_cols = np.interp(col_indices, extraction_cols, xshift_1d)

                # Evaluate the B-spline model at interpolated wavelength positions
                try:
                    predicted_values = bspline_model.value(xshift_at_cols)[0]
                    # Assign the predicted values to the pixel map
                    pixel_map[row, col_indices] = predicted_values
                except Exception as e:
                    logger.debug(f"Error evaluating B-spline for fiber {fiber_idx}, row {row}: {str(e)}")
                    continue

            processed_fibers += 1

            if processed_fibers % 50 == 0:
                logger.debug(f"Processed {processed_fibers}/{len(fiber_fits)} fibers")

        # Set unassigned pixels (outside fiber traces) to 1.0 for proper flat field normalization
        nan_mask = np.isnan(pixel_map)
        pixel_map[nan_mask] = 1.0

        # Check statistics
        nan_count = np.sum(nan_mask)  # Count of pixels that were NaN (now set to 1.0)
        total_pixels = pixel_map.size
        traced_pixels = total_pixels - nan_count

        logger.info(f"Pixel map normalization: {traced_pixels}/{total_pixels} traced pixels, "
                   f"{nan_count}/{total_pixels} untraced pixels set to 1.0 "
                   f"({100*nan_count/total_pixels:.1f}% untraced)")

        return pixel_map

    def generate_normalized_flat_from_bspline_fits(self, flat_dict_calibrated, fit_results, trace_dir):
        """Generate normalized flat field maps using notebook method.

        Creates a 24-extension MEF file where each extension contains:
            normalized_flat = flat_counts / bspline_predictions

        This removes the spectral shape while preserving pixel-to-pixel variations.

        Args:
            flat_dict_calibrated: Dictionary with wavelength-calibrated flat extractions
            fit_results: Dictionary from calculate_fits_all_extensions with B-spline fits
            trace_dir: Directory containing trace files with fiberimg

        Returns:
            str: Path to normalized flat field MEF file
        """
        logger.info("Generating normalized flat field maps using B-spline division method")

        # Create output MEF file
        output_file = os.path.join(self.output_dir, 'normalized_flat_field.fits')

        # Create primary HDU
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['OBJECT'] = 'Normalized Flat Field'
        primary_hdu.header['METHOD'] = 'Bspline Division'
        primary_hdu.header['COMMENT'] = 'Flat counts divided by B-spline fit'
        primary_hdu.header['NEXTEN'] = (24, 'Number of extensions')

        hdu_list = [primary_hdu]

        # Process each of 24 extensions
        extensions = list(idx_lookup.keys())  # List of (channel, bench, side) tuples

        for ext_idx, (channel, bench, side) in enumerate(sorted(extensions, key=lambda x: idx_lookup[x])):
            ext_key = f"{channel}{bench}{side}"
            logger.info(f"Processing extension {ext_idx+1}/24: {ext_key}")

            try:
                # Find matching extraction in flat_dict_calibrated
                extraction_idx = None
                for i, meta in enumerate(flat_dict_calibrated['metadata']):
                    if (meta['channel'] == channel and
                        str(meta['bench']) == str(bench) and
                        meta['side'] == side):
                        extraction_idx = i
                        break

                if extraction_idx is None:
                    logger.error(f"No extraction found for {ext_key}")
                    # Create empty extension filled with 1.0
                    normalized_data = np.ones((4112, 4096), dtype=np.float32)
                else:
                    # Load trace file for this extension
                    trace_file = os.path.join(trace_dir, f'LLAMAS_{channel}_{bench}_{side}_traces.pkl')
                    if not os.path.exists(trace_file):
                        logger.error(f"Trace file not found: {trace_file}")
                        normalized_data = np.ones((4112, 4096), dtype=np.float32)
                    else:
                        with open(trace_file, 'rb') as f:
                            traces = pickle.load(f)
                        fib_img = traces.fiberimg

                        # Get flat extraction data
                        flat_data = flat_dict_calibrated['extractions'][extraction_idx]

                        # Get B-spline fit results for this extension
                        ext_results = fit_results.get(ext_key, {})

                        # Initialize normalized image to 1.0 (no correction)
                        normalized_data = np.ones_like(fib_img, dtype=np.float32)

                        # Track bad pixels
                        bad_pixel_count = 0

                        # Iterate over all fibers in this extension
                        for fiber_idx in ext_results.keys():
                            # Get B-spline predictions (smooth continuum)
                            y_predicted = ext_results[fiber_idx]['y_predicted']

                            # Get actual flat counts
                            fiber_counts = flat_data.counts[fiber_idx]

                            # Normalize: divide counts by B-spline fit
                            # This removes spectral shape, keeps pixel-to-pixel variations
                            normalized_flat_1d = fiber_counts / y_predicted

                            # Get pixels belonging to this fiber from trace
                            fiber_mask = fib_img == fiber_idx
                            fiber_rows, fiber_cols = np.where(fiber_mask)

                            # Get unique columns (spectral direction)
                            unique_cols = np.unique(fiber_cols)

                            # Map 1D spectral data to 2D image columns
                            for spectral_idx, col in enumerate(unique_cols):
                                # Find all rows in this column for this fiber
                                rows_in_col = fiber_rows[fiber_cols == col]

                                # Assign normalized value
                                if spectral_idx < len(normalized_flat_1d):
                                    value = normalized_flat_1d[spectral_idx]

                                    # Handle NaN/Inf
                                    if np.isnan(value) or np.isinf(value):
                                        normalized_data[rows_in_col, col] = 1.0
                                        bad_pixel_count += len(rows_in_col)
                                    else:
                                        normalized_data[rows_in_col, col] = value

                        logger.info(f"  {ext_key}: Bad pixels = {bad_pixel_count}")

                # Create HDU for this extension
                hdu = fits.ImageHDU(data=normalized_data)
                hdu.header['EXTNAME'] = ext_key
                hdu.header['CHANNEL'] = channel.upper()
                hdu.header['BENCH'] = str(bench)
                hdu.header['SIDE'] = side
                hdu.header['EXTVER'] = ext_idx + 1

                # Add statistics
                traced_mask = normalized_data != 1.0
                if np.any(traced_mask):
                    traced_values = normalized_data[traced_mask]
                    hdu.header['DATAMEAN'] = float(np.mean(traced_values))
                    hdu.header['DATAMED'] = float(np.median(traced_values))
                    hdu.header['DATAMIN'] = float(np.min(traced_values))
                    hdu.header['DATAMAX'] = float(np.max(traced_values))
                    hdu.header['NPIX'] = int(np.sum(traced_mask))
                    n_bad = int(bad_pixel_count if 'bad_pixel_count' in locals() else 0)
                    hdu.header['NBADFPIX'] = n_bad

                hdu_list.append(hdu)

            except Exception as e:
                logger.error(f"Error processing {ext_key}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Add empty extension
                empty_data = np.ones((4112, 4096), dtype=np.float32)
                hdu = fits.ImageHDU(data=empty_data)
                hdu.header['EXTNAME'] = ext_key
                hdu.header['CHANNEL'] = channel.upper()
                hdu.header['BENCH'] = str(bench)
                hdu.header['SIDE'] = side
                hdu.header['EXTVER'] = ext_idx + 1
                hdu.header['ERROR'] = str(e)[:68]  # FITS header value limit
                hdu_list.append(hdu)

        # Write MEF file
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(output_file, overwrite=True)
        logger.info(f"✓ Normalized flat field saved: {output_file}")

        return output_file

    def generate_thresholds(self):
        """Generate thresholds for flat fielding based on science data.
=======
        return pixel_map, bad_pixels
    
    def generate_all_pixel_maps(self):
>>>>>>> ce70df31f7d94c3f2abe4d2b0d98f1c4cd12533c

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
