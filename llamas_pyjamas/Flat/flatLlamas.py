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
    
    # # Step 5: Generate pixel maps for each channel/bench combination
    # logger.info("Step 5: Generating pixel maps for each channel/bench combination")
    
    # pixel_map_results = threshold_processor.generate_all_pixel_maps() #generate_complete_pixel_maps()
    

    # # Step 6: Create normalized flat field FITS file using notebook method
    # logger.info("Step 6: Creating normalized flat field FITS file using B-spline division method")

    # try:
    #     logger.info("Starting normalized flat field creation...")
    #     normalized_flat_field_file = threshold_processor.generate_normalized_flat_from_bspline_fits(
    #         flat_dict_calibrated=flat_dict_calibrated,
    #         fit_results=fit_results,
    #         trace_dir=trace_dir
    #     )
    #     logger.info(f"✓ Successfully created normalized flat field: {os.path.basename(normalized_flat_field_file)}")

    # except Exception as e:
    #     logger.error(f"CRITICAL ERROR: Failed to create normalized flat field: {str(e)}")
    #     import traceback
    #     logger.error(f"Full traceback: {traceback.format_exc()}")
    #     normalized_flat_field_file = None

    #     # Also raise the exception to ensure calling code knows about the failure
    #     raise
    
    # # Step 7: Generate true 2D pixel QE maps (if requested)
    # pixel_qe_file = None
    # if pixel_qe_mode:
    #     logger.info("Step 7: Generating true 2D pixel QE flat field maps")
    #     try:
    #         flat_files = {
    #             'red': red_flat_file,
    #             'green': green_flat_file,
    #             'blue': blue_flat_file
    #         }
    #         pixel_qe_file = threshold_processor.generate_2d_pixel_qe_map(
    #             flat_dict_calibrated=flat_dict_calibrated,
    #             fit_results=fit_results,
    #             trace_dir=trace_dir,
    #             flat_files=flat_files,
    #             bias_file=use_bias
    #         )
    #         logger.info(f"Successfully created pixel QE flat field: {os.path.basename(pixel_qe_file)}")
    #     except Exception as e:
    #         logger.error(f"Failed to create pixel QE flat field: {str(e)}")
    #         import traceback
    #         logger.error(f"Full traceback: {traceback.format_exc()}")
    #         pixel_qe_file = None

    # results = {
    #     'combined_flat_file': combined_flat_file,
    #     'calibrated_flat_file': calibrated_flat_file,
    #     'fit_results': fit_results,
    #     'pixel_map_results': pixel_map_results,
    #     'pixel_qe_file': pixel_qe_file,
    #     'processing_status': 'completed'
    # }

    # logger.info("Complete flat field processing workflow finished successfully")
    # return results



        
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
                # Red channel has fringing; needs tight spacing to follow the ripples
                current_bkspace = 8.0 
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
    
    # def calculate_fits_all_extensions(self, extraction_file):
    #     """Calculate pixel thresholds for flat fielding.

    #     This method calculates the pixel thresholds based on the flat field data
    #     and returns the threshold values.

    #     Returns:
    #         list: List of threshold values for each pixel.
    #     """
    #     logger.info(f"Calculating fits for all extensions in {extraction_file}")
        
    #     # Load the extraction data
    #     try:
    #         with open(extraction_file, 'rb') as f:
    #             extraction_data = pickle.load(f)
            
    #         logger.info(f"Successfully loaded extraction data")
    #     except Exception as e:
    #         logger.error(f"Failed to load extraction data: {str(e)}")
    #         raise
        
    #     extract_objs = extraction_data['extractions']
    #     metadata = extraction_data['metadata']
        
    #     logger.info(f"Found {len(extract_objs)} extraction objects")
    #     for i, meta in enumerate(metadata):
    #         logger.info(f"Extension {i}: {meta.get('bench', 'UNKNOWN')}{meta.get('side', 'UNKNOWN')} {meta.get('channel', 'UNKNOWN')}")
        
    #     # Dictionary to store results
    #     results = {}
        
    #     for ext_idx, item in enumerate(extract_objs):
            
    #         # Assuming item has a 'counts' attribute which is a 2D array
            
    #         ext_metadata = metadata[ext_idx]
    #         benchside = f"{ext_metadata['bench']}{ext_metadata['side']}"
    #         channel = ext_metadata['channel']
            
    #         logger.info(f"Processing extension {ext_idx}: {channel} {benchside}")
            
    #         # Create a key for this combination
    #         ext_key = f"{channel}{benchside}"
    #         results[ext_key] = {}
            
    #         nfibers = item.counts.shape[0]
    #         logger.info(f"Processing {nfibers} fibers for {ext_key}")
            
    #         for fiber_idx in range(nfibers):
    #             try:
    #                 logger.debug(f"Processing fiber {fiber_idx}")
                    
    #                 # Use fit_spectrum_to_xshift for this fiber
    #                 fiber_fit = fit_spectrum_to_xshift(item, fiber_idx)
                    
    #                 # Get the bspline model from the fit result
    #                 bspline_model = fiber_fit['bspline_model']
                    
    #                 # Get the fitted values at the original x coordinates
    #                 y_predicted = fiber_fit['y_fit']
                    
    #                 # Calculate residuals (actual - predicted)
    #                 residuals = fiber_fit['residuals']
    #                 rms_residual = fiber_fit['rms_residual']
    #                 relative_rms = fiber_fit['relative_rms']

    #                 # Store results for this fiber
    #                 results[ext_key][fiber_idx] = {
    #                     'xshift': item.xshift[fiber_idx, :],  # Original xshift array for direct pixel mapping
    #                     'xshift_clean': fiber_fit['xshift_clean'],
    #                     'counts_clean': fiber_fit['counts_clean'],
    #                     'y_predicted': y_predicted,
    #                     'residuals': residuals,
    #                     'rms_residual': rms_residual,
    #                     'relative_rms': relative_rms,
    #                     'xmodel': fiber_fit['xmodel'],
    #                     'y_fit': fiber_fit['y_fit'],
    #                     'bspline_model': bspline_model
    #                 }
                    
    #             except Exception as e:
    #                 logger.error(f"Error processing fiber {fiber_idx}: {str(e)}")
    #                 continue
            
    #         logger.info(f"Completed processing {len(results[ext_key])} fibers for {ext_key}")
        
    #     # Save the results
    #     output_file = os.path.splitext(os.path.basename(extraction_file))[0] + '_fits.pkl'
    #     output_path = os.path.join(self.output_dir, output_file)
        
    #     try:
    #         with open(output_path, 'wb') as f:
    #             pickle.dump(results, f)
    #         logger.info(f"Saved fitting results to {output_path}")
    #     except Exception as e:
    #         logger.error(f"Error saving results: {str(e)}")

    #     self.fit_results = results
        
    #     return results

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

            # Collect values for the histogram (excluding the 1.0 fillers later)
            all_ratios_for_plot.extend(ratio_1d.tolist())

            # 3. Project the 1D ratio back to the 2D detector grid
            # Find pixels on the CCD assigned to this fiber index
            mask = (fiber_img == fiber_idx)
            rows, cols = np.where(mask)

            # Map the ratio to the coordinates. 
            # Note: c (column) corresponds to the spectral index in the 1D array
            for r, c in zip(rows, cols):
                if c < len(ratio_1d):
                    pixel_map[r, c] = ratio_1d[c]

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
    
    # def _generate_single_pixel_map(self, ext_name, extraction_obj, trace_obj, ext_results):
    #     """
    #     Generate a single pixel map for one extension using B-spline fits and trace object.

    #     FIXED: Uses interpolation to map 2D image columns to extraction wavelengths.
    #     Previously, direct indexing assumed col_indices could index xshift array,
    #     which caused incorrect wavelength mapping.

    #     Args:
    #         fiber_fits (dict): Dictionary of B-spline fits for each fiber
    #         trace_obj: Trace object containing fiberimg and other trace information

    #     Returns:
    #         np.ndarray: 2D pixel map with flat field values
    #     """
    #     logger.debug(f"Generating pixel map for {trace_obj.channel} {trace_obj.bench}{trace_obj.side}")

    #     # Get the fiber image from the trace object
    #     fib_img = trace_obj.fiberimg

    #     mask_nonneg1 = fib_img != -1
    #     n_nonneg1 = np.count_nonzero(mask_nonneg1)
    #     vals = fib_img[mask_nonneg1]
    
        
    #     # Create an empty array matching the shape of the fiber image
    #     pixel_map = np.ones_like(fib_img, dtype=np.float32)

    #     # Dictionary to store bad pixel information
    #     bad_pixels = {
    #         'coords': [],  # List of (row, col) tuples
    #         'fiber_idx': [],  # Which fiber the bad pixel belongs to
    #         'spectral_idx': [],  # Position in the 1D spectrum
    #         'reason': []  # Why it's bad (e.g., 'nan_in_normalized', 'inf_in_normalized', etc.)
    #         }

    #     # Checking the number of fibres matched the fit results
    #     key_len = (ext_results.keys())
    #     _fibs = np.unique(vals, return_counts=True)
    #     fib_len = len(_fibs)
    #     logger.debug(f"Extension {ext_name}: Found {fib_len} unique fibers in trace image")
    #     if key_len != fib_len:
    #         logger.warning(f"Extension {ext_name}: Mismatch in number of fibers between trace image ({fib_len}) and fit results ({key_len})")   


    #     for fiber_idx in ext_results.keys():

    #         y_predicted = ext_results[fiber_idx]['y_predicted']
    #         fibre_counts = extraction_obj.counts[fiber_idx]
    #         try:
    #             normalised_flat = fibre_counts / y_predicted
            
    #             fibre_mask = fib_img == fiber_idx
    #             fibre_rows, fibre_cols = np.where(fibre_mask)

    #             unique_cols = np.unique(fibre_cols)

    #             for spectral_idx, col in enumerate(unique_cols):
    #                 rows_in_col = fibre_rows[fibre_cols == col]

    #                 if col < len(normalised_flat):
    #                     norm_value = normalised_flat[col]

    #                     if np.isnan(norm_value):
    #                         pixel_map[rows_in_col, col] = 1.0
    #                         for row in rows_in_col:
    #                             bad_pixels['coords'].append((row, col))
    #                             bad_pixels['fiber_idx'].append(fiber_idx)
    #                             bad_pixels['spectral_idx'].append(spectral_idx)
    #                             bad_pixels['reason'].append('nan_in_normalized')
                            
    #                     elif np.isinf(norm_value):
    #                         pixel_map[rows_in_col, col] = 1.0

    #                         for row in rows_in_col:
    #                             bad_pixels['coords'].append((row, col))
    #                             bad_pixels['fiber_idx'].append(fiber_idx)
    #                             bad_pixels['spectral_idx'].append(spectral_idx)
    #                             bad_pixels['reason'].append('inf_in_normalized')
                            
    #                     else:
    #                         pixel_map[rows_in_col, col] = norm_value
                        
    #         except Exception as e:
    #             logger.error(f"Error normalizing fiber {fiber_idx}: {str(e)}")
    #             continue

    #         # Print summary of bad pixels
    #         print(f"Total bad pixels found: {len(bad_pixels['coords'])}")
    #         print(f"Fibers affected: {len(set(bad_pixels['fiber_idx']))}")

    #         # Get counts by reason
            
    #         reason_counts = Counter(bad_pixels['reason'])
    #         print("Bad pixel breakdown:")
    #         for reason, count in reason_counts.items():
    #             print(f"  {reason}: {count}")

        # return pixel_map, bad_pixels
    
    import matplotlib.pyplot as plt

# Inside the Thresholding class...

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
                                if col < len(normalized_flat_1d):
                                    value = normalized_flat_1d[col]

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
    
    def generate_2d_pixel_qe_map(self, flat_dict_calibrated, fit_results, trace_dir,
                                    flat_files, bias_file=None,
                                    clip_low=0.5, clip_high=1.5,
                                    min_profile_weight=0.01):
        """Generate a true 2D pixel-level QE flat field map.

        Unlike generate_normalized_flat_from_bspline_fits() which assigns a single
        normalization value to all spatial pixels within a fibre, this method computes
        a per-pixel correction by dividing the raw 2D flat image by the expected model:

            pixel_qe(row, col) = flat_2d(row, col) / [bspline_pred(col) * profimg(row, col)]

        This isolates the per-pixel quantum efficiency relative to the fibre's mean,
        capturing intra-fibre spatial QE variations that are lost in the 1D approach.

        Parameters
        ----------
        flat_dict_calibrated : dict
            Dictionary with wavelength-calibrated flat extractions (from arcTransfer).
        fit_results : dict
            Dictionary from calculate_fits_all_extensions with B-spline fits per fibre.
        trace_dir : str
            Directory containing trace files with fiberimg and profimg.
        flat_files : dict
            Dictionary mapping color channel to FITS file path, e.g.
            {'red': '/path/to/red.fits', 'green': '/path/to/green.fits', 'blue': '/path/to/blue.fits'}
        bias_file : str, optional
            Path to bias file for subtraction. If None, uses default.
        clip_low : float, optional
            Lower clip bound for pixel QE values. Default 0.5.
        clip_high : float, optional
            Upper clip bound for pixel QE values. Default 1.5.
        min_profile_weight : float, optional
            Minimum profimg value to trust. Pixels below this get correction=1.0. Default 0.01.

        Returns
        -------
        str
            Path to the output pixel_qe_flat_field.fits MEF file.
        """
        from llamas_pyjamas.Flat.flatProcessing import load_bias_subtracted_flat_2d

        logger.info("Generating TRUE 2D pixel QE flat field maps")
        logger.info(f"  clip_low={clip_low}, clip_high={clip_high}, min_profile_weight={min_profile_weight}")

        output_file = os.path.join(self.output_dir, 'pixel_qe_flat_field.fits')

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['OBJECT'] = 'Pixel QE Flat Field'
        primary_hdu.header['METHOD'] = '2D Pixel QE (bspline * profimg division)'
        primary_hdu.header['COMMENT'] = 'True per-pixel QE correction: flat_2d / (bspline_pred * profimg)'
        primary_hdu.header['CLIPLO'] = (clip_low, 'Lower clip bound for pixel QE')
        primary_hdu.header['CLIPHI'] = (clip_high, 'Upper clip bound for pixel QE')
        primary_hdu.header['MINPROF'] = (min_profile_weight, 'Min profile weight threshold')
        primary_hdu.header['NEXTEN'] = (24, 'Number of extensions')

        hdu_list = [primary_hdu]

        extensions = list(idx_lookup.keys())

        for ext_idx, (channel, bench, side) in enumerate(sorted(extensions, key=lambda x: idx_lookup[x])):
            ext_key = f"{channel}{bench}{side}"
            logger.info(f"Processing extension {ext_idx+1}/24: {ext_key}")

            try:
                # Find matching extraction
                extraction_idx = None
                for i, meta in enumerate(flat_dict_calibrated['metadata']):
                    if (meta['channel'] == channel and
                        str(meta['bench']) == str(bench) and
                        meta['side'] == side):
                        extraction_idx = i
                        break

                if extraction_idx is None:
                    logger.error(f"No extraction found for {ext_key}")
                    hdu_list.append(self._make_empty_qe_hdu(ext_key, channel, bench, side, ext_idx))
                    continue

                # Load trace file
                trace_file = os.path.join(trace_dir, f'LLAMAS_{channel}_{bench}_{side}_traces.pkl')
                if not os.path.exists(trace_file):
                    logger.error(f"Trace file not found: {trace_file}")
                    hdu_list.append(self._make_empty_qe_hdu(ext_key, channel, bench, side, ext_idx))
                    continue

                with open(trace_file, 'rb') as f:
                    trace_obj = pickle.load(f)

                fiberimg = trace_obj.fiberimg
                profimg = trace_obj.profimg

                # Load bias-subtracted 2D flat
                flat_fits_file = flat_files.get(channel.lower())
                if flat_fits_file is None:
                    logger.error(f"No flat file provided for channel {channel}")
                    hdu_list.append(self._make_empty_qe_hdu(ext_key, channel, bench, side, ext_idx))
                    continue

                flat_2d = load_bias_subtracted_flat_2d(
                    flat_fits_file, channel, bench, side, bias_file=bias_file
                )

                # Get B-spline fit results
                ext_results = fit_results.get(ext_key, {})
                if not ext_results:
                    logger.warning(f"No fit results for {ext_key}")
                    hdu_list.append(self._make_empty_qe_hdu(ext_key, channel, bench, side, ext_idx))
                    continue

                # Normalize profimg so it sums to 1 per fibre per column.
                # The extraction uses a weighted mean: sum(frame*profimg)/sum(profimg),
                # so bspline_pred ≈ average pixel value. To reconstruct per-pixel expected
                # values we need profimg_norm where sum(profimg_norm) = 1 across each fibre,
                # so that bspline_pred * profimg_norm gives the fraction of flux at each pixel.
                profimg_norm = np.zeros_like(profimg, dtype=np.float64)

                # Compute per-fibre per-column profile sums (vectorized per fibre)
                for fiber_idx in ext_results.keys():
                    fiber_rows, fiber_cols = np.where(fiberimg == fiber_idx)
                    if len(fiber_rows) == 0:
                        continue

                    # For each unique column, normalize the profile to sum to 1
                    unique_cols, col_inverse = np.unique(fiber_cols, return_inverse=True)

                    # Sum profimg per column for this fibre
                    prof_vals = profimg[fiber_rows, fiber_cols]
                    col_sums = np.zeros(len(unique_cols), dtype=np.float64)
                    np.add.at(col_sums, col_inverse, prof_vals)

                    # Normalize: divide each pixel's profimg by its column sum
                    safe_sums = np.where(col_sums > 0, col_sums, 1.0)
                    profimg_norm[fiber_rows, fiber_cols] = prof_vals / safe_sums[col_inverse]

                # Build expected model image: bspline_pred(col) * profimg_norm(row, col)
                model_2d = np.zeros_like(flat_2d, dtype=np.float64)

                for fiber_idx in ext_results.keys():
                    bspline_model = ext_results[fiber_idx]['bspline_model']
                    xshift = ext_results[fiber_idx]['xshift']

                    # Evaluate bspline at all spectral columns
                    valid_xshift = np.isfinite(xshift)
                    bspline_vals = np.zeros(len(xshift), dtype=np.float64)
                    if np.any(valid_xshift):
                        bspline_vals[valid_xshift] = bspline_model.value(xshift[valid_xshift])[0]

                    # Get all pixels belonging to this fibre
                    fiber_rows, fiber_cols = np.where(fiberimg == fiber_idx)

                    if len(fiber_rows) == 0:
                        continue

                    # Vectorized: model = bspline_at_column * normalized_profile_weight
                    model_2d[fiber_rows, fiber_cols] = (
                        bspline_vals[fiber_cols] * profimg_norm[fiber_rows, fiber_cols]
                    )

                # Compute pixel QE: flat_2d / model_2d
                pixel_qe = np.ones_like(flat_2d, dtype=np.float32)

                # Only compute where model is positive and both arrays are finite
                valid = (
                    (model_2d > 0) &
                    np.isfinite(flat_2d) &
                    np.isfinite(model_2d) &
                    (profimg_norm >= min_profile_weight)
                )
                pixel_qe[valid] = (flat_2d[valid] / model_2d[valid]).astype(np.float32)

                # Clip extreme values
                pixel_qe = np.clip(pixel_qe, clip_low, clip_high)

                # Reset non-fibre and low-profile pixels to 1.0
                non_fibre = (fiberimg == -1) | (profimg_norm < min_profile_weight)
                pixel_qe[non_fibre] = 1.0

                # Handle any remaining NaN/Inf
                bad_mask = ~np.isfinite(pixel_qe)
                pixel_qe[bad_mask] = 1.0

                # Statistics on traced pixels
                traced_mask = valid & ~non_fibre
                n_bad = int(np.sum(bad_mask & ~non_fibre))

                logger.info(f"  {ext_key}: traced pixels={int(np.sum(traced_mask))}, "
                           f"bad pixels={n_bad}")

                # Create HDU
                hdu = fits.ImageHDU(data=pixel_qe)
                hdu.header['EXTNAME'] = ext_key
                hdu.header['CHANNEL'] = channel.upper()
                hdu.header['BENCH'] = str(bench)
                hdu.header['SIDE'] = side
                hdu.header['EXTVER'] = ext_idx + 1
                hdu.header['METHOD'] = '2D Pixel QE'

                if np.any(traced_mask):
                    traced_values = pixel_qe[traced_mask]
                    hdu.header['DATAMEAN'] = float(np.mean(traced_values))
                    hdu.header['DATAMED'] = float(np.median(traced_values))
                    hdu.header['DATAMIN'] = float(np.min(traced_values))
                    hdu.header['DATAMAX'] = float(np.max(traced_values))
                    hdu.header['DATASTD'] = float(np.std(traced_values))
                    hdu.header['NPIX'] = int(np.sum(traced_mask))
                    hdu.header['NBADFPIX'] = n_bad

                hdu_list.append(hdu)

            except Exception as e:
                logger.error(f"Error processing {ext_key}: {str(e)}")
                import traceback
                traceback.print_exc()
                hdu_list.append(self._make_empty_qe_hdu(ext_key, channel, bench, side, ext_idx,
                                                         error=str(e)))

        # Write MEF file
        hdul = fits.HDUList(hdu_list)
        hdul.writeto(output_file, overwrite=True)
        logger.info(f"Pixel QE flat field saved: {output_file}")

        return output_file

    @staticmethod
    def _make_empty_qe_hdu(ext_key, channel, bench, side, ext_idx, error=None):
        """Create an empty (all 1.0) QE HDU for extensions that couldn't be processed."""
        empty_data = np.ones((4112, 4096), dtype=np.float32)
        hdu = fits.ImageHDU(data=empty_data)
        hdu.header['EXTNAME'] = ext_key
        hdu.header['CHANNEL'] = channel.upper()
        hdu.header['BENCH'] = str(bench)
        hdu.header['SIDE'] = side
        hdu.header['EXTVER'] = ext_idx + 1
        if error:
            hdu.header['ERROR'] = str(error)[:68]
        return hdu

    def generate_all_pixel_maps(self):

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


    def generate_pypeit_flat_field(self, flat_dict_calibrated, trace_dir, flat_files,
                                  bias_file=None, verbose=False):
        """Generate flat field products using PypeIt-style methodology.

        This uses the fiber-aggregated approach: per-fiber bspline spectral
        fitting in log-space, plus a per-detector pixel sensitivity map built
        by stacking all fibers by fractional cross-dispersion position.

        Parameters
        ----------
        flat_dict_calibrated : dict
            Wavelength-calibrated flat extractions with metadata.
        trace_dir : str
            Directory containing trace pickle files.
        flat_files : dict
            Maps channel to raw FITS path, e.g.
            {'red': '/path/red.fits', 'green': '...', 'blue': '...'}.
        bias_file : str, optional
            Path to bias file.
        verbose : bool
            Enable verbose logging.

        Returns
        -------
        dict
            Output file paths and processing results.
        """
        from llamas_pyjamas.Flat.flatPypeit import PypeItFlatField

        pypeit_ff = PypeItFlatField(output_dir=self.output_dir, verbose=verbose)
        return pypeit_ff.generate_all_maps(
            flat_dict_calibrated, trace_dir, flat_files, bias_file=bias_file
        )

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
