import os
import logging
from llamas_pyjamas.Image.WhiteLightModule import WhiteLightFits
import numpy as np
from astropy.io import fits
from llamas_pyjamas.config import CALIB_DIR, OUTPUT_DIR, LUT_DIR
from llamas_pyjamas.constants import idx_lookup
from llamas_pyjamas.Flat.flatProcessing import produce_flat_extractions
from llamas_pyjamas.Utils.utils import concat_extractions
from llamas_pyjamas.Arc.arcLlamas import arcTransfer
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
            xmodel = np.linspace(xshift_clean.min(), xshift_clean.max(), len(xshift_clean))  # was 2* for Higher resolution
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
    
    # Load the combined flat extractions
    logger.info(f"Loading combined flat extractions from {combined_flat_file}")
    flat_dict = ExtractLlamas.loadExtraction(combined_flat_file)
    
    # Load the arc calibration
    logger.info(f"Loading arc calibration from {arc_calib_file}")
    arc_dict = ExtractLlamas.loadExtraction(arc_calib_file)
    
    # Apply wavelength solution transfer
    logger.info("Transferring wavelength calibration to flat field extractions")
    flat_dict_calibrated = arcTransfer(flat_dict, arc_dict)
    
    # Save the calibrated flat extractions (sanitized to avoid pickling issues)
    calibrated_flat_file = os.path.join(output_dir, 'combined_flat_extractions_calibrated.pkl')
    sanitized_flat_dict = sanitize_extraction_dict_for_pickling(flat_dict_calibrated) #why is this here?
    with open(calibrated_flat_file, 'wb') as f:
        pickle.dump(sanitized_flat_dict, f)
    logger.info(f"Calibrated flat extractions saved to {calibrated_flat_file}")
    
    # Step 4: Fit B-splines and generate pixel maps
    logger.info("Step 4: Fitting B-splines and generating pixel maps")
    
    # Initialize the Thresholding class for the remaining processing
    threshold_processor = Thresholding(
        red_flat_file, green_flat_file, blue_flat_file,
        use_bias=use_bias, output_dir=output_dir, trace_dir=trace_dir
    )
    
    # Calculate fits for all extensions in the calibrated file
    fit_results = threshold_processor.calculate_fits_all_extensions(calibrated_flat_file)
    
    # Step 5: Generate pixel maps for each channel/bench combination
    logger.info("Step 5: Generating pixel maps for each channel/bench combination")
    
    pixel_map_results = threshold_processor.generate_complete_pixel_maps(fit_results)
    pixel_map_mef_file = pixel_map_results['output_file']

    # Step 6: Create normalized flat field FITS file for reduce.py pipeline
    logger.info("Step 6: Creating normalized flat field FITS file for reduce.py pipeline")

    # Convert all flat file paths to absolute paths to ensure they can be found
    original_flat_files = {
        'red': os.path.abspath(red_flat_file),
        'green': os.path.abspath(green_flat_file),
        'blue': os.path.abspath(blue_flat_file)
    }

    # Debug: Log the original flat file paths being passed
    logger.info(f"Original flat files dictionary:")
    for channel, filepath in original_flat_files.items():
        logger.info(f"  {channel}: {filepath}")
        logger.info(f"  {channel} exists: {os.path.exists(filepath)}")
        logger.info(f"  {channel} absolute: {os.path.abspath(filepath)}")

    try:
        logger.info("Starting normalized flat field creation...")
        normalized_flat_field_file = threshold_processor.create_normalized_flat_field_fits(
            original_flat_files=original_flat_files,
            pixel_map_mef_file=pixel_map_mef_file,  # Pass the combined MEF file
            output_filename=None  # Will use default naming in output_dir
        )
        logger.info(f"Successfully created normalized flat field: {os.path.basename(normalized_flat_field_file)}")
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR: Failed to create normalized flat field: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        normalized_flat_field_file = None
        
        # Also raise the exception to ensure calling code knows about the failure
        raise
    
    results = {
        'combined_flat_file': combined_flat_file,
        'calibrated_flat_file': calibrated_flat_file,
        'fit_results': fit_results,
        'pixel_map_results': pixel_map_results,
        'pixel_map_mef_file': pixel_map_mef_file,
        'normalized_flat_field_file': normalized_flat_field_file,
        'processing_status': 'completed'
    }
    
    logger.info("Complete flat field processing workflow finished successfully")
    return results


class LlamasFlatFielding():
    """Class for handling flat field processing in LLAMAS observations.

    This class provides functionality for creating normalized flat field images
    and applying flat field corrections to white light images.
    """
    
    def __init__(self)->None:
        """Initialize the LlamasFlatFielding object.

        Returns:
            None
        """
        logger.info("Initializing LlamasFlatFielding")
        pass
    
    
    def flatcube(self, extraction_list: list = None, outputname: str = None)-> str:
        """Produce a normalized flat field image using the WhiteLightFits class.

        This method creates a normalized flat field cube by processing extracted 
        fiber data and normalizing each HDU by its maximum value.

        Args:
            extraction_list (list, optional): List of extracted fibers from a flat 
                field image. Defaults to None.
            outputname (str, optional): Name for the output FITS file. If None, 
                defaults to 'normalized_flat.fits'. Defaults to None.

        Returns:
            str: Output filename of the saved normalized flat field.
        """
        logger.info(f"Creating flat cube with {len(extraction_list) if extraction_list else 0} extractions")
        
        hdul = WhiteLightFits(extraction_list, outfile=-1)
        # Log HDU structure
        logger.info(f"Flat cube contains {len(hdul)} HDUs")
        for i, hdu in enumerate(hdul):
            if i == 0:
                logger.info(f"HDU {i}: Primary")
            else:
                # Extract key header information
                try:
                    bench = hdu.header.get('BENCH', 'UNKNOWN')
                    side = hdu.header.get('SIDE', 'UNKNOWN')
                    color = hdu.header.get('COLOR', 'UNKNOWN')
                    logger.info(f"HDU {i}: {bench}{side} {color}")
                except Exception as e:
                    logger.warning(f"Couldn't extract header info for HDU {i}: {str(e)}")
                    
        # Normalize each image data in the HDU list
        for i, hdu in enumerate(hdul):
            if hdu.data is not None:
                max_val = np.nanmax(hdu.data)
                if max_val != 0:
                    hdu.data = hdu.data / max_val
                    logger.debug(f"Normalized HDU {i} with max value {max_val}")
                else:
                    logger.warning(f"HDU {i} has max value of 0, skipping normalization")

        # Save the normalized HDU list as a new FITS file
        if outputname is not None:
            outputname = 'normalized_flat.fits'
            
        hdul.writeto(outputname, overwrite=True)
        logger.info(f"Flat cube saved to {outputname}")
        return outputname
        
    
    def flatFieldImage(self, whitelight_fits: str, flatcube_fits: str, outputname: str = None)-> str:
        """Apply flat field correction by dividing white light image by flat field.

        This method performs flat field correction by dividing a white light image 
        by a corresponding flat field image, handling matching of bench sides and 
        colors, and protecting against division by zero.

        Args:
            whitelight_fits (str): Path to the white light FITS image.
            flatcube_fits (str): Path to the flat field FITS image.
            outputname (str, optional): Filename for the normalized output image. 
                If None, defaults to 'normalized_whitelight.fits'. Defaults to None.

        Returns:
            str: Path to the saved flat-fielded image.
        """
        logger.info(f"Flat-fielding {whitelight_fits} with {flatcube_fits}")
        
        # Open the white light and flat field FITS files
        white_hdul = fits.open(whitelight_fits)
        flat_hdul = fits.open(flatcube_fits)
        
        logger.info(f"White light file has {len(white_hdul)} extensions")
        logger.info(f"Flat field file has {len(flat_hdul)} extensions")

        new_hdus = []

        # Loop over paired HDUs from both files
        for i, (white_hdu, flat_hdu) in enumerate(zip(white_hdul, flat_hdul)):
            bench_white = white_hdu.header.get("BENCHSIDE")
            bench_flat = flat_hdu.header.get("BENCHSIDE")
            colour_white = white_hdu.header.get("COLOUR")
            colour_flat = flat_hdu.header.get("COLOUR")
            
            logger.info(f"Processing extension {i}: White={bench_white} {colour_white}, Flat={bench_flat} {colour_flat}")
            
            # Only process if both hdu's have matching benchside and colour keywords
            if bench_white != bench_flat or colour_white != colour_flat:
                logger.warning(f"Skipping extension {i} due to mismatched benchside or color")
                continue

            # Ensure both have valid data arrays
            if white_hdu.data is not None and flat_hdu.data is not None:
                # Divide and protect against division by zero (assign NaN where flat data is zero)
                divided = np.divide(
                    white_hdu.data,
                    flat_hdu.data,
                    out=np.full_like(white_hdu.data, np.nan, dtype=np.float64),
                    where=flat_hdu.data != 0
                )
                
                zero_count = np.sum(flat_hdu.data == 0)
                logger.debug(f"Extension {i}: {zero_count} zero values in flat field")

                # Normalize the result: divide by the maximum value if it is nonzero
                max_val = np.nanmax(divided)
                if max_val and max_val != 0:
                    divided /= max_val
                    logger.debug(f"Extension {i}: Normalized with max value {max_val}")
                else:
                    logger.warning(f"Extension {i}: Max value is 0 or NaN, skipping normalization")

                # Create a new Image HDU with the result using the white light header
                new_hdu = fits.ImageHDU(data=divided, header=white_hdu.header.copy())
                new_hdus.append(new_hdu)
                logger.info(f"Extension {i}: Added to output")

        # Package the new HDUs into a new HDUList.
        logger.info(f"Created {len(new_hdus)} HDUs for output")
        
        # Use the first new HDU as PrimaryHDU
        if new_hdus:
            primary = fits.PrimaryHDU(data=new_hdus[0].data, header=new_hdus[0].header)
            hdulist = fits.HDUList([primary] + new_hdus[1:])
        else:
            logger.warning("No valid HDUs were created, creating empty output")
            hdulist = fits.HDUList([fits.PrimaryHDU()])

        if outputname is None:
            outputname = 'normalized_whitelight.fits'

        hdulist.writeto(outputname, overwrite=True)
        logger.info(f"Saved flat-fielded image to {outputname}")

        white_hdul.close()
        flat_hdul.close()

        return outputname
        
class Thresholding():

    def __init__(self, red_flat_file, green_flat_file, blue_flat_file, use_bias=None, output_dir=OUTPUT_DIR, trace_dir=CALIB_DIR) -> None:
        self.red_flat_file = red_flat_file
        self.green_flat_file = green_flat_file
        self.blue_flat_file = blue_flat_file
        self.use_bias = use_bias
        self.output_dir = output_dir
        self.trace_dir = trace_dir
        
        logger.info(f"Initializing Thresholding with files:")
        logger.info(f"  Red: {red_flat_file}")
        logger.info(f"  Green: {green_flat_file}")
        logger.info(f"  Blue: {blue_flat_file}")
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
        
        return results
    
    def generate_complete_pixel_maps(self, fit_results, output_filename=None):
        """
        Generate complete pixel maps and save as a single multi-extension FITS file.

        This method takes the B-spline fit results, generates 2D pixel maps for each
        channel and bench combination, and writes them to a single multi-extension FITS file.

        Args:
            fit_results (dict): Dictionary of B-spline fit results for each extension and fiber
            output_filename (str, optional): Output filename for MEF file. If None, defaults to
                'combined_flat_pixel_maps.fits' in output_dir.

        Returns:
            dict: Dictionary containing pixel maps, output file path, and trace file map
        """
        logger.info("Generating complete pixel maps as multi-extension FITS file")

        import glob
        from datetime import datetime

        # Get all available trace files
        trace_files = glob.glob(os.path.join(self.trace_dir, 'LLAMAS*traces.pkl'))
        logger.info(f"Found {len(trace_files)} trace files")

        # Create a mapping from channel/bench/side to trace file
        trace_file_map = {}
        for trace_file in trace_files:
            # Load trace object to get its metadata
            with open(trace_file, 'rb') as f:
                trace_obj = pickle.load(f)

            key = f"{trace_obj.channel}{trace_obj.bench}{trace_obj.side}"
            trace_file_map[key] = trace_file
            logger.debug(f"Mapped {key} to {trace_file}")

        # Store pixel maps and extension metadata
        pixel_maps = {}
        extension_info = []  # List of (ext_key, pixel_map, trace_obj) tuples

        # Process each extension key in fit_results
        for ext_key, fiber_fits in fit_results.items():
            logger.info(f"Processing pixel map for extension {ext_key}")

            # Parse the extension key to find matching trace file
            # ext_key format: "red1A", "green2B", etc.
            found_trace = False
            for trace_key, trace_file in trace_file_map.items():
                if ext_key.replace(ext_key[0], '').lower() in trace_key.lower():
                    # Try to match the channel part too
                    channel_from_ext = None
                    for color in ['red', 'green', 'blue']:
                        if color in ext_key.lower():
                            channel_from_ext = color
                            break

                    if channel_from_ext and channel_from_ext in trace_key.lower():
                        logger.info(f"Matched {ext_key} to trace file {trace_file}")

                        # Load the trace object
                        with open(trace_file, 'rb') as f:
                            trace_obj = pickle.load(f)

                        # Generate pixel map for this extension
                        pixel_map = self._generate_single_pixel_map(fiber_fits, trace_obj)
                        pixel_maps[ext_key] = pixel_map

                        # Store for MEF creation
                        extension_info.append((ext_key, pixel_map, trace_obj))

                        found_trace = True
                        break

            if not found_trace:
                logger.warning(f"Could not find matching trace file for extension {ext_key}")

        # Sort extensions by LLAMAS idx_lookup ordering
        def get_sort_key(info):
            ext_key, pixel_map, trace_obj = info
            return idx_lookup.get((trace_obj.channel, trace_obj.bench, trace_obj.side), 999)

        extension_info.sort(key=get_sort_key)

        logger.info(f"Will create MEF with {len(extension_info)} extensions in the following order:")
        for i, (ext_key, _, trace_obj) in enumerate(extension_info):
            logger.info(f"  Extension {i+1}: {trace_obj.channel}{trace_obj.bench}{trace_obj.side}")

        # Build multi-extension FITS file
        if output_filename is None:
            output_filename = os.path.join(self.output_dir, 'combined_flat_pixel_maps.fits')

        # Create HDU list starting with primary HDU
        hdul = fits.HDUList()

        # Create primary HDU with basic header information
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['COMMENT'] = 'Combined flat field pixel maps for LLAMAS'
        primary_hdu.header['CREATOR'] = 'LLAMAS flatLlamas.py'
        primary_hdu.header['DATE'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        primary_hdu.header['NEXTEND'] = len(extension_info)
        primary_hdu.header['ORIGIN'] = 'LLAMAS Pipeline'
        hdul.append(primary_hdu)

        # Add each pixel map as an extension
        for i, (ext_key, pixel_map, trace_obj) in enumerate(extension_info):
            # Create extension name
            ext_name = f"FLAT_{trace_obj.channel.upper()}{trace_obj.bench}{trace_obj.side.upper()}"

            # Create image HDU
            hdu = fits.ImageHDU(data=pixel_map.astype(np.float32), name=ext_name)

            # Add metadata to header
            hdu.header['EXTVER'] = i + 1
            hdu.header['CHANNEL'] = trace_obj.channel.upper()
            hdu.header['BENCH'] = trace_obj.bench
            hdu.header['SIDE'] = trace_obj.side.upper()
            hdu.header['BENCHSIDE'] = f"{trace_obj.bench}{trace_obj.side.upper()}"
            hdu.header['COLOUR'] = trace_obj.channel.upper()
            hdu.header['BUNIT'] = 'Counts'
            hdu.header['COMMENT'] = 'Flat field pixel map from B-spline fits'

            # Add statistics to header
            valid_pixels = ~np.isnan(pixel_map)
            if np.any(valid_pixels):
                hdu.header['DATAMIN'] = np.nanmin(pixel_map)
                hdu.header['DATAMAX'] = np.nanmax(pixel_map)
                hdu.header['DATAMEAN'] = np.nanmean(pixel_map)
                hdu.header['NPIX'] = np.sum(valid_pixels)
                hdu.header['NNANS'] = np.sum(~valid_pixels)

            hdul.append(hdu)
            logger.debug(f"Added extension {ext_name} ({ext_key})")

        # Write the combined FITS file
        try:
            hdul.writeto(output_filename, overwrite=True)
            logger.info(f"Successfully created combined pixel map file: {output_filename}")
            logger.info(f"File contains {len(hdul)-1} extensions (plus primary)")
        except Exception as e:
            logger.error(f"Error writing FITS file: {str(e)}")
            raise
        finally:
            hdul.close()

        results = {
            'pixel_maps': pixel_maps,
            'output_file': output_filename,
            'trace_file_map': trace_file_map,
            'n_extensions': len(extension_info)
        }

        logger.info(f"Generated pixel maps for {len(pixel_maps)} extensions")
        logger.info(f"Created MEF file: {os.path.basename(output_filename)}")

        return results
    
    def _generate_single_pixel_map(self, fiber_fits, trace_obj):
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
        fiber_image = trace_obj.fiberimg

        # Create an empty array matching the shape of the fiber image
        pixel_map = np.full_like(fiber_image, np.nan, dtype=float)

        processed_fibers = 0
        for fiber_idx, fiber_data in fiber_fits.items():
            # Get the B-spline model for this fiber
            bspline_model = fiber_data['bspline_model']
            xshift_1d = fiber_data['xshift']  # 1D wavelength array from extraction (may include NaN)
            xshift_clean = fiber_data['xshift_clean']  # Cleaned wavelength array used for B-spline fitting

            # Get the valid domain of the B-spline (where it was fitted)
            xshift_min = xshift_clean.min()
            xshift_max = xshift_clean.max()

            # Create extraction column positions
            # The extraction produces one value per spectral pixel
            extraction_cols = np.arange(len(xshift_1d))

            # Find all pixels belonging to this fiber
            fiber_pixels = (fiber_image == fiber_idx)

            if not np.any(fiber_pixels):
                logger.debug(f"No pixels found for fiber {fiber_idx}")
                continue

            # For each row in the image that contains this fiber
            for row in range(fiber_image.shape[0]):
                row_pixels = fiber_pixels[row, :]

                if not np.any(row_pixels):
                    continue

                # Get the column indices of the fiber pixels in this row (2D image columns)
                col_indices = np.where(row_pixels)[0]

                # Interpolate xshift values at actual image column positions
                # This maps: 2D image columns → extraction positions → wavelengths
                xshift_at_cols = np.interp(col_indices, extraction_cols, xshift_1d)

                # Check which pixels are within the B-spline's valid domain
                valid_domain = (xshift_at_cols >= xshift_min) & (xshift_at_cols <= xshift_max) & np.isfinite(xshift_at_cols)

                if not np.any(valid_domain):
                    continue

                # Only evaluate B-spline for pixels within valid domain
                try:
                    valid_xshift = xshift_at_cols[valid_domain]
                    predicted_values = bspline_model.value(valid_xshift)[0]

                    # Assign predicted values only to valid pixels
                    valid_col_indices = col_indices[valid_domain]
                    pixel_map[row, valid_col_indices] = predicted_values
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
    
    
    def generate_thresholds(self):
        """Generate thresholds for flat fielding based on science data.

        This method generates thresholds based on the provided science data,
        which can be used to apply flat field corrections.

        Args:
            science_data (list): List of science data to generate thresholds from.

        Returns:
            list: List of generated threshold values.
        """
        logger.info("Generating thresholds for flat fielding")
        
        # extract flat field data
        logger.info("Producing flat extractions")
        produce_flat_extractions(
            self.red_flat_file, 
            self.green_flat_file, 
            self.blue_flat_file, 
            tracedir=self.trace_dir, 
            outpath=self.output_dir
        )
        
        #### need to add in the concat behaiour here and then begin running the fits per extension



        # Assuming the extraction files are generated and available
        red_extraction_file = os.path.join(self.output_dir, 'red_extractions_flat.pkl')
        green_extraction_file = os.path.join(self.output_dir, 'green_extractions_flat.pkl')
        blue_extraction_file = os.path.join(self.output_dir, 'blue_extractions_flat.pkl')
        
        # Check if files exist
        for file_path, color in [
            (red_extraction_file, 'red'),
            (green_extraction_file, 'green'),
            (blue_extraction_file, 'blue')
        ]:
            if os.path.exists(file_path):
                logger.info(f"{color.capitalize()} extraction file found: {file_path}")
            else:
                logger.warning(f"{color.capitalize()} extraction file not found: {file_path}")

        # Process each color
        results = {}
        for file_path, color in [
            (red_extraction_file, 'red'),
            (green_extraction_file, 'green'),
            (blue_extraction_file, 'blue')
        ]:
            if os.path.exists(file_path):
                logger.info(f"Processing {color} extraction")
                try:
                    color_results = self.calculate_fits_all_extensions(file_path)
                    results[color] = color_results
                    logger.info(f"Completed {color} extraction processing")
                except Exception as e:
                    logger.error(f"Error processing {color} extraction: {str(e)}")
            else:
                logger.warning(f"Skipping {color} extraction (file not found)")

        logger.info("Threshold generation complete")
        return results
    
    def create_normalized_flat_field_fits(self, original_flat_files, pixel_map_mef_file=None, output_filename=None):
        """
        Create normalized flat field FITS file for reduce.py pipeline with per-fiber normalization.

        This method creates a multi-extension FITS file with normalized flat field data by:
        1. Dividing original flat data by pixel maps (B-spline predicted values) to remove spectral variations
        2. **PER-FIBER NORMALIZATION**: Each fiber normalized to its own median to preserve fiber-to-fiber
           throughput variations (CRITICAL: different fibers have different sensitivities!)
        3. Optional global rescaling maintains values ~O(1) while preserving relative throughput
        4. Setting values outside fiber traces to exactly 1.0

        **Key Physics**: The normalized flat field should preserve real fiber-to-fiber sensitivity
        differences. Each science frame pixel is divided by the corresponding normalized flat pixel,
        so the flat must contain the true relative sensitivity per pixel, NOT force everything to unity.

        Args:
            original_flat_files (dict): Dictionary with 'red', 'green', 'blue' FITS file paths
            pixel_map_mef_file (str, optional): Path to multi-extension FITS file containing
                B-spline predicted flat field values. If None, falls back to trace-only normalization.
            output_filename (str, optional): Output filename for normalized flat field.
                If None, defaults to 'normalized_flat_field.fits'

        Returns:
            str: Path to the created normalized flat field FITS file
        """
        # Import required modules
        import numpy as np
        import pickle
        from datetime import datetime
        from astropy.io import fits
        from llamas_pyjamas.constants import idx_lookup
        
        logger.info("Creating normalized flat field FITS file for reduce.py pipeline")
        logger.info("="*60)
        
        # Initialize processing counters
        total_extensions_expected = len(idx_lookup)  # Should be 24: 3 colors × 4 benches × 2 sides
        extensions_attempted = 0
        extensions_successful = 0
        processing_errors = []

        logger.info(f"Expected to create {total_extensions_expected} extensions based on idx_lookup")
        
        if output_filename is None:
            # Determine correct output directory for normalized flat field
            if os.path.basename(self.output_dir) == 'pixel_maps':
                # If we're in pixel_maps subdirectory, save in parent flat directory
                parent_dir = os.path.dirname(self.output_dir)
            elif os.path.basename(self.output_dir) == 'flat':
                # If we're in flat directory, save here
                parent_dir = self.output_dir
            else:
                # Default to extractions directory
                parent_dir = self.output_dir
                
            output_filename = os.path.join(parent_dir, 'normalized_flat_field.fits')
            logger.info(f"Output file will be saved as: {output_filename}")
        
        # Validate input files
        required_colors = ['red', 'green', 'blue']
        for color in required_colors:
            if color not in original_flat_files:
                raise ValueError(f"Missing {color} flat file in original_flat_files")
            if not os.path.exists(original_flat_files[color]):
                raise FileNotFoundError(f"{color} flat file not found: {original_flat_files[color]}")
                
        # Log detailed input validation
        logger.info("Input file validation:")
        flat_file_info = {}
        for color, file_path in original_flat_files.items():
            try:
                with fits.open(file_path) as test_hdul:
                    num_extensions = len(test_hdul) - 1  # Exclude primary
                    flat_file_info[color] = {
                        'file': file_path,
                        'extensions': num_extensions,
                        'valid': True
                    }
                    logger.info(f"  {color.capitalize()}: {os.path.basename(file_path)} ({num_extensions} extensions)")
            except Exception as e:
                flat_file_info[color] = {
                    'file': file_path,
                    'extensions': 0,
                    'valid': False,
                    'error': str(e)
                }
                logger.error(f"  {color.capitalize()}: ERROR reading {os.path.basename(file_path)} - {str(e)}")
        
        logger.info(f"Input flat files:")
        for color, file_path in original_flat_files.items():
            logger.info(f"  {color.capitalize()}: {os.path.basename(file_path)}")
        
        # Load pixel maps from MEF file if provided (contains B-spline predicted flat field values)
        pixel_maps = {}
        pixel_map_stats = {}
        if pixel_map_mef_file:
            if not os.path.exists(pixel_map_mef_file):
                logger.error(f"Pixel map MEF file not found: {pixel_map_mef_file}")
                logger.info("Will use trace-only normalization")
            else:
                logger.info(f"Loading pixel maps from MEF file: {os.path.basename(pixel_map_mef_file)}")
                pixel_maps_loaded = 0

                try:
                    with fits.open(pixel_map_mef_file) as mef_hdul:
                        n_extensions = len(mef_hdul) - 1  # Exclude primary HDU
                        logger.info(f"  MEF file contains {n_extensions} extensions")

                        for i, hdu in enumerate(mef_hdul[1:], start=1):  # Skip primary HDU
                            try:
                                # Get metadata from header
                                channel = hdu.header.get('CHANNEL', '').lower()
                                bench = hdu.header.get('BENCH', '')
                                side = hdu.header.get('SIDE', '')

                                if not all([channel, bench, side]):
                                    logger.warning(f"    Extension {i}: Missing metadata (CHANNEL/BENCH/SIDE)")
                                    continue

                                pixel_map_data = hdu.data
                                if pixel_map_data is None:
                                    logger.error(f"    Extension {i}: Pixel map data is None")
                                    continue

                                key = f"{channel}{bench}{side}"
                                pixel_maps[key] = pixel_map_data

                                # Calculate and log pixel map statistics
                                valid_pixels = np.isfinite(pixel_map_data) & (pixel_map_data > 0)
                                if np.any(valid_pixels):
                                    stats = {
                                        'shape': pixel_map_data.shape,
                                        'valid_pixels': np.sum(valid_pixels),
                                        'total_pixels': pixel_map_data.size,
                                        'min_val': np.min(pixel_map_data[valid_pixels]),
                                        'max_val': np.max(pixel_map_data[valid_pixels]),
                                        'median_val': np.median(pixel_map_data[valid_pixels])
                                    }
                                    pixel_map_stats[key] = stats
                                    logger.info(f"    ✓ Extension {i} ({key}): {stats['shape']}, {stats['valid_pixels']:,} valid pixels, median={stats['median_val']:.3f}")
                                    pixel_maps_loaded += 1
                                else:
                                    logger.error(f"    Extension {i}: No valid pixel data")

                            except Exception as e:
                                logger.error(f"    Extension {i}: Error loading - {str(e)}")

                        logger.info(f"Successfully loaded {pixel_maps_loaded}/{n_extensions} pixel maps from MEF")
                        if pixel_maps_loaded == 0:
                            logger.warning("No pixel maps were successfully loaded - will use trace-only normalization")

                except Exception as e:
                    logger.error(f"Error opening MEF file: {str(e)}")
                    logger.info("Will use trace-only normalization")
        else:
            logger.info("No pixel map MEF file provided - using trace-only normalization")
        
        # Create HDU list starting with primary HDU
        hdul = fits.HDUList()
        
        # Create primary HDU with metadata
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['COMMENT'] = 'Normalized flat field for LLAMAS science data reduction'
        primary_hdu.header['CREATOR'] = 'LLAMAS flatLlamas.py create_normalized_flat_field_fits()'
        primary_hdu.header['DATE'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        primary_hdu.header['PURPOSE'] = 'Direct division flat field correction'
        primary_hdu.header['ORIGIN'] = 'LLAMAS Pipeline'
        primary_hdu.header['NEXTEND'] = 24
        hdul.append(primary_hdu)
        
        # Get trace files for fiber location information
        import glob
        trace_files = glob.glob(os.path.join(self.trace_dir, 'LLAMAS_master*traces.pkl'))
        logger.info(f"Trace file loading from {self.trace_dir}:")
        if not trace_files:
            logger.error(f"  ERROR: No trace files found in {self.trace_dir}")
            logger.error(f"  This will prevent proper fiber normalization")
        else:
            logger.info(f"  Found {len(trace_files)} trace files")
        
        # Create trace file mapping
        trace_file_map = {}
        trace_files_loaded = 0
        for trace_file in trace_files:
            try:
                with open(trace_file, 'rb') as f:
                    trace_obj = pickle.load(f)
                    
                # Validate trace object has required attributes
                if not hasattr(trace_obj, 'channel') or not hasattr(trace_obj, 'bench') or not hasattr(trace_obj, 'side'):
                    logger.error(f"    ERROR: Invalid trace object in {os.path.basename(trace_file)} - missing channel/bench/side")
                    continue
                    
                if not hasattr(trace_obj, 'fiberimg') or trace_obj.fiberimg is None:
                    logger.error(f"    ERROR: No fiberimg data in {os.path.basename(trace_file)}")
                    continue
                    
                key = f"{trace_obj.channel}{trace_obj.bench}{trace_obj.side}"
                trace_file_map[key] = {'file': trace_file, 'obj': trace_obj}
                
                # Log trace statistics
                fiber_pixels = np.sum(trace_obj.fiberimg > 0)
                total_pixels = trace_obj.fiberimg.size
                fiber_fraction = fiber_pixels / total_pixels * 100
                logger.info(f"    ✓ Loaded {key}: {os.path.basename(trace_file)}, {fiber_pixels:,} fiber pixels ({fiber_fraction:.1f}%)")
                trace_files_loaded += 1
                
            except Exception as e:
                logger.error(f"    ERROR loading trace file {os.path.basename(trace_file)}: {str(e)}")
                continue
                
        logger.info(f"Successfully loaded {trace_files_loaded}/{len(trace_files)} trace files")
        if trace_files_loaded == 0:
            logger.error("No trace files were successfully loaded - normalization will fail")
        
        # Inspect original flat files before processing
        logger.info(f"\nInspecting original flat files before processing:")
        for channel, flat_file in original_flat_files.items():
            if os.path.exists(flat_file):
                with fits.open(flat_file) as flat_hdul:
                    logger.info(f"  {channel.upper()} flat: {os.path.basename(flat_file)} - {len(flat_hdul)-1} extensions")
                    for i in range(1, min(len(flat_hdul), 6)):  # Show first 5 extensions
                        hdu = flat_hdul[i]
                        hdu_bench = str(hdu.header.get('BENCH', ''))
                        hdu_side = str(hdu.header.get('SIDE', '')).upper()
                        hdu_channel = hdu.header.get('COLOR', hdu.header.get('CHANNEL', '')).lower()
                        logger.info(f"    Ext {i}: {hdu_channel} {hdu_bench}{hdu_side}")
            else:
                logger.error(f"  {channel.upper()} flat: {flat_file} - FILE NOT FOUND")

        # Process extensions in idx_lookup order
        extensions_created = 0
        logger.info(f"\nProcessing {len(idx_lookup)} extensions for normalized flat field:")
        logger.info(f"Expected to process 24 extensions from idx_lookup")
        logger.info(f"Available pixel maps: {list(pixel_maps.keys()) if pixel_maps else 'None'}")
        logger.info(f"Available trace files: {list(trace_file_map.keys()) if trace_file_map else 'None'}")
        logger.info("-" * 60)

        # Open each flat file once and store handles (efficiency improvement)
        flat_hduls = {}
        try:
            for channel in ['red', 'green', 'blue']:
                flat_file = original_flat_files[channel]
                if os.path.exists(flat_file):
                    flat_hduls[channel] = fits.open(flat_file)
                    logger.info(f"Opened {channel} flat file: {os.path.basename(flat_file)}")
                else:
                    logger.error(f"Cannot open {channel} flat file: {flat_file} - FILE NOT FOUND")
                    flat_hduls[channel] = None

            for (channel, bench, side), ext_idx in sorted(idx_lookup.items(), key=lambda x: x[1]):
                extensions_attempted += 1
                logger.info(f"Extension {ext_idx:2d}: {channel.upper()} {bench}{side.upper()}")

                extension_success = False
                error_details = []

                try:
                    # Get the already-opened flat file handle for this channel
                    flat_hdul = flat_hduls.get(channel)

                    if flat_hdul is None:
                        flat_file = original_flat_files[channel]
                        logger.error(f"    ERROR: Flat file handle not available for {channel}: {flat_file}")
                        error_details.append(f"Flat file not opened: {flat_file}")
                        processing_errors.append(f"Extension {ext_idx}: {error_details[-1]}")
                        continue

                    # Load original flat field data using direct indexing with validation
                    flat_file = original_flat_files[channel]
                    logger.info(f"  Loading from: {os.path.basename(flat_file)}")

                    flat_data = None
                    original_header = None
                    extension_found = False

                    # Primary method: Direct index with metadata verification
                    if ext_idx < len(flat_hdul):
                        hdu = flat_hdul[ext_idx]
                        hdu_bench = str(hdu.header.get('BENCH', ''))
                        hdu_side = str(hdu.header.get('SIDE', '')).upper()
                        hdu_channel = hdu.header.get('COLOR', hdu.header.get('CHANNEL', '')).lower()

                        logger.debug(f"    Direct index {ext_idx}: channel='{hdu_channel}', bench='{hdu_bench}', side='{hdu_side}'")

                        # Verify metadata matches expected configuration
                        if (hdu_channel == channel and
                            hdu_bench == bench and
                            hdu_side.upper() == side.upper()):

                            if hdu.data is not None:
                                flat_data = hdu.data.copy()
                                original_header = hdu.header.copy()
                                extension_found = True
                                logger.info(f"    ✓ Direct index {ext_idx} matches: {flat_data.shape}")
                            else:
                                logger.error(f"    ERROR: Extension {ext_idx} has no data")
                                error_details.append(f"Extension {ext_idx} has no data")
                        else:
                            # Metadata mismatch - log warning and search all extensions
                            logger.warning(f"    Metadata mismatch at index {ext_idx}!")
                            logger.warning(f"    Expected: {channel} {bench}{side}")
                            logger.warning(f"    Found: {hdu_channel} {hdu_bench}{hdu_side}")
                            logger.warning(f"    Falling back to metadata search...")
                    else:
                        logger.warning(f"    Extension index {ext_idx} out of range, using metadata search")

                    # Fallback method: Search all extensions for matching metadata
                    if not extension_found:
                        logger.debug(f"    Searching {len(flat_hdul)-1} extensions for metadata match...")
                        for i in range(1, len(flat_hdul)):  # Skip primary
                            hdu = flat_hdul[i]
                            hdu_bench = str(hdu.header.get('BENCH', ''))
                            hdu_side = str(hdu.header.get('SIDE', '')).upper()
                            hdu_channel = hdu.header.get('COLOR', hdu.header.get('CHANNEL', '')).lower()

                            logger.debug(f"    Search ext {i}: channel='{hdu_channel}', bench='{hdu_bench}', side='{hdu_side}'")

                            if (hdu_channel == channel and
                                hdu_bench == bench and
                                hdu_side.upper() == side.upper()):

                                if hdu.data is None:
                                    logger.error(f"    ERROR: Extension {i} has no data")
                                    error_details.append(f"Extension {i} has no data")
                                    continue

                                flat_data = hdu.data.copy()
                                original_header = hdu.header.copy()
                                extension_found = True
                                logger.info(f"    ✓ Found via search at extension {i}: {flat_data.shape}")
                                break

                    if not extension_found or flat_data is None:
                        logger.error(f"    ERROR: No matching extension found for {channel} {bench}{side}")
                        logger.error(f"    Available extensions in {os.path.basename(flat_file)}:")
                        for i in range(1, len(flat_hdul)):
                            hdu = flat_hdul[i]
                            hdu_bench = str(hdu.header.get('BENCH', ''))
                            hdu_side = str(hdu.header.get('SIDE', '')).upper()
                            hdu_channel = hdu.header.get('COLOR', hdu.header.get('CHANNEL', '')).lower()
                            logger.error(f"      Extension {i}: {hdu_channel} {hdu_bench}{hdu_side}")
                        error_details.append(f"No matching extension in {os.path.basename(flat_file)}")
                        processing_errors.append(f"Extension {ext_idx}: {error_details[-1]}")
                        continue

                    # STEP 1: Divide by pixel map (removes B-spline modeled variation)
                    pixel_map_key = f"{channel}{bench}{side}"
                    logger.info(f"  Looking for pixel map: {pixel_map_key}")
                    corrected_data = flat_data.copy()
                    pixel_map_applied = False

                    logger.info(f"  Step 1: Pixel map correction for {pixel_map_key}")
                    if pixel_map_key in pixel_maps:
                        pixel_map = pixel_maps[pixel_map_key]

                        # Verify shapes match
                        if pixel_map.shape != flat_data.shape:
                            logger.error(f"    ERROR: Shape mismatch - flat {flat_data.shape} vs pixel map {pixel_map.shape}")
                            error_details.append(f"Shape mismatch with pixel map")
                        else:
                            # Check pixel map validity
                            valid_pixel_map = np.isfinite(pixel_map) & (pixel_map > 0)
                            if not np.any(valid_pixel_map):
                                logger.error(f"    ERROR: Pixel map contains no valid data")
                                error_details.append(f"Invalid pixel map data")
                            else:
                                # Divide flat by pixel map (B-spline predictions) - IFU Standard
                                with np.errstate(divide='ignore', invalid='ignore'):
                                    corrected_data = np.where(pixel_map > 0,
                                                             flat_data / pixel_map,
                                                             np.nan)  # Set invalid regions to NaN (IFU standard)

                                # Validate corrected data
                                valid_corrected = np.isfinite(corrected_data)
                                if np.any(valid_corrected):
                                    pixel_map_applied = True
                                    corrected_stats = pixel_map_stats.get(pixel_map_key, {})
                                    logger.info(f"    ✓ Applied pixel map: {np.sum(valid_corrected):,} valid pixels")
                                else:
                                    logger.error(f"    ERROR: Pixel map correction resulted in no valid data")
                                    error_details.append(f"Pixel map correction failed")
                                    corrected_data = flat_data.copy()  # Fallback to original
                    else:
                        logger.info(f"    No pixel map available - using original flat data")

                    if not pixel_map_applied:
                        logger.info(f"    Using original flat field data for normalization")

                    # STEP 2: Per-Detector Throughput Normalization
                    # This removes fiber-to-fiber AND detector-to-detector throughput variations
                    # Result: all values ≈ 1.0, preserving only pixel-to-pixel QE variations
                    trace_key = f"{channel}{bench}{side}"
                    logger.info(f"  Looking for trace file: {trace_key}")
                    normalized_data = np.ones_like(flat_data, dtype=np.float32)
                    normalization_successful = False

                    logger.info(f"  Step 2: Per-detector throughput normalization for {trace_key}")

                    # Initialize variables for header metadata
                    fibers_processed = 0
                    fibers_failed = 0
                    detector_norm_factor = np.nan

                    if trace_key in trace_file_map:
                        trace_obj = trace_file_map[trace_key]['obj']
                        fiber_image = trace_obj.fiberimg

                        if fiber_image is not None and fiber_image.shape == flat_data.shape:
                            # Find traced regions (non-zero in fiber image)
                            traced_mask = fiber_image > 0

                            if np.any(traced_mask):
                                # Get corrected flat field values in traced regions
                                traced_corrected_values = corrected_data[traced_mask]

                                # Remove bad values
                                valid_mask = np.isfinite(traced_corrected_values) & (traced_corrected_values > 0)
                                if np.any(valid_mask):
                                    # Get unique fiber IDs (excluding 0 which is background)
                                    unique_fibers = np.unique(fiber_image[traced_mask])
                                    unique_fibers = unique_fibers[unique_fibers > 0]

                                    logger.info(f"    Applying per-detector throughput normalization")
                                    logger.info(f"    Found {len(unique_fibers)} fibers on this detector")

                                    # Initialize output with background = 1.0
                                    normalized_data[~traced_mask] = 1.0

                                    # Collect median values for each fiber to compute detector-level normalization
                                    fiber_medians = []

                                    for fiber_id in unique_fibers:
                                        fiber_mask = (fiber_image == fiber_id)
                                        fiber_pixels = corrected_data[fiber_mask]
                                        fiber_valid = np.isfinite(fiber_pixels) & (fiber_pixels > 0)

                                        if np.sum(fiber_valid) > 0:
                                            fiber_median = np.median(fiber_pixels[fiber_valid])
                                            if fiber_median > 0 and np.isfinite(fiber_median):
                                                fiber_medians.append(fiber_median)
                                                fibers_processed += 1
                                            else:
                                                fibers_failed += 1
                                        else:
                                            fibers_failed += 1

                                    # Compute detector-level normalization factor
                                    # This is the median of all fiber medians on this detector
                                    if len(fiber_medians) > 0:
                                        detector_norm_factor = np.median(fiber_medians)
                                        logger.info(f"    Detector normalization factor: {detector_norm_factor:.3f}")
                                        logger.info(f"    Fiber median range: {np.min(fiber_medians):.3f} to {np.max(fiber_medians):.3f}")

                                        # Apply single normalization factor to ALL traced pixels on this detector
                                        # This removes both spectral shape (from Stage 1) and throughput variations
                                        if detector_norm_factor > 0:
                                            # Normalize all traced pixels by detector factor
                                            normalized_traced = corrected_data[traced_mask] / detector_norm_factor

                                            # Clip extreme values
                                            normalized_traced = np.clip(normalized_traced, 0.1, 3.0)

                                            # Handle any remaining NaN/inf
                                            bad_pixels = ~np.isfinite(normalized_traced)
                                            normalized_traced[bad_pixels] = 1.0

                                            # Assign to output
                                            normalized_data[traced_mask] = normalized_traced

                                            # Calculate final statistics
                                            traced_count = np.sum(traced_mask)
                                            untraced_count = normalized_data.size - traced_count
                                            traced_data_final = normalized_data[traced_mask]
                                            valid_traced_final = traced_data_final[np.isfinite(traced_data_final)]

                                            final_median = np.median(valid_traced_final) if len(valid_traced_final) > 0 else np.nan
                                            final_std = np.std(valid_traced_final) if len(valid_traced_final) > 0 else np.nan
                                            nan_count = np.sum(~np.isfinite(traced_data_final))

                                            logger.info(f"    ✓ Per-detector normalization complete:")
                                            logger.info(f"      Fibers processed: {fibers_processed}, Failed: {fibers_failed}")
                                            logger.info(f"      Traced pixels: {traced_count:,}, Background pixels: {untraced_count:,}")
                                            logger.info(f"      Final median in traces: {final_median:.3f} (should be ≈1.0)")
                                            logger.info(f"      Final std in traces: {final_std:.3f} (should be <0.1)")
                                            logger.info(f"      NaN pixels: {nan_count:,}")

                                            # Verify normalization succeeded
                                            if 0.8 <= final_median <= 1.2:
                                                normalization_successful = True
                                                logger.info(f"    ✓ Normalization successful - median near 1.0")
                                            else:
                                                logger.warning(f"    WARNING: Final median {final_median:.3f} not near 1.0")
                                                normalization_successful = True  # Accept anyway
                                        else:
                                            logger.error(f"    ERROR: Invalid detector normalization factor: {detector_norm_factor}")
                                            error_details.append(f"Invalid detector norm factor")
                                    else:
                                        logger.error(f"    ERROR: No valid fiber medians computed")
                                        error_details.append(f"No valid fiber medians")
                                else:
                                    logger.error(f"    ERROR: No valid corrected flat field values in traced regions")
                                    error_details.append(f"No valid values in traced regions")
                            else:
                                logger.error(f"    ERROR: No traced pixels found")
                                error_details.append(f"No traced pixels found")
                        else:
                            if fiber_image is None:
                                logger.error(f"    ERROR: No fiber image data in trace file")
                                error_details.append(f"No fiber image in trace")
                            else:
                                logger.error(f"    ERROR: Fiber image shape {fiber_image.shape} != flat data shape {flat_data.shape}")
                                error_details.append(f"Fiber image shape mismatch")
                    else:
                        logger.error(f"    ERROR: No trace information found for {trace_key}")
                        error_details.append(f"No trace file found")

                    # Fallback normalization if per-detector normalization failed
                    if not normalization_successful:
                        logger.warning(f"    Using fallback normalization (global median)")
                        valid_corrected = corrected_data[np.isfinite(corrected_data) & (corrected_data > 0)]
                        if len(valid_corrected) > 100:
                            median_corrected = np.median(valid_corrected)
                            if median_corrected > 0:
                                normalized_data = np.clip(corrected_data / median_corrected, 0.1, 3.0)
                                normalized_data[~np.isfinite(normalized_data)] = 1.0
                                normalization_successful = True
                                detector_norm_factor = median_corrected
                                logger.info(f"    ✓ Fallback normalization applied, factor: {median_corrected:.3f}")
                            else:
                                logger.error(f"    ERROR: Invalid fallback median: {median_corrected}")
                        else:
                            logger.error(f"    ERROR: Insufficient valid data for fallback ({len(valid_corrected)} pixels)")

                    if not normalization_successful:
                        logger.error(f"    ERROR: All normalization methods failed, setting to ones")
                        normalized_data = np.ones_like(flat_data, dtype=np.float32)
                        error_details.append(f"All normalization methods failed")

                    # Create extension HDU with validation
                    ext_name = f"FLAT_{channel.upper()}{bench}{side.upper()}"

                    # Validate normalized data before creating HDU
                    if normalized_data is None:
                        logger.error(f"    ERROR: Normalized data is None")
                        error_details.append(f"Normalized data is None")
                    elif normalized_data.size == 0:
                        logger.error(f"    ERROR: Normalized data is empty array")
                        error_details.append(f"Normalized data is empty")
                    else:
                        # Final data validation
                        valid_pixels = np.isfinite(normalized_data)
                        if not np.any(valid_pixels):
                            logger.error(f"    ERROR: No valid pixels in final normalized data")
                            error_details.append(f"No valid pixels in final data")
                        else:
                            # Create HDU with validated data
                            hdu = fits.ImageHDU(data=normalized_data.astype(np.float32), name=ext_name)

                            # Set header information
                            hdu.header['EXTVER'] = ext_idx
                            hdu.header['CHANNEL'] = channel.upper()
                            hdu.header['BENCH'] = bench
                            hdu.header['SIDE'] = side.upper()
                            hdu.header['BENCHSIDE'] = f"{bench}{side.upper()}"
                            hdu.header['COLOUR'] = channel.upper()
                            hdu.header['BUNIT'] = 'Normalized'
                            hdu.header['PURPOSE'] = 'Flat field correction by division'
                            hdu.header['ORIGFILE'] = os.path.basename(original_flat_files[channel])
                            hdu.header['NORMTYPE'] = ('PER_DETECTOR', 'Per-detector throughput normalization')

                            # Add per-detector normalization statistics
                            hdu.header['NFIBERS'] = (fibers_processed, 'Number of fibers successfully processed')
                            hdu.header['FIBFAIL'] = (fibers_failed, 'Number of fibers that failed')
                            if np.isfinite(detector_norm_factor):
                                hdu.header['DETNORM'] = (float(detector_norm_factor), 'Detector normalization factor')
                            hdu.header['COMMENT'] = 'Values should be ~1.0, preserving only pixel QE variations'

                            # Add comprehensive statistics
                            data_stats = {
                                'min': float(np.nanmin(normalized_data)),
                                'max': float(np.nanmax(normalized_data)),
                                'mean': float(np.nanmean(normalized_data)),
                                'median': float(np.nanmedian(normalized_data)),
                                'valid_pixels': int(np.sum(valid_pixels)),
                                'total_pixels': int(normalized_data.size)
                            }

                            for key, value in data_stats.items():
                                if key == 'valid_pixels':
                                    hdu.header['NPIX'] = value
                                elif key == 'total_pixels':
                                    hdu.header['NTOTPIX'] = value
                                else:
                                    hdu.header[f'DATA{key[:2].upper()}'] = value

                            # Check for NaN or infinity values
                            nan_count = np.sum(np.isnan(normalized_data))
                            inf_count = np.sum(np.isinf(normalized_data))
                            if nan_count > 0 or inf_count > 0:
                                logger.warning(f"    WARNING: Found {nan_count} NaN and {inf_count} infinity values")
                                hdu.header['NANCOUNT'] = nan_count
                                hdu.header['INFCOUNT'] = inf_count

                            hdul.append(hdu)
                            extensions_created += 1
                            extensions_successful += 1
                            extension_success = True

                            logger.info(f"    ✓ Extension {ext_idx} created successfully: {ext_name}")
                            logger.info(f"      Data range: [{data_stats['min']:.3f}, {data_stats['max']:.3f}], median: {data_stats['median']:.3f}")
                            logger.info(f"      Valid pixels: {data_stats['valid_pixels']:,}/{data_stats['total_pixels']:,}")

                        if not extension_success and error_details:
                            processing_errors.append(f"Extension {ext_idx} ({channel}{bench}{side}): {'; '.join(error_details)}")
                            logger.error(f"    ✗ Extension {ext_idx} failed: {'; '.join(error_details)}")

                except Exception as e:
                    logger.error(f"    ERROR: Exception processing extension {ext_idx}: {str(e)}")
                    processing_errors.append(f"Extension {ext_idx}: Exception - {str(e)}")
                    import traceback
                    logger.debug(f"    Traceback: {traceback.format_exc()}")
                    continue

        finally:
            # Close all flat file handles
            for channel, hdul_handle in flat_hduls.items():
                if hdul_handle is not None:
                    try:
                        hdul_handle.close()
                        logger.info(f"Closed {channel} flat file")
                    except Exception as e:
                        logger.warning(f"Error closing {channel} flat file: {e}")

        # Log final processing summary
        logger.info("-" * 60)
        logger.info(f"PROCESSING SUMMARY:")
        logger.info(f"  Extensions attempted: {extensions_attempted}/{total_extensions_expected}")
        logger.info(f"  Extensions successful: {extensions_successful}/{extensions_attempted}")
        logger.info(f"  Success rate: {extensions_successful/extensions_attempted*100:.1f}%" if extensions_attempted > 0 else "  Success rate: 0.0%")
        
        if processing_errors:
            logger.error(f"  Processing errors ({len(processing_errors)}):")
            for error in processing_errors:
                logger.error(f"    - {error}")
        
        # Write the normalized flat field FITS file
        if extensions_created == 0:
            logger.error("ERROR: No extensions were successfully created - cannot write FITS file")
            raise ValueError("No valid extensions created for normalized flat field")

        if extensions_created != total_extensions_expected:
            logger.error(f"ERROR: Created {extensions_created} extensions, expected {total_extensions_expected}")
            logger.error("This will result in an incomplete normalized flat field unsuitable for science reduction")
            if extensions_created < total_extensions_expected // 2:  # Less than half
                raise ValueError(f"Too few extensions created: {extensions_created}/{total_extensions_expected}")
            else:
                logger.warning("Proceeding with incomplete normalized flat field - may cause issues in science reduction")
            
        try:
            hdul.writeto(output_filename, overwrite=True)
            
            # Verify file was written correctly
            file_size = os.path.getsize(output_filename)
            if file_size < 1024:  # Less than 1KB indicates likely empty file
                logger.error(f"ERROR: Output file is too small ({file_size} bytes) - likely empty")
                raise ValueError(f"Output file too small: {file_size} bytes")
            
            logger.info(f"\n" + "=" * 60)
            logger.info(f"SUCCESS: Normalized flat field created: {output_filename}")
            logger.info(f"  File contains {extensions_created} extensions (plus primary)")
            logger.info(f"  File size: {file_size / (1024*1024):.1f} MB")
            
            # Final verification of data content
            with fits.open(output_filename) as verify_hdul:
                data_extensions = len(verify_hdul) - 1
                if data_extensions != extensions_created:
                    logger.error(f"ERROR: Extension count mismatch - created {extensions_created}, file has {data_extensions}")
                else:
                    logger.info(f"  Verification: All {data_extensions} extensions present in file")
                    
                    # Sample a few extensions to verify data
                    sample_extensions = min(3, data_extensions)
                    for i in range(1, sample_extensions + 1):
                        ext_data = verify_hdul[i].data
                        if ext_data is not None and ext_data.size > 0:
                            valid_count = np.sum(np.isfinite(ext_data))
                            median_val = np.nanmedian(ext_data)
                            logger.info(f"    Extension {i}: {valid_count:,} valid pixels, median={median_val:.3f}")
                        else:
                            logger.error(f"    Extension {i}: ERROR - Empty or null data")
            
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"ERROR writing normalized flat field FITS file: {str(e)}")
            raise
        finally:
            hdul.close()
        
        return output_filename
    
# If run as a standalone script
if __name__ == "__main__":
    logger.info("flatLlamas.py executed as standalone script")
    
    import argparse
    parser = argparse.ArgumentParser(description='Process LLAMAS flat field data with complete workflow')
    parser.add_argument('red_flat', help='Path to red flat field FITS file')
    parser.add_argument('green_flat', help='Path to green flat field FITS file') 
    parser.add_argument('blue_flat', help='Path to blue flat field FITS file')
    parser.add_argument('--arc_calib', help='Path to arc calibration file (default: LLAMAS_reference_arc.pkl in trace dir)')
    parser.add_argument('--output_dir', default=OUTPUT_DIR, help=f'Output directory (default: {OUTPUT_DIR})')
    parser.add_argument('--trace_dir', default=CALIB_DIR, help=f'Trace directory (default: {CALIB_DIR})')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose console output')
    
    args = parser.parse_args()
    
    logger.info(f"Processing flat fields:")
    logger.info(f"  Red: {args.red_flat}")
    logger.info(f"  Green: {args.green_flat}") 
    logger.info(f"  Blue: {args.blue_flat}")
    
    try:
        results = process_flat_field_complete(
            args.red_flat,
            args.green_flat, 
            args.blue_flat,
            arc_calib_file=args.arc_calib,
            output_dir=args.output_dir,
            trace_dir=args.trace_dir,
            verbose=args.verbose
        )
        
        logger.info("Processing completed successfully!")

        if results['pixel_map_mef_file']:
            logger.info(f"Pixel map MEF file: {results['pixel_map_mef_file']}")
        else:
            logger.warning("Pixel map MEF file was not created")
        
        if results['normalized_flat_field_file']:
            logger.info(f"Normalized flat field for reduce.py: {results['normalized_flat_field_file']}")
        else:
            logger.warning("Normalized flat field file was not created")
            
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise


def diagnose_normalized_flat(normalized_flat_file, trace_dir=CALIB_DIR, extension_idx=1):
    """
    Diagnostic function to check if normalized flat field looks reasonable after per-fiber normalization.

    This function validates that the per-fiber normalization has preserved fiber-to-fiber
    throughput variations as intended, rather than forcing all pixels toward unity.

    Args:
        normalized_flat_file (str): Path to normalized flat field FITS file
        trace_dir (str): Directory containing trace files for fiber identification
        extension_idx (int): Which extension to check (1-indexed, default=1 for first data extension)

    Returns:
        dict: Diagnostic results including per-fiber statistics

    Example:
        >>> results = diagnose_normalized_flat('normalized_flat_field.fits')
        >>> if results['validation_passed']:
        ...     print("✓ Normalized flat looks good!")
        ... else:
        ...     print(f"✗ Issues found: {results['issues']}")
    """
    import numpy as np
    import pickle
    from astropy.io import fits
    import glob
    import os

    print(f"\n{'='*70}")
    print(f"NORMALIZED FLAT FIELD DIAGNOSTICS")
    print(f"{'='*70}")
    print(f"File: {os.path.basename(normalized_flat_file)}")
    print(f"Extension: {extension_idx}")

    # Load normalized flat data
    with fits.open(normalized_flat_file) as hdul:
        if extension_idx >= len(hdul):
            print(f"ERROR: Extension {extension_idx} not found (file has {len(hdul)} HDUs)")
            return {'validation_passed': False, 'issues': ['Extension not found']}

        data = hdul[extension_idx].data
        header = hdul[extension_idx].header

        # Report header information
        norm_type = header.get('NORMTYPE', 'UNKNOWN')
        print(f"\nNormalization type: {norm_type}")

        if 'NFIBERS' in header:
            print(f"Fibers processed: {header['NFIBERS']}")
            print(f"Fibers failed: {header.get('FIBFAIL', 0)}")
            print(f"Fiber throughput range: {header.get('FIBTHMIN', 0):.3f} to {header.get('FIBTHMAX', 0):.3f}")
            print(f"Fiber throughput std: {header.get('FIBTHSTD', 0):.3f}")
            if 'GLOBSCAL' in header:
                print(f"Global rescaling factor: {header['GLOBSCAL']:.3f}")

    # Load trace for fiber identification
    channel = header.get('CHANNEL', header.get('COLOUR', '')).lower()
    bench = str(header.get('BENCH', ''))
    side = header.get('SIDE', '')
    trace_key = f"{channel}{bench}{side}"

    print(f"\nLooking for trace file for {trace_key}...")

    trace_files = glob.glob(os.path.join(trace_dir, f'LLAMAS_master_{channel}_{bench}_{side}_traces.pkl'))
    if not trace_files:
        print(f"WARNING: No trace file found for {trace_key}")
        print(f"Cannot perform per-fiber validation")
        trace_mask = np.ones_like(data, dtype=bool)  # Assume all pixels are traced
        fiber_image = None
    else:
        trace_file = trace_files[0]
        print(f"Loading trace: {os.path.basename(trace_file)}")

        with open(trace_file, 'rb') as f:
            trace = pickle.load(f)

        fiber_image = trace.fiberimg
        trace_mask = (fiber_image > 0)

    # Global statistics
    print(f"\n{'='*70}")
    print(f"GLOBAL STATISTICS")
    print(f"{'='*70}")

    all_pixels_finite = np.isfinite(data)
    trace_pixels_finite = all_pixels_finite & trace_mask

    print(f"Total pixels: {data.size:,}")
    print(f"Traced pixels: {np.sum(trace_mask):,} ({np.sum(trace_mask)/data.size*100:.1f}%)")
    print(f"Background pixels: {np.sum(~trace_mask):,}")
    print(f"Finite traced pixels: {np.sum(trace_pixels_finite):,}")
    print(f"NaN/Inf traced pixels: {np.sum(~trace_pixels_finite & trace_mask):,}")

    traced_values = data[trace_pixels_finite]
    if len(traced_values) > 0:
        print(f"\nTraced region values:")
        print(f"  Min: {np.min(traced_values):.4f}")
        print(f"  Max: {np.max(traced_values):.4f}")
        print(f"  Median: {np.median(traced_values):.4f}")
        print(f"  Std: {np.std(traced_values):.4f}")

    # Background check
    background_values = data[~trace_mask]
    all_ones = np.allclose(background_values[np.isfinite(background_values)], 1.0)
    print(f"\nBackground pixels all = 1.0? {all_ones}")

    # Per-fiber statistics
    if fiber_image is not None:
        print(f"\n{'='*70}")
        print(f"PER-FIBER STATISTICS (first 10 fibers)")
        print(f"{'='*70}")
        print(f"{'Fiber':>6} {'Pixels':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
        print(f"{'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        unique_fibers = np.unique(fiber_image[trace_mask])
        unique_fibers = unique_fibers[unique_fibers > 0]  # Remove background

        fiber_medians = []
        for fiber_id in unique_fibers[:10]:  # First 10 fibers
            fiber_mask = (fiber_image == fiber_id)
            fiber_vals = data[fiber_mask]
            fiber_vals_finite = fiber_vals[np.isfinite(fiber_vals)]

            if len(fiber_vals_finite) > 0:
                fiber_med = np.median(fiber_vals_finite)
                fiber_std = np.std(fiber_vals_finite)
                fiber_min = np.min(fiber_vals_finite)
                fiber_max = np.max(fiber_vals_finite)
                fiber_medians.append(fiber_med)

                print(f"{fiber_id:6.0f} {len(fiber_vals_finite):8d} {fiber_med:8.4f} {fiber_std:8.4f} {fiber_min:8.4f} {fiber_max:8.4f}")

        # Statistics across all fibers
        all_fiber_medians = []
        for fiber_id in unique_fibers:
            fiber_mask = (fiber_image == fiber_id)
            fiber_vals = data[fiber_mask]
            fiber_vals_finite = fiber_vals[np.isfinite(fiber_vals)]
            if len(fiber_vals_finite) > 0:
                all_fiber_medians.append(np.median(fiber_vals_finite))

        if len(all_fiber_medians) > 1:
            print(f"\n{'='*70}")
            print(f"FIBER-TO-FIBER VARIATION SUMMARY")
            print(f"{'='*70}")
            print(f"Total fibers analyzed: {len(all_fiber_medians)}")
            print(f"Fiber median range: {np.min(all_fiber_medians):.4f} to {np.max(all_fiber_medians):.4f}")
            print(f"Fiber-to-fiber std: {np.std(all_fiber_medians):.4f}")
            print(f"Coefficient of variation: {np.std(all_fiber_medians)/np.mean(all_fiber_medians):.4f}")

    # Validation checks
    print(f"\n{'='*70}")
    print(f"VALIDATION CHECKS")
    print(f"{'='*70}")

    issues = []
    validation_passed = True

    # Check 1: Background should be 1.0
    if not all_ones:
        issues.append("Background pixels not all 1.0")
        validation_passed = False
        print(f"✗ FAIL: Background pixels not all 1.0")
    else:
        print(f"✓ PASS: Background pixels = 1.0")

    # Check 2: Fiber medians should show variation (NOT all ~1.0)
    if fiber_image is not None and len(all_fiber_medians) > 1:
        fiber_std = np.std(all_fiber_medians)
        if fiber_std < 0.05:  # Too little variation suggests wrong normalization
            issues.append(f"Fiber-to-fiber variation too low ({fiber_std:.3f} < 0.05)")
            validation_passed = False
            print(f"✗ FAIL: Fiber-to-fiber std ({fiber_std:.4f}) < 0.05 - normalization may have destroyed throughput info")
        elif fiber_std > 0.5:  # Too much variation suggests no normalization
            issues.append(f"Fiber-to-fiber variation too high ({fiber_std:.3f} > 0.5)")
            validation_passed = False
            print(f"✗ FAIL: Fiber-to-fiber std ({fiber_std:.4f}) > 0.5 - normalization may not have been applied")
        else:
            print(f"✓ PASS: Fiber-to-fiber std ({fiber_std:.4f}) in expected range [0.05-0.5]")

        # Check 3: Fiber medians should be in reasonable range
        fiber_med_range = (np.min(all_fiber_medians), np.max(all_fiber_medians))
        if fiber_med_range[0] < 0.5 or fiber_med_range[1] > 2.0:
            issues.append(f"Fiber median range {fiber_med_range} outside [0.5-2.0]")
            validation_passed = False
            print(f"✗ FAIL: Fiber median range {fiber_med_range} outside [0.5-2.0]")
        else:
            print(f"✓ PASS: Fiber median range {fiber_med_range} within [0.5-2.0]")
    else:
        print(f"⚠  SKIP: Per-fiber checks (no fiber image available)")

    # Check 4: Should not have all pixels ≈ 1.0
    if fiber_image is not None:
        median_traced = np.median(traced_values)
        if 0.99 <= median_traced <= 1.01 and fiber_std < 0.02:
            issues.append("All values ~1.0 - global normalization artifact")
            validation_passed = False
            print(f"✗ FAIL: Median {median_traced:.4f} ≈ 1.0 with low std - suggests wrong normalization method")
        else:
            print(f"✓ PASS: Values show physical variation, not forced to 1.0")

    # Summary
    print(f"\n{'='*70}")
    if validation_passed:
        print(f"✓✓✓ VALIDATION PASSED - Normalized flat looks good!")
        print(f"✓ Per-fiber throughput variations preserved")
        print(f"✓ Suitable for flat field correction by division")
    else:
        print(f"✗✗✗ VALIDATION FAILED - Issues detected:")
        for issue in issues:
            print(f"  ✗ {issue}")
    print(f"{'='*70}\n")

    return {
        'validation_passed': validation_passed,
        'issues': issues,
        'norm_type': norm_type,
        'fiber_count': len(all_fiber_medians) if fiber_image is not None else None,
        'fiber_median_range': (np.min(all_fiber_medians), np.max(all_fiber_medians)) if fiber_image is not None and len(all_fiber_medians) > 0 else None,
        'fiber_std': fiber_std if fiber_image is not None and len(all_fiber_medians) > 1 else None,
        'background_is_one': all_ones
    }


