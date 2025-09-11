import os
import logging
from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas
from llamas_pyjamas.Image.WhiteLightModule import WhiteLightFits
import numpy as np
from astropy.io import fits
import llamas_pyjamas.GUI.guiExtract as ge
from llamas_pyjamas.config import CALIB_DIR, OUTPUT_DIR, LUT_DIR
from llamas_pyjamas.constants import idx_lookup
from llamas_pyjamas.Flat.flatProcessing import produce_flat_extractions
from llamas_pyjamas.Utils.utils import concat_extractions
from llamas_pyjamas.Arc.arcLlamas import arcTransfer
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

from scipy.interpolate import BSpline, make_interp_spline
from pypeit.core.fitting import iterfit
from pypeit import bspline
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


# Alternative simpler version if the above is too complex:
def fit_spectrum_simple(extraction, fiber_index, maxiter=6):
    """Simplified version with aggressive breakpoint spacing for sharp features."""
    
    xshift = extraction.xshift[fiber_index, :]
    counts = extraction.counts[fiber_index, :]
    
    mask = np.isfinite(counts)
    xshift_clean = xshift[mask]
    counts_clean = counts[mask]
    
    # Very aggressive breakpoint spacing for sharp spectral lines
    xrange = xshift_clean.max() - xshift_clean.min()
    bkspace = xrange / max(100, len(xshift_clean) // 15)  # Many breakpoints
    
    logger.info(f"Using aggressive bkspace: {bkspace:.4f}")
    
    sset, outmask = iterfit(
        xshift_clean, counts_clean,
        maxiter=maxiter,
        nord=3,  # Cubic splines
        upper=3.0,
        lower=3.0,
        kwargs_bspline={'bkspace': bkspace}
    )
    
    # High-resolution model
    xmodel = np.linspace(xshift_clean.min(), xshift_clean.max(), len(xshift)*3)
    y_fit = sset.value(xmodel)[0]
    
    return {
        'xshift_clean': xshift_clean,
        'counts_clean': counts_clean,
        'xmodel': xmodel,
        'y_fit': y_fit,
        'bspline_model': sset,
        'outmask': outmask,
        'bkspace_used': bkspace
    }


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
        logger.error(f"Arc calibration file not found: {arc_calib_file}")
        raise FileNotFoundError(f"Missing arc calibration file: {arc_calib_file}")
    
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
    sanitized_flat_dict = sanitize_extraction_dict_for_pickling(flat_dict_calibrated)
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
    output_files = pixel_map_results['output_files']
    
    # Step 6: Create normalized flat field FITS file for reduce.py pipeline
    logger.info("Step 6: Creating normalized flat field FITS file for reduce.py pipeline")
    
    original_flat_files = {
        'red': red_flat_file,
        'green': green_flat_file,
        'blue': blue_flat_file
    }
    
    try:
        normalized_flat_field_file = threshold_processor.create_normalized_flat_field_fits(
            original_flat_files=original_flat_files,
            output_filename=None  # Will use default naming in output_dir
        )
        logger.info(f"Successfully created normalized flat field: {os.path.basename(normalized_flat_field_file)}")
    except Exception as e:
        logger.error(f"Failed to create normalized flat field: {str(e)}")
        normalized_flat_field_file = None
    
    results = {
        'combined_flat_file': combined_flat_file,
        'calibrated_flat_file': calibrated_flat_file,
        'fit_results': fit_results,
        'pixel_map_results': pixel_map_results,
        'output_files': output_files,
        'combined_mef_file': pixel_map_results.get('combined_mef_file'),
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

                    # Calculate statistics for thresholding
                    # median_residual = np.median(residuals)
                    # std_residual = np.std(residuals)
                    
                    # logger.debug(f"Fiber {fiber_idx}: median residual={median_residual:.4f}, std residual={std_residual:.4f}")
                    
                    # Store results for this fiber
                    results[ext_key][fiber_idx] = {
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
    
    def generate_pixel_map(self, fit_results, trace_object):
        """
        Generate a map of predicted flat field values for each pixel based on the B-spline fits.

        Args:
            fit_results (dict): Dictionary of B-spline fit results for each extension and fiber
            trace_object (TraceLlamas): Trace object containing the fiber image map

        Returns:
            dict: Dictionary of predicted flat field image arrays for each channel/benchside
        """
        logger.info(f"Generating pixel map for {trace_object.channel} channel")
        
        # Initialize a dictionary to store the resulting maps for each channel/benchside
        pixel_maps = {}

        # Get the fiber image from the trace object
        fiber_image = trace_object.fiberimg
        channel = trace_object.channel
        bench = trace_object.bench
        side = trace_object.side
        
        logger.info(f"Trace object: {channel} {bench}{side}, fiber image shape: {fiber_image.shape}")
        logger.info(f"Number of fibers in trace: {trace_object.nfibers}")
        
        
        # Loop through each extension in the fit results
        for ext_key, fibers in fit_results.items():
            logger.info(f"Processing fit results for {ext_key} with {len(fibers)} fibers")
            
            # Create an empty array matching the shape of the fiber image
            # Initialize with NaN to easily identify unprocessed pixels
            pixel_map = np.full_like(fiber_image, np.nan, dtype=float)

            # Process each fiber in this extension
            processed_count = 0
            for fiber_idx, fiber_data in fibers.items():
                # Get the B-spline model for this fiber
                bspline_model = fiber_data['bspline_model']

                # Find all pixels belonging to this fiber
                fiber_pixels = (fiber_image == fiber_idx)
                pixel_count = np.sum(fiber_pixels)

                if not np.any(fiber_pixels):
                    logger.warning(f"No pixels found for fiber {fiber_idx} in {ext_key}")
                    continue
                
                logger.debug(f"Processing fiber {fiber_idx} with {pixel_count} pixels")
                
                # For each row in the image that contains this fiber
                for row in range(fiber_image.shape[0]):
                    # Get the pixels in this row that belong to this fiber
                    row_pixels = fiber_pixels[row, :]

                    if not np.any(row_pixels):
                        continue
                    
                    # Get the column indices of the fiber pixels in this row
                    col_indices = np.where(row_pixels)[0]

                    # Get the range of x-shift values for this fiber
                    xshift_min = np.min(fiber_data['xshift_clean'])
                    xshift_max = np.max(fiber_data['xshift_clean'])

                    # Map the column indices to x-shift values
                    xshift_values = np.interp(
                        col_indices,
                        [0, fiber_image.shape[1] - 1],
                        [xshift_min, xshift_max]
                    )

                    # Evaluate the B-spline model at these x-shift values
                    predicted_values = bspline_model.value(xshift_values)[0]

                    # Assign the predicted values to the pixel map
                    pixel_map[row, col_indices] = predicted_values
                
                processed_count += 1
                if processed_count % 50 == 0:
                    logger.info(f"Processed {processed_count}/{len(fibers)} fibers")

            # Check for unassigned pixels
            nan_count = np.sum(np.isnan(pixel_map))
            logger.info(f"Pixel map for {ext_key} has {nan_count} unassigned pixels out of {pixel_map.size}")
            
            # Store the completed pixel map for this extension
            pixel_maps[ext_key] = pixel_map

            # # Save the pixel map as a FITS file for inspection
            # hdu = fits.PrimaryHDU(data=pixel_map)
            # hdu.header['EXTNAME'] = ext_key
            # hdu.header['COMMENT'] = 'Predicted flat field values from B-spline fits'
            # output_file = os.path.join(self.output_dir, f'predicted_flat_{channel}_{bench}{side}.fits')
            # hdu.writeto(output_file, overwrite=True)
            # logger.info(f"Saved predicted flat field map for {ext_key} to {output_file}")

        return pixel_maps
    
    def generate_complete_pixel_maps(self, fit_results, output_fits=True, create_mef=True):
        """
        Generate complete pixel maps with FITS output for each channel/bench combination.
        
        This method takes the B-spline fit results and generates 2D pixel maps for each
        channel and bench combination, saving them as FITS files.
        
        Args:
            fit_results (dict): Dictionary of B-spline fit results for each extension and fiber
            output_fits (bool): Whether to save pixel maps as FITS files. Default True.
            create_mef (bool): Whether to create combined multi-extension FITS file. Default True.
            
        Returns:
            dict: Dictionary containing pixel maps and output file paths
        """
        logger.info("Generating complete pixel maps with FITS output")
        
        import glob
        from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas
        
        # Get all available trace files
        trace_files = glob.glob(os.path.join(self.trace_dir, 'LLAMAS_master*traces.pkl'))
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
        
        pixel_maps = {}
        output_files = []
        
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
                        
                        # Save as FITS file if requested
                        if output_fits:
                            output_filename = os.path.join(
                                self.output_dir, 
                                f'flat_pixel_map_{ext_key}.fits'
                            )
                            
                            # Create FITS HDU
                            hdu = fits.PrimaryHDU(data=pixel_map.astype(np.float32))
                            hdu.header['EXTNAME'] = ext_key
                            hdu.header['CHANNEL'] = trace_obj.channel
                            hdu.header['BENCH'] = trace_obj.bench
                            hdu.header['SIDE'] = trace_obj.side
                            hdu.header['COMMENT'] = 'Flat field pixel map from B-spline fits'
                            hdu.header['BUNIT'] = 'Counts'
                            
                            # Add statistics to header
                            valid_pixels = ~np.isnan(pixel_map)
                            if np.any(valid_pixels):
                                hdu.header['DATAMIN'] = np.nanmin(pixel_map)
                                hdu.header['DATAMAX'] = np.nanmax(pixel_map)
                                hdu.header['DATAMEAN'] = np.nanmean(pixel_map)
                                hdu.header['NPIX'] = np.sum(valid_pixels)
                                hdu.header['NNANS'] = np.sum(~valid_pixels)
                            
                            hdu.writeto(output_filename, overwrite=True)
                            output_files.append(output_filename)
                            logger.info(f"Saved pixel map to {output_filename}")
                        
                        found_trace = True
                        break
            
            if not found_trace:
                logger.warning(f"Could not find matching trace file for extension {ext_key}")
        
        # Create combined multi-extension FITS file if requested
        combined_mef_file = None
        if create_mef and output_files:
            try:
                logger.info("Creating combined multi-extension FITS file")
                combined_mef_file = self.combine_pixel_maps_to_mef(output_files)
                logger.info(f"Successfully created combined MEF file: {combined_mef_file}")
            except Exception as e:
                logger.error(f"Failed to create combined MEF file: {str(e)}")
                # Don't raise the exception - individual files are still valid
        
        results = {
            'pixel_maps': pixel_maps,
            'output_files': output_files,
            'combined_mef_file': combined_mef_file,
            'trace_file_map': trace_file_map
        }
        
        logger.info(f"Generated pixel maps for {len(pixel_maps)} extensions")
        logger.info(f"Saved {len(output_files)} individual FITS files")
        if combined_mef_file:
            logger.info(f"Created combined MEF file: {os.path.basename(combined_mef_file)}")
        
        return results
    
    def _generate_single_pixel_map(self, fiber_fits, trace_obj):
        """
        Generate a single pixel map for one extension using B-spline fits and trace object.
        
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
            
            # Find all pixels belonging to this fiber
            fiber_pixels = (fiber_image == fiber_idx)
            
            if not np.any(fiber_pixels):
                logger.debug(f"No pixels found for fiber {fiber_idx}")
                continue
            
            # Get the xshift range for this fiber
            xshift_min = np.min(fiber_data['xshift_clean'])
            xshift_max = np.max(fiber_data['xshift_clean'])
            
            # For each row in the image that contains this fiber
            for row in range(fiber_image.shape[0]):
                row_pixels = fiber_pixels[row, :]
                
                if not np.any(row_pixels):
                    continue
                
                # Get the column indices of the fiber pixels in this row
                col_indices = np.where(row_pixels)[0]
                
                # Map the column indices to x-shift values
                # This assumes a linear mapping from pixel column to xshift
                xshift_values = np.interp(
                    col_indices,
                    [0, fiber_image.shape[1] - 1],
                    [xshift_min, xshift_max]
                )
                
                # Evaluate the B-spline model at these x-shift values
                try:
                    predicted_values = bspline_model.value(xshift_values)[0]
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
    
    def combine_pixel_maps_to_mef(self, pixel_map_files, output_filename=None):
        """
        Combine individual pixel map FITS files into a single multi-extension FITS file.
        
        This method takes individual pixel map files and combines them into a single
        multi-extension FITS file with extensions ordered by color, bench, and side
        to match the raw science frame structure.
        
        Args:
            pixel_map_files (list): List of individual pixel map FITS file paths
            output_filename (str, optional): Output filename for combined MEF file.
                If None, defaults to 'combined_pixel_maps.fits'
                
        Returns:
            str: Path to the created multi-extension FITS file
        """
        logger.info("Combining individual pixel maps into multi-extension FITS file")
        
        if not pixel_map_files:
            logger.error("No pixel map files provided")
            raise ValueError("No pixel map files to combine")
            
        # Default output filename
        if output_filename is None:
            output_filename = os.path.join(self.output_dir, 'combined_flat_pixel_maps.fits')
        
        # Parse filenames to extract channel/bench/side information
        file_info = []
        for file_path in pixel_map_files:
            if not os.path.exists(file_path):
                logger.warning(f"Pixel map file not found: {file_path}")
                continue
                
            filename = os.path.basename(file_path)
            
            # Parse filename like: flat_pixel_map_red1A.fits
            if filename.startswith('flat_pixel_map_') and filename.endswith('.fits'):
                channel_bench_side = filename.replace('flat_pixel_map_', '').replace('.fits', '')
                
                # Extract channel
                channel = None
                for color in ['blue', 'green', 'red']:
                    if channel_bench_side.lower().startswith(color):
                        channel = color
                        bench_side = channel_bench_side[len(color):]
                        break
                
                if channel and len(bench_side) >= 2:
                    bench = bench_side[0]  # First character is bench number
                    side = bench_side[1]   # Second character is side (A or B)
                    
                    file_info.append({
                        'file_path': file_path,
                        'channel': channel,
                        'bench': bench,
                        'side': side,
                        'sort_key': f"{channel}_{bench}{side}"
                    })
                    logger.debug(f"Parsed {filename}: {channel} {bench}{side}")
                else:
                    logger.warning(f"Could not parse filename: {filename}")
            else:
                logger.warning(f"Unexpected filename format: {filename}")
        
        if not file_info:
            logger.error("No valid pixel map files found")
            raise ValueError("No valid pixel map files to combine")
        
        # Sort by LLAMAS idx_lookup ordering from constants.py
        file_info.sort(key=lambda x: idx_lookup.get((x['channel'], x['bench'], x['side']), 999))
        
        logger.info(f"Will combine {len(file_info)} pixel maps in the following order:")
        for i, info in enumerate(file_info):
            logger.info(f"  Extension {i+1}: {info['channel']}{info['bench']}{info['side']}")
        
        # Create HDU list starting with primary HDU
        hdul = fits.HDUList()
        
        # Create primary HDU with basic header information
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['COMMENT'] = 'Combined flat field pixel maps for LLAMAS'
        primary_hdu.header['CREATOR'] = 'LLAMAS flatLlamas.py'
        primary_hdu.header['DATE'] = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        primary_hdu.header['NEXTEND'] = len(file_info)
        primary_hdu.header['ORIGIN'] = 'LLAMAS Pipeline'
        hdul.append(primary_hdu)
        
        # Add each pixel map as an extension
        for i, info in enumerate(file_info):
            try:
                # Read the individual pixel map file
                logger.debug(f"Reading {info['file_path']}")
                with fits.open(info['file_path']) as pixel_hdul:
                    pixel_data = pixel_hdul[0].data
                    pixel_header = pixel_hdul[0].header.copy()
                
                # Create extension name
                ext_name = f"FLAT_{info['channel'].upper()}{info['bench']}{info['side'].upper()}"
                
                # Create image HDU
                hdu = fits.ImageHDU(data=pixel_data.astype(np.float32), 
                                  header=pixel_header, 
                                  name=ext_name)
                
                # Add additional metadata to header
                hdu.header['EXTVER'] = i + 1
                hdu.header['CHANNEL'] = info['channel'].upper()
                hdu.header['BENCH'] = info['bench']
                hdu.header['SIDE'] = info['side'].upper()
                hdu.header['BENCHSIDE'] = f"{info['bench']}{info['side'].upper()}"
                hdu.header['COLOUR'] = info['channel'].upper()
                hdu.header['ORIGFILE'] = os.path.basename(info['file_path'])
                
                # Ensure BUNIT is set consistently
                if 'BUNIT' not in hdu.header:
                    hdu.header['BUNIT'] = 'Counts'
                
                hdul.append(hdu)
                logger.debug(f"Added extension {ext_name} from {info['file_path']}")
                
            except Exception as e:
                logger.error(f"Error reading pixel map file {info['file_path']}: {str(e)}")
                continue
        
        # Write the combined FITS file
        try:
            hdul.writeto(output_filename, overwrite=True)
            logger.info(f"Successfully created combined pixel map file: {output_filename}")
            logger.info(f"File contains {len(hdul)-1} extensions (plus primary)")
            
            # Log file size for reference
            file_size = os.path.getsize(output_filename)
            logger.info(f"Combined file size: {file_size / (1024*1024):.1f} MB")
            
        except Exception as e:
            logger.error(f"Error writing combined FITS file: {str(e)}")
            raise
        finally:
            hdul.close()
        
        return output_filename
    
    def create_normalized_flat_field_fits(self, original_flat_files, output_filename=None):
        """
        Create normalized flat field FITS file for reduce.py pipeline.
        
        This method creates a multi-extension FITS file with normalized flat field data
        where values within fiber traces are multipliers around 1.0 and values outside
        fiber traces are exactly 1.0, suitable for direct division with science frames.
        
        Args:
            original_flat_files (dict): Dictionary with 'red', 'green', 'blue' FITS file paths
            output_filename (str, optional): Output filename for normalized flat field.
                If None, defaults to 'normalized_flat_field.fits'
                
        Returns:
            str: Path to the created normalized flat field FITS file
        """
        logger.info("Creating normalized flat field FITS file for reduce.py pipeline")
        
        if output_filename is None:
            output_filename = os.path.join(self.output_dir, 'normalized_flat_field.fits')
        
        # Validate input files
        required_colors = ['red', 'green', 'blue']
        for color in required_colors:
            if color not in original_flat_files:
                raise ValueError(f"Missing {color} flat file in original_flat_files")
            if not os.path.exists(original_flat_files[color]):
                raise FileNotFoundError(f"{color} flat file not found: {original_flat_files[color]}")
        
        logger.info(f"Input flat files:")
        for color, file_path in original_flat_files.items():
            logger.info(f"  {color.capitalize()}: {os.path.basename(file_path)}")
        
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
        if not trace_files:
            logger.warning(f"No trace files found in {self.trace_dir}")
        
        # Create trace file mapping
        trace_file_map = {}
        for trace_file in trace_files:
            try:
                with open(trace_file, 'rb') as f:
                    trace_obj = pickle.load(f)
                key = f"{trace_obj.channel}{trace_obj.bench}{trace_obj.side}"
                trace_file_map[key] = {'file': trace_file, 'obj': trace_obj}
                logger.debug(f"Mapped trace {key} to {os.path.basename(trace_file)}")
            except Exception as e:
                logger.warning(f"Error loading trace file {trace_file}: {str(e)}")
                continue
        
        # Process extensions in idx_lookup order
        extensions_created = 0
        for (channel, bench, side), ext_idx in sorted(idx_lookup.items(), key=lambda x: x[1]):
            logger.debug(f"Processing extension {ext_idx}: {channel} {bench}{side}")
            
            try:
                # Load original flat field data for this channel
                flat_file = original_flat_files[channel]
                with fits.open(flat_file) as flat_hdul:
                    # Find matching extension in original flat file
                    flat_data = None
                    for i in range(1, len(flat_hdul)):  # Skip primary
                        hdu = flat_hdul[i]
                        hdu_bench = hdu.header.get('BENCH', '')
                        hdu_side = hdu.header.get('SIDE', '')
                        hdu_channel = hdu.header.get('COLOR', hdu.header.get('CHANNEL', '')).lower()
                        
                        if (hdu_channel == channel and 
                            hdu_bench == bench and 
                            hdu_side == side):
                            flat_data = hdu.data.copy()
                            original_header = hdu.header.copy()
                            break
                    
                    if flat_data is None:
                        logger.warning(f"No matching extension found for {channel} {bench}{side} in {flat_file}")
                        continue
                
                # Get trace information for fiber locations
                trace_key = f"{channel}{bench}{side}"
                if trace_key in trace_file_map:
                    trace_obj = trace_file_map[trace_key]['obj']
                    fiber_image = trace_obj.fiberimg
                    
                    # Create normalized flat field data
                    normalized_data = np.ones_like(flat_data, dtype=np.float32)
                    
                    if fiber_image is not None and fiber_image.shape == flat_data.shape:
                        # Find traced regions (non-zero in fiber image)
                        traced_mask = fiber_image > 0
                        
                        if np.any(traced_mask):
                            # Get flat field values in traced regions
                            traced_flat_values = flat_data[traced_mask]
                            
                            # Remove bad values
                            valid_mask = np.isfinite(traced_flat_values) & (traced_flat_values > 0)
                            if np.any(valid_mask):
                                valid_traced_values = traced_flat_values[valid_mask]
                                
                                # Normalize to median = 1.0 within traced regions
                                median_flat = np.median(valid_traced_values)
                                if median_flat > 0:
                                    normalization_factor = 1.0 / median_flat
                                    normalized_traced_values = flat_data[traced_mask] * normalization_factor
                                    
                                    # Clip extreme values to reasonable range
                                    normalized_traced_values = np.clip(normalized_traced_values, 0.1, 5.0)
                                    
                                    # Set normalized values in traced regions
                                    normalized_data[traced_mask] = normalized_traced_values
                                    
                                    traced_count = np.sum(traced_mask)
                                    untraced_count = normalized_data.size - traced_count
                                    logger.debug(f"  {channel}{bench}{side}: {traced_count:,} traced pixels, "
                                               f"{untraced_count:,} untraced pixels (={1.0})")
                                else:
                                    logger.warning(f"Invalid median flat value for {channel}{bench}{side}")
                            else:
                                logger.warning(f"No valid flat field values for {channel}{bench}{side}")
                        else:
                            logger.warning(f"No traced pixels found for {channel}{bench}{side}")
                    else:
                        logger.warning(f"Fiber image shape mismatch or missing for {channel}{bench}{side}")
                        # Use original flat data normalized to median = 1.0
                        valid_flat = flat_data[np.isfinite(flat_data) & (flat_data > 0)]
                        if len(valid_flat) > 0:
                            median_flat = np.median(valid_flat)
                            if median_flat > 0:
                                normalized_data = np.clip(flat_data / median_flat, 0.1, 5.0)
                                normalized_data[~np.isfinite(normalized_data)] = 1.0
                else:
                    logger.warning(f"No trace information found for {channel}{bench}{side}")
                    # Fallback: use original flat data normalized to median = 1.0
                    valid_flat = flat_data[np.isfinite(flat_data) & (flat_data > 0)]
                    if len(valid_flat) > 0:
                        median_flat = np.median(valid_flat)
                        if median_flat > 0:
                            normalized_data = np.clip(flat_data / median_flat, 0.1, 5.0)
                            normalized_data[~np.isfinite(normalized_data)] = 1.0
                
                # Create extension HDU
                ext_name = f"FLAT_{channel.upper()}{bench}{side.upper()}"
                hdu = fits.ImageHDU(data=normalized_data, name=ext_name)
                
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
                
                # Add statistics
                valid_pixels = np.isfinite(normalized_data)
                if np.any(valid_pixels):
                    hdu.header['DATAMIN'] = float(np.nanmin(normalized_data))
                    hdu.header['DATAMAX'] = float(np.nanmax(normalized_data))
                    hdu.header['DATAMEAN'] = float(np.nanmean(normalized_data))
                    hdu.header['DATAMD'] = float(np.nanmedian(normalized_data))
                    hdu.header['NPIX'] = int(np.sum(valid_pixels))
                
                hdul.append(hdu)
                extensions_created += 1
                logger.debug(f"Created extension {ext_idx}: {ext_name}")
                
            except Exception as e:
                logger.error(f"Error creating extension for {channel} {bench}{side}: {str(e)}")
                continue
        
        # Write the normalized flat field FITS file
        try:
            hdul.writeto(output_filename, overwrite=True)
            logger.info(f"Successfully created normalized flat field: {output_filename}")
            logger.info(f"File contains {extensions_created} extensions (plus primary)")
            
            # Log file size
            file_size = os.path.getsize(output_filename)
            logger.info(f"Normalized flat field file size: {file_size / (1024*1024):.1f} MB")
            
        except Exception as e:
            logger.error(f"Error writing normalized flat field FITS file: {str(e)}")
            raise
        finally:
            hdul.close()
        
        return output_filename
    
    def apply_thresholds(self, science_data):
        """Apply calculated thresholds to the flat field data.

        This method applies the provided thresholds to the flat field data,
        modifying the data in place or returning a new modified dataset.

        Args:
            thresholds (list): List of threshold values to apply.

        Returns:
            None: The method modifies the flat field data in place.
        """
        logger.info("Applying thresholds to science data")
        # Placeholder for actual threshold application logic
        # This should be replaced with the actual implementation
        pass


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
        logger.info(f"Generated {len(results['output_files'])} individual FITS files:")
        for output_file in results['output_files']:
            logger.info(f"  {output_file}")
        
        if results['combined_mef_file']:
            logger.info(f"Combined multi-extension FITS file: {results['combined_mef_file']}")
        else:
            logger.warning("Combined multi-extension FITS file was not created")
        
        if results['normalized_flat_field_file']:
            logger.info(f"Normalized flat field for reduce.py: {results['normalized_flat_field_file']}")
        else:
            logger.warning("Normalized flat field file was not created")
            
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
    
    