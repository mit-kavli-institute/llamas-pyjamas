import os
import logging
from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas
from Image.WhiteLightModule import WhiteLightFits
import numpy as np
from astropy.io import fits
import llamas_pyjamas.GUI.guiExtract as ge
from llamas_pyjamas.config import CALIB_DIR, OUTPUT_DIR
from llamas_pyjamas.Flat.flatProcessing import produce_flat_extractions

from scipy.interpolate import BSpline, make_interp_spline
from pypeit.core.fitting import iterfit
from pypeit import bspline
import pickle
from datetime import datetime

# Set up logging
log_dir = os.path.join(OUTPUT_DIR, 'logs')
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = os.path.join(log_dir, f'flatLlamas_{timestamp}.log')

# Configure the logger
logger = logging.getLogger('flatLlamas')
logger.setLevel(logging.DEBUG)

# Create handlers
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()

# Set levels
file_handler.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)

# Create formatters
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_formatter = logging.Formatter('%(levelname)s: %(message)s')

# Add formatters to handlers
file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info(f"Logging initialized. Log file: {log_file}")


def fit_spectrum_to_xshift(extraction, fiber_index, maxiter=6, bkspace=0.5):
    # Get the xshift and counts data for a specific fiber
    xshift = extraction.xshift[fiber_index, :]
    counts = extraction.counts[fiber_index, :]
    
    logger.debug(f"Fitting fiber {fiber_index} with xshift shape {xshift.shape}")
    
    # Remove any NaN values that might cause fitting issues
    # valid_indices = ~np.isnan(counts) & ~np.isnan(xshift)
    # xshift_clean = xshift[valid_indices]
    # counts_clean = counts[valid_indices]
    
    logger.debug(f"After NaN removal: {len(xshift)} valid points")

    # Using pypeit's iterfit (more robust for spectral data)
    try:
        sset, outmask = iterfit(
            xshift, counts, 
            maxiter=maxiter,
            kwargs_bspline={'bkspace': bkspace}  # Adjust bkspace based on your data sampling
        )
        
        # Create an evaluation grid with finer sampling if needed
        xmodel = np.linspace(np.min(xshift), np.max(xshift), len(xshift))

        y_fit = sset.value(xmodel)[0]
        
        logger.debug(f"Fit successful for fiber {fiber_index}")
        
        return {
            'xshift_clean': xshift,
            'counts_clean': counts,
            'xmodel': xmodel,
            'y_fit': y_fit,
            'bspline_model': sset  # The pypeit bspline model object
        }
    except Exception as e:
        logger.error(f"Error fitting fiber {fiber_index}: {str(e)}")
        raise



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
                    residuals = fiber_fit['counts_clean'] - y_predicted

                    # Calculate statistics for thresholding
                    median_residual = np.median(residuals)
                    std_residual = np.std(residuals)
                    
                    logger.debug(f"Fiber {fiber_idx}: median residual={median_residual:.4f}, std residual={std_residual:.4f}")
                    
                    # Store results for this fiber
                    results[ext_key][fiber_idx] = {
                        'xshift_clean': fiber_fit['xshift_clean'],
                        'counts_clean': fiber_fit['counts_clean'],
                        'y_predicted': y_predicted,
                        'residuals': residuals,
                        'median_residual': median_residual,
                        'std_residual': std_residual,
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
    
    