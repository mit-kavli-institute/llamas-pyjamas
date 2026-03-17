# llamas_pyjamas/utils.py
"""Utils module for llamas_pyjamas package.
This module provides various utility functions for logging, file handling,
data processing, and visualization related to the llamas_pyjamas project.
Functions:
    setup_logger(name, log_filename=None):
        Setup logger with file and console handlers.
    create_peak_lookup(peaks):
        Create lookup table mapping peak index to y position.
    save_combs_to_fits(comb_dict: dict, outfile: str = 'comb_profiles.fits') -> None:
        Save dictionary of comb profiles to multi-extension FITS file.
    grab_peak_pos_from_LUT(file, channel, benchside, fiber=None):
        Retrieve peak positions from a lookup table (LUT) file.
    create_peak_lookups(peaks, benchside=None):
        Create lookup dict with special handling for 2A missing fiber.
    dump_LUT(channel, hdu, trace_obj):
        Dump lookup table (LUT) data from HDU to a JSON file.
    flip_b_side_positions():
        Flip B-side positions in the LUT.
    flip_positions():
        Flip positions in the LUT based on predefined criteria.
    plot_trace(traceobj):
        Plot trace data from a trace object.
    plot_traces_on_image(traceobj, data, zscale=False):
        Plot traces overlaid on raw data with optional zscale.
    check_bspline_negative_values(fit_results, pixel_map_files=None, output_dir='diagnostics'):
        Fast scanning function to identify negative values in B-spline fit results.
    plot_bspline_fit(fit_results, extension_key, fiber_idx, output_file=None):
        Plot individual fiber B-spline fits with original data for inspection.
    check_reference_arc_wavelength_ranges(arc_file=None, verbose=True):
        Check wavelength ranges in the LLAMAS reference arc calibration file.
    check_extraction_wavelength_ranges(extraction_file, reference_arc_file=None, verbose=True):
        Check wavelength ranges in extraction file and compare to reference arc.
"""
import os
import logging
from datetime import datetime
import numpy as np
from astropy.io import fits
from llamas_pyjamas.config import CALIB_DIR, LUT_DIR
import json
import matplotlib.pyplot as plt
import traceback
from matplotlib import cm
from typing import Union
#from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas

import glob
import pickle
import cloudpickle

# Module-level logger for utility functions
# This logger can be used by all functions in this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add console handler if no handlers exist
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

def setup_logger(name, log_filename=None)-> logging.Logger:
    """Setup logger with file and console handlers.

    Creates a logger with both file and console output, automatically creating
    a logs directory if it doesn't exist.

    Args:
        name (str): Logger name (usually __name__).
        log_filename (str, optional): Custom log filename. If None, a default 
            filename will be used. Defaults to None.

    Returns:
        logging.Logger: Configured logger instance.
    """
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create default filename if none provided
    if log_filename is None:
        log_filename = f"{name.replace('.', '_')}.log"
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create file handler
    log_file = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set desired console log level
    
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add only file handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def check_header(fits_file, color=None, bench=None, side=None) -> bool:
    """
    Check the FITS file header for color, bench, and side values.

    Args:
        fits_file (str): Path to the FITS file.
        color (str, optional): Expected color value. Defaults to None.
        bench (str, optional): Expected bench value. Defaults to None.
        side (str, optional): Expected side value. Defaults to None.

    Returns:
        bool: True if all specified checks pass, False otherwise.
    """
    hdu = fits.open(fits_file)
    primary = hdu[0].header

    if 'CAM_NAME' in primary:
        cam_name = primary['CAM_NAME']
        bench_h = cam_name.split('_')[0][0]
        side_h = cam_name.split('_')[0][1]
        color_h = cam_name.split('_')[1].lower()
    else:
        bench_h = primary.get('BENCH')
        side_h = primary.get('SIDE')
        color_h = primary.get('COLOR', '').lower()

    if bench is not None and bench_h != bench:
        logger.error(f'Bench mismatch in {fits_file}: header {bench_h} vs expected {bench}')
        return False
    if side is not None and side_h != side:
        logger.error(f'Side mismatch in {fits_file}: header {side_h} vs expected {side}')
        return False
    if color is not None and color_h != color:
        logger.error(f'Color mismatch in {fits_file}: header {color_h} vs expected {color}')
        return False

    return True


def concat_extractions(pkl_files: list, outfile: str) -> None:
    """Concatenate multiple pickle files containing extraction data.

    This function reads multiple pickle files, concatenates their contents,
    and saves the combined data into a new pickle file.

    Args:
        pkl_files (list): List of paths to pickle files to concatenate.
        outfile (str): Path to the output pickle file.

    Returns:
        None
    """

    assert len(pkl_files) > 0, "No pickle files provided for concatenation."

    combined_data = {'extractions': [], 'metadata': [], 'primary_header': None}

    for pkl_file in pkl_files:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
            if combined_data['primary_header'] is None:
                combined_data['primary_header'] = data.get('primary_header', None)
            if 'extractions' in data:
                combined_data['extractions'].extend(data['extractions'])
            if 'metadata' in data:
                combined_data['metadata'].extend(data['metadata'])
    
    with open(outfile, 'wb') as f:
        cloudpickle.dump(combined_data, f)
    
    return


def is_wavelength_solution_useable(wvln_solution_dict, extraction_idx=0):
    """Check if wavelength solution xshift has any non-zero values."""
    xshifts = wvln_solution_dict['extractions'][extraction_idx].xshift
    return not np.all(xshifts == 0)


def create_peak_lookup(peaks: list) -> dict:
    """Create lookup table mapping peak index to y position.

    Args:
        peaks (list): List of peak y-positions.

    Returns:
        dict: Dictionary mapping peak index to y position.
    """
    peak_pos = {idx: y_pos for idx, y_pos in enumerate(peaks)}
    
    return peak_pos

def save_combs_to_fits(comb_dict: dict, outfile: str = 'comb_profiles.fits') -> None:
    """Save dictionary of comb profiles to multi-extension FITS file.

    Creates a FITS file with each comb profile as a separate extension.

    Args:
        comb_dict (dict): Dictionary of comb profiles with format {'label': comb_array}.
        outfile (str, optional): Output filename. Defaults to 'comb_profiles.fits'.

    Returns:
        None
    """
    # Create HDU list with empty primary
    hdul = fits.HDUList([fits.PrimaryHDU()])
    
    # Add each comb as an extension
    for label, comb in comb_dict.items():
        # Create ImageHDU for this comb
        hdu = fits.ImageHDU(data=comb.astype(np.float32))
        # Clean label for FITS extension name (remove spaces, special chars)
        ext_name = label.replace(' ', '_').upper()
        hdu.header['EXTNAME'] = ext_name
        hdu.header['LABEL'] = label
        hdu.header['BUNIT'] = 'Counts'
        hdul.append(hdu)
    
    # Write to file
    outpath = os.path.join(CALIB_DIR, outfile)
    hdul.writeto(outpath, overwrite=True)
    
    return



def grab_peak_pos_from_LUT(file: str, channel: str, benchside: str, fiber=None) -> Union[list, int]:
    """Retrieve peak positions from a Look-Up Table (LUT) file.

    Args:
        file (str): Path to the LUT file in JSON format.
        channel (str): The channel to retrieve data for.
        benchside (str): The benchside to retrieve data for.
        fiber (int, optional): Specific fiber to retrieve the peak position for. 
            If None, returns all peak positions. Defaults to None.

    Returns:
        Union[list, int]: A list of peak positions if fiber is not specified, 
            otherwise the peak position for the specified fiber.
    """

    with open(file, 'r') as f:
        lut = json.load(f)
    peak_positions = [int(val) for val in lut[channel][benchside].values()]
    if not fiber:
        return peak_positions
    
    return peak_positions[fiber]


def create_peak_lookups(peaks: np.ndarray, benchside=None)-> dict:
    """Create a lookup dictionary for peak positions with special handling for missing fibers.

    This function processes an array of peak positions and inserts additional peaks 
    at specific indices based on the provided benchside. It handles special cases 
    for benchsides '2A' and '2B' where fibers may be missing.

    Args:
        peaks (np.ndarray): An array of peak positions.
        benchside (str, optional): The benchside identifier, either '2A' or '2B'. 
            If None, no special handling is applied. Defaults to None.

    Returns:
        dict: A dictionary where the keys are the indices of the peaks (as strings) 
            and the values are the peak positions (as integers).

    Note:
        - **Benchside '2A' Handling**: Calculates average spacing and inserts new 
          peaks at indices 269 and 298.
        - **Benchside '2B' Handling**: Calculates average spacing and inserts new 
          peak at index 48.

    Example:
        >>> peaks = np.array([100, 200, 300, 400])
        >>> create_peak_lookups(peaks, benchside='2A')
        {'0': 100, '1': 200, '2': 300, '3': 400, '269': 250, '298': 450}
    """
    
    if benchside == '2A':
        # Calculate average peak spacing
        peak_diffs = np.diff(peaks)
        avg_spacing = np.mean(peak_diffs)
        
        # Find index 269 and insert new peak
        peak_list = peaks.tolist()
        idx_269 = 269
        idx_298 = 298
        
        if idx_269 < len(peak_list):
            new_peak_pos = int(peak_list[idx_269] + avg_spacing)
            peak_list.insert(idx_269 + 1, new_peak_pos)
            peaks = np.array(peak_list)
            
        if idx_298 < len(peak_list):
            new_peak_pos = int(peak_list[idx_298] + avg_spacing)
            peak_list.insert(idx_298 + 1, new_peak_pos)
            peaks = np.array(peak_list)
    
    if benchside == '2B':
        # Calculate average peak spacing
        peak_diffs = np.diff(peaks)
        avg_spacing = np.mean(peak_diffs)
        
        # Find index 269 and insert new peak
        peak_list = peaks.tolist()
        idx_48 = 48
        
        if idx_48 < len(peak_list):
            new_peak_pos = int(peak_list[idx_48] + avg_spacing)
            peak_list.insert(idx_48 + 1, new_peak_pos)
            peaks = np.array(peak_list)
        
        
    # Create lookup with regular Python ints
    return {str(idx): int(pos) for idx, pos in enumerate(peaks)}

def dump_LUT(channel: str, hdu, trace_obj: 'TraceLlamas')-> None:
    """
    Dumps the Look-Up Table (LUT) for a given channel and HDU (Header Data Unit) into a JSON file.
    Parameters:
    channel (str): The channel name to be processed.
    hdu (astropy.io.fits.HDUList): The HDU list containing the FITS file data.
    trace_obj (TraceObject): An object that processes HDU data and extracts LUT information.
    Returns:
    None
    Raises:
    FileNotFoundError: If the 'traceLUT.json' file is not found when attempting to load the master LUT.
    Notes:
    - The function initializes or loads a master LUT from 'traceLUT.json'.
    - It processes each relevant HDU to extract LUT information and updates the master LUT.
    - The updated master LUT is saved back to 'traceLUT.json'.
    """

    # Initialize or load master LUT
    try:
        if 'CAM_NAME' in hdu[1].header:
            channel_hdu_idx = [i for i in range(1, len(hdu)) if channel in hdu[i].header['CAM_NAME'].lower()]
        else:
            channel_hdu_idx = [i for i in range(1, len(hdu)) if channel in hdu[i].header['COLOR'].lower()]
        
        with open('traceLUT.json', 'r') as f:
            master_lut = json.load(f)
    except FileNotFoundError:
        master_lut = {channel: {}}

    # Loop through and add each LUT
    for i in channel_hdu_idx:
        
        if 'CAM_NAME' in hdu[i].header:
            cam_name = hdu[i].header['CAM_NAME']
            bench = cam_name.split('_')[0][0]
            side = cam_name.split('_')[0][1]
            
        else:
            channel = hdu[i].header['COLOR'].lower()
            bench = hdu[i].header['BENCH']
            side  = hdu[i].header['SIDE']
            
        benchside = f"{bench}{side}"
        
        trace_obj.process_hdu_data(hdu[i].data, dict(hdu[i].header), find_LUT=True)
        comb = trace_obj.orig_comb
        peaks = trace_obj.first_peaks
        # Convert numpy array to regular Python types
        peaks_dict = create_peak_lookups(peaks, benchside=benchside)
        
        master_lut["combs"][channel][benchside] = trace_obj.orig_comb.tolist()
        # Add to master LUT
        master_lut["fib_pos"][channel][benchside] = peaks_dict
        print(f"Added {benchside} peaks to LUT")

    # Save updated master LUT 
    with open('traceLUT.json', 'w') as f:
        json.dump(master_lut, f, indent=4)
    
    return


def trace_dump_LUT(trace_obj: 'TraceLlamas')-> None:
    """
    Dumps the Look-Up Table (LUT) for a given channel and HDU (Header Data Unit) into a JSON file.
    Parameters:
    channel (str): The channel name to be processed.
    hdu (astropy.io.fits.HDUList): The HDU list containing the FITS file data.
    trace_obj (TraceObject): An object that processes HDU data and extracts LUT information.
    Returns:
    None
    Raises:
    FileNotFoundError: If the 'traceLUT.json' file is not found when attempting to load the master LUT.
    Notes:
    - The function initializes or loads a master LUT from 'traceLUT.json'.
    - It processes each relevant HDU to extract LUT information and updates the master LUT.
    - The updated master LUT is saved back to 'traceLUT.json'.
    """

    # Initialize or load master LUT
    try:
        print(os.path.join(LUT_DIR, "traceLUT.json"))
        with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'r') as f:
            master_lut = json.load(f)
    except FileNotFoundError:
        print(f'No LUT file found, for {os.path.join(LUT_DIR, "traceLUT.json")}')
        master_lut = {channel: {}}

    benchside = trace_obj.benchside
    channel = trace_obj.channel
    comb = trace_obj.orig_comb
    peaks = trace_obj.first_peaks
    # Convert numpy array to regular Python types
    peaks_dict = create_peak_lookups(peaks, benchside=benchside)
    
    master_lut["combs"][channel][benchside] = trace_obj.orig_comb.tolist()
    # Add to master LUT
    master_lut["fib_pos"][channel][benchside] = peaks_dict
    print(f"Added {channel} {benchside} peaks to LUT")

    # Save updated master LUT 
    with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'w') as f:
        json.dump(master_lut, f, indent=4)
    
    return






def flip_b_side_positions()-> None:
    """
    Flips the fiber positions for the 'B' side in the lookup table (LUT) for each color.
    This function reads a JSON file containing a lookup table (LUT) of fiber positions,
    processes the positions for each color ('green', 'blue', 'red'), and flips the positions
    for the 'B' side. The flipped positions are then saved to a new JSON file.
    The LUT is expected to have the following structure:
    {
        "fib_pos": {
            "color": {
                "benchside": {
                    "fiber_number": "position"
                }
            }
        }
    }
    The function performs the following steps:
    1. Loads the LUT from 'LUT/traceLUT.json'.
    2. Iterates over each color ('green', 'blue', 'red').
    3. For each color, checks if the 'B' side exists in the LUT.
    4. If the 'B' side exists, flips the fiber positions by reversing the fiber numbers.
    5. Saves the updated LUT to 'LUT/traceLUT_copy.json'.
    Raises:
        FileNotFoundError: If the LUT file does not exist.
        json.JSONDecodeError: If the LUT file is not a valid JSON.
    """

    # Load LUT
    with open('LUT/traceLUT.json', 'r') as f:
        lut = json.load(f)
    # Process each color
    for color in ['green', 'blue', 'red']:
        if color not in lut['fib_pos']:
            continue
        # Process each benchside
        for benchside in lut['fib_pos'][color].keys():
            if 'B' in benchside:
                positions = lut['fib_pos'][color][benchside]
                # Get max fiber number
                max_fiber = max(int(k) for k in positions.keys())
                # Create new flipped mapping
                flipped = {}
                for k, v in positions.items():
                    new_key = str(max_fiber - int(k))
                    flipped[new_key] = v
                # Replace original mapping
                lut['fib_pos'][color][benchside] = flipped
    # Save updated LUT
    with open('LUT/traceLUT_copy.json', 'w') as f:
        json.dump(lut, f, indent=4)
        
def flip_positions()-> None:
    """
    Flips the positions of fibers in the lookup table (LUT) based on the specified configuration.
    This function reads a JSON file containing the LUT, processes each color and benchside,
    and flips the positions of the fibers if specified in the `flipped` dictionary. The updated
    LUT is then saved back to the JSON file.
    The `flipped` dictionary determines which fibers to flip for each color and benchside.
    The keys in the dictionary are in the format "{color}{side}" (e.g., "greenA", "blueB"),
    and the values are booleans indicating whether to flip the positions (True) or not (False).
    Example of `flipped` dictionary:
    
    flipped = {"greenA": False, "greenB": True, "blueA": True, "blueB": False, "redA": False, "redB": True}
    
    The function will flip the positions of the fibers for each specified color and benchside,
    and save the updated LUT back to the JSON file.
    Raises:
        FileNotFoundError: If the LUT JSON file is not found.
        json.JSONDecodeError: If there is an error decoding the JSON file.
    """

    # flipped = {"greenA":True, "greenB":False, "blueA": False, "blueB":True, "redA":True, "redB":False}
    #Attempt to check on 23rd Jan 2025
    flipped = {"greenA":False, "greenB":True, "blueA": True, "blueB":False, "redA":False, "redB":True}
    ##temp one
    #flipped = {"greenA":True, "greenB":False, "blueA": True, "blueB":False, "redA":True, "redB":False}
    print(f'flipping {flipped}')
    # Load LUT
    with open('LUT/traceLUT.json', 'r') as f:
        lut = json.load(f)
    
    # Process each color
    for color in ['green', 'blue', 'red']:
        if color not in lut['fib_pos']:
            continue
            
        # Process each benchside
        for benchside in lut['fib_pos'][color].keys():
            side = 'A' if 'A' in benchside else 'B'
            lookup_key = f"{color}{side}"
            
            # Only flip if specified in flipped dict
            if flipped.get(lookup_key, False):
                positions = lut['fib_pos'][color][benchside]
                
                # Get max fiber number
                max_fiber = max(int(k) for k in positions.keys())
                
                # Create new flipped mapping
                flipped_positions = {}
                for k, v in positions.items():
                    new_key = str(max_fiber - int(k))
                    flipped_positions[new_key] = v
                    
                # Replace original mapping
                lut['fib_pos'][color][benchside] = flipped_positions
                print(f"Flipped {color} {benchside}")
    
    # Save updated LUT
    with open('LUT/traceLUT.json', 'w') as f:
        json.dump(lut, f, indent=4)


def copy_mastercalib_trace(channel: str, bench: str, side: str,
                          mastercalib_dir: str, target_dir: str) -> str:
    """
    Copy mastercalib trace to target directory, replacing the invalid user trace.

    Args:
        channel: Color channel (red/green/blue)
        bench: Bench number
        side: Side letter (A/B)
        mastercalib_dir: Source mastercalib directory
        target_dir: Destination user trace directory

    Returns:
        str: Path to copied trace file

    Raises:
        FileNotFoundError: If mastercalib trace not found

    Example:
        >>> copy_mastercalib_trace('green', '4', 'B', '/calib', '/user/traces')
        '/user/traces/LLAMAS_green_4_B_traces.pkl'
    """
    import shutil

    # Find mastercalib trace
    mastercalib_filename = f'LLAMAS_master_{channel.lower()}_{bench}_{side}_traces.pkl'
    mastercalib_path = os.path.join(mastercalib_dir, mastercalib_filename)

    if not os.path.exists(mastercalib_path):
        raise FileNotFoundError(f"Mastercalib trace not found: {mastercalib_path}")

    # Copy to target directory using USER naming convention (without "master_" prefix)
    # This replaces the invalid user trace file
    user_filename = f'LLAMAS_{channel.lower()}_{bench}_{side}_traces.pkl'
    target_path = os.path.join(target_dir, user_filename)

    # Use shutil.copy2 to preserve metadata
    shutil.copy2(mastercalib_path, target_path)
    logger.info(f"Replaced invalid trace with mastercalib: {user_filename}")

    return target_path


def validate_and_fix_trace_fibres(trace_dir: str, mastercalib_dir: str = CALIB_DIR) -> dict:
    """
    Validate each trace file individually and copy mastercalib fallback when needed.

    This function checks each trace file for correct fiber count. If a trace has
    the wrong number of fibers, it automatically copies the corresponding mastercalib
    trace as a fallback, allowing the pipeline to continue with a hybrid trace set.

    Args:
        trace_dir: User trace directory to validate
        mastercalib_dir: Mastercalib directory for fallback traces

    Returns:
        dict: {
            'valid_traces': [(channel, bench, side, filepath), ...],
            'invalid_traces': [(channel, bench, side, expected, actual), ...],
            'fallback_used': [(channel, bench, side, mastercalib_path, copied_path), ...],
            'all_valid': bool
        }

    Example:
        >>> results = validate_and_fix_trace_fibres('/user/traces')
        >>> if not results['all_valid']:
        ...     print(f"Replaced {len(results['fallback_used'])} invalid traces")
    """
    # Expected fiber counts per benchside
    N_fib = {'1A': 298, '1B': 300, '2A': 298, '2B': 297,
             '3A': 298, '3B': 300, '4A': 300, '4B': 298}

    # Get all trace files in directory
    files = glob.glob(os.path.join(trace_dir, '*.pkl'))

    if not files:
        logger.warning(f"No trace files found in {trace_dir}")
        return {
            'valid_traces': [],
            'invalid_traces': [],
            'fallback_used': [],
            'all_valid': False
        }

    valid_traces = []
    invalid_traces = []
    fallback_used = []

    for pkl_file in files:
        try:
            # Load trace object
            with open(pkl_file, "rb") as file:
                traceobj = pickle.load(file)

            # Extract trace information
            shape = traceobj.traces.shape[0]
            channel = traceobj.channel.lower()
            bench = str(traceobj.bench)
            side = traceobj.side
            benchside = f"{bench}{side}"

            # Get expected fiber count
            expected_fibers = N_fib.get(benchside)

            if expected_fibers is None:
                logger.warning(f"Unknown benchside {benchside} in {os.path.basename(pkl_file)}")
                continue

            # Check if fiber count matches
            if shape == expected_fibers:
                valid_traces.append((channel, bench, side, pkl_file))
                logger.debug(f"✓ {channel}{bench}{side}: {shape}/{expected_fibers} fibers (valid)")
            else:
                invalid_traces.append((channel, bench, side, expected_fibers, shape))
                logger.warning(f"✗ {channel}{bench}{side}: {shape}/{expected_fibers} fibers (invalid)")

                # Copy mastercalib fallback
                try:
                    copied_path = copy_mastercalib_trace(
                        channel, bench, side,
                        mastercalib_dir, trace_dir
                    )
                    fallback_used.append((channel, bench, side, os.path.join(mastercalib_dir, f'LLAMAS_master_{channel}_{bench}_{side}_traces.pkl'), copied_path))
                    logger.info(f"✓ Copied mastercalib fallback for {channel}{bench}{side}")
                except FileNotFoundError as e:
                    logger.error(f"✗ Could not find mastercalib fallback for {channel}{bench}{side}: {e}")

        except Exception as e:
            logger.error(f"Error validating trace file {pkl_file}: {e}")
            continue

    all_valid = len(invalid_traces) == 0

    return {
        'valid_traces': valid_traces,
        'invalid_traces': invalid_traces,
        'fallback_used': fallback_used,
        'all_valid': all_valid
    }


def count_trace_fibres(mastercalib_dir: str = CALIB_DIR) -> bool:
    """
    Checks if all trace files have the correct number of fibers for their benchside.

    DEPRECATED: Use validate_and_fix_trace_fibres() for more detailed validation
    with automatic fallback support.

    Args:
        mastercalib_dir (str): Directory containing the trace calibration pickle files.
            Defaults to CALIB_DIR from config.

    Returns:
        bool: True if all trace files have the correct fiber count, False otherwise.
    """
    N_fib = {'1A':298, '1B':300, '2A':298, '2B':297, '3A':298, '3B':300, '4A':300, '4B':298}
    files = glob.glob(mastercalib_dir+'/*.pkl')
    assert type(files) == list, 'File extraction did not return as list'

    all_match = True

    for idx, pkl in enumerate(files):
        with open(pkl, "rb") as file:
            traceobj = pickle.load(file)
        shape = traceobj.traces.shape[0]
        channel = traceobj.channel
        benchside = f"{traceobj.bench}{traceobj.side}"
        req = N_fib[benchside]

        if shape != req:
            all_match = False

        print(f'Channel {channel} Benchside {benchside} trace has {shape} fibres and requires {req}')

    return all_match
        
        
def flip_blue_combs()-> None:
    """
    Flips all blue combs in the lookup table (LUT) file.
    
    This function reads the LUT JSON file, processes all benchsides ('1A' through '4B')
    in the blue combs section, and flips those comb arrays. The updated LUT is then 
    saved back to the JSON file.
    
    The function performs the following steps:
    1. Loads the LUT from 'LUT/traceLUT.json'.
    2. Checks if 'combs' and 'blue' keys exist in the LUT structure.
    3. For each benchside (1A-4B) in the blue combs, flips the comb array.
    4. Saves the updated LUT back to the original file.
    
    Raises:
        FileNotFoundError: If the LUT file does not exist.
        json.JSONDecodeError: If the LUT file is not a valid JSON.
        KeyError: If the expected structure ('combs' > 'blue') is not found.
    """
    # Load LUT
    with open('LUT/traceLUT.json', 'r') as f:
        lut = json.load(f)
    
    # Check if 'combs' and 'blue' keys exist
    if 'combs' not in lut:
        print("Combs data not found in LUT")
        return
        
    if 'blue' not in lut['combs']:
        print("Blue comb data not found in LUT")
        return
    
    # Process each benchside for blue combs
    for benchside in lut['combs']['blue'].keys():
        # Get the comb as a list
        comb = lut['combs']['blue'][benchside]
        
        # Flip the comb array
        flipped_comb = list(reversed(comb))
        
        # Replace the original comb with the flipped one
        lut['combs']['blue'][benchside] = flipped_comb
        print(f"Flipped blue comb for benchside {benchside}")
    
    # Save the updated LUT
    with open('LUT/traceLUT.json', 'w') as f:
        json.dump(lut, f, indent=4)
    
    print("Blue combs successfully flipped and saved to LUT/traceLUT.json")
    
    
def check_image_properties(hdu: fits.HDUList, start_idx: int = 1) -> None:

    for i in range(start_idx, len(hdu)):
        color = hdu[i].header['COLOR']
        bench = hdu[i].header['BENCh']
        side = hdu[i].header['SIDE']
        data = hdu[i].data
        has_nan = np.isnan(data).any()
        has_neg = np.any(data < 0)
        print(f'color {color} bench {bench} side {side}')
        print(f'Has NaN: {has_nan}')
        print(f'Has negative value {has_neg}')


def check_bspline_negative_values(fit_results, pixel_map_files=None, output_dir='diagnostics'):
    """
    Fast scanning function to identify and summarize negative values in B-spline fit results.

    This function checks B-spline fit results for negative predicted values, which should not
    occur in flat field calibration. It also optionally checks pixel map FITS files for
    negative pixels. Results are summarized without generating plots for performance.

    Args:
        fit_results (dict): Dictionary from calculate_fits_all_extensions() containing
            B-spline fits for each extension and fiber
        pixel_map_files (list, optional): List of pixel map FITS file paths to check
            for negative values. Defaults to None.
        output_dir (str, optional): Directory for saving diagnostic report.
            Defaults to 'diagnostics'.

    Returns:
        dict: Dictionary mapping (extension_key, fiber_idx) -> negative_info for
            problematic fibers found
    """
    import os

    print("="*80)
    print("B-SPLINE NEGATIVE VALUE CHECKER")
    print("="*80)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    problematic_fibers = {}
    total_fibers = 0
    total_negative_predictions = 0

    # Check fit results for negative predictions
    print(f"\nChecking B-spline fit results...")
    print(f"Found {len(fit_results)} extensions to check")

    for ext_key, extension_data in fit_results.items():
        print(f"\nExtension {ext_key}:")
        extension_problems = 0

        for fiber_idx, fiber_data in extension_data.items():
            total_fibers += 1

            # Check y_predicted values
            y_predicted = fiber_data.get('y_predicted', [])
            if len(y_predicted) == 0:
                continue

            negative_mask = np.array(y_predicted) < 0
            negative_count = np.sum(negative_mask)

            if negative_count > 0:
                min_value = np.min(y_predicted)
                negative_fraction = negative_count / len(y_predicted) * 100

                problematic_fibers[(ext_key, fiber_idx)] = {
                    'negative_count': negative_count,
                    'total_points': len(y_predicted),
                    'negative_fraction': negative_fraction,
                    'min_value': min_value,
                    'rms_residual': fiber_data.get('rms_residual', np.nan),
                    'relative_rms': fiber_data.get('relative_rms', np.nan)
                }

                extension_problems += 1
                total_negative_predictions += negative_count

        print(f"  {len(extension_data)} fibers checked, {extension_problems} with negative predictions")

    # Check pixel map files if provided
    pixel_map_negatives = 0
    total_pixels_checked = 0

    if pixel_map_files:
        print(f"\nChecking {len(pixel_map_files)} pixel map files...")

        for pixel_map_file in pixel_map_files:
            if not os.path.exists(pixel_map_file):
                print(f"  WARNING: File not found: {os.path.basename(pixel_map_file)}")
                continue

            try:
                with fits.open(pixel_map_file) as hdul:
                    data = hdul[0].data
                    if data is not None:
                        negative_pixels = np.sum(data < 0)
                        total_pixels = data.size
                        total_pixels_checked += total_pixels
                        pixel_map_negatives += negative_pixels

                        if negative_pixels > 0:
                            min_value = np.min(data)
                            print(f"  {os.path.basename(pixel_map_file)}: {negative_pixels:,} negative pixels "
                                  f"({negative_pixels/total_pixels*100:.2f}%), min={min_value:.3f}")
                        else:
                            print(f"  {os.path.basename(pixel_map_file)}: No negative pixels")
            except Exception as e:
                print(f"  ERROR reading {os.path.basename(pixel_map_file)}: {str(e)}")

    # Print summary
    print(f"\n" + "="*60)
    print(f"SUMMARY")
    print(f"="*60)
    print(f"Total fibers checked: {total_fibers}")
    print(f"Fibers with negative predictions: {len(problematic_fibers)}")
    print(f"Total negative prediction points: {total_negative_predictions}")

    if pixel_map_files:
        print(f"Total pixels checked in pixel maps: {total_pixels_checked:,}")
        print(f"Negative pixels found: {pixel_map_negatives:,}")

    if problematic_fibers:
        print(f"\nPROBLEMATIC FIBERS:")
        for (ext_key, fiber_idx), info in sorted(problematic_fibers.items()):
            print(f"  {ext_key} fiber {fiber_idx}: {info['negative_count']} negative points "
                  f"({info['negative_fraction']:.1f}%), min={info['min_value']:.3f}, "
                  f"RMS={info['rms_residual']:.3f}")
    else:
        print(f"\n✅ No negative values found in B-spline predictions!")

    # Write detailed report
    report_file = os.path.join(output_dir, 'negative_values_report.txt')
    with open(report_file, 'w') as f:
        f.write("B-SPLINE NEGATIVE VALUE REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total fibers checked: {total_fibers}\n")
        f.write(f"Fibers with negative predictions: {len(problematic_fibers)}\n")
        f.write(f"Total negative prediction points: {total_negative_predictions}\n\n")

        if pixel_map_files:
            f.write(f"Pixel map files checked: {len(pixel_map_files)}\n")
            f.write(f"Total pixels checked: {total_pixels_checked:,}\n")
            f.write(f"Negative pixels found: {pixel_map_negatives:,}\n\n")

        if problematic_fibers:
            f.write("DETAILED PROBLEMATIC FIBER LIST:\n")
            f.write("-" * 40 + "\n")
            for (ext_key, fiber_idx), info in sorted(problematic_fibers.items()):
                f.write(f"Extension: {ext_key}, Fiber: {fiber_idx}\n")
                f.write(f"  Negative points: {info['negative_count']}/{info['total_points']} "
                        f"({info['negative_fraction']:.1f}%)\n")
                f.write(f"  Minimum value: {info['min_value']:.6f}\n")
                f.write(f"  RMS residual: {info['rms_residual']:.6f}\n")
                f.write(f"  Relative RMS: {info['relative_rms']:.6f}\n\n")

    print(f"\nDetailed report saved to: {report_file}")

    return problematic_fibers


def assess_pixel_maps(pixel_map_file, output_dir=None, extensions=None, save_figs=True):
    """Assess pixel-to-pixel flat field maps for imprint patterns and data quality.

    For each extension in ``pixel_map_file`` this function computes:

    - Global statistics (mean, median, std, fraction of pixels outside [0.9, 1.1])
    - 2D image display with a diverging colormap centred at 1.0
    - Spatial profile: median collapsed along the spectral axis to reveal
      cross-fiber sensitivity gradients or block-level imprint patterns
    - Spectral profile: median collapsed along the spatial axis to reveal
      large-scale spectral residuals or fringing structure
    - Power spectrum of the spectral profile via FFT to quantify fringing
      frequency and amplitude (especially important for the red channel)
    - Histogram of pixel values

    A multi-panel PNG is saved per extension and a plain-text summary report is
    written listing per-extension statistics with flags for anomalous structure.

    Args:
        pixel_map_file (str): Path to the ``pixel_maps.fits`` MEF file produced by
            ``process_flat_field_complete``.
        output_dir (str, optional): Directory for output PNGs and the summary report.
            Defaults to the directory containing ``pixel_map_file``.
        extensions (list of str, optional): Subset of EXTNAME values to assess
            (e.g. ``['red1A', 'red2B']``).  If ``None`` all extensions are assessed.
        save_figs (bool): Whether to write PNG files.  Defaults to True.

    Returns:
        dict: ``{ext_name: {'mean': float, 'std': float, 'median': float,
                            'frac_outside': float, 'dominant_fft_freq': float,
                            'dominant_fft_amp': float}}``
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from astropy.io import fits

    if not os.path.exists(pixel_map_file):
        raise FileNotFoundError(f"Pixel map file not found: {pixel_map_file}")

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(pixel_map_file))
    os.makedirs(output_dir, exist_ok=True)

    summary = {}
    report_lines = [
        f"Pixel Map Assessment: {os.path.basename(pixel_map_file)}",
        "=" * 70,
        f"{'Extension':<12} {'Mean':>8} {'Median':>8} {'Std':>8} {'%Outside':>9} "
        f"{'FFT_freq':>9} {'FFT_amp':>9} {'Flags'}",
        "-" * 70,
    ]

    with fits.open(pixel_map_file) as hdul:
        image_exts = [h for h in hdul if h.data is not None and h.data.ndim == 2]

        for hdu in image_exts:
            ext_name = hdu.header.get('EXTNAME', 'UNKNOWN')

            if extensions is not None and ext_name not in extensions:
                continue

            data = hdu.data.astype(np.float64)

            # ------------------------------------------------------------------
            # Statistics
            # ------------------------------------------------------------------
            flat_data = data.ravel()
            finite_mask = np.isfinite(flat_data)
            flat_finite = flat_data[finite_mask]

            mean_val  = float(np.mean(flat_finite))
            median_val = float(np.median(flat_finite))
            std_val   = float(np.std(flat_finite))
            frac_outside = float(np.mean((flat_finite < 0.9) | (flat_finite > 1.1)))

            # ------------------------------------------------------------------
            # Collapsed profiles
            # ------------------------------------------------------------------
            # Spatial profile: median along spectral axis (axis=1 → along columns)
            # Result shape = (nrows,) — one value per detector row
            spatial_profile = np.nanmedian(data, axis=1)

            # Spectral profile: median along spatial axis (axis=0 → along rows)
            # Result shape = (ncols,) — one value per spectral pixel
            spectral_profile = np.nanmedian(data, axis=0)

            # ------------------------------------------------------------------
            # Power spectrum of spectral profile
            # ------------------------------------------------------------------
            sp_finite = spectral_profile[np.isfinite(spectral_profile)]
            sp_detrended = sp_finite - np.mean(sp_finite)
            fft_coeffs = np.fft.rfft(sp_detrended)
            fft_power  = np.abs(fft_coeffs) ** 2
            freqs      = np.fft.rfftfreq(len(sp_finite))

            # Ignore DC (index 0)
            if len(fft_power) > 1:
                peak_idx = int(np.argmax(fft_power[1:]) + 1)
                dominant_fft_freq = float(freqs[peak_idx])
                dominant_fft_amp  = float(np.sqrt(fft_power[peak_idx]) / len(sp_finite))
            else:
                dominant_fft_freq = 0.0
                dominant_fft_amp  = 0.0

            summary[ext_name] = {
                'mean': mean_val,
                'median': median_val,
                'std': std_val,
                'frac_outside': frac_outside,
                'dominant_fft_freq': dominant_fft_freq,
                'dominant_fft_amp': dominant_fft_amp,
            }

            # ------------------------------------------------------------------
            # Flags
            # ------------------------------------------------------------------
            flags = []
            if std_val > 0.05:
                flags.append('HIGH_STD')
            if dominant_fft_amp > 0.01:
                flags.append('FRINGING')
            if abs(mean_val - 1.0) > 0.02:
                flags.append('OFFSET')
            flag_str = ','.join(flags) if flags else 'OK'

            report_lines.append(
                f"{ext_name:<12} {mean_val:>8.4f} {median_val:>8.4f} {std_val:>8.4f} "
                f"{frac_outside*100:>8.2f}% {dominant_fft_freq:>9.4f} "
                f"{dominant_fft_amp:>9.5f}  {flag_str}"
            )

            # ------------------------------------------------------------------
            # Multi-panel figure
            # ------------------------------------------------------------------
            if save_figs:
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle(
                    f"Pixel Map Assessment: {ext_name}\n"
                    f"mean={mean_val:.4f}  median={median_val:.4f}  "
                    f"std={std_val:.4f}  {frac_outside*100:.1f}% outside [0.9, 1.1]",
                    fontsize=12
                )

                # Panel 1: 2D image
                ax = axes[0, 0]
                vmin = float(np.nanpercentile(data, 2))
                vmax = float(np.nanpercentile(data, 98))
                # Centre the diverging colourmap at 1.0
                half_range = max(abs(vmin - 1.0), abs(vmax - 1.0), 0.01)
                im = ax.imshow(data, origin='lower', aspect='auto',
                               cmap='RdBu_r',
                               vmin=1.0 - half_range, vmax=1.0 + half_range)
                plt.colorbar(im, ax=ax, label='Pixel ratio')
                ax.set_title('2D Pixel Map')
                ax.set_xlabel('Spectral pixel')
                ax.set_ylabel('Spatial pixel')

                # Panel 2: Spatial profile
                ax = axes[0, 1]
                ax.plot(spatial_profile, np.arange(len(spatial_profile)),
                        color='steelblue', lw=0.8)
                ax.axvline(1.0, color='red', ls='--', lw=1.5, label='Ideal (1.0)')
                ax.set_title('Spatial Profile\n(median along spectral axis)')
                ax.set_xlabel('Median pixel ratio')
                ax.set_ylabel('Detector row')
                ax.legend(fontsize=8)

                # Panel 3: Spectral profile
                ax = axes[0, 2]
                ax.plot(np.arange(len(spectral_profile)), spectral_profile,
                        color='darkorange', lw=0.8)
                ax.axhline(1.0, color='red', ls='--', lw=1.5, label='Ideal (1.0)')
                ax.set_title('Spectral Profile\n(median along spatial axis)')
                ax.set_xlabel('Spectral pixel')
                ax.set_ylabel('Median pixel ratio')
                ax.legend(fontsize=8)

                # Panel 4: Power spectrum
                ax = axes[1, 0]
                ax.semilogy(freqs[1:], fft_power[1:], color='purple', lw=0.8)
                ax.axvline(dominant_fft_freq, color='red', ls='--', lw=1.5,
                           label=f'Peak f={dominant_fft_freq:.4f}')
                ax.set_title('Power Spectrum of Spectral Profile\n(fringing indicator)')
                ax.set_xlabel('Spatial frequency (cycles/pixel)')
                ax.set_ylabel('Power')
                ax.legend(fontsize=8)

                # Panel 5: Histogram
                ax = axes[1, 1]
                clip_lo, clip_hi = 0.5, 1.5
                hist_data = flat_finite[(flat_finite > clip_lo) & (flat_finite < clip_hi)]
                ax.hist(hist_data, bins=200, color='#2ab0ff', edgecolor='none', alpha=0.8)
                ax.axvline(1.0, color='red', ls='--', lw=1.5, label='Ideal (1.0)')
                ax.axvline(mean_val, color='green', ls=':', lw=1.5,
                           label=f'Mean={mean_val:.4f}')
                ax.set_title(f'Pixel Value Histogram\nstd={std_val:.4f}')
                ax.set_xlabel('Pixel ratio (Actual / Model)')
                ax.set_ylabel('Count')
                ax.legend(fontsize=8)

                # Panel 6: Text summary
                ax = axes[1, 2]
                ax.axis('off')
                summary_text = (
                    f"Extension: {ext_name}\n\n"
                    f"Statistics:\n"
                    f"  Mean:    {mean_val:.5f}\n"
                    f"  Median:  {median_val:.5f}\n"
                    f"  Std:     {std_val:.5f}\n"
                    f"  % outside [0.9,1.1]: {frac_outside*100:.2f}%\n\n"
                    f"Fringing (spectral FFT):\n"
                    f"  Peak freq:  {dominant_fft_freq:.5f} cyc/px\n"
                    f"  Peak amp:   {dominant_fft_amp:.5f}\n\n"
                    f"Flags: {flag_str}"
                )
                ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                        fontsize=10, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

                plt.tight_layout()
                out_png = os.path.join(output_dir, f"assessment_{ext_name}.png")
                plt.savefig(out_png, dpi=120, bbox_inches='tight')
                plt.close(fig)
                print(f"  Saved: {os.path.basename(out_png)}")

    # ------------------------------------------------------------------
    # Write summary report
    # ------------------------------------------------------------------
    report_lines.append("=" * 70)
    flagged = [k for k, v in summary.items()
               if v['std'] > 0.05 or v['dominant_fft_amp'] > 0.01 or abs(v['mean'] - 1.0) > 0.02]
    report_lines.append(f"\nFlagged extensions ({len(flagged)}): {', '.join(flagged) or 'none'}")
    report_lines.append(
        "\nFlag definitions:\n"
        "  HIGH_STD  : std > 0.05 (excessive pixel-to-pixel structure)\n"
        "  FRINGING  : dominant FFT amplitude > 0.01 (spectral ripple)\n"
        "  OFFSET    : |mean - 1.0| > 0.02 (calibration level offset)\n"
    )

    report_file = os.path.join(output_dir, 'pixel_map_assessment.txt')
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines) + '\n')
    print(f"\nAssessment report saved to: {report_file}")

    return summary


def plot_bspline_fit(fit_results, extension_key, fiber_idx, output_file=None):
    """
    Plot individual fiber B-spline fits with original data for detailed inspection.

    This function creates a comprehensive plot showing the original flat field data,
    B-spline fit, and residuals for a specific fiber. Negative predictions are
    highlighted in red for easy identification.

    Args:
        fit_results (dict): B-spline fit results dictionary from calculate_fits_all_extensions()
        extension_key (str): Extension identifier (e.g., 'red1A', 'green2B')
        fiber_idx (int): Fiber number to plot
        output_file (str, optional): Filename to save plot. If None, displays plot.
            Defaults to None.

    Returns:
        None
    """

    # Validate inputs
    if extension_key not in fit_results:
        print(f"ERROR: Extension '{extension_key}' not found in fit results")
        print(f"Available extensions: {list(fit_results.keys())}")
        return

    if fiber_idx not in fit_results[extension_key]:
        print(f"ERROR: Fiber {fiber_idx} not found in extension '{extension_key}'")
        print(f"Available fibers: {list(fit_results[extension_key].keys())}")
        return

    # Get fiber data
    fiber_data = fit_results[extension_key][fiber_idx]

    # Extract data arrays
    xshift_clean = np.array(fiber_data.get('xshift_clean', []))
    counts_clean = np.array(fiber_data.get('counts_clean', []))
    y_predicted = np.array(fiber_data.get('y_predicted', []))
    residuals = np.array(fiber_data.get('residuals', []))
    rms_residual = fiber_data.get('rms_residual', np.nan)
    relative_rms = fiber_data.get('relative_rms', np.nan)

    # Check for negative predictions
    negative_mask = y_predicted < 0
    negative_count = np.sum(negative_mask)
    min_prediction = np.min(y_predicted) if len(y_predicted) > 0 else 0

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Original data and B-spline fit
    ax1.scatter(xshift_clean, counts_clean, alpha=0.6, s=20, color='blue',
               label='Original Data')
    ax1.plot(xshift_clean, y_predicted, 'r-', linewidth=2, label='B-spline Fit')

    # Highlight negative predictions
    if negative_count > 0:
        negative_x = xshift_clean[negative_mask]
        negative_y = y_predicted[negative_mask]
        ax1.scatter(negative_x, negative_y, color='red', s=50, marker='x',
                   linewidth=3, label=f'Negative Predictions ({negative_count})')

    ax1.set_xlabel('X-shift')
    ax1.set_ylabel('Counts')
    ax1.set_title(f'B-spline Fit: {extension_key.upper()} Fiber {fiber_idx}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'RMS Residual: {rms_residual:.4f}\n'
    stats_text += f'Relative RMS: {relative_rms:.4f}\n'
    stats_text += f'Min Prediction: {min_prediction:.4f}\n'
    stats_text += f'Data Points: {len(xshift_clean)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Residuals
    ax2.scatter(xshift_clean, residuals, alpha=0.6, s=20, color='green')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('X-shift')
    ax2.set_ylabel('Residuals (Data - Fit)')
    ax2.set_title(f'Fit Residuals: {extension_key.upper()} Fiber {fiber_idx}')
    ax2.grid(True, alpha=0.3)

    # Add residual statistics
    if len(residuals) > 0:
        residual_std = np.std(residuals)
        residual_mean = np.mean(residuals)
        ax2.axhline(y=residual_mean + 2*residual_std, color='red', linestyle=':', alpha=0.7, label='+2σ')
        ax2.axhline(y=residual_mean - 2*residual_std, color='red', linestyle=':', alpha=0.7, label='-2σ')
        ax2.legend()

    plt.tight_layout()

    # Save or display plot
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        plt.close()
    else:
        plt.show()

    # Print summary information
    print(f"\nFiber Summary: {extension_key.upper()} Fiber {fiber_idx}")
    print(f"  Data points: {len(xshift_clean)}")
    print(f"  RMS residual: {rms_residual:.6f}")
    print(f"  Relative RMS: {relative_rms:.6f}")
    print(f"  Min prediction: {min_prediction:.6f}")
    if negative_count > 0:
        print(f"  ⚠️  WARNING: {negative_count} negative predictions found!")
        print(f"  Negative fraction: {negative_count/len(y_predicted)*100:.1f}%")
    else:
        print(f"  ✅ No negative predictions")

def plot_spline_fibre(pkl_file: str, extension: str, fiber_number: int) -> None:
    """
    Load B-spline fit results from a pickle file and plot a specific fiber.

    This function reads B-spline fit results from a specified pickle file,
    lists available extensions and fibers, and generates a plot for the
    specified extension and fiber number.

    Args:
        pkl_file (str): Path to the pickle file containing fit results.
        extension (str): Extension identifier (e.g., 'red1A', 'green2B').
        fiber_number (int): Fiber number to plot.

    Returns:
        None
    """
    import pickle
    import os

    print(f"Loading B-spline fit results from: {pkl_file}")

    if not os.path.exists(pkl_file):
        print(f"ERROR: File not found: {pkl_file}")
        return
# Load fit results
    with open(pkl_file, 'rb') as f:
        fit_results = pickle.load(f)
    # See what's available
    print("Available extensions:")
    for ext_key in fit_results.keys():
        fiber_count = len(fit_results[ext_key])
        fiber_range = f"{min(fit_results[ext_key].keys())}-{max(fit_results[ext_key].keys())}"
        print(f"  {ext_key}: {fiber_count} fibers (range: {fiber_range})")
    # Plot specific fiber (change these values)
    extension = 'green2B'  # Change this
    fiber_number = 0    # Change this
    plot_bspline_fit(fit_results, extension, fiber_number, f'{extension}_fiber_{fiber_number}.png')
    return


# Example usage functions and documentation
def check_reference_arc_wavelength_ranges(arc_file=None, verbose=True):
    """
    Check wavelength ranges in the LLAMAS reference arc calibration file.

    This function loads the reference arc file used for wavelength calibration
    and reports the wavelength coverage for each extension and channel.

    Args:
        arc_file (str, optional): Path to the reference arc pickle file.
            If None, uses the default LLAMAS_reference_arc.pkl from LUT_DIR.
            Defaults to None.
        verbose (bool, optional): If True, prints detailed output for each extension.
            If False, only prints channel summaries. Defaults to True.

    Returns:
        dict: Dictionary containing wavelength ranges organized by channel:
            {
                'extensions': [
                    {
                        'index': int,
                        'bench': str,
                        'side': str,
                        'channel': str,
                        'nfibers': int,
                        'wave_min': float,
                        'wave_max': float,
                        'has_wavelength_data': bool
                    },
                    ...
                ],
                'channels': {
                    'red': {'min': float, 'max': float},
                    'green': {'min': float, 'max': float},
                    'blue': {'min': float, 'max': float}
                }
            }

    Example:
        >>> from llamas_pyjamas.Utils.utils import check_reference_arc_wavelength_ranges
        >>> ranges = check_reference_arc_wavelength_ranges()
        >>> print(f"Red channel: {ranges['channels']['red']['min']:.1f} - {ranges['channels']['red']['max']:.1f} Å")
    """
    import sys
    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

    # Use default file if none provided
    if arc_file is None:
        arc_file = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')

    # Check if file exists
    if not os.path.exists(arc_file):
        print(f"ERROR: Reference arc file not found: {arc_file}")
        return None

    # Load the reference arc
    try:
        arc_data = ExtractLlamas.loadExtraction(arc_file)
    except Exception as e:
        print(f"ERROR: Failed to load reference arc file: {e}")
        return None

    extractions = arc_data['extractions']
    metadata = arc_data['metadata']

    # Initialize results
    results = {
        'extensions': [],
        'channels': {}
    }

    if verbose:
        print("=" * 60)
        print("WAVELENGTH RANGES BY EXTENSION")
        print("=" * 60)

    # Process each extension
    for i in range(len(extractions)):
        ext = extractions[i]
        meta = metadata[i]

        channel = meta['channel']
        bench = meta['bench']
        side = meta['side']
        nfibers = meta['nfibers']

        ext_info = {
            'index': i,
            'bench': str(bench),
            'side': side,
            'channel': channel,
            'nfibers': nfibers,
            'has_wavelength_data': False
        }

        if hasattr(ext, 'wave') and ext.wave is not None and np.any(ext.wave > 0):
            # Get min/max wavelengths (excluding zeros)
            wave_min = np.min(ext.wave[ext.wave > 0])
            wave_max = np.max(ext.wave)

            ext_info['wave_min'] = wave_min
            ext_info['wave_max'] = wave_max
            ext_info['has_wavelength_data'] = True

            if verbose:
                print(f"Extension {i:2d}: {bench}{side} {channel:6s} | "
                      f"{wave_min:7.2f} - {wave_max:7.2f} Å | "
                      f"{nfibers} fibers")

            # Update channel ranges
            if channel not in results['channels']:
                results['channels'][channel] = {'min': wave_min, 'max': wave_max}
            else:
                results['channels'][channel]['min'] = min(results['channels'][channel]['min'], wave_min)
                results['channels'][channel]['max'] = max(results['channels'][channel]['max'], wave_max)
        else:
            if verbose:
                print(f"Extension {i:2d}: {bench}{side} {channel:6s} | NO WAVELENGTH DATA")

        results['extensions'].append(ext_info)

    # Print channel summary
    if verbose:
        print("=" * 60)
    print("\nWAVELENGTH COVERAGE BY CHANNEL:")
    print("-" * 60)

    for channel in ['red', 'green', 'blue']:
        if channel in results['channels']:
            print(f"{channel.upper():6s}: {results['channels'][channel]['min']:7.2f} - "
                  f"{results['channels'][channel]['max']:7.2f} Å")
        else:
            print(f"{channel.upper():6s}: No wavelength data found")

    if verbose:
        print("=" * 60)

    return results


def check_extraction_wavelength_ranges(extraction_file, reference_arc_file=None, verbose=True):
    """
    Check wavelength ranges in an extraction file and compare to reference arc.

    This function loads an extraction pickle file (science or arc data) and reports
    the wavelength coverage for each extension and channel. It can optionally compare
    against the reference arc calibration to identify any extrapolation.

    Args:
        extraction_file (str): Path to the extraction pickle file to check.
        reference_arc_file (str, optional): Path to the reference arc pickle file.
            If None, uses the default LLAMAS_reference_arc.pkl from LUT_DIR.
            Defaults to None.
        verbose (bool, optional): If True, prints detailed output for each extension.
            If False, only prints channel summaries. Defaults to True.

    Returns:
        dict: Dictionary containing wavelength ranges and comparison results:
            {
                'extensions': [...],  # Same format as check_reference_arc_wavelength_ranges
                'channels': {...},
                'comparison': {
                    'red': {'below': float, 'above': float, 'within': bool},
                    'green': {...},
                    'blue': {...}
                }
            }

    Example:
        >>> from llamas_pyjamas.Utils.utils import check_extraction_wavelength_ranges
        >>> results = check_extraction_wavelength_ranges('science_extract.pkl')
        >>> if results['comparison']['red']['within']:
        >>>     print("Red channel is within calibration range")
    """
    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

    # Load the extraction file
    if not os.path.exists(extraction_file):
        print(f"ERROR: Extraction file not found: {extraction_file}")
        return None

    try:
        data = ExtractLlamas.loadExtraction(extraction_file)
    except Exception as e:
        print(f"ERROR: Failed to load extraction file: {e}")
        return None

    extractions = data['extractions']
    metadata = data['metadata']

    # Load reference arc if comparison requested
    arc_ranges = None
    if reference_arc_file or reference_arc_file is None:
        if reference_arc_file is None:
            reference_arc_file = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')

        if os.path.exists(reference_arc_file):
            try:
                arc_data = ExtractLlamas.loadExtraction(reference_arc_file)
                arc_extractions = arc_data['extractions']
                arc_metadata = arc_data['metadata']

                # Build arc ranges
                arc_ranges = {}
                for channel in ['red', 'green', 'blue']:
                    channel_exts = [i for i, m in enumerate(arc_metadata) if m['channel'] == channel]
                    if channel_exts:
                        all_mins = []
                        all_maxs = []
                        for i in channel_exts:
                            ext = arc_extractions[i]
                            if hasattr(ext, 'wave') and ext.wave is not None and np.any(ext.wave > 0):
                                all_mins.append(np.min(ext.wave[ext.wave > 0]))
                                all_maxs.append(np.max(ext.wave))
                        if all_mins and all_maxs:
                            arc_ranges[channel] = (min(all_mins), max(all_maxs))
            except Exception as e:
                print(f"Warning: Could not load reference arc: {e}")

    # Initialize results
    results = {
        'extensions': [],
        'channels': {},
        'comparison': {}
    }

    if verbose:
        print("=" * 80)
        print("EXTRACTION WAVELENGTH RANGES")
        print("=" * 80)

    # Process each extension
    channels_data = {'red': [], 'green': [], 'blue': []}

    for i in range(len(extractions)):
        ext = extractions[i]
        meta = metadata[i]

        channel = meta['channel']
        bench = meta['bench']
        side = meta['side']
        nfibers = meta['nfibers']

        ext_info = {
            'index': i,
            'bench': str(bench),
            'side': side,
            'channel': channel,
            'nfibers': nfibers,
            'has_wavelength_data': False
        }

        if hasattr(ext, 'wave') and ext.wave is not None and np.any(ext.wave > 0):
            wave_min = np.min(ext.wave[ext.wave > 0])
            wave_max = np.max(ext.wave)

            ext_info['wave_min'] = wave_min
            ext_info['wave_max'] = wave_max
            ext_info['has_wavelength_data'] = True

            channels_data[channel].append({'min': wave_min, 'max': wave_max})

            if verbose:
                status = ""
                if arc_ranges and channel in arc_ranges:
                    arc_min, arc_max = arc_ranges[channel]
                    if wave_min < arc_min or wave_max > arc_max:
                        status = " ⚠️  OUTSIDE ARC RANGE"

                print(f"Extension {i:2d}: {bench}{side} {channel:6s} | "
                      f"{wave_min:7.2f} - {wave_max:7.2f} Å | "
                      f"{nfibers} fibers{status}")

            # Update channel ranges
            if channel not in results['channels']:
                results['channels'][channel] = {'min': wave_min, 'max': wave_max}
            else:
                results['channels'][channel]['min'] = min(results['channels'][channel]['min'], wave_min)
                results['channels'][channel]['max'] = max(results['channels'][channel]['max'], wave_max)
        else:
            if verbose:
                print(f"Extension {i:2d}: {bench}{side} {channel:6s} | NO WAVELENGTH DATA")

        results['extensions'].append(ext_info)

    # Compare to arc ranges
    if arc_ranges:
        if verbose:
            print("=" * 80)
            print("\nCOMPARISON TO REFERENCE ARC:")
            print("-" * 80)

        for channel in ['red', 'green', 'blue']:
            if channel in results['channels'] and channel in arc_ranges:
                ext_min = results['channels'][channel]['min']
                ext_max = results['channels'][channel]['max']
                arc_min, arc_max = arc_ranges[channel]

                below = max(0, arc_min - ext_min)
                above = max(0, ext_max - arc_max)
                within = (ext_min >= arc_min) and (ext_max <= arc_max)

                results['comparison'][channel] = {
                    'below': below,
                    'above': above,
                    'within': within
                }

                if verbose:
                    print(f"\n{channel.upper():6s}:")
                    print(f"  Extraction range: {ext_min:.4f} - {ext_max:.4f} Å")
                    print(f"  Arc range:        {arc_min:.2f} - {arc_max:.2f} Å")

                    if within:
                        print(f"  Status: ✅ WITHIN arc calibration range")
                    else:
                        if below > 0:
                            print(f"  Status: ⚠️  BELOW arc range by {below:.4f} Å")
                        if above > 0:
                            print(f"  Status: ⚠️  ABOVE arc range by {above:.4f} Å")

    if verbose:
        print("=" * 80)

    return results


def is_wavelength_solution_useable(arc_dict):
    """Check if an arc calibration dictionary has useable wavelength solutions.

    Validates that wavelength solutions exist and are properly calibrated
    for use in wavelength transfer operations. This function samples a few
    fibers from each extension to verify the wavelength data is valid.

    Parameters
    ----------
    arc_dict : dict
        Dictionary containing 'extractions' (list of ExtractLlamas objects)
        and 'metadata' for each extension. Typically loaded via
        ExtractLlamas.loadExtraction().

    Returns
    -------
    bool
        True if wavelength solutions are useable, False otherwise.

    Notes
    -----
    This function uses validate_wavelength_solution() from Arc/arcValidation.py
    to check wavelength data quality including:
    - Non-zero wavelength values
    - No NaN/Inf values
    - Monotonically increasing wavelengths
    - Reasonable wavelength ranges for each channel
    """
    from llamas_pyjamas.Arc.arcValidation import validate_wavelength_solution

    extractions = arc_dict.get('extractions', [])
    metadata = arc_dict.get('metadata', [])

    if not extractions:
        logger.warning("No extractions found in arc_dict")
        return False

    for idx, extraction in enumerate(extractions):
        meta = metadata[idx] if idx < len(metadata) else {}
        channel = meta.get('channel', 'red')

        # Check if wave attribute exists
        if not hasattr(extraction, 'wave') or extraction.wave is None:
            logger.warning(f"Extension {idx} has no wavelength data")
            return False

        # Check a sample of fibers (first, middle, last)
        n_fibers = extraction.wave.shape[0] if extraction.wave.ndim > 1 else 1
        if n_fibers == 0:
            logger.warning(f"Extension {idx} has no fibers")
            return False

        sample_fibers = [0, n_fibers // 2, n_fibers - 1]

        for fiber_idx in sample_fibers:
            if fiber_idx >= n_fibers:
                continue
            result = validate_wavelength_solution(extraction, channel, fiber_idx)
            if not result['valid']:
                logger.warning(f"Extension {idx} fiber {fiber_idx} invalid: {result['errors']}")
                return False

    return True


def _bspline_diagnostic_usage_examples():
    """
    Example usage for B-spline diagnostic functions.

    This function is not meant to be called directly, but serves as documentation
    for how to use the check_bspline_negative_values and plot_bspline_fit functions.
    """

    # Example 1: Check for negative values after flat field processing
    """
    import pickle
    import glob
    from llamas_pyjamas.Utils.utils import check_bspline_negative_values, plot_bspline_fit

    # Load fit results from flat field processing
    with open('output/combined_flat_extractions_fits.pkl', 'rb') as f:
        fit_results = pickle.load(f)

    # Find pixel map files
    pixel_map_files = glob.glob('output/flat_pixel_map_*.fits')

    # Run negative value check
    problematic_fibers = check_bspline_negative_values(
        fit_results=fit_results,
        pixel_map_files=pixel_map_files,
        output_dir='diagnostics/'
    )

    # Plot specific problematic fibers
    for (ext_key, fiber_idx), info in problematic_fibers.items():
        if info['negative_fraction'] > 10:  # Only plot fibers with >10% negative values
            print(f"Plotting problematic fiber: {ext_key} fiber {fiber_idx}")
            plot_bspline_fit(fit_results, ext_key, fiber_idx,
                            output_file=f'diagnostics/fiber_{ext_key}_{fiber_idx}.png')
    """

    # Example 2: Plot any fiber for inspection
    """
    # Plot a specific fiber for inspection (doesn't need to be problematic)
    plot_bspline_fit(fit_results, 'red1A', 150, 'inspection_red1A_150.png')

    # Plot without saving (display interactively)
    plot_bspline_fit(fit_results, 'green2B', 200)
    """

    # Example 3: Command line usage
    """
    # After running flat field processing, check from command line:
    cd /path/to/llamas_pyjamas
    python -c "
    from llamas_pyjamas.Utils.utils import check_bspline_negative_values
    import pickle
    import glob

    with open('output/combined_flat_extractions_fits.pkl', 'rb') as f:
        fit_results = pickle.load(f)

    pixel_maps = glob.glob('output/flat_pixel_map_*.fits')

    problems = check_bspline_negative_values(fit_results, pixel_maps, 'diagnostics/')
    print(f'Found {len(problems)} problematic fibers')
    "
    """
    pass


# =========================================================================
# Pixel flat-field diagnostic functions
# =========================================================================

def plot_flat_correction_ratio(uncorrected_pkl, corrected_pkl, output_file=None,
                               n_fibers=5):
    """Compare uncorrected vs flat-corrected extractions fiber by fiber.

    For each of the 24 extensions the ratio ``corrected / uncorrected`` is
    plotted for *n_fibers* representative fibers.  A perfect correction
    produces a smooth curve near 1.0; high-frequency residual structure
    indicates under- or over-correction.

    Parameters
    ----------
    uncorrected_pkl : str
        Path to batch extraction pickle **before** flat correction.
    corrected_pkl : str
        Path to batch extraction pickle **after** flat correction.
    output_file : str, optional
        Path to save the figure.  If ``None``, displays interactively.
    n_fibers : int
        Number of representative fibers per extension (evenly spaced).

    Returns
    -------
    dict
        ``{ext_label: {'median_ratio': float, 'rms_residual': float}}``
    """
    import matplotlib
    if output_file:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    uncorr_data = _load_extraction_pkl(uncorrected_pkl)
    corr_data = _load_extraction_pkl(corrected_pkl)

    uncorr_exts, uncorr_meta = uncorr_data['extractions'], uncorr_data['metadata']
    corr_exts, corr_meta = corr_data['extractions'], corr_data['metadata']

    n_ext = min(len(uncorr_exts), len(corr_exts))
    color_map = {'red': '#d62728', 'green': '#2ca02c', 'blue': '#1f77b4'}

    ncols = 6
    nrows = int(np.ceil(n_ext / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_2d(axes).ravel()

    stats = {}
    for idx in range(n_ext):
        ax = axes[idx]
        u_ext = uncorr_exts[idx]
        c_ext = corr_exts[idx]
        meta = uncorr_meta[idx]
        channel = meta['channel']
        label = f"{channel[:3].upper()}{meta['bench']}{meta['side']}"

        nfib = min(u_ext.counts.shape[0], c_ext.counts.shape[0])
        fib_idxs = np.linspace(0, nfib - 1, n_fibers, dtype=int)

        all_ratios = []
        for fib in fib_idxs:
            u_counts = u_ext.counts[fib, :]
            c_counts = c_ext.counts[fib, :]
            valid = u_counts > 0
            ratio = np.ones_like(u_counts, dtype=float)
            ratio[valid] = c_counts[valid] / u_counts[valid]
            ax.plot(ratio, lw=0.5, alpha=0.7,
                    color=color_map.get(channel, 'gray'))
            all_ratios.append(ratio[valid])

        ax.axhline(1.0, color='k', ls='--', lw=0.8, alpha=0.5)
        ax.set_ylim(0.85, 1.15)
        ax.set_title(label, fontsize=8)

        combined = np.concatenate(all_ratios) if all_ratios else np.array([1.0])
        stats[label] = {
            'median_ratio': float(np.median(combined)),
            'rms_residual': float(np.std(combined)),
        }

    for ax in axes[n_ext:]:
        ax.set_visible(False)

    fig.suptitle('Flat Correction Ratio (corrected / uncorrected)', fontsize=12)
    fig.supxlabel('Spectral pixel')
    fig.supylabel('Ratio')
    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_file}")
    else:
        plt.show()

    return stats


def plot_channel_flat_summary(extraction_pkl, output_file=None, filter_size=12):
    """Per-channel histogram of fiber RMS after dividing out a smooth model.

    For every fiber in every extension the extracted counts are divided by
    a median+Gaussian smooth model to isolate high-frequency residuals.
    The RMS of those residuals is histogrammed per color channel (red |
    green | blue).  If one channel has systematically higher RMS, the
    signal threshold or filter size may need tuning.

    Parameters
    ----------
    extraction_pkl : str
        Path to a batch extraction pickle (e.g. the corrected flat
        extraction or a corrected science extraction).
    output_file : str, optional
        Save path for the figure.
    filter_size : int
        Median filter size used to build the smooth model.

    Returns
    -------
    dict
        ``{'red': {'median_rms': float, 'fibers': int}, ...}``
    """
    from scipy.ndimage import median_filter as _mf, gaussian_filter as _gf
    import matplotlib
    if output_file:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = _load_extraction_pkl(extraction_pkl)
    extractions, metadata = data['extractions'], data['metadata']

    channel_rms = {'red': [], 'green': [], 'blue': []}
    color_hex = {'red': '#d62728', 'green': '#2ca02c', 'blue': '#1f77b4'}

    for ext, meta in zip(extractions, metadata):
        channel = meta['channel'].lower()
        if channel not in channel_rms:
            continue
        for fib_idx in range(ext.counts.shape[0]):
            counts = ext.counts[fib_idx, :].astype(np.float64)
            smooth = _gf(_mf(counts, size=filter_size), sigma=2.0)
            valid = smooth > 0
            if np.sum(valid) < 10:
                continue
            ratio = np.ones_like(counts)
            ratio[valid] = counts[valid] / smooth[valid]
            rms = float(np.std(ratio[valid]))
            channel_rms[channel].append(rms)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    stats = {}
    for ax, (ch, rms_list) in zip(axes, channel_rms.items()):
        arr = np.array(rms_list) if rms_list else np.array([0.0])
        ax.hist(arr * 100, bins=60, range=(0, 6), color=color_hex[ch],
                alpha=0.75, edgecolor='none')
        med = float(np.median(arr)) * 100
        ax.axvline(med, color='k', ls='--', lw=1.5,
                   label=f'median = {med:.2f}%')
        ax.set_title(f'{ch.capitalize()} ({len(rms_list)} fibers)', fontsize=10)
        ax.set_xlabel('Fiber RMS (%)')
        ax.legend(fontsize=8)
        stats[ch] = {'median_rms': med, 'fibers': len(rms_list)}

    axes[0].set_ylabel('Fiber count')
    fig.suptitle('Per-Channel Flat Residual RMS', fontsize=12)
    plt.tight_layout()

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_file}")
    else:
        plt.show()

    return stats


def plot_fiber_consistency(extraction_pkl, output_file=None):
    """Fiber-to-fiber scatter as a function of wavelength pixel.

    For each extension the standard deviation across fibers at each
    spectral pixel is computed.  After good flat correction the
    fiber-to-fiber scatter should be dominated by photon noise, not
    systematic structure.  Lines are colored by channel so that
    red/green/blue can be compared directly.

    Parameters
    ----------
    extraction_pkl : str
        Path to a batch extraction pickle (e.g. corrected flat lamp
        extraction).
    output_file : str, optional
        Save path for the figure.

    Returns
    -------
    dict
        ``{ext_label: {'mean_scatter': float, 'max_scatter': float}}``
    """
    import matplotlib
    if output_file:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = _load_extraction_pkl(extraction_pkl)
    extractions, metadata = data['extractions'], data['metadata']
    color_hex = {'red': '#d62728', 'green': '#2ca02c', 'blue': '#1f77b4'}

    fig, ax = plt.subplots(figsize=(14, 5))
    stats = {}

    for ext, meta in zip(extractions, metadata):
        channel = meta['channel'].lower()
        label = f"{channel[:3].upper()}{meta['bench']}{meta['side']}"
        counts = ext.counts.astype(np.float64)

        # Normalize each fiber by its own median to remove throughput
        fiber_medians = np.nanmedian(counts, axis=1, keepdims=True)
        fiber_medians[fiber_medians == 0] = 1.0
        normed = counts / fiber_medians

        # Scatter across fibers at each spectral pixel
        scatter = np.nanstd(normed, axis=0)

        ax.plot(scatter * 100, lw=0.6, alpha=0.5,
                color=color_hex.get(channel, 'gray'), label=label)

        finite = scatter[np.isfinite(scatter)]
        stats[label] = {
            'mean_scatter': float(np.mean(finite)) * 100 if len(finite) else 0.0,
            'max_scatter': float(np.max(finite)) * 100 if len(finite) else 0.0,
        }

    ax.set_xlabel('Spectral pixel')
    ax.set_ylabel('Fiber-to-fiber scatter (%)')
    ax.set_title('Fiber-to-Fiber Consistency After Flat Correction')
    ax.set_ylim(0, 15)

    # De-duplicate legend entries by color
    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        ch = l[:3]
        if ch not in seen:
            seen[ch] = h
    ax.legend(seen.values(), seen.keys(), fontsize=8, loc='upper right')

    plt.tight_layout()
    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_file}")
    else:
        plt.show()

    return stats


def plot_flat_residual_map(pixel_map_file, output_file=None):
    """4x6 grid of 2D pixel map images for all 24 extensions.

    Each panel shows the sensitivity map with a diverging colormap
    centred at 1.0.  Residual structure along traces indicates the flat
    is not fully removing pixel QE variations; structure perpendicular to
    traces indicates trace-edge artifacts.

    Parameters
    ----------
    pixel_map_file : str
        Path to ``pixel_maps.fits`` (24-extension MEF).
    output_file : str, optional
        Save path for the figure.

    Returns
    -------
    dict
        ``{ext_name: {'median': float, 'std': float, 'frac_outside': float}}``
    """
    import matplotlib
    if output_file:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not os.path.exists(pixel_map_file):
        raise FileNotFoundError(f"Pixel map file not found: {pixel_map_file}")

    with fits.open(pixel_map_file) as hdul:
        image_exts = [(h.header.get('EXTNAME', f'EXT{i}'),
                       h.data.astype(np.float64))
                      for i, h in enumerate(hdul)
                      if h.data is not None and h.data.ndim == 2]

    n_ext = len(image_exts)
    ncols = 6
    nrows = max(1, int(np.ceil(n_ext / ncols)))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_2d(axes).ravel()

    stats = {}
    for i, (ext_name, data) in enumerate(image_exts):
        ax = axes[i]

        im = ax.imshow(data, aspect='auto', origin='lower',
                       vmin=0.90, vmax=1.10, cmap='coolwarm')
        ax.set_title(ext_name, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

        active = data[(data != 1.0) & np.isfinite(data)]
        med = float(np.median(active)) if len(active) else 1.0
        std = float(np.std(active)) if len(active) else 0.0
        frac = float(np.mean((active < 0.9) | (active > 1.1))) if len(active) else 0.0
        stats[ext_name] = {'median': med, 'std': std, 'frac_outside': frac}

    for ax in axes[n_ext:]:
        ax.set_visible(False)

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Sensitivity')
    fig.suptitle('Pixel-to-Pixel Sensitivity Maps', fontsize=13, y=1.01)
    plt.tight_layout(rect=[0, 0, 0.91, 1])

    if output_file:
        fig.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_file}")
    else:
        plt.show()

    # Print summary table
    print(f"\n{'Extension':<12} {'Median':>8} {'Std (%)':>8} {'%Outside':>9}")
    print("-" * 40)
    for name, s in stats.items():
        flag = ' <-- CHECK' if s['std'] > 0.02 or s['frac_outside'] > 0.01 else ''
        print(f"{name:<12} {s['median']:>8.4f} {s['std']*100:>7.3f}% "
              f"{s['frac_outside']*100:>8.2f}%{flag}")

    return stats


def _load_extraction_pkl(pkl_path):
    """Load a batch extraction pickle, returning the dict with
    'extractions' and 'metadata' keys."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Extraction file not found: {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = cloudpickle.load(f)
    return data
