from   astropy.io import fits
import scipy
import numpy as np
import os
from datetime import datetime
from   matplotlib import pyplot as plt
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl
import pickle, cloudpickle
import logging
import argparse, glob
import ray, multiprocessing, psutil
import traceback

import pkg_resources
from pathlib import Path

from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR, BIAS_DIR
from llamas_pyjamas.Trace.traceLlamasMaster import _grab_bias_hdu, TraceRay
import llamas_pyjamas.Trace.traceLlamasMaster as traceLlamasMaster
import sys
sys.modules['traceLlamasMaster'] = traceLlamasMaster

from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions, load_extractions
from llamas_pyjamas.Image.WhiteLightModule import WhiteLight, WhiteLightFits, WhiteLightQuickLook
import time

from llamas_pyjamas.File.llamasIO import process_fits_by_color

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, log_filename=f'extractLlamas_{timestamp}.log')



####################################################################################

def ExtractLlamasCube(infits, tracefits, optimal=True):

    assert infits.endswith('.fits'), 'File must be a .fits file'  
    # hdu = fits.open(infits)
    hdu = process_fits_by_color(infits)

    # Find the trace files
    basefile = os.path.basename(tracefits).split('.fits')[0]
    trace_files = glob.glob(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))
    extraction_file = os.path.basename(infits).split('mef.fits')[0] + 'extract.pkl'

    if len(trace_files) == 0:
        logger.error("No trace files found for the indicated file root!")
        return None
    
    hdu_trace_pairs = match_hdu_to_traces(hdu, trace_files)
    logger.debug(f"HDU trace pairs: {hdu_trace_pairs}")

    extraction_list = []

    logger.info(f"Saving extractions to {extraction_file}")

    counter = 1
    for hdu_index, file in hdu_trace_pairs:

        logger.debug(f"Extracting extension number {counter} of 24")
        hdr = hdu[hdu_index].header 
        bias = np.nanmedian(hdu[hdu_index].data.astype(float))  
        
        try:
            with open(file, mode='rb') as f:
                tracer = pickle.load(f)
    
            extraction = ExtractLlamas(tracer, hdu[hdu_index].data.astype(float)-bias, hdu[hdu_index].header)
            extraction_list.append(extraction)
            
        except Exception as e:
            logger.error(f"Error extracting trace from {file}")
            logger.error(traceback.format_exc())
        counter += 1
        
    logger.debug(f'Extraction list = {extraction_list}')        
    filename = save_extractions(extraction_list, savefile=extraction_file)
    logger.info(f'extraction saved filename = {filename}')

    return None


def get_trace_file(channel, bench, side, trace_dir):
    """
    Find trace file for specific camera configuration.

    Args:
        channel: Color channel (red/green/blue)
        bench: Bench number
        side: Side letter (A/B)
        trace_dir: Directory containing trace files

    Returns:
        str: Path to trace file

    Raises:
        FileNotFoundError: If trace file not found
    """
    # Standard trace file naming: LLAMAS_master_{channel}_{bench}_{side}_traces.pkl
    trace_filename = f'LLAMAS_master_{channel.lower()}_{bench}_{side}_traces.pkl'
    trace_path = os.path.join(trace_dir, trace_filename)

    if os.path.exists(trace_path):
        return trace_path

    # Try alternate naming without "master"
    alt_filename = f'LLAMAS_{channel.lower()}_{bench}_{side}_traces.pkl'
    alt_path = os.path.join(trace_dir, alt_filename)

    if os.path.exists(alt_path):
        return alt_path

    raise FileNotFoundError(
        f"Trace file not found for {channel}{bench}{side} in {trace_dir}\n"
        f"  Tried: {trace_filename}, {alt_filename}"
    )


def match_hdu_to_traces(hdu_list, trace_files, start_idx=1):
    """Match HDU extensions to their corresponding trace files"""
    matches = []

    # Skip primary HDU (index 0)
    #### need to be super careful with this starting index
    for idx in range(start_idx, len(hdu_list)):

        header = hdu_list[idx].header

        # Get color and benchside from header
        if 'COLOR' in header:
            color = header['COLOR'].lower()
            bench = header['BENCH']
            side = header['SIDE']
        else:
            camname = header['CAM_NAME']
            color = camname.split('_')[1].lower()
            bench = camname.split('_')[0][0]
            side = camname.split('_')[0][1]

        benchside = f"{bench}{side}"
        pattern = f"{color}_{bench}_{side}_traces"

        # Find matching trace file
        matching_trace = next(
            (tf for tf in trace_files
             if pattern in os.path.basename(tf)),
            None
        )
        print(f'HDU {idx}: Looking for pattern "{pattern}" -> {os.path.basename(matching_trace) if matching_trace else "NOT FOUND"}')
        if matching_trace:
            matches.append((idx, matching_trace))
        else:
            logger.warning(f"No matching trace found for HDU {idx}: {color} {benchside}, pattern: {pattern}")
            print(f"  Available trace files: {[os.path.basename(tf) for tf in trace_files]}")

    return matches

# Define a Ray remote function for processing a single trace extraction.
@ray.remote

def process_trace(hdu_data, header, trace_file, method='optimal', use_bias=None):
    """
    Process a single HDU: subtract bias, load the trace from a trace file, and create an ExtractLlamas object.
    Returns the extraction object or None if there is an error.
    """
    try:
        # Compute the bias from the current extension data.
        if 'COLOR' in header:
            color = header['COLOR'].lower()
            bench = header['BENCH']
            side = header['SIDE']
        else:
            camname = header['CAM_NAME']
            color = camname.split('_')[1].lower()
            bench = camname.split('_')[0][0]
            side = camname.split('_')[0][1]

        if use_bias is None:    
            bias_file = os.path.join(BIAS_DIR, 'combined_bias.fits')

        elif use_bias is str:
            if os.path.isfile(use_bias):
                bias_file = use_bias
        else:
            bias_file = os.path.join(BIAS_DIR, 'combined_bias.fits')


        print(f'Bias file: {bias_file}')
        #### fix the directory here!
        bias = _grab_bias_hdu(bench=bench, side=side, color=color, dir=bias_file)
        
        bias_data = bias.data #np.median(bias.data[20:50])
            
        # Load the trace object from the pickle file using cloudpickle for better compatibility
        with open(trace_file, mode='rb') as f:
            tracer = cloudpickle.load(f)
        # Create an ExtractLlamas object; note the subtraction of the bias.
        if (method == 'optimal'):
            extraction = ExtractLlamas(tracer, hdu_data.astype(float)-bias_data, header, optimal=True)
        elif (method == 'boxcar'):
            extraction = ExtractLlamas(tracer, hdu_data.astype(float)-bias_data, header, optimal=False)
        return extraction
    except Exception as e:
        logger.error(f"Error extracting trace from {trace_file}")
        logger.error(traceback.format_exc())
        return None


def make_writable(extraction_obj):
    """Convert a Ray-returned extraction object to a writable version."""
    import copy
    
    # First approach: Fix class references for TraceRay objects
    try:
        if hasattr(extraction_obj, 'tracer') and hasattr(extraction_obj.tracer, '__class__'):
            tracer_class = extraction_obj.tracer.__class__
            if tracer_class.__name__ == 'TraceRay' and tracer_class.__module__ == 'traceLlamasMaster':
                # Replace the tracer with a corrected class reference
                correct_class = sys.modules['traceLlamasMaster'].TraceRay
                if tracer_class is not correct_class:
                    # Create a new object with the correct class
                    extraction_obj.tracer.__class__ = correct_class
    except Exception as e:
        logger.warning(f"Could not fix tracer class reference: {e}")
    
    # Second approach: Deep copy
    try:
        return copy.deepcopy(extraction_obj)
    except:
        pass
    
    # Third approach: Cloudpickle and unpickle for better compatibility
    try:
        # This forces a complete serialization and deserialization
        pickled = cloudpickle.dumps(extraction_obj)
        return cloudpickle.loads(pickled)
    except:
        pass
    
    # Fourth approach: If the object has a to_dict method, use it
    if hasattr(extraction_obj, 'to_dict') and callable(extraction_obj.to_dict):
        try:
            obj_dict = extraction_obj.to_dict()
            # Assuming there's a from_dict or similar constructor
            if hasattr(type(extraction_obj), 'from_dict') and callable(type(extraction_obj).from_dict):
                return type(extraction_obj).from_dict(obj_dict)
        except:
            pass
    
    # If all else fails, return the original object and log a warning
    logger.warning(f"Could not make object of type {type(extraction_obj)} writable")
    return extraction_obj

def is_placeholder_camera(data):
    """
    Check if HDU data represents a non-functional camera (filled with 1.0).
    
    Args:
        data: numpy array of HDU data
        
    Returns:
        bool: True if this is a placeholder camera
    """
    if data is None:
        return True
    
    # Check if all values are 1.0 (or very close due to floating point)
    return np.allclose(data, 1.0, rtol=1e-5)

def compute_detector_background(data, rows=(30, 50)):
    """
    Compute median background from specified detector rows.
    
    Args:
        data: numpy array of detector data
        rows: tuple of (start_row, end_row) for background region
        
    Returns:
        float: median background value
    """
    upper_det = data[rows[0]:rows[1], :]
    upper_background_value = np.median(upper_det)
    return upper_background_value

def normalize_detector_backgrounds(hdu_data_list, rows=(30, 50)):
    """
    Normalize detector backgrounds to a common reference level.
    Instead of subtracting to zero, we subtract the difference from the minimum
    background level to preserve absolute counts.
    
    Args:
        hdu_data_list: list of (hdu_index, data_array) tuples
        rows: tuple of (start_row, end_row) for background region
        
    Returns:
        dict: {hdu_index: offset_to_subtract}
    """
    backgrounds = {}
    
    # Compute background for each detector
    for hdu_index, data in hdu_data_list:
        if not is_placeholder_camera(data):
            bg = compute_detector_background(data, rows)
            backgrounds[hdu_index] = bg
    
    if not backgrounds:
        return {}
    
    # Find minimum background (our reference level)
    min_background = min(backgrounds.values())
    
    # Calculate offsets: subtract only the DIFFERENCE from minimum
    offsets = {}
    for hdu_index, bg in backgrounds.items():
        offsets[hdu_index] = bg - min_background
    
    return offsets

##Main function currently used by the Quicklook for full extraction


def GUI_extract(file: fits.BinTableHDU, flatfiles: str = None, output_dir: str = None, use_bias: str = None, method='optimal', trace_dir=None, mastercalib_trace_dir=None) -> None:

    """
    Extracts data from a FITS file using calibration files and saves the extracted data.

    Supports hybrid trace selection: uses user traces for real camera data and
    mastercalib traces for placeholder extensions (missing cameras).

    Parameters:
    file (str): Path to the FITS file to be processed. Must have a .fits extension.
    flatfiles (str, optional): Path to the flat files for generating new traces. Defaults to None.
    output_dir (str, optional): Output directory for extraction files. Defaults to None.
    use_bias (str, optional): Path to bias file for calibration. Defaults to None.
    method (str, optional): Extraction method ('optimal' or 'boxcar'). Defaults to 'optimal'.
    trace_dir (str, optional): User trace directory for real extensions. Defaults to None.
    mastercalib_trace_dir (str, optional): Mastercalib trace directory for placeholder extensions.
                                           Defaults to CALIB_DIR if None.

    Returns:
    Tuple[str, int]: (extraction_file_path, number_of_placeholder_extensions)
    """
    start_time = time.perf_counter()  # Start timer
    
    try:
        print(f'file is {file}')
        assert file.endswith('.fits'), 'File must be a .fits file'
        
        master_pkls = glob.glob(os.path.join(CALIB_DIR, '*.pkl'))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if not master_pkls:
            raise ValueError("No master calibration files found in CALIB_DIR")
        
        #getting package sources for Ray
        # Get absolute path to llamas_pyjamas package
        package_path = pkg_resources.resource_filename('llamas_pyjamas', '')
        package_root = os.path.dirname(package_path)

        # Configure Ray runtime environment
        runtime_env = {
            "py_modules": [package_root],
            "env_vars": {"PYTHONPATH": f"{package_root}:{os.environ.get('PYTHONPATH', '')}"},
            "excludes": [
                str(Path(DATA_DIR) / "**"),  # Exclude DATA_DIR and all subdirectories
                "**/*.fits",                 # Exclude all FITS files anywhere
                "**/*.pkl",                  # Exclude all pickle files anywhere
                "**/.git/**",               # Exclude git directory
                "**/*.zip/**",
                "**/*.tar.gz/**",
                "**/mastercalib*/**",
                "**/*.zip",
                
            ]
        }

        # Initialize Ray
        num_cpus = int(os.environ.get('LLAMAS_RAY_CPUS', 8))
        ray.shutdown()
        ray.init(num_cpus=num_cpus, runtime_env=runtime_env)

        # Import placeholder detection utilities
        from llamas_pyjamas.DataModel.validate import get_placeholder_extension_indices

        #opening the fitsfile
        hdu = process_fits_by_color(file)

        primary_hdr = hdu[0].header

        extraction_file = os.path.basename(file).split('mef.fits')[0] + 'extract.pkl'

        #Defining the base filename
        #basefile = os.path.basename(file).split('.fits')[0]
        basefile = os.path.basename(file).split('.fits')[0]
        masterfile = 'LLAMAS_master'
        if use_bias is not None:
            if os.path.isfile(use_bias):
                masterbiasfile = use_bias
            else:
                raise ValueError(f"Bias file {use_bias} does not exist.")
        else:
            masterbiasfile = os.path.join(BIAS_DIR, 'combined_bias.fits')

        #Debug statements
        print(f'basefile = {basefile}')
        print(f'masterfile = {masterfile}')
        print(f'Bias file is {masterbiasfile}')

        # Default mastercalib location
        if mastercalib_trace_dir is None:
            mastercalib_trace_dir = CALIB_DIR

        # Set default user trace directory
        if not trace_dir:
            trace_dir = CALIB_DIR
            print(f'No trace_dir specified, using CALIB_DIR: {CALIB_DIR}')
        else:
            print(f'Using specified trace_dir: {trace_dir}')

        # Identify placeholder extensions
        placeholder_indices = get_placeholder_extension_indices(file)

        if placeholder_indices:
            print(f"\n{'='*60}")
            print(f"HYBRID TRACE EXTRACTION")
            print(f"{'='*60}")
            print(f"Detected {len(placeholder_indices)} placeholder extensions (missing cameras)")
            print(f"  Real extensions: Will use traces from {trace_dir}")
            print(f"  Placeholder extensions: Will use mastercalib traces from {mastercalib_trace_dir}")
            print(f"{'='*60}\n")

        #Running the extract routine with hybrid trace selection
        extraction_list = []

        # Build HDU-trace pairs with per-extension trace directory selection
        hdu_trace_pairs = []
        for idx in range(1, len(hdu)):
            header = hdu[idx].header

            # Get camera configuration
            if 'COLOR' in header:
                channel = header['COLOR'].lower()
                bench = header['BENCH']
                side = header['SIDE']
            else:
                camname = header['CAM_NAME']
                channel = camname.split('_')[1].lower()
                bench = camname.split('_')[0][0]
                side = camname.split('_')[0][1]

            # Select trace directory based on placeholder status
            if idx in placeholder_indices:
                active_trace_dir = mastercalib_trace_dir
                trace_source = "mastercalib"
            else:
                active_trace_dir = trace_dir
                trace_source = "user"

            # Find trace file
            try:
                trace_file = get_trace_file(channel, bench, side, active_trace_dir)
                hdu_trace_pairs.append((idx, trace_file))
                print(f"  Extension {idx:2d} ({channel:5s}{bench}{side}): Using {trace_source:12s} trace")
            except FileNotFoundError as e:
                logger.error(f"  Extension {idx:2d} ({channel:5s}{bench}{side}): Trace file not found - {e}")
                # Don't add to pairs - will skip this extension
        #print(hdu_trace_pairs)

        ### Testing ray usage

        # Process each HDU-trace pair in parallel using Ray.
        
        hdu_data_pairs = []
        for hdu_index, trace_file in hdu_trace_pairs:
            
            hdu_data = hdu[hdu_index].data.copy()
            hdu_data_pairs.append((hdu_index, hdu_data))
            
        # Compute detector-to-detector normalization offsets
        offsets = normalize_detector_backgrounds(hdu_data_pairs, rows=(30, 50))
            
            # Get the data and header from the current extension.
        futures = []
        for (hdu_index, trace_file), (_, hdu_data) in zip(hdu_trace_pairs, hdu_data_pairs):
        
            hdr = hdu[hdu_index].header
            
            # Check if this is a placeholder camera and skip background subtraction if so
            if is_placeholder_camera(hdu_data):
                logger.warning(f"Extension {hdu_index} is a placeholder camera (filled with 1.0), skipping background subtraction")
                # Still process but don't subtract background
            elif hdu_index in offsets:
                # Apply normalization offset (not full background subtraction!)
                offset = offsets[hdu_index]
                hdu_data = hdu_data - offset
                logger.info(f"Extension {hdu_index}: Applied detector normalization offset of {offset:.2f}")

            
            #future = process_trace.remote(hdu_data, hdr, trace_file, use_bias=use_bias)
            if (method == 'optimal'):
                future = process_trace.remote(hdu_data, hdr, trace_file, method='optimal', use_bias=use_bias)
            elif (method == 'boxcar'):
                future = process_trace.remote(hdu_data, hdr, trace_file, method='boxcar', use_bias=use_bias)
            futures.append(future)
        
        # Wait for all remote tasks to complete.
        raw_extraction_list = ray.get(futures)
        #extraction_list = [ex for ex in extraction_list if ex is not None]

        # Post-process to make objects writable
        extraction_list = []
        for ex in raw_extraction_list:
            if ex is not None:
                writable_ex = make_writable(ex)
                extraction_list.append(writable_ex)


        print(f'Extraction list = {extraction_list}')
        if output_dir:
            if os.path.exists(output_dir):        
                filename = save_extractions(extraction_list, primary_header=primary_hdr, savefile=extraction_file, save_dir=output_dir)
        else:
            filename = save_extractions(extraction_list, primary_header=primary_hdr, savefile=extraction_file, save_dir=OUTPUT_DIR)
        #print(f'extraction saved filename = {filename}')
        print(f'extraction saved filename = {extraction_file}')
        # if output_dir:
        #     if os.path.exists(output_dir):
                
        
        # else:
        if output_dir:
            obj, metadata = load_extractions(os.path.join(output_dir, extraction_file))
        else:
            obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, extraction_file))
        print(f'obj = {obj}')
        outfile = basefile+'_whitelight.fits'
        
        if output_dir and os.path.exists(output_dir):
            outfile = os.path.join(output_dir, outfile)
        else:
            outfile = os.path.join(OUTPUT_DIR, outfile)
                
        white_light_file = WhiteLightFits(obj, metadata, outfile=outfile)
        print(f'white_light_file = {white_light_file}')

        # Summary statistics
        if placeholder_indices:
            real_count = len(hdu_trace_pairs) - len(placeholder_indices)
            print(f"\n{'='*60}")
            print(f"EXTRACTION SUMMARY")
            print(f"{'='*60}")
            print(f"Total extracted: {len(extraction_list)} spectra")
            print(f"  Real camera data: {real_count} spectra (user traces)")
            print(f"  Placeholder data: {len(placeholder_indices)} spectra (mastercalib traces)")
            print(f"{'='*60}\n")

    except Exception as e:
        traceback.print_exc()
        return

    end_time = time.perf_counter()  # End timer
    elapsed = end_time - start_time
    # Log or print out the elapsed time
    print(f"Full GUI extraction process completed in {elapsed:.2f} seconds.")

    return extraction_file, white_light_file

def make_ifuimage(extraction_file, flat=False):
    obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, extraction_file))
    outfile = 'test_whitelight.fits'
    white_light_file = WhiteLightFits(obj, outfile=outfile)
    print(f'white_light_file = {white_light_file}')


def box_extract(file, flat=False):
    try:
    
        assert file.endswith('.fits'), 'File must be a .fits file'
        
        master_pkls = glob.glob(os.path.join(CALIB_DIR, '*.pkl'))
        
        if not master_pkls:
            raise ValueError("No master calibration files found in CALIB_DIR")
        
        #getting package sources for Ray
        # Get absolute path to llamas_pyjamas package
        package_path = pkg_resources.resource_filename('llamas_pyjamas', '')
        package_root = os.path.dirname(package_path)

        # Configure Ray runtime environment
        runtime_env = {
            "py_modules": [package_root],
            "env_vars": {"PYTHONPATH": f"{package_root}:{os.environ.get('PYTHONPATH', '')}"},
            "excludes": [
                str(Path(DATA_DIR) / "**"),  # Exclude DATA_DIR and all subdirectories
                "**/*.fits",                 # Exclude all FITS files anywhere
                "**/*.pkl",                  # Exclude all pickle files anywhere
                "**/.git/**",               # Exclude git directory
                "**/*.zip/**",
                "**/*.tar.gz/**",
                "**/mastercalib*/**",
            ]
        }

        # Initialize Ray
        ray.shutdown()
        ray.init(runtime_env=runtime_env)

        #opening the fitsfile
        hdu = process_fits_by_color(file)

        #Defining the base filename
        basefile = os.path.basename(file).split('.fits')[0]

        #Debug statements
        print(f'basefile = {basefile}')

        if flat:
            print(f'Running trace routine with flat fielding')
            ##Running the trace routine
            main(file)
            trace_files = glob.glob(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))
        else:
            print('Using master traces')
            trace_files = glob.glob(os.path.join(CALIB_DIR, f'*traces.pkl'))

        #print(os.path.join(CALIB_DIR, f'{basefile}*traces.pkl'))

        #Running the extract routine
        #This code should isolate to only the traces for the given fitsfile
        
        print(f'trace_files = {trace_files}')
        extraction_list = []
        
        hdu_trace_pairs = match_hdu_to_traces(hdu, trace_files)
        #print(hdu_trace_pairs)
        
        #for file in trace_files:
        for hdu_index, file in hdu_trace_pairs:
            hdr = hdu[hdu_index].header
            
            data = hdu[hdu_index].data.astype(float)
            
            bias = np.nanmedian(data)
            
            #print(f'hdu_index {hdu_index}, file {file}, {hdr['CAM_NAME']}')
            
            try:
                with open(file, mode='rb') as f:
                    tracer = pickle.load(f)
      
                extraction = ExtractLlamas(tracer, data-bias, hdu[hdu_index].header, optimal=False)
                extraction_list.append(extraction)
                
            except Exception as e:
                print(f"Error extracting trace from {file}")
                print(traceback.format_exc())
        
        print(f'Extraction list = {extraction_list}')        
        filename = save_extractions(extraction_list)
        print(f'extraction saved filename = {filename}')

        obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, filename))
        print(f'obj = {obj}')
        outfile = basefile + '_whitelight.fits'
        white_light_file = WhiteLightFits(obj, outfile=outfile)
        print(f'white_light_file = {white_light_file}')
    
    except Exception as e:
        traceback.print_exc()
        return
    
    
    return 
