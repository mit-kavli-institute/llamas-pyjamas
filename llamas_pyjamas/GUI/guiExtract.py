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
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR

from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions, load_extractions
from llamas_pyjamas.Image.WhiteLight import WhiteLight, WhiteLightFits, WhiteLightQuickLook
import time

# Set up logging
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, log_filename=f'extractLlamas_{timestamp}.log')



####################################################################################

def ExtractLlamasCube(infits, tracefits, optimal=True):

    assert infits.endswith('.fits'), 'File must be a .fits file'  
    hdu = fits.open(infits)

    # Find the trace files
    basefile = os.path.basename(tracefits).split('.fits')[0]
    trace_files = glob.glob(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))
    extraction_file = os.path.basename(infits).split('mef.fits')[0] + 'extract.pkl'

    if len(trace_files) == 0:
        logger.error("No trace files found for the indicated file root!")
        return None
    
    hdu_trace_pairs = match_hdu_to_traces(hdu, trace_files)
    print(hdu_trace_pairs)

    extraction_list = []

    print(f"Saving extractions to {extraction_file}")

    counter = 1
    for hdu_index, file in hdu_trace_pairs:

        print(f"Extracting extension number {counter} of 24")
        hdr = hdu[hdu_index].header 
        bias = np.nanmedian(hdu[hdu_index].data.astype(float))  
        
        try:
            with open(file, mode='rb') as f:
                tracer = pickle.load(f)
    
            extraction = ExtractLlamas(tracer, hdu[hdu_index].data.astype(float)-bias, hdu[hdu_index].header)
            extraction_list.append(extraction)
            
        except Exception as e:
            print(f"Error extracting trace from {file}")
            print(traceback.format_exc())
        counter += 1
        
    print(f'Extraction list = {extraction_list}')        
    filename = save_extractions(extraction_list, savefile=extraction_file)
    print(f'extraction saved filename = {filename}')

    return None


def match_hdu_to_traces(hdu_list, trace_files):
    """Match HDU extensions to their corresponding trace files"""
    matches = []
    
    # Skip primary HDU (index 0)
    for idx in range(1, len(hdu_list)):
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
        #print(f'HDU {idx}: {color} {benchside} -> {matching_trace}')
        if matching_trace:
            matches.append((idx, matching_trace))
        else:
            print(f"No matching trace found for HDU {idx}: {color} {benchside}")
            
    return matches

# Define a Ray remote function for processing a single trace extraction.
@ray.remote
def process_trace(hdu_data, header, trace_file):
    """
    Process a single HDU: subtract bias, load the trace from a trace file, and create an ExtractLlamas object.
    Returns the extraction object or None if there is an error.
    """
    try:
        # Compute the bias from the current extension data.
        bias = np.nanmedian(hdu_data.astype(float))
        # Load the trace object from the pickle file.
        with open(trace_file, mode='rb') as f:
            tracer = pickle.load(f)
        # Create an ExtractLlamas object; note the subtraction of the bias.
        extraction = ExtractLlamas(tracer, hdu_data.astype(float), header)
        return extraction
    except Exception as e:
        print(f"Error extracting trace from {trace_file}")
        print(traceback.format_exc())
        return None


def make_writable(extraction_obj):
    """Convert a Ray-returned extraction object to a writable version."""
    import copy
    import pickle
    
    # First approach: Deep copy
    try:
        return copy.deepcopy(extraction_obj)
    except:
        pass
    
    # Second approach: Pickle and unpickle
    try:
        # This forces a complete serialization and deserialization
        pickled = pickle.dumps(extraction_obj)
        return pickle.loads(pickled)
    except:
        pass
    
    # Third approach: If the object has a to_dict method, use it
    if hasattr(extraction_obj, 'to_dict') and callable(extraction_obj.to_dict):
        try:
            obj_dict = extraction_obj.to_dict()
            # Assuming there's a from_dict or similar constructor
            if hasattr(type(extraction_obj), 'from_dict') and callable(type(extraction_obj).from_dict):
                return type(extraction_obj).from_dict(obj_dict)
        except:
            pass
    
    # If all else fails, return the original object and log a warning
    print(f"Warning: Could not make object of type {type(extraction_obj)} writable")
    return extraction_obj



##Main function currently used by the Quicklook for full extraction

def GUI_extract(file: fits.BinTableHDU, flatfiles: str = None, bias: str = None) -> None:
    """
    Extracts data from a FITS file using calibration files and saves the extracted data.
    Parameters:
    file (str): Path to the FITS file to be processed. Must have a .fits extension.
    flatfiles (str, optional): Path to the flat files for generating new traces. Defaults to None.
    biasfiles (str, optional): Path to the bias files for calibration. Defaults to None.
    
    Returns:
    None
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
                
            ]
        }

        # Initialize Ray
        num_cpus = 5
        ray.shutdown()
        ray.init(num_cpus=num_cpus, runtime_env=runtime_env)

        #opening the fitsfile
        hdu = fits.open(file)

        

        extraction_file = os.path.basename(file).split('mef.fits')[0] + 'extract.pkl'

        #Defining the base filename
        #basefile = os.path.basename(file).split('.fits')[0]
        basefile = os.path.basename(file).split('.fits')[0]
        masterfile = 'LLAMAS_master'
        masterbiasfile = os.path.join(CALIB_DIR, 'combined_bias.fits')

        #Debug statements
        print(f'basefile = {basefile}')
        print(f'masterfile = {masterfile}')
        print(f'Bias file is {masterbiasfile}')


        bias_hdu = None
        #if bias == None:
            #opening the masterbias

            ##not implementing this yet
            #bias_hdu = fits.open(masterbiasfile)
            #assert len(hdu) == len(bias_hdu), 'Number of extensions in the bias and fits file do not match'

        
        trace_files = glob.glob(os.path.join(CALIB_DIR, f'{masterfile}*traces.pkl'))
        print(f'Using master traces {trace_files}')
        
        #Running the extract routine
        #This code should isolate to only the traces for the given fitsfile
        
        extraction_list = []
        
        hdu_trace_pairs = match_hdu_to_traces(hdu, trace_files)
        #print(hdu_trace_pairs)

        ### Testing ray usage

        # Process each HDU-trace pair in parallel using Ray.
        futures = []
        for hdu_index, trace_file in hdu_trace_pairs:
            
            hdu_data = hdu[hdu_index].data
            
            # Get the data and header from the current extension.
            
            hdr = hdu[hdu_index].header
            future = process_trace.remote(hdu_data, hdr, trace_file)
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
        filename = save_extractions(extraction_list, savefile=extraction_file)
        #print(f'extraction saved filename = {filename}')
        print(f'extraction saved filename = {extraction_file}')

        # obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, filename))
        obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, extraction_file))
        print(f'obj = {obj}')
        outfile = basefile+'_whitelight.fits'
        white_light_file = WhiteLightFits(obj, metadata, outfile=outfile)
        print(f'white_light_file = {white_light_file}')
    
    except Exception as e:
        traceback.print_exc()
        return
    
    end_time = time.perf_counter()  # End timer
    elapsed = end_time - start_time
    # Log or print out the elapsed time
    print(f"Full GUI extraction process completed in {elapsed:.2f} seconds.")

    return 

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
        hdu = fits.open(file)

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
