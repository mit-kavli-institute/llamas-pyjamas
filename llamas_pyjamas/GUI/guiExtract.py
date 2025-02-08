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


ray.init(ignore_reinit_error=True)


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


##Main function currently used by the Quicklook for full extraction

def GUI_extract(file: fits.BinTableHDU, flatfiles: str = None, biasfiles: str = None) -> None:
    """
    Extracts data from a FITS file using calibration files and saves the extracted data.
    Parameters:
    file (str): Path to the FITS file to be processed. Must have a .fits extension.
    flatfiles (str, optional): Path to the flat files for generating new traces. Defaults to None.
    biasfiles (str, optional): Path to the bias files for calibration. Defaults to None.
    
    Returns:
    None
    """
    
    
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
        ray.shutdown()
        ray.init(runtime_env=runtime_env)

        #opening the fitsfile
        hdu = fits.open(file)

        extraction_file = os.path.basename(file).split('mef.fits')[0] + 'extract.pkl'

        #Defining the base filename
        #basefile = os.path.basename(file).split('.fits')[0]
        basefile = os.path.basename(file).split('.fits')[0]
        masterfile = 'LLAMAS_master'
        #Debug statements
        print(f'basefile = {basefile}')
        print(f'masterfile = {masterfile}')

        
        trace_files = glob.glob(os.path.join(CALIB_DIR, f'{masterfile}*traces.pkl'))
        print(f'Using master traces {trace_files}')

        #Running the extract routine
        #This code should isolate to only the traces for the given fitsfile
        
        extraction_list = []
        
        hdu_trace_pairs = match_hdu_to_traces(hdu, trace_files)
        #print(hdu_trace_pairs)
        
        #for file in trace_files:
        for hdu_index, file in hdu_trace_pairs:
            hdr = hdu[hdu_index].header

            if 'CAM_NAME' in hdr:
                cam_name = hdr['CAM_NAME']
                channel = cam_name.split('_')[1].lower()
                bench = cam_name.split('_')[0][0]
                side = cam_name.split('_')[0][1]
            
            else:
                channel = hdr['COLOR'].lower()
                bench = hdr['BENCH']
                side  = hdr['SIDE']

            
            bias = np.nanmedian(hdu[hdu_index].data.astype(float))
            
            #print(f'hdu_index {hdu_index}, file {file}, {hdr['CAM_NAME']}')
            
            try:
                with open(file, mode='rb') as f:
                    tracer = pickle.load(f)
      
                extraction = ExtractLlamas(tracer, hdu[hdu_index].data.astype(float)-bias, hdu[hdu_index].header)
                extraction_list.append(extraction)
                
                
            except Exception as e:
                print(f"Error extracting trace from {file}")
                print(traceback.format_exc())
        
        print(f'Extraction list = {extraction_list}')        
        filename = save_extractions(extraction_list, savefile=extraction_file)
        #print(f'extraction saved filename = {filename}')
        print(f'extraction saved filename = {extraction_file}')

        # obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, filename))
        obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, extraction_file))
        print(f'obj = {obj}')
        outfile = basefile + '_whitelight.fits'
        white_light_file = WhiteLightFits(obj, outfile=outfile)
        print(f'white_light_file = {white_light_file}')
    
    except Exception as e:
        traceback.print_exc()
        return
    
    
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
