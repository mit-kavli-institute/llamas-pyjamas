import os
import sys
import numpy as np
import pickle
import ray
import pkg_resources
import glob
import traceback
from pathlib import Path
from datetime import datetime
import argparse


# Get package root and add to path before other imports
package_root = Path().absolute().parent
sys.path.append(str(package_root))


from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR
from llamas_pyjamas.Trace.traceLlamasMulti import main as run_trace# type: ignore
from astropy.io import fits

sys.path.append(BASE_DIR+'/')


ray.init(ignore_reinit_error=True)

from llamas_pyjamas.Trace.traceLlamasMulti import main # type: ignore
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions, load_extractions # type: ignore
from llamas_pyjamas.Image.WhiteLight import WhiteLight, WhiteLightFits, WhiteLightQuickLook # type: ignore
from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.File.llamasIO import llamasAllCameras, llamasOneCamera, getBenchSideChannel # type: ignore
from llamas_pyjamas.QA.llamasQA import plot_ds9 # type: ignore

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, f'traceLlamasMulti_{timestamp}.log')


# Define the function to run traceLlamasMulti.py
def run_trace_script(file):
    run_trace(file)

def match_hdu_to_traces(hdu_list: list, trace_files: list) -> list:
    """
    Match HDU extensions to their corresponding trace files.
    Parameters:
    hdu_list (list): A list of HDU (Header Data Unit) objects from a FITS file.
    trace_files (list): A list of file paths to trace files.
    Returns:
    list: A list of tuples where each tuple contains the index of the HDU and the matching trace file path.
    The function skips the primary HDU (index 0) and processes the remaining HDUs. It extracts the color, bench, 
    and side information from the HDU header to form a pattern. This pattern is then used to find a matching 
    trace file from the provided list of trace files. If a matching trace file is found, it is added to the 
    matches list along with the HDU index. If no matching trace file is found, a message is printed.
    """
    

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



### DO not use this method as it is outdated and likely wrong due to 4am coding skills
def main_extract(file):
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

    
        basename = os.path.basename(file).split('.fits')[0]
        
        hdu = fits.open(file)
        hdr = hdu[0].header
        
        ##creating a new HDU file from the WhiteLightFits method
        # Create HDU list
        new_hdul = fits.HDUList()
        primary_hdu = fits.PrimaryHDU()
        new_hdul.append(primary_hdu)
        
        green_idx = 0
        
        
        # for index, item in enumerate(hdu[1:]):
            
        #     if not 'COLOR' in item.header:
        #         camname = item.header['CAM_NAME']
        #         channel = camname.split('_')[1]
        #         bench = camname.split('_')[0][0]
        #         side = camname.split('_')[0][1]
        #     else:
        #         bench = item.header['BENCH']
        #         side = item.header['SIDE']
        #         channel = item.header['COLOR']
            
        #     data = item.data
        
        green_hdu_idx = [i for i in range(1, len(hdu)) if 'green' in hdu[i].header['CAM_NAME'].lower()]
        blue_hdu_idx = [i for i in range(1, len(hdu)) if 'blue' in hdu[i].header['CAM_NAME'].lower()]
        red_hdu_idx = [i for i in range(1, len(hdu)) if 'red' in hdu[i].header['CAM_NAME'].lower()]
        
        green_extraction_img = np.zeros((43, 46, 8))
        blue_extraction_img = np.zeros((43, 46, 8))
        red_extraction_img = np.zeros((43, 46, 8))
        
        
        hdu_trace_pairs = match_hdu_to_traces(hdu, master_pkls)
        for i, idx in enumerate(green_hdu_idx):
            pkl = next(trace for hdu_idx, trace in hdu_trace_pairs if hdu_idx == idx)        
            data = hdu[idx].data

            whitelight, xdata, ydata, flux = WhiteLightQuickLook(pkl, data)
            green_extraction_img[:, :, i] = whitelight
        
        for i, idx in enumerate(blue_hdu_idx):
            pkl = next(trace for hdu_idx, trace in hdu_trace_pairs if hdu_idx == idx)        
            data = hdu[idx].data

            whitelight, xdata, ydata, flux = WhiteLightQuickLook(pkl, data)
            blue_extraction_img[:, :, i] = whitelight
            
        for i, idx in enumerate(red_hdu_idx):
            pkl = next(trace for hdu_idx, trace in hdu_trace_pairs if hdu_idx == idx)        
            data = hdu[idx].data

            whitelight, xdata, ydata, flux = WhiteLightQuickLook(pkl, data)
            red_extraction_img[:, :, i] = whitelight
            
            
            # if channel.lower() == 'green':
            #     print(f'green idx = {green_idx}')
            #     print('hitting green condition')
                
            #     search_term = f'{channel}_{bench}_{side}_traces.pkl'
            #     print(f'search_term = {search_term}')
                
            #     # Find matching pickle file
            #     matching_pkl = [pkl for pkl in master_pkls if search_term in pkl][0]
            #     print(f'matching_pkls = {matching_pkl}')
            #     if not matching_pkl:
            #         raise ValueError(f"No matching pickle file found for {search_term}")

            #     #for index, matching_pkl in enumerate(matching_pkls):
            #         #print(f'index {index}')
            #     whitelight, xdata, ydata, flux = WhiteLightQuickLook(matching_pkl, data)
            #     print(f'Whitelight generated')
            #     extractions[:, :, green_idx] = whitelight
            #     green_idx += 1
            #     print(f'green_idx = {green_idx}')
                    
        # print(f'Extractions = {extractions}')
        # print(f'{np.shape(extractions)}')
        stitched_image_green = np.nansum(green_extraction_img, axis=2)
        stitched_image_blue = np.nansum(blue_extraction_img, axis=2)
        stitched_image_red = np.nansum(red_extraction_img, axis=2)
        
        green_hdu = fits.ImageHDU(data=stitched_image_green.astype(np.float32), name=f'green')
        new_hdul.append(green_hdu)
        
        blue_hdu = fits.ImageHDU(data=stitched_image_green.astype(np.float32), name=f'blue')
        new_hdul.append(blue_hdu)
        
        red_hdu = fits.ImageHDU(data=stitched_image_green.astype(np.float32), name=f'red')
        new_hdul.append(red_hdu)
                    
        fitsfilebase = file.split('/')[-1]
        white_light_file = fitsfilebase.replace('.fits', '_whitelight.fits')
        print(f'Writing white light file to {white_light_file}')
        # Write to file
        new_hdul.writeto(os.path.join(OUTPUT_DIR, white_light_file), overwrite=True)
                 
    
    except Exception as e:
        traceback.print_exc()
    
    return 

##Main function currently used by the Quicklook for full extraction

def brute_extract(file: fits.BinTableHDU, flatfiles: str = None, biasfiles: str = None) -> None:
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

        #Defining the base filename
        basefile = os.path.basename(file).split('.fits')[0]

        #Debug statements
        print(f'basefile = {basefile}')

        if flatfiles:
            try:
                print(f'generating new traces routine with flat file {flatfiles}')
                ##Running the trace routine
                main(file)
                trace_files = glob.glob(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))
            
            except Exception as e:
                traceback.print_exc()
        else:
            trace_files = glob.glob(os.path.join(CALIB_DIR, f'*traces.pkl'))
            print(f'Using master traces {trace_files}')

        #Running the extract routine
        #This code should isolate to only the traces for the given fitsfile
        
        extraction_list = []
        
        hdu_trace_pairs = match_hdu_to_traces(hdu, trace_files)
        #print(hdu_trace_pairs)
        
        #for file in trace_files:
        for hdu_index, file in hdu_trace_pairs:
            hdr = hdu[hdu_index].header
            
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


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process LLAMAS FITS files using Ray multiprocessing.')
    parser.add_argument('filename', type=str, help='Path to input FITS file')
    parser.add_argument('--flat', action='store_true', help='Flag to indicate flat fielding')
    
    args = parser.parse_args()
    
    #brute_extract(args.filename, flat=args.flat)
    box_extract(args.filename, flat=args.flat)
    #main_extract(args.filename)