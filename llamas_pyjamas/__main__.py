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
from .config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR
from astropy.io import fits
# Get package root and add to path before other imports
package_root = Path().absolute().parent
sys.path.append(str(package_root))
sys.path.append(BASE_DIR+'/')


#ray.init(ignore_reinit_error=True)

from llamas_pyjamas.Trace.traceLlamasMulti import main # type: ignore
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions, load_extractions # type: ignore
from llamas_pyjamas.Image.WhiteLight import WhiteLight, WhiteLightFits, WhiteLightQuickLook # type: ignore
from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.File.llamasIO import llamasAllCameras, llamasOneCamera, getBenchSideChannel # type: ignore
from llamas_pyjamas.QA.llamasQA import plot_ds9 # type: ignore

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, f'traceLlamasMulti_{timestamp}.log')


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
            ]
        }

        # Initialize Ray
        # ray.shutdown()
        # ray.init(runtime_env=runtime_env)

        ##Running the trace routine
        #main(file)

        #Defining the base filename
        basefile = os.path.basename(file).split('.fits')[0]

        #Debug statements
        #print(f'basefile = {basefile}')
        #print(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))

        ##Running the extract routine
        #This code should isolate to only the traces for the given fitsfile
        # trace_files = glob.glob(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))
        # print(f'trace_files = {trace_files}')
        # extraction_list = []

        # for file in trace_files:
        #     try:
        #         with open(file, mode='rb') as f:
        #             tracer = pickle.load(f)
        #         extraction = ExtractLlamas(tracer)
        #         extraction_list.append(extraction)
        #     except Exception as e:
        #         print(f"Error extracting trace from {file}")
        #         print(traceback.format_exc())

        # filename = save_extractions(extraction_list)
        # print(f'extraction saved filename = {filename}')

        # obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, filename))
        # print(f'obj = {obj}')
        # white_light_file = WhiteLightFits(obj)
        # print(f'white_light_file = {white_light_file}')
        
        hdu = fits.open(file)
        hdr = hdu[0].header
        
        ##creating a new HDU file from the WhiteLightFits method
        # Create HDU list
        new_hdul = fits.HDUList()
        primary_hdu = fits.PrimaryHDU()
        new_hdul.append(primary_hdu)
        
        green_idx = 0
        extractions = np.zeros((43, 46, 8))
        for index, item in enumerate(hdu):
            if index == 0:
                continue
            bench = item.header['BENCH']
            side = item.header['SIDE']
            channel = item.header['COLOR']
            data = item.data
            
            if channel.lower() == 'green':
                print(f'green idx = {green_idx}')
                print('hitting green condition')
                
                search_term = f'{channel}_{bench}_{side}_traces.pkl'
                print(f'search_term = {search_term}')
                
                # Find matching pickle file
                matching_pkl = [pkl for pkl in master_pkls if search_term in pkl][0]
                print(f'matching_pkls = {matching_pkl}')
                if not matching_pkl:
                    raise ValueError(f"No matching pickle file found for {search_term}")

                #for index, matching_pkl in enumerate(matching_pkls):
                    #print(f'index {index}')
                whitelight, xdata, ydata, flux = WhiteLightQuickLook(matching_pkl, data)
                print(f'Whitelight generated')
                #breakpoint()
                extractions[:, :, green_idx] = whitelight
                green_idx += 1
                print(f'green_idx = {green_idx}')
                    
        print(f'Extractions = {extractions}')
        print(f'{np.shape(extractions)}')
        stitched_image = np.nansum(extractions, axis=2)
        
        _hdu = fits.ImageHDU(data=stitched_image.astype(np.float32), name=f'{channel}')
        new_hdul.append(_hdu)
                    
        fitsfilebase = file.split('/')[-1]
        white_light_file = fitsfilebase.replace('.fits', '_whitelight.fits')
        print(f'Writing white light file to {white_light_file}')
        # Write to file
        new_hdul.writeto(os.path.join(OUTPUT_DIR, white_light_file), overwrite=True)
                 
    
    except Exception as e:
        traceback.print_exc()
    
    return extractions

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process LLAMAS FITS files using Ray multiprocessing.')
    parser.add_argument('filename', type=str, help='Path to input FITS file')
    args = parser.parse_args()
    
    main_extract(args.filename)