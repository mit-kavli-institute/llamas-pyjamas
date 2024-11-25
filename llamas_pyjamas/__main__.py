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
from config import BASE_DIR, OUTPUT_DIR, DATA_DIR

# Get package root and add to path before other imports
package_root = Path().absolute().parent
sys.path.append(str(package_root))
sys.path.append(BASE_DIR+'/')


ray.init(ignore_reinit_error=True)
from llamas_pyjamas.Trace.traceLlamasMulti import main # type: ignore
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions, load_extractions # type: ignore
from llamas_pyjamas.Image.WhiteLight import WhiteLight, WhiteLightFits
from llamas_pyjamas.Utils.utils import setup_logger


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, f'traceLlamasMulti_{timestamp}.log')


def main_extract(file):
    try:
    
        assert file.endswith('.fits'), 'File must be a .fits file'

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
        ray.shutdown()
        ray.init(runtime_env=runtime_env)

        ##Running the trace routine
        main(file)

        #Defining the base filename
        basefile = os.path.basename(file).split('.fits')[0]

        #Debug statements
        #print(f'basefile = {basefile}')
        #print(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))

        ##Running the extract routine
        #This code should isolate to only the traces for the given fitsfile
        trace_files = glob.glob(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))
        print(f'trace_files = {trace_files}')
        extraction_list = []

        for file in trace_files:
            try:
                with open(file, mode='rb') as f:
                    tracer = pickle.load(f)
                extraction = ExtractLlamas(tracer)
                extraction_list.append(extraction)
            except Exception as e:
                print(f"Error extracting trace from {file}")
                print(traceback.format_exc())

        filename = save_extractions(extraction_list)
        print(f'extraction saved filename = {filename}')

        obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, filename))
        print(f'obj = {obj}')
        white_light_file = WhiteLightFits(obj)
        print(f'white_light_file = {white_light_file}')
    
    except Exception as e:
        traceback.print_exc()
    
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process LLAMAS FITS files using Ray multiprocessing.')
    parser.add_argument('filename', type=str, help='Path to input FITS file')
    args = parser.parse_args()
    
    main_extract(args.filename)