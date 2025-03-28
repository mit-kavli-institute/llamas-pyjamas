import os
import glob
import pdb
import argparse
import ray
import pkg_resources
from pathlib import Path
import logging
# from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR
from llamas_pyjamas.File.llamasIO import process_fits_by_color


from llamas_pyjamas.GUI.guiExtract import process_trace, make_writable
from llamas_pyjamas.Extract.extractLlamas import save_extractions
from llamas_pyjamas.GUI.guiExtract import match_hdu_to_traces



def reduce_flat(filename, idxs, tracedir=None, channel=None) -> None:
    """Reduce the flat field image and save the extractions to a pickle file.

    :param filename: the flat field image to reduce, must be of type FITS
    :type filename: str
    :param idxs: the indices of the HDUs to reduce
    :type idxs: list
    :param tracedir: directory to use for the tracing files, defaults to None
    :type tracedir: str, optional
    :param channel: the spectrograph channel from the fits image to process, defaults to None
    :type channel: str, optional
    """
    
    assert type(idxs) == list, 'idxs must be a list of integers'
    package_path = pkg_resources.resource_filename('llamas_pyjamas', '')
    package_root = os.path.dirname(package_path)
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
    num_cpus = 8
    ray.shutdown()
    ray.init(num_cpus=num_cpus, runtime_env=runtime_env)    
    
    
    _extractions = []
    _hdus = process_fits_by_color(filename)
    channel_hdus = [_hdus[idx] for idx in idxs]
    
    masterfile = 'LLAMAS_master'
    extraction_file = os.path.splitext(filename)[0] + '_extractions_flat.pkl'
    
    if type(channel) == str:
        extraction_file = f'{channel}_extractions_flat.pkl'
        
    if not tracedir:
        trace_files = glob.glob(os.path.join(CALIB_DIR, f'{masterfile}*traces.pkl'))
    else:
        print('Currently only using mastercalib files to implement this...check again later')
        trace_files = glob.glob(os.path.join(CALIB_DIR, f'{masterfile}*traces.pkl'))
    
    _hdu_trace_pairs = match_hdu_to_traces(channel_hdus, trace_files, start_idx=0)
    print("HDU-Trace pairs:", _hdu_trace_pairs)
    futures = []
    for hdu_index, trace_file in _hdu_trace_pairs:
        logging.info(f'Processing HDU {hdu_index} with trace file {trace_file}')   
        hdu_data = channel_hdus[hdu_index].data
        hdr = channel_hdus[hdu_index].header
       
        future = process_trace.remote(hdu_data, hdr, trace_file)
        futures.append(future)
    
    _extractions = ray.get(futures)
    ray.shutdown()
    # Post-process to make objects writable
    extraction_list = []
    for ex in _extractions:
        if ex is not None:
            writable_ex = make_writable(ex)
            extraction_list.append(writable_ex)    

    extracted_filename = save_extractions(extraction_list, savefile=extraction_file)
    print(f'Extractions saved to {extracted_filename}')
    
    return 


def produce_flat_extractions(red_flat, green_flat, blue_flat, tracedir=None) -> None:
    """Produce flat field extractions for each color channel.

    :param red_flat: file to use for red channel flat field extraction
    :type red_flat: str
    :param green_flat: file to use for green channel flat field extraction
    :type green_flat: str
    :param blue_flat: file to use for blue channel flat field extraction
    :type blue_flat: str
    :param tracedir: Directory to use find the trace files, defaults to None
    :type tracedir: str, optional
    """

    #isoltation the extensions by colour
    red_idxs = [1, 4, 7, 10, 13, 16, 19, 22]
    green_idxs = [2, 5, 8, 11, 14, 17, 20, 23]
    blue_idxs = [3, 6, 9, 12, 15, 18, 21, 24]

    # Reduce the red flat field image
    reduce_flat(red_flat, red_idxs, tracedir=tracedir, channel='red')
    # Reduce the green flat field image
    reduce_flat(green_flat, green_idxs, tracedir=tracedir, channel='green')
    # Reduce the blue flat field image
    reduce_flat(blue_flat, blue_idxs, tracedir=tracedir, channel='blue')
    
    print('Flat field extractions complete.')
    
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process LLAMAS FITS files using Ray multiprocessing.')
    
    parser.add_argument('filename', type=str, help='Path to input FITS file')
    parser.add_argument('--mastercalib', action='store_true', help='Use master calibration')
    parser.add_argument('--channel', type=str, choices=['red', 'green', 'blue'], help='Specify the color channel to use')
    parser.add_argument('--outpath', type=str, help='Path to save output files')
    args = parser.parse_args()
    
    red_idxs = [1, 4, 7, 10, 13, 16, 19, 22]
    green_idxs = [2, 5, 8, 11, 14, 17, 20, 23]
    blue_idxs = [3, 6, 9, 12, 15, 18, 21, 24]
    
    if args.channel == 'red':
        reduce_flat(args.filename, red_idxs, channel=args.channel)
    elif args.channel == 'green':
        reduce_flat(args.filename, green_idxs, channel=args.channel)
    elif args.channel == 'blue':
        reduce_flat(args.filename, blue_idxs, channel=args.channel)
    else:
        raise ValueError('Channel must be red, green, or blue')
