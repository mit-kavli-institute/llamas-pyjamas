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
from llamas_pyjamas.Image.WhiteLightModule import WhiteLightFits

from llamas_pyjamas.GUI.guiExtract import process_trace, make_writable
from llamas_pyjamas.Extract.extractLlamas import save_extractions
from llamas_pyjamas.GUI.guiExtract import match_hdu_to_traces
import pickle
from typing import List, Tuple
from astropy.io import fits
import numpy as np

from llamas_pyjamas.Trace.traceLlamasMaster import _grab_bias_hdu

from llamas_pyjamas.constants import RED_IDXS, GREEN_IDXS, BLUE_IDXS

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
    print(f'Processing {filename} with indices {idxs}')
    _hdus = process_fits_by_color(filename)
    print(f'Length of _hdus: {len(_hdus)}')
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


def produce_flat_extractions(red_flat, green_flat, blue_flat, tracedir=None, custom=None) -> Tuple[List, List]:
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

    # Reduce the red flat field image
    reduce_flat(red_flat, RED_IDXS, tracedir=tracedir, channel='red')
    # Reduce the green flat field image
    reduce_flat(green_flat, GREEN_IDXS, tracedir=tracedir, channel='green')
    # Reduce the blue flat field image
    reduce_flat(blue_flat, BLUE_IDXS, tracedir=tracedir, channel='blue')
    
    print('Flat field extractions complete.')


    red_extraction, green_extraction, blue_extraction = None, None, None
    if custom:
        red_extraction = os.path.splitext(red_flat)[0] + '_extractions_flat.pkl'
        green_extraction = os.path.splitext(green_flat)[0] + '_extractions_flat.pkl'
        blue_extraction = os.path.splitext(blue_flat)[0] + '_extractions_flat.pkl'
    else:
        red_extraction = os.path.join(OUTPUT_DIR, 'red_extractions_flat.pkl')
        green_extraction = os.path.join(OUTPUT_DIR, 'green_extractions_flat.pkl')
        blue_extraction = os.path.join(OUTPUT_DIR, 'blue_extractions_flat.pkl')

    all_extractions = []
    all_metadata = []
    for fname in [red_extraction, green_extraction, blue_extraction]:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        all_extractions.extend(data.get('extractions', []))
        all_metadata.extend(data.get('metadata', []))
    print("Total extractions:", len(all_extractions))
    print("Total metadata entries:", len(all_metadata))
    print(f"All metadata is {all_metadata}")

    return all_extractions, all_metadata


def produce_normalised_whitelight(red_flat, green_flat, blue_flat, tracedir=None, custom=None) -> None:


    extraction_list, metadata = produce_flat_extractions(red_flat, green_flat, blue_flat, tracedir=tracedir, custom=custom)

    outfile = 'flat_whitelight.fits'
    white_light_file = WhiteLightFits(extraction_list, metadata, outfile=outfile)
    print(f'white_light_file = {white_light_file}')

    hdu = fits.open(os.path.join(OUTPUT_DIR, outfile), mode='update')
    # Indices of HDUs containing image data to normalize
    image_extensions = [1, 3, 5]
    for index, item in enumerate(hdu):
        if index == 0:
            continue
        elif index in image_extensions:
            print(f'HDU {index}: {item}')
            image = hdu[index].data
            max_val = np.nanmax(image)
            if np.isnan(image).all() or np.isnan(max_val):
                logging.warning(f'All values are NaN for HDU {index}. Skipping normalization.')
                normalized_image = image
            elif max_val != 0:
                normalized_image = image / max_val
            else:
                print(f'All values are zero for HDU {index}. Skipping normalization.')
                normalized_image = image

            hdu[index].data = normalized_image
            print(f'Normalized image; new max value: {np.nanmax(normalized_image)}')
    # Saving the normalised image
    hdu.flush()  # ensure changes are written to disk
    hdu.close()
    
    return None


def apply_flat_field(science_file, flat_file, output_file):
    """
    Divide the science image by the normalized flat field image and write the result as a new FITS file.
    
    Args:
        science_file (str): Path to the science FITS file.
        flat_file (str): Path to the normalized flat field FITS file.
        output_file (str): Path for the new, flat-field-corrected FITS file.
    """
    with fits.open(science_file) as sci_hdus, fits.open(flat_file) as flat_hdus:
        # Copy the primary header from the science file to preserve metadata
        new_primary = fits.PrimaryHDU(header=sci_hdus[0].header)
        new_hdus = []
        # Iterate over each HDU; assumes both files have matching HDUs.
        image_extensions = [1, 3, 5]
        for idx, item in enumerate(image_extensions):
            sci_data = sci_hdus[item].data
            #print(f'HDU {idx}: {sci_hdus[idx].data}')
            
            flat_data = flat_hdus[item].data
            #print(f'HDU {idx}: {sci_hdus[idx].data}')
            # If both HDUs contain image data, perform the division.
            
            if sci_data is not None and flat_data is not None:
                # Convert to float32 to carry out the division
                sci_data = sci_data.astype(np.float32)
                flat_data = flat_data.astype(np.float32)
                # Avoid division by zero by placing NaN where flat field is zero.
                with np.errstate(divide='ignore', invalid='ignore'):
                    #corrected_data = np.where(flat_data != 0, sci_data / flat_data, np.nan)
                    corrected_data = np.divide(sci_data, flat_data, out=np.zeros_like(sci_data), where=flat_data != 0)
                    
                # Use the science header (or you can update it accordingly)
                new_hdu = fits.ImageHDU(data=corrected_data, header=sci_hdus[idx].header)
            else:
                print(f'warning: HDU {idx} in science or flat file is None. Skipping this HDU.')
            new_hdus.append(new_hdu)
        # Add the primary header HDU to the beginning of the list
        new_hdus.insert(0, new_primary)
        print(f'Length of new_hdus: {len(new_hdus)}')
        # Assemble the new HDU list and write to file.
        new_hdul = fits.HDUList(new_hdus)
        new_hdul.writeto(output_file, overwrite=True)
        print(f"Flat-field corrected FITS file saved as: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Process LLAMAS FITS files using Ray multiprocessing.'
    )
    parser.add_argument(
        'filenames',
        type=str,
        nargs='+',
        help='Path(s) to input FITS file. Supply one file for single-channel extraction (use --channel or --all) or exactly three files for red, green, and blue channels.'
    )
    parser.add_argument('--mastercalib', action='store_true', help='Use master calibration')
    parser.add_argument(
        '--channel',
        type=str,
        choices=['red', 'green', 'blue'],
        help='Specify the color channel to use for a single file extraction'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Extract all channels from a single file'
    )
    parser.add_argument(
        '--outpath',
        type=str,
        help='Path to save output files'
    )
    
    args = parser.parse_args()
    print(f'args.filenames {args.filenames}')
    # Multiple files provided: use produce_flat_extractions if exactly three files given.
    if len(args.filenames) > 1:
        if len(args.filenames) != 3:
            parser.error("When providing multiple files, exactly three files are required for red, green, and blue channels.")
        #produce_flat_extractions(args.filenames[0], args.filenames[1], args.filenames[2], tracedir=args.outpath)
        produce_normalised_whitelight(args.filenames[0], args.filenames[1], args.filenames[2], tracedir=args.outpath)
    else:
        # Single file provided. Must supply either --channel or --all.
        if not (args.channel or args.all):
            parser.error("For a single file extraction, you must supply --channel or --all.")
        if args.channel:
            if args.channel == 'red':
                idxs = RED_IDXS
            elif args.channel == 'green':
                idxs = GREEN_IDXS
            elif args.channel == 'blue':
                idxs = BLUE_IDXS
            reduce_flat(args.filenames[0], idxs, tracedir=args.outpath, channel=args.channel)
        elif args.all:
            # Process all channels for the single file.
            if len(args.filenames) != 1:
                parser.error("When using --all, only one file should be provided.")
            print("Processing all channels for the single file...")
            reduce_flat(args.filenames[0], RED_IDXS, tracedir=args.outpath, channel='red')
            reduce_flat(args.filenames[0], GREEN_IDXS, tracedir=args.outpath, channel='green')
            reduce_flat(args.filenames[0], BLUE_IDXS, tracedir=args.outpath, channel='blue')

