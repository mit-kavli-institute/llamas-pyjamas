import os
import glob
from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR
from llamas_pyjamas.File.llamasIO import process_fits_by_color


from llamas_pyjamas.GUI.guiExtract import process_trace, save_extractions
from llamas_pyjamas.GUI.guiExtract import match_hdu_to_traces



def produce_flat_extractions(red_flat, green_flat, blue_flat, tracedir=None) -> None:
    """Produce the flat extractions for each fibre in the red, green and blue flat field images.

    Args:
        red_flat (str): path to the red flat field image
        green_flat (str): path to the green flat field image
        blue_flat (str): path to the blue flat field image
    """

    #isoltation the extensions by colour
    red_idxs = [1, 4, 7, 10, 13, 16, 19, 22]
    green_idxs = [2, 5, 8, 11, 14, 17, 20, 23]
    blue_idxs = [3, 6, 9, 12, 15, 18, 21, 24]

    red_extractions = []
    green_extractions = []
    blue_extractions = []


    #Open the flat field image to ensure trimming and approproate flips
    red_hdus = [process_fits_by_color(red_flat)[idx] for idx in red_idxs]
    green_hdus = [process_fits_by_color(green_flat)[idx] for idx in green_idxs]
    blue_hdus = [process_fits_by_color(blue_flat)[idx] for idx in blue_idxs]

    masterfile = 'LLAMAS_master'
    extraction_file = 'extractions_flat.pkl'

    if not tracedir:
        trace_files = glob.glob(os.path.join(CALIB_DIR, f'{masterfile}*traces.pkl'))
    else:
        print('Currently only using mastercalib files to implement this...check again later')
        trace_files = glob.glob(os.path.join(CALIB_DIR, f'{masterfile}*traces.pkl'))

    

    # Process the red flat field image
    red_hdu_trace_pairs = match_hdu_to_traces(red_hdus, trace_files)
    for hdu, trace in red_hdu_trace_pairs:
        extraction = process_trace(hdu, trace, tracedir=tracedir)
        red_extractions.append(extraction)
    red_filename = save_extractions(red_extractions, savefile='red_'+extraction_file)
    print(f'Red extractions saved to {red_filename}')
    
    #process the green flat field image
    green_hdu_trace_pairs = match_hdu_to_traces(green_hdus, trace_files)
    for hdu, trace in green_hdu_trace_pairs:
        extraction = process_trace(hdu, trace, tracedir=tracedir)
        green_extractions.append(extraction)
    green_filename = save_extractions(green_extractions, savefile='green_'+extraction_file)
    print(f'Green extractions saved to {green_filename}')

    blue_hdu_trace_pairs = match_hdu_to_traces(blue_hdus, trace_files)
    for hdu, trace in blue_hdu_trace_pairs:
        extraction = process_trace(hdu, trace, tracedir=tracedir)
        blue_extractions.append(extraction)
    blue_filename = save_extractions(blue_extractions, savefile='blue_'+extraction_file)
    print(f'Blue extractions saved to {blue_filename}')

