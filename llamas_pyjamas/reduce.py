import os
import argparse
import pickle
import traceback
from datetime import datetime

from llamas_pyjamas.Trace.traceLlamasMaster import run_ray_tracing
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR, BIAS_DIR, LUT_DIR
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions
import llamas_pyjamas.GUI.guiExtract as ge
from llamas_pyjamas.File.llamasIO import process_fits_by_color
from llamas_pyjamas.File.llamasRSS import update_ra_dec_in_fits
import llamas_pyjamas.Arc.arcLlamas as arc
from llamas_pyjamas.File.llamasRSS import RSSgeneration
from llamas_pyjamas.Utils.utils import count_trace_fibres, setup_logger
from llamas_pyjamas.Cube.cubeConstruct import CubeConstructor


_linefile = os.path.join(LUT_DIR, '')





### This needs to be edited to handle a trace file per channel
#This has been independently tested
def generate_traces(red_flat, green_flat, blue_flat, output_dir, bias=None):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    assert os.path.exists(red_flat), "Red flat file does not exist."
    assert os.path.exists(green_flat), "Green flat file does not exist."
    assert os.path.exists(blue_flat), "Blue flat file does not exist."

    run_ray_tracing(red_flat, outpath=output_dir, channel='red', use_bias=bias)
    run_ray_tracing(green_flat, outpath=output_dir, channel='green', use_bias=bias)
    run_ray_tracing(blue_flat, outpath=output_dir, channel='blue', use_bias=bias)
    print(f"Traces generated and saved to {output_dir}")

    return


###need to edit GUI extract to give custom output_dir
#currently designed to use skyflats
#only used for generating new wl solutions
def extract_flat_field(flat_file_dir, output_dir, use_bias=None):
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ge.GUI_extract(flat_file_dir, output_dir=output_dir, use_bias=use_bias)

    return


def run_extraction(science_file, output_dir, use_bias=None, trace_dir=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert os.path.exists(science_file), "Science file does not exist."
    if type(science_file) is list:
        for file in science_file:
            assert os.path.exists(file), f"Science file {file} does not exist."
            extraction_file_path = ge.GUI_extract(file, output_dir=output_dir, use_bias=use_bias, trace_dir=trace_dir)
    else:
        assert os.path.exists(science_file), "Science file does not exist."
        extraction_file_path, _ = ge.GUI_extract(science_file, output_dir=output_dir, use_bias=use_bias, trace_dir=trace_dir)

    return  extraction_file_path


def calc_wavelength_soln(arc_file, output_dir, bias=None):

    ge.GUI_extract(arc_file, use_bias=bias, output_dir=output_dir)

    arc_picklename = os.path.join(output_dir, os.path.basename(arc_file).replace('_mef.fits', '_extract.pkl'))

    with open(arc_picklename, 'rb') as fp:
        batch_data = pickle.load(fp)
    
    arcdict = ExtractLlamas.loadExtraction(arc_picklename)
    arcspec, metadata = arcdict['extractions'], arcdict['metadata']

    arc.shiftArcX(arc_picklename)

    return arcdict



def relative_throughput(shift_picklename, flat_picklename):

    arc.fiberRelativeThroughput(flat_picklename, shift_picklename)
    ### need to add code in to return the name of the throughput file
    return


def correct_wavelengths(science_extraction_file, soln=None):
    if soln is None:
        # Load the reference arc dictionary if not provided
        arcdict = ExtractLlamas.loadExtraction(os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl'))
    
    _science = ExtractLlamas.loadExtraction(science_extraction_file)
    extractions, metadata, primary_hdr = _science['extractions'], _science['metadata'], _science['primary_header']
    print(f'extractions: {extractions}')
    print(f'metadata: {metadata}')
    std_wvcal = arc.arcTransfer(_science, arcdict,)
    
    print(f'std_wvcal: {std_wvcal}')
    print(f'std_wvcal metadata: {std_wvcal.get('metadata', {})}')
    
    return std_wvcal, primary_hdr



def construct_cube(rss_files, output_dir, wavelength_range=None, dispersion=1.0, spatial_sampling=0.75):
    """
    Construct IFU data cubes from RSS files.
    
    This function can handle both:
    1. Single RSS files with multiple channels
    2. Multiple channel-specific RSS files with names like:
       "_extract_RSS_blue.fits", "_extract_RSS_green.fits", "_extract_RSS_red.fits"
    
    Parameters:
        rss_files (str or list): Path to RSS FITS file(s) or base paths
        output_dir (str): Directory to save output cubes
        wavelength_range (tuple, optional): Min/max wavelength range for output cubes
        dispersion (float): Wavelength dispersion in Angstroms/pixel
        spatial_sampling (float): Spatial sampling in arcsec/pixel
        
    Returns:
        list: Paths to constructed cube files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if isinstance(rss_files, str):
        rss_files = [rss_files]
        
    cube_files = []
    
    # Create a single logger for all cube construction
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logger = setup_logger(__name__, f'CubeConstruct_{timestamp}.log')
    logger.info(f"Starting cube construction for {len(rss_files)} RSS files/base paths")
    
    for rss_file in rss_files:
        # Get base name for output files and detect channel if present
        base_name = os.path.splitext(os.path.basename(rss_file))[0]
        channel = None
        
        # Check for channel-specific naming pattern
        for color in ['red', 'green', 'blue']:
            if f'_extract_RSS_{color}' in base_name:
                # Extract the base name and channel color
                channel = color
                base_name = base_name.split(f'_extract_RSS_{color}')[0]
                logger.info(f"Detected {color} channel file, using base name: {base_name}")
                break
        
        if channel:
            print(f"Processing {channel} channel RSS file with base name: {base_name}")
        else:
            print(f"Processing RSS file (no specific channel detected): {base_name}")
            
        logger.info(f"Constructing channel cubes from RSS file: {rss_file}")
        print(f"Constructing channel cubes from RSS file: {rss_file}")
        
        # Pass the common logger to the constructor
        constructor = CubeConstructor(logger=logger)
        
        # Construct one cube per channel
        channel_cubes = constructor.construct_cube_from_rss(
            rss_file,
            wavelength_range=wavelength_range,
            dispersion=dispersion,
            spatial_sampling=spatial_sampling
        )
        
        if channel_cubes:
            # Log which channels were found
            logger.info(f"Found channels for {os.path.basename(rss_file)}: {list(channel_cubes.keys())}")
            
            # Save each channel cube
            saved_paths = constructor.save_channel_cubes(
                channel_cubes,
                output_prefix=os.path.join(output_dir, f"{base_name}"),
                header_info={'ORIGIN': 'LLAMAS Pipeline', 'SPAXELSZ': spatial_sampling},
                spatial_sampling=spatial_sampling
            )
            
            # Add saved paths to the list
            for channel, path in saved_paths.items():
                print(f"  - Channel {channel} cube saved: {path}")
                cube_files.append(path)
        else:
            print(f"  No valid channel cubes constructed for {rss_file}")
            logger.warning(f"No valid channel cubes constructed for {rss_file}")
    
    return cube_files

def main(config_path):
    print("This is a placeholder for the reduce module.")
    # You can add functionality here as needed.
    with open(config_path, 'r') as f:
        config = {}
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # Skip empty lines and comments
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Handle quoted values
                if value.startswith('"') and value.endswith('"') or value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]  # Remove quotes
                
                # Handle lists of quoted values
                elif ',' in value:
                    items = []
                    for item in value.split(','):
                        item = item.strip()
                        # Remove quotes from each item if present
                        if item.startswith('"') and item.endswith('"') or item.startswith("'") and item.endswith("'"):
                            item = item[1:-1]
                        items.append(item)
                    value = items
                    
                config[key] = value
        
        print(f"Loaded configuration from {config_path}")
    print("Configuration:", config)


        
    if not config.get('output_dir'):
        output_dir = os.path.join(BASE_DIR, 'reduced')
    else:
        output_dir = config.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)
        
    if bool(config.get('generate_new_wavelength_soln')) == True:
        print("Generating new wavelength solution.")
        extract_flat_field(config.get('flat_file_dir'), config.get('output_dir'), bias_file=config.get('bias_file'))
        if 'arc_file' not in config:
            raise ValueError("No arc file provided in the configuration.")
        relative_throughput(config.get('shift_picklename'), config.get('flat_picklename'))
        arcdict = calc_wavelength_soln(config['arc_file'], config.get('output_dir'), bias=config.get('bias_file'))
        config['arcdict'] = arcdict
        
    
    # Set default for trace_output_dir if not present
    if 'trace_output_dir' not in config:
        trace_path = os.path.join(output_dir, 'traces')
        os.makedirs(trace_path, exist_ok=True)
        config['trace_output_dir'] = trace_path
    else:
        trace_path = config['trace_output_dir']
        
    #set default for extraction_output_dir if not present
    if 'extraction_output_dir' not in config:
        extraction_path = os.path.join(output_dir, 'extractions')
        os.makedirs(extraction_path, exist_ok=True)
        config['extraction_output_dir'] = extraction_path
    else:
        extraction_path = config['extraction_output_dir']
    
    try:
        
        generate_traces(config.get('red_flat_file'), config.get('green_flat_file'), config.get('blue_flat_file'), 
                       config.get('trace_output_dir'), bias=config.get('bias_file'))
        
        # Check if the generated traces have the correct number of fibers
        print("Checking fiber counts in generated traces...")
        if not count_trace_fibres(config.get('trace_output_dir')):
            print("Generated traces have incorrect fiber counts. Using traces from CALIB_DIR instead.")
            config['trace_output_dir'] = CALIB_DIR
        else:
            print("Generated traces have correct fiber counts. Proceeding with new traces.")
        
        
       # Process science files by color if they're provided as a list
        if 'science_files' not in config:
            raise ValueError("No science files provided in the configuration.")
    
        if isinstance(config['science_files'], list):
            print(f'Found {len(config["science_files"])} science files to process.')
        
            for i, science_file in enumerate(config['science_files']):
                print(f"Processing science file {i+1}/{len(config['science_files'])}: {science_file}")
                if not os.path.exists(science_file):
                    raise FileNotFoundError(f"Science file {science_file} does not exist.")
                # Process each science file by color
                extracted_file = run_extraction(science_file, extraction_path, use_bias=config.get('bias_file'), trace_dir=config.get('trace_output_dir'))
                print(f"Extraction completed for {science_file}. Output file: {extracted_file}")
        else:
            extracted_file = run_extraction(config.get('science_files'), extraction_path, use_bias=config.get('bias_file'), trace_dir=config.get('trace_output_dir'))
            print(f"Extraction completed. Used traces {config.get('trace_output_dir')} Output file: {extracted_file}")

        # print("Correcting wavelengths in the extracted file...")
        # correction_path = os.path.join(extraction_path, extracted_file)
        pkl_files = [os.path.join(extraction_path, f) for f in os.listdir(extraction_path) if f.endswith('.pkl') and 'corrected' not in f]
        
        for index, file in enumerate(pkl_files):
            print(f"Processing extraction file {index+1}/{len(pkl_files)}: {file}")
            correction_path = file
            if not os.path.exists(correction_path):
                raise FileNotFoundError(f"Extraction file {correction_path} does not exist.")
            
            # Correct wavelengths for each extraction file
            corr_extractions, primary_hdr = correct_wavelengths(correction_path, soln=config.get('arcdict'))
            
            corr_extraction_list = corr_extractions['extractions']
            
            # Save the corrected extractions using the current file's base name
            base_name = os.path.splitext(os.path.basename(file))[0]
            savefile = os.path.join(extraction_path, f'{base_name}_corrected_extractions.pkl')
            save_extractions(corr_extraction_list, primary_header=primary_hdr, savefile=savefile, save_dir=extraction_path, prefix='LLAMASExtract_batch_corrected')

            # Create a logger for RSS generation
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            rss_logger = setup_logger(__name__, f'RSSgeneration_{timestamp}.log')
            rss_logger.info(f"Starting RSS generation for {base_name}")
            
            #RSS generation
            rss_gen = RSSgeneration(logger=rss_logger)
            rss_output_file = os.path.join(extraction_path, f'{base_name}_RSS.fits')
            new_rss_outputs = rss_gen.generate_rss(savefile, rss_output_file)
            rss_logger.info(f"RSS file generated: {new_rss_outputs}")
            print(f"RSS file generated: {new_rss_outputs}")

        # Updating RA and Dec in RSS files
        for rss_output_file in new_rss_outputs:
            update_ra_dec_in_fits(rss_output_file, logger=rss_logger)

        # Cube construction from RSS files
        print("Constructing cubes from RSS files...")
        # First check for files with 'extract_RSS' in the name
        rss_files = [os.path.join(extraction_path, f) for f in os.listdir(extraction_path) 
                if 'extract_RSS' in f and f.endswith('.fits')]
        
        # If none found, fall back to the original pattern
        if not rss_files:
            print(f"Found {len(rss_files)} RSS files for cube construction")

        if 'cube_output_dir' not in config:
            cube_output_dir = os.path.join(output_dir, 'cubes')
        else:
            cube_output_dir = config.get('cube_output_dir')
        os.makedirs(cube_output_dir, exist_ok=True)


        if rss_files:
            cube_files = construct_cube(
                rss_files, 
                cube_output_dir,
                wavelength_range=config.get('wavelength_range'),
                dispersion=config.get('dispersion', 1.0),
                spatial_sampling=config.get('spatial_sampling', 0.75)
            )
            print(f"Cubes constructed: {cube_files}")
        else:
            print("No RSS files found for cube construction")
                
        
        
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce module placeholder")
    parser.add_argument('config_file', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    
    main(args.config_file)