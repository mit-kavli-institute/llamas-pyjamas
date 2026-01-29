"""LLAMAS data reduction pipeline main module.

This module provides the primary functions for reducing LLAMAS (Large Lens Array 
Multi-Object Spectrograph) observations through a complete pipeline from trace generation 
to final data products including RSS files and data cubes.

Functions:
    generate_traces: Generate fiber traces from flat field observations.
    extract_flat_field: Extract flat field spectra for calibration.
    run_extraction: Extract science spectra from observations.
    calc_wavelength_soln: Calculate wavelength solutions from arc lamp observations.
    relative_throughput: Calculate relative throughput corrections.
    correct_wavelengths: Apply wavelength corrections to extracted spectra.
    construct_cube: Create 3D data cubes from RSS files.
    main: Main pipeline function that processes complete observations.

Example:
    Run the complete reduction pipeline::
    
        python reduce.py --config config.txt
"""

import os
import argparse
import pickle
import traceback
import multiprocessing
from datetime import datetime

from llamas_pyjamas.Trace.traceLlamasMaster import run_ray_tracing
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR, BIAS_DIR, LUT_DIR
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions
import llamas_pyjamas.GUI.guiExtract as ge
from llamas_pyjamas.File.llamasIO import process_fits_by_color
from llamas_pyjamas.File.llamasRSS import update_ra_dec_in_fits
import llamas_pyjamas.Arc.arcLlamasMulti as arc
from llamas_pyjamas.File.llamasRSS import RSSgeneration
from llamas_pyjamas.Utils.utils import count_trace_fibres, setup_logger
from llamas_pyjamas.Flat.flatLlamas import process_flat_field_complete
from llamas_pyjamas.Cube.cubeConstruct import CubeConstructor
from llamas_pyjamas.Bias.llamasBias import BiasLlamas
from llamas_pyjamas.Cube.crr_cube_constructor import CRRCubeConstructor, CRRCubeConfig
from llamas_pyjamas.Cube.rss_to_crr_adapter import load_rss_as_crr_data, combine_channels_for_crr
from llamas_pyjamas.DataModel.validate import validate_and_fix_extensions
from astropy.io import fits
import numpy as np


_linefile = os.path.join(LUT_DIR, '')





def generate_traces(red_flat, green_flat, blue_flat, output_dir, bias=None):
    """Generate fiber traces from flat field observations for all three channels.
    
    Args:
        red_flat: Path to red flat field FITS file.
        green_flat: Path to green flat field FITS file. 
        blue_flat: Path to blue flat field FITS file.
        output_dir: Directory to save trace files.
        bias: Optional bias frame for correction.
        
    Raises:
        AssertionError: If any of the flat field files do not exist.
    """
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    assert os.path.exists(red_flat), "Red flat file does not exist."
    assert os.path.exists(green_flat), "Green flat file does not exist."
    assert os.path.exists(blue_flat), "Blue flat file does not exist."

    run_ray_tracing(red_flat, outpath=output_dir, channel='red', use_bias=bias, is_master_calib=False)
    run_ray_tracing(green_flat, outpath=output_dir, channel='green', use_bias=bias, is_master_calib=False)
    run_ray_tracing(blue_flat, outpath=output_dir, channel='blue', use_bias=bias, is_master_calib=False)
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


def run_extraction(science_file, output_dir, use_bias=None, trace_dir=None, mastercalib_trace_dir=None):
    """
    Run spectrum extraction with hybrid trace support.

    Args:
        science_file: Path to science FITS file or list of paths
        output_dir: Output directory for extractions
        use_bias: Bias file for calibration
        trace_dir: User trace directory (for real extensions)
        mastercalib_trace_dir: Mastercalib trace directory (for placeholder extensions)

    Returns:
        str: Path to extraction file
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Default mastercalib location if not specified
    if mastercalib_trace_dir is None:
        mastercalib_trace_dir = CALIB_DIR

    assert os.path.exists(science_file), "Science file does not exist."
    if type(science_file) is list:
        for file in science_file:
            assert os.path.exists(file), f"Science file {file} does not exist."
            extraction_file_path = ge.GUI_extract(
                file,
                output_dir=output_dir,
                use_bias=use_bias,
                trace_dir=trace_dir,
                mastercalib_trace_dir=mastercalib_trace_dir
            )
    else:
        assert os.path.exists(science_file), "Science file does not exist."
        extraction_file_path, _ = ge.GUI_extract(
            science_file,
            output_dir=output_dir,
            use_bias=use_bias,
            trace_dir=trace_dir,
            mastercalib_trace_dir=mastercalib_trace_dir
        )

    return  extraction_file_path


#this isn't quite right -> nneeds checking
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
    print(f'std_wvcal metadata: {std_wvcal.get("metadata", {})}')

    return std_wvcal, primary_hdr


def process_flat_field_calibration(red_flat, green_flat, blue_flat, trace_dir, output_dir,
                                  arc_calib_file=None, verbose=False):
    """Generate flat field pixel maps for science frame correction.

    Args:
        red_flat (str): Path to red flat field FITS file
        green_flat (str): Path to green flat field FITS file
        blue_flat (str): Path to blue flat field FITS file
        trace_dir (str): Directory containing trace files
        output_dir (str): Output directory for flat field products
        arc_calib_file (str, optional): Path to arc calibration file
        verbose (bool): Enable verbose output
        flat_method (str): Flat fielding method - 'standard' or 'pypeit' (default: 'standard')

    Returns:
        list: List of flat field pixel map FITS file paths
    """
    # Use flat field output directory directly (no nested pixel_maps subdirectory)
    flat_output_dir = output_dir
    os.makedirs(flat_output_dir, exist_ok=True)

    print(f"  Red flat: {os.path.basename(red_flat)}")
    print(f"  Green flat: {os.path.basename(green_flat)}")
    print(f"  Blue flat: {os.path.basename(blue_flat)}")
    print(f"  Output directory: {flat_output_dir}")

    # Run the appropriate flat field workflow based on method
    try:
        
        from llamas_pyjamas.Flat.flatLlamas import process_flat_field_complete
        print("Using standard flat fielding approach")
        results = process_flat_field_complete(
            red_flat, green_flat, blue_flat,
            arc_calib_file=arc_calib_file,
            output_dir=flat_output_dir,
            trace_dir=trace_dir,
            verbose=verbose
        )


        # Standard method returns normalized flat field file
        normalized_flat = results.get('normalized_flat_field_file')
        pixel_map_files = [normalized_flat] if normalized_flat else []

        print(f"Successfully generated {len(pixel_map_files)} flat field pixel maps")
        # This should be using the normalised fits image
        return pixel_map_files

    except Exception as e:
        print(f"Error in flat field calibration: {str(e)}")
        import traceback
        traceback.print_exc()
        return []




def construct_cube(rss_files, output_dir, wavelength_range=None, dispersion=1.0, spatial_sampling=0.75, use_crr=True, crr_config=None, parallel=False):
    """
    Construct IFU data cubes from RSS files using either traditional or CRR method.
    
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
        use_crr (bool): Use CRR (Covariance-regularized Reconstruction) method
        crr_config (CRRCubeConfig, optional): CRR configuration parameters
        parallel (bool): Use parallel processing for CRR method
        
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
        
        # Check for channel-specific naming pattern (handle both old and new patterns)
        for color in ['red', 'green', 'blue']:
            if f'_extract_RSS_{color}' in base_name:
                # Extract the base name and channel color
                channel = color
                base_name = base_name.split(f'_extract_RSS_{color}')[0]
                logger.info(f"Detected {color} channel file, using base name: {base_name}")
                break
        
        # If no channel-specific pattern found, check for generic RSS pattern
        if not channel and '_RSS' in base_name:
            # This handles the new flat-corrected naming pattern: {base}_flat_corrected_RSS_{channel}.fits
            # or fallback pattern: {base}_RSS.fits (single file with all channels)
            if '_flat_corrected_RSS' in base_name:
                base_name = base_name.split('_flat_corrected_RSS')[0]
                logger.info(f"Detected flat-corrected RSS file, using base name: {base_name}")
            elif '_RSS' in base_name:
                base_name = base_name.split('_RSS')[0]
                logger.info(f"Detected generic RSS file, using base name: {base_name}")
        
        if channel:
            print(f"Processing {channel} channel RSS file with base name: {base_name}")
        else:
            print(f"Processing RSS file (no specific channel detected): {base_name}")
            
        if use_crr:
            # Use CRR (Covariance-regularized Reconstruction) method
            logger.info(f"Constructing CRR cube from RSS file: {rss_file}")
            print(f"Constructing CRR cube from RSS file: {rss_file}")
            
            try:
                # Convert RSS to CRR format
                rss_data = load_rss_as_crr_data(rss_file)
                
                # Create CRR configuration
                if crr_config is None:
                    crr_config = CRRCubeConfig(
                        pixel_scale=spatial_sampling,
                        use_sky_subtraction=False  # Sky subtraction handled in pipeline
                    )
                
                # Construct CRR cube
                if parallel:
                    from llamas_pyjamas.Cube.crr_parallel import parallel_cube_construction
                    logger.info("Using parallel CRR reconstruction")
                    crr_cube = parallel_cube_construction(rss_data, crr_config)
                else:
                    logger.info("Using serial CRR reconstruction")
                    constructor = CRRCubeConstructor(crr_config)
                    crr_cube = constructor.reconstruct_cube(rss_data)
                
                # Save CRR cube
                output_suffix = "_crr_cube"
                if parallel:
                    output_suffix += "_parallel"
                
                cube_filename = f"{base_name}{output_suffix}.fits"
                cube_path = os.path.join(output_dir, cube_filename)
                
                crr_cube.save_to_fits(cube_path)
                cube_files.append(cube_path)
                
                print(f"  - CRR cube saved: {cube_path}")
                logger.info(f"CRR cube saved: {cube_path}")
                
                # Log quality metrics
                if hasattr(crr_cube, 'quality_metrics') and crr_cube.quality_metrics:
                    coverage = crr_cube.quality_metrics.get('total_coverage_fraction', 0)
                    print(f"  - Coverage: {coverage:.1%}")
                    logger.info(f"CRR cube quality: {coverage:.1%} coverage")
                
            except Exception as e:
                logger.error(f"CRR cube construction failed: {e}")
                print(f"  ERROR: CRR cube construction failed: {e}")
                print("  Falling back to traditional cube construction...")
                
                # Fall back to traditional method
                use_crr = False
        
        if not use_crr:
            # Use traditional cube construction method
            logger.info(f"Constructing traditional channel cubes from RSS file: {rss_file}")
            print(f"Constructing traditional channel cubes from RSS file: {rss_file}")
            
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
    """
    Main entry point for the data reduction pipeline.
    This function reads a configuration file containing key-value pairs and processes the parameters to drive a series of data reduction operations. The configuration file is expected to include various settings such as file paths for flat-fields, science files, bias files, and other optional parameters. It sets up the necessary output directories (like the reduced data directory, traces, and extractions), performs trace generation, handles science file extractions, corrects wavelengths, and finally constructs a data cube.
    The configuration file format:
        - Each non-empty line should contain a key and a value separated by '='.
        - Lines starting with '#' are treated as comments and skipped.
        - Values may not be quoted, but can be comma-separated lists.
    Args:
        config_path (str): The file path to the configuration file.
    Raises:
        ValueError: If required configuration keys (e.g., 'arc_file' or 'science_files') are missing.
        FileNotFoundError: If specified science or extraction files do not exist.
        Exception: Propagates any other exceptions that occur during processing, with a traceback printed for debugging purposes.
    Examples:
        >>> main("/path/to/config.cfg")
        Loaded configuration from /path/to/config.cfg
        Configuration: {...}
    """
    print(f"Loading configuration from {config_path}")
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
                    
                # Handle boolean values
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                    
                config[key] = value
        
        
    print("Configuration:", config)

    #code to handle bias file input
    if 'bias_file' not in config:
        raise ValueError("No bias file provided in the configuration.")
    
    if isinstance(config['bias_file'], str):
        bias_file = config['bias_file']
        if 'bias_dir' in config:
            # If bias_file is just a basename, join with bias_dir
            if bias_file == os.path.basename(bias_file):
                bias_file = os.path.join(config['bias_dir'], bias_file)
            bias = BiasLlamas(bias_file)
        else:
            # bias_dir not provided; ensure bias_file is an absolute path
            if not os.path.isabs(bias_file):
                raise ValueError("Bias file is provided as a relative path and 'bias_dir' is missing in configuration.")
            bias = BiasLlamas(bias_file)

    elif isinstance(config['bias_file'], list):
        
        bias_list = config['bias_file']
        print(f"Using a list of bias files {bias_list}")
        if 'bias_dir' in config:
            bias_dir = config['bias_dir']
            updated_bias_list = []
            for b in bias_list:
                # If each bias file is provided as a basename, join with bias_dir
                if b == os.path.basename(b):
                    updated_bias_list.append(os.path.join(bias_dir, b))
                else:
                    updated_bias_list.append(b)
            bias_list = updated_bias_list
        else:
            # Ensure all bias file paths in the list are absolute
            for b in bias_list:
                if not os.path.isabs(b):
                    raise ValueError("One or more bias files are provided as relative paths and 'bias_dir' is missing in configuration.")
        
        
        bias = BiasLlamas(bias_list)
    bias_file = bias.master_bias()



        
    # Parse CRR cube configuration (defaults to True if not specified)
    use_crr_cube = config.get('CRR_cube', False)  # Default to True
    if isinstance(use_crr_cube, str):
        use_crr_cube = use_crr_cube.lower() == 'true'
        
    _gen_cubes = config.get('generate_cubes', True)  # Default to True

    print(f"CRR cube reconstruction: {'enabled' if use_crr_cube else 'disabled'}")

    # Configure Ray CPU usage globally
    ray_num_cpus = config.get('ray_num_cpus', multiprocessing.cpu_count())
    if isinstance(ray_num_cpus, str):
        ray_num_cpus = int(ray_num_cpus)
    os.environ['LLAMAS_RAY_CPUS'] = str(ray_num_cpus)
    print(f"Configuring pipeline to use {ray_num_cpus} Ray cores")


    if not config.get('output_dir'):
        output_dir = os.path.join(BASE_DIR, 'reduced')
    else:
        output_dir = config.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)

    ### Checking for arc file or master wavelength solution
        
    if bool(config.get('generate_new_wavelength_soln')) == True:
        print("Generating new wavelength solution.")
        extract_flat_field(config.get('flat_file_dir'), config.get('output_dir'), bias_file=bias_file)
        if 'arc_file' not in config:
            raise ValueError("No arc file provided in the configuration.")
        relative_throughput(config.get('shift_picklename'), config.get('flat_picklename'))
        arcdict = calc_wavelength_soln(config['arc_file'], config.get('output_dir'), bias=bias_file)
        config['arcdict'] = arcdict

    else:
        arcdict = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')
        if not os.path.exists(arcdict):
            raise FileNotFoundError(f"Reference arc file not found at {arcdict}")
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
    
    # Note: Pixel maps will be created in extractions/flat/ directory during flat field processing
    # No need to pre-create a separate pixel_maps directory
    try:
        
        # =====================================================================
        # CENTRALIZED TRACE DIRECTORY SELECTION
        # This is the SINGLE decision point for traces used throughout pipeline
        # =====================================================================

        print("\n" + "="*60)
        print("TRACE DIRECTORY SELECTION")
        print("="*60)

        final_trace_dir = None
        trace_source = None

        # Check if we should use existing traces or generate new ones
        if config.get('use_existing_traces', False) and os.path.exists(config.get('trace_output_dir')):
            # Check if trace files exist in the specified directory
            import glob
            existing_traces = glob.glob(os.path.join(config.get('trace_output_dir'), '*.pkl'))
            if existing_traces:
                print(f"Found {len(existing_traces)} existing trace files in {config.get('trace_output_dir')}")

                # Validate existing traces with per-trace fallback
                from llamas_pyjamas.Utils.utils import validate_and_fix_trace_fibres

                print("Validating existing traces...")
                validation_results = validate_and_fix_trace_fibres(
                    config.get('trace_output_dir'),
                    mastercalib_dir=CALIB_DIR
                )

                if validation_results['all_valid']:
                    print("✓ All existing traces validated successfully")
                    final_trace_dir = config.get('trace_output_dir')
                    trace_source = "existing_validated"
                else:
                    # Some traces invalid - fallbacks were copied
                    invalid_count = len(validation_results['invalid_traces'])
                    fallback_count = len(validation_results['fallback_used'])

                    print(f"⚠️  Found {invalid_count} trace(s) with incorrect fiber counts")
                    print(f"✓  Copied {fallback_count} mastercalib fallback trace(s)")

                    # Print details about invalid traces
                    for channel, bench, side, expected, actual in validation_results['invalid_traces']:
                        print(f"  ✗ {channel}{bench}{side}: Expected {expected} fibers, found {actual}")

                    # Print details about fallbacks used
                    for channel, bench, side, _master_path, _copied_path in validation_results['fallback_used']:
                        print(f"  ✓ {channel}{bench}{side}: Using mastercalib fallback (copied to trace_dir)")

                    # Still use trace_output_dir (now contains mix of user + mastercalib)
                    final_trace_dir = config.get('trace_output_dir')
                    trace_source = "existing_partial_fallback"

                    print(f"\n✓ Continuing with hybrid trace set ({len(validation_results['valid_traces'])} user + {fallback_count} mastercalib)")
            else:
                print("No existing trace files found in specified directory")
                final_trace_dir = None  # Will generate new or fallback
                trace_source = "existing_missing"
        else:
            print("Generating new traces...")

            # Validate flat field files before trace generation
            from llamas_pyjamas.DataModel.validate import validate_and_fix_extensions

            print("Validating flat field files...")
            red_flat_validated = validate_and_fix_extensions(
                config.get('red_flat_file'),
                output_file=None,  # Fix in-place
                backup=True
            )
            green_flat_validated = validate_and_fix_extensions(
                config.get('green_flat_file'),
                output_file=None,
                backup=True
            )
            blue_flat_validated = validate_and_fix_extensions(
                config.get('blue_flat_file'),
                output_file=None,
                backup=True
            )

            generate_traces(red_flat_validated, green_flat_validated, blue_flat_validated,
                           config.get('trace_output_dir'), bias=config.get('bias_file'))

            # Validate newly generated traces with per-trace fallback
            from llamas_pyjamas.Utils.utils import validate_and_fix_trace_fibres

            print("Validating newly generated traces...")
            validation_results = validate_and_fix_trace_fibres(
                config.get('trace_output_dir'),
                mastercalib_dir=CALIB_DIR
            )

            if validation_results['all_valid']:
                print("✓ Generated traces validated successfully")
                final_trace_dir = config.get('trace_output_dir')
                trace_source = "generated_validated"
            else:
                # Some traces invalid - fallbacks were copied
                invalid_count = len(validation_results['invalid_traces'])
                fallback_count = len(validation_results['fallback_used'])

                print(f"⚠️  Found {invalid_count} generated trace(s) with incorrect fiber counts")
                print(f"✓  Copied {fallback_count} mastercalib fallback trace(s)")

                # Print details about invalid traces
                for channel, bench, side, expected, actual in validation_results['invalid_traces']:
                    print(f"  ✗ {channel}{bench}{side}: Expected {expected} fibers, found {actual}")

                # Print details about fallbacks used
                for channel, bench, side, _master_path, _copied_path in validation_results['fallback_used']:
                    print(f"  ✓ {channel}{bench}{side}: Using mastercalib fallback (copied to trace_dir)")

                # Still use trace_output_dir (now contains mix of user + mastercalib)
                final_trace_dir = config.get('trace_output_dir')
                trace_source = "generated_partial_fallback"

                print(f"\n✓ Continuing with hybrid trace set ({len(validation_results['valid_traces'])} generated + {fallback_count} mastercalib)")

        # Final fallback to mastercalib if needed
        if final_trace_dir is None:
            print(f"\n⚠️  FALLBACK: Using mastercalib traces from {CALIB_DIR}")

            # Validate mastercalib traces exist and are valid
            if count_trace_fibres(CALIB_DIR):
                final_trace_dir = CALIB_DIR
                trace_source = "mastercalib"
                print("✓ Mastercalib traces validated successfully")
            else:
                raise RuntimeError(f"FATAL: Even mastercalib traces in {CALIB_DIR} are invalid or missing!")

        # Update config to reflect final decision
        config['trace_output_dir'] = final_trace_dir

        # Clear status reporting
        print("\n" + "="*60)
        print("TRACE SELECTION FINAL DECISION")
        print("="*60)
        print(f"Selected trace directory: {final_trace_dir}")
        print(f"Trace source: {trace_source}")
        print(f"All pipeline steps will use traces from: {final_trace_dir}")
        print("="*60)
        
        ################################################
        ############ FLAT FIELD PROCESSING #############
        ################################################


        #function to concat the flat field file extensions
        # function to extract them all
        # process to apply wvln solution to the combined extractions
        # function to generate pixel maps from the combined extractions
        if config.get('apply_flat_field_correction', False):
            try:
                # I might need to check the arc file being used here
                flat_pixel_maps = process_flat_field_complete(red_flat_file=config.get('red_flat_file'),
                                            green_flat_file=config.get('green_flat_file'),
                                            blue_flat_file=config.get('blue_flat_file'),
                                            arc_calib_file=config.get('arc_calib_file'),
                                            output_dir=os.path.join(extraction_path, 'flat_temp'),
                                            trace_dir=config.get('trace_output_dir'),
                                            verbose=False)
            except Exception as e:
                print(f"Error during preliminary flat field processing: {str(e)}")
                import traceback
                traceback.print_exc()

        if flat_pixel_maps:
            print(f"\nGenerated {len(flat_pixel_maps)} flat field pixel maps:")
        else:
            print("WARNING: No flat field pixel maps generated. Proceeding without flat field correction.")
        

        
       # Apply flat field corrections to science files before extraction
        # First, validate all science files for missing extensions
        

        print("\n" + "="*60)
        print("VALIDATING SCIENCE FILES")
        print("="*60)

        validated_science_files = []
        if isinstance(config['science_files'], list):
            for science_file in config['science_files']:
                print(f"Validating: {os.path.basename(science_file)}")
                validated_file = validate_and_fix_extensions(
                    science_file,
                    output_file=None,  # Fix in-place
                    backup=True
                )
                validated_science_files.append(validated_file)
        else:
            print(f"Validating: {os.path.basename(config['science_files'])}")
            validated_file = validate_and_fix_extensions(
                config['science_files'],
                output_file=None,
                backup=True
            )
            validated_science_files = validated_file

        science_files_to_process = validated_science_files

        if config.get('apply_flat_field_correction', True) and flat_pixel_maps:
            print("\n" + "="*60)
            print("APPLYING FLAT FIELD CORRECTIONS")
            print("="*60)
            
            flat_corrected_files = []
            # Output flat-corrected files directly to main output directory to avoid unnecessary subdirectories
            flat_output_dir = output_dir
            
            overall_stats = {'total_corrected': 0, 'total_skipped': 0, 'total_errors': 0}
            
            if isinstance(config['science_files'], list):
                for i, science_file in enumerate(config['science_files']):
                    print(f"\nFlat-correcting science file {i+1}/{len(config['science_files'])}:")
                    print(f"Input: {os.path.basename(science_file)}")
                    
                    if not os.path.exists(science_file):
                        raise FileNotFoundError(f"Science file {science_file} does not exist.")
                    
                    # corrected_file, stats = apply_flat_field_correction(
                    #     science_file, 
                    #     flat_pixel_maps, 
                    #     flat_output_dir,
                    #     validate_matching=config.get('validate_flat_matching', True),
                    #     require_all_matches=config.get('require_all_flat_matches', True)
                    # )
                    
                    if corrected_file:
                        flat_corrected_files.append(corrected_file)
                        # Accumulate statistics
                        overall_stats['total_corrected'] += stats['corrected']
                        overall_stats['total_skipped'] += stats['skipped'] 
                        overall_stats['total_errors'] += stats['errors']
                    else:
                        print(f"ERROR: Failed to flat-correct {science_file}")
                        # Use original file as fallback
                        flat_corrected_files.append(science_file)
                        
                science_files_to_process = flat_corrected_files
            else:
                # Single science file
                science_file = config['science_files']
                if not os.path.exists(science_file):
                    raise FileNotFoundError(f"Science file {science_file} does not exist.")
                
                print(f"\nFlat-correcting science file: {os.path.basename(science_file)}")
                # corrected_file, stats = apply_flat_field_correction(
                #     science_file,
                #     flat_pixel_maps,
                #     flat_output_dir,
                #     validate_matching=config.get('validate_flat_matching', True),
                #     require_all_matches=config.get('require_all_flat_matches', False)
                # )
                
                if corrected_file:
                    science_files_to_process = corrected_file
                    overall_stats['total_corrected'] = stats['corrected']
                    overall_stats['total_skipped'] = stats['skipped']
                    overall_stats['total_errors'] = stats['errors']
                else:
                    print(f"ERROR: Failed to flat-correct {science_file}, using original")
                    science_files_to_process = science_file
            
            print("\n" + "="*60)
            print("FLAT FIELD CORRECTION SUMMARY")
            print("="*60)
            files_processed = len(config['science_files']) if isinstance(config['science_files'], list) else 1
            print(f"Files processed: {files_processed}")
            print(f"Total extensions corrected: {overall_stats['total_corrected']}")
            print(f"Total extensions skipped: {overall_stats['total_skipped']}")
            print(f"Total extensions with errors: {overall_stats['total_errors']}")
        
        # Process science files (now potentially flat-corrected) for extraction
        if 'science_files' not in config:
            raise ValueError("No science files provided in the configuration.")
    
        # Track whether files were flat-corrected
        were_flat_corrected = config.get('apply_flat_field_correction', True) and flat_pixel_maps and len(flat_pixel_maps) > 0
        
        if isinstance(science_files_to_process, list):
            print(f'\nFound {len(science_files_to_process)} science files to process for extraction.')

            for i, science_file in enumerate(science_files_to_process):
                print(f"Extracting science file {i+1}/{len(science_files_to_process)}: {os.path.basename(science_file)}")
                # Process each science file with hybrid trace support
                extracted_file = run_extraction(
                    science_file,
                    extraction_path,
                    use_bias=config.get('bias_file'),
                    trace_dir=final_trace_dir,              # User traces
                    mastercalib_trace_dir=CALIB_DIR         # Mastercalib fallback
                )
                print(f"Extraction completed for {os.path.basename(science_file)}. Output file: {extracted_file}")
        else:
            print(f"Extracting science file: {os.path.basename(science_files_to_process)}")
            extracted_file = run_extraction(
                science_files_to_process,
                extraction_path,
                use_bias=config.get('bias_file'),
                trace_dir=final_trace_dir,                  # User traces
                mastercalib_trace_dir=CALIB_DIR             # Mastercalib fallback
            )
            print(f"Extraction completed. Used traces from {final_trace_dir} with mastercalib fallback. Output file: {extracted_file}")

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
            
            # Construct RSS output filename with flat correction traceability
            flat_suffix = '_flat_corrected' if were_flat_corrected else ''
            rss_output_file = os.path.join(extraction_path, f'{base_name}{flat_suffix}_RSS.fits')
            
            if were_flat_corrected:
                rss_logger.info(f"Science data was flat-field corrected - RSS files will include '_flat_corrected' suffix")
                print(f"Science data was flat-field corrected - RSS files will include '_flat_corrected' suffix")
            
            #RSS generation
            rss_gen = RSSgeneration(logger=rss_logger)
            new_rss_outputs = rss_gen.generate_rss(savefile, rss_output_file)
            rss_logger.info(f"RSS file generated: {new_rss_outputs}")
            print(f"RSS file generated: {new_rss_outputs}")

        # Updating RA and Dec in RSS files
        for rss_output_file in new_rss_outputs:
            update_ra_dec_in_fits(rss_output_file, logger=rss_logger)

        # Cube construction from RSS files
        print("Constructing cubes from RSS files...")
        # Look for RSS files (both flat-corrected and uncorrected patterns)
        rss_files = [os.path.join(extraction_path, f) for f in os.listdir(extraction_path) 
                if ('extract_RSS' in f or '_RSS' in f) and f.endswith('.fits')]
        
        if rss_files:
            print(f"Found {len(rss_files)} RSS files for cube construction:")
            for rss_file in rss_files:
                is_flat_corrected = '_flat_corrected' in os.path.basename(rss_file)
                status = " (flat-corrected)" if is_flat_corrected else " (original)"
                print(f"  - {os.path.basename(rss_file)}{status}")
        else:
            print(f"Found {len(rss_files)} RSS files for cube construction")

        if 'cube_output_dir' not in config:
            cube_output_dir = os.path.join(output_dir, 'cubes')
        else:
            cube_output_dir = config.get('cube_output_dir')
        os.makedirs(cube_output_dir, exist_ok=True)


        if rss_files and _gen_cubes:
            cube_files = construct_cube(
                rss_files, 
                cube_output_dir,
                wavelength_range=config.get('wavelength_range'),
                dispersion=config.get('dispersion', 1.0),
                spatial_sampling=config.get('spatial_sampling', 0.75),
                use_crr=use_crr_cube,
                parallel=config.get('CRR_parallel', False)
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