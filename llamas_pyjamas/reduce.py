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
import llamas_pyjamas.Arc.arcLlamas as arc
from llamas_pyjamas.File.llamasRSS import RSSgeneration
from llamas_pyjamas.Utils.utils import count_trace_fibres, setup_logger
from llamas_pyjamas.Cube.cubeConstruct import CubeConstructor
from llamas_pyjamas.Cube.crr_cube_constructor import CRRCubeConstructor, CRRCubeConfig
from llamas_pyjamas.Cube.rss_to_crr_adapter import load_rss_as_crr_data, combine_channels_for_crr
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
    print(f'std_wvcal metadata: {std_wvcal.get('metadata', {})}')
    
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
        
    Returns:
        list: List of flat field pixel map FITS file paths
    """
    from llamas_pyjamas.Flat.flatLlamas import process_flat_field_complete
    
    # Use flat field output directory directly (no nested pixel_maps subdirectory)
    flat_output_dir = output_dir
    os.makedirs(flat_output_dir, exist_ok=True)
    
    print(f"Processing flat field calibration...")
    print(f"  Red flat: {os.path.basename(red_flat)}")
    print(f"  Green flat: {os.path.basename(green_flat)}")
    print(f"  Blue flat: {os.path.basename(blue_flat)}")
    print(f"  Output directory: {flat_output_dir}")
    
    # Run the complete flat field workflow
    try:
        results = process_flat_field_complete(
            red_flat, green_flat, blue_flat,
            arc_calib_file=arc_calib_file,
            output_dir=flat_output_dir,
            trace_dir=trace_dir,
            verbose=verbose
        )
        
        pixel_map_files = results.get('output_files', [])
        print(f"Successfully generated {len(pixel_map_files)} flat field pixel maps")
        
        return pixel_map_files
        
    except Exception as e:
        print(f"Error in flat field calibration: {str(e)}")
        import traceback
        traceback.print_exc()
        return []


def build_flat_field_map(flat_pixel_maps, science_file):
    """
    Build mapping between science extensions and corresponding flat field pixel maps.
    
    Args:
        flat_pixel_maps (list): List of flat field pixel map file paths
        science_file (str): Path to science FITS file
        
    Returns:
        dict: {extension_index: {'flat_file': path, 'match_info': details}}
    """
    flat_map = {}
    
    print(f"Building flat field mapping for: {os.path.basename(science_file)}")
    print(f"Available flat field pixel maps: {len(flat_pixel_maps)}")
    
    # Open science file to get extension headers
    try:
        with fits.open(science_file) as sci_hdul:
            for ext_idx in range(1, len(sci_hdul)):  # Skip primary HDU
                sci_hdu = sci_hdul[ext_idx]
                
                # Extract science extension metadata
                sci_channel = sci_hdu.header.get('COLOR', '').lower()
                sci_bench = sci_hdu.header.get('BENCH', '')
                sci_side = sci_hdu.header.get('SIDE', '')
                
                # Alternative naming if COLOR not present
                if not sci_channel:
                    cam_name = sci_hdu.header.get('CAM_NAME', '')
                    if cam_name:
                        parts = cam_name.split('_')
                        if len(parts) >= 2:
                            sci_channel = parts[1].lower()
                            if len(parts[0]) >= 2:
                                sci_bench = parts[0][0]
                                sci_side = parts[0][1]
                
                # Build expected flat field key
                sci_key = f"{sci_channel}{sci_bench}{sci_side}"
                
                # Find matching flat field pixel map
                matching_flat = None
                best_match_score = 0
                
                for flat_file in flat_pixel_maps:
                    flat_basename = os.path.basename(flat_file)
                    
                    # Check if flat file contains the science key
                    if sci_key.lower() in flat_basename.lower():
                        # Try to verify match by reading flat file header
                        try:
                            with fits.open(flat_file) as flat_hdul:
                                flat_channel = flat_hdul[0].header.get('CHANNEL', '').lower()
                                flat_bench = flat_hdul[0].header.get('BENCH', '')
                                flat_side = flat_hdul[0].header.get('SIDE', '')
                                
                                # Calculate match score
                                match_score = 0
                                if flat_channel == sci_channel:
                                    match_score += 3
                                if flat_bench == sci_bench:
                                    match_score += 2  
                                if flat_side == sci_side:
                                    match_score += 1
                                
                                # Perfect match verification
                                if (flat_channel == sci_channel and 
                                    flat_bench == sci_bench and 
                                    flat_side == sci_side):
                                    matching_flat = flat_file
                                    best_match_score = match_score
                                    break
                                elif match_score > best_match_score:
                                    # Keep track of best partial match
                                    matching_flat = flat_file
                                    best_match_score = match_score
                                    
                        except Exception as e:
                            print(f"Warning: Could not read flat file {flat_file}: {str(e)}")
                            continue
                    
                    # Fallback: try pattern matching in filename
                    elif not matching_flat:
                        # Look for individual components in filename
                        name_lower = flat_basename.lower()
                        component_matches = 0
                        if sci_channel in name_lower:
                            component_matches += 1
                        if sci_bench.lower() in name_lower:
                            component_matches += 1
                        if sci_side.lower() in name_lower:
                            component_matches += 1
                            
                        if component_matches >= 2:  # At least 2 out of 3 match
                            matching_flat = flat_file
                            best_match_score = component_matches
                
                # Store mapping result
                flat_map[ext_idx] = {
                    'flat_file': matching_flat,
                    'science_key': sci_key,
                    'science_channel': sci_channel,
                    'science_bench': sci_bench, 
                    'science_side': sci_side,
                    'match_found': matching_flat is not None,
                    'match_score': best_match_score
                }
    
    except Exception as e:
        print(f"Error reading science file {science_file}: {str(e)}")
        return {}
    
    return flat_map


def validate_flat_field_matching(flat_map, science_file):
    """
    Validate flat field matching and provide detailed reporting.
    
    Args:
        flat_map (dict): Flat field mapping from build_flat_field_map()
        science_file (str): Path to science FITS file
        
    Returns:
        dict: {'valid': bool, 'report': str, 'warnings': list, 'stats': dict}
    """
    total_extensions = len(flat_map)
    matched_extensions = sum(1 for v in flat_map.values() if v['match_found'])
    perfect_matches = sum(1 for v in flat_map.values() if v.get('match_score', 0) >= 6)
    
    validation_report = []
    warnings = []
    
    validation_report.append(f"Flat Field Validation for: {os.path.basename(science_file)}")
    validation_report.append(f"Total extensions: {total_extensions}")
    validation_report.append(f"Matched extensions: {matched_extensions}")
    validation_report.append(f"Perfect matches: {perfect_matches}")
    validation_report.append(f"Unmatched extensions: {total_extensions - matched_extensions}")
    validation_report.append("-" * 60)
    
    # Group extensions by channel for better reporting
    channels = {}
    for ext_idx, mapping in flat_map.items():
        channel = mapping.get('science_channel', 'unknown')
        if channel not in channels:
            channels[channel] = {'matched': 0, 'total': 0, 'extensions': []}
        channels[channel]['total'] += 1
        channels[channel]['extensions'].append((ext_idx, mapping))
        if mapping['match_found']:
            channels[channel]['matched'] += 1
    
    # Report by channel
    for channel, info in sorted(channels.items()):
        validation_report.append(f"\n{channel.upper()} Channel: {info['matched']}/{info['total']} matched")
        for ext_idx, mapping in sorted(info['extensions']):
            if mapping['match_found']:
                flat_name = os.path.basename(mapping['flat_file'])
                score = mapping.get('match_score', 0)
                score_str = f"(score: {score})" if score < 6 else "(perfect)"
                validation_report.append(
                    f"  ✓ Extension {ext_idx:2d} ({mapping['science_key']}): {flat_name} {score_str}"
                )
            else:
                validation_report.append(
                    f"  ✗ Extension {ext_idx:2d} ({mapping['science_key']}): No matching flat found"
                )
                warnings.append(f"Extension {ext_idx} ({mapping['science_key']}) has no flat field correction")
    
    # Check for critical channels missing
    critical_missing = []
    critical_channels = ['red', 'green', 'blue']
    
    for ext_idx, mapping in flat_map.items():
        if (not mapping['match_found'] and 
            mapping['science_channel'] in critical_channels):
            critical_missing.append(mapping['science_key'])
    
    # Determine validity
    coverage_ratio = matched_extensions / total_extensions if total_extensions > 0 else 0
    is_valid = coverage_ratio >= 0.5 and len(critical_missing) == 0  # At least 50% coverage and no critical missing
    
    if critical_missing:
        warnings.append(f"Critical channels missing flat corrections: {critical_missing}")
    
    if coverage_ratio < 0.8:
        warnings.append(f"Low flat field coverage: {coverage_ratio:.1%}")
    
    stats = {
        'total_count': total_extensions,
        'matched_count': matched_extensions,
        'perfect_matches': perfect_matches,
        'coverage_ratio': coverage_ratio,
        'critical_missing': len(critical_missing)
    }
    
    return {
        'valid': is_valid,
        'report': '\n'.join(validation_report),
        'warnings': warnings,
        'stats': stats
    }


def apply_flat_field_correction(science_file, flat_pixel_maps, output_dir, 
                               validate_matching=True, require_all_matches=False):
    """
    Apply flat field corrections to a science frame with robust cross-checking.
    
    Args:
        science_file (str): Path to science FITS file
        flat_pixel_maps (list): List of flat field pixel map files
        output_dir (str): Output directory for corrected files
        validate_matching (bool): Whether to validate matching before processing
        require_all_matches (bool): Whether to require all extensions to have matches
        
    Returns:
        tuple: (output_file_path, correction_statistics)
    """
    
    print(f"Applying flat field correction to: {os.path.basename(science_file)}")
    
    # Step 1: Build flat field mapping with cross-checking
    flat_map = build_flat_field_map(flat_pixel_maps, science_file)
    
    if not flat_map:
        print("ERROR: Could not build flat field mapping")
        return None, {'corrected': 0, 'skipped': 0, 'errors': 1}
    
    # Step 2: Validate matching if requested
    if validate_matching:
        validation = validate_flat_field_matching(flat_map, science_file)
        print("\n" + validation['report'])
        
        if validation['warnings']:
            for warning in validation['warnings']:
                print(f"WARNING: {warning}")
        
        if require_all_matches and not validation['valid']:
            raise ValueError("Not all extensions have matching flat fields and require_all_matches=True")
        
        print(f"\nProceeding with {validation['stats']['matched_count']}/{validation['stats']['total_count']} extensions matched")
        print("=" * 60)
    
    # Step 3: Apply corrections with verified matching
    try:
        with fits.open(science_file) as sci_hdul:
            corrected_hdus = [sci_hdul[0].copy()]  # Primary header
            
            correction_stats = {'corrected': 0, 'skipped': 0, 'errors': 0}
            
            for ext_idx in range(1, len(sci_hdul)):
                sci_hdu = sci_hdul[ext_idx]
                mapping = flat_map.get(ext_idx, {})
                
                if mapping.get('match_found') and sci_hdu.data is not None:
                    try:
                        # Apply flat field correction with verified matching
                        with fits.open(mapping['flat_file']) as flat_hdul:
                            flat_data = flat_hdul[0].data
                            
                            # Shape verification
                            if flat_data.shape != sci_hdu.data.shape:
                                print(f"  ERROR: Shape mismatch for extension {ext_idx} "
                                      f"(sci: {sci_hdu.data.shape}, flat: {flat_data.shape})")
                                raise ValueError(f"Shape mismatch: sci={sci_hdu.data.shape}, flat={flat_data.shape}")
                            
                            # Check for valid flat field data
                            valid_flat_pixels = np.isfinite(flat_data) & (flat_data > 0)
                            if not np.any(valid_flat_pixels):
                                print(f"  ERROR: No valid flat field data for extension {ext_idx}")
                                raise ValueError("No valid flat field data")
                            
                            # Perform division with NaN protection
                            corrected_data = np.divide(
                                sci_hdu.data.astype(np.float32),
                                flat_data.astype(np.float32),
                                out=np.full_like(sci_hdu.data, np.nan, dtype=np.float32),
                                where=flat_data > 0
                            )
                            
                            # Create new HDU with corrected data
                            new_hdu = fits.ImageHDU(
                                data=corrected_data,
                                header=sci_hdu.header.copy()
                            )
                            
                            # Add correction metadata
                            new_hdu.header['FLATCORR'] = (True, 'Flat field corrected')
                            new_hdu.header['FLATFILE'] = (os.path.basename(mapping['flat_file']), 'Flat field file used')
                            new_hdu.header['FLATKEY'] = (mapping['science_key'], 'Flat field matching key')
                            new_hdu.header['FLATSCORE'] = (mapping.get('match_score', 0), 'Flat field match score')
                            
                            # Add statistics
                            valid_pixels = np.isfinite(corrected_data)
                            if np.any(valid_pixels):
                                new_hdu.header['FLATMEAN'] = (float(np.nanmean(corrected_data)), 'Mean after flat correction')
                                new_hdu.header['FLATMED'] = (float(np.nanmedian(corrected_data)), 'Median after flat correction')
                                new_hdu.header['FLATVPIX'] = (int(np.sum(valid_pixels)), 'Valid pixels after flat correction')
                            
                            corrected_hdus.append(new_hdu)
                            correction_stats['corrected'] += 1
                            
                            print(f"  ✓ Extension {ext_idx:2d} ({mapping['science_key']}): Corrected with {os.path.basename(mapping['flat_file'])}")
                            
                    except Exception as e:
                        print(f"  ✗ Extension {ext_idx:2d} ({mapping.get('science_key', 'UNKNOWN')}): Error - {str(e)}")
                        # Copy original on error
                        new_hdu = sci_hdu.copy()
                        new_hdu.header['FLATCORR'] = (False, f'Flat correction failed: {str(e)}')
                        new_hdu.header['FLATKEY'] = (mapping.get('science_key', 'UNKNOWN'), 'Failed flat correction')
                        corrected_hdus.append(new_hdu)
                        correction_stats['errors'] += 1
                        
                else:
                    # No flat correction applied
                    new_hdu = sci_hdu.copy()
                    new_hdu.header['FLATCORR'] = (False, 'No matching flat field found')
                    if mapping:
                        new_hdu.header['FLATKEY'] = (mapping.get('science_key', 'UNKNOWN'), 'No matching flat found')
                    corrected_hdus.append(new_hdu)
                    correction_stats['skipped'] += 1
                    print(f"  - Extension {ext_idx:2d} ({mapping.get('science_key', 'UNKNOWN')}): Skipped (no flat)")
            
            # Save corrected science frame
            base_name = os.path.splitext(os.path.basename(science_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_flat_corrected.fits")
            
            corrected_hdul = fits.HDUList(corrected_hdus)
            
            # Add summary to primary header
            corrected_hdul[0].header['FLATSTAT'] = (
                f"{correction_stats['corrected']}C/{correction_stats['skipped']}S/{correction_stats['errors']}E",
                'Flat correction stats: Corrected/Skipped/Errors'
            )
            corrected_hdul[0].header['FLATTIME'] = (
                datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
                'Flat field correction timestamp'
            )
            
            corrected_hdul.writeto(output_file, overwrite=True)
            
            print(f"\nFlat field correction summary:")
            print(f"  - Corrected: {correction_stats['corrected']} extensions")
            print(f"  - Skipped: {correction_stats['skipped']} extensions") 
            print(f"  - Errors: {correction_stats['errors']} extensions")
            print(f"  - Output file: {os.path.basename(output_file)}")
            
            return output_file, correction_stats
            
    except Exception as e:
        print(f"ERROR: Failed to process science file {science_file}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, {'corrected': 0, 'skipped': 0, 'errors': 1}


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
                    
                # Handle boolean values
                elif value.lower() in ('true', 'false'):
                    value = value.lower() == 'true'
                    
                config[key] = value
        
        print(f"Loaded configuration from {config_path}")
    print("Configuration:", config)

    # Parse CRR cube configuration (defaults to True if not specified)
    use_crr_cube = config.get('CRR_cube', True)  # Default to True
    if isinstance(use_crr_cube, str):
        use_crr_cube = use_crr_cube.lower() == 'true'
    
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

                # Validate existing traces have correct fiber counts
                print("Validating existing traces...")
                if count_trace_fibres(config.get('trace_output_dir')):
                    final_trace_dir = config.get('trace_output_dir')
                    trace_source = "existing_validated"
                    print("✓ Existing traces validated successfully")
                else:
                    print("✗ Existing traces have incorrect fiber counts")
                    final_trace_dir = None  # Will fallback to mastercalib
                    trace_source = "existing_failed"
            else:
                print("No existing trace files found in specified directory")
                final_trace_dir = None  # Will generate new or fallback
                trace_source = "existing_missing"
        else:
            print("Generating new traces...")
            generate_traces(config.get('red_flat_file'), config.get('green_flat_file'), config.get('blue_flat_file'),
                           config.get('trace_output_dir'), bias=config.get('bias_file'))

            # Validate newly generated traces
            print("Validating newly generated traces...")
            if count_trace_fibres(config.get('trace_output_dir')):
                final_trace_dir = config.get('trace_output_dir')
                trace_source = "generated_validated"
                print("✓ Generated traces validated successfully")
            else:
                print("✗ Generated traces have incorrect fiber counts")
                final_trace_dir = None  # Will fallback to mastercalib
                trace_source = "generated_failed"

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
        
        # Generate flat field pixel maps if flat correction is enabled
        flat_pixel_maps = []
        if config.get('apply_flat_field_correction', True):  # Default to True
            print("\n" + "="*60)
            print("FLAT FIELD PROCESSING")
            print("="*60)
            
            # Create proper flat field directory structure: extractions/flat/
            flat_field_dir = config.get('flat_field_output_dir', os.path.join(extraction_path, 'flat'))
            os.makedirs(flat_field_dir, exist_ok=True)
            
            flat_pixel_maps = process_flat_field_calibration(
                config.get('red_flat_file'), 
                config.get('green_flat_file'), 
                config.get('blue_flat_file'),
                config.get('trace_output_dir'),
                flat_field_dir,
                arc_calib_file=config.get('arc_calib_file'),
                verbose=config.get('verbose_flat_processing', False)
            )
            
            if flat_pixel_maps:
                print(f"\nGenerated {len(flat_pixel_maps)} flat field pixel maps:")
                for flat_file in flat_pixel_maps:
                    print(f"  - {os.path.basename(flat_file)}")
            else:
                print("WARNING: No flat field pixel maps generated. Proceeding without flat field correction.")
        
        
       # Apply flat field corrections to science files before extraction
        science_files_to_process = config['science_files']
        
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
                    
                    corrected_file, stats = apply_flat_field_correction(
                        science_file, 
                        flat_pixel_maps, 
                        flat_output_dir,
                        validate_matching=config.get('validate_flat_matching', True),
                        require_all_matches=config.get('require_all_flat_matches', False)
                    )
                    
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
                corrected_file, stats = apply_flat_field_correction(
                    science_file,
                    flat_pixel_maps,
                    flat_output_dir,
                    validate_matching=config.get('validate_flat_matching', True),
                    require_all_matches=config.get('require_all_flat_matches', False)
                )
                
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
                # Process each science file by color
                extracted_file = run_extraction(science_file, extraction_path, use_bias=config.get('bias_file'), trace_dir=config.get('trace_output_dir'))
                print(f"Extraction completed for {os.path.basename(science_file)}. Output file: {extracted_file}")
        else:
            print(f"Extracting science file: {os.path.basename(science_files_to_process)}")
            extracted_file = run_extraction(science_files_to_process, extraction_path, use_bias=config.get('bias_file'), trace_dir=config.get('trace_output_dir'))
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


        if rss_files:
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