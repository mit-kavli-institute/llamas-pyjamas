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
import logging
import gc
from datetime import datetime

from llamas_pyjamas.Trace.traceLlamasMaster import run_ray_tracing
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR, BIAS_DIR, LUT_DIR, SLOW_BIAS_FILE, FAST_BIAS_FILE
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions
import llamas_pyjamas.GUI.guiExtract as ge
from llamas_pyjamas.File.llamasIO import process_fits_by_color
from llamas_pyjamas.File.llamasRSS import update_ra_dec_in_fits
import llamas_pyjamas.Arc.arcLlamasMulti as arc
from llamas_pyjamas.File.llamasRSS import RSSgeneration
from llamas_pyjamas.Utils.utils import count_trace_fibres, check_header, configure_pipeline_logging, setup_logger
from llamas_pyjamas.Cube.cubeConstruct import CubeConstructor
from llamas_pyjamas.Bias.llamasBias import BiasLlamas
from llamas_pyjamas.Cube.crr_cube_constructor import CRRCubeConstructor, CRRCubeConfig
from llamas_pyjamas.Cube.rss_to_crr_adapter import load_rss_as_crr_data, combine_channels_for_crr
from llamas_pyjamas.Cube.simple_cube_constructor import SimpleCubeConstructor
from astropy.io import fits
import numpy as np

import shutil

_cached_reference_arc = None

logger = logging.getLogger(__name__)

from llamas_pyjamas.DataModel.validate import validate_and_fix_extensions, get_placeholder_extension_indices, validate_for_gui
from llamas_pyjamas.Flat.flatLlamas import process_flat_field_complete, process_pixel_flat_simple
from llamas_pyjamas.Flat.fibreFlat import (compute_fibre_flat_lamp_only,
                                           compute_fibre_flat_twilight,
                                           reduce_twilight_flat,
                                           apply_fibre_flat_to_rss)

_linefile = os.path.join(LUT_DIR, '')


def validate_input_files(file_list):

    for file in file_list:
        assert os.path.exists(file), f"Input file does not exist: {file}"
        validate_and_fix_extensions(file, 
                                    output_file=None, backup=True)

    return True

def get_input_files_from_config(config, file_keys=None):
      """
      Extract all input file paths from a configuration dictionary.
      
      Parameters
      ----------
      config : dict
          Configuration dictionary with file paths
      file_keys : list, optional
          List of config keys that contain file paths. If None, uses default keys.
          
      Returns
      -------
      list
          List of all file paths found in the config
      """
      if file_keys is None:
          file_keys = ['science_files', 'slow_bias_file', 'fast_bias_file',
                       'red_flat_file', 'green_flat_file', 'blue_flat_file',
                       'twilight_flat']

      input_files = []

      for key in file_keys:
          if key in config:
              value = config[key]
              # If it's a list, extend the input_files list
              if isinstance(value, list):
                  input_files.extend(value)
              # If it's a single string, append it
              elif isinstance(value, str):
                  input_files.append(value)

      return input_files


def validate_pipeline_config(config: dict, config_path: str) -> bool:
    """
    Pre-flight validation of the pipeline configuration.

    Checks package directories, required master bias files, required config keys,
    and file/directory path existence before any processing begins.  On failure,
    prints a clear itemised summary and calls sys.exit(1).

    Parameters
    ----------
    config : dict
        Parsed configuration dictionary from the config .txt file.
    config_path : str
        Path to the config file (used in error messages for context).

    Returns
    -------
    bool
        True if all checks pass (warnings are allowed).
    """
    import sys

    errors = []
    warnings = []

    print("\n" + "=" * 60)
    print("PIPELINE PRE-FLIGHT CHECK")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Package directory checks — fail immediately if missing
    # ------------------------------------------------------------------
    if not os.path.isdir(CALIB_DIR):
        errors.append(
            f"Package mastercalib directory not found: '{CALIB_DIR}'. "
            f"This directory is required for trace fallbacks."
        )
    if not os.path.isdir(LUT_DIR):
        errors.append(
            f"Package LUT directory not found: '{LUT_DIR}'. "
            f"This directory is required for the reference arc and trace LUT."
        )

    if errors:
        for e in errors:
            print(f"  \u2717  ERROR: {e}")
        print(f"\nPipeline pre-flight check FAILED — {len(errors)} error(s).")
        print("Fix the above issues and re-run.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 2. Master bias file checks
    # ------------------------------------------------------------------
    # Check packaged default locations
    for label, path in [('slow_master_bias', SLOW_BIAS_FILE),
                         ('fast_master_bias', FAST_BIAS_FILE)]:
        if not os.path.isfile(path):
            errors.append(
                f"Required master bias not found: '{path}'. "
                f"Create it from raw bias frames using BiasLlamas and place it in the Bias/ directory, "
                f"or set 'slow_bias_file'/'fast_bias_file' in your config to a custom path."
            )

    # Check user-supplied overrides if present
    for key in ('slow_bias_file', 'fast_bias_file'):
        if key in config:
            p = config[key]
            if not os.path.isfile(p):
                errors.append(f"'{key}' path does not exist: '{p}'")

    # ------------------------------------------------------------------
    # 3. Required config keys
    # ------------------------------------------------------------------
    required_keys = ['science_files', 'red_flat_file', 'green_flat_file', 'blue_flat_file']
    for key in required_keys:
        if key not in config:
            errors.append(f"Required key '{key}' is missing from config.")

    if config.get('generate_new_wavelength_soln') is True and 'arc_file' not in config:
        errors.append(
            "'generate_new_wavelength_soln' is true but 'arc_file' is not set in config."
        )

    # ------------------------------------------------------------------
    # 4. File path existence checks
    # ------------------------------------------------------------------
    file_keys = ['science_files', 'red_flat_file', 'green_flat_file', 'blue_flat_file',
                 'twilight_flat']
    if config.get('generate_new_wavelength_soln') is True:
        file_keys.append('arc_file')

    for key in file_keys:
        if key not in config:
            continue  # already caught by required check or truly optional
        val = config[key]
        paths = val if isinstance(val, list) else [val]
        for p in paths:
            if not os.path.isfile(p):
                errors.append(f"'{key}' path does not exist: '{p}'")

    # Directory keys — warn only (pipeline creates them if missing)
    dir_keys = ['output_dir', 'trace_output_dir', 'extraction_output_dir',
                'flat_field_output_dir', 'log_output_dir', 'cube_output_dir', 'flat_file_dir']
    for key in dir_keys:
        if key in config and config[key]:
            p = config[key]
            if not os.path.isdir(p):
                warnings.append(
                    f"'{key}' directory does not exist yet: '{p}' — it will be created."
                )

    # ------------------------------------------------------------------
    # 5. LUT reference arc check
    # ------------------------------------------------------------------
    if not config.get('generate_new_wavelength_soln'):
        ref_arc = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')
        if not os.path.isfile(ref_arc):
            errors.append(
                f"Reference arc not found: '{ref_arc}'. "
                f"Set 'generate_new_wavelength_soln = true' and provide 'arc_file' to generate one, "
                f"or place the reference arc pickle at the expected location."
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    for w in warnings:
        print(f"  \u26a0  WARNING: {w}")
    for e in errors:
        print(f"  \u2717  ERROR: {e}")

    if errors:
        print(
            f"\nPipeline pre-flight check FAILED — "
            f"{len(errors)} error(s), {len(warnings)} warning(s)."
        )
        print("Fix the above issues and re-run.")
        sys.exit(1)

    if warnings:
        print(f"\nPipeline pre-flight check passed with {len(warnings)} warning(s).")
    else:
        print("\nPipeline pre-flight check passed.")

    return True


def copy_mastercalib_traces_for_placeholders(flat_file, trace_dir, channel, placeholder_indices=None):
    """
    Copy mastercalib traces for placeholder extensions identified in a flat file.

    Args:
        flat_file: Path to flat field FITS file
        trace_dir: Directory where traces should be copied to
        channel: Channel name ('red', 'green', 'blue')
        placeholder_indices: Optional list of placeholder indices (will detect if None)

    Returns:
        int: Number of traces copied
    """
    if placeholder_indices is None:
        placeholder_indices = get_placeholder_extension_indices(flat_file)

    if len(placeholder_indices) == 0:
        return 0  # No placeholders to handle

    print(f"Copying {len(placeholder_indices)} mastercalib traces for missing {channel} cameras")

    traces_copied = 0
    with fits.open(flat_file) as hdul:
        for idx in placeholder_indices:
            hdu = hdul[idx]

            # Extract camera metadata
            bench = hdu.header['BENCH']
            side = hdu.header['SIDE']
            cam_channel = hdu.header['COLOR'].lower()

            # Only copy if the channel matches
            if cam_channel != channel.lower():
                continue

            # Determine mastercalib trace filename
            master_trace_file = f'LLAMAS_master_{channel.lower()}_{bench}_{side}_traces.pkl'
            master_trace_path = os.path.join(CALIB_DIR, master_trace_file)

            # Target filename (user trace naming convention)
            user_trace_file = f'LLAMAS_{channel.lower()}_{bench}_{side}_traces.pkl'
            user_trace_path = os.path.join(trace_dir, user_trace_file)

            # Copy mastercalib trace to user trace directory
            if os.path.exists(master_trace_path):
                shutil.copy2(master_trace_path, user_trace_path)
                print(f"  ✓ Copied mastercalib trace for {channel}{bench}{side}")
                traces_copied += 1
            else:
                print(f"  ✗ WARNING: Mastercalib trace not found: {master_trace_file}")

    return traces_copied



def generate_traces(red_flat, green_flat, blue_flat, output_dir,
                    slow_bias=None, fast_bias=None, missing_cams=False):
    """Generate fiber traces from flat field observations for all three channels.

    This function intelligently handles missing camera extensions by:
    1. Detecting placeholder extensions in each flat field file
    2. Running trace generation only for real camera data
    3. Copying mastercalib traces for placeholder extensions

    Args:
        red_flat: Path to red flat field FITS file.
        green_flat: Path to green flat field FITS file.
        blue_flat: Path to blue flat field FITS file.
        output_dir: Directory to save trace files.
        slow_bias: Path to SLOW-mode master bias FITS file.
        fast_bias: Path to FAST-mode master bias FITS file.
        missing_cams: Deprecated parameter, kept for backwards compatibility.

    Raises:
        AssertionError: If any of the flat field files do not exist.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    assert os.path.exists(red_flat), "Red flat file does not exist."
    assert os.path.exists(green_flat), "Green flat file does not exist."
    assert os.path.exists(blue_flat), "Blue flat file does not exist."

    print("\n" + "="*60)
    print("TRACE GENERATION WITH PLACEHOLDER DETECTION")
    print("="*60)

    # Process each channel with placeholder detection
    channels = [
        ('red', red_flat),
        ('green', green_flat),
        ('blue', blue_flat)
    ]

    for channel, flat_file in channels:
        print(f"\n--- Processing {channel.upper()} channel ---")

        # Detect placeholder extensions
        placeholder_indices = get_placeholder_extension_indices(flat_file)

        if placeholder_indices:
            print(f"Found {len(placeholder_indices)} placeholder extensions in {os.path.basename(flat_file)}")
            print(f"Will skip trace generation for placeholder indices: {placeholder_indices}")
        else:
            print(f"No placeholder extensions found in {os.path.basename(flat_file)}")

        # Run trace generation, skipping placeholder extensions
        # slow_bias/fast_bias are forwarded; run_ray_tracing selects per READ-MDE via GUI_extract
        run_ray_tracing(
            flat_file,
            outpath=output_dir,
            channel=channel,
            slow_bias=slow_bias,
            fast_bias=fast_bias,
            is_master_calib=False,
            skip_extension_indices=placeholder_indices
        )

        # Copy mastercalib traces for placeholder extensions
        if placeholder_indices:
            traces_copied = copy_mastercalib_traces_for_placeholders(
                flat_file,
                output_dir,
                channel,
                placeholder_indices
            )
            if traces_copied > 0:
                print(f"Successfully copied {traces_copied} mastercalib traces for {channel} placeholders")

    print("\n" + "="*60)
    print(f"All traces generated and saved to {output_dir}")
    print("="*60)

    return


###need to edit GUI extract to give custom output_dir
#currently designed to use skyflats
#only used for generating new wl solutions
def extract_flat_field(flat_file_dir, output_dir, slow_bias=None, fast_bias=None):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ge.GUI_extract(flat_file_dir, output_dir=output_dir, slow_bias=slow_bias, fast_bias=fast_bias)

    return


def run_extraction(science_file, output_dir, slow_bias=None, fast_bias=None,
                   trace_dir=None, mastercalib_trace_dir=None):
    """
    Run spectrum extraction with hybrid trace support.

    Args:
        science_file: Path to science FITS file or list of paths
        output_dir: Output directory for extractions
        slow_bias: Path to SLOW-mode master bias FITS file
        fast_bias: Path to FAST-mode master bias FITS file
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
                slow_bias=slow_bias,
                fast_bias=fast_bias,
                trace_dir=trace_dir,
                mastercalib_trace_dir=mastercalib_trace_dir
            )
    else:
        assert os.path.exists(science_file), "Science file does not exist."
        extraction_file_path, _ = ge.GUI_extract(
            science_file,
            output_dir=output_dir,
            slow_bias=slow_bias,
            fast_bias=fast_bias,
            trace_dir=trace_dir,
            mastercalib_trace_dir=mastercalib_trace_dir
        )

    return  extraction_file_path


#this isn't quite right -> nneeds checking
def calc_wavelength_soln(arc_file, output_dir, slow_bias=None, fast_bias=None):

    ge.GUI_extract(arc_file, output_dir=output_dir, slow_bias=slow_bias, fast_bias=fast_bias)

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
    # TODO: when arc processing pipeline is wired up, use soln to generate/load a custom arc solution
    global _cached_reference_arc
    if _cached_reference_arc is None:
        logger.info("Loading reference arc (first call, will be cached)")
        _cached_reference_arc = ExtractLlamas.loadExtraction(
            os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl'))
    arcdict = _cached_reference_arc

    _science = ExtractLlamas.loadExtraction(science_extraction_file)
    extractions = _science['extractions']
    metadata    = _science['metadata']
    primary_hdr = _science.get('primary_header')
    print(f'extractions: {extractions}')
    print(f'metadata: {metadata}')
    std_wvcal = arc.arcTransfer(_science, arcdict,)

    print(f'std_wvcal: {std_wvcal}')
    print(f'std_wvcal metadata: {std_wvcal.get("metadata", {})}')

    return std_wvcal, primary_hdr


def _process_flat_for_rss(flat_files, flat_pixel_maps, output_dir,
                          trace_dir, arc_dict_config,
                          timestamp, label='flat',
                          slow_bias=None, fast_bias=None):
    """Apply pixel flat → extract → wavelength-calibrate → generate RSS for a flat frame.

    Used for both dome flats (fibre-flat RSS) and twilight flats.
    The pixel-to-pixel flat map *must* be applied to the raw flat frame before
    extraction so that extracted spectra represent pure fibre throughput
    (detector per-pixel sensitivity removed).

    Parameters
    ----------
    flat_files : list of str
        Raw flat FITS files (dome or twilight).
    flat_pixel_maps : list of str
        Pixel flat maps generated from the dome flat.
    output_dir : str
        Directory for all intermediate and output files.
    slow_bias : str or None
        Path to SLOW-mode master bias FITS (passed to run_extraction).
    fast_bias : str or None
        Path to FAST-mode master bias FITS (passed to run_extraction).
    trace_dir : str or None
        Directory containing fibre trace files.
    arc_dict_config : object or None
        Arc solution config forwarded to correct_wavelengths.
    timestamp : str
        Timestamp string used in log-file names.
    label : str
        Short label used in log filenames (e.g. 'dome', 'twilight').

    Returns
    -------
    list of str
        RSS output file paths (may be empty on failure).
    """
    rss_outputs = []
    for flat_file in flat_files:
        if flat_file is None:
            continue

        # Step 1: pixel-to-pixel flat correction (2D)
        corr_file, _ = apply_flat_field_correction(
            flat_file, flat_pixel_maps, output_dir,
            validate_matching=True, require_all_matches=False
        )
        if not corr_file:
            print(f"  WARNING: Pixel flat correction failed for {os.path.basename(flat_file)} — skipping")
            continue

        # Step 2: extract (bias subtraction is handled inside run_extraction)
        # run_extraction returns only a basename (from GUI_extract); join with output_dir
        pkl_basename = run_extraction(corr_file, output_dir,
                                      slow_bias=slow_bias, fast_bias=fast_bias,
                                      trace_dir=trace_dir, mastercalib_trace_dir=CALIB_DIR)
        if not pkl_basename:
            print(f"  WARNING: Extraction failed for {os.path.basename(corr_file)} — skipping")
            continue
        pkl = os.path.join(output_dir, pkl_basename)

        # Step 3: wavelength-calibrate and save corrected pickle
        corr_extr, primary_hdr = correct_wavelengths(pkl, soln=arc_dict_config)
        base = os.path.splitext(os.path.basename(pkl))[0]
        corr_pkl = os.path.join(output_dir, f'{base}_corrected_extractions.pkl')
        save_extractions(
            corr_extr['extractions'], primary_header=primary_hdr,
            savefile=corr_pkl, save_dir=output_dir,
            prefix=f'LLAMASExtract_{label}_corrected',
        )

        # Step 4: generate RSS
        rss_base = os.path.join(
            output_dir,
            os.path.basename(corr_pkl).replace('_corrected_extractions.pkl', '_RSS.fits')
        )
        rss_logger = setup_logger(__name__, f'{label}_RSS_{timestamp}.log')
        rss_gen = RSSgeneration(logger=rss_logger)
        out = rss_gen.generate_rss(corr_pkl, rss_base)
        if out:
            rss_outputs.extend(out)

    return rss_outputs


def process_flat_field_calibration(red_flat, green_flat, blue_flat, trace_dir, output_dir,
                                  arc_calib_file=None, verbose=False, method='simple',
                                  filter_size=12, signal_thresholds=None,
                                  clip_range=(0.90, 1.10)):
    """Generate flat field pixel maps for science frame correction.

    Args:
        red_flat (str): Path to red flat field FITS file
        green_flat (str): Path to green flat field FITS file
        blue_flat (str): Path to blue flat field FITS file
        trace_dir (str): Directory containing trace files
        output_dir (str): Output directory for flat field products
        arc_calib_file (str, optional): Path to arc calibration file
        verbose (bool): Enable verbose output
        method (str): 'simple' (default, median+Gaussian in wavelength space)
            or 'bspline' (legacy arc+B-spline fitting).
        filter_size (int): Median filter size (simple method only).
        signal_thresholds (dict, optional): Per-channel thresholds (simple method only).
        clip_range (tuple): Sensitivity clip range (simple method only).

    Returns:
        list: List of flat field pixel map FITS file paths
    """
    flat_output_dir = output_dir
    os.makedirs(flat_output_dir, exist_ok=True)

    print(f"Processing flat field calibration (method={method})")
    print(f"  Red flat: {os.path.basename(red_flat)}")
    print(f"  Green flat: {os.path.basename(green_flat)}")
    print(f"  Blue flat: {os.path.basename(blue_flat)}")
    print(f"  Output directory: {flat_output_dir}")

    try:
        if method == 'simple':
            results = process_pixel_flat_simple(
                red_flat, green_flat, blue_flat,
                arc_calib_file=arc_calib_file,
                output_dir=flat_output_dir,
                trace_dir=trace_dir,
                verbose=verbose,
                filter_size=filter_size,
                signal_thresholds=signal_thresholds,
                clip_range=clip_range,
            )
        elif method == 'bspline':
            results = process_flat_field_complete(
                red_flat, green_flat, blue_flat,
                arc_calib_file=arc_calib_file,
                output_dir=flat_output_dir,
                trace_dir=trace_dir,
                verbose=verbose,
            )
        else:
            raise ValueError(
                f"Unknown flat field method: {method!r}. "
                f"Use 'simple' or 'bspline'.")

        pixel_map_files = [results['pixel_map_file']]
        print(f"Flat field calibration complete: {os.path.basename(pixel_map_files[0])}")
        print(f"  MEF file with 24 extensions generated")
        return pixel_map_files

    except Exception as e:
        print(f"Error in flat field calibration: {str(e)}")
        logger.error(f"Flat field calibration failed: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return []


def _extract_channel_from_filename(path):
    """Extract the colour channel name ('red', 'green', or 'blue') from a filename.

    Parses the last occurrence of _red, _green, or _blue before the extension.

    Args:
        path (str): File path.

    Returns:
        str or None: Channel name lower-cased, or None if not found.
    """
    basename = os.path.basename(path).lower()
    for ch in ('red', 'green', 'blue'):
        if f'_{ch}' in basename or basename.endswith(f'{ch}.fits'):
            return ch
    return None


def _find_matching_flat_rss(flat_rss_list, channel):
    """Return the flat RSS path that matches a given colour channel.

    Args:
        flat_rss_list (list): List of flat RSS file paths from RSSgeneration.
        channel (str or None): Channel to match ('red', 'green', 'blue').

    Returns:
        str or None: Path of the matching flat RSS file, or None.
    """
    if not flat_rss_list or channel is None:
        return None
    for path in flat_rss_list:
        if f'_{channel}' in os.path.basename(path).lower():
            return path
    return None


def build_flat_field_map(flat_pixel_maps, science_file):
    """
    Build mapping between science extensions and corresponding flat field pixel maps.

    Handles multi-extension FITS (MEF) flat field files by matching
    extension metadata exactly (channel, bench, side).

    Args:
        flat_pixel_maps (list): List with single MEF file path
        science_file (str): Path to science FITS file

    Returns:
        dict: {sci_ext_idx: {'flat_ext_idx': N, 'match_found': bool, ...}}
    """
    flat_map = {}

    if len(flat_pixel_maps) != 1:
        print(f"ERROR: Expected single MEF flat file, got {len(flat_pixel_maps)} files")
        return {}

    flat_mef_file = flat_pixel_maps[0]
    print(f"Building flat field map from MEF: {os.path.basename(flat_mef_file)}")

    try:
        with fits.open(flat_mef_file) as flat_hdul, fits.open(science_file) as sci_hdul:
            # Iterate over science extensions (skip primary at index 0)
            for sci_ext_idx in range(1, len(sci_hdul)):
                if sci_hdul[sci_ext_idx].data is None:
                    continue

                # Get science metadata - try multiple header keys
                sci_header = sci_hdul[sci_ext_idx].header
                sci_channel = sci_header.get('CHANNEL', sci_header.get('COLOR', '')).lower()
                sci_bench = str(sci_header.get('BENCH', ''))
                sci_side = sci_header.get('SIDE', '').upper()

                # Alternative naming if metadata not directly present
                if not sci_channel or not sci_bench:
                    cam_name = sci_header.get('CAM_NAME', '')
                    if cam_name:
                        parts = cam_name.split('_')
                        if len(parts) >= 2:
                            sci_channel = parts[1].lower() if not sci_channel else sci_channel
                            if len(parts[0]) >= 2 and not sci_bench:
                                sci_bench = parts[0][0]
                                sci_side = parts[0][1]

                sci_key = f"{sci_channel}{sci_bench}{sci_side}"

                # Search flat extensions for EXACT match
                matching_flat_ext = None

                for flat_ext_idx in range(1, len(flat_hdul)):
                    if flat_hdul[flat_ext_idx].data is None:
                        continue

                    flat_header = flat_hdul[flat_ext_idx].header
                    flat_channel = flat_header.get('CHANNEL', '').lower()
                    flat_bench = str(flat_header.get('BENCH', ''))
                    flat_side = flat_header.get('SIDE', '').upper()

                    # Exact match required (no scoring)
                    if (flat_channel == sci_channel and
                        flat_bench == sci_bench and
                        flat_side == sci_side):
                        matching_flat_ext = flat_ext_idx
                        print(f"  ✓ Sci ext {sci_ext_idx} ({sci_key}) → Flat ext {flat_ext_idx}")
                        break

                # Store mapping
                flat_map[sci_ext_idx] = {
                    'flat_file': flat_mef_file,
                    'flat_ext_idx': matching_flat_ext,
                    'science_key': sci_key,
                    'science_channel': sci_channel,
                    'science_bench': sci_bench,
                    'science_side': sci_side,
                    'match_found': matching_flat_ext is not None,
                    'is_mef': True
                }

                if matching_flat_ext is None:
                    print(f"  ✗ Sci ext {sci_ext_idx} ({sci_key}) → No flat match found")

        # Summary
        matched = sum(1 for m in flat_map.values() if m['match_found'])
        print(f"✓ Matched {matched}/{len(flat_map)} science extensions to flat field")

    except Exception as e:
        print(f"ERROR building flat field map: {e}")
        logger.error(f"Failed to build flat field map: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
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


def _align_shapes(science_data, flat_data):
    """Align science and flat data shapes by trimming the larger array.

    LLAMAS detectors occasionally produce frames with an extra row or column
    (e.g. 2049x2048 vs 2048x2048).  This trims the larger array to match
    the smaller, keeping data from the origin corner.

    Parameters
    ----------
    science_data : ndarray
        2D science frame.
    flat_data : ndarray
        2D flat field / pixel map.

    Returns
    -------
    science_out, flat_out : ndarray
        Arrays with matching shapes.
    trimmed : bool
        True if any trimming was performed.
    """
    if science_data.shape == flat_data.shape:
        return science_data, flat_data, False

    common_rows = min(science_data.shape[0], flat_data.shape[0])
    common_cols = min(science_data.shape[1], flat_data.shape[1])

    return (science_data[:common_rows, :common_cols],
            flat_data[:common_rows, :common_cols],
            True)


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
    flat_map = build_flat_field_map(flat_pixel_maps, science_file) # this doesn't seem righ
    
    if not flat_map:
        print("ERROR: Could not build flat field mapping")
        logger.error("Could not build flat field mapping")
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
        # Open flat MEF file once (outside the loop for efficiency)
        flat_mef_file = flat_pixel_maps[0]

        with fits.open(science_file) as sci_hdul, fits.open(flat_mef_file) as flat_hdul:
            corrected_hdus = [sci_hdul[0].copy()]  # Primary header

            correction_stats = {'corrected': 0, 'skipped': 0, 'errors': 0}

            for ext_idx in range(1, len(sci_hdul)):
                sci_hdu = sci_hdul[ext_idx]
                mapping = flat_map.get(ext_idx, {})

                if mapping.get('match_found') and sci_hdu.data is not None:
                    try:
                        # Get flat data from MEF extension
                        flat_ext_idx = mapping['flat_ext_idx']
                        flat_data = flat_hdul[flat_ext_idx].data

                        # Shape alignment (handles extra-row science frames)
                        sci_aligned, flat_aligned, was_trimmed = _align_shapes(
                            sci_hdu.data, flat_data)
                        if was_trimmed:
                            print(f"  NOTE: Shape mismatch for ext {ext_idx} "
                                  f"(sci: {sci_hdu.data.shape}, flat: {flat_data.shape}). "
                                  f"Trimmed to {sci_aligned.shape}.")

                        # Perform division with NaN protection
                        corrected_trimmed = np.divide(
                            sci_aligned.astype(np.float32),
                            flat_aligned.astype(np.float32),
                            out=np.full_like(sci_aligned, np.nan, dtype=np.float32),
                            where=(flat_aligned > 0)
                        )

                        # Handle bad pixels from division
                        bad_pixels = ~np.isfinite(corrected_trimmed)
                        n_bad = np.sum(bad_pixels)
                        if n_bad > 0:
                            corrected_trimmed[bad_pixels] = sci_aligned[bad_pixels]

                        # Embed corrected data back into original science shape
                        if was_trimmed and sci_hdu.data.shape != sci_aligned.shape:
                            corrected_data = sci_hdu.data.astype(np.float32).copy()
                            corrected_data[:sci_aligned.shape[0],
                                           :sci_aligned.shape[1]] = corrected_trimmed
                        else:
                            corrected_data = corrected_trimmed

                        # Create new HDU with corrected data
                        new_hdu = fits.ImageHDU(
                            data=corrected_data,
                            header=sci_hdu.header.copy(),
                            name=sci_hdu.name
                        )

                        # Add correction metadata
                        new_hdu.header['FLATCORR'] = (True, 'Flat field corrected')
                        new_hdu.header['FLATFILE'] = (os.path.basename(flat_mef_file), 'Flat field MEF file')
                        new_hdu.header['FLATEXT'] = (flat_ext_idx, 'Flat field extension index')
                        new_hdu.header['FLATKEY'] = (mapping['science_key'], 'Flat field matching key')
                        new_hdu.header['BADFPIX'] = (int(n_bad), 'Bad pixels from flat division')

                        # Add statistics
                        valid_pixels = np.isfinite(corrected_data)
                        if np.any(valid_pixels):
                            new_hdu.header['FLATMEAN'] = (float(np.mean(corrected_data[valid_pixels])), 'Mean after flat correction')
                            new_hdu.header['FLATMED'] = (float(np.median(corrected_data[valid_pixels])), 'Median after flat correction')
                            new_hdu.header['FLATVPIX'] = (int(np.sum(valid_pixels)), 'Valid pixels after flat correction')

                        corrected_hdus.append(new_hdu)
                        correction_stats['corrected'] += 1

                        print(f"  ✓ Extension {ext_idx:2d} ({mapping['science_key']}): Corrected (bad pixels: {n_bad})")

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
        logger.error(f"Failed to process science file {science_file}: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        return None, {'corrected': 0, 'skipped': 0, 'errors': 1}


def construct_cube(rss_files, output_dir, wavelength_range=None, dispersion=1.0, spatial_sampling=0.75,
                   use_crr=True, crr_config=None, parallel=False, cube_method='traditional',
                   cube_pixel_size=0.3, cube_fiber_pitch=0.75, cube_wave_sampling=1.0,
                   cube_radius=1.5, cube_min_weight=0.01, cube_grid_method='oversampled'):
    """
    Construct IFU data cubes from RSS files using simple, traditional, or CRR method.

    This function can handle both:
    1. Single RSS files with multiple channels
    2. Multiple channel-specific RSS files with names like:
       "_extract_RSS_blue.fits", "_extract_RSS_green.fits", "_extract_RSS_red.fits"

    Parameters:
        rss_files (str or list): Path to RSS FITS file(s) or base paths
        output_dir (str): Directory to save output cubes
        wavelength_range (tuple, optional): Min/max wavelength range for output cubes
        dispersion (float): Wavelength dispersion in Angstroms/pixel (legacy, for traditional/CRR)
        spatial_sampling (float): Spatial sampling in arcsec/pixel (legacy, for traditional/CRR)
        use_crr (bool): Use CRR (Covariance-regularized Reconstruction) method (overrides cube_method)
        crr_config (CRRCubeConfig, optional): CRR configuration parameters
        parallel (bool): Use parallel processing for CRR method
        cube_method (str): Cube construction method: 'simple' (default), 'crr', or 'traditional'
        cube_pixel_size (float): Spatial pixel size in arcsec (for simple method)
        cube_fiber_pitch (float): Fiber pitch in arcsec (for simple method)
        cube_wave_sampling (float): Wavelength sampling factor (for simple method)
        cube_radius (float): Interpolation radius in arcsec (for simple method)
        cube_min_weight (float): Minimum weight threshold (for simple method)
        cube_grid_method (str): Spatial grid method for simple constructor:
            'oversampled' (default), 'native_hex', or 'nearest_hex'

    Returns:
        list: Paths to constructed cube files
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if isinstance(rss_files, str):
        rss_files = [rss_files]
        
    cube_files = []

    # Track base names already processed by the traditional method to avoid
    # redundant work when rss_files contains multiple channel-specific files
    # (e.g. _RSS_blue, _RSS_green, _RSS_red) that share the same base name.
    processed_traditional_bases = set()

    # Create a single logger for all cube construction
    logger = logging.getLogger(__name__)
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

        # Determine which method to use
        # CRR flag overrides cube_method for backward compatibility
        use_simple = (cube_method == 'simple') and not use_crr
        use_traditional = (cube_method == 'traditional') and not use_crr

        if use_simple:
            # Use SimpleCubeConstructor (MUSE-like method)
            logger.info(f"Constructing simple MUSE-like cube from RSS file: {rss_file}")
            print(f"Constructing simple MUSE-like cube from RSS file: {rss_file}")

            try:
                # Initialize simple cube constructor
                constructor = SimpleCubeConstructor(
                    fiber_pitch=cube_fiber_pitch,
                    pixel_size=cube_pixel_size,
                    wave_sampling=cube_wave_sampling,
                    grid_method=cube_grid_method
                )

                # Load RSS file (reads FIBERMAP extension for fiber identity)
                constructor.load_rss_file(rss_file)

                # Match fibers to IFU positions via FiberMap_LUT
                constructor.match_fibers_to_fibermap()

                # Create wavelength grid
                wave_min, wave_max = None, None
                if wavelength_range is not None:
                    wave_min, wave_max = wavelength_range
                constructor.create_wavelength_grid(wave_min=wave_min, wave_max=wave_max)

                # Create spatial grid
                constructor.create_spatial_grid()

                # Construct cube
                constructor.construct_cube(radius=cube_radius, min_weight=cube_min_weight)

                # Create WCS (use default RA/Dec = 0 for now, could be enhanced later)
                constructor.create_wcs(ra_center=0.0, dec_center=0.0)

                # Save cube
                if channel:
                    cube_filename = f"{base_name}_cube_{channel}.fits"
                else:
                    cube_filename = f"{base_name}_cube.fits"

                cube_path = os.path.join(output_dir, cube_filename)
                constructor.save_cube(cube_path, overwrite=True)
                cube_files.append(cube_path)

                print(f"  - Simple cube saved: {cube_path}")
                logger.info(f"Simple cube saved: {cube_path}")

                # Log quality metrics
                valid_spaxels = np.sum(np.any(constructor.cube_weight > 0, axis=0))
                total_spaxels = constructor.cube_weight.shape[1] * constructor.cube_weight.shape[2]
                coverage = valid_spaxels / total_spaxels if total_spaxels > 0 else 0
                print(f"  - Coverage: {coverage:.1%} ({valid_spaxels}/{total_spaxels} spaxels)")
                logger.info(f"Simple cube quality: {coverage:.1%} coverage ({valid_spaxels}/{total_spaxels} spaxels)")

            except Exception as e:
                logger.error(f"Simple cube construction failed: {e}")
                print(f"  ERROR: Simple cube construction failed: {e}")
                print("  Falling back to traditional cube construction...")
                traceback.print_exc()

                # Fall back to traditional method
                use_simple = False
                use_traditional = True

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
                use_traditional = True

        if use_traditional or (not use_simple and not use_crr):
            # Deduplicate: construct_cube_from_rss strips the channel suffix and
            # discovers all 3 colour files, so processing any one of them builds
            # all 3 cubes.  Skip if this base name has already been handled.
            trad_base = os.path.splitext(rss_file)[0]
            for color in ['red', 'green', 'blue']:
                for pattern in [f'_extract_RSS_{color}', f'_flat_corrected_RSS_{color}']:
                    if pattern in trad_base:
                        trad_base = trad_base.split(pattern)[0]
                        break

            if trad_base in processed_traditional_bases:
                logger.info(f"Skipping {rss_file}: base '{os.path.basename(trad_base)}' already processed")
                print(f"  Skipping {os.path.basename(rss_file)} (already built all channels for this base)")
                continue
            processed_traditional_bases.add(trad_base)

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

    # Configure pipeline logging — log file goes next to the config file
    if 'log_output_dir' in config:
        log_dir = config['log_output_dir']
    else:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(config_path)), 'logs')
    log_file = configure_pipeline_logging(log_dir)
    logger = logging.getLogger(__name__)
    logger.info(f"Pipeline started. Config: {config_path}")

    # Pre-flight validation — exits cleanly if anything is wrong
    validate_pipeline_config(config, config_path)

    input_files = get_input_files_from_config(config)
    validate_input_files(input_files)

    # Resolve master bias files — user overrides take priority, otherwise use package defaults.
    # Missing extensions are handled per-detector in process_trace() via inter-fibre fallback;
    # validate_for_gui() is NOT applied to bias files to avoid creating *_GUI_version.fits
    # artefacts inside the package Bias/ directory.
    _slow_bias_path = config.get('slow_bias_file', SLOW_BIAS_FILE)
    _fast_bias_path = config.get('fast_bias_file', FAST_BIAS_FILE)
    slow_bias_file = _slow_bias_path if os.path.isfile(_slow_bias_path) else None
    fast_bias_file = _fast_bias_path if os.path.isfile(_fast_bias_path) else None
    if slow_bias_file is None:
        logger.warning(f"Slow bias file not found at '{_slow_bias_path}' — inter-fibre fallback will be used")
    if fast_bias_file is None:
        logger.warning(f"Fast bias file not found at '{_fast_bias_path}' — inter-fibre fallback will be used")

        
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
        logger.info("Stage: Generating new wavelength solution")
        extract_flat_field(config.get('flat_file_dir'), config.get('output_dir'),
                           slow_bias=slow_bias_file, fast_bias=fast_bias_file)
        if 'arc_file' not in config:
            raise ValueError("No arc file provided in the configuration.")
        relative_throughput(config.get('shift_picklename'), config.get('flat_picklename'))
        arcdict = calc_wavelength_soln(config['arc_file'], config.get('output_dir'),
                                       slow_bias=slow_bias_file, fast_bias=fast_bias_file)
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
        if config.get('use_existing_traces', True) and os.path.exists(config.get('trace_output_dir')):
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
            logger.info("Stage: Generating new traces")

            # Validate flat field files before trace generation
            

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
                           config.get('trace_output_dir'),
                           slow_bias=slow_bias_file, fast_bias=fast_bias_file)

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
        
        # Generate flat field pixel maps if flat correction is enabled
        flat_pixel_maps = []
        flat_field_method = config.get('flat_field_method', 'simple')
        if config.get('apply_flat_field_correction', True):
            print("\n" + "="*60)
            print(f"FLAT FIELD PROCESSING (method={flat_field_method})")
            print("="*60)

            # Create proper flat field directory structure: extractions/flat/
            flat_field_dir = config.get('flat_field_output_dir', os.path.join(extraction_path, 'flat'))
            os.makedirs(flat_field_dir, exist_ok=True)

            # Parse optional simple-method parameters from config
            filter_size = int(config.get('flat_filter_size', 12))
            clip_min = float(config.get('flat_clip_min', 0.90))
            clip_max = float(config.get('flat_clip_max', 1.10))

            # Parse per-channel thresholds if provided
            signal_thresholds = None
            if any(f'flat_signal_threshold_{c}' in config for c in ('red', 'green', 'blue')):
                signal_thresholds = {
                    'red':   float(config.get('flat_signal_threshold_red', 5000)),
                    'green': float(config.get('flat_signal_threshold_green', 8000)),
                    'blue':  float(config.get('flat_signal_threshold_blue', 5000)),
                }

            flat_pixel_maps = process_flat_field_calibration(
                config.get('red_flat_file'),
                config.get('green_flat_file'),
                config.get('blue_flat_file'),
                config.get('trace_output_dir'),
                flat_field_dir,
                arc_calib_file=config.get('arc_calib_file'),
                verbose=config.get('verbose_flat_processing', False),
                method=flat_field_method,
                filter_size=filter_size,
                signal_thresholds=signal_thresholds,
                clip_range=(clip_min, clip_max),
            )

            if flat_pixel_maps:
                print(f"\nGenerated {len(flat_pixel_maps)} flat field pixel maps:")
            else:
                print("WARNING: No flat field pixel maps generated. Proceeding without flat field correction.")
                logger.warning("No flat field pixel maps generated. Proceeding without flat field correction.")

            # --- Generate flat RSS for fibre-to-fibre correction ---
            # The flat frames (dome or twilight) must have the pixel map applied
            # before extraction so that extracted spectra represent pure fibre
            # throughput (detector per-pixel sensitivity removed).
            config['flat_rss_outputs'] = []
            timestamp_ff = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Prefer twilight flat when specified
            twilight_file = config.get('twilight_flat')

            if twilight_file is not None and flat_pixel_maps:
                print("\nTwilight flat specified — building fibre-flat RSS from twilight flat...")
                twi_dir = os.path.join(flat_field_dir, 'twilight')
                os.makedirs(twi_dir, exist_ok=True)
                twi_rss = _process_flat_for_rss(
                    [twilight_file],
                    flat_pixel_maps, twi_dir,
                    trace_dir=final_trace_dir,
                    arc_dict_config=config.get('arcdict'),
                    timestamp=timestamp_ff, label='twilight',
                    slow_bias=slow_bias_file, fast_bias=fast_bias_file,
                )
                if twi_rss:
                    config['flat_rss_outputs'] = twi_rss
                    print(f"  Twilight flat RSS: {twi_rss}")
                else:
                    print("  WARNING: Twilight flat RSS generation failed.")

            if not config['flat_rss_outputs']:
                # Fallback: dome flats — also need pixel correction before extraction
                dome_files = [config.get('red_flat_file'),
                              config.get('green_flat_file'),
                              config.get('blue_flat_file')]
                if any(f is not None for f in dome_files) and flat_pixel_maps:
                    if twilight_file is not None:
                        msg = ("No twilight flat RSS — building fibre-flat RSS from "
                               "pixel-corrected dome flats.")
                    else:
                        msg = ("No twilight_flat specified — using dome flat RSS "
                               "for fibre-to-fibre correction.\n"
                               "        Provide twilight_flat in config for best results.")
                    print(f"\n  NOTE: {msg}")
                    dome_dir = os.path.join(flat_field_dir, 'dome_rss')
                    os.makedirs(dome_dir, exist_ok=True)
                    dome_rss = _process_flat_for_rss(
                        [f for f in dome_files if f is not None],
                        flat_pixel_maps, dome_dir,
                        trace_dir=final_trace_dir,
                        arc_dict_config=config.get('arcdict'),
                        timestamp=timestamp_ff, label='dome',
                        slow_bias=slow_bias_file, fast_bias=fast_bias_file,
                    )
                    if dome_rss:
                        config['flat_rss_outputs'] = dome_rss
                        print(f"  Dome flat RSS: {dome_rss}")
                    else:
                        print("  WARNING: Dome flat RSS generation also failed — "
                              "fibre-to-fibre correction will be skipped.")
                else:
                    print("  WARNING: No flat files available or pixel maps missing — "
                          "skipping flat RSS generation.")


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
                    
                    corrected_file, stats = apply_flat_field_correction(
                        science_file,
                        flat_pixel_maps,
                        flat_output_dir,
                        validate_matching=config.get('validate_flat_matching', True),
                        require_all_matches=config.get('require_all_flat_matches', True)
                    )
                    
                    if corrected_file:
                        flat_corrected_files.append(corrected_file)
                        # Accumulate statistics
                        overall_stats['total_corrected'] += stats['corrected']
                        overall_stats['total_skipped'] += stats['skipped'] 
                        overall_stats['total_errors'] += stats['errors']
                    else:
                        print(f"ERROR: Failed to flat-correct {science_file}")
                        logger.error(f"Failed to flat-correct {science_file}")
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
                    logger.error(f"Failed to flat-correct {science_file}, using original")
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
        
        # Build a list of (pkl_path, original_science_fits_path) pairs during extraction.
        # This avoids re-globbing (which could pick up flat/calibration pkls) and keeps
        # each pkl explicitly linked to the original science file for header extraction.
        science_pkl_pairs = []  # list of (pkl_path, original_science_fits_path)

        # Normalise original science files to a list for uniform handling
        original_science_files = config['science_files']
        if not isinstance(original_science_files, list):
            original_science_files = [original_science_files]

        if isinstance(science_files_to_process, list):
            print(f'\nFound {len(science_files_to_process)} science files to process for extraction.')
            logger.info(f"Stage: Extracting {len(science_files_to_process)} science files")

            for i, (science_file, orig_file) in enumerate(zip(science_files_to_process, original_science_files)):
                print(f"Extracting science file {i+1}/{len(science_files_to_process)}: {os.path.basename(science_file)}")
                # Process each science file with hybrid trace support
                extracted_basename = run_extraction(
                    science_file,
                    extraction_path,
                    slow_bias=slow_bias_file,
                    fast_bias=fast_bias_file,
                    trace_dir=final_trace_dir,              # User traces
                    mastercalib_trace_dir=CALIB_DIR         # Mastercalib fallback
                )
                print(f"Extraction completed for {os.path.basename(science_file)}. Output file: {extracted_basename}")
                if extracted_basename:
                    science_pkl_pairs.append((os.path.join(extraction_path, extracted_basename), orig_file))
        else:
            print(f"Extracting science file: {os.path.basename(science_files_to_process)}")
            extracted_basename = run_extraction(
                science_files_to_process,
                extraction_path,
                slow_bias=slow_bias_file,
                fast_bias=fast_bias_file,
                trace_dir=final_trace_dir,                  # User traces
                mastercalib_trace_dir=CALIB_DIR             # Mastercalib fallback
            )
            print(f"Extraction completed. Used traces from {final_trace_dir} with mastercalib fallback. Output file: {extracted_basename}")
            if extracted_basename:
                science_pkl_pairs.append((os.path.join(extraction_path, extracted_basename), original_science_files[0]))

        for index, (correction_path, orig_science_file) in enumerate(science_pkl_pairs):
            print(f"Processing extraction file {index+1}/{len(science_pkl_pairs)}: {correction_path}")
            if not os.path.exists(correction_path):
                raise FileNotFoundError(f"Extraction file {correction_path} does not exist.")

            # Correct wavelengths for each extraction file
            corr_extractions, _ = correct_wavelengths(correction_path, soln=config.get('arcdict'))

            corr_extraction_list = corr_extractions['extractions']

            # Read primary header directly from the original science FITS file.
            # This guarantees correct RA/DEC and telescope pointing regardless of what
            # may have been stored in intermediate pickle files (flat-corrected copies,
            # concat'd flat pkls, etc. all share the same primary HDU data but the
            # pkl chain is fragile — the original file is authoritative).
            with fits.open(orig_science_file) as _sci_hdul:
                primary_hdr = _sci_hdul[0].header.copy()

            # Save the corrected extractions using the current file's base name
            base_name = os.path.splitext(os.path.basename(correction_path))[0]
            savefile = os.path.join(extraction_path, f'{base_name}_corrected_extractions.pkl')
            save_extractions(corr_extraction_list, primary_header=primary_hdr, savefile=savefile, save_dir=extraction_path, prefix='LLAMASExtract_batch_corrected')

            # Create a logger for RSS generation
            rss_logger = logging.getLogger(__name__ + '.rss')
            rss_logger.info(f"Starting RSS generation for {base_name}")
            
            rss_output_file = os.path.join(extraction_path, f'{base_name}_RSS.fits')
            
            #RSS generation
            rss_gen = RSSgeneration(logger=rss_logger)
            new_rss_outputs = rss_gen.generate_rss(savefile, rss_output_file)
            rss_logger.info(f"RSS file generated: {new_rss_outputs}")
            print(f"RSS file generated: {new_rss_outputs}")

            # Updating RA and Dec in RSS files
            for rss_output_file in new_rss_outputs:
                update_ra_dec_in_fits(rss_output_file, logger=rss_logger)

            # Free wavelength-corrected extractions to reduce memory pressure
            del corr_extractions, corr_extraction_list
            gc.collect()

        # ── Fibre-to-fibre flat correction on RSS files ──
        if were_flat_corrected and config.get('apply_fibre_flat', True):
            flat_field_dir = config.get('flat_field_output_dir',
                                        os.path.join(extraction_path, 'flat'))
            smooth_models_file = os.path.join(flat_field_dir,
                                              'flat_smooth_models.fits')

            if os.path.exists(smooth_models_file):
                print("\n" + "=" * 60)
                print("FIBRE-TO-FIBRE FLAT CORRECTION")
                print("=" * 60)

                twilight_file = config.get('twilight_flat')
                corrections_file = None

                if twilight_file and os.path.exists(twilight_file):
                    # Branch A: Twilight + Lamp
                    print(f"Using twilight flat: {os.path.basename(twilight_file)}")
                    try:
                        twi_extractions = reduce_twilight_flat(
                            twilight_file,
                            flat_pixel_maps[0] if flat_pixel_maps else None,
                            final_trace_dir,
                            config.get('arcdict'),
                            slow_bias_file,
                            extraction_path,
                            fast_bias=fast_bias_file,
                        )
                        corrections_file = compute_fibre_flat_twilight(
                            twi_extractions, smooth_models_file,
                            flat_field_dir,
                            integration_range=config.get(
                                'fibre_flat_integration_range'),
                            poly_order=config.get(
                                'fibre_flat_poly_order', None),
                        )
                        del twi_extractions
                        gc.collect()
                        print("Fibre flat computed (twilight + lamp method)")
                    except Exception as e:
                        print(f"WARNING: Twilight reduction failed: {e}")
                        logger.warning(f"Twilight reduction failed: {e}", exc_info=True)
                        print("Falling back to lamp-only fibre flat")
                        traceback.print_exc()
                        # Clean up if twi_extractions was created before failure
                        try:
                            del twi_extractions
                        except NameError:
                            pass
                        gc.collect()
                        corrections_file = None

                if corrections_file is None:
                    # Branch B: Lamp-only fallback
                    if twilight_file:
                        print("Twilight flat not available or failed — "
                              "using lamp-only fallback")
                    corrections_file = compute_fibre_flat_lamp_only(
                        smooth_models_file, flat_field_dir)
                    print("Fibre flat computed (lamp-only method)")

                # Apply to all RSS files in the extraction directory
                rss_to_correct = [
                    os.path.join(extraction_path, f)
                    for f in os.listdir(extraction_path)
                    if f.endswith('.fits') and '_RSS' in f
                    and '_FF' not in f
                ]
                for rss_file in rss_to_correct:
                    ff_output = rss_file.replace('.fits', '_FF.fits')
                    apply_fibre_flat_to_rss(rss_file, corrections_file,
                                           ff_output)
                    print(f"  FF RSS: {os.path.basename(ff_output)}")
            else:
                print("WARNING: Smooth models file not found — "
                      "skipping fibre-to-fibre flat correction")
                logger.warning("Smooth models file not found — skipping fibre-to-fibre flat correction")

        # Cube construction from RSS files
        print("Constructing cubes from RSS files...")
        logger.info("Stage: Constructing cubes from RSS files")
        # Look for RSS files — prefer FF (fibre-flat corrected) when available
        all_rss = [os.path.join(extraction_path, f)
                   for f in os.listdir(extraction_path)
                   if f.endswith('.fits') and '_RSS' in f]
        ff_rss = [f for f in all_rss if '_FF' in os.path.basename(f)]
        non_ff_rss = [f for f in all_rss if '_FF' not in os.path.basename(f)]
        rss_files = ff_rss if ff_rss else non_ff_rss
        
        if rss_files:
            print(f"Found {len(rss_files)} RSS files for cube construction:")
            for rss_file in rss_files:
                basename = os.path.basename(rss_file)
                if '_FF' in basename:
                    status = " (fibre-flat corrected)"
                elif '_flat_corrected' in basename:
                    status = " (pixel-flat corrected)"
                else:
                    status = ""
                print(f"  - {basename}{status}")
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
                parallel=config.get('CRR_parallel', False),
                cube_method=config.get('cube_method', 'traditional'),
                cube_pixel_size=float(config.get('cube_pixel_size', 0.3)),
                cube_fiber_pitch=float(config.get('cube_fiber_pitch', 0.75)),
                cube_wave_sampling=float(config.get('cube_wave_sampling', 1.0)),
                cube_radius=float(config.get('cube_radius', 1.5)),
                cube_min_weight=float(config.get('cube_min_weight', 0.01)),
                cube_grid_method=config.get('cube_grid_method', 'oversampled')
            )
            print(f"Cubes constructed: {cube_files}")
        else:
            print("No RSS files found for cube construction")
                
        
        
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce module placeholder")
    parser.add_argument('config_file', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    
    main(args.config_file)