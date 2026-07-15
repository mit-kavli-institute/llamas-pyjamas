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
import re
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
from llamas_pyjamas.Utils.rayManager import resolve_run_temp_dir, prune_stale, check_inputs_reachable, preflight_disk_check, cleanup_scratch, init_ray
from llamas_pyjamas.Cube.cubeConstruct import CubeConstructor
from llamas_pyjamas.Bias.llamasBias import BiasLlamas
from llamas_pyjamas.Cube.crr_cube_constructor import CRRCubeConstructor, CRRCubeConfig
from llamas_pyjamas.Cube.rss_to_crr_adapter import load_rss_as_crr_data, combine_channels_for_crr
from llamas_pyjamas.Cube.simple_cube_constructor import SimpleCubeConstructor
from astropy.io import fits
import numpy as np

import shutil

_cached_reference_arc = None
_cached_reference_arc_path = None

logger = logging.getLogger(__name__)

from llamas_pyjamas.DataModel.validate import validate_and_fix_extensions, get_placeholder_extension_indices, validate_for_gui
from llamas_pyjamas.Flat.flatLlamas import process_flat_field_complete, process_pixel_flat_simple
from llamas_pyjamas.Sky.skyLlamas import skyModel_1d
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
                       'twilight_flat', 'red_twilight_flat',
                       'green_twilight_flat', 'blue_twilight_flat']

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
                 'twilight_flat', 'red_twilight_flat', 'green_twilight_flat',
                 'blue_twilight_flat']
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

    ge.GUI_extract(flat_file_dir, output_dir=output_dir, slow_bias=slow_bias, fast_bias=fast_bias,
                   remove_cosmic_rays=False)

    return


def run_extraction(science_file, output_dir, slow_bias=None, fast_bias=None,
                   trace_dir=None, mastercalib_trace_dir=None,
                   remove_cosmic_rays=True, mask_output_dir=None, edge_bias=None):
    """
    Run spectrum extraction with hybrid trace support.

    Args:
        science_file: Path to science FITS file or list of paths
        output_dir: Output directory for extractions
        slow_bias: Path to SLOW-mode master bias FITS file
        fast_bias: Path to FAST-mode master bias FITS file
        trace_dir: User trace directory (for real extensions)
        mastercalib_trace_dir: Mastercalib trace directory (for placeholder extensions)
        remove_cosmic_rays: Enable L.A.Cosmic cosmic ray removal before extraction
        mask_output_dir: Output directory for cosmic ray mask FITS files
        edge_bias: Optional dict controlling the per-frame edge-bias (DC offset)
            correction (see build_edge_bias_config); forwarded to GUI_extract.
            None => process_trace uses its built-in defaults (Tier 1 on).

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
                mastercalib_trace_dir=mastercalib_trace_dir,
                remove_cosmic_rays=remove_cosmic_rays,
                mask_output_dir=mask_output_dir,
                edge_bias=edge_bias
            )
    else:
        assert os.path.exists(science_file), "Science file does not exist."
        extraction_file_path, _ = ge.GUI_extract(
            science_file,
            output_dir=output_dir,
            slow_bias=slow_bias,
            fast_bias=fast_bias,
            trace_dir=trace_dir,
            mastercalib_trace_dir=mastercalib_trace_dir,
            remove_cosmic_rays=remove_cosmic_rays,
            mask_output_dir=mask_output_dir,
            edge_bias=edge_bias
        )

    # GUI_extract returns just the basename; make it a full path
    if extraction_file_path and not os.path.isabs(extraction_file_path):
        extraction_file_path = os.path.join(output_dir, extraction_file_path)

    return extraction_file_path


def apply_twilight_throughput(extraction_dict, twilight_dir):
    """Set per-fibre relative_throughput from the twilight (sky) flat extraction.

    The twilight illuminates the fibres with the same geometry as the night sky
    (unlike the dome lamp), and — because it is extracted with the same method
    as the science in the same run — its per-fibre response is directly
    applicable (validated 2026-07: predicts the stable on-sky 5577 response at
    corr 0.97 on cameras where the lamp throughput reads 0.40). Objects never
    contaminate it (dedicated sky-illuminated frames).

    Finds the newest twilight extraction pickle in ``twilight_dir`` and, for
    every camera present in both, replaces ``relative_throughput`` with the
    fibre's median twilight counts normalised to the camera median (clipped to
    [0.1, 3.0]; out-of-range/dead fibres get 1.0). Cameras without a twilight
    counterpart keep their existing (arc-carried) values.

    Returns the number of cameras updated (0 => nothing done / no twilight).
    """
    import glob as _glob
    cands = sorted(
        _glob.glob(os.path.join(twilight_dir, '*_extract_corrected_extractions.pkl'))
        + _glob.glob(os.path.join(twilight_dir, '*_extract.pkl')),
        key=os.path.getmtime, reverse=True)
    if not cands:
        return 0
    twi = ExtractLlamas.loadExtraction(cands[0])
    twi_map = {}
    for e, m in zip(twi['extractions'], twi['metadata']):
        twi_map[(m['channel'], str(m['bench']), str(m['side']).upper())] = e
    n_done = 0
    for e, m in zip(extraction_dict['extractions'], extraction_dict['metadata']):
        et = twi_map.get((m['channel'], str(m['bench']), str(m['side']).upper()))
        if et is None:
            continue
        counts = np.asarray(et.counts, dtype=float)
        wl = np.nanmedian(counts[:, 700:1500], axis=1)
        good = np.isfinite(wl) & (wl > 0)
        if good.sum() < 30:
            continue
        tp = wl / np.nanmedian(wl[good])
        tp = np.where(np.isfinite(tp) & (tp > 0.1) & (tp < 3.0), tp, 1.0)
        n = min(np.asarray(e.counts).shape[0], tp.size)
        e.relative_throughput = tp[:n].copy()
        n_done += 1
    logger.info(f"apply_twilight_throughput: updated {n_done} cameras from "
                f"{os.path.basename(cands[0])}")
    return n_done


def build_edge_bias_config(config, extraction_path, flat_field_dir=None):
    """Assemble the edge-bias (per-frame DC offset) settings from the pipeline config.

    Returns a small, Ray-serialisable dict consumed by GUI_extract/process_trace.
    The DC offset is measured from unilluminated pixels of each frame and
    subtracted on top of the master bias to track the nightly temperature drift.
    """
    return {
        'enabled':       bool(config.get('edge_bias_dc_correction', True)),
        'min_distance':  float(config.get('edge_bias_min_distance', 20)),
        'min_pixels':    int(config.get('edge_bias_min_pixels', 500)),
        'use_flat_mask': bool(config.get('edge_bias_use_flat_mask', False)),
        'flat_frac':     float(config.get('edge_bias_flat_frac', 0.1)),
        'flat_mask_dir': flat_field_dir or os.path.join(extraction_path, 'flat'),
        'qa_csv':        os.path.join(extraction_path, 'edge_bias_levels.csv'),
    }


def _extract_sky_frame(sky_file, extraction_path, slow_bias, fast_bias, trace_dir,
                       remove_cosmic_rays, mask_output_dir, config, flat_pixel_maps=None):
    """Extract a dedicated blank-sky MEF and return a ``*_corrected_extractions.pkl``.

    Mirrors the science extraction path (validate -> optional flat-correct ->
    extract -> wavelength-correct -> save) so the result can be passed straight
    to :func:`skyModel_1d` as ``sky_extraction_file`` for the ``'frame'`` method.
    """
    if not os.path.exists(sky_file):
        raise FileNotFoundError(f"Sky frame {sky_file} does not exist.")
    print("\n" + "=" * 60)
    print(f"EXTRACTING DEDICATED SKY FRAME: {os.path.basename(sky_file)}")
    print("=" * 60)

    orig_sky_file = sky_file
    with fits.open(orig_sky_file) as _sky_hdul:
        sky_primary_hdr = _sky_hdul[0].header.copy()

    sky_file = validate_and_fix_extensions(sky_file, output_file=None, backup=True)

    # Optional flat-field correction, matching the science path so the sky and
    # science counts share the same detector response.
    if flat_pixel_maps:
        corrected, _stats = apply_flat_field_correction(
            sky_file, flat_pixel_maps, extraction_path,
            validate_matching=config.get('validate_flat_matching', True),
            require_all_matches=config.get('require_all_flat_matches', True))
        if corrected:
            sky_file = corrected

    extracted = run_extraction(
        sky_file, extraction_path, slow_bias=slow_bias, fast_bias=fast_bias,
        trace_dir=trace_dir, mastercalib_trace_dir=CALIB_DIR,
        remove_cosmic_rays=remove_cosmic_rays, mask_output_dir=mask_output_dir,
        edge_bias=build_edge_bias_config(config, extraction_path))
    if not extracted:
        raise RuntimeError(f"Extraction of sky frame {sky_file} produced no output")

    corr_extractions, _ = correct_wavelengths(extracted, soln=config.get('arcdict'))
    base_name = os.path.splitext(os.path.basename(extracted))[0]
    sky_savefile = os.path.join(extraction_path, f'{base_name}_corrected_extractions.pkl')
    save_extractions(corr_extractions['extractions'], primary_header=sky_primary_hdr,
                     savefile=sky_savefile, save_dir=extraction_path,
                     prefix='LLAMASExtract_sky_corrected')
    print(f"Sky frame extraction complete: {os.path.basename(sky_savefile)}")
    return sky_savefile


#this isn't quite right -> nneeds checking
def calc_wavelength_soln(arc_file, output_dir, slow_bias=None, fast_bias=None):

    ge.GUI_extract(arc_file, output_dir=output_dir, slow_bias=slow_bias, fast_bias=fast_bias,
                   remove_cosmic_rays=False)

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
    """Transfer the wavelength solution onto a science extraction.

    ``soln`` is the arc solution to use: a pickle path (e.g. the refined
    product from refineArcX/refineArcX2D), an already-loaded arcdict, or None
    for the packaged reference arc. Loads are cached keyed on the source path
    so successive science files reuse one load.

    NOTE: this function previously ignored ``soln`` entirely (a leftover TODO)
    and always used the packaged reference arc — silently discarding any
    refined solution, so refine_arc had no effect on science reductions.
    """
    global _cached_reference_arc, _cached_reference_arc_path

    # stdout (not just logger): when run via `python -m llamas_pyjamas.reduce`
    # this module is `__main__`, whose logger does not propagate into the
    # llamas_pyjamas log file — prints are the reliable record of which arc
    # solution was actually used.
    print(f"correct_wavelengths: soln = "
          f"{'<in-memory dict>' if isinstance(soln, dict) else repr(soln)}")

    if isinstance(soln, dict):
        arcdict = soln
    else:
        if isinstance(soln, str) and soln:
            if not os.path.exists(soln):
                print(f"correct_wavelengths: WARNING soln path not found "
                      f"({soln}); using packaged reference arc")
                logger.warning(f"correct_wavelengths: soln path not found "
                               f"({soln}); using packaged reference arc")
                arc_path = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')
            else:
                arc_path = soln
        else:
            arc_path = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')

        if _cached_reference_arc is None or _cached_reference_arc_path != arc_path:
            print(f"correct_wavelengths: loading arc solution {arc_path}")
            logger.info(f"Loading arc solution {os.path.basename(arc_path)} "
                        f"(cached for subsequent calls)")
            _cached_reference_arc = ExtractLlamas.loadExtraction(arc_path)
            _cached_reference_arc_path = arc_path
        else:
            print(f"correct_wavelengths: using cached arc solution "
                  f"{os.path.basename(arc_path)}")
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


def _pointing_from_header(header):
    """Extract (ra_deg, dec_deg, pa_deg) from an RSS/science primary header (F6).

    Prefers the decimal ``RA``/``DEC`` keywords, falling back to the sexagesimal
    HIERARCH ``TEL RA``/``TEL DEC`` pair, and reads the field rotation from
    ``TEL PA`` (then ``TEL ROT``). Returns ``(0.0, 0.0, 0.0)`` if nothing usable
    is present so cube construction degrades to the old placeholder behaviour.
    """
    if header is None:
        return 0.0, 0.0, 0.0
    ra = header.get('RA')
    dec = header.get('DEC')
    try:
        ra, dec = float(ra), float(dec)
        if not (np.isfinite(ra) and np.isfinite(dec)):
            raise ValueError
    except (TypeError, ValueError):
        ra = dec = None
    if ra is None or dec is None:
        tra, tdec = header.get('TEL RA'), header.get('TEL DEC')
        if tra and tdec:
            try:
                from astropy.coordinates import SkyCoord
                import astropy.units as u
                c = SkyCoord(str(tra), str(tdec), unit=(u.hourangle, u.deg))
                ra, dec = float(c.ra.deg), float(c.dec.deg)
            except Exception:
                return 0.0, 0.0, 0.0
        else:
            return 0.0, 0.0, 0.0
    pa = header.get('TEL PA', header.get('TEL ROT', 0.0))
    try:
        pa = float(pa)
        if not np.isfinite(pa):
            pa = 0.0
    except (TypeError, ValueError):
        pa = 0.0
    return float(ra), float(dec), pa


def resolve_twilight_files(config):
    """Resolve the twilight flat to use for each colour (F1).

    Supports both the per-colour keys ``{red,green,blue}_twilight_flat`` and the
    singular ``twilight_flat`` key, with the per-colour key taking precedence and
    the singular key as a shared fallback. Returns ``{color: path_or_None}`` for
    ``red``/``green``/``blue``.  A single LLAMAS twilight MEF contains all
    cameras, so multiple colours commonly resolve to the same file.
    """
    shared = config.get('twilight_flat')
    return {color: (config.get(f'{color}_twilight_flat') or shared)
            for color in ('red', 'green', 'blue')}


def _process_flat_for_rss(flat_files, flat_pixel_maps, output_dir,
                          trace_dir, arc_dict_config,
                          timestamp, label='flat',
                          slow_bias=None, fast_bias=None, edge_bias=None):
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
                                      trace_dir=trace_dir, mastercalib_trace_dir=CALIB_DIR,
                                      remove_cosmic_rays=False, edge_bias=edge_bias)
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
                                  clip_range=(0.90, 1.10), use_bias=None,
                                  saturation_threshold=None, unillum_frac=0.1):
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
                use_bias=use_bias,
                output_dir=flat_output_dir,
                trace_dir=trace_dir,
                verbose=verbose,
                filter_size=filter_size,
                signal_thresholds=signal_thresholds,
                clip_range=clip_range,
                saturation_threshold=saturation_threshold,
                unillum_frac=unillum_frac,
            )
        elif method == 'bspline':
            results = process_flat_field_complete(
                red_flat, green_flat, blue_flat,
                arc_calib_file=arc_calib_file,
                use_bias=use_bias,
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
                   cube_radius=1.5, cube_min_weight=0.01, cube_grid_method='oversampled',
                   name_suffix='', resume=False):
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
        name_suffix (str): Tag appended to each output cube filename (e.g. '_skysub'),
            so multiple cube sets (standard vs sky-subtracted) can coexist in the same
            output directory without overwriting. Default '' (no suffix).

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

        # Resume: skip RSS files whose output cube already exists (set clobber=true
        # to force a rebuild). The filename is deterministic from base_name/channel,
        # so this is checked before any (expensive) construction.
        if resume:
            if channel:
                _existing_cube = os.path.join(output_dir, f"{base_name}_cube_{channel}{name_suffix}.fits")
            else:
                _existing_cube = os.path.join(output_dir, f"{base_name}_cube{name_suffix}.fits")
            if os.path.exists(_existing_cube):
                print(f"  RESUME: cube already exists, skipping: {os.path.basename(_existing_cube)}")
                logger.info(f"Resume: skipping existing cube {_existing_cube}")
                cube_files.append(_existing_cube)
                continue

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

                # Create WCS from the real telescope pointing in the RSS header
                # (F6) — previously hardcoded to (0,0), leaving cubes with no
                # on-sky astrometry.
                _ra_c, _dec_c, _pa = _pointing_from_header(
                    getattr(constructor, 'primary_header', None))
                constructor.create_wcs(ra_center=_ra_c, dec_center=_dec_c,
                                       rotation_deg=_pa)
                if _ra_c == 0.0 and _dec_c == 0.0:
                    logger.warning("construct_cube: no telescope pointing in RSS "
                                   "header; cube WCS left at placeholder (0,0)")
                else:
                    logger.info(f"construct_cube: cube WCS centred at RA={_ra_c:.5f}, "
                                f"DEC={_dec_c:.5f}, PA={_pa:.3f} deg")

                # Save cube
                if channel:
                    cube_filename = f"{base_name}_cube_{channel}{name_suffix}.fits"
                else:
                    cube_filename = f"{base_name}_cube{name_suffix}.fits"

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
                
                cube_filename = f"{base_name}{output_suffix}{name_suffix}.fits"
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
                    output_prefix=os.path.join(output_dir, f"{base_name}{name_suffix}"),
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


def _frame_label(science_file):
    """Compact human label for a science exposure, e.g. '02-49 SCI22'.

    Used only for the curated terminal reporter's per-frame sub-status.
    """
    import re as _re
    base = os.path.basename(str(science_file))
    m = _re.search(r'(\d{2}-\d{2})-[\d.]+_((?:SCI|CAL|sci|cal)\d+)', base)
    if m:
        return f"{m.group(1)} {m.group(2).upper()}"
    return os.path.splitext(base)[0][:24]


def _science_stem(science_file):
    """Return a stable identifier for a science exposure.

    The stem is preserved as a substring through the whole per-file product
    naming chain (``{stem}_mef_flat_corrected_extract_RSS_{color}[...]_FF.fits``),
    so it can be used to detect on disk whether a given science file has already
    been reduced. Strips the extension and a trailing ``_mef``.
    """
    base = os.path.splitext(os.path.basename(science_file))[0]
    if base.endswith('_mef'):
        base = base[:-len('_mef')]
    return base


def _has_rss_product(extraction_dir, stem):
    """True if a base RSS product (pre fibre-flat) exists for ``stem``.

    Matches ``*_RSS*.fits`` files that belong to this science stem but are not
    themselves fibre-flat (``_FF``) products. Presence of the RSS means the
    extraction -> wavelength-correction -> sky -> RSS-generation chain for this
    file already completed.
    """
    if not os.path.isdir(extraction_dir):
        return False
    for f in os.listdir(extraction_dir):
        if (stem in f and f.endswith('.fits')
                and '_RSS' in f and '_FF' not in f):
            return True
    return False


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

                # Strip inline comments: a '#' preceded by whitespace begins a
                # trailing comment (e.g. "0.3   # pixel size"). The required
                # leading whitespace means a '#' that is part of a value or path
                # is left intact.
                value = re.split(r'\s+#', value, maxsplit=1)[0].strip()

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

    # Resume control: by default the pipeline reuses any intermediate products
    # already on disk (flat pixel maps, per-science RSS/_FF products, cubes) and
    # skips the stages that produced them, so a re-run only does the work that is
    # actually missing. Set ``clobber = true`` in the config to force every stage
    # to run from scratch (including trace regeneration; normally traces are also
    # reused when ``use_existing_traces`` is true, which is the default).
    clobber = bool(config.get('clobber', False))
    resume = not clobber
    if clobber:
        print("clobber=True — all stages will run from scratch (no resume).")
    else:
        print("Resume enabled — existing intermediate products will be reused "
              "(set clobber=true to force a full re-run).")

    # Configure pipeline logging — log file goes next to the config file
    if 'log_output_dir' in config:
        log_dir = config['log_output_dir']
    else:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(config_path)), 'logs')
    log_file = configure_pipeline_logging(log_dir, retention=int(config.get('log_retention', 10)))
    logger = logging.getLogger(__name__)
    logger.info(f"Pipeline started. Config: {config_path}")

    # ── Curated terminal reporting (terminal_verbose = false, default) ──────
    # Route the driver's ~hundreds of print()/logger.info lines to the log file
    # and show a compact live phase summary on the terminal instead. Ray worker
    # output is already off the terminal (log_to_driver=False). Set
    # terminal_verbose = true to restore the full firehose for debugging.
    import sys as _sys, atexit as _atexit
    from llamas_pyjamas.Utils.reporter import (PipelineReporter, StdoutToLog,
                                               ReporterLogHandler)
    terminal_verbose = bool(config.get('terminal_verbose', False))
    _sf = config.get('science_files')
    _n_frames = len(_sf) if isinstance(_sf, list) else (1 if _sf else 0)
    # Planned phases: bias, trace, [arc], [flat], extract, wave+sky, finalise.
    _total_phases = 5
    if config.get('refine_arc', False) and not config.get('generate_new_wavelength_soln'):
        _total_phases += 1
    if config.get('apply_flat_field_correction', True) and config.get('apply_pixel_flat', True):
        _total_phases += 1
    reporter = PipelineReporter(n_frames=_n_frames, enabled=not terminal_verbose)
    _orig_stdout = _sys.stdout
    _restored = {'done': False}
    if not terminal_verbose:
        _sys.stdout = StdoutToLog(logging.getLogger('llamas_pyjamas.stdout'))
        _parent = logging.getLogger('llamas_pyjamas')
        for _h in list(_parent.handlers):
            if isinstance(_h, logging.StreamHandler) and not isinstance(_h, logging.FileHandler):
                _parent.removeHandler(_h)
        _parent.addHandler(ReporterLogHandler(reporter))

    def _restore_terminal(ok=True):
        if _restored['done']:
            return
        _restored['done'] = True
        try:
            reporter.finish(ok=ok)
        except Exception:
            pass
        _sys.stdout = _orig_stdout
    _atexit.register(_restore_terminal)
    reporter.start(total_phases=_total_phases)

    # --- Ray / temp scratch management ---------------------------------------
    # Redirect all of Ray's temp output (session logs, object-store spill, and the
    # runtime_env package uploads) into one short, owned, per-run directory so it can
    # be bounded and reliably cleaned up. Registers atexit + SIGTERM backstops so the
    # scratch is removed even if the pipeline fails early, and reclaims any leftovers
    # from crashed prior runs. See llamas_pyjamas/Utils/rayManager.py.
    ray_temp_dir = resolve_run_temp_dir(config)
    prune_stale(config)

    # Fail fast if any input lives on an unresponsive filesystem (an offline
    # cloud mount — Box/iCloud/Dropbox — or a dead network share). Otherwise the
    # first os.stat in validate_pipeline_config below blocks indefinitely with no
    # message. This only flags paths whose stat never returns; genuinely missing
    # files are reported by the existence checks that follow.
    check_inputs_reachable(get_input_files_from_config(config),
                           timeout_s=float(config.get('input_timeout_s', 15)))

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

        
    # Parse cosmic ray removal configuration
    remove_cosmic_rays = config.get('remove_cosmic_rays', True)
    mask_output_dir = config.get('mask_output_dir', None)
    if remove_cosmic_rays:
        logger.info("Cosmic ray removal enabled (L.A.Cosmic)")
    else:
        logger.info("Cosmic ray removal disabled")

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

    # Configure Ray object store memory (default 8 GB)
    ray_object_store_mb = config.get('ray_object_store_mb', 8192)
    if isinstance(ray_object_store_mb, str):
        ray_object_store_mb = int(ray_object_store_mb)
    os.environ['LLAMAS_RAY_OBJECT_STORE_MB'] = str(ray_object_store_mb)
    print(f"Configuring Ray object store memory to {ray_object_store_mb} MB")

    # ── Extraction method (applies to EVERY extraction in the run) ──────────
    # 'boxcar' (default): trace-following aperture with fractional pixel weights
    # at the edges (continuous across half-pixel trace crossings). Interim
    # default until the Horne optimal extraction is rebuilt — the current
    # profile-weighted 'optimal' is unstable on cameras with blended profiles.
    # Threaded via environment so science, arc, flat, twilight and sky
    # extractions all use the SAME method — mixing methods breaks the
    # per-fibre throughput calibration.
    extraction_method = str(config.get('extraction_method', 'boxcar')).strip().lower()
    if extraction_method not in ('optimal', 'boxcar', 'horne', 'legacy'):
        print(f"WARNING: unknown extraction_method '{extraction_method}'; using 'boxcar'")
        extraction_method = 'boxcar'
    boxcar_halfwidth = float(config.get('boxcar_halfwidth', 2.5))
    os.environ['LLAMAS_EXTRACT_METHOD'] = extraction_method
    os.environ['LLAMAS_BOXCAR_HALFWIDTH'] = str(boxcar_halfwidth)
    # Detector read noise [DN] for Horne variance weighting (read_noise config key).
    os.environ['LLAMAS_READ_NOISE'] = str(float(config.get('read_noise', 3.5)))
    print(f"Extraction method: {extraction_method}"
          + (f" (fractional aperture half-width {boxcar_halfwidth} px)"
             if extraction_method == 'boxcar' else ""))

    # Optional per-task memory reservation for extraction (default 0 = none).
    # A non-zero value throttles extraction concurrency on low-RAM machines; leave
    # at 0 to schedule purely on ray_num_cpus and avoid the infeasible-memory deadlock.
    ray_task_memory_mb = config.get('ray_task_memory_mb', 0)
    if isinstance(ray_task_memory_mb, str):
        ray_task_memory_mb = int(ray_task_memory_mb)
    os.environ['LLAMAS_RAY_TASK_MEMORY_MB'] = str(ray_task_memory_mb)
    print(f"Configuring Ray per-task memory reservation to {ray_task_memory_mb} MB")


    if not config.get('output_dir'):
        output_dir = os.path.join(BASE_DIR, 'reduced')
    else:
        output_dir = config.get('output_dir')
    os.makedirs(output_dir, exist_ok=True)

    # Wavelength/xshift QA (QA/waveQA.py): per-run HTML report + CSV scorecard.
    # Enabled by default; set wavelength_qa = false to disable. The directory is
    # created lazily on first QA write.
    wavelength_qa = bool(config.get('wavelength_qa', True))
    config['qa_output_dir'] = (config.get('qa_output_dir')
                               or os.path.join(output_dir, 'QA'))
    print(f"Wavelength QA: enabled={wavelength_qa}, dir={config['qa_output_dir']}")

    # Pre-flight disk-space gate: warn when space is tight, abort *before any work*
    # when it is guaranteed insufficient (so no partial output is left behind).
    _preflight_inputs = []
    for _k in ('science_files', 'red_flat_file', 'green_flat_file', 'blue_flat_file',
               'arc_file', 'flat_file'):
        _v = config.get(_k)
        if isinstance(_v, list):
            _preflight_inputs.extend(_v)
        elif _v:
            _preflight_inputs.append(_v)
    preflight_disk_check(config, output_dir, ray_temp_dir, _preflight_inputs)

    # Start THE one Ray session for this run with the canonical config (object store
    # clamped to 30% RAM, py_modules bundle uploaded once, temp/spill in the owned
    # scratch dir). Every stage below calls init_ray(reuse=True) and attaches to this
    # session; it is torn down once at the end via the finally: cleanup_scratch().
    init_ray(config)

    ### Checking for arc file or master wavelength solution

    arc_qa_records = None
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
        # NOTE: refine_arc dispatch now happens AFTER trace selection (below),
        # because the 2d method can extract the night's own arc exposures
        # ({red,green,blue}_arc_file) as its line source, which needs traces.

        
    
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

    # Per-frame edge-bias (DC offset) settings, threaded into every extraction so
    # science and flats are corrected identically. See build_edge_bias_config.
    edge_bias_cfg = build_edge_bias_config(config, extraction_path)
    print(f"Edge-bias DC correction: enabled={edge_bias_cfg['enabled']}, "
          f"min_distance={edge_bias_cfg['min_distance']}px, "
          f"use_flat_mask={edge_bias_cfg['use_flat_mask']}")

    # ── BIAS-FIRST PREPROCESSING (unconditional) ────────────────────────────
    # Bias subtraction is ALWAYS the first data operation for EVERY frame —
    # science, arc, flat, twilight and sky — before flat division, tracing or
    # extraction. Each raw MEF gets (1) the mode-appropriate (READ-MDE) 2D
    # master bias and (2) the per-frame edge DC measured from the
    # unilluminated stripes outside the fibre bundle, then is written to
    # {output_dir}/bias_corrected/ with BIASSUB/EDGE* header stamps that make
    # every downstream stage skip its internal bias handling (exactly-once).
    # Not configurable: any other order is unphysical.
    #
    # WHY: the flat-field division previously ran on frames still carrying
    # their ~500 DN bias pedestal, so percent-level spectral structure in the
    # pixel flat was multiplied by the pedestal instead of the signal,
    # imprinting tens-of-DN fake emission lines (the spurious blue "sky
    # lines" of 2026-07). With the pedestal removed first, the same flat
    # structure perturbs the signal by <1 DN.
    from llamas_pyjamas.Bias.biasFirst import bias_correct_frame
    reporter.phase("🧾", "Bias subtraction")
    print("\n" + "=" * 60)
    print("BIAS-FIRST PREPROCESSING (master bias + edge DC, before all else)")
    print("=" * 60)
    _bias_corr_dir = os.path.join(output_dir, 'bias_corrected')

    def _bias_first(path):
        """Bias-correct one raw frame (resume-aware); returns the new path."""
        if not path or not os.path.exists(path):
            return path
        _expected = os.path.join(
            _bias_corr_dir,
            os.path.splitext(os.path.basename(path))[0] + '_bias_corrected.fits')
        if resume and os.path.exists(_expected):
            print(f"RESUME: reusing bias-corrected {os.path.basename(_expected)}")
            return _expected
        return bias_correct_frame(
            path, _bias_corr_dir,
            slow_bias=slow_bias_file, fast_bias=fast_bias_file,
            trace_dir=config.get('trace_output_dir'),
            mastercalib_trace_dir=CALIB_DIR,
            edge_bias=edge_bias_cfg)

    # Calibration frames (single-path keys)
    for _k in ('red_flat_file', 'green_flat_file', 'blue_flat_file',
               'red_arc_file', 'green_arc_file', 'blue_arc_file',
               'twilight_flat', 'red_twilight_flat', 'green_twilight_flat',
               'blue_twilight_flat'):
        if config.get(_k):
            config[_k] = _bias_first(config[_k])

    # Sky frames (list or comma-separated string)
    _sky_frames = config.get('sky_frame_files')
    if _sky_frames:
        if isinstance(_sky_frames, str):
            _sky_frames = [s.strip() for s in _sky_frames.split(',') if s.strip()]
        config['sky_frame_files'] = [_bias_first(_f) for _f in _sky_frames]

    # Science frames (list or single path)
    _sci = config.get('science_files')
    if isinstance(_sci, list):
        config['science_files'] = [_bias_first(_f) for _f in _sci]
    elif _sci:
        config['science_files'] = _bias_first(_sci)
    print("=" * 60 + "\n")

    # Note: Pixel maps will be created in extractions/flat/ directory during flat field processing
    # No need to pre-create a separate pixel_maps directory
    try:
        reporter.phase("🔦", "Tracing fibres")

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
        if resume and config.get('use_existing_traces', True) and os.path.exists(config.get('trace_output_dir')):
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

        # ── Arc xshift refinement (moved after trace selection so the 2d
        # method can extract the night's own arc exposures) ──
        if config.get('refine_arc', False) and not config.get('generate_new_wavelength_soln'):
            reporter.phase("🌈", "Arc wavelength refinement")
            refine_channels = config.get('refine_arc_channels', None)
            if isinstance(refine_channels, str):
                refine_channels = [c.strip() for c in refine_channels.split(',')]
            ch_label = ','.join(refine_channels) if refine_channels else 'all'
            arc_qa_records = [] if wavelength_qa else None
            refine_method = str(config.get('refine_arc_method', 'perfiber')).strip().lower()
            if refine_method == '2d':
                from llamas_pyjamas.Arc.arcSurface import refineArcX2D

                # Night-of arc line source: extract the same-afternoon arc
                # exposures ({red,green,blue}_arc_file) through the night's
                # traces so the refined solution aligns to the observation
                # epoch rather than the packaged reference arc's. Extraction
                # is resume-aware (existing *_extract.pkl in arcs/ reused).
                line_source = None
                if bool(config.get('refine_arc_use_night_arcs', True)):
                    night_arcs = {}
                    for _ch in ('red', 'green', 'blue'):
                        _f = config.get(f'{_ch}_arc_file')
                        if _f and os.path.exists(_f):
                            night_arcs[_ch] = _f
                    if night_arcs:
                        arc_extract_dir = os.path.join(output_dir, 'arcs')
                        os.makedirs(arc_extract_dir, exist_ok=True)
                        line_source = {}
                        for _ch, _f in night_arcs.items():
                            _base = os.path.splitext(os.path.basename(_f))[0] + '_extract.pkl'
                            _pkl = os.path.join(arc_extract_dir, _base)
                            if resume and os.path.exists(_pkl):
                                print(f"RESUME: reusing extracted night arc for {_ch}: {_base}")
                            else:
                                print(f"Extracting night arc for {_ch}: {os.path.basename(_f)}")
                                run_extraction(_f, arc_extract_dir,
                                               slow_bias=slow_bias_file,
                                               fast_bias=fast_bias_file,
                                               trace_dir=final_trace_dir,
                                               mastercalib_trace_dir=CALIB_DIR,
                                               remove_cosmic_rays=False,
                                               edge_bias=edge_bias_cfg)
                            if os.path.exists(_pkl):
                                line_source[_ch] = _pkl
                            else:
                                print(f"WARNING: night-arc extraction missing for {_ch} "
                                      f"({_pkl}) — that channel refines against the "
                                      f"reference arc spectra")
                        if not line_source:
                            line_source = None
                    else:
                        print("refine_arc_use_night_arcs=true but no "
                              "{red,green,blue}_arc_file found — refining against "
                              "the reference arc spectra")

                print(f"Refining arc xshift with 2D surface + per-fibre "
                      f"perturbations (channels: {ch_label}, "
                      f"line source: {'night arcs' if line_source else 'reference arc'})...")
                arcdict = refineArcX2D(
                    arcdict, channels=refine_channels,
                    qa_collector=arc_qa_records,
                    surface_order_x=int(config.get('arc_surface_order_x', 3)),
                    surface_order_fiber=int(config.get('arc_surface_order_fiber', 2)),
                    perturb_order=int(config.get('arc_perturb_order', 0)),
                    perturb_min_lines=int(config.get('arc_perturb_min_lines', 8)),
                    perturb_shrink_lines=float(config.get('arc_perturb_shrink_lines', 5)),
                    use_unidentified_peaks=bool(config.get('arc_use_unidentified_peaks', True)),
                    blend_min_sep=float(config.get('arc_catalog_min_sep', 8)),
                    line_source=line_source)
            else:
                from llamas_pyjamas.Arc.arcLlamas import refineArcX
                print(f"Refining arc xshift with sub-pixel centroiding (channels: {ch_label})...")
                arcdict = refineArcX(arcdict, channels=refine_channels,
                                     qa_collector=arc_qa_records)
            print(f"Using refined arc: {os.path.basename(arcdict)}")
            config['arcdict'] = arcdict

        # Arc-level wavelength QA: xshift structure of the solution in use (and
        # arc-line fit residuals when refine_arc ran). QA must never kill a run.
        if wavelength_qa:
            try:
                from llamas_pyjamas.QA import waveQA
                waveQA.xshift_structure_qa(config['arcdict'],
                                           qa_dir=config['qa_output_dir'],
                                           label='arc', emit='png')
                if arc_qa_records:
                    waveQA.arc_residual_qa(arc_qa_records,
                                           qa_dir=config['qa_output_dir'],
                                           label='arc', emit='png')
            except Exception as _qa_exc:
                logger.warning(f"Wavelength QA (arc stage) failed: {_qa_exc}")

        # Generate flat field pixel maps if flat correction is enabled
        if config.get('apply_flat_field_correction', True) and config.get('apply_pixel_flat', True):
            reporter.phase("💡", "Flat field")
        flat_pixel_maps = []
        flat_field_method = config.get('flat_field_method', 'simple')
        # Resume: flat_pixel_maps is always a single-element list holding the path
        # to pixel_maps.fits (see process_flat_field_calibration). If both that MEF
        # and the flat_smooth_models.fits (needed by the fibre-flat stage) already
        # exist, we can skip the expensive flat extraction entirely and just point
        # flat_pixel_maps at the existing file.
        _flat_field_dir = config.get('flat_field_output_dir',
                                     os.path.join(extraction_path, 'flat'))
        _flat_products_present = (
            os.path.exists(os.path.join(_flat_field_dir, 'pixel_maps.fits'))
            and os.path.exists(os.path.join(_flat_field_dir, 'flat_smooth_models.fits')))
        if config.get('apply_flat_field_correction', True) and resume and _flat_products_present:
            print("\n" + "="*60)
            print("FLAT FIELD PROCESSING — RESUME (existing products found)")
            print("="*60)
            print(f"Using existing pixel_maps.fits + flat_smooth_models.fits in {_flat_field_dir}")
            print("Skipping flat field generation (set clobber=true to force).")
            logger.info("Resume: skipping flat field generation (products present)")
            flat_pixel_maps = [os.path.join(_flat_field_dir, 'pixel_maps.fits')]
        elif config.get('apply_flat_field_correction', True):
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
                use_bias=slow_bias_file,
                filter_size=filter_size,
                signal_thresholds=signal_thresholds,
                clip_range=(clip_min, clip_max),
                saturation_threshold=config.get('pixel_flat_saturation_threshold'),
                unillum_frac=edge_bias_cfg['flat_frac'],
            )

            if flat_pixel_maps:
                print(f"\nGenerated {len(flat_pixel_maps)} flat field pixel maps:")
            else:
                print("WARNING: No flat field pixel maps generated. Proceeding without flat field correction.")

            """
            # NOTE: This is old code for computing per-fiber relative throughput from the flat field and updating arcdict.
            # May be slated for removal 

            # Compute per-fiber relative throughput from the flat extractions and
            # update arcdict so arcTransfer carries the correct values into the science.
            # Prefer the wavelength-calibrated flat pkl (xshift populated) so that
            # the ratio-on-common-xshift method in fiberRelativeThroughput works.
            _calib_flat_pkl = os.path.join(flat_field_dir, 'combined_flat_extractions_calibrated.pkl')
            _raw_flat_pkl   = os.path.join(flat_field_dir, 'combined_flat_extractions.pkl')
            flat_pkl = _calib_flat_pkl if os.path.exists(_calib_flat_pkl) else _raw_flat_pkl
            if os.path.exists(flat_pkl) and flat_pixel_maps:
                print("\nComputing per-fiber relative throughput from flat field...")
                # Copy arcdict into flat_field_dir so the _shifted_tp output lands there
                # rather than next to the LUT master calibration file.
                import shutil as _shutil
                arc_for_tp = os.path.join(flat_field_dir,
                                          os.path.basename(arcdict))
                _shutil.copy2(arcdict, arc_for_tp)
                arc.fiberRelativeThroughput(flat_pkl, arc_for_tp)
                tp_arcdict = arc_for_tp.replace('.pkl', '_shifted_tp.pkl')
                if os.path.exists(tp_arcdict):
                    arcdict = tp_arcdict
                    config['arcdict'] = arcdict
                    print(f"arcdict updated with throughputs: {os.path.basename(arcdict)}")
                else:
                    print("WARNING: throughput pkl not found after fiberRelativeThroughput — using original arcdict")
            else:
                print("WARNING: combined flat extraction not found, skipping throughput computation")


                logger.warning("No flat field pixel maps generated. Proceeding without flat field correction.")

            """

            # --- Generate flat RSS for fibre-to-fibre correction ---
            # The flat frames (dome or twilight) must have the pixel map applied
            # before extraction so that extracted spectra represent pure fibre
            # throughput (detector per-pixel sensitivity removed).
            config['flat_rss_outputs'] = []
            timestamp_ff = datetime.now().strftime('%Y%m%d_%H%M%S')

            # Prefer twilight flat when specified (per-colour aware, F1).
            twilight_by_color = resolve_twilight_files(config)
            twilight_files = list(dict.fromkeys(
                f for f in twilight_by_color.values() if f))  # distinct, ordered
            twilight_file = twilight_files[0] if twilight_files else None

            if twilight_files and flat_pixel_maps:
                print("\nTwilight flat specified — building fibre-flat RSS from twilight flat...")
                twi_dir = os.path.join(flat_field_dir, 'twilight')
                os.makedirs(twi_dir, exist_ok=True)
                twi_rss = _process_flat_for_rss(
                    twilight_files,
                    flat_pixel_maps, twi_dir,
                    trace_dir=final_trace_dir,
                    arc_dict_config=config.get('arcdict'),
                    timestamp=timestamp_ff, label='twilight',
                    slow_bias=slow_bias_file, fast_bias=fast_bias_file,
                    edge_bias=edge_bias_cfg,
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
                        edge_bias=edge_bias_cfg,
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

        if (config.get('apply_flat_field_correction', True)
                and config.get('apply_pixel_flat', True)
                and flat_pixel_maps):
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

                    # Resume: reuse the existing flat-corrected file if present, keeping
                    # this list aligned with original_science_files for the extraction zip.
                    _existing_corr = os.path.join(
                        flat_output_dir,
                        os.path.splitext(os.path.basename(science_file))[0] + '_flat_corrected.fits')
                    if resume and os.path.exists(_existing_corr):
                        print(f"RESUME: reusing existing flat-corrected file "
                              f"{os.path.basename(_existing_corr)}")
                        flat_corrected_files.append(_existing_corr)
                        continue

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

                _existing_corr = os.path.join(
                    flat_output_dir,
                    os.path.splitext(os.path.basename(science_file))[0] + '_flat_corrected.fits')
                if resume and os.path.exists(_existing_corr):
                    print(f"RESUME: reusing existing flat-corrected file "
                          f"{os.path.basename(_existing_corr)}")
                    science_files_to_process = _existing_corr
                else:
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

        # Control whether large 2D detector images are included in pickled extractions.
        # Default is slim (strips ~7 GB of per-pixel arrays).  Set full_extraction_pickle=true
        # in the config to keep everything for QA/troubleshooting.
        ExtractLlamas._slim_pickle = not config.get('full_extraction_pickle', False)
        print(f"Extraction pickle mode: {'FULL (large 2D arrays included)' if not ExtractLlamas._slim_pickle else 'SLIM (2D detector images stripped)'}")

        # Track whether files were flat-corrected
        were_flat_corrected = bool(
            config.get('apply_flat_field_correction', True)
            and config.get('apply_pixel_flat', True)
            and flat_pixel_maps)
        
        # Build a list of (pkl_path, original_science_fits_path) pairs during extraction.
        # This avoids re-globbing (which could pick up flat/calibration pkls) and keeps
        # each pkl explicitly linked to the original science file for header extraction.
        science_pkl_pairs = []  # list of (pkl_path, original_science_fits_path)

        # Normalise original science files to a list for uniform handling
        original_science_files = config['science_files']
        if not isinstance(original_science_files, list):
            original_science_files = [original_science_files]

        reporter.phase("✳️", "Extracting spectra")
        if isinstance(science_files_to_process, list):
            print(f'\nFound {len(science_files_to_process)} science files to process for extraction.')
            logger.info(f"Stage: Extracting {len(science_files_to_process)} science files")

            for i, (science_file, orig_file) in enumerate(zip(science_files_to_process, original_science_files)):
                reporter.frame(i + 1, len(science_files_to_process), _frame_label(orig_file))
                # Resume: if this exposure already has an RSS product on disk, its
                # whole extraction -> wavelength -> sky -> RSS chain is done. Skip it
                # (not appending to science_pkl_pairs skips its post-processing too).
                if resume and _has_rss_product(extraction_path, _science_stem(orig_file)):
                    print(f"RESUME: RSS product already exists for "
                          f"{os.path.basename(orig_file)} — skipping extraction/RSS.")
                    logger.info(f"Resume: skipping extraction for {orig_file} (RSS present)")
                    continue
                print(f"Extracting science file {i+1}/{len(science_files_to_process)}: {os.path.basename(science_file)}")
                # Process each science file with hybrid trace support
                extracted_basename = run_extraction(
                    science_file,
                    extraction_path,
                    slow_bias=slow_bias_file,
                    fast_bias=fast_bias_file,
                    trace_dir=final_trace_dir,              # User traces
                    mastercalib_trace_dir=CALIB_DIR,        # Mastercalib fallback
                    remove_cosmic_rays=remove_cosmic_rays,
                    mask_output_dir=mask_output_dir,
                    edge_bias=edge_bias_cfg
                )
                print(f"Extraction completed for {os.path.basename(science_file)}. Output file: {extracted_basename}")
                if extracted_basename:
                    science_pkl_pairs.append((os.path.join(extraction_path, extracted_basename), orig_file))
        elif resume and _has_rss_product(extraction_path, _science_stem(original_science_files[0])):
            print(f"RESUME: RSS product already exists for "
                  f"{os.path.basename(original_science_files[0])} — skipping extraction/RSS.")
            logger.info(f"Resume: skipping extraction for {original_science_files[0]} (RSS present)")
        else:
            print(f"Extracting science file: {os.path.basename(science_files_to_process)}")
            extracted_basename = run_extraction(
                science_files_to_process,
                extraction_path,
                slow_bias=slow_bias_file,
                fast_bias=fast_bias_file,
                trace_dir=final_trace_dir,                  # User traces
                mastercalib_trace_dir=CALIB_DIR,            # Mastercalib fallback
                remove_cosmic_rays=remove_cosmic_rays,
                mask_output_dir=mask_output_dir,
                edge_bias=edge_bias_cfg
            )
            print(f"Extraction completed. Used traces from {final_trace_dir} with mastercalib fallback. Output file: {extracted_basename}")
            if extracted_basename:
                science_pkl_pairs.append((os.path.join(extraction_path, extracted_basename), original_science_files[0]))

        # ── Sky-fibre selection setup (base model skyModel_1d) ──
        # Resolve which fibres/regions build the sky model. 'frame' extracts a
        # dedicated blank-sky MEF here (once, reused for every science file);
        # 'skymap' preloads the user sky map; 'dimmest'/'middle-third' need no setup.
        sky_selection_method = str(config.get('sky_selection_method', 'stratified')).lower()
        sky_n_fibres = int(config.get('sky_n_fibres', 20))
        sky_map_obj = None
        # Backward-compatible: an explicit sky_extraction_file still works.
        sky_frame_extraction_file = config.get('sky_extraction_file', None)

        if sky_selection_method == 'skymap':
            sky_map_path = config.get('sky_map_file', None)
            if sky_map_path:
                from llamas_pyjamas.Sky import skySelect
                print(f"Loading sky map for fibre selection: {sky_map_path}")
                sky_map_obj = skySelect.load_sky_map(sky_map_path)
            else:
                print("WARNING: sky_selection_method='skymap' but no sky_map_file set; "
                      "falling back to 'quantile'")
                sky_selection_method = 'stratified'

        if sky_selection_method == 'frame' and not sky_frame_extraction_file:
            sky_frame_files = config.get('sky_frame_files', None)
            if isinstance(sky_frame_files, str):
                sky_frame_files = [s.strip() for s in sky_frame_files.split(',') if s.strip()]
            if sky_frame_files:
                if len(sky_frame_files) > 1:
                    print(f"NOTE: {len(sky_frame_files)} sky frames supplied; using the "
                          f"first ({os.path.basename(sky_frame_files[0])}). Multi-frame "
                          "combination is not yet implemented.")
                    logger.warning("sky_frame_files: using only the first of %d frames",
                                   len(sky_frame_files))
                use_flat = (config.get('apply_flat_field_correction', True)
                            and config.get('apply_pixel_flat', True) and flat_pixel_maps)
                sky_frame_extraction_file = _extract_sky_frame(
                    sky_frame_files[0], extraction_path,
                    slow_bias_file, fast_bias_file, final_trace_dir,
                    remove_cosmic_rays, mask_output_dir, config,
                    flat_pixel_maps=flat_pixel_maps if use_flat else None)
            else:
                print("WARNING: sky_selection_method='frame' but no sky_frame_files set; "
                      "falling back to 'quantile'")
                sky_selection_method = 'stratified'

        reporter.phase("🌌", "Wavelength & sky subtraction")
        for index, (correction_path, orig_science_file) in enumerate(science_pkl_pairs):
            reporter.frame(index + 1, len(science_pkl_pairs), _frame_label(orig_science_file))
            print(f"Processing extraction file {index+1}/{len(science_pkl_pairs)}: {correction_path}")
            if not os.path.exists(correction_path):
                raise FileNotFoundError(f"Extraction file {correction_path} does not exist.")

            # Correct wavelengths for each extraction file
            corr_extractions, _ = correct_wavelengths(correction_path, soln=config.get('arcdict'))

            # Per-fibre throughput from the twilight (sky) flat, extracted with
            # the same method as the science (default on; twilight_throughput =
            # false reverts to the arc-carried lamp values).
            if config.get('twilight_throughput', True):
                _twi_dir = os.path.join(
                    config.get('flat_field_output_dir',
                               os.path.join(extraction_path, 'flat')), 'twilight')
                _n_twi = apply_twilight_throughput(corr_extractions, _twi_dir) \
                    if os.path.isdir(_twi_dir) else 0
                if _n_twi:
                    print(f"Twilight throughput applied to {_n_twi} cameras")
                else:
                    print("Twilight throughput: no twilight extraction found — "
                          "keeping arc-carried values")

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

            # Optionally refine per-fiber xshift using sky line centroids
            sky_x_refine = config.get('sky_line_refinement', False)
            rss_input_file = savefile
            if sky_x_refine:
                from llamas_pyjamas.Sky.skyLlamas import refineSkyX
                sky_x_channels = config.get('sky_line_channels', None)
                if isinstance(sky_x_channels, str):
                    sky_x_channels = [c.strip() for c in sky_x_channels.split(',')]
                ch_label = ','.join(sky_x_channels) if sky_x_channels else 'all'
                print(f"Refining xshift from sky lines (channels: {ch_label})...")
                savefile = refineSkyX(savefile, channels=sky_x_channels)
                print(f"Sky xshift refinement complete: {os.path.basename(savefile)}")

            # Optionally run sky subtraction, populating the .sky attribute on each fiber
            sky_subtract = config.get('sky_subtract', True)
            rss_input_file = savefile
            if sky_subtract:
                print(f"Running sky subtraction on {os.path.basename(savefile)} "
                      f"(selection='{sky_selection_method}')...")
                sky1d_file = skyModel_1d(savefile, color=None,
                                         sky_extraction_file=sky_frame_extraction_file,
                                         show_plots=config.get('sky_qa_plots', False),
                                         selection_method=sky_selection_method,
                                         n_sky_fibres=sky_n_fibres,
                                         sky_map=sky_map_obj,
                                         arc_soln=config.get('arcdict'))
                rss_input_file = sky1d_file
                print(f"Sky subtraction complete. Sky model saved to {os.path.basename(sky1d_file)}")

                # Per-science wavelength QA: final xshift (incl. refineSkyX if
                # enabled) + populated .sky. Writes one HTML report + CSV per
                # science frame into qa_output_dir. Never fatal. Note: lives
                # inside the loop that resume skips, so QA regenerates only when
                # extraction reruns; the standalone CLI
                # (python -m llamas_pyjamas.QA.waveQA) covers existing products.
                if wavelength_qa:
                    try:
                        from llamas_pyjamas.QA import waveQA
                        waveQA.run_wavelength_qa(
                            sky1d_file,
                            qa_dir=config['qa_output_dir'],
                            run_label=base_name.split('_flat_corrected')[0],
                            arc_qa_records=arc_qa_records)
                    except Exception as _qa_exc:
                        logger.warning(f"Wavelength QA failed for {base_name}: {_qa_exc}")

                # Remove superseded intermediate pkls to save disk space.
                # _corrected_extractions.pkl is now superseded by sky1d (via skyX if used)
                corrected_pkl = os.path.join(extraction_path, f'{base_name}_corrected_extractions.pkl')
                for _old in ([savefile] if sky_x_refine and savefile != sky1d_file else []) + \
                            ([corrected_pkl] if corrected_pkl != rss_input_file and os.path.exists(corrected_pkl) else []):
                    try:
                        os.remove(_old)
                        print(f"Removed intermediate file: {os.path.basename(_old)}")
                    except OSError:
                        pass
            elif wavelength_qa:
                # No sky subtraction: still QA the xshift structure of the
                # wavelength-corrected extraction (sky panels will be skipped).
                try:
                    from llamas_pyjamas.QA import waveQA
                    waveQA.run_wavelength_qa(
                        savefile,
                        qa_dir=config['qa_output_dir'],
                        run_label=base_name.split('_flat_corrected')[0],
                        arc_qa_records=arc_qa_records)
                except Exception as _qa_exc:
                    logger.warning(f"Wavelength QA failed for {base_name}: {_qa_exc}")

            # Optionally build a NOFLAT comparison extraction (from the original, pre-flat FITS)
            noflat_rss_file = None
            print(f"\nNOFLAT check: noflat_comparison={config.get('noflat_comparison', False)!r}, "
                  f"were_flat_corrected={were_flat_corrected!r}")
            if config.get('noflat_comparison', False):
                if not were_flat_corrected:
                    print("NOFLAT comparison requested but pixel flat was not applied "
                          "(apply_pixel_flat=false or no flat maps available) — skipping NOFLAT extension")
                else:
                    orig_science = config.get('science_files')
                    if isinstance(orig_science, list):
                        print("NOFLAT comparison: list of science files not supported — skipping NOFLAT extension")
                    elif orig_science and os.path.exists(orig_science):
                        print("\nBuilding NOFLAT comparison extraction from original (pre-flat) FITS...")
                        try:
                            noflat_extracted = run_extraction(
                                orig_science, extraction_path,
                                trace_dir=final_trace_dir,
                                mastercalib_trace_dir=CALIB_DIR,
                                edge_bias=edge_bias_cfg
                            )
                            noflat_corr_dict, noflat_hdr = correct_wavelengths(
                                noflat_extracted, soln=config.get('arcdict'))
                            noflat_objs = noflat_corr_dict['extractions']

                            # Copy sky model from the main (flat-corrected) extraction
                            main_dict = ExtractLlamas.loadExtraction(rss_input_file)
                            main_objs = main_dict['extractions']
                            for nf_obj, main_obj in zip(noflat_objs, main_objs):
                                nf_obj.sky = getattr(main_obj, 'sky', np.zeros_like(nf_obj.counts))

                            noflat_rss_file = os.path.join(extraction_path, f'{base_name}_noflat_extractions.pkl')
                            save_extractions(noflat_objs, primary_header=noflat_hdr, savefile=noflat_rss_file)
                            print(f"NOFLAT extraction saved: {os.path.basename(noflat_rss_file)}")

                            # The raw _extract.pkl for the original FITS is now superseded
                            try:
                                os.remove(noflat_extracted)
                            except OSError:
                                pass
                        except Exception as _nf_err:
                            import traceback as _tb
                            print(f"WARNING: NOFLAT extraction failed: {_nf_err}")
                            _tb.print_exc()
                            print("Proceeding without NOFLAT extension.")
                            noflat_rss_file = None
                    else:
                        print(f"NOFLAT comparison: original science file not found "
                              f"({orig_science!r}) — skipping NOFLAT extension")

            # Create a logger for RSS generation
            rss_logger = logging.getLogger(__name__ + '.rss')
            rss_logger.info(f"Starting RSS generation for {base_name}")

            rss_output_file = os.path.join(extraction_path, f'{base_name}_RSS.fits')

            # RSS generation (subtract_sky mirrors sky_subtract; SKY extension always written)
            rss_gen = RSSgeneration(logger=rss_logger)
            print(f"Calling generate_rss: noflat_file={noflat_rss_file!r}")
            new_rss_outputs = rss_gen.generate_rss(rss_input_file, rss_output_file,
                                                    subtract_sky=sky_subtract,
                                                    noflat_file=noflat_rss_file)
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

            # RSS files that still need a fibre-flat product. On resume, drop any
            # that already have their _FF.fits so the (expensive) twilight
            # reduction below is skipped entirely when nothing is pending.
            _rss_all = [
                os.path.join(extraction_path, f)
                for f in os.listdir(extraction_path)
                if f.endswith('.fits') and '_RSS' in f and '_FF' not in f
            ]
            if resume:
                _rss_pending = [r for r in _rss_all
                                if not os.path.exists(r.replace('.fits', '_FF.fits'))]
            else:
                _rss_pending = _rss_all

            if os.path.exists(smooth_models_file) and not _rss_pending:
                print("RESUME: all _FF products present — skipping fibre-to-fibre "
                      "flat correction (set clobber=true to force).")
                logger.info("Resume: skipping fibre-flat (all _FF present)")
            elif os.path.exists(smooth_models_file):
                print("\n" + "=" * 60)
                print("FIBRE-TO-FIBRE FLAT CORRECTION")
                print("=" * 60)

                # F1: resolve twilight flats PER COLOUR. The previous code read
                # only the singular ``twilight_flat`` key and silently ignored
                # the per-colour keys ``{red,green,blue}_twilight_flat`` that the
                # LLAMAS config templates use, forcing a lamp-only fallback.
                twilight_by_color = resolve_twilight_files(config)
                # Group colours by the distinct file that serves them (a single
                # LLAMAS twilight MEF holds all cameras, so several colours often
                # share one file).
                files_to_colors = {}
                for color in ('red', 'green', 'blue'):
                    tfile = twilight_by_color.get(color)
                    if tfile and os.path.exists(tfile):
                        files_to_colors.setdefault(tfile, []).append(color)
                corrections_file = None

                if files_to_colors:
                    # Branch A: Twilight + Lamp (per colour)
                    summary = ", ".join(
                        f"{'/'.join(cs)}={os.path.basename(f)}"
                        for f, cs in files_to_colors.items())
                    print(f"Using twilight flat(s): {summary}")
                    logger.info(f"Twilight flats (per colour): {summary}")
                    try:
                        merged_ext, merged_meta = [], []
                        merged_primary = None
                        used_colors = set()
                        for tfile, colors in files_to_colors.items():
                            twi = reduce_twilight_flat(
                                tfile,
                                flat_pixel_maps[0] if flat_pixel_maps else None,
                                final_trace_dir,
                                config.get('arcdict'),
                                slow_bias_file,
                                extraction_path,
                                fast_bias=fast_bias_file,
                                edge_bias=edge_bias_cfg,
                            )
                            merged_primary = twi.get('primary_header',
                                                     merged_primary)
                            # Keep only the channels this file is assigned to, so
                            # a per-colour twilight flat contributes only its
                            # colour(s) to the merged set.
                            for ext, meta in zip(twi['extractions'],
                                                 twi['metadata']):
                                if meta['channel'] in colors:
                                    merged_ext.append(ext)
                                    merged_meta.append(meta)
                                    used_colors.add(meta['channel'])
                            del twi
                            gc.collect()
                        missing = {'red', 'green', 'blue'} - used_colors
                        if missing:
                            logger.warning(
                                f"No twilight flat for colour(s) "
                                f"{sorted(missing)}; those benchsides will use "
                                f"the lamp-derived T_i fallback within the "
                                f"twilight method.")
                            print(f"  NOTE: colour(s) {sorted(missing)} have no "
                                  f"twilight flat — lamp-only T_i for those.")
                        twi_extractions = {
                            'extractions': merged_ext,
                            'metadata': merged_meta,
                            'primary_header': merged_primary,
                        }
                        corrections_file = compute_fibre_flat_twilight(
                            twi_extractions, smooth_models_file,
                            flat_field_dir,
                            integration_range=config.get(
                                'fibre_flat_integration_range'),
                            poly_order=config.get(
                                'fibre_flat_poly_order', None),
                        )
                        del twi_extractions, merged_ext, merged_meta
                        gc.collect()
                        print(f"Fibre flat computed (twilight + lamp method; "
                              f"colours: {sorted(used_colors)})")
                    except Exception as e:
                        print(f"WARNING: Twilight reduction failed: {e}")
                        logger.warning(f"Twilight reduction failed: {e}", exc_info=True)
                        print("Falling back to lamp-only fibre flat")
                        traceback.print_exc()
                        # Best-effort cleanup of any partial intermediates.
                        try:
                            del twi_extractions
                        except NameError:
                            pass
                        try:
                            del merged_ext, merged_meta
                        except NameError:
                            pass
                        gc.collect()
                        corrections_file = None

                if corrections_file is None:
                    # Branch B: Lamp-only fallback
                    if files_to_colors:
                        print("Twilight flat not available or failed — "
                              "using lamp-only fallback")
                    else:
                        logger.warning(
                            "No twilight flat provided (checked twilight_flat and "
                            "{red,green,blue}_twilight_flat). Falling back to "
                            "lamp-only fibre-to-fibre flat. Science cubes will "
                            "contain artificial spatial illumination gradients "
                            "from the lamp.")
                    corrections_file = compute_fibre_flat_lamp_only(
                        smooth_models_file, flat_field_dir)
                    print("Fibre flat computed (lamp-only method)")

                # Apply to the pending RSS files (resume-filtered above).
                for rss_file in _rss_pending:
                    ff_output = rss_file.replace('.fits', '_FF.fits')
                    apply_fibre_flat_to_rss(rss_file, corrections_file,
                                           ff_output)
                    print(f"  FF RSS: {os.path.basename(ff_output)}")
            else:
                print("WARNING: Smooth models file not found — "
                      "skipping fibre-to-fibre flat correction")
                logger.warning("Smooth models file not found — skipping fibre-to-fibre flat correction")

        # ── Sky-subtraction framework (post fibre-flat, per-colour) ──
        # Builds on the base SKY model already in the FF files and writes
        # LLAMAS_{name}_RSS_{color}_FF_SKYSUB.fits. Disabled by default.
        if config.get('sky_framework', False):
            from llamas_pyjamas.Sky.skySubtract import subtract_sky_all_colors
            from llamas_pyjamas.Sky.skyConfig import SkySubtractConfig
            ff_for_sky = [
                os.path.join(extraction_path, f)
                for f in os.listdir(extraction_path)
                if f.endswith('_FF.fits') and '_RSS' in f
            ]
            # Resume: skip FF files that already have their _FF_SKYSUB.fits product.
            if resume:
                _sky_pending = [f for f in ff_for_sky
                                if not os.path.exists(f.replace('_FF.fits', '_FF_SKYSUB.fits'))]
                if ff_for_sky and not _sky_pending:
                    print("RESUME: all _FF_SKYSUB products present — skipping sky framework "
                          "(set clobber=true to force).")
                    logger.info("Resume: skipping sky framework (all _FF_SKYSUB present)")
                ff_for_sky = _sky_pending
            if ff_for_sky:
                print("\n" + "=" * 60)
                print("SKY-SUBTRACTION FRAMEWORK")
                print("=" * 60)
                sky_cfg = SkySubtractConfig.from_pipeline_config(config)
                skysub_files = subtract_sky_all_colors(ff_for_sky, config=sky_cfg)
                for sf in skysub_files:
                    print(f"  SKYSUB RSS: {os.path.basename(sf)}")
                logger.info(f"Sky framework produced: {skysub_files}")
            else:
                print("Sky framework enabled but no *_FF.fits files found — skipping")
                logger.warning("sky_framework=True but no _FF.fits files present")

        # Cube construction from RSS files
        reporter.phase("🧊", "RSS & cubes")
        print("Constructing cubes from RSS files...")
        logger.info("Stage: Constructing cubes from RSS files")
        all_rss = [os.path.join(extraction_path, f)
                   for f in os.listdir(extraction_path)
                   if f.endswith('.fits') and '_RSS' in f]
        # Two independent cube sets, both built by default:
        #   - standard set from the fibre-flat (_FF) RSS  -> ..._cube_{color}.fits
        #   - sky-subtracted set from the framework (_FF_SKYSUB) RSS -> ..._cube_{color}_skysub.fits
        # _FF_SKYSUB files end in '_FF_SKYSUB.fits' (not '_FF.fits'), so the two lists are disjoint.
        skysub_rss = [f for f in all_rss if '_FF_SKYSUB' in os.path.basename(f)]
        ff_rss = [f for f in all_rss if os.path.basename(f).endswith('_FF.fits')]
        non_ff_rss = [f for f in all_rss if '_FF' not in os.path.basename(f)]
        # Standard set uses the _FF RSS; falls back to plain RSS only if fibre-flat
        # produced no _FF files at all.
        standard_rss = ff_rss if ff_rss else non_ff_rss

        if 'cube_output_dir' not in config:
            cube_output_dir = os.path.join(output_dir, 'cubes')
        else:
            cube_output_dir = config.get('cube_output_dir')
        os.makedirs(cube_output_dir, exist_ok=True)

        gen_standard = config.get('generate_standard_cubes', True)
        gen_skysub = config.get('generate_skysub_cubes', True)

        def _build_cubes(rss_list, name_suffix):
            return construct_cube(
                rss_list,
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
                cube_grid_method=config.get('cube_grid_method', 'oversampled'),
                name_suffix=name_suffix,
                resume=resume,
            )

        cube_files = []
        if _gen_cubes:
            if gen_standard and standard_rss:
                print(f"Building standard cubes from {len(standard_rss)} RSS file(s)...")
                logger.info(f"Building standard cube set from {len(standard_rss)} RSS files")
                cube_files += _build_cubes(standard_rss, '')
            elif gen_standard:
                print("generate_standard_cubes=True but no standard RSS files found")
                logger.warning("generate_standard_cubes=True but no standard RSS files found")

            if gen_skysub and skysub_rss:
                print(f"Building sky-subtracted (_skysub) cubes from {len(skysub_rss)} RSS file(s)...")
                logger.info(f"Building sky-subtracted cube set from {len(skysub_rss)} RSS files")
                cube_files += _build_cubes(skysub_rss, '_skysub')
            elif gen_skysub:
                print("generate_skysub_cubes=True but no *_FF_SKYSUB.fits found "
                      "(did the sky framework run?)")
                logger.warning("generate_skysub_cubes=True but no _FF_SKYSUB RSS files found "
                               "(sky framework may not have run)")

            if cube_files:
                print(f"Cubes constructed: {cube_files}")
            else:
                print("No cubes constructed (no matching RSS files or both sets disabled)")
        else:
            print("Cube generation disabled (generate_cubes=False)")
                
        
        
    except Exception as e:
        traceback.print_exc()
        print(f"An error occurred: {e}")
        logger.error(f"Pipeline failed: {e}", exc_info=True)
    finally:
        # Restore the terminal (finalise the reporter, un-redirect stdout) before
        # teardown so the completion line and any traceback render normally.
        # exc_info is set iff an exception is propagating through this finally.
        _restore_terminal(ok=(_sys.exc_info()[0] is None))
        # Prompt teardown: shut Ray down and remove the run scratch dir (honours
        # cleanup_scratch=false). The atexit/SIGTERM backstops guarantee this also
        # runs for failures before this try (validate/flat/arc) and on signals.
        cleanup_scratch()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reduce module placeholder")
    parser.add_argument('config_file', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    
    main(args.config_file)