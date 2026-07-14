from   astropy.io import fits
import scipy
import numpy as np
import os
from datetime import datetime
from   matplotlib import pyplot as plt
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl
import pickle, cloudpickle
import logging
import argparse, glob
import ray, multiprocessing, psutil
from llamas_pyjamas.Utils.rayManager import init_ray
import traceback

import pkg_resources
from pathlib import Path

import llamas_pyjamas
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR, BIAS_DIR
from llamas_pyjamas.Trace.traceLlamasMaster import _grab_bias_hdu, TraceRay
import llamas_pyjamas.Trace.traceLlamasMaster as traceLlamasMaster
import sys
sys.modules['traceLlamasMaster'] = traceLlamasMaster

from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions, load_extractions
from llamas_pyjamas.Image.WhiteLightModule import WhiteLight, WhiteLightFits, WhiteLightQuickLook
import time

from llamas_pyjamas.File.llamasIO import process_fits_by_color
from llamas_pyjamas.DataModel.validate import get_placeholder_extension_indices, validate_for_gui
from llamas_pyjamas.Bias import BiasNotFoundError, BiasReadModeError, generate_fallback_bias_hdu
from llamas_pyjamas.Bias.biasChecking import (build_interfibre_mask,
                                              build_topbottom_stripe_mask,
                                              measure_edge_dc_offset)
from llamas_pyjamas.Masking.cosmicLlamas import clean_cosmic_rays, save_cosmic_ray_masks

# Set up logging
logger = logging.getLogger(__name__)



####################################################################################

def ExtractLlamasCube(infits, tracefits, optimal=True):

    assert infits.endswith('.fits'), 'File must be a .fits file'
    # hdu = fits.open(infits)
    hdu, _ = process_fits_by_color(infits)

    # Find the trace files
    basefile = os.path.basename(tracefits).split('.fits')[0]
    trace_files = glob.glob(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))
    extraction_file = os.path.splitext(os.path.basename(infits))[0] + '_extract.pkl'

    if len(trace_files) == 0:
        logger.error("No trace files found for the indicated file root!")
        return None
    
    hdu_trace_pairs = match_hdu_to_traces(hdu, trace_files)
    logger.debug(f"HDU trace pairs: {hdu_trace_pairs}")

    extraction_list = []

    logger.info(f"Saving extractions to {extraction_file}")

    counter = 1
    for hdu_index, file in hdu_trace_pairs:

        logger.debug(f"Extracting extension number {counter} of 24")
        hdr = hdu[hdu_index].header 
        bias = np.nanmedian(hdu[hdu_index].data.astype(float))  
        
        try:
            with open(file, mode='rb') as f:
                tracer = pickle.load(f)
    
            extraction = ExtractLlamas(tracer, hdu[hdu_index].data.astype(float)-bias, hdu[hdu_index].header)
            extraction_list.append(extraction)
            
        except Exception as e:
            logger.error(f"Error extracting trace from {file}")
            logger.error(traceback.format_exc())
        counter += 1
        
    logger.debug(f'Extraction list = {extraction_list}')        
    filename = save_extractions(extraction_list, savefile=extraction_file)
    logger.info(f'extraction saved filename = {filename}')

    return None


def get_trace_file(channel, bench, side, trace_dir):
    """
    Find trace file for specific camera configuration.

    Args:
        channel: Color channel (red/green/blue)
        bench: Bench number
        side: Side letter (A/B)
        trace_dir: Directory containing trace files

    Returns:
        str: Path to trace file

    Raises:
        FileNotFoundError: If trace file not found
    """
    # Standard trace file naming: LLAMAS_master_{channel}_{bench}_{side}_traces.pkl
    trace_filename = f'LLAMAS_master_{channel.lower()}_{bench}_{side}_traces.pkl'
    trace_path = os.path.join(trace_dir, trace_filename)

    if os.path.exists(trace_path):
        return trace_path

    # Try alternate naming without "master"
    alt_filename = f'LLAMAS_{channel.lower()}_{bench}_{side}_traces.pkl'
    alt_path = os.path.join(trace_dir, alt_filename)

    if os.path.exists(alt_path):
        return alt_path

    raise FileNotFoundError(
        f"Trace file not found for {channel}{bench}{side} in {trace_dir}\n"
        f"  Tried: {trace_filename}, {alt_filename}"
    )


def match_hdu_to_traces(hdu_list, trace_files, start_idx=1):
    """Match HDU extensions to their corresponding trace files"""
    matches = []

    # Skip primary HDU (index 0)
    #### need to be super careful with this starting index
    for idx in range(start_idx, len(hdu_list)):

        header = hdu_list[idx].header

        # Get color and benchside from header
        if 'COLOR' in header:
            color = header['COLOR'].lower()
            bench = header['BENCH']
            side = header['SIDE']
        else:
            camname = header['CAM_NAME']
            color = camname.split('_')[1].lower()
            bench = camname.split('_')[0][0]
            side = camname.split('_')[0][1]

        benchside = f"{bench}{side}"
        pattern = f"{color}_{bench}_{side}_traces"

        # Find matching trace file
        matching_trace = next(
            (tf for tf in trace_files
             if pattern in os.path.basename(tf)),
            None
        )
        print(f'HDU {idx}: Looking for pattern "{pattern}" -> {os.path.basename(matching_trace) if matching_trace else "NOT FOUND"}')
        if matching_trace:
            matches.append((idx, matching_trace))
        else:
            logger.warning(f"No matching trace found for HDU {idx}: {color} {benchside}, pattern: {pattern}")
            print(f"  Available trace files: {[os.path.basename(tf) for tf in trace_files]}")

    return matches

# Define a Ray remote function for processing a single trace extraction.
# NOTE: no hard `memory=` reservation here. A fixed per-task memory reservation
# can deadlock extraction when Ray's auto-detected logical `memory` pool (≈ free
# RAM − object store) drops below the reservation, in which case the task becomes
# permanently unschedulable. Concurrency is governed by `num_cpus` instead. To opt
# back into a reservation on low-RAM machines, set `ray_task_memory_mb` in the
# config (applied via `.options(memory=...)` at dispatch).
def _load_unillum_mask(mask_dir, bench, side, color, shape):
    """Load the per-camera flat-derived unilluminated mask extension, or None.

    Looks for ``unillum_mask.fits`` in ``mask_dir`` (written by the flat stage)
    and returns the boolean 2-D mask whose BENCH/SIDE/COLOR match this detector.
    """
    if not mask_dir:
        return None
    path = os.path.join(mask_dir, 'unillum_mask.fits')
    if not os.path.exists(path):
        return None
    try:
        with fits.open(path) as hml:
            for h in hml[1:]:
                hh = h.header
                if (str(hh.get('BENCH', '')).strip() == str(bench)
                        and str(hh.get('SIDE', '')).strip().upper() == str(side).upper()
                        and str(hh.get('COLOR', '')).strip().lower() == str(color).lower()):
                    if h.data is None:
                        return None
                    m = np.asarray(h.data).astype(bool)
                    return m if m.shape == tuple(shape) else None
    except Exception as exc:
        logger.warning(f"_load_unillum_mask: failed to read {path}: {exc}")
    return None


def _apply_edge_bias(data, tracer, header, bench, side, color, edge_bias):
    """Measure and subtract a per-extension DC offset from unilluminated pixels.

    Runs *after* master-bias subtraction so it captures the residual nightly DC
    drift the static master bias misses. The mask is the top/bottom stripes
    outside the fibre stack (always), optionally unioned with a flat-derived
    per-pixel dark mask (Tier 2). Writes EDGE* header keywords and returns
    ``(corrected_data, stats_dict)``. The ``stats_dict`` also carries the
    stripes-only and stripes+flat candidate levels for QA comparison.
    """
    eb = edge_bias or {}
    dmin = float(eb.get('min_distance', 20.0))
    min_px = int(eb.get('min_pixels', 500))
    stats = {'edge_dc': 0.0, 'edge_npix': 0, 'edge_dmin': dmin, 'edge_src': 'disabled',
             'level_stripes': float('nan'), 'level_combined': float('nan'),
             'bench': bench, 'side': side, 'color': color}

    # Guard against double application: bias-first preprocessing (biasFirst.py)
    # already measured and subtracted the edge DC and stamped EDGEDC/EDGESRC.
    if 'EDGEDC' in header and str(header.get('EDGESRC', '')) not in ('', 'disabled'):
        stats.update(edge_dc=float(header.get('EDGEDC', 0.0)),
                     edge_npix=int(header.get('EDGENPIX', 0)),
                     edge_src='pre-applied')
        logger.info(f"Edge-bias {bench}{side} {color}: already applied upstream "
                    f"(EDGEDC={header.get('EDGEDC', 0.0):.2f} DN) — skipping")
        return data, stats

    if not eb.get('enabled', True):
        header['EDGEDC']   = (0.0, 'Edge-bias DC offset subtracted (DN)')
        header['EDGENPIX'] = (0, 'Edge-bias mask pixel count')
        header['EDGEDMIN'] = (dmin, 'Edge-bias min distance from fibre (px)')
        header['EDGESRC']  = ('disabled', 'Edge-bias mask source')
        return data, stats

    stripe_mask = build_topbottom_stripe_mask(tracer, data.shape, min_distance=dmin)
    lvl_s, n_s, st_s = measure_edge_dc_offset(data, stripe_mask, min_pixels=min_px)
    stats['level_stripes'] = lvl_s if st_s == 'ok' else float('nan')

    chosen_lvl, chosen_n, chosen_st, src = lvl_s, n_s, st_s, 'stripes'

    # Tier 2: optionally union with the flat-derived L/R dark zones.
    if eb.get('use_flat_mask', False):
        flat_mask = _load_unillum_mask(eb.get('flat_mask_dir'), bench, side, color, data.shape)
        if (flat_mask is not None and hasattr(tracer, 'fiberimg')
                and tracer.fiberimg is not None):
            onfib = tracer.fiberimg >= 0
            combined = stripe_mask | (onfib & flat_mask)
            lvl_c, n_c, st_c = measure_edge_dc_offset(data, combined, min_pixels=min_px)
            stats['level_combined'] = lvl_c if st_c == 'ok' else float('nan')
            chosen_lvl, chosen_n, chosen_st, src = lvl_c, n_c, st_c, 'stripes+flat'
        else:
            logger.info(
                f"Edge-bias: flat mask requested but unavailable for "
                f"{bench}{side} {color}; using stripes only")

    if chosen_st == 'ok':
        corrected = data - chosen_lvl
        edge_dc, edge_src = float(chosen_lvl), src
    else:
        corrected = data
        edge_dc, edge_src = 0.0, ('skipped_placeholder' if chosen_n > 0 else 'skipped')

    stats.update(edge_dc=edge_dc, edge_npix=int(chosen_n), edge_src=edge_src)
    header['EDGEDC']   = (edge_dc, 'Edge-bias DC offset subtracted (DN)')
    header['EDGENPIX'] = (int(chosen_n), 'Edge-bias mask pixel count')
    header['EDGEDMIN'] = (dmin, 'Edge-bias min distance from fibre (px)')
    header['EDGESRC']  = (edge_src, 'Edge-bias mask source')
    logger.info(f"Edge-bias {bench}{side} {color}: subtracted {edge_dc:.2f} DN "
                f"(src={edge_src}, n={chosen_n})")
    return corrected, stats


def _write_edge_qa_rows(qa_csv, science_file, primary_hdr, results):
    """Append one edge-bias QA row per extension to ``qa_csv`` (driver-side).

    Writing here (after ray.get) rather than inside the Ray workers avoids
    concurrent-append races on the shared CSV. Logs both the stripes-only and
    stripes+flat candidate levels so the two can be compared during refinement.
    """
    import csv
    def _fmt(v):
        return '' if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.3f}"
    ts = primary_hdr.get('DATE-OBS', primary_hdr.get('DATE', ''))
    wth = primary_hdr.get('WTH-TEMP', '')
    os.makedirs(os.path.dirname(qa_csv) or '.', exist_ok=True)
    is_new = not os.path.exists(qa_csv)
    with open(qa_csv, 'a', newline='') as fh:
        w = csv.writer(fh)
        if is_new:
            w.writerow(['timestamp', 'science_file', 'camera', 'bench', 'side', 'color',
                        'edge_dc', 'edge_npix', 'level_stripes', 'level_stripes_plus_flat',
                        'edge_src', 'wth_temp'])
        for r in results:
            if not r:
                continue
            st = r.get('edge_stats')
            if not st:
                continue
            cam = f"{st['color']}{st['bench']}{st['side']}"
            w.writerow([ts, os.path.basename(str(science_file)), cam,
                        st['bench'], st['side'], st['color'],
                        _fmt(st['edge_dc']), st['edge_npix'],
                        _fmt(st['level_stripes']), _fmt(st['level_combined']),
                        st['edge_src'], wth])


def select_bias_for_extension(hdu_data, header, tracer, use_bias, bench, side, color):
    """Resolve the master-bias HDU for one extension, with verification + fallback.

    Selection logic (shared by process_trace and Bias.biasFirst so the two can
    never drift apart):
    1. Try to load the master bias extension from ``use_bias`` (mode-checked
       against the frame's READ-MDE).
    2. Verify bench/side/color match; discard on mismatch.
    3. If no valid master bias remains, build an inter-fibre/test-region
       fallback bias from the frame itself.

    Returns the chosen bias HDU (never None).
    """
    _BIAS_INTERFIBRE_LOG_THRESHOLD = 10.0  # DN — divergence above which a diagnostic warning is logged

    frame_mode = header.get('READ-MDE', None)
    bias = None

    if use_bias is not None:
        bias_file = os.fspath(use_bias)
        print(f'Bias file: {bias_file}')
        try:
            bias = _grab_bias_hdu(bench=bench, side=side, color=color,
                                  dir=bias_file, required_readmode=frame_mode)
            logger.info(f"Loaded master bias for {bench}{side} {color}")
        except (BiasNotFoundError, BiasReadModeError) as e:
            logger.warning(f"Master bias failed for {bench}{side} {color}: {e}")
            bias = None
    else:
        logger.info(f"No bias file path received for {bench}{side} {color}; using inter-fibre fallback")

    if bias is not None:
        # --- Bench/side/color verification before subtraction ---
        bias_bench = str(bias.header.get('BENCH', '')).strip()
        bias_side  = str(bias.header.get('SIDE',  '')).strip().upper()
        bias_color = str(bias.header.get('COLOR', '')).strip().lower()
        # CAM_NAME fallback if individual keywords are absent
        if not (bias_bench and bias_side and bias_color) and 'CAM_NAME' in bias.header:
            cam = bias.header['CAM_NAME']
            parts = cam.split('_')
            bias_color = parts[1].lower() if len(parts) >= 2 else ''
            bias_bench = parts[0][0] if parts else ''
            bias_side  = parts[0][1].upper() if parts and len(parts[0]) >= 2 else ''

        if (bias_bench and bias_side and bias_color and
                (bias_bench != str(bench) or bias_side != str(side).upper() or
                 bias_color != str(color).lower())):
            logger.error(
                f"Bias BSC mismatch: expected {bench}{side} {color}, "
                f"got {bias_bench}{bias_side} {bias_color} — using inter-fibre fallback"
            )
            bias = None

    if bias is not None:
        # --- Inter-fibre diagnostic (informational only) ---
        # Log the divergence between the raw frame's inter-fibre median and the
        # master bias's inter-fibre median.  For science frames, this divergence
        # is dominated by sky background in the gaps and can legitimately be
        # hundreds of DN — it does NOT indicate a bad bias file.  The master
        # bias is never discarded here; hard failures (missing file, read-mode
        # mismatch, wrong detector) are handled earlier.
        try:
            gap_mask = build_interfibre_mask(tracer, hdu_data.shape, image_type='science')
            n_gap = int(gap_mask.sum())
            if n_gap >= 100:
                frame_if_level = float(np.nanmedian(hdu_data.astype(float)[gap_mask]))
                bias_if_level  = float(np.nanmedian(bias.data[gap_mask]))
                divergence = abs(frame_if_level - bias_if_level)
                if divergence > _BIAS_INTERFIBRE_LOG_THRESHOLD:
                    logger.warning(
                        f"Master bias inter-fibre divergence for {bench}{side} {color}: "
                        f"|frame_if={frame_if_level:.2f} - bias_if={bias_if_level:.2f}| = "
                        f"{divergence:.2f} DN "
                        f"(expected for sky-illuminated frames; master bias retained)"
                    )
                else:
                    logger.info(
                        f"Master bias inter-fibre check OK for {bench}{side} {color}: "
                        f"divergence={divergence:.2f} DN"
                    )
            else:
                logger.warning(
                    f"Inter-fibre diagnostic skipped for {bench}{side} {color}: "
                    f"only {n_gap} gap pixels (< 100)"
                )
        except Exception as exc:
            logger.warning(
                f"Inter-fibre bias diagnostic failed for {bench}{side} {color} "
                f"({exc}); continuing with master bias"
            )

    if bias is None:
        logger.warning(
            f"No valid master bias for {bench}{side} {color} — "
            f"using inter-fibre/test-region fallback."
        )
        bias = generate_fallback_bias_hdu(hdu_data, tracer=tracer)
        logger.info(f"Fallback bias source: {bias.header['BIASSRC']}")

    return bias


@ray.remote
def process_trace(hdu_data, header, trace_file, hdu_index, method='optimal', use_bias=None,
                  remove_cosmic_rays=True, edge_bias=None):
    """
    Process a single HDU: subtract bias, load the trace from a trace file, and create an ExtractLlamas object.

    The tracer is loaded first so its fiberimg can be used for an inter-fibre
    bias quality check and, when needed, an inter-fibre fallback bias estimate.

    Bias selection logic:
    1. Try to load master bias extension from use_bias file.
    2. If loaded, verify bench/side/color match.
    3. If verified, run inter-fibre quality check: if the master bias inter-fibre
       median deviates from the raw frame's inter-fibre median by more than
       _BIAS_INTERFIBRE_THRESHOLD DN, discard the master bias.
    4. If no valid master bias remains, call generate_fallback_bias_hdu(hdu_data, tracer)
       which cross-validates inter-fibre vs test-region (rows 30-50) estimates and
       selects the better one.

    Returns a dict containing:
        'extraction': ExtractLlamas object (or None if error)
        'detector_background': median background after bias subtraction
        'hdu_index': the HDU index for this extraction
    """
    _BIAS_INTERFIBRE_LOG_THRESHOLD = 10.0  # DN — divergence above which a diagnostic warning is logged

    try:
        # Parse camera identifiers
        if 'COLOR' in header:
            color = header['COLOR'].lower()
            bench = header['BENCH']
            side = header['SIDE']
        else:
            camname = header['CAM_NAME']
            color = camname.split('_')[1].lower()
            bench = camname.split('_')[0][0]
            side = camname.split('_')[0][1]

        # Load the trace object first — needed for inter-fibre bias quality check
        with open(trace_file, mode='rb') as f:
            tracer = cloudpickle.load(f)

        # Guard against double-subtraction (bias-first preprocessing stamps BIASSUB)
        if header.get('BIASSUB', False):
            logger.info(
                f"BIASSUB already set for {bench}{side} {color} — skipping bias subtraction"
            )
            bias_subtracted_data = hdu_data.astype(float)
        else:
            # use_bias is the resolved bias file path from GUI_extract (may be None)
            bias = select_bias_for_extension(hdu_data, header, tracer, use_bias,
                                             bench, side, color)
            bias_data = bias.data

            # Compute bias-subtracted data
            bias_subtracted_data = hdu_data.astype(float) - bias_data

            # Record bias subtraction in the header
            header['BIASSUB'] = (True, 'True = bias was subtracted')
            header['BIASSRC'] = (bias.header.get('BIASSRC', 'master_bias'), 'Source of bias subtracted')
            header['BIASLVL'] = (float(np.nanmedian(bias_data)), 'Median bias level subtracted (DN)')

        # --- Per-frame edge-bias (DC offset) correction, on top of master bias ---
        # Measures the residual DC level from unilluminated pixels of THIS frame and
        # subtracts it, tracking the nightly temperature drift the static master bias
        # cannot. No-op (records EDGESRC) when disabled or on placeholder extensions.
        bias_subtracted_data, edge_stats = _apply_edge_bias(
            bias_subtracted_data, tracer, header, bench, side, color, edge_bias)

        # Cosmic ray removal (after bias subtraction, before extraction)
        cr_mask = None
        if remove_cosmic_rays:
            bias_subtracted_data, cr_mask = clean_cosmic_rays(
                bias_subtracted_data,
                color=color, bench=bench, side=side
            )
            header['CRCLEAN'] = (True, 'Cosmic rays cleaned with L.A.Cosmic')
            header['CRNPIX'] = (int(cr_mask.sum()), 'Number of CR pixels cleaned')

        # Compute detector background AFTER bias subtraction
        detector_background = compute_detector_background(bias_subtracted_data, rows=(30, 50))
        print(f"{bench}{side} {color}: Detector bg after bias = {detector_background:.2f}")

        # Create an ExtractLlamas object with bias-subtracted data
        extraction = ExtractLlamas(tracer, bias_subtracted_data, header, method=method)

        return {
            'extraction': extraction,
            'detector_background': detector_background,
            'hdu_index': hdu_index,
            'cr_mask': cr_mask,
            'edge_stats': edge_stats
        }
    except Exception as e:
        logger.error(f"Error extracting trace from {trace_file}")
        logger.error(traceback.format_exc())
        return {'extraction': None, 'detector_background': None, 'hdu_index': hdu_index,
                'cr_mask': None, 'edge_stats': None}


def make_writable(extraction_obj):
    """Convert a Ray-returned extraction object to a writable version."""
    import copy
    
    # First approach: Fix class references for TraceRay objects
    try:
        if hasattr(extraction_obj, 'tracer') and hasattr(extraction_obj.tracer, '__class__'):
            tracer_class = extraction_obj.tracer.__class__
            if tracer_class.__name__ == 'TraceRay' and tracer_class.__module__ == 'traceLlamasMaster':
                # Replace the tracer with a corrected class reference
                correct_class = sys.modules['traceLlamasMaster'].TraceRay
                if tracer_class is not correct_class:
                    # Create a new object with the correct class
                    extraction_obj.tracer.__class__ = correct_class
    except Exception as e:
        logger.warning(f"Could not fix tracer class reference: {e}")
    
    # Second approach: Deep copy
    try:
        return copy.deepcopy(extraction_obj)
    except:
        pass
    
    # Third approach: Cloudpickle and unpickle for better compatibility
    try:
        # This forces a complete serialization and deserialization
        pickled = cloudpickle.dumps(extraction_obj)
        return cloudpickle.loads(pickled)
    except:
        pass
    
    # Fourth approach: If the object has a to_dict method, use it
    if hasattr(extraction_obj, 'to_dict') and callable(extraction_obj.to_dict):
        try:
            obj_dict = extraction_obj.to_dict()
            # Assuming there's a from_dict or similar constructor
            if hasattr(type(extraction_obj), 'from_dict') and callable(type(extraction_obj).from_dict):
                return type(extraction_obj).from_dict(obj_dict)
        except:
            pass
    
    # If all else fails, return the original object and log a warning
    logger.warning(f"Could not make object of type {type(extraction_obj)} writable")
    return extraction_obj

def is_placeholder_camera(data):
    """
    Check if HDU data represents a non-functional camera (filled with 1.0).
    
    Args:
        data: numpy array of HDU data
        
    Returns:
        bool: True if this is a placeholder camera
    """
    if data is None:
        return True
    
    # Check if all values are 1.0 (or very close due to floating point)
    return np.allclose(data, 1.0, rtol=1e-5)

def compute_detector_background(data, rows=(30, 50)):
    """
    Compute median background from specified detector rows.
    
    Args:
        data: numpy array of detector data
        rows: tuple of (start_row, end_row) for background region
        
    Returns:
        float: median background value
    """
    upper_det = data[rows[0]:rows[1], :]
    upper_background_value = np.median(upper_det)
    return upper_background_value

##Main function currently used by the Quicklook for full extraction


def GUI_extract(file: fits.BinTableHDU, flatfiles: str = None, output_dir: str = None,
                slow_bias: str = None, fast_bias: str = None,
                method=None, trace_dir=None, mastercalib_trace_dir=None,
                remove_cosmic_rays=True, mask_output_dir=None, force_refresh=False,
                edge_bias=None) -> None:

    """
    Extracts data from a FITS file using calibration files and saves the extracted data.

    Supports hybrid trace selection: uses user traces for real camera data and
    mastercalib traces for placeholder extensions (missing cameras).

    The correct bias file is selected per-file based on the READ-MDE primary header keyword.
    If slow_bias and/or fast_bias are provided they take priority over the package defaults.
    If neither is provided the function falls back to package-default paths in BIAS_DIR.
    If no bias files exist at all, process_trace() applies a test-region (rows 30-50) fallback.

    Parameters:
    file (str): Path to the FITS file to be processed. Must have a .fits extension.
    flatfiles (str, optional): Path to the flat files for generating new traces. Defaults to None.
    output_dir (str, optional): Output directory for extraction files. Defaults to None.
    slow_bias (str, optional): Path to the SLOW-mode master bias FITS file. Defaults to None.
    fast_bias (str, optional): Path to the FAST-mode master bias FITS file. Defaults to None.
    method (str, optional): Extraction method ('optimal' or 'boxcar'). None (default)
                            resolves from the LLAMAS_EXTRACT_METHOD environment
                            variable (set once by reduce.py from the config's
                            extraction_method key), falling back to 'optimal'.
                            The env mechanism keeps EVERY extraction in a run
                            (science, arc, flat, twilight, sky) on the same
                            method — mixing methods breaks the per-fibre
                            throughput calibration.
    trace_dir (str, optional): User trace directory for real extensions. Defaults to None.
    mastercalib_trace_dir (str, optional): Mastercalib trace directory for placeholder extensions.
                                           Defaults to CALIB_DIR if None.

    Returns:
    Tuple[str, int]: (extraction_file_path, number_of_placeholder_extensions)
    """
    start_time = time.perf_counter()  # Start timer

    if method is None:
        method = os.environ.get('LLAMAS_EXTRACT_METHOD', 'optimal').strip().lower()
    if method not in ('optimal', 'boxcar', 'horne'):
        logger.warning(f"Unknown extraction method '{method}'; using 'optimal'")
        method = 'optimal'
    print(f'Extraction method: {method}')
    
    try:
        print(f'file is {file}')
        assert file.endswith('.fits'), 'File must be a .fits file'
        
        master_pkls = glob.glob(os.path.join(CALIB_DIR, '*.pkl'))
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        if not master_pkls:
            raise ValueError("No master calibration files found in CALIB_DIR")
        
        # Per-task memory reservation for extraction (0 ⇒ schedule purely on CPUs).
        task_mem_mb = int(os.environ.get('LLAMAS_RAY_TASK_MEMORY_MB', 0))

        # Attach to (or start) the one consolidated Ray session — object store clamped
        # to 30% RAM, temp/spill redirected into the owned scratch dir, package uploaded
        # once per run. In the pipeline this attaches to the session reduce.py started;
        # the GUI passes force_refresh=True so each click re-bundles source (per-click reset).
        init_ray(force_refresh=force_refresh)

        # Import placeholder detection utilities
        from llamas_pyjamas.DataModel.validate import get_placeholder_extension_indices, validate_for_gui

        # Validate and create GUI version if needed (preserves original file)
        file = validate_for_gui(file)

        #opening the fitsfile
        hdu, _ = process_fits_by_color(file)

        primary_hdr = hdu[0].header

        # Determine bias file based on READ-MDE header keyword
        read_mode = primary_hdr.get('READ-MDE', None)
        if read_mode is not None:
            read_mode = read_mode.strip().upper()
            logger.info(f"Detected READ-MDE: {read_mode}")

        extraction_file = os.path.splitext(os.path.basename(file))[0] + '_extract.pkl'

        #Defining the base filename
        basefile = os.path.basename(file).split('.fits')[0]
        masterfile = 'LLAMAS_master'

        if slow_bias is not None or fast_bias is not None:
            # Caller supplied explicit paths — use READ-MDE to pick the right one per file
            if read_mode == 'FAST' and fast_bias is not None:
                masterbiasfile = fast_bias
                logger.info(f"READ-MDE=FAST: using caller-supplied fast bias: {masterbiasfile}")
            elif slow_bias is not None:
                masterbiasfile = slow_bias
                logger.info(f"READ-MDE={read_mode}: using caller-supplied slow bias: {masterbiasfile}")
            else:
                # FAST data but no fast_bias supplied by caller
                logger.warning(
                    f"READ-MDE=FAST but no fast_bias supplied; "
                    f"process_trace() will use test-region fallback if no bias can be loaded"
                )
                masterbiasfile = None  # process_trace() handles the fallback
        else:
            # No caller paths supplied — check package-default locations in BIAS_DIR
            slow_default = os.path.join(BIAS_DIR, 'slow_master_bias.fits')
            fast_default = os.path.join(BIAS_DIR, 'fast_master_bias.fits')
            if read_mode == 'FAST' and os.path.isfile(fast_default):
                masterbiasfile = fast_default
                logger.info(f"READ-MDE=FAST: auto-selected fast bias from BIAS_DIR: {masterbiasfile}")
            elif os.path.isfile(slow_default):
                masterbiasfile = slow_default
                logger.info(f"READ-MDE={read_mode}: auto-selected slow bias from BIAS_DIR: {masterbiasfile}")
            else:
                # Neither package default exists — process_trace() will use test-region fallback
                masterbiasfile = None
                logger.warning(
                    "No master bias files found in BIAS_DIR; "
                    "process_trace() will use test-region fallback bias per extension"
                )

        # Validate bias file structure (add placeholders for missing cameras)
        if masterbiasfile is not None:
            masterbiasfile = validate_for_gui(masterbiasfile)

        #Debug statements
        print(f'basefile = {basefile}')
        print(f'masterfile = {masterfile}')
        print(f'Bias file is {masterbiasfile}')

        # Default mastercalib location
        if mastercalib_trace_dir is None:
            mastercalib_trace_dir = CALIB_DIR

        # Set default user trace directory
        if not trace_dir:
            trace_dir = CALIB_DIR
            print(f'No trace_dir specified, using CALIB_DIR: {CALIB_DIR}')
        else:
            print(f'Using specified trace_dir: {trace_dir}')

        # Identify placeholder extensions
        placeholder_indices = get_placeholder_extension_indices(file)

        if placeholder_indices:
            print(f"\n{'='*60}")
            print(f"HYBRID TRACE EXTRACTION")
            print(f"{'='*60}")
            print(f"Detected {len(placeholder_indices)} placeholder extensions (missing cameras)")
            print(f"  Real extensions: Will use traces from {trace_dir}")
            print(f"  Placeholder extensions: Will use mastercalib traces from {mastercalib_trace_dir}")
            print(f"{'='*60}\n")

        #Running the extract routine with hybrid trace selection
        extraction_list = []

        # Build HDU-trace pairs with per-extension trace directory selection
        hdu_trace_pairs = []
        for idx in range(1, len(hdu)):
            header = hdu[idx].header

            # Get camera configuration
            if 'COLOR' in header:
                channel = header['COLOR'].lower()
                bench = header['BENCH']
                side = header['SIDE']
            else:
                camname = header['CAM_NAME']
                channel = camname.split('_')[1].lower()
                bench = camname.split('_')[0][0]
                side = camname.split('_')[0][1]

            # Select trace directory based on placeholder status
            if idx in placeholder_indices:
                active_trace_dir = mastercalib_trace_dir
                trace_source = "mastercalib"
            else:
                active_trace_dir = trace_dir
                trace_source = "user"

            # Find trace file
            try:
                trace_file = get_trace_file(channel, bench, side, active_trace_dir)
                hdu_trace_pairs.append((idx, trace_file))
                print(f"  Extension {idx:2d} ({channel:5s}{bench}{side}): Using {trace_source:12s} trace")
            except FileNotFoundError as e:
                logger.error(f"  Extension {idx:2d} ({channel:5s}{bench}{side}): Trace file not found - {e}")
                # Don't add to pairs - will skip this extension
        #print(hdu_trace_pairs)

        ### Process each HDU-trace pair in parallel using Ray

        # Optional per-task memory reservation (opt-in via `ray_task_memory_mb`).
        # Default 0 ⇒ schedule on CPU only, so concurrency = num_cpus.
        extract_fn = (process_trace.options(memory=task_mem_mb * 1024 * 1024)
                      if task_mem_mb > 0 else process_trace)

        futures = []
        for hdu_index, trace_file in hdu_trace_pairs:
            hdu_data = hdu[hdu_index].data.copy()
            hdr = hdu[hdu_index].header

            future = extract_fn.remote(hdu_data, hdr, trace_file, hdu_index, method=method, use_bias=masterbiasfile, remove_cosmic_rays=remove_cosmic_rays, edge_bias=edge_bias)
            futures.append(future)

        # Wait for all remote tasks to complete
        results = ray.get(futures)

        # Edge-bias QA CSV (driver-side, race-free). Logged whenever a qa_csv path
        # is supplied, regardless of whether Tier 2 is on, so both candidate levels
        # can be compared over the night.
        try:
            _eb = edge_bias or {}
            if _eb.get('qa_csv') and _eb.get('enabled', True):
                _write_edge_qa_rows(_eb['qa_csv'], file, primary_hdr, results)
        except Exception as _qa_exc:
            logger.warning(f"Edge-bias QA logging failed: {_qa_exc}")

        # First, make all extraction objects writable (Ray returns read-only objects)
        writable_results = []
        for result in results:
            if result is not None and result.get('extraction') is not None:
                writable_ex = make_writable(result['extraction'])
                writable_results.append({
                    'extraction': writable_ex,
                    'detector_background': result['detector_background'],
                    'hdu_index': result['hdu_index'],
                    'cr_mask': result.get('cr_mask')
                })

        # Set placeholder cameras to zero (no real data)
        for result in writable_results:
            hdu_idx = result['hdu_index']
            hdu_data = hdu[hdu_idx].data
            if is_placeholder_camera(hdu_data):
                result['extraction'].counts[:] = 0.0
                logger.info(f"Extension {hdu_idx}: Placeholder camera set to 0 (no real data)")

        # Save cosmic ray masks if CR removal was enabled
        if remove_cosmic_rays:
            cr_masks = {}
            for result in writable_results:
                if result.get('cr_mask') is not None:
                    cr_masks[result['hdu_index']] = result['cr_mask']
            if cr_masks:
                mask_dir = mask_output_dir if mask_output_dir else (output_dir or OUTPUT_DIR)
                save_cosmic_ray_masks(cr_masks, primary_hdr, file, mask_dir)

        # Build extraction list from writable results
        extraction_list = [result['extraction'] for result in writable_results]


        print(f'Extraction list = {extraction_list}')
        if output_dir:
            if os.path.exists(output_dir):        
                filename = save_extractions(extraction_list, primary_header=primary_hdr, savefile=extraction_file, save_dir=output_dir)
        else:
            filename = save_extractions(extraction_list, primary_header=primary_hdr, savefile=extraction_file, save_dir=OUTPUT_DIR)
        #print(f'extraction saved filename = {filename}')
        print(f'extraction saved filename = {extraction_file}')
        # if output_dir:
        #     if os.path.exists(output_dir):
                
        
        # else:
        if output_dir:
            obj, metadata = load_extractions(os.path.join(output_dir, extraction_file))
        else:
            obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, extraction_file))
        print(f'obj = {obj}')
        outfile = basefile+'_whitelight.fits'
        
        if output_dir and os.path.exists(output_dir):
            outfile = os.path.join(output_dir, outfile)
        else:
            outfile = os.path.join(OUTPUT_DIR, outfile)
                
        white_light_file = WhiteLightFits(obj, metadata, outfile=outfile)
        print(f'white_light_file = {white_light_file}')

        # Summary statistics
        if placeholder_indices:
            real_count = len(hdu_trace_pairs) - len(placeholder_indices)
            print(f"\n{'='*60}")
            print(f"EXTRACTION SUMMARY")
            print(f"{'='*60}")
            print(f"Total extracted: {len(extraction_list)} spectra")
            print(f"  Real camera data: {real_count} spectra (user traces)")
            print(f"  Placeholder data: {len(placeholder_indices)} spectra (mastercalib traces)")
            print(f"{'='*60}\n")

    except Exception as e:
        traceback.print_exc()
        return

    end_time = time.perf_counter()  # End timer
    elapsed = end_time - start_time
    # Log or print out the elapsed time
    print(f"Full GUI extraction process completed in {elapsed:.2f} seconds.")

    return extraction_file, white_light_file

def make_ifuimage(extraction_file, flat=False):
    obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, extraction_file))
    outfile = 'test_whitelight.fits'
    white_light_file = WhiteLightFits(obj, outfile=outfile)
    print(f'white_light_file = {white_light_file}')


def box_extract(file, flat=False, remove_cosmic_rays=True, mask_output_dir=None):
    try:
    
        assert file.endswith('.fits'), 'File must be a .fits file'
        
        master_pkls = glob.glob(os.path.join(CALIB_DIR, '*.pkl'))
        
        if not master_pkls:
            raise ValueError("No master calibration files found in CALIB_DIR")
        
        # Attach to (or start) the one consolidated Ray session (see Utils/rayManager.py).
        init_ray()

        #opening the fitsfile
        hdu, _ = process_fits_by_color(file)

        #Defining the base filename
        basefile = os.path.basename(file).split('.fits')[0]

        #Debug statements
        print(f'basefile = {basefile}')

        if flat:
            print(f'Running trace routine with flat fielding')
            ##Running the trace routine
            main(file)
            trace_files = glob.glob(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))
        else:
            print('Using master traces')
            trace_files = glob.glob(os.path.join(CALIB_DIR, f'*traces.pkl'))

        #print(os.path.join(CALIB_DIR, f'{basefile}*traces.pkl'))

        #Running the extract routine
        #This code should isolate to only the traces for the given fitsfile

        print(f'trace_files = {trace_files}')
        extraction_list = []
        original_file = file  # preserve before loop variable shadows it
        
        hdu_trace_pairs = match_hdu_to_traces(hdu, trace_files)
        #print(hdu_trace_pairs)

        cr_masks = {}
        #for file in trace_files:
        for hdu_index, file in hdu_trace_pairs:
            hdr = hdu[hdu_index].header

            data = hdu[hdu_index].data.astype(float)

            bias = np.nanmedian(data)
            bias_subtracted = data - bias

            if remove_cosmic_rays:
                color = hdr.get('COLOR', '').lower()
                bench = hdr.get('BENCH', '')
                side = hdr.get('SIDE', '')
                bias_subtracted, cr_mask = clean_cosmic_rays(
                    bias_subtracted, color=color, bench=bench, side=side
                )
                cr_masks[hdu_index] = cr_mask
                hdr['CRCLEAN'] = (True, 'Cosmic rays cleaned with L.A.Cosmic')
                hdr['CRNPIX'] = (int(cr_mask.sum()), 'Number of CR pixels cleaned')

            #print(f'hdu_index {hdu_index}, file {file}, {hdr['CAM_NAME']}')

            try:
                with open(file, mode='rb') as f:
                    tracer = pickle.load(f)

                extraction = ExtractLlamas(tracer, bias_subtracted, hdu[hdu_index].header, optimal=False)
                extraction_list.append(extraction)
                
            except Exception as e:
                print(f"Error extracting trace from {file}")
                print(traceback.format_exc())
        
        # Save cosmic ray masks
        if remove_cosmic_rays and cr_masks:
            primary_hdr = hdu[0].header
            mask_dir = mask_output_dir if mask_output_dir else OUTPUT_DIR
            save_cosmic_ray_masks(cr_masks, primary_hdr, original_file, mask_dir)

        print(f'Extraction list = {extraction_list}')
        filename = save_extractions(extraction_list)
        print(f'extraction saved filename = {filename}')

        obj, metadata = load_extractions(os.path.join(OUTPUT_DIR, filename))
        print(f'obj = {obj}')
        outfile = basefile + '_whitelight.fits'
        white_light_file = WhiteLightFits(obj, outfile=outfile)
        print(f'white_light_file = {white_light_file}')
    
    except Exception as e:
        traceback.print_exc()
        return
    
    
    return 
