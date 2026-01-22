"""
Integration module for PypeIt-style flat fielding with LLAMAS detector layout.

This module provides the translation layer between LLAMAS's data structures
(extracted 1D spectra per fiber) and PypeIt's expected format (2D detector images
with fiber IDs and wavelength solutions).
"""

import os
import pickle
import numpy as np
from astropy.io import fits
from llamas_pyjamas.constants import idx_lookup, N_fib
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas


def get_detector_info(extension_index):
    """
    Get detector information from extension index.

    LLAMAS has 24 extensions cycling through Red, Green, Blue for benches 1A-4B:
    Extension 1: Red 1A, 2: Green 1A, 3: Blue 1A
    Extension 4: Red 1B, 5: Green 1B, 6: Blue 1B
    ...
    Extension 22: Red 4B, 23: Green 4B, 24: Blue 4B

    Args:
        extension_index (int): FITS extension index (1-24)

    Returns:
        dict: {'channel': str, 'bench': str, 'side': str, 'n_fibers': int}
    """
    # Map extension to channel
    color_cycle = ['red', 'green', 'blue']
    bench_numbers = ['1', '1', '2', '2', '3', '3', '4', '4']  # 1A, 1B, 2A, 2B, ...
    bench_sides = ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B']

    # Extension 1-24 maps to color 0-23
    ext_idx = extension_index - 1  # Convert to 0-indexed

    channel = color_cycle[ext_idx % 3]
    bench_group = ext_idx // 3  # Which bench group (0-7)
    bench = bench_numbers[bench_group]
    side = bench_sides[bench_group]

    bench_key = f"{bench}{side}"
    n_fibers = N_fib.get(bench_key, 300)

    return {
        'channel': channel,
        'bench': bench,
        'side': side,
        'bench_key': bench_key,
        'n_fibers': n_fibers,
        'extension_index': extension_index
    }


def load_flat_field_images(red_flat, green_flat, blue_flat):
    """
    Load all flat field images and organize by detector.

    Args:
        red_flat (str): Path to red flat FITS file
        green_flat (str): Path to green flat FITS file
        blue_flat (str): Path to blue flat FITS file

    Returns:
        list: List of dicts with flat field data for each detector (24 total)
    """
    detectors = []

    with fits.open(red_flat) as red_hdul, \
         fits.open(green_flat) as green_hdul, \
         fits.open(blue_flat) as blue_hdul:

        # Map extensions to HDUs
        color_hduls = {
            'red': red_hdul,
            'green': green_hdul,
            'blue': blue_hdul
        }

        for ext_idx in range(1, 25):  # Extensions 1-24
            det_info = get_detector_info(ext_idx)

            # Get the corresponding HDU for this detector
            channel = det_info['channel']
            hdul = color_hduls[channel]

            # Find the matching extension in the color file
            # Extensions cycle: red, green, blue, so we need the right one
            # Extension 1 (Red 1A) is in red file ext 1
            # Extension 2 (Green 1A) is in green file ext 1
            # Extension 3 (Blue 1A) is in blue file ext 1
            # Extension 4 (Red 1B) is in red file ext 2

            color_ext_idx = (ext_idx - 1) // 3 + 1  # Which extension in color file

            if color_ext_idx < len(hdul):
                flat_data = hdul[color_ext_idx].data
                flat_header = hdul[color_ext_idx].header

                detector = {
                    'detector_idx': ext_idx - 1,  # 0-indexed for PypeIt
                    'extension_idx': ext_idx,      # 1-indexed for FITS
                    'flat_data': flat_data,
                    'header': flat_header,
                    **det_info
                }
                detectors.append(detector)
            else:
                print(f"Warning: Extension {ext_idx} ({channel} {det_info['bench']}{det_info['side']}) not found")

    return detectors


def get_trace_file_for_detector(trace_dir, channel, bench, side):
    """
    Get the trace file path for a specific detector.

    Args:
        trace_dir (str): Directory containing trace files
        channel (str): Color channel ('red', 'green', 'blue')
        bench (str): Bench number ('1', '2', '3', '4')
        side (str): Side ('A', 'B')

    Returns:
        str: Path to trace file, or None if not found
    """
    import glob

    # Try standard LLAMAS naming: LLAMAS_{channel}_{bench}_{side}_traces.pkl
    trace_pattern = f"LLAMAS_{channel}_{bench}_{side}_traces.pkl"
    trace_path = os.path.join(trace_dir, trace_pattern)

    if os.path.exists(trace_path):
        return trace_path

    # Try mastercalib naming: LLAMAS_master_{channel}_{bench}_{side}_traces.pkl
    master_pattern = f"LLAMAS_master_{channel}_{bench}_{side}_traces.pkl"
    master_path = os.path.join(trace_dir, master_pattern)

    if os.path.exists(master_path):
        return master_path

    # Try glob pattern to match any LLAMAS trace file for this detector
    glob_pattern = f"LLAMAS*{channel}_{bench}_{side}_traces.pkl"
    matching_traces = glob.glob(os.path.join(trace_dir, glob_pattern))

    if matching_traces:
        return matching_traces[0]

    # If nothing found, return None (caller will handle the error)
    return None


def create_fiber_id_array(trace, flat_shape):
    """
    Create a 1D fiber ID array for a detector.

    Since LLAMAS has multiple fibers per detector (not one per row),
    we create a simplified mapping where each fiber gets assigned to
    its central row position.

    Args:
        trace: TraceLlamas object with fiber positions
        flat_shape: (ny, nx) shape of flat field image

    Returns:
        np.ndarray: 1D array of fiber IDs (length = ny)
    """
    ny, nx = flat_shape
    fiber_ids = np.full(ny, -1, dtype=int)  # -1 for gaps between fibers

    # For each fiber, mark its central position
    for fiber_idx in range(trace.nfibers):
        # Get fiber trace (y-position at each x)
        # Use middle column for fiber position
        mid_x = nx // 2

        if hasattr(trace, 'fiber_traces') and trace.fiber_traces is not None:
            fiber_y = int(np.round(trace.fiber_traces[fiber_idx][mid_x]))
        else:
            # Fallback: estimate fiber positions
            fiber_y = int(fiber_idx * ny / trace.nfibers)

        if 0 <= fiber_y < ny:
            fiber_ids[fiber_y] = fiber_idx

    return fiber_ids


def create_wavelength_array_2d(trace, arc_dict, flat_shape):
    """
    Create a 2D wavelength array matching the detector image shape.

    Maps wavelength solutions from extracted 1D spectra to 2D detector coordinates.

    Args:
        trace: TraceLlamas object with fiber positions
        arc_dict: Dictionary with wavelength calibration
        flat_shape: (ny, nx) shape of flat field image

    Returns:
        np.ndarray: 2D wavelength array (ny, nx)
    """
    ny, nx = flat_shape
    wavelengths = np.zeros((ny, nx))

    # Extract wavelength information from arc_dict
    # Structure: arc_dict['extractions'] contains ExtractLlamas objects
    extractions = arc_dict.get('extractions', [])

    if not extractions:
        # Fallback: create a simple wavelength array
        # Typical LLAMAS wavelength range
        wave_min = 3500  # Angstroms
        wave_max = 9500
        wavelength_1d = np.linspace(wave_min, wave_max, nx)
        wavelengths = np.tile(wavelength_1d, (ny, 1))
        return wavelengths

    # Find matching extraction for this detector
    matching_extraction = None
    for extraction in extractions:
        if (extraction.channel == trace.channel and
            extraction.bench == trace.bench and
            extraction.side == trace.side):
            matching_extraction = extraction
            break

    if matching_extraction is None:
        # Fallback
        wave_min = 3500
        wave_max = 9500
        wavelength_1d = np.linspace(wave_min, wave_max, nx)
        wavelengths = np.tile(wavelength_1d, (ny, 1))
        return wavelengths

    # Map wavelength from fibers to 2D
    if hasattr(matching_extraction, 'wave') and matching_extraction.wave is not None:
        for fiber_idx in range(min(trace.nfibers, matching_extraction.wave.shape[0])):
            fiber_wave = matching_extraction.wave[fiber_idx]  # 1D array (nx,)

            # Get fiber trace
            if hasattr(trace, 'fiber_traces') and trace.fiber_traces is not None:
                # Map to detector coordinates
                for x in range(min(nx, len(fiber_wave))):
                    fiber_y = int(np.round(trace.fiber_traces[fiber_idx][x]))
                    if 0 <= fiber_y < ny:
                        wavelengths[fiber_y, x] = fiber_wave[x]

                        # Fill in adjacent rows (fiber has width ~5 pixels)
                        for dy in range(-2, 3):
                            y = fiber_y + dy
                            if 0 <= y < ny:
                                wavelengths[y, x] = fiber_wave[x]

    return wavelengths


def process_flat_with_pypeit(red_flat, green_flat, blue_flat, trace_dir, output_dir,
                             arc_calib_file=None, reference_fiber=150, verbose=False):
    """
    Process flat field using PypeIt method with proper LLAMAS detector mapping.

    Args:
        red_flat (str): Path to red flat FITS file
        green_flat (str): Path to green flat FITS file
        blue_flat (str): Path to blue flat FITS file
        trace_dir (str): Directory containing trace files
        output_dir (str): Output directory for flat field products
        arc_calib_file (str, optional): Path to arc calibration file
        reference_fiber (int, optional): Reference fiber for normalization
        verbose (bool, optional): Enable verbose output

    Returns:
        dict: Results dictionary with output_files list
    """
    from llamas_pyjamas.Flat.flatLlamas_pypeit import MultiDetectorFlatField
    from llamas_pyjamas.config import LUT_DIR

    print("Loading flat field images...")
    detectors = load_flat_field_images(red_flat, green_flat, blue_flat)

    if not detectors:
        raise ValueError("No detector data loaded")

    print(f"Loaded {len(detectors)} detector extensions")

    # Load arc calibration
    if arc_calib_file is None:
        arc_calib_file = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')

    print(f"Loading arc calibration from {arc_calib_file}")
    arc_dict = ExtractLlamas.loadExtraction(arc_calib_file)

    # Initialize PypeIt processor
    print(f"Initializing PypeIt flat field processor (reference fiber: {reference_fiber})")
    ff = MultiDetectorFlatField(n_detectors=len(detectors), reference_fiber=reference_fiber)

    # Process each detector
    output_files = []

    for det in detectors:
        print(f"\nProcessing detector {det['extension_idx']}: "
              f"{det['channel']} {det['bench']}{det['side']}")

        # Get trace file
        trace_file = get_trace_file_for_detector(
            trace_dir, det['channel'], det['bench'], det['side']
        )

        if trace_file is None or not os.path.exists(trace_file):
            print(f"  Warning: Trace file not found for {det['channel']} {det['bench']}{det['side']}")
            print(f"  Expected in: {trace_dir}")
            print(f"  Skipping detector {det['extension_idx']}")
            continue

        # Load trace
        with open(trace_file, 'rb') as f:
            trace = pickle.load(f)

        # Create fiber ID array
        fiber_ids = create_fiber_id_array(trace, det['flat_data'].shape)

        # Create wavelength array
        wavelengths = create_wavelength_array_2d(trace, arc_dict, det['flat_data'].shape)

        # Process with PypeIt method
        try:
            pixel_flat, illum_flat, flat_model = ff.process_detector(
                det['flat_data'],
                fiber_ids,
                wavelengths,
                det['detector_idx'],
                variance=None
            )

            # Save output in LLAMAS format
            output_filename = (f"{det['channel']}{det['bench']}{det['side']}_"
                             f"normalized_flat_pypeit.fits")
            output_path = os.path.join(output_dir, output_filename)

            # Create FITS file
            hdu = fits.PrimaryHDU(pixel_flat)
            hdu.header['CHANNEL'] = det['channel']
            hdu.header['BENCH'] = det['bench']
            hdu.header['SIDE'] = det['side']
            hdu.header['FLATMETH'] = 'pypeit'
            hdu.header['NFIBERS'] = det['n_fibers']
            hdu.header['DETIDX'] = det['detector_idx']
            hdu.header['EXTIDX'] = det['extension_idx']
            hdu.header['COMMENT'] = 'PypeIt-style flat field calibration'

            hdu.writeto(output_path, overwrite=True)
            output_files.append(output_path)

            print(f"  ✓ Saved: {output_filename}")

        except Exception as e:
            print(f"  ✗ Error processing detector {det['extension_idx']}: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            continue

    # Normalize across detectors
    print("\nNormalizing across detectors...")
    ff.normalize_multi_detector(scale_to_reference=True)

    # Save normalized versions
    print(f"\nSaved {len(output_files)} flat field pixel maps")

    results = {
        'output_files': output_files,
        'n_detectors_processed': len(output_files),
        'processing_status': 'completed'
    }

    return results
