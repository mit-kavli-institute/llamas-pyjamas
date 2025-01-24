# llamas_pyjamas/utils.py
"""Utils module for llamas_pyjamas package.
This module provides various utility functions for logging, file handling, 
data processing, and visualization related to the llamas_pyjamas project.
Functions:
    setup_logger(name, log_filename=None):
        Setup logger with file and console handlers.
    create_peak_lookup(peaks):
        Create lookup table mapping peak index to y position.
    save_combs_to_fits(comb_dict: dict, outfile: str = 'comb_profiles.fits') -> None:
        Save dictionary of comb profiles to multi-extension FITS file.
    grab_peak_pos_from_LUT(file, channel, benchside, fiber=None):
        Retrieve peak positions from a lookup table (LUT) file.
    create_peak_lookups(peaks, benchside=None):
        Create lookup dict with special handling for 2A missing fiber.
    dump_LUT(channel, hdu, trace_obj):
        Dump lookup table (LUT) data from HDU to a JSON file.
    flip_b_side_positions():
        Flip B-side positions in the LUT.
    flip_positions():
        Flip positions in the LUT based on predefined criteria.
    plot_trace(traceobj):
        Plot trace data from a trace object.
    plot_traces_on_image(traceobj, data, zscale=False):
        Plot traces overlaid on raw data with optional zscale.
"""
import os
import logging
from datetime import datetime
import numpy as np
from astropy.io import fits
from llamas_pyjamas.config import CALIB_DIR
import json
import matplotlib.pyplot as plt
import traceback
from matplotlib import cm
from astropy.visualization import ZScaleInterval
from typing import Union
from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas

def setup_logger(name, log_filename=None)-> logging.Logger:
    """
    Setup logger with file and console handlers
    Args:
        name: Logger name (usually __name__)
        log_filename: Optional custom log filename
    """
    # Create logs directory
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create file handler
    log_file = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set desired console log level
    
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add only file handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_peak_lookup(peaks: list) -> dict:
    """Create lookup table mapping peak index to y position"""
    peak_pos = {idx: y_pos for idx, y_pos in enumerate(peaks)}
    
    return peak_pos

def save_combs_to_fits(comb_dict: dict, outfile: str = 'comb_profiles.fits') -> None:
    """
    Save dictionary of comb profiles to multi-extension FITS file
    
    Args:
        comb_dict (dict): Dictionary of comb profiles {'label': comb_array}
        outfile (str): Output filename
    """
    # Create HDU list with empty primary
    hdul = fits.HDUList([fits.PrimaryHDU()])
    
    # Add each comb as an extension
    for label, comb in comb_dict.items():
        # Create ImageHDU for this comb
        hdu = fits.ImageHDU(data=comb.astype(np.float32))
        # Clean label for FITS extension name (remove spaces, special chars)
        ext_name = label.replace(' ', '_').upper()
        hdu.header['EXTNAME'] = ext_name
        hdu.header['LABEL'] = label
        hdu.header['BUNIT'] = 'Counts'
        hdul.append(hdu)
    
    # Write to file
    outpath = os.path.join(CALIB_DIR, outfile)
    hdul.writeto(outpath, overwrite=True)
    
    return



def grab_peak_pos_from_LUT(file: str, channel: str, benchside: str, fiber=None) -> Union[list, int]:
    """
    Retrieve peak positions from a Look-Up Table (LUT) file.
    Args:
        file (str): Path to the LUT file in JSON format.
        channel (str): The channel to retrieve data for.
        benchside (str): The benchside to retrieve data for.
        fiber (int, optional): Specific fiber to retrieve the peak position for. Defaults to None.
    Returns:
        list or int: A list of peak positions if fiber is not specified, otherwise the peak position for the specified fiber.
    """

    with open(file, 'r') as f:
        lut = json.load(f)
    peak_positions = [int(val) for val in lut[channel][benchside].values()]
    if not fiber:
        return peak_positions
    
    return peak_positions[fiber]


def create_peak_lookups(peaks: np.ndarray, benchside=None)-> dict:
    """
    Create a lookup dictionary for peak positions with special handling for missing fibers in specific benchsides.
    This function processes an array of peak positions and inserts additional peaks at specific indices based on the 
    provided benchside. The function handles two benchsides, '2A' and '2B', each with its own specific logic for 
    inserting new peaks. The resulting peaks are then converted into a lookup dictionary where the keys are the 
    indices (as strings) and the values are the peak positions (as integers).
    Parameters:
    -----------
    peaks : numpy.ndarray
        An array of peak positions.
    benchside : str, optional
        The benchside identifier, either '2A' or '2B'. If None, no special handling is applied.
    Returns:
    --------
    dict
        A dictionary where the keys are the indices of the peaks (as strings) and the values are the peak positions 
        (as integers).
    Benchside '2A' Handling:
    ------------------------
    - Calculates the average spacing between peaks.
    - Inserts a new peak at index 269 and 298 based on the average spacing.
    Benchside '2B' Handling:
    ------------------------
    - Calculates the average spacing between peaks.
    - Inserts a new peak at index 48 based on the average spacing.
    Example:
    --------
    peaks = np.array([100, 200, 300, 400])
    create_peak_lookups(peaks, benchside='2A')
    {'0': 100, '1': 200, '2': 300, '3': 400, '4': 250, '5': 450}
    """
    
    if benchside == '2A':
        # Calculate average peak spacing
        peak_diffs = np.diff(peaks)
        avg_spacing = np.mean(peak_diffs)
        
        # Find index 269 and insert new peak
        peak_list = peaks.tolist()
        idx_269 = 269
        idx_298 = 298
        
        if idx_269 < len(peak_list):
            new_peak_pos = int(peak_list[idx_269] + avg_spacing)
            peak_list.insert(idx_269 + 1, new_peak_pos)
            peaks = np.array(peak_list)
            
        if idx_298 < len(peak_list):
            new_peak_pos = int(peak_list[idx_298] + avg_spacing)
            peak_list.insert(idx_298 + 1, new_peak_pos)
            peaks = np.array(peak_list)
    
    if benchside == '2B':
        # Calculate average peak spacing
        peak_diffs = np.diff(peaks)
        avg_spacing = np.mean(peak_diffs)
        
        # Find index 269 and insert new peak
        peak_list = peaks.tolist()
        idx_48 = 48
        
        if idx_48 < len(peak_list):
            new_peak_pos = int(peak_list[idx_48] + avg_spacing)
            peak_list.insert(idx_48 + 1, new_peak_pos)
            peaks = np.array(peak_list)
        
        
    # Create lookup with regular Python ints
    return {str(idx): int(pos) for idx, pos in enumerate(peaks)}

def dump_LUT(channel: str, hdu, trace_obj: TraceLlamas)-> None:
    """
    Dumps the Look-Up Table (LUT) for a given channel and HDU (Header Data Unit) into a JSON file.
    Parameters:
    channel (str): The channel name to be processed.
    hdu (astropy.io.fits.HDUList): The HDU list containing the FITS file data.
    trace_obj (TraceObject): An object that processes HDU data and extracts LUT information.
    Returns:
    None
    Raises:
    FileNotFoundError: If the 'traceLUT.json' file is not found when attempting to load the master LUT.
    Notes:
    - The function initializes or loads a master LUT from 'traceLUT.json'.
    - It processes each relevant HDU to extract LUT information and updates the master LUT.
    - The updated master LUT is saved back to 'traceLUT.json'.
    """

    # Initialize or load master LUT
    try:
        if 'CAM_NAME' in hdu[1].header:
            channel_hdu_idx = [i for i in range(1, len(hdu)) if channel in hdu[i].header['CAM_NAME'].lower()]
        else:
            channel_hdu_idx = [i for i in range(1, len(hdu)) if channel in hdu[i].header['COLOR'].lower()]
        
        with open('traceLUT.json', 'r') as f:
            master_lut = json.load(f)
    except FileNotFoundError:
        master_lut = {channel: {}}

    # Loop through and add each LUT
    for i in channel_hdu_idx:
        
        if 'CAM_NAME' in hdu[i].header:
            cam_name = hdu[i].header['CAM_NAME']
            bench = cam_name.split('_')[0][0]
            side = cam_name.split('_')[0][1]
            
        else:
            channel = hdu[i].header['COLOR'].lower()
            bench = hdu[i].header['BENCH']
            side  = hdu[i].header['SIDE']
            
        benchside = f"{bench}{side}"
        
        trace_obj.process_hdu_data(hdu[i].data, dict(hdu[i].header), find_LUT=True)
        comb = trace_obj.orig_comb
        peaks = trace_obj.first_peaks
        # Convert numpy array to regular Python types
        peaks_dict = create_peak_lookups(peaks, benchside=benchside)
        
        master_lut["combs"][channel][benchside] = trace_obj.orig_comb.tolist()
        # Add to master LUT
        master_lut["fib_pos"][channel][benchside] = peaks_dict
        print(f"Added {benchside} peaks to LUT")

    # Save updated master LUT 
    with open('traceLUT.json', 'w') as f:
        json.dump(master_lut, f, indent=4)
    
    return




def flip_b_side_positions()-> None:
    """
    Flips the fiber positions for the 'B' side in the lookup table (LUT) for each color.
    This function reads a JSON file containing a lookup table (LUT) of fiber positions,
    processes the positions for each color ('green', 'blue', 'red'), and flips the positions
    for the 'B' side. The flipped positions are then saved to a new JSON file.
    The LUT is expected to have the following structure:
    {
        "fib_pos": {
            "color": {
                "benchside": {
                    "fiber_number": "position"
                }
            }
        }
    }
    The function performs the following steps:
    1. Loads the LUT from 'LUT/traceLUT.json'.
    2. Iterates over each color ('green', 'blue', 'red').
    3. For each color, checks if the 'B' side exists in the LUT.
    4. If the 'B' side exists, flips the fiber positions by reversing the fiber numbers.
    5. Saves the updated LUT to 'LUT/traceLUT_copy.json'.
    Raises:
        FileNotFoundError: If the LUT file does not exist.
        json.JSONDecodeError: If the LUT file is not a valid JSON.
    """

    # Load LUT
    with open('LUT/traceLUT.json', 'r') as f:
        lut = json.load(f)
    # Process each color
    for color in ['green', 'blue', 'red']:
        if color not in lut['fib_pos']:
            continue
        # Process each benchside
        for benchside in lut['fib_pos'][color].keys():
            if 'B' in benchside:
                positions = lut['fib_pos'][color][benchside]
                # Get max fiber number
                max_fiber = max(int(k) for k in positions.keys())
                # Create new flipped mapping
                flipped = {}
                for k, v in positions.items():
                    new_key = str(max_fiber - int(k))
                    flipped[new_key] = v
                # Replace original mapping
                lut['fib_pos'][color][benchside] = flipped
    # Save updated LUT
    with open('LUT/traceLUT_copy.json', 'w') as f:
        json.dump(lut, f, indent=4)
        
def flip_positions()-> None:
    """
    Flips the positions of fibers in the lookup table (LUT) based on the specified configuration.
    This function reads a JSON file containing the LUT, processes each color and benchside,
    and flips the positions of the fibers if specified in the `flipped` dictionary. The updated
    LUT is then saved back to the JSON file.
    The `flipped` dictionary determines which fibers to flip for each color and benchside.
    The keys in the dictionary are in the format "{color}{side}" (e.g., "greenA", "blueB"),
    and the values are booleans indicating whether to flip the positions (True) or not (False).
    Example of `flipped` dictionary:
    
    flipped = {"greenA": False, "greenB": True, "blueA": True, "blueB": False, "redA": False, "redB": True}
    
    The function will flip the positions of the fibers for each specified color and benchside,
    and save the updated LUT back to the JSON file.
    Raises:
        FileNotFoundError: If the LUT JSON file is not found.
        json.JSONDecodeError: If there is an error decoding the JSON file.
    """

    # flipped = {"greenA":True, "greenB":False, "blueA": False, "blueB":True, "redA":True, "redB":False}
    #Attempt to check on 23rd Jan 2025
    flipped = {"greenA":False, "greenB":True, "blueA": True, "blueB":False, "redA":False, "redB":True}
    ##temp one
    #flipped = {"greenA":True, "greenB":False, "blueA": True, "blueB":False, "redA":True, "redB":False}
    print(f'flipping {flipped}')
    # Load LUT
    with open('LUT/traceLUT.json', 'r') as f:
        lut = json.load(f)
    
    # Process each color
    for color in ['green', 'blue', 'red']:
        if color not in lut['fib_pos']:
            continue
            
        # Process each benchside
        for benchside in lut['fib_pos'][color].keys():
            side = 'A' if 'A' in benchside else 'B'
            lookup_key = f"{color}{side}"
            
            # Only flip if specified in flipped dict
            if flipped.get(lookup_key, False):
                positions = lut['fib_pos'][color][benchside]
                
                # Get max fiber number
                max_fiber = max(int(k) for k in positions.keys())
                
                # Create new flipped mapping
                flipped_positions = {}
                for k, v in positions.items():
                    new_key = str(max_fiber - int(k))
                    flipped_positions[new_key] = v
                    
                # Replace original mapping
                lut['fib_pos'][color][benchside] = flipped_positions
                print(f"Flipped {color} {benchside}")
    
    # Save updated LUT
    with open('LUT/traceLUT.json', 'w') as f:
        json.dump(lut, f, indent=4)


def plot_trace(traceobj: TraceLlamas)-> None:
    """
    Plots the trace data from a given trace object.
    Parameters:
    traceobj (object): An object containing trace data. It should have the following attributes:
        - tracearr (numpy.ndarray): A 2D array where each row represents a trace.
        - xtracefit (numpy.ndarray): A 2D array where the first row represents the x-coordinates for the trace.
        - traces (list or numpy.ndarray): A list or array of traces to be plotted.
    The function iterates over each trace in tracearr, plots the y-coordinates against the x-coordinates,
    and attempts to plot an additional trace using the traces attribute. If an error occurs during plotting,
    it prints an error message with the index of the trace that caused the error.
    The plot is displayed using plt.show() at the end.
    """
    
    for i in range(len(traceobj.tracearr[:, 0])):
        ypos = traceobj.tracearr[i, :]
        xpos = traceobj.xtracefit[0, :]
        plt.plot(xpos, ypos, ".")
        # Move line plot inside loop
        try:
            plt.plot(np.arange(2048), traceobj.traces[i])
        except:
            print(f"ERROR {i}")
    plt.show()
    
    

def plot_traces_on_image(traceobj: TraceLlamas, data: np.ndarray, zscale=False)-> None:
    """
    Plot traces overlaid on raw data with optional zscale.

    Parameters:
    -----------
    traceobj : object
        An object containing trace information. It should have the following attributes:
        - tracearr: A 2D numpy array where each row represents a fiber trace.
        - xtracefit: A 2D numpy array containing the x-coordinates for the trace fits.
        - traces: A list or array of trace lines.
        - channel: A string representing the channel information.
        - bench: A string representing the bench information.
        - side: A string representing the side information.
    data : numpy.ndarray
        A 2D array representing the raw data image on which the traces will be overlaid.
    zscale : bool, optional
        If True, apply zscale to the image data for better contrast. Default is False.

    Returns:
    --------
    None
        This function does not return any value. It displays a plot with the traces overlaid on the raw data.

    Notes:
    ------
    - The function uses matplotlib for plotting and astropy.visualization for zscale interval.
    - Each trace is plotted with a unique color from the viridis colormap.
    - A vertical red dashed line is plotted at the midpoint of the image.
    - Trace indices are annotated next to the red vertical line.
    - If an error occurs while plotting a trace line, the error is printed and the plotting stops.
    """
    
    fig, ax = plt.subplots(figsize=(12, 8))

    # Apply zscale if requested
    if zscale:
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(data)
    else:
        vmin, vmax = None, None

    # Plot raw data
    im = ax.imshow(data, origin='lower', aspect='auto', cmap='gray', vmin=vmin, vmax=vmax)
    plt.colorbar(im)

    # Generate color gradient
    colors = cm.viridis(np.linspace(0, 1, len(traceobj.tracearr[:, 0])))

    # Plot each fiber trace
    for i, color in enumerate(colors):
        ypos = traceobj.tracearr[i, :]
        xpos = traceobj.xtracefit[0, :]
        ax.plot(xpos, ypos, ".", color=color, label=f"Trace {i}")

        # Plot trace line
        try:
            ax.plot(np.arange(2048), traceobj.traces[i], color=color, label=f"Trace Line {i}")
        except Exception as e:
            traceback.print_exc()
            print(f"ERROR {i}: {e}")
            break

        # Add trace index number next to the red vertical line
        midpoint = data.shape[1] // 2
        ypos_midpoint = min(midpoint, len(ypos) - 1)
        ax.text(midpoint + 5, ypos[ypos_midpoint], f'{i}', color='red', fontsize=8, verticalalignment='center')

    # Plot vertical red line at the midpoint of NAXIS2
    midpoint = data.shape[1] // 2
    ax.axvline(midpoint, color='red', linestyle='--', label='Midpoint')
    plt.legend()
    ax.set_title(f'{traceobj.channel} {traceobj.bench}{traceobj.side} Traces')
    plt.tight_layout()
    plt.show()
    
