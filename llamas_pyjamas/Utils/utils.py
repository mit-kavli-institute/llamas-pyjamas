# llamas_pyjamas/utils.py
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

def setup_logger(name, log_filename=None):
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

def create_peak_lookup(peaks):
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

def grab_peak_pos_from_LUT(file, channel, benchside, fiber=None):
    with open(file, 'r') as f:
        lut = json.load(f)
    peak_positions = [int(val) for val in lut[channel][benchside].values()]
    if not fiber:
        return peak_positions
    
    return peak_positions[fiber]


def create_peak_lookups(peaks, benchside=None):
    """Create lookup dict with special handling for 2A missing fiber"""
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

def dump_LUT(channel, hdu, trace_obj):
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




def flip_b_side_positions():
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
        
def flip_positions():
    flipped = {"greenA":True, "greenB":False, "blueA": False, "blueB":True, "redA":True, "redB":False}
    ##temp one
    #flipped = {"greenA":True, "greenB":False, "blueA": True, "blueB":False, "redA":True, "redB":False}
    
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


def plot_trace(traceobj):
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
    
    

def plot_traces_on_image(traceobj, data, zscale=False):
    """Plot traces overlaid on raw data with optional zscale"""
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
        ax.text(midpoint + 5, ypos[ypos_midpoint], f'{i}', color='red', fontsize=8, verticalalignment='bottom')

    # Plot vertical red line at the midpoint of NAXIS2
    midpoint = data.shape[1] // 2
    ax.axvline(midpoint, color='red', linestyle='--', label='Midpoint')

    ax.set_title(f'{traceobj.channel} {traceobj.bench}{traceobj.side} Traces')
    plt.tight_layout()
    plt.show()
    
