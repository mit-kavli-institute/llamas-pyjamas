"""
This module provides various functions for plotting and analyzing data related to the llamas_pyjamas pipeline 
using FITS files and matplotlib for visualization. It includes functions to plot images in DS9, 
generate QA plots for fiber tracing, plot combined templates, and compare combs between different 
channels and bench sides.

Functions:
    plot_ds9(image_array: fits, samp=False) -> None:
        Plots a FITS image array in DS9, with optional SAMP integration.
    plot_trace_qa(trace_obj, save_dir=None):
        Plots QA for fiber tracing, including individual fiber profiles.
    plot_comb_template(fitsfile, channel):
        Plots combined templates for a specified channel from a FITS file.
    plot_master_comb(channel):
        Plots master combs for a specified channel using data from a lookup table.
    compare_combs(channel1, benchside1, channel2, benchside2):
        Compares combs between two specified channels and bench sides.
"""
from astropy.io import fits
import numpy as np
import subprocess
from io import BytesIO
import traceback
from astropy.samp import SAMPIntegratedClient
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from llamas_pyjamas.Trace.traceLlamasMulti import TraceLlamas
from llamas_pyjamas.config import LUT_DIR
import json

from matplotlib import cm
from astropy.visualization import ZScaleInterval
from typing import Union

def plot_ds9(image_array: fits, samp=False) -> None:
    
    if isinstance(image_array, np.ndarray):

    
        hdu = fits.PrimaryHDU(image_array)
        hdul = fits.HDUList([hdu])
        
        if (samp == True):
                
            try:
                tmpfile = os.environ['PWD']+"/tmp.fits"
                print(tmpfile)
                hdul.writeto(tmpfile, overwrite=True)
            except:
                print("Error writing file")
            try:
                
                ds9 = SAMPIntegratedClient()
                ds9.connect()
                key = ds9.get_private_key()
                clients = ds9.hub.get_registered_clients(key)
                client_id = clients[-1]
                for c in clients:
                    metadata = ds9.get_metadata(c)
                    if (metadata['samp.name'] == 'ds9'):
                        print(f"Binding client ID {c} to ds9")
                        client_id = c
                ds9.ecall_and_wait(client_id,"ds9.set","10",cmd="frame 1")
                ds9.ecall_and_wait(client_id,"ds9.set","10",cmd=f"fits {tmpfile}")
                ds9.ecall_and_wait(client_id,"ds9.set","10",cmd="zoom to fit")

                os.remove(tmpfile)

            except:
                print("ERROR: Problem connecting to the ds9 SAMP hub")

            ds9.disconnect()

        else:

            try:

                fits_file = BytesIO()
                hdul.writeto(fits_file)
                fits_file.seek(0)

                # Open DS9 using subprocess
                process = subprocess.Popen(
                    ['ds9', '-'],
                    stdin=subprocess.PIPE,  # Pipe data into DS9's stdin
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )

                # Send the FITS data to DS9 via stdin
                process.communicate(input=fits_file.read())
            
            except Exception as e:
                traceback.print_exc()
                return

    return

def plot_trace_qa(trace_obj: TraceLlamas, save_dir=None)-> None:
    """Plot QA for fiber tracing"""

    # Plot 2: Individual fiber profiles
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    axes = axes.ravel()
    
    for i, sset in enumerate(trace_obj.bspline_ssets[:16]):  # Plot first 16 fibers
        ax = axes[i]
        yy = np.linspace(-5, 5, 100)
        profile = sset.value(yy)[0]
        ax.plot(yy, profile, 'b-', label='Fit')
        ax.set_title(f'Fiber {i}')
        ax.grid(True)
    
    plt.tight_layout()
    #plt.savefig(os.path.join(save_dir, f'profiles_{trace_obj.bench}_{trace_obj.side}.png'))
    plt.show()
    

    return



def plot_comb_template(fitsfile: str, channel)-> None:
    
    trace = TraceLlamas(fitsfile)
    hdu = fits.open(fitsfile)
    
    #find the hdu extensions for the channel we want
    #channel_hdu_idx = [i for i in range(1, len(hdu)) if hdu[i].header['COLOR'] == channel]
    try:
        channel_hdu_idx = [i for i in range(1, len(hdu)) if channel in hdu[i].header['CAM_NAME'].lower()]
    except:
        channel_hdu_idx = [i for i in range(1, len(hdu)) if channel in hdu[i].header['COLOR'].lower()]
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    # Process green HDUs
    for i, (ax, hdu_idx) in enumerate(zip(axes, channel_hdu_idx)):
        try:
            # Process HDU data (adding 1 to skip primary)
            trace.process_hdu_data(hdu[hdu_idx].data, dict(hdu[hdu_idx].header))
            comb = trace.orig_comb
            peaks = trace.orig_peaks
            peak_heights = trace.orig_pkht
            
            # Plot data
            ax.plot(comb, 'b-', alpha=0.6, label='Data')
            ax.plot(peaks, peak_heights, 'rx', label='Peaks')
            
            # Add vertical lines from peaks to x-axis
            for idx, (peak, height) in enumerate(zip(peaks, peak_heights)):
                ax.vlines(peak, 0, height, colors='r', linestyles=':', alpha=0.5)
                
                ax.text(peak, height + 100, str(int(idx)), 
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color='red',
                       fontsize=8)
            
            ax.set_title(f'HDU {hdu_idx}: {trace.bench}{trace.side}')
            ax.grid(True, alpha=0.3)
            
            if i == 0:  # Legend only on first plot
                ax.legend()
                
        except Exception as e:
            print(f"Error processing HDU {hdu_idx+1}: {e}")
            continue

    plt.suptitle('Green Channel Trace Profiles', fontsize=14)
    plt.tight_layout()
    plt.show()
        
    return


def plot_master_comb(channel: str)-> None:
    
    pkht_value = 10000
    
    with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'rb') as f:
        master_LUT = json.load(f)
    #find the hdu extensions for the channel we want
    #channel_hdu_idx = [i for i in range(1, len(hdu)) if hdu[i].header['COLOR'] == channel]
    #channel_hdu_idx = [i for i in range(1, len(hdu)) if channel in hdu[i].header['CAM_NAME'].lower()]
    fig, axes = plt.subplots(2, 4, figsize=(15, 9), constrained_layout=True)
    axes = axes.flatten()
    
    benchsides = ['1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B']
    
    try:
        for idx, bench in enumerate(benchsides):
            print(idx)
            comb = np.array(master_LUT['combs'][channel][bench])
            peaks = master_LUT['fib_pos'][channel][bench]
            
            master_peaks = [int(pos) for pos in peaks.values()]
            
            
            peak_heights = comb[master_peaks]#np.full_like(master_peaks, pkht_value)

            # Plot data
            axes[idx].plot(comb, 'b-', alpha=0.6, label='Data')
            axes[idx].plot(master_peaks, peak_heights, 'rx', label='Peaks')

            # Add vertical lines from peaks to x-axis
            for peak_idx, (peak, height) in enumerate(zip(master_peaks, peak_heights)):
                axes[idx].vlines(peak, 0, height, colors='r', linestyles=':', alpha=0.5)

                axes[idx].text(peak, height + 100, str(int(peak_idx)), 
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       color='red',
                       fontsize=8)

            axes[idx].set_title(f'{channel}: {bench}')
            axes[idx].grid(True, alpha=0.3)

            if idx == 0:  # Legend only on first plot
                axes[idx].legend()
            
    except Exception as e:
        traceback.print_exc()
        print(f"Error plotting master combs: {e}")
        return
        

    plt.suptitle('Green Channel Trace Profiles', fontsize=14)
    plt.tight_layout()
    plt.show()
        
    return


def compare_combs(channel1: str, benchside1: str, channel2: str, benchside2: str)-> None:
    with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'rb') as f:
        master_LUT = json.load(f)
    
    # Create vertical subplot layout
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    try:
        # Plot first benchside
        comb1 = np.array(master_LUT['combs'][channel1][benchside1])
        peaks1 = master_LUT['fib_pos'][channel1][benchside1]
        master_peaks1 = [int(pos) for pos in peaks1.values()]
        peak_heights1 = comb1[master_peaks1]
        
        ax1.plot(comb1, 'b-', alpha=0.6, label='Data')
        ax1.plot(master_peaks1, peak_heights1, 'rx', label='Peaks')
        
        for peak_idx, (peak, height) in enumerate(zip(master_peaks1, peak_heights1)):
            ax1.vlines(peak, 0, height, colors='r', linestyles=':', alpha=0.5)
            ax1.text(peak, height + 100, str(peak_idx),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    color='red', fontsize=8)
        
        ax1.set_title(f'{channel1}: {benchside1}')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot second benchside
        comb2 = np.array(master_LUT['combs'][channel2][benchside2])
        peaks2 = master_LUT['fib_pos'][channel2][benchside2]
        master_peaks2 = [int(pos) for pos in peaks2.values()]
        peak_heights2 = comb2[master_peaks2]
        
        ax2.plot(comb2, 'b-', alpha=0.6, label='Data')
        ax2.plot(master_peaks2, peak_heights2, 'rx', label='Peaks')
        
        for peak_idx, (peak, height) in enumerate(zip(master_peaks2, peak_heights2)):
            ax2.vlines(peak, 0, height, colors='r', linestyles=':', alpha=0.5)
            ax2.text(peak, height + 100, str(peak_idx),
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    color='red', fontsize=8)
        
        ax2.set_title(f'{channel2}: {benchside2}')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
    except Exception as e:
        traceback.print_exc()
        print(f"Error comparing combs: {e}")
        return
    
    plt.suptitle(f'Comparing Channel Combs', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return


def plot_trace(traceobj: 'TraceLlamas')-> None:
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
    
    

def plot_traces_on_image(traceobj: 'TraceLlamas', data: np.ndarray, zscale=False)-> None:
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
        #ax.text(midpoint + 5, ypos[ypos_midpoint], f'{i}', color='red', fontsize=8, verticalalignment='center')

    # Plot vertical red line at the midpoint of NAXIS2
    midpoint = data.shape[1] // 2
    ax.axvline(midpoint, color='red', linestyle='--', label='Midpoint')
    #plt.legend()
    ax.set_title(f'{traceobj.channel} {traceobj.bench}{traceobj.side} Traces')
    plt.tight_layout()
    plt.show()


def plot_fiber_masks_on_image(traceobj, data, zscale=False):
    """Plot fiber masks overlaid on raw data with optional zscale"""
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
    colors = cm.viridis(np.linspace(0, 1, len(traceobj.traces)))

    # Plot each fiber mask
    for i, color in enumerate(colors):
        ytrace = traceobj.traces[i]
        yy = np.outer(np.arange(traceobj.naxis2), np.ones(traceobj.naxis1)) - np.outer(np.ones(traceobj.naxis2), ytrace)
        profmask = np.abs(yy) < 2

        # Overlay the fiber mask
        mask_overlay = np.ma.masked_where(~profmask, profmask)
        ax.imshow(mask_overlay, origin='lower', aspect='auto', cmap='cool', alpha=0.5)

        # Add trace index number next to the trace line
        midpoint = data.shape[1] // 2
        ypos_midpoint = ytrace[midpoint] if midpoint < len(ytrace) else ytrace[-1]
        ax.text(midpoint + 5, ypos_midpoint, f'{i}', color='red', fontsize=8, verticalalignment='center')

    # Plot vertical red line at the midpoint of NAXIS2
    midpoint = data.shape[1] // 2
    ax.axvline(midpoint, color='red', linestyle='--', label='Midpoint')
    plt.legend()
    ax.set_title(f'{traceobj.channel} {traceobj.bench}{traceobj.side} Fiber Masks')
    plt.tight_layout()
    plt.show()
    