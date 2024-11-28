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

def plot_trace_qa(trace_obj, save_dir=None):
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



def plot_comb_template(fitsfile, channel):
    
    trace = TraceLlamas(fitsfile)
    hdu = fits.open(fitsfile)
    
    #find the hdu extensions for the channel we want
    #channel_hdu_idx = [i for i in range(1, len(hdu)) if hdu[i].header['COLOR'] == channel]
    channel_hdu_idx = [i for i in range(1, len(hdu)) if channel in hdu[i].header['CAM_NAME'].lower()]
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