from astropy.io import fits
import numpy as np
import subprocess
from io import BytesIO
import traceback
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def plot_ds9(image_array: fits) -> None:
    
    if isinstance(image_array, np.ndarray):

    
        hdu = fits.PrimaryHDU(image_array)
        hdul = fits.HDUList([hdu])
        
        fits_file = BytesIO()
        hdul.writeto(fits_file)
        fits_file.seek(0)
        
        try:
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