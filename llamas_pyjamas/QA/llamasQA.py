from astropy.io import fits
import numpy as np
import subprocess
from io import BytesIO
import traceback

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