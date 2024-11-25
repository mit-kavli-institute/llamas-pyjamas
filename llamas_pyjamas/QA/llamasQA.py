from astropy.io import fits
import numpy as np
import subprocess
from io import BytesIO
import traceback
from astropy.samp import SAMPIntegratedClient
import os

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