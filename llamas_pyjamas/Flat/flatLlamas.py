
from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas
from Image.WhiteLightModule import WhiteLightFits
import numpy as np
from astropy.io import fits

class LlamasFlatFielding():
    """Class for handling flat field processing in LLAMAS observations.

    This class provides functionality for creating normalized flat field images
    and applying flat field corrections to white light images.
    """
    
    def __init__(self)->None:
        """Initialize the LlamasFlatFielding object.

        Returns:
            None
        """
        pass
    
    
    def flatcube(self, extraction_list: list = None, outputname: str = None)-> str:
        """Produce a normalized flat field image using the WhiteLightFits class.

        This method creates a normalized flat field cube by processing extracted 
        fiber data and normalizing each HDU by its maximum value.

        Args:
            extraction_list (list, optional): List of extracted fibers from a flat 
                field image. Defaults to None.
            outputname (str, optional): Name for the output FITS file. If None, 
                defaults to 'normalized_flat.fits'. Defaults to None.

        Returns:
            str: Output filename of the saved normalized flat field.
        """
        
        hdul = WhiteLightFits(extraction_list, outfile=-1)
        # Normalize each image data in the HDU list
        for hdu in hdul:
            if hdu.data is not None:
                max_val = np.nanmax(hdu.data)
                if max_val != 0:
                    hdu.data = hdu.data / max_val

        # Save the normalized HDU list as a new FITS file
        if outputname is not None:
            outputname = 'normalized_flat.fits'
            
        hdul.writeto(outputname, overwrite=True)
        return outputname
        
    
    def flatFieldImage(self, whitelight_fits: str, flatcube_fits: str, outputname: str = None)-> str:
        """Apply flat field correction by dividing white light image by flat field.

        This method performs flat field correction by dividing a white light image 
        by a corresponding flat field image, handling matching of bench sides and 
        colors, and protecting against division by zero.

        Args:
            whitelight_fits (str): Path to the white light FITS image.
            flatcube_fits (str): Path to the flat field FITS image.
            outputname (str, optional): Filename for the normalized output image. 
                If None, defaults to 'normalized_whitelight.fits'. Defaults to None.

        Returns:
            str: Path to the saved flat-fielded image.
        """
        # Open the white light and flat field FITS files
        white_hdul = fits.open(whitelight_fits)
        flat_hdul = fits.open(flatcube_fits)

        new_hdus = []

        # Loop over paired HDUs from both files
        for white_hdu, flat_hdu in zip(white_hdul, flat_hdul):
            bench_white = white_hdu.header.get("BENCHSIDE")
            bench_flat = flat_hdu.header.get("BENCHSIDE")
            colour_white = white_hdu.header.get("COLOUR")
            colour_flat = flat_hdu.header.get("COLOUR")
            
            # Only process if both hdu's have matching benchside and colour keywords
            if bench_white != bench_flat or colour_white != colour_flat:
                continue

            # Ensure both have valid data arrays
            if white_hdu.data is not None and flat_hdu.data is not None:
                # Divide and protect against division by zero (assign NaN where flat data is zero)
                divided = np.divide(
                    white_hdu.data,
                    flat_hdu.data,
                    out=np.full_like(white_hdu.data, np.nan, dtype=np.float64),
                    where=flat_hdu.data != 0
                )

                # Normalize the result: divide by the maximum value if it is nonzero
                max_val = np.nanmax(divided)
                if max_val and max_val != 0:
                    divided /= max_val

                # Create a new Image HDU with the result using the white light header
                new_hdu = fits.ImageHDU(data=divided, header=white_hdu.header.copy())
                new_hdus.append(new_hdu)

        # Package the new HDUs into a new HDUList.
        # Use the first new HDU as PrimaryHDU
        if new_hdus:
            primary = fits.PrimaryHDU(data=new_hdus[0].data, header=new_hdus[0].header)
            hdulist = fits.HDUList([primary] + new_hdus[1:])
        else:
            hdulist = fits.HDUList([fits.PrimaryHDU()])

        if outputname is None:
            outputname = 'normalized_whitelight.fits'

        hdulist.writeto(outputname, overwrite=True)

        white_hdul.close()
        flat_hdul.close()

        return outputname
        
