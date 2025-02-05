
from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas
from llamas_pyjamas.Image.WhiteLight import WhiteLightFits
import numpy as np
from astropy.io import fits

class LlamasFlatFielding():
    """
    """
    
    def __init__(self)->None:
        pass
    
    
    def flatcube(self, extraction_list: list = None, outputname: str = None)-> str:
        """A function to produce a normalised flat field image, using the WhiteLightFits class.

        Args:
            extraction_list (list): list of extracted fibres from a flat field image

        Returns:
            str: output filename
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
        pass
    
    def flatFieldImage(self, whitelight_fits: str, flatcube_fits: str, outputname: str = None)-> str:
        """A function to divide a white light image by a flat field image.

        Args:
            whitelight_fits (str): path to the white light image
            flatcube_fits (str): path to the flat field image
            outputname (str, optional): normalised output image filename. Defaults to None.

        Returns:
            str: _description_
        """
        pass
