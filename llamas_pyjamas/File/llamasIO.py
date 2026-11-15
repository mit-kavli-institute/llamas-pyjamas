


"""FITS file I/O module for the llamas-pyjamas project.

This module provides classes and functions to handle FITS files for the llamas-pyjamas project.
It includes functionality to read and process data from multiple cameras stored in a FITS file,
with support for coordinate transformations and metadata extraction.

Classes:
    llamasOneCamera: Represents a single camera's data and metadata.
    llamasAllCameras: Represents all cameras' data and metadata from a FITS file.

Functions:
    getBenchSideChannel: Retrieves data for a specific bench, side, and channel from a FITS file.
    process_fits_by_color: Processes FITS data with color-based image transformations.
    update_ra_dec_in_header: Converts telescope coordinates to decimal degrees in FITS headers.

Example:
    Basic usage for reading multi-camera FITS data::
    
        # Read data from all cameras in a FITS file
        cameras = llamasAllCameras('observation.fits')
        
        # Access specific camera data
        for ext in cameras.extensions:
            print(f"Camera {ext.bench}-{ext.side}-{ext.channel}: {ext.data.shape}")
"""
import logging
import os
import re

import numpy as np
from astropy.io import fits # type: ignore

logger = logging.getLogger(__name__)

_DATASEC_RE = re.compile(r'\[(\d+):(\d+),\s*(\d+):(\d+)\]')


def header_color(header):
    """Lower-case channel of an extension, from COLOR or else CAM_NAME; None if neither."""
    if 'COLOR' in header:
        return str(header['COLOR']).lower()
    if 'CAM_NAME' in header:
        return header['CAM_NAME'].split('_')[1].lower()
    return None


def datasec_slices(header, shape):
    """(rows, cols) slices of the DATASEC region, or the whole array.

    A DATASEC that doesn't parse, or that runs past the data, is ignored, exactly as in
    ``process_fits_by_color``.
    """
    match = _DATASEC_RE.match(str(header.get('DATASEC', '')))
    if match:
        x1, x2, y1, y2 = map(int, match.groups())
        if x2 <= shape[1] and y2 <= shape[0]:
            return slice(y1 - 1, y2), slice(x1 - 1, x2)
    return slice(0, shape[0]), slice(0, shape[1])


def orient_by_color(data, color):
    """Flip a detector image between raw and pipeline orientation.

    Blue is flipped in x and y, green in x, red not at all. Each flip is its own inverse,
    so the same call takes a pipeline-orientation image back to raw orientation.
    """
    if color == 'blue':
        return np.flipud(np.fliplr(data))
    if color == 'green':
        return np.fliplr(data)
    return data


def trim_and_orient(data, header):
    """The DATASEC region of a raw extension in pipeline orientation (as process_fits_by_color)."""
    rows, cols = datasec_slices(header, data.shape)
    return orient_by_color(data[rows, cols], header_color(header))


def to_raw(oriented, header, raw):
    """Copy of ``raw`` (float32) with its DATASEC region replaced by ``oriented``, flipped back.

    Rows and columns outside DATASEC (the metadata row) are left as they are.
    """
    out = np.array(raw, dtype=np.float32)
    rows, cols = datasec_slices(header, out.shape)
    out[rows, cols] = orient_by_color(oriented, header_color(header))
    return out
##############################################################

class llamasOneCamera:
    """A class to represent a single camera in the llamas system.
    
    Attributes:
        header (str): The header information of the camera.
        data (int): The data associated with the camera.
        bench (int): The bench number of the camera.
        side (str): The side of the camera.
        channel (str): The color channel of the camera.
    """


    def __init__(self):
        self.header =   ''
        self.data   =   0
        self.bench  =   -1
        self.side   =   ''
        self.channel =  ''
        

    def readhdu(self, hdu: fits.HDUList) -> None:
        """Read header and data from the given HDU and extract metadata.
        
        Reads the header and data from the given HDU (Header Data Unit) and assigns
        them to the instance variables. Additionally, extracts specific header 
        information such as 'BENCH', 'SIDE', and 'COLOR'.
        
        Args:
            hdu: The HDU object from which to read the header and data.
        """

        self.header = hdu.header
        self.data   = hdu.data
        self.bench  = self.header['BENCH']
        self.side   = self.header['SIDE']
        self.channel = self.header['COLOR']
        self.index  = -1

##############################################################

class llamasAllCameras:
    """A class to handle multiple camera data from a FITS file.
    
    Attributes:
        header: The header of the primary HDU in the FITS file.
        Next (int): The number of extensions (HDUs) in the FITS file.
        extensions (list): A list of llamasOneCamera objects, each representing an extension.
    """


    def __init__(self, fitsfile: str) -> None:

        with fits.open(fitsfile) as hdulist:
            self.header     = hdulist[0].header
            self.Next       = len(hdulist) - 1
            self.extensions = []
            for hdu in hdulist[1:]:
                thiscam = llamasOneCamera()
                thiscam.readhdu(hdu)
                self.extensions.append(thiscam)

        hdulist.close()

##############################################################
            
def getBenchSideChannel(fitsfile: str, bench: str, side: str, channel: str)-> None:
    """Extract data from a FITS file for a specific bench, side, and channel.
    
    Args:
        fitsfile: The path to the FITS file.
        bench: The bench identifier to match in the FITS file header.
        side: The side identifier to match in the FITS file header.
        channel: The channel identifier to match in the FITS file header.
        
    Returns:
        The data from the FITS file that matches the specified bench, side, and channel.
    """


    hdul = fits.open(fitsfile)
    for hdu in hdul[1:]:
        if (hdu.header['BENCH'] == bench):
            if (hdu.header['SIDE'] == side):
                if (hdu.header['COLOR']=='Color'):
                    return(hdu.data)
                

def process_fits_by_color(fits_file, output_file=None):
    """Process a FITS file and transform image data based on color attribute.

    The function applies the following transformations:
    - Blue: Flip both x and y axes
    - Green: Flip only x axis (horizontally)
    - Red: No transformation

    Args:
        fits_file: Path to the FITS file.
        output_file: Path to also save the processed FITS file to. If None (the
                     default), nothing is written: the pipeline only uses the returned
                     HDUs, and a ``_trimmed`` copy of every frame read cost 0.4 GB each.

    Returns:
        tuple: (HDUList, str or None) containing the processed HDU list and the written
               path (None when nothing was written), or (None, None) if processing fails.
    """
    try:
        # Open the FITS file
        with fits.open(fits_file) as hdul:
            # Create a new HDU list for the result
            result_hdus = fits.HDUList()

            # Add the primary HDU without changes
            result_hdus.append(hdul[0].copy())

            # Trim to DATASEC and flip each extension (trim_and_orient, shared with
            # bias-first and the dark so the three can never disagree).
            for i in range(1, len(hdul)):
                hdu = hdul[i].copy()  # Create a copy to avoid modifying the original
                if hdu.data is not None:
                    match = _DATASEC_RE.match(str(hdu.header.get('DATASEC', '')))
                    if match and (int(match.group(2)) > hdu.data.shape[1]
                                  or int(match.group(4)) > hdu.data.shape[0]):
                        logger.warning(f"Warning: DATASEC dimensions {hdu.header['DATASEC']} "
                                       f"exceed data dimensions {hdu.data.shape} for HDU {i}")
                    hdu.data = trim_and_orient(hdu.data, hdu.header)
                result_hdus.append(hdu)

        if output_file is not None:
            result_hdus.writeto(output_file, overwrite=True)
            print(f"Processed FITS file saved to: {output_file}")

        return result_hdus, output_file

    except Exception as e:
        print(f"Error processing FITS file: {e}")
        return None, None  
    
    
    
### Function to convert telescope RA and DEC into decimeral degrees for the FITS header
def update_ra_dec_in_header(header, save_to_file=None) -> fits.Header:
    """Extract and convert RA/DEC coordinates from FITS header to decimal degrees.
    
    Extract RA and DEC from HIERARCH TEL RA and HIERARCH TEL DEC in a FITS header,
    convert them to decimal degrees, and update the header's RA and DEC keywords.
    Only updates if RA and DEC are not already set.
    
    Args:
        header: The FITS header to process.
        save_to_file: Path to FITS file to update with the new header.
    
    Returns:
        The updated header.
    """
    from astropy.coordinates import SkyCoord
    
    # Check if RA and DEC are already set
    if 'RA' in header and 'DEC' in header:
        # If RA and DEC values already exist, no change needed
        return header
    
    # Check if the telescope coordinates exist in the header
    if 'HIERARCH TEL RA' in header and 'HIERARCH TEL DEC' in header:
        # Extract sexagesimal coordinates
        ra_sex = header['HIERARCH TEL RA']
        dec_sex = header['HIERARCH TEL DEC']
        
        # Convert to decimal degrees
        coord = SkyCoord(ra_sex, dec_sex, unit=('hourangle', 'deg'))
        ra_decimal = coord.ra.deg
        dec_decimal = coord.dec.deg
        
        # Update the header with decimal values
        header['RA'] = ra_decimal
        header['DEC'] = dec_decimal
        
        # Add a comment to indicate the conversion
        header.comments['RA'] = 'Decimal degrees, converted from HIERARCH TEL RA'
        header.comments['DEC'] = 'Decimal degrees, converted from HIERARCH TEL DEC'
    
    # If a file path is provided, update the primary header in that file
    if save_to_file:
        with fits.open(save_to_file, mode='update') as hdul:
            hdul[0].header = header
            hdul.flush()
    
    return header