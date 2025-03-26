


"""
This module provides classes and functions to handle FITS files for the llamas-pyjamas project.
It includes functionality to read and process data from multiple cameras stored in a FITS file.
Classes:
    llamasOneCamera: Represents a single camera's data and metadata.
    llamasAllCameras: Represents all cameras' data and metadata from a FITS file.
Functions:
    getBenchSideChannel(fitsfile, bench, side, channel): Retrieves data for a specific bench, side, and channel from a FITS file.

    Represents a single camera's data and metadata.
    Attributes:
        header (str): The header information from the FITS file.
        data (numpy.ndarray): The data from the FITS file.
        bench (int): The bench number from the header.
        side (str): The side information from the header.
        channel (str): The channel information from the header.

        Initializes a new instance of the llamasOneCamera class.

        self.header = ''
        self.data = 0
        self.bench = -1
        self.side = ''
        self.channel = ''

        Reads data and metadata from a given HDU (Header/Data Unit).
        Args:
            hdu (astropy.io.fits.HDUList): The HDU to read data from.

        self.data = hdu.data
        self.bench = self.header['BENCH']
        self.side = self.header['SIDE']
        self.index = -1
    
    Represents all cameras' data and metadata from a FITS file.
    Attributes:
        header (str): The primary header information from the FITS file.
        Next (int): The number of extensions in the FITS file.
        extensions (list): A list of llamasOneCamera instances for each extension in the FITS file.
    
        
        Initializes a new instance of the llamasAllCameras class and reads data from the given FITS file.
        Args:
            fitsfile (str): The path to the FITS file to read.
        
            self.header = hdulist[0].header
            self.Next = len(hdulist) - 1
    
    Retrieves data for a specific bench, side, and channel from a FITS file.
    Args:
        fitsfile (str): The path to the FITS file to read.
        bench (int): The bench number to filter by.
        side (str): The side information to filter by.
        channel (str): The channel information to filter by.
    Returns:
        numpy.ndarray: The data for the specified bench, side, and channel.
                if (hdu.header['COLOR'] == 'Color'):
                    return hdu.data
"""
from astropy.io import fits # type: ignore
##############################################################

class llamasOneCamera:
    """
    A class to represent a single camera in the llamas system.
    Attributes
    ----------
    header : str
        The header information of the camera.
    data : int
        The data associated with the camera.
    bench : int
        The bench number of the camera.
    side : str
        The side of the camera.
    channel : str
        The color channel of the camera.
    Methods
    -------
    __init__():
        Initializes the llamasOneCamera with default values.
    readhdu(hdu):
        Reads the header and data from the given HDU (Header Data Unit).
    """


    def __init__(self):
        self.header =   ''
        self.data   =   0
        self.bench  =   -1
        self.side   =   ''
        self.channel =  ''

    def readhdu(self, hdu: fits.HDUList) -> None:
        """
        Reads the header and data from the given HDU (Header Data Unit) and assigns
        them to the instance variables. Additionally, extracts specific header 
        information such as 'BENCH', 'SIDE', and 'COLOR'.
        Parameters:
        hdu (astropy.io.fits.HDUList): The HDU object from which to read the header and data.
        Attributes:
        header (astropy.io.fits.Header): The header of the HDU.
        data (numpy.ndarray): The data of the HDU.
        bench (str): The 'BENCH' value from the header.
        side (str): The 'SIDE' value from the header.
        channel (str): The 'COLOR' value from the header.
        index (int): An index initialized to -1.
        """

        self.header = hdu.header
        self.data   = hdu.data
        self.bench  = self.header['BENCH']
        self.side   = self.header['SIDE']
        self.channel = self.header['COLOR']
        self.index  = -1

##############################################################

class llamasAllCameras:
    """
    A class to handle multiple camera data from a FITS file.
    Attributes:
    -----------
    header : astropy.io.fits.header.Header
        The header of the primary HDU in the FITS file.
    Next : int
        The number of extensions (HDUs) in the FITS file.
    extensions : list
        A list of llamasOneCamera objects, each representing an extension in the FITS file.
    Methods:
    --------
    __init__(fitsfile)
        Initializes the llamasAllCameras object by reading the FITS file and storing the header and extensions.
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
    """
    Extracts and returns the data from a FITS file for a specific bench, side, and channel.
    Parameters:
    fitsfile (str): The path to the FITS file.
    bench (str): The bench identifier to match in the FITS file header.
    side (str): The side identifier to match in the FITS file header.
    channel (str): The channel identifier to match in the FITS file header.
    Returns:
    numpy.ndarray: The data from the FITS file that matches the specified bench, side, and channel.
    """


    hdul = fits.open(fitsfile)
    for hdu in hdul[1:]:
        if (hdu.header['BENCH'] == bench):
            if (hdu.header['SIDE'] == side):
                if (hdu.header['COLOR']=='Color'):
                    return(hdu.data)
                

def process_fits_by_color(fits_file):
    """
    Process a FITS file, transform image data based on color attribute, and return a table of HDUs.
    
    The function applies the following transformations:
    - Blue: Flip both x and y axes
    - Green: Flip only x axis (horizontally)
    - Red: No transformation
    
    Parameters:
    ----------
    fits_file : str
        Path to the FITS file
        
    Returns:
    -------
    astropy.io.fits.HDUList
        HDUList with color-based transformations applied
    """
    from astropy.io import fits
    import numpy as np
    
    try:
        # Open the FITS file
        with fits.open(fits_file) as hdul:
            # Create a new HDU list for the result
            result_hdus = fits.HDUList()
            
            # Add the primary HDU without changes
            result_hdus.append(hdul[0].copy())
            
            # Process each extension HDU
            for i in range(1, len(hdul)):
                hdu = hdul[i].copy()  # Create a copy to avoid modifying the original
                
                if 'DATASEC' in hdu.header:
                    datasec = hdu.header['DATASEC']
                    
                    # Parse the DATASEC string '[x1:x2, y1:y2]'
                    import re
                    match = re.match(r'\[(\d+):(\d+),\s*(\d+):(\d+)\]', datasec)
                    if match:
                        x1, x2, y1, y2 = map(int, match.groups())

                        # Convert from FITS 1-based indexing to Python 0-based indexing
                        x1 -= 1
                        y1 -= 1

                        # Trim the data
                        original_shape = hdu.data.shape
                        if x2 <= original_shape[1] and y2 <= original_shape[0]:
                            hdu.data = hdu.data[y1:y2, x1:x2]
                            print(f"Trimmed HDU {i} from {original_shape} to {hdu.data.shape} based on DATASEC={datasec}")
                        else:
                            print(f"Warning: DATASEC dimensions {datasec} exceed data dimensions {original_shape} for HDU {i}")
                
                # Check if the HDU has data
                if hdu.data is not None:
                    # Determine the color
                    color = None
                    
                    # If COLOR is in the header, use it directly
                    if 'COLOR' in hdu.header:
                        color = hdu.header['COLOR'].lower()
                        side  = hdu.header['SIDE']
                    # If COLOR is not in the header but CAM_NAME is, extract color from CAM_NAME
                    elif 'CAM_NAME' in hdu.header:
                        camname = hdu.header['CAM_NAME']
                        color = camname.split('_')[1].lower()
                        side = camname.split('_')[0][1]
                    
                    # Apply transformations based on color if we have determined it
                    if color:
                        if color == 'blue':
                            # Flip both x and y axes
                            #hdu.data = np.flip(hdu.data, (0, 1))
                            hdu.data = np.fliplr(hdu.data)
                            hdu.data = np.flipud(hdu.data)
                        elif color == 'green':
                            # Flip only x axis (flip horizontally)
                            hdu.data = np.fliplr(hdu.data)
                        # No change for red
                        
                
                result_hdus.append(hdu)
        
        return result_hdus
    
    except Exception as e:
        print(f"Error processing FITS file: {e}")
        return None  