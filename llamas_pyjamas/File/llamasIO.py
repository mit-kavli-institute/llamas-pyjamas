


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
                


    