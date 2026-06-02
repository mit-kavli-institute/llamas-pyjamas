import numpy as np


class dataCube():
    """
        DataContainer to hold the products of a datacube

        The datamodel attributes are:

        Args:
            flux (`numpy.ndarray`_):
                The science datacube (nwave, nspaxel_y, nspaxel_x)
            sig (`numpy.ndarray`_):
                The error datacube (nwave, nspaxel_y, nspaxel_x)
            bpm (`numpy.ndarray`_):
                The bad pixel mask of the datacube (nwave, nspaxel_y, nspaxel_x).
                True values indicate a bad pixel
            wave (`numpy.ndarray`_):
                A 1D numpy array containing the wavelength array for convenience (nwave)
                
                """
    
    
    datamodel = {'flux': dict(otype=np.ndarray, atype=np.floating,
                              descr='Flux datacube in units of counts/s/Ang/arcsec^2 or '
                                    '10^-17 erg/s/cm^2/Ang/arcsec^2'),
                 'sig': dict(otype=np.ndarray, atype=np.floating,
                             descr='Error datacube (matches units of flux)'),
                 'bpm': dict(otype=np.ndarray, atype=np.uint8,
                             descr='Bad pixel mask of the datacube (0=good, 1=bad)'),
                 'wave': dict(otype=np.ndarray, atype=np.floating,
                              descr='Wavelength of each slice in the spectral direction. '
                                    'The units are Angstroms.')}
    
    def __init__(self):
        
        self._ivar = None
        self._wcs = None
        self.head0 = None  # This contains the primary header of the spec2d used to make the datacube
        
        pass
        