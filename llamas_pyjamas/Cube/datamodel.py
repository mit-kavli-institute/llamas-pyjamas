"""Data model for LLAMAS IFU data cubes.

Defines the :class:`dataCube` container that holds the flux, error, bad-pixel
mask, and wavelength arrays produced during cube construction, along with the
``datamodel`` describing each array's expected type and units.
"""

import numpy as np


class dataCube():
    """
        DataContainer to hold the products of a datacube

        The datamodel attributes are:

        Args:
            flux (``numpy.ndarray``):
                The science datacube (nwave, nspaxel_y, nspaxel_x)
            sig (``numpy.ndarray``):
                The error datacube (nwave, nspaxel_y, nspaxel_x)
            bpm (``numpy.ndarray``):
                The bad pixel mask of the datacube (nwave, nspaxel_y, nspaxel_x).
                True values indicate a bad pixel
            wave (``numpy.ndarray``):
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
        """Initialise an empty data cube container.

        Sets up the placeholder attributes for inverse variance, WCS, and the
        primary header; the flux/error/mask/wavelength arrays are populated
        separately during cube construction.

        Attributes:
            _ivar: Inverse variance array (None until populated).
            _wcs: World Coordinate System for the cube (None until populated).
            head0: Primary header of the spec2d frame used to build the cube
                (None until populated).
        """
        self._ivar = None
        self._wcs = None
        self.head0 = None  # This contains the primary header of the spec2d used to make the datacube
        
        pass
        