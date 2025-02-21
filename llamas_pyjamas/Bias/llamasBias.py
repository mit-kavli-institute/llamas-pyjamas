

"""
This module provides functionality to create a master bias frame from a directory of bias images or a list of files.
Classes:
    BiasLlamas: A class to handle bias image processing and creation of a master bias frame.
Usage:
    To use this module, instantiate the BiasLlamas class with a directory path containing bias images or a list of bias image files.
    Then call the `master_bias` method to create and save the master bias frame.
Example:
    bias_llamas = BiasLlamas('/path/to/bias/images')

    A class to handle bias image processing and creation of a master bias frame.
    Attributes:
        bias_path (str): The directory path containing bias images.
        files (list): A list of bias image files.
    Methods:
        master_bias(): Creates and saves the master bias frame from the provided bias images.

        Initializes the BiasLlamas class with the provided input data.
        Args:
            input_data (str or list): A directory path containing bias images or a list of bias image files.
        Raises:
            ValueError: If the provided directory does not exist or no .fits files are found.
            TypeError: If the input data is not a string or a list.

        Creates and saves the master bias frame from the provided bias images.
        The method reads bias images from the specified directory or list, combines them using average method with sigma clipping,
        and saves the combined bias frame as 'combined_bias.fit' in the specified directory.
        Raises:
            ValueError: If no bias images are found in the specified directory or list.
        """
import os
from astropy.nddata import CCDData
from astropy.stats import mad_std

import ccdproc as ccdp
import matplotlib.pyplot as plt
import numpy as np
import argparse
from astropy.io import fits
from llamas_pyjamas.config import CALIB_DIR



class BiasLlamas:
    """
    A class to handle the creation of a master bias frame from a directory of bias images or a list of files.
    Attributes:
    -----------
    bias_path : str
        The directory path containing bias images.
    files : list
        A list of bias image files.
    Methods:
    --------
    __init__(input_data):
        Initializes the BiasLlamas object with a directory path or a list of files.
    master_bias():
        Creates a master bias frame by combining bias images using sigma clipping and saves the result.
    """

    def __init__(self, input_data) -> None:
        
        if isinstance(input_data, str):
            if not os.path.isdir(input_data):
                raise ValueError(f"The directory {input_data} does not exist.")
            self.bias_path = input_data
            self.files = [f for f in os.listdir(input_data) if f.endswith('.fits')]
            
        elif isinstance(input_data, list):
            self.files = input_data
            self.bias_path = CALIB_DIR
        
        else:
            raise TypeError("Input must be a string (directory path) or a list of files.")
        
        if not self.files:
            raise ValueError("No .fits files found in the provided directory or list.")
        
        return
    

    def master_bias(self) -> None:
        """
        Create a master bias frame from a list of bias images.
        Each input FITS file is assumed to have a primary HDU and one or more extensions,
        where each extension contains a bias image from a detector.
        The function loops over all extensions (determined dynamically), 
        averages the corresponding images across all files, and builds a new FITS file with:
          - A primary HDU (using the primary header from the first file)
          - One ImageHDU per extension, where the data is the mean of all corresponding exposures.
        The resulting combined FITS file is written to 'combined_bias.fits' in the bias_path.
        """
        # Open first file to get primary header and number of extensions
        with fits.open(self.files[0]) as hdulist_first:
            primary_hdr = hdulist_first[0].header.copy()
            num_ext = len(hdulist_first) - 1  # excluding primary HDU
        
        combined_hdus = []
        # Create primary HDU using the primary header
        primary_hdu = fits.PrimaryHDU(header=primary_hdr)
        combined_hdus.append(primary_hdu)

        # Loop over each extension (1, 2, ... num_ext)
        for ext in range(1, num_ext + 1):
            data_list = []
            header_list = []
            for file in self.files:
                with fits.open(file) as hdul:
                    # Read the data and header for current extension
                    data = hdul[ext].data
                    hdr = hdul[ext].header.copy()
                    data_list.append(data)
                    header_list.append(hdr)
            # Stack and compute mean (using nanmean to avoid any NaN issues)
            combined_data = np.nanmedian(np.array(data_list), axis=0)
            combined_header = header_list[0]
            combined_header['COMBINED'] = True
            # Create an ImageHDU for the combined extension data
            image_hdu = fits.ImageHDU(data=combined_data, header=combined_header, name=f'EXT{ext}')
            combined_hdus.append(image_hdu)
        
        # Assemble all HDUs into an HDUList and write to file
        hdul_out = fits.HDUList(combined_hdus)
        out_filename = os.path.join(self.bias_path, 'combined_bias.fits')
        hdul_out.writeto(out_filename, overwrite=True)

 