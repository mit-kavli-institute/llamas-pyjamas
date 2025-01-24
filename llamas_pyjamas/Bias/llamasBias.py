
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
        
        else:
            raise TypeError("Input must be a string (directory path) or a list of files.")
        
        if not self.files:
            raise ValueError("No .fits files found in the provided directory or list.")
        
        return
    
    def master_bias(self)-> None:
        """
        Create a master bias frame from a directory of bias images.
        This function reads bias images from the specified directory, combines them using
        sigma-clipping and averaging, and writes the resulting master bias frame to a file.
        Parameters:
        None
        Returns:
        None
        """

        
        bias_images = ccdp.ImageFileCollection(self.bias_path)
        calibrated_biases = bias_images.files_filtered(imagetyp='bias', include_path=True)
        
        combined_bias = ccdp.combine(calibrated_biases,
                             method='average',
                             sigma_clip=True, sigma_clip_low_thresh=5, sigma_clip_high_thresh=5,
                             sigma_clip_func=np.ma.median, sigma_clip_dev_func=mad_std,
                             mem_limit=350e6
                            )

        combined_bias.meta['combined'] = True

        combined_bias.write(self.bias_path / 'combined_bias.fit')

        if __name__ == "__main__":

            parser = argparse.ArgumentParser(description="Create a master bias frame from a directory of bias images or a list of files.")
            parser.add_argument('input_data', type=str, help="Directory path containing bias images or a list of bias image files.")
            
            args = parser.parse_args()
            
            bias_llamas = BiasLlamas(args.input_data)
            bias_llamas.master_bias()