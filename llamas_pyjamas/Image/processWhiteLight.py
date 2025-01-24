
"""
Module: processWhiteLight
This module provides functions for processing white light images, including 
removing striping patterns and applying quartile bias correction.
Functions:
- quartile_bias(frame, quartile=20): Applies a quartile bias correction to the input frame.
- remove_striping(image, axis=0, smoothing=None): Removes striping patterns from the input image.
Dependencies:
- numpy
- pickle
- matplotlib.pyplot
- scipy.interpolate.LinearNDInterpolator
- astropy.io.fits
- astropy.table.Table
- os
- matplotlib.tri.Triangulation
- matplotlib.tri.LinearTriInterpolator
- llamas_pyjamas.Utils.utils
- datetime
- traceback
    Apply a quartile bias correction to the input frame.
        frame (numpy.ndarray): The input 2D array representing the image frame.
        quartile (int, optional): The quartile percentage to use for bias correction. 
                                  Default is 20.
    Returns:
        numpy.ndarray: The bias-corrected frame.
    pass
    Remove striping pattern from the input image.
        image (numpy.ndarray): The input 2D array representing the image.
        axis (int, optional): The axis along which to remove stripes. 
                              0 for vertical stripes, 1 for horizontal. Default is 0.
        smoothing (int or None, optional): Optional smoothing window for the pattern. 
                                           If None, no smoothing is applied. Default is None.
    Returns:
        numpy.ndarray: The image with the striping pattern removed.
    pass
"""
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from ..Extract.extractLlamas import ExtractLlamas
from ..QA import plot_ds9
from ..config import OUTPUT_DIR
from astropy.io import fits
from astropy.table import Table
import os
from matplotlib.tri import Triangulation, LinearTriInterpolator
from llamas_pyjamas.Utils.utils import setup_logger
from datetime import datetime
import traceback

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, f'ProcessWhiteLight_{timestamp}.log')


def quartile_bias(frame: np.ndarray, quartile=20)-> np.ndarray:
    """
    Adjusts the input frame by subtracting the specified quartile value and setting negative values to zero.
    Parameters:
    frame (numpy.ndarray): The input data frame to be processed.
    quartile (int, optional): The percentile value to be used as the threshold. Default is 20.
    Returns:
    numpy.ndarray: The processed data frame with values adjusted based on the specified quartile.
    """

    threshold = np.nanpercentile(frame, quartile)
    cleaned_data = frame - threshold
    cleaned_data[cleaned_data < 0] = 0
    
    return cleaned_data

def remove_striping(image: np.ndarray, axis=0, smoothing=None)-> np.ndarray:
    """
    Remove striping pattern from image
    
    Args:
        image: 2D numpy array
        axis: 0 for vertical stripes, 1 for horizontal
        smoothing: Optional smoothing window for pattern
    """
    # Make copy to avoid modifying original
    cleaned = image.copy()
    
    # Calculate median along opposite axis
    pattern = np.nanmedian(image, axis=axis)
    
    # Optional smoothing of pattern
    if smoothing:
        from scipy.ndimage import gaussian_filter1d
        pattern = gaussian_filter1d(pattern, smoothing)
    
    # Reshape pattern to match image dimensions
    if axis == 0:
        pattern = pattern.reshape(1, -1)  # Row pattern
    else:
        pattern = pattern.reshape(-1, 1)  # Column pattern
        
    # Subtract pattern
    cleaned = image - pattern
    
    # Handle negatives
    cleaned[cleaned < 0] = 0
    
    return cleaned