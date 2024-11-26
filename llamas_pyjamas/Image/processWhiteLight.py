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


def quartile_bias(frame, quartile=20):
    threshold = np.nanpercentile(frame, quartile)
    cleaned_data = frame - threshold
    cleaned_data[cleaned_data < 0] = 0
    
    return cleaned_data

def remove_striping(image, axis=0, smoothing=None):
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