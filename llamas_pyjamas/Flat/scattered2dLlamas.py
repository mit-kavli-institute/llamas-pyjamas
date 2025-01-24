

"""
Module: scattered2dLlamas
This module provides functionality to process and analyze 2D scattered light data from Llamas Pyjamas project. 
It includes functions to read FITS files, perform profile fitting, and generate fitted surfaces.
Functions:
    - scattered2dLlamas: Main function to process 2D scattered light data.
    Process and analyze 2D scattered light data.
    This function reads dark and profile FITS files, performs profile fitting, 
    and generates fitted surfaces. It also uses DS9 for visualization.
    Returns:
        tuple: A tuple containing:
            - fitimg (numpy.ndarray): The image after profile fitting and dark subtraction.
            - fitsurface (numpy.ndarray): The fitted surface.
            - profdata-dark (numpy.ndarray): The profile data after dark subtraction.
"""
from   ..Trace import traceLlamas
import numpy as np
#import pyds9
from   matplotlib import pyplot as plt
from   typing import Tuple
from   astropy.io import fits
from   pypeit.core import fitting as f


def scattered2dLlamas()->Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    Processes 2D scattered light data from Llamas.
    This function performs the following steps:
    1. Reads a dark frame from a FITS file.
    2. Reads a profile data frame from another FITS file.
    3. Uses the profile data to trace and fit the profile.
    4. Subtracts the dark frame from the profile data.
    5. Sets pixels in the fit image to zero where the profile fit is greater than zero.
    6. Displays the fit image using DS9.
    7. Creates x and y coordinate images.
    8. Identifies pixels to fit a 2D polynomial surface.
    9. Fits a 2D polynomial surface to the identified pixels.
    10. Evaluates the fitted surface.
    11. Displays the fitted surface using DS9.
    12. Returns the fit image, the fitted surface, and the dark-subtracted profile data.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - fitimg: The fit image with pixels set to zero where the profile fit is greater than zero.
            - fitsurface: The evaluated 2D polynomial surface.
            - profdata-dark: The dark-subtracted profile data.
    """


    darkfile = 'bllamas_0003_Camera00_DARK_60s_Loff.fits'
    hdu = fits.open(darkfile)
    dark = hdu[0].data
    hdu.close()
    
    proffile = '20200210_190852_Camera00_Blue_MASK.fits'
    hdu = fits.open(proffile)
    profdata = hdu[0].data
    hdu.close()
    
    trace  = traceLlamas.TraceLlamas(proffile, \
                         spectrograph=0,channel='blue',mph=1500)

    profimg = trace.profileFit()[0]

    fitimg = profdata - dark
    fitimg[np.where(profimg > 0)] = 0

    ds9 = pyds9.DS9(target='DS9:*', start=True, wait=10, verify=True)
    ds9.set_np2arr(fitimg)

    ximg = np.outer(np.ones(trace.naxis2),np.arange(trace.naxis1))
    yimg = np.outer(np.arange(trace.naxis2),np.ones(trace.naxis1))

    fitpix = np.where(np.logical_and(profimg == 0,ximg > 40))

    c,minx, maxx, miny,maxy = \
        f.polyfit2d_general(ximg[fitpix],yimg[fitpix],fitimg[fitpix],(2,2))

    fitsurface = f.evaluate_fit(c,'polynomial2d',ximg,x2=yimg)

    ds9.set_np2arr(fitsurface)
    
    return fitimg,fitsurface,profdata-dark

