

"""
Module: scattered2dLlamas
2D scattered-light modelling: fits a polynomial surface to the inter-fibre pixels of a
dark-subtracted frame.
"""
# from   ..Trace import traceLlamas
from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas
from llamas_pyjamas.Image.WhiteLightModule import WhiteLightFits
import numpy as np
#import pyds9
from   matplotlib import pyplot as plt
from   typing import Tuple
from   astropy.io import fits
from   pypeit.core import fitting as f


def scattered2dLlamas()->Tuple[np.ndarray, np.ndarray, np.ndarray]: 
    """
    Fit a 2D polynomial scattered-light surface to a dark-subtracted profile frame.

    Pixels under the fibre profile fit are zeroed and a (2, 2) polynomial is fit to the
    remaining inter-fibre pixels; the image and surface are displayed in DS9.
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


