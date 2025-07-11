"""Module for processing and analyzing white light images from LLAMAS.

This module provides functions for processing white light images from the LLAMAS 
instrument, including color channel isolation, FITS file creation, and fiber mapping.

Functions:
    color_isolation: Isolates blue, green, and red channels from extraction objects.
    WhiteLightFits: Creates a FITS file from extraction objects.
    WhiteLight: Generates a white light image from extraction objects.
    WhiteLightQuickLook: Generates a quick look white light image.
    WhiteLightHex: Creates hexagonal grid white light images.
    FiberMap: Maps a fiber to its x and y coordinates.
    FiberMap_LUT: Looks up fiber coordinates using a lookup table.
    plot_fibermap: Plots the fiber map for the LLAMAS instrument.
    fibermap_table: Generates a table of fiber mappings.
    rerun: Reruns the white light generation process.
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.QA import plot_ds9
from llamas_pyjamas.config import OUTPUT_DIR, CALIB_DIR
from astropy.io import fits
from astropy.table import Table
import os
from matplotlib.tri import Triangulation, LinearTriInterpolator
from llamas_pyjamas.Utils.utils import setup_logger
from datetime import datetime
import traceback
from llamas_pyjamas.config import LUT_DIR
from typing import Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from llamas_pyjamas.File.llamasIO import process_fits_by_color

from matplotlib.patches import RegularPolygon
import matplotlib.cm as cm
from matplotlib.colors import Normalize



timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, f'WhiteLight_{timestamp}.log')

orig_fibre_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LLAMAS_FiberMap_revA.dat')
fibre_map_path = os.path.join(LUT_DIR, 'LLAMAS_FiberMap_rev04.dat')
print(f'Fibre map path: {fibre_map_path}')
fibermap_lut = Table.read(fibre_map_path, format='ascii.fixed_width')


def color_isolation(extractions: list, metadata: dict)-> Tuple[list, list, list]:
    """Isolate blue, green, and red channels from extraction objects.

    This function separates extraction objects by their color channels and returns 
    both the extraction objects and their corresponding metadata.

    Args:
        extractions (list): A list of extraction objects loaded from ExtractLlamas.
        metadata (dict): Dictionary containing metadata for each extraction.

    Returns:
        tuple: A tuple containing six lists:
            - blue_extractions (list): Blue channel extraction objects.
            - green_extractions (list): Green channel extraction objects.
            - red_extractions (list): Red channel extraction objects.
            - blue_meta (list): Metadata for blue channel extractions.
            - green_meta (list): Metadata for green channel extractions.
            - red_meta (list): Metadata for red channel extractions.
    """

    blue_extractions = [ext for ext in extractions if ext.channel.lower() == 'blue']
    green_extractions = [ext for ext in extractions if ext.channel.lower() == 'green']
    red_extractions = [ext for ext in extractions if ext.channel.lower() == 'red']

    blue_meta = [meta for meta in metadata if meta['channel'].lower() == 'blue']
    green_meta = [meta for meta in metadata if meta['channel'].lower() == 'green']
    red_meta = [meta for meta in metadata if meta['channel'].lower() == 'red']

    
    return blue_extractions, green_extractions, red_extractions, blue_meta, green_meta, red_meta


def WhiteLightFits(extraction_array: list, metadata: dict, outfile=None)-> str:
    """Process extraction data to create a white light FITS file.

    This function takes an array of extracted color data and creates a FITS file 
    containing white light images for each bench/side/channel combination.

    Args:
        extraction_array (list): A list of extracted color data arrays.
        metadata (dict): Dictionary containing metadata for each extraction.
        outfile (str, optional): The output file path for the white light FITS file. 
            If None, the output file name is generated based on the input file name. 
            Defaults to None.

    Returns:
        str: The file path of the created white light FITS file.

    Note:
        - The function assumes that all extraction objects came from the same original file.
        - The function processes blue, green, and red data if they exist in the extraction array.
        - The function creates a primary HDU and additional HDUs for each color data and their corresponding tables.
        - The function writes the created HDU list to a FITS file in the specified output directory.
    """

    
    blue, green, red, blue_meta, green_meta, red_meta = color_isolation(extraction_array, metadata)
    print(blue, green, red)
    fitsfile = None
    ###For now assuming that all extraction objects came from the same original file
    if all(not color for color in [blue, green, red]):
        logger.error('No blue, green, or red extractions found. Exiting...')
        return
    
    # Create HDU list
    hdul = fits.HDUList()
    primary_hdu = fits.PrimaryHDU()
    
    
    fitsfile = blue[0].fitsfile if blue else green[0].fitsfile if green else red[0].fitsfile
    primary_hdu.header['ORIGFILE'] = os.path.basename(fitsfile)
    hdul.append(fits.PrimaryHDU())

    # Process blue data if exists
    if blue:
        
        blue_whitelight, blue_x, blue_y, blue_flux = WhiteLight(blue, blue_meta, ds9plot=False)
        blue_hdu = fits.ImageHDU(data=blue_whitelight.astype(float), name='BLUE')
        hdul.append(blue_hdu)
        
        blue_tab = fits.BinTableHDU.from_columns([
            fits.Column(name='XDATA', format='E', array=blue_x.astype(np.float32)),
            fits.Column(name='YDATA', format='E', array=blue_y.astype(np.float32)),
            fits.Column(name='FLUX', format='E', array=blue_flux.astype(np.float32))
        ], name='BLUE_TAB', nrows=len(blue_whitelight))
        hdul.append(blue_tab)
    
    # Process green data if exists
    if green:
      
        green_whitelight, green_x, green_y, green_flux = WhiteLight(green, green_meta, ds9plot=False)
        green_hdu = fits.ImageHDU(data=green_whitelight.astype(float), name='GREEN')
        hdul.append(green_hdu)
        
        green_tab = fits.BinTableHDU.from_columns([
            fits.Column(name='XDATA', format='E', array=green_x.astype(float)),
            fits.Column(name='YDATA', format='E', array=green_y.astype(float)),
            fits.Column(name='FLUX', format='E', array=green_flux.astype(float))
        ], name='GREEN_TAB')
        hdul.append(green_tab)
    
    # Process red data if exists
    if red:
        red_whitelight, red_x, red_y, red_flux = WhiteLight(red, red_meta, ds9plot=False)
        red_hdu = fits.ImageHDU(data=red_whitelight.astype(np.float32), name='RED')
        hdul.append(red_hdu)
        
        red_tab = fits.BinTableHDU.from_columns([
            fits.Column(name='XDATA', format='E', array=red_x.astype(np.float32)),
            fits.Column(name='YDATA', format='E', array=red_y.astype(np.float32)),
            fits.Column(name='FLUX', format='E', array=red_flux.astype(np.float32))
        ], name='RED_TAB', nrows=len(red_whitelight))
        hdul.append(red_tab)
    if not outfile:
        fitsfilebase = fitsfile.split('/')[-1]
        white_light_file = fitsfilebase.replace('.fits', '_whitelight.fits')
    
    #code added in to allow for normalisation in flat fielding process
    elif outfile == -1:
        return hdul
    else:
        white_light_file = outfile
    
    print(f'Writing white light file to {white_light_file}')
    # Write to file
    hdul.writeto(os.path.join(OUTPUT_DIR, white_light_file), overwrite=True)
    
    return white_light_file


def WhiteLight(extraction_array: list, metadata: list, ds9plot=True)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a white light image from an array of extraction files or objects.
    Parameters:
    extraction_array (list): A list of extraction files (str) or ExtractLlamas objects.
    ds9plot (bool, optional): If True, plot the white light image using DS9. Default is True.
    Returns:
    tuple: A tuple containing:
        - whitelight (numpy.ndarray): The interpolated white light image.
        - xdata (numpy.ndarray): The x-coordinates of the fiber positions.
        - ydata (numpy.ndarray): The y-coordinates of the fiber positions.
        - flux (numpy.ndarray): The flux values for each fiber.
    Raises:
    AssertionError: If extraction_array is not a list.
    TypeError: If an element in extraction_array is not a string or ExtractLlamas object.
    """

    
    assert type(extraction_array) == list, 'Extraction array must be a list of extraction files'
    
    xdata = np.array([])
    ydata = np.array([])
    flux  = np.array([])
    
    for extraction_obj, meta in zip(extraction_array, metadata):
        channel = meta['channel']
        side = meta['side']
        counts = extraction_obj.counts
        #Might need to put in side condition here as well it depends on the outcome
        # if channel == 'blue':
        #     counts = np.flipud(extraction_obj.counts)
    
        if isinstance(extraction_obj, str):
            extraction, _ = ExtractLlamas.loadExtraction(extraction_obj)
            logger.info(f'Loaded extraction object {extraction.bench}{extraction.side}')
        elif isinstance(extraction_obj, ExtractLlamas):
            extraction = extraction_obj
        else:
            raise TypeError(f"Unexpected type: {type(extraction_obj)}. Must be string or ExtractLlamas object")
        
        
        nfib, naxis1 = np.shape(extraction.counts)
        
        for ifib in range(nfib):
            benchside = f'{extraction.bench}{extraction.side}'
            
            
            try:
                x, y = FiberMap_LUT(benchside,ifib)
            except Exception as e:
                logger.info(f'Fiber {ifib} not found in fiber map for bench {benchside} for color {extraction.channel}')
                logger.error(traceback.format_exc())
                continue
            
            
            # thisflux = np.nansum(extraction.counts[ifib])
            thisflux = np.nansum(counts[ifib])
            flux = np.append(flux, thisflux)
            xdata = np.append(xdata,x)
            ydata = np.append(ydata,y)

    flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux, fill_value=np.nan)
        
    if (False):
        xx = np.arange(53)
        yy = np.arange(53)
    else:

        subsample = 1.5

        xx = 1.0/subsample * np.arange(53*subsample)
        yy = 1.0/subsample * np.arange(53*subsample)

    x_grid, y_grid = np.meshgrid(xx/subsample, yy/subsample)
    
    whitelight = flux_interpolator(x_grid, y_grid)
    # whitelight = np.fliplr(whitelight)
    if (ds9plot):
        #ds9 = pyds9.DS9(target='DS9:*', start=True, wait=10, verify=True)
        #ds9.set_np2arr(whitelight)
        plot_ds9(whitelight)

    return whitelight, xdata, ydata, flux

def WhiteLightQuickLook(tracefile: str, data)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a quick look white light image from trace data and image data.
    Parameters:
    tracefile (str): Path to the trace file containing the trace object.
    data (numpy.ndarray): Image data array.
    Returns:
    tuple: A tuple containing:
        - whitelight (numpy.ndarray): Interpolated white light image.
        - xdata (numpy.ndarray): Array of x-coordinates for fibers.
        - ydata (numpy.ndarray): Array of y-coordinates for fibers.
        - flux (numpy.ndarray): Array of flux values for fibers.
    Notes:
    - The function reads the trace object from the provided tracefile.
    - It uses a fiber map lookup table (FiberMap_LUT) to get x and y coordinates for each fiber.
    - The flux for each fiber is calculated by summing the data values where the fiber image matches the fiber index.
    - A linear interpolator (LinearNDInterpolator) is used to create the white light image.
    - Optionally, the white light image can be plotted using DS9 (if ds9plot is set to True).
    """

    #    hdul = fits.open(data)

    # Each trace object represents one camera / side pair
    
    with open(tracefile, "rb") as fp:
        traceobj = pickle.load(fp)
    fiberimg = traceobj.fiberimg
    nfib     = traceobj.nfibers
    
    xdata = np.array([])
    ydata = np.array([])
    flux  = np.array([])
    for ifib in range(nfib):
        benchside = f'{traceobj.bench}{traceobj.side}'
        channel = f'{traceobj.channel}'
        try:
            x, y = FiberMap_LUT(benchside,ifib)
        except Exception as e:
            logger.info(f'Fiber {ifib} not found in fiber map for bench {benchside}')
            logger.error(traceback.format_exc())
            continue
        
        thisflux = np.nansum(data[fiberimg == ifib])
        flux = np.append(flux, thisflux)
        xdata = np.append(xdata,x)
        ydata = np.append(ydata,y)

    flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux, fill_value=np.nan)
        
    xx = np.arange(46)
    yy = np.arange(43)
    x_grid, y_grid = np.meshgrid(xx, yy)
    
    whitelight = flux_interpolator(x_grid, y_grid)

    ds9plot = False
    if (ds9plot):
        plot_ds9(whitelight, samp=True)

    return whitelight, xdata, ydata, flux

        
def WhiteLightHex(extraction_array, ds9plot=True):
    pass

    ## placeholder for eventual hexagonal grid inclusion

    return

   
def FiberMap(bench: str, infiber: int)-> Tuple[float, float]:
    """
    Calculate the fiber map coordinates for a given bench and fiber number.
    Parameters:
    bench (str): The bench identifier, which can be '1A', '2A', '3A', '4A', '1B', '2B', '3B', or '4B'.
    infiber (int): The fiber number.
    Returns:
    tuple: A tuple containing the x and y coordinates of the fiber.
    """


    n_right    = 23
    n_left     = 23
    ncols = n_left + n_right
    n_vertical = 51
    dy         = 0.8660254037 # == np.sin(60), but hardwire the factor for time saving
    dx         = 1.0

    wrap = 0
    if (bench == '1A'):
        x0 = 0.0
        y0 = 0.0
        Nfib=298
    elif (bench == '2A'):
        x0 = np.floor_divide(n_right,2)
        y0 = 6.0
        wrap = 23
        Nfib=300
    elif (bench == '3A'):
        x0 = 0.0
        y0 = 13.0
        Nfib=298
    elif (bench == '4A'):
        x0 = np.floor_divide(n_right,2)
        y0 = 19
        wrap = 23
        Nfib=300

        
    elif (bench == '1B'):
        x0 = 0.0
        y0 = 45
        Nfib = 300#298
        wrap = 23
    elif (bench == '2B'):
        x0 = np.floor_divide(n_right,2)
        y0 = 39.0
        Nfib = 298#300
    elif (bench == '3B'):
        x0 = 0.0
        y0 = 32.0
        Nfib=300#298
        wrap = 23
    elif (bench == '4B'):
        x0 = np.floor_divide(n_right,2)
        y0 = 26.0
        Nfib = 298#300
        
    fiber = infiber
        
    if (fiber % 2 == 1):
        xoffset = n_left
        wrap -= 1
    else:
        xoffset = 0


    if ('A' in bench):
        y_rownum = np.floor_divide((fiber+wrap), (n_left+n_right))
        y_value  = (y0 + y_rownum) * dy

        # Account for the fact that the fibers snake back and forth within the two
        # sides of the IFU. Even rows go right to left, odd rows go left to right

        x_index = np.floor_divide(((fiber+wrap) % (n_left+n_right)),2)

        if ((bench == '1A') or (bench == '2A')):
            offset = 1
        else:
            offset = 0

        if (((y_rownum+y0+offset) % 2) == 0):
            if ((fiber % 2) == 1):
                x_fiber = (n_left+n_right) - x_index - 1
            else:
                x_fiber = n_left - x_index - 1
        else:
            if ((fiber % 2) == 1):
                x_fiber = n_left + 0.5 + x_index
            else:
                x_fiber = 0.5 + x_index

        x_value = x_fiber - 0.5
        x_fiber = ncols - x_fiber
        y_value = n_vertical*dy-y_value

    elif ('B' in bench):

        y_rownum = np.floor_divide((fiber+wrap), ncols)
        y_value  = (y0 + y_rownum) * dy

        # Account for the fact that the fibers snake back and forth within the two
        # sides of the IFU. Even rows go right to left, odd rows go left to right

        x_index = np.floor_divide(((fiber+wrap) % (n_left+n_right)),2)

        if ((bench == '2B') or (bench == '1B')):
            offset = 1
        else:
            offset = 0

        if (((y_rownum+y0+offset) % 2) == 0):
            if ((fiber % 2) == 0):
                # Even fiber number
                x_fiber = (n_left+n_right) - x_index
            else:
                # Odd fiber numbers
                x_fiber = n_left - x_index

            x_fiber -= 0.5
            # x_value = x_fiber - 0.5

            #if (bench=='2B'):
            #    x_fiber+=0.5
            
        else:
            if ((fiber % 2) == 0):
                x_fiber = n_left + 0.5 + x_index
            else:
                x_fiber = 0.5 + x_index

            x_fiber += 0.5

            #if (bench=='2B'):
            #    x_fiber-=0.5
            

        y_value = n_vertical*dy-y_value
        
    # return(x_value,y_value)

    y_final = n_vertical-int(y0+y_rownum)
    x_final = x_fiber

    if ((bench == '1B') or (bench == '2B') or (bench == '3A') or (bench == '4A')):
        if (y_final % 2 == 0):
            x_final += 0.5
        else:
            x_final -= 0.5   

    # return(x_fiber,n_vertical-int(y0+y_rownum))
    return(x_final, y_final)

def FiberMap_LUT(bench: str, fiber: int)-> Tuple[float, float]:

    #if (np.logical_and(bench == '2B',fiber >= 49)):
    #    fiber += 1
    
    fiber_row = fibermap_lut[np.logical_and(fibermap_lut['bench']==bench, \
                                            fibermap_lut['fiber']==fiber)]
    #breakpoint()
    try:
        return(fiber_row['xpos'][0],fiber_row['ypos'][0])
    except:
        return(-1,-1)

def plot_fibermap(outpath: str)-> None:
    """
    Plots the fiber map for the LLAMAS IFU, showing the mapping of fibers to different configurations.
    This function generates a plot with the fiber numbers annotated at their respective positions for 
    different configurations (1A, 2A, 3A, 4A, 1B, 2B, 3B, 4B). It also includes directional annotations 
    (N for North, E for East) and saves the plot as an image file.
    Annotations:
    - Fibers in configuration 1A, 3A, 4B, and 2B are plotted in black and red.
    - Fibers in configuration 2A, 4A, 3B, and 1B are plotted in blue and green.
    - Directional annotations for North and East are included.
    The plot is saved as "fiber.png" in the specified directory.
    Returns:
        None
    """


    # 1A - N=298
    # 2A - N=300
    # 3A - all fibers fine (N=298)
    # 4A - all fibers fine (N=300)
    # 4B - all fibers fine (N=298)
    # 3B - all fibers fine (N=300)
    # 2B - fiber 49 (zero index) is broken / dead (N=297 good + 1 dead)
    # 1B - all fibers fine (N=300)
    
    fig, ax = plt.subplots(1)
    fibernum_a = np.arange(298)
    fibernum_b = np.arange(300)

    fs = 4
    for fiber in fibernum_a:
        x, y = FiberMap_LUT('1A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='k')

        x, y = FiberMap_LUT('3A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='r')

        x, y = FiberMap_LUT('4B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='k')

        x, y = FiberMap_LUT('2B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='r')
        
        
    for fiber in fibernum_b:
        x, y = FiberMap_LUT('2A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='b')

        x, y = FiberMap_LUT('4A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='g')

        x, y = FiberMap_LUT('3B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='b')

        x, y = FiberMap_LUT('1B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='g')
        
        
    ax.set_xlim(-2,75)
    ax.set_ylim(-2,47)
    ax.set_aspect('equal')

    ax.text(49,2.5,"1B")
    ax.text(49,7.5,"2B")
    ax.text(49,12.5,"3B")
    ax.text(49,17.5,"4B")
    ax.text(49,24.5,"4A")
    ax.text(49,29.5,"3A")
    ax.text(49,34.5,"2A")
    ax.text(49,39.5,"1A")

    fs = 8

    ax.text(52.25, 40.5, "R-1", fontsize=fs)
    ax.text(52.25, 39.5, "G-2", fontsize=fs)
    ax.text(52.25, 38.5, "B-3", fontsize=fs)

    ax.text(52.25, 35.5, "R-7", fontsize=fs)
    ax.text(52.25, 34.5, "G-8", fontsize=fs)
    ax.text(52.25, 33.5, "B-9", fontsize=fs)

    ax.text(52.25, 30.5, "R-13", fontsize=fs)
    ax.text(52.25, 29.5, "G-14", fontsize=fs)
    ax.text(52.25, 28.5, "B-15", fontsize=fs)

    ax.text(52.25, 25.5, "R-19", fontsize=fs)
    ax.text(52.25, 24.5, "G-20", fontsize=fs)
    ax.text(52.25, 23.5, "B-21", fontsize=fs)

    ax.text(52.25, 3.5, "R-4", fontsize=fs)
    ax.text(52.25, 2.5, "G-5", fontsize=fs)
    ax.text(52.25, 1.5, "B-6", fontsize=fs)

    ax.text(52.25, 8.5, "R-10", fontsize=fs)
    ax.text(52.25, 7.5, "G-11", fontsize=fs)
    ax.text(52.25, 6.5, "B-12", fontsize=fs)

    ax.text(52.25, 13.5, "R-16", fontsize=fs)
    ax.text(52.25, 12.5, "G-17", fontsize=fs)
    ax.text(52.25, 11.5, "B-18", fontsize=fs)

    ax.text(52.25, 18.5, "R-22", fontsize=fs)
    ax.text(52.25, 17.5, "G-23", fontsize=fs)
    ax.text(52.25, 16.5, "B-24", fontsize=fs)

    ax.annotate('', xy=(73,20), xytext=(60,20),
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))

    ax.annotate('', xy=(60,25), xytext=(60,20),
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))
    ax.text(71, 15.5, "N", fontsize=12)
    ax.text(60, 27, "E", fontsize=12)

    plt.title("LLAMAS IFU Fiber to slit / FITS extension mapping (Rev 04)")

    plt.tight_layout()
    fig.savefig(outpath, dpi=600)
    plt.show()


def fibermap_table()-> Table:
    """
    Generates a fiber map table for the LLAMAS instrument and writes it to a file.
    The function creates a table with columns: 'bench', 'fiber', 'xindex', 'yindex', 'xpos', and 'ypos'.
    It populates the table with fiber positions for both A and B sides of the instrument, using the FiberMap function
    to get the x and y indices for each fiber. The y position is adjusted by the sine of 60 degrees.
    The table is then written to a file named 'LLAMAS_FiberMap_rev02_updated.dat' in fixed-width ASCII format.
    Returns:
        astropy.table.Table: The populated fiber map table.
    """


    fiber_table = Table(names=('bench','fiber','xindex','yindex','xpos','ypos'),\
                        dtype=('S4','i4','i4','i4','f4','f4'))

    # A sides - DONE!
    
    for ifib in range(298):
        ix, iy = FiberMap('1A',ifib)
        fiber_table.add_row(['1A',int(ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(300):
        ix, iy = FiberMap('2A',ifib)
        fiber_table.add_row(['2A',int(ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(298):
        ix, iy = FiberMap('3A',ifib)
        fiber_table.add_row(['3A',int(ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(300):
        ix, iy = FiberMap('4A',ifib)
        fiber_table.add_row(['4A',int(ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    # B Sides
        
    for ifib in range(300):
        ix, iy = FiberMap('1B',ifib)
        fiber_table.add_row(['1B',int(299-ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(298):
        ix, iy = FiberMap('2B',ifib)
        fiber_table.add_row(['2B',int(297-ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(300):
        ix, iy = FiberMap('3B',ifib)
        fiber_table.add_row(['3B',int(299-ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(298):
        ix, iy = FiberMap('4B',ifib)
        fiber_table.add_row(['4B',int(297-ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    fiber_table.write('LLAMAS_FiberMap_rev03.dat', format='ascii.fixed_width', overwrite=True)
        
    return(fiber_table)
    
def rerun():
    """
    Reruns the WhiteLight process with a set of predefined extractions.
    This function loads several extraction files using the ExtractLlamas class and then
    runs the WhiteLight process with these extractions.
    The following extraction files are loaded:
    - Extract_1A.pkl
    - Extract_2A.pkl
    - Extract_3A.pkl
    - Extract_4A.pkl
    - Extract_1B.pkl
    - Extract_2B.pkl
    - Extract_3B.pkl
    - Extract_4B.pkl
    The loaded extractions are then passed to the WhiteLight function for processing.
    """

    extraction1a = ('Extract_1A.pkl')
    extraction2a = ExtractLlamas.loadExtraction('Extract_2A.pkl')
    extraction3a = ExtractLlamas.loadExtraction('Extract_3A.pkl')
    extraction4a = ExtractLlamas.loadExtraction('Extract_4A.pkl')

    extraction1b = ExtractLlamas.loadExtraction('Extract_1B.pkl')
    extraction2b = ExtractLlamas.loadExtraction('Extract_2B.pkl')
    extraction3b = ExtractLlamas.loadExtraction('Extract_3B.pkl')
    extraction4b = ExtractLlamas.loadExtraction('Extract_4B.pkl')
    

    WhiteLight([extraction1a, extraction2a, extraction3a, extraction4a, extraction1b, extraction2b, extraction3b, extraction4b])



######### Testing qucik whitelight

def QuickWhiteLight(trace_list, data_list, metadata=None, ds9plot=False):
    """
    Generate a white light image by directly summing unmasked fiber values without extraction.
    
    Parameters:
    -----------
    trace_list : list
        A list of TraceLlamas objects containing the fiber trace information.
    data_list : list
        A list of data arrays corresponding to each trace object.
    metadata : list, optional
        Optional metadata for each trace/data pair.
    ds9plot : bool, optional
        If True, display the resulting white light image using DS9. Default is False.
    
    Returns:
    --------
    tuple
        A tuple containing:
        - whitelight (numpy.ndarray): The interpolated white light image.
        - xdata (numpy.ndarray): The x-coordinates of the fiber positions.
        - ydata (numpy.ndarray): The y-coordinates of the fiber positions.
        - flux (numpy.ndarray): The flux values for each fiber.
    """

    xdata = np.array([])
    ydata = np.array([])
    flux = np.array([])
    
    for trace_obj, data, meta in zip(trace_list, data_list, metadata if metadata else [None]*len(trace_list)):
        # Get bench and side information
        bench = trace_obj.bench
        side = trace_obj.side
        channel = trace_obj.channel if hasattr(trace_obj, 'channel') else meta.get('channel') if meta else None
        
        # Process each fiber
        for ifib in range(trace_obj.nfibers):
            # Get fiber mask from the trace object
            fiber_mask = trace_obj.fiberimg == ifib
            
            if not np.any(fiber_mask):
                continue  # Skip if no pixels for this fiber
            
            # Get bench-side identifier
            benchside = f'{bench}{side}'
            
            try:
                # Map fiber to physical coordinates
                x, y = FiberMap_LUT(benchside, ifib)
                if x == -1 and y == -1:
                    continue  # Skip if fiber mapping not found
            except Exception as e:
                logger.info(f'Fiber {ifib} not found in fiber map for bench {benchside}')
                logger.error(traceback.format_exc())
                continue
            
            # Sum the flux directly from masked values in the data
            thisflux = np.nansum(data[fiber_mask])
            
            # Record the position and flux
            flux = np.append(flux, thisflux)
            xdata = np.append(xdata, x)
            ydata = np.append(ydata, y)
    
    # Create interpolated image
    flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux, fill_value=np.nan)
    
    # Define grid for interpolation
    subsample = 1.5
    xx = 1.0/subsample * np.arange(53*subsample)
    yy = 1.0/subsample * np.arange(53*subsample)
    x_grid, y_grid = np.meshgrid(xx, yy)
    
    # Generate white light image
    whitelight = flux_interpolator(x_grid, y_grid)
    
    # Optional DS9 plot
    if ds9plot:
        plot_ds9(whitelight)
    
    return whitelight, xdata, ydata, flux

def QuickWhiteLightCube(science_file, bias: str = None, ds9plot: bool = False, outfile: str = None) -> str:
        """
        Generates a cube FITS file with quick-look white light images for each color.
        The function groups the mastercalib dictionary by color (keys: blue, green, red),
        calls QuickWhiteLight for each color group, and creates an HDU for the image and an
        associated binary table HDU with fiber positions and flux data.
        
        Parameters:
            mastercalib (dict): Dictionary with keys 'blue', 'green', and 'red'. For each key, 
                the value should be a dict with the following entries:
                    'traces'   - list of trace objects,
                    'data'     - list of corresponding data arrays,
                    'metadata' - (optional) list of metadata dictionaries.
            ds9plot (bool, optional): If True, display each generated white light image using DS9.
                                        Default is True.
            outfile (str, optional): Output FITS file name. If None, a file name is generated
                                     with the current timestamp.
        
        Returns:
            str: The file path of the created quick-look white light cube FITS file.
        """

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Assuming DATA_DIR is defined and mastercalib is a subdirectory under DATA_DIR

        trace_objs = []

        # Open the science FITS file and create the output HDU list
        science_hdul = process_fits_by_color(science_file) #fits.open(science_file)

        if not bias:
            bias_hdul = process_fits_by_color(os.path.join(CALIB_DIR, 'combined_bias.fits'))
        else:
            try:
                bias_hdul = process_fits_by_color(bias)
            except Exception as e:
                logger.error(f"Error processing bias file {bias}: {e}")
                raise ValueError(f"Could not process bias file {bias}. Ensure it is a valid FITS file.")
            
        if len(science_hdul) != len(bias_hdul):
            logger.error(f"Science file has {len(science_hdul)} extensions while bias file has {len(bias_hdul)} extensions.")
            raise ValueError("Science and bias FITS files must have the same number of extensions. Please use compatible files.")

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['COMMENT'] = "Quick White Light Cube created from science file extensions."
        hdul = fits.HDUList([primary_hdu])

        blue_traces = []
        green_traces = []
        red_green = []
        
        blue_data = []
        green_data = []
        red_data = []

        blue_meta = []
        green_meta = []
        red_meta = []

        # Loop over each extension (skip primary) to process data
        for i, ext in enumerate(science_hdul[1:], start=1):
            bias = bias_hdul[i].data
            bias_data = np.median(bias[20:50])
            data = ext.data - bias_data
            
            
            if 'COLOR' in ext.header:
                header = ext.header
                color = header.get('COLOR', '').lower()
                bench = header.get('BENCH', '')
                side = header.get('SIDE', '')
                benchside = f'{bench}{side}'
            else:
                header = ext.header
                # Parse the CAM_NAME to determine color, bench, and side
                cam_name = header.get('CAM_NAME', '')
                if cam_name:
                    # Example format: '1A_Red' -> bench='1', side='A', color='red'
                    parts = cam_name.split('_')
                    if len(parts) >= 2:
                        benchside = parts[0]
                        color = parts[1].lower()  # Convert 'Red' to 'red'
                        if len(benchside) >= 2:
                            bench = benchside[0]
                            side = benchside[1]

            print(f'Processing extension {i}: {benchside} {color}')
            # Determine the corresponding trace file based on benchside and color
            #LLAMAS_master_blue_1_A_traces.pkl
            trace_filename = f"LLAMAS_master_{color}_{bench}_{side}_traces.pkl"
            trace_filepath = os.path.join(CALIB_DIR, trace_filename)
            if not os.path.exists(trace_filepath):
                logger.info(f"Trace file {trace_filepath} not found for {benchside} {color}. Skipping extension.")
                continue
            
            with open(trace_filepath, "rb") as f:
                trace_obj = pickle.load(f)
            
            # Build trace and data lists for QuickWhiteLight processing
            
            
            metadata = {'channel': color, 'bench': bench, 'side': side}
            if color == 'blue':
                blue_traces.append(trace_obj)
                blue_data.append(data)
                blue_meta.append(metadata)
            elif color == 'green':
                green_traces.append(trace_obj)
                green_data.append(data)
                green_meta.append(metadata)
            elif color == 'red':
                red_green.append(trace_obj)
                red_data.append(data)
                red_meta.append(metadata)
            
            # After processing all science_hdul extensions, generate white light images for each color

        whitelight_results = {}
        for col, traces_list, data_list, meta_list in [
            ('blue', blue_traces, blue_data, blue_meta),
            ('green', green_traces, green_data, green_meta),
            ('red', red_green, red_data, red_meta)
        ]:
            if traces_list and data_list:
                wl, xdata, ydata, flux = QuickWhiteLight(traces_list, data_list, meta_list, ds9plot=ds9plot)
                whitelight_results[col] = (wl, xdata, ydata, flux)
            else:
                logger.info(f"No data found for {col} color.")
                whitelight_results[col] = (None, None, None, None)

        for color in ['blue', 'green', 'red']:
            wl, xdata, ydata, flux = whitelight_results[color]
            if wl is None:
                continue
            # Create an image HDU for the white light image
            image_hdu = fits.ImageHDU(data=wl.astype(np.float32), name=color.upper())
            hdul.append(image_hdu)
            
            # Create a binary table HDU with the fiber x, y positions and flux data
            tab_hdu = fits.BinTableHDU.from_columns([
                fits.Column(name='XDATA', format='E', array=np.array(xdata, dtype=np.float32)),
                fits.Column(name='YDATA', format='E', array=np.array(ydata, dtype=np.float32)),
                fits.Column(name='FLUX',  format='E', array=np.array(flux, dtype=np.float32))
            ], name=f'{color.upper()}_TAB')
            hdul.append(tab_hdu)
         
        
        science_hdul.close()

        
        # Determine output file name
        if outfile is None:
            filename = os.path.basename(science_file)
            name, ext = os.path.splitext(filename)
            white_light_file = f"{name}_quickwhitelight.fits"
        else:
            white_light_file = outfile
            
        
        # Write the FITS file to disk.
        outpath = os.path.join(OUTPUT_DIR, white_light_file)
        hdul.writeto(outpath, overwrite=True)
        
        print(f'Quick white light cube saved to {outpath}')
        return outpath


def WhiteLightHex(extraction_file, ds9plot=False, median=False, mask=None, 
                 zscale=True, scale_min=None, scale_max=None, colorbar=True, 
                 colormap='viridis', fig=None, ax=None, **kwargs):
    """
    Create a hexagonal grid white light image without interpolation between fibers.
    Each fiber is represented as a discrete hexagon with its measured value.
    
    Parameters
    ----------
    extraction_list : list
        List of ExtractLlamas objects
    metadata : list, optional
        Metadata for each extraction object, by default None
    ds9plot : bool, optional
        If True, display the image with DS9, by default False
    median : bool, optional
        If True, use median instead of mean for combining extractions, by default False
    mask : ndarray, optional
        Mask to apply to the data, by default None
    zscale : bool, optional
        If True, use zscale for display, by default True
    scale_min : float, optional
        Minimum value for display scaling, by default None
    scale_max : float, optional
        Maximum value for display scaling, by default None
    colorbar : bool, optional
        If True, display colorbar, by default True
    colormap : str, optional
        Colormap to use, by default 'viridis'
    fig : matplotlib.figure.Figure, optional
        Figure to plot on, by default None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
        
    Returns
    -------
    ndarray
        2D hexagonal grid image
    """


    
    # Initialize data structures
    xdata = np.array([])
    ydata = np.array([])
    flux = np.array([])
    bench_sides = np.array([])


    if isinstance(extraction_file, str):
        print(f'Type is str-> loading file {extraction_file}')
        extraction, _ = ExtractLlamas.loadExtraction(extraction_file)
        logger.info(f'Loaded extraction object from file: {extraction_file}')
    elif isinstance(extraction_obj, ExtractLlamas):
        print(f'Type is ExtractLlamas-> using object')
        extraction = extraction_obj
    else:
        raise TypeError(f"Unexpected type: {type(extraction_obj)}. Must be string or ExtractLlamas object")
    

    extract_obj = ExtractLlamas.loadExtraction(extraction_file)
    extraction_list = extract_obj['extractions']
    metadata = extract_obj['metadata'] if 'metadata' in extract_obj else None
    
    # Process extraction list
    for i, extraction_obj in enumerate(extraction_list):
        meta = metadata[i] if metadata else None
        
        channel = extraction_obj.channel
        side = extraction_obj.side
        counts = extraction_obj.counts
        
        nfib, naxis1 = np.shape(counts)
        
        for ifib in range(nfib):
            benchside = f'{extraction_obj.bench}{extraction_obj.side}'
            
            try:
                x, y = FiberMap_LUT(benchside, ifib)
                if x == -1 and y == -1:
                    continue  # Skip if fiber mapping not found
            except Exception as e:
                logger.info(f'Fiber {ifib} not found in fiber map for bench {benchside}')
                logger.error(traceback.format_exc())
                continue
            
            # Get fiber value
            thisflux = np.nansum(counts[ifib])
            if mask is not None and len(mask) > 0:
                thisflux = thisflux * (1 - mask[ifib])
                
            # Store data
            flux = np.append(flux, thisflux)
            xdata = np.append(xdata, x)
            ydata = np.append(ydata, y)
            bench_sides = np.append(bench_sides, benchside)
    
    # Create figure if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine colormap scaling
    if zscale:
        from astropy.visualization import ZScaleInterval
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(flux)
    else:
        vmin = scale_min if scale_min is not None else np.nanmin(flux)
        vmax = scale_max if scale_max is not None else np.nanmax(flux)
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(colormap)
    
    # Compute hexagon size based on fiber spacing (using median distance between adjacent fibers)
    x_sorted = np.sort(np.unique(xdata))
    if len(x_sorted) > 1:
        x_diffs = np.diff(x_sorted)
        hex_size = np.nanmedian(x_diffs) / 1.5  # Adjust to prevent overlap
    else:
        hex_size = 0.5  # Default if we can't compute
    
    # Create a grid to store hexagonal values for DS9 display
    x_range = (np.max(xdata) - np.min(xdata)) + 2*hex_size
    y_range = (np.max(ydata) - np.min(ydata)) + 2*hex_size
    x_min, y_min = np.min(xdata) - hex_size, np.min(ydata) - hex_size
    
    # Create grid with higher resolution for DS9
    grid_scale = 5  # Higher resolution for better hexagon approximation
    hex_grid = np.full(
        (int(y_range * grid_scale) + 1, int(x_range * grid_scale) + 1),
        np.nan
    )
    
    # Plot hexagons for each fiber
    for i in range(len(xdata)):
        x, y = xdata[i], ydata[i]
        value = flux[i]
        
        if np.isnan(value):
            continue
            
        color = cmap(norm(value))
        
        # Create hexagon patch for matplotlib
        hex_patch = RegularPolygon(
            (x, y), 
            numVertices=6, 
            radius=hex_size,
            orientation=np.pi/6,  # 30 degrees rotation
            facecolor=color, 
            edgecolor='black', 
            linewidth=0.5,
            alpha=1.0
        )
        ax.add_patch(hex_patch)
        
        # Fill corresponding area in hex_grid for DS9
        # Convert hexagon vertices to grid coordinates
        for phi in np.linspace(0, 2*np.pi, 60):  # 60 points around hexagon
            hx = x + hex_size * np.cos(phi)
            hy = y + hex_size * np.sin(phi)
            
            # Convert to grid indices
            ix = int((hx - x_min) * grid_scale)
            iy = int((hy - y_min) * grid_scale)
            
            # Check bounds and set value
            if (0 <= ix < hex_grid.shape[1] and 0 <= iy < hex_grid.shape[0]):
                hex_grid[iy, ix] = value
    
    # Fill in the interior of hexagons in grid (simple flood fill)
    from scipy import ndimage
    # Create a binary mask of valid points
    mask = ~np.isnan(hex_grid)
    # Label connected regions
    labels, num = ndimage.label(mask)
    # Fill holes in each labeled region
    for i in range(1, num+1):
        region = labels == i
        values = hex_grid[region]
        if len(values) > 0:
            median_value = np.nanmedian(values)
            # Fill entire region with median value
            hex_grid[region] = median_value
    
    # Set axis limits
    ax.set_xlim(np.min(xdata) - 2*hex_size, np.max(xdata) + 2*hex_size)
    ax.set_ylim(np.min(ydata) - 2*hex_size, np.max(ydata) + 2*hex_size)
    ax.set_aspect('equal')
    
    # Add colorbar if requested
    if colorbar:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Flux')
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Hexagonal Fiber Grid (No Interpolation)')
    
    # Show the plot
    if ds9plot:
        # Display in DS9
        try:
            from llamas_pyjamas.QA import plot_ds9
            plot_ds9(hex_grid, samp=True)
        except Exception as e:
            logger.error(f"Error displaying in DS9: {e}")
            plt.tight_layout()
            plt.show()
    else:
        plt.tight_layout()
        plt.show()
    
    return hex_grid




def plot_spaxelmap(outpath: str)-> None:
    """
    Plots the fiber map for the LLAMAS IFU in arcseconds, showing the hexagonal spaxel layout.
    
    This function generates a plot with hexagonal patches representing each fiber position for 
    different configurations (1A, 2A, 3A, 4A, 1B, 2B, 3B, 4B). The positions are scaled by the 
    spatial pitch of 0.75" to create a spaxel map with physical units. It also includes directional 
    annotations (N for North, E for East) and saves the plot as an image file.
    
    Color coding:
    - Fibers in configuration 1A are plotted in black
    - Fibers in configuration 3A are plotted in red
    - Fibers in configuration 4B are plotted in purple
    - Fibers in configuration 2B are plotted in orange
    - Fibers in configuration 2A are plotted in blue
    - Fibers in configuration 4A are plotted in green
    - Fibers in configuration 3B are plotted in cyan
    - Fibers in configuration 1B are plotted in magenta
    
    The plot is saved as a high-resolution image in the specified path.
    
    Returns:
        None
    """
    import matplotlib.patches as mpatches
    
    # Spatial pitch in arcseconds
    SPAXEL_PITCH = 0.75
    
    # Size of hexagon (radius to vertex)
    HEX_SIZE = 0.4 * SPAXEL_PITCH
    
    # Configuration colors
    colors = {
        '1A': 'black',
        '3A': 'red',
        '4B': 'purple',
        '2B': 'orange',
        '2A': 'blue',
        '4A': 'green',
        '3B': 'cyan',
        '1B': 'magenta'
    }
    
    # 1A - N=298
    # 2A - N=300
    # 3A - all fibers fine (N=298)
    # 4A - all fibers fine (N=300)
    # 4B - all fibers fine (N=298)
    # 3B - all fibers fine (N=300)
    # 2B - fiber 49 (zero index) is broken / dead (N=297 good + 1 dead)
    # 1B - all fibers fine (N=300)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    fibernum_a = np.arange(298)
    fibernum_b = np.arange(300)
    
    # Create legend handles
    legend_handles = []
    
    # Plot fibers for configurations with 298 fibers
    for config in ['1A', '3A', '4B', '2B']:
        for fiber in fibernum_a:
            x, y = FiberMap_LUT(config, int(fiber))
            # Convert to arcseconds
            x_arcsec, y_arcsec = x * SPAXEL_PITCH, y * SPAXEL_PITCH
            
            # Create and add hexagon patch
            hex_patch = mpatches.RegularPolygon(
                (x_arcsec, y_arcsec),  # center coordinates
                numVertices=6,         # hexagon
                radius=HEX_SIZE,       # size
                orientation=0,         # flat top
                facecolor='none',      # transparent fill
                edgecolor=colors[config],
                linewidth=0.5,
                alpha=0.7
            )
            ax.add_patch(hex_patch)
        
        # Add to legend
        legend_handles.append(mpatches.Patch(color=colors[config], label=config))
    
    # Plot fibers for configurations with 300 fibers
    for config in ['2A', '4A', '3B', '1B']:
        for fiber in fibernum_b:
            x, y = FiberMap_LUT(config, int(fiber))
            # Convert to arcseconds
            x_arcsec, y_arcsec = x * SPAXEL_PITCH, y * SPAXEL_PITCH
            
            # Create and add hexagon patch
            hex_patch = mpatches.RegularPolygon(
                (x_arcsec, y_arcsec),  # center coordinates
                numVertices=6,         # hexagon
                radius=HEX_SIZE,       # size
                orientation=0,         # flat top
                facecolor='none',      # transparent fill
                edgecolor=colors[config],
                linewidth=0.5,
                alpha=0.7
            )
            ax.add_patch(hex_patch)
        
        # Add to legend
        legend_handles.append(mpatches.Patch(color=colors[config], label=config))
    
    # Adjust axis limits to reflect the new scale
    ax.set_xlim(-2 * SPAXEL_PITCH, 75 * SPAXEL_PITCH)
    ax.set_ylim(-2 * SPAXEL_PITCH, 47 * SPAXEL_PITCH)
    ax.set_aspect('equal')
    
    # Add axis labels
    ax.set_xlabel('Arcseconds')
    ax.set_ylabel('Arcseconds')
    
    # Add legend
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.7)
    
    # Extension labels - scale positions by spatial pitch
    fs = 8
    ax.text(52.25 * SPAXEL_PITCH, 40.5 * SPAXEL_PITCH, "R-1", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 39.5 * SPAXEL_PITCH, "G-2", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 38.5 * SPAXEL_PITCH, "B-3", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 35.5 * SPAXEL_PITCH, "R-7", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 34.5 * SPAXEL_PITCH, "G-8", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 33.5 * SPAXEL_PITCH, "B-9", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 30.5 * SPAXEL_PITCH, "R-13", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 29.5 * SPAXEL_PITCH, "G-14", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 28.5 * SPAXEL_PITCH, "B-15", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 25.5 * SPAXEL_PITCH, "R-19", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 24.5 * SPAXEL_PITCH, "G-20", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 23.5 * SPAXEL_PITCH, "B-21", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 3.5 * SPAXEL_PITCH, "R-4", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 2.5 * SPAXEL_PITCH, "G-5", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 1.5 * SPAXEL_PITCH, "B-6", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 8.5 * SPAXEL_PITCH, "R-10", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 7.5 * SPAXEL_PITCH, "G-11", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 6.5 * SPAXEL_PITCH, "B-12", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 13.5 * SPAXEL_PITCH, "R-16", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 12.5 * SPAXEL_PITCH, "G-17", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 11.5 * SPAXEL_PITCH, "B-18", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 18.5 * SPAXEL_PITCH, "R-22", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 17.5 * SPAXEL_PITCH, "G-23", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 16.5 * SPAXEL_PITCH, "B-24", fontsize=fs)

    # Direction arrows - scale positions by spatial pitch
    ax.annotate('', xy=(73 * SPAXEL_PITCH, 20 * SPAXEL_PITCH), 
                xytext=(60 * SPAXEL_PITCH, 20 * SPAXEL_PITCH),
                arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))

    ax.annotate('', xy=(60 * SPAXEL_PITCH, 25 * SPAXEL_PITCH), 
                xytext=(60 * SPAXEL_PITCH, 20 * SPAXEL_PITCH),
                arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))
    
    ax.text(71 * SPAXEL_PITCH, 15.5 * SPAXEL_PITCH, "N", fontsize=12)
    ax.text(60 * SPAXEL_PITCH, 27 * SPAXEL_PITCH, "E", fontsize=12)

    # Update title to reflect physical units
    plt.title("LLAMAS IFU Hexagonal Spaxel Map (0.75\"/spaxel)")

    plt.tight_layout()
    fig.savefig(outpath, dpi=600)
    plt.show()
