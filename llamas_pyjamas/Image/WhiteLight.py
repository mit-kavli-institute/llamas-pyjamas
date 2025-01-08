import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.QA import plot_ds9
from llamas_pyjamas.config import OUTPUT_DIR
from astropy.io import fits
from astropy.table import Table
import os
from matplotlib.tri import Triangulation, LinearTriInterpolator
from llamas_pyjamas.Utils.utils import setup_logger
from datetime import datetime
import traceback
from llamas_pyjamas.config import LUT_DIR

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, f'WhiteLight_{timestamp}.log')

orig_fibre_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LLAMAS_FiberMap_revA.dat')
fibre_map_path = os.path.join(LUT_DIR, 'LLAMAS_FiberMap_rev02.dat')
print(f'Fibre map path: {fibre_map_path}')
fibermap_lut = Table.read(fibre_map_path, format='ascii.fixed_width')


def color_isolation(extractions):
    """A function that takes in a list of extraction objects and isolates the blue, green, and red channels

    Args:
        extractions (list): A list of extraction objects loaded from ExtractLlamas
    """
    blue_extractions = [ext for ext in extractions if ext.channel.lower() == 'blue']
    green_extractions = [ext for ext in extractions if ext.channel.lower() == 'green']
    red_extractions = [ext for ext in extractions if ext.channel.lower() == 'red']
    
    return blue_extractions, green_extractions, red_extractions


def WhiteLightFits(extraction_array, outfile=None):
    
    blue, green, red = color_isolation(extraction_array)
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
        
        blue_whitelight, blue_x, blue_y, blue_flux = WhiteLight(blue, ds9plot=False)
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
      
        green_whitelight, green_x, green_y, green_flux = WhiteLight(green, ds9plot=False)
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
        red_whitelight, red_x, red_y, red_flux = WhiteLight(red, ds9plot=False)
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
    else:
        white_light_file = outfile
    
    print(f'Writing white light file to {white_light_file}')
    # Write to file
    hdul.writeto(os.path.join(OUTPUT_DIR, white_light_file), overwrite=True)
    
    return white_light_file


def WhiteLight(extraction_array, ds9plot=True):
    
    assert type(extraction_array) == list, 'Extraction array must be a list of extraction files'
    
    xdata = np.array([])
    ydata = np.array([])
    flux  = np.array([])
    
    for extraction_obj in extraction_array:
        if isinstance(extraction_obj, str):
            extraction = ExtractLlamas.loadExtraction(extraction_obj)
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
            thisflux = np.nansum(extraction.counts[ifib])
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

    x_grid, y_grid = np.meshgrid(xx, yy)
    
    whitelight = flux_interpolator(x_grid, y_grid)
    whitelight = np.fliplr(whitelight)
    if (ds9plot):
        #ds9 = pyds9.DS9(target='DS9:*', start=True, wait=10, verify=True)
        #ds9.set_np2arr(whitelight)
        plot_ds9(whitelight)

    return whitelight, xdata, ydata, flux

def WhiteLightQuickLook(tracefile, data):
        
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

   
def FiberMap(bench, infiber):

    n_right    = 23
    n_left     = 23
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
        # sides of the IFU. Even rows go right to left, off rows go left to right

        x_index = np.floor_divide(((fiber+wrap) % (n_left+n_right)),2)

        if ((y_rownum % 2) == 0):
            if ((fiber % 2) == 0):
                x_fiber = (n_left+n_right) - x_index
            else:
                x_fiber = n_left - x_index
        else:
            if ((fiber % 2) == 0):
                x_fiber = n_left + 0.5 + x_index
            else:
                x_fiber = 0.5 + x_index

        x_value = x_fiber - 0.5
        y_value = n_vertical*dy-y_value

    elif ('B' in bench):

        ncols = n_left + n_right
        y_rownum = np.floor_divide((fiber+wrap), ncols)
        y_value  = (y0 + y_rownum) * dy

        # Account for the fact that the fibers snake back and forth within the two
        # sides of the IFU. Even rows go right to left, odd rows go left to right

        x_index = np.floor_divide(((fiber+wrap) % (n_left+n_right)),2)

        if ((y_rownum % 2) == 0):
            if ((fiber % 2) == 0):
                x_fiber = (n_left+n_right) - x_index
            else:
                x_fiber = n_left - x_index
            x_fiber -= 0.5
            x_value = x_fiber - 0.5

            if (bench=='2B'):
                x_fiber+=0.5
            
        else:
            if ((fiber % 2) == 0):
                x_fiber = n_left + 0.5 + x_index
            else:
                x_fiber = 0.5 + x_index
            x_fiber += 0.5

            if (bench=='2B'):
                x_fiber-=0.5
            

        y_value = n_vertical*dy-y_value
        
    # return(x_value,y_value)
    return(x_fiber,n_vertical-int(y0+y_rownum))

def FiberMap_LUT(bench, fiber):

    #if (np.logical_and(bench == '2B',fiber >= 49)):
    #    fiber += 1
    
    fiber_row = fibermap_lut[np.logical_and(fibermap_lut['bench']==bench, \
                                            fibermap_lut['fiber']==fiber)]
    #breakpoint()
    try:
        return(fiber_row['xpos'][0],fiber_row['ypos'][0])
    except:
        return(-1,-1)

def plot_fibermap():

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

    for fiber in fibernum_a:
        x, y = FiberMap_LUT('1A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=2, horizontalalignment='center',verticalalignment='center', color='k')

        x, y = FiberMap_LUT('3A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=2, horizontalalignment='center',verticalalignment='center', color='r')

        x, y = FiberMap_LUT('4B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=2, horizontalalignment='center',verticalalignment='center', color='k')

        x, y = FiberMap_LUT('2B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=2, horizontalalignment='center',verticalalignment='center', color='r')
        
        
    for fiber in fibernum_b:
        x, y = FiberMap_LUT('2A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=2, horizontalalignment='center',verticalalignment='center', color='b')

        x, y = FiberMap_LUT('4A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=2, horizontalalignment='center',verticalalignment='center', color='g')

        x, y = FiberMap_LUT('3B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=2, horizontalalignment='center',verticalalignment='center', color='b')

        x, y = FiberMap_LUT('1B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=2, horizontalalignment='center',verticalalignment='center', color='g')
        
        
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

    ax.text(52.25, 40.5, "R-1", fontsize=7)
    ax.text(52.25, 39.5, "G-2", fontsize=7)
    ax.text(52.25, 38.5, "B-3", fontsize=7)

    ax.text(52.25, 35.5, "R-7", fontsize=7)
    ax.text(52.25, 34.5, "G-8", fontsize=7)
    ax.text(52.25, 33.5, "B-9", fontsize=7)

    ax.text(52.25, 30.5, "R-13", fontsize=7)
    ax.text(52.25, 29.5, "G-14", fontsize=7)
    ax.text(52.25, 28.5, "B-15", fontsize=7)

    ax.text(52.25, 25.5, "R-19", fontsize=7)
    ax.text(52.25, 24.5, "G-20", fontsize=7)
    ax.text(52.25, 23.5, "B-21", fontsize=7)

    ax.text(52.25, 3.5, "R-4", fontsize=7)
    ax.text(52.25, 2.5, "G-5", fontsize=7)
    ax.text(52.25, 1.5, "B-6", fontsize=7)

    ax.text(52.25, 8.5, "R-10", fontsize=7)
    ax.text(52.25, 7.5, "G-11", fontsize=7)
    ax.text(52.25, 6.5, "B-12", fontsize=7)

    ax.text(52.25, 13.5, "R-16", fontsize=7)
    ax.text(52.25, 12.5, "G-17", fontsize=7)
    ax.text(52.25, 11.5, "B-18", fontsize=7)

    ax.text(52.25, 18.5, "R-22", fontsize=7)
    ax.text(52.25, 17.5, "G-23", fontsize=7)
    ax.text(52.25, 16.5, "B-24", fontsize=7)

    ax.annotate('', xy=(73,20), xytext=(60,20),
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))

    ax.annotate('', xy=(60,25), xytext=(60,20),
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))
    ax.text(71, 15.5, "N", fontsize=12)
    ax.text(60, 27, "E", fontsize=12)

    plt.title("LLAMAS IFU Fiber to slit / FITS extension mapping (Rev 02)")

    plt.tight_layout()
    fig.savefig("/Users/simcoe/fiber.png", dpi=600)
    plt.show()


def fibermap_table():

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

    fiber_table.write('LLAMAS_FiberMap_rev02_updated.dat', format='ascii.fixed_width', overwrite=True)
        
    return(fiber_table)
    
def rerun():

    extraction1a = ('Extract_1A.pkl')
    extraction2a = ExtractLlamas.loadExtraction('Extract_2A.pkl')
    extraction3a = ExtractLlamas.loadExtraction('Extract_3A.pkl')
    extraction4a = ExtractLlamas.loadExtraction('Extract_4A.pkl')

    extraction1b = ExtractLlamas.loadExtraction('Extract_1B.pkl')
    extraction2b = ExtractLlamas.loadExtraction('Extract_2B.pkl')
    extraction3b = ExtractLlamas.loadExtraction('Extract_3B.pkl')
    extraction4b = ExtractLlamas.loadExtraction('Extract_4B.pkl')
    

    WhiteLight([extraction1a, extraction2a, extraction3a, extraction4a, extraction1b, extraction2b, extraction3b, extraction4b])
