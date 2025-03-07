from   astropy.io import fits
import scipy
import numpy as np
import llamas_pyjamas.Extract.extractLlamas as extract
from   matplotlib import pyplot as plt
from   scipy.ndimage import generic_filter
from   scipy.signal import correlate
from   astropy.table import Table
import os
from   llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR
# from   pypeit.core.arc import detect_peaks,iter_continuum
# from   pypeit.core import pydl
# from   pypeit.par import pypeitpar
# from   pypeit.core.wavecal import autoid

from pypeit.core.wavecal.wvutils import xcorr_shift_stretch
from pypeit import bspline
import warnings

###############################################################################3

def reidentifyArc(shifted_arc, reference_linelist=os.path.join(CALIB_DIR,'llamas_ThAr_ref_arclines.fits')):

    arcdict = extract.ExtractLlamas.loadExtraction(shifted_arc)
    arcspec = arcdict['extractions']
    metadata = arcdict['metadata']

    fits_ext = 18

    if (metadata[fits_ext]['channel'] == 'red'):
        reference_arc = os.path.join(CALIB_DIR,'llamasr_ThAr_reference.fits')
        linelist = Table.read(reference_linelist, 3)

    refarc = interpolateNaNs(Table.read(reference_arc)['ThAr'])
    refarc -= np.nanmedian(refarc)
    thisarc = interpolateNaNs(arcspec[fits_ext].counts[150])
    thisarc -= np.nanmedian(thisarc)

    success, shift, stretch, stretch2, _, _, _ = \
        xcorr_shift_stretch(thisarc, refarc, stretch_func='linear')
        
#    for i in range(100,200):
#        plt.plot(arcspec[fits_ext].xshift[i],arcspec[fits_ext].counts[i],'.',markersize=1)

    
    xin = arcspec[fits_ext].xshift.flatten()
    yin = arcspec[fits_ext].counts.flatten()
    indices = np.argsort(xin)
    xin = xin[indices]
    yin = yin[indices]
    yin[np.where(np.isnan(yin))] = 0
    # yin = interpolateNaNs(yin)
    plt.plot(xin, yin, '.', markersize=1, color='c')
    print(np.isnan(yin).any())

    sset = bspline.bspline(xin, bkspace=1.2)
    res, yfit = sset.fit(xin, yin, np.ones(len(xin)))
    xmodel = (np.arange(4096)/2).astype(float)
    ymodel = sset.value(xmodel)[0]
    plt.plot(xmodel, ymodel, color='b')

    for line in linelist['xpos_fiber150']:
        # plt.plot([line+bestlag,line+bestlag],[0,np.max(refarc)],'-',color='red', alpha=0.5)
        newx = line*stretch+shift
        plt.plot([newx,newx],[0,np.max(refarc)],'-',color='red', alpha=0.5)
    plt.show()

    return(xin, yin)

def shiftArcX(arc_extraction_pickle):

    warnings.filterwarnings("ignore", category=UserWarning, module="astropy.stats.sigma_clipping")
    arcdict = extract.ExtractLlamas.loadExtraction(arc_extraction_pickle)
    arcspec = arcdict['extractions']
    metadata = arcdict['metadata']

    # We will use fiber #150 in spectrograph 4A as the reference (near the center of the IFU)
    # This corresponds to extension 18 (red), 19 (green) and 20 (blue) in the arc extraction object

    # Red first
    red_refspec = interpolateNaNs(arcspec[18].counts[150])

    fits_ext = 18
    x = np.arange(2048)

    for ifiber in range(0,metadata[fits_ext]['nfibers']):
    # for ifiber in range(0,5):
        print(f"Shifting fiber number {ifiber} in extension {metadata[fits_ext]['bench']}{metadata[fits_ext]['side']}")
        func = 'quadratic'
        success, shift, stretch, stretch2, _, _, _ = \
            xcorr_shift_stretch(red_refspec, interpolateNaNs(arcspec[fits_ext].counts[ifiber]), \
                                stretch_func=func)
        
        if (success == 1):
            arcspec[fits_ext].xshift[ifiber,:] = (x*stretch+x**2*stretch2)+shift
        else:
            print("....Warning: arc shift failed for this fiber!")
            arcspec[fits_ext].xshift[ifiber,:] = x

    # Re-save with new information populated
    sv = arc_extraction_pickle.replace('.pkl','_shifted.pkl')
    extract.save_extractions(arcspec, savefile=sv)

def nan_median_filter(input_spectrum):
    values = input_spectrum[~np.isnan(input_spectrum)]  # Remove NaNs
    return np.median(values) if values.size else np.nan

def interpolateNaNs(input_spectrum):
    array_filtered = generic_filter(input_spectrum, nan_median_filter, size=3)
    return(array_filtered)

def fiberRelativeThroughput(flat_extraction_pickle, arc_extraction_pickle):

    flatdict = extract.ExtractLlamas.loadExtraction(flat_extraction_pickle)
    flatspec = flatdict['extractions']
    metadata = flatdict['metadata']

    arcdict = extract.ExtractLlamas.loadExtraction(arc_extraction_pickle)
    arcspec = arcdict['extractions']

    fits_ext = 18
    reference_fiber = 150

    refspec = flatspec[fits_ext].counts[reference_fiber]
    gd = (arcspec[fits_ext].xshift[reference_fiber] > 100) & (arcspec[fits_ext].xshift[reference_fiber] < 2048-100)
    reference_flux = np.nansum(refspec[gd])
    print(reference_flux)

    for ifiber in range(0,metadata[fits_ext]['nfibers']):   
        spec = flatspec[fits_ext].counts[ifiber]
        gd = (arcspec[fits_ext].xshift[ifiber] > 100) & (arcspec[fits_ext].xshift[ifiber] < 2048-100)
        flux = np.nansum(spec[gd])
        arcspec[fits_ext].relative_throughput[ifiber] = flux/reference_flux
        print(f'Throughput = {flux/reference_flux}')

    plt.hist(arcspec[fits_ext].relative_throughput, cumulative=True, density=True, bins=25)
    plt.show()

    sv = arc_extraction_pickle.replace('.pkl','_shifted_tp.pkl')
    extract.save_extractions(arcspec, savefile=sv)