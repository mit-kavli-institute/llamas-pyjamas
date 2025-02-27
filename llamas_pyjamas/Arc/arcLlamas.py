from   astropy.io import fits
import scipy
import numpy as np
import llamas_pyjamas.Extract.extractLlamas as extract
from   matplotlib import pyplot as plt
# from   pypeit.core.arc import detect_peaks,iter_continuum
# from   pypeit.core import pydl
# from   pypeit.par import pypeitpar
# from   pypeit.core.wavecal import autoid

from pypeit.core.wavecal.wvutils import xcorr_shift_stretch
import warnings

###############################################################################3

def shiftArcX(arc_extraction_pickle):

    warnings.filterwarnings("ignore", category=UserWarning, module="astropy.stats.sigma_clipping")
    arcdict = extract.load_extractions(arc_extraction_pickle)
    arcspec = arcdict['extraction']
    metadata = arcdict['metadata']

    # We will use fiber #150 in spectrograph 4A as the reference (near the center of the IFU)
    # This corresponds to extension 18 (red), 19 (green) and 20 (blue) in the arc extraction object

    # Red first
    red_refspec = arcspec[18].counts[150]

    fits_ext = 18
    x = np.arange(2048)

    for ifiber in range(0,metadata[fits_ext]['nfibers']):
    # for ifiber in range(0,5):
        print(f"Shifting fiber number {ifiber} in extension {metadata[fits_ext]['bench']}{metadata[fits_ext]['side']}")
        func = 'quadratic'
        success, shift, stretch, stretch2, _, _, _ = \
            xcorr_shift_stretch(red_refspec, arcspec[fits_ext].counts[ifiber], stretch_func=func)
        
        if (success == 1):
            arcspec[fits_ext].xshift[ifiber,:] = (x*stretch+x**2*stretch2)+shift
        else:
            print("....Warning: arc shift failed for this fiber!")
            arcspec[fits_ext].xshift[ifiber,:] = x

    # Re-save with new information populated
    sv = arc_extraction_pickle.replace('.pkl','_shifted.pkl')
    extract.save_extractions(arcspec, savefile=sv)

def fiberRelativeThroughput(arc_extraction_pickle):

    arcspec, metadata = extract.load_extractions(arc_extraction_pickle)
    fits_ext = 18

    for ifiber in range(0,metadata[fits_ext]['nfibers']):   
        trace = arcspec[fits_ext].trace
        
