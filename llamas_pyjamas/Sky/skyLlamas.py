import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.Extract.extractLlamas import save_extractions
import llamas_pyjamas.Arc.arcLlamas as arc
from llamas_pyjamas.QA import plot_ds9
from llamas_pyjamas.config import OUTPUT_DIR, CALIB_DIR
from pypeit.core.fitting import iterfit
from astropy.io import fits
from astropy.table import Table
import os

def skyModel_1d(science_extraction_file, color, sky_extraction_file=None):
    """
    Create a 1D sky model from the sky extraction.

    Parameters:
    science_extraction (ExtractLlamas): Extracted science data.
    sky_extraction (ExtractLlamas): Extracted data from a blank sky reference field

    Note that as a default, the code assumes that the user estimates the sky from 
    the science field itself. If a separate sky field is provided, it will be used instead.

    Either way, the user must generate a mask to exclude sources from the sky estimation.

    Returns:
    sky_model (np.ndarray): 1D sky model.
    """

    if sky_extraction_file is None:
        sky_extraction_file = science_extraction_file 

    print("Loading science extraction from ", science_extraction_file)
    science_dict = ExtractLlamas.loadExtraction(science_extraction_file)
    science = science_dict['extractions']
    science_metadata = science_dict['metadata']
    hdr = science_dict['primary_header']

    print("Loading sky extraction and arc from ", sky_extraction_file)
    if (sky_extraction_file == science_extraction_file):
        sky_dict = science_dict
    else:
        sky_dict = ExtractLlamas.loadExtraction(sky_extraction_file)

    arc_dict = ExtractLlamas.loadExtraction(os.path.join(OUTPUT_DIR, 'LLAMAS_reference_arc.pkl'))
    sky_wvcal = arc.arcTransfer(sky_dict, arc_dict)
    
    sky = sky_wvcal['extractions']
    sky_metadata = sky_wvcal['metadata']

    # Find the brightest fibers in the std star spectrum



    ########## CORE ROUTINE ##########################

    if (color == None):
        allcolors = ['red','green','blue']
    else:
        allcolors = [color]

    for color in allcolors:
    
        extension = np.array([])
        fiber = np.array([])
        counts = np.array([])

        print("Generating sky model for color ", color)

        for i in range(len(sky)):
            if (sky_metadata[i]['channel'] == color):
                for thisfiber in range(sky_metadata[i]['nfibers']):
                    fiber = np.append(fiber, thisfiber)
                    extension = np.append(extension, i)
                    tt = np.sum(sky[i].counts[thisfiber]/sky[i].relative_throughput[thisfiber])
                    counts = np.append(counts, tt)

        # Sort in descending order
        idx = np.argsort(-counts)

        extension = extension[idx].astype(int)
        fiber = fiber[idx].astype(int)
        counts = counts[idx]

        sky_fitx = np.array([])
        sky_fity = np.array([])

        # Create a fitting array.  Right now, we are using fibers 900-1800 from 0-2400 sorted by brightness
        for i in range(1500,1800):
            sky_fitx = np.append(sky_fitx, sky[extension[i]].xshift[fiber[i],:])
            sky_fity = np.append(sky_fity, sky[extension[i]].counts[fiber[i],:])
            

        # Re-sort in order of increasing wavelength (becuase the append operation above mixes the wave order)
        idx = np.argsort(sky_fitx)
        sky_fitx = sky_fitx[idx]
        sky_fity = sky_fity[idx]

        # Filter out bad pixels before the fit
        gd = (~np.isnan(sky_fity))
        sky_fitx = sky_fitx[gd]
        sky_fity = sky_fity[gd]


        print("Fitting sky with ", len(sky_fitx), " points")
        sset, outmask = iterfit(sky_fitx, sky_fity, maxiter=6, kwargs_bspline={'bkspace':0.5})

        plt.plot(sky_fitx, sky_fity, '.', markersize=0.1, label='data', color='k')
        y = sset.value(sky_fitx)[0]
        plt.plot(sky_fitx, y, color='r')
        plt.ylim(0,100)
        plt.show()

        # Now apply the sky model to all science fibers
        print("Applying sky model to science fibers")
        for i in range(len(counts)):
            skymodel = sset.value(sky[extension[i]].xshift[fiber[i],:])[0]
            scale = np.nanmedian(sky[extension[i]].counts[fiber[i],:])/np.nanmedian(skymodel)
            skymodel = skymodel * scale    
            science[extension[i]].sky[fiber[i],:] = skymodel

    #########################################


    # Plot some QA
    try:
        for i in range(1600,1605):
            plt.plot(sky[extension[i]].wave[fiber[i],:], sky[extension[i]].counts[fiber[i],:], '.', markersize=0.5, label='data', color='k')
            plt.plot(sky[extension[i]].wave[fiber[i],:], science[extension[i]].sky[fiber[i],:], color='r')
            plt.ylim(0, np.nanmax(sky[extension[i]].counts[fiber[i],:])*1.2)   
            plt.title('Ext '+str(extension[i])+' Fiber '+str(fiber[i]))
            plt.xlabel('Wavelength (Angstroms)')
            plt.ylabel('Counts')
            plt.show()  

            """ 
                skymodel = sset.value(sky_fitx)[0]
                #plt.plot(sky_fitx, sky_fity-skymodel, '.', markersize=0.1, label='std', color='k')
                #plt.plot(sky_fitx, np.sqrt(skymodel + (2.5**2)/1.2), color='r')
                #plt.plot(sky_fitx, -1.0 * np.sqrt(skymodel + (2.5**2)/1.2), color='r') 

                plt.plot(sky_fitx, sky_fity, '.', markersize=0.1, label='std', color='k')
                plt.plot(sky_fitx, skymodel, color='r')

                plt.show()
            """
    except: 
        print("Error in plotting")

    # Save the extraction object with sky attribute populated back out to disk

    outputfile = science_extraction_file.replace('_extractions.pkl','_sky1d_extractions.pkl')
    print("Saving science extraction with sky model to ", outputfile)
    save_extractions(science, primary_header=hdr, savefile=outputfile)
