import llamas_pyjamas.Arc.arcLlamas as arc
import llamas_pyjamas.Extract.extractLlamas as extract
import os
from   llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR, LUT_DIR
import matplotlib.pyplot as plt
import numpy as np
from pypeit.core.fitting import iterfit

def calcThroughput(std_dict, arcdict):

    std_extract = std_dict['extractions']
    std_metadata = std_dict['metadata']
    std_wvcal = arc.arcTransfer(std_dict, arcdict)

    # Find the brightest fibers in the std star spectrum

    extension = np.array([])
    fiber = np.array([])
    counts = np.array([])

    for i in range(len(std_extract)):
        if (std_metadata[i]['channel'] == 'red'):
            for thisfiber in range(std_metadata[i]['nfibers']):
                fiber = np.append(fiber, thisfiber)
                extension = np.append(extension, i)
                keep = np.where(std_extract[i].wave[thisfiber] > 3400)
                tt = np.sum(std_extract[i].counts[thisfiber, keep])
                counts = np.append(counts, tt)

    # Sort in descending order
    idx = np.argsort(-counts)

    extension = extension[idx].astype(int)
    fiber = fiber[idx].astype(int)
    counts = counts[idx]

    sky_fitx = np.array([])
    sky_fity = np.array([])

    for i in range(900,1100):
        sky_fitx = np.append(sky_fitx, std_wvcal['extractions'][extension[i]].xshift[fiber[i],:])
        sky_fity = np.append(sky_fity, std_wvcal['extractions'][extension[i]].counts[fiber[i],:])#  / std_wvcal['extractions'][extension[i]].relative_throughput[fiber[i]])

    idx = np.argsort(sky_fitx)
    sky_fitx = sky_fitx[idx]
    sky_fity = sky_fity[idx]

    plt.plot(sky_fitx, sky_fity, '.', markersize=0.5, label='std', color='r')
    plt.show()

    print(sky_fity)

    sset, outmask = iterfit(sky_fitx, sky_fity, maxiter=6, kwargs_bspline={'bkspace':0.5})

    for i in range(0,15):
        fiberflux = std_wvcal['extractions'][extension[i]].counts[fiber[i],:] / std_wvcal['extractions'][extension[i]].relative_throughput[fiber[i]]
        skymodel = sset(std_wvcal['extractions'][extension[i]].wave[fiber[i],:])
        #skybg = std_wvcal['extractions'][extension[i]].counts[fiber[i+1000],:] / std_wvcal['extractions'][extension[i]].relative_throughput[fiber[i]]
        if (i == 0):
            apsum = fiberflux - skymodel
        else:
            apsum += fiberflux - skymodel
         
    plt.plot(std_wvcal['extractions'][extension[0]].wave[fiber[0],:], apsum, label='std', color='r')
    plt.show()

    return(sky_fitx)