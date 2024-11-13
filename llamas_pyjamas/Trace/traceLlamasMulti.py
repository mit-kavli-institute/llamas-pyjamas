
import os
import sys
from   astropy.io import fits
import scipy
import numpy as np
import time
from   matplotlib import pyplot as plt
import traceback
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl
from pypeit.core.fitting import iterfit
from   pypeit.core import fitting
from pypeit.bspline.bspline import bspline
import pickle

import ray
from typing import List, Set, Dict, Tuple, Optional

#Initalising ray for multiprocessing
ray.init()



class TraceLlamas:
    
    def __init__(self, 
                 flat_fitsfile: str, 
                 spectrograph: Optional[str] = None,
                 channel: Optional[str] = None,
                 mph: Optional[int] = None):
        
        self.flat_fitsfile = flat_fitsfile
        self.spectrograph = spectrograph
        self.channel = channel
        self.mph = mph
        
        self.xmin     = 200
        self.fitspace = 10
        
            
    def generate_valleys(self, tslice: float) -> Tuple[np.ndarray, ...]:

        tmp            = detect_peaks(tslice,mpd=2,threshold=10,show=False,valley=True)
        valley_indices = np.ndarray(len(tmp))
        valley_depths  = np.ndarray(len(tmp))
        valley_indices = tmp.astype(float)
        valley_depths  = tslice[tmp].astype(float)
        invvar         = np.ones(len(tmp))
        
        return valley_indices, valley_depths, invvar
    
    def fit_bspline(self, valley_indices: np.ndarray, valley_depths: np.ndarray, invvar: np.ndarray) -> Tuple[np.ndarray, ...]:
        
        sset = bspline(valley_indices,everyn=2) #pydl.bspline(valley_indices,everyn=2)
        res, yfit = sset.fit(valley_indices, valley_depths, invvar)
        
        x_model = np.arange(self.naxis2).astype(float)
        y_model = sset.value(x_model)[0]
        
        return x_model, y_model
    
    def fit_grid_single(self, xtmp, x_model, y_model) -> Tuple[float, bspline, int, np.ndarray]:
        
        ytrace = np.median(self.data[:,xtmp.astype(int)-7:xtmp.astype(int)+7],axis=1)
        valleys = detect_peaks(ytrace,mpd=2,show=False,valley=True)
        
        nvalley = len(valleys)

        valley_indices = valleys.astype(float)
        valley_depths  = ytrace[valleys].astype(float)
        invvar         = np.ones(nvalley)
        
        # Fit out scattered light / continuum at each x
            
        sset = bspline(valley_indices,everyn=2)
        
        res, yfit = sset.fit(valley_indices, valley_depths, invvar)
        
        comb = ytrace.astype(float)-y_model
        
        return comb, sset, res, yfit
        
    #@ray.remote   
    def process_hdu_data(self, hdu: fits.HDUList, hdu_index: int) -> dict:
        """Processes data from a specific HDU array."""
        
        try:
            self.bspline_ssets = []

            self.hdr = hdu.header
            self.data =hdu.data.astype(float)

            self.naxis1 = self.hdr['naxis1']
            self.naxis2 = self.hdr['naxis2']

            middle_row = int(self.naxis1/2)
            tslice = np.sum(self.data[:,middle_row-5:middle_row+4],axis=1).astype(float)


            valley_indices, valley_depths, invvar = self.generate_valleys(tslice)



            x_model, y_model = self.fit_bspline(valley_indices, valley_depths, invvar)
            print(f'x_model, y_model {x_model}, {y_model}')
            if not self.mph:
                peaks = detect_peaks(tslice-y_model,mpd=2,threshold=10,mph=1500,show=False)
            else:
                peaks = detect_peaks(tslice-y_model,mpd=2,threshold=10,mph=self.mph,show=False)


            self.nfibers  = len(peaks)
            self.xmax     = self.naxis1-100

            n_tracefit = np.floor((self.xmax-self.xmin)/self.fitspace).astype(int)
            xtrace = self.xmin + self.fitspace * np.arange(n_tracefit)

            tracearr = np.zeros(shape=(self.nfibers,n_tracefit))
            print("NFibers = {}".format(self.nfibers))

            if (self.channel=='blue'):
                min_pkheight = 50
            else:
                min_pkheight = 500
            itrace = 0

            for index, item in enumerate(xtrace):
                comb, sset, res, yfit = self.fit_grid_single(item, x_model, y_model)
                breakpoint
                if index == 0:
                    peaks = detect_peaks(comb,mpd=2,mph=min_pkheight,show=False,valley=False)
                else:
                    peaks = tracearr[:,itrace-1].astype(int)

                ifiber = 0

                #try:

                for pk_guess in peaks:

                    if pk_guess == 0:
                        continue
                    pk_centroid = \
                        np.sum(np.multiply(comb[pk_guess-2:pk_guess+3],pk_guess-2+np.arange(5))) \
                        / np.sum(comb[pk_guess-2:pk_guess+3])


                    if (np.abs(pk_centroid-pk_guess) < 1.5):
                        tracearr[ifiber,itrace] = pk_centroid
                    else:
                        tracearr[ifiber,itrace] = pk_guess
                    ifiber += 1
               
                itrace += 1

            self.xtracefit = np.outer(np.ones(ifiber),xtrace)
            self.tracearr  = tracearr
            self.tset      = pydl.xy2traceset(self.xtracefit, self.tracearr, maxdev=0.5)

            x2          = np.outer(np.ones(ifiber),np.arange(self.naxis1))
            self.traces = pydl.traceset2xy(self.tset,xpos=x2)[1]
        
        except Exception as e:
            traceback.print_exc()
            result = {"HDU": hdu_index, "status": "failed", "error":str(e)}
            print(result)
            return result
            
        result = {"HDU": hdu_index, "status": "success"}
        return result
    
    @ray.remote
    def process_hdu_data_ray(self, hdu_data: fits.HDUList, hdu_index: int) -> dict:
        return self.process_hdu_data(hdu_data, hdu_index)

    
    def run_ray_processing(self) -> list:
        """Processes all HDUs in parallel using Ray and returns a log of results."""
        ledger = []
        # Open FITS file to access HDUs
        with fits.open(self.flat_fitsfile) as hdu_list:
            # Prepare Ray tasks for each HDU
            futures = [self.process_hdu_data.remote(hdu_list[i], i) for i in range(len(hdu_list))]
            # Execute tasks and collect results
            ledger = ray.get(futures)  # Retrieve results from Ray tasks
        # Print and return the ledger of HDU processing statuses
        print("Processing completed. Ledger:")
        for entry in ledger:
            print(entry)
        return ledger
    
        