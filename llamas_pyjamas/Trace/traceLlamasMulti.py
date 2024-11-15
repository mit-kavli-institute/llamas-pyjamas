
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
import logging
import ray
from typing import List, Set, Dict, Tuple, Optional

#Initalising ray for multiprocessing
# ray.init()



class TraceLlamas:
    
    def __init__(self, 
                 fitsfile: str,
                 mph: Optional[int] = None):
        
        self.flat_fitsfile = fitsfile
        self.mph = mph
        
        self.xmin     = 200
        self.fitspace = 10
        self.min_pkheight = 500
        
        self.window = 11
        
        return
        
            
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
        
        ytrace = np.median(self.data[:,xtmp.astype(int)-self.window:xtmp.astype(int)+self.window],axis=1)
        valleys = detect_peaks(ytrace,mpd=2,show=False,valley=True)
        
        nvalley = len(valleys)

        valley_indices = valleys.astype(float)
        valley_depths  = ytrace[valleys].astype(float)
        invvar         = np.ones(nvalley)
        
        if self.channel == 'blue':
            sset = bspline(valley_indices,everyn=2) 
        else:
            sset = bspline(valley_indices,everyn=4)
            
        res, yfit = sset.fit(valley_indices, valley_depths, invvar)
        y_model = sset.value(x_model)[0]
        comb = ytrace.astype(float)-y_model
        
        return comb, sset, res, yfit
        
    #@ray.remote   
    def process_hdu_data(self, hdu: fits.HDUList) -> dict:
        """Processes data from a specific HDU array."""
        
        try:
            self.bspline_ssets = []

            self.hdr = hdu.header
            self.data = hdu.data.astype(float)

            self.naxis1 = self.hdr['naxis1']
            self.naxis2 = self.hdr['naxis2']
            
            self.channel = self.hdr['COLOR']
            self.bench = self.hdr['BENCH']
            self.side  = self.hdr['SIDE']
            
            if self.channel == 'red':
                logging.warning("Red channel selected which is not yet supported.")
                return 0

            middle_row = int(self.naxis1/2)
            tslice = np.median(self.data[:,middle_row-5:middle_row+4],axis=1).astype(float)


            valley_indices, valley_depths, invvar = self.generate_valleys(tslice)
            
            
            x_model = np.arange(self.naxis2).astype(float)
            if (self.channel != 'blue'):
                
                sset = bspline(valley_indices,everyn=2)
                res, yfit = sset.fit(valley_indices, valley_depths, invvar)
                x_model = np.arange(self.naxis2).astype(float)
                y_model = sset.value(x_model)[0]
                
            else:
                y_model = np.zeros(self.naxis2)
                
            comb = tslice-y_model
            peaks = detect_peaks(comb,mpd=2,threshold=10,show=False, valley=False)
            pkht = comb[peaks]

            if not (self.channel=='blue'):
                self.min_pkheight = 10000

            self.min_pkheight = 0.3 * np.median(pkht)
            

            self.nfibers  = len(peaks)
            self.xmax     = self.naxis1-100

            n_tracefit = np.floor((self.xmax-self.xmin)/self.fitspace).astype(int)
            xtrace = self.xmin + self.fitspace * np.arange(n_tracefit)
            
            tracearr = np.zeros(shape=(self.nfibers,n_tracefit))
            logging.info("NFibers = {}".format(self.nfibers))
            
            for itrace, item in enumerate(xtrace):
                comb, sset, res, yfit = self.fit_grid_single(item, x_model, y_model)
                
                if itrace == 0:
                    peaks = detect_peaks(comb,mpd=2,mph=self.min_pkheight*2*self.window/self.naxis1,show=False,valley=False)
                else:
                    peaks = tracearr[:,itrace-1].astype(int)


                for ifiber, pk_guess in enumerate(peaks):
                    if pk_guess == 0:
                        continue
                    pk_centroid = \
                        np.sum(np.multiply(comb[pk_guess-2:pk_guess+3],pk_guess-2+np.arange(5))) \
                        / np.sum(comb[pk_guess-2:pk_guess+3])


                    if (np.abs(pk_centroid-pk_guess) < 1.5):
                        tracearr[ifiber,itrace] = pk_centroid
                    else:
                        tracearr[ifiber,itrace] = pk_guess

            self.xtracefit = np.outer(np.ones(ifiber),xtrace)
            self.tracearr  = tracearr
            self.tset      = pydl.xy2traceset(self.xtracefit, self.tracearr, maxdev=0.5)

            x2          = np.outer(np.ones(ifiber),np.arange(self.naxis1))
            self.traces = pydl.traceset2xy(self.tset,xpos=x2)[1]
        
        except Exception as e:
            traceback.print_exc()
            result = {"status": "failed", "error":str(e)}
            logging.warning(result)
            return result
            
        result = {"status": "success"}
        return result
    
    

    
    
    
    
@ray.remote
class TraceRay(TraceLlamas):
    
    def __init__(self, fitsfile: str) -> None:
        pass
    
    def process_hdu_data(self, hdu: fits.HDUList) -> dict:
        super().process_hdu_data(hdu)
        return
    
    @staticmethod
    def run(self, n_cpu: int = 5) -> None:
        #NUMBER_OF_CORES = multiprocessing.cpu_count()

        #initalise ray
        ray.init(ignore_reinit_error=True)

        # Start a timer to capture the total elapsed computation time.
        start_time = time.monotonic()

        futures = []
        results = []

        # Launch all of the actors and call the `calculate` method on them but do not
        # wait for the results. This is so that all the actors get started at the same
        # time. We will later wait for the results in another loop.
        for i in range(n_cpu):
            hdu_processor = TraceRay.remote()
            future = hdu_processor.process_hdu_data.remote()
            futures.append(future)

        # Now wait for the results of all the calculations.
        for index, future in enumerate(futures):
            result, elapsed_time = ray.get(future)
            results.append(result)
            print(
                f"Actor index: {index}. Result: {result}. Elapsed time: {elapsed_time} seconds"
            )

        print(f"Total elapsed time: {time.monotonic() - start_time} seconds")

        ray.shutdown()
    
    
    
    

    
        