
import os
import sys
from   astropy.io import fits
import scipy
import numpy as np
import time
import psutil
from   matplotlib import pyplot as plt
import traceback
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl
from pypeit.core.fitting import iterfit
from   pypeit.core import fitting
from pypeit.bspline.bspline import bspline
import pickle, h5py
import logging
import ray
from typing import List, Set, Dict, Tuple, Optional
import multiprocessing
import argparse
import cloudpickle
from scipy.signal import find_peaks

# Enable DEBUG for your specific logger
logger = logging.getLogger(__name__)
logging.basicConfig(filename='tracing.log')
logger.setLevel(logging.DEBUG)

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

        # 1A    298 (Green) / 298 Blue
        # 1B    300 (Green) / 300 Blue
        # 2A    299 (Green) / 299 Blue - potentially 2 lost fibers
        # 2B    297 (Green) / 297 Blue - 1 dead fiber
        # 3A    298 (Green) / 298 Blue
        # 3B    300 (Green) / 300 Blue
        # 4A    300 (Green) / 300 Blue

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
            sset = bspline(valley_indices,everyn=4) 
        else:
            sset = bspline(valley_indices,everyn=2)
            
        res, yfit = sset.fit(valley_indices, valley_depths, invvar)
        y_model = sset.value(x_model)[0]
        comb = ytrace.astype(float)-y_model
        
        return comb, sset, res, yfit
         
    def process_hdu_data(self, hdu_data: np.ndarray, hdu_header: dict) -> dict:
        """Processes data from a specific HDU array."""
        
        try:
            self.bspline_ssets = []

            self.hdr = hdu_header
            self.data = hdu_data.astype(float)
            #case sensitive when concerted into a dict for ray processing
            self.naxis1 = self.hdr['NAXIS1']
            self.naxis2 = self.hdr['NAXIS2']
            
            self.channel = self.hdr['COLOR']
            self.bench = self.hdr['BENCH']
            self.side  = self.hdr['SIDE']
            
            if self.channel == 'red':
                logger.warning("Red channel selected which is not yet supported.")
                result = {"status": "unable to process red channel"}
                return result


            #print(f'Processing {self.channel} channel, {self.bench} bench, {self.side} side')
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
                
            if not (self.channel=='blue'):
                self.min_pkheight = 10000
            else:
                self.min_pkheight = 100

            comb = tslice-y_model
            # peaks = detect_peaks(comb,mpd=1,mph=self.min_pkheight,threshold=0,show=True,valley=False)
            peaks, _ = find_peaks(comb,distance=2,height=100,threshold=None)

            peaks = peaks[np.logical_and(peaks > 20, peaks < 2020)]

            pkht = comb[peaks]
            #plt.plot(comb)
            #plt.plot(peaks, pkht, '+')
            #plt.title(f"{len(peaks)} Peaks")
            #plt.show()

            self.min_pkheight = 0.3 * np.median(pkht)
            self.nfibers  = len(peaks)
            self.xmax     = self.naxis1-100

            n_tracefit = np.floor((self.xmax-self.xmin)/self.fitspace).astype(int)
            xtrace = self.xmin + self.fitspace * np.arange(n_tracefit)
            
            tracearr = np.zeros(shape=(self.nfibers,n_tracefit))
            logger.info(f"Bench {self.bench}{self.side} - {self.channel}")
            logger.info(f"NFibers = {self.nfibers}")
            
            for itrace, item in enumerate(xtrace):
                comb, sset, res, yfit = self.fit_grid_single(item, x_model, y_model)
                
                if itrace == 0:
                    peaks = detect_peaks(comb,mpd=2,mph=self.min_pkheight*2*self.window/self.naxis1,show=False,valley=False)
                else:
                    peaks = tracearr[:,itrace-1].astype(int)

                for ifiber, pk_guess in enumerate(peaks):
                    if ifiber >= self.nfibers:
                        logger.warning(f"ifiber {ifiber} exceeds nfibers {self.nfibers} for channel {self.channel} Bench {self.bench} side {self.side}, skipping")
                        continue
                    
                    if pk_guess -2 < 0:
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
            result = {"status": "failed", "error":str(e), 'channel': {self.channel}, 'bench': {self.bench}, 'side': {self.side}}
            logger.warning(result)
            return result
            
        result = {"status": "success"}
        return result
    
    def profileFit(self):
        
        ref = self.data[12,:]

        # Use a working copy of "data" so as not to overwrite that with normalized data
        data_work = np.copy(self.data) 
        for i in range(self.naxis2):
            if (i != 12):
                data_work[i,:] = self.data[i,:] - ref

        fiberimg = np.zeros(self.data.shape,dtype=int)   # Lists the fiber # of each pixel
        profimg  = np.zeros(self.data.shape,dtype=float) # Profile weighting function
        bpmask   = np.zeros(self.data.shape,dtype=bool)  # bad pixel mask
        
        for index, item in enumerate(self.traces):
            
            ytrace = item 

            # Generate a curved "y" image
            yy = np.outer(np.arange(self.naxis2),np.ones(self.naxis1)) \
                - np.outer(np.ones(self.naxis2),ytrace)

            # Normalize out the spectral shape of the lamp for profile fitting
            for i in range(self.naxis1):
                norm = np.sum(data_work[np.where(np.abs(yy[:,i]) < 2.0),i])
                data_work[:,i] = data_work[:,i] / norm

            # Generate a mask of pixels that are
            # (a) within 4 pixels of the profile center for this fiber and
            # (b) not NaNs or Infs
            # Also generate an inverse variance array that is presently flat weighting

            infmask = np.ones(data_work.shape,dtype=bool)
            NaNmask = np.ones(data_work.shape,dtype=bool)
            badmask = np.ones(data_work.shape,dtype=bool)
            profmask = np.zeros(data_work.shape,dtype=bool)
            invvar = np.ones(data_work.shape,dtype=float)
            
            infmask[np.where(np.isinf(data_work))]  = False
            NaNmask[np.where(np.isnan(data_work))] = False
            badmask[np.where(data_work > 20)] = False
            badmask[np.where(data_work < -5)] = False
            profmask[np.where(np.abs(yy) < 3)] = True

            inprof = np.where(infmask & profmask & NaNmask & badmask)

            # Fit the fiber spatial profile with a bspline
            sset,outmask = iterfit(yy[inprof],data_work[inprof],maxiter=6, \
                        invvar=invvar[inprof],kwargs_bspline={'bkspace':0.33})
            
            
            fiberimg[np.where(profmask == True)] = index#ifiber
            bpmask[np.where(infmask == False)]   = True
            profimg[inprof] = profimg[inprof] + sset.value(yy[inprof])[0]

            self.bspline_ssets.append(sset)

        return (fiberimg, profimg, bpmask)
    
    def saveTraces(self, outfile='LLAMASTrace.pkl'):
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        os.makedirs(save_dir, exist_ok=True)
        
        outpath = os.path.join(save_dir, outfile)

        if ('.pkl' in outfile):
            with open(outpath,'wb') as fp:
               cloudpickle.dump(self, fp)

        if ('.h5' in outfile):
            with h5py.File(outpath, 'w') as f:
                # Stack all 2D data arrays along a new axis
                data_stack = np.stack([trace.data for trace in objlist], axis=0)
                f.create_dataset('data', data=data_stack)  

                # Save other attributes
                f.create_dataset('naxis1', data=[llama.naxis1 for llama in objlist])
                f.create_dataset('naxis2', data=[llama.naxis2 for llama in objlist])
                dt = h5py.string_dtype(encoding='utf-8')
                f.create_dataset('bench', data=[llama.bench for llama in objlist])
        return

    
    
@ray.remote
class TraceRay(TraceLlamas):
    
    def __init__(self, fitsfile: str) -> None:
        super().__init__(fitsfile)
        
        return
    
    
    def process_hdu_data(self, hdu_data: np.ndarray, hdu_header: dict) -> dict:
        start_time = time.time()
        
        result = super().process_hdu_data(hdu_data, hdu_header)
        if result["status"] != "success":
                return result
        
        self.fiberimg, self.profimg, self.bpmask = super().profileFit()
        
        outfile = f'{self.channel}_{self.bench}_{self.side}_traces.pkl'
        print(f'outfile: {outfile}')
        super().saveTraces(outfile)
        
        elapsed_time = time.time() - start_time
        return 


def main(fitsfile: str) -> None:
    
    NUMBER_OF_CORES = multiprocessing.cpu_count() 
    ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
    
    print(f"\nStarting with {NUMBER_OF_CORES} cores available")
    print(f"Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    futures = []
    results = []    
    
    with fits.open(fitsfile) as hdul:
        hdus = [(hdu.data, dict(hdu.header)) for hdu in hdul if hdu.data is not None]
        
    hdu_processors = [TraceRay.remote(fitsfile) for _ in range(len(hdus))]
    print(f"\nProcessing {len(hdus)} HDUs with {NUMBER_OF_CORES} cores")
        
    #hdu_processor = TraceRay.remote(fitsfile)
        
    for index, ((hdu_data, hdu_header), processor) in enumerate(zip(hdus, hdu_processors)):
        future = processor.process_hdu_data.remote(hdu_data, hdu_header)
        futures.append(future)
    
    # Monitor processing
    total_jobs = len(futures)
    completed = 0
        
    # Monitor processing
    print("\nProcessing Status:")
    while futures:
        
        # Print current CPU usage every 5 seconds
        if completed % 5 == 0:
            print(f"CPU Usage: {psutil.cpu_percent(percpu=True)}%")
            print(f"Progress: {completed}/{total_jobs} jobs complete")
        
        
        done_id, futures = ray.wait(futures)
        result = ray.get(done_id[0])
        results.append(result)
        completed += 1
        
    print(f"\nAll {total_jobs} jobs complete")
    print(f"Final CPU Usage: {psutil.cpu_percent(percpu=True)}%")
    
    ray.shutdown()
    
    return
    
    
if __name__ == "__main__":  
    
    parser = argparse.ArgumentParser(description='Process LLAMAS FITS files using Ray multiprocessing.')
    parser.add_argument('filename', type=str, help='Path to input FITS file')
    args = parser.parse_args()
      
    NUMBER_OF_CORES = multiprocessing.cpu_count() 
    ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
    
    print(f"\nStarting with {NUMBER_OF_CORES} cores available")
    print(f"Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    futures = []
    results = []    
    
    fitsfile = args.filename
    with fits.open(fitsfile) as hdul:
        hdus = [(hdu.data, dict(hdu.header)) for hdu in hdul if hdu.data is not None]
        
    hdu_processors = [TraceRay.remote(fitsfile) for _ in range(len(hdus))]
    print(f"\nProcessing {len(hdus)} HDUs with {NUMBER_OF_CORES} cores")
        
    #hdu_processor = TraceRay.remote(fitsfile)
        
    for index, ((hdu_data, hdu_header), processor) in enumerate(zip(hdus, hdu_processors)):
        future = processor.process_hdu_data.remote(hdu_data, hdu_header)
        futures.append(future)
    
    # Monitor processing
    total_jobs = len(futures)
    completed = 0
        
    # Monitor processing
    print("\nProcessing Status:")
    while futures:
        
        # Print current CPU usage every 5 seconds
        if completed % 5 == 0:
            print(f"CPU Usage: {psutil.cpu_percent(percpu=True)}%")
            print(f"Progress: {completed}/{total_jobs} jobs complete")
        
        
        done_id, futures = ray.wait(futures)
        result = ray.get(done_id[0])
        results.append(result)
        completed += 1
        
    print(f"\nAll {total_jobs} jobs complete")
    print(f"Final CPU Usage: {psutil.cpu_percent(percpu=True)}%")
    
    ray.shutdown()
    
    
    
    

    
        