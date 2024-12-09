
import os
import sys
from   astropy.io import fits
import scipy
import numpy as np
import json
import time
from datetime import datetime
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
from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, LUT_DIR
import pkg_resources
from pathlib import Path
import rpdb


# Enable DEBUG for your specific logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add timestamp to log filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#logger = setup_logger(__name__, f'traceLlamasMulti_{timestamp}.log')


def get_fiber_position(channel:str, benchside: str, fiber: str) -> int:
    json_file = os.path.join(LUT_DIR, 'traceLUT.json')
    with open(json_file, 'r') as f:
        lut = json.load(f)
    
    # Access nested structure: GreenLUT -> 1A -> fiber_number
    position = lut[channel][benchside][str(fiber)]
    return position


def cross_correlate_combs(comb1: np.ndarray, comb2: np.ndarray) -> Tuple[float, np.ndarray]:
    """Cross correlate two combs to find the offset between them."""
    # Normalize combs
    comb1_norm = (comb1 - np.mean(comb1)) / np.std(comb1)
    comb2_norm = (comb2 - np.mean(comb2)) / np.std(comb2)
    # Calculate cross correlation
    correlation = np.correlate(comb1_norm, comb2_norm, mode='full')
    # Find peak of correlation
    max_corr = np.argmax(correlation)
    offset = max_corr - len(comb1) + 1
    return offset, correlation


class TraceLlamas:
    
    def __init__(self, 
                 fitsfile: str,
                 mph: Optional[int] = None,
                 master_trace: Optional[str] = None
                 ):
        
        self.fitsfile = fitsfile
        self.mph = mph
        self.xmin     = 200
        self.fitspace = 10
        self.min_pkheight = 500
        self.window = 11 #can update to 15
        self.offset_cutoff = 3
        

        # 1A    298 (Green) / 298 Blue
        # 1B    300 (Green) / 300 Blue
        # 2A    299 (Green) / 299 Blue - potentially 2 lost fibers (only found one in comb)
        # 2B    297 (Green) / 297 Blue - 1 dead fiber
        # 3A    298 (Green) / 298 Blue
        # 3B    300 (Green) / 300 Blue
        # 4A    300 (Green) / 300 Blue

        return
    
    def insert_dead_fibers(self, LUT, benchside, pkhts):
        dead_fibers = LUT.get('dead_fibers', {}).get(benchside, [])
        dead_fibers = sorted(dead_fibers)

        pkhts = pkhts.tolist()
        
        for fiber in dead_fibers:
            #pos = get_fiber_position(channel, benchside, fiber)
            pkhts.insert(fiber, 0)
        
        return pkhts
    
    
            
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
        
        y_model = sset.value(self.x_model)[0]
        
        return self.x_model, y_model
    
    
    # takes in location of xwindow, and the y_model which the the dark base level along the comb
    def fit_grid_single(self, xtmp) -> Tuple[float, bspline, int, np.ndarray]:
        
        #defining a new yslice for a given xwindow point
        ytrace = np.median(self.data[:,xtmp.astype(int)-self.window:xtmp.astype(int)+self.window],axis=1)
        #detect the valleys along this new slice
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
        y_model = sset.value(self.x_model)[0]
        comb = ytrace.astype(float)-y_model
        
        return comb, sset, res, yfit
    
    def find_initial_comb(self):
        
        middle_row = int(self.naxis1/2)
        #straight up to _+ 15 pixels on either side
        tslice = np.median(self.data[:,middle_row-5:middle_row+4],axis=1).astype(float)
        valley_indices, valley_depths, invvar = self.generate_valleys(tslice)
        
        self.x_model = np.arange(self.naxis2).astype(float)
        
        if (self.channel != 'blue'):
            sset = bspline(valley_indices,everyn=2, nord=2)
            res, yfit = sset.fit(valley_indices, valley_depths, invvar)
            
            #x_model = np.arange(self.naxis2).astype(float)
            y_model = sset.value(self.x_model)[0]
            
        else:
            y_model = np.zeros(self.naxis2)
            
        if not (self.channel=='blue'):
            self.min_pkheight = 10000
        else:
            self.min_pkheight = 100
        #subtracting the dark level from the tslice
        self.comb = tslice - y_model
        
        # peaks = detect_peaks(comb,mpd=1,mph=self.min_pkheight,threshold=0,show=True,valley=False)
        
        #refinding the peaks after subtracting the valleys from the tslice
        
        self.orig_peaks, _ = find_peaks(self.comb,distance=2,height=100,threshold=None, prominence=500)
        
        return self.comb
    
         
    def process_hdu_data(self, hdu_data: np.ndarray, hdu_header: dict, find_LUT=False) -> dict:
        """Processes data from a specific HDU array."""
        
        try:
            self.bspline_ssets = []

            self.hdr = hdu_header
            self.data = hdu_data.astype(float)
            #case sensitive when concerted into a dict for ray processing
            
            if not 'COLOR' in self.hdr:
                camname = self.hdr['CAM_NAME']
                self.channel = camname.split('_')[1].lower()
                self.bench = camname.split('_')[0][0]
                self.side = camname.split('_')[0][1]
            else:
                self.channel = self.hdr['COLOR'].lower()
                self.bench = self.hdr['BENCH']
                self.side  = self.hdr['SIDE']
            
            if self.channel == 'red':
                logger.warning("Red channel selected which is not yet supported.")
            
            self.naxis1 = self.hdr['NAXIS1']
            self.naxis2 = self.hdr['NAXIS2']
            
            self.benchside = f'{self.bench}{self.side}'
            

            #print(f'Processing {self.channel} channel, {self.bench} bench, {self.side} side')
            #finding the inital comb for the data we are trying to fit
            self.comb = self.find_initial_comb()
            self.orig_comb = self.comb
            if find_LUT:
                return
            #make sure peaks aren't too close to the edge
            #peaks = peaks[np.logical_and(peaks > 20, peaks < 2020)]
            
            #find the height value of the comb using the new peak locations
            #pkht = self.comb[peaks]
            
            
            
            #code which opens the trace LUT, and updates peaks and pkhts arrays to account for dead fibers
            with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'r') as f:
                LUT = json.load(f)
                self.LUT = LUT    
        
            self.master_comb = np.array(LUT['combs'][self.channel.lower()][self.benchside])
            masterpeaks_dict = LUT["fib_pos"][self.channel.lower()][self.benchside]
            
            self.master_peaks = [int(pos) for pos in masterpeaks_dict.values()]
            
            #these quantities are for debugging the tracing process to generate QA plots, might not be needed later on
            #self.pkht = self.insert_dead_fibers(LUT, self.benchside, pkht)
            #self.orig_pkht = self.pkht
            
            
            #assert len(self.pkht) == len(self.master_peaks), "Length of peak heights does not match master peaks"
            
            
            offset, _ = cross_correlate_combs(self.comb, self.master_comb)
            
            if np.abs(offset) > self.offset_cutoff:
                #find a way to print this error to terminal still and the logger
                print(f"Offset of {offset} exceeds cutoff of {self.offset_cutoff}")
                logger.error(f"Offset of {offset} exceeds cutoff of {self.offset_cutoff}")
                return 1
            
            #update the peaks to the master peaks
            self.updated_peaks = self.master_peaks + offset

            #self.min_pkheight = 0.3 * np.median(pkht)
            
            self.nfibers  = len(self.updated_peaks)
            self.xmax     = self.naxis1-100

            ###Note to self: window was originall fitspace here, in principle should be the same thing
            #finding the number of windows along x we want for new tslice
            n_tracefit = np.floor((self.xmax-self.xmin)/self.window).astype(int)
            
            xtrace = self.xmin + self.window * np.arange(n_tracefit)
            
            tracearr = np.zeros(shape=(self.nfibers,n_tracefit))
            
            logger.info(f"Bench {self.bench}{self.side} - {self.channel}")
            logger.info(f"NFibers = {self.nfibers}")
            
            for itrace, item in enumerate(xtrace):
                
                if itrace == 0:
                    peaks = self.updated_peaks
                else:
                    peaks = tracearr[:,itrace-1].astype(int)

                for ifiber, pk_guess in enumerate(peaks):
                    if ifiber >= self.nfibers:
                        logger.warning(f"ifiber {ifiber} exceeds nfibers {self.nfibers} for channel {self.channel} Bench {self.bench} side {self.side}, skipping")
                        continue
                    #breakpoint()
                    
                    #if the guess is too close to the edge, skip
                    ###Shouldn't this not be needed if we exclude peaks too close to the edge?
                    if pk_guess -2 < 0:
                        print('Peak guess too close condition hit')
                        continue
                    
                    #taking the weighted sum
                    
                    pk_centroid = \
                        np.nansum(np.multiply(self.comb[pk_guess-2:pk_guess+3],pk_guess-2+np.arange(5))) \
                        / np.nansum(self.comb[pk_guess-2:pk_guess+3])

                    #if the updated peak diverges too far from the peak guess then use the peak guess
                    if (np.abs(pk_centroid-pk_guess) < 1.5):
                        tracearr[ifiber,itrace] = pk_centroid
                    else:
                        tracearr[ifiber,itrace] = pk_guess

            #defines the coordinates of the trace along the x axis
            self.xtracefit = np.outer(np.ones(ifiber),xtrace)
            #defining the y (peak coord) of the trace for that window along xthe x axis for every fiber
            self.tracearr  = tracearr
            #defines the traces by fitting a spline along the x axis
            
            self.tset      = pydl.xy2traceset(self.xtracefit, self.tracearr, maxdev=0.5)
            
            x2          = np.outer(np.ones(ifiber),np.arange(self.naxis1))
            #interpolates the traces to give an x,y position for each fiber along the naxis
            
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
                norm = np.nansum(data_work[np.where(np.abs(yy[:,i]) < 2.0),i])
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
            ##this is where we ajust the width of the profile mask in pixels
            profmask[np.where(np.abs(yy) < 2)] = True

            inprof = np.where(infmask & profmask & NaNmask & badmask)

            # Fit the fiber spatial profile with a bspline
            
            sset,outmask = iterfit(yy[inprof],data_work[inprof],maxiter=6, \
                        invvar=invvar[inprof],kwargs_bspline={'bkspace':0.33})
            
            
            fiberimg[np.where(profmask == True)] = index#ifiber
            bpmask[np.where(infmask == False)]   = True
            profimg[inprof] = profimg[inprof] + sset.value(yy[inprof])[0]

            self.bspline_ssets.append(sset)
        
        self.fiberimg, self.profimg, self.bpmask = fiberimg, profimg, bpmask
        
        return (fiberimg, profimg, bpmask)
    
    def saveTraces(self, outfile='LLAMASTrace.pkl'):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        outpath = os.path.join(OUTPUT_DIR, outfile)
        print(f'outpath: {outpath}')

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
        self.fitsfile = os.path.basename(fitsfile)
        print(f'fitsfile: {self.fitsfile}')
        return
    
    
    def process_hdu_data(self, hdu_data: np.ndarray, hdu_header: dict) -> dict:
        start_time = time.time()
        
        result = super().process_hdu_data(hdu_data, hdu_header)
        if result["status"] != "success":
                return result
        
        self.fiberimg, self.profimg, self.bpmask = super().profileFit()
        
        origfile = self.fitsfile.split('.fits')[0]
        color = self.channel.lower()
        print(f'color: {color}')
        self.outfile = f'{origfile}_{self.channel.lower()}_{self.bench}_{self.side}_traces.pkl'
        print(f'outfile: {self.outfile}')
        super().saveTraces(self.outfile)
        
        elapsed_time = time.time() - start_time
        return 


def main(fitsfile: str) -> None:

    NUMBER_OF_CORES = multiprocessing.cpu_count() 
    # ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
    # Initialize Ray with logging config
    ray.shutdown()  # Clear any existing Ray instances
    ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
    
    print(f"\nStarting with {NUMBER_OF_CORES} cores available")
    print(f"Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    futures = []
    results = []    
    
    with fits.open(fitsfile) as hdul:
        hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul if hdu.data.astype(float) is not None]
        
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
    os.environ["RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S"] = "1200"
    parser = argparse.ArgumentParser(description='Process LLAMAS FITS files using Ray multiprocessing.')
    parser.add_argument('filename', type=str, help='Path to input FITS file')
    args = parser.parse_args()
      
    NUMBER_OF_CORES = multiprocessing.cpu_count() 
    # ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
    ray.shutdown()  # Clear any existing Ray instances
    ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
    
    print(f"\nStarting with {NUMBER_OF_CORES} cores available")
    print(f"Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    futures = []
    results = []    
    
    fitsfile = args.filename
    with fits.open(fitsfile) as hdul:
        hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul if hdu.data is not None]
        
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
    
    
    
    

    
        