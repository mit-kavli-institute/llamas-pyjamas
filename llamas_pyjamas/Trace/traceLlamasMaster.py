
"""
Module: traceLlamasMaster
This module provides functionality for processing and tracing fiber positions in LLAMAS FITS files using Ray multiprocessing to produce a set of Master traces. 
It includes classes and functions to handle FITS file data, detect peaks, fit B-splines, and generate trace profiles for fibers.
Classes:
    TraceLlamas: 
        A class for processing and tracing fiber positions in LLAMAS FITS files. 
        It includes methods for detecting peaks, fitting B-splines, generating trace profiles, and saving trace data.
    TraceRay(TraceLlamas): 
        A subclass of TraceLlamas that uses Ray for parallel processing of HDU data in FITS files.
Functions:
    run_ray_tracing(fitsfile: str, channel: str = None) -> None:
        Runs the tracing process on the specified FITS file using Ray multiprocessing.
Usage:
    This module can be run as a standalone script to process LLAMAS FITS files. 
    Use the command line arguments to specify the input FITS file and optional parameters.
Example:
    python traceLlamasMaster.py filename.fits --channel blue

"""
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
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, LUT_DIR, CALIB_DIR
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
    def get_fiber_position(channel: str, benchside: str, fiber: str) -> int:
        """
        Retrieve the position of a fiber from a lookup table (LUT) based on the given channel, benchside, and fiber.
        Args:
            channel (str): The channel name (red, green or blue).
            benchside (str): The benchside identifier (e.g., '1A').
            fiber (str): The fiber identifier (e.g., 'fiber_number').
        Returns:
            int: The position of the specified fiber.
        Raises:
            FileNotFoundError: If the LUT file is not found.
            KeyError: If the specified channel, benchside, or fiber is not found in the LUT.
        """

    json_file = os.path.join(LUT_DIR, 'traceLUT.json')
    with open(json_file, 'r') as f:
        lut = json.load(f)
    
    # Access nested structure: GreenLUT -> 1A -> fiber_number
    position = lut[channel][benchside][str(fiber)]
    return position



class TraceLlamas:
    """
    A class used to trace and process fiber data from FITS files.
    Attributes
    ----------
    fitsfile : str
        The path to the FITS file.
    mph : Optional[int]
        Minimum peak height for peak detection.
    master_trace : Optional[str]
        Path to the master trace file.
    xmin : int
        Minimum x-coordinate for trace fitting.
    fitspace : int
        Space for fitting.
    min_pkheight : int
        Minimum peak height for peak detection.
    window : int
        Window size for median calculation.
    offset_cutoff : int
        Cutoff for offset calculation.
    Methods
    -------
    insert_dead_fibers(LUT, benchside, pkhts)
        Inserts dead fibers into the peak heights array.
    generate_valleys(tslice: float) -> Tuple[np.ndarray, ...]
        Generates valleys in the given slice.
    fit_bspline(valley_indices: np.ndarray, valley_depths: np.ndarray, invvar: np.ndarray) -> Tuple[np.ndarray, ...]
        Fits a B-spline to the given valleys.
    fit_grid_single(xtmp) -> Tuple[float, bspline, int, np.ndarray]
        Fits a grid to a single x-window.
    find_comb(rownum=None)
        Finds the comb for the given row number.
    process_hdu_data(hdu_data: np.ndarray, hdu_header: dict, find_LUT=False) -> dict
        Processes data from a specific HDU array.
    profileFit()
        Fits the profile of the data.
    saveTraces(outfile='LLAMASTrace.pkl')
        Saves the traces to a file.
    """

    
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
    
    def insert_dead_fibers(self, LUT: dict, benchside: str, pkhts: list) -> list:
        """
        Inserts zero values into the pkhts list at positions specified by dead fibers.
        This method retrieves the list of dead fibers for a given benchside from the LUT (Lookup Table),
        sorts the list, and then inserts a zero value into the pkhts list at each position corresponding
        to a dead fiber.
        Args:
            LUT (dict): A lookup table containing information about dead fibers.
            benchside (str): The benchside identifier to retrieve dead fibers for.
            pkhts (list): A list of pkhts values where zeros will be inserted at positions of dead fibers.
        Returns:
            list: The updated pkhts list with zeros inserted at positions of dead fibers.
        """

        dead_fibers = LUT.get('dead_fibers', {}).get(benchside, [])
        dead_fibers = sorted(dead_fibers)

        pkhts = pkhts.tolist()
        
        for fiber in dead_fibers:
            pkhts.insert(fiber, 0)
        
        return pkhts
    
    
            
    def generate_valleys(self, tslice: float) -> Tuple[np.ndarray, ...]:
        """
        Detects valleys in the given data slice and returns their indices, depths, and an array of ones.
        Parameters:
        -----------
        tslice : float
            The data slice in which to detect valleys.
        Returns:
        --------
        Tuple[np.ndarray, ...]
            A tuple containing:
            - valley_indices (np.ndarray): The indices of the detected valleys.
            - valley_depths (np.ndarray): The depths of the detected valleys.
            - invvar (np.ndarray): An array of ones with the same length as the number of detected valleys.
        Notes:
        ------
        If no valleys are detected, a log message is generated indicating that no valleys were found and suggesting to check the threshold.
        """


        tmp            = detect_peaks(tslice,mpd=2,threshold=10,show=False,valley=True)
        
        if len(tmp) == 0:
            logger.info("No valleys detected - check threshold")
        
        valley_indices = np.ndarray(len(tmp))
        valley_depths  = np.ndarray(len(tmp))
        
        valley_indices = tmp.astype(float)
        valley_depths  = tslice[tmp].astype(float)
        
        invvar         = np.ones(len(tmp))
        
        return valley_indices, valley_depths, invvar
    
    def fit_bspline(self, valley_indices: np.ndarray, valley_depths: np.ndarray, invvar: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Fit a B-spline to the given valley indices and depths.
        Parameters:
        valley_indices (np.ndarray): Array of indices where valleys are located.
        valley_depths (np.ndarray): Array of depths corresponding to the valley indices.
        invvar (np.ndarray): Inverse variance of the valley depths.
        Returns:
        Tuple[np.ndarray, ...]: A tuple containing the x_model and the fitted y_model values.
        """

        
        sset = bspline(valley_indices,everyn=2) #pydl.bspline(valley_indices,everyn=2)
        res, yfit = sset.fit(valley_indices, valley_depths, invvar)
        
        y_model = sset.value(self.x_model)[0]
        
        return self.x_model, y_model
    
    
    # takes in location of xwindow, and the y_model which the the dark base level along the comb
    def fit_grid_single(self, xtmp: int) -> Tuple[float, bspline, int, np.ndarray]:
        """
        Fits a B-spline to a slice of the data at a given x-window point.
        Parameters:
        -----------
        xtmp : int
            The x-coordinate around which the y-slice is taken.
        Returns:
        --------
        comb : np.ndarray
            The combined result of the y-slice minus the B-spline model.
        sset : bspline
            The B-spline object fitted to the data.
        res : int
            The result of the B-spline fitting process.
        yfit : np.ndarray
            The fitted y-values from the B-spline model.
        """

        
        #defining a new yslice for a given xwindow point
        ytrace = np.median(self.data[:,xtmp.astype(int)-self.window:xtmp.astype(int)+self.window],axis=1)
        #detect the valleys along this new slice
        valleys = detect_peaks(ytrace,mpd=2,show=False,valley=True)
        
        nvalley = len(valleys)

        valley_indices = valleys.astype(float)
        valley_depths  = ytrace[valleys].astype(float)
        invvar         = np.ones(nvalley)
        

        sset = bspline(valley_indices,everyn=2)
            
        res, yfit = sset.fit(valley_indices, valley_depths, invvar)
        y_model = sset.value(self.x_model)[0]
        comb = ytrace.astype(float)-y_model
        
        return comb, sset, res, yfit
    
    def find_comb(self, rownum=None)-> np.ndarray:
        """
        Find and process the comb of peaks in the data.
        This function extracts a slice of the data around the specified row number,
        identifies valleys, fits a B-spline to the valleys, and then finds peaks
        in the residuals after subtracting the fitted B-spline.
        Parameters:
        rownum (int, optional): The row number around which to extract the data slice.
                                If None, the middle row is used.
        Returns:
        numpy.ndarray: The comb of peaks in the data slice after subtracting the fitted B-spline.
        """

        
        # When in doubt, extract in the middle
        if (rownum == None):
            rownum = int(self.naxis1/2)

        rownum = int(rownum)

        #straight up to _+ 15 pixels on either side
        tslice = np.median(self.data[:,rownum-8:rownum+7],axis=1).astype(float)
        valley_indices, valley_depths, invvar = self.generate_valleys(tslice)
        
        self.x_model = np.arange(self.naxis2).astype(float)
        
        
        sset = bspline(valley_indices,everyn=2, nord=2)
        res, yfit = sset.fit(valley_indices, valley_depths, invvar)
        
        #x_model = np.arange(self.naxis2).astype(float)
        self.y_model = sset.value(self.x_model)[0]
        
        self.min_pkheight = 10000    
        if self.channel.lower() == 'blue':
            self.min_pkheight = 1000
        

        self.comb = tslice - self.y_model
        
        self.peaks, self.peak_properties = find_peaks(self.comb,distance=2,height=100,threshold=None, prominence=500)
        self.pkht = self.peak_properties['peak_heights']
        
        return self.comb
    
         
    def process_hdu_data(self, hdu_data: np.ndarray, hdu_header: dict, find_LUT=False) -> dict:
        """
        Processes data from a specific HDU (Header Data Unit) array.
        Parameters:
        -----------
        hdu_data : np.ndarray
            The data array from the HDU to be processed.
        hdu_header : dict
            The header information associated with the HDU data.
        find_LUT : bool, optional
            Flag to determine whether to find and use a Look-Up Table (LUT) for processing (default is False).
        Returns:
        --------
        dict
            A dictionary containing the status of the processing. If successful, returns {"status": "success"}.
            If an error occurs, returns {"status": "failed", "error": str(e), 'channel': self.channel, 'bench': self.bench, 'side': self.side}.
        Description:
        ------------
        This method processes the HDU data by performing the following steps:
        1. Initializes the bspline_ssets list.
        2. Extracts and processes header information to determine the channel, bench, and side.
        3. Extracts the dimensions of the data (NAXIS1 and NAXIS2).
        4. Finds the initial comb for the data to be fitted.
        5. Updates peaks and peak heights, ensuring they are not too close to the edge.
        6. Optionally, opens and uses a trace Look-Up Table (LUT) to account for dead fibers.
        7. Determines the number of fibers and the maximum x-axis value for tracing.
        8. Fits combs from the midpoint forward and then backward, updating peak positions.
        9. Defines the coordinates of the trace along the x-axis and fits a spline along the x-axis for each fiber.
        10. Interpolates the traces to give x, y positions for each fiber along the naxis.
        Exceptions:
        -----------
        If an error occurs during processing, the method catches the exception, logs the error, and returns a dictionary with the status "failed" and the error message.
        """
        
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
                      
            self.naxis1 = self.hdr['NAXIS1']
            self.naxis2 = self.hdr['NAXIS2']
            
            self.benchside = f'{self.bench}{self.side}'
            

            #print(f'Processing {self.channel} channel, {self.bench} bench, {self.side} side')
            #finding the inital comb for the data we are trying to fit
            
            ######New code to subtract background from the data
            # n_rows = 12
            # top_rows = self.data[-n_rows:, :]
            # background = np.median(top_rows)
            
            # self.data = self.data - background

            self.comb = self.find_comb(rownum=self.naxis1/2)
            self.orig_comb = self.comb
            
            self.first_peaks = self.peaks
            self.first_pkht = self.pkht
            

            #make sure peaks aren't too close to the edge
            self.updated_peaks = self.first_peaks[np.logical_and(self.first_peaks > 20, self.first_peaks < 2020)]
            pkhts = self.first_pkht[np.logical_and(self.first_peaks > 20, self.first_peaks < 2020)]
            
            
            # #code which opens the trace LUT, and updates peaks and pkhts arrays to account for dead fibers
            # with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'r') as f:
            #     LUT = json.load(f)
            #     self.LUT = LUT    
        
            # self.master_comb = np.array(LUT['combs'][self.channel.lower()][self.benchside])
            # masterpeaks_dict = LUT["fib_pos"][self.channel.lower()][self.benchside]
            
            # self.master_peaks = [int(pos) for pos in masterpeaks_dict.values()]
            
            #these quantities are for debugging the tracing process to generate QA plots, might not be needed later on
            # self.pkht = self.insert_dead_fibers(LUT, self.benchside, pkhts)
            # self.orig_pkht = self.pkht

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


            ######## Fit combs from the midpoint forward ########
            mid_index = int(n_tracefit / 2)
            tt = xtrace[mid_index:]
            for itrace, thisx in enumerate(tt):
                thiscomb = self.find_comb(thisx)

                if itrace == 0:
                    
                    peaks = np.array(self.updated_peaks)
                else:
                    peaks = tracearr[:,mid_index+itrace-1].astype(int)

                for ifiber, pk_guess in enumerate(peaks):
                    if ifiber >= self.nfibers:
                        logger.warning(f"ifiber {ifiber} exceeds nfibers {self.nfibers} for channel {self.channel} Bench {self.bench} side {self.side}, skipping")
                        continue
                    
                    #if the guess is too close to the edge, skip
                    ###Shouldn't this not be needed if we exclude peaks too close to the edge?
                    if pk_guess -2 < 0:
                        print('Peak guess too close condition hit')
                        continue
                    
                    #taking the weighted sum  
                    pk_centroid = \
                        np.nansum(np.multiply(thiscomb[pk_guess-2:pk_guess+3],pk_guess-2+np.arange(5))) \
                        / np.nansum(thiscomb[pk_guess-2:pk_guess+3])

                    #if the updated peak diverges too far from the peak guess then use the peak guess
                    if (np.abs(pk_centroid-pk_guess) < 1.5):
                        tracearr[ifiber,mid_index+itrace] = pk_centroid
                    else:
                        tracearr[ifiber,mid_index+itrace] = pk_guess


            ######### Now go back and fit from the midpoint backward ######
            for itrace, thisx in enumerate(reversed(xtrace[0:mid_index])):
                thiscomb = self.find_comb(thisx)

                if itrace == 0:
                    peaks = np.array(self.updated_peaks)
                else:
                    peaks = tracearr[:,mid_index-itrace+1].astype(int)

                for ifiber, pk_guess in enumerate(peaks):
                    if ifiber >= self.nfibers:
                        logger.warning(f"ifiber {ifiber} exceeds nfibers {self.nfibers} for channel {self.channel} Bench {self.bench} side {self.side}, skipping")
                        continue
                    
                    #if the guess is too close to the edge, skip
                    ###Shouldn't this not be needed if we exclude peaks too close to the edge?
                    if pk_guess -2 < 0:
                        print('Peak guess too close condition hit')
                        continue
                    
                    #taking the weighted sum  
                    pk_centroid = \
                        np.nansum(np.multiply(thiscomb[pk_guess-2:pk_guess+3],pk_guess-2+np.arange(5))) \
                        / np.nansum(thiscomb[pk_guess-2:pk_guess+3])

                    #if the updated peak diverges too far from the peak guess then use the peak guess
                    if (np.abs(pk_centroid-pk_guess) < 1.5):
                        tracearr[ifiber,mid_index-itrace-1] = pk_centroid
                    else:
                        tracearr[ifiber,mid_index-itrace-1] = pk_guess


            #defines the coordinates of the trace along the x axis
            self.xtracefit = np.outer(np.ones(self.nfibers),xtrace)
            #defining the y (peak coord) of the trace for that window along xthe x axis for every fiber
            self.tracearr  = tracearr
            #defines the traces by fitting a spline along the x axis
            
            self.tset      = pydl.xy2traceset(self.xtracefit, self.tracearr, maxdev=0.5)
            
            x2          = np.outer(np.ones(self.nfibers),np.arange(self.naxis1))
            #interpolates the traces to give an x,y position for each fiber along the naxis
            
            self.traces = pydl.traceset2xy(self.tset,xpos=x2)[1]


        except Exception as e:
            traceback.print_exc()
            result = {"status": "failed", "error":str(e), 'channel': {self.channel}, 'bench': {self.bench}, 'side': {self.side}}
            logger.warning(result)
            return result
            
        result = {"status": "success"}
        return result
    
    def profileFit(self)-> Tuple[np.ndarray, ...]:
        """
        Fits the spatial profile of fibers in the data.
        This method normalizes the data, generates masks for bad pixels, and fits
        the spatial profile of fibers using a bspline. The results are stored in
        the instance variables `fiberimg`, `profimg`, and `bpmask`.
        Returns:
            tuple: A tuple containing:
                - fiberimg (ndarray): An array listing the fiber number of each pixel.
                - profimg (ndarray): An array containing the profile weighting function.
                - bpmask (ndarray): A boolean array representing the bad pixel mask.
        """

        
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
    
    def saveTraces(self, outfile='LLAMASTrace.pkl')-> None:
        os.makedirs(CALIB_DIR, exist_ok=True)
        outpath = os.path.join(CALIB_DIR, outfile)
        
            
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
    """
    TraceRay class is designed to execute TraceLlamas using ray multiprocessing.
    Attributes:
        fitsfile (str): The name of the FITS file being processed.
    Methods:
        __init__(fitsfile: str) -> None:
            Initializes the TraceRay instance with the given FITS file.
        process_hdu_data(hdu_data: np.ndarray, hdu_header: dict) -> dict:
            Processes the HDU data and header, fits the profile, saves the traces, and returns the result.
    """

    
    def __init__(self, fitsfile: str) -> None:
        super().__init__(fitsfile)
        self.fitsfile = os.path.basename(fitsfile)
        print(f'fitsfile: {self.fitsfile}')
        return
    
    
    def process_hdu_data(self, hdu_data: np.ndarray, hdu_header: dict) -> dict:
        start_time = time.time()
        
        result = super().process_hdu_data(hdu_data, hdu_header)

               
        self.fiberimg, self.profimg, self.bpmask = super().profileFit()
        
        origfile = self.fitsfile.split('.fits')[0]
        color = self.channel.lower()
        print(f'color: {color}')
        self.outfile = f'LLAMAS_master_{self.channel.lower()}_{self.bench}_{self.side}_traces.pkl'
        print(f'outfile: {self.outfile}')
        super().saveTraces(self.outfile)
        
        elapsed_time = time.time() - start_time
        return 
        if result["status"] != "success":
                return result

def run_ray_tracing(fitsfile: str, channel: str = None) -> None:

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
        if channel is not None and 'COLOR' in hdul[1].header:
            hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul if hdu.data.astype(float) is not None and hdu.header['COLOR'].lower() == channel.lower()]
        elif channel is not None and 'CAM_NAME' in hdul[1].header:
            hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul if hdu.data.astype(float) is not None and hdu.header['CAM_NAME'].split('_')[1].lower() == channel.lower()]
        else:
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
    parser.add_argument('--mastercalib', action='store_true', help='Use master calibration')
    parser.add_argument('--channel', type=str, choices=['red', 'green', 'blue'], help='Specify the color channel to use')
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
        if args.channel is not None and 'COLOR' in hdul[1].header:
            hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul if hdu.data is not None and hdu.header['COLOR'].lower() == args.channel.lower()]
        elif args.channel is not None and 'CAM_NAME' in hdul[1].header:
            hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul if hdu.data is not None and hdu.header['CAM_NAME'].split('_')[1].lower() == args.channel.lower()]
        else:
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
    
    
    
    

    
        