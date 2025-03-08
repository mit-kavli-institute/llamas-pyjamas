
"""
Module: traceLlamasMulti
This module provides functionality for processing and tracing fibers in LLAMAS FITS files using Ray multiprocessing. 
It includes classes and methods for detecting peaks, fitting B-splines, and generating fiber profiles.
The main class, TraceLlamas, handles the core processing logic, including:
- Loading and processing FITS files
- Detecting peaks and valleys in the data
- Fitting B-splines to the detected features
- Generating fiber profiles and bad pixel masks
- Saving the processed traces to output files
The module relies on a set of master traces already being produced to apply cross-correlation methods of matching the combs. 
This ensures accurate alignment and tracing of fibers across different channels and bench sides.
Classes:
- TraceLlamas: Core class for processing and tracing fibers in LLAMAS FITS files.
- TraceRay: Subclass of TraceLlamas that integrates Ray for parallel processing.
Functions:
- run_ray_tracing: Function to initiate Ray-based tracing on a given FITS file.
Usage:
- The module can be run as a standalone script with command-line arguments to specify the input FITS file and other options.
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

from llamas_pyjamas.File.llamasIO import process_fits_by_color

# Enable DEBUG for your specific logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add timestamp to log filename
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#logger = setup_logger(__name__, f'traceLlamasMulti_{timestamp}.log')


def get_fiber_position(channel:str, benchside: str, fiber: str) -> int: 
    """
    Retrieve the position of a fiber from a lookup table (LUT) based on the given channel, benchside, and fiber identifier.
    Args:
        channel (str): The channel name to look up in the LUT (red, blue or green).
        benchside (str): The benchside identifier to look up in the LUT (e.g., '1A').
        fiber (str): The fiber identifier to look up in the LUT.
    Returns:
        int: The position of the specified fiber.
    Raises:
        FileNotFoundError: If the LUT file does not exist.
        KeyError: If the specified channel, benchside, or fiber is not found in the LUT.
    """

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
    """
    A class to trace and process fiber positions in a FITS file.
    Attributes:
    -----------
    fitsfile : str
        The path to the FITS file to be processed.
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
        Cutoff for offset in cross-correlation.
    Methods:
    --------
    insert_dead_fibers(LUT, benchside, pkhts):
        Inserts dead fibers into the peak heights array.
    generate_valleys(tslice: float) -> Tuple[np.ndarray, ...]:
        Generates valleys in the given slice.
    fit_bspline(valley_indices: np.ndarray, valley_depths: np.ndarray, invvar: np.ndarray) -> Tuple[np.ndarray, ...]:
        Fits a B-spline to the given valleys.
    fit_grid_single(xtmp) -> Tuple[float, bspline, int, np.ndarray]:
        Fits a grid to a single x-window.
    find_comb(rownum=None):
        Finds the comb for the given row number.
    process_hdu_data(hdu_data: np.ndarray, hdu_header: dict, find_LUT=False) -> dict:
        Processes data from a specific HDU array.
    profileFit():
        Fits the profile of the fibers.
    saveTraces(outfile='LLAMASTrace.pkl'):
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
    
    def insert_dead_fibers(self, LUT: dict, benchside: str, pkhts: np.ndarray) -> np.ndarray:
        """
        Inserts dead fibers into the pkhts list at the positions specified in the LUT.
        Parameters:
        - LUT (dict): Lookup table containing information about dead fibers.
        - benchside (str): The benchside identifier to look up dead fibers in the LUT.
        - pkhts (list): List of peak heights to insert dead fibers into.
        Returns:
        - list: The updated pkhts list with dead fibers inserted at the specified positions.
        """

        dead_fibers = LUT.get('dead_fibers', {}).get(benchside, [])
        dead_fibers = sorted(dead_fibers)

        pkhts = pkhts.tolist()
        
        for fiber in dead_fibers:
            #pos = get_fiber_position(channel, benchside, fiber)
            pkhts.insert(fiber, 0)
        
        return pkhts
    
    
            
    def generate_valleys(self, tslice: float) -> Tuple[np.ndarray, ...]:
        """
        Generate valleys from a given slice of the raw image.
        This method detects valleys in the provided slice data and returns
        the indices and depths of the valleys, along with an array of inverse variances.
        Parameters:
        tslice (float): The slice data from which valleys are to be detected.
        Returns:
        Tuple[np.ndarray, ...]: A tuple containing:
            - valley_indices (np.ndarray): The indices of the detected valleys.
            - valley_depths (np.ndarray): The depths of the detected valleys.
            - invvar (np.ndarray): An array of ones with the same length as the number of detected valleys.
        """


        tmp            = detect_peaks(tslice,mpd=2,threshold=10,show=False,valley=True)
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
        invvar (np.ndarray): Array of inverse variances for the valley depths.
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
        Fits a grid to a single x-window point and returns the residuals, bspline object, 
        fit result, and fitted y-values.
        Parameters:
        -----------
        xtmp : int
            The x-window point around which the fitting is performed.
        Returns:
        --------
        comb : np.ndarray
            The residuals after subtracting the model from the y-trace.
        sset : bspline
            The bspline object used for fitting.
        res : int
            The result of the bspline fit.
        yfit : np.ndarray
            The fitted y-values from the bspline model.
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
        Finds and returns the comb for a given row number in the data.
        Parameters:
        rownum (int, optional): The row number to process. If None, defaults to the middle row.
        Returns:
        numpy.ndarray: The computed comb for the specified row.
        Notes:
        - If `rownum` is not provided, it defaults to the middle row of the data.
        - The method extracts a slice of the data around the specified row and computes the valleys.
        - A B-spline is fitted to the valley depths, and the comb is calculated by subtracting the fitted model from the slice.
        - If an exception occurs, a default minimum peak height and master comb are used.
        """

        try:
            # When in doubt, extract in the middle
            if (rownum == None):
                rownum = int(self.naxis1/2)

            rownum = int(rownum)

            #straight up to _+ 15 pixels on either side
            tslice = np.median(self.data[:,rownum-5:rownum+4],axis=1).astype(float)
            valley_indices, valley_depths, invvar = self.generate_valleys(tslice)

            self.x_model = np.arange(self.naxis2).astype(float)


            sset = bspline(valley_indices,everyn=2, nord=2)
            res, yfit = sset.fit(valley_indices, valley_depths, invvar)

            #x_model = np.arange(self.naxis2).astype(float)
            y_model = sset.value(self.x_model)[0]


            self.min_pkheight = 10000
            if self.channel.lower() == 'blue':
                logger.info(f"Setting min peak height to 1000 for blue channel")
                self.min_pkheight = 1000

            self.comb = tslice - y_model
        
            #self.orig_peaks, _ = find_peaks(self.comb,distance=2,height=100,threshold=None, prominence=500)
        except:
            self.min_pkheight = 1000
            self.comb = self.master_comb
            
            
        
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
            If True, the function will only find the LUT (Lookup Table) and return without further processing. Default is False.
        Returns:
        --------
        dict
            A dictionary containing the status of the processing. If successful, returns {"status": "success"}.
            If an error occurs, returns {"status": "failed", "error": str(e), 'channel': self.channel, 'bench': self.bench, 'side': self.side}.
        Raises:
        -------
        Exception
            If any error occurs during the processing, it will be caught and logged, and the function will return a failure status.
        Notes:
        ------
        - The function processes the HDU data by extracting relevant information from the header and data array.
        - It handles different cases based on the presence of specific header keys ('COLOR', 'CAM_NAME').
        - The function reads a trace LUT (Lookup Table) from a JSON file and updates peaks and peak heights arrays.
        - It performs cross-correlation to find offsets and updates the comb and peaks accordingly.
        - The function fits traces from the midpoint forward and backward, updating the trace array.
        - It defines the coordinates of the trace along the x-axis and fits a spline along the x-axis for each fiber.
        - The function interpolates the traces to give x, y positions for each fiber along the naxis.
        Example:
        --------
        result = process_hdu_data(hdu_data, hdu_header, find_LUT=True)
        """
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
                      
            self.naxis1 = self.hdr['NAXIS1']
            self.naxis2 = self.hdr['NAXIS2']
            
            self.benchside = f'{self.bench}{self.side}'
            
            #code which opens the trace LUT, and updates peaks and pkhts arrays to account for dead fibers
            with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'r') as f:
                LUT = json.load(f)
                self.LUT = LUT    
        
            self.master_comb = np.array(LUT['combs'][self.channel.lower()][self.benchside])
            masterpeaks_dict = LUT["fib_pos"][self.channel.lower()][self.benchside]
            
            self.master_peaks = [int(pos) for pos in masterpeaks_dict.values()]
            

            #print(f'Processing {self.channel} channel, {self.bench} bench, {self.side} side')
            #finding the inital comb for the data we are trying to fit
            
            self.comb = self.find_comb(rownum=self.naxis1/2)
            
                
            
            #make sure peaks aren't too close to the edge
            #peaks[np.logical_and(peaks > 20, peaks < 2020)]
            
            
        
            #assert len(self.pkht) == len(self.master_peaks), "Length of peak heights does not match master peaks"
            
            offset, _ = cross_correlate_combs(self.comb, self.master_comb)
            
            if np.abs(offset) > self.offset_cutoff:
                #find a way to print this error to terminal still and the logger
                print(f"Offset of {offset} exceeds cutoff of {self.offset_cutoff} for channel {self.channel} Bench {self.bench} side {self.side}")
                #plt.plot(self.comb, color='blue')
                #plt.plot(self.master_comb, color='green')
                #plt.show()
                logger.error(f"Offset of {offset} exceeds cutoff of {self.offset_cutoff}")
                
                
                #offset=0
                #return 1
            
            #update the peaks to the master peaks
            self.updated_comb = np.array(self.master_comb) + offset
            self.updated_peaks = np.array(self.master_peaks) + offset

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
                    peaks = self.updated_comb
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
                    peaks = self.updated_comb
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
            # Check if traces are generated
            # if self.traces is None or len(self.traces) == 0:
            #     logger.error(f"Traces not generated for channel {self.channel} Bench {self.bench} side {self.side}")
            #     result = {"status": "failed", "error": "Traces not generated", 'channel': self.channel, 'bench': self.bench, 'side': self.side}
            #     return result

        except Exception as e:
            traceback.print_exc()
            result = {"status": "failed", "error":str(e), 'channel': {self.channel}, 'bench': {self.bench}, 'side': {self.side}}
            logger.warning(result)
            return result
            
        result = {"status": "success"}
        return result
    
    def profileFit(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fits the spatial profile of fibers in the data.
        This method normalizes the data, generates masks for bad pixels, and fits 
        the fiber spatial profile using a bspline. It updates the fiber image, 
        profile image, and bad pixel mask attributes of the class.
        Returns:
            tuple: A tuple containing:
                - fiberimg (numpy.ndarray): An array listing the fiber number of each pixel.
                - profimg (numpy.ndarray): An array representing the profile weighting function.
                - bpmask (numpy.ndarray): A boolean array representing the bad pixel mask.
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
            
            
            self.profmask = profmask
            self.inprof = inprof

            fiberimg[np.where(profmask == True)] = index#ifiber
            bpmask[np.where(infmask == False)]   = True
            profimg[inprof] = profimg[inprof] + sset.value(yy[inprof])[0]

            self.bspline_ssets.append(sset)
        
        self.fiberimg, self.profimg, self.bpmask = fiberimg, profimg, bpmask
        
        return (fiberimg, profimg, bpmask)
    
    def saveTraces(self, outfile='LLAMASTrace.pkl')-> None:
        """
        Save trace data to a specified file format.
        Parameters:
        outfile (str): The name of the output file. Default is 'LLAMASTrace.pkl'.
                       Supported formats are '.pkl' for pickle files and '.h5' for HDF5 files.
        This method saves the trace data of the object to a file in the specified format.
        If the output file is a pickle file ('.pkl'), the entire object is serialized using cloudpickle.
        If the output file is an HDF5 file ('.h5'), the trace data and other attributes are saved in datasets.
        The output file is saved in a directory named after the base name of the FITS file with '_traces' appended.
        Raises:
        OSError: If there is an error creating the output directory or writing the file.
        Example:
        >>> obj.saveTraces('output_traces.h5')
        """

        name = os.path.basename(self.fitsfile).replace('.fits', '_traces')
        self.traceloc = os.path.join(OUTPUT_DIR, name)
        os.makedirs(self.traceloc, exist_ok=True)
        outpath = os.path.join(self.traceloc, outfile)
        
            
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
    TraceRay class inherits from TraceLlamas and is used to process and trace data from a FITS file.
    Attributes:
        fitsfile (str): The name of the FITS file being processed.
    Methods:
        __init__(fitsfile: str) -> None:
            Initializes the TraceRay object with the given FITS file.
        process_hdu_data(hdu_data: np.ndarray, hdu_header: dict) -> dict:
            Processes the HDU data and header, performs profile fitting, saves the traces, and returns the result.
    """

    
    def __init__(self, fitsfile: str) -> None:
        super().__init__(fitsfile)
        self.fitsfile = os.path.basename(fitsfile)
        print(f'fitsfile: {self.fitsfile}')
        return
    
    
    def process_hdu_data(self, hdu_data: np.ndarray, hdu_header: dict) -> dict:
        """
        Processes HDU (Header Data Unit) data and performs profile fitting.
        Parameters:
        hdu_data (np.ndarray): The HDU data to be processed.
        hdu_header (dict): The header information associated with the HDU data.
        Returns:
        dict: A dictionary containing the result of the processing, including status and any relevant data.
        This method performs the following steps:
        1. Records the start time of the processing.
        2. Calls the superclass's process_hdu_data method to process the HDU data.
        3. Performs profile fitting by calling the superclass's profileFit method and stores the results in instance variables.
        4. Constructs the output file name based on the fitsfile, channel, bench, and side attributes.
        5. Saves the traces to the constructed output file by calling the superclass's saveTraces method.
        6. Calculates the elapsed time for the processing.
        7. Returns the result of the processing if the status is not "success".
        Note:
        - The method currently has a return statement before checking the result status, which may need to be adjusted.
        """

        start_time = time.time()
        
        result = super().process_hdu_data(hdu_data, hdu_header)

    
        self.fiberimg, self.profimg, self.bpmask = super().profileFit()
        
        origfile = self.fitsfile.split('.fits')[0]
        color = self.channel.lower()
        print(f'color: {color}')
        self.outfile = f'{origfile}_{self.channel.lower()}_{self.bench}_{self.side}_traces.pkl'
        print(f'outfile: {self.outfile}')
        super().saveTraces(self.outfile)
        
        elapsed_time = time.time() - start_time
        if result["status"] != "success":
                return result

def run_ray_tracing(fitsfile: str) -> None:
    """
    Perform ray tracing on a FITS file using parallel processing with Ray.
    This function initializes Ray with the number of available CPU cores, processes
    each HDU (Header Data Unit) in the FITS file using a remote TraceRay actor, and
    monitors the processing status, including CPU usage.
    Parameters:
    fitsfile (str): The path to the FITS file to be processed.
    Returns:
    None
    """


    NUMBER_OF_CORES = multiprocessing.cpu_count() 
    # ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
    # Initialize Ray with logging config
    ray.shutdown()  # Clear any existing Ray instances
    ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
    
    print(f"\nStarting with {NUMBER_OF_CORES} cores available")
    print(f"Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    futures = []
    results = []    
    
    # with fits.open(fitsfile) as hdul:
    hdul = process_fits_by_color(fitsfile)
    hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul[1:] if hdu.data.astype(float) is not None]
        
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

    # with fits.open(fitsfile) as hdul:
    hdul = process_fits_by_color(fitsfile)
    
    if args.channel is not None and 'COLOR' in hdul[1].header:
        hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul if hdu.data is not None and hdu.header['COLOR'].lower() == args.channel.lower()]
    elif args.channel is not None and 'CAM_NAME' in hdul[1].header:
        hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul if hdu.data is not None and hdu.header['CAM_NAME'].split('_')[1].lower() == args.channel.lower()]
    else:
        hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul if hdu.data is not None]



    #with fits.open(fitsfile) as hdul:
    #    hdus = [(hdu.data.astype(float), dict(hdu.header)) for hdu in hdul if hdu.data is not None and hdu.header['COLOR'].lower() == args.channel.lower()]
        
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
    
    
    
    

    
        