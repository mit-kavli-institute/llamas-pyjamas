
"""Module for extracting data from LLAMAS observations.

This module provides functionality for extracting data from LLAMAS (Large Lens Array 
Multi-Object Spectrograph) observations. It includes classes and functions for performing 
optimal and boxcar extractions, saving and loading extraction results, and parallel 
processing using Ray.

Classes:
    ExtractLlamas: A class for extracting data from LLAMAS observations using optimal 
        or boxcar methods.
    ExtractLlamasRay: A Ray remote class for parallel processing of LLAMAS extractions.

Functions:
    save_extractions: Save multiple extraction objects to a single file.
    load_extractions: Load a batch of extraction objects from a file.
    parse_args: Parse command-line arguments for input pkl files.

Example:
    This module can be run as a script to process LLAMAS pkl files using parallel 
    processing with Ray::

        python extractLlamas.py *.pkl
"""
from   astropy.io import fits
import scipy
import numpy as np
import os
from datetime import datetime
from   matplotlib import pyplot as plt
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl
import pickle, cloudpickle
import logging
import argparse, glob
import ray, multiprocessing, psutil
import traceback
from typing import Tuple
import json
import pkg_resources
from pathlib import Path

####################################################################################

from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR, LUT_DIR
from llamas_pyjamas.Trace.traceLlamas import TraceLlamas

ray.init(ignore_reinit_error=True)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, log_filename=f'extractLlamas_{timestamp}.log')

class ExtractLlamas:
    """A class used to extract data from LLAMAS observations.

    This class handles the extraction of spectral data from LLAMAS observations using 
    either optimal or boxcar extraction methods.

    Attributes:
        trace (TraceLlamas): An instance of the TraceLlamas class containing trace information.
        bench (str): The bench identifier from the trace.
        side (str): The side identifier from the trace.
        channel (str): The channel identifier from the trace.
        fitsfile (str): The FITS file associated with the trace.
        counts (np.ndarray): An array to store the extracted counts.
        hdr (dict): Header information from the FITS file.
        frame (np.ndarray): The data frame from the FITS file.
        x (np.ndarray): An array representing the x-axis.
        xshift (np.ndarray): An array to store x-axis shifts.
        wave (np.ndarray): An array to store wavelength information.
        ximage (np.ndarray): An array representing the x-axis image.
        relative_throughput (np.ndarray): Array storing relative throughput for each fiber.
        fiberid (np.ndarray): Array storing fiber IDs.
    """

    def __init__(self,trace: TraceLlamas, hdu_data: np.ndarray, hdr: dict,optimal=True) -> None:
        """Initialize the ExtractLlamas object.

        Args:
            trace (TraceLlamas): An instance of the TraceLlamas class containing trace information.
            hdu_data (np.ndarray): The HDU data array from the FITS file.
            hdr (dict): Header information from the FITS file.
            optimal (bool, optional): If True, use optimal extraction. If False, use boxcar extraction. 
                Defaults to True.

        Returns:
            None
        """

        if (trace is None or hdu_data is None or hdr is None):
            # Instantiate a blank object that can be used for a deep copy
            self.trace = None
            self.bench = None
            self.side = None
            self.channel = None
            self.fitsfile = None     
            self.counts = None
            self.hdr    = None
            self.frame  = None
            self.x      = None
            self.xshift = None
            self.wave   = None
            self.counts = None
            self.ximage = None
            self.relative_throughput = None

        else:
            self.trace = trace
            self.bench = trace.bench
            self.side = trace.side
            self.channel = trace.channel
            self.fitsfile = self.trace.fitsfile
            ##put in a check here for hdu against trace attributes when I have more brain capacity        
            self.counts = np.zeros(shape=(trace.nfibers,trace.naxis1))
            self.hdr    = hdr
            self.frame  = hdu_data.astype(float)
            self.x      = np.arange(trace.naxis1)
            # xshift and wave will be populated only after an arc solution
            self.xshift = np.zeros(shape=(trace.nfibers,trace.naxis1))
            self.wave   = np.zeros(shape=(trace.nfibers,trace.naxis1))
            self.counts = np.zeros(shape=(trace.nfibers,trace.naxis1))
            self.ximage = np.outer(np.ones(trace.naxis2),np.arange(trace.naxis1))
            self.relative_throughput = np.zeros(shape=(trace.nfibers))
            self.fiberid = np.zeros(shape=(trace.nfibers))

            print(f'Optimal {optimal}')
            print(f'bench {self.bench} self.side {self.side} channel {self.channel}')
            
            benchside = str(self.bench) + str(self.side)
            with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'r') as f:
                LUT = json.load(f)
                self.LUT = LUT 
            
            
            # Safely handle missing entries in the LUT
            try:
                if 'dead_fibers' in self.LUT and benchside in self.LUT['dead_fibers']:
                    self.dead_fibers = self.LUT['dead_fibers'][benchside]
                    print(f'Dead fibers: {self.dead_fibers}')
                else:
                    self.dead_fibers = []  # Default to empty list if entry doesn't exist
                    logger.info(f"No dead fibers entry found in LUT for bench={self.bench}, side={self.side}")
            except Exception as e:
                self.dead_fibers = []  # Default to empty list if any error occurs
                logger.warning(f"Error accessing dead fibers in LUT: {e}")


            for ifiber in range(trace.nfibers):    

                if (optimal == True):
                    # Optimally weighted extraction (a la Horne et al ~1986)
                    #logger.info("..Optimally Extracting fiber #{}".format(ifiber))
                    x_spec,f_spec,weights = self.isolateProfile(ifiber)
                    if x_spec is None:
                        continue
                
                    extracted = np.zeros(self.trace.naxis1)
                    for i in range(self.trace.naxis1):
                        thisx = (x_spec == i)
                        if np.nansum(thisx) > 0:
                            extracted[i] = np.nansum(f_spec[thisx]*weights[thisx])/np.nansum(weights[thisx])
                        #handles case where there are no elements
                        else:
                            extracted[i] = 0.0

                    self.counts[ifiber,:] = extracted
                
                elif optimal == False:
                    # Boxcar Extraction - fast!
                    logger.info("..Boxcar extracting fiber #{}".format(ifiber))
                    x_spec,f_spec,weights = self.isolateProfile(ifiber, boxcar=True)
                    
                    if x_spec is None:
                        continue
                
                    extracted = np.zeros(self.trace.naxis1)
                    for i in range(self.trace.naxis1):
                        thisx = (x_spec == i)
                        
                        if np.nansum(thisx) > 0:
                            extracted[i] = np.nansum(f_spec[thisx]*weights[thisx])/np.nansum(weights[thisx])
                        #handles case where there are no elements
                        else:
                            extracted[i] = 0.0

                    self.counts[ifiber,:] = extracted
            self.old_count_shape = self.counts.shape
            logger.info(f'Benchside {benchside} counts shape {self.counts.shape}')
            # Process the dead fibers by inserting dummy arrays at specific indices
            # if self.dead_fibers:
            #     logger.info(f'Processing dead fibers: {self.dead_fibers}')
                

            #     # Sort dead fibers in descending order to avoid index shifting
            #     # when we insert multiple rows
            #     for dead_idx in sorted(self.dead_fibers, reverse=True):
            #         # Create a row of zeros for the dead fiber
            #         dummy_counts = np.zeros(trace.naxis1)
                    
            #         self.counts = np.insert(self.counts, dead_idx, dummy_counts, axis=0)
            
            if self.dead_fibers:
                logger.info(f'Processing dead fibers: {self.dead_fibers}')
                
                # Create new array with space for dead fibers
                total_fibers = trace.nfibers + len(self.dead_fibers)
                new_counts = np.zeros((total_fibers, trace.naxis1))
                
                # Copy data from original counts array to correct positions in new array
                current_idx = 0
                dead_set = set(self.dead_fibers)  # Convert to set for faster lookup
                
                for i in range(total_fibers):
                    if i in dead_set:
                        # Leave zeros for dead fiber positions
                        logger.info(f"Inserting dead fiber at index {i}")
                        continue
                    else:
                        # Copy data from original array if position exists
                        if current_idx < len(self.counts):
                            new_counts[i] = self.counts[current_idx]
                            current_idx += 1
                
                # Replace the counts array with the new one
                self.counts = new_counts
                logger.info(f'New counts shape after dead fiber insertion: {self.counts.shape}')

                    
                
                
                    

    def isolateProfile(self,ifiber, boxcar=False)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Isolate the profile for a given fiber.

        Args:
            ifiber (int): The fiber index to isolate the profile for.
            boxcar (bool, optional): If True, use boxcar extraction. If False, use profile 
                extraction. Defaults to False.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - x_spec (np.ndarray or None): The x-coordinates of the spectrum for the given fiber.
                - f_spec (np.ndarray or None): The flux values of the spectrum for the given fiber.
                - weights (np.ndarray or None): The weights for the spectrum extraction, either 
                  boxcar or profile-based.

        Note:
            If no profile is found for the given fiber, the function will return (None, None, None) 
            and log a warning.
        """

        #profile  = self.trace.profimg[ifiber]
        inprofile = self.trace.fiberimg == ifiber
        profile = self.trace.profimg[self.trace.fiberimg == ifiber]
            
        #inprof = np.where(profile > 0)
        if inprofile.size == 0:
            logger.warning("No profile for fiber #{}".format(ifiber))
            return None,None,None
        
        x_spec = self.ximage[inprofile]
        f_spec = self.frame[inprofile]
        if boxcar == True:
            weights = np.where(inprofile, 1, 0)[inprofile]#[weights]
            
        elif boxcar == False:
            weights = self.trace.profimg[inprofile]#self.trace.profimg[inprofile]
        
        return x_spec,f_spec,weights


    def saveExtraction(self, save_dir: str)-> None:
        """Save the current extraction object to a file in the specified directory.

        This method constructs the output file path using the object's attributes
        (channel, side, bench) and saves the object using cloudpickle.

        Args:
            save_dir (str): The directory where the extraction file will be saved.
                Note that this argument is overridden and the file is always saved 
                in the 'output' directory relative to the script's location.

        Returns:
            None
        """

        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        outfile = f'LLAMASExtract_{self.channel}_{self.side}_{self.bench}.pkl'
        outpath = os.path.join(save_dir, outfile)
        logger.info(f'outpath {outpath}')
        with open(outpath,'wb') as fp:
            cloudpickle.dump(self, fp)
        return

    def loadExtraction(infile: str)-> object:
        """Load an object from a pickle file.

        Args:
            infile (str): The path to the input file containing the pickled object.

        Returns:
            object: The object loaded from the pickle file.
        """

        with open(infile,'r+b') as fp:
            object = pickle.load(fp)
        return(object)


def save_extractions(extraction_list, savefile=None, save_dir=None, prefix='LLAMASExtract_batch')-> str:
    """Save multiple extraction objects to single file.

    Args:
        extraction_list (list): List of extraction objects to save.
        savefile (str, optional): Specific filename to use. If None, a timestamped 
            filename will be generated. Defaults to None.
        save_dir (str, optional): Directory to save the file. If None, uses the 
            default output directory. Defaults to None.
        prefix (str, optional): Prefix for the filename when savefile is None. 
            Defaults to 'LLAMASExtract_batch'.

    Returns:
        str: Path to the saved file.
    """
    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
    os.makedirs(save_dir, exist_ok=True)
    
    # Create metadata for each extraction
    batch_data = {
        'extractions': extraction_list,
        'metadata': [{
            'channel': ext.channel,
            'bench': ext.bench,
            'side': ext.side,
            'nfibers': ext.trace.nfibers
        } for ext in extraction_list]
    }
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = f'{prefix}_{timestamp}.pkl'
    if (savefile != None):
        outpath = os.path.join(save_dir, savefile)
    else:   
        outpath = os.path.join(save_dir, outfile)
    
    logger.info(f'Saving batch extraction to: {outpath}')
    with open(outpath, 'wb') as fp:
        cloudpickle.dump(batch_data, fp)
    return outpath

@staticmethod
def load_extractions(infile: str)-> Tuple[list, list]:
    """Load batch of extraction objects.

    Args:
        infile (str): Path to the file containing the batch extraction data.

    Returns:
        tuple[list, list]: A tuple containing:
            - extractions (list): List of extraction objects.
            - metadata (list): List of metadata dictionaries for each extraction.
    """
    with open(infile, 'rb') as fp:
        batch_data = cloudpickle.load(fp)
    
    logger.info(f"Loaded {len(batch_data['extractions'])} extractions")
    return batch_data['extractions'], batch_data['metadata']

@ray.remote
class ExtractLlamasRay(ExtractLlamas):
    """Ray remote class for parallel processing of LLAMAS extractions.

    ExtractLlamasRay is a subclass of ExtractLlamas that handles the extraction 
    process for LLAMAS data using Ray for parallel processing.

    Attributes:
        files (list): A list of files to be processed.
    """

    
    def __init__(self, files) -> None:
        """Initialize the ExtractLlamasRay instance.

        Args:
            files (list): A list of files to be processed.

        Returns:
            None
        """
        self.files = files
        pass
        
   
    def process_extractions(self, tracepkl: "TraceLlamas") -> None:
        """Process the extraction of LLAMAS traces from a given pickle file.

        Args:
            tracepkl (TraceLlamas): The path to the pickle file containing the LLAMAS trace data.

        Returns:
            None
        """

        with open(tracepkl, "rb") as tracer:
            trace = pickle.load(tracer)
        
        extraction = super.__init__(trace)
        extraction.saveExtraction()
        return

def parse_args()-> list:
    """Parse command-line arguments to process LLAMAS pkl files.

    This function sets up an argument parser to accept one or more file paths
    (with support for wildcards like *.pkl), expands the wildcards, validates
    the files, and returns a list of valid .pkl files.

    Returns:
        list: A list of valid .pkl file paths.

    Raises:
        ValueError: If no .pkl files are found.
    """

    parser = argparse.ArgumentParser(description='Process LLAMAS pkl files.')
    parser.add_argument('files', nargs='+', help='Path to input pkl files (supports wildcards like *.pkl)')
    args = parser.parse_args()
    
    # Expand wildcards and validate files
    pkl_files = []
    for pattern in args.files:
        matched_files = glob.glob(pattern)
        if not matched_files:
            print(f"Warning: No files found matching pattern {pattern}")
            continue
        pkl_files.extend([f for f in matched_files if f.endswith('.pkl')])
    
    if not pkl_files:
        raise ValueError("No .pkl files found!")
        
    return pkl_files


if __name__ == '__main__':
    # Example of how to run the extraction

   ##Need to edit this and the remote class so that it runs the extraction through ray not just multiple files at once.
    files = parse_args()
    
    NUMBER_OF_CORES = multiprocessing.cpu_count() 
    ray.init(ignore_reinit_error=True, num_cpus=NUMBER_OF_CORES)
    
    print(f"\nStarting with {NUMBER_OF_CORES} cores available")
    print(f"Current CPU Usage: {psutil.cpu_percent(interval=1)}%")
    
    futures = []
    results = []
    
    extraction_processors = [ExtractLlamasRay.remote(file) for file in files]
    
    print(f"\nProcessing {len(files)} files with {NUMBER_OF_CORES} cores")
    for processor in extraction_processors:
        futures.append(processor.process_extractions.remote())
    
    #Monitor processing
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
    