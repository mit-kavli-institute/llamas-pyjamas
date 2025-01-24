
"""
Module: extractLlamas
This module provides functionality for extracting data from LLAMAS (Large Lens Array Multi-Object Spectrograph) 
observations. It includes classes and functions for performing optimal and boxcar extractions, saving and loading 
extraction results, and parallel processing using Ray.
Classes:
    ExtractLlamas: 
        A class for extracting data from LLAMAS observations using optimal or boxcar methods.
    ExtractLlamasRay: 
        A Ray remote class for parallel processing of LLAMAS extractions.
Functions:
    save_extractions(extraction_list, savefile=None, save_dir=None, prefix='LLAMASExtract_batch'):
        Save multiple extraction objects to a single file.
    load_extractions(infile):
        Load a batch of extraction objects from a file.
    parse_args():
        Parse command-line arguments for input pkl files.
Usage:
    This module can be run as a script to process LLAMAS pkl files using parallel processing with Ray.
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

import pkg_resources
from pathlib import Path

####################################################################################

from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR
from llamas_pyjamas.Trace.traceLlamas import TraceLlamas




ray.init(ignore_reinit_error=True)


timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, log_filename=f'extractLlamas_{timestamp}.log')

class ExtractLlamas:
    """
    A class used to extract data from Llamas.
    Attributes
    ----------
    trace : TraceLlamas
        An instance of the TraceLlamas class containing trace information.
    bench : str
        The bench identifier from the trace.
    side : str
        The side identifier from the trace.
    channel : str
        The channel identifier from the trace.
    fitsfile : str
        The FITS file associated with the trace.
    counts : np.ndarray
        An array to store the extracted counts.
    hdr : dict
        Header information from the FITS file.
    frame : np.ndarray
        The data frame from the FITS file.
    x : np.ndarray
        An array representing the x-axis.
    xshift : np.ndarray
        An array to store x-axis shifts.
    wave : np.ndarray
        An array to store wavelength information.
    ximage : np.ndarray
        An array representing the x-axis image.
    Methods
    -------
    __init__(trace: "TraceLlamas", hdu_data: np.ndarray, hdr: dict, optimal=True) -> None
        Initializes the ExtractLlamas object with trace, hdu_data, hdr, and optimal extraction flag.
    isolateProfile(ifiber, boxcar=False)
        Isolates the profile for a given fiber and returns the x_spec, f_spec, and weights.
    saveExtraction(save_dir)
        Saves the extracted data to a specified directory.
    loadExtraction(infile)
        Loads the extracted data from a specified file.
    """

    def __init__(self,trace: TraceLlamas, hdu_data: np.ndarray, hdr: dict,optimal=True) -> None:
        self.trace = trace
        self.bench = trace.bench
        self.side = trace.side
        self.channel = trace.channel
        self.fitsfile = self.trace.fitsfile
        print(f'Optimal {optimal}')
        
        ##put in a check here for hdu against trace attributes when I have more brain capacity
        
        self.counts = np.zeros(shape=(trace.nfibers,trace.naxis1))
        
        self.hdr    = hdr#trace.hdr
        self.frame  = hdu_data.astype(float)
        self.x      = np.arange(trace.naxis1)

        # xshift and wave will be populated only after an arc solution
        self.xshift = np.zeros(shape=(trace.nfibers,trace.naxis1))
        self.wave   = np.zeros(shape=(trace.nfibers,trace.naxis1))
        self.counts = np.zeros(shape=(trace.nfibers,trace.naxis1))

        self.ximage = np.outer(np.ones(trace.naxis2),np.arange(trace.naxis1))

        for ifiber in range(trace.nfibers):

            if (optimal == True):
                # Optimally weighted extraction (a la Horne et al ~1986)
                logger.info("..Optimally Extracting fiber #{}".format(ifiber))
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
                

    def isolateProfile(self,ifiber, boxcar=False)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Isolate the profile for a given fiber.
        Parameters:
        -----------
        ifiber : int
            The fiber index to isolate the profile for.
        boxcar : bool, optional
            If True, use boxcar extraction. If False, use profile extraction. Default is False.
        Returns:
        --------
        x_spec : numpy.ndarray or None
            The x-coordinates of the spectrum for the given fiber.
        f_spec : numpy.ndarray or None
            The flux values of the spectrum for the given fiber.
        weights : numpy.ndarray or None
            The weights for the spectrum extraction, either boxcar or profile-based.
        Notes:
        ------
        If no profile is found for the given fiber, the function will return (None, None, None) and log a warning.
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
        """
        Save the current extraction object to a file in the specified directory.
        This method constructs the output file path using the object's attributes
        (channel, side, bench) and saves the object using cloudpickle.
        Args:
            save_dir (str): The directory where the extraction file will be saved.
                    Note that this argument is overridden and the file is
                    always saved in the 'output' directory relative to the
                    script's location.
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
        """
        Load an object from a pickle file.
        Args:
            infile (str): The path to the input file containing the pickled object.
        Returns:
            object: The object loaded from the pickle file.
        """

        with open(infile,'rb') as fp:
            object = pickle.load(fp)
        return(object)


def save_extractions(extraction_list, savefile=None, save_dir=None, prefix='LLAMASExtract_batch')-> str:
    """Save multiple extraction objects to single file"""
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
    """Load batch of extraction objects"""
    with open(infile, 'rb') as fp:
        batch_data = cloudpickle.load(fp)
    
    logger.info(f"Loaded {len(batch_data['extractions'])} extractions")
    return batch_data['extractions'], batch_data['metadata']

@ray.remote
class ExtractLlamasRay(ExtractLlamas):
    """
    ExtractLlamasRay is a subclass of ExtractLlamas that handles the extraction process for llama data using Ray.
    Attributes:
        files (list): A list of files to be processed.
    Methods:
        __init__(files):
            Initializes the ExtractLlamasRay instance with the provided files.
        process_extractions(tracepkl: "TraceLlamas") -> None:
            Processes the extractions from the given TraceLlamas pickle file.
    """

    
    def __init__(self, files) -> None:
        self.files = files
        pass
        
   
    def process_extractions(self, tracepkl: "TraceLlamas") -> None:
        """
        Processes the extraction of llama traces from a given pickle file.
        Args:
            tracepkl (TraceLlamas): The path to the pickle file containing the llama trace data.
        Returns:
            None
        """

        with open(tracepkl, "rb") as tracer:
            trace = pickle.load(tracer)
        
        extraction = super.__init__(trace)
        extraction.saveExtraction()
        return

def parse_args()-> list:
    """
    Parse command-line arguments to process LLAMAS pkl files.
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
    