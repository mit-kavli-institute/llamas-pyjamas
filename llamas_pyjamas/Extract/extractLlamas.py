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
####################################################################################

from llamas_pyjamas.Utils.utils import setup_logger

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, log_filename=f'extractLlamas_{timestamp}.log')

class ExtractLlamas:

    def __init__(self,trace: "TraceLlamas") -> None:
        self.trace = trace
        self.bench = trace.bench
        self.side = trace.side
        self.channel = trace.channel
        self.fitsfile = self.trace.fitsfile
        
        if self.channel == 'red':
            logger.warning("Red channel may not extract correctly")
        
        self.counts = np.zeros(shape=(trace.nfibers,trace.naxis1))
        
        self.hdr   = trace.hdr
        self.frame = trace.data.astype(float)
        
        self.x  = np.arange(trace.naxis1)
        self.xx = np.outer(np.ones(trace.naxis2),np.arange(trace.naxis1))

        for ifiber in range(trace.nfibers):

            logger.info("..Extracting fiber #{}".format(ifiber))
            x_spec,f_spec,invvar = self.isolateProfile(ifiber)
            if x_spec is None:
                continue
            
            extracted = np.zeros(self.trace.naxis1)
            for i in range(self.trace.naxis1):
                thisx = (x_spec == i)
                if np.nansum(thisx) > 0:
                    extracted[i] = np.nansum(f_spec[thisx]*invvar[thisx])/np.nansum(invvar[thisx])
                #handles case where there are no elements
                else:
                    extracted[i] = 0.0

            self.counts[ifiber,:] = extracted
            
            
    def isolateProfile(self,ifiber):
        #profile  = self.trace.profimg[ifiber]
        weights = self.trace.fiberimg == ifiber
        profile = self.trace.profimg[self.trace.fiberimg == ifiber]
            
        #inprof = np.where(profile > 0)
        if weights.size == 0:
            logger.warning("No profile for fiber #{}".format(ifiber))
            return None,None,None
        
        x_spec = self.xx[weights]
        f_spec = self.frame[weights]
        invvar = self.trace.profimg[weights]
        
        return x_spec,f_spec,invvar


    def saveExtraction(self, save_dir):
        save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output')
        outfile = f'LLAMASExtract_{self.channel}_{self.side}_{self.bench}.pkl'
        outpath = os.path.join(save_dir, outfile)
        logger.info(f'outpath {outpath}')
        with open(outpath,'wb') as fp:
            cloudpickle.dump(self, fp)
        return

    def loadExtraction(infile):
        with open(infile,'rb') as fp:
            object = pickle.load(fp)
        return(object)
    
def save_extractions(extraction_list, save_dir=None, prefix='LLAMASExtract_batch'):
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
    outpath = os.path.join(save_dir, outfile)
    
    logger.info(f'Saving batch extraction to: {outpath}')
    with open(outpath, 'wb') as fp:
        cloudpickle.dump(batch_data, fp)
    return outpath

@staticmethod
def load_extractions(infile):
    """Load batch of extraction objects"""
    with open(infile, 'rb') as fp:
        batch_data = cloudpickle.load(fp)
    
    logger.info(f"Loaded {len(batch_data['extractions'])} extractions")
    return batch_data['extractions'], batch_data['metadata']

@ray.remote
class ExtractLlamasRay(ExtractLlamas):
    
    def __init__(self, files) -> None:
        self.files = files
        pass
        
   
    def process_extractions(self, tracepkl: "TraceLlamas") -> None:
        with open(tracepkl, "rb") as tracer:
            trace = pickle.load(tracer)
        
        extraction = super.__init__(trace)
        extraction.saveExtraction()
        return

def parse_args():
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

def main(files):
    pass

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
    