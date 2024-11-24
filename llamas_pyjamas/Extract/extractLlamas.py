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
####################################################################################
from llamas_pyjamas.Utils.utils import setup_logger


# Enable DEBUG for your specific logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, log_filename=f'extractLlamas_{timestamp}.log')

class ExtractLlamas:

    def __init__(self,trace):
        self.trace = trace
        self.bench = trace.bench
        self.side = trace.side
        self.channel = trace.channel
        
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
                if np.sum(thisx) > 0:
                    extracted[i] = np.sum(f_spec[thisx]*invvar[thisx])/np.sum(invvar[thisx])
                #handles case where there are no elements
                else:
                    extracted[i] = 0.0

            self.counts[ifiber,:] = extracted
            
            
    def isolateProfile(self,ifiber):
        profile  = self.trace.profimg[ifiber]
            
        inprof = np.where(profile > 0)
        if inprof[0].size == 0:
            logger.warning("No profile for fiber #{}".format(ifiber))
            return None,None,None
        
        x_spec = self.xx[ifiber][inprof]
        f_spec = self.frame[ifiber][inprof]
        invvar = profile[inprof]
        
        return x_spec,f_spec,invvar


    def saveExtraction(self):
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

def parse_args():
    parser = argparse.ArgumentParser(description='Process LLAMAS pkl files.')
    parser.add_argument('files', nargs='+', help='Path to input pkl files (supports wildcards like *.pkl)')
    args = parser.parse_args()
    
    # Expand wildcards and validate files
    pkl_files = []
    for pattern in args.files:
        matched_files = glob(pattern)
        if not matched_files:
            print(f"Warning: No files found matching pattern {pattern}")
            continue
        pkl_files.extend([f for f in matched_files if f.endswith('.pkl')])
    
    if not pkl_files:
        raise ValueError("No .pkl files found!")
        
    return pkl_files



if __name__ == '__main__':
    # Example of how to run the extraction
   
    files = parse_args()