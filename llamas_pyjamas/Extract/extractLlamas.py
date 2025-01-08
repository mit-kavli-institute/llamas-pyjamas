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

####################################################################################

from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.config import BASE_DIR, OUTPUT_DIR, DATA_DIR, CALIB_DIR

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
logger = setup_logger(__name__, log_filename=f'extractLlamas_{timestamp}.log')

class ExtractLlamas:

    def __init__(self,trace: "TraceLlamas", hdu_data: np.ndarray, hdr: dict,optimal=True) -> None:
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
                

    def isolateProfile(self,ifiber, boxcar=False):
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

####################################################################################

def ExtractLlamasCube(infits, tracefits, optimal=True):

    assert infits.endswith('.fits'), 'File must be a .fits file'  
    hdu = fits.open(infits)

    # Find the trace files
    basefile = os.path.basename(tracefits).split('.fits')[0]
    trace_files = glob.glob(os.path.join(OUTPUT_DIR, f'{basefile}*traces.pkl'))
    extraction_file = os.path.basename(infits).split('mef.fits')[0] + 'extract.pkl'

    if len(trace_files) == 0:
        logger.error("No trace files found for the indicated file root!")
        return None
    
    hdu_trace_pairs = match_hdu_to_traces(hdu, trace_files)
    print(hdu_trace_pairs)

    extraction_list = []

    print(f"Saving extractions to {extraction_file}")

    counter = 1
    for hdu_index, file in hdu_trace_pairs:

        print(f"Extracting extension number {counter} of 24")
        hdr = hdu[hdu_index].header 
        bias = np.nanmedian(hdu[hdu_index].data.astype(float))  
        
        try:
            with open(file, mode='rb') as f:
                tracer = pickle.load(f)
    
            extraction = ExtractLlamas(tracer, hdu[hdu_index].data.astype(float)-bias, hdu[hdu_index].header)
            extraction_list.append(extraction)
            
        except Exception as e:
            print(f"Error extracting trace from {file}")
            print(traceback.format_exc())
        counter += 1
        
    print(f'Extraction list = {extraction_list}')        
    filename = save_extractions(extraction_list, savefile=extraction_file)
    print(f'extraction saved filename = {filename}')

    return None


def save_extractions(extraction_list, savefile=None, save_dir=None, prefix='LLAMASExtract_batch'):
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

def match_hdu_to_traces(hdu_list, trace_files):
    """Match HDU extensions to their corresponding trace files"""
    matches = []
    
    # Skip primary HDU (index 0)
    for idx in range(1, len(hdu_list)):
        header = hdu_list[idx].header
        
        # Get color and benchside from header
        if 'COLOR' in header:
            color = header['COLOR'].lower()
            bench = header['BENCH']
            side = header['SIDE']
        else:
            camname = header['CAM_NAME']
            color = camname.split('_')[1].lower()
            bench = camname.split('_')[0][0]
            side = camname.split('_')[0][1]
            
        benchside = f"{bench}{side}"
        pattern = f"{color}_{bench}_{side}_traces"
        
        # Find matching trace file
        matching_trace = next(
            (tf for tf in trace_files 
             if pattern in os.path.basename(tf)),
            None
        )
        #print(f'HDU {idx}: {color} {benchside} -> {matching_trace}')
        if matching_trace:
            matches.append((idx, matching_trace))
        else:
            print(f"No matching trace found for HDU {idx}: {color} {benchside}")
            
    return matches

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
    