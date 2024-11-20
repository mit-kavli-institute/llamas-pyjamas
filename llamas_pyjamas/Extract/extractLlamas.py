from   astropy.io import fits
import scipy
import numpy as np
import time
from   matplotlib import pyplot as plt
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl
import pickle
import logging
####################################################################################

# Enable DEBUG for your specific logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class ExtractLlamas:

    def __init__(self,trace):

        self.bench = trace.bench
        self.side = trace.side
        self.channel = trace.channel
        
        if self.channel == 'red':
            logger.warning("Red channel not implemented yet...ending extraction")
            return
        
        self.counts = np.zeros(shape=(trace.nfibers,trace.naxis1))
        
        self.hdr   = trace.hdr
        frame = trace.data.astype(float)
        
        x  = np.arange(trace.naxis1)
        xx = np.outer(np.ones(trace.naxis2),np.arange(trace.naxis1))

        for ifiber in range(trace.nfibers):

            logger.info("..Extracting fiber #{}".format(ifiber))
            
            profile  = trace.profimg[ifiber]
            
            inprof = np.where(profile > 0)
            if inprof[0].size == 0:
                logger.warning("No profile for fiber #{}".format(ifiber))
                continue
            x_spec = xx[ifiber][inprof]
            f_spec = frame[ifiber][inprof]
            invvar = profile[inprof]
            
            extracted = np.zeros(trace.naxis1)
            for i in range(trace.naxis1):
                thisx = (x_spec == i)
                if np.sum(thisx) > 0:
                    extracted[i] = np.sum(f_spec[thisx]*invvar[thisx])/np.sum(invvar[thisx])
                #handles case where there are no elements
                else:
                    extracted[i] = 0.0

            self.counts[ifiber,:] = extracted


    def saveExtraction(self):
        outfile = f'LLAMASExtract_{self.channel}_{self.side}_{self.bench}.pkl'
        with open(outfile,'wb') as fp:
            pickle.dump(self, fp)

    def loadExtraction(infile):
        with open(infile,'rb') as fp:
            object = pickle.load(fp)
        return(object)

