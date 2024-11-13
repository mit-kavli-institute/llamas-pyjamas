from   astropy.io import fits
import scipy
import numpy as np
import pyds9
import time
from   matplotlib import pyplot as plt
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl
import pickle

####################################################################################

class ExtractLlamas:

    def __init__(self,fitsfile,trace):

        self.counts = np.zeros(shape=(trace.nfibers,trace.naxis1))
        self.bench = trace.spectrograph

        with fits.open(fitsfile) as hdu:
            self.hdr   = hdu[0].header
            frame = hdu[0].data.astype(np.float)
        
        x  = np.arange(trace.naxis1)
        xx = np.outer(np.ones(trace.naxis2),np.arange(trace.naxis1))

        for ifiber in range(trace.nfibers):

            print("..Extracting fiber #{}".format(ifiber))
            
            profimg  = trace.fiberProfileImg(ifiber)
            
            inprof = np.where(profimg > 0)
            x_spec = xx[inprof]
            f_spec = frame[inprof]
            invvar = profimg[inprof]
            
            extracted = np.zeros(trace.naxis1)
            for i in range(trace.naxis1):
                thisx        = np.where(x_spec == i)
                extracted[i] = np.sum(f_spec[thisx]*invvar[thisx])/np.sum(invvar[thisx])

            self.counts[ifiber,:] = extracted


    def saveExtraction(self, outfile='LLAMASExtract.pkl'):
        with open(outfile,'wb') as fp:
            pickle.dump(self, fp)

    def loadExtraction(infile):
        with open(infile,'rb') as fp:
            object = pickle.load(fp)
        return(object)

