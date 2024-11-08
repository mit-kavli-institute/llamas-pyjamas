from   astropy.io import fits
import scipy
import numpy as np
import extractLlamas 
import pickle
from   matplotlib import pyplot as plt
from   pypeit.core.arc import detect_peaks,iter_continuum
from   pypeit.core import pydl
from   pypeit.par import pypeitpar
from   pypeit.core.wavecal import autoid


###############################################################################3

class ArcLlamas:

    def __init__(self,arc_fitsfile,trace):

        print("...ExtractingArcSpectrum...")
        self.arcspec = extractLlamas.ExtractLlamas(arc_fitsfile,trace)
        
        print("...Removing underlying continuum...")
        self.removeArcContinuum()
        
        llamas_lamps = ['ArI','NeI','KrI']
        print("...Setting up wavelength fitting parameters...")
        arcsol_params = pypeitpar.WavelengthSolutionPar(lamps=llamas_lamps)

        arc_input = np.transpose(self.arcspec.counts)


        if (True):
            print("...Brute force wavelength fit (only do this if no reference arc available...")
            self.arcsol = autoid.HolyGrail(arc_input, par=arcsol_params, \
                                      islinelist=False, use_unknowns=False)

        if (False):
            f = open("wavesol.pickle","wb")
            pickle.dump(self.arcsol,f)
            f.close()

            #
            # Note: to load this solution:
            #   import pickle
            #   f = open("wavesol.pickle","rb")
            #   tt = pickle.load(f)
            #   f.close()
            #   
            #   Now you can access the object as tt
            
    def removeArcContinuum(self):

        nfibers,naxis1 = self.arcspec.counts.shape

        for ifiber in range(nfibers):
            contin = iter_continuum(self.arcspec.counts[ifiber,:],npoly=12)
            self.arcspec.counts[ifiber,:] -= contin[0]

        

