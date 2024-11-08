from   astropy.io import fits
import scipy
import numpy as np
import pyds9
import time
from   matplotlib import pyplot as plt
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl


###############################################################################3

class TraceLlamas:

    def __init__(self,flat_fitsfile,spectrograph=None,channel=None):
        

        with fits.open(flat_fitsfile) as hdu:
            hdr = hdu[0].header
            raw = hdu[0].data

        self.data = raw.astype(np.float)
            
        self.naxis1 = hdr['naxis1']
        self.naxis2 = hdr['naxis2']

        # Take a vertical slice through traces at center of array, and find the "valleys"
        # corresponding to the continuum

        middle_row = int(self.naxis1/2)
        tslice = np.sum(raw[:,middle_row-5:middle_row+4],axis=1).astype(np.float)

        tmp            = detect_peaks(tslice,mpd=2,threshold=10,show=False,valley=True)
        valley_indices = np.ndarray(len(tmp))
        valley_depths  = np.ndarray(len(tmp))
        valley_indices = tmp.astype(np.float)
        valley_depths  = tslice[tmp].astype(np.float)
        invvar         = np.ones(len(tmp))
        
        # Fit out the scatterd light continuum, using uniform weighting

        sset = pydl.bspline(valley_indices,everyn=2)
        res, yfit = sset.fit(valley_indices, valley_depths, invvar)
        
        x_model = np.arange(self.naxis2).astype(np.float)
        y_model = sset.value(x_model)[0]
        
        # Now find trace peaks

        peaks = detect_peaks(tslice-y_model,mpd=2,threshold=10,mph=1500,show=False)

        self.nfibers  = len(peaks)
        self.xmin     = 100
        self.xmax     = self.naxis1-self.xmin
        self.fitspace = 5

        # We will explicitly fit a vertical slice for peak positions every
        # self.fitspace pixels in the horizontal direction
        n_tracefit = np.floor((self.xmax-self.xmin)/self.fitspace).astype(np.int)
        xtrace = self.xmin + self.fitspace * np.arange(n_tracefit)

        tracearr = np.zeros(shape=(self.nfibers,n_tracefit))
        print("NFibers = {}".format(self.nfibers))
        
        itrace = 0

        print("...Generating the trace fitting grid...")
        
        # Loop over each slice to populate tracearr, which is the fitting grid

        for xtmp in xtrace:
            ytrace = np.median(raw[:,xtmp.astype(np.int)-7:xtmp.astype(np.int)+7],axis=1)
            
            valleys = detect_peaks(ytrace,mpd=2,show=False,valley=True)
            nvalley = len(valleys)

            valley_indices = valleys.astype(np.float)
            valley_depths  = ytrace[valleys].astype(np.float)
            invvar         = np.ones(nvalley)

            # Fit out scattered light / continuum at each x
            
            sset = pydl.bspline(valley_indices,everyn=2)
            res, yfit = sset.fit(valley_indices, valley_depths, invvar)
            
            y_model = sset.value(x_model)[0]

            comb = ytrace.astype(np.float)-y_model

            # This is a first guess at peaks, which we will then centroid
            # Don't guess at every location, just the first, and then 
            # firts guess is the location of the last column
            if (itrace == 0):
                peaks = detect_peaks(comb,mpd=2,mph=40,show=False,valley=False)
            else:
                peaks = tracearr[:,itrace-1].astype(np.int)

            # Centroid all of the peaks for this comb
            ifiber = 0
            for pk_guess in peaks:
                pk_centroid = \
                    np.sum(np.multiply(comb[pk_guess-2:pk_guess+3],pk_guess-2+np.arange(5))) \
                    / np.sum(comb[pk_guess-2:pk_guess+3])
                if (np.abs(pk_centroid-pk_guess) < 1.5):
                    tracearr[ifiber,itrace] = pk_centroid
                else:
                    tracearr[ifiber,itrace] = pk_guess
                ifiber += 1
        
            itrace += 1

        # Solve a trace set for all of the fibers
        print("...Solving for the trace functions...")
        self.xtracefit = np.outer(np.ones(ifiber),xtrace)
        self.tracearr  = tracearr
        self.tset      = pydl.xy2traceset(self.xtracefit,self.tracearr,maxdev=0.5)

        # This is a QA plot for tracing the profile centroids
        if (False):
            for i in range(ifiber):
                plt.plot(xtrace, tracearr[i,:])
                plt.plot(xtrace,tset.yfit.T[:,i])

        # Full traces from solution for all pixels
        x2          = np.outer(np.ones(ifiber),np.arange(self.naxis1))
        self.traces = pydl.traceset2xy(self.tset,xpos=x2)[1]

        
    def profileFit(self):
        
        ref = self.data[12,:]

        # Use a working copy of "data" so as not to overwrite that with normalized data
        data_work = np.copy(self.data) 
        for i in range(self.naxis2):
            if (i != 12):
                data_work[i,:] = self.data[i,:] - ref

        fiberimg = np.zeros(self.data.shape,dtype=np.int)   # Lists the fiber # of each pixel
        profimg  = np.zeros(self.data.shape,dtype=np.float) # Profile weighting function
        bpmask   = np.zeros(self.data.shape,dtype=np.bool)  # bad pixel mask
        
        print("...Solving profile weights for Fiber #")
        for ifiber in range(self.nfibers):

            print(".....{}".format(ifiber))
            ytrace = self.traces[ifiber,:]

            # Generate a curved "y" image
            yy = np.outer(np.arange(self.naxis2),np.ones(self.naxis1)) \
                - np.outer(np.ones(self.naxis2),ytrace)
            
            # Normalize out the spectral shape of the lamp for profile fitting
            for i in range(self.naxis1):
                norm = np.sum(data_work[np.where(np.abs(yy[:,i]) < 2.0),i])
                data_work[:,i] = data_work[:,i] / norm

            # Generate a mask of pixels that are
            # (a) within 4 pixels of the profile center for this fiber and
            # (b) not NaNs or Infs
            # Also generate an inverse variance array that is presently flat weighting

            infmask = np.ones(data_work.shape,dtype=np.bool)
            NaNmask = np.ones(data_work.shape,dtype=np.bool)
            profmask = np.zeros(data_work.shape,dtype=np.bool)
            invvar = np.ones(data_work.shape,dtype=np.float)
            
            infmask[np.where(np.isinf(data_work))]  = False
            NaNmask[np.where(np.isnan(data_work))] = False
            profmask[np.where(np.abs(yy) < 3)] = True

            inprof = np.where(infmask & profmask & NaNmask)

            # Fit the fiber spatial profile with a bspline
            sset,outmask = pydl.iterfit(yy[inprof],data_work[inprof],maxiter=6, \
                        invvar=invvar[inprof],kwargs_bspline={'bkspace':0.33})

            # QA plot showing pixel vals and fit
            if (ifiber == 130):
#                plt.clf()
                xx = -3.0 + np.arange(100) * 6.0/100.0
                plt.plot(yy[inprof],data_work[inprof],',')
                plt.plot(xx,sset.value(xx)[0])
                plt.xlim([-5,5])
                plt.ylim([-0.05,0.5])
                plt.show()
                
            fiberimg[np.where(profmask == True)] = ifiber
            bpmask[np.where(infmask == False)]   = True
#            profimg[inprof] = profimg[inprof] + sset.value(yy[inprof])[0]

            # QA: Show the profile image in a DS9 window
            if (False):
                ds9 = pyds9.DS9(target='DS9:*', start=True, wait=10, verify=True)
                ds9.set_np2arr(profimg)

        return (fiberimg, profimg, bpmask)
