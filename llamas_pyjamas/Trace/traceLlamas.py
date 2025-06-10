from   astropy.io import fits # type: ignore
import numpy as np # type: ignore
#import pyds9 # type: ignore
import time
from   matplotlib import pyplot as plt # type: ignore
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl
from   pypeit.core.fitting import iterfit # type: ignore
from   pypeit import bspline
from   llamas_pyjamas.File import llamasIO
import pickle, h5py

###############################################################################3

class TraceLlamas:
    """Class for tracing fiber positions in LLAMAS observations.

    This class handles the tracing of fiber positions and profile fitting for 
    LLAMAS spectrograph data.

    Attributes:
        data (np.ndarray): The image data array.
        naxis1 (int): The x-axis dimension of the data.
        naxis2 (int): The y-axis dimension of the data.
        bench (str): The bench identifier.
        side (str): The side identifier. 
        channel (str): The channel identifier (red, green, blue).
        nfibers (int): Number of fibers detected.
        xmin (int): Minimum x position for trace fitting.
        xmax (int): Maximum x position for trace fitting.
        fitspace (int): Spacing between trace fitting points.
        traces (np.ndarray): Array containing the traced fiber positions.
        bspline_ssets (list): List of B-spline sets for profile fitting.
    """

    def __init__(self, data=0, naxis1=0, naxis2=0, bench='', side='', channel=''):
        """Initialize the TraceLlamas object.

        Args:
            data (np.ndarray, optional): The image data array. Defaults to 0.
            naxis1 (int, optional): The x-axis dimension of the data. Defaults to 0.
            naxis2 (int, optional): The y-axis dimension of the data. Defaults to 0.
            bench (str, optional): The bench identifier. Defaults to ''.
            side (str, optional): The side identifier. Defaults to ''.
            channel (str, optional): The channel identifier. Defaults to ''.

        Returns:
            None
        """
        self.data = data
        self.naxis1 = naxis1
        self.naxis2 = naxis2
        self.bench = bench
        self.side  = side
        self.channel = channel

    def traceSingleCamera(self, dataobj, mph=None):
        """Trace fiber positions for a single camera.

        This method traces the fiber positions across the detector for a single camera
        by detecting peaks in vertical slices and fitting trace functions.

        Args:
            dataobj: Data object containing header and image data.
            mph (float, optional): Minimum peak height for peak detection. If None, 
                an automatic threshold will be calculated. Defaults to None.

        Returns:
            int: Returns 0 if the channel is red (saturated flats), otherwise traces 
                are stored in the object attributes.
        """

        hdr = dataobj.header
        raw = dataobj.data

        self.data = raw.astype(float)
        self.bspline_ssets = []
            
        self.naxis1 = hdr['NAXIS1']
        self.naxis2 = hdr['NAXIS2']
        self.bench = hdr['BENCH']
        self.side  = hdr['SIDE']
        self.channel = hdr['COLOR']

        print(f"Tracing Bench {self.bench}{self.side} ({self.channel})")

        if (self.channel == 'red'):
            print('Red flats are saturated\n\n')
            return(0)

        # Take a vertical slice through traces at center of array, and find the "valleys"
        # corresponding to the continuum

        middle_row = int(self.naxis1/2)
        tslice = np.median(raw[:,middle_row-5:middle_row+4],axis=1).astype(float)

        tmp            = detect_peaks(tslice,mpd=2,threshold=10,show=False,valley=True)
        valley_indices = np.ndarray(len(tmp))
        valley_depths  = np.ndarray(len(tmp))
        
        valley_indices = tmp.astype(float)
        valley_depths  = tslice[tmp].astype(float)
        
        invvar         = np.ones(len(tmp))
        
        # Fit out the scatterd light continuum, using uniform weighting

        x_model = np.arange(self.naxis2).astype(float)
        if (self.channel != 'blue'):
            sset = bspline.bspline(valley_indices,everyn=2)
            res, yfit = sset.fit(valley_indices, valley_depths, invvar)
            x_model = np.arange(self.naxis2).astype(float)
            y_model = sset.value(x_model)[0]
        else:
            y_model = np.zeros(self.naxis2)

        # Now find trace peaks

        comb = tslice-y_model
        peaks = detect_peaks(comb,mpd=2,threshold=10,show=False, valley=False)
        pkht = comb[peaks]

        if (self.channel=='blue'):
            min_pkheight = 500
            print('blue')
        else:
            min_pkheight = 10000

        min_pkheight = 0.3 * np.median(pkht)

        if (mph!=None):
            peaks = detect_peaks(tslice-y_model,mpd=2,mph=mph,show=False, valley=False)
        else:
            peaks = detect_peaks(tslice-y_model,mpd=2,mph=min_pkheight,show=False, valley=False)

        self.nfibers  = len(peaks)
        self.xmin     = 200
        self.xmax     = self.naxis1-100
        self.fitspace = 10

        # ds9 = pyds9.DS9()
        # ds9.set_np2arr(raw)

        # We will explicitly fit a vertical slice for peak positions every
        # self.fitspace pixels in the horizontal direction
        n_tracefit = np.floor((self.xmax-self.xmin)/self.fitspace).astype(int)
        xtrace = self.xmin + self.fitspace * np.arange(n_tracefit)

        tracearr = np.zeros(shape=(self.nfibers,n_tracefit))
        print("NFibers = {}".format(self.nfibers))

        itrace = 0

        print("...Generating the trace fitting grid...")
        
        # Loop over each slice to populate tracearr, which is the fitting grid

        for xtmp in xtrace:

            window = 7
            window = 11
            ytrace = np.median(raw[:,xtmp.astype(int)-window:xtmp.astype(int)+window],axis=1)
            
            valleys = detect_peaks(ytrace,mpd=2,show=False,valley=True)
            nvalley = len(valleys)

            valley_indices = valleys.astype(float)
            valley_depths  = ytrace[valleys].astype(float)
            invvar         = np.ones(nvalley)

            # Fit out scattered light / continuum at each x
            if (self.channel != 'blue'):
                sset = bspline.bspline(valley_indices,everyn=2)
                res, yfit = sset.fit(valley_indices, valley_depths, invvar)
                y_model = sset.value(x_model)[0]
            else:
                sset = bspline.bspline(valley_indices,everyn=4)
                res, yfit = sset.fit(valley_indices, valley_depths, invvar)
                y_model = sset.value(x_model)[0]
                # y_model = np.zeros(self.naxis2) + np.quantile(ytrace,[0.02])[0]
            comb = ytrace.astype(float)-y_model

            # This is a first guess at peaks, which we will then centroid
            # Don't guess at every location, just the first, and then 
            # firts guess is the location of the last column
            if (itrace == 0):
                peaks = detect_peaks(comb,mpd=2,mph=min_pkheight*2*window/self.naxis1,show=False,valley=False)
            else:
                peaks = tracearr[:,itrace-1].astype(int)

            # Centroid all of the peaks for this comb
            ifiber = 0
            try:
                for pk_guess in peaks:
                    if pk_guess == 0:
                        continue
                    pk_centroid = \
                        np.sum(np.multiply(comb[pk_guess-2:pk_guess+3],pk_guess-2+np.arange(5))) \
                        / np.sum(comb[pk_guess-2:pk_guess+3])
                    if (np.abs(pk_centroid-pk_guess) < 1.5):
                        tracearr[ifiber,itrace] = pk_centroid
                    else:
                        tracearr[ifiber,itrace] = pk_guess
                    ifiber += 1
            except:
                plt.plot(comb)
                plt.show()
                
                # print(peaks)
                
            itrace += 1

        # Solve a trace set for all of the fibers
        print("...Solving for the trace functions...")
        self.xtracefit = np.outer(np.ones(ifiber),xtrace)
        self.tracearr  = tracearr
        self.tset      = pydl.xy2traceset(self.xtracefit,self.tracearr,maxdev=0.5)

        # This is a QA plot for tracing the profile centroids
        if (False):
            plt.clf()
            for i in range(ifiber):
                plt.plot(xtrace, tracearr[i,:])
                plt.plot(xtrace,self.tset.yfit.T[:,i])

        # Full traces from solution for all pixels
        x2          = np.outer(np.ones(ifiber),np.arange(self.naxis1))
        self.traces = pydl.traceset2xy(self.tset,xpos=x2)[1]

        print("...All done [traceSingleCamera]!")
        print("")
        
    def profileFit(self):
        """Fit spatial profiles for each fiber.

        This method fits B-spline profiles to each fiber, normalizing out spectral 
        variations and generating profile weight images for optimal extraction.

        Returns:
            tuple: A tuple containing:
                - fiberimg (np.ndarray): Image where each pixel contains the fiber number.
                - profimg (np.ndarray): Profile weight image for optimal extraction.
                - bpmask (np.ndarray): Bad pixel mask.
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

            infmask = np.ones(data_work.shape,dtype=bool)
            NaNmask = np.ones(data_work.shape,dtype=bool)
            badmask = np.ones(data_work.shape,dtype=bool)
            profmask = np.zeros(data_work.shape,dtype=bool)
            invvar = np.ones(data_work.shape,dtype=float)
            
            infmask[np.where(np.isinf(data_work))]  = False
            NaNmask[np.where(np.isnan(data_work))] = False
            badmask[np.where(data_work > 20)] = False
            badmask[np.where(data_work < -5)] = False
            profmask[np.where(np.abs(yy) < 3)] = True

            inprof = np.where(infmask & profmask & NaNmask & badmask)

            # Fit the fiber spatial profile with a bspline
            sset,outmask = iterfit(yy[inprof],data_work[inprof],maxiter=6, \
                        invvar=invvar[inprof],kwargs_bspline={'bkspace':0.33})

            # QA plot showing pixel vals and fit
            if (False):
                xx = -3.0 + np.arange(100) * 6.0/100.0
                plt.plot(yy[inprof],data_work[inprof],',')
                plt.plot(xx,sset.value(xx)[0])
                plt.xlim([-5,5])
                plt.ylim([-0.05,0.5])
                plt.show()
                
            fiberimg[np.where(profmask == True)] = ifiber
            bpmask[np.where(infmask == False)]   = True
            profimg[inprof] = profimg[inprof] + sset.value(yy[inprof])[0]

            self.bspline_ssets.append(sset)
            
            # QA: Show the profile image in a DS9 window
            if (True):
                ds9 = pyds9.DS9(target='DS9:*', start=True, wait=10, verify=True)
                ds9.set_np2arr(profimg)

        return (fiberimg, profimg, bpmask)

    ######
    #
    # Delivers a profile image just for one fiber.
    #
    
    def fiberProfileImg(self,fiber_index):
        """Generate a profile image for a single fiber.

        This method creates a profile weight image for a specific fiber using 
        the previously fitted B-spline profile.

        Args:
            fiber_index (int): The index of the fiber to generate the profile for.

        Returns:
            np.ndarray: 2D profile image with weights for the specified fiber.
        """

        profimg  = np.zeros((self.naxis2,self.naxis1),dtype=float)
        ytrace = self.traces[fiber_index,:]
        
        yy = np.outer(np.arange(self.naxis2),np.ones(self.naxis1)) \
            - np.outer(np.ones(self.naxis2),ytrace)


        profmask = np.zeros(profimg.shape,dtype=bool)
        profmask[np.where(np.abs(yy) < 3)] = True
        
        inprof = np.where(profmask)
        profimg[inprof] = self.bspline_ssets[fiber_index].value(yy[inprof])[0]

        return profimg


def saveTraces(objlist, outfile='LLAMASTrace.h5'):
    """Save a list of TraceLlamas objects to file.

    This function saves TraceLlamas objects to either pickle (.pkl) or HDF5 (.h5) format.

    Args:
        objlist (list): List of TraceLlamas objects to save.
        outfile (str, optional): Output filename. Defaults to 'LLAMASTrace.h5'.

    Returns:
        None
    """

    if ('.pkl' in outfile):
        with open(outfile,'wb') as fp:
           pickle.dump(objlist, fp)

    if ('.h5' in outfile):
        with h5py.File(outfile, 'w') as f:
            # Stack all 2D data arrays along a new axis
            data_stack = np.stack([trace.data for trace in objlist], axis=0)
            f.create_dataset('data', data=data_stack)  
    
            # Save other attributes
            f.create_dataset('naxis1', data=[llama.naxis1 for llama in objlist])
            f.create_dataset('naxis2', data=[llama.naxis2 for llama in objlist])
            dt = h5py.string_dtype(encoding='utf-8')
            f.create_dataset('bench', data=[llama.bench for llama in objlist])


def loadTraces(infile):
    """Load TraceLlamas objects from file.

    This function loads TraceLlamas objects from either pickle (.pkl) or HDF5 (.h5) format.

    Args:
        infile (str): Input filename to load from.

    Returns:
        list or object: List of TraceLlamas objects (for .h5 files) or the loaded 
            object (for .pkl files).
    """

    if ('pkl' in infile):
        with open(infile,'rb') as fp:
            object = pickle.load(fp)
        return(object)
        
    if ('h5' in infile):
        with h5py.File(infile, 'r') as f:
            data_stack = f['data'][:]  # Load the 3D data array
            naxis1 = f['naxis1'][:]
            naxis2 = f['naxis2'][:]
            bench = f['bench'][:]
    
            # Reconstruct the array of TraceLlamas objects
            loaded_llama_traces = [
                TraceLlamas(data=data_stack[i], naxis1=naxis1[i], naxis2=naxis2[i], bench=bench[i])
                for i in range(len(data_stack))]
        return(loaded_llama_traces)

def traceAllCameras(dataobj_all):
    """Trace fibers for all cameras in a dataset.

    This function traces fibers for all camera extensions in the provided data object.

    Args:
        dataobj_all: Data object containing multiple camera extensions.

    Returns:
        list: List of TraceLlamas objects, one for each camera extension.
    """

    all_traces = []

    for thisobj in dataobj_all.extensions:
        thistrace = TraceLlamas()
        thistrace.traceSingleCamera(thisobj)
        all_traces.append(thistrace)
    
    return(all_traces)