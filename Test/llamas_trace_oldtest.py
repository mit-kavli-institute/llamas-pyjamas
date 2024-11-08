from   astropy.io import fits
import scipy
import numpy as np
from   matplotlib import pyplot as plt
import pydl
import pyds9

###########

def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):
    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height (if parameter
        `valley` is False) or peaks that are smaller than maximum peak height
         (if parameter `valley` is True).
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`

    The function can handle NaN's

    See this IPython Notebook [1]_.

    __author__ = "Marcos Duarte, https://github.com/demotu/BMC"
    __version__ = "1.0.5"
    __license__ = "MIT"

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=-1.2, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)

    Version history
    ---------------
    '1.0.5':
        The sign of `mph` is inverted if parameter `valley` is True

    """
    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
        if mph is not None:
            mph = -mph
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan - 1, indnan + 1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size - 1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind] - x[ind - 1], x[ind] - x[ind + 1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                       & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
            if mph is not None:
                mph = -mph
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind


def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
    """Plot results of the detect_peaks function, see its help."""


    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(x, 'b', lw=1)
    if ind.size:
        label = 'valley' if valley else 'peak'
        label = label + 's' if ind.size > 1 else label
        ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                label='%d %s' % (ind.size, label))
        ax.legend(loc='best', framealpha=.5, numpoints=1)
    ax.set_xlim(-.02 * x.size, x.size * 1.02 - 1)
    ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
    yrange = ymax - ymin if ymax > ymin else 1
    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.1 * yrange)
    ax.set_xlabel('Data #', fontsize=14)
    ax.set_ylabel('Amplitude', fontsize=14)
    mode = 'Valley detection' if valley else 'Peak detection'
    ax.set_title("%s (mph=%s, mpd=%f, threshold=%s, edge='%s')"
                 % (mode, str(mph), mpd, str(threshold), edge))
    # plt.grid()
    plt.show()


#########

#with fits.open("output_20200103_110827_blue_flat.fits") as hdu:
with fits.open("20200108_150501_Camera01_red_FLAT.fits") as hdu:
    hdr = hdu[0].header
    raw = hdu[0].data

y = np.sum(raw[:,600:610],axis=1)

valleys = detect_peaks(y,mpd=2,threshold=10,show=False, valley=True)
nvalley = len(valleys)

valleyind    = np.ndarray(nvalley)
valleydepths = np.ndarray(nvalley)

valleyind =    valleys.astype(np.float)
valleydepths = y[valleys].astype(np.float)
invvar = np.ones(nvalley)

# Fit a bspline to the scattered light background
sset = pydl.bspline(valleyind,everyn=2)
res, yfit = sset.fit(valleyind, valleydepths, invvar)

x_model = np.arange(1150).astype(np.float)
y_model = sset.value(x_model)[0]

#plt.plot(valleyind, valleydepths)
#plt.plot(x_model, y_model)
#plt.show()

peaks   = detect_peaks(y.astype(np.float)-y_model,mpd=2,threshold=10,mph=2500)

nfibers = len(peaks)
xmin     = 100
xmax     = 1180
# xmax     = 150
fitspace = 5.0

n_tracefit = np.floor((xmax-xmin)/fitspace).astype(np.int)
xtrace = xmin + fitspace * np.arange(n_tracefit)

tracearr = np.zeros(shape=(nfibers,n_tracefit))
itrace = 0

print("Generating the trace fitting grid")

for xtmp in xtrace:

    ytrace = np.median(raw[:,xtmp.astype(np.int)-7:xtmp.astype(np.int)+7],axis=1)
    
    valleys = detect_peaks(ytrace,mpd=2,show=False,valley=True)
    nvalley = len(valleys)

    valleyind    = np.ndarray(nvalley)
    valleydepths = np.ndarray(nvalley)

    valleyind =    valleys.astype(np.float)
    valleydepths = ytrace[valleys].astype(np.float)
    invvar = np.ones(nvalley)

    # Fit a bspline to the scattered light background
    sset = pydl.bspline(valleyind,everyn=2)
    res, yfit = sset.fit(valleyind, valleydepths, invvar)

    x_model = np.arange(1150).astype(np.float)
    y_model = sset.value(x_model)[0]

    comb  = ytrace.astype(np.float)-y_model

    if (itrace == 0):
        peaks = detect_peaks(comb,mpd=2,mph=50,show=False,valley=False)
    else:
        peaks = tracearr[:,itrace-1].astype(np.int)

    ifiber = 0
    for pk_guess in peaks:
        pk_centroid = \
            np.sum(np.multiply(comb[pk_guess-2:pk_guess+3],pk_guess-2+np.arange(5))) \
            / np.sum(comb[pk_guess-2:pk_guess+3])
        if (np.abs(pk_centroid-pk_guess) < 1.5):
            tracearr[ifiber,itrace] = pk_centroid
        else:
            tracearr[ifiber,itrace] = pk_guess
        ifiber = ifiber+1
        
    itrace = itrace+1

    if (xtmp == 1000 and False):
        plt.plot(ytrace)
        plt.plot(y_model)
        plt.plot(comb)
        plt.plot(tracearr[:,itrace-1],750+np.zeros(len(peaks)),'+')
        
# Solve a trace set for all of the fibers
x = np.outer(np.ones(ifiber),xtrace)
tset = pydl.xy2traceset(x,tracearr)

# This is a QA plot for tracing the profile centroids
if (False):
    for i in range(ifiber):
        plt.plot(xtrace, tracearr[i,:])
        plt.plot(xtrace,tset.yfit.T[:,i])

# Full traces from solution for all pixels
x2 = np.outer(np.ones(ifiber),np.arange(1240))
profiles = pydl.traceset2xy(tset,xpos=x2)[1]

########################

# Profile Fitting

# Subtract off a lazy-man's "overscan" to get rid of scattered light
# Fix this!
data = raw.astype(np.float)
ref = data[12,:]
for i in range(1150):
    if (i != 12):
        data[i,:] = data[i,:] - ref

fiberimg = np.zeros(data.shape,dtype=np.int)   # Lists the fiber # of each pixel
profimg  = np.zeros(data.shape,dtype=np.float) # Profile weighting function
bpmask   = np.zeros(data.shape,dtype=np.bool)  # bad pixel mask

ifiber = 5
yprof = profiles[ifiber,:]

# Generate a curved "y" image
yy = np.outer(np.arange(1150),np.ones(1240)) - np.outer(np.ones(1150),yprof)

# Normalize out the spectral shape of the lamp when fitting the profile        

data_ref = np.copy(data)
for i in range(1240):
    norm = np.sum(data[np.where(np.abs(yy[:,i]) < 2.0),i])
    data[:,i] = data[:,i] / norm
    
# Generate a mask of pixels that are
# (a) within 4 pixels of the profile center for this fiber and
# (b) not NaNs or Infs
# Also generate an inverse variance array that is presently flat weighting

infmask = np.ones(data.shape,dtype=np.bool)
infmask[np.where(np.isinf(data))]  = False
Nanmask = np.ones(data.shape,dtype=np.bool)
Nanmask[np.where(np.isnan(data))]  = False
profmask = np.zeros(data.shape,dtype=np.bool)
profmask[np.where(np.abs(yy) < 4)] = True

inprof = np.where(infmask & profmask & Nanmask)
invvar = np.ones(data.shape,dtype=np.float)

# Fit the fiber spatial profile with a bspline
sset,outmask = pydl.iterfit(yy[inprof],data[inprof],maxiter=6,invvar=invvar[inprof],kwargs_bspline={'bkspace':0.33})

# QA plot showing pixel vals and fit
xx = -4.0 + np.arange(100) * 8.0/100.0
plt.plot(yy[inprof],data[inprof],',')
plt.plot(xx,sset.value(xx)[0])
plt.xlim([-5,5])
plt.ylim([-0.05,0.5])

fiberimg[np.where(profmask == True)] = ifiber
bpmask[np.where(infmask == True)] = True
profimg[inprof] = profimg[inprof] + sset.value(yy[inprof])[0]

# QA: Show the profile image in a DS9 window
if (False):
    ds9 = pyds9.DS9(target='DS9:*', start=True, wait=10, verify=True)
    ds9.set_np2arr(profimg)

stop
    
#######################

# Extract a flat field image or an arc

with fits.open("20200108_150215_Camera01_red_KrNeAr.fits") as hdu:
    hdr = hdu[0].header
    arc = hdu[0].data.astype(np.float)

x_rect = np.outer(np.ones(1150),np.arange(1240))

fibermask = (fiberimg == ifiber)
inprof    = (fibermask & infmask)

x_spec = x_rect[inprof]
# f_spec = data_ref[inprof] / profimg[inprof]
f_spec = arc[inprof]# / profimg[inprof]
invvar = profimg[inprof]
inmask = infmask[inprof]

# sset,outmask = pydl.iterfit(x_spec,f_spec,maxiter=6,invvar=invvar,kwargs_bspline={'bkspace':1.0})
# extracted[i] = sset.value(np.arange(1240))

extracted = np.zeros(1240)
for i in range(1240):
    thisx = np.where(x_spec == i)
    extracted[i] = np.sum(f_spec[thisx]*invvar[thisx])/np.sum(invvar[thisx])
    
plt.plot(x_spec, f_spec, ',')
plt.plot(np.arange(1240),extracted)
    
plt.show()
             
#######################

# Wavelength solution

ee = np.zeros((1240,1))
ee[:,0]=extracted
llamas_lamps = ['NeI','KrI','ArI','ArII']
arcsol_params = pypeitpar.WavelengthSolutionPar(lamps=llamas_lamps,rms_threshold=0.25)

# HolyGrail is an initial fit, later on we will want to solve via reidentify
arcsol = autoid.HolyGrail(ee, par=arcsol_params, islinelist=False, use_unknowns=False)
autoid.arc_fit_qa(arcsol._all_final_fit['0'],outfile='qa.pdf')

#######################

