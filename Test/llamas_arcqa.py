from   astropy.io import fits
import scipy
import numpy as np
import extractLlamas 
import pickle
from   matplotlib import pyplot as plt
from   pypeit.core.arc import detect_peaks
from   pypeit.core import pydl
from   pypeit import utils
from   pypeit.par import pypeitpar
from   pypeit.core.wavecal import autoid

f = open("llamas_redwavesol.pickle","rb")
tt = pickle.load(f)
f.close()

autoid.arc_fit_qa(tt.arcsol._all_final_fit['100'],outfile='qa.pdf')

for i in range(164):
    istr = str(i)

    wv  = tt.arcsol._all_final_fit[istr]['wave_soln']
    arc = tt.arcsol._all_final_fit[istr]['spec']

    fitwv = np.where(np.logical_and((wv>7712),(wv<7740)))

    tmp = utils.func_fit(wv[fitwv],arc[fitwv],'gaussian',4)

    print("(FWHM,R) = {},{}".format(tmp[3]*2.35,tmp[2]/(tmp[3]*2.35)))
    
#    plt.plot(wv[fitwv], \
#             (arc[fitwv]-tmp[0])/tmp[1],',')

    invvar = np.ones(arc.shape)
    ydata = (arc-tmp[0])/tmp[1]

    plt.plot(wv,ydata,',')

sset,outmaskmask = pydl.iterfit(wv,ydata,maxiter=3, \
                             invvar=invvar,kwargs_bspline={'bkspace':1.0})
xx = 5500+np.arange(4000)
y_model = sset.value(xx)[0]

    plt.plot(xx,y_model)
    
plt.show()
