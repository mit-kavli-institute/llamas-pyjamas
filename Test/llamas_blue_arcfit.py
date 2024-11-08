import numpy as np
import matplotlib.pyplot as plt
from pypeit import utils

roughpix = [274,326,388,405,410,1113,1136,1140]
waves = [3949,4044,4159,4191,4201, 5520,5562,5570]
ions = ['ArI','ArI','ArI','ArI','ArI','KrI','KrI','KrI']

finepix = np.array([274.41938009,326.39792279,387.89143782,405.2505614,410.21744749,1112.65867627,1135.15309381,1139.48954288])
nistvacwave=np.array([3950.097,4045.561,4159.762,4191.894,4201.858,5522.04373,5563.76977,5571.83623])
#Air:    [5520.5104,5562.22534,5570.28944]

ff = 'polynomial'
mask,coeff = utils.robust_polyfit_djs(finepix,nistvacwave,4,function=ff)

y_model = utils.func_val(coeff,finepix,ff) 


sigma = np.std(nistvacwave-y_model)
print("Sigma = {}".format(sigma))

xx = np.arange(1240)
y_all = utils.func_val(coeff,xx,ff) 

#plt.plot(finepix,nistvacwave-y_model, '+')
plt.plot(xx,y_all)


plt.show()

llamas_lamps = ['ArI','NeI','KrI']
arcsol_params = pypeitpar.WavelengthSolutionPar(lamps=llamas_lamps,reid_arxiv='./llamas_blue_WvcalTemplate.fits',func='polynomial',n_first=2,n_final=2,sigdetect=3.5)
slit = 10
tt = autoid.full_template(arcspec.counts[slit],arcsol_params,[slit],0,1,nsnippet=1) 
