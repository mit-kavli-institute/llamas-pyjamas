from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from pypeit.core.wave import airtovac
from   pypeit.core.arc import detect_peaks,iter_continuum,detect_lines
import astropy.units as u
from pypeit import utils

from LlamasPipeline.Trace.traceLlamas import TraceLlamas
from LlamasPipeline.Extract.extractLlamas import ExtractLlamas
from LlamasPipeline.Arc.arcLlamas import ArcLlamas


fig, ax = plt.subplots(2,1)

# tt = Table.read('KrI.tab',format='ascii')
# tt = Table.read('ArI.tab',format='ascii')
tt = Table.read('NeI.tab',format='ascii')

wv = tt[tt['col2'] >= 300]['col1'] * 10

wv_vac = airtovac(wv*u.AA).value

height = tt[tt['col2'] >= 20]['col2']

for i in range(len(wv)):
    
    ax[0].plot([wv[i],wv[i]],[0,height[i]])
    label = f"{wv_vac[i]:7.2f}"
    ax[0].text(wv[i],1.05*height[i],label,rotation=90)
    
ax[0].set_yscale('log')
ax[0].set_xlim(6900,4800)
# ax[0].set_xlim(6900, 10000)
ax[1].set_xlim(0,2048)


# trace  = TraceLlamas("Green_flat.fits",spectrograph=0,channel='green',mph=1500)
#trace  = TraceLlamas("Red_flat.fits",spectrograph=0,channel='green',mph=1500)
#trace.profileFit()
trace = TraceLlamas.loadTrace('Trace_1A.pkl')

arcfile_Kr = '/Users/simcoe/GIT/LlamasPipeline/Test/FinalAlignmentData/Bench_1a_Final_Alignment_Data/Green/REF_fIFU_Arc/Kr/20240506_135311_Camera3_1A_Green_1.7s-2.0e_Kr_Long.fits'
arcfile_Ar = '/Users/simcoe/GIT/LlamasPipeline/Test/FinalAlignmentData/Bench_1a_Final_Alignment_Data/Green/REF_fIFU_Arc/Ar/20240506_140929_Camera3_1A_Green_1.7s-2.0e_Ar_Long.fits'
arcfile_Ne = '/Users/simcoe/GIT/LlamasPipeline/Test/FinalAlignmentData/Bench_1a_Final_Alignment_Data/Green/REF_fIFU_Arc/Ne/20240506_143226_Camera3_1A_Green_0.05s-2.0e_Ne_Short.fits'

arcfile = arcfile_Ne

arcspec = ExtractLlamas(arcfile,trace)

nfibers,naxis1 = arcspec.counts.shape
for ifiber in range(nfibers):
    contin = iter_continuum(arcspec.counts[ifiber,:],npoly=12)
    arcspec.counts[ifiber,:] -= contin[0]

    
x = np.arange(2048)
ax[1].plot(x,arcspec.counts[10])
pks = detect_lines(arcspec.counts[10], sigdetect=5, fwhm=3, cont_subtract=True)

centroids = pks[2]
heights = pks[1]

for i in range(len(centroids)):
    label = f"{centroids[i]:7.2f}"
    ax[1].text(centroids[i], heights[i], label, rotation=90)
    print(f"Added centroid: {label}")


# GREEN CHANNEL
xpk  = np.array([1200.83,1182.05,1049.22,914.80,862.37,846.15,839.80,721.13,709.45,520.47,550.97,591.89,271.97,308.13,349.50,156.63,177.37,207.10,125.21,1453.52,1542.77,1749.00,1872.11])
wvpk = np.array([5651.13,5674.03,5834.47,5995.51,6057.80,6076.94,6084.55,6224.46,6238.08,6458.07,6422.80,6375.35,6741.96,6701.08,6654.07,6871.53,6848.29,6814.99,6904.67,5340.60,5229.63,4970.47,4813.98])

# RED CHANNEL

#xpk = np.array([1812.52, 1898.85, 1457.20, 1523.62, 1567.69, 1609.21, 999.97, 1010.32, 1071.31, 910.01, 810.04, 817.26, 714.04, 749.73, 755.08, 443.12, 449.60, 522.68, 120.18, 180.88, 228.56, 304.00, 370.74])
#wvpk = np.array([9660.43, 9787.19, 9125.47, 9227.03, 9294.08, 9356.79, 8410.52, 8426.56, 8523.78, 8266.79, 8105.92, 8117.54, 7950.36, 8008.36, 8016.99, 7505.94, 7516.72, 7637.21, 6967.35, 7069.17, 7149.01, 7274.94, 7386.01])


reorder = np.argsort(xpk)
xpk = xpk[reorder]
wvpk = wvpk[reorder]

ff = 'polynomial'
mask,coeff = utils.robust_polyfit_djs(xpk,wvpk,5,function=ff,upper=3,lower=3)
y_model = utils.func_val(coeff,xpk,ff) 

plt.plot(xpk, wvpk-y_model, '+')
#plt.plot(xpk, y_model)
plt.plot([0,2048],[0,0])

rms = np.std(wvpk[mask]-y_model[mask])
print("RMS = {}".format(rms))

waves = utils.func_val(coeff,np.arange(naxis1),ff)


plt.show()
