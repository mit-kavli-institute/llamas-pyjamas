from   astropy.table import Table
from   astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from   pypeit.core.wave import airtovac
from   pypeit.core.arc import detect_peaks,iter_continuum,detect_lines
import astropy.units as u
from   pypeit import utils
from   scipy.signal import correlate as cross_correlate

from LlamasPipeline.Trace.traceLlamas import TraceLlamas
from LlamasPipeline.Extract.extractLlamas import ExtractLlamas
from LlamasPipeline.Arc.arcLlamas import ArcLlamas

# Read in the Argon table
# tt = Table.read('ArI.tab',format='ascii')
tt = Table.read('KrI.tab',format='ascii')
wv = tt[tt['col2'] >= 20]['col1'] * 10
wv_vac = airtovac(wv*u.AA).value
height = tt[tt['col2'] >= 20]['col2']

print("Reading traces")
#trace  = TraceLlamas("Red_flat.fits",spectrograph=0,channel='green',mph=1500)
#trace.profileFit()
trace = TraceLlamas.loadTrace('Trace_1A.pkl')

print("Extracting arc traces")
arcfile = '/Users/simcoe/GIT/LlamasPipeline/Test/FinalAlignmentData/Bench_1a_Final_Alignment_Data/Green/REF_fIFU_Arc/Kr/20240506_135311_Camera3_1A_Green_1.7s-2.0e_Kr_Long.fits'
arcspec = ExtractLlamas(arcfile,trace)
arcspec.saveExtraction(outfile='KrI.pkl')
# arcspec = ExtractLlamas.loadExtraction('KrI.pkl')

print("Subtracting scattered light")
nfibers,naxis1 = arcspec.counts.shape
for ifiber in range(nfibers):
    contin = iter_continuum(arcspec.counts[ifiber,:],npoly=12)
    arcspec.counts[ifiber,:] -= contin[0]

########### REIDENTIFY #################

# RED CHANNEL REFERENCE PEAKS
xpk_old = np.array([1812.52, 1898.85, 1457.20, 1523.62, 1567.69, 1609.21, 999.97, 1010.32, 1071.31, 910.01, 810.04, 817.26, 714.04, 749.73, 755.08, 443.12, 449.60, 522.68, 120.18, 180.88, 228.56, 304.00, 370.74])

wvpk = np.array([9660.43, 9787.19, 9125.47, 9227.03, 9294.08, 9356.79, 8410.52, 8426.56, 8523.78, 8266.79, 8105.92, 8117.54, 7950.36, 8008.36, 8016.99, 7505.94, 7516.72, 7637.21, 6967.35, 7069.17, 7149.01, 7274.94, 7386.01])

# GREEN KrI CHANNEL REFERENCE PEAKS
xpk_old  = np.array([128.21,  179.21,  280.69,  322.30,  388.71,  493.66,  524.26,  565.23,  755.48,  819.63,  991.41,  1240.07, 682.96,  694.42,  813.36,  836.03,  888.46,  1023.03, 1127.18, 1174.89, 1280.67, 1304.86])

wvpk     = np.array([6871.53, 6814.99, 6701.08, 6654.07, 6578.24, 6458.07, 6422.80, 6375.35, 6153.11, 6076.94, 5872.54, 5571.84, 6238.08, 6224.46, 6084.55, 6057.80, 5995.51, 5834.47, 5709.10, 5651.13, 5522.04, 5492.46])

######################################

sort_ind = np.argsort(xpk_old)
xpk_old = xpk_old[sort_ind]
wvpk = wvpk[sort_ind]

x = np.arange(2048)
reference_arc = arcspec.counts[10]

wv_image = np.zeros([2048, nfibers])
arc_image = np.zeros([2048, nfibers])


for ifiber in range(nfibers):


    
    test_arc = arcspec.counts[ifiber]

    ccorr = cross_correlate(test_arc, reference_arc,mode='full')
    pks = detect_lines(ccorr, sigdetect=5, fwhm=3, cont_subtract=True)
    centroids = pks[2]
    heights   = pks[1]

    lag_ind = np.where(heights==np.max(heights))
    lag = centroids[lag_ind][0] - 2048

    pks = detect_lines(test_arc, sigdetect=5, fwhm=3, cont_subtract=True)
    centroids = pks[2]
    heights = pks[1]

    if (False):
        plt.plot(x+lag,reference_arc)
        plt.plot(x,test_arc)
        plt.plot(centroids, heights, '+')
        plt.ylim(-100,2000)

        for line in xpk_old:
            plt.plot([line+lag,line+lag],[0,2000],alpha=0.2)
        
        plt.show()

    new_xpk = np.array([])
    xpk = xpk_old + lag
    
    for peak in xpk:
        offsets = np.abs(centroids-peak)
        match = centroids[np.where(offsets == np.min(offsets))][0]
        # print(peak, match,peak-match)
        new_xpk = np.append(new_xpk, match)

    ff = 'polynomial'
    mask,coeff = utils.robust_polyfit_djs(new_xpk,wvpk,3,function=ff,upper=3,lower=3)
    y_model = utils.func_val(coeff,new_xpk,ff) 
    rms = np.std(wvpk[mask]-y_model[mask])
    print(f'...Reidentifying fiber {ifiber}\tRMS={rms}')

    waves   = utils.func_val(coeff,np.arange(naxis1),ff)
    
    wv_image[:,ifiber]   = waves
    arc_image[:,ifiber]  = test_arc
    
    if (False):
        plt.plot(new_xpk, wvpk-y_model, '+')
        plt.plot([0,2048],[0,0])
        plt.show()


        
hdr = fits.Header()
hdr_hdu = fits.PrimaryHDU(hdr)
wv_hdu = fits.ImageHDU(wv_image)
arc_hdu = fits.ImageHDU(arc_image)
hdu_list = fits.HDUList([hdr_hdu,wv_hdu,arc_hdu])
hdu_list.writeto("llamasg_arcref2d.fits", overwrite=True)



