"""Decompose the clean-central-region blue residual: spectrally-COHERENT fixed spatial pattern
(C = mean over lambda) vs the plane-VARYING part (D-C). Is the varying part just photon noise
(-> systematic is spatial) or does it spike at sky lines (-> sky-subtraction residual)?"""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_erosion, median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
RR='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01/combined'
OUT='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/sky_diagnostics/blue_decomp_qa.png'
TH=np.deg2rad(145.0)

fig,ax=plt.subplots(2,3,figsize=(16,9))
for i,(fld,fname) in enumerate((('J0958','J0958+1347_cube_blue.fits'),('J2151','J2151+0235_cube_blue.fits'))):
    with fits.open(f'{RR}/{fname}') as h:
        data=np.asarray(h[0].data,float); var=np.asarray(h['VAR'].data,float)
        nexp=np.asarray(h['NEXP'].data,float)
        wave=(np.asarray(h['WAVELENGTH'].data,float) if 'WAVELENGTH' in h else np.arange(data.shape[0]))
    wl=np.nanmean(data,axis=0)
    central=binary_erosion(nexp>=0.9*np.nanmax(nexp),iterations=2)&np.isfinite(wl)
    blank=central&(wl<np.nanpercentile(wl[central],60))
    C=wl                                                     # coherent spatial pattern (lambda-mean)
    coh=np.nanstd(C[blank])
    # per-plane: varying spatial RMS (D-C) and photon, vs wavelength
    vary=np.array([np.nanstd((data[k]-C)[blank]) for k in range(data.shape[0])])
    phot=np.array([np.nanmedian(np.sqrt(var[k][blank])) for k in range(data.shape[0])])
    spec=np.array([np.nanmean(data[k][blank]) for k in range(data.shape[0])])   # mean blank spectrum (sky resid)
    good=np.isfinite(vary)&np.isfinite(phot)
    vmed=np.nanmedian(vary[good]); pmed=np.nanmedian(phot[good])
    excess=np.sqrt(max(vmed**2-pmed**2,0))                   # plane-varying systematic beyond photon
    cohfrac=coh**2/(coh**2+vmed**2)
    # does the varying RMS track sky lines? correlate vary with |spec - smooth(spec)| (line strength)
    sm=median_filter(np.nan_to_num(spec),size=31,mode='nearest'); lines=np.abs(spec-sm)
    r_lines=np.corrcoef(vary[good],lines[good])[0,1] if good.sum()>50 else np.nan
    print(f"{fld} blue central: coh(spatial)={coh:.2e}  vary/plane={vmed:.2e}  photon={pmed:.2e}  "
          f"vary-excess={excess:.2e}  coherent-frac={cohfrac:.0%}  corr(vary,skylines)={r_lines:+.2f}")
    # panels: coherent pattern high-pass (striping?), and vary/photon/lines vs wavelength
    hp=(C-median_filter(np.nan_to_num(C,nan=np.nanmedian(C[blank])),size=9))
    hpm=np.where(blank,hp,np.nan); hl=np.nanpercentile(np.abs(hpm[np.isfinite(hpm)]),97)
    ax[i,0].imshow(hpm,origin='lower',vmin=-hl,vmax=hl,cmap='RdBu_r')
    ax[i,0].set_title(f'{fld} blue: coherent spatial pattern (high-pass)')
    ax[i,1].plot(wave,vary,lw=.6,label='varying RMS/plane'); ax[i,1].plot(wave,phot,lw=.8,label='photon')
    ax[i,1].axhline(coh,color='r',ls='--',lw=.8,label=f'coherent floor {coh:.1e}')
    ax[i,1].legend(fontsize=7); ax[i,1].set_title(f'{fld} blue: RMS vs wavelength'); ax[i,1].set_xlabel('A')
    ax[i,2].plot(wave,spec,lw=.6); ax[i,2].axhline(0,color='k',lw=.4)
    ax[i,2].set_title(f'{fld} blue: mean blank spectrum (sky resid)'); ax[i,2].set_xlabel('A')
fig.suptitle('Blue clean-central residual: coherent spatial pattern vs plane-varying (sky-line?) part',fontsize=12)
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("\nwrote",OUT)
