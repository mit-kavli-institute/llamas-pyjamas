"""Green -> photon floor: clean-central-region decomposition (coherent spatial floor vs photon vs
plane-varying/sky-line) on corrected rev01 green cubes, + a benchside-edge check to see how much of
the green systematic is the shared slit-edge per-fibre bias vs green-specific sky lines."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_erosion, median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
ND='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17'
RR=f'{ND}/reduced_rev01/combined'; EX=f'{ND}/reduced_rev01/extractions'
OUT=f'{ND}/sky_diagnostics/green_decomp_qa.png'
fields=[('J2151','J2151+0235_cube_green.fits'),('J0958','J0958+1347_cube_green.fits')]

fig,ax=plt.subplots(2,2,figsize=(13,9))
for i,(fld,fn) in enumerate(fields):
    with fits.open(f'{RR}/{fn}') as h:
        data=np.asarray(h[0].data,float); var=np.asarray(h['VAR'].data,float)
        nexp=np.asarray(h['NEXP'].data,float); wave=np.asarray(h['WAVELENGTH'].data,float)
    wl=np.nanmean(data,axis=0)
    central=binary_erosion(nexp>=0.9*np.nanmax(nexp),iterations=2)&np.isfinite(wl)
    blank=central&(wl<np.nanpercentile(wl[central],60))
    coh=np.nanstd(wl[blank])
    vary=np.array([np.nanstd((data[k]-wl)[blank]) for k in range(data.shape[0])])
    phot=np.array([np.nanmedian(np.sqrt(var[k][blank])) for k in range(data.shape[0])])
    spec=np.array([np.nanmean(data[k][blank]) for k in range(data.shape[0])])
    g=np.isfinite(vary)&np.isfinite(phot)
    n_ok=np.sum(np.isfinite(var)&(var>0),axis=0)
    propFB=np.nanmedian((np.sqrt(np.nansum(np.where(np.isfinite(var)&(var>0),var,0),axis=0))/np.maximum(n_ok,1))[blank])
    sm=median_filter(np.nan_to_num(spec),size=31,mode='nearest'); lines=np.abs(spec-sm)
    rl=np.corrcoef(vary[g],lines[g])[0,1] if g.sum()>50 else np.nan
    print(f"{fld} green (clean central): coh(spatial)={coh:.2e}  photon(FB)={propFB:.2e}  coh/phot={coh/propFB:.1f}"
          f"  vary/plane={np.nanmedian(vary[g]):.2e}  corr(vary,skylines)={rl:+.2f}")
    # coherent-pattern high-pass map
    hp=wl-median_filter(np.nan_to_num(wl,nan=np.nanmedian(wl[blank])),size=9)
    hpm=np.where(blank,hp,np.nan); hl=np.nanpercentile(np.abs(hpm[np.isfinite(hpm)]),97)
    ax[i,0].imshow(hpm,origin='lower',vmin=-hl,vmax=hl,cmap='RdBu_r')
    ax[i,0].set_title(f'{fld} green coherent pattern (high-pass), coh={coh:.1e}')
    ax[i,1].plot(wave,vary,lw=.5,label='vary/plane'); ax[i,1].plot(wave,phot,lw=.7,label='photon')
    ax[i,1].axhline(coh,color='r',ls='--',lw=.8,label=f'coh floor {coh:.1e}')
    ax[i,1].legend(fontsize=7); ax[i,1].set_title(f'{fld} green RMS vs wavelength'); ax[i,1].set_xlabel('A')

    # benchside-edge check from one RSS
    rf=[x for x in sorted(glob.glob(f'{EX}/*_RSS_green.fits')) if fld in str(fits.getheader(x,0).get('OBJECT',''))][0]
    with fits.open(rf) as h:
        F=np.asarray(h['FLAM'].data,float); S=np.asarray(h['SKY'].data,float); msk=np.asarray(h['MASK'].data)
        bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
    ok=(msk==0)&np.isfinite(F)
    wlf=np.array([np.nanmedian(F[j][ok[j]]) if ok[j].sum()>50 else np.nan for j in range(F.shape[0])])
    bl=np.isfinite(wlf)&(wlf<np.nanpercentile(wlf[np.isfinite(wlf)],70))
    edge_ex=[]
    for b in sorted(set(bs)):
        idx=np.where(bs==b)[0]; rank=np.argsort(idx)
        isedge=np.zeros(len(bs),bool); isedge[idx[(rank<25)|(rank>=idx.size-25)]]=True
        me=bl&isedge&(bs==b); mi=bl&~isedge&(bs==b)
        if me.sum()>5 and mi.sum()>5: edge_ex.append(np.nanmedian(wlf[me])-np.nanmedian(wlf[mi]))
    print(f"   benchside edge-minus-interior residual: median={np.nanmedian(edge_ex):+.2e} spread={np.nanstd(edge_ex):.2e} (n={len(edge_ex)})")
fig.suptitle('Green clean-central noise decomposition (corrected rev01 cubes)',fontsize=12)
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("\nwrote",OUT)
