"""Blue-channel stacking efficiency in the FULLY-COVERED central region (excludes NaN stripes AND
their boundaries via erosion). Compares 'all covered' vs 'eroded full-coverage' blank estimates, and
asks how close blue gets to the photon floor there."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_erosion
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
RR='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01/combined'
OUT='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/sky_diagnostics/blue_central_qa.png'

def load(path):
    with fits.open(path) as h:
        data=np.asarray(h[0].data,float)
        var=np.asarray(h['VAR'].data,float) if 'VAR' in h else None
        nexp=np.asarray(h['NEXP'].data,float) if 'NEXP' in h else None
    return data,var,nexp

def metrics(data,var,mask):
    mid=data.shape[0]//2
    wl=np.nanmean(data,axis=0)
    blank=mask&np.isfinite(wl)
    blank&=(wl<np.nanpercentile(wl[blank],60))                # low-flux (avoid sources) within region
    def emp(N):
        a=max(0,mid-N//2); b=min(data.shape[0],a+N)
        return np.nanstd(np.nanmean(data[a:b],axis=0)[blank])
    coh=np.nanstd(wl[blank])
    prop=np.nan
    if var is not None:
        n_ok=np.sum(np.isfinite(var)&(var>0),axis=0)
        pmean=np.sqrt(np.nansum(np.where(np.isfinite(var)&(var>0),var,0),axis=0))/np.maximum(n_ok,1)
        prop=np.nanmedian(pmean[blank])
    return dict(n=int(blank.sum()),coh=coh,e10=emp(10),e100=emp(100),prop=prop,blank=blank)

print(f"{'cube / region':<34} {'nblank':>6} {'coh.floor':>10} {'emp@10':>9} {'photon(FB)':>11} {'coh/phot':>8}")
figrows=[]
for fld,fname in (('J0958','J0958+1347_cube_blue.fits'),('J2151','J2151+0235_cube_blue.fits')):
    data,var,nexp=load(f'{RR}/{fname}')
    covered=np.isfinite(np.nanmean(data,axis=0))
    mx=np.nanmax(nexp) if nexp is not None else 1
    full=(nexp>=0.9*mx) if nexp is not None else covered      # fullest-coverage spaxels
    central=binary_erosion(full,iterations=2)                 # drop NaN-boundary spaxels
    for lab,m in (('all covered',covered),('full-cov ERODED central',central)):
        r=metrics(data,var,m)
        cp=r['coh']/r['prop'] if r['prop'] else float('nan')
        print(f"{fld+' blue / '+lab:<34} {r['n']:>6} {r['coh']:>10.2e} {r['e10']:>9.2e} {r['prop']:>11.2e} {cp:>8.1f}")
    figrows.append((fld,data,nexp,covered,central,metrics(data,var,central)['blank']))

fig,ax=plt.subplots(2,3,figsize=(15,9))
for i,(fld,data,nexp,covered,central,blank) in enumerate(figrows):
    wl=np.nanmean(data,axis=0)
    lo,hi=ZScaleInterval().get_limits(wl[np.isfinite(wl)])
    ax[i,0].imshow(wl,origin='lower',vmin=lo,vmax=hi,cmap='inferno'); ax[i,0].set_title(f'{fld} blue white-light')
    ax[i,1].imshow(nexp,origin='lower',cmap='viridis'); ax[i,1].set_title(f'{fld} NEXP (coverage depth)')
    ov=np.zeros(wl.shape); ov[covered]=1; ov[central]=2; ov[blank]=3
    ax[i,2].imshow(ov,origin='lower',cmap='cividis'); ax[i,2].set_title(f'{fld}: covered=1, central=2, blank used=3')
fig.suptitle('Blue: full-coverage central region (excl. NaN-stripe boundaries) for the noise estimate',fontsize=12)
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("\nwrote",OUT)
