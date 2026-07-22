"""Confirm the J2151 blue residual is an IFU-area (benchside footprint) EDGE effect: show white-light
all-covered (residual visible) vs coverage-excluded (clean overlap), and map which benchside's edge it
sits on + whether those are slit-edge fibres."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
ND='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17'
RR=f'{ND}/reduced_rev01/combined'; SD=f'{ND}/sky_diagnostics'

with fits.open(f'{RR}/J2151+0235_cube_blue.fits') as h:
    data=np.asarray(h[0].data,float); nexp=np.asarray(h['NEXP'].data,float); wave=np.asarray(h['WAVELENGTH'].data,float)
wl_all=np.nanmean(data[wave>=3600],axis=0)
wl_fix=np.where(nexp<0.7*np.nanmax(nexp),np.nan,wl_all)

f=[x for x in sorted(glob.glob(f'{ND}/reduced_rev01/extractions/*_RSS_blue.fits'))
   if 'J2151' in str(fits.getheader(x,0).get('OBJECT',''))][0]
with fits.open(f) as h:
    F=np.asarray(h['FLAM'].data,float); S=np.asarray(h['SKY'].data,float); msk=np.asarray(h['MASK'].data); wv=np.asarray(h['WAVE'].data,float)
    fw=h['FIBERWCS'].data if 'FIBERWCS' in [x.name for x in h] else h['FIBERMAP'].data
    bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
    ra=np.asarray(fw['RA_FIBERMAP'] if 'RA_FIBERMAP' in fw.names else fw['RA'],float)
    dec=np.asarray(fw['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fw.names else fw['DEC'],float)
ok=(msk==0)&np.isfinite(F)&(wv>=3600)
wlf=np.array([np.nanmedian((F)[i][ok[i]]) if ok[i].sum()>50 else np.nan for i in range(F.shape[0])])
obj=np.array([np.nanmedian((F[i]-0)[ok[i]]) for i in range(F.shape[0])])  # proxy
blank=np.isfinite(wlf)&(wlf<np.nanpercentile(wlf[np.isfinite(wlf)],70))
cosd=np.cos(np.deg2rad(np.nanmedian(dec))); dx=(ra-np.nanmedian(ra))*cosd*3600; dy=(dec-np.nanmedian(dec))*3600

# per-benchside: is the residual concentrated in one benchside's EDGE fibres?
print(f"{'bs':>4} {'medWL(blank)':>12} {'edge-fib medWL':>15} {'n':>4}")
for b in sorted(set(bs)):
    m=blank&(bs==b)
    if m.sum()<20: continue
    idx=np.where(bs==b)[0]; rank=np.argsort(idx); edgepos=(rank<25)|(rank>=idx.size-25)
    isedge=np.zeros(len(bs),bool); isedge[idx[edgepos]]=True
    me=m&isedge
    print(f"{b:>4} {np.nanmedian(wlf[m]):>12.2e} {np.nanmedian(wlf[me]) if me.sum()>5 else np.nan:>15.2e} {int(m.sum()):>4}")

fig,ax=plt.subplots(1,3,figsize=(16,5))
lo,hi=ZScaleInterval().get_limits(wl_all[np.isfinite(wl_all)])
ax[0].imshow(wl_all,origin='lower',vmin=lo,vmax=hi,cmap='inferno'); ax[0].set_title('J2151 blue WL, ALL covered (edge residual)')
ax[1].imshow(wl_fix,origin='lower',vmin=lo,vmax=hi,cmap='inferno'); ax[1].set_title('J2151 blue WL, coverage-excluded (clean overlap)')
ub=sorted(set(bs)); cm=plt.cm.tab10
for j,b in enumerate(ub):
    m=(bs==b)&np.isfinite(dx)
    ax[2].scatter(dx[m],dy[m],s=10,color=cm(j%10),label=b)
ax[2].set_aspect('equal'); ax[2].invert_xaxis(); ax[2].legend(fontsize=6,ncol=2)
ax[2].set_title('IFU areas: fibres on sky by benchside (single exp)')
fig.suptitle('J2151 blue: edge residual (benchside footprint edge) vs clean max-coverage overlap',fontsize=12)
fig.savefig(f'{SD}/j2151_edge_peek_qa.png',dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("wrote",f'{SD}/j2151_edge_peek_qa.png')
