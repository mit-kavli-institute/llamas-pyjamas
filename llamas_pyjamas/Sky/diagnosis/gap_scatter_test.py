"""Gap-flux decomposition (green 1A): gap = a*local_fibre_flux (scattering skirt, flat-absorbed)
+ dome(Y) (true additive halo). Regress out the skirt; does the residual dome match the fibre floor?"""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, pickle, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
OUT=f'{ND}/gap_scatter_qa.png'

mef=f'{ND}/reduced_ped_off/bias_corrected/LLAMAS_2026-05-17_02-49-56.7_SCI22_mef_bias_corrected.fits'
img=None
with fits.open(mef) as h:
    for hd in h[1:]:
        hdr=hd.header
        if (str(hdr.get('COLOR','')).lower()=='green' and str(hdr.get('BENCH','')).strip()=='1'
                and str(hdr.get('SIDE','')).strip().upper()=='A'):
            img=np.asarray(hd.data,float); break
with open(f'{ND}/reduced_new/traces/LLAMAS_green_1_A_traces.pkl','rb') as fh:
    traces=np.asarray(getattr(pickle.load(fh),'traces'),float)
nfib,ncol=traces.shape

xs=np.arange(0,ncol,2)
trI=np.clip(np.round(traces[:,xs]).astype(int),0,img.shape[0]-1)
spec=np.nanmedian(img[trI,xs[None,:]],axis=0)
locol=spec<np.nanpercentile(spec,30)

def trimmed(v):
    v=v[np.isfinite(v)]
    if v.size<10: return np.nan
    lo,hi=np.nanpercentile(v,[10,90]); w=v[(v>=lo)&(v<=hi)]
    return float(np.mean(w)) if w.size else np.nan

gap=np.full((nfib-1,xs.size),np.nan)
for k,x in enumerate(xs):
    y=traces[:,x]
    for p in range(nfib-1):
        if not (np.isfinite(y[p]) and np.isfinite(y[p+1])): continue
        if y[p+1]-y[p]<6.0: continue
        ymid=int(round(0.5*(y[p]+y[p+1])))
        if 0<ymid<img.shape[0]: gap[p,k]=img[ymid,x]
g=np.array([trimmed(gap[p,locol]) for p in range(nfib-1)])

# per-fibre between-line continuum + floor from the pkl (1A)
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
pkl=sorted(glob.glob(f'{ND}/reduced_ped_off/extractions/LLAMAS_2026-05-17_02-49-56.7*sky1d_extractions.pkl'))[0]
d=ExtractLlamas.loadExtraction(pkl)
for ext,md in zip(d['extractions'],d['metadata']):
    if str(md.get('channel','')).lower()=='green' and str(md['bench'])=='1' and str(md['side']).upper()=='A':
        C=np.asarray(ext.counts,float); S=np.asarray(ext.sky,float)
        nf=C.shape[0]; f1a=np.full(nf,np.nan); cont=np.full(nf,np.nan)
        for i in range(nf):
            fin=np.isfinite(S[i])&np.isfinite(C[i])
            if fin.sum()<200: continue
            lo=fin&(S[i]<np.nanpercentile(S[i][fin],30))
            if lo.sum()>50:
                f1a[i]=np.nanmedian((C[i]-S[i])[lo]); cont[i]=np.nanmedian(C[i][lo])
        break
pf=0.5*(f1a[:-1]+f1a[1:]); pc=0.5*(cont[:-1]+cont[1:])
m=np.isfinite(g)&np.isfinite(pf)&np.isfinite(pc)
# 1. skirt regression: gap ~ a*localflux + b
A=np.column_stack([pc[m],np.ones(m.sum())])
coef,*_=np.linalg.lstsq(A,g[m],rcond=None)
skirt=A@coef; dome=g[m]-skirt
print(f"skirt fit: gap = {coef[0]:.4f} * local_fibre_flux + {coef[1]:+.2f}   "
      f"(skirt explains {1-np.nanstd(dome)/np.nanstd(g[m]):.0%} of gap variance)")
# 2. residual dome vs floor
fm=pf[m]-np.nanmedian(pf[m]); dm=dome-np.nanmedian(dome)
# smooth both along slit to beat 1-ADU quantization noise
ds=median_filter(dm,size=21,mode='nearest'); fs=median_filter(fm,size=21,mode='nearest')
r_raw=np.corrcoef(dm,fm)[0,1]; r_sm=np.corrcoef(ds,fs)[0,1]
sl=float(np.sum(ds*fs)/np.sum(ds*ds))
print(f"dome(Y) vs fibre floor: corr raw={r_raw:+.3f}  smoothed={r_sm:+.3f}  slope(sm)={sl:.2f} "
      f"(2D-halo prediction ~5 px)")
fig,ax=plt.subplots(1,3,figsize=(15,4.4))
ax[0].plot(pc[m],g[m],'.',ms=3); ax[0].plot(pc[m],skirt,'r.',ms=2)
ax[0].set_xlabel('local fibre continuum (counts)'); ax[0].set_ylabel('gap flux (ADU/px)')
ax[0].set_title(f'skirt: gap ~ {coef[0]:.3f}*flux {coef[1]:+.1f}')
idx=np.where(m)[0]
ax[1].plot(idx,fs,'.',ms=3,label='fibre floor (smoothed)')
ax[1].plot(idx,ds*max(sl,1e-9) if sl>0 else ds,'.',ms=3,label='gap dome (scaled)')
ax[1].axhline(0,color='k',lw=.5); ax[1].legend(fontsize=8); ax[1].set_xlabel('slit position')
ax[1].set_title(f'dome vs floor (corr={r_sm:+.2f})')
ax[2].scatter(ds,fs,s=6,alpha=.4); ax[2].set_xlabel('gap dome (ADU/px, smoothed)')
ax[2].set_ylabel('fibre floor (counts, smoothed)'); ax[2].set_title('per slit position')
fig.suptitle('Gap-flux decomposition: local skirt + dome(Y) — does the dome match the floor? (green 1A)')
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig)
print("wrote",OUT)
