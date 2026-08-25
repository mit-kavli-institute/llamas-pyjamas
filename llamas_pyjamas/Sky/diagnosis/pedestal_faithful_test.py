"""Definitive pedestal test: compare striping (single-frame + 3-frame stack) between reduced_ped_off
and reduced_ped_on (J1613 green). Also confirm the pedestal was actually applied (off vs on SKY differ,
SKYPED header). QA PNG."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from llamas_pyjamas.Combine.superRSS import FibreTable
from llamas_pyjamas.Combine.coadd import coadd_image
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'; OUT=f'{ND}/pedestal_result_qa.png'

def files(v): return sorted(glob.glob(f'{ND}/reduced_ped_{v}/extractions/*_RSS_green.fits'))
print("off green RSS:", len(files('off')), " on green RSS:", len(files('on')))

# --- confirm pedestal applied: same frame, off vs on SKY should differ; check SKYPED header ---
fo, fn = files('off')[0], files('on')[0]
with fits.open(fo) as h: sky_off=np.asarray(h['SKY'].data,float); names_off=[x.name for x in h]
with fits.open(fn) as h:
    sky_on=np.asarray(h['SKY'].data,float); hdr_on=h[0].header
    plane='FLAM' if 'FLAM' in [x.name for x in h] else 'SKYSUB'
d=np.abs(sky_on-sky_off); print(f"SKY off-vs-on max|Δ|={np.nanmax(d):.3g}  median|Δ|={np.nanmedian(d):.3g}  "
      f"SKYPED header={hdr_on.get('SKYPED','(absent)')}  compare plane={plane}")

def load(v):
    RA=[];DEC=[];EXP=[];WL=[];VAR=[];NP=[]
    for ei,f in enumerate(files(v)):
        with fits.open(f) as h:
            nm=[x.name for x in h]; pl='FLAM' if 'FLAM' in nm else 'SKYSUB'
            dat=np.asarray(h[pl].data,float); msk=np.asarray(h['MASK'].data)
            ep='FLAM_ERR' if pl=='FLAM' else 'ERROR'; err=np.asarray(h[ep].data,float)
            fm=h['FIBERWCS'].data if 'FIBERWCS' in nm else h['FIBERMAP'].data   # provisional RA/DEC ok
            ra=np.asarray(fm['RA_FIBERMAP'] if 'RA_FIBERMAP' in fm.names else fm['RA'],float)
            dec=np.asarray(fm['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fm.names else fm['DEC'],float)
        ok=(msk==0)&np.isfinite(dat)
        for i in range(dat.shape[0]):
            if ok[i].sum()<50: continue
            RA.append(ra[i]);DEC.append(dec[i]);EXP.append(ei);WL.append(np.nanmedian(dat[i][ok[i]]))
            VAR.append(np.nanmedian(err[i][ok[i]])**2 if np.isfinite(err[i][ok[i]]).any() else 1.0);NP.append(int(ok[i].sum()))
    return list(map(np.array,(RA,DEC,EXP,WL,VAR,NP)))
def coadd(A,sel=None):
    RA,DEC,EXP,WL,VAR,NP=A
    if sel is None: sel=np.ones(RA.size,bool)
    return coadd_image(FibreTable(ra=RA[sel],dec=DEC[sel],value=WL[sel],var=VAR[sel],
        solid_angle=np.full(int(sel.sum()),0.44),exposure=EXP[sel],
        channel=np.array(['green']*int(sel.sum())),npix=NP[sel]),units='flux',weighting='ivar')
def strp(im):
    core=im.nexp>=np.nanmax(im.nexp)-0.5; dd=im.data
    src=dd>np.nanpercentile(dd[core&np.isfinite(dd)],90); bp=core&np.isfinite(dd)&~src
    hp=dd-median_filter(np.nan_to_num(dd,nan=np.nanmedian(dd)),size=9)
    return hp,(np.nanstd(hp[bp]) if bp.sum()>30 else np.nan),bp
R={}
for v in ('off','on'):
    A=load(v); imS=coadd(A)
    per=[strp(coadd(A,A[2]==e))[1] for e in sorted(set(A[2].tolist()))]
    hpS,rS,bpS=strp(imS)
    R[v]=dict(imS=imS,hpS=hpS,rS=rS,bpS=bpS,per=per,rpf=float(np.nanmean(per)))
    print(f"{v}: per-frame stripeRMS={[f'{x:.2e}' for x in per]}  mean={np.nanmean(per):.3e}  stack={rS:.3e}")
print(f"\n=== PEDESTAL RESULT (J1613 green) ===")
print(f"per-frame mean: off {R['off']['rpf']:.3e} -> on {R['on']['rpf']:.3e}  ({100*(1-R['on']['rpf']/R['off']['rpf']):+.0f}%)")
print(f"3-frame stack : off {R['off']['rS']:.3e} -> on {R['on']['rS']:.3e}  ({100*(1-R['on']['rS']/R['off']['rS']):+.0f}%)  [provisional WCS]")

fig,ax=plt.subplots(2,2,figsize=(11,10))
lo,hi=ZScaleInterval().get_limits(R['off']['imS'].data[np.isfinite(R['off']['imS'].data)])
for j,v in enumerate(('off','on')):
    ax[0,j].imshow(R[v]['imS'].data,origin='lower',vmin=lo,vmax=hi,cmap='inferno'); ax[0,j].set_title(f'stack white-light — pedestal {v.upper()}')
hl,hh=ZScaleInterval().get_limits(R['off']['hpS'][np.isfinite(R['off']['hpS'])])
for j,v in enumerate(('off','on')):
    ax[1,j].imshow(np.where(R[v]['bpS'],R[v]['hpS'],np.nan),origin='lower',vmin=hl,vmax=hh,cmap='RdBu_r')
    ax[1,j].set_title(f'high-pass {v.upper()} (RMS {R[v]["rS"]:.2e})')
fig.suptitle(f'Faithful pedestal test — J1613 green 3-frame stack (stack {100*(1-R["on"]["rS"]/R["off"]["rS"]):+.0f}%)',fontsize=13)
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig);print("wrote",OUT)
