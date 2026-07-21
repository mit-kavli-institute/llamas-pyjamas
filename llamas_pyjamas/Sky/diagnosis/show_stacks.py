"""Visual off/on comparison: single-frame and 3-frame-stack white-light + high-pass images,
pedestal OFF vs template ON, common color scales."""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from llamas_pyjamas.Combine.superRSS import FibreTable
from llamas_pyjamas.Combine.coadd import coadd_image
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
OUT=f'{ND}/pedestal_visual_qa.png'

def load(v):
    RA=[];DEC=[];EXP=[];WL=[];VAR=[];NP=[]
    for ei,f in enumerate(sorted(glob.glob(f'{ND}/reduced_ped_{v}/extractions/*_RSS_green.fits'))):
        with fits.open(f) as h:
            dat=np.asarray(h['SKYSUB'].data,float); msk=np.asarray(h['MASK'].data)
            err=np.asarray(h['ERROR'].data,float)
            fm=h['FIBERWCS'].data if 'FIBERWCS' in [x.name for x in h] else h['FIBERMAP'].data
            ra=np.asarray(fm['RA_FIBERMAP'] if 'RA_FIBERMAP' in fm.names else fm['RA'],float)
            dec=np.asarray(fm['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fm.names else fm['DEC'],float)
        ok=(msk==0)&np.isfinite(dat)
        for i in range(dat.shape[0]):
            if ok[i].sum()<50: continue
            RA.append(ra[i]);DEC.append(dec[i]);EXP.append(ei);WL.append(np.nanmedian(dat[i][ok[i]]))
            VAR.append(np.nanmedian(err[i][ok[i]])**2 if np.isfinite(err[i][ok[i]]).any() else 1.0);NP.append(int(ok[i].sum()))
    return list(map(np.array,(RA,DEC,EXP,WL,VAR,NP)))
def co(A,sel=None):
    RA,DEC,EXP,WL,VAR,NP=A
    if sel is None: sel=np.ones(RA.size,bool)
    return coadd_image(FibreTable(ra=RA[sel],dec=DEC[sel],value=WL[sel],var=VAR[sel],
        solid_angle=np.full(int(sel.sum()),0.44),exposure=EXP[sel],
        channel=np.array(['green']*int(sel.sum())),npix=NP[sel]),units='flux',weighting='ivar')
def hp(im):
    d=im.data; core=im.nexp>=np.nanmax(im.nexp)-0.5
    src=d>np.nanpercentile(d[core&np.isfinite(d)],90); bp=core&np.isfinite(d)&~src
    h=d-median_filter(np.nan_to_num(d,nan=np.nanmedian(d)),size=9)
    return np.where(bp,h,np.nan),np.nanstd(h[bp])
res={}
for v in ('off','on'):
    A=load(v)
    res[v]=dict(f0=co(A,A[2]==0),st=co(A))
fig,ax=plt.subplots(2,3,figsize=(16.5,10))
lo,hi=ZScaleInterval().get_limits(res['off']['st'].data[np.isfinite(res['off']['st'].data)])
h0,_=hp(res['off']['f0']); hl=np.nanpercentile(np.abs(h0),95)
for r,v in enumerate(('off','on')):
    f0h,r0=hp(res[v]['f0']); sth,rs=hp(res[v]['st'])
    ax[r,0].imshow(res[v]['st'].data,origin='lower',vmin=lo,vmax=hi,cmap='inferno')
    ax[r,0].set_title(f'{v.upper()}: 3-frame stack white-light')
    ax[r,1].imshow(f0h,origin='lower',vmin=-hl,vmax=hl,cmap='RdBu_r')
    ax[r,1].set_title(f'{v.upper()}: SINGLE frame high-pass (RMS {r0:.2f})')
    ax[r,2].imshow(sth,origin='lower',vmin=-hl,vmax=hl,cmap='RdBu_r')
    ax[r,2].set_title(f'{v.upper()}: STACK high-pass (RMS {rs:.2f})')
fig.suptitle('Template pedestal OFF (top) vs ON (bottom) — J1613 green, registered, common scales',fontsize=13)
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig)
print('wrote',OUT)
