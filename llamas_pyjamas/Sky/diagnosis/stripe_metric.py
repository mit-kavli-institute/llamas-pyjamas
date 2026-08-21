"""Stripe-specific figure of merit: amplitude of the banding along the 145-deg stripe-normal axis
(std of the binned 1-D projection of the high-pass image) — insensitive to isotropic noise/holes."""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
from llamas_pyjamas.Combine.superRSS import FibreTable
from llamas_pyjamas.Combine.coadd import coadd_image
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
TH=np.deg2rad(145.0)

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
def stripe_amp(im):
    d=im.data; core=im.nexp>=np.nanmax(im.nexp)-0.5
    src=d>np.nanpercentile(d[core&np.isfinite(d)],90); bp=core&np.isfinite(d)&~src
    h=d-median_filter(np.nan_to_num(d,nan=np.nanmedian(d)),size=9)
    ny,nx=h.shape; yy,xx=np.mgrid[0:ny,0:nx]
    p=(xx*np.cos(TH)+yy*np.sin(TH))[bp&np.isfinite(h)]; v=h[bp&np.isfinite(h)]
    nb=60; pe=np.linspace(p.min(),p.max(),nb+1)
    prof=np.array([np.nanmedian(v[(p>=pe[k])&(p<pe[k+1])]) if ((p>=pe[k])&(p<pe[k+1])).sum()>4
                   else np.nan for k in range(nb)])
    return float(np.nanstd(prof))
print(f"{'':6s} {'single-frame':>13s} {'3-frame stack':>14s}   (stripe amplitude along 145 deg)")
res={}
for v in ('off','on'):
    A=load(v); s0=stripe_amp(co(A,A[2]==0)); ss=stripe_amp(co(A))
    res[v]=(s0,ss)
    print(f"{v:>6s} {s0:>13.3f} {ss:>14.3f}")
print(f"\nstripe reduction: single {100*(1-res['on'][0]/res['off'][0]):+.0f}%   "
      f"stack {100*(1-res['on'][1]/res['off'][1]):+.0f}%")
