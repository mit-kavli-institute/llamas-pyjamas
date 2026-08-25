"""Blue + red off/on 8-frame stack comparison (same stripe metric + visual as green)."""
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
TH=np.deg2rad(145.0)

def load(v,chan):
    RA=[];DEC=[];EXP=[];WL=[];VAR=[];NP=[]
    for ei,f in enumerate(sorted(glob.glob(f'{ND}/reduced_ped_{v}/extractions/*_RSS_{chan}.fits'))):
        fg=f.replace(f'_RSS_{chan}',f'_RSS_green')          # registered WCS lives in green sibling
        with fits.open(f) as h:
            dat=np.asarray(h['SKYSUB'].data,float); msk=np.asarray(h['MASK'].data)
            err=np.asarray(h['ERROR'].data,float)
            names=[x.name for x in h]
            fm=h['FIBERWCS'].data if 'FIBERWCS' in names else None
        if fm is None:
            with fits.open(fg) as hg:
                fm=hg['FIBERWCS'].data if 'FIBERWCS' in [x.name for x in hg] else hg['FIBERMAP'].data
        ra=np.asarray(fm['RA_FIBERMAP'] if 'RA_FIBERMAP' in fm.names else fm['RA'],float)
        dec=np.asarray(fm['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fm.names else fm['DEC'],float)
        ok=(msk==0)&np.isfinite(dat)
        for i in range(dat.shape[0]):
            if ok[i].sum()<50: continue
            RA.append(ra[i]);DEC.append(dec[i]);EXP.append(ei);WL.append(np.nanmedian(dat[i][ok[i]]))
            VAR.append(np.nanmedian(err[i][ok[i]])**2 if np.isfinite(err[i][ok[i]]).any() else 1.0);NP.append(int(ok[i].sum()))
    return list(map(np.array,(RA,DEC,EXP,WL,VAR,NP)))
def co(A):
    RA,DEC,EXP,WL,VAR,NP=A
    return coadd_image(FibreTable(ra=RA,dec=DEC,value=WL,var=VAR,solid_angle=np.full(RA.size,0.44),
        exposure=EXP,channel=np.array(['x']*RA.size),npix=NP),units='flux',weighting='ivar')
def hpimg(im):
    d=im.data; core=im.nexp>=np.nanmax(im.nexp)-0.5
    src=d>np.nanpercentile(d[core&np.isfinite(d)],90); bp=core&np.isfinite(d)&~src
    return d-median_filter(np.nan_to_num(d,nan=np.nanmedian(d)),size=9),bp
def stripe(im):
    h,bp=hpimg(im); ny,nx=h.shape; yy,xx=np.mgrid[0:ny,0:nx]
    m=bp&np.isfinite(h); p=(xx*np.cos(TH)+yy*np.sin(TH))[m]; v=h[m]
    nb=80; pe=np.linspace(p.min(),p.max(),nb+1)
    prof=np.array([np.nanmedian(v[(p>=pe[k])&(p<pe[k+1])]) if ((p>=pe[k])&(p<pe[k+1])).sum()>4 else np.nan for k in range(nb)])
    return float(np.nanstd(prof))
fig,ax=plt.subplots(2,2,figsize=(12,10.5))
for j,chan in enumerate(('blue','red')):
    ims={}
    for v in ('off','on'): ims[v]=co(load(v,chan))
    s_off,s_on=stripe(ims['off']),stripe(ims['on'])
    print(f"{chan}: 8-frame stack stripe amplitude  off {s_off:.3f} -> on {s_on:.3f}  ({100*(1-s_on/s_off):+.0f}%)")
    h0,bp0=hpimg(ims['off']); hl=np.nanpercentile(np.abs(h0[bp0]),97)
    for r,v in enumerate(('off','on')):
        h,bp=hpimg(ims[v])
        ax[r,j].imshow(np.where(bp,h,np.nan),origin='lower',vmin=-hl,vmax=hl,cmap='RdBu_r')
        ax[r,j].set_title(f'{chan} {v.upper()}: stack high-pass (stripe {stripe(ims[v]):.2f})')
fig.suptitle('BLUE and RED 8-frame stacks — pedestal OFF (top) vs template ON (bottom)',fontsize=13)
fig.savefig(f'{ND}/pedestal8_bluered_qa.png',dpi=110,bbox_inches='tight',facecolor='white')
print('wrote',f'{ND}/pedestal8_bluered_qa.png')
