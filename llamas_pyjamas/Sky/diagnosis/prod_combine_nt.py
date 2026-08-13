"""Fair production combine: transparency DISABLED for both sets (the OFF set's auto-scales are
broken by the floor: 2.26/0.094 outliers), stripe metric over nexp>=4 region. Writes *_noscale cubes."""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, os, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
from llamas_pyjamas.Combine.superRSS import build_super_rss
from llamas_pyjamas.Combine.cube import combine_field_cubes
TH=np.deg2rad(145.0)
def stripe(wl,nexp):
    core=nexp>=4
    src=wl>np.nanpercentile(wl[core&np.isfinite(wl)],90); bp=core&np.isfinite(wl)&~src
    h=wl-median_filter(np.nan_to_num(wl,nan=np.nanmedian(wl)),size=9)
    ny,nx=h.shape; yy,xx=np.mgrid[0:ny,0:nx]
    m=bp&np.isfinite(h); p=(xx*np.cos(TH)+yy*np.sin(TH))[m]; vv=h[m]
    nb=80; pe=np.linspace(p.min(),p.max(),nb+1)
    prof=np.array([np.nanmedian(vv[(p>=pe[k])&(p<pe[k+1])]) if ((p>=pe[k])&(p<pe[k+1])).sum()>4 else np.nan for k in range(nb)])
    return float(np.nanstd(prof)),np.where(bp,h,np.nan)
cubes={}
for v in ('off','on'):
    paths=sorted(glob.glob(f'{ND}/reduced_ped_{v}/extractions/*_RSS_green.fits'))
    print(f'=== super-RSS {v} (no transparency) ===', flush=True)
    sr=build_super_rss(paths)                      # NO apply_scales
    cc=combine_field_cubes(sr,units='sb')
    outdir=f'{ND}/reduced_ped_{v}/combined'
    for c,cu in cc.items():
        out=f'{outdir}/J1613_cube_{c}_noscale.fits'; cu.write(out); print('wrote',out, flush=True)
    cubes[v]=cc
fig,ax=plt.subplots(3,3,figsize=(16,15))
for r,chan in enumerate(('blue','green','red')):
    W={};H={};S={}
    for v in ('off','on'):
        cu=cubes[v][chan]; wl=cu.white_light(); S[v],H[v]=stripe(wl,cu.nexp); W[v]=wl
    print(f"{chan}: no-transparency stack stripe (nexp>=4)  off {S['off']:.4g} -> on {S['on']:.4g}  ({100*(1-S['on']/S['off']):+.0f}%)")
    lo,hi=ZScaleInterval().get_limits(W['off'][np.isfinite(W['off'])])
    ax[r,0].imshow(W['off'],origin='lower',vmin=lo,vmax=hi,cmap='inferno'); ax[r,0].set_title(f'{chan} OFF white-light')
    ax[r,1].imshow(W['on'],origin='lower',vmin=lo,vmax=hi,cmap='inferno'); ax[r,1].set_title(f'{chan} ON white-light')
    hl=np.nanpercentile(np.abs(H['off'][np.isfinite(H['off'])]),97)
    ax[r,2].imshow(H['on'],origin='lower',vmin=-hl,vmax=hl,cmap='RdBu_r')
    ax[r,2].set_title(f"{chan} ON high-pass ({100*(1-S['on']/S['off']):+.0f}%)")
fig.suptitle('PRODUCTION combine, NO transparency scaling (fair off/on): 8 dithers',fontsize=13)
fig.savefig(f'{ND}/prodcombine_nt_qa.png',dpi=110,bbox_inches='tight',facecolor='white')
print('wrote',f'{ND}/prodcombine_nt_qa.png')
