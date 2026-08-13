"""Production combine on the off/on 8-frame reductions: sync registered WCS green->blue/red,
build_super_rss (bad-fibre masking) + transparency + combine_field_cubes per set; write cubes,
compare white-lights (stripe metric + PNG)."""
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
from llamas_pyjamas.Combine.transparency import transparency_scales

# 1. sync registered green WCS into blue/red siblings (fibre sky positions are channel-independent)
for v in ('off','on'):
    for fg in sorted(glob.glob(f'{ND}/reduced_ped_{v}/extractions/*_RSS_green.fits')):
        with fits.open(fg) as hg:
            if 'FIBERWCS' not in [x.name for x in hg]: continue
            fw=hg['FIBERWCS'].copy(); fm=hg['FIBERMAP'].data
        for chan in ('blue','red'):
            fc=fg.replace('_RSS_green',f'_RSS_{chan}')
            if not os.path.exists(fc): continue
            with fits.open(fc,mode='update') as hc:
                names=[x.name for x in hc]
                if 'FIBERWCS' in names:
                    hc['FIBERWCS'].data=fw.data; hc['FIBERWCS'].header.update(fw.header)
                else: hc.append(fw.copy())
                hc['FIBERMAP'].data['RA']=fm['RA']; hc['FIBERMAP'].data['DEC']=fm['DEC']; hc.flush()
print('WCS synced to blue/red in both sets', flush=True)

# 2. production combine per set
cubes={}
for v in ('off','on'):
    paths=sorted(glob.glob(f'{ND}/reduced_ped_{v}/extractions/*_RSS_green.fits'))
    print(f'=== building super-RSS ({v}, {len(paths)} exposures) ===', flush=True)
    sr=build_super_rss(paths)
    try: sr.apply_scales(transparency_scales(sr))
    except Exception as e: print('transparency skipped:',e)
    cc=combine_field_cubes(sr,units='sb')
    outdir=f'{ND}/reduced_ped_{v}/combined'; os.makedirs(outdir,exist_ok=True)
    for c,cu in cc.items():
        out=f'{outdir}/J1613_cube_{c}.fits'; cu.write(out); print('wrote',out, flush=True)
    cubes[v]=cc

# 3. white-light comparison + stripe metric per channel
TH=np.deg2rad(145.0)
def stripe(wl,nexp):
    core=nexp>=np.nanmax(nexp)-0.5
    src=wl>np.nanpercentile(wl[core&np.isfinite(wl)],90); bp=core&np.isfinite(wl)&~src
    h=wl-median_filter(np.nan_to_num(wl,nan=np.nanmedian(wl)),size=9)
    ny,nx=h.shape; yy,xx=np.mgrid[0:ny,0:nx]
    m=bp&np.isfinite(h); p=(xx*np.cos(TH)+yy*np.sin(TH))[m]; vv=h[m]
    nb=80; pe=np.linspace(p.min(),p.max(),nb+1)
    prof=np.array([np.nanmedian(vv[(p>=pe[k])&(p<pe[k+1])]) if ((p>=pe[k])&(p<pe[k+1])).sum()>4 else np.nan for k in range(nb)])
    return float(np.nanstd(prof)),np.where(bp,h,np.nan)
fig,ax=plt.subplots(3,3,figsize=(16,15))
for r,chan in enumerate(('blue','green','red')):
    W={}; H={}; S={}
    for v in ('off','on'):
        cu=cubes[v][chan]; wl=cu.white_light()
        S[v],H[v]=stripe(wl,cu.nexp); W[v]=wl
    print(f"{chan}: production-combine stack stripe amplitude  off {S['off']:.4g} -> on {S['on']:.4g}  ({100*(1-S['on']/S['off']):+.0f}%)")
    lo,hi=ZScaleInterval().get_limits(W['off'][np.isfinite(W['off'])])
    ax[r,0].imshow(W['off'],origin='lower',vmin=lo,vmax=hi,cmap='inferno'); ax[r,0].set_title(f'{chan} OFF white-light')
    ax[r,1].imshow(W['on'],origin='lower',vmin=lo,vmax=hi,cmap='inferno'); ax[r,1].set_title(f'{chan} ON white-light')
    hl=np.nanpercentile(np.abs(H['off'][np.isfinite(H['off'])]),97)
    ax[r,2].imshow(H['on'],origin='lower',vmin=-hl,vmax=hl,cmap='RdBu_r')
    ax[r,2].set_title(f"{chan} ON high-pass ({100*(1-S['on']/S['off']):+.0f}%)")
fig.suptitle('PRODUCTION combine (bad-fibre masking + transparency): pedestal OFF vs template ON, 8 dithers',fontsize=13)
fig.savefig(f'{ND}/prodcombine_qa.png',dpi=110,bbox_inches='tight',facecolor='white')
print('wrote',f'{ND}/prodcombine_qa.png')
print('DONE')
