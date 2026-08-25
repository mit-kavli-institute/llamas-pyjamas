"""Compare the auto transparency scales for the off vs on 8-frame sets (suspect for the ON
white-light blotchiness), and difference the off/on white-lights like-for-like."""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
from llamas_pyjamas.Combine.superRSS import build_super_rss
from llamas_pyjamas.Combine.transparency import transparency_scales
for v in ('off','on'):
    paths=sorted(glob.glob(f'{ND}/reduced_ped_{v}/extractions/*_RSS_green.fits'))
    sr=build_super_rss(paths)
    sc=transparency_scales(sr)
    print(f'{v}: transparency scales = {[round(x,3) for x in sc.values()]}')
# like-for-like white-light difference (green)
wl={}
for v in ('off','on'):
    with fits.open(f'{ND}/reduced_ped_{v}/combined/J1613_cube_green.fits') as h:
        cube=np.asarray(h[0].data,float); wl[v]=np.nanmedian(cube,axis=0)
ny=min(wl['off'].shape[0],wl['on'].shape[0]); nx=min(wl['off'].shape[1],wl['on'].shape[1])
d=wl['off'][:ny,:nx]-wl['on'][:ny,:nx]
fig,ax=plt.subplots(1,3,figsize=(16,4.8))
for j,(img,t) in enumerate([(wl['off'],'OFF green white-light'),(wl['on'],'ON green white-light'),(d,'OFF - ON (what changed)')]):
    lo,hi=ZScaleInterval().get_limits(img[np.isfinite(img)])
    im=ax[j].imshow(img,origin='lower',vmin=lo,vmax=hi,cmap='inferno' if j<2 else 'RdBu_r')
    ax[j].set_title(t); fig.colorbar(im,ax=ax[j],shrink=0.8)
fig.savefig(f'{ND}/prodcombine_diff_qa.png',dpi=110,bbox_inches='tight',facecolor='white')
print('wrote',f'{ND}/prodcombine_diff_qa.png')
