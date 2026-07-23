"""2D (fibre x pixel) sky-line residual QA. For one green camera, show the base sky-subtracted residual
counts-sky BEFORE vs AFTER the pkl-domain per-line refine, in a pixel window around 5577 [OI] (and a
wider OH region), same colour scale. Reveals coherent 2D structure (line tilt / per-fibre shift, the
across-slit wing asymmetry) that per-fibre 1D stats average over."""
import sys; sys.path.insert(0,'/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import pickle, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from llamas_pyjamas.Sky.skyLineRefine import refine_sky_lines_pkl
D='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01_pkl/extractions'
OUT='/Users/simcoe/.claude/jobs/8fc668fa/tmp'
stamp='08-24-02.8'   # J2151 (largest residual -> most visible)

obj=pickle.load(open(glob.glob(f'{D}/*{stamp}*sky1d*extractions.pkl')[0],'rb'))
sci=obj['extractions']; md=obj['metadata']
# pick green bench=2 side=A
gi=[i for i,m in enumerate(md) if str(m.get('channel')).lower()=='green' and str(m.get('bench'))=='2' and str(m.get('side')).upper()=='A'][0]
base=np.asarray(sci[gi].sky,float).copy()
refine_sky_lines_pkl(sci,{},metadata=md)
e=sci[gi]; W=np.asarray(e.wave,float); C=np.asarray(e.counts,float); Sa=np.asarray(e.sky,float)
nf=W.shape[0]

def window(lam, halfpix):
    col=int(np.nanmedian([np.argmin(np.abs(W[f]-lam)) for f in range(nf) if np.all(np.isfinite(W[f]))]))
    return slice(max(0,col-halfpix), min(W.shape[1],col+halfpix))

def show(ax, img, title, vlim):
    im=ax.imshow(img, origin='lower', aspect='auto', cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    ax.set_title(title, fontsize=9); ax.set_xlabel('pixel'); ax.set_ylabel('fibre'); return im

fig,ax=plt.subplots(2,3,figsize=(15,9))
for row,(lam,hp,tag) in enumerate([(5577.34,35,'5577 [OI]'),(6300.30,45,'6300 [OI]')]):
    sl=window(lam,hp)
    rb=(C-base)[:,sl]; ra=(C-Sa)[:,sl]; diff=rb-ra
    vlim=np.nanpercentile(np.abs(rb[np.isfinite(rb)]),97)
    show(ax[row,0], rb, f'{tag}  BEFORE (counts-base sky)', vlim)
    show(ax[row,1], ra, f'{tag}  AFTER (pkl per-line refine)', vlim)
    im=show(ax[row,2], diff, f'{tag}  removed (before-after)', vlim)
    fig.colorbar(im, ax=ax[row,2], fraction=0.046)
    print(f'{tag}: window RMS  before={np.nanstd(rb):.1f}  after={np.nanstd(ra):.1f} cts '
          f'({100*(1-np.nanstd(ra)/np.nanstd(rb)):.0f}% lower)')
fig.suptitle(f'J2151 green 2A — 2D sky-line residual (fibre x pixel), base vs pkl-refine', fontsize=11)
fig.tight_layout(); p=f'{OUT}/qa_2d_residual.png'; fig.savefig(p,dpi=95); plt.close(fig)
print('wrote',p)
