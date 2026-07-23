"""2D sky-line residual QA in the FIXED-XSHIFT frame. Each fibre's residual (counts-sky) is resampled
onto a common xshift grid (the coordinate the base sky is built in), so a sky line straightens into a
vertical band and the across-slit shift/asymmetry reads directly as vertical (fibre) variation.
BEFORE (base sky) vs AFTER (pkl per-line refine) vs removed, for 4 prominent green lines/sets."""
import sys; sys.path.insert(0,'/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import pickle, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from llamas_pyjamas.Sky.skyLineRefine import refine_sky_lines_pkl
D='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01_pkl/extractions'
OUT='/Users/simcoe/.claude/jobs/8fc668fa/tmp'
stamp='08-24-02.8'                                    # J2151 green 2A
LINES=[('5577 [OI]',5577.34,8.0),('NaD 5890',5889.95,9.0),('6300 [OI]',6300.30,9.0),('OH/O2 ~6863',6863.0,14.0)]

obj=pickle.load(open(glob.glob(f'{D}/*{stamp}*sky1d*extractions.pkl')[0],'rb'))
sci=obj['extractions']; md=obj['metadata']
gi=[i for i,m in enumerate(md) if str(m.get('channel')).lower()=='green' and str(m.get('bench'))=='2' and str(m.get('side')).upper()=='A'][0]
base=np.asarray(sci[gi].sky,float).copy()
refine_sky_lines_pkl(sci,{},metadata=md)
e=sci[gi]; W=np.asarray(e.wave,float); C=np.asarray(e.counts,float); Sa=np.asarray(e.sky,float); X=np.asarray(e.xshift,float)
nf=W.shape[0]; mid=nf//2
RB=C-base; RA=C-Sa                                     # residuals

def resample(R, grid):
    img=np.full((nf,grid.size),np.nan)
    for f in range(nf):
        xf=X[f]; r=R[f]; ok=np.isfinite(xf)&np.isfinite(r)
        if ok.sum()<50: continue
        o=np.argsort(xf[ok]); img[f]=np.interp(grid, xf[ok][o], r[ok][o], left=np.nan, right=np.nan)
    return img

fig,ax=plt.subplots(len(LINES),3,figsize=(15,4.2*len(LINES)))
for row,(tag,lam,dx) in enumerate(LINES):
    pix=np.argmin(np.abs(W[mid]-lam)); x0=X[mid][pix]
    grid=np.linspace(x0-dx, x0+dx, int(dx*8))
    ib=resample(RB,grid); ia=resample(RA,grid); idf=ib-ia
    vlim=np.nanpercentile(np.abs(ib[np.isfinite(ib)]),97) or 1.0
    ext=[grid[0]-x0,grid[-1]-x0,0,nf]
    for col,(img,ttl) in enumerate([(ib,'BEFORE'),(ia,'AFTER refine'),(idf,'removed')]):
        a=ax[row,col]; im=a.imshow(img,origin='lower',aspect='auto',cmap='RdBu_r',vmin=-vlim,vmax=vlim,extent=ext)
        a.set_title(f'{tag}  {ttl}',fontsize=9); a.set_xlabel('xshift - line center');
        if col==0: a.set_ylabel('fibre')
        if col==2: fig.colorbar(im,ax=a,fraction=0.046)
    rb=np.nanstd(ib); ra=np.nanstd(ia)
    print(f'{tag}: xshift-window RMS before={rb:.1f} after={ra:.1f} cts ({100*(1-ra/rb):.0f}% lower)')
fig.suptitle('J2151 green 2A — 2D residual in FIXED-XSHIFT frame (line = vertical band): base vs pkl-refine',fontsize=11)
fig.tight_layout(); p=f'{OUT}/qa_2d_residual_xshift.png'; fig.savefig(p,dpi=92); plt.close(fig)
print('wrote',p)
