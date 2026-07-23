"""QA figure for the pkl-domain per-line OH refinement: stacked 5577 residual line profile BEFORE
(base sky) vs AFTER (skyLineRefine), per field, + the at-line fractional-residual distribution."""
import sys; sys.path.insert(0,'/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import pickle, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from llamas_pyjamas.Sky.skyLineRefine import refine_sky_lines_pkl
D='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01_pkl/extractions'
OUT='/Users/simcoe/.claude/jobs/8fc668fa/tmp'
LAM=5577.34; HALF=4.0; GRID=np.arange(-5,5.01,0.25)
FRAMES={'J0958':'00-05-50.8','J1613':'02-49-56.7','J2151':'08-24-02.8'}

def lineonly(y,xr):
    a=np.polyfit([xr[0],xr[-1]],[np.median(y[:2]),np.median(y[-2:])],1); return y-np.polyval(a,xr)

fig,ax=plt.subplots(1,2,figsize=(14,5))
cols={'J0958':'C0','J1613':'C1','J2151':'C2'}
allb=[]; alla=[]
for tag,stamp in FRAMES.items():
    obj=pickle.load(open(glob.glob(f'{D}/*{stamp}*sky1d*extractions.pkl')[0],'rb'))
    sci=obj['extractions']; md=obj['metadata']
    base={i:np.asarray(e.sky,float).copy() for i,e in enumerate(sci) if str(md[i].get('channel')).lower()=='green'}
    refine_sky_lines_pkl(sci,{},metadata=md)
    pb=[]; pa=[]
    for i,e in enumerate(sci):
        if str(md[i].get('channel')).lower()!='green': continue
        W=np.asarray(e.wave,float); C=np.asarray(e.counts,float); Sa=np.asarray(e.sky,float); Sb=base[i]
        for fb in range(W.shape[0]):
            w=W[fb]; sel=np.where(np.abs(w-LAM)<HALF)[0]
            if sel.size<7: continue
            c=C[fb][sel]; sb=Sb[fb][sel]; sa=Sa[fb][sel]; xr=np.arange(sel.size)
            if not(np.all(np.isfinite(c))and np.all(np.isfinite(sb))and np.all(np.isfinite(sa))): continue
            cb=lineonly(c,xr); sbb=lineonly(sb,xr); sba=lineonly(sa,xr); amp=np.nanmax(sbb)
            if not np.isfinite(amp) or amp<50: continue
            cen=np.sum(xr*np.clip(sbb,0,None))/np.clip(sbb,0,None).sum(); off=xr-cen
            rb=(cb-sbb)/amp; ra=(cb-sba)/amp
            pb.append(np.interp(GRID,off,rb,left=np.nan,right=np.nan)); pa.append(np.interp(GRID,off,ra,left=np.nan,right=np.nan))
            allb.append(np.nanstd(rb)); alla.append(np.nanstd(ra))
    mb=np.nanmedian(np.vstack(pb),0); ma=np.nanmedian(np.vstack(pa),0)
    ax[0].plot(GRID,mb,'--',color=cols[tag],lw=1.5,alpha=.7,label=f'{tag} base')
    ax[0].plot(GRID,ma,'-',color=cols[tag],lw=2.2,label=f'{tag} refined')
ax[0].axhline(0,color='k',lw=.5); ax[0].set_xlabel('pixel offset from line center')
ax[0].set_ylabel('median residual / line amp'); ax[0].set_title('green 5577 residual line profile: base (dashed) vs refined (solid)')
ax[0].legend(fontsize=8,ncol=3)
allb=np.array(allb); alla=np.array(alla)
ax[1].hist(allb,bins=np.linspace(0,0.5,50),histtype='step',lw=2,color='0.5',label=f'base (med {np.median(allb):.3f})')
ax[1].hist(alla,bins=np.linspace(0,0.5,50),histtype='step',lw=2,color='C3',label=f'refined (med {np.median(alla):.3f})')
ax[1].set_xlabel('at-line fractional residual (RMS/amp)'); ax[1].set_ylabel('N fibres'); ax[1].legend()
ax[1].set_title(f'5577 at-line residual: {100*(1-np.median(alla)/np.median(allb)):.0f}% lower (3 fields)')
fig.tight_layout(); p=f'{OUT}/qa_line_refine.png'; fig.savefig(p,dpi=95); plt.close(fig)
print('wrote',p)
