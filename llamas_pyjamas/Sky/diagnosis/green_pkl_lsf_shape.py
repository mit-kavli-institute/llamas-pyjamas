"""Characterize the residual LSF SHAPE vs slit position (native/xshift domain), to define what a
static per-slit-position template must model. Green 5577, per fibre: fit counts_line ~ a*sky + b*sky'
+ c*sky'' (amplitude+shift+width), take the leftover residual, normalize by line amp, align to the
sky-line centroid, and median-stack in slit-position bins. If the stacked profiles differ across the
slit (e.g. asymmetry flipping edge-to-edge), that across-slit LSF structure is the template target."""
import sys; sys.path.insert(0,'/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import pickle, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01_pkl/extractions'
OUT='/Users/simcoe/.claude/jobs/8fc668fa/tmp'
LAM=5577.34; HALF=4.0
FRAMES=['00-05-50.8','02-49-56.7','08-24-02.8']   # J0958,J1613,J2151
GRID=np.arange(-5,5.01,0.25)                        # common pixel-offset grid
NB=5                                                # slit bins
prof_acc=[[] for _ in range(NB)]; rms2=[]; rms_lsf=[]; ranks=[]
for stamp in FRAMES:
    obj=pickle.load(open(glob.glob(f'{D}/*{stamp}*sky1d*extractions.pkl')[0],'rb'))
    ex=obj['extractions']; md=obj['metadata']
    for i,e in enumerate(ex):
        if str(md[i].get('channel')).lower()!='green': continue
        W=np.asarray(e.wave,float); C=np.asarray(e.counts,float); S=np.asarray(e.sky,float)
        nf=W.shape[0]
        for fb in range(nf):
            w=W[fb]; sel=np.where(np.abs(w-LAM)<HALF)[0]
            if sel.size<7: continue
            c=C[fb][sel]; s=S[fb][sel]; xr=np.arange(sel.size)
            if not (np.all(np.isfinite(c))and np.all(np.isfinite(s))): continue
            def sub(y):
                a=np.polyfit([xr[0],xr[-1]],[np.median(y[:2]),np.median(y[-2:])],1); return y-np.polyval(a,xr)
            cb=sub(c); sb=sub(s); amp=np.nanmax(sb)
            if not np.isfinite(amp) or amp<50: continue
            s1=np.gradient(sb); s2=np.gradient(s1)
            B=np.vstack([sb,s1,s2]).T
            coef,*_=np.linalg.lstsq(B,cb,rcond=None)
            resid=cb-B@coef                                     # after amp+shift+width
            cen=np.sum(xr*np.clip(sb,0,None))/np.clip(sb,0,None).sum()
            off=xr-cen
            rn=np.interp(GRID,off,resid/amp,left=np.nan,right=np.nan)
            rank=(fb/(nf-1))
            b=min(NB-1,int(rank*NB)); prof_acc[b].append(rn); ranks.append(rank)
            rms2.append(np.std(cb)/amp)                         # before (amp only baseline ~raw)
            rms_lsf.append(np.std(resid)/amp)                   # after amp+shift+width
rms2=np.array(rms2); rms_lsf=np.array(rms_lsf)
print(f'green 5577 (3 fields, {len(rms_lsf)} fibres):')
print(f'  frac resid: raw≈{np.median(rms2):.3f}  after amp+shift+width fit={np.median(rms_lsf):.3f} '
      f'({100*(1-np.median(rms_lsf)/np.median(rms2)):.0f}% removed by 2nd-order model)')
fig,ax=plt.subplots(1,2,figsize=(14,5))
cols=plt.cm.viridis(np.linspace(0,1,NB))
for b in range(NB):
    if not prof_acc[b]: continue
    m=np.nanmedian(np.vstack(prof_acc[b]),axis=0)
    ax[0].plot(GRID,m,color=cols[b],lw=2,label=f'slit bin {b} ({b/NB:.1f}-{(b+1)/NB:.1f})')
ax[0].axhline(0,color='k',lw=.5); ax[0].set_xlabel('pixel offset from line center')
ax[0].set_ylabel('median residual / line amp (after amp+shift+width)')
ax[0].set_title('green 5577 leftover LSF-shape residual vs slit position'); ax[0].legend(fontsize=8)
# residual magnitude vs slit
ranks=np.array(ranks); bins=np.linspace(0,1,11); bc=0.5*(bins[:-1]+bins[1:])
bl=np.array([np.nanmedian(rms_lsf[(ranks>=bins[k])&(ranks<bins[k+1])]) for k in range(len(bc))])
ax[1].plot(bc,bl,'o-'); ax[1].set_xlabel('slit position'); ax[1].set_ylabel('frac resid after amp+shift+width')
ax[1].set_title('leftover LSF residual magnitude vs slit position'); ax[1].set_ylim(0,(np.nanpercentile(bl,100) or 0.2)*1.2 if np.isfinite(np.nanpercentile(bl,100)) else 0.2)
fig.tight_layout(); p=f'{OUT}/green_pkl_lsf_shape.png'; fig.savefig(p,dpi=90); plt.close(fig)
print('FIG',p)
