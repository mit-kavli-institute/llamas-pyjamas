"""Phase B prototype: static per-(camera, slit-position) LSF-residual template.
Build (held-out on J0958+J1613): after the Phase-A per-line refine, stack the leftover line-residual
PROFILE (fixed-xshift/offset frame, normalized by line amp) per camera and slit-position bin, robust
median (rejects object positives). Validate on J2151 (held out): does adding the template push the
at-line residual below the ~12% Phase-A floor, and does it GENERALIZE across fields (field-independent)?"""
import sys; sys.path.insert(0,'/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import pickle, glob, warnings; warnings.filterwarnings('ignore')
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from llamas_pyjamas.Sky.skyLineRefine import refine_fibre
D='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01_pkl/extractions'
OUT='/Users/simcoe/.claude/jobs/8fc668fa/tmp'
BUILD=['00-05-50.8','00-33-44.3','01-01-33.7','01-32-28.6','01-55-56.0',        # J0958 x5
       '02-49-56.7','03-30-45.4','04-13-54.3','04-56-10.5','05-43-06.1','06-22-48.3','07-03-13.7','07-43-54.7']  # J1613 x8
VALID=['08-24-02.8','08-49-49.1','09-28-44.4','10-08-24.8']                     # J2151 x4
LINES=[5577.34,6300.30]           # isolated singlets for the LSF shape
OFF=np.arange(-6,6.01,0.3); RANKBINS=8; AMPFLOOR=200.0; PADPIX=12

def refined_green(e):
    S=np.asarray(e.sky,float); C=np.asarray(e.counts,float); R=S.copy()
    for fb in range(S.shape[0]):
        R[fb]=S[fb]+refine_fibre(C[fb],S[fb])
    return R

def iter_profiles(stamps):
    """yield (cam, rankbin, amp, profile_on_OFF, resid_window, off_window) for green lines."""
    for stamp in stamps:
        obj=pickle.load(open(glob.glob(f'{D}/*{stamp}*sky1d*extractions.pkl')[0],'rb'))
        sci=obj['extractions']; md=obj['metadata']
        for i,e in enumerate(sci):
            m=md[i]
            if str(m.get('channel')).lower()!='green': continue
            cam=f"{m.get('bench')}{m.get('side')}"
            W=np.asarray(e.wave,float); C=np.asarray(e.counts,float); X=np.asarray(e.xshift,float)
            Sr=refined_green(e); nf=W.shape[0]
            for fb in range(nf):
                w0=W[fb]
                if not np.all(np.isfinite(w0)): continue
                rbin=min(RANKBINS-1,int((fb/(nf-1))*RANKBINS))
                for lam in LINES:
                    if not (np.nanmin(w0)<lam<np.nanmax(w0)): continue
                    pix=int(np.argmin(np.abs(w0-lam))); lo=max(0,pix-PADPIX); hi=min(w0.size,pix+PADPIX+1)
                    sl=slice(lo,hi); xr=np.arange(hi-lo,dtype=float)
                    sky=Sr[fb][sl]; c=C[fb][sl]; xsh=X[fb][sl]
                    if not(np.all(np.isfinite(sky))and np.all(np.isfinite(c))and np.all(np.isfinite(xsh))): continue
                    a=np.polyfit([xr[0],xr[-1]],[np.median(sky[:2]),np.median(sky[-2:])],1); skl=sky-np.polyval(a,xr)
                    amp=np.nanmax(skl)
                    if amp<AMPFLOOR: continue
                    x0=X[fb][pix]; off=xsh-x0
                    resid=c-sky; b=np.polyfit([xr[0],xr[-1]],[np.median(resid[:2]),np.median(resid[-2:])],1)
                    resid=resid-np.polyval(b,xr)                 # remove local DC
                    prof=np.interp(OFF, off, resid/amp, left=np.nan, right=np.nan)
                    yield cam, rbin, amp, prof, resid, off

# --- build (held out) ---
store=defaultdict(lambda: defaultdict(list))
for cam,rbin,amp,prof,_,_ in iter_profiles(BUILD):
    store[cam][rbin].append(prof)
template={}
for cam in store:
    T=np.full((RANKBINS,OFF.size),0.0)
    for rb in range(RANKBINS):
        if store[cam].get(rb):
            T[rb]=np.nanmedian(np.vstack(store[cam][rb]),axis=0)   # robust to object positives
    template[cam]=np.nan_to_num(T)
print(f'built LSF template: {len(template)} green cameras, {RANKBINS} slit bins x {OFF.size} offsets')

# --- validate on held-out J2151: core/wing/full, fixed-amp vs fitted-amp template (best case) ---
WINS={'core |off|<2':lambda o:np.abs(o)<2.0,'wing 2-5':lambda o:(np.abs(o)>=2)&(np.abs(o)<5),'full |off|<5':lambda o:np.abs(o)<5.0}
res={w:{'A':[],'Tfix':[],'Tfit':[]} for w in WINS}
for cam,rbin,amp,prof,resid,off in iter_profiles(VALID):
    tprof=template.get(cam)
    if tprof is None or not np.isfinite(amp): continue
    tcorr_fix=np.interp(off, OFF, tprof[rbin], left=0,right=0)*amp
    for wn,wf in WINS.items():
        m=wf(off)
        if m.sum()<5: continue
        r=resid[m]; a=amp
        res[wn]['A'].append(np.nanstd(r)/a)
        res[wn]['Tfix'].append(np.nanstd(r-tcorr_fix[m])/a)
        t=tprof[rbin]; tv=np.interp(off[m],OFF,t,left=0,right=0)*a
        d=float(np.dot(r,tv)/np.dot(tv,tv)) if np.dot(tv,tv)>0 else 0.0   # fitted template amplitude
        res[wn]['Tfit'].append(np.nanstd(r-d*tv)/a)
print('J2151 (HELD OUT) median at-line frac-resid:')
for wn in WINS:
    A=np.median(res[wn]['A']); Tf=np.median(res[wn]['Tfix']); Tt=np.median(res[wn]['Tfit'])
    print(f'  {wn:14s}: PhaseA={A:.3f}  +Tfixed={Tf:.3f} ({100*(1-Tf/A):+.0f}%)  +Tfitted={Tt:.3f} ({100*(1-Tt/A):+.0f}%)')
fa=np.array(res['full |off|<5']['A']); fat=np.array(res['full |off|<5']['Tfit'])

# --- QA ---
fig,ax=plt.subplots(1,2,figsize=(14,5))
cam0=sorted(template)[0]; cols=plt.cm.viridis(np.linspace(0,1,RANKBINS))
for rb in range(RANKBINS):
    ax[0].plot(OFF,template[cam0][rb],color=cols[rb],lw=1.8,label=f'slit {rb/RANKBINS:.2f}-{(rb+1)/RANKBINS:.2f}')
ax[0].axhline(0,color='k',lw=.5); ax[0].set_title(f'LSF-residual template (green {cam0}) vs slit position')
ax[0].set_xlabel('xshift offset from line center'); ax[0].set_ylabel('residual / line amp'); ax[0].legend(fontsize=7,ncol=2)
ax[1].hist(fa,bins=np.linspace(0,0.4,50),histtype='step',lw=2,color='0.5',label=f'Phase A (med {np.median(fa):.3f})')
ax[1].hist(fat,bins=np.linspace(0,0.4,50),histtype='step',lw=2,color='C3',label=f'A+template (med {np.median(fat):.3f})')
ax[1].set_xlabel('at-line frac-resid (J2151, held out)'); ax[1].set_ylabel('N'); ax[1].legend()
ax[1].set_title('held-out validation: LSF template on top of Phase A')
fig.tight_layout(); p=f'{OUT}/lsf_template_proto.png'; fig.savefig(p,dpi=95); plt.close(fig)
print('wrote',p)
