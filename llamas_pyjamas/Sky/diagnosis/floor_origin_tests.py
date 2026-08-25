"""Two discriminators for the along-slit floor origin (J1613 green, ped_off pkl):
A) floor_i vs MEASURED OH-line width per fibre (second moment on the xshift grid) — LSF-width
   wing-redistribution predicts a strong smoothed correlation along the slit.
B) floor amplitude vs DISTANCE from the nearest sky line — wing contamination decays with
   distance; a true scattered/veil continuum does not."""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from scipy.ndimage import median_filter, distance_transform_edt
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
OUT=f'{ND}/lsf_floor_qa.png'
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

pkl=sorted(glob.glob(f'{ND}/reduced_ped_off/extractions/LLAMAS_2026-05-17_02-49-56.7*sky1d_extractions.pkl'))[0]
d=ExtractLlamas.loadExtraction(pkl)
Ns=[2,5,10,20,40]
print(f"{'cam':>4} {'nblank':>6} {'corr(floor,width)sm':>19}  arch-std vs line-distance N={Ns}")
res={}
for ext,md in zip(d['extractions'],d['metadata']):
    if str(md.get('channel','')).lower()!='green': continue
    cam=f"{md['bench']}{md['side']}"
    C=np.asarray(ext.counts,float); S=np.asarray(ext.sky,float); X=np.asarray(ext.xshift,float)
    nf=C.shape[0]
    wl=np.nanmedian(C,axis=1)
    good=np.isfinite(wl)&(wl>0)&np.any(np.isfinite(S)&(S!=0),axis=1)
    blank=good&(wl<=np.nanpercentile(wl[good],60))
    j0=np.where(blank)[0][int(blank.sum()//2)]
    s0=np.nan_to_num(S[j0]); x0g=X[j0]
    # 3 strongest isolated line positions (xshift units) from the model
    cand=np.argsort(s0)[::-1]; picks=[]
    for c in cand:
        if len(picks)>=3: break
        if all(abs(c-p)>40 for p in picks) and 60<c<s0.size-60: picks.append(c)
    line_x=[x0g[c] for c in picks]
    # A) per-fibre line width (flux-weighted 2nd moment over +-6 px, continuum ring 8-12 px)
    width=np.full(nf,np.nan); floor=np.full(nf,np.nan)
    fldist={N: np.full(nf,np.nan) for N in Ns}
    dx=np.arange(-12,13,1.0)
    for i in np.where(blank)[0]:
        xi=X[i]; ci=np.nan_to_num(C[i]); si=np.nan_to_num(S[i])
        ok=np.isfinite(C[i])&np.isfinite(S[i])
        if ok.sum()<300: continue
        ws=[]; wgt=[]
        for lx in line_x:
            prof=np.interp(lx+dx, xi, ci)
            cont=np.median(prof[np.abs(dx)>=8])
            p=np.clip(prof-cont,0,None)
            core=np.abs(dx)<=6
            F=p[core].sum()
            if F<50: continue
            mu=(p[core]*dx[core]).sum()/F
            ws.append(np.sqrt(max((p[core]*(dx[core]-mu)**2).sum()/F,1e-3))); wgt.append(F)
        if ws: width[i]=np.average(ws,weights=wgt)
        # B) floor vs distance from lines (line mask from the fibre's own sky model)
        med=np.nanmedian(si[ok]) if np.isfinite(si[ok]).any() else 0
        linemask=si>max(1.5*med,np.nanpercentile(si[ok],70))
        dist=distance_transform_edt(~linemask)
        r=ci-si
        for N in Ns:
            m=ok&(dist>N)
            if m.sum()>40: fldist[N][i]=np.nanmedian(r[m])
        m0=ok&(si<np.nanpercentile(si[ok],30))
        if m0.sum()>50: floor[i]=np.nanmedian(r[m0])
    m=blank&np.isfinite(width)&np.isfinite(floor)
    if m.sum()<30: continue
    fs=median_filter(floor[m],size=15,mode='nearest'); wsm=median_filter(width[m],size=15,mode='nearest')
    r_sm=np.corrcoef(fs,wsm)[0,1]
    stds=[float(np.nanstd(fldist[N][m])) for N in Ns]
    print(f"{cam:>4} {int(m.sum()):>6} {r_sm:>+19.2f}  {['%.2f'%s for s in stds]}")
    res[cam]=dict(idx=np.where(m)[0],fs=fs,ws=wsm,stds=stds)
show=[b for b in ('1A','2A','3B','4B') if b in res]
fig,ax=plt.subplots(2,len(show),figsize=(4.4*len(show),8),squeeze=False)
for j,b in enumerate(show):
    R=res[b]; a=ax[0,j]; a2=a.twinx()
    a.plot(R['fs'],'.',ms=3,color='C0'); a2.plot(R['ws'],'.',ms=3,color='C1')
    a.set_title(f"{b}: floor(C0) vs line width(C1), smoothed")
    ax[1,j].plot(Ns,R['stds'],'o-'); ax[1,j].set_xlabel('min px from any line'); ax[1,j].set_ylabel('arch std')
    ax[1,j].set_title(f'{b}: floor amplitude vs line distance')
fig.suptitle('Floor origin: LSF width tracking (top) and line-wing decay (bottom) — J1613 green')
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig)
print("wrote",OUT)
