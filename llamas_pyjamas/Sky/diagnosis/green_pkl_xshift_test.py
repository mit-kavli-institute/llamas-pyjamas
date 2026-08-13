"""DECISIVE xshift-vs-wavelength test (pkl domain). For green 5577 [OI], per fibre, measure the
line-centroid misalignment between the extracted COUNTS and the base SKY model IN NATIVE PIXELS
(the xshift domain the base sky is built in): dcent = centroid(counts) - centroid(sky).
If the fibre-to-fibre shift residual seen in the RSS/wavelength frame (field-dependent, +/- across
fields) is an ARTIFACT of wave-grid resampling, dcent here should be ~0 mean, small spread, and NOT
field-dependent -> the refinement belongs in the pkl/xshift domain. Also reports the native at-line
fractional residual (counts-sky)/amp to compare with the RSS-frame ~9-11%."""
import sys; sys.path.insert(0,'/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import pickle, glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
D='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01_pkl/extractions'
OUT='/Users/simcoe/.claude/jobs/8fc668fa/tmp'
LAM=5577.34; HALF=4.0   # Angstrom half-window
FRAMES={'J0958':'00-05-50.8','J1613':'02-49-56.7','J2151':'08-24-02.8'}

def frame_stats(tag, stamp):
    f=glob.glob(f'{D}/*{stamp}*sky1d*extractions.pkl')[0]
    obj=pickle.load(open(f,'rb')); ex=obj['extractions']; md=obj['metadata']
    rows=[]
    for i,e in enumerate(ex):
        m=md[i]
        if str(m.get('channel')).lower()!='green': continue
        bs=f"{m.get('bench')}{m.get('side')}"
        W=np.asarray(e.wave,float); C=np.asarray(e.counts,float); S=np.asarray(e.sky,float)
        nf=W.shape[0]
        for fb in range(nf):
            w=W[fb]; sel=np.where(np.abs(w-LAM)<HALF)[0]
            if sel.size<7: continue
            idx=sel; c=C[fb][idx]; s=S[fb][idx]
            if not (np.all(np.isfinite(c))and np.all(np.isfinite(s))): continue
            # continuum via endpoints (linear)
            xr=np.arange(idx.size)
            def sub(y):
                a=np.polyfit([xr[0],xr[-1]],[np.median(y[:2]),np.median(y[-2:])],1)
                return y-np.polyval(a,xr)
            cb=sub(c); sb=sub(s)
            amp_s=np.nanmax(sb)
            if not np.isfinite(amp_s) or amp_s<50: continue        # line detected in the model
            cp=np.clip(cb,0,None); sp=np.clip(sb,0,None)
            if cp.sum()<=0 or sp.sum()<=0: continue
            cen_c=np.sum(idx*cp)/cp.sum(); cen_s=np.sum(idx*sp)/sp.sum()
            dcent=cen_c-cen_s                                        # native-pixel misalignment
            resid=c-s; fres=np.std(resid)/amp_s                     # native at-line fractional residual
            rows.append((bs, fb, dcent, fres, amp_s))
    bs_arr=np.array([r[0] for r in rows]); dc=np.array([r[2] for r in rows])
    fr=np.array([r[3] for r in rows])
    print(f'{tag}: nfib={len(rows)}  dcent(native px) mean={np.mean(dc):+.4f} med={np.median(dc):+.4f} '
          f'MAD={1.4826*np.median(np.abs(dc-np.median(dc))):.4f}  |native frac_resid med={np.median(fr):.3f}')
    return rows

fig,ax=plt.subplots(1,2,figsize=(14,5))
allrows={}
for tag,stamp in FRAMES.items():
    rows=frame_stats(tag,stamp); allrows[tag]=rows
    dc=np.array([r[2] for r in rows])
    ax[0].hist(dc,bins=np.linspace(-1,1,61),histtype='step',lw=2,label=f'{tag} (med {np.median(dc):+.3f})')
    # vs slit rank within benchside
    bslist=sorted(set(r[0] for r in rows));
    rank=[]; dcs=[]
    for bs in bslist:
        fbs=sorted([r[1] for r in rows if r[0]==bs]);
        for r in rows:
            if r[0]==bs:
                rr=(fbs.index(r[1])/(len(fbs)-1)) if len(fbs)>1 else 0.5
                rank.append(rr); dcs.append(r[2])
    rank=np.array(rank); dcs=np.array(dcs)
    bins=np.linspace(0,1,11); bc=0.5*(bins[:-1]+bins[1:])
    binned=np.array([np.nanmedian(dcs[(rank>=bins[k])&(rank<bins[k+1])]) for k in range(len(bc))])
    ax[1].plot(bc,binned,'o-',label=tag)
ax[0].axvline(0,color='k',lw=.5); ax[0].set_xlabel('dcent = centroid(counts)-centroid(sky), native px')
ax[0].set_title('5577 native-pixel (xshift-domain) line misalignment'); ax[0].legend()
ax[1].axhline(0,color='k',lw=.5); ax[1].set_xlabel('slit position (within benchside)'); ax[1].set_ylabel('median dcent (native px)')
ax[1].set_title('native dcent vs slit position (cf. RSS shift was field-dep +/-)'); ax[1].legend()
fig.tight_layout(); p=f'{OUT}/green_pkl_xshift_test.png'; fig.savefig(p,dpi=90); plt.close(fig)
print('FIG',p)
