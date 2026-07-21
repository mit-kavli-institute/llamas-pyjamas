"""Investigation (a): why does the flat leave per-fibre banding in COUNTS?
Non-circular discriminator, per green camera (J1613 frame, ped_off pkl):
  t_OH,i  = per-fibre NIGHT-SKY throughput from OH line amplitudes (floor-immune)
  tp_i    = twilight relative_throughput (what the pipeline uses)
  ratio_i = t_OH/tp  -> twilight-vs-sky throughput mismatch
  pred_i  = cont_i * (1 - k/ratio_i), k=median(ratio)  -> predicted BETWEEN-LINE residual
  f_i     = measured between-line residual (what drives the striping)
If f ~ pred (corr high, slope ~1): root cause = twilight throughput mismatch (multiplicative).
If ratio flat but f banded: real per-fibre additive floor (scattered light)."""
import glob, os, warnings; warnings.filterwarnings('ignore')
import numpy as np
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
OUT=f'{ND}/flatinvest_qa.png'

# peek at the twilight gradient model (does the pipeline already model a twilight sky gradient?)
from astropy.io import fits
tg=f'{ND}/reduced_new/extractions/flat/twilight_gradient_model.fits'
if os.path.exists(tg):
    with fits.open(tg) as h:
        print("twilight_gradient_model.fits HDUs:")
        for hd in h:
            print("   ", hd.name, None if hd.data is None else np.shape(hd.data))

pkls=sorted(glob.glob(f'{ND}/reduced_ped_off/extractions/*sky1d*.pkl'))
print("sky1d pkls:", [os.path.basename(p) for p in pkls])
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
d=ExtractLlamas.loadExtraction(pkls[0])
science=d['extractions']; meta=d['metadata']

profs={}; allf=[]; allp=[]
print(f"\n{'cam':>4} {'nblank':>6} {'std(f)':>8} {'corr(f,pred)':>12} {'slope':>6} {'std(f-pred)':>11} {'explained':>9}")
for ext,md in zip(science,meta):
    if str(md.get('channel','')).lower()!='green': continue
    cam=f"{md['bench']}{md['side']}"
    C=np.asarray(ext.counts,float); S=np.asarray(ext.sky,float)
    tp=np.asarray(ext.relative_throughput,float)
    nf=C.shape[0]
    wl=np.nanmedian(C,axis=1)
    good=np.isfinite(wl)&(wl>0)&np.isfinite(tp)&(tp>0)&np.any(np.isfinite(S)&(S!=0),axis=1)
    if good.sum()<30: continue
    L=np.full(nf,np.nan); f=np.full(nf,np.nan); cont=np.full(nf,np.nan)
    for i in np.where(good)[0]:
        s=S[i]; fin=np.isfinite(s)&np.isfinite(C[i])
        if fin.sum()<200: good[i]=False; continue
        hi=fin&(s>np.nanpercentile(s[fin],85)); lo=fin&(s<np.nanpercentile(s[fin],30))
        if hi.sum()<20 or lo.sum()<50: good[i]=False; continue
        cont[i]=np.nanmedian(C[i][lo])                    # between-line counts level
        L[i]=np.nanmedian(C[i][hi])-cont[i]               # OH-line amplitude (floor-immune)
        f[i]=np.nanmedian((C[i]-s)[lo])                   # measured between-line residual
    blank=good&np.isfinite(L)&(L>0)&(wl<=np.nanpercentile(wl[good],60))
    if blank.sum()<20: continue
    tO=L/np.nanmedian(L[blank])
    tpn=tp/np.nanmedian(tp[blank])
    ratio=tO/tpn
    k=np.nanmedian(ratio[blank])
    pred=cont*(1.0-k/ratio)                               # lines predict the continuum residual
    m=blank&np.isfinite(f)&np.isfinite(pred)
    ff,pp=f[m],pred[m]
    r=np.corrcoef(ff,pp)[0,1]
    a=float(np.sum(pp*ff)/np.sum(pp*pp)) if np.sum(pp*pp)>0 else np.nan
    expl=1-np.nanstd(ff-pp)/np.nanstd(ff)
    print(f"{cam:>4} {int(m.sum()):>6} {np.nanstd(ff):>8.3f} {r:>12.3f} {a:>6.2f} {np.nanstd(ff-pp):>11.3f} {expl:>9.1%}")
    profs[cam]=dict(idx=np.where(m)[0],tpn=tpn[m],tO=tO[m],f=ff,pred=pp)
    allf.append(ff); allp.append(pp)
allf=np.concatenate(allf); allp=np.concatenate(allp)
R=np.corrcoef(allf,allp)[0,1]; A=float(np.sum(allp*allf)/np.sum(allp*allp))
print(f"\nALL cameras pooled: corr(f,pred)={R:+.3f}  slope={A:.2f}  "
      f"std(f)={np.nanstd(allf):.3f} -> std(f-pred)={np.nanstd(allf-allp):.3f} "
      f"({1-np.nanstd(allf-allp)/np.nanstd(allf):.1%} of banding explained)")

show=[c for c in ('1A','2A','4A') if c in profs] or list(profs)[:3]
fig,ax=plt.subplots(2,len(show)+1,figsize=(4.4*(len(show)+1),8))
for j,cam in enumerate(show):
    P=profs[cam]
    ax[0,j].plot(P['idx'],P['tpn'],'.',ms=3,label='tp (twilight)')
    ax[0,j].plot(P['idx'],P['tO'],'.',ms=3,label='t (night-sky OH)')
    ax[0,j].set_title(f'{cam}: throughput, twilight vs sky'); ax[0,j].legend(fontsize=7)
    ax[1,j].plot(P['idx'],P['f'],'.',ms=3,label='measured f')
    ax[1,j].plot(P['idx'],P['pred'],'.',ms=3,label='pred from OH/tp mismatch')
    ax[1,j].axhline(0,color='k',lw=.5); ax[1,j].set_title(f'{cam}: between-line residual'); ax[1,j].legend(fontsize=7)
    ax[1,j].set_xlabel('fibre index (slit position)')
ax[0,-1].axis('off')
ax[1,-1].scatter(allp,allf,s=4,alpha=.3)
lim=np.nanpercentile(np.abs(allp),99)
ax[1,-1].plot([-lim,lim],[-lim,lim],'r--',lw=1)
ax[1,-1].set_xlabel('predicted (OH/twilight mismatch)'); ax[1,-1].set_ylabel('measured f')
ax[1,-1].set_title(f'all cams: corr={R:+.2f}, slope={A:.2f}')
fig.suptitle('Investigation (a): does twilight-throughput mismatch (from OH lines) predict the banding?',fontsize=12)
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig)
print("wrote",OUT)
