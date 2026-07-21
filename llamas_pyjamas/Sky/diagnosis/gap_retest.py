"""Upgraded gap-vs-floor test (RS's methodology question): stack the inter-trace gap profile over
the 8 J1613 bias-corrected frames (trimmed mean -> sub-ADU sensitivity), and correlate against the
HIGH-SNR floor template (noJ1613) along-slit profile. Green 1A and 2A."""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, pickle, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
from llamas_pyjamas.Sky.skyFloorTemplate import load_template
tpl=load_template(f'{ND}/reduced/floor_template_noJ1613_green.fits','green')

mefs=sorted(glob.glob(f'{ND}/reduced_ped_off/bias_corrected/LLAMAS_2026-05-17_*_SCI22_mef_bias_corrected.fits'))
print(f'{len(mefs)} bias-corrected science frames')

fig,ax=plt.subplots(1,2,figsize=(12,4.6))
for j,(bench,side) in enumerate((('1','A'),('2','A'))):
    cam=f'{bench}{side}'
    with open(f'{ND}/reduced_new/traces/LLAMAS_green_{bench}_{side}_traces.pkl','rb') as fh:
        traces=np.asarray(getattr(pickle.load(fh),'traces'),float)
    nfib,ncol=traces.shape
    xs=np.arange(0,ncol,2)
    per_frame=[]
    for mef in mefs:
        img=None
        with fits.open(mef) as h:
            for hd in h[1:]:
                hh=hd.header
                if (str(hh.get('COLOR','')).lower()=='green' and str(hh.get('BENCH','')).strip()==bench
                        and str(hh.get('SIDE','')).strip().upper()==side):
                    img=np.asarray(hd.data,float); break
        if img is None: continue
        trI=np.clip(np.round(traces[:,xs]).astype(int),0,img.shape[0]-1)
        spec=np.nanmedian(img[trI,xs[None,:]],axis=0)
        locol=spec<np.nanpercentile(spec,30)
        g=np.full(nfib-1,np.nan)
        for p in range(nfib-1):
            y0,y1=traces[p,xs],traces[p+1,xs]
            okp=np.isfinite(y0)&np.isfinite(y1)&((y1-y0)>=6.0)&locol
            if okp.sum()<20: continue
            ym=np.round(0.5*(y0[okp]+y1[okp])).astype(int)
            xv=xs[okp]; inb=(ym>0)&(ym<img.shape[0])
            vals=img[ym[inb],xv[inb]]
            lo,hi=np.nanpercentile(vals,[10,90]); w=vals[(vals>=lo)&(vals<=hi)]
            if w.size>10: g[p]=float(np.mean(w))
        per_frame.append(g)
    G=np.nanmean(np.stack(per_frame),axis=0)              # 8-frame stacked gap profile (ADU/px)
    # template along-slit profile at pair positions
    Tprof=np.nanmean(tpl[cam],axis=1)
    Tp=0.5*(Tprof[:-1]+Tprof[1:])
    m=np.isfinite(G)&np.isfinite(Tp)
    gm=G[m]-np.nanmedian(G[m]); tm=Tp[m]-np.nanmedian(Tp[m])
    gs=median_filter(gm,size=15,mode='nearest'); ts=median_filter(tm,size=15,mode='nearest')
    r=np.corrcoef(gs,ts)[0,1]
    # amplitude accounting: floor amplitude per 5-px extraction vs gap variation
    print(f'{cam}: corr(stacked gap profile, template profile) = {r:+.3f}   '
          f'gap profile std={np.nanstd(gs):.3f} ADU/px  template std={np.nanstd(ts):.2f} cts '
          f'(2D-halo would need gap std ~ {np.nanstd(ts)/5:.2f} ADU/px tracking the template)')
    a=ax[j]; a2=a.twinx()
    a.plot(np.where(m)[0],ts,'.',ms=3,color='C0'); a2.plot(np.where(m)[0],gs,'.',ms=3,color='C1')
    a.set_title(f'{cam}: template floor (C0, cts) vs 8-frame gap (C1, ADU/px), corr={r:+.2f}')
    a.set_xlabel('slit position')
fig.suptitle('Robust re-test: does the between-trace gap light track the floor template?',fontsize=12)
fig.savefig(f'{ND}/gap_retest_qa.png',dpi=110,bbox_inches='tight',facecolor='white')
print('wrote',f'{ND}/gap_retest_qa.png')
