"""Phase 3a concept check (J1613 green): compare TWO additive-correction scopes against the striping —
(A) per-CAMERA constant pedestal, (B) per-camera SLIT-POSITION profile (captures the edge dips) —
both measured from blank fibres, in white-light. QA PNG + signal-preservation check. Approximate
(FLAM white-light, existing RSS); faithful test = re-reduce with sky_pedestal=True."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from llamas_pyjamas.Combine.superRSS import FibreTable
from llamas_pyjamas.Combine.coadd import coadd_image
RED='/Users/simcoe/data/LLAMAS/may26/ut20260516_17/reduced'; OUTPNG=f'{RED}/pedestal_concept_qa.png'

fs=[f for f in sorted(glob.glob(f'{RED}/extractions/*_RSS_green.fits'))
    if 'J1613' in str(fits.getheader(f,0).get('OBJECT',''))]
RA=[];DEC=[];EXP=[];WB=[];WA=[];WS=[];VAR=[];NPIX=[]
qso_b=qso_a=qso_w=None; qso_br=-np.inf
for ei,f in enumerate(fs):
    with fits.open(f) as h:
        flam=np.asarray(h['FLAM'].data,float); ferr=np.asarray(h['FLAM_ERR'].data,float)
        cnt=np.asarray(h['COUNTS'].data,float); msk=np.asarray(h['MASK'].data); wave=np.asarray(h['WAVE'].data,float)
        bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
        fw=h['FIBERWCS'].data
        ra=np.asarray(fw['RA_FIBERMAP'] if 'RA_FIBERMAP' in fw.names else fw['RA'],float)
        dec=np.asarray(fw['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fw.names else fw['DEC'],float)
    ok=(msk==0)&np.isfinite(flam); nf=flam.shape[0]
    white=np.array([np.nanmedian(flam[i][ok[i]]) if ok[i].sum()>50 else np.nan for i in range(nf)])
    bright=np.nansum(np.where(np.isfinite(cnt),cnt,0),axis=1)
    wcam=white.copy(); wslit=white.copy()
    for b in np.unique(bs):
        idx=np.where(bs==b)[0]
        if idx.size<40: continue
        rank=np.arange(idx.size)                              # within-camera position ~ slit position
        wb=white[idx]; br=bright[idx]
        faint=np.isfinite(wb)&np.isfinite(br)&(br<np.nanpercentile(br[np.isfinite(br)],50))  # blank half
        interior=(rank>=25)&(rank<idx.size-25)
        # (A) per-camera constant: interior blank median
        m=faint&interior
        if m.sum()>=5: wcam[idx]=wb-np.nanmedian(wb[m])
        # (B) slit-position profile: rolling median of blank white vs rank, subtracted at each rank
        if faint.sum()>=15:
            r_b=rank[faint]; w_b=wb[faint]; o=np.argsort(r_b)
            prof=np.interp(rank, r_b[o],
                           median_filter(w_b[o], size=min(11, (w_b.size//2)*2+1), mode='nearest'))
            wslit[idx]=wb-prof
    for i in range(nf):
        if not (ok[i].sum()>50): continue
        RA.append(ra[i]);DEC.append(dec[i]);EXP.append(ei);WB.append(white[i]);WA.append(wcam[i]);WS.append(wslit[i])
        VAR.append(np.nanmedian(ferr[i][ok[i]])**2 if np.isfinite(ferr[i][ok[i]]).any() else 1.0); NPIX.append(int(ok[i].sum()))
        if bright[i]>qso_br: qso_br=bright[i]; qso_b=flam[i].copy(); qso_w=wave[i].copy()
RA=np.array(RA);DEC=np.array(DEC);EXP=np.array(EXP);WB=np.array(WB);WA=np.array(WA);WS=np.array(WS)
VAR=np.array(VAR);NPIX=np.array(NPIX);sol=np.full(RA.size,0.44);ch=np.array(['green']*RA.size)
def co(v): return coadd_image(FibreTable(ra=RA,dec=DEC,value=v,var=VAR,solid_angle=sol,exposure=EXP,channel=ch,npix=NPIX),units='flux',weighting='ivar')
def strp(im):
    core=im.nexp>=np.nanmax(im.nexp)-0.5; d=im.data
    src=d>np.nanpercentile(d[core&np.isfinite(d)],90); bp=core&np.isfinite(d)&~src
    hp=d-median_filter(np.nan_to_num(d,nan=np.nanmedian(d)),size=9); return hp,np.nanstd(hp[bp]),bp
imB=co(WB);imA=co(WA);imS=co(WS)
hpB,rB,bp=strp(imB); _,rA,_=strp(imA); hpS,rS,_=strp(imS)
print(f"striping RMS  baseline={rB:.3e}")
print(f"  (A) per-CAMERA constant : {rA:.3e}  ({100*(1-rA/rB):+.0f}%)")
print(f"  (B) per-SLIT profile    : {rS:.3e}  ({100*(1-rS/rB):+.0f}%)")

fig,ax=plt.subplots(2,3,figsize=(16,9))
lo,hi=ZScaleInterval().get_limits(imB.data[np.isfinite(imB.data)])
ax[0,0].imshow(imB.data,origin='lower',vmin=lo,vmax=hi,cmap='inferno');ax[0,0].set_title('white-light BASELINE')
ax[0,1].imshow(imS.data,origin='lower',vmin=lo,vmax=hi,cmap='inferno');ax[0,1].set_title('AFTER slit-profile (B)')
diff=imB.data-imS.data; dl,dh=ZScaleInterval().get_limits(diff[np.isfinite(diff)])
c=ax[0,2].imshow(diff,origin='lower',cmap='RdBu_r');ax[0,2].set_title('REMOVED by (B)');fig.colorbar(c,ax=ax[0,2])
hl,hh=ZScaleInterval().get_limits(hpB[np.isfinite(hpB)])
ax[1,0].imshow(np.where(bp,hpB,np.nan),origin='lower',vmin=hl,vmax=hh,cmap='RdBu_r');ax[1,0].set_title(f'high-pass BASELINE (RMS {rB:.1e})')
ax[1,1].imshow(np.where(bp,hpS,np.nan),origin='lower',vmin=hl,vmax=hh,cmap='RdBu_r');ax[1,1].set_title(f'high-pass AFTER (B) (RMS {rS:.1e}, {100*(1-rS/rB):+.0f}%)')
ax[1,2].text(0.02,0.7,f"striping RMS reduction:\n (A) per-camera const: {100*(1-rA/rB):+.0f}%\n (B) per-slit profile: {100*(1-rS/rB):+.0f}%",
             transform=ax[1,2].transAxes,fontsize=12,va='top'); ax[1,2].axis('off')
fig.suptitle('Phase 3a concept check (J1613 green): does an additive continuum correction remove the striping?',fontsize=13)
fig.savefig(OUTPNG,dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("wrote",OUTPNG)
