"""Does the striping appear when gridding a SINGLE frame (no co-add)? And is it in the fibres or made
by the hex->grid resampling? Plus a multiplicative per-camera test. J1613 green, FLAM white-light."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from llamas_pyjamas.Combine.superRSS import FibreTable
from llamas_pyjamas.Combine.coadd import coadd_image
RED='/Users/simcoe/data/LLAMAS/may26/ut20260516_17/reduced'; OUTPNG=f'{RED}/singleframe_qa.png'

fs=[f for f in sorted(glob.glob(f'{RED}/extractions/*_RSS_green.fits'))
    if 'J1613' in str(fits.getheader(f,0).get('OBJECT',''))]
RA=[];DEC=[];EXP=[];WB=[];SK=[];BS=[];VAR=[];NPIX=[]
for ei,f in enumerate(fs):
    with fits.open(f) as h:
        flam=np.asarray(h['FLAM'].data,float); ferr=np.asarray(h['FLAM_ERR'].data,float)
        sky=np.asarray(h['SKY'].data,float); msk=np.asarray(h['MASK'].data)
        bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
        fw=h['FIBERWCS'].data
        ra=np.asarray(fw['RA_FIBERMAP'] if 'RA_FIBERMAP' in fw.names else fw['RA'],float)
        dec=np.asarray(fw['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fw.names else fw['DEC'],float)
    ok=(msk==0)&np.isfinite(flam)
    for i in range(flam.shape[0]):
        if ok[i].sum()<50: continue
        RA.append(ra[i]);DEC.append(dec[i]);EXP.append(ei);WB.append(np.nanmedian(flam[i][ok[i]]))
        SK.append(np.nanmedian(sky[i][ok[i]])); BS.append(f"{ei}_{bs[i]}")
        VAR.append(np.nanmedian(ferr[i][ok[i]])**2 if np.isfinite(ferr[i][ok[i]]).any() else 1.0); NPIX.append(int(ok[i].sum()))
RA=np.array(RA);DEC=np.array(DEC);EXP=np.array(EXP);WB=np.array(WB);SK=np.array(SK)
BS=np.array(BS);VAR=np.array(VAR);NPIX=np.array(NPIX);sol=np.full(RA.size,0.44);ch=np.array(['green']*RA.size)
def co(sel,val=None):
    v=WB if val is None else val
    return coadd_image(FibreTable(ra=RA[sel],dec=DEC[sel],value=v[sel],var=VAR[sel],solid_angle=sol[sel],
                       exposure=EXP[sel],channel=ch[sel],npix=NPIX[sel]),units='flux',weighting='ivar')
def strp(im):
    core=im.nexp>=np.nanmax(im.nexp)-0.5; d=im.data
    src=d>np.nanpercentile(d[core&np.isfinite(d)],90); bp=core&np.isfinite(d)&~src
    hp=d-median_filter(np.nan_to_num(d,nan=np.nanmedian(d)),size=9); return hp,(np.nanstd(hp[bp]) if bp.sum()>30 else np.nan),bp

# SINGLE frame vs STACK
im0=co(EXP==0); hp0,r0,bp0=strp(im0)
im1=co(EXP==1); hp1,r1,bp1=strp(im1)
imS=co(np.ones(RA.size,bool)); hpS,rS,bpS=strp(imS)
print(f"striping RMS  single frame0={r0:.3e}  single frame1={r1:.3e}  8-frame stack={rS:.3e}")

# MULTIPLICATIVE per-camera test: correct residual by per-camera throughput (SKY-level) scale
gmed=np.nanmedian(SK[SK>0])
fac=np.ones(RA.size)
for cam in np.unique(BS):
    m=BS==cam; cmed=np.nanmedian(SK[m])
    if cmed>0: fac[m]=gmed/cmed
WBm=WB*fac
imM=co(np.ones(RA.size,bool),WBm); hpM,rM,_=strp(imM)
print(f"multiplicative per-camera renorm: stack RMS={rM:.3e}  ({100*(1-rM/rS):+.0f}% vs baseline stack)")
print(f"(per-camera SKY-scale spread: {np.nanstd([np.nanmedian(SK[BS==c])/gmed for c in np.unique(BS)]):.3f})")

fig,ax=plt.subplots(2,3,figsize=(16,9))
lo,hi=ZScaleInterval().get_limits(im0.data[np.isfinite(im0.data)])
ax[0,0].imshow(im0.data,origin='lower',vmin=lo,vmax=hi,cmap='inferno');ax[0,0].set_title('SINGLE frame 0 — white-light')
hl,hh=ZScaleInterval().get_limits(hp0[np.isfinite(hp0)])
ax[0,1].imshow(np.where(bp0,hp0,np.nan),origin='lower',vmin=hl,vmax=hh,cmap='RdBu_r');ax[0,1].set_title(f'SINGLE frame 0 high-pass (RMS {r0:.1e})')
# raw fibres (pre-grid) for frame 0
m0=EXP==0; cosd=np.cos(np.deg2rad(np.nanmedian(DEC))); dx=(RA[m0]-np.nanmedian(RA[m0]))*cosd*3600; dy=(DEC[m0]-np.nanmedian(DEC[m0]))*3600
vlo,vhi=np.nanpercentile(WB[m0],[5,95])
sc=ax[0,2].scatter(dx,dy,c=np.clip(WB[m0],vlo,vhi),s=9,cmap='inferno');ax[0,2].set_aspect('equal');ax[0,2].invert_xaxis()
ax[0,2].set_title('SINGLE frame 0 — RAW fibres (pre-grid)');fig.colorbar(sc,ax=ax[0,2])
ax[1,0].imshow(np.where(bpS,hpS,np.nan),origin='lower',vmin=hl,vmax=hh,cmap='RdBu_r');ax[1,0].set_title(f'8-frame STACK high-pass (RMS {rS:.1e})')
ax[1,1].imshow(np.where(bpS,hpM,np.nan) if hpM.shape==bpS.shape else hpM,origin='lower',vmin=hl,vmax=hh,cmap='RdBu_r')
ax[1,1].set_title(f'STACK, per-camera MULT renorm ({100*(1-rM/rS):+.0f}%)')
ax[1,2].text(0.02,0.75,f"striping RMS:\n single frame 0: {r0:.2e}\n single frame 1: {r1:.2e}\n 8-frame stack: {rS:.2e}\n mult renorm:   {rM:.2e}",
             transform=ax[1,2].transAxes,fontsize=12,va='top',family='monospace');ax[1,2].axis('off')
fig.suptitle('Does the striping appear in a SINGLE frame? (J1613 green)',fontsize=13)
fig.savefig(OUTPNG,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig);print("wrote",OUTPNG)
