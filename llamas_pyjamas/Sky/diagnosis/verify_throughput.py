"""Verify the striping mechanism: (1) is the fibre-flat CORRECTION normalised PER CAMERA (median ~1,
so absolute per-benchside gain is discarded)? (2) do per-benchside FLAM-residual steps track the
per-benchside throughput the flat should have removed? J1613 green."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
RED='/Users/simcoe/data/LLAMAS/may26/ut20260516_17/reduced'; OUTPNG=f'{RED}/throughput_qa.png'

# (1) fibre-flat CORRECTION: per-camera median (green cameras)
ff=f'{RED}/extractions/flat/fibre_flat_corrections.fits'
print("=== fibre_flat_corrections: per-green-camera median CORRECTION (==1 => per-camera normalized) ===")
cam_med={}
with fits.open(ff) as h:
    for hd in h:
        if hd.name.lower().startswith('green_') and not hd.name.lower().endswith('_ref'):
            cor=np.asarray(hd.data['CORRECTION'],float)
            m=np.nanmedian(cor); cam_med[hd.name]=m
            print(f"  {hd.name:12s} median CORRECTION={m:.4f}  (over {cor.shape[0]} fibres)")
vals=np.array(list(cam_med.values()))
print(f"  -> spread of per-camera median CORRECTION: {np.nanstd(vals):.4f} "
      f"(tiny => each camera normalized to ~1 independently => absolute gain discarded)")

# (2) per-benchside blank-fibre throughput (COUNTS) and residual (FLAM), science frame
f=[x for x in sorted(glob.glob(f'{RED}/extractions/*_RSS_green.fits'))
   if 'J1613' in str(fits.getheader(x,0).get('OBJECT',''))][0]
with fits.open(f) as h:
    C=np.asarray(h['COUNTS'].data,float); S=np.asarray(h['SKY'].data,float); F=np.asarray(h['FLAM'].data,float)
    msk=np.asarray(h['MASK'].data); bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
ok=(msk==0)&np.isfinite(F); keep=ok.sum(1)>50
c=np.array([np.nanmedian(C[i][ok[i]]) if keep[i] else np.nan for i in range(C.shape[0])])
s=np.array([np.nanmedian(S[i][ok[i]]) if keep[i] else np.nan for i in range(S.shape[0])])
fl=np.array([np.nanmedian(F[i][ok[i]]) if keep[i] else np.nan for i in range(F.shape[0])])
obj=c-s; blank=keep&(np.abs(obj)<np.nanpercentile(np.abs(obj[np.isfinite(obj)]),50))
ub=sorted(set(bs))
print("\n=== per-benchside (blank fibres): COUNTS (throughput proxy) vs FLAM residual ===")
print(f"  {'bs':>4} {'medCOUNTS':>10} {'medFLAM':>11} {'n':>4}")
cc=[]; ffl=[]
for b in ub:
    m=blank&(bs==b); mc=np.nanmedian(c[m]); mf=np.nanmedian(fl[m]); cc.append(mc); ffl.append(mf)
    print(f"  {b:>4} {mc:>10.1f} {mf:>+11.3e} {int(m.sum()):>4}")
cc=np.array(cc); ffl=np.array(ffl)
print(f"  COUNTS per-benchside spread: {100*np.nanstd(cc)/np.nanmean(cc):.1f}%   "
      f"corr(benchside COUNTS, benchside FLAM residual) = {np.corrcoef(cc,ffl)[0,1]:+.2f}")

fig,ax=plt.subplots(1,3,figsize=(16,4.6))
ax[0].bar(range(len(cam_med)),vals); ax[0].axhline(1,color='r',ls='--')
ax[0].set_xticks(range(len(cam_med))); ax[0].set_xticklabels(list(cam_med),rotation=45,fontsize=7)
ax[0].set_title('fibre-flat median CORRECTION per green camera\n(all ~1 => per-camera normalized)'); ax[0].set_ylim(0.9,1.1)
ax[1].bar(range(len(ub)),cc); ax[1].set_xticks(range(len(ub))); ax[1].set_xticklabels(ub)
ax[1].set_title('per-benchside blank COUNTS (throughput steps)')
ax[2].bar(range(len(ub)),ffl); ax[2].set_xticks(range(len(ub))); ax[2].set_xticklabels(ub); ax[2].axhline(0,color='k',lw=.6)
ax[2].set_title('per-benchside blank FLAM residual (striping)')
fig.suptitle('Striping mechanism: per-camera-normalized flat leaves per-benchside gain steps (J1613 green)',fontsize=12)
fig.savefig(OUTPNG,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig);print("\nwrote",OUTPNG)
