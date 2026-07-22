"""What IS the slit-edge per-fibre residual? Per fibre in the FINAL SKYSUB (post flat/framework/
pedestal), split into CONTINUUM (between-line) vs OH-LINE residual, vs distance from the benchside
slit edge. Continuum-biased edges -> additive/scattered (per-slit-position correction); line-biased
edges -> LSF-across-slit mis-subtraction (derivative refinement)."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
ND='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01/extractions'
OUT='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/sky_diagnostics/edge_fibre_diag_qa.png'

fig,ax=plt.subplots(1,2,figsize=(13,5))
for fld,col in (('J2151','C0'),('J0958','C1')):
    f=[x for x in sorted(glob.glob(f'{ND}/*_RSS_green.fits')) if fld in str(fits.getheader(x,0).get('OBJECT',''))][0]
    with fits.open(f) as h:
        C=np.asarray(h['COUNTS'].data,float); S=np.asarray(h['SKY'].data,float)
        SS=np.asarray(h['SKYSUB'].data,float); msk=np.asarray(h['MASK'].data)
        bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
    ok=(msk==0)&np.isfinite(SS)&np.isfinite(S)
    nf=C.shape[0]; cont=np.full(nf,np.nan); line=np.full(nf,np.nan); obj=np.full(nf,np.nan); dist=np.full(nf,np.nan)
    for b in sorted(set(bs)):
        idx=np.where(bs==b)[0]
        for r,i in enumerate(idx):
            dist[i]=min(r, idx.size-1-r)                  # fibres from nearest slit edge
    for i in range(nf):
        m=ok[i]
        if m.sum()<200: continue
        hi=m&(S[i]>np.nanpercentile(S[i][m],85)); lo=m&(S[i]<np.nanpercentile(S[i][m],30))
        if lo.sum()>50: cont[i]=np.nanmedian(SS[i][lo])
        if hi.sum()>20: line[i]=np.nanmedian(SS[i][hi])
        obj[i]=np.nanmedian(C[i][m])-np.nanmedian(S[i][m])
    blank=np.isfinite(obj)&(np.abs(obj)<np.nanpercentile(np.abs(obj[np.isfinite(obj)]),50))
    # aggregate blank fibres by distance-from-edge
    ds=np.arange(0,20)
    cont_d=[np.nanmedian(cont[blank&(dist==d)]) for d in ds]
    line_d=[np.nanmedian(line[blank&(dist==d)]) for d in ds]
    ax[0].plot(ds,cont_d,'o-',ms=3,color=col,label=f'{fld}')
    ax[1].plot(ds,line_d,'o-',ms=3,color=col,label=f'{fld}')
    edge=blank&(dist<=2); ctr=blank&(dist>=10)
    print(f"{fld} green (final SKYSUB, blank fibres):")
    print(f"  CONTINUUM resid: edge(d<=2)={np.nanmedian(cont[edge]):+.2e}  center(d>=10)={np.nanmedian(cont[ctr]):+.2e}"
          f"  edge-center={np.nanmedian(cont[edge])-np.nanmedian(cont[ctr]):+.2e}")
    print(f"  OH-LINE  resid: edge(d<=2)={np.nanmedian(line[edge]):+.2e}  center(d>=10)={np.nanmedian(line[ctr]):+.2e}"
          f"  edge-center={np.nanmedian(line[edge])-np.nanmedian(line[ctr]):+.2e}")
ax[0].axhline(0,color='k',lw=.4); ax[0].set_title('CONTINUUM residual vs distance from slit edge'); ax[0].set_xlabel('fibres from edge'); ax[0].legend()
ax[1].axhline(0,color='k',lw=.4); ax[1].set_title('OH-LINE residual vs distance from slit edge'); ax[1].set_xlabel('fibres from edge'); ax[1].legend()
fig.suptitle('Slit-edge per-fibre residual: continuum (additive?) vs OH-line (LSF?) — green',fontsize=12)
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("\nwrote",OUT)
