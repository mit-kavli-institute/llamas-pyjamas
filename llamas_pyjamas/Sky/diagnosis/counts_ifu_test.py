"""(1) Is the banding in COUNTS (pre-sky) or only in SKYSUB/FLAM (post-sky)? -> flat/extraction artifact
vs sky residual. (2) Map the banding onto IFU X/Y and benchside. Single J1613 green frame."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
RED='/Users/simcoe/data/LLAMAS/may26/ut20260516_17/reduced'; OUTPNG=f'{RED}/counts_ifu_qa.png'
TH=np.deg2rad(145.0)

f=[x for x in sorted(glob.glob(f'{RED}/extractions/*_RSS_green.fits'))
   if 'J1613' in str(fits.getheader(x,0).get('OBJECT',''))][0]
with fits.open(f) as h:
    C=np.asarray(h['COUNTS'].data,float); S=np.asarray(h['SKY'].data,float)
    SS=np.asarray(h['SKYSUB'].data,float); F=np.asarray(h['FLAM'].data,float); msk=np.asarray(h['MASK'].data)
    bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
    fw=h['FIBERWCS'].data
    ra=np.asarray(fw['RA_FIBERMAP'] if 'RA_FIBERMAP' in fw.names else fw['RA'],float)
    dec=np.asarray(fw['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fw.names else fw['DEC'],float)
    X=np.asarray(fw['X_FIBERMAP'],float) if 'X_FIBERMAP' in fw.names else None
    Y=np.asarray(fw['Y_FIBERMAP'],float) if 'Y_FIBERMAP' in fw.names else None
ok=(msk==0)&np.isfinite(SS); keep=ok.sum(1)>50
def wl(P): return np.array([np.nanmedian(P[i][ok[i]]) if keep[i] else np.nan for i in range(P.shape[0])])
c,s,ss,fl=wl(C),wl(S),wl(SS),wl(F)
obj=c-s                                                   # object excess
blank=keep&np.isfinite(obj)&(np.abs(obj)<np.nanpercentile(np.abs(obj[np.isfinite(obj)]),50))
cosd=np.cos(np.deg2rad(np.nanmedian(dec))); dx=(ra-np.nanmedian(ra))*cosd*3600; dy=(dec-np.nanmedian(dec))*3600
proj=dx*np.cos(TH)+dy*np.sin(TH)

def detrend(v, m):
    """residual after removing the smooth trend along the projection axis (isolates the banding)."""
    out=np.full(v.size,np.nan); idx=np.where(m&np.isfinite(v))[0]
    o=idx[np.argsort(proj[idx])]; sm=median_filter(v[o],size=15,mode='nearest'); out[o]=v[o]-sm
    return out
dc,ds,dss,df=detrend(c,blank),detrend(s,blank),detrend(ss,blank),detrend(fl,blank)
def amp(d,ref):
    a=np.nanstd(d[blank]); return a, a/abs(np.nanmedian(ref[blank]))
print("=== banding amplitude per plane (blank fibres, detrended along 145 deg) ===")
for nm,d,ref in [('COUNTS',dc,c),('SKY',ds,s),('SKYSUB',dss,ss),('FLAM',df,fl)]:
    a,fr=amp(d,ref); print(f"  {nm:7s} band_amp={a:.3e}  fractional={fr:.3%}  (median={np.nanmedian(ref[blank]):.3e})")
def corr(a,b):
    m=blank&np.isfinite(a)&np.isfinite(b); return np.corrcoef(a[m],b[m])[0,1] if m.sum()>20 else np.nan
print("\n=== does the striping track the COUNTS/throughput pattern? (detrended, blank) ===")
print(f"  corr(SKYSUB band, COUNTS band) = {corr(dss,dc):+.3f}")
print(f"  corr(FLAM   band, COUNTS band) = {corr(df,dc):+.3f}")
print(f"  corr(FLAM   band, SKY    band) = {corr(df,ds):+.3f}   (SKY ~ throughput x sky)")
print(f"  corr(SKYSUB band, SKY    band) = {corr(dss,ds):+.3f}")

fig,ax=plt.subplots(2,3,figsize=(16,9))
# (1) banding projections
o=np.argsort(proj[blank]); pj=proj[blank][o]
for nm,d,sc in [('COUNTS/med',dc/np.nanmedian(c[blank]),1),('SKY/med',ds/np.nanmedian(s[blank]),1),('FLAM(abs)',df/np.nanmax(np.abs(df[blank])),1)]:
    ax[0,0].plot(pj,(d[blank][o])*sc,'.-',ms=2,lw=.5,label=nm)
ax[0,0].legend(fontsize=8);ax[0,0].set_title('banding vs projection (normalized)');ax[0,0].set_xlabel('arcsec')
# sky map colored by benchside
ub=sorted(set(bs)); cmap=plt.cm.tab10
for j,b in enumerate(ub):
    m=blank&(bs==b); ax[0,1].scatter(dx[m],dy[m],s=8,color=cmap(j%10),label=b)
ax[0,1].set_aspect('equal');ax[0,1].invert_xaxis();ax[0,1].legend(fontsize=6,ncol=2);ax[0,1].set_title('SKY position, colored by benchside')
# sky map colored by FLAM residual (the bands)
vlo,vhi=np.nanpercentile(df[blank],[5,95]); sc=ax[0,2].scatter(dx[blank],dy[blank],c=np.clip(df[blank],vlo,vhi),s=8,cmap='RdBu_r')
ax[0,2].set_aspect('equal');ax[0,2].invert_xaxis();ax[0,2].set_title('SKY position, FLAM banding residual');fig.colorbar(sc,ax=ax[0,2])
if X is not None:
    for j,b in enumerate(ub):
        m=blank&(bs==b); ax[1,0].scatter(X[m],Y[m],s=8,color=cmap(j%10))
    ax[1,0].set_title('IFU X/Y, colored by benchside');ax[1,0].set_aspect('equal')
    sc=ax[1,1].scatter(X[blank],Y[blank],c=np.clip(df[blank],vlo,vhi),s=8,cmap='RdBu_r')
    ax[1,1].set_title('IFU X/Y, FLAM banding residual');ax[1,1].set_aspect('equal');fig.colorbar(sc,ax=ax[1,1])
else:
    ax[1,0].text(.1,.5,'no X_FIBERMAP',transform=ax[1,0].transAxes)
ax[1,2].scatter(c[blank],fl[blank],s=6,alpha=.4);ax[1,2].set_xlabel('COUNTS wl (~throughput x sky)');ax[1,2].set_ylabel('FLAM wl (striping)')
ax[1,2].set_title(f'FLAM vs COUNTS (blank)  corr(band)={corr(df,dc):+.2f}')
fig.suptitle('COUNTS-vs-SKYSUB banding + IFU/benchside mapping (J1613 green single frame)',fontsize=13)
fig.savefig(OUTPNG,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig);print("\nwrote",OUTPNG)
