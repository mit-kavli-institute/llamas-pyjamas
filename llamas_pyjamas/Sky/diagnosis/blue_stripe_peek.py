"""Peek at the bright stripe through the two J0958 blue point sources: is it an IFU fibre-ROW
(spatial systematic / bad flat / scattered) or the slit-NEIGHBOURS of the source fibres (detector
cross-talk from the bright QSOs)? Stack white-light + per-fibre RSS on IFU-XY and on sky."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
ND='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17'
RR=f'{ND}/reduced_rev01/combined'; SD=f'{ND}/sky_diagnostics'

# --- stack white-light (>3600, coverage-masked) ---
with fits.open(f'{RR}/J0958+1347_cube_blue.fits') as h:
    data=np.asarray(h[0].data,float); nexp=np.asarray(h['NEXP'].data,float); wave=np.asarray(h['WAVELENGTH'].data,float)
wl=np.nanmean(data[wave>=3600],axis=0)
wl=np.where(nexp<0.7*np.nanmax(nexp),np.nan,wl)
# two brightest peaks (sources)
flat=np.where(np.isfinite(wl),wl,-np.inf); pk=[]
tmp=flat.copy()
for _ in range(2):
    iy,ix=np.unravel_index(np.argmax(tmp),tmp.shape); pk.append((ix,iy))
    tmp[max(0,iy-4):iy+5,max(0,ix-4):ix+5]=-np.inf
print("cube source peaks (x,y):",pk)

# --- one J0958 blue RSS: per-fibre white-light + IFU coords + sky ---
f=[x for x in sorted(glob.glob(f'{RR}/../extractions/*_RSS_blue.fits')) if False]  # placeholder
f=[x for x in sorted(glob.glob(f'{ND}/reduced_rev01/extractions/*_RSS_blue.fits'))
   if 'J0958' in str(fits.getheader(x,0).get('OBJECT',''))][0]
with fits.open(f) as h:
    F=np.asarray(h['FLAM'].data,float) if 'FLAM' in [x.name for x in h] else np.asarray(h['SKYSUB'].data,float)
    msk=np.asarray(h['MASK'].data); wv=np.asarray(h['WAVE'].data,float)
    fw=h['FIBERWCS'].data if 'FIBERWCS' in [x.name for x in h] else h['FIBERMAP'].data
    bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
    X=np.asarray(fw['X_FIBERMAP'],float) if 'X_FIBERMAP' in fw.names else None
    Y=np.asarray(fw['Y_FIBERMAP'],float) if 'Y_FIBERMAP' in fw.names else None
    ra=np.asarray(fw['RA_FIBERMAP'] if 'RA_FIBERMAP' in fw.names else fw['RA'],float)
    dec=np.asarray(fw['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fw.names else fw['DEC'],float)
ok=(msk==0)&np.isfinite(F)&(wv>=3600)
wlf=np.array([np.nanmedian(F[i][ok[i]]) if ok[i].sum()>50 else np.nan for i in range(F.shape[0])])
srcfib=np.argsort(np.nan_to_num(wlf,nan=-np.inf))[::-1][:6]
print("brightest fibres (index, benchside, IFU X,Y):")
for i in srcfib:
    print(f"  fib {i:4d}  {bs[i]:>3}  X={X[i]:.1f} Y={Y[i]:.1f}  wl={wlf[i]:.2e}")

fig,ax=plt.subplots(1,3,figsize=(16,5))
lo,hi=ZScaleInterval().get_limits(wl[np.isfinite(wl)])
ax[0].imshow(wl,origin='lower',vmin=lo,vmax=hi,cmap='inferno')
for (x,y) in pk: ax[0].plot(x,y,'c+',ms=14,mew=2)
ax[0].plot([pk[0][0],pk[1][0]],[pk[0][1],pk[1][1]],'c--',lw=.8)
ax[0].set_title('J0958 blue stack WL (+ = sources, line through them)')
g=np.isfinite(wlf)&np.isfinite(X)&np.isfinite(Y)
vlo,vhi=np.nanpercentile(wlf[g],[5,97])
sc=ax[1].scatter(X[g],Y[g],c=np.clip(wlf[g],vlo,vhi),s=12,cmap='inferno')
ax[1].scatter(X[srcfib],Y[srcfib],facecolors='none',edgecolors='cyan',s=60,lw=1.5)
ax[1].set_aspect('equal'); ax[1].set_title('RSS fibres on IFU X/Y (o=brightest fibres)'); fig.colorbar(sc,ax=ax[1])
cosd=np.cos(np.deg2rad(np.nanmedian(dec))); dx=(ra-np.nanmedian(ra))*cosd*3600; dy=(dec-np.nanmedian(dec))*3600
sc=ax[2].scatter(dx[g],dy[g],c=np.clip(wlf[g],vlo,vhi),s=12,cmap='inferno'); ax[2].set_aspect('equal'); ax[2].invert_xaxis()
ax[2].scatter(dx[srcfib],dy[srcfib],facecolors='none',edgecolors='cyan',s=60,lw=1.5)
ax[2].set_title('RSS fibres on SKY (single exposure)')
fig.suptitle('J0958 blue: is the stripe an IFU fibre-row or the source fibres cross-talk?',fontsize=12)
fig.savefig(f'{SD}/blue_stripe_peek_qa.png',dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("wrote",f'{SD}/blue_stripe_peek_qa.png')
