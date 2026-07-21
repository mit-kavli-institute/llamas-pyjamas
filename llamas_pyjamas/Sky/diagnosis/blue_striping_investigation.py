"""Phase 2b: what drives the coherent BLUE stack striping? (a) wavelength localization from the cube,
(b) is a small SKYSUB(counts) residual amplified by the large blue flux-cal into big FLAM striping?"""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
RED='/Users/simcoe/data/LLAMAS/may26/ut20260516_17/reduced'
OUT='/Users/simcoe/.claude/jobs/8fc668fa/tmp/skydiag'
field='J1613'

# ---------- A. wavelength localization of blue striping (cube) ----------
cf=glob.glob(f'{RED}/combined/{field}+*_cube_blue.fits')[0]
with fits.open(cf) as h:
    cube=np.asarray(h[0].data,float); var=np.asarray(h['VAR'].data,float) if 'VAR' in h else None
    nexp=np.asarray(h['NEXP'].data,float) if 'NEXP' in h else None
    wave=np.asarray(h['WAVELENGTH'].data['WAVELENGTH'],float) if 'WAVELENGTH' in h else np.arange(cube.shape[0])
nw=cube.shape[0]; core=(nexp>=np.nanmax(nexp)-0.5) if nexp is not None else np.isfinite(cube[0])
wl=np.nanmedian(cube,axis=0); src=wl>np.nanpercentile(wl[core&np.isfinite(wl)],90); bp=core&np.isfinite(wl)&~src
nb=10; edges=np.linspace(0,nw,nb+1).astype(int); wc=[]; rms=[]; snr=[]
for k in range(nb):
    sl=slice(edges[k],edges[k+1]); img=np.nanmedian(cube[sl],axis=0)
    hp=img-median_filter(np.nan_to_num(img,nan=np.nanmedian(img)),size=9)
    r=np.nanstd(hp[bp]); wc.append(np.nanmedian(wave[sl]))
    nz=np.nanmedian(np.sqrt(np.nanmedian(var[sl],axis=0))[bp])/np.sqrt(edges[k+1]-edges[k]) if var is not None else np.nan
    rms.append(r); snr.append(r/nz if nz else np.nan)
print("=== blue striping vs wavelength (cube) ===")
for w,r,s in zip(wc,rms,snr): print(f"  {w:7.1f} A  stripeRMS={r:.2e}  /noise={s:.1f}")

# ---------- B. SKYSUB(counts) vs FLAM(flux): flux-cal amplification? ----------
fs=[f for f in sorted(glob.glob(f'{RED}/extractions/*_RSS_blue.fits'))
    if field in str(fits.getheader(f,0).get('OBJECT',''))]
ss_wl=[]; fl_wl=[]; sens=[]; objx=[]; bs=None
for f in fs:
    with fits.open(f) as h:
        ss=np.asarray(h['SKYSUB'].data,float); fl=np.asarray(h['FLAM'].data,float)
        cnt=np.asarray(h['COUNTS'].data,float); sky=np.asarray(h['SKY'].data,float); msk=np.asarray(h['MASK'].data)
        if bs is None: bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
    ok=(msk==0)&np.isfinite(ss)&np.isfinite(fl)&(ss!=0.0); nf=ss.shape[0]
    a=np.full(nf,np.nan); b=np.full(nf,np.nan); s=np.full(nf,np.nan); ob=np.full(nf,np.nan)
    for i in range(nf):
        o=ok[i]
        if o.sum()<50: continue
        a[i]=np.nanmean(ss[i][o])                 # white-light SKYSUB residual (counts)
        b[i]=np.nanmean(fl[i][o])                 # white-light FLAM residual (flux) -> feeds the stack
        with np.errstate(divide='ignore',invalid='ignore'):
            s[i]=np.nanmedian(np.abs(fl[i][o]/ss[i][o]))   # |FLAM/SKYSUB| ~ sensfunc*flat amplification
        ob[i]=np.nanmedian(cnt[i][o])-np.nanmedian(sky[i][o])
    ss_wl.append(a); fl_wl.append(b); sens.append(s); objx.append(ob)
ssw=np.nanmedian(ss_wl,axis=0); flw=np.nanmedian(fl_wl,axis=0); sensf=np.nanmedian(sens,axis=0); obj=np.nanmedian(objx,axis=0)
blank=np.isfinite(obj)&(np.abs(obj)<np.nanpercentile(np.abs(obj[np.isfinite(obj)]),50))
print("\n=== per-fibre white-light residual, BLUE blank fibres ===")
print(f"  SKYSUB(counts): median={np.nanmedian(ssw[blank]):+.3g}  scatter(std across fibres)={np.nanstd(ssw[blank]):.3g}")
print(f"  FLAM(flux):     median={np.nanmedian(flw[blank]):+.3g}  scatter(std across fibres)={np.nanstd(flw[blank]):.3g}")
print(f"  |FLAM/SKYSUB| amplification: median={np.nanmedian(sensf[blank]):.3g}  90pct={np.nanpercentile(sensf[blank],90):.3g}")
g=blank&np.isfinite(ssw)&np.isfinite(flw)
print(f"  corr(SKYSUB resid, FLAM resid)={np.corrcoef(ssw[g],flw[g])[0,1]:+.3f}")
print("  per-benchside FLAM white-light residual scatter (drives striping):")
for bb in sorted(set(bs)):
    m=blank&(bs==bb)&np.isfinite(flw)
    if m.sum()>15: print(f"    {bb}: median={np.nanmedian(flw[m]):+.2e}  std={np.nanstd(flw[m]):.2e} (n={int(m.sum())})")

fig,ax=plt.subplots(1,2,figsize=(12,4.6))
ax[0].plot(wc,rms,'o-'); ax[0].set_xlabel('wavelength (A)'); ax[0].set_ylabel('stack stripe RMS')
ax[0].set_title('BLUE striping vs wavelength')
ax[1].scatter(np.abs(ssw[g]),np.abs(flw[g]),c=sensf[g],s=8,cmap='viridis')
ax[1].set_xlabel('|SKYSUB resid| (counts)'); ax[1].set_ylabel('|FLAM resid| (flux)')
ax[1].set_title('flux-cal amplification (color=|FLAM/SKYSUB|)')
fig.savefig(f'{OUT}/blue_investigation.png',dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("\nwrote blue_investigation.png")
