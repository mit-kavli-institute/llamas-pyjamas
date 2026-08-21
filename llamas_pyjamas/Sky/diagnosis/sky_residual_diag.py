"""Phase 2 pass 6 (J1613 green): clean sky-residual on BLANK fibres only (object excess ~ 0), so the
'floor' is a true sky-subtraction residual, not object continuum. Per-camera offset vs sky-scaling.
Plus: measure the stacked striping directly in blank (source-masked) regions."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
RED='/Users/simcoe/data/LLAMAS/may26/ut20260516_17/reduced'
OUT='/Users/simcoe/.claude/jobs/8fc668fa/tmp/skydiag'

# ---- per-fibre floor, sky level, object excess (8 J1613 green) ----
fs=[f for f in sorted(glob.glob(f'{RED}/extractions/*_RSS_green.fits'))
    if 'J1613' in str(fits.getheader(f,0).get('OBJECT',''))]
floors=[]; skyl=[]; obj=[]; bs=None
for f in fs:
    with fits.open(f) as h:
        sky=np.asarray(h['SKY'].data,float); ss=np.asarray(h['SKYSUB'].data,float)
        cnt=np.asarray(h['COUNTS'].data,float); msk=np.asarray(h['MASK'].data)
        if bs is None: bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
    ok=(msk==0)&np.isfinite(ss)&np.isfinite(sky); nf=sky.shape[0]
    fl=np.full(nf,np.nan); sl=np.full(nf,np.nan); ob=np.full(nf,np.nan)
    for i in range(nf):
        o=ok[i]
        if o.sum()<50: continue
        lf=o&(sky[i]<=np.nanpercentile(sky[i][o],30))
        if lf.sum()>20: fl[i]=np.nanmedian(ss[i][lf])
        sl[i]=np.nanmedian(sky[i][o]); ob[i]=np.nanmedian(cnt[i][o])-sl[i]
    floors.append(fl); skyl.append(sl); obj.append(ob)
floor=np.nanmedian(floors,axis=0); sky_lev=np.nanmedian(skyl,axis=0); objx=np.nanmedian(obj,axis=0)

# BLANK fibres: object excess consistent with zero (below the per-field noise)
oscale=np.nanpercentile(np.abs(objx[np.isfinite(objx)]),50)
blank=np.isfinite(objx)&(np.abs(objx)<max(2*oscale, np.nanpercentile(objx[np.isfinite(objx)],40)))
print(f"blank fibres: {int(blank.sum())}/{floor.size} (|object excess| small)")
print(f"[blank] sky residual floor: median={np.nanmedian(floor[blank]):+.3g}  MAD={np.nanmedian(np.abs(floor[blank]-np.nanmedian(floor[blank]))):.3g}")
g=blank&np.isfinite(floor)&np.isfinite(sky_lev)&(sky_lev>0)
print(f"[blank] corr(floor, SKY level) = {np.corrcoef(sky_lev[g],floor[g])[0,1]:+.3f}  (sky-scaling error if strong +)")
print("\n=== BLANK-fibre residual per benchside: constant offset vs sky-fraction ===")
print(f"  {'bs':>4} {'med_floor':>10} {'frac/SKY':>10} {'n':>4}")
for b in sorted(set(bs)):
    m=(bs==b)&g
    if m.sum()<15: continue
    frac=float(np.sum(sky_lev[m]*floor[m])/np.sum(sky_lev[m]**2))
    print(f"  {b:>4} {np.nanmedian(floor[m]):>+10.3g} {frac:>10.4f} {int(m.sum()):>4}")

# ---- stacked striping in blank (source-masked) regions ----
with fits.open(f'{RED}/combined/J1613+0808_cube_green.fits') as h:
    cube=np.asarray(h[0].data,float); nexp=np.asarray(h['NEXP'].data,float) if 'NEXP' in h else None
white=np.nanmedian(cube,axis=0)
core=(nexp>=4) if nexp is not None else np.isfinite(white)
src=white>np.nanpercentile(white[core&np.isfinite(white)],90)   # mask bright sources
blankpix=core&np.isfinite(white)&~src
hp=white-median_filter(np.nan_to_num(white,nan=np.nanmedian(white)),size=9)
noise_floor=np.nanmedian(np.abs(hp[blankpix]))*1.4826
print(f"\n[stack] blank-region high-pass striping RMS = {np.nanstd(hp[blankpix]):.3g}  "
      f"(robust {noise_floor:.3g}); blank pixels {int(blankpix.sum())}")
fig,ax=plt.subplots(1,2,figsize=(11,4.6))
ax[0].plot(floor[blank],'.',ms=3); ax[0].axhline(0,color='k',lw=.6); ax[0].set_ylim(-15,15)
ax[0].set_title('BLANK-fibre sky residual (per fibre)'); ax[0].set_xlabel('fibre index')
from astropy.visualization import ZScaleInterval
hpm=np.where(blankpix,hp,np.nan); lo,hi=ZScaleInterval().get_limits(hpm[np.isfinite(hpm)])
im=ax[1].imshow(hpm,origin='lower',vmin=lo,vmax=hi,cmap='RdBu_r'); ax[1].set_title('stack high-pass, blank regions (striping)')
fig.colorbar(im,ax=ax[1]); fig.savefig(f'{OUT}/blank_residual.png',dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("wrote blank_residual.png")
