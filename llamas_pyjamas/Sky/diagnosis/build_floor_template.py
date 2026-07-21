"""Rebuild the green floor template from ONLY the long quasar exposures (J1613/J2151/J0958, 2200s):
the 4s standards frames are read-noise-dominated (sky~2 counts) and dilute the template. Compare
against the all-frames template. Positive-outlier rejection across 17 frames / 3 fields."""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, os, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
OUT=f'{ND}/floor_template_qa.png'
TPL=f'{ND}/reduced/floor_template_green.fits'
from llamas_pyjamas.Sky.skyFloorTemplate import build_floor_template, save_template, load_template

old = load_template(TPL,'green') if os.path.exists(TPL) else {}

files=[]
for f in sorted(glob.glob(f'{ND}/reduced/extractions/*_RSS_green.fits')):
    if any(t in str(fits.getheader(f,0).get('OBJECT','')) for t in ('J1613','J2151','J0958')):
        files.append(f)
print(f"building from {len(files)} QUASAR frames (2200s)...")
templates,diag=build_floor_template(files,'green')
save_template(TPL,templates,diag,'green')

sky=np.array([m['skylev'] for m in diag['frames']])
print(f"sky level range {np.nanmin(sky):.0f}-{np.nanmax(sky):.0f}")
print(f"{'cam':>4} {'T rms':>7} {'rejfrac':>8} {'amp spread':>10} {'corr(new,old)':>13}")
for cam in sorted(templates):
    T=templates[cam]; a=diag['amplitudes'][cam]
    co=np.nan
    if cam in old and old[cam].shape==T.shape:
        m=np.isfinite(T)&np.isfinite(old[cam])
        co=np.corrcoef(T[m],old[cam][m])[0,1]
    print(f"{cam:>4} {np.nanstd(T):>7.2f} {diag['reject_frac'][cam]:>8.1%} "
          f"{np.nanstd(a)/abs(np.nanmean(a)):>10.1%} {co:>+13.2f}")

fig,ax=plt.subplots(1,3,figsize=(16,4.6))
for cam in sorted(templates):
    ax[0].plot(np.nanmean(templates[cam],axis=1),lw=1,label=cam)
ax[0].axhline(0,color='k',lw=.4); ax[0].legend(fontsize=7,ncol=2)
ax[0].set_title(f'template (quasar frames only, n={len(files)}): along-slit profile'); ax[0].set_xlabel('fibre (slit pos)')
for cam in ('1A','2A','4B'):
    if cam in templates: ax[1].plot(np.nanmean(templates[cam],axis=0),lw=1.2,label=cam)
ax[1].axhline(0,color='k',lw=.4); ax[1].legend(fontsize=8)
ax[1].set_title('wavelength shape (slit-avg)'); ax[1].set_xlabel('λ bin (blue→red)')
T=templates['1A']
im=ax[2].imshow(T,origin='lower',aspect='auto',cmap='RdBu_r',
                vmin=-np.nanpercentile(np.abs(T),95),vmax=np.nanpercentile(np.abs(T),95))
ax[2].set_title('1A template (fibre × λ)'); ax[2].set_xlabel('λ bin'); ax[2].set_ylabel('fibre')
fig.colorbar(im,ax=ax[2])
fig.suptitle('Static floor template — long quasar exposures only, positive-outlier rejection',fontsize=12)
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig)
print("wrote",OUT,"and",TPL)
