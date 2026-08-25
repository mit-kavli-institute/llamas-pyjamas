"""Why are there systematics at NaN-coverage boundaries? Is it (a) honest higher NOISE (fewer fibres
-> empirical RMS ~ sqrt(VAR)) or (b) a coherent BIAS/ring (empRMS >> sqrt(VAR), or a mean offset that
VAR doesn't predict)? And does a NEXP threshold clean it? J0958 + J2151 blue, >3600A."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_erosion, binary_dilation, median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
RR='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01/combined'
OUT='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/sky_diagnostics/boundary_diag_qa.png'

fig,ax=plt.subplots(2,3,figsize=(16,9))
for i,(fld,fn) in enumerate((('J0958','J0958+1347_cube_blue.fits'),('J2151','J2151+0235_cube_blue.fits'))):
    with fits.open(f'{RR}/{fn}') as h:
        data=np.asarray(h[0].data,float); var=np.asarray(h['VAR'].data,float)
        nexp=np.asarray(h['NEXP'].data,float); wave=np.asarray(h['WAVELENGTH'].data,float)
    sl=wave>=3600
    wl=np.nanmean(data[sl],axis=0)
    n_ok=np.sum(np.isfinite(var[sl])&(var[sl]>0),axis=0)
    prop=np.sqrt(np.nansum(np.where(np.isfinite(var[sl])&(var[sl]>0),var[sl],0),axis=0))/np.maximum(n_ok,1)
    covered=np.isfinite(wl); mx=np.nanmax(nexp)
    interior=binary_erosion(nexp>=0.9*mx,iterations=2)&covered
    # boundary = covered spaxels within 2 px of a coverage drop (edge of NaN / low-NEXP)
    full=nexp>=0.9*mx
    boundary=covered & binary_dilation(~full,iterations=2) & ~binary_erosion(~full,iterations=0)
    boundary&=covered&(~interior)
    # low-flux (blank) within each, to isolate noise/systematic from sources
    def blank(m):
        mm=m&np.isfinite(wl)
        return mm&(wl<np.nanpercentile(wl[mm],70)) if mm.sum()>30 else mm
    bi,bb=blank(interior),blank(boundary)
    def stats(m):
        return (np.nanmean(wl[m]), np.nanstd(wl[m]), np.nanmedian(prop[m]))
    mi,si,pi=stats(bi); mb,sb,pb=stats(bb)
    print(f"{fld} blue (>3600A):")
    print(f"  INTERIOR : mean={mi:+.2e}  empRMS={si:.2e}  sqrt(VAR)={pi:.2e}  empRMS/prop={si/pi:.1f}  n={int(bi.sum())}")
    print(f"  BOUNDARY : mean={mb:+.2e}  empRMS={sb:.2e}  sqrt(VAR)={pb:.2e}  empRMS/prop={sb/pb:.1f}  n={int(bb.sum())}")
    print(f"  boundary mean OFFSET vs interior = {mb-mi:+.2e}  ({(mb-mi)/si:+.1f} interior-sigma)")
    # does VAR already know the boundary is noisier? (ratio of prop boundary/interior)
    print(f"  sqrt(VAR) boundary/interior = {pb/pi:.1f}  (>1 => VAR already accounts for fewer fibres)")
    # NEXP threshold sweep: empRMS/prop as we tighten coverage
    print(f"  {'NEXP>=':>7} {'nblank':>6} {'empRMS':>9} {'empRMS/prop':>11}")
    for frac in (0.0,0.5,0.7,0.9,1.0):
        m=blank(covered&(nexp>=frac*mx))
        if m.sum()<30: continue
        _,s,p=stats(m); print(f"  {frac*mx:>7.0f} {int(m.sum()):>6} {s:>9.2e} {s/p:>11.1f}")
    # panels
    lo,hi=ZScaleInterval().get_limits(wl[np.isfinite(wl)])
    ax[i,0].imshow(wl,origin='lower',vmin=lo,vmax=hi,cmap='inferno'); ax[i,0].set_title(f'{fld} blue WL (>3600)')
    ov=np.zeros(wl.shape); ov[covered]=1; ov[interior]=2; ov[boundary]=3
    ax[i,1].imshow(ov,origin='lower',cmap='cividis'); ax[i,1].set_title(f'{fld}: interior=2 boundary=3')
    ratio=np.where(covered,wl/np.maximum(prop,1e-30),np.nan)
    rl=np.nanpercentile(np.abs(ratio[covered]),98)
    im=ax[i,2].imshow(ratio,origin='lower',vmin=-rl,vmax=rl,cmap='RdBu_r'); ax[i,2].set_title(f'{fld}: WL / sqrt(VAR) (SNR)')
    fig.colorbar(im,ax=ax[i,2])
fig.suptitle('NaN-boundary systematics: honest noise (empRMS~sqrt(VAR)) vs bias (offset / excess)',fontsize=12)
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("\nwrote",OUT)
