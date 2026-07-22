"""Does the blank-region cube RMS fall as sqrt(N) when averaging spectral planes?
Decompose: empirical spatial RMS(window) vs propagated noise (from VAR) vs the spectrally-coherent
floor. If empirical RMS is flat while propagated falls ~sqrt(N), the noise is systematic-limited
(a spectrally-coherent spatial pattern = the sky-subtraction residual), not photon-limited."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
ND='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17'
fs=(glob.glob(f'{ND}/reduced/**/*J0958*cube*green*.fits',recursive=True)
    or glob.glob(f'{ND}/reduced/**/*J2151*cube*green*.fits',recursive=True)
    or glob.glob(f'{ND}/reduced/**/*cube*green*.fits',recursive=True))
f=fs[0]; print("cube:",f.split('/')[-1])
with fits.open(f) as h:
    data=np.asarray(h[0].data,float)                       # (nplane, ny, nx)
    var=np.asarray(h['VAR'].data,float) if 'VAR' in h else None
    cov=np.asarray(h['COVERAGE'].data,float) if 'COVERAGE' in h else None
    wave=np.asarray(h['WAVELENGTH'].data,float) if 'WAVELENGTH' in h else np.arange(data.shape[0])
nplane,ny,nx=data.shape
wl=np.nanmean(data,axis=0)                                  # white light per spaxel
covered=np.isfinite(wl)
if cov is not None: covered&=(cov>=np.nanmax(cov)*0.6)      # well-covered spaxels only
# blank = covered, low flux (avoid sources), and finite
thr=np.nanpercentile(wl[covered],60)
blank=covered&(wl<thr)
print(f"cube {nplane} planes x {ny}x{nx}; blank spaxels used = {int(blank.sum())}")

mid=nplane//2
def collapse_rms(N):
    a=max(0,mid-N//2); b=min(nplane,a+N)
    im=np.nanmean(data[a:b],axis=0)                        # == CubeViewer collapse
    emp=np.nanstd(im[blank])                               # empirical spatial RMS over blank spaxels
    prop=np.nan
    if var is not None:
        # propagated 1-sigma of the N-plane mean, per spaxel: sqrt(sum var)/N ; median over blank
        vslab=var[a:b]; n_ok=np.sum(np.isfinite(vslab)&(vslab>0),axis=0)
        pmean=np.sqrt(np.nansum(np.where(np.isfinite(vslab)&(vslab>0),vslab,0),axis=0))/np.maximum(n_ok,1)
        prop=np.nanmedian(pmean[blank])
    return emp,prop,b-a
print(f"\n{'Nplanes':>8} {'empRMS(spatial)':>16} {'propNoise(VAR)':>15} {'emp/prop':>9}")
for N in (1,3,10,30,100,300,nplane):
    emp,prop,n=collapse_rms(N)
    print(f"{n:>8} {emp:>16.3e} {prop:>15.3e} {emp/prop if prop else float('nan'):>9.1f}")

# spectrally-coherent floor = spatial RMS of the all-plane mean pattern (the systematic)
coh=np.nanstd(wl[blank])
# single-plane spatial RMS (median over planes)
s1=np.nanmedian([np.nanstd(data[k][blank]) for k in range(mid-50,mid+50)])
print(f"\ncoherent floor (spatial RMS of full-band mean pattern) = {coh:.3e}")
print(f"typical single-plane spatial RMS                        = {s1:.3e}")
print(f"-> coherent/single-plane = {coh/s1:.2f}  (near 1 => systematic-dominated even per plane)")

# spectral autocorrelation of blank spaxels (are planes independent?)
bs=data[:, blank]                                          # (nplane, nblank)
bs=bs-np.nanmean(bs,axis=0,keepdims=True)
lag1=np.nanmean([np.corrcoef(bs[:-1,j],bs[1:,j])[0,1] for j in range(0,bs.shape[1],max(1,bs.shape[1]//200)) if np.isfinite(bs[:,j]).all()])
print(f"mean spectral lag-1 autocorr of blank spaxels = {lag1:+.2f}  (high => planes NOT independent)")
