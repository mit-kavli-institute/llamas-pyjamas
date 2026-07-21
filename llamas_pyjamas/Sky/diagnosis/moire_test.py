"""Moiré (hex fibres x rotated cardinal grid) vs real per-fibre banding, single J1613 green frame.
(1) pixel-scale sweep: does the stripe PERIOD (arcsec) change with pixscale? -> moiré.
(2) RAW-fibre 1-D projection (no grid): is the banding present in the fibre VALUES themselves? -> real.
"""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from llamas_pyjamas.Combine.superRSS import FibreTable
from llamas_pyjamas.Combine.coadd import coadd_image
RED='/Users/simcoe/data/LLAMAS/may26/ut20260516_17/reduced'; OUTPNG=f'{RED}/moire_qa.png'

f=[x for x in sorted(glob.glob(f'{RED}/extractions/*_RSS_green.fits'))
   if 'J1613' in str(fits.getheader(x,0).get('OBJECT',''))][0]
with fits.open(f) as h:
    flam=np.asarray(h['FLAM'].data,float); ferr=np.asarray(h['FLAM_ERR'].data,float); msk=np.asarray(h['MASK'].data)
    fw=h['FIBERWCS'].data
    ra=np.asarray(fw['RA_FIBERMAP'] if 'RA_FIBERMAP' in fw.names else fw['RA'],float)
    dec=np.asarray(fw['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fw.names else fw['DEC'],float)
ok=(msk==0)&np.isfinite(flam); keep=ok.sum(1)>50
val=np.array([np.nanmedian(flam[i][ok[i]]) if keep[i] else np.nan for i in range(flam.shape[0])])
var=np.array([np.nanmedian(ferr[i][ok[i]])**2 if keep[i] and np.isfinite(ferr[i][ok[i]]).any() else 1.0 for i in range(flam.shape[0])])
g=keep&np.isfinite(val)&np.isfinite(ra)&np.isfinite(dec)
ra,dec,val,var=ra[g],dec[g],val[g],var[g]
cosd=np.cos(np.deg2rad(np.nanmedian(dec))); dx=(ra-np.nanmedian(ra))*cosd*3600; dy=(dec-np.nanmedian(dec))*3600
exp=np.zeros(ra.size,int); sol=np.full(ra.size,0.44); ch=np.array(['green']*ra.size)

def grid(px):
    im=coadd_image(FibreTable(ra=ra,dec=dec,value=val,var=var,solid_angle=sol,exposure=exp,channel=ch,
                   npix=np.full(ra.size,100)),units='flux',weighting='ivar',pixscale=px)
    d=im.data; core=np.isfinite(d); src=d>np.nanpercentile(d[core],90); bp=core&~src
    hp=d-median_filter(np.nan_to_num(d,nan=np.nanmedian(d)),size=9)
    return d,hp,bp

# (2) find stripe-normal angle from RAW fibres: maximize banding power of binned medians
def banding_power(theta):
    p=dx*np.cos(theta)+dy*np.sin(theta)
    bins=np.linspace(np.nanpercentile(p,2),np.nanpercentile(p,98),40)
    med=np.array([np.nanmedian(val[(p>=bins[k])&(p<bins[k+1])]) for k in range(len(bins)-1)])
    med=med-median_filter(np.nan_to_num(med,nan=np.nanmedian(med)),size=7)   # remove smooth trend
    return np.nanstd(med), bins, med
thetas=np.linspace(0,np.pi,37); powers=[banding_power(t)[0] for t in thetas]
th=thetas[int(np.nanargmax(powers))]
bp_amp,bins,med=banding_power(th)
smooth_amp=np.nanstd(val)-0  # scale ref
print(f"RAW-fibre banding (no grid): stripe-normal angle={np.rad2deg(th):.0f} deg, "
      f"banding amp in raw values={bp_amp:.3e}  (fibre-value scatter={np.nanstd(val):.3e})")

# (1) pixel-scale sweep: stripe period in arcsec via 1-D projection FFT of the high-pass
print("pixscale sweep (single frame):")
imgs={}
for px in (0.3,0.5,0.75):
    d,hp,bpm=grid(px); imgs[px]=(d,hp,bpm)
    ny,nx=hp.shape; yy,xx=np.mgrid[0:ny,0:nx]
    proj_coord=(xx*np.cos(th)+yy*np.sin(th))
    mask=bpm&np.isfinite(hp)
    order=np.argsort(proj_coord[mask]); pc=proj_coord[mask][order]; hv=hp[mask][order]
    # bin along projection, FFT for dominant period
    nb=80; pe=np.linspace(pc.min(),pc.max(),nb+1); prof=np.array([np.nanmean(hv[(pc>=pe[k])&(pc<pe[k+1])]) for k in range(nb)])
    prof=np.nan_to_num(prof-np.nanmean(prof)); F=np.abs(np.fft.rfft(prof))
    kpk=1+int(np.argmax(F[1:len(F)//1])); period_bins=nb/kpk; step=(pc.max()-pc.min())/nb*px
    period_arcsec=period_bins*step
    print(f"  px={px}\"  stripeRMS={np.nanstd(hp[bpm]):.3e}  dominant period ~ {period_arcsec:.2f}\"")

fig,ax=plt.subplots(2,3,figsize=(16,9))
for j,px in enumerate((0.3,0.5,0.75)):
    d,hp,bpm=imgs[px]; hl,hh=ZScaleInterval().get_limits(hp[np.isfinite(hp)])
    ax[0,j].imshow(np.where(bpm,hp,np.nan),origin='lower',vmin=hl,vmax=hh,cmap='RdBu_r')
    ax[0,j].set_title(f'single-frame high-pass, px={px}"')
vlo,vhi=np.nanpercentile(val,[5,95])
sc=ax[1,0].scatter(dx,dy,c=np.clip(val,vlo,vhi),s=10,cmap='inferno');ax[1,0].set_aspect('equal');ax[1,0].invert_xaxis()
ax[1,0].set_title('RAW fibres (values on sky)');fig.colorbar(sc,ax=ax[1,0])
ctr=0.5*(bins[1:]+bins[:-1]); ax[1,1].plot(ctr,med,'o-',ms=3)
ax[1,1].set_title(f'RAW-value banding vs projection ({np.rad2deg(th):.0f}deg)\namp={bp_amp:.2e}');ax[1,1].set_xlabel('arcsec along stripe-normal')
ax[1,2].plot(np.rad2deg(thetas),powers);ax[1,2].axvline(np.rad2deg(th),color='r',ls='--')
ax[1,2].set_title('banding power vs projection angle');ax[1,2].set_xlabel('deg')
fig.suptitle('Moiré vs real per-fibre banding — single J1613 green frame',fontsize=13)
fig.savefig(OUTPNG,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig);print("wrote",OUTPNG)
