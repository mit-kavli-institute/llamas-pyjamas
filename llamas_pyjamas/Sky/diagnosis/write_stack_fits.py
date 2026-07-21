"""Write DS9-inspectable FITS of the 8-frame stacks: pedestal OFF, template ON, and the DIFFERENCE
(off - on = what the pedestal removed). Uses the same loader/co-add as the metric; CoaddImage.write
gives white-light + VAR + COVERAGE + NEXP extensions with WCS."""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from llamas_pyjamas.Combine.superRSS import FibreTable
from llamas_pyjamas.Combine.coadd import coadd_image
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'

def load(v):
    RA=[];DEC=[];EXP=[];WL=[];VAR=[];NP=[]
    for ei,f in enumerate(sorted(glob.glob(f'{ND}/reduced_ped_{v}/extractions/*_RSS_green.fits'))):
        with fits.open(f) as h:
            dat=np.asarray(h['SKYSUB'].data,float); msk=np.asarray(h['MASK'].data)
            err=np.asarray(h['ERROR'].data,float)
            fm=h['FIBERWCS'].data if 'FIBERWCS' in [x.name for x in h] else h['FIBERMAP'].data
            ra=np.asarray(fm['RA_FIBERMAP'] if 'RA_FIBERMAP' in fm.names else fm['RA'],float)
            dec=np.asarray(fm['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fm.names else fm['DEC'],float)
        ok=(msk==0)&np.isfinite(dat)
        for i in range(dat.shape[0]):
            if ok[i].sum()<50: continue
            RA.append(ra[i]);DEC.append(dec[i]);EXP.append(ei);WL.append(np.nanmedian(dat[i][ok[i]]))
            VAR.append(np.nanmedian(err[i][ok[i]])**2 if np.isfinite(err[i][ok[i]]).any() else 1.0);NP.append(int(ok[i].sum()))
    return list(map(np.array,(RA,DEC,EXP,WL,VAR,NP)))

ims={}
for v in ('off','on'):
    A=load(v)
    # ON gridded onto the OFF grid center for a pixel-matched difference
    ctr=None
    if 'off' in ims:
        w=ims['off'].wcs; ny0,nx0=ims['off'].data.shape
        sk=w.pixel_to_world(nx0/2-0.5,ny0/2-0.5); ctr=(float(sk.ra.deg),float(sk.dec.deg))
    im=coadd_image(FibreTable(ra=A[0],dec=A[1],value=A[3],var=A[4],
        solid_angle=np.full(A[0].size,0.44),exposure=A[2],
        channel=np.array(['green']*A[0].size),npix=A[5]),units='flux',weighting='ivar',
        center=ctr)
    ims[v]=im
    out=f'{ND}/pedestal8_stack_{v}.fits'
    im.write(out); print('wrote',out)
# difference on the common grid (shapes should match; crop to common if not)
a,b=ims['off'].data,ims['on'].data
ny,nx=min(a.shape[0],b.shape[0]),min(a.shape[1],b.shape[1])
diff=a[:ny,:nx]-b[:ny,:nx]
hdr=ims['off'].wcs.to_header(); hdr['COMMENT']='pedestal OFF minus template ON (what was removed)'
fits.PrimaryHDU(diff.astype(np.float32),header=hdr).writeto(f'{ND}/pedestal8_stack_diff.fits',overwrite=True)
print('wrote',f'{ND}/pedestal8_stack_diff.fits')
