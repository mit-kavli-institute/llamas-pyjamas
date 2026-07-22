"""Concept test: does removing the per-slit-position continuum ramp (measured from blank fibres
INCLUDING edges) drop the green coherent white-light floor? Per exposure/camera, subtract the
along-slit continuum-residual profile from each fibre's white-light, re-stack, compare coherent floor."""
import sys; sys.path.insert(0,'/Users/simcoe/GIT/llamas-pyjamas')
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import binary_erosion, median_filter
from llamas_pyjamas.Combine.superRSS import FibreTable
from llamas_pyjamas.Combine.coadd import coadd_image
ND='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01/extractions'

def coh_floor(RA,DEC,EXP,WL,VAR,NP):
    ft=FibreTable(ra=RA,dec=DEC,value=WL,var=VAR,solid_angle=np.full(RA.size,0.44),
                  exposure=EXP,channel=np.array(['green']*RA.size),npix=NP)
    im=coadd_image(ft,units='flux',weighting='ivar')
    d=im.data; core=im.nexp>=np.nanmax(im.nexp)-0.5
    src=d>np.nanpercentile(d[core&np.isfinite(d)],90); bp=core&np.isfinite(d)&~src
    return float(np.nanstd(d[bp]))

for fld in ('J2151','J0958'):
    files=[x for x in sorted(glob.glob(f'{ND}/*_RSS_green.fits')) if fld in str(fits.getheader(x,0).get('OBJECT',''))]
    RA=[];DEC=[];EXP=[];WB=[];WA=[];VAR=[];NP=[]
    for ei,f in enumerate(files):
        with fits.open(f) as h:
            SS=np.asarray(h['SKYSUB'].data,float); S=np.asarray(h['SKY'].data,float)
            C=np.asarray(h['COUNTS'].data,float); msk=np.asarray(h['MASK'].data)
            er=np.asarray(h['ERROR'].data,float)
            bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
            fw=h['FIBERWCS'].data if 'FIBERWCS' in [x.name for x in h] else h['FIBERMAP'].data
            ra=np.asarray(fw['RA_FIBERMAP'] if 'RA_FIBERMAP' in fw.names else fw['RA'],float)
            dec=np.asarray(fw['DEC_FIBERMAP'] if 'DEC_FIBERMAP' in fw.names else fw['DEC'],float)
        ok=(msk==0)&np.isfinite(SS)&np.isfinite(S); nf=SS.shape[0]
        wl=np.array([np.nanmedian(SS[i][ok[i]&(S[i]<np.nanpercentile(S[i][ok[i]],30))]) if ok[i].sum()>200 else np.nan for i in range(nf)])
        obj=np.array([np.nanmedian(C[i][ok[i]])-np.nanmedian(S[i][ok[i]]) if ok[i].sum()>200 else np.nan for i in range(nf)])
        blank=np.isfinite(wl)&np.isfinite(obj)&(np.abs(obj)<np.nanpercentile(np.abs(obj[np.isfinite(obj)]),50))
        # per-benchside along-slit continuum profile from blank fibres (incl edges), running median
        prof=np.zeros(nf)
        for b in sorted(set(bs)):
            idx=np.where(bs==b)[0]; rank=np.empty(idx.size,int); rank[np.argsort(idx)]=np.arange(idx.size)
            bmask=blank[idx]
            for k,i in enumerate(idx):
                near=idx[(np.abs(rank-rank[k])<=3)&bmask]
                if near.size>=3: prof[i]=np.nanmedian(wl[near])
        var=np.array([np.nanmedian(er[i][ok[i]])**2 if ok[i].sum()>200 and np.isfinite(er[i][ok[i]]).any() else 1.0 for i in range(nf)])
        for i in range(nf):
            if ok[i].sum()<200: continue
            RA.append(ra[i]);DEC.append(dec[i]);EXP.append(ei);WB.append(wl[i]);WA.append(wl[i]-prof[i]);VAR.append(var[i]);NP.append(int(ok[i].sum()))
    RA,DEC,EXP,WB,WA,VAR,NP=map(np.array,(RA,DEC,EXP,WB,WA,VAR,NP))
    cb=coh_floor(RA,DEC,EXP,WB,VAR,NP); ca=coh_floor(RA,DEC,EXP,WA,VAR,NP)
    print(f"{fld} green coherent floor (counts):  before={cb:.3f}  after edge-ramp removal={ca:.3f}  ({100*(1-ca/cb):+.0f}%)")
