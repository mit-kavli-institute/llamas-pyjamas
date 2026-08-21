"""Green bright-sky-line residual diagnosis, in the sky-subtracted RSS frame (rev01).
For each bright line, per fibre: take the SKYSUB residual in a window around the line and the base-SKY
line template; project the residual onto the shift(s'), width(s''), asym(s''') derivative basis of the
template (the same basis the framework's derivative fit uses, but here measuring what SURVIVES it).
Report: fractional residual (RMS/amp), which derivative-order dominates, and how it varies ACROSS THE
SLIT (fibre rank within benchside). Isolates arc-solution shift (would vanish in xshift) vs real LSF."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
SRC='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01/extractions'
OUT='/Users/simcoe/.claude/jobs/8fc668fa/tmp'
LINES={'5577[OI]':5577.34,'NaD 5890':5889.95,'6300[OI]':6300.30}
HALF=6.0   # Angstrom half-window

def pick(fld):
    return [x for x in sorted(glob.glob(f'{SRC}/*_RSS_green.fits')) if fld in str(fits.getheader(x,0).get('OBJECT',''))][0]

def analyze(path, fld):
    h=fits.open(path)
    SS=np.asarray(h['SKYSUB'].data,float); SKY=np.asarray(h['SKY'].data,float)
    W=np.asarray(h['WAVE'].data,float); M=np.asarray(h['MASK'].data)
    bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
    nf=SS.shape[0]
    # slit rank within benchside
    rank=np.full(nf,np.nan)
    for b in set(bs):
        idx=np.where(bs==b)[0]; rank[idx]=(np.argsort(np.argsort(idx))/(idx.size-1))  # 0..1 across slit
    rows=[]
    for name,lam in LINES.items():
        for i in range(nf):
            w=W[i]; sel=np.isfinite(w)&(np.abs(w-lam)<HALF)&(M[i]==0)
            if sel.sum()<9: continue
            s=SKY[i][sel]; d=SS[i][sel]
            amp=np.nanmax(s)-np.nanmin(s)
            if not np.isfinite(amp) or amp<=0: continue
            # derivative basis of the template (wavelength-grid index)
            s1=np.gradient(s); s2=np.gradient(s1); s3=np.gradient(s2)
            B=np.vstack([s,s1,s2,s3]).T
            fin=np.all(np.isfinite(B),1)&np.isfinite(d)
            if fin.sum()<8: continue
            coef,*_=np.linalg.lstsq(B[fin],d[fin],rcond=None)
            model=B[fin]@coef
            rms_raw=np.std(d[fin]); rms_after=np.std(d[fin]-model)
            rows.append(dict(line=name,fib=i,bs=bs[i],rank=rank[i],amp=amp,
                             fresid=rms_raw/amp, a=coef[0],shift=coef[1],width=coef[2],asym=coef[3],
                             rms_raw=rms_raw,rms_after=rms_after))
    return rows

def summarize(rows,fld):
    import collections
    print(f'\n=== {fld} green ===')
    for name in LINES:
        r=[x for x in rows if x['line']==name]
        if not r: continue
        fr=np.array([x['fresid'] for x in r]); amp=np.array([x['amp'] for x in r])
        sh=np.abs([x['shift'] for x in r]); wd=np.abs([x['width'] for x in r]); asy=np.abs([x['asym'] for x in r])
        rr=np.array([x['rms_raw'] for x in r]); ra=np.array([x['rms_after'] for x in r])
        print(f'{name:10s} nfib={len(r):4d}  frac_resid(RMS/amp) med={np.median(fr):.3f}  '
              f'residRMS med={np.median(rr):.3g}  after shift/width/asym fit={np.median(ra):.3g} '
              f'({100*(1-np.median(ra)/np.median(rr)):.0f}% explained)  '
              f'|shift|/|width|/|asym| med={np.median(sh):.2g}/{np.median(wd):.2g}/{np.median(asy):.2g}')

# figure: 5577 across-slit
fig,ax=plt.subplots(2,2,figsize=(14,9))
for col,fld in enumerate(('J2151','J0958')):
    rows=analyze(pick(fld),fld); summarize(rows,fld)
    r=[x for x in rows if x['line']=='5577[OI]']
    rank=np.array([x['rank'] for x in r]); fr=np.array([x['fresid'] for x in r])
    shift=np.array([x['shift'] for x in r]); width=np.array([x['width'] for x in r])
    a=ax[0,col]; a.scatter(rank,fr,s=6,alpha=.4); a.set_title(f'{fld} 5577 fractional residual vs slit pos')
    a.set_xlabel('slit position (0=edge,1=edge, within benchside)'); a.set_ylabel('residual RMS / line amp'); a.set_ylim(0,np.nanpercentile(fr,98))
    b=ax[1,col]
    # bin shift & width coeff vs slit
    bins=np.linspace(0,1,11); bc=0.5*(bins[:-1]+bins[1:])
    def binned(y):
        return np.array([np.nanmedian(y[(rank>=bins[k])&(rank<bins[k+1])]) for k in range(len(bc))])
    b.plot(bc,binned(shift),'o-',label='shift (s\')'); b.plot(bc,binned(width),'s-',label='width (s\'\')')
    b.axhline(0,color='k',lw=.5); b.set_title(f'{fld} 5577 residual shift/width vs slit pos'); b.set_xlabel('slit position'); b.legend()
fig.tight_layout(); p=f'{OUT}/green_line_resid.png'; fig.savefig(p,dpi=90); plt.close(fig)
print('\nFIG',p)
