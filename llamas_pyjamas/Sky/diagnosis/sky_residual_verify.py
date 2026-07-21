"""Phase 2 verification: does the additive per-camera floor + slit-edge-dip signature hold across
FIELDS (J1613/J2151/J0958) and CHANNELS (blue/green/red)? Blank-fibre residual only."""
import glob, warnings, collections; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
RED='/Users/simcoe/data/LLAMAS/may26/ut20260516_17/reduced'
OUT='/Users/simcoe/.claude/jobs/8fc668fa/tmp/skydiag'; import os; os.makedirs(OUT,exist_ok=True)
FIELDS=['J1613','J2151','J0958']; CHANS=['blue','green','red']

def field_files(field, chan):
    out=[]
    for f in sorted(glob.glob(f'{RED}/extractions/*_RSS_{chan}.fits')):
        try:
            if field in str(fits.getheader(f,0).get('OBJECT','')): out.append(f)
        except Exception: pass
    return out

def per_fibre(files):
    floors=[]; skyl=[]; obj=[]; bs=None
    for f in files:
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
    return np.nanmedian(floors,axis=0), np.nanmedian(skyl,axis=0), np.nanmedian(obj,axis=0), bs

def edge_dip(floor, bs, blank, nedge=15):
    """median floor at benchside-block edges minus interior (negative => edge over-subtraction)."""
    edges=[]; inter=[]
    for b in sorted(set(bs)):
        idx=np.where(bs==b)[0]
        e=np.r_[idx[:nedge], idx[-nedge:]]; ii=idx[nedge:-nedge]
        e=e[blank[e]]; ii=ii[blank[ii]]
        if e.size>5 and ii.size>5:
            edges.append(np.nanmedian(floor[e])); inter.append(np.nanmedian(floor[ii]))
    if not edges: return np.nan
    return float(np.nanmedian(edges)-np.nanmedian(inter))

def stripe_rms(field, chan):
    cf=f'{RED}/combined/{field}+*_cube_{chan}.fits'; g=glob.glob(cf)
    if not g: return np.nan
    with fits.open(g[0]) as h:
        cube=np.asarray(h[0].data,float); nexp=np.asarray(h['NEXP'].data,float) if 'NEXP' in h else None
    white=np.nanmedian(cube,axis=0); core=(nexp>=np.nanmax(nexp)-0.5) if nexp is not None else np.isfinite(white)
    if not np.isfinite(white[core]).any(): return np.nan
    src=white>np.nanpercentile(white[core&np.isfinite(white)],90)
    bp=core&np.isfinite(white)&~src
    hp=white-median_filter(np.nan_to_num(white,nan=np.nanmedian(white)),size=9)
    return float(np.nanstd(hp[bp])) if bp.sum()>50 else np.nan

print(f"{'field':>6} {'chan':>6} {'nexp':>4} {'nblank':>7} {'medfloor':>9} {'edge_dip':>9} "
      f"{'corrSKY':>8} {'worst benchsides':>26} {'stripeRMS':>10}")
summary=collections.defaultdict(dict)
for field in FIELDS:
    figrows=[]
    for chan in CHANS:
        files=field_files(field,chan)
        if not files: continue
        floor,sky_lev,obj,bs=per_fibre(files)
        oscale=np.nanpercentile(np.abs(obj[np.isfinite(obj)]),50) if np.isfinite(obj).any() else 0
        blank=np.isfinite(obj)&(np.abs(obj)<max(2*oscale,np.nanpercentile(obj[np.isfinite(obj)],40)))
        g=blank&np.isfinite(floor)&np.isfinite(sky_lev)&(sky_lev>0)
        cs=np.corrcoef(sky_lev[g],floor[g])[0,1] if g.sum()>20 else np.nan
        med=np.nanmedian(floor[blank])
        dip=edge_dip(floor,bs,blank)
        bsoff=sorted([(b,float(np.nanmedian(floor[(bs==b)&blank]))) for b in sorted(set(bs))],
                     key=lambda r:-abs(r[1]))[:3]
        worst=' '.join(f'{b}{v:+.1f}' for b,v in bsoff)
        srms=stripe_rms(field,chan)
        print(f"{field:>6} {chan:>6} {len(files):>4} {int(blank.sum()):>7} {med:>+9.2f} {dip:>+9.2f} "
              f"{cs:>+8.2f} {worst:>26} {srms:>10.2e}")
        figrows.append((chan,floor,blank))
        summary[field][chan]=dict(med=med,dip=dip,corr=cs,worst=worst,srms=srms)
    if figrows:
        fig,axes=plt.subplots(len(figrows),1,figsize=(11,2.4*len(figrows)),squeeze=False)
        for k,(chan,floor,blank) in enumerate(figrows):
            ax=axes[k,0]; ff=np.where(blank,floor,np.nan)
            ax.plot(ff,'.',ms=2); ax.axhline(0,color='k',lw=.5); ax.set_ylim(-15,15)
            ax.set_ylabel(f'{chan}\nfloor'); ax.set_xlim(0,ff.size)
        axes[0,0].set_title(f'{field}: blank-fibre sky residual per fibre (blue/green/red)')
        fig.savefig(f'{OUT}/verify_{field}.png',dpi=110,bbox_inches='tight',facecolor='white'); plt.close(fig)
print("\nwrote verify_<field>.png for fields with data")
