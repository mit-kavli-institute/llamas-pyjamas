"""Is the per-camera along-slit floor profile STATIC (instrumental -> calibrate once) or
field-dependent (ambient/target-scattered -> per-frame)? Compare smoothed profiles across
frames within J1613 and across fields (J1613/J2151/J0958), green, production RSS."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
OUT=f'{ND}/floor_stability_qa.png'

def profiles(path):
    with fits.open(path) as h:
        C=np.asarray(h['COUNTS'].data,float); S=np.asarray(h['SKY'].data,float)
        SS=np.asarray(h['SKYSUB'].data,float); msk=np.asarray(h['MASK'].data)
        bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
    ok=(msk==0)&np.isfinite(SS)&np.isfinite(S)
    nf=C.shape[0]; fl=np.full(nf,np.nan); obj=np.full(nf,np.nan)
    for i in range(nf):
        m=ok[i]
        if m.sum()<200: continue
        lo=m&(S[i]<np.nanpercentile(S[i][m],30))
        if lo.sum()>50: fl[i]=np.nanmedian(SS[i][lo])
        obj[i]=np.nanmedian(C[i][m])-np.nanmedian(S[i][m])
    blank=np.isfinite(obj)&(np.abs(obj)<np.nanpercentile(np.abs(obj[np.isfinite(obj)]),50))
    out={}
    for b in sorted(set(bs)):
        m=(bs==b)
        v=np.where(blank&np.isfinite(fl)&m, fl, np.nan)[m]
        # fill gaps by interpolation then smooth -> comparable profiles
        idx=np.arange(v.size); g=np.isfinite(v)
        if g.sum()<40: continue
        vi=np.interp(idx,idx[g],v[g])
        out[b]=median_filter(vi,size=21,mode='nearest')
    return out

# frames: 3x J1613 (dithers), 1x J2151, 1x J0958
frames=[]
for field,n in (('J1613',3),('J2151',1),('J0958',1)):
    got=0
    for f in sorted(glob.glob(f'{ND}/reduced/extractions/*_RSS_green.fits')):
        if field in str(fits.getheader(f,0).get('OBJECT','')):
            frames.append((field,f)); got+=1
            if got>=n: break
labels=[f'{fd}:{f.split("_RSS")[0][-9:]}' for fd,f in frames]
profs=[profiles(f) for _,f in frames]
cams=sorted(set.intersection(*[set(p) for p in profs]))
print("frames:",labels)
print(f"\n{'cam':>4} {'within-J1613 corr':>17} {'cross-field corr':>16} {'profile rms':>11}")
w_all=[]; x_all=[]
for cam in cams:
    P=[p[cam]-np.nanmean(p[cam]) for p in profs]
    L=min(len(x) for x in P); P=[x[:L] for x in P]
    within=[np.corrcoef(P[a],P[b])[0,1] for a in range(3) for b in range(a+1,3)]
    cross=[np.corrcoef(P[a],P[b])[0,1] for a in range(3) for b in (3,4)]
    w,x=np.nanmean(within),np.nanmean(cross)
    w_all.append(w); x_all.append(x)
    print(f"{cam:>4} {w:>17.2f} {x:>16.2f} {np.nanstd(P[0]):>11.2f}")
print(f"\nMEAN within-field corr = {np.nanmean(w_all):+.2f}   cross-field corr = {np.nanmean(x_all):+.2f}")
print("(both high => STATIC instrumental template; within>>cross => field/target-dependent)")

show=[c for c in ('1A','2A','3B','4B') if c in cams]
fig,ax=plt.subplots(1,len(show),figsize=(4.4*len(show),4.2),squeeze=False)
for j,cam in enumerate(show):
    for k,(lab,p) in enumerate(zip(labels,profs)):
        v=p[cam]-np.nanmean(p[cam])
        ax[0,j].plot(v,lw=1,label=lab if j==0 else None,
                     ls='-' if k<3 else '--')
    ax[0,j].axhline(0,color='k',lw=.4); ax[0,j].set_title(cam); ax[0,j].set_xlabel('slit position')
fig.legend(loc='upper right',fontsize=7)
fig.suptitle('Floor profile stability: J1613 dithers (solid) vs other fields (dashed), green')
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig)
print("wrote",OUT)
