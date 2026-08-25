"""RS's chromaticity hypothesis: is per-fibre throughput wavelength-dependent?
Per blank fibre (green): OH-derived throughput in the BLUE half vs RED half of the channel
(chrom_OH = t_b/t_r), cross-checked against the same ratio from the chromatic fibre-flat
CORRECTION(lambda) table (chrom_flat). Agreement => real chromatic throughput, already encoded in
the RSS flat but IGNORED by the sky model's scalar relative_throughput."""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
OUT=f'{ND}/chromaticity_qa.png'
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

pkl=sorted(glob.glob(f'{ND}/reduced_ped_off/extractions/*sky1d*.pkl'))[0]
d=ExtractLlamas.loadExtraction(pkl)
science=d['extractions']; meta=d['metadata']
ffc=fits.open(f'{ND}/reduced_new/extractions/flat/fibre_flat_corrections.fits')

def oh_amp(C,S,i,half):
    """OH-line amplitude for fibre i restricted to a wavelength half ('b'|'r')."""
    fin=np.isfinite(S[i])&np.isfinite(C[i])
    nz=np.where(fin)[0]
    if nz.size<200: return np.nan
    mid=nz[nz.size//2]
    m=fin.copy(); m[mid:]=False if half=='b' else m[mid:]
    if half=='b': m=fin&(np.arange(C.shape[1])<mid)
    else:         m=fin&(np.arange(C.shape[1])>=mid)
    s=S[i]
    hi=m&(s>np.nanpercentile(s[m],85)); lo=m&(s<np.nanpercentile(s[m],30))
    if hi.sum()<10 or lo.sum()<25: return np.nan
    return np.nanmedian(C[i][hi])-np.nanmedian(C[i][lo])

print(f"{'cam':>4} {'nblank':>6} {'std(chromOH)':>12} {'std(chromFLAT)':>14} {'corr':>6}")
allo=[]; allf=[]
for ext,md in zip(science,meta):
    if str(md.get('channel','')).lower()!='green': continue
    cam=f"{md['bench']}{md['side']}"
    C=np.asarray(ext.counts,float); S=np.asarray(ext.sky,float)
    nf=C.shape[0]
    wl=np.nanmedian(C,axis=1)
    good=np.isfinite(wl)&(wl>0)
    blank=good&(wl<=np.nanpercentile(wl[good],60))
    # OH-derived blue/red throughput ratio
    tb=np.full(nf,np.nan); tr=np.full(nf,np.nan)
    for i in np.where(blank)[0]:
        tb[i]=oh_amp(C,S,i,'b'); tr[i]=oh_amp(C,S,i,'r')
    m=blank&np.isfinite(tb)&np.isfinite(tr)&(tr>0)&(tb>0)
    chromOH=np.full(nf,np.nan)
    chromOH[m]=(tb[m]/np.nanmedian(tb[m]))/(tr[m]/np.nanmedian(tr[m]))
    # fibre-flat CORRECTION(λ) blue/red ratio for the same camera
    try:
        tab=ffc[f"green_{md['bench']}_{md['side']}"].data
        cor=np.asarray(tab['CORRECTION'],float)                    # (nfib, 2048)
        nw=cor.shape[1]
        cb=np.nanmedian(cor[:,:nw//2],axis=1); cr=np.nanmedian(cor[:,nw//2:],axis=1)
        chromF=np.full(nf,np.nan)
        k=min(nf,cor.shape[0])
        with np.errstate(divide='ignore',invalid='ignore'):
            cf=(cb[:k]/np.nanmedian(cb[:k]))/(cr[:k]/np.nanmedian(cr[:k]))
        chromF[:k]=cf
    except KeyError:
        chromF=np.full(nf,np.nan)
    mm=m&np.isfinite(chromF)
    r=np.corrcoef(chromOH[mm],chromF[mm])[0,1] if mm.sum()>20 else np.nan
    print(f"{cam:>4} {int(mm.sum()):>6} {np.nanstd(chromOH[m]):>12.4f} {np.nanstd(chromF[mm]):>14.4f} {r:>+6.2f}")
    allo.append(chromOH[mm]); allf.append(chromF[mm])
ao=np.concatenate(allo); af=np.concatenate(allf)
R=np.corrcoef(ao,af)[0,1]
print(f"\nALL: corr(chrom_OH, chrom_flat)={R:+.3f}  n={ao.size}")
print(f"chromaticity amplitude: std(t_blue/t_red) OH={np.nanstd(ao):.4f}  flat={np.nanstd(af):.4f}")
print("(scalar relative_throughput assumes this ratio == 1 for every fibre)")

fig,ax=plt.subplots(1,2,figsize=(11,4.6))
ax[0].scatter(af,ao,s=5,alpha=.3)
lim=[np.nanpercentile(af,1),np.nanpercentile(af,99)]
ax[0].plot(lim,lim,'r--',lw=1)
ax[0].set_xlabel('chromaticity from fibre-flat CORRECTION(λ)'); ax[0].set_ylabel('chromaticity from night-sky OH')
ax[0].set_title(f'per-fibre t_blue/t_red: flat vs OH  (corr={R:+.2f})')
ax[1].hist(ao,bins=60,alpha=.6,label='OH-derived'); ax[1].hist(af,bins=60,alpha=.6,label='flat-derived')
ax[1].axvline(1,color='k',lw=.7); ax[1].legend(); ax[1].set_xlabel('t_blue/t_red per fibre')
ax[1].set_title('achromatic assumption = delta at 1')
fig.suptitle('Is per-fibre throughput chromatic? (J1613 green blank fibres)')
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig)
print("wrote",OUT)
