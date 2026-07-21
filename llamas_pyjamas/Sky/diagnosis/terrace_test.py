"""Does 'smooth 2D field + per-camera mean removal' explain the striping?
(1) TOY: smooth dome on the IFU, subtract per-band means -> show terracing = stripes.
(2) DATA: fit measured blank-fibre floor(x,y) [J1613 green, production RSS] with ONE smooth 2D
    surface (cubic, 9 dof) + 8 per-camera offsets, vs a null model of 8 INDEPENDENT per-camera
    cubic slit profiles (32 dof). If the single surface wins/matches with fewer dof and its
    within-band slices reproduce every camera's shape, the terracing mechanism holds."""
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
OUT=f'{ND}/terrace_qa.png'

# ---------------- (1) toy demo ----------------
ny,nx=48,46
yy,xx=np.mgrid[0:ny,0:nx]
F=6*np.exp(-0.5*(((xx-18)/18.)**2+((yy-28)/20.)**2))          # smooth dome, few counts
bands=(yy//6).astype(int)                                      # 8 horizontal 6-row bands
Fr=F.copy()
for b in range(8): Fr[bands==b]-=F[bands==b].mean()            # per-band mean removal

# ---------------- (2) data fit ----------------
f=f'{ND}/reduced/extractions/LLAMAS_2026-05-17_02-49-56.7_RSS_green.fits'
with fits.open(f) as h:
    C=np.asarray(h['COUNTS'].data,float); S=np.asarray(h['SKY'].data,float)
    SS=np.asarray(h['SKYSUB'].data,float); msk=np.asarray(h['MASK'].data)
    bs=np.array([str(b).strip() for b in h['FIBERMAP'].data['BENCHSIDE']])
    fw=h['FIBERWCS'].data
    X=np.asarray(fw['X_FIBERMAP'],float); Y=np.asarray(fw['Y_FIBERMAP'],float)
ok=(msk==0)&np.isfinite(SS)&np.isfinite(S)
nf=C.shape[0]; floor=np.full(nf,np.nan); obj=np.full(nf,np.nan)
for i in range(nf):
    m=ok[i]
    if m.sum()<200: continue
    lo=m&(S[i]<np.nanpercentile(S[i][m],30))
    if lo.sum()>50: floor[i]=np.nanmedian(SS[i][lo])
    obj[i]=np.nanmedian(C[i][m])-np.nanmedian(S[i][m])
blank=(np.isfinite(obj)&(np.abs(obj)<np.nanpercentile(np.abs(obj[np.isfinite(obj)]),50))
       &np.isfinite(floor)&np.isfinite(X)&np.isfinite(Y))
ub=sorted(set(bs))
xn=(X-np.nanmean(X[blank]))/np.nanstd(X[blank]); yn=(Y-np.nanmean(Y[blank]))/np.nanstd(Y[blank])
# model A: ONE smooth cubic surface (no constant) + 8 per-camera offsets  -> 9+8 dof
polyA=[xn,yn,xn**2,xn*yn,yn**2,xn**3,xn**2*yn,xn*yn**2,yn**3]
ind=[(bs==b).astype(float) for b in ub]
A=np.column_stack(polyA+ind)
# model B (null): 8 independent per-camera cubics in within-camera rank -> 32 dof
cols=[]
for b in ub:
    r=np.zeros(nf); mb=bs==b
    r[mb]=np.linspace(-1,1,mb.sum())
    for p in range(4):
        c=np.zeros(nf); c[mb]=r[mb]**p; cols.append(c)
B=np.column_stack(cols)
w=np.where(blank)[0]
def fit(M):
    coef,*_=np.linalg.lstsq(M[w],floor[w],rcond=None)
    mod=M@coef
    resid=floor[w]-mod[w]
    # one 3-sigma clip pass
    s=1.4826*np.nanmedian(np.abs(resid-np.nanmedian(resid)))
    keep=w[np.abs(resid)<3*s]
    coef,*_=np.linalg.lstsq(M[keep],floor[keep],rcond=None)
    mod=M@coef
    r2=1-np.nanvar(floor[keep]-mod[keep])/np.nanvar(floor[keep])
    return coef,mod,r2,keep
coefA,modA,r2A,keepA=fit(A)
coefB,modB,r2B,keepB=fit(B)
print(f"model A (ONE smooth 2D surface + 8 offsets, 17 dof): R2 = {r2A:.3f}")
print(f"model B (8 independent per-camera cubics,   32 dof): R2 = {r2B:.3f}")
Fsurf=np.column_stack(polyA)@coefA[:9]                        # the fitted smooth field (up to const)
offs={b:coefA[9+k] for k,b in enumerate(ub)}
print("fitted per-camera offsets:", {b:round(float(offs[b]),2) for b in ub})
# self-consistency: offsets should ~ per-band means of the fitted surface (terracing prediction)
pred_off=np.array([np.nanmean(Fsurf[blank&(bs==b)]) for b in ub])
fit_off=np.array([-offs[b] for b in ub])                       # floor = F - c_b -> c_b = -offset
cc=np.corrcoef(pred_off,fit_off)[0,1]
print(f"terracing self-consistency corr(<F>_band, fitted c_b) = {cc:+.2f}")
qi=int(np.nanargmax(obj)); print(f"surface peak vs QSO: QSO at IFU ({X[qi]:.1f},{Y[qi]:.1f})")

fig,ax=plt.subplots(2,3,figsize=(15.5,8.6))
im=ax[0,0].imshow(F,origin='lower',cmap='inferno'); ax[0,0].set_title('TOY: smooth field F on IFU')
fig.colorbar(im,ax=ax[0,0])
im=ax[0,1].imshow(Fr,origin='lower',cmap='RdBu_r',vmin=-3,vmax=3)
for b in range(1,8): ax[0,1].axhline(6*b-0.5,color='k',lw=.3)
ax[0,1].set_title('TOY: F − per-band mean = TERRACES (stripes)'); fig.colorbar(im,ax=ax[0,1])
sc=ax[0,2].scatter(X[blank],Y[blank],c=np.clip(floor[blank],-8,8),s=9,cmap='RdBu_r')
ax[0,2].plot(X[qi],Y[qi],'k*',ms=12); ax[0,2].set_title('DATA: measured floor on IFU (* = QSO)')
fig.colorbar(sc,ax=ax[0,2])
sc=ax[1,0].scatter(X[blank],Y[blank],c=np.clip(Fsurf[blank]-np.nanmean(Fsurf[blank]),-8,8),s=9,cmap='RdBu_r')
ax[1,0].plot(X[qi],Y[qi],'k*',ms=12); ax[1,0].set_title(f'fitted smooth surface F (17-dof model, R2={r2A:.2f})')
fig.colorbar(sc,ax=ax[1,0])
sc=ax[1,1].scatter(X[blank],Y[blank],c=np.clip(modA[blank],-8,8),s=9,cmap='RdBu_r')
ax[1,1].set_title('model A prediction: F − c_b (terraced)'); fig.colorbar(sc,ax=ax[1,1])
resA=floor-modA
sc=ax[1,2].scatter(X[blank],Y[blank],c=np.clip(resA[blank],-8,8),s=9,cmap='RdBu_r')
ax[1,2].set_title('DATA − model A (what terracing fails to explain)'); fig.colorbar(sc,ax=ax[1,2])
fig.suptitle('Terracing test: does ONE smooth IFU field + per-camera mean removal reproduce the stripes?',fontsize=12)
fig.savefig(OUT,dpi=110,bbox_inches='tight',facecolor='white');plt.close(fig)
print("wrote",OUT)
