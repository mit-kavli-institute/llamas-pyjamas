"""FULL 8-frame J1613 test, template EXCLUDING J1613 (plan A):
1. build floor templates (3 channels) from J2151+J0958 quasar frames only (9 frames)
2. extend both configs to all 8 J1613 dithers; point ON at the noJ1613 templates
3. delete ON sky-stage products (all frames get the new template treatment); OFF keeps its 3 cached
4. reduce OFF then ON (resumable, flat/traces cached)
5. Gaia-register the 8 OFF frames (block solve, both rotation groups); sync WCS OFF->ON
6. stripe-specific metric + visual QA on the 8-frame stacks
"""
import sys; sys.path.insert(0, '/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, os, subprocess, warnings; warnings.filterwarnings('ignore')
import numpy as np
from astropy.io import fits
ND='/Users/simcoe/data/LLAMAS/may26/ut20260516_17'
T='/Users/simcoe/.claude/jobs/8fc668fa/tmp'
WT='/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking'
PY='/opt/anaconda3/envs/llamas_reduce/bin/python'

# ---- 1. templates without J1613 ----
from llamas_pyjamas.Sky.skyFloorTemplate import build_floor_template, save_template
for chan in ('green','blue','red'):
    out=f'{ND}/reduced/floor_template_noJ1613_{chan}.fits'
    if os.path.exists(out): print(f'{chan}: noJ1613 template exists'); continue
    files=[f for f in sorted(glob.glob(f'{ND}/reduced/extractions/*_RSS_{chan}.fits'))
           if any(t in str(fits.getheader(f,0).get('OBJECT','')) for t in ('J2151','J0958'))]
    print(f'{chan}: building noJ1613 template from {len(files)} frames...', flush=True)
    tpl,diag=build_floor_template(files,chan)
    save_template(out,tpl,diag,chan)
    print(f'{chan}: wrote {out}', flush=True)

# ---- 2. configs: all 8 J1613 dithers ----
times=['02-49-56.7','03-30-45.4','04-13-54.3','04-56-10.5','05-43-06.1','06-22-48.3','07-03-13.7','07-43-54.7']
mefs=[f'{ND}/LLAMAS_2026-05-17_{t}_SCI22_mef.fits' for t in times]
missing=[m for m in mefs if not os.path.exists(m)]
assert not missing, f'missing raw MEFs: {missing}'
for v in ('off','on'):
    cfg=f'{T}/config_ped_{v}.txt'; txt=open(cfg).read()
    if 'FULL8' not in txt:
        with open(cfg,'a') as f:
            f.write('\n# FULL8 test\nscience_files = '+', '.join(mefs)+'\n')
            if v=='on':
                f.write(f'sky_pedestal_template = {ND}/reduced/floor_template_noJ1613_{{channel}}.fits\n')
        print(f'config_ped_{v} -> 8 frames')

# ---- 3. reset ON sky-stage products ----
n=0
for pat in ('*_sky1d_extractions.pkl','*_sky1dped_extractions.pkl','*_RSS_*.fits','*_whitelight_fullpipeline.fits'):
    for f in glob.glob(f'{ND}/reduced_ped_on/extractions/{pat}'): os.remove(f); n+=1
print(f'deleted {n} ON sky-stage products')

# ---- 4. reduce ----
env=dict(os.environ, PYTHONPATH=WT)
for v in ('off','on'):
    print(f'=== reducing {v} (8 frames) ===', flush=True)
    with open(f'{T}/reduce_{v}.log','w') as log:
        rc=subprocess.run([PY,'-m','llamas_pyjamas.reduce',f'{T}/config_ped_{v}.txt'],
                          cwd=WT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    print(f'{v}: rc={rc}', flush=True)
    if rc!=0: sys.exit(f'{v} reduction failed')

# ---- 5. register OFF, sync WCS -> ON ----
from llamas_pyjamas.Utils.register import register_exposures
offs=sorted(glob.glob(f'{ND}/reduced_ped_off/extractions/*_RSS_green.fits'))
print(f'registering {len(offs)} OFF frames...', flush=True)
try: register_exposures(offs)
except Exception as e: print('registration issue:',e)
for fo in offs:
    fn=fo.replace('reduced_ped_off','reduced_ped_on')
    if not os.path.exists(fn): print('missing ON:',fn); continue
    with fits.open(fo) as ho:
        fw=ho['FIBERWCS'].copy() if 'FIBERWCS' in [x.name for x in ho] else None
        fm=ho['FIBERMAP'].data
    with fits.open(fn,mode='update') as hn:
        names=[x.name for x in hn]
        if fw is not None:
            if 'FIBERWCS' in names: hn['FIBERWCS'].data=fw.data; hn['FIBERWCS'].header.update(fw.header)
            else: hn.append(fw)
        hn['FIBERMAP'].data['RA']=fm['RA']; hn['FIBERMAP'].data['DEC']=fm['DEC']; hn.flush()
print('WCS synced OFF->ON')

# ---- 6. stripe metric + visual on the 8-frame stacks ----
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use('Agg'); matplotlib.rcParams['text.usetex']=False
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from llamas_pyjamas.Combine.superRSS import FibreTable
from llamas_pyjamas.Combine.coadd import coadd_image
TH=np.deg2rad(145.0)
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
def co(A,sel=None):
    RA,DEC,EXP,WL,VAR,NP=A
    if sel is None: sel=np.ones(RA.size,bool)
    return coadd_image(FibreTable(ra=RA[sel],dec=DEC[sel],value=WL[sel],var=VAR[sel],
        solid_angle=np.full(int(sel.sum()),0.44),exposure=EXP[sel],
        channel=np.array(['green']*int(sel.sum())),npix=NP[sel]),units='flux',weighting='ivar')
def hpimg(im):
    d=im.data; core=im.nexp>=np.nanmax(im.nexp)-0.5
    src=d>np.nanpercentile(d[core&np.isfinite(d)],90); bp=core&np.isfinite(d)&~src
    h=d-median_filter(np.nan_to_num(d,nan=np.nanmedian(d)),size=9)
    return h,bp
def stripe(im):
    h,bp=hpimg(im); ny,nx=h.shape; yy,xx=np.mgrid[0:ny,0:nx]
    m=bp&np.isfinite(h); p=(xx*np.cos(TH)+yy*np.sin(TH))[m]; v=h[m]
    nb=80; pe=np.linspace(p.min(),p.max(),nb+1)
    prof=np.array([np.nanmedian(v[(p>=pe[k])&(p<pe[k+1])]) if ((p>=pe[k])&(p<pe[k+1])).sum()>4 else np.nan for k in range(nb)])
    return float(np.nanstd(prof))
R={}
for v in ('off','on'):
    A=load(v); R[v]=dict(st=co(A),f0=co(A,A[2]==0),nfr=int(A[2].max())+1)
s_off,s_on=stripe(R['off']['st']),stripe(R['on']['st'])
f_off,f_on=stripe(R['off']['f0']),stripe(R['on']['f0'])
print(f"\n=== FULL {R['off']['nfr']}-FRAME RESULT (stripe amplitude, 145 deg) ===")
print(f"single frame: off {f_off:.3f} -> on {f_on:.3f}  ({100*(1-f_on/f_off):+.0f}%)")
print(f"8-frame stack: off {s_off:.3f} -> on {s_on:.3f}  ({100*(1-s_on/s_off):+.0f}%)")
fig,ax=plt.subplots(2,2,figsize=(12,10.5))
lo,hi=ZScaleInterval().get_limits(R['off']['st'].data[np.isfinite(R['off']['st'].data)])
h0,bp0=hpimg(R['off']['st']); hl=np.nanpercentile(np.abs(h0[bp0]),97)
for r,v in enumerate(('off','on')):
    ax[r,0].imshow(R[v]['st'].data,origin='lower',vmin=lo,vmax=hi,cmap='inferno')
    ax[r,0].set_title(f'{v.upper()}: 8-frame stack white-light')
    h,bp=hpimg(R[v]['st'])
    ax[r,1].imshow(np.where(bp,h,np.nan),origin='lower',vmin=-hl,vmax=hl,cmap='RdBu_r')
    ax[r,1].set_title(f'{v.upper()}: stack high-pass (stripe amp {stripe(R[v]["st"]):.2f})')
fig.suptitle('FULL 8-dither J1613 stack — pedestal OFF vs template ON (template excludes J1613)',fontsize=13)
fig.savefig(f'{ND}/pedestal8_visual_qa.png',dpi=110,bbox_inches='tight',facecolor='white')
print('wrote',f'{ND}/pedestal8_visual_qa.png')
print('ALL DONE')
