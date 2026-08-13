"""Build the green LSF-residual template and validate the INTEGRATED refine (Phase A vs A+B) held-out,
then build + save the production template from all 17 frames and round-trip check."""
import sys; sys.path.insert(0,'/Users/simcoe/GIT/llamas-pyjamas/.claude/worktrees/exposure-stacking')
import glob, warnings; warnings.filterwarnings('ignore')
import numpy as np
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.Sky.skyLineTemplate import build_line_template, save_template, load_template
from llamas_pyjamas.Sky.skyLineRefine import refine_sky_lines_pkl
D='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/reduced_rev01_pkl/extractions'
CAL='/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17/calib'
def pk(stamp): return glob.glob(f'{D}/*{stamp}*sky1d*extractions.pkl')[0]
J0958=['00-05-50.8','00-33-44.3','01-01-33.7','01-32-28.6','01-55-56.0']
J1613=['02-49-56.7','03-30-45.4','04-13-54.3','04-56-10.5','05-43-06.1','06-22-48.3','07-03-13.7','07-43-54.7']
J2151=['08-24-02.8','08-49-49.1','09-28-44.4','10-08-24.8']
LINES=[5577.34,6300.30]; HALF=4.0

def lineonly(y,xr):
    a=np.polyfit([xr[0],xr[-1]],[np.median(y[:2]),np.median(y[-2:])],1); return y-np.polyval(a,xr)
def measure(sci, md):
    out=[]
    for i,e in enumerate(sci):
        if str(md[i].get('channel')).lower()!='green': continue
        W=np.asarray(e.wave,float); C=np.asarray(e.counts,float); S=np.asarray(e.sky,float)
        for fb in range(W.shape[0]):
            w=W[fb]
            for lam in LINES:
                if not(np.nanmin(w)<lam<np.nanmax(w)): continue
                sel=np.where(np.abs(w-lam)<HALF)[0]
                if sel.size<7: continue
                c=C[fb][sel]; s=S[fb][sel]; xr=np.arange(sel.size)
                if not(np.all(np.isfinite(c))and np.all(np.isfinite(s))): continue
                cb=lineonly(c,xr); sb=lineonly(s,xr); amp=np.nanmax(sb)
                if amp<50: continue
                out.append(np.nanstd(cb-sb)/amp)
    return np.array(out)

# --- held-out build (J0958+J1613) ---
print('building held-out template (J0958+J1613)...', flush=True)
tmpl_ho, off, _ = build_line_template([pk(s) for s in J0958+J1613], 'green')
print(f'held-out template cameras: {sorted(tmpl_ho)}', flush=True)

# --- integrated validation on held-out J2151 ---
base=[]; A=[]; AB=[]
for s in J2151:
    d=ExtractLlamas.loadExtraction(pk(s)); base+=list(measure(d['extractions'],d['metadata']))
    d=ExtractLlamas.loadExtraction(pk(s)); refine_sky_lines_pkl(d['extractions'],{},d['metadata']); A+=list(measure(d['extractions'],d['metadata']))
    d=ExtractLlamas.loadExtraction(pk(s)); refine_sky_lines_pkl(d['extractions'],{},d['metadata'],templates=tmpl_ho,offgrid=off); AB+=list(measure(d['extractions'],d['metadata']))
base,A,AB=map(np.array,(base,A,AB))
print(f'\nINTEGRATED held-out J2151 at-line frac-resid (green 5577+6300, n={len(base)}):')
print(f'  base   = {np.median(base):.3f}')
print(f'  PhaseA = {np.median(A):.3f}  ({100*(1-np.median(A)/np.median(base)):+.0f}% vs base)')
print(f'  A+tmpl = {np.median(AB):.3f}  ({100*(1-np.median(AB)/np.median(A)):+.0f}% vs A, {100*(1-np.median(AB)/np.median(base)):+.0f}% vs base)')

# --- production build (all 17) + save + round-trip ---
print('\nbuilding production template (all 17 frames)...', flush=True)
tmpl_all, off2, diag = build_line_template([pk(s) for s in J0958+J1613+J2151], 'green')
outp=f'{CAL}/line_template_green.fits'
save_template(outp, tmpl_all, off2, 'green', diag)
t2,o2=load_template(outp,'green')
print(f'saved {outp}; round-trip cameras match: {sorted(t2)==sorted(tmpl_all)}; off match: {np.allclose(o2,off2)}')
print('DONE')
