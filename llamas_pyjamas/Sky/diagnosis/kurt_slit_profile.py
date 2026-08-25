"""Test RS's constraint: line-shape (kurtosis) should vary SMOOTHLY and SYMMETRICALLY along the slit
(per camera), not randomly. Per green camera, measure per-fibre 5577 excess-kurtosis for each of the 21
rev02 frames, AVERAGE over frames (LSF is instrument-stable; counts unaffected by sky refine), then test:
  - smoothness: lag-1 autocorrelation of frame-averaged kurtosis vs fibre index (physical slit order)
  - symmetry: quadratic fit k = A*idx^2 + B*idx + C -> curvature sign + vertex + R^2
A high lag-1 autocorr + good parabola = RS is right (smooth, symmetric-about-centre). Read-only.
"""
import sys, glob, warnings; warnings.filterwarnings('ignore')
import os as _os  # repo root = three levels up from this diagnosis script
sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..')))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
ND = '/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17'; OUT = _os.environ.get('LLAMAS_DIAG_OUT', _os.path.join(_os.path.dirname(__file__), 'figures'))
W, MW, AMP_MIN, LAM = 8, 5, 1500.0, 5577.34
frames = sorted(glob.glob(f'{ND}/reduced_rev02/extractions/*sky1d_extractions.pkl'))
print(f'{len(frames)} rev02 frames')

def exkurt(x, p):
    wgt = np.clip(p, 0.0, None); Sw = wgt.sum()
    if Sw <= 0: return np.nan
    mu = (wgt * x).sum() / Sw; d = x - mu; m2 = (wgt * d ** 2).sum() / Sw
    return (wgt * d ** 4).sum() / Sw / m2 ** 2 - 3.0 if m2 > 0 else np.nan

# accumulate kurt[cam] -> list of (nfibre) arrays, one per frame
acc = {}
for fr in frames:
    d = ExtractLlamas.loadExtraction(fr)
    for i, m in enumerate(d['extractions'] if False else d['metadata']):
        if str(m.get('channel')).lower() != 'green': continue
        e = d['extractions'][i]; cam = f"{m.get('bench')}{m.get('side')}"
        C = np.asarray(e.counts, float); WV = np.asarray(e.wave, float); XS = np.asarray(e.xshift, float)
        S = np.asarray(e.sky, float); nf = C.shape[0]
        kv = np.full(nf, np.nan)
        for fb in range(nf):
            wv = WV[fb]; sel = np.where(np.abs(wv - LAM) < 8)[0]
            if sel.size < 3: continue
            pk = sel[int(np.nanargmax(S[fb][sel]))]
            if pk - W < 0 or pk + W >= wv.size: continue
            idx = np.arange(pk - W, pk + W + 1); xx = XS[fb][idx]; cc = C[fb][idx]
            if not (np.all(np.isfinite(xx)) and np.all(np.isfinite(cc))): continue
            P = cc - np.polyval(np.polyfit([xx[0], xx[-1]], [np.median(cc[:2]), np.median(cc[-2:])], 1), xx)
            if np.nanmax(P) < AMP_MIN: continue
            mi = np.arange(W - MW, W + MW + 1); kv[fb] = exkurt(xx[mi] - xx[W], P[mi])
        acc.setdefault(cam, []).append(kv)

def lag1(a):
    a = np.asarray(a, float); m = np.isfinite(a[:-1]) & np.isfinite(a[1:])
    return np.corrcoef(a[:-1][m], a[1:][m])[0, 1] if m.sum() >= 8 else np.nan

cams = sorted(acc)
fig, ax = plt.subplots(2, 4, figsize=(20, 9)); axf = ax.ravel()
print('\ncam   nfib  lag1-autocorr   quad A (curv)   vertex(frac)   parab R^2   kurt(centre vs ends)')
for k, cam in enumerate(cams):
    K = np.vstack(acc[cam])                      # (nframe, nfib)
    mk = np.nanmean(K, 0); sem = np.nanstd(K, 0) / np.sqrt(np.sum(np.isfinite(K), 0).clip(1))
    fi = np.arange(mk.size); good = np.isfinite(mk)
    a1 = lag1(mk)
    # quadratic fit
    x = fi[good]; y = mk[good]
    if x.size > 10:
        A, B, Cc = np.polyfit(x, y, 2); yhat = np.polyval([A, B, Cc], x)
        r2 = 1 - np.nansum((y - yhat) ** 2) / np.nansum((y - np.nanmean(y)) ** 2)
        vtx = (-B / (2 * A)) / mk.size
        # centre vs ends
        cen = np.nanmedian(mk[good][(x > 0.4 * mk.size) & (x < 0.6 * mk.size)])
        end = np.nanmedian(mk[good][(x < 0.15 * mk.size) | (x > 0.85 * mk.size)])
    else:
        A = r2 = vtx = cen = end = np.nan
    print(f'{cam}   {good.sum():4d}   {a1:+.2f}          {A:+.2e}     {vtx:+.2f}         {r2:+.2f}       '
          f'{cen:+.2f} vs {end:+.2f}')
    axf[k].errorbar(fi[good], mk[good], yerr=sem[good], fmt='.', ms=3, alpha=0.5, color='C0', elinewidth=0.5)
    if np.isfinite(A):
        xs = np.linspace(x.min(), x.max(), 100); axf[k].plot(xs, np.polyval([A, B, Cc], xs), '-', color='C3', lw=2)
    # running median
    from scipy.ndimage import median_filter
    mm = mk.copy(); mm[~good] = np.nanmedian(mk[good]); axf[k].plot(fi, median_filter(mm, 15), '-', color='C2', lw=1, alpha=0.7)
    axf[k].set_title(f'green {cam}: 5577 kurtosis vs fibre  (lag1={a1:+.2f}, R²={r2:+.2f})', fontsize=9)
    axf[k].set_xlabel('fibre index (slit order)'); axf[k].set_ylabel('excess kurtosis')
fig.suptitle('Per-fibre 5577 line kurtosis along the slit, per green camera (21-frame average)', fontsize=13)
fig.tight_layout(); pth = f'{OUT}/kurt_slit_profile.png'; fig.savefig(pth, dpi=100); print('\nFIG', pth)
print('\nlag1>>0 = smooth fibre-to-fibre (RS right); R^2 high + consistent curvature sign = symmetric trend.')
