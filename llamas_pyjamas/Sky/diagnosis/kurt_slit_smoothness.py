"""Quantify how SMOOTH the along-slit kurtosis trend really is, separating the smooth signal from
per-fibre measurement noise. Per green camera (21-frame-avg 5577 kurtosis): (a) order-4 polynomial R^2
(fraction of the frame-averaged fibre variation that is a smooth function of slit position); (b) lag-1
autocorrelation after binning fibres in groups of 8 (beats down per-fibre noise -> reveals the true
fibre-to-fibre coherence RS expects to be 'extremely high'). Saves mk arrays for fix-prototyping."""
import sys, glob, warnings; warnings.filterwarnings('ignore')
import os as _os  # repo root = three levels up from this diagnosis script
sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..')))
import numpy as np
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
ND = '/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17'; OUT = _os.environ.get('LLAMAS_DIAG_OUT', _os.path.join(_os.path.dirname(__file__), 'figures'))
W, MW, AMP_MIN, LAM = 8, 5, 1500.0, 5577.34
frames = sorted(glob.glob(f'{ND}/reduced_rev02/extractions/*sky1d_extractions.pkl'))

def exkurt(x, p):
    wgt = np.clip(p, 0.0, None); Sw = wgt.sum()
    if Sw <= 0: return np.nan
    mu = (wgt * x).sum() / Sw; d = x - mu; m2 = (wgt * d ** 2).sum() / Sw
    return (wgt * d ** 4).sum() / Sw / m2 ** 2 - 3.0 if m2 > 0 else np.nan

acc = {}
for fr in frames:
    d = ExtractLlamas.loadExtraction(fr)
    for i, m in enumerate(d['metadata']):
        if str(m.get('channel')).lower() != 'green': continue
        e = d['extractions'][i]; cam = f"{m.get('bench')}{m.get('side')}"
        C = np.asarray(e.counts, float); WV = np.asarray(e.wave, float); XS = np.asarray(e.xshift, float)
        S = np.asarray(e.sky, float); kv = np.full(C.shape[0], np.nan)
        for fb in range(C.shape[0]):
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

def binned(a, b=8):
    n = (a.size // b) * b
    return np.nanmean(a[:n].reshape(-1, b), 1)

def lag1(a):
    a = np.asarray(a, float); m = np.isfinite(a[:-1]) & np.isfinite(a[1:])
    return np.corrcoef(a[:-1][m], a[1:][m])[0, 1] if m.sum() >= 5 else np.nan

save = {}
print('cam    poly4 R^2   binned(8) lag1   raw lag1   noise σ (frame-to-frame)   trend range')
polyr, blag = [], []
for cam in sorted(acc):
    K = np.vstack(acc[cam]); mk = np.nanmean(K, 0); save[cam] = mk
    fi = np.arange(mk.size); g = np.isfinite(mk)
    c = np.polyfit(fi[g], mk[g], 4); yh = np.polyval(c, fi[g])
    r2 = 1 - np.nansum((mk[g] - yh) ** 2) / np.nansum((mk[g] - np.nanmean(mk[g])) ** 2)
    bl = lag1(binned(mk)); rl = lag1(mk)
    noise = np.nanmedian(np.nanstd(K, 0) / np.sqrt(np.sum(np.isfinite(K), 0).clip(1)))
    trng = np.nanpercentile(np.polyval(c, fi[g]), 95) - np.nanpercentile(np.polyval(c, fi[g]), 5)
    polyr.append(r2); blag.append(bl)
    print(f'{cam}    {r2:+.2f}        {bl:+.2f}            {rl:+.2f}       {noise:.3f}                    {trng:.2f}')
print(f'\nmedian across cameras:  poly4 R^2={np.nanmedian(polyr):+.2f}   binned lag1={np.nanmedian(blag):+.2f}')
np.savez(f'{OUT}/kurt_slit_mk.npz', **save)
print('saved per-camera frame-averaged kurtosis profiles ->', f'{OUT}/kurt_slit_mk.npz')
print('\npoly4 R^2 = smooth fraction of the trend; binned lag1 -> ~1 = fibre-to-fibre coherence once')
print('per-fibre noise is averaged down (confirms the underlying variation is smooth, not random).')
