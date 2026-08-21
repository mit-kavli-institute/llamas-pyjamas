"""Robustness + red extension for the skew/kurtosis -> ring test. Adds: Spearman (robust to the
nonlinearity), an amp (S/N) confound check, a kurtosis-deviation-quartile binned ring table (the clean
summary of the mechanism), and the red channel. Read-only on the ring-test pkls."""
import sys, glob, warnings; warnings.filterwarnings('ignore')
import os as _os  # repo root = three levels up from this diagnosis script
sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..')))
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

ND = '/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17'; OUT = _os.environ.get('LLAMAS_DIAG_OUT', _os.path.join(_os.path.dirname(__file__), 'figures'))
W, MW, AMP_MIN = 8, 5, 1500.0
LINES = {'green': {'5577': 5577.34, '6300': 6300.30},
         'red':   {'6300': 6300.30, '6863': 6863.96, '7276': 7276.41, '7340': 7340.89,
                   '7750': 7750.65, '7913': 7913.71, '7993': 7993.33, '8399': 8399.17}}
frames = sorted(glob.glob(f'{ND}/reduced_ringtest/extractions/*sky1d_extractions.pkl'))

def moments(x, p):
    wgt = np.clip(p, 0.0, None); Sw = wgt.sum()
    if Sw <= 0: return np.nan, np.nan
    mu = (wgt * x).sum() / Sw; d = x - mu
    m2 = (wgt * d ** 2).sum() / Sw
    if m2 <= 0: return np.nan, np.nan
    return (wgt * d ** 3).sum() / Sw / m2 ** 1.5, (wgt * d ** 4).sum() / Sw / m2 ** 2 - 3.0

def collect(channel, lines):
    rows = {k: [] for k in lines}
    for fr in frames:
        d = ExtractLlamas.loadExtraction(fr)
        for i, e in enumerate(d['extractions']):
            if str(d['metadata'][i].get('channel')).lower() != channel: continue
            C = np.asarray(e.counts, float); S = np.asarray(e.sky, float)
            WV = np.asarray(e.wave, float); XS = np.asarray(e.xshift, float); nf = C.shape[0]
            for fb in range(nf):
                wv = WV[fb]; c = C[fb]; s = S[fb]; x = XS[fb]
                for name, lam in lines.items():
                    sel = np.where(np.abs(wv - lam) < 8)[0]
                    if sel.size < 3: continue
                    pk = sel[int(np.nanargmax(s[sel]))]
                    if pk - W < 0 or pk + W >= wv.size: continue
                    idx = np.arange(pk - W, pk + W + 1)
                    xx = x[idx]; cc = c[idx]; rr = (c - s)[idx]
                    if not (np.all(np.isfinite(xx)) and np.all(np.isfinite(cc)) and np.all(np.isfinite(rr))): continue
                    P = cc - np.polyval(np.polyfit([xx[0], xx[-1]], [np.median(cc[:2]), np.median(cc[-2:])], 1), xx)
                    amp = np.nanmax(P)
                    if amp < AMP_MIN: continue
                    mi = np.arange(W - MW, W + MW + 1)
                    sk, exk = moments(xx[mi] - xx[W], P[mi])
                    if not np.isfinite(sk): continue
                    ring = (rr[W] - np.mean([rr[W - 2], rr[W + 2], rr[W - 3], rr[W + 3]])) / amp
                    rows[name].append((fb / max(1, nf - 1), amp, sk, exk, ring))
    return rows

def sp(a, b):
    a = np.asarray(a); b = np.asarray(b); m = np.isfinite(a) & np.isfinite(b)
    return spearmanr(a[m], b[m]).correlation if m.sum() >= 8 else np.nan

print('chan line   N     skew(μ±σ)      exkurt(μ±σ)    Spear(ring,exk-μ) Spear(ring,skew) '
      'confound Spear(ring,amp)   ring by exkurt-dev quartile [Q1flat..Q4sharp]')
for chan, lines in LINES.items():
    rows = collect(chan, lines)
    for name in lines:
        A = np.array(rows[name])
        if len(A) < 20:
            print(f'{chan:5s} {name}  {len(A):4d}   (too few)'); continue
        slit, amp, skew, exk, ring = A.T
        exk_dev = exk - np.nanmedian(exk)
        q = np.percentile(exk_dev, [25, 50, 75])
        binq = [np.nanmedian(ring[exk_dev <= q[0]]),
                np.nanmedian(ring[(exk_dev > q[0]) & (exk_dev <= q[1])]),
                np.nanmedian(ring[(exk_dev > q[1]) & (exk_dev <= q[2])]),
                np.nanmedian(ring[exk_dev > q[2]])]
        print(f'{chan:5s} {name}  {len(A):4d}   {np.nanmean(skew):+.2f}±{np.nanstd(skew):.2f}   '
              f'{np.nanmean(exk):+.2f}±{np.nanstd(exk):.2f}   {sp(ring, exk_dev):+.2f}             '
              f'{sp(ring, skew):+.2f}            {sp(ring, amp):+.2f}            '
              f'[{binq[0]:+.3f} {binq[1]:+.3f} {binq[2]:+.3f} {binq[3]:+.3f}]')
print('\nQuartile ring should climb monotonically flat->sharp if kurtosis-mismatch drives the ring;')
print('Spear(ring,amp) near 0 rules out an S/N/brightness confound.')
