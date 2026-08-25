"""Money plot: for green 5577/6300, bin fibres by line-kurtosis deviation and plot the MEDIAN residual
profile (counts-sky)/amp vs px-from-peak per bin. If kurtosis-mismatch drives the ring, flat-line
fibres and sharp-line fibres should show OPPOSITE-sign peak residuals. Read-only, ring-test green."""
import sys, glob, warnings; warnings.filterwarnings('ignore')
import os as _os  # repo root = three levels up from this diagnosis script
sys.path.insert(0, _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..', '..', '..')))
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
ND = '/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17'; OUT = _os.environ.get('LLAMAS_DIAG_OUT', _os.path.join(_os.path.dirname(__file__), 'figures'))
W, MW, AMP_MIN = 8, 5, 1500.0
LINES = {'5577': 5577.34, '6300': 6300.30}
frames = sorted(glob.glob(f'{ND}/reduced_ringtest/extractions/*sky1d_extractions.pkl'))

def moments(x, p):
    wgt = np.clip(p, 0.0, None); Sw = wgt.sum()
    if Sw <= 0: return np.nan
    mu = (wgt * x).sum() / Sw; d = x - mu; m2 = (wgt * d ** 2).sum() / Sw
    return (wgt * d ** 4).sum() / Sw / m2 ** 2 - 3.0 if m2 > 0 else np.nan

prof = {k: [] for k in LINES}; exks = {k: [] for k in LINES}
for fr in frames:
    d = ExtractLlamas.loadExtraction(fr)
    for i, e in enumerate(d['extractions']):
        if str(d['metadata'][i].get('channel')).lower() != 'green': continue
        C = np.asarray(e.counts, float); S = np.asarray(e.sky, float)
        WV = np.asarray(e.wave, float); XS = np.asarray(e.xshift, float)
        for fb in range(C.shape[0]):
            wv = WV[fb]; c = C[fb]; s = S[fb]
            for name, lam in LINES.items():
                sel = np.where(np.abs(wv - lam) < 8)[0]
                if sel.size < 3: continue
                pk = sel[int(np.nanargmax(s[sel]))]
                if pk - W < 0 or pk + W >= wv.size: continue
                idx = np.arange(pk - W, pk + W + 1); xx = XS[fb][idx]; cc = c[idx]; rr = (c - s)[idx]
                if not (np.all(np.isfinite(xx)) and np.all(np.isfinite(cc)) and np.all(np.isfinite(rr))): continue
                P = cc - np.polyval(np.polyfit([xx[0], xx[-1]], [np.median(cc[:2]), np.median(cc[-2:])], 1), xx)
                amp = np.nanmax(P)
                if amp < AMP_MIN: continue
                mi = np.arange(W - MW, W + MW + 1); exk = moments(xx[mi] - xx[W], P[mi])
                if not np.isfinite(exk): continue
                prof[name].append(rr / amp); exks[name].append(exk)

fig, ax = plt.subplots(1, 2, figsize=(13, 5)); px = np.arange(-W, W + 1)
for j, name in enumerate(LINES):
    Pr = np.array(prof[name]); ek = np.array(exks[name]); dev = ek - np.median(ek)
    q = np.percentile(dev, [25, 75])
    flat = dev <= q[0]; sharp = dev > q[1]; mid = (dev > q[0]) & (dev <= q[1])
    ax[j].plot(px, np.nanmedian(Pr[flat], 0), '-o', ms=3, color='C0', label=f'flattest 25% (exk-μ<{q[0]:+.2f}), N={flat.sum()}')
    ax[j].plot(px, np.nanmedian(Pr[mid], 0), '-', color='0.6', label=f'middle 50%')
    ax[j].plot(px, np.nanmedian(Pr[sharp], 0), '-o', ms=3, color='C3', label=f'sharpest 25% (exk-μ>{q[1]:+.2f}), N={sharp.sum()}')
    ax[j].axhline(0, color='k', lw=0.5); ax[j].axvline(0, color='k', lw=0.3)
    ax[j].set_title(f'green {name}: median residual/amp by line kurtosis'); ax[j].set_xlabel('px from line peak')
    ax[j].set_ylabel('(counts - base_sky) / amp'); ax[j].legend(fontsize=8)
fig.tight_layout(); pth = f'{OUT}/ring_shape_bins.png'; fig.savefig(pth, dpi=110); print('FIG', pth)
# numeric: peak residual for flat vs sharp
for name in LINES:
    Pr = np.array(prof[name]); ek = np.array(exks[name]); dev = ek - np.median(ek)
    q = np.percentile(dev, [25, 75])
    print(f'{name}: peak resid/amp  flat={np.nanmedian(Pr[dev<=q[0]],0)[W]:+.3f}  '
          f'sharp={np.nanmedian(Pr[dev>q[1]],0)[W]:+.3f}')
