#!/usr/bin/env python
"""Interactive 2-panel sky-fit viewer for one fitting block (one camera).

Top:    throughput-corrected sky counts (counts/throughput) vs xshift, scatter over all sky fibres in
        the block, with the base bspline fit overplotted (= .sky/throughput, the actual pipeline fit).
        In xshift the sky lines of every fibre align on the horizontal axis.
Bottom: residual after sky subtraction, (counts - sky)/throughput vs xshift.
Panels share the x-axis, so zoom/pan (matplotlib toolbar) stays synced — zoom onto a line to see the ring.

NOISE FLOOR: the +/-noise envelope is computed LIVE from the sky model, NOT read from the pkl's
`errors` array. Those were written at extraction with the legacy aperture_pix=9 (a leftover 9-px
window) and over-count read noise by ~1.34x. Here the read-noise term uses the corrected method-aware
aperture (boxcar -> 2*halfwidth = 5 px) and the per-camera gain/RN from the lab table
(Config/detector_lab_props.csv via props_for_header) — i.e. the values the FIXED pipeline will write:
    sigma_counts = sqrt( sky_e + aperture_pix * RN^2 ) / gain ,  sky_e = clip(sky,0) * gain
This matches the corrected ERROR extension so residual scatter can be judged against the true floor.

Usage (run locally for the interactive window):
    python red_skyfit_viewer.py [--cam 1A] [--channel red] [--pkl PATH] [--all] [--save out.png]
                                [--fiber 40,90,140 | --spread N] [--halfwidth 2.5] [--rn E] [--gain G]
"""
import argparse, glob, os, sys, warnings
warnings.filterwarnings('ignore')
import numpy as np
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, DEFAULT_GAIN, DEFAULT_READNOISE
from llamas_pyjamas.Utils.detectorProps import props_for_header

# effective_aperture_pix lives on the error-model fix branch; fall back to a local copy so this
# diagnostic works whether or not the installed checkout carries it yet (keep in sync with
# extractLlamas.effective_aperture_pix).
try:
    from llamas_pyjamas.Extract.extractLlamas import effective_aperture_pix
except ImportError:
    def effective_aperture_pix(method='boxcar', boxcar_halfwidth=2.5, trace=None):
        method = str(method).lower(); half = float(boxcar_halfwidth)
        if method in ('optimal', 'horne'):
            ea = getattr(trace, 'extraction_aperture', None)
            return float(ea) if ea else 2.0 * half
        return 2.0 * half

ND = '/Users/simcoe/DATA/LLAMAS/may26/ut20260516_17'
ap = argparse.ArgumentParser()
ap.add_argument('--cam', default='1A'); ap.add_argument('--channel', default='red')
ap.add_argument('--pkl', default=None); ap.add_argument('--all', action='store_true',
                help='show all fibres (default: only the faint sky fibres = the fit block)')
ap.add_argument('--save', default=None)
ap.add_argument('--backend', default=None, help='force a matplotlib backend, e.g. TkAgg or macosx')
ap.add_argument('--fiber', default=None, help='per-fibre view: comma-separated fibre indices, e.g. 40,90,140')
ap.add_argument('--spread', type=int, default=0, help='per-fibre view of N sky fibres evenly across the slit')
ap.add_argument('--method', default='boxcar', help='extraction method for the RN aperture (boxcar/optimal)')
ap.add_argument('--halfwidth', type=float, default=2.5, help='boxcar halfwidth (px); RN aperture = 2*halfwidth')
ap.add_argument('--rn', type=float, default=None, help='override read noise (e-); default = lab table per camera')
ap.add_argument('--gain', type=float, default=None, help='override gain (e-/ADU); default = lab table per camera')
a = ap.parse_args()

import matplotlib
if a.save:
    matplotlib.use('Agg')
else:
    # Force an INTERACTIVE backend — some launch contexts default to Agg (whose show() is a
    # silent no-op, i.e. "started but no window"). macOS native first, then TkAgg (tkinter present).
    for _bk in ([a.backend] if a.backend else []) + ['macosx', 'TkAgg', 'QtAgg']:
        try:
            matplotlib.use(_bk, force=True); break
        except Exception:
            continue
import matplotlib.pyplot as plt
if not a.save:
    print(f'matplotlib backend = {matplotlib.get_backend()} (interactive={matplotlib.is_interactive()})')

pkl = a.pkl or sorted(glob.glob(f'{ND}/reduced_ringtest/extractions/*sky1d_extractions.pkl'))[0]
d = ExtractLlamas.loadExtraction(pkl); md = d['metadata']
idx = [i for i, m in enumerate(md) if str(m.get('channel')).lower() == a.channel.lower()
       and f"{m.get('bench')}{m.get('side')}".upper() == a.cam.upper()]
if not idx:
    raise SystemExit(f'camera {a.channel} {a.cam} not found in {os.path.basename(pkl)}')
e = d['extractions'][idx[0]]
C = np.asarray(e.counts, float); X = np.asarray(e.xshift, float); S = np.asarray(e.sky, float)
tp = np.asarray(getattr(e, 'relative_throughput', np.ones(C.shape[0])), float)
nf = C.shape[0]

# ---- CORRECTED per-camera noise model (not the stale pkl `errors`) ----
_hdr = getattr(e, 'hdr', None)
_g, _rn, _src = props_for_header(_hdr, DEFAULT_GAIN, DEFAULT_READNOISE)
GAIN = float(a.gain) if a.gain is not None else float(_g)
RN = float(a.rn) if a.rn is not None else float(_rn)
APER = effective_aperture_pix(a.method, a.halfwidth)     # boxcar -> 2*halfwidth = 5 px
NOISE_LABEL = f'sky·g + {APER:.0f}·RN²  (g={GAIN:.2f}, RN={RN:.2f} e-, src={_src})'
print(f'noise model: aperture={APER:.1f} px, gain={GAIN:.3f} e-/ADU, RN={RN:.2f} e- '
      f'(source={_src}); read-noise floor = sqrt({APER:.0f})·RN/gain = {np.sqrt(APER)*RN/GAIN:.2f} counts')

def noise_floor(sky_counts):
    """Corrected per-pixel 1-sigma noise in COUNTS from the sky model: sqrt(sky_e + aper·RN²)/gain."""
    sky_e = np.clip(np.asarray(sky_counts, float), 0.0, None) * GAIN
    return np.sqrt(sky_e + APER * RN ** 2) / GAIN

def robust_sigma(v):
    """IQR-based sigma (IQR/1.349) — outlier-robust RMS."""
    v = v[np.isfinite(v)]
    if v.size < 8:
        return np.nan
    q1, q3 = np.percentile(v, [25, 75])
    return (q3 - q1) / 1.349

# fit block = faint (sky) fibres, unless --all
bright = np.nanmedian(np.where(np.isfinite(C), C, np.nan), 1)
fin = np.isfinite(bright) & (bright != 0) & np.isfinite(tp) & (tp > 0)
if a.all:
    sel = fin
else:
    sel = fin & (bright <= np.nanpercentile(bright[fin], 60))
fibs = np.where(sel)[0]
if fibs.size == 0:
    raise SystemExit(f'{a.channel} {a.cam}: no usable fibres (dead/blank camera?) — nothing to plot.')

# ---- per-fibre photon test (raw counts; no throughput mixing) ----
per = None
if a.fiber:
    per = [int(x) for x in str(a.fiber).split(',') if x.strip() != '']
elif a.spread:
    per = list(fibs[np.linspace(0, len(fibs) - 1, min(a.spread, len(fibs))).astype(int)])
if per:
    nrow = len(per)
    fig, ax = plt.subplots(nrow, 2, sharex=True, figsize=(15, 2.4 * nrow), squeeze=False)
    for ri, fb in enumerate(per):
        w = X[fb]; c = C[fb]; s = S[fb]; t = float(tp[fb])
        m = np.isfinite(w) & np.isfinite(c) & np.isfinite(s)
        oo = np.argsort(w[m]); ws = w[m][oo]
        a0, a1 = ax[ri, 0], ax[ri, 1]
        a0.scatter(w[m], c[m], s=3, alpha=0.5, color='0.4'); a0.plot(ws, s[m][oo], '-', color='C3', lw=0.8)
        a0.set_ylabel(f'fib {fb}\ncounts')
        rr = (c - s)[m]                                   # residual (counts)
        floor = noise_floor(s[m])                         # corrected √(sky·g + aper·RN²)/g
        pois = np.sqrt(np.clip(s[m] * GAIN, 0, None)) / GAIN   # photon-only √(sky_e)/g
        a1.scatter(w[m], rr, s=3, alpha=0.5, color='C0')
        a1.plot(ws, floor[oo], '-', color='C1', lw=1.1, label=f'±noise √({NOISE_LABEL})')
        a1.plot(ws, -floor[oo], '-', color='C1', lw=1.1)
        a1.plot(ws, pois[oo], '--', color='C3', lw=0.8, alpha=0.7, label='±√sky (photon only)')
        a1.plot(ws, -pois[oo], '--', color='C3', lw=0.8, alpha=0.7)
        a1.axhline(0, color='k', lw=0.4)
        if ri == 0:
            a1.legend(fontsize=6, loc='upper right')
        cont = s[m] < np.nanpercentile(s[m], 40)          # between-line (low-sky) pixels
        sig = robust_sigma(rr[cont])                      # robust residual sigma (IQR)
        r_flr = sig / max(np.nanmedian(floor[cont]), 1e-6)
        r_pois = sig / max(np.nanmedian(pois[cont]), 1e-6)
        a1.set_title(f'fib {fb} tp={t:.2f}  between-line IQR-σ/noise={r_flr:.2f}  (/√sky={r_pois:.1f})', fontsize=8)
        a1.set_ylim(*np.nanpercentile(rr, [1, 99]))
    ax[-1, 0].set_xlabel('xshift (px)'); ax[-1, 1].set_xlabel('xshift (px)')
    fig.suptitle(f'{a.channel} {a.cam} per-fibre — counts+fit (left) | residual + corrected noise floor (right)   [{NOISE_LABEL}]')
    fig.tight_layout()
    if a.save:
        fig.savefig(a.save, dpi=110); print('wrote', a.save)
    else:
        print(f'backend={matplotlib.get_backend()}; per-fibre {a.channel} {a.cam}: {per}')
        plt.show()
    sys.exit(0)

def flat(arr):
    return arr[fibs].ravel()
xs = flat(X); ct = (C[fibs] / tp[fibs, None]).ravel(); sk = (S[fibs] / tp[fibs, None]).ravel()
tp_pt = (np.ones((len(fibs), C.shape[1])) * tp[fibs, None]).ravel()   # per-point throughput
ok = np.isfinite(xs) & np.isfinite(ct) & np.isfinite(sk) & np.isfinite(tp_pt) & (tp_pt > 0)
xs, ct, sk, tp_pt = xs[ok], ct[ok], sk[ok], tp_pt[ok]
res = ct - sk
o = np.argsort(xs)                                        # for the fit-curve line
# Poisson expectation on the /throughput residual. sky per fibre (counts) = sk*tp; photon sigma
# (counts) = sqrt(sky_e)/g = sqrt(sk*tp*g)/g; the residual is /tp -> divide by tp.
sky_fibre = sk * tp_pt                                    # sky in counts, per point
exp_sig = np.sqrt(np.clip(sky_fibre * GAIN, 0, None)) / GAIN / tp_pt          # photon only, /tp
floor_pt = noise_floor(sky_fibre) / tp_pt                                     # corrected √(sky·g+aper·RN²)/g, /tp
from scipy.stats import binned_statistic
nb = max(200, int((xs.max() - xs.min()) / 1.0))          # ~1-px bins
med_exp, edges, _ = binned_statistic(xs, exp_sig, statistic='median', bins=nb)
med_flr, _, _ = binned_statistic(xs, floor_pt, statistic='median', bins=nb)
rob_res, _, _ = binned_statistic(xs, res, statistic=robust_sigma, bins=nb)  # robust (IQR) residual sigma
bc = 0.5 * (edges[:-1] + edges[1:])

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14, 9))
ax1.scatter(xs, ct, s=2, alpha=0.18, color='0.4', rasterized=True, label='sky-fibre counts / throughput')
ax1.plot(xs[o], sk[o], '-', color='C3', lw=0.8, label='base bspline fit (.sky/throughput)')
ax1.set_ylabel('counts / throughput'); ax1.legend(loc='upper right')
ax1.set_title(f'{a.channel} {a.cam} — {len(fibs)} {"" if a.all else "sky "}fibres — {os.path.basename(pkl)}')
lo, hi = np.nanpercentile(ct, [1, 99.5]); ax1.set_ylim(lo - 0.05 * (hi - lo), hi + 0.1 * (hi - lo))
ax2.scatter(xs, res, s=2, alpha=0.18, color='C0', rasterized=True)
ax2.plot(bc, med_flr, '-', color='C1', lw=1.3, label=f'±noise floor √({NOISE_LABEL}) /tp')
ax2.plot(bc, -med_flr, '-', color='C1', lw=1.3)
ax2.plot(bc, med_exp, '--', color='C3', lw=0.8, alpha=0.7, label='±√sky (photon only) /tp')
ax2.plot(bc, -med_exp, '--', color='C3', lw=0.8, alpha=0.7)
ax2.plot(bc, rob_res, '-', color='C2', lw=1.0, alpha=0.9, label='residual IQR-σ')
ax2.plot(bc, -rob_res, '-', color='C2', lw=1.0, alpha=0.9)
ax2.axhline(0, color='k', lw=0.5); ax2.set_ylabel('residual (counts - sky) / throughput'); ax2.set_xlabel('xshift (px)')
ax2.legend(loc='upper right')
rr = np.nanpercentile(np.abs(res), 99); ax2.set_ylim(-rr, rr)
fig.tight_layout()
if a.save:
    fig.savefig(a.save, dpi=110); print('wrote', a.save)
else:
    print(f'{a.channel} {a.cam}: {len(fibs)} fibres, {xs.size} points. Zoom onto a sky line to inspect the ring.')
    plt.show()
