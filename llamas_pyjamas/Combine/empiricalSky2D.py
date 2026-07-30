"""
llamas_pyjamas.Combine.empiricalSky2D
=====================================
Empirical multi-dither sky subtraction on the BIAS+FLAT-CORRECTED 2D DETECTOR FRAMES (pre-extraction).

Rationale (RS): the post-extraction b-spline model cannot see the sky's true 2D optical footprint —
per-fibre LSF/optics aberrations (worst in the red) and inter-fibre scattered light. Subtracting the
sky on the raw 2D frame, from the dithers where each fibre lands on BLANK sky, matches that footprint
pixel-for-pixel (same detector pixel = same optics), then the frame is re-extracted. OH airglow varies
between dithers, so a residual remains after the overall per-frame sky-level scaling — that residual is
small vs the total sky and is cleaned afterward by the existing b-spline/PCA sky stage on the
re-extracted RSS.

Assumptions:
  * flexure negligible (instrument flexure-compensation) -> the per-dither 2D frames of a field align
    pixel-for-pixel, and a fixed trace/fiberimg maps pixels -> fibres in every dither.
  * blank-vs-source per fibre is decided by self-reference from a light trace-collapse of the 2D frame
    (median of the fibre's own pixels); the faintest dithers = on blank sky.

Per detector (channel,bench,side): flip into trace space, blank-select per fibre, build the per-pixel
leave-one-out median of the blank dithers (scaled to each target frame's overall sky level; inter-fibre
GAP pixels are sky in every dither), subtract, un-flip, write a sky-subtracted 2D MEF for re-extraction.
"""
import logging
import os
import pickle
import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

MIN_BLANK = 3
BLANK_FRAC = 0.6
FLIP = {'green': 'lr', 'blue': 'udlr', 'red': 'none'}


def _apply_flip(a, color):
    f = FLIP.get(str(color).lower(), 'none')
    if f == 'lr':
        return np.fliplr(a)
    if f == 'udlr':
        return np.flipud(np.fliplr(a))
    return a


def _load_trace(trace_dir, color, bench, side):
    p = os.path.join(trace_dir, f"LLAMAS_{color}_{bench}_{side}_traces.pkl")
    if not os.path.exists(p):
        return None
    return pickle.load(open(p, "rb"))


def build_camera_2d_sky(stack, fiberimg, min_blank=MIN_BLANK, blank_frac=BLANK_FRAC):
    """Per-pixel empirical 2D sky for one detector across a field's dithers (all in trace space).

    stack : (ndith, ny, nx) bias+flat-corrected 2D frames (flexure-aligned).
    fiberimg : (ny, nx) fibre id per pixel (>=0), -1 in inter-fibre gaps.

    Returns emp_sky (ndith, ny, nx) (leave-one-out per target frame), nblank (nfib,), S (ndith,).
    """
    stack = np.asarray(stack, float)
    ndith, ny, nx = stack.shape
    nfib = int(fiberimg.max()) + 1

    # per-(dither,fibre) continuum from a trace-collapse (blank selection self-reference)
    cont = np.full((ndith, nfib), np.nan)
    for f in range(nfib):
        ys, xs = np.where(fiberimg == f)
        if xs.size == 0:
            continue
        vals = stack[:, ys, xs]                             # (ndith, npix_f)
        cont[:, f] = np.nanmedian(vals, axis=1)
    # per-frame overall sky level (median over fibres); robust to a source-minority IFU
    S = np.nanmedian(cont, axis=1)
    S_safe = np.where(np.isfinite(S) & (S > 0), S, np.nan)

    # blank dithers per fibre = faintest n_blank_target
    n_blank_target = min(max(int(min_blank), int(round(blank_frac * ndith))), ndith)
    blank_of = {}                                          # fibre -> list of blank dither indices
    nblank = np.zeros(nfib, dtype=int)
    for f in range(nfib):
        bi = cont[:, f]
        order = np.argsort(np.where(np.isfinite(bi), bi, np.inf))
        blank = [d for d in order[:n_blank_target] if np.isfinite(bi[d]) and np.isfinite(S_safe[d])]
        if len(blank) >= min_blank:
            blank_of[f] = blank
            nblank[f] = len(blank)

    # per-dither blank map projected to pixels (gaps = sky in every dither)
    all_d = [d for d in range(ndith) if np.isfinite(S_safe[d])]
    blankimg = np.zeros((ndith, ny, nx), bool)
    gap = fiberimg < 0
    fid = np.where(gap, 0, fiberimg)
    for d in range(ndith):
        # fibre blank in dither d?
        fibre_blank = np.zeros(nfib, bool)
        for f, bl in blank_of.items():
            fibre_blank[f] = d in bl
        bm = fibre_blank[fid]
        bm[gap] = np.isfinite(S_safe[d])                   # gaps: always sky (when frame usable)
        blankimg[d] = bm

    # DUAL per-frame scaling (RS): the empirical sky is right up to a per-frame amplitude, but that
    # amplitude differs for the OH lines (airglow ~low-dimensional in intensity, ~one brightness scale)
    # vs the less-variable inter-line continuum. So build a leave-one-out sky SHAPE (native-level median
    # of the blank dithers) and fit TWO amplitudes to the target frame's own blank pixels: a_oh on OH
    # pixels, a_cont on continuum pixels. Collapses the OH-variability residual the single scale left.
    OH_FACTOR = 3.0
    emp_sky = np.full_like(stack, np.nan)
    amps = {}
    acc = np.empty((ndith, ny, nx), np.float32)

    def _amp(shape, target, px):
        x = shape[px]; y = target[px]
        m = np.isfinite(x) & np.isfinite(y) & (x != 0)
        if m.sum() < 50:
            return np.nan
        a = float(np.nansum(x[m] * y[m]) / np.nansum(x[m] ** 2))
        return float(np.clip(a, 0.0, 5.0))

    for t in range(ndith):
        if not np.isfinite(S_safe[t]):
            continue
        acc[:] = np.nan
        for d in all_d:
            if d == t:
                continue
            acc[d] = np.where(blankimg[d], stack[d], np.nan)     # native level (no pre-scaling)
        shape = np.nanmedian(acc, axis=0)                         # LOO sky shape @ ~median blank level
        bt = blankimg[t] & np.isfinite(shape) & np.isfinite(stack[t])
        if bt.sum() < 500:                                        # too few blank px: single-scale fallback
            lvl = np.nanmedian(shape[bt]) if bt.any() else np.nan
            emp_sky[t] = shape * (S_safe[t] / lvl) if (lvl and lvl > 0) else shape
            continue
        cont_level = np.nanmedian(shape[bt])
        oh = shape > OH_FACTOR * cont_level                       # OH-line vs continuum pixels
        a_cont = _amp(shape, stack[t], bt & ~oh)
        a_oh = _amp(shape, stack[t], bt & oh)
        if not np.isfinite(a_cont):
            a_cont = 1.0
        if not np.isfinite(a_oh):
            a_oh = a_cont
        amps[t] = (round(a_cont, 3), round(a_oh, 3))
        emp_sky[t] = shape * np.where(oh, a_oh, a_cont)
    logger.info("empiricalSky2D: %d/%d fibres >=%d blank of %d; dual-scale amps (a_cont,a_oh)=%s",
                int((nblank >= min_blank).sum()), nfib, min_blank, ndith, amps)
    return emp_sky, nblank, S


def subtract_field_2d(frame_files, trace_dir, out_dir=None, suffix='_EMPSKY2D',
                      channels=('red', 'green', 'blue'), min_blank=MIN_BLANK, blank_frac=BLANK_FRAC):
    """Build + subtract the 2D empirical sky for a field. frame_files are the field's per-dither
    bias+flat-corrected 2D MEFs (same detector layout). Writes one sky-subtracted MEF per dither
    (SCI = 2D_frame - empirical_sky) for re-extraction. Non-destructive (new files). Returns paths."""
    out_dir = out_dir or os.path.dirname(frame_files[0])
    hduls = [fits.open(f) for f in frame_files]
    ndith = len(hduls)
    # output copies (start from the inputs; we overwrite each extension's data)
    outs = [h[0].header.copy() for h in hduls]
    # accumulate per-extension sky-subtracted data
    sub = [{} for _ in range(ndith)]                       # per dither: extindex -> 2D array
    n_ext = len(hduls[0])
    for i in range(1, n_ext):
        eh = hduls[0][i].header
        color = str(eh.get('COLOR', '')).strip().lower()
        bench = eh.get('BENCH'); side = str(eh.get('SIDE', '')).strip()
        if color not in channels:
            for d in range(ndith):
                sub[d][i] = np.asarray(hduls[d][i].data, float)
            continue
        trace = _load_trace(trace_dir, color, bench, side)
        if trace is None or trace.fiberimg is None:
            logger.warning("empiricalSky2D: no trace for %s_%s_%s; passthrough", color, bench, side)
            for d in range(ndith):
                sub[d][i] = np.asarray(hduls[d][i].data, float)
            continue
        # into trace space (fiberimg space)
        stack = np.stack([_apply_flip(np.asarray(hduls[d][i].data, float), color) for d in range(ndith)])
        emp, nblank, S = build_camera_2d_sky(stack, trace.fiberimg, min_blank, blank_frac)
        for d in range(ndith):
            sky = emp[d]
            skysub = np.where(np.isfinite(sky), stack[d] - sky, stack[d])   # fallback: unchanged where no sky
            sub[d][i] = _apply_flip(skysub, color)          # back to raw space
    written = []
    for d, f in enumerate(frame_files):
        hd = [fits.PrimaryHDU(header=outs[d])]
        for i in range(1, n_ext):
            src = hduls[d][i]
            hdu = fits.ImageHDU(sub[d][i].astype(np.float32), header=src.header.copy())
            hdu.header['EMPSKY2D'] = (True, 'empirical multi-dither 2D sky subtracted')
            hd.append(hdu)
        out = os.path.join(out_dir, os.path.basename(f).replace('.fits', f'{suffix}.fits'))
        fits.HDUList(hd).writeto(out, overwrite=True)
        written.append(out)
    for h in hduls:
        h.close()
    logger.info("empiricalSky2D: wrote %d sky-subtracted 2D frames", len(written))
    return written
