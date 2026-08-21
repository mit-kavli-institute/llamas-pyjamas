"""
llamas_pyjamas.Arc.arcSurface
=============================
Hybrid 2D xshift refinement (``refine_arc_method = 2d``).

The per-fibre global-quadratic xshift (shiftArcX) mis-tracks trace curvature
near the detector edges (fibre-to-fibre deviation rms ~0.7 px center vs
~2.4-3.1 px at the edge thirds on every camera), and the per-fibre cubic
refinement (refineArcX) is data-starved (~15 identified catalog lines per
fibre) and extrapolates noisily beyond the outermost matched line.

Physics: the camera optics impose a smooth, low-order curvature field shared
by all fibres of a detector, while slit assembly tolerances perturb each fibre
approximately rigidly (offset, possibly a small tilt) — not with arbitrary
curvature. This module therefore fits, per camera:

1. a robust 2D Legendre surface ``xshift_target(x, fibre_coord)`` to arc-line
   centroids pooled across ALL fibres — including catalog peaks without
   wavelength IDs, which constrain the alignment even though they play no role
   in the wavelength solution (~7-14x more data than refineArcX uses); then
2. a small per-fibre perturbation (offset by default), shrunk toward zero, fit
   to each fibre's residuals about the surface. Fibres with too few matched
   lines get the surface only; fibres whose quadratic failed entirely
   (identity xshift) inherit the surface — strictly better than ``xshift=x``.

Matching is two-pass: pass 1 uses the existing quadratic to predict line
positions for confidently-fittable fibres; pass 2 re-predicts every fibre's
line positions from the pass-1 surface (accurate at the edges and for identity
fibres, because the surface interpolates across fibres) and refits.

The output pickle (``*_refined2d.pkl``) carries updated ``.xshift`` and
``.wave`` arrays and is drop-in for arcTransfer, exactly like refineArcX's
product.

QA-record note: records appended to ``qa_collector`` use ``status='refined'``
whenever fit residuals exist (QA/waveQA.arc_residual_qa splits on that exact
string) with an auxiliary ``mode`` key ('surface+perturb' or 'surface_only').
Fibres that received the surface but matched zero lines get
``status='surface_no_matches'`` — they count as fallbacks in the QA scorecard
(no measured residuals) even though their xshift was improved.

Standalone usage (writes next to the input; symlink into scratch to keep
outputs out of LUT/):

    python -m llamas_pyjamas.Arc.arcSurface <arc.pkl> --qa-dir /path/QA
"""

import argparse
import os
import time
import warnings

import numpy as np
from astropy.table import Table

from llamas_pyjamas.config import LUT_DIR
import llamas_pyjamas.Extract.extractLlamas as extract
from llamas_pyjamas.Arc.arcLlamas import interpolateNaNs

from pypeit.core.wavecal.wvutils import arc_lines_from_spec, xcorr_shift_stretch
from pypeit.core.fitting import robust_fit

# Reference extensions per channel (fiber 150 of bench 4A), as in arcLlamas.py
ARC_REF_EXT = {'red': 18, 'green': 19, 'blue': 20}
REF_FIBER = 150


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_peak_catalog(channel, use_unidentified=True, blend_min_sep=8.0):
    """Load LUT/{channel}_peaks.csv.

    Returns an array of catalog peak positions (reference-fibre pixel frame).
    With ``use_unidentified=False`` only rows with a positive Wavelength are
    kept (the subset refineArcX uses); otherwise every detected peak counts —
    unidentified lines still constrain the xshift alignment.

    Blend filter: unidentified peaks whose nearest catalog neighbour is closer
    than ``blend_min_sep`` px (~2x the centroiding FWHM) are dropped — blended
    lines yield biased centroids regardless of how they are matched. Measured
    on the reference arc, blends inflated red's fit residuals from ~0.4 px to
    ~1.1 px. Identified (curated) lines are always kept.
    """
    tbl = Table.read(os.path.join(LUT_DIR, f'{channel}_peaks.csv'))
    if 'PeakHeight' in tbl.colnames and 'Height' not in tbl.colnames:
        tbl.rename_column('PeakHeight', 'Height')

    pixels = np.asarray(tbl['Pixel'], dtype=float)
    wl_col = tbl['Wavelength']
    if hasattr(wl_col, 'filled'):
        wavelength = np.asarray(wl_col.filled(np.nan), dtype=float)
    else:
        wavelength = np.array(
            [float(v) if str(v).strip() not in ('', '--', 'nan') else np.nan
             for v in wl_col])

    good = np.isfinite(pixels)
    identified = good & np.isfinite(wavelength) & (wavelength > 0)
    if not use_unidentified:
        return pixels[identified]

    # Distance from each peak to its nearest catalog neighbour (full list)
    p = pixels[good]
    nearest = np.full(p.size, np.inf)
    if p.size > 1:
        order = np.argsort(p)
        ps = p[order]
        gaps = np.diff(ps)
        near_sorted = np.minimum(np.append(gaps, np.inf),
                                 np.insert(gaps, 0, np.inf))
        nearest[order] = near_sorted
    isolated = nearest >= float(blend_min_sep)

    keep = identified[good] | isolated
    n_drop = int(np.sum(~keep))
    if n_drop:
        print(f"  {channel} catalog: dropped {n_drop} blended unidentified "
              f"peaks (< {blend_min_sep:.0f} px separation); "
              f"{int(keep.sum())} kept")
    return p[keep]


def _fiber_coords(extraction, nfibers, camlabel):
    """Continuous fibre-axis coordinate, normalized to [-1, 1].

    Prefers the physical slit coordinate — each fibre's trace centroid
    (detector y) at the centre column — over the bare fibre index, mirroring
    the fibreFlat surface-fit pattern. Falls back to fibre index when the
    trace is unavailable on the pickle.
    """
    fc_raw = None
    source = 'fiber_index'
    try:
        traces = extraction.trace.traces
        mid = traces.shape[1] // 2
        cand = np.asarray(traces[:nfibers, mid], dtype=float)
        if cand.shape[0] == nfibers and np.isfinite(cand).sum() >= 2:
            # Patch isolated NaNs by interpolation over fibre index
            idx = np.arange(nfibers, dtype=float)
            bad = ~np.isfinite(cand)
            if bad.any():
                cand[bad] = np.interp(idx[bad], idx[~bad], cand[~bad])
            if np.ptp(cand) > 0:
                fc_raw = cand
                source = 'trace_y'
    except Exception:
        pass
    if fc_raw is None:
        fc_raw = np.arange(nfibers, dtype=float)

    lo, hi = fc_raw.min(), fc_raw.max()
    span = hi - lo if hi > lo else 1.0
    fc = 2.0 * (fc_raw - lo) / span - 1.0
    print(f"  {camlabel}: fibre coordinate source = {source}")
    return fc


def _detect_lines(spec, sigdetect, fwhm):
    """Detect arc-line centroids on one fibre spectrum.

    Duplicated from refineArcX (arcLlamas.py detection block) so that function
    stays byte-identical; keep the two in sync deliberately.
    Returns (tcent, ecent) arrays or (None, None) on failure.
    """
    spec = interpolateNaNs(spec)
    spec = np.nan_to_num(spec, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="All pixels rejected")
            warnings.filterwarnings("ignore", message=".*invalid values.*")
            warnings.filterwarnings("ignore", category=UserWarning,
                                    module="astropy.stats")
            tcent, ecent, _, _, _ = arc_lines_from_spec(
                spec, sigdetect=sigdetect, fwhm=fwhm)
    except Exception:
        return None, None
    if len(tcent) == 0:
        return None, None
    return np.asarray(tcent, dtype=float), np.asarray(ecent, dtype=float)


def _match_lines(tcent, ecent, pred_pixels, targets, match_tol):
    """Match catalog lines to detected centroids with ambiguity rejection.

    For each catalog line (predicted at ``pred_pixels[j]``, with xshift target
    ``targets[j]``): accept the nearest centroid if it is within ``match_tol``
    AND the second-nearest centroid is farther than ``2*match_tol`` from the
    prediction (otherwise the pairing is confusable — critical for the red
    catalog, whose ~10 px mean line spacing rivals the tolerance). Assignments
    are then made one-to-one, keeping the closest claim per centroid.

    Returns (matched_pixels, matched_targets, matched_weights).
    """
    if tcent is None or len(tcent) == 0:
        return np.array([]), np.array([]), np.array([])

    claims = {}  # centroid index -> (distance, pixel, target, weight)
    for j in range(len(pred_pixels)):
        d = np.abs(tcent - pred_pixels[j])
        nearest = int(np.argmin(d))
        if d[nearest] >= match_tol:
            continue
        if len(d) > 1:
            second = np.partition(d, 1)[1]
            if second < 2.0 * match_tol:
                continue  # confusable — reject
        w = 1.0 / max(float(ecent[nearest]), 1e-4)
        prev = claims.get(nearest)
        if prev is None or d[nearest] < prev[0]:
            claims[nearest] = (float(d[nearest]), float(tcent[nearest]),
                               float(targets[j]), w)

    if not claims:
        return np.array([]), np.array([]), np.array([])
    vals = list(claims.values())
    mp = np.array([v[1] for v in vals])
    mt = np.array([v[2] for v in vals])
    mw = np.array([v[3] for v in vals])
    order = np.argsort(mp)
    return mp[order], mt[order], mw[order]


def _fit_surface(px, fc, target, w, order_x, order_fiber, naxis1):
    """Robust 2D Legendre surface fit xshift_target(x, fibre_coord).

    Explicit domain bounds keep the Legendre basis identical across passes and
    fibres regardless of where the samples happen to lie.
    """
    return robust_fit(px, target, order=(order_x, order_fiber), x2=fc,
                      function='legendre2d', weights=w,
                      lower=3, upper=3, maxiter=10,
                      minx=0.0, maxx=float(naxis1 - 1),
                      minx2=-1.0, maxx2=1.0)


def _soft_threshold(value, se, min_snr):
    """Significance-gate a fitted coefficient by soft-thresholding at min_snr*se.

    Genuine offsets (|value| well above their standard error se) survive, merely
    reduced by the threshold; offsets consistent with zero collapse to zero.
    With min_snr <= 0 this is a no-op (returns value unchanged). This is what
    stops the per-fibre perturbation from injecting fibre-to-fibre noise when the
    matched lines don't actually constrain a real slit-tolerance offset.
    """
    if min_snr <= 0 or not np.isfinite(se) or se <= 0:
        return value
    thr = float(min_snr) * float(se)
    return float(np.sign(value) * max(0.0, abs(value) - thr))


def _fit_perturbation(px, resid, w, order, shrink_lines, naxis1, min_snr=0.0):
    """Small per-fibre correction about the surface, shrunk toward zero.

    order 0: weighted-mean offset with a zero-prior worth ``shrink_lines``
    median-weight lines. order 1: ridge-regularized offset + tilt. Each fitted
    coefficient is then significance-gated (soft-thresholded at ``min_snr`` times
    its standard error) so noise-level offsets do not perturb the fibre.
    Returns a callable evaluating the perturbation on a pixel grid.
    """
    if px.size == 0:
        return lambda xg: np.zeros_like(np.asarray(xg, dtype=float))
    lam = float(shrink_lines) * float(np.median(w))
    sw = float(np.sum(w))
    # Effective sample size and weighted residual scatter -> coefficient errors.
    n_eff = (sw ** 2) / float(np.sum(w ** 2)) if np.sum(w ** 2) > 0 else px.size
    if order <= 0:
        offset = float(np.sum(w * resid) / (sw + lam))
        wvar = float(np.sum(w * (resid - offset) ** 2) / sw) if sw > 0 else 0.0
        se = np.sqrt(max(wvar, 0.0) / max(n_eff, 1.0))
        offset = _soft_threshold(offset, se, min_snr)
        return lambda xg: np.full(np.asarray(xg, dtype=float).shape, offset)
    # order 1: ridge on [1, xn]
    half = (naxis1 - 1) / 2.0
    xn = px / half - 1.0
    A = np.column_stack([np.ones_like(xn), xn])
    W = np.diag(w)
    lhs = A.T @ W @ A + lam * np.eye(2)
    rhs = A.T @ (w * resid)
    try:
        c = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        c = np.zeros(2)
    if min_snr > 0:
        model = c[0] + c[1] * xn
        wvar = float(np.sum(w * (resid - model) ** 2) / sw) if sw > 0 else 0.0
        try:
            cov = np.linalg.inv(lhs) * wvar
            se = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            se = np.zeros(2)
        c = np.array([_soft_threshold(c[0], se[0], min_snr),
                      _soft_threshold(c[1], se[1], min_snr)])
    return lambda xg: c[0] + c[1] * (np.asarray(xg, dtype=float) / half - 1.0)


def _monotonic(arr):
    return bool(np.all(np.diff(arr) > 0))


def _cluster_consensus(vals, cams, tol, min_cams):
    """Group common-frame line positions and keep those seen in enough cameras.

    ``vals`` are candidate line xshifts pooled across cameras, ``cams`` the
    parallel camera index of each. Positions within ``tol`` form a cluster; a
    cluster confirmed by at least ``min_cams`` distinct cameras is a real line
    (noise does not repeat coherently across cameras) and its median position is
    returned. This is how blue-end lines missing from the catalog get harvested
    without any wavelength ID.
    """
    vals = np.asarray(vals, float)
    cams = np.asarray(cams)
    if vals.size == 0:
        return np.array([])
    order = np.argsort(vals)
    vals, cams = vals[order], cams[order]
    splits = np.where(np.diff(vals) > tol)[0] + 1
    out = []
    for g in np.split(np.arange(vals.size), splits):
        if len(set(cams[g].tolist())) >= min_cams:
            out.append(float(np.median(vals[g])))
    return np.array(out)


def _apply_edge_extrap(vals, x, x_lo, x_hi, mode='linear', slope_win=50):
    """Replace the surface beyond the constrained range with a controlled tail.

    Past the outermost matched arc line the Legendre surface is unconstrained and
    can swing several pixels (measured 2-6 px on blue), which misaligns the sky
    model at the wavelength extremes and produces broad P-Cygni residuals. This
    continues each fibre's xshift with a C1 linear (or quadratic) extension using
    the local slope/curvature at the boundary, so every fibre extrapolates the
    same smooth, controlled way. ``mode='none'`` disables (raw polynomial tail).
    """
    if mode == 'none':
        return vals
    out = vals.copy()
    n = len(x)
    hi = int(np.clip(np.floor(x_hi), 0, n - 1))
    lo = int(np.clip(np.ceil(x_lo), 0, n - 1))
    # High-x (long-wavelength / "red") end
    if hi < n - 1 and hi - slope_win >= 0:
        if mode == 'quadratic':
            c = np.polyfit(x[hi - slope_win:hi + 1], vals[hi - slope_win:hi + 1], 2)
            out[hi + 1:] = np.polyval(c, x[hi + 1:])
        else:
            slope = (vals[hi] - vals[hi - slope_win]) / (x[hi] - x[hi - slope_win])
            out[hi + 1:] = vals[hi] + slope * (x[hi + 1:] - x[hi])
    # Low-x (short-wavelength / "blue") end
    if lo > 0 and lo + slope_win < n:
        if mode == 'quadratic':
            c = np.polyfit(x[lo:lo + slope_win + 1], vals[lo:lo + slope_win + 1], 2)
            out[:lo] = np.polyval(c, x[:lo])
        else:
            slope = (vals[lo + slope_win] - vals[lo]) / (x[lo + slope_win] - x[lo])
            out[:lo] = vals[lo] + slope * (x[:lo] - x[lo])
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def refineArcX2D(arc_extraction_shifted_pickle, channels=None, qa_collector=None,
                 surface_order_x=3, surface_order_fiber=2,
                 perturb_order=0, perturb_min_lines=8, perturb_shrink_lines=5.0,
                 perturb_min_snr=0.0,
                 use_unidentified_peaks=True, blend_min_sep=8.0,
                 match_tol_pass1=4.0, match_tol_pass2=4.0,
                 sigdetect=5.0, fwhm=4.0, line_source=None, detection_cache=None,
                 edge_extrap='linear', harvest_lines=True, harvest_sigdetect=3.0,
                 harvest_min_cameras=3, harvest_tol=2.0):
    """Hybrid per-camera 2D surface + per-fibre perturbation xshift refinement.

    Saves ``<input>_refined2d.pkl`` (same conventions as refineArcX) and
    returns its path. See module docstring for the algorithm and QA-record
    semantics. The default pipeline never calls this: it is opt-in via
    ``refine_arc_method = 2d``.

    line_source (dict, optional): ``{channel: extraction pickle path or dict}``
        of NIGHT-OF arc exposures (e.g. the afternoon ThAr set, extracted with
        the night's traces). When given, line centroids are detected on THOSE
        spectra — so the refined solution aligns to the instrument's state at
        the observation epoch — while the reference pickle still provides the
        line identifications and the xshift->wavelength mapping (defined in
        xshift space, epoch-independent). Each camera is bootstrapped by
        cross-correlating the night arc's reference-fibre spectrum against the
        packaged arc's, so multi-pixel epoch offsets cannot defeat pass-1
        matching. Without line_source, centroids come from the reference
        pickle's own (old-epoch) spectra as before.
    """
    # Accept either a pickle path or an already-loaded arcdict. The dict form
    # (operated on in place, returned instead of saved) lets callers run many
    # refinements without disk I/O — e.g. a parameter sweep.
    if isinstance(arc_extraction_shifted_pickle, dict):
        arcdict = arc_extraction_shifted_pickle
        _save_result = False
    else:
        arcdict = extract.ExtractLlamas.loadExtraction(arc_extraction_shifted_pickle)
        _save_result = True
    arcspec = arcdict['extractions']
    metadata = arcdict['metadata']

    def _qa(channel, bench, side, fiber, status, n_detected=0, n_matched=0,
            pixels=None, resid_pix=None, mode=''):
        if qa_collector is not None:
            qa_collector.append({
                'channel': channel, 'bench': bench, 'side': side,
                'fiber': fiber, 'status': status, 'mode': mode,
                'n_detected': int(n_detected), 'n_matched': int(n_matched),
                'pixels': (np.asarray(pixels, dtype=float)
                           if pixels is not None else np.array([])),
                'resid_pix': (np.asarray(resid_pix, dtype=float)
                              if resid_pix is not None else np.array([])),
            })

    for channel in (channels if channels is not None else ['red', 'green', 'blue']):
        ref_ext = ARC_REF_EXT[channel]
        naxis1 = arcspec[ref_ext].xshift.shape[1]
        x = np.arange(naxis1, dtype=float)

        catalog_pixels = _load_peak_catalog(channel, use_unidentified_peaks,
                                            blend_min_sep=blend_min_sep)
        ref_xshift = arcspec[ref_ext].xshift[REF_FIBER, :]
        ref_wave = arcspec[ref_ext].wave[REF_FIBER, :]
        # Catalog positions expressed in the reference xshift frame (mirrors
        # refineArcX's ref_line_xshift construction).
        ref_line_xshift = np.interp(catalog_pixels, x, ref_xshift)
        n_catalog = ref_line_xshift.size

        # --- Harvest extra line targets (self-calibrating) ---
        # The hand-built catalog under-samples the wavelength extremes (blue: 6 of
        # ~24 available lines below the mid-blue), leaving the xshift fit under-
        # constrained there -> misaligned sky -> edge residuals. Detect lines on
        # every real camera's reference fibre at ``harvest_sigdetect``, map each to
        # the common xshift frame via that camera's own xshift, and keep those
        # confirmed across >= ``harvest_min_cameras`` cameras. These add xshift
        # constraints only; the wavelength solution (wv_fit below) is unchanged.
        det_sig = harvest_sigdetect if harvest_lines else sigdetect
        if harvest_lines:
            hv, hc = [], []
            for e_idx in range(len(arcspec)):
                if metadata[e_idx]['channel'] != channel:
                    continue
                o = arcspec[e_idx]
                rf = min(REF_FIBER, o.counts.shape[0] - 1)
                s150 = np.nan_to_num(o.counts[rf, :])
                if np.nanmax(s150) <= 1:
                    continue  # placeholder camera
                xs150 = o.xshift[rf, :]
                if not _monotonic(xs150):
                    continue
                tc, _ = _detect_lines(s150, harvest_sigdetect, fwhm)
                if tc is None:
                    continue
                cm = np.interp(tc, x, xs150)   # common-frame xshift of each line
                hv.append(cm)
                hc.append(np.full(cm.size, e_idx))
            if hv:
                harvested = _cluster_consensus(np.concatenate(hv),
                                               np.concatenate(hc),
                                               harvest_tol, harvest_min_cameras)
                # keep only harvested targets not already covered by the catalog
                new = np.array([t for t in harvested
                                if ref_line_xshift.size == 0
                                or np.min(np.abs(ref_line_xshift - t)) > harvest_tol])
                if new.size:
                    ref_line_xshift = np.sort(np.concatenate([ref_line_xshift, new]))
                print(f"  {channel}: harvested {new.size} extra line targets "
                      f"(catalog {n_catalog}, total {ref_line_xshift.size})")

        # Global xshift -> wavelength map from the reference fibre, exactly as
        # refineArcX builds wv_fit.
        sort_idx = np.argsort(ref_xshift)
        wv_fit = robust_fit(ref_xshift[sort_idx], ref_wave[sort_idx],
                            function='legendre', order=5, lower=3, upper=3,
                            maxdev=5)

        min_samples = max(20 * (surface_order_x + 1) * (surface_order_fiber + 1),
                          200)

        # Night-of line source for this channel (e.g. afternoon ThAr arcs)
        src_dict = None
        src_lookup = {}
        if line_source and line_source.get(channel):
            src = line_source[channel]
            src_dict = (src if isinstance(src, dict)
                        else extract.ExtractLlamas.loadExtraction(src))
            src_lookup = {(str(m['bench']), m['side']): j
                          for j, m in enumerate(src_dict['metadata'])
                          if m['channel'] == channel}
            print(f"  {channel}: detecting line centroids on the NIGHT arc "
                  f"({len(src_lookup)} cameras available)")

        for fits_ext in range(len(arcspec)):
            if metadata[fits_ext]['channel'] != channel:
                continue

            nfibers = metadata[fits_ext]['nfibers']
            bench = metadata[fits_ext]['bench']
            side = metadata[fits_ext]['side']
            cam = f"{bench}{side} {channel}"
            t0 = time.time()

            orig_xshift = arcspec[fits_ext].xshift.copy()
            counts = arcspec[fits_ext].counts

            # Placeholder / unusable extension guards
            if (np.count_nonzero(orig_xshift) == 0
                    or not np.isfinite(orig_xshift).any()
                    or np.nanstd(np.nan_to_num(counts)) == 0):
                print(f"  {cam}: placeholder/unusable — skipped")
                for i in range(nfibers):
                    _qa(channel, bench, side, i, 'skipped_placeholder')
                continue

            fc = _fiber_coords(arcspec[fits_ext], nfibers, cam)

            # Night-arc line source for this camera, with a per-camera epoch
            # bootstrap: cross-correlate the night arc's reference-fibre
            # spectrum against the packaged arc's so pass-1 matching survives
            # multi-pixel offsets between the two epochs.
            det_counts = counts
            n_det_fibers = nfibers
            boot_t = None  # old-frame pixel as a function of new-frame pixel
            if src_dict is not None:
                j = src_lookup.get((str(bench), side))
                src_counts = (src_dict['extractions'][j].counts
                              if j is not None else None)
                if (src_counts is None
                        or np.nanstd(np.nan_to_num(src_counts)) == 0):
                    print(f"  {cam}: night arc unavailable/placeholder — "
                          f"using reference-arc spectra for this camera")
                else:
                    det_counts = src_counts
                    n_det_fibers = min(nfibers, src_counts.shape[0])
                    rf = min(REF_FIBER, n_det_fibers - 1)
                    old_spec = np.nan_to_num(interpolateNaNs(counts[rf, :]),
                                             nan=0.0, posinf=0.0, neginf=0.0)
                    new_spec = np.nan_to_num(interpolateNaNs(src_counts[rf, :]),
                                             nan=0.0, posinf=0.0, neginf=0.0)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            success, shift, stretch, stretch2, _, _, _ = \
                                xcorr_shift_stretch(old_spec, new_spec,
                                                    stretch_func='quadratic')
                    except Exception:
                        success = 0
                    if success == 1:
                        boot_t = x * stretch + x ** 2 * stretch2 + shift
                        if not _monotonic(boot_t):
                            boot_t = None
                        else:
                            print(f"  {cam}: epoch bootstrap shift={shift:+.2f}px "
                                  f"stretch={(stretch - 1) * 1e3:+.3f}e-3")
                    if boot_t is None:
                        print(f"  {cam}: bootstrap xcorr failed — assuming "
                              f"zero epoch offset for pass 1")

            # Detection once per fibre (cached in-run; reused by both passes).
            # ``detection_cache`` (optional): a {(bench, side): [(tcent, ecent), ...]}
            # dict of pre-computed centroids. Detection depends only on the line
            # source and (sigdetect, fwhm), so when those are fixed across calls
            # (e.g. a parameter sweep over fit-only knobs) the caller can supply
            # the cache to skip the expensive re-detection. On a miss the
            # detections are computed and stored back into the dict for reuse.
            cache_key = (str(bench), side)
            cached = detection_cache.get(cache_key) if detection_cache is not None else None
            n_det_total = 0
            if cached is not None:
                detections = cached
                for tcent, _ in detections:
                    if tcent is not None:
                        n_det_total += len(tcent)
            else:
                detections = []
                for i in range(nfibers):
                    if i < n_det_fibers:
                        tcent, ecent = _detect_lines(det_counts[i, :], det_sig, fwhm)
                    else:
                        tcent, ecent = None, None
                    detections.append((tcent, ecent))
                    if tcent is not None:
                        n_det_total += len(tcent)
                if detection_cache is not None:
                    detection_cache[cache_key] = detections

            identity = np.array([np.allclose(orig_xshift[i], x)
                                 for i in range(nfibers)])

            # ---- Pass 1: sample from fibres with a usable quadratic ----
            p_px, p_fc, p_t, p_w = [], [], [], []
            for i in range(nfibers):
                tcent, ecent = detections[i]
                if tcent is None or identity[i]:
                    continue
                if not _monotonic(orig_xshift[i]):
                    continue
                pred = np.interp(ref_line_xshift, orig_xshift[i], x)
                if boot_t is not None:
                    # map old-frame predictions into the night arc's frame
                    pred = np.interp(pred, boot_t, x)
                mp, mt, mw = _match_lines(tcent, ecent, pred, ref_line_xshift,
                                          match_tol_pass1)
                if mp.size:
                    p_px.append(mp); p_t.append(mt); p_w.append(mw)
                    p_fc.append(np.full(mp.size, fc[i]))

            n_pass1 = int(sum(a.size for a in p_px))
            if n_pass1 < min_samples:
                print(f"  {cam}: only {n_pass1} pass-1 samples "
                      f"(< {min_samples}) — extension left unchanged")
                for i in range(nfibers):
                    _qa(channel, bench, side, i, 'fallback_surface')
                continue

            surf = _fit_surface(np.concatenate(p_px), np.concatenate(p_fc),
                                np.concatenate(p_t), np.concatenate(p_w),
                                surface_order_x, surface_order_fiber, naxis1)

            # ---- Pass 2: re-predict every fibre from the surface, refit ----
            fiber_matches = [None] * nfibers
            q_px, q_fc, q_t, q_w = [], [], [], []
            for i in range(nfibers):
                tcent, ecent = detections[i]
                if tcent is None:
                    continue
                surf_i = surf.eval(x, x2=np.full(naxis1, fc[i]))
                if not _monotonic(surf_i):
                    continue
                pred2 = np.interp(ref_line_xshift, surf_i, x)
                mp, mt, mw = _match_lines(tcent, ecent, pred2, ref_line_xshift,
                                          match_tol_pass2)
                fiber_matches[i] = (mp, mt, mw)
                if mp.size:
                    q_px.append(mp); q_t.append(mt); q_w.append(mw)
                    q_fc.append(np.full(mp.size, fc[i]))

            n_pass2 = int(sum(a.size for a in q_px))
            if n_pass2 >= min_samples:
                surf = _fit_surface(np.concatenate(q_px), np.concatenate(q_fc),
                                    np.concatenate(q_t), np.concatenate(q_w),
                                    surface_order_x, surface_order_fiber,
                                    naxis1)

            # Camera-wide constrained x-range (outermost matched arc lines);
            # the surface is only trustworthy inside this, so xshift is
            # extrapolated in a controlled way beyond it (see _apply_edge_extrap).
            _pooled = (np.concatenate(q_px) if q_px
                       else (np.concatenate(p_px) if p_px
                             else np.array([0.0, naxis1 - 1.0])))
            x_lo_c, x_hi_c = float(np.min(_pooled)), float(np.max(_pooled))

            # ---- Stage C: per-fibre perturbation + assembly ----
            n_perturb = n_surface_only = n_nomatch = n_revert = 0
            for i in range(nfibers):
                surf_i = surf.eval(x, x2=np.full(naxis1, fc[i]))
                matches = fiber_matches[i]
                tcent, _ = detections[i]
                n_det = 0 if tcent is None else len(tcent)

                if matches is None or matches[0].size == 0:
                    new_xshift = surf_i
                    mode, n_m = 'surface_only', 0
                    resid = None
                    pixels = None
                else:
                    mp, mt, mw = matches
                    resid_surface = mt - surf.eval(mp, x2=np.full(mp.size, fc[i]))
                    if mp.size >= perturb_min_lines:
                        perturb = _fit_perturbation(mp, resid_surface, mw,
                                                    perturb_order,
                                                    perturb_shrink_lines,
                                                    naxis1,
                                                    min_snr=perturb_min_snr)
                        mode = 'surface+perturb'
                        n_perturb += 1
                    else:
                        perturb = lambda xg: np.zeros_like(
                            np.asarray(xg, dtype=float))
                        mode = 'surface_only'
                        n_surface_only += 1
                    new_xshift = surf_i + perturb(x)
                    resid = mt - (surf.eval(mp, x2=np.full(mp.size, fc[i]))
                                  + perturb(mp))
                    pixels = mp
                    n_m = int(mp.size)

                # Controlled extrapolation past the outermost matched line
                # (kills the multi-px polynomial swing -> edge sky residuals).
                new_xshift = _apply_edge_extrap(new_xshift, x, x_lo_c, x_hi_c,
                                                edge_extrap)

                if not _monotonic(new_xshift):
                    n_revert += 1
                    _qa(channel, bench, side, i, 'fallback_monotonic',
                        n_detected=n_det, n_matched=n_m, mode=mode)
                    continue  # leave orig xshift/wave on this fibre

                arcspec[fits_ext].xshift[i, :] = new_xshift
                arcspec[fits_ext].wave[i, :] = wv_fit.eval(new_xshift)

                if resid is not None:
                    _qa(channel, bench, side, i, 'refined',
                        n_detected=n_det, n_matched=n_m,
                        pixels=pixels, resid_pix=resid, mode=mode)
                else:
                    n_nomatch += 1
                    _qa(channel, bench, side, i, 'surface_no_matches',
                        n_detected=n_det, mode='surface_only')

            n_identity_rescued = int(identity.sum())
            print(f"  {cam}: surface fit on {n_pass2 or n_pass1} samples "
                  f"(pass1 {n_pass1}); perturb {n_perturb}, surface-only "
                  f"{n_surface_only}, no-match {n_nomatch}, reverted {n_revert}, "
                  f"identity-rescued {n_identity_rescued}  "
                  f"[{time.time() - t0:.1f}s]")

        # Free the night-arc extraction before moving to the next channel
        src_dict = None
        src_lookup = {}

    if not _save_result:
        return arcdict
    sv = arc_extraction_shifted_pickle.replace('.pkl', '_refined2d.pkl')
    print(f"Saving 2D-refined arc solution to {sv}")
    extract.save_extractions(arcspec, savefile=sv)
    return sv


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Hybrid 2D surface xshift refinement (standalone).')
    parser.add_argument('pickle', help='Arc extraction pickle (xshift populated)')
    parser.add_argument('--channels', default=None,
                        help='Comma-separated subset, e.g. red,green')
    parser.add_argument('--order-x', type=int, default=3)
    parser.add_argument('--order-fiber', type=int, default=2)
    parser.add_argument('--perturb-order', type=int, default=0)
    parser.add_argument('--qa-dir', default=None,
                        help='Also run waveQA (xshift structure + arc residuals) '
                             'into this directory')
    parser.add_argument('--label', default='arc2d')
    args = parser.parse_args()

    channels = ([c.strip() for c in args.channels.split(',')]
                if args.channels else None)
    records = []
    out = refineArcX2D(args.pickle, channels=channels, qa_collector=records,
                       surface_order_x=args.order_x,
                       surface_order_fiber=args.order_fiber,
                       perturb_order=args.perturb_order)
    print(f"Output: {out}  ({len(records)} QA records)")

    if args.qa_dir:
        from llamas_pyjamas.QA import waveQA
        waveQA.xshift_structure_qa(out, qa_dir=args.qa_dir, label=args.label,
                                   emit='png')
        waveQA.arc_residual_qa(records, qa_dir=args.qa_dir, label=args.label,
                               emit='png')
        # CSV scorecard for diffing against baseline/arcrefine
        waveQA.run_wavelength_qa(out, qa_dir=args.qa_dir, run_label=args.label,
                                 arc_qa_records=records)


if __name__ == '__main__':
    main()
