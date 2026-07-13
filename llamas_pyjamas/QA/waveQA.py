"""
llamas_pyjamas.QA.waveQA
========================
Per-run quality-assurance diagnostics for the wavelength calibration / xshift.

Motivation: the per-fibre xshift (global quadratic from cross-correlation against
a single reference fibre) can mis-track trace curvature near the detector edges,
which shows up as large sky-subtraction residuals. This module quantifies that so
algorithm changes can be scored against a baseline.

Products (written to a QA directory as PNGs + one CSV per run):

1. ``xshift_structure_qa``  -- per-camera deviation of each fibre's xshift from
   the detector-median curve (per-fibre DC offset removed, isolating *curvature*
   disagreement), with identity-fallback fibres flagged.
2. ``sky_line_residual_qa`` -- per-camera post-sky-subtraction residuals at
   bright OH sky-line pixels vs (fibre, x): the direct view of the symptom.
3. ``write_wavelength_qa_summary`` -- one CSV row per camera with center-vs-edge
   RMS metrics, for diffing a baseline run against an improved one.

Standalone usage (no pipeline run needed; reads extraction pickles):

    python -m llamas_pyjamas.QA.waveQA <..._sky1d_extractions.pkl> \
        --qa-dir /path/to/QA --label baseline

Notes
-----
- Inputs are extraction pickles (``ExtractLlamas.loadExtraction`` dicts). RSS
  FITS files are out of scope for v1: they do not carry ``xshift``.
- Arc-line residual QA (from refineArcX) is added separately; its CSV columns
  are left empty when no collector records are supplied.
"""

import argparse
import base64
import csv
import io
import logging
import os
from datetime import datetime, timezone

import numpy as np

logger = logging.getLogger(__name__)

# Fraction of the dispersion axis treated as each "edge" third.
EDGE_FRAC = 1.0 / 3.0

# OH-line detection parameters (mirroring Sky/skyConfig.py defaults used by skyQA).
OH_SIGDETECT = 5.0
CONTINUUM_WINDOW_PIX = 6

# CSV schema — one row per camera. Arc columns stay empty until arc QA records
# are supplied (Phase B: refineArcX collector).
CSV_COLUMNS = [
    'run_label', 'timestamp_utc', 'source_file', 'bench', 'side', 'channel',
    'nfibers', 'n_identity_fallback',
    'xshift_dev_rms_center_pix', 'xshift_dev_rms_edge_pix',
    'xshift_edge_center_ratio', 'max_fiber_discontinuity_pix',
    'arc_n_refined', 'arc_n_fallback', 'arc_resid_rms_pix',
    'arc_resid_rms_center_pix', 'arc_resid_rms_edge_pix',
    'sky_rms_before', 'sky_rms_after', 'sky_improvement',
    'sky_rms_center', 'sky_rms_edge', 'n_sky_quality_flagged',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _camera_key(meta):
    """'1A_red' style key from a metadata dict."""
    return f"{meta.get('bench', '?')}{meta.get('side', '?')}_{meta.get('channel', '?')}"


def _resolve_qa_dir(qa_dir, product_path):
    """QA output dir: explicit qa_dir, else '<run output dir>/QA' next to the product.

    Extraction pickles live in ``{output_dir}/extractions/``, so the default is
    the sibling ``{output_dir}/QA``.
    """
    if not qa_dir:
        base = os.path.dirname(os.path.dirname(os.path.abspath(product_path)))
        qa_dir = os.path.join(base, 'QA')
    os.makedirs(qa_dir, exist_ok=True)
    return qa_dir


def _thirds(n):
    """(left, center, right) slices splitting the dispersion axis into thirds."""
    a = int(n * EDGE_FRAC)
    b = int(n * (1.0 - EDGE_FRAC))
    return slice(0, a), slice(a, b), slice(b, n)


def _load(extraction_input):
    """Accept a pickle path or an already-loaded extraction dict."""
    if isinstance(extraction_input, dict):
        return extraction_input, '<in-memory>'
    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
    return ExtractLlamas.loadExtraction(extraction_input), extraction_input


def _rms(values):
    """RMS of the finite entries of ``values`` (nan when empty)."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.sqrt(np.mean(v ** 2))) if v.size else float('nan')


def _fig_prefix(label):
    return f"{label}_" if label else ""


def _emit_figure(fig, qa_dir, fname, emit):
    """Render a figure per the emit mode and return the stats-dict fragment.

    emit: 'b64' (default; embed in the HTML report), 'png' (write file only),
    or 'both'.
    """
    import matplotlib.pyplot as plt
    out = {}
    if emit in ('png', 'both'):
        path = os.path.join(qa_dir, fname)
        fig.savefig(path, dpi=120)
        out['figure'] = path
    if emit in ('b64', 'both'):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110)
        out['figure_b64'] = base64.b64encode(buf.getvalue()).decode('ascii')
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Product 1: xshift structure QA
# ---------------------------------------------------------------------------

def xshift_structure_qa(extraction_input, qa_dir=None, label='', emit='png'):
    """Per-camera xshift curvature-disagreement diagnostics.

    For each camera, computes ``delta = xshift - x`` per fibre, subtracts the
    detector-median delta curve and each fibre's own median offset. What remains
    ('dev') is fibre-to-fibre disagreement in the *shape* of the wavelength
    shift — exactly the quantity that breaks sky subtraction when it grows at
    the detector edges. Identity-fallback fibres (xcorr failure sets
    ``xshift[i,:] = x``) are excluded from the reference curve and flagged.

    Returns ``{cam_key: {metrics..., 'figure': path or None}}``.
    """
    arcdict, src_path = _load(extraction_input)
    extractions = arcdict['extractions']
    metadata = arcdict.get('metadata',
                           [{} for _ in extractions])
    qa_dir = _resolve_qa_dir(qa_dir, src_path if src_path != '<in-memory>' else '.')

    results = {}
    for obj, meta in zip(extractions, metadata):
        cam = _camera_key(meta)
        xshift = getattr(obj, 'xshift', None)
        if xshift is None or xshift.ndim != 2 or xshift.shape[0] == 0:
            results[cam] = {'note': 'no_xshift'}
            continue
        if not np.isfinite(xshift).any() or np.count_nonzero(xshift) == 0:
            results[cam] = {'note': 'xshift_unpopulated'}
            continue

        nfib, nx = xshift.shape
        x = np.arange(nx, dtype=float)
        left, center, right = _thirds(nx)

        delta = xshift - x[None, :]
        identity = np.array([np.allclose(xshift[i], x) for i in range(nfib)])

        good = ~identity
        ref_rows = delta[good] if good.any() else delta
        median_curve = np.nanmedian(ref_rows, axis=0)
        dev = delta - median_curve[None, :]
        dev = dev - np.nanmedian(dev, axis=1, keepdims=True)

        dev_good = dev[good] if good.any() else dev
        rms_center = float(np.nanstd(dev_good[:, center]))
        edge_vals = np.concatenate(
            [dev_good[:, left].ravel(), dev_good[:, right].ravel()])
        rms_edge = float(np.nanstd(edge_vals[np.isfinite(edge_vals)]))
        ratio = rms_edge / rms_center if rms_center > 0 else float('nan')
        # Largest adjacent-fibre jump in the dev field (median |difference| per pair)
        if nfib > 1:
            pair_jump = np.nanmedian(np.abs(np.diff(dev, axis=0)), axis=1)
            max_disc = float(np.nanmax(pair_jump))
        else:
            max_disc = float('nan')

        stats = {
            'nfibers': nfib,
            'n_identity_fallback': int(identity.sum()),
            'xshift_dev_rms_center_pix': rms_center,
            'xshift_dev_rms_edge_pix': rms_edge,
            'xshift_edge_center_ratio': ratio,
            'max_fiber_discontinuity_pix': max_disc,
            'figure': None,
        }

        try:
            import matplotlib
            matplotlib.use('Agg', force=False)
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(11, 8),
                                     gridspec_kw={'height_ratios': [1.4, 1]})

            ax = axes[0]
            vlim = np.nanpercentile(np.abs(dev[np.isfinite(dev)]), 98)
            vlim = max(vlim, 0.1)
            im = ax.imshow(dev, aspect='auto', origin='lower',
                           cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                           extent=[0, nx, -0.5, nfib - 0.5])
            for i in np.where(identity)[0]:
                ax.plot([-0.005 * nx], [i], marker='>', color='red',
                        markersize=4, clip_on=False)
            for edge_x in (nx * EDGE_FRAC, nx * (1 - EDGE_FRAC)):
                ax.axvline(edge_x, color='k', ls=':', lw=0.8, alpha=0.6)
            fig.colorbar(im, ax=ax, label='xshift deviation (px)')
            ax.set_ylabel('Fibre index')
            ax.set_title(
                f"{label + ' ' if label else ''}xshift curvature deviation — {cam}\n"
                f"rms center={rms_center:.3f}px  edge={rms_edge:.3f}px  "
                f"(ratio {ratio:.2f})   identity-fallback fibres: {int(identity.sum())}")

            ax = axes[1]
            cmap = plt.get_cmap('viridis')
            for i in range(nfib):
                if identity[i]:
                    continue
                ax.plot(x, dev[i], color=cmap(i / max(nfib - 1, 1)),
                        lw=0.4, alpha=0.35)
            for edge_x in (nx * EDGE_FRAC, nx * (1 - EDGE_FRAC)):
                ax.axvline(edge_x, color='k', ls=':', lw=0.8, alpha=0.6)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xlim(0, nx)
            ax.set_ylim(-3 * vlim, 3 * vlim)
            ax.set_xlabel('x (detector column)')
            ax.set_ylabel('deviation (px)')

            fig.tight_layout()
            stats.update(_emit_figure(
                fig, qa_dir, f"{_fig_prefix(label)}xshiftQA_{cam}.png", emit))
        except Exception as e:
            logger.warning(f"waveQA: xshift figure failed for {cam} ({e})")

        results[cam] = stats
        logger.info(
            f"waveQA xshift {cam}: rms center={rms_center:.3f} edge={rms_edge:.3f} "
            f"ratio={ratio:.2f} identity={int(identity.sum())}/{nfib}")

    return results


# ---------------------------------------------------------------------------
# Product 2: arc-line residual QA (from refineArcX qa_collector records)
# ---------------------------------------------------------------------------

def arc_residual_qa(arc_qa_records, qa_dir=None, label='arc', emit='png',
                    naxis1=2048):
    """Per-camera arc-line fit residual diagnostics from refineArcX records.

    ``arc_qa_records``: list of per-fiber dicts appended by
    ``refineArcX(..., qa_collector=[...])`` — status, matched line pixels, and
    fit residuals (in xshift/pixel units, reference-fiber frame).

    Returns ``{cam_key: {metrics..., 'figure'/'figure_b64'}}``.
    """
    if not arc_qa_records:
        return {}
    qa_dir = _resolve_qa_dir(qa_dir, '.') if qa_dir else _resolve_qa_dir(None, '.')

    by_cam = {}
    for rec in arc_qa_records:
        cam = f"{rec.get('bench','?')}{rec.get('side','?')}_{rec.get('channel','?')}"
        by_cam.setdefault(cam, []).append(rec)

    left, center, right = _thirds(naxis1)
    results = {}
    for cam, recs in by_cam.items():
        refined = [r for r in recs if r.get('status') == 'refined']
        fallback = [r for r in recs if r.get('status') != 'refined']

        all_px = (np.concatenate([r['pixels'] for r in refined])
                  if refined else np.array([]))
        all_res = (np.concatenate([r['resid_pix'] for r in refined])
                   if refined else np.array([]))

        in_center = (all_px >= center.start) & (all_px < center.stop)
        stats = {
            'arc_n_refined': len(refined),
            'arc_n_fallback': len(fallback),
            'arc_resid_rms_pix': _rms(all_res),
            'arc_resid_rms_center_pix': _rms(all_res[in_center]),
            'arc_resid_rms_edge_pix': _rms(all_res[~in_center]),
        }

        try:
            import matplotlib
            matplotlib.use('Agg', force=False)
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 1, figsize=(11, 7))

            ax = axes[0]
            cmap = plt.get_cmap('viridis')
            nfib_tot = max((r['fiber'] for r in recs), default=1) + 1
            for r in refined:
                if r['pixels'].size:
                    ax.plot(r['pixels'], r['resid_pix'], '.', markersize=2,
                            color=cmap(r['fiber'] / max(nfib_tot - 1, 1)),
                            alpha=0.5)
            for span in ((0, naxis1 * EDGE_FRAC),
                         (naxis1 * (1 - EDGE_FRAC), naxis1)):
                ax.axvspan(span[0], span[1], color='0.9', zorder=0)
            ax.axhline(0, color='k', lw=0.6)
            ax.set_xlim(0, naxis1)
            ax.set_xlabel('matched line pixel (x)')
            ax.set_ylabel('xshift residual (px)')
            ax.set_title(
                f"{label + ' ' if label else ''}arc-line fit residuals — {cam}\n"
                f"rms={stats['arc_resid_rms_pix']:.3f}px "
                f"(center {stats['arc_resid_rms_center_pix']:.3f} / "
                f"edge {stats['arc_resid_rms_edge_pix']:.3f})   "
                f"refined {len(refined)} / fallback {len(fallback)} fibres")

            ax = axes[1]
            fib_idx = [r['fiber'] for r in refined]
            fib_rms = [_rms(r['resid_pix']) for r in refined]
            ax.plot(fib_idx, fib_rms, '.', color='tab:blue', markersize=4,
                    label='refined (per-fibre rms)')
            if fallback:
                fx = [r['fiber'] for r in fallback]
                ax.plot(fx, np.zeros(len(fx)), 'rx', markersize=5,
                        label='fallback (no refinement)')
            ax.set_xlabel('Fibre index')
            ax.set_ylabel('rms (px)')
            ax.legend(fontsize=8)

            fig.tight_layout()
            stats.update(_emit_figure(
                fig, qa_dir, f"{_fig_prefix(label)}arcResidQA_{cam}.png", emit))
        except Exception as e:
            logger.warning(f"waveQA: arc figure failed for {cam} ({e})")

        results[cam] = stats
        logger.info(
            f"waveQA arc {cam}: rms={stats['arc_resid_rms_pix']:.3f}px "
            f"refined={len(refined)} fallback={len(fallback)}")

    return results


# ---------------------------------------------------------------------------
# Product 3: sky-line residual QA (the money plot)
# ---------------------------------------------------------------------------

def sky_line_residual_qa(sky1d_input, qa_dir=None, label='', emit='png'):
    """Per-camera post-sky-subtraction residuals at OH sky-line pixels.

    Uses each camera's own ``.sky`` model (populated by skyModel_1d) to locate
    OH-line pixels, then measures the line-contrast RMS of
    ``counts - sky`` (per-fibre continuum removed) at those pixels — before vs
    after sky subtraction, and center vs edge thirds of the detector.

    Returns ``{cam_key: {metrics..., 'figure': path or None}}``.
    """
    from llamas_pyjamas.Sky.skyScale import _continuum, _line_mask

    scidict, src_path = _load(sky1d_input)
    extractions = scidict['extractions']
    metadata = scidict.get('metadata', [{} for _ in extractions])
    qa_dir = _resolve_qa_dir(qa_dir, src_path if src_path != '<in-memory>' else '.')

    results = {}
    for obj, meta in zip(extractions, metadata):
        cam = _camera_key(meta)
        counts = getattr(obj, 'counts', None)
        sky = getattr(obj, 'sky', None)
        if counts is None or counts.ndim != 2 or counts.shape[0] == 0:
            results[cam] = {'note': 'no_counts'}
            continue
        if sky is None or np.count_nonzero(np.nan_to_num(sky)) == 0:
            results[cam] = {'note': 'no_sky_model'}
            continue
        if np.count_nonzero(np.nan_to_num(counts)) == 0:
            results[cam] = {'note': 'placeholder_camera'}
            continue
        if sky.shape != counts.shape:
            # Seen in real data: counts and sky rows disagree (e.g. 300 vs 298).
            # Trim to the common fibre count and flag it — worth investigating
            # upstream, but QA must not crash on it.
            logger.warning(
                f"waveQA sky {cam}: counts {counts.shape} vs sky {sky.shape} "
                f"shape mismatch — trimming to common rows")
            n_common = min(counts.shape[0], sky.shape[0])
            counts = counts[:n_common]
            sky = sky[:n_common]

        nfib, nx = counts.shape
        left, center, right = _thirds(nx)

        # OH-line pixel mask from the camera's median sky model
        template = np.nanmedian(sky, axis=0)
        cont_t = _continuum(template, CONTINUUM_WINDOW_PIX)
        line_px = _line_mask(np.nan_to_num(template) - cont_t, OH_SIGDETECT)
        if not line_px.any():
            results[cam] = {'note': 'no_oh_lines_detected'}
            continue

        resid = counts - sky
        # Per-fibre line-contrast RMS: remove each fibre's smooth continuum so
        # object continuum does not masquerade as a sky-line residual.
        rms_before = np.full(nfib, np.nan)
        rms_after = np.full(nfib, np.nan)
        resid_cs = np.full_like(resid, np.nan, dtype=float)
        for i in range(nfib):
            ci = np.nan_to_num(counts[i])
            ri = np.nan_to_num(resid[i])
            before_cs = ci - _continuum(ci, CONTINUUM_WINDOW_PIX)
            after_cs = ri - _continuum(ri, CONTINUUM_WINDOW_PIX)
            resid_cs[i] = after_cs
            rms_before[i] = _rms(before_cs[line_px])
            rms_after[i] = _rms(after_cs[line_px])

        sky_quality = getattr(obj, 'sky_quality', None)
        if sky_quality is not None and len(sky_quality) > nfib:
            sky_quality = np.asarray(sky_quality)[:nfib]
        n_flagged = int(np.count_nonzero(sky_quality)) if sky_quality is not None else 0

        med_before = float(np.nanmedian(rms_before))
        med_after = float(np.nanmedian(rms_after))
        improvement = med_before / med_after if med_after > 0 else float('nan')

        def _third_rms(sl):
            m = np.zeros(nx, dtype=bool)
            m[sl] = True
            m &= line_px
            return _rms(resid_cs[:, m]) if m.any() else float('nan')

        sky_rms_center = _third_rms(center)
        sky_rms_edge = _rms([v for v in
                             [_third_rms(left), _third_rms(right)]
                             if np.isfinite(v)])

        stats = {
            'nfibers': nfib,
            'sky_rms_before': med_before,
            'sky_rms_after': med_after,
            'sky_improvement': improvement,
            'sky_rms_center': sky_rms_center,
            'sky_rms_edge': sky_rms_edge,
            'n_sky_quality_flagged': n_flagged,
            'n_oh_pixels': int(line_px.sum()),
            'figure': None,
        }

        try:
            import matplotlib
            matplotlib.use('Agg', force=False)
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(11, 10),
                                     gridspec_kw={'height_ratios': [1.5, 1, 1]})

            ax = axes[0]
            img = np.where(line_px[None, :], resid_cs, np.nan)
            finite = img[np.isfinite(img)]
            vlim = max(np.nanpercentile(np.abs(finite), 98) if finite.size else 1.0,
                       1e-3)
            im = ax.imshow(img, aspect='auto', origin='lower', cmap='RdBu_r',
                           vmin=-vlim, vmax=vlim, extent=[0, nx, -0.5, nfib - 0.5])
            for edge_x in (nx * EDGE_FRAC, nx * (1 - EDGE_FRAC)):
                ax.axvline(edge_x, color='k', ls=':', lw=0.8, alpha=0.6)
            fig.colorbar(im, ax=ax, label='sky-line residual (counts)')
            ax.set_ylabel('Fibre index')
            ax.set_title(
                f"{label + ' ' if label else ''}OH sky-line residuals — {cam}\n"
                f"line RMS before={med_before:.1f} after={med_after:.1f} "
                f"(x{improvement:.1f})   center={sky_rms_center:.1f} "
                f"edge={sky_rms_edge:.1f}   flagged fibres: {n_flagged}")

            ax = axes[1]
            fidx = np.arange(nfib)
            ax.plot(fidx, rms_before, color='0.7', lw=0.8, label='before')
            ax.plot(fidx, rms_after, color='tab:blue', lw=0.9, label='after')
            if sky_quality is not None:
                bad = np.where(np.asarray(sky_quality) != 0)[0]
                ax.plot(bad, rms_after[bad], 'r.', markersize=5,
                        label='sky_quality flagged')
            ax.set_yscale('log')
            ax.set_xlabel('Fibre index')
            ax.set_ylabel('OH-line RMS')
            ax.legend(fontsize=8)

            ax = axes[2]
            colmed = np.nanmedian(np.abs(resid_cs), axis=0)
            ax.plot(np.arange(nx), colmed, color='tab:blue', lw=0.6)
            ax.plot(np.where(line_px)[0], colmed[line_px], '.', color='tab:orange',
                    markersize=2, label='OH-line pixels')
            for edge_x in (nx * EDGE_FRAC, nx * (1 - EDGE_FRAC)):
                ax.axvline(edge_x, color='k', ls=':', lw=0.8, alpha=0.6)
            ax.set_xlabel('x (detector column)')
            ax.set_ylabel('median |residual|')
            ax.legend(fontsize=8)

            fig.tight_layout()
            stats.update(_emit_figure(
                fig, qa_dir, f"{_fig_prefix(label)}skyLineQA_{cam}.png", emit))
        except Exception as e:
            logger.warning(f"waveQA: sky figure failed for {cam} ({e})")

        results[cam] = stats
        logger.info(
            f"waveQA sky {cam}: OH RMS before={med_before:.1f} after={med_after:.1f} "
            f"center={sky_rms_center:.1f} edge={sky_rms_edge:.1f}")

    return results


# ---------------------------------------------------------------------------
# Product 3b: sky-line xshift ERROR QA (measured from the science frame itself)
# ---------------------------------------------------------------------------

def _sky_template_peaks(obj, nfibers, ref_fiber=150, fiber_half_width=10,
                        sigdetect=5.0, fwhm=4.0):
    """Detect sky-line peaks in a per-camera template (refineSkyX's method).

    Builds a high-S/N template from ref_fiber +/- fiber_half_width in xshift
    space and returns the peak positions in xshift coordinates (or None).
    Mirrors Sky/skyLlamas.refineSkyX:90-127 — keep in sync deliberately.
    """
    rf = min(max(ref_fiber, 0), nfibers - 1)
    t_lo, t_hi = max(0, rf - fiber_half_width), min(nfibers, rf + fiber_half_width + 1)
    txs = np.concatenate([obj.xshift[tf, :] for tf in range(t_lo, t_hi)])
    tct = np.concatenate([np.nan_to_num(obj.counts[tf, :], nan=0.0, posinf=0.0,
                                        neginf=0.0) for tf in range(t_lo, t_hi)])
    order = np.argsort(txs)
    txs, tct = txs[order], tct[order]
    n_grid = 2048
    grid = np.linspace(txs[0], txs[-1], n_grid)
    binned = np.interp(grid, txs, tct)
    try:
        import warnings as _w
        from pypeit.core.wavecal.wvutils import arc_lines_from_spec
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            tcent, _, _, _, _ = arc_lines_from_spec(binned, sigdetect=sigdetect,
                                                    fwhm=fwhm)
    except Exception:
        return None
    if len(tcent) == 0:
        return None
    return np.interp(tcent, np.arange(n_grid, dtype=float), grid)


def sky_xshift_error_qa(sky1d_input, qa_dir=None, label='', emit='png',
                        ref_fiber=150, fiber_half_width=10,
                        sigdetect=5.0, fwhm=4.0, match_tol=5.0):
    """Measure the ACTUAL xshift error on a science frame from its own sky lines.

    The arc-based QA measures how well the model fits the (reference-epoch) arc
    exposure. This measures what matters on sky: for each fibre, sky-line
    centroids are detected on the science spectrum, mapped through that fibre's
    xshift, and compared to a per-camera sky template — if xshift were perfect,
    every fibre's sky lines would land exactly on the template positions.
    Residuals therefore capture BOTH model error and any epoch drift (flexure)
    between the reference arc and the science exposure. This is refineSkyX's
    internal measurement, surfaced as QA (computed there and discarded).

    Returns {cam_key: {metrics..., 'figure'/'figure_b64'}}.
    """
    import warnings as _w
    from pypeit.core.wavecal.wvutils import arc_lines_from_spec

    scidict, src_path = _load(sky1d_input)
    extractions = scidict['extractions']
    metadata = scidict.get('metadata', [{} for _ in extractions])
    qa_dir = _resolve_qa_dir(qa_dir, src_path if src_path != '<in-memory>' else '.')

    results = {}
    for obj, meta in zip(extractions, metadata):
        cam = _camera_key(meta)
        counts = getattr(obj, 'counts', None)
        xshift = getattr(obj, 'xshift', None)
        if (counts is None or xshift is None or counts.shape[0] == 0
                or np.count_nonzero(np.nan_to_num(counts)) == 0
                or np.count_nonzero(xshift) == 0):
            results[cam] = {'note': 'no_data'}
            continue

        # Guard the known counts-vs-xshift row mismatch (e.g. 300 vs 298)
        nfib = min(counts.shape[0], xshift.shape[0])
        nx = counts.shape[1]
        x = np.arange(nx, dtype=float)
        left, center, right = _thirds(nx)

        tmpl_peaks = _sky_template_peaks(obj, nfib, ref_fiber, fiber_half_width,
                                         sigdetect, fwhm)
        if tmpl_peaks is None:
            results[cam] = {'note': 'no_template_lines'}
            continue

        all_px, all_fib, all_err = [], [], []
        fib_offset = np.full(nfib, np.nan)
        n_nolines = 0
        for i in range(nfib):
            spec = np.nan_to_num(counts[i, :], nan=0.0, posinf=0.0, neginf=0.0)
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    tcent, ecent, _, _, _ = arc_lines_from_spec(
                        spec, sigdetect=sigdetect, fwhm=fwhm)
            except Exception:
                n_nolines += 1
                continue
            if len(tcent) == 0:
                n_nolines += 1
                continue
            fib_xs = np.interp(tcent, x, xshift[i, :])
            errs, pxs = [], []
            for pk in tmpl_peaks:
                d = np.abs(fib_xs - pk)
                nearest = int(np.argmin(d))
                if d[nearest] < match_tol:
                    errs.append(fib_xs[nearest] - pk)   # + = fibre lines land redward of template
                    pxs.append(tcent[nearest])
            if not errs:
                n_nolines += 1
                continue
            errs = np.array(errs); pxs = np.array(pxs)
            all_err.append(errs); all_px.append(pxs)
            all_fib.append(np.full(errs.size, i))
            fib_offset[i] = np.median(errs)

        if not all_err:
            results[cam] = {'note': 'no_sky_lines_matched'}
            continue
        all_err = np.concatenate(all_err)
        all_px = np.concatenate(all_px)
        all_fib = np.concatenate(all_fib)

        in_center = (all_px >= center.start) & (all_px < center.stop)
        finite_off = fib_offset[np.isfinite(fib_offset)]
        stats = {
            'skyx_err_rms_pix': _rms(all_err),
            'skyx_err_rms_center_pix': _rms(all_err[in_center]),
            'skyx_err_rms_edge_pix': _rms(all_err[~in_center]),
            'skyx_median_abs_err_pix': float(np.median(np.abs(all_err))),
            'skyx_frac_fibres_off_gt0p5': float(np.mean(np.abs(finite_off) > 0.5)),
            'skyx_frac_fibres_off_gt1': float(np.mean(np.abs(finite_off) > 1.0)),
            'skyx_camera_median_offset_pix': float(np.median(finite_off)),
            'n_fibres_no_sky_lines': n_nolines,
            'n_matched_lines': int(all_err.size),
        }

        try:
            import matplotlib
            matplotlib.use('Agg', force=False)
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(3, 1, figsize=(11, 10))

            ax = axes[0]
            cmap = plt.get_cmap('viridis')
            ax.scatter(all_px, all_err, s=3, c=all_fib / max(nfib - 1, 1),
                       cmap=cmap, alpha=0.5)
            for yv, st in ((0, '-'), (0.5, ':'), (-0.5, ':'), (1, '--'), (-1, '--')):
                ax.axhline(yv, color='k', ls=st, lw=0.7, alpha=0.6)
            for edge_x in (nx * EDGE_FRAC, nx * (1 - EDGE_FRAC)):
                ax.axvline(edge_x, color='k', ls=':', lw=0.8, alpha=0.5)
            ax.set_xlim(0, nx)
            ax.set_ylim(-3, 3)
            ax.set_xlabel('detector pixel of sky line')
            ax.set_ylabel('xshift error (px)')
            ax.set_title(
                f"{label + ' ' if label else ''}sky-line xshift ERROR (measured on frame) — {cam}\n"
                f"rms={stats['skyx_err_rms_pix']:.2f}px (ctr {stats['skyx_err_rms_center_pix']:.2f} / "
                f"edge {stats['skyx_err_rms_edge_pix']:.2f})  med|err|={stats['skyx_median_abs_err_pix']:.2f}  "
                f"fibres off >0.5px: {100*stats['skyx_frac_fibres_off_gt0p5']:.0f}%  "
                f">1px: {100*stats['skyx_frac_fibres_off_gt1']:.0f}%")

            ax = axes[1]
            ax.plot(np.arange(nfib), fib_offset, '.', markersize=3, color='tab:blue')
            for yv, st in ((0, '-'), (0.5, ':'), (-0.5, ':')):
                ax.axhline(yv, color='k', ls=st, lw=0.7, alpha=0.6)
            ax.set_xlabel('fibre index')
            ax.set_ylabel('median offset (px)')
            ax.set_ylim(-3, 3)

            ax = axes[2]
            ax.hist(all_err[np.abs(all_err) < 4], bins=100, color='tab:blue', alpha=0.8)
            for yv in (0.5, -0.5, 1, -1):
                ax.axvline(yv, color='k', ls=':' if abs(yv) < 1 else '--', lw=0.7)
            ax.set_xlabel('xshift error (px)')
            ax.set_ylabel('N lines')

            fig.tight_layout()
            stats.update(_emit_figure(
                fig, qa_dir, f"{_fig_prefix(label)}skyXerrQA_{cam}.png", emit))
        except Exception as e:
            logger.warning(f"waveQA: skyXerr figure failed for {cam} ({e})")

        results[cam] = stats
        logger.info(
            f"waveQA skyXerr {cam}: rms={stats['skyx_err_rms_pix']:.2f}px "
            f"med|err|={stats['skyx_median_abs_err_pix']:.2f}px "
            f"fibres>0.5px={100*stats['skyx_frac_fibres_off_gt0p5']:.0f}%")

    return results


# ---------------------------------------------------------------------------
# ds9 region files: model-predicted line positions on the raw detector
# ---------------------------------------------------------------------------

def make_line_region_files(extraction_input, out_dir, cameras=None, lines='sky',
                           fiber_step=1, ref_fiber=150, fiber_half_width=10,
                           sigdetect=5.0, fwhm=4.0):
    """Write ds9 region files of model-predicted line positions per camera.

    For each fibre, predicts the raw-detector (x, y) of each line — sky-template
    lines (``lines='sky'``, from the frame itself) or catalog arc lines
    (``lines='arc'``) — by inverting that fibre's xshift and reading the trace
    y-position. Load the .reg over the matching camera extension of the raw
    MEF (or flat-corrected) image: offsets between crosses and observed lines
    ARE the xshift error, fibre by fibre.

    One file per camera: ``{out_dir}/lines_{cam}.reg`` (image coordinates).
    Returns list of paths written.
    """
    scidict, _ = _load(extraction_input)
    extractions = scidict['extractions']
    metadata = scidict.get('metadata', [{} for _ in extractions])
    os.makedirs(out_dir, exist_ok=True)

    written = []
    for obj, meta in zip(extractions, metadata):
        cam = _camera_key(meta)
        if cameras is not None and cam not in cameras:
            continue
        counts = getattr(obj, 'counts', None)
        xshift = getattr(obj, 'xshift', None)
        trace = getattr(obj, 'trace', None)
        traces = getattr(trace, 'traces', None) if trace is not None else None
        if (counts is None or xshift is None or traces is None
                or np.count_nonzero(xshift) == 0
                or np.count_nonzero(np.nan_to_num(counts)) == 0):
            continue

        nfib, nx = xshift.shape
        x = np.arange(nx, dtype=float)

        if lines == 'sky':
            targets = _sky_template_peaks(obj, nfib, ref_fiber, fiber_half_width,
                                          sigdetect, fwhm)
            if targets is None:
                continue
        else:
            channel = meta.get('channel', 'red')
            catalog_pixels = _load_peak_catalog_for_regions(channel)
            ref_xs = xshift[min(ref_fiber, nfib - 1), :]
            targets = np.interp(catalog_pixels, x, ref_xs)

        path = os.path.join(out_dir, f"lines_{cam}.reg")
        n_pts = 0
        with open(path, 'w') as fh:
            fh.write("# Region file format: DS9\n")
            fh.write(f"# Model-predicted {lines} line positions — camera {cam}\n")
            fh.write("global color=green point=cross width=1\nimage\n")
            for i in range(0, nfib, max(1, fiber_step)):
                xs_i = xshift[i, :]
                if not np.all(np.diff(xs_i) > 0):
                    continue
                px_pred = np.interp(targets, xs_i, x, left=np.nan, right=np.nan)
                for xp in px_pred:
                    if not np.isfinite(xp):
                        continue
                    col = int(round(xp))
                    if 0 <= col < traces.shape[1] and i < traces.shape[0]:
                        yp = traces[i, col]
                        if np.isfinite(yp):
                            # ds9 image coords are 1-indexed
                            fh.write(f"point({xp + 1:.2f},{yp + 1:.2f})\n")
                            n_pts += 1
        print(f"  {cam}: {n_pts} predicted positions -> {path}")
        written.append(path)
    return written


def _load_peak_catalog_for_regions(channel):
    """Catalog peak pixels for region files (avoids importing Arc at module load)."""
    from llamas_pyjamas.Arc.arcSurface import _load_peak_catalog
    return _load_peak_catalog(channel, use_unidentified=True)


# ---------------------------------------------------------------------------
# Product 4: summary CSV
# ---------------------------------------------------------------------------

def write_wavelength_qa_summary(qa_dir, per_camera_rows, run_label='',
                                csv_name='wavelengthQA_summary.csv'):
    """Write the per-camera scorecard CSV and print an aligned console table.

    ``per_camera_rows``: list of dicts already merged per camera (missing keys
    are written as empty fields). Returns the CSV path.
    """
    os.makedirs(qa_dir, exist_ok=True)
    csv_path = os.path.join(qa_dir, csv_name)
    with open(csv_path, 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        for row in per_camera_rows:
            out = {}
            for k in CSV_COLUMNS:
                v = row.get(k, '')
                if isinstance(v, float):
                    out[k] = '' if not np.isfinite(v) else f"{v:.4f}"
                else:
                    out[k] = v
            writer.writerow(out)

    # Console table (compact subset)
    cols = ['camera', 'xsh_ctr', 'xsh_edge', 'ratio', 'n_ident',
            'sky_ctr', 'sky_edge', 'flagged']
    print(f"\nWavelength QA summary ({run_label or 'run'}):")
    print("  " + "  ".join(f"{c:>9s}" for c in cols))
    for row in per_camera_rows:
        cam = f"{row.get('bench','?')}{row.get('side','?')}_{row.get('channel','?')}"
        def _f(key):
            v = row.get(key, '')
            return f"{v:9.3f}" if isinstance(v, float) and np.isfinite(v) else f"{'—':>9s}"
        print("  " + "  ".join([
            f"{cam:>9s}", _f('xshift_dev_rms_center_pix'), _f('xshift_dev_rms_edge_pix'),
            _f('xshift_edge_center_ratio'),
            f"{row.get('n_identity_fallback', '—'):>9}",
            _f('sky_rms_center'), _f('sky_rms_edge'),
            f"{row.get('n_sky_quality_flagged', '—'):>9}",
        ]))
    print(f"  -> {csv_path}\n")
    return csv_path


# ---------------------------------------------------------------------------
# Single-file HTML report
# ---------------------------------------------------------------------------

_CHANNEL_ORDER = {'red': 0, 'green': 1, 'blue': 2}


def _row_sort_key(row):
    return (_CHANNEL_ORDER.get(row.get('channel'), 3),
            str(row.get('bench', '')), str(row.get('side', '')))


def _fmt_num(v, nd=2):
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float) and np.isfinite(v):
        return f"{v:.{nd}f}"
    return '&mdash;'


def write_wavelength_qa_html(qa_dir, rows, cam_sections, run_label='',
                             html_name=None):
    """Write ONE self-contained HTML report: scorecard table + per-camera figures.

    ``rows``: per-camera metric dicts (CSV schema).
    ``cam_sections``: {cam_key: {'xshift_b64', 'sky_b64', 'notes': [str,...]}}.
    Images are embedded as base64 data URIs — no external files. Returns path.
    """
    os.makedirs(qa_dir, exist_ok=True)
    if html_name is None:
        html_name = f"{run_label or 'run'}_wavelengthQA.html"
    path = os.path.join(qa_dir, html_name)

    rows = sorted(rows, key=_row_sort_key)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

    def _cell(v, warn=False, nd=2):
        style = ' style="background:#ffd9d9;font-weight:bold"' if warn else ''
        return f"<td{style}>{_fmt_num(v, nd)}</td>"

    table_rows = []
    for r in rows:
        cam = f"{r.get('bench','?')}{r.get('side','?')}_{r.get('channel','?')}"
        ratio = r.get('xshift_edge_center_ratio', float('nan'))
        improv = r.get('sky_improvement', float('nan'))
        flagged = r.get('n_sky_quality_flagged', 0) or 0
        n_ident = r.get('n_identity_fallback', 0) or 0
        table_rows.append(
            "<tr>"
            f'<td><a href="#{cam}">{cam}</a></td>'
            f"<td>{r.get('nfibers','')}</td>"
            + _cell(n_ident, warn=isinstance(n_ident, (int, np.integer)) and n_ident > 0)
            + _cell(r.get('xshift_dev_rms_center_pix', float('nan')))
            + _cell(r.get('xshift_dev_rms_edge_pix', float('nan')))
            + _cell(ratio, warn=isinstance(ratio, float) and np.isfinite(ratio) and ratio > 2)
            + _cell(r.get('max_fiber_discontinuity_pix', float('nan')))
            + _cell(r.get('sky_rms_before', float('nan')), nd=1)
            + _cell(r.get('sky_rms_after', float('nan')), nd=1)
            + _cell(improv, warn=isinstance(improv, float) and np.isfinite(improv) and improv < 1)
            + _cell(r.get('sky_rms_center', float('nan')), nd=1)
            + _cell(r.get('sky_rms_edge', float('nan')), nd=1)
            + _cell(flagged, warn=isinstance(flagged, (int, np.integer)) and flagged > 0)
            + "</tr>")

    sections = []
    for r in rows:
        cam = f"{r.get('bench','?')}{r.get('side','?')}_{r.get('channel','?')}"
        sec = cam_sections.get(cam, {})
        parts = [f'<h2 id="{cam}">{cam}</h2>']
        for note in sec.get('notes', []):
            parts.append(f'<p class="note">{note}</p>')
        for key, alt in (('xshift_b64', 'xshift structure'),
                         ('arc_b64', 'arc-line residuals'),
                         ('sky_b64', 'sky-line residuals')):
            b64 = sec.get(key)
            if b64:
                parts.append(
                    f'<img alt="{cam} {alt}" src="data:image/png;base64,{b64}">')
        parts.append('<p><a href="#top">&uarr; back to summary</a></p>')
        sections.append('\n'.join(parts))

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Wavelength QA — {run_label}</title>
<style>
 body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 24px;
        max-width: 1200px; }}
 table {{ border-collapse: collapse; font-size: 13px; }}
 th, td {{ border: 1px solid #ccc; padding: 3px 8px; text-align: right; }}
 th {{ background: #f0f0f0; position: sticky; top: 0; }}
 td:first-child {{ text-align: left; }}
 img {{ max-width: 100%; border: 1px solid #ddd; margin: 6px 0; }}
 h2 {{ border-top: 2px solid #444; padding-top: 12px; margin-top: 28px; }}
 .note {{ color: #888; font-style: italic; }}
 .legend {{ font-size: 12px; color: #555; }}
</style></head><body id="top">
<h1>Wavelength / xshift QA &mdash; {run_label}</h1>
<p class="legend">Generated {ts}. Source: {rows[0].get('source_file','') if rows else ''}.
Red cells: identity-fallback fibres present, xshift edge/center ratio &gt; 2,
sky made worse (improvement &lt; 1), or sky_quality-flagged fibres. Click a camera to jump
to its figures.</p>
<table>
<tr><th>camera</th><th>nfib</th><th>n_ident</th>
<th>xsh rms ctr (px)</th><th>xsh rms edge (px)</th><th>edge/ctr</th><th>max fib jump (px)</th>
<th>sky rms before</th><th>sky rms after</th><th>improv &times;</th>
<th>sky ctr</th><th>sky edge</th><th>flagged</th></tr>
{''.join(table_rows)}
</table>
{''.join(sections)}
</body></html>
"""
    with open(path, 'w') as fh:
        fh.write(html)
    logger.info(f"waveQA: wrote report {path}")
    return path


# ---------------------------------------------------------------------------
# Orchestrator + CLI
# ---------------------------------------------------------------------------

def run_wavelength_qa(extraction_file, qa_dir=None, run_label=None,
                      arc_qa_records=None, keep_pngs=False):
    """Run all wavelength QA products on one extraction pickle.

    Writes ONE self-contained HTML report (scorecard + all per-camera figures,
    images embedded) plus the machine-readable CSV. Set ``keep_pngs=True`` to
    additionally write the individual PNG files.

    ``arc_qa_records`` (optional): per-fibre records collected from refineArcX;
    consumed by the arc-residual QA once available (Phase B). Currently unused
    when None — the CSV arc columns stay empty.

    Returns the merged per-camera stats dict.
    """
    scidict, src_path = _load(extraction_file)
    qa_dir = _resolve_qa_dir(qa_dir, src_path if src_path != '<in-memory>' else '.')
    if run_label is None:
        run_label = os.path.basename(str(extraction_file)).split('_extract')[0]

    emit = 'both' if keep_pngs else 'b64'
    xstats = xshift_structure_qa(scidict, qa_dir=qa_dir, label=run_label, emit=emit)
    sstats = sky_line_residual_qa(scidict, qa_dir=qa_dir, label=run_label, emit=emit)
    astats = (arc_residual_qa(arc_qa_records, qa_dir=qa_dir, label=run_label,
                              emit=emit)
              if arc_qa_records else {})

    metadata = scidict.get('metadata', [])
    ts = datetime.now(timezone.utc).isoformat(timespec='seconds')
    rows = []
    cam_sections = {}
    for meta in metadata:
        cam = _camera_key(meta)
        row = {
            'run_label': run_label,
            'timestamp_utc': ts,
            'source_file': os.path.basename(str(extraction_file)),
            'bench': meta.get('bench', ''),
            'side': meta.get('side', ''),
            'channel': meta.get('channel', ''),
            'nfibers': meta.get('nfibers', ''),
        }
        skip_keys = ('figure', 'figure_b64', 'note', 'n_oh_pixels')
        row.update({k: v for k, v in xstats.get(cam, {}).items()
                    if k not in skip_keys})
        row.update({k: v for k, v in sstats.get(cam, {}).items()
                    if k not in skip_keys + ('nfibers',)})
        row.update({k: v for k, v in astats.get(cam, {}).items()
                    if k not in skip_keys})
        rows.append(row)

        notes = []
        for name, st in (('xshift', xstats.get(cam, {})),
                         ('sky', sstats.get(cam, {}))):
            if st.get('note'):
                notes.append(f"{name}: {st['note']}")
        cam_sections[cam] = {
            'xshift_b64': xstats.get(cam, {}).get('figure_b64'),
            'arc_b64': astats.get(cam, {}).get('figure_b64'),
            'sky_b64': sstats.get(cam, {}).get('figure_b64'),
            'notes': notes,
        }

    write_wavelength_qa_summary(qa_dir, sorted(rows, key=_row_sort_key),
                                run_label=run_label,
                                csv_name=f"{run_label}_wavelengthQA.csv")
    report = write_wavelength_qa_html(qa_dir, rows, cam_sections,
                                      run_label=run_label)
    print(f"  QA report: {report}")
    return {'xshift': xstats, 'sky': sstats, 'qa_dir': qa_dir, 'report': report}


def main():
    parser = argparse.ArgumentParser(
        description='Standalone wavelength/xshift QA on extraction pickles.')
    parser.add_argument('pickles', nargs='+',
                        help='Extraction pickle(s), ideally *_sky1d_extractions.pkl')
    parser.add_argument('--qa-dir', default=None,
                        help='QA output directory (default: <run dir>/QA)')
    parser.add_argument('--label', default=None,
                        help='Run label prefix for figures/CSV (default: from filename)')
    parser.add_argument('--pngs', action='store_true',
                        help='Also write individual PNG files (default: HTML report only)')
    parser.add_argument('--sky-error', action='store_true',
                        help='Run the sky-line xshift-error QA (measures xshift error '
                             'from the frame\'s own sky lines) instead of the full report')
    parser.add_argument('--regions', default=None, metavar='DIR',
                        help='Write ds9 region files of model-predicted line positions '
                             'per camera into DIR (loads over the raw MEF extensions)')
    parser.add_argument('--regions-lines', default='sky', choices=['sky', 'arc'],
                        help="Line source for --regions (default: sky)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(message)s')
    for pkl in args.pickles:
        label = args.label
        if label and len(args.pickles) > 1:
            label = f"{label}_{os.path.basename(pkl).split('_')[1]}"
        print(f"\n=== Wavelength QA: {os.path.basename(pkl)} ===")
        if args.regions:
            make_line_region_files(pkl, args.regions, lines=args.regions_lines)
        if args.sky_error:
            sky_xshift_error_qa(pkl, qa_dir=args.qa_dir, label=label or 'skyXerr')
        if not (args.regions or args.sky_error):
            run_wavelength_qa(pkl, qa_dir=args.qa_dir, run_label=label,
                              keep_pngs=args.pngs)


if __name__ == '__main__':
    main()
