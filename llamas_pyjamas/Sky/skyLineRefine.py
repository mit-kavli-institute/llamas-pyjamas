"""pkl-domain (native-pixel / xshift) per-fibre OH sky-line refinement.

The base B-spline sky model (``skyModel_1d``) is built in the **xshift** coordinate, where each fibre's
sky lines are aligned to the pooled template by construction. The production *framework* refinement
(``Sky/skySubtract.py`` -> ``skyScale.scale_sky_per_fiber``) instead runs on the wavelength-resampled
RSS, so it re-encodes the per-fibre arc-solution (wavelength) error as a spurious fibre-to-fibre line
SHIFT -- a field-dependent artifact that does not average down in stacks (diagnosed in
``Sky/diagnosis/green_pkl_xshift_test.py``: the field-dependent shift collapses to one field-independent
curve in native/xshift space).

This module relocates that refinement into the pkl/extraction domain. It reuses the identical, tested
per-fibre fit (:func:`scale_sky_per_fiber` -- amplitude ``alpha`` + damped derivative shift/width basis)
but applies it to the **native-pixel** ``counts - sky`` residual against the native ``sky`` template, so
the derivative basis is d/d(native pixel) ~ d/d(xshift) and the line is already aligned. The fitted
line-only correction is folded into ``.sky`` (``sky += correction``) so the written RSS ``SKYSUB`` =
``counts - sky`` carries it. Reversible + config-gated (``sky_line_refine``, default off).

This is the amplitude+shift+width relocation. The static per-slit-position LSF-residual *template*
(the across-slit wing asymmetry left over at the ~12% level, see ``green_pkl_lsf_shape.py``) is a planned
augmentation layered on top of this.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.ndimage import label

from llamas_pyjamas.Sky.skyScale import _continuum, _line_mask

logger = logging.getLogger(__name__)

# defaults (pkl/native-pixel domain)
CONT_WIN = 6          # rolling-median continuum half-width (px); 2*win+1 kernel
SIGDETECT = 5.0       # OH-line detection sigma on the sky template
PAD = 3               # pixels of wing added on each side of a detected line segment
AMP_FLOOR = 50.0      # min line amplitude (counts) to refine a segment
AMP_CLIP = 0.6        # clip |alpha| (amplitude residual as a fraction of the line)


def _cfg(config, key, default):
    if config is None:
        return default
    try:
        return config.get(key, default)
    except AttributeError:
        return getattr(config, key, default)


def refine_fibre(counts_1d, sky_1d, *, cont_win=CONT_WIN, sigdetect=SIGDETECT, pad=PAD,
                 amp_floor=AMP_FLOOR, amp_clip=AMP_CLIP, deriv=True,
                 xshift_1d=None, tprof=None, offgrid=None, tmpl_clip=(-1.0, 3.0)):
    """Per-line OH residual correction for one fibre (native pixels / xshift domain).

    Detects OH lines in the base ``sky`` template, then for EACH line segment independently fits the
    base sky-subtracted residual ``counts - sky`` to the local line shape and its pixel-space
    derivatives ``d ~ alpha*S + beta*S'`` (per-line, unlike the RSS framework's single global per-fibre
    fit). When a static LSF-residual template ``tprof`` (on ``offgrid``, from :mod:`Sky.skyLineTemplate`)
    and ``xshift_1d`` are given, a ``delta * amp * T(xshift-line_centre)`` term REPLACES the width
    ``gamma*S''`` term (the empirical template subsumes it) with a per-line fitted, clipped amplitude
    ``delta``. Returns a (n_pix,) additive correction to ADD to the sky model (line-localised).
    """
    c = np.asarray(counts_1d, float); s = np.asarray(sky_1d, float)
    n = s.size
    corr = np.zeros(n)
    fin = np.isfinite(c) & np.isfinite(s)
    if fin.sum() < 20:
        return corr
    use_T = tprof is not None and xshift_1d is not None and offgrid is not None
    xs = np.asarray(xshift_1d, float) if use_T else None
    s_line = s - _continuum(np.where(fin, s, 0.0), cont_win)
    mask = _line_mask(s_line, sigdetect) & fin
    if not mask.any():
        return corr
    lab, nlab = label(mask)
    resid = c - s                                             # base sky-subtracted residual
    for k in range(1, nlab + 1):
        idx = np.where(lab == k)[0]
        lo = max(0, idx[0] - pad); hi = min(n, idx[-1] + 1 + pad)
        seg = np.arange(lo, hi)
        sl = s_line[seg]
        amp = np.nanmax(sl)
        if not np.isfinite(amp) or amp < amp_floor:
            continue
        d = resid[seg]
        good = np.isfinite(d) & np.isfinite(sl)
        if good.sum() < 5:
            continue
        # --- Phase A: alpha (amplitude) + beta (shift) + gamma (width) + continuum nuisance ---
        cols = [sl]
        if deriv:
            cols += [np.gradient(sl), np.gradient(np.gradient(sl))]
        n_corr = len(cols)
        xr = np.arange(seg.size, dtype=float); xr -= xr.mean()
        cols += [np.ones_like(sl), xr]                        # continuum nuisance: ABSORBED, NOT applied
        B = np.column_stack(cols)
        try:
            coef, *_ = np.linalg.lstsq(B[good], d[good], rcond=None)
        except np.linalg.LinAlgError:
            continue
        coef = np.asarray(coef, float)
        coef[0] = float(np.clip(coef[0], -amp_clip, amp_clip))   # clip amplitude residual
        cA = B[:, :n_corr] @ coef[:n_corr]                    # Phase-A line correction
        corr[seg] += cA
        # --- Phase B: fit the static LSF-residual template amplitude on the Phase-A LEFTOVER ---
        # (two-stage: keeps gamma; delta is a single well-determined param on what Phase A leaves)
        if use_T and np.all(np.isfinite(xs[seg])):
            x0 = xs[seg][int(np.nanargmax(sl))]               # line centre in xshift
            t_col = amp * np.interp(xs[seg] - x0, offgrid, tprof, left=0.0, right=0.0)
            rA = d - cA
            gg = good & np.isfinite(t_col)
            den = float(np.dot(t_col[gg], t_col[gg]))
            if den > 0:
                delta = float(np.clip(np.dot(rA[gg], t_col[gg]) / den, tmpl_clip[0], tmpl_clip[1]))
                corr[seg] += delta * t_col
    return corr


def refine_sky_lines_pkl(science, config, metadata=None, templates=None, offgrid=None):
    """Refine per-fibre OH-line residuals in the pkl/xshift domain (in place on each camera's ``.sky``).

    ``science`` is the list of per-camera extraction objects from a sky-subtracted pkl (``.sky`` already
    populated by ``skyModel_1d``; ``.counts`` present). For each live fibre :func:`refine_fibre` fits the
    per-line residual in NATIVE pixels and the correction is added to ``.sky`` so the written RSS
    ``SKYSUB = counts - sky`` carries it. Blue skips the derivative (amplitude-only) by default.

    ``templates`` (``{channel: {benchside: (n_slitbin, n_off)}}`` from
    :func:`Sky.skyLineTemplate.load_template`, keyed by channel since a benchside like ``1A`` exists in
    every channel) and ``offgrid`` enable the Phase-B static LSF-residual template: each fibre's slit-bin
    profile is passed to :func:`refine_fibre` and fit with a per-line amplitude. Returns ``science``
    (mutated); each camera gets ``.sky_line_refine`` ((n_fib,n_pix)) for provenance.
    """
    cont_win = int(_cfg(config, "sky_line_cont_window", CONT_WIN))
    sigdet = float(_cfg(config, "sky_line_sigdetect", SIGDETECT))
    amp_clip = float(_cfg(config, "sky_line_amp_clip", AMP_CLIP))
    skip_deriv = [str(c).lower() for c in _cfg(config, "sky_line_deriv_skip_colors", ["blue"])]
    n_applied = 0
    for j, e in enumerate(science):
        sky = getattr(e, "sky", None)
        counts = getattr(e, "counts", None)
        if sky is None or counts is None:
            continue
        sky = np.asarray(sky, float); counts = np.asarray(counts, float)
        if not np.any(np.isfinite(sky)) or np.nanmax(np.abs(np.nan_to_num(sky))) == 0:
            continue                                          # placeholder / missing camera
        md = metadata[j] if metadata is not None and j < len(metadata) else {}
        color = str(md.get("channel", getattr(e, "channel", ""))).lower()
        deriv = color not in skip_deriv
        cam = f"{md.get('bench', getattr(e, 'bench', ''))}{md.get('side', getattr(e, 'side', ''))}"
        T = templates.get(color, {}).get(cam) if templates is not None else None
        xshift = np.asarray(e.xshift, float) if T is not None and hasattr(e, "xshift") else None
        nfib = sky.shape[0]
        corr = np.zeros_like(sky)
        for fb in range(nfib):
            tprof = None
            if T is not None:
                sb = min(T.shape[0] - 1, int((fb / max(1, nfib - 1)) * T.shape[0]))
                tprof = T[sb]
            corr[fb] = refine_fibre(counts[fb], sky[fb], cont_win=cont_win, sigdetect=sigdet,
                                    amp_clip=amp_clip, deriv=deriv,
                                    xshift_1d=(xshift[fb] if xshift is not None else None),
                                    tprof=tprof, offgrid=offgrid)
        e.sky = sky + corr
        e.sky_line_refine = corr
        n_applied += 1
    logger.info("skyLineRefine: pkl-domain per-line OH refinement on %d/%d cameras (template=%s)",
                n_applied, len(science), templates is not None)
    return science


def _load_channel_templates(config):
    """Load per-channel LSF-residual templates from ``config['sky_line_template']`` (a path with a
    ``{channel}`` placeholder). Returns ``({channel: {cam: T}}, offgrid)`` or ``(None, None)``."""
    import os
    tpath = _cfg(config, "sky_line_template", None)
    if not tpath:
        return None, None
    from llamas_pyjamas.Sky.skyLineTemplate import load_template
    templates = {}; offgrid = None
    for chan in ("green", "red", "blue"):
        p = tpath.format(channel=chan) if "{channel}" in str(tpath) else tpath
        if os.path.exists(p):
            try:
                t, off = load_template(p, chan)
                templates[chan] = t; offgrid = off
            except Exception as exc:                          # noqa: BLE001
                logger.warning("skyLineRefine: failed to load template %s (%s)", p, exc)
    return (templates or None), offgrid


def apply_line_refine_file(pkl_file, config):
    """Load a ``*_sky1d[...]_extractions.pkl``, apply :func:`refine_sky_lines_pkl`, save a new pkl.

    Thin pkl-I/O wrapper for the pipeline (mirrors ``skyPedestal.apply_pedestal_file``). Loads the static
    LSF-residual template per channel from ``config['sky_line_template']`` if set. Returns the path to the
    written ``*_lr_extractions.pkl``. Leaves the input untouched so the step is reversible.
    """
    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions
    d = ExtractLlamas.loadExtraction(pkl_file)
    science = d["extractions"]
    hdr = d.get("primary_header")
    templates, offgrid = _load_channel_templates(config)
    refine_sky_lines_pkl(science, config, metadata=d.get("metadata"),
                         templates=templates, offgrid=offgrid)
    if hdr is not None:
        hdr["SKYLR"] = (True, "pkl-domain (xshift) OH-line refinement applied")
        hdr["SKYLRTMP"] = (templates is not None, "LSF-residual template applied")
    if "_extractions.pkl" in pkl_file:
        out = pkl_file.replace("_extractions.pkl", "_lr_extractions.pkl")
    else:
        out = pkl_file.replace(".pkl", "_lr.pkl")
    save_extractions(science, primary_header=hdr, savefile=out)
    logger.info("skyLineRefine: wrote %s", out)
    return out
