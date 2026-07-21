"""
llamas_pyjamas.Sky.skyPedestal
==============================
Per-camera additive continuum pedestal (sky-refine Phase 3a) — **HYPOTHESIS UNDER TEST**.

The base B-spline sky model (``skyModel_1d``) captures the OH lines but leaves a small positive
additive continuum floor in each camera (Phase 2 diagnosis: worst benchsides 1A/2A in green, 1A in
red; ``Sky/diagnosis/DIAGNOSIS.md``). That floor is coherent per camera, so the rotated dithers
cross-hatch it into the stacked-image striping — and in blue the large flux calibration amplifies it.

This module estimates the floor from **known-blank fibres** and adds it to the sky model so it is
subtracted, in the **counts / pkl domain, BEFORE flux calibration**. It is a separate post-step:
``skyModel_1d`` is untouched, and the whole thing is **config-gated, default OFF** (``sky_pedestal``).

CENTRAL RISK — could subtract real diffuse emission. Two guards:
1. it is a **continuum** term (median-filtered), so narrow line emission (e.g. Lyα) is preserved;
2. it is measured only from **conservatively-selected blank fibres** (the faintest per camera).

Validate (Sky/DESIGN.md Phase 3) that it actually reduces the striping AND preserves real signal
before trusting it. If it does not, the whole ``pedestal-fix`` branch is discarded.

Public API
----------
estimate_pedestal(counts, sky, blank_mask, *, cont_window, clip_negative) -> pedestal[nwave]
apply_continuum_pedestal(science, config) -> list  (adds the pedestal to each camera's .sky in place)
"""

import logging
import numpy as np
from scipy.ndimage import median_filter

logger = logging.getLogger(__name__)

# Defaults (overridable via the pipeline config).
PEDESTAL_WINDOW = 51        # continuum median-filter width (pixels); >> OH/line widths
PEDESTAL_NFIBRES = 40       # scope='camera': the N faintest fibres per camera
SLIT_WINDOW = 15            # scope='slit': running-median half-width along the slit (fibres)
BLANK_PCT = 60.0            # scope='slit': blank = faintest this-percent of live fibres
MIN_BLANK = 5               # need at least this many blank fibres to estimate a pedestal


def estimate_pedestal(counts, sky, blank_mask, *, cont_window=PEDESTAL_WINDOW,
                      clip_negative=False):
    """Per-camera additive continuum pedestal from blank fibres.

    Parameters
    ----------
    counts, sky : array_like (n_fiber, n_wave)
        The camera's extracted counts and the base sky model already applied
        (per fibre ``sky = spline(xshift) * throughput``).
    blank_mask : array_like[bool] (n_fiber,)
        Fibres to measure the floor from (known blank — the faintest, so they
        carry sky + instrumental floor but no object).
    cont_window : int
        Median-filter width (pixels) that removes residual OH and any narrow
        line emission, leaving a smooth continuum — this is what preserves Lyα.
    clip_negative : bool
        If True, clip the pedestal at 0 (the diagnosed floor is positive; edge
        over-subtraction — a *negative* residual — is handled separately, not here).

    Returns
    -------
    np.ndarray (n_wave,)
        The additive continuum to ADD to the sky model (subtracted from the data).
    """
    counts = np.asarray(counts, float)
    sky = np.asarray(sky, float)
    resid = counts - sky                                   # residual after the base sky model
    idx = np.where(np.asarray(blank_mask, bool))[0]
    nwave = resid.shape[1]
    if idx.size < MIN_BLANK:
        logger.warning("skyPedestal: only %d blank fibres (<%d); pedestal=0", idx.size, MIN_BLANK)
        return np.zeros(nwave)
    win = max(3, int(cont_window) | 1)                     # odd window
    cont = np.empty((idx.size, nwave))
    for k, i in enumerate(idx):
        cont[k] = median_filter(np.nan_to_num(resid[i], nan=0.0), size=win, mode="nearest")
    ped = np.nanmedian(cont, axis=0)                       # robust across blank fibres
    ped = np.nan_to_num(ped, nan=0.0)
    if clip_negative:
        ped = np.clip(ped, 0.0, None)
    return ped


def estimate_slit_pedestal(counts, sky, blank_mask, *, cont_window=PEDESTAL_WINDOW,
                           slit_window=SLIT_WINDOW, clip_negative=False):
    """Per-fibre additive continuum pedestal, SMOOTH ALONG THE SLIT (scope='slit').

    The diagnosed residual floor is not constant per camera — it is a smooth, low-order profile
    along the slit (arch / tilt / step; Sky/DESIGN.md investigation (a)). Estimate it per fibre:
    take each *blank* fibre's continuum residual (median-filtered in wavelength, so narrow line
    emission never enters), then for EVERY fibre fit a LOCAL LINEAR profile over the blank fibres
    within ``slit_window`` of it in slit position — smooth along the slit, correct at the slit
    ends (a running median is biased there: its one-sided window pulls toward the interior,
    exactly where the real dips are steepest), and interpolated from blank neighbours under
    object fibres (an object's own continuum is never in the estimate).

    Returns
    -------
    np.ndarray (n_fiber, n_wave)
        The additive continuum to ADD to each fibre's sky model.
    """
    counts = np.asarray(counts, float)
    sky = np.asarray(sky, float)
    resid = counts - sky
    nfib, nwave = resid.shape
    idx = np.where(np.asarray(blank_mask, bool))[0]
    if idx.size < MIN_BLANK:
        logger.warning("skyPedestal(slit): only %d blank fibres (<%d); pedestal=0",
                       idx.size, MIN_BLANK)
        return np.zeros((nfib, nwave))
    win = max(3, int(cont_window) | 1)
    cont = np.empty((idx.size, nwave))
    for k, i in enumerate(idx):
        cont[k] = median_filter(np.nan_to_num(resid[i], nan=0.0), size=win, mode="nearest")
    W = max(3, int(slit_window))
    ped = np.empty((nfib, nwave))
    for i in range(nfib):
        d = idx - i
        sel = np.where(np.abs(d) <= W)[0]
        if sel.size < 4:                                    # sparse blanks: take the nearest few
            sel = np.argsort(np.abs(d))[:max(4, MIN_BLANK)]
        x = d[sel].astype(float)
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, cont[sel], rcond=None)   # (2, n_wave)
        ped[i] = coef[0]                                    # local linear fit evaluated AT fibre i
    ped = np.nan_to_num(ped, nan=0.0)
    if clip_negative:
        ped = np.clip(ped, 0.0, None)
    return ped


def _blank_mask_frac(sci, pct):
    """Blank fibres as the faintest ``pct`` percent of live fibres — spans the whole slit
    (needed for the along-slit running median), unlike a fixed dimmest-N which can cluster."""
    from llamas_pyjamas.Sky.skySelect import build_sky_mask
    counts = np.asarray(sci.counts, float)
    tp = getattr(sci, "relative_throughput", None)
    if tp is not None:
        tp = np.asarray(tp, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            bright = np.nansum(np.where(tp[:, None] > 0, counts / tp[:, None], np.nan), axis=1)
    else:
        bright = np.nansum(counts, axis=1)
    finite = np.isfinite(bright) & (bright != 0)
    dead = getattr(sci, "dead_fibers", None) or []
    for d in dead:
        if 0 <= d < finite.size:
            finite[d] = False
    if not finite.any():
        return finite
    cut = np.nanpercentile(bright[finite], float(pct))
    mask = finite & (bright <= cut)
    return build_sky_mask(method="manual", explicit=mask).mask


def _blank_mask(sci, n_fibres):
    """Conservative blank-fibre mask for one camera: the ``n_fibres`` faintest live fibres.

    Uses throughput-corrected white-light as the brightness proxy (matching the base sky model), via
    the shared provider so 'blank' means the same thing everywhere."""
    from llamas_pyjamas.Sky.skySelect import build_sky_mask
    counts = np.asarray(sci.counts, float)
    tp = getattr(sci, "relative_throughput", None)
    if tp is not None:
        tp = np.asarray(tp, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            bright = np.nansum(np.where(tp[:, None] > 0, counts / tp[:, None], np.nan), axis=1)
    else:
        bright = np.nansum(counts, axis=1)
    finite = np.isfinite(bright) & (bright != 0)
    dead = getattr(sci, "dead_fibers", None) or []
    for d in dead:
        if 0 <= d < finite.size:
            finite[d] = False
    return build_sky_mask(bright, finite, method="dimmest", n_fibres=int(n_fibres)).mask


def apply_continuum_pedestal(science, config):
    """Add a per-camera additive continuum pedestal to each science camera's ``.sky`` (in place).

    ``science`` is the list of per-camera extraction objects from a sky-subtracted pkl (``.sky``
    already populated by ``skyModel_1d``). Config keys (all optional):

    * ``sky_pedestal_scope``    'slit' (default: per-fibre profile, smooth along the slit —
                                matches the diagnosed residual shape) or 'camera' (one constant
                                profile per camera; falsified for the striping, kept for comparison)
    * ``sky_pedestal_window``   continuum median-filter width in wavelength (px), default 51
    * ``sky_pedestal_slit_window``  scope='slit': running-median half-width along the slit (fibres), default 15
    * ``sky_pedestal_blank_pct``    scope='slit': blank = faintest this-% of live fibres, default 60
    * ``sky_pedestal_nfibres``  scope='camera': blank fibres per camera, default 40
    * ``sky_pedestal_clip_negative``  clip pedestal at 0, default False

    Returns ``science`` (mutated). Each camera also gets a ``.sky_pedestal`` attribute
    ((n_wave,) for 'camera', (n_fib, n_wave) for 'slit') for provenance/diagnostics.
    Placeholder/empty cameras are skipped.
    """
    win = int(config.get("sky_pedestal_window", PEDESTAL_WINDOW))
    nfib = int(config.get("sky_pedestal_nfibres", PEDESTAL_NFIBRES))
    clip = bool(config.get("sky_pedestal_clip_negative", False))
    scope = str(config.get("sky_pedestal_scope", "slit")).lower()
    slit_win = int(config.get("sky_pedestal_slit_window", SLIT_WINDOW))
    blank_pct = float(config.get("sky_pedestal_blank_pct", BLANK_PCT))
    n_applied = 0
    for sci in science:
        counts = getattr(sci, "counts", None)
        sky = getattr(sci, "sky", None)
        if counts is None or sky is None:
            continue
        counts = np.asarray(counts, float)
        if not np.any(np.isfinite(counts)) or np.nanmax(np.abs(np.nan_to_num(counts))) == 0:
            continue                                       # placeholder / missing camera
        if scope == "slit":
            # smooth along-slit per-fibre profile (the diagnosed shape of the residual floor)
            blank = _blank_mask_frac(sci, blank_pct)
            ped = estimate_slit_pedestal(counts, np.asarray(sky, float), blank,
                                         cont_window=win, slit_window=slit_win,
                                         clip_negative=clip)
            sci.sky = np.asarray(sky, float) + ped         # (n_fib, n_wave)
        else:
            # scope='camera': one constant profile per camera (falsified for the striping;
            # kept for comparison/experimentation)
            blank = _blank_mask(sci, nfib)
            ped = estimate_pedestal(counts, np.asarray(sky, float), blank,
                                    cont_window=win, clip_negative=clip)
            sci.sky = np.asarray(sky, float) + ped[None, :]
        sci.sky_pedestal = ped
        n_applied += 1
    logger.info("skyPedestal: applied continuum pedestal (scope=%s) on %d/%d cameras "
                "(wave window=%d px)", scope, n_applied, len(science), win)
    return science


def apply_pedestal_file(sky1d_file, config):
    """Load a ``*_sky1d_extractions.pkl``, add the continuum pedestal, and save a new pkl.

    Thin pkl-I/O wrapper around :func:`apply_continuum_pedestal` for the pipeline. Returns the path
    to the written ``*_sky1dped_extractions.pkl`` (the RSS is then built from it). Leaves the input
    file untouched so the step is reversible."""
    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, save_extractions
    d = ExtractLlamas.loadExtraction(sky1d_file)
    science = d["extractions"]
    hdr = d.get("primary_header")
    apply_continuum_pedestal(science, config)
    if hdr is not None:
        hdr["SKYPED"] = (True, "per-camera additive continuum pedestal subtracted")
        hdr["SKYPEDWN"] = (int(config.get("sky_pedestal_window", PEDESTAL_WINDOW)),
                           "pedestal continuum window (px)")
    out = sky1d_file.replace("_sky1d_extractions.pkl", "_sky1dped_extractions.pkl")
    if out == sky1d_file:
        out = sky1d_file.replace(".pkl", "_ped.pkl")
    save_extractions(science, primary_header=hdr, savefile=out)
    logger.info("skyPedestal: wrote %s", out)
    return out
