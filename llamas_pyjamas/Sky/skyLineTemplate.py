"""Static per-(camera, slit-position) OH-line LSF-residual template (Phase B).

After the Phase-A per-line refine (:mod:`Sky.skyLineRefine`), a small (~1-3 % of line amplitude) but
COHERENT residual remains: an across-slit **LSF wing asymmetry** (line tilt / optical aberration), which
flips sign from one slit edge to the other and is **static/field-independent** (a template built on some
fields cleans a held-out field). This module calibrates that residual SHAPE once, per camera, as a
function of slit position and offset-from-line-centre (in the fixed-xshift frame), from many
frames/fields with positive-outlier rejection (objects move between dithers -> rejected).

It is applied by :func:`Sky.skyLineRefine.refine_fibre` as a ``delta * T[cam, slit](offset)`` basis term
with a per-line FITTED amplitude (held-out tests: fixed amplitude gives ~+1-5 %, fitted ~+18-24 % on top
of Phase A -- static shape, variable amplitude). Diagnosis/validation: ``Sky/diagnosis/lsf_template_proto.py``.
"""
from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from astropy.io import fits

from llamas_pyjamas.Sky.skyLineRefine import refine_fibre

logger = logging.getLogger(__name__)

N_SLITBIN = 8                          # slit-position bins per camera
OFF = np.round(np.arange(-6.0, 6.0001, 0.3), 4)   # offset-from-line-centre grid (xshift px), 41 pts
AMP_FLOOR = 200.0                      # min line amplitude (counts) to contribute a profile
PAD_PIX = 12                           # native-pixel half-window around each line
# isolated bright singlets used to calibrate the LSF shape (avoid blends/doublets, e.g. NaD)
LINES_BY_CHANNEL = {"green": [5577.34, 6300.30], "red": [], "blue": []}


def _line_profile(w, c, sr, x, lam):
    """Normalised Phase-A residual profile of one line on the OFF grid (or None)."""
    if not (np.nanmin(w) < lam < np.nanmax(w)):
        return None
    pix = int(np.argmin(np.abs(w - lam)))
    lo = max(0, pix - PAD_PIX); hi = min(w.size, pix + PAD_PIX + 1)
    sl = slice(lo, hi); xr = np.arange(hi - lo, dtype=float)
    sky = sr[sl]; cc = c[sl]; xsh = x[sl]
    if not (np.all(np.isfinite(sky)) and np.all(np.isfinite(cc)) and np.all(np.isfinite(xsh))):
        return None
    a = np.polyfit([xr[0], xr[-1]], [np.median(sky[:2]), np.median(sky[-2:])], 1)
    amp = np.nanmax(sky - np.polyval(a, xr))
    if not np.isfinite(amp) or amp < AMP_FLOOR:
        return None
    resid = cc - sky
    b = np.polyfit([xr[0], xr[-1]], [np.median(resid[:2]), np.median(resid[-2:])], 1)
    resid = resid - np.polyval(b, xr)                       # local DC removed
    x0 = x[pix]
    return np.interp(OFF, xsh - x0, resid / amp, left=np.nan, right=np.nan)


def build_line_template(pkl_files, channel="green"):
    """Build the per-(camera, slit-bin) LSF-residual template from a list of sky1d pkls.

    Returns ``(templates, off, diag)`` where ``templates`` maps benchside -> ``(N_SLITBIN, len(OFF))``
    (normalised residual / line amp), ``off`` is :data:`OFF`, and ``diag`` carries per-camera counts.
    """
    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
    lines = LINES_BY_CHANNEL.get(channel, [])
    if not lines:
        logger.warning("skyLineTemplate: no calibration lines defined for channel %r", channel)
    stacks = defaultdict(lambda: [[] for _ in range(N_SLITBIN)])
    for f in pkl_files:
        d = ExtractLlamas.loadExtraction(f)
        sci = d["extractions"]; md = d.get("metadata", [])
        for i, e in enumerate(sci):
            m = md[i] if i < len(md) else {}
            if str(m.get("channel", getattr(e, "channel", ""))).lower() != channel:
                continue
            cam = f"{m.get('bench', getattr(e, 'bench', ''))}{m.get('side', getattr(e, 'side', ''))}"
            W = np.asarray(e.wave, float); C = np.asarray(e.counts, float)
            X = np.asarray(e.xshift, float); S = np.asarray(e.sky, float)
            nf = W.shape[0]
            for fb in range(nf):
                sr = S[fb] + refine_fibre(C[fb], S[fb])     # Phase-A refined sky
                sb = min(N_SLITBIN - 1, int((fb / max(1, nf - 1)) * N_SLITBIN))
                for lam in lines:
                    prof = _line_profile(W[fb], C[fb], sr, X[fb], lam)
                    if prof is not None:
                        stacks[cam][sb].append(prof)
    templates = {}; diag = {"n_prof": {}}
    for cam, bins in stacks.items():
        T = np.zeros((N_SLITBIN, OFF.size))
        n = np.zeros(N_SLITBIN, int)
        for sb in range(N_SLITBIN):
            if bins[sb]:
                A = np.vstack(bins[sb])
                T[sb] = np.nanmedian(A, axis=0)             # median rejects object positives
                n[sb] = A.shape[0]
        templates[cam] = np.nan_to_num(T)
        diag["n_prof"][cam] = n.tolist()
        logger.info("skyLineTemplate[%s]: %s built from %d profiles", channel, cam, int(n.sum()))
    return templates, OFF, diag


def save_template(path, templates, off, channel="green", diag=None):
    """Write the template to FITS: primary header carries the grid, one ImageHDU per camera."""
    hdus = [fits.PrimaryHDU()]
    h0 = hdus[0].header
    h0["CHANNEL"] = channel; h0["NSLITBIN"] = N_SLITBIN
    h0["NOFF"] = len(off); h0["OFF0"] = float(off[0]); h0["OFFSTEP"] = float(off[1] - off[0])
    for cam, T in templates.items():
        hdu = fits.ImageHDU(np.asarray(T, np.float32), name=cam)
        if diag and cam in diag.get("n_prof", {}):
            hdu.header["NPROF"] = int(np.sum(diag["n_prof"][cam]))
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    logger.info("skyLineTemplate: wrote %s (%d cameras)", path, len(templates))


def load_template(path, channel="green"):
    """Load a template written by :func:`save_template`. Returns ``(templates, off)``."""
    templates = {}
    with fits.open(path) as h:
        h0 = h[0].header
        n = int(h0["NOFF"]); off = h0["OFF0"] + h0["OFFSTEP"] * np.arange(n)
        for hdu in h[1:]:
            if hdu.data is not None:
                templates[hdu.name] = np.asarray(hdu.data, float)
    return templates, off
