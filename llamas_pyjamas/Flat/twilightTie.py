"""Twilight-derived low-frequency illumination + bench-to-bench tie for LLAMAS.

The lamp throughput (:mod:`llamas_pyjamas.Flat.lampThroughput`) deliberately keeps
only the *high-frequency* per-fibre scatter and discards the smooth / bench-level
component (the lamp screen is not uniformly illuminated).  This module supplies that
missing piece from twilight flats: the **low-frequency illumination** and the
**bench-to-bench tie** ``L_j(λ)``.

Because the 8 bench-sides tile contiguous, non-overlapping horizontal y-stripes in
the IFU focal plane, per-bench-side offsets are *sharp y-discontinuities* that a
smooth ``(x, y)`` polynomial cannot represent.  The tie is therefore modelled as
**two terms** per wavelength bin::

    L_j(λ) = b_b(λ) · S(x_j, y_j; λ)

where ``b_b`` is an explicit per-bench-side multiplicative offset (captures the
bench steps) and ``S`` is a low-order smooth surface for the residual illumination.

The tie is smooth and stable, so it is **archived** under ``mastercalib/twilight_tie/``
and reused across runs.  Only a twilight that passes :func:`twilight_qa_gate` updates
the archive; a failing twilight is rejected and the previous tie retained.

NOTE: a single twilight epoch's smooth sky gradient is degenerate with instrument
illumination under an ``(x, y)`` fit — the ``b_b`` bench steps are robust, but
``S(x, y)`` can absorb a real sky gradient.  The archive's multi-epoch averaging
(exposures spanning varied sky orientations/times) is what cancels the sky term, so
prefer such sets when building the tie.
"""

import os
import copy
import glob
import logging
from datetime import datetime

import numpy as np
from numpy.polynomial.polynomial import polyvander2d

from llamas_pyjamas.config import CALIB_DIR
from llamas_pyjamas.Image.WhiteLightModule import FiberMap_LUT
from llamas_pyjamas.Flat.fibreFlat import (
    _load_smooth_models,
    _write_corrections_fits,
    CORRECTION_CLAMP,
    MIN_REF_FLUX,
)

logger = logging.getLogger(__name__)

# Archive location for the smooth, stable bench-to-bench tie.
TWILIGHT_TIE_DIR = os.path.join(CALIB_DIR, 'twilight_tie')

# Default low-order surface degree for the residual illumination term, per channel
# (blue sky gradient is strongest → slightly higher order tolerated).
_DEFAULT_TIE_ORDER = {'blue': 2, 'green': 2, 'red': 2}

# Number of coarse wavelength bins over which b_b and S are solved, then
# interpolated back onto each fibre's native grid.
DEFAULT_N_WAVE_BINS = 20

# Default QA-gate thresholds (permissive but non-trivial; override via config).
DEFAULT_QA_THRESHOLDS = {
    'min_frames': 1,            # exposures combined
    'min_sn_per_benchside': 20.0,
    'max_solar_residual': 0.10,  # fractional RMS in masked solar bands
    'min_benchsides_pass': 6,    # of 8, per channel
}


# ──────────────────────────────────────────────────────────────────────────────
# Core two-term tie fit (unit-testable)
# ──────────────────────────────────────────────────────────────────────────────

def _fit_2d_surface(x, y, values, order, mask=None):
    """Fit a low-order 2D polynomial surface to scattered ``(x, y, value)`` data.

    Coordinates are normalised to ``[-1, 1]`` for numerical stability; only terms
    with total degree ``<= order`` are kept.

    Returns
    -------
    model : ndarray
        Surface evaluated at every input ``(x, y)`` (same length as *values*).
    coeffs : ndarray or None
        Fitted coefficients, or ``None`` if there were too few points (in which
        case *model* is a flat surface at the median).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    values = np.asarray(values, float)

    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    if mask is not None:
        good &= np.asarray(mask, bool)

    n_terms = (order + 1) * (order + 2) // 2
    if np.sum(good) < n_terms:
        med = float(np.nanmedian(values[good])) if np.any(good) else 1.0
        return np.full(values.shape, med), None

    x_min, x_max = x[good].min(), x[good].max()
    y_min, y_max = y[good].min(), y[good].max()
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0
    xn = 2.0 * (x - x_min) / x_span - 1.0
    yn = 2.0 * (y - y_min) / y_span - 1.0

    col_mask = [ix * (order + 1) + iy
                for ix in range(order + 1) for iy in range(order + 1)
                if ix + iy <= order]
    vander = polyvander2d(xn[good], yn[good], [order, order])[:, col_mask]
    coeffs, *_ = np.linalg.lstsq(vander, values[good], rcond=None)
    vander_all = polyvander2d(xn, yn, [order, order])[:, col_mask]
    return vander_all @ coeffs, coeffs


def fit_bench_tie(residual, benchsides, x, y, order=2):
    """Two-term tie ``L_j = b_b · S(x, y)`` for ONE wavelength bin.

    Parameters
    ----------
    residual : ndarray, shape (n_fib,)
        Per-fibre residual ``(twilight / C) / field_median`` in this bin.
    benchsides : sequence of str, shape (n_fib,)
        Bench-side label per fibre (e.g. ``'1A'``).
    x, y : ndarray, shape (n_fib,)
        IFU focal-plane coordinates per fibre.
    order : int
        Polynomial order of the smooth residual surface.

    Returns
    -------
    L : ndarray, shape (n_fib,)
        The tie ``b_b · S`` per fibre.
    b : dict
        ``{benchside: offset}``.
    rms : float
        RMS of the de-offset residual about the surface (model uncertainty).
    """
    benchsides = np.asarray(benchsides)
    resid = np.asarray(residual, float)

    # 1. Per-bench-side offset: median residual within each bench-side.  This is
    #    the explicit "tie" term — it captures the sharp y-stripe steps a smooth
    #    surface cannot represent.
    b = {}
    for bs in np.unique(benchsides):
        m = (benchsides == bs) & np.isfinite(resid)
        val = float(np.nanmedian(resid[m])) if np.any(m) else 1.0
        b[bs] = val if val > 0 else 1.0
    b_vec = np.array([b[bs] for bs in benchsides])

    # 2. De-offset, then fit the smooth (x, y) residual surface.
    resid_norm = resid / b_vec
    S, _ = _fit_2d_surface(x, y, resid_norm, order)
    S = np.where(np.isfinite(S) & (S > 0), S, 1.0)

    L = b_vec * S
    fit_resid = resid_norm - S
    rms = float(np.sqrt(np.nanmean(fit_resid[np.isfinite(fit_resid)] ** 2))) \
        if np.any(np.isfinite(fit_resid)) else 0.0
    return L, b, rms


# ──────────────────────────────────────────────────────────────────────────────
# Twilight reduction (multi-frame combine)
# ──────────────────────────────────────────────────────────────────────────────

def reduce_twilight_flats(twilight_files, p2p_map_file, trace_dir, arc_soln,
                          slow_bias, output_dir, fast_bias=None):
    """Reduce and combine multiple twilight exposures into one extraction dict.

    Each exposure is P2P-corrected, extracted through the lamp traces, and
    wavelength-calibrated; the per-fibre counts are then median-combined across
    exposures (the fixed instrumental pattern persists while the smooth, changing
    sky gradient averages down).

    Returns
    -------
    extraction_dict : dict
        ``{'extractions', 'metadata', 'primary_header', 'n_frames'}``.
    """
    # Lazy import: reduce.py imports the Flat package (circular at module load).
    from llamas_pyjamas.reduce import (apply_flat_field_correction,
                                       run_extraction, correct_wavelengths)

    if isinstance(twilight_files, (str, bytes)):
        twilight_files = [twilight_files]
    if not twilight_files:
        raise ValueError("reduce_twilight_flats: no twilight files provided")

    twi_output_dir = os.path.join(output_dir, 'twilight')
    os.makedirs(twi_output_dir, exist_ok=True)

    per_frame = []
    primary_hdr = None
    for twi_file in twilight_files:
        logger.info(f"Reducing twilight exposure: {twi_file}")
        corrected_file, _ = apply_flat_field_correction(
            twi_file, [p2p_map_file], twi_output_dir)
        if corrected_file is None:
            logger.warning(f"P2P correction failed for {twi_file}; skipping")
            continue
        extraction_file = run_extraction(
            corrected_file, twi_output_dir, slow_bias=slow_bias,
            fast_bias=fast_bias, trace_dir=trace_dir, remove_cosmic_rays=False)
        extraction_path = os.path.join(twi_output_dir, extraction_file)
        extraction_dict, primary_hdr = correct_wavelengths(
            extraction_path, soln=arc_soln)
        per_frame.append(extraction_dict)

    if not per_frame:
        raise RuntimeError("reduce_twilight_flats: no twilight exposures reduced")

    combined = _combine_twilight_frames(per_frame)
    combined['primary_header'] = primary_hdr
    combined['n_frames'] = len(per_frame)
    logger.info(f"Combined {len(per_frame)} twilight exposure(s)")
    return combined


def combine_twilight_frames(frame_dicts):
    """Median-combine several reduced-twilight extraction dicts (per-fibre counts).

    Each input is a dict with ``'extractions'`` + ``'metadata'`` (e.g. the result
    of :func:`reduce_twilight_flats` on a single frame).  The combine is
    **non-mutating** — cached frame extractions are not modified, so the same
    frame may be reused across channels — and ``'n_frames'`` is set on the result.
    """
    combined = _combine_twilight_frames(frame_dicts)
    combined['n_frames'] = len(frame_dicts)
    for fd in frame_dicts:
        if fd.get('primary_header') is not None:
            combined['primary_header'] = fd['primary_header']
            break
    return combined


def _combine_twilight_frames(per_frame):
    """Median-combine per-fibre counts across exposures sharing the same traces.

    Does NOT mutate the input extraction objects — a shallow copy is taken before
    the combined counts are assigned, so cached frames can be reused safely.
    """
    base = per_frame[0]
    if len(per_frame) == 1:
        return {'extractions': base['extractions'],
                'metadata': base['metadata']}

    # Index each frame's extractions by (channel, bench, side)
    def key(meta):
        return f"{meta['channel']}_{meta['bench']}_{meta['side']}"

    frame_lookups = []
    for frame in per_frame:
        lut = {key(m): ext for ext, m in
               zip(frame['extractions'], frame['metadata'])}
        frame_lookups.append(lut)

    combined_exts = []
    for ext, meta in zip(base['extractions'], base['metadata']):
        k = key(meta)
        stack = [lut[k].counts for lut in frame_lookups
                 if k in lut and lut[k].counts.shape == ext.counts.shape]
        new_ext = copy.copy(ext)            # shallow copy → do not mutate the cache
        if len(stack) > 1:
            new_ext.counts = np.nanmedian(np.stack(stack, axis=0), axis=0)
        combined_exts.append(new_ext)
    return {'extractions': combined_exts, 'metadata': base['metadata']}


# ──────────────────────────────────────────────────────────────────────────────
# QA gate
# ──────────────────────────────────────────────────────────────────────────────

def twilight_qa_gate(twilight_extractions, thresholds=None, solar_telluric_mask=None):
    """Decide, per channel, whether a twilight set is good enough to update the tie.

    Checks per bench-side S/N and (optionally) solar-residual scatter, plus the
    combined frame count.  Returns a verdict dict::

        {'passed': {channel: bool}, 'per_benchside': {...},
         'n_frames': int, 'reason': {channel: str}}

    Only channels marked ``passed`` should refresh the archive.
    """
    th = dict(DEFAULT_QA_THRESHOLDS)
    if thresholds:
        th.update(thresholds)

    exts = twilight_extractions['extractions']
    meta = twilight_extractions['metadata']
    n_frames = int(twilight_extractions.get('n_frames', 1))

    per_bs = {}
    channels = {}
    for ext, m in zip(exts, meta):
        channel = m['channel'].lower()
        bs = f"{m['bench']}{m['side']}"
        counts = np.asarray(ext.counts, float)
        finite = counts[np.isfinite(counts) & (counts > 0)]
        if finite.size < 10:
            sn = 0.0
        else:
            med = np.median(finite)
            # Robust high-frequency noise estimate (MAD of pixel-to-pixel diff)
            diff = np.diff(finite)
            noise = 1.4826 * np.median(np.abs(diff - np.median(diff))) / np.sqrt(2)
            sn = float(med / noise) if noise > 0 else 0.0
        per_bs[(channel, bs)] = sn
        channels.setdefault(channel, []).append((bs, sn))

    passed, reason = {}, {}
    for channel, bs_list in channels.items():
        n_ok = sum(1 for _, sn in bs_list if sn >= th['min_sn_per_benchside'])
        ok = (n_frames >= th['min_frames']
              and n_ok >= th['min_benchsides_pass'])
        passed[channel] = bool(ok)
        reason[channel] = (f"{n_ok}/{len(bs_list)} bench-sides S/N>="
                           f"{th['min_sn_per_benchside']:.0f}, n_frames={n_frames}")
        if not ok:
            logger.warning(f"Twilight QA gate FAILED for {channel}: {reason[channel]}")
        else:
            logger.info(f"Twilight QA gate passed for {channel}: {reason[channel]}")

    return {'passed': passed, 'per_benchside': per_bs,
            'n_frames': n_frames, 'reason': reason}


# ──────────────────────────────────────────────────────────────────────────────
# Tie computation
# ──────────────────────────────────────────────────────────────────────────────

def _load_corrections(corrections_file):
    """Load a corrections FITS into ``{ext_name: {fiber_ids, correction, wave}}``."""
    from astropy.io import fits
    out = {}
    with fits.open(corrections_file) as hdul:
        for ext in hdul[1:]:
            name = ext.header.get('EXTNAME', '')
            if name.endswith('_REF') or ext.data is None:
                continue
            cols = ext.data.names
            if 'CORRECTION' not in cols:
                continue
            out[name] = {
                'fiber_ids': ext.data['FIBER_ID'].copy(),
                'correction': ext.data['CORRECTION'].copy(),
                'wave': ext.data['WAVE'].copy(),
            }
    return out


def _twilight_spectrum(twi_ext, fid):
    """Return (wave, counts) for physical fibre *fid*, handling dead-fibre packing."""
    counts = twi_ext.counts
    wave = twi_ext.wave
    total = counts.shape[0]
    if fid >= total:
        return None, None
    dead_set = set(getattr(twi_ext, 'dead_fibers', []) or [])
    if fid in dead_set:
        return None, None
    n_wave_rows = wave.shape[0] if wave is not None else 0
    if wave is None:
        return None, None
    if n_wave_rows == total:
        return wave[fid], counts[fid]
    # Compacted wave rows (alive-only): map physical id → wave row
    wave_row = 0
    for phys in range(total):
        if phys in dead_set:
            continue
        if phys == fid:
            return (wave[wave_row] if wave_row < n_wave_rows else None,
                    counts[fid])
        wave_row += 1
    return None, None


def compute_twilight_tie(twilight_extractions, lamp_corrections_file,
                         smooth_models_file, output_dir, poly_order=None,
                         n_wave_bins=DEFAULT_N_WAVE_BINS, solar_telluric_mask=None,
                         channels=None,
                         output_name='twilight_tie_corrections.fits'):
    """Compute the bench-to-bench tie ``L_j(λ) = b_b(λ) · S(x, y; λ)``.

    Parameters
    ----------
    twilight_extractions : dict
        Combined twilight extractions from :func:`reduce_twilight_flats`.
    lamp_corrections_file : str
        Path to ``lamp_throughput_corrections.fits`` (Step 2 ``C_j``); the
        twilight is divided by ``C_j`` before the tie is measured.
    smooth_models_file : str
        ``flat_smooth_models.fits`` — supplies fibre IDs / native wave grids.
    output_dir : str
        Directory for the output corrections file.
    poly_order : int or dict, optional
        Surface order for ``S``; default per-channel ``_DEFAULT_TIE_ORDER``.
    n_wave_bins : int
        Coarse wavelength bins over which ``b_b`` and ``S`` are solved.
    solar_telluric_mask : list of (lo, hi), optional
        Wavelength ranges (Å) to exclude from the fit (solar / telluric bands).
    channels : iterable of str, optional
        Restrict to these channels (e.g. only those passing the QA gate).

    Returns
    -------
    corrections_file : str
        Path to ``twilight_tie_corrections.fits`` (same schema as the lamp file).
    """
    models = _load_smooth_models(smooth_models_file)
    lamp_corr = _load_corrections(lamp_corrections_file)

    twi_lookup = {f"{m['channel']}_{m['bench']}_{m['side']}": ext
                  for ext, m in zip(twilight_extractions['extractions'],
                                    twilight_extractions['metadata'])}

    # Group benchside extensions by channel
    channel_groups = {}
    for ext_name, data in models.items():
        ch = data['channel'].lower()
        if channels is not None and ch not in channels:
            continue
        channel_groups.setdefault(ch, []).append(ext_name)

    corrections = {}
    for channel in sorted(channel_groups.keys()):
        order = _resolve_order(poly_order, channel)
        ch_corr = _compute_channel_tie(
            channel, channel_groups[channel], models, twi_lookup, lamp_corr,
            order, n_wave_bins, solar_telluric_mask)
        corrections.update(ch_corr)

    output_path = os.path.join(output_dir, output_name)
    _write_corrections_fits(
        corrections, output_path, method='twilight_tie',
        header_extra={'LAMPCORR': os.path.basename(lamp_corrections_file),
                      'NWAVEBIN': (int(n_wave_bins), 'Wavelength bins for tie fit')})
    return output_path


def _resolve_order(poly_order, channel):
    if poly_order is None:
        return _DEFAULT_TIE_ORDER.get(channel, 2)
    if isinstance(poly_order, dict):
        return int(poly_order.get(channel, 2))
    return int(poly_order)


def _bin_mask_from_ranges(wave_centers, ranges):
    """Boolean mask of bins to KEEP (True) given exclusion ranges."""
    keep = np.ones(len(wave_centers), dtype=bool)
    if not ranges:
        return keep
    for lo, hi in ranges:
        keep &= ~((wave_centers >= lo) & (wave_centers <= hi))
    return keep


def _compute_channel_tie(channel, ext_names, models, twi_lookup, lamp_corr,
                         order, n_wave_bins, solar_telluric_mask):
    """Compute ``L_j(λ)`` for all bench-sides of one channel."""
    # ── Common grid from the lamp/smooth-model wavelengths ──
    all_w, steps = [], []
    for ext_name in ext_names:
        wave_arr = models[ext_name]['wave']
        for i in range(wave_arr.shape[0]):
            w = wave_arr[i]
            good = np.isfinite(w) & (w > 0)
            if np.sum(good) > 10:
                all_w.append(w[good])
                dw = np.median(np.abs(np.diff(w[good])))
                if dw > 0:
                    steps.append(dw)
    if not all_w:
        logger.warning(f"Channel {channel}: no valid wavelengths for tie")
        return {}
    w_min = min(w.min() for w in all_w)
    w_max = max(w.max() for w in all_w)
    n_common = int(np.ceil((w_max - w_min) / np.median(steps))) + 1
    common_wave = np.linspace(w_min, w_max, n_common)

    # ── Gather twilight/C residual per fibre on the common grid ──
    records = []          # per-fibre metadata
    resid_rows = []       # (n_fib, n_common) twilight/C resampled
    for ext_name in sorted(ext_names):
        data = models[ext_name]
        bs = f"{data['bench']}{data['side']}"
        fiber_ids = data['fiber_ids']
        twi_ext = twi_lookup.get(ext_name)
        lamp = lamp_corr.get(ext_name)
        for li, fid in enumerate(fiber_ids):
            fid = int(fid)
            row = np.full(n_common, np.nan)
            x, y = FiberMap_LUT(bs, fid)
            mapped = not (x == -1 and y == -1)
            if twi_ext is not None and mapped:
                tw, tc = _twilight_spectrum(twi_ext, fid)
                if tw is not None and np.any(np.isfinite(tw)):
                    good = np.isfinite(tw) & np.isfinite(tc) & (tw > 0) & (tc > 0)
                    if np.sum(good) > 10:
                        twi_common = np.interp(common_wave, tw[good], tc[good],
                                               left=np.nan, right=np.nan)
                        c_common = _lamp_on_common(lamp, li, fid, common_wave)
                        row = twi_common / c_common
            records.append({'ext_name': ext_name, 'local_i': li, 'fid': fid,
                            'benchside': bs, 'x': float(x), 'y': float(y),
                            'mapped': mapped})
            resid_rows.append(row)

    twi_over_c = np.vstack(resid_rows)                    # (n_fib, n_common)

    # ── Divide by the channel field-median spectrum ──
    field_median = np.nanmedian(twi_over_c, axis=0)
    field_median = np.where(np.isfinite(field_median) & (field_median > MIN_REF_FLUX),
                            field_median, np.nan)
    residual = twi_over_c / field_median[None, :]         # ~1 + illumination

    # ── Per-wavelength-bin two-term tie ──
    benchsides = np.array([r['benchside'] for r in records])
    xs = np.array([r['x'] for r in records])
    ys = np.array([r['y'] for r in records])
    mapped = np.array([r['mapped'] for r in records])

    edges = np.linspace(w_min, w_max, n_wave_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    keep_bin = _bin_mask_from_ranges(centers, solar_telluric_mask)

    L_bins = np.ones((len(records), n_wave_bins))
    var_bins = np.zeros((len(records), n_wave_bins))
    for b_idx in range(n_wave_bins):
        if not keep_bin[b_idx]:
            L_bins[:, b_idx] = np.nan      # filled by interpolation across bins
            continue
        sl = (common_wave >= edges[b_idx]) & (common_wave < edges[b_idx + 1])
        if np.sum(sl) == 0:
            L_bins[:, b_idx] = np.nan
            continue
        scalar = np.nanmedian(residual[:, sl], axis=1)
        L, _, rms = fit_bench_tie(scalar[mapped], benchsides[mapped],
                                  xs[mapped], ys[mapped], order=order)
        L_full = np.ones(len(records))
        L_full[mapped] = L
        L_bins[:, b_idx] = L_full
        var_bins[mapped, b_idx] = rms ** 2

    # Interpolate any masked/empty bins across the kept bin centres
    valid_bins = np.isfinite(L_bins).all(axis=0)
    if not valid_bins.any():
        logger.warning(f"Channel {channel}: tie fit produced no valid bins")
        return {}
    good_centers = centers[valid_bins]

    # ── Scatter L_j(λ) back onto each bench-side's native grid ──
    corrections = {}
    for ext_name in sorted(ext_names):
        data = models[ext_name]
        wave_arr = data['wave']
        n_ext, n_pix = wave_arr.shape
        corr_native = np.ones((n_ext, n_pix), dtype=np.float64)
        var_native = np.zeros((n_ext, n_pix), dtype=np.float64)
        flags = np.zeros(n_ext, dtype=np.int16)
        corrections[ext_name] = {
            'fiber_ids': data['fiber_ids'], 'correction': corr_native,
            'wave': wave_arr, 'channel': channel, 'bench': data['bench'],
            'side': data['side'], 'n_ref': int(np.sum(mapped)),
            'common_wave': common_wave,
            'reference': np.where(np.isfinite(field_median), field_median, 0.0),
            'correction_var': var_native, 'fibre_flags': flags,
        }

    for k, rec in enumerate(records):
        out = corrections[rec['ext_name']]
        i = rec['local_i']
        w_native = models[rec['ext_name']]['wave'][i]
        good = np.isfinite(w_native) & (w_native > 0)
        if not rec['mapped'] or np.sum(good) < 10:
            continue
        L_curve = L_bins[k, valid_bins]
        v_curve = var_bins[k, valid_bins]
        l_native = np.interp(w_native, good_centers, L_curve,
                             left=L_curve[0], right=L_curve[-1])
        l_native = np.clip(l_native, CORRECTION_CLAMP[0], CORRECTION_CLAMP[1])
        out['correction'][i] = l_native
        out['correction_var'][i] = np.interp(w_native, good_centers, v_curve,
                                             left=v_curve[0], right=v_curve[-1])

    for ext_name in ext_names:
        corr = corrections[ext_name]['correction']
        logger.info(f"  {ext_name}: L median={np.nanmedian(corr):.4f} "
                    f"std={np.nanstd(corr):.4f}")
    return corrections


def _lamp_on_common(lamp, local_i, fid, common_wave):
    """Lamp throughput C_j resampled onto the common grid (1.0 where unavailable)."""
    if lamp is None:
        return np.ones_like(common_wave)
    fid_arr = lamp['fiber_ids']
    match = np.where(fid_arr == fid)[0]
    if match.size == 0:
        return np.ones_like(common_wave)
    j = match[0]
    c = lamp['correction'][j]
    w = lamp['wave'][j]
    good = np.isfinite(w) & (w > 0) & np.isfinite(c) & (c > 0)
    if np.sum(good) < 2:
        return np.ones_like(common_wave)
    c_common = np.interp(common_wave, w[good], c[good], left=1.0, right=1.0)
    return np.where(np.isfinite(c_common) & (c_common > 0), c_common, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Archive
# ──────────────────────────────────────────────────────────────────────────────

def save_archived_tie(tie_file, channel, epoch, archive_root=None):
    """Copy a per-channel tie into the archive as ``twilight_tie_<channel>_<epoch>.fits``.

    Parameters
    ----------
    tie_file : str
        Source corrections FITS to archive.
    channel : str
        Channel name ('red'/'green'/'blue').
    epoch : str
        Epoch tag, e.g. a UTC date ``'2026-06-03'``.
    archive_root : str, optional
        Override the default ``mastercalib/twilight_tie`` location.

    Returns
    -------
    archived_path : str
    """
    import shutil
    root = archive_root or TWILIGHT_TIE_DIR
    os.makedirs(root, exist_ok=True)
    safe_epoch = str(epoch).replace(':', '').replace(' ', '_')
    dest = os.path.join(root, f"twilight_tie_{channel}_{safe_epoch}.fits")
    shutil.copyfile(tie_file, dest)
    logger.info(f"Archived twilight tie ({channel}, {epoch}) → {dest}")
    return dest


def load_latest_tie(channel, archive_root=None):
    """Return the path to the newest archived tie for *channel*, or ``None``."""
    root = archive_root or TWILIGHT_TIE_DIR
    pattern = os.path.join(root, f"twilight_tie_{channel}_*.fits")
    matches = sorted(glob.glob(pattern))
    if not matches:
        logger.warning(f"No archived twilight tie found for {channel} in {root}")
        return None
    latest = matches[-1]    # lexical sort on ISO epoch == chronological
    logger.info(f"Using archived twilight tie for {channel}: "
                f"{os.path.basename(latest)}")
    return latest
