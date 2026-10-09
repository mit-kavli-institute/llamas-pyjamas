"""Half-IFU mask awareness: header state, fibre halves, data check and header stamps.

LLAMAS can observe with a mask in the frontend that blocks one half of the IFU field. Consecutive
fibres on a benchside alternate between the two halves of the fibre map in x, so with the mask in
roughly every other trace on the detector is dark, and the faint-first sky selections would build
the sky model from the dark fibres (a near-zero sky that leaves the open half unsubtracted).

This module is the single place that knows:

* the header state (``HIERARCH FRONTEND IFU MASK-STATE``: EMPTY / HOME / TOP / BOTTOM; present but
  undefined in 2025 data) and the request it implies together with the config switches
  ``IFU_mask``, ``IFU_mask_force_off`` (hidden) and ``IFU_mask_half`` (hidden);
* which physical fibres each state blocks (a fibre-map ``xpos`` split);
* a data check on extracted counts that confirms -- or, for blank-header data, decides -- which
  half is dark (:func:`measure_halves`);
* the FITS keywords that carry the final decision from the extraction pickle to the RSS and cubes
  (:func:`stamp_header` / :func:`decision_from_header`), and the RSS MASK bit / FIBERMAP column
  that flag the blocked fibres (:data:`MASK_IFU_BLOCKED`, :data:`FIBERMAP_COLUMN`).

Geometry (``LUT/LLAMAS_FiberMap_rev04.dat``): low-x is ``xpos <= 23.0``, high-x is
``xpos >= 23.5`` (23 fibres of each per row); equivalently high-x <=> ``(physical id +
[side == 'B'])`` is even, for all 2392 fibres. Pipeline arrays are LIVE-row indexed, so rows must be
mapped to physical fibre ids with :func:`llamas_pyjamas.Utils.deadfibers.live_fibre_ids` (as
:func:`object_phys_ids` does) before any lookup here.

Run ``python -m llamas_pyjamas.Utils.ifuMask FILES...`` for a quick per-file assessment.
"""

import argparse
import logging
import os
import sys
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits

from llamas_pyjamas.config import LUT_DIR
from llamas_pyjamas.Utils.deadfibers import live_fibre_ids
from llamas_pyjamas.Utils.rayManager import _as_bool

logger = logging.getLogger(__name__)

MASK_STATE_KEY = 'HIERARCH FRONTEND IFU MASK-STATE'
FRONTEND_STATE_KEY = 'HIERARCH FRONTEND STATE'

FULL_FIELD_STATE = 'EMPTY'
# HOME: the mask covers the WHOLE field of view, so no sky light reaches the IFU (instrument team,
# 2026-10-09). Lamp calibrations are lit through the frontend CALIBRATION path and are taken with
# HOME as part of normal operations, as are biases and darks; only inputs that need sky light are
# unusable with HOME.
CLOSED_STATE = 'HOME'
HIGHX, LOWX = 'HIGHX', 'LOWX'
# Fibre-map half blocked by each half-field state. TOP was measured on 2026-10-09 on
# LLAMAS_2026-01-07_09-16-45.2 (1 s twilight): 16/16 red+green cameras dark in high-x with
# fingerprint-registered master traces, and 21/21 cameras (dark/bright 0.03) through the
# pipeline's own extraction. BOTTOM -> LOWX is assumed by symmetry (no BOTTOM frame was
# available); the data check warns if a frame disagrees.
BLOCKED_HALF = {'TOP': HIGHX, 'BOTTOM': LOWX}
KNOWN_STATES = (FULL_FIELD_STATE, CLOSED_STATE) + tuple(BLOCKED_HALF)

# RSS MASK bit for fibres behind the half-IFU mask. Bits 1 and 2 are the fibre flat's
# (Flat/fibreFlat.py MASK_FLAT_LAMP_ONLY / MASK_UNMAPPED_FIBRE_INTERP); every MASK consumer rejects
# any non-zero value.
MASK_IFU_BLOCKED = np.int16(4)
# FIBERMAP column carrying the same flag. Always written (all False when unmasked) so stacks of
# RSS files share one schema.
FIBERMAP_COLUMN = 'IFU_BLOCKED'

# Fibre-map x boundary between the halves (xpos takes values ... 23.0 | 23.5 ...).
X_SPLIT = 23.25
# Open fibres within this x distance of the boundary stay out of sky pools and out of the data
# check: the mask edge may partly vignette them.
SKY_EDGE_GUARD = 1.0

# Pre-flight roles. HOME is fatal for the sky-light roles and normal for the lamp roles; biases
# are never assessed.
TWILIGHT_ROLES = ('twilight', 'red_twilight', 'green_twilight', 'blue_twilight')
SKY_LIGHT_ROLES = ('science', 'standard', 'sky_frame') + TWILIGHT_ROLES
LAMP_ROLES = ('flat', 'red_flat', 'green_flat', 'blue_flat',
              'arc', 'red_arc', 'green_arc', 'blue_arc')
CALIBRATION_ROLES = LAMP_ROLES + TWILIGHT_ROLES
_CONFIG_ROLE_KEYS = (
    ('sky_frame_files', 'sky_frame'),
    ('red_flat_file', 'red_flat'), ('green_flat_file', 'green_flat'),
    ('blue_flat_file', 'blue_flat'),
    ('twilight_flat', 'twilight'), ('red_twilight_flat', 'red_twilight'),
    ('green_twilight_flat', 'green_twilight'), ('blue_twilight_flat', 'blue_twilight'),
    ('arc_file', 'arc'), ('red_arc_file', 'red_arc'), ('green_arc_file', 'green_arc'),
    ('blue_arc_file', 'blue_arc'),
)

# Requests: what the header + config ask for, before the data check.
REQ_CLOSED = 'CLOSED'      # HOME on a sky-light input: fatal
REQ_HEADER = 'HEADER'      # header says TOP/BOTTOM
REQ_CONFIG = 'CONFIG'      # blank header + IFU_mask = true
REQ_NONE = 'NONE'
REQ_FORCEOFF = 'FORCEOFF'  # hidden IFU_mask_force_off = true

# Data-check thresholds on per-camera dark/bright ratios of the per-half medians.
CONT_MASKED = 0.3          # continuum ratio below which a camera votes "masked"
CONT_UNMASKED = 0.6        # continuum ratio above which a camera votes "unmasked"
LINE_MASKED = 0.3          # sky-line ratio confirming the mask
LINE_UNMASKED = 0.6        # sky-line ratio refuting it (sky lines are uniform without a mask)
VOTE_FRAC = 0.8            # fraction of usable cameras that must agree
MIN_CAMERAS = 3            # fewer usable cameras -> inconclusive
MIN_LINE_CAMERAS = 2       # cameras with measurable sky lines needed for the line ratio
SIG_MIN = 5.0              # bright-half median must exceed this many robust errors
# Sky lines for the line ratio (air wavelengths, A): the green [OI] singlets (as
# Sky/skyLineTemplate.LINES_BY_CHANNEL) and bright red OH lines. A fixed list keeps the check
# offline-safe (no line-atlas download at run time); blue has no sky lines worth using.
SKY_LINES = {
    'green': (5577.34, 6300.30),
    'red': (6863.96, 7316.28, 7340.89, 7750.64, 7794.11, 7913.71, 7993.33, 8344.60, 8399.17,
            8430.17, 8827.10, 8885.85, 8943.40),
}
_PEAK_HALF = 2             # line-core window half-width [px]
_SIDE_IN, _SIDE_OUT = 6, 14   # continuum sidebands [px from the line pixel]


# ----------------------------------------------------------------------------
# Header state and request
# ----------------------------------------------------------------------------
def read_mask_state(header):
    """Normalised ``FRONTEND IFU MASK-STATE`` of a primary header.

    Returns 'EMPTY' / 'HOME' / 'TOP' / 'BOTTOM', ``''`` when the card is missing, blank or
    undefined (2025 data), or the stripped upper-case raw value when it is something else.
    """
    if header is None:
        return ''
    value = None
    for key in (MASK_STATE_KEY, 'FRONTEND IFU MASK-STATE'):
        try:
            value = header.get(key)
        except Exception:
            value = None
        if value is not None:
            break
    if value is None or isinstance(value, bool):
        return ''
    return str(value).strip().upper()


def _frontend_state(header):
    """Normalised ``FRONTEND STATE`` (e.g. 'CALIBRATION', 'ADC'), ``''`` when absent."""
    try:
        value = header.get(FRONTEND_STATE_KEY)
    except Exception:
        value = None
    return '' if value is None else str(value).strip().upper()


def request_for(state, ifu_mask=False, force_off=False, role='science'):
    """What the header ``state`` and the config ask for (see the REQ_* constants).

    HOME is CLOSED (fatal) for an input that needs sky light and NONE for a lamp calibration; it
    wins over ``force_off``. TOP/BOTTOM win over ``IFU_mask = false``. ``IFU_mask`` only acts on a
    blank, missing or unrecognised state.
    """
    if state == CLOSED_STATE:
        return REQ_NONE if role in LAMP_ROLES else REQ_CLOSED
    if force_off:
        return REQ_FORCEOFF
    if state in BLOCKED_HALF:
        return REQ_HEADER
    if state == FULL_FIELD_STATE:
        return REQ_NONE
    return REQ_CONFIG if ifu_mask else REQ_NONE


def config_switches(config):
    """``(IFU_mask, IFU_mask_force_off, IFU_mask_half)`` from a parsed pipeline config.

    ``IFU_mask_half`` is 'TOP' / 'BOTTOM' (the header vocabulary) or None.
    """
    config = config or {}
    ifu_mask = _as_bool(config.get('IFU_mask'), False)
    force_off = _as_bool(config.get('IFU_mask_force_off'), False)
    half = str(config.get('IFU_mask_half') or '').strip().upper() or None
    if half is not None and half not in BLOCKED_HALF:
        if half not in _WARNED_HALF:             # read once per frame: warn only once
            _WARNED_HALF.add(half)
            logger.warning("IFU_mask_half = %r not understood (use top or bottom); ignored",
                           config.get('IFU_mask_half'))
        half = None
    return ifu_mask, force_off, half


_WARNED_HALF = set()


# ----------------------------------------------------------------------------
# Fibre geometry
# ----------------------------------------------------------------------------
_XPOS = None


def _xpos_table():
    """Cached ``(benchside, physical fibre) -> xpos`` from the fibre-map LUT (the same file
    WhiteLightModule loads; read here directly to keep this module light)."""
    global _XPOS
    if _XPOS is None:
        from astropy.table import Table
        t = Table.read(os.path.join(LUT_DIR, 'LLAMAS_FiberMap_rev04.dat'),
                       format='ascii.fixed_width')
        _XPOS = {(str(b).strip(), int(f)): float(x)
                 for b, f, x in zip(t['bench'], t['fiber'], t['xpos'])}
    return _XPOS


def fibre_xpos(benchsides, phys_ids):
    """Fibre-map ``xpos`` per (benchside, physical id); NaN for fibres not in the LUT.

    ``benchsides`` is one benchside string (e.g. '1A') or one per fibre.
    """
    lut = _xpos_table()
    ids = np.atleast_1d(np.asarray(phys_ids)).astype(int)
    bs = np.broadcast_to(np.asarray(benchsides, dtype=str), ids.shape)
    return np.array([lut.get((str(b).strip(), int(i)), np.nan) for b, i in zip(bs, ids)],
                    dtype=float)


def fibre_halves(benchsides, phys_ids):
    """'HIGHX' / 'LOWX' per fibre ('' when the fibre is not in the LUT)."""
    x = fibre_xpos(benchsides, phys_ids)
    out = np.full(x.shape, '', dtype='<U5')
    out[x > X_SPLIT] = HIGHX
    out[x < X_SPLIT] = LOWX
    return out


def blocked_mask(benchsides, phys_ids, blocked):
    """True for fibres in the ``blocked`` half ('HIGHX' / 'LOWX'); all False for anything else.

    Fibres missing from the LUT are never flagged (no data is discarded on a lookup miss).
    """
    x = fibre_xpos(benchsides, phys_ids)
    if blocked == HIGHX:
        return x > X_SPLIT
    if blocked == LOWX:
        return x < X_SPLIT
    return np.zeros(x.shape, dtype=bool)


def sky_pool_mask(benchsides, phys_ids, blocked, guard=SKY_EDGE_GUARD):
    """True for fibres usable as sky with the ``blocked`` half masked: the open half, minus the
    fibres within ``guard`` of the mask edge. Fibres missing from the LUT are excluded."""
    x = fibre_xpos(benchsides, phys_ids)
    if blocked == HIGHX:
        open_half = x < X_SPLIT
    elif blocked == LOWX:
        open_half = x > X_SPLIT
    else:
        open_half = np.isfinite(x)
    return open_half & (np.abs(x - X_SPLIT) >= guard)


def object_benchside(obj):
    """Benchside string ('1A') of an extraction object (``bench`` may be int or str)."""
    return f"{getattr(obj, 'bench', '')}{getattr(obj, 'side', '')}"


def object_phys_ids(obj, n_rows=None):
    """Physical fibre id of each live row of an extraction object."""
    if n_rows is None:
        n_rows = np.asarray(obj.counts).shape[0]
    return live_fibre_ids(n_rows, getattr(obj, 'dead_fibers', None) or [])


def rss_blocked_rows(source, n_rows=None):
    """Per-row IFU-blocked flag of an RSS.

    ``source`` is an HDUList (FIBERMAP column, else MASK bit 4), a FIBERMAP table/record array or
    a dict of FIBERMAP columns (column only), or None. Returns all False (length ``n_rows`` when given) when nothing
    carries the flag.
    """
    fibermap = mask = None
    if isinstance(source, fits.HDUList):
        if 'FIBERMAP' in source:
            fibermap = source['FIBERMAP'].data
        if 'MASK' in source:
            mask = source['MASK'].data
    else:
        fibermap = source
    names = ()
    if isinstance(fibermap, dict):                 # e.g. CubeConstructor's FIBERMAP column dict
        names = tuple(fibermap.keys())
    elif fibermap is not None:
        names = getattr(fibermap, 'names', None) or getattr(getattr(fibermap, 'dtype', None),
                                                              'names', None) or ()
    if FIBERMAP_COLUMN in names:
        return np.asarray(fibermap[FIBERMAP_COLUMN], dtype=bool)
    if mask is not None:
        m = np.asarray(mask)
        if m.ndim == 2:
            return np.any((m.astype(np.int64) & int(MASK_IFU_BLOCKED)) != 0, axis=1)
    if n_rows is None:
        if isinstance(fibermap, dict):
            n_rows = len(next(iter(fibermap.values()), []))
        else:
            n_rows = len(fibermap) if fibermap is not None else 0
    return np.zeros(int(n_rows), dtype=bool)


# ----------------------------------------------------------------------------
# Data check
# ----------------------------------------------------------------------------
@dataclass
class HalfMeasurement:
    """Result of :func:`measure_halves`."""
    verdict: str = 'NOTRUN'           # MASKED-HIGHX / MASKED-LOWX / UNMASKED / INCONCLUSIVE
    dark: Optional[str] = None        # HIGHX / LOWX when masked
    cont_ratio: float = np.nan        # median over usable cameras of dark/bright continuum
    line_ratio: float = np.nan        # median sky-line ratio (NaN when unmeasurable)
    n_cameras: int = 0                # cameras with a significant bright half
    n_votes: int = 0                  # cameras voting masked with the majority dark half


def _half_stats(values, sel):
    """(median, robust error of the median) of ``values[sel]`` over finite entries."""
    v = np.asarray(values, float)[sel]
    v = v[np.isfinite(v)]
    if v.size < 5:
        return np.nan, np.nan
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    return med, 1.4826 * mad / np.sqrt(v.size)


def _sky_lines(channel, wmin, wmax):
    """Sky-line wavelengths [A] for the line ratio, at least 20 A inside ``[wmin, wmax]``."""
    return [w for w in SKY_LINES.get(channel, ()) if wmin + 20.0 < w < wmax - 20.0]


def _line_amplitudes(wave, flux, channel):
    """Per-fibre summed sky-line amplitude (mean over the line core minus the sideband median),
    NaN where unmeasurable. The core MEAN, not its maximum: the maximum of pure noise is biased
    high by the same amount in both halves, which would make a line-free frame (e.g. twilight)
    look like it has equal "lines" in the masked and open halves."""
    if wave is None:
        return None
    wave = np.asarray(wave, float)
    if wave.shape != flux.shape or not np.isfinite(wave).any():
        return None
    lines = _sky_lines(channel, np.nanmin(wave), np.nanmax(wave))
    if not lines:
        return None
    n, ncol = flux.shape
    amp = np.zeros(n)
    good = np.zeros(n, dtype=bool)
    peak_off = np.arange(-_PEAK_HALF, _PEAK_HALF + 1)
    side_off = np.concatenate([np.arange(-_SIDE_OUT, -_SIDE_IN + 1),
                               np.arange(_SIDE_IN, _SIDE_OUT + 1)])
    for lam in lines:
        dist = np.where(np.isfinite(wave), np.abs(wave - lam), np.inf)
        pix = np.argmin(dist, axis=1)
        ok = np.isfinite(dist[np.arange(n), pix]) & (pix >= _SIDE_OUT) & (pix < ncol - _SIDE_OUT)
        ip = np.clip(pix[:, None] + peak_off[None, :], 0, ncol - 1)
        isd = np.clip(pix[:, None] + side_off[None, :], 0, ncol - 1)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            peak = np.nanmean(np.take_along_axis(flux, ip, axis=1), axis=1)
            cont = np.nanmedian(np.take_along_axis(flux, isd, axis=1), axis=1)
        a = peak - cont
        ok &= np.isfinite(a)
        amp[ok] += a[ok]
        good |= ok
    amp[~good] = np.nan
    return amp


def measure_halves(extractions):
    """Classify a frame as half-masked or not from its extracted counts.

    Per real camera (placeholder cameras are skipped) the medians of the two fibre-map x halves
    are compared, without the fibres within :data:`SKY_EDGE_GUARD` of the edge, for (a) the
    throughput-corrected continuum and (b) the summed sky-line amplitude. The line ratio is what
    stops a bright object filling one half from mimicking the mask: sky lines are uniform across an
    unmasked field. ``extractions`` are ExtractLlamas-like objects (``counts``, ``wave``,
    ``relative_throughput``, ``dead_fibers``, ``channel``, ``bench``, ``side``).
    """
    cams = []
    for obj in extractions:
        counts = getattr(obj, 'counts', None)
        if counts is None:
            continue
        counts = np.asarray(counts, float)
        if counts.ndim != 2 or not np.isfinite(counts).any() \
                or np.allclose(np.nan_to_num(counts), 0.0):
            continue
        n = counts.shape[0]
        x = fibre_xpos(object_benchside(obj), object_phys_ids(obj, n))
        keep = np.abs(x - X_SPLIT) >= SKY_EDGE_GUARD
        sel = {HIGHX: keep & (x > X_SPLIT), LOWX: keep & (x < X_SPLIT)}
        tp = getattr(obj, 'relative_throughput', None)
        tp = np.ones(n) if tp is None else np.asarray(tp, float).ravel()
        if tp.size != n:
            tp = np.ones(n)
        tp = np.where(np.isfinite(tp) & (tp > 0), tp, 1.0)
        flux = counts / tp[:, None]
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            cont = np.nanmedian(flux, axis=1)
        stats = {h: _half_stats(cont, s) for h, s in sel.items()}
        (m_hi, e_hi), (m_lo, e_lo) = stats[HIGHX], stats[LOWX]
        if not (np.isfinite(m_hi) and np.isfinite(m_lo)):
            continue
        dark, bright = (HIGHX, LOWX) if m_hi < m_lo else (LOWX, HIGHX)
        m_b, e_b = stats[bright]
        if not (m_b > 0 and m_b > SIG_MIN * e_b):
            continue                              # no measurable light in this camera
        cam = {'dark': dark, 'ratio': stats[dark][0] / m_b, 'amp': None}
        amp = _line_amplitudes(getattr(obj, 'wave', None), flux,
                               str(getattr(obj, 'channel', '')).lower())
        if amp is not None:
            a = {h: _half_stats(amp, s) for h, s in sel.items()}
            a_b = max(a[HIGHX][0], a[LOWX][0])
            err_b = a[HIGHX][1] if a[HIGHX][0] >= a[LOWX][0] else a[LOWX][1]
            if np.isfinite(a_b) and a_b > 0 and a_b > SIG_MIN * err_b:
                cam['amp'] = (a[HIGHX][0], a[LOWX][0])
        cams.append(cam)

    meas = HalfMeasurement(verdict='INCONCLUSIVE', n_cameras=len(cams))
    if not cams:
        return meas
    meas.cont_ratio = float(np.median([c['ratio'] for c in cams]))
    if len(cams) < MIN_CAMERAS:
        return meas

    votes = Counter(c['dark'] for c in cams if c['ratio'] < CONT_MASKED)
    dark, n_votes = votes.most_common(1)[0] if votes else (None, 0)
    meas.n_votes = n_votes

    def _line_ratio(dark_half):
        vals = []
        for c in cams:
            if c['amp'] is None:
                continue
            a_hi, a_lo = c['amp']
            if dark_half == HIGHX:
                vals.append(a_hi / a_lo if a_lo > 0 else np.nan)
            elif dark_half == LOWX:
                vals.append(a_lo / a_hi if a_hi > 0 else np.nan)
            else:
                vals.append(min(a_hi, a_lo) / max(a_hi, a_lo) if max(a_hi, a_lo) > 0 else np.nan)
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.median(vals)) if len(vals) >= MIN_LINE_CAMERAS else np.nan

    if dark is not None and n_votes >= VOTE_FRAC * len(cams):
        meas.line_ratio = _line_ratio(dark)
        if not np.isfinite(meas.line_ratio) or meas.line_ratio < LINE_MASKED:
            meas.verdict, meas.dark = f'MASKED-{dark}', dark
        elif meas.line_ratio > LINE_UNMASKED:
            meas.verdict = 'UNMASKED'
        return meas
    meas.line_ratio = _line_ratio(None)
    if np.isfinite(meas.line_ratio) and meas.line_ratio > LINE_UNMASKED:
        meas.verdict = 'UNMASKED'
    elif sum(c['ratio'] > CONT_UNMASKED for c in cams) >= VOTE_FRAC * len(cams):
        meas.verdict = 'UNMASKED'
    return meas


# ----------------------------------------------------------------------------
# Final decision and header stamps
# ----------------------------------------------------------------------------
@dataclass
class IFUMaskDecision:
    """The per-exposure outcome carried by the header stamps."""
    state: str = ''
    request: str = REQ_NONE
    masked: bool = False
    blocked: Optional[str] = None     # HIGHX / LOWX
    source: str = 'NONE'              # TABLE / DATA / OVERRIDE / NONE
    verdict: str = 'NOTRUN'
    cont_ratio: float = np.nan
    line_ratio: float = np.nan
    notes: List[Tuple[int, str]] = field(default_factory=list)   # (logging level, message)


def finalise(request, state, meas=None, half_override=None):
    """Combine the request with the data check into an :class:`IFUMaskDecision`.

    * HEADER: the mask is on (the header is authoritative); the blocked half comes from
      :data:`BLOCKED_HALF` unless the data conclusively show the other half dark.
    * CONFIG: the data decide per frame (mixed nights work); when inconclusive the hidden
      ``IFU_mask_half`` is used, otherwise the frame is left unrestricted with a warning.
    * NONE / FORCEOFF: no handling; a frame that looks half-masked is reported.
    """
    meas = meas or HalfMeasurement()
    dec = IFUMaskDecision(state=state, request=request, verdict=meas.verdict,
                          cont_ratio=meas.cont_ratio, line_ratio=meas.line_ratio)
    looks_masked = meas.dark is not None
    if request == REQ_CLOSED:
        raise ValueError("IFU mask state HOME on a sky-light frame: no light reaches the IFU")
    if request == REQ_HEADER:
        table = BLOCKED_HALF[state]
        dec.masked, dec.blocked, dec.source = True, table, 'TABLE'
        if looks_masked and meas.dark != table:
            dec.blocked, dec.source = meas.dark, 'DATA'
            dec.notes.append((logging.WARNING,
                              f"header says {state} ({table} blocked) but the data show {meas.dark} "
                              f"dark (continuum ratio {meas.cont_ratio:.2f}); using the data -- "
                              f"please report this frame (the TOP/BOTTOM table may be wrong)"))
        elif meas.verdict == 'UNMASKED':
            dec.notes.append((logging.WARNING,
                              f"header says {state} but the frame does not look half-masked "
                              f"(continuum ratio {meas.cont_ratio:.2f}); applying the header"))
    elif request == REQ_CONFIG:
        if looks_masked:
            dec.masked, dec.blocked, dec.source = True, meas.dark, 'DATA'
            if half_override and BLOCKED_HALF[half_override] != meas.dark:
                dec.notes.append((logging.WARNING,
                                  f"IFU_mask_half = {half_override.lower()} disagrees with the data "
                                  f"({meas.dark} dark); using the data"))
        elif meas.verdict == 'UNMASKED':
            dec.notes.append((logging.INFO,
                              "IFU_mask = true but this frame does not look half-masked; "
                              "no mask handling"))
        elif half_override:
            dec.masked, dec.blocked, dec.source = True, BLOCKED_HALF[half_override], 'OVERRIDE'
        else:
            dec.notes.append((logging.WARNING,
                              "IFU_mask = true but the data cannot tell which half is blocked "
                              "(too little signal); no mask handling -- set IFU_mask_half = "
                              "top|bottom to force it"))
    elif looks_masked:
        if request == REQ_FORCEOFF:
            dec.notes.append((logging.INFO,
                              f"frame looks half-masked ({meas.dark} dark) but IFU_mask_force_off "
                              f"= true"))
        elif state == FULL_FIELD_STATE:
            dec.notes.append((logging.WARNING,
                              f"header says EMPTY but the frame looks half-masked ({meas.dark} dark, "
                              f"continuum ratio {meas.cont_ratio:.2f}); no mask handling"))
        else:
            dec.notes.append((logging.WARNING,
                              f"frame looks half-masked ({meas.dark} dark, continuum ratio "
                              f"{meas.cont_ratio:.2f}) but its header has no mask state; set "
                              f"IFU_mask = true if the mask was in"))
    return dec


def decide_for_frame(primary_header, extractions, config, role='science'):
    """Request from ``primary_header`` + config, data check on ``extractions``, final decision."""
    ifu_mask, force_off, half = config_switches(config)
    state = read_mask_state(primary_header)
    request = request_for(state, ifu_mask, force_off, role=role)
    return finalise(request, state, measure_halves(extractions), half)


def log_decision(dec, label, log=None):
    """Emit the decision's notes and one summary line."""
    log = log or logger
    for level, msg in dec.notes:
        log.log(level, "IFU mask %s: %s", label, msg)
    ratios = ''
    if np.isfinite(dec.cont_ratio):
        ratios = f" (continuum {dec.cont_ratio:.2f}"
        ratios += f", lines {dec.line_ratio:.2f})" if np.isfinite(dec.line_ratio) else ")"
    outcome = (f"{dec.blocked} blocked [{dec.source}]" if dec.masked else "no mask handling")
    log.info("IFU mask %s: state=%s request=%s data=%s%s -> %s", label, dec.state or 'BLANK',
             dec.request, dec.verdict, ratios, outcome)


def stamp_header(header, dec):
    """Write the decision into a primary header (the extraction pickle's, copied to the RSS)."""
    header['IFUMSTAT'] = ((dec.state or 'BLANK')[:60], 'FRONTEND IFU MASK-STATE as read')
    header['IFUMREQ'] = (dec.request, 'IFU mask request: HEADER/CONFIG/NONE/FORCEOFF')
    header['IFUMASK'] = (bool(dec.masked), 'Half-IFU mask handling applied')
    header['IFUMBLK'] = (dec.blocked if dec.masked else 'NONE',
                         'Blocked fibre-map half (HIGHX: xpos >= 23.5)')
    header['IFUMSRC'] = (dec.source if dec.masked else 'NONE',
                         'Blocked half from TABLE/DATA/OVERRIDE')
    header['IFUMDATA'] = (dec.verdict, 'IFU mask data-check verdict')
    for key, val, comment in (('IFUMRAT', dec.cont_ratio, 'Dark/bright half continuum ratio'),
                              ('IFUMLRAT', dec.line_ratio, 'Dark/bright half sky-line ratio')):
        if np.isfinite(val):
            header[key] = (round(float(val), 4), comment)
        elif key in header:
            del header[key]
    return header


STAMP_KEYS = ('IFUMSTAT', 'IFUMREQ', 'IFUMASK', 'IFUMBLK', 'IFUMSRC', 'IFUMDATA', 'IFUMRAT',
              'IFUMLRAT', 'NIFUMSK')


def stamp_items(header):
    """``{key: (value, comment)}`` of the IFU-mask stamps present in ``header`` (for copying to
    cube headers)."""
    if header is None:
        return {}
    return {k: (header[k], header.comments[k]) for k in STAMP_KEYS if k in header}


def stamp_items_from_file(path):
    """:func:`stamp_items` of a FITS file's primary header; ``{}`` if it cannot be read."""
    try:
        return stamp_items(fits.getheader(path, 0))
    except Exception:
        return {}


def file_is_half_masked(path):
    """True when a FITS file's primary header carries ``IFUMASK = T``."""
    try:
        return bool(fits.getval(path, 'IFUMASK', 0))
    except Exception:
        return False


def decision_from_header(header):
    """The stamped :class:`IFUMaskDecision`, or None when ``header`` carries no stamp.

    Only stamped headers act: calibration products and pickles written before the stamp exist are
    never treated as masked.
    """
    if header is None:
        return None
    try:
        if 'IFUMASK' not in header:
            return None
        masked = bool(header['IFUMASK'])
        blocked = str(header.get('IFUMBLK', 'NONE')).strip().upper()
    except Exception:
        return None
    if blocked not in (HIGHX, LOWX):
        masked, blocked = False, None
    return IFUMaskDecision(state=str(header.get('IFUMSTAT', '')).strip(),
                           request=str(header.get('IFUMREQ', REQ_NONE)).strip(),
                           masked=masked, blocked=blocked if masked else None,
                           source=str(header.get('IFUMSRC', 'NONE')).strip(),
                           verdict=str(header.get('IFUMDATA', 'NOTRUN')).strip(),
                           cont_ratio=float(header.get('IFUMRAT', np.nan)),
                           line_ratio=float(header.get('IFUMLRAT', np.nan)))


# ----------------------------------------------------------------------------
# Pre-flight assessment
# ----------------------------------------------------------------------------
@dataclass
class PreflightRow:
    role: str
    path: str
    state: str
    request: str
    frontend: str = ''


def _as_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [s.strip() for s in value.split(',') if s.strip()]
    return [str(v).strip() for v in value if str(v).strip()]


def input_roles(config):
    """``(role, path)`` for every input of a pipeline config that the mask can affect.

    Standards (``flux_standard_files``, already folded into ``science_files``) are tagged
    'standard'. Biases are not assessed.
    """
    stds = set(_as_list(config.get('flux_standard_files')))
    out = [('standard' if p in stds else 'science', p)
           for p in _as_list(config.get('science_files'))]
    for key, role in _CONFIG_ROLE_KEYS:
        out.extend((role, p) for p in _as_list(config.get(key)))
    return out


def assess_inputs(config, ifu_mask=False, force_off=False):
    """Read every input's mask state; one :class:`PreflightRow` per (role, file)."""
    rows = []
    for role, path in input_roles(config):
        try:
            hdr = fits.getheader(path, 0)
        except Exception:
            rows.append(PreflightRow(role, path, 'UNREADABLE', REQ_NONE))
            continue
        state = read_mask_state(hdr)
        rows.append(PreflightRow(role, path, state,
                                 request_for(state, ifu_mask, force_off, role=role),
                                 _frontend_state(hdr)))
    return rows


def closed_mask_failures(rows):
    """Rows whose input needs sky light but reports HOME (the whole field of view is covered)."""
    return [r for r in rows if r.request == REQ_CLOSED]


def format_closed_failure(failures):
    """Concise terminal message naming every file that failed the HOME check."""
    width = max(len(r.role) for r in failures)
    lines = [f"IFU mask closed (FRONTEND IFU MASK-STATE = HOME): no sky light reaches the IFU in "
             f"{len(failures)} input frame(s):"]
    lines += [f"  {r.role:<{width}}  {os.path.basename(r.path)}" for r in failures]
    lines.append("Remove these from the config and re-run.")
    return '\n'.join(lines)


def preflight_report(rows, ifu_mask=False, force_off=False):
    """``(summary, warnings, table_lines)`` for the pre-flight log and terminal."""
    warnings, table = [], []
    for r in rows:
        name = os.path.basename(r.path)
        table.append(f"  {r.role:<14} {(r.state or 'BLANK'):<8} {r.request:<8} {name}")
        if r.state == 'UNREADABLE':
            warnings.append(f"{r.role} {name}: header unreadable; IFU mask state unknown")
        elif r.state and r.state not in KNOWN_STATES:
            warnings.append(f"{r.role} {name}: unrecognised IFU mask state {r.state!r}; "
                            f"treated as missing")
        if r.state == CLOSED_STATE and r.role in LAMP_ROLES and r.frontend \
                and r.frontend != 'CALIBRATION':
            warnings.append(f"{r.role} {name}: mask HOME but FRONTEND STATE = {r.frontend} "
                            f"(not CALIBRATION) -- the lamp path may not be in use and the frame "
                            f"may be dark")
        if r.state in BLOCKED_HALF:
            if r.role in CALIBRATION_ROLES:
                warnings.append(f"{r.role} {name}: taken with the half-IFU mask in ({r.state}); "
                                f"full-IFU calibrations are assumed -- traces, fibre flats and arc "
                                f"solutions are not mask-aware yet")
            elif r.role == 'standard':
                warnings.append(f"flux standard {name}: half-IFU mask in ({r.state}); the "
                                f"sensitivity function may be biased if the star sits near the "
                                f"mask edge")
    sky_rows = [r for r in rows if r.role in ('science', 'standard', 'sky_frame')]
    n_empty_ignored = sum(1 for r in sky_rows if ifu_mask and r.state == FULL_FIELD_STATE)
    if n_empty_ignored:
        warnings.append(f"IFU_mask = true ignored for {n_empty_ignored} frame(s) whose header "
                        f"reports EMPTY (full field)")
    if ifu_mask and not force_off and not any(r.request == REQ_CONFIG for r in sky_rows):
        warnings.append("IFU_mask = true has no effect: every science/standard/sky frame reports "
                        "its mask state in the header")
    n_forced = sum(1 for r in sky_rows if r.request == REQ_FORCEOFF and r.state in BLOCKED_HALF)
    if n_forced:
        warnings.append(f"IFU_mask_force_off = true: half-mask handling DISABLED for {n_forced} "
                        f"frame(s) whose header reports TOP/BOTTOM")

    by_state = Counter(r.state for r in sky_rows if r.request == REQ_HEADER)
    n_cfg = sum(1 for r in sky_rows if r.request == REQ_CONFIG)
    n_cal_masked = sum(1 for r in rows if r.role in CALIBRATION_ROLES and r.state in BLOCKED_HALF)
    parts = [f"{len(sky_rows)} science/standard/sky frame(s) checked"]
    if by_state:
        parts.append("half-masked by header: " + ", ".join(f"{k} {v}"
                                                           for k, v in sorted(by_state.items())))
    if n_cfg:
        parts.append(f"{n_cfg} left to the data check (IFU_mask = true, blank header)")
    if not by_state and not n_cfg:
        parts.append("none half-masked")
    parts.append(f"{n_cal_masked} masked calibration(s)" if n_cal_masked
                 else "calibrations full-field")
    return "IFU mask: " + "; ".join(parts), warnings, table


# ----------------------------------------------------------------------------
# CLI: quick per-file assessment
# ----------------------------------------------------------------------------
def _cli_role(header):
    """Role of a raw frame from PRODCATG (CAL.R-BIA/DRK bias/dark, CAL.R-FLT/ARC lamp,
    CAL.R-SKY twilight, else science)."""
    cat = str(header.get('PRODCATG') or '').strip().upper()
    if cat.startswith(('CAL.R-BIA', 'CAL.R-DRK')):
        return 'bias/dark'
    if cat.startswith('CAL.R-FLT'):
        return 'flat'
    if cat.startswith('CAL.R-ARC'):
        return 'arc'
    if cat.startswith('CAL.R-SKY'):
        return 'twilight'
    return 'science'


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Report the half-IFU mask state (HIERARCH FRONTEND IFU MASK-STATE) of LLAMAS "
                    "frames. Exits 1 if a frame that needs sky light reports HOME.")
    ap.add_argument('files', nargs='+', help='raw LLAMAS MEF files')
    ap.add_argument('--ifu-mask', action='store_true',
                    help='treat blank-header frames as masked (the IFU_mask config switch)')
    args = ap.parse_args(argv)

    failures = []
    print(f"{'role':<10} {'state':<8} {'assessment':<44} file")
    for path in args.files:
        try:
            hdr = fits.getheader(path, 0)
        except Exception as exc:
            print(f"{'?':<10} {'?':<8} {'unreadable: ' + str(exc)[:32]:<44} {os.path.basename(path)}")
            continue
        state, role = read_mask_state(hdr), _cli_role(hdr)
        shown_role = 'lamp' if role in LAMP_ROLES else role
        if role == 'bias/dark':
            req = REQ_NONE
        else:
            req = request_for(state, args.ifu_mask, role=role)
        if state == CLOSED_STATE:
            text = ('normal (mask closed)' if req != REQ_CLOSED
                    else 'FAIL: mask closed, no sky light')
        elif req == REQ_HEADER:
            text = f'half-masked ({BLOCKED_HALF[state]} blocked)'
        elif req == REQ_CONFIG:
            text = 'half-masked via IFU_mask (half from data)'
        elif state == FULL_FIELD_STATE:
            text = 'full field'
        elif not state:
            text = 'blank header: full field unless IFU_mask'
        else:
            text = 'unrecognised state: treated as blank'
        print(f"{shown_role:<10} {(state or 'BLANK'):<8} {text:<44} {os.path.basename(path)}")
        if req == REQ_CLOSED:
            failures.append(PreflightRow(shown_role, path, state, req))
    if failures:
        print()
        print(format_closed_failure(failures))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
