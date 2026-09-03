"""
Calibration-frame assessment and recommendation for LLAMAS reductions.

Scans a directory of raw LLAMAS MEF files, scores every arc, LDLS flat and
twilight (sky) flat per channel, and recommends which file to assign to each
of the nine calibration attributes in ``Utils/reduxSetupGUI.py``
(red/green/blue x flat/twilight/arc). The goal is to predict, BEFORE a
multi-hour reduction, the two classic old-data failures:

* **Trace failure across a channel** - the pipeline traces fibres with
  ``find_peaks(comb, distance=5, height=100, prominence=500)`` on a 5-column
  median slice at the detector centre (``Trace/traceLlamasMaster.py``), and
  the trace validator accepts a benchside ONLY when the detected peak count
  exactly equals the expected fibre count. Faint fibres below the fixed
  prominence cut, or saturated (flat-topped, merging) fibre peaks, make the
  count wrong and the pipeline silently falls back to mastercalib traces -
  or writes no trace at all. This script reproduces the same peak detection
  on each candidate flat and predicts PASS/FAIL per benchside.

* **Arc refinement failure** - ``Arc/arcLlamas.refineArcX`` needs at least 6
  catalog lines matched within a 10 px shift budget per fibre. Weak lamp
  exposures (too few detectable lines), saturated lines, and epoch shifts
  larger than 10 px (e.g. blue ~ -206 px on 2024 commissioning data) all
  force per-fibre fallbacks. This script detects lines in each candidate
  arc, matches them against the ``LUT/{red,green,blue}_peaks.csv`` catalogs,
  and estimates the epoch shift.

READ-ONLY: no file in the input directory is opened for writing and nothing
is written there. Outputs (report + config snippet) go to --output-dir
(default: the current working directory).

Frame classification uses the primary-header PRODCATG when present
(CAL.R-FLT lamp flat, CAL.R-SKY twilight, CAL.R-ARC arc, CAL.R-BIA bias),
falling back to OBJECT free-text heuristics, then to the observing log
(--obs-log, GUI tab-separated or handwritten xlsx, matched by filename
timestamp) - old data commonly has OBJECT='None' and no PRODCATG, so the log
is often the only source of frame types. Files that remain unclassifiable
are listed as UNKNOWN for manual review. Blank READ-MDE comes from the log
when consistent, else is inferred statistically via the sibling
readmode_assess module.

Usage:
    python calib_assess.py /path/to/night
    python calib_assess.py /path/to/night --obs-log LLAMAS_obs-log_20241128.xlsx \\
        --output-dir ./assess --report assess.json -v

Exit codes:
    0  every attribute with candidates has a GOOD recommendation
    1  some attributes only have MARGINAL/POOR candidates
    2  at least one attribute has no viable candidate (or none at all)
    3  system error
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy.io import fits
from scipy.ndimage import percentile_filter, uniform_filter1d
from scipy.signal import find_peaks

# Bare sibling imports (same convention as fix_exposure_info.py): importing
# the llamas_pyjamas package would eagerly pull the full pipeline (Ray, PypeIt).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import readmode_assess  # noqa: E402
import obs_log          # noqa: E402

EXIT_OK, EXIT_MARGINAL, EXIT_UNVIABLE, EXIT_ERROR = 0, 1, 2, 3

# ---------------------------------------------------------------------------
# Constants mirrored from the pipeline (do not import pipeline modules).
# ---------------------------------------------------------------------------

# Expected fibre count per benchside - llamas_pyjamas/constants.py:6 (N_fib),
# identical to the dict used by Utils/utils.py:validate_and_fix_trace_fibres.
N_FIB = {'1A': 298, '1B': 300, '2A': 298, '2B': 297,
         '3A': 298, '3B': 300, '4A': 300, '4B': 298}

# Trace peak detection - Trace/traceLlamasMaster.py:640 and its edge cut.
TRACE_PEAK_KW = dict(distance=5, height=100, prominence=500)
TRACE_EDGE_LO, TRACE_EDGE_HI = 20, 2020
# Peaks with prominence in [PROM_CUT, PROM_MARGINAL) are "one bad night from
# failing"; peaks detected at PROM_FLOOR but below PROM_CUT are fibres the
# pipeline will drop (the documented F7 failure on red benchsides).
PROM_CUT, PROM_MARGINAL, PROM_FLOOR = 500.0, 1000.0, 150.0

SAT_LEVEL = 63000.0        # QA convention (qa_config_cal.yaml, extract_stats.py)
EXPECTED_SHAPE = (2048, 2048)

# Arc refinement - Arc/arcLlamas.py:162-164.
ARC_MIN_LINES = 6
ARC_MATCH_TOL = 10.0
ARC_MAX_SHIFT = 300        # px scanned when estimating the epoch shift
ARC_SIGDETECT = 5.0        # S/N used for line detection (pypeit sigdetect)
ARC_PROM_FLOOR = 25.0      # ADU; absolute prominence floor - real ThAr lines
                           # are 100s-1000s ADU, noise peaks are not
ARC_SCAN_TOL = 3.0         # px; tight tolerance for the shift scan so chance
                           # noise alignments cannot fabricate an epoch shift

PRODCATG_KIND = {
    'CAL.R-BIA': 'bias', 'CAL.R-DRK': 'dark', 'CAL.R-FLT': 'flat',
    'CAL.R-SKY': 'twilight', 'CAL.R-ARC': 'arc',
    'SCI.R-SL': 'science', 'SCI.R-DT': 'science',
}
KIND_PRODCATG = {'flat': 'CAL.R-FLT', 'twilight': 'CAL.R-SKY', 'arc': 'CAL.R-ARC'}

# Order matters: 'twilight'/'sky' must win over 'flat' ("twilight flat").
OBJECT_HEURISTICS = [
    ('arc', ('arc', 'thar', 'th-ar', 'wavelength')),
    ('twilight', ('twilight', 'twi ', 'twiflat', 'sky flat', 'skyflat', 'sky_flat')),
    ('flat', ('flat', 'ldls', 'dome', 'quartz')),
    ('bias', ('bias', 'zero')),
    ('dark', ('dark',)),
]

# Log-row classification: the log's object field is written by the observer
# during calibration blocks, so a bare 'sky' safely means twilight there
# (unlike a science OBJECT header, where 'sky' would be ambiguous).
LOG_HEURISTICS = [
    ('arc', ('arc', 'thar', 'th-ar')),
    ('twilight', ('twilight', 'twi', 'sky')),
    ('flat', ('flat', 'ldls', 'dome', 'quartz')),
    ('bias', ('bias', 'zero')),
    ('dark', ('dark',)),
]

# Files derived by the pipeline; never calibration candidates.
SKIP_NAME_TOKENS = ('white', 'diff', '_trimmed', 'corrected', 'master',
                    '_rss', '_extract', '_cube', 'whitelight')

CHANNELS = ('red', 'green', 'blue')
BENCHSIDES = tuple(N_FIB)

# The nine reduxSetupGUI attributes -> the exact config keys reduce.py reads.
ROLE_CONFIG_KEY = {
    ('red', 'flat'): 'red_flat_file',
    ('green', 'flat'): 'green_flat_file',
    ('blue', 'flat'): 'blue_flat_file',
    ('red', 'twilight'): 'red_twilight_flat',
    ('green', 'twilight'): 'green_twilight_flat',
    ('blue', 'twilight'): 'blue_twilight_flat',
    ('red', 'arc'): 'red_arc_file',
    ('green', 'arc'): 'green_arc_file',
    ('blue', 'arc'): 'blue_arc_file',
}

_LUT_DIR = Path(__file__).resolve().parents[1] / 'LUT'

VERDICT_ORDER = {'GOOD': 0, 'MARGINAL': 1, 'POOR': 2, 'UNUSABLE': 3}


class CalibAssessError(Exception):
    """System-level problem (bad input dir, missing LUT catalogs)."""


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------

def load_line_catalogs(lut_dir: Path | None = None) -> dict:
    """Load identified (Wavelength > 0) line pixel positions per channel."""
    lut_dir = Path(lut_dir) if lut_dir else _LUT_DIR
    catalogs = {}
    for channel in CHANNELS:
        path = lut_dir / f"{channel}_peaks.csv"
        if not path.is_file():
            raise CalibAssessError(f"line catalog not found: {path}")
        pix = []
        with open(path, newline='') as fh:
            for row in csv.DictReader(fh):
                wave = (row.get('Wavelength') or '').strip()
                try:
                    if wave and float(wave) > 0:
                        pix.append(float(row['Pixel']))
                except (TypeError, ValueError):
                    continue
        if not pix:
            raise CalibAssessError(f"no identified lines in {path}")
        catalogs[channel] = np.sort(np.asarray(pix))
    return catalogs


def load_level_bands(baselines_path=None) -> dict | None:
    """Per-detector expected edge-background / frame-level bands, by PRODCATG
    and readmode, from QA/baselines/qa_thresholds_derived.json. Optional -
    returns None (with level checks skipped) when unavailable."""
    path = (Path(baselines_path) if baselines_path
            else readmode_assess.DEFAULT_BASELINES)
    if not path.is_file() or path.suffix.lower() in ('.yaml', '.yml'):
        return None
    try:
        with open(path) as fh:
            return json.load(fh).get('per_detector')
    except (OSError, json.JSONDecodeError):
        return None


def load_dead_fibre_note(lut_dir: Path | None = None) -> str:
    lut_dir = Path(lut_dir) if lut_dir else _LUT_DIR
    try:
        with open(lut_dir / 'traceLUT.json') as fh:
            dead = json.load(fh).get('dead_fibers', {})
        if dead:
            parts = [f"{bs}:{len(v)}" for bs, v in sorted(dead.items())]
            return ("known dead fibres already folded into expected counts: "
                    + ", ".join(parts))
    except (OSError, json.JSONDecodeError):
        pass
    return ""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_frame(header) -> tuple[str, str]:
    """Return (kind, how) where kind is flat/twilight/arc/bias/dark/science/
    unknown and how records the evidence used."""
    prodcatg = str(header.get('PRODCATG', '') or '').strip().upper()
    if prodcatg in PRODCATG_KIND:
        return PRODCATG_KIND[prodcatg], f"PRODCATG={prodcatg}"
    obj = str(header.get('OBJECT', '') or '').strip().lower()
    if obj:
        for kind, tokens in OBJECT_HEURISTICS:
            if any(tok in obj for tok in tokens):
                return kind, f"OBJECT~'{obj}'"
    if prodcatg:
        return 'unknown', f"unrecognised PRODCATG={prodcatg}"
    return 'unknown', "no PRODCATG, OBJECT uninformative"


def _classify_log_row(row) -> str | None:
    text = ' '.join(filter(None, (row.object_name, row.obs_type))).lower()
    if not text:
        return None
    for kind, tokens in LOG_HEURISTICS:
        if any(tok in text for tok in tokens):
            return kind
    if row.obs_type and any(t in row.obs_type.lower()
                            for t in ('sci', 'std', 'pr')):
        return 'science'
    return None


def classify_from_log(name: str, log_index: dict) -> tuple[str, str, str | None,
                                                           float | None]:
    """Classify a file from its observing-log row(s), matched by filename
    stem. Returns (kind, how, readmode, exptime); conflicting rows for the
    same stem (data-entry duplicates) yield 'unknown'."""
    stem = obs_log.normalize_stem(name)
    rows = log_index.get(stem, []) if stem else []
    if not rows:
        return 'unknown', 'not in obs log', None, None
    kinds = {k for k in (_classify_log_row(r) for r in rows) if k}
    if not kinds:
        return 'unknown', 'obs-log row has no classifiable type', None, None
    if len(kinds) > 1:
        return ('unknown',
                f"conflicting obs-log rows ({len(rows)} entries)", None, None)
    kind = kinds.pop()
    modes = {r.read_mode for r in rows if r.read_mode}
    exps = {r.exp_time for r in rows if r.exp_time is not None}
    return (kind, f"obs log: {rows[0].describe()}",
            modes.pop() if len(modes) == 1 else None,
            exps.pop() if len(exps) == 1 else None)


def _exptime(header):
    for key in ('SEXPTIME', 'REXPTIME', 'EXPTIME'):
        try:
            value = header.get(key)
            if value is not None:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Per-extension metrics
# ---------------------------------------------------------------------------

_DATASEC_RE = re.compile(r'\[(\d+):(\d+),\s*(\d+):(\d+)\]')


def orient_extension(data: np.ndarray, header, channel: str) -> np.ndarray:
    """DATASEC trim + per-colour flips, mirroring File/llamasIO.py:155-200,
    so measured x positions live in the same frame as the LUT catalogs."""
    if 'DATASEC' in header:
        match = _DATASEC_RE.match(str(header['DATASEC']))
        if match:
            x1, x2, y1, y2 = map(int, match.groups())
            if x2 <= data.shape[1] and y2 <= data.shape[0]:
                data = data[y1 - 1:y2, x1 - 1:x2]
    if channel == 'blue':
        data = np.flipud(np.fliplr(data))
    elif channel == 'green':
        data = np.fliplr(data)
    return data


def comb_slice(data: np.ndarray) -> np.ndarray:
    """The pipeline's trace comb input: 5-column median at the detector
    centre (traceLlamasMaster.py:731 with rownum = naxis1/2)."""
    mid = data.shape[1] // 2
    return np.nanmedian(data[:, mid - 3:mid + 2], axis=1)


def _baseline(profile: np.ndarray, size: int = 51) -> np.ndarray:
    """Smooth valley baseline (running low percentile) standing in for the
    pipeline's bspline fit through comb valleys. Peak prominence is nearly
    baseline-independent; only the height=100 term needs the subtraction."""
    return percentile_filter(profile, percentile=10, size=size, mode='nearest')


def flat_extension_metrics(data: np.ndarray, benchside: str) -> dict:
    """Trace-predictive metrics for one flat/twilight extension."""
    expected = N_FIB[benchside]
    tslice = comb_slice(data)
    comb = tslice - _baseline(tslice)

    def peak_count(prominence):
        kw = dict(TRACE_PEAK_KW)
        kw['prominence'] = prominence
        peaks, props = find_peaks(comb, **kw)
        keep = (peaks > TRACE_EDGE_LO) & (peaks < TRACE_EDGE_HI)
        return peaks[keep], {k: v[keep] for k, v in props.items()}

    peaks, props = peak_count(PROM_CUT)
    floor_peaks, _ = peak_count(PROM_FLOOR)
    n_peaks = int(peaks.size)
    n_marginal = int(np.count_nonzero(props['prominences'] < PROM_MARGINAL)) \
        if n_peaks else 0
    n_faint_missed = max(0, int(floor_peaks.size) - n_peaks)
    n_sat_peaks = int(np.count_nonzero(tslice[peaks] >= SAT_LEVEL)) if n_peaks else 0

    finite = data[np.isfinite(data)]
    sat_frac = float(np.count_nonzero(finite > SAT_LEVEL) / finite.size) \
        if finite.size else 0.0
    signal = float(np.nanmedian(props['prominences'])) if n_peaks else \
        max(float(np.nanmax(comb[TRACE_EDGE_LO:TRACE_EDGE_HI])), 0.0)

    reasons = []
    notes = []
    if n_sat_peaks:
        reasons.append(f"{n_sat_peaks} saturated fibre peak(s) at trace column")
    if n_peaks < expected:
        detail = f"{n_peaks}/{expected} fibre peaks"
        if n_faint_missed:
            detail += (f" ({n_faint_missed} fibre(s) visible but below the "
                       f"pipeline prominence cut)")
        reasons.append(detail)
    elif n_peaks > expected:
        # The pipeline recovers small over-counts by matching the comb against
        # the fibremap and dropping whatever lands off a live slit slot
        # (traceLlamasMaster resolve_trace_slots); a large excess means
        # ghosts/scattered light and is not safe.
        excess = n_peaks - expected
        if excess <= 3:
            notes.append(f"{excess} extra peak(s) - pipeline drops spacing "
                         f"outliers, still expected to trace")
        else:
            reasons.append(f"{n_peaks}/{expected} fibre peaks "
                           f"({excess} extra - ghost/scattered-light "
                           f"contamination)")
    trace_pass = not reasons
    reasons.extend(notes)

    return {
        'n_peaks': n_peaks, 'expected': expected,
        'n_marginal': n_marginal, 'n_faint_missed': n_faint_missed,
        'n_sat_peaks': n_sat_peaks, 'sat_frac': sat_frac,
        'median_prominence': signal, 'frame_median': float(np.nanmedian(data)),
        'trace_pass': trace_pass, 'reasons': reasons,
    }


def _robust_sigma(values: np.ndarray) -> float:
    med = np.nanmedian(values)
    return float(1.4826 * np.nanmedian(np.abs(values - med))) or 1.0


def arc_extension_metrics(data: np.ndarray, catalog: np.ndarray) -> dict:
    """Line detection + catalog matching for one arc extension."""
    # Brightest fibre row: arc lines are sparse in x, so a row's MEAN sees
    # them while a median/percentile would not (blue has almost no catalog
    # lines near the detector centre). Smooth and take a +-2 row band as a
    # single-fibre spectrum.
    profile = uniform_filter1d(np.nanmean(data, axis=1), size=3)
    row = int(np.nanargmax(profile))
    row = min(max(row, 2), data.shape[0] - 3)
    spec = np.nanmedian(data[row - 2:row + 3, :], axis=0)
    resid = spec - _baseline(spec)
    sigma = _robust_sigma(resid)
    threshold = max(ARC_SIGDETECT * sigma, ARC_PROM_FLOOR)
    lines, props = find_peaks(resid, prominence=threshold, distance=4)
    if lines.size > 3 * catalog.size:
        # keep only the brightest plausible lines for matching
        order = np.argsort(props['prominences'])[::-1][:3 * catalog.size]
        lines = np.sort(lines[order])
    n_sat_lines = int(np.count_nonzero(spec[lines] >= SAT_LEVEL)) if lines.size else 0

    shift, n_matched = estimate_shift(lines.astype(float), catalog)

    finite = data[np.isfinite(data)]
    sat_frac = float(np.count_nonzero(finite > SAT_LEVEL) / finite.size) \
        if finite.size else 0.0
    return {
        'n_lines': int(lines.size), 'catalog_size': int(catalog.size),
        'n_matched': n_matched, 'shift': shift,
        'n_sat_lines': n_sat_lines, 'sat_frac': sat_frac,
        'peak_row': row,
    }


def estimate_shift(detected: np.ndarray, catalog: np.ndarray,
                   tol: float = ARC_MATCH_TOL,
                   max_shift: int = ARC_MAX_SHIFT) -> tuple[float | None, int]:
    """Best global shift s (detected ~ catalog + s) and the matched-line count.

    Two-stage: the shift is FOUND with a tight tolerance (ARC_SCAN_TOL) so a
    handful of noise peaks cannot fabricate an alignment when the scan
    maximises over hundreds of candidate shifts; the matched count is then
    taken at the pipeline's tol (MATCH_TOL) around the refined shift, which
    is the number refineArcX actually gets to work with. Ties prefer smaller
    |s|. Returns (refined shift, n_matched) or (None, 0)."""
    if detected.size == 0 or catalog.size == 0:
        return None, 0
    best_s, best_n = 0, 0
    for s in range(-max_shift, max_shift + 1):
        dist = np.abs(detected[None, :] - (catalog[:, None] + s))
        n = int(np.count_nonzero(dist.min(axis=1) <= ARC_SCAN_TOL))
        if n > best_n or (n == best_n and abs(s) < abs(best_s)):
            best_n, best_s = n, s
    if best_n < min(4, catalog.size):
        return None, 0            # no credible alignment
    dist = np.abs(detected[None, :] - (catalog[:, None] + best_s))
    nearest = dist.argmin(axis=1)
    matched = dist.min(axis=1) <= ARC_SCAN_TOL
    offsets = detected[nearest[matched]] - catalog[matched]
    shift = float(np.median(offsets))
    final = np.abs(detected[None, :] - (catalog[:, None] + shift))
    n_matched = int(np.count_nonzero(final.min(axis=1) <= tol))
    return shift, n_matched


def level_band_notes(bands: dict | None, kind: str, readmode: str | None,
                     det_label: str, frame_median: float) -> list[str]:
    """Compare the frame-level median with the QA observed band, when one
    exists for this PRODCATG/readmode/detector. Informational only."""
    prodcatg = KIND_PRODCATG.get(kind)
    if not bands or not prodcatg or not readmode:
        return []
    entry = bands.get(prodcatg, {}).get(readmode, {}).get(det_label)
    if not entry:
        return []
    obs = entry.get('full_level', {}).get('observed_med', {})
    lo, hi = obs.get('min'), obs.get('max')
    if lo is None or hi is None:
        return []
    if frame_median < lo:
        return [f"level {frame_median:.0f} below QA band [{lo:.0f},{hi:.0f}]"]
    if frame_median > hi:
        return [f"level {frame_median:.0f} above QA band [{lo:.0f},{hi:.0f}]"]
    return []


# ---------------------------------------------------------------------------
# Per-file assessment
# ---------------------------------------------------------------------------

def resolve_readmode(path: Path, header, baselines_path=None,
                     log_mode: str | None = None) -> tuple[str | None, str]:
    """Header READ-MDE, else obs-log value, else statistical inference
    (accepted at >= MEDIUM confidence)."""
    raw = header.get('READ-MDE')
    if raw is not None:
        text = str(raw).strip().upper()
        if text in ('FAST', 'SLOW'):
            return text, 'header'
    if log_mode in ('FAST', 'SLOW'):
        return log_mode, 'obs log'
    try:
        result = readmode_assess.assess_readmode(str(path),
                                                 baselines_path=baselines_path)
    except (readmode_assess.ReadmodeAssessError, OSError):
        return None, 'inference failed'
    if result['mode'] and \
            readmode_assess.CONFIDENCE_ORDER[result['confidence']] >= \
            readmode_assess.CONFIDENCE_ORDER['MEDIUM']:
        return result['mode'], f"inferred ({result['confidence']})"
    return None, 'undecided'


def assess_file(path: Path, kind: str, catalogs: dict, bands: dict | None,
                readmode: str | None) -> dict:
    """Assess every extension of one calibration candidate."""
    per_bench = {ch: {} for ch in CHANNELS}
    problems = []
    with fits.open(str(path), mode='readonly', memmap=False) as hdul:
        mapping, map_warnings = readmode_assess.map_extensions(hdul)
        problems.extend(map_warnings)
        for det_label, idx in mapping:
            bench, side, color = det_label.split('.')
            channel = color.lower()
            benchside = f"{bench}{side}"
            if channel not in per_bench or benchside not in N_FIB:
                continue
            raw = hdul[idx].data
            if raw.shape != EXPECTED_SHAPE:
                per_bench[channel][benchside] = {
                    'status': 'BAD_GEOMETRY',
                    'reasons': [f"shape {raw.shape} != {EXPECTED_SHAPE}"]}
                continue
            data = np.asarray(raw, dtype=np.float64)
            if np.nanmin(data) == np.nanmax(data):
                per_bench[channel][benchside] = {
                    'status': 'PLACEHOLDER', 'reasons': ['placeholder/constant frame']}
                continue
            data = orient_extension(data, hdul[idx].header, channel)
            if kind in ('flat', 'twilight'):
                metrics = flat_extension_metrics(data, benchside)
            else:
                metrics = arc_extension_metrics(data, catalogs[channel])
            metrics['status'] = 'OK'
            metrics.setdefault('reasons', [])
            metrics['reasons'].extend(
                level_band_notes(bands, kind, readmode,
                                 det_label, float(np.nanmedian(data))))
            per_bench[channel][benchside] = metrics

    for channel in CHANNELS:
        missing = [bs for bs in BENCHSIDES if bs not in per_bench[channel]]
        if missing:
            problems.append(f"{channel}: missing extension(s) {', '.join(missing)}")
    return {'per_bench': per_bench, 'problems': problems}


# ---------------------------------------------------------------------------
# Channel summaries & verdicts
# ---------------------------------------------------------------------------

def summarize_flat_channel(rows: dict, kind: str) -> dict:
    """Aggregate one file's 8 benchsides of one channel (flat/twilight)."""
    present = [bs for bs in BENCHSIDES if bs in rows]
    ok_rows = [rows[bs] for bs in present if rows[bs].get('status') == 'OK']
    n_pass = sum(1 for r in ok_rows if r.get('trace_pass'))
    n_sat = sum(r.get('n_sat_peaks', 0) for r in ok_rows)
    sat_frac_max = max((r.get('sat_frac', 0.0) for r in ok_rows), default=0.0)
    n_marginal = sum(r.get('n_marginal', 0) for r in ok_rows)
    n_faint = sum(r.get('n_faint_missed', 0) for r in ok_rows)
    med_prom = float(np.median([r['median_prominence'] for r in ok_rows])) \
        if ok_rows else 0.0
    n_bad = len(BENCHSIDES) - len(ok_rows)

    reasons = []
    if n_bad:
        reasons.append(f"{n_bad} benchside(s) missing/placeholder/bad-geometry")
    if n_sat:
        reasons.append(f"{n_sat} saturated fibre peak(s)")
    if n_faint:
        reasons.append(f"{n_faint} fibre(s) below the pipeline prominence cut")

    if kind == 'flat':
        if n_pass == len(BENCHSIDES):
            verdict = 'GOOD'
        elif n_pass >= 6:
            verdict = 'MARGINAL'
        elif n_pass >= 1:
            verdict = 'POOR'
        else:
            verdict = 'UNUSABLE'
        if verdict != 'GOOD':
            reasons.insert(0, f"predicted trace PASS on {n_pass}/8 benchsides")
    else:  # twilight: throughput frame - signal & saturation over exact count
        if not ok_rows:
            verdict, lead = 'UNUSABLE', "no usable benchsides"
        elif n_sat or sat_frac_max > 0.005:
            verdict, lead = 'POOR', "saturated"
        elif med_prom < PROM_FLOOR:
            verdict, lead = 'POOR', f"very low signal (median fibre prominence {med_prom:.0f})"
        elif med_prom < PROM_CUT or n_bad:
            verdict, lead = 'MARGINAL', f"low signal (median fibre prominence {med_prom:.0f})"
        else:
            verdict, lead = 'GOOD', ""
        if lead:
            reasons.insert(0, lead)

    return {'verdict': verdict, 'n_pass': n_pass, 'n_sat_peaks': n_sat,
            'sat_frac_max': sat_frac_max, 'n_marginal': n_marginal,
            'n_faint_missed': n_faint, 'median_prominence': med_prom,
            'reasons': reasons}


def summarize_arc_channel(rows: dict) -> dict:
    present = [bs for bs in BENCHSIDES if bs in rows]
    ok_rows = [rows[bs] for bs in present if rows[bs].get('status') == 'OK']
    n_bad = len(BENCHSIDES) - len(ok_rows)
    matched = [r['n_matched'] for r in ok_rows]
    shifts = [r['shift'] for r in ok_rows if r['shift'] is not None]
    n_sat = sum(r.get('n_sat_lines', 0) for r in ok_rows)
    catalog_size = ok_rows[0]['catalog_size'] if ok_rows else 0
    min_matched = min(matched) if matched else 0
    med_shift = float(np.median(shifts)) if shifts else None

    reasons = []
    if n_bad:
        reasons.append(f"{n_bad} benchside(s) missing/placeholder/bad-geometry")
    if n_sat:
        reasons.append(f"{n_sat} saturated arc line(s)")
    ref_missing = [bs for bs in ('4A',) if bs not in rows
                   or rows[bs].get('status') != 'OK']
    if ref_missing:
        reasons.append("bench 4A (hard-coded refinement reference camera) unusable")

    if not ok_rows or min_matched < ARC_MIN_LINES:
        verdict = 'UNUSABLE'
        reasons.insert(0, f"only {min_matched}/{catalog_size} catalog lines "
                          f"matched (pipeline needs >= {ARC_MIN_LINES})")
    elif min_matched < max(ARC_MIN_LINES + 2, int(0.5 * catalog_size)) or n_sat \
            or ref_missing or n_bad:
        verdict = 'MARGINAL'
        if min_matched < max(ARC_MIN_LINES + 2, int(0.5 * catalog_size)):
            reasons.insert(0, f"weak line coverage: {min_matched}/{catalog_size} matched")
    else:
        verdict = 'GOOD'

    return {'verdict': verdict, 'min_matched': min_matched,
            'catalog_size': catalog_size, 'median_shift': med_shift,
            'n_sat_lines': n_sat, 'reasons': reasons}


def summarize_channel(kind: str, rows: dict) -> dict:
    if kind == 'arc':
        return summarize_arc_channel(rows)
    return summarize_flat_channel(rows, kind)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_candidates(kind: str, channel: str, files: list[dict]) -> list[dict]:
    """Sort candidate files for one attribute, best first."""
    def key(entry):
        s = entry['summary'][channel]
        if kind == 'arc':
            # an unsaturated arc beats a saturated one regardless of raw
            # matched count - clipped lines have unusable centroids
            return (VERDICT_ORDER[s['verdict']], int(s['n_sat_lines'] > 0),
                    -s['min_matched'], s['n_sat_lines'])
        if kind == 'flat':
            return (VERDICT_ORDER[s['verdict']], -s['n_pass'],
                    s['n_faint_missed'], s['n_marginal'], s['sat_frac_max'])
        return (VERDICT_ORDER[s['verdict']], s['sat_frac_max'],
                -s['median_prominence'])
    return sorted(files, key=key)


def role_status(ranked: list[dict], channel: str) -> str:
    """'good' | 'marginal' | 'none' for exit-code accounting."""
    if not ranked:
        return 'none'
    best = ranked[0]['summary'][channel]['verdict']
    if best == 'GOOD':
        return 'good'
    if best in ('MARGINAL', 'POOR'):
        return 'marginal'
    return 'none'


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _fmt_metrics(kind: str, summary: dict) -> str:
    if kind == 'arc':
        shift = summary['median_shift']
        shift_txt = f"{shift:+.1f}px" if shift is not None else "shift n/a"
        return (f"lines {summary['min_matched']}/{summary['catalog_size']} "
                f"{shift_txt} sat {summary['n_sat_lines']}")
    if kind == 'flat':
        return (f"trace {summary['n_pass']}/8 faint-miss "
                f"{summary['n_faint_missed']} satpk {summary['n_sat_peaks']} "
                f"satfrac {summary['sat_frac_max'] * 100:.2f}%")
    return (f"signal {summary['median_prominence']:.0f} "
            f"satfrac {summary['sat_frac_max'] * 100:.2f}% "
            f"trace {summary['n_pass']}/8")


def print_report(roles: dict, warnings: list[str], unknown: list[dict],
                 skipped: list[str], verbose: bool):
    print("=" * 72)
    print("CALIBRATION ASSESSMENT")
    print("=" * 72)
    for (channel, kind), info in roles.items():
        label = f"{channel.capitalize()} {kind.capitalize()}"
        print(f"\n{label}  [config key: {ROLE_CONFIG_KEY[(channel, kind)]}]")
        if not info['ranked']:
            print("  (no candidates found)")
            continue
        for rank, entry in enumerate(info['ranked'], start=1):
            summary = entry['summary'][channel]
            marker = "->" if rank == 1 and info['status'] != 'none' else "  "
            exptime = entry['exptime']
            exp_txt = f"{exptime:.1f}s" if exptime is not None else "?s"
            print(f"  {marker} {rank}. [{summary['verdict']:<8}] "
                  f"{entry['name']}  ({exp_txt}, "
                  f"{entry['readmode'] or 'readmode?'}) "
                  f"{_fmt_metrics(kind, summary)}")
            for reason in summary['reasons']:
                print(f"        - {reason}")
            if verbose:
                for bs in BENCHSIDES:
                    row = entry['per_bench'][channel].get(bs)
                    if row is None:
                        print(f"          {bs}: MISSING")
                    elif row.get('status') != 'OK':
                        print(f"          {bs}: {row['status']}")
                    elif kind == 'arc':
                        shift = row['shift']
                        stxt = f"{shift:+.1f}" if shift is not None else "n/a"
                        print(f"          {bs}: lines {row['n_lines']} "
                              f"matched {row['n_matched']}/{row['catalog_size']} "
                              f"shift {stxt}")
                    else:
                        print(f"          {bs}: peaks {row['n_peaks']}/"
                              f"{row['expected']} "
                              f"{'PASS' if row['trace_pass'] else 'FAIL'} "
                              f"faint-miss {row['n_faint_missed']} "
                              f"satpk {row['n_sat_peaks']}")

    if warnings:
        print("\nWARNINGS")
        for w in warnings:
            print(f"  ! {w}")
    if unknown:
        print("\nUNCLASSIFIED FILES (assign manually)")
        for entry in unknown:
            print(f"  ? {entry['name']}: {entry['how']}")
    if skipped and verbose:
        print(f"\nSkipped {len(skipped)} derived/product file(s)")
    print()


def write_snippet(path: Path, roles: dict, input_dir: Path):
    lines = [
        f"# calib_assess recommendations - generated "
        f"{datetime.now():%Y-%m-%d %H:%M} from {input_dir}",
        "# Paste into a reduxSetupGUI config (key = value format), or use as",
        "# a checklist when assigning attributes in the GUI.",
    ]
    for (channel, kind), info in roles.items():
        key = ROLE_CONFIG_KEY[(channel, kind)]
        if info['status'] == 'none':
            why = ("no candidates found" if not info['ranked'] else
                   "; ".join(info['ranked'][0]['summary'][channel]['reasons'])
                   or "no viable candidate")
            lines.append(f"# {key}: no viable candidate ({why})")
        else:
            best = info['ranked'][0]
            lines.append(f"{key} = {best['path']}")
    path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _epoch_shift_warnings(roles: dict) -> list[str]:
    """The epoch shift is a property of the night, common to all arcs of a
    channel - a shift beyond the 10 px match budget means refineArcX will
    fall back on every fibre regardless of which arc is chosen."""
    warnings = []
    for channel in CHANNELS:
        info = roles.get((channel, 'arc'))
        if not info:
            continue
        # Saturated arcs give distorted line centroids and bogus shifts, so
        # take the estimate from the unsaturated candidate with the best
        # line coverage (falling back to all candidates).
        cands = [e['summary'][channel] for e in info['ranked']
                 if e['summary'][channel]['median_shift'] is not None]
        if not cands:
            continue
        clean = [s for s in cands if s['n_sat_lines'] == 0]
        pool = clean or cands
        best = max(pool, key=lambda s: s['min_matched'])
        shift = best['median_shift']
        spread = max(s['median_shift'] for s in pool) - \
            min(s['median_shift'] for s in pool)
        if len(pool) > 1 and spread > 5.0:
            warnings.append(
                f"{channel} arc shift estimates disagree across candidates "
                f"(spread {spread:.0f} px) - trusting the best-matched "
                f"unsaturated frame ({shift:+.1f} px)")
        if abs(shift) > ARC_MATCH_TOL:
            warnings.append(
                f"{channel} arc epoch shift ~{shift:+.0f} px exceeds the "
                f"{ARC_MATCH_TOL:.0f} px refinement budget - arc refinement "
                f"will fail on every fibre without an epoch bootstrap "
                f"(shifted reference arc / peaks.csv rebuild)")
    return warnings


def _bias_sanity(bias_entries: list[dict], needed_modes: set[str]) -> list[str]:
    have = {e['readmode'] for e in bias_entries if e['readmode']}
    warnings = []
    for mode in sorted(m for m in needed_modes if m):
        if mode not in have:
            warnings.append(
                f"no {mode} bias frames found, but {mode} frames need a "
                f"{mode} master bias (tracing subtracts a readmode-matched bias)")
    if not bias_entries:
        warnings.append("no bias frames found in this directory")
    return warnings


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Assess raw LLAMAS calibration frames (arcs, LDLS flats, "
                    "twilight flats) and recommend files for the reduxSetupGUI "
                    "attributes. Read-only.")
    parser.add_argument('input_dir', help="directory of raw *_mef.fits files")
    parser.add_argument('--glob', default='*.fits', help="filename pattern "
                        "(default: *.fits; derived products are skipped)")
    parser.add_argument('--output-dir', default='.',
                        help="where the snippet/report are written (default: cwd)")
    parser.add_argument('--obs-log', help="observing log (GUI tab-separated "
                        "obs.log or xlsx) used to classify files whose headers "
                        "lack PRODCATG/OBJECT - the norm for old data")
    parser.add_argument('--log-sheet', help="xlsx sheet name (default: first)")
    parser.add_argument('--report', help="also write a JSON report to this path "
                        "(relative to --output-dir unless absolute)")
    parser.add_argument('--no-snippet', action='store_true',
                        help="do not write calib_assess_recommendations.txt")
    parser.add_argument('--baselines',
                        help="override QA baselines JSON (levels + readmode bands)")
    parser.add_argument('--science-readmode', choices=('FAST', 'SLOW'),
                        help="readmode of the science frames, for the bias "
                             "coverage check (default: read from science headers)")
    parser.add_argument('-v', '--verbose', action='store_true',
                        help="per-benchside detail in the report")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input_dir).expanduser()
    if not input_dir.is_dir():
        print(f"error: not a directory: {input_dir}", file=sys.stderr)
        return EXIT_ERROR
    output_dir = Path(args.output_dir).expanduser()

    try:
        catalogs = load_line_catalogs()
    except CalibAssessError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    bands = load_level_bands(args.baselines)

    log_index = {}
    if args.obs_log:
        try:
            rows = obs_log.load_obs_log(args.obs_log, sheet=args.log_sheet)
            log_index = obs_log.build_log_index(rows)
        except obs_log.ObsLogError as exc:
            print(f"error: cannot read obs log: {exc}", file=sys.stderr)
            return EXIT_ERROR

    paths = sorted(input_dir.glob(args.glob))
    skipped = [p.name for p in paths
               if any(tok in p.name.lower() for tok in SKIP_NAME_TOKENS)]
    paths = [p for p in paths if p.name not in set(skipped)]
    if not paths:
        print(f"error: no FITS files match {args.glob!r} in {input_dir}",
              file=sys.stderr)
        return EXIT_ERROR

    candidates = {'flat': [], 'twilight': [], 'arc': []}
    bias_entries, science_modes, unknown = [], set(), []
    warnings: list[str] = []

    print(f"Scanning {len(paths)} file(s) in {input_dir} ...")
    for path in paths:
        try:
            header = fits.getheader(str(path), 0)
        except OSError as exc:
            warnings.append(f"{path.name}: unreadable ({exc})")
            continue
        kind, how = classify_frame(header)
        log_mode = log_exp = None
        if kind == 'unknown' and log_index:
            kind, how, log_mode, log_exp = classify_from_log(path.name, log_index)
        exptime = _exptime(header)
        entry = {'path': str(path), 'name': path.name, 'kind': kind,
                 'how': how,
                 'exptime': exptime if exptime is not None else log_exp}
        if kind in candidates:
            readmode, rm_how = resolve_readmode(path, header, args.baselines,
                                                log_mode)
            entry['readmode'], entry['readmode_source'] = readmode, rm_how
            print(f"  {path.name}: {kind} ({how}), assessing ...")
            try:
                result = assess_file(path, kind, catalogs, bands, readmode)
            except (OSError, ValueError) as exc:
                warnings.append(f"{path.name}: assessment failed ({exc})")
                continue
            entry['per_bench'] = result['per_bench']
            entry['problems'] = result['problems']
            entry['summary'] = {ch: summarize_channel(kind, result['per_bench'][ch])
                                for ch in CHANNELS}
            candidates[kind].append(entry)
        elif kind == 'bias':
            entry['readmode'] = resolve_readmode(path, header, args.baselines,
                                                 log_mode)[0]
            bias_entries.append(entry)
        elif kind == 'science':
            raw = str(header.get('READ-MDE') or '').strip().upper()
            mode = raw if raw in ('FAST', 'SLOW') else log_mode
            if mode:
                science_modes.add(mode)
        elif kind == 'unknown':
            unknown.append(entry)

    roles = {}
    for kind in ('flat', 'twilight', 'arc'):
        for channel in CHANNELS:
            ranked = rank_candidates(kind, channel, candidates[kind])
            roles[(channel, kind)] = {'ranked': ranked,
                                      'status': role_status(ranked, channel)}

    warnings.extend(_epoch_shift_warnings(roles))
    needed = set(science_modes)
    if args.science_readmode:
        needed.add(args.science_readmode)
    for (channel, kind), info in roles.items():
        if info['status'] != 'none':
            mode = info['ranked'][0].get('readmode')
            if mode:
                needed.add(mode)
    warnings.extend(_bias_sanity(bias_entries, needed))
    dead_note = load_dead_fibre_note()
    if dead_note and args.verbose:
        warnings.append(dead_note)

    print_report(roles, warnings, unknown, skipped, args.verbose)

    output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_snippet:
        snippet = output_dir / 'calib_assess_recommendations.txt'
        write_snippet(snippet, roles, input_dir)
        print(f"Config snippet written to {snippet}")
    if args.report:
        report_path = Path(args.report)
        if not report_path.is_absolute():
            report_path = output_dir / report_path
        payload = {
            'generated': datetime.now().isoformat(timespec='seconds'),
            'input_dir': str(input_dir),
            'roles': {f"{ch}_{kind}": {
                'config_key': ROLE_CONFIG_KEY[(ch, kind)],
                'status': info['status'],
                'recommended': (info['ranked'][0]['path']
                                if info['status'] != 'none' and info['ranked']
                                else None),
                'ranked': [{'file': e['path'], 'exptime': e['exptime'],
                            'readmode': e.get('readmode'),
                            'summary': e['summary'][ch]}
                           for e in info['ranked']],
            } for (ch, kind), info in roles.items()},
            'files': [e for kind in candidates for e in candidates[kind]],
            'bias_frames': [{'file': e['path'], 'readmode': e['readmode']}
                            for e in bias_entries],
            'unknown_files': [{'file': e['path'], 'why': e['how']}
                              for e in unknown],
            'warnings': warnings,
        }
        report_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"JSON report written to {report_path}")

    statuses = [info['status'] for info in roles.values()]
    if 'none' in statuses:
        return EXIT_UNVIABLE
    if 'marginal' in statuses:
        return EXIT_MARGINAL
    return EXIT_OK


if __name__ == '__main__':
    sys.exit(main())
