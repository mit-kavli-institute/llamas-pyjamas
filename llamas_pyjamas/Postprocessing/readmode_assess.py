"""
Statistical FAST/SLOW readout-mode inference for LLAMAS raw MEF files.

Older LLAMAS data (pre-2026) lacks the READ-MDE primary-header keyword. This
module infers the mode from the per-detector bias level measured in the
unilluminated top/bottom stripes of each 2048x2048 detector image - LLAMAS
CCDs have no overscan, and these stripes are the same regions the QA baseline
suite uses (``QA/baselines/scripts/extract_stats.py``), so the per-detector
FAST/SLOW reference bands in ``QA/baselines/qa_thresholds_derived.json`` apply
directly to raw frames.

One file has exactly one readout mode across all 24 detectors, so ~21 usable
detectors vote independently and the votes are aggregated with abstention
rules that make the result robust to per-detector background/illumination
differences and epoch drift:

* the stripe statistic is ``min(median(bottom), median(top))`` - illumination
  only ever adds signal, so the cleaner stripe wins;
* each detector is compared two-sided against ITS OWN FAST and SLOW bands
  (band ordering is not uniform across detectors - never use a global
  threshold; the LLAMAS_QL 1000 DN rule misclassifies SLOW frames as FAST);
* detectors far outside both bands abstain (epoch drift, illuminated stripes,
  dead cameras), as do detectors whose two bands overlap and detectors with
  no measured baselines.

Validated against known-mode ut20241128-29 frames: SLOW bias 21/21 correct,
FAST bias 21/21, 1800 s SLOW science 20/20 (one abstention), FAST LDLS flat
19/20 with the single dissent flagged marginal.

Standalone usage:  python readmode_assess.py file1_mef.fits [file2 ...] [-v]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits

logger = logging.getLogger(__name__)

# Stripe regions in RAW frame orientation, identical to extract_stats.py
BOT = (slice(2, 28), slice(100, 1948))
TOP = (slice(2020, 2046), slice(100, 1948))
EXPECTED_SHAPE = (2048, 2048)

MIN_GAP = 30.0          # ADU; detector excluded if its FAST/SLOW bands are closer
NEITHER_MAX = 150.0     # ADU; abstain if measured value is further than this from BOTH bands
MARGINAL = 20.0         # ADU; vote counted but flagged if |d_fast - d_slow| is below this
STRIPE_DIFF_WARN = 25.0  # ADU; |bottom - top| above this suggests an illumination gradient
ELEV_WARN = 20.0        # ADU; median elevation above the voted band suggests illuminated stripes
BLEED_P95_WARN = 500.0  # ADU; stripe p95 this far above its median suggests bleed

MIN_VOTERS = 5
UNDECIDED_FRAC = 0.60
CONFIDENCE_ORDER = {'UNDECIDED': 0, 'LOW': 1, 'MEDIUM': 2, 'HIGH': 3}

_QA_DIR = Path(__file__).resolve().parents[1] / 'QA'
DEFAULT_BASELINES = _QA_DIR / 'baselines' / 'qa_thresholds_derived.json'
FALLBACK_YAML = _QA_DIR / 'qa_config_cal.yaml'
# The YAML cannot distinguish measured bands from generated placeholders, so
# the known-bad detectors are hard-coded for that path only. The JSON path
# derives exclusions from the data (absent = unmeasured, gap test = overlap).
YAML_EXCLUDE = {'4.A.Red', '1.A.Blue', '4.A.Blue'}

# Extension order of a complete raw MEF, mirroring constants.idx_lookup
# (red/green/blue within each bench-side 1A,1B,...,4B). Used only as a
# last-resort fallback when extension headers carry no identity cards.
POSITIONAL_ORDER = [
    f"{bench}.{side}.{color}"
    for bench in '1234' for side in 'AB' for color in ('Red', 'Green', 'Blue')
]


class ReadmodeAssessError(Exception):
    """Raised when the reference bands cannot be loaded at all."""


def load_bands(baselines_path: str | Path | None = None) -> dict:
    """
    Load per-detector FAST/SLOW bias-level bands.

    Returns {'bands': {det: {'FAST': (lo, hi, med), 'SLOW': (...), 'gap': g}},
             'excluded_overlap': [det...], 'source': str}.
    """
    if baselines_path is not None:
        path = Path(baselines_path)
        if not path.is_file():
            raise ReadmodeAssessError(f"Baselines file not found: {path}")
        if path.suffix.lower() in ('.yaml', '.yml'):
            return _load_bands_yaml(path)
        return _load_bands_json(path)
    if DEFAULT_BASELINES.is_file():
        return _load_bands_json(DEFAULT_BASELINES)
    if FALLBACK_YAML.is_file():
        logger.warning(f"{DEFAULT_BASELINES.name} not found; falling back to "
                       f"{FALLBACK_YAML.name} bands")
        return _load_bands_yaml(FALLBACK_YAML)
    raise ReadmodeAssessError(
        f"No QA baselines found ({DEFAULT_BASELINES} or {FALLBACK_YAML}); "
        f"pass an explicit path.")


def _finish_bands(raw: dict, source: str) -> dict:
    bands, excluded = {}, []
    for det, modes in raw.items():
        f_lo, f_hi, f_med = modes['FAST']
        s_lo, s_hi, s_med = modes['SLOW']
        gap = max(s_lo - f_hi, f_lo - s_hi)
        if gap < MIN_GAP:
            excluded.append(det)
        else:
            bands[det] = {'FAST': modes['FAST'], 'SLOW': modes['SLOW'], 'gap': gap}
    if not bands:
        raise ReadmodeAssessError(f"No usable detector bands in {source}")
    return {'bands': bands, 'excluded_overlap': sorted(excluded), 'source': source}


def _load_bands_json(path: Path) -> dict:
    with open(path) as fh:
        data = json.load(fh)
    try:
        bia = data['per_detector']['CAL.R-BIA']
        fast, slow = bia['FAST'], bia['SLOW']
    except KeyError as exc:
        raise ReadmodeAssessError(
            f"{path} lacks per_detector/CAL.R-BIA/FAST+SLOW: {exc}") from exc
    raw = {}
    for det in sorted(set(fast) & set(slow)):
        entry = {}
        for mode, table in (('FAST', fast), ('SLOW', slow)):
            eb = table[det].get('edge_bg', {})
            obs = eb.get('observed', {})
            # The observed intervals are the source of truth the calibration
            # was validated on; the top-level min/max are padded QA tolerance
            # bands whose padding shrinks vote margins and overlaps 3.A.Red.
            lo, hi = obs.get('min'), obs.get('max')
            if lo is None or hi is None:
                lo, hi = eb.get('min'), eb.get('max')
            if lo is None or hi is None:
                entry = None
                break
            entry[mode] = (float(lo), float(hi), float(obs.get('med', (lo + hi) / 2.0)))
        if entry:
            raw[det] = entry
    return _finish_bands(raw, str(path))


def _load_bands_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ReadmodeAssessError(
            "Reading YAML baselines requires PyYAML (pip install pyyaml); "
            "prefer the derived JSON baselines.") from exc
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    table = (cfg.get('lookup_tables') or {}).get('edge_bg_BIAS')
    if not isinstance(table, dict):
        raise ReadmodeAssessError(f"{path} has no lookup_tables/edge_bg_BIAS table")
    raw = {}
    for det, modes in table.items():
        if det in YAML_EXCLUDE or not isinstance(modes, dict):
            continue
        entry = {}
        for mode in ('FAST', 'SLOW'):
            band = modes.get(mode) or modes.get(mode.lower())
            if not isinstance(band, dict) or 'min' not in band or 'max' not in band:
                entry = None
                break
            lo, hi = float(band['min']), float(band['max'])
            entry[mode] = (lo, hi, (lo + hi) / 2.0)
        if entry:
            raw[det] = entry
    return _finish_bands(raw, str(path))


def interval_dist(value: float, band: tuple[float, float, float]) -> float:
    """Distance from value to the [lo, hi] interval; 0 inside."""
    lo, hi = band[0], band[1]
    if value < lo:
        return lo - value
    if value > hi:
        return value - hi
    return 0.0


def map_extensions(hdul) -> tuple[list[tuple[str, int]], list[str]]:
    """
    Map image extensions to detector labels ("1.A.Green") by their
    BENCH/SIDE/COLOR header cards, falling back to CAM_NAME ("1A_red"), then
    to position. Never trusts HDU index when headers are available - real
    files can have missing extensions (the fast master bias has only 22).
    """
    mapping: list[tuple[str, int]] = []
    warnings: list[str] = []
    seen: set[str] = set()
    image_ordinal = 0
    for idx, hdu in enumerate(hdul):
        if idx == 0 or getattr(hdu, 'data', None) is None or not hasattr(hdu, 'header'):
            continue
        if getattr(hdu.data, 'ndim', 0) != 2:
            continue
        image_ordinal += 1
        hdr = hdu.header
        det = None
        try:
            if 'BENCH' in hdr and 'SIDE' in hdr and 'COLOR' in hdr:
                det = (f"{str(hdr['BENCH']).strip()}."
                       f"{str(hdr['SIDE']).strip().upper()}."
                       f"{str(hdr['COLOR']).strip().capitalize()}")
            elif 'CAM_NAME' in hdr:
                cam = str(hdr['CAM_NAME']).strip()
                benchside, color = cam.split('_', 1)
                det = (f"{benchside[0]}.{benchside[1].upper()}."
                       f"{color.strip().capitalize()}")
        except (ValueError, IndexError):
            det = None
        if det is None:
            if image_ordinal <= len(POSITIONAL_ORDER):
                det = POSITIONAL_ORDER[image_ordinal - 1]
                warnings.append(
                    f"HDU {idx}: no BENCH/SIDE/COLOR or CAM_NAME; assumed {det} "
                    f"by position (unreliable if extensions are missing)")
            else:
                warnings.append(f"HDU {idx}: unidentifiable extension skipped")
                continue
        if det in seen:
            warnings.append(f"HDU {idx}: duplicate detector {det}; first occurrence kept")
            continue
        seen.add(det)
        mapping.append((det, idx))
    return mapping, warnings


def assess_readmode(source, exposure_type: str | None = None,
                    baselines_path: str | Path | None = None) -> dict:
    """
    Infer the readout mode of one raw LLAMAS MEF file.

    Args:
        source: path to the file, or an open HDUList (must have been opened
            with memmap=False - raw frames carry BZERO/BSCALE).
        exposure_type: optional hint ('Science', 'calibration', ...) from the
            obs log or the user. Science frames get an advisory SLOW prior -
            it pre-selects the prompt default when the vote is UNDECIDED but
            never alters the tally.
        baselines_path: override for the QA baselines (JSON or YAML).

    Returns a dict:
        mode        'FAST' | 'SLOW' | None
        confidence  'HIGH' | 'MEDIUM' | 'LOW' | 'UNDECIDED'
        numeric     {votes_fast, votes_slow, n_usable, n_neither, n_excluded,
                     n_placeholder, frac, median_margin}
        per_detector  list of per-detector measurement dicts (for -v display)
        warnings    list of strings
        prior_note  advisory string or None
    """
    ref = load_bands(baselines_path)
    if isinstance(source, (str, Path)):
        with fits.open(str(source), mode='readonly', memmap=False) as hdul:
            return _assess_hdul(hdul, ref, exposure_type)
    return _assess_hdul(source, ref, exposure_type)


def _assess_hdul(hdul, ref: dict, exposure_type: str | None) -> dict:
    bands = ref['bands']
    mapping, warnings = map_extensions(hdul)
    if not mapping:
        warnings.append("no identifiable 2-D image extensions")

    per_detector = []
    for det, idx in mapping:
        data = hdul[idx].data
        if data.shape != EXPECTED_SHAPE:
            per_detector.append({'detector': det, 'hdu_index': idx,
                                 'status': 'BAD_GEOMETRY',
                                 'note': f"shape {data.shape} != {EXPECTED_SHAPE}"})
            continue
        d = np.asarray(data, dtype=np.float64)
        if np.nanmin(d) == np.nanmax(d):
            per_detector.append({'detector': det, 'hdu_index': idx,
                                 'status': 'PLACEHOLDER',
                                 'note': f"constant frame ({d.flat[0]:.0f} ADU)"})
            continue
        bot, top = d[BOT], d[TOP]
        bot_med = float(np.nanmedian(bot))
        top_med = float(np.nanmedian(top))
        bot_p95 = float(np.nanpercentile(bot, 95))
        # Illumination/background only ever ADD signal: the cleaner stripe wins.
        measured = min(bot_med, top_med)
        row = {'detector': det, 'hdu_index': idx, 'bot_med': bot_med,
               'top_med': top_med, 'measured': measured,
               'stripe_diff': abs(bot_med - top_med),
               'bleed_excess': bot_p95 - bot_med,
               'vote': None, 'margin': None, 'elevation': None, 'note': ''}
        if det not in bands:
            row['status'] = ('EXCLUDED_OVERLAP' if det in ref['excluded_overlap']
                             else 'NO_BANDS')
            per_detector.append(row)
            continue
        d_fast = interval_dist(measured, bands[det]['FAST'])
        d_slow = interval_dist(measured, bands[det]['SLOW'])
        row['d_fast'], row['d_slow'] = d_fast, d_slow
        if min(d_fast, d_slow) > NEITHER_MAX or d_fast == d_slow:
            row['status'] = 'NEITHER'
        else:
            vote = 'FAST' if d_fast < d_slow else 'SLOW'
            margin = abs(d_fast - d_slow)
            row['vote'] = vote
            row['margin'] = margin
            row['elevation'] = measured - bands[det][vote][2]
            row['status'] = 'MARGINAL' if margin < MARGINAL else 'VOTE'
        per_detector.append(row)

    votes = [r for r in per_detector if r.get('vote')]
    n_fast = sum(1 for r in votes if r['vote'] == 'FAST')
    n_slow = len(votes) - n_fast
    n_usable = len(votes)
    winner = 'FAST' if n_fast > n_slow else ('SLOW' if n_slow > n_fast else None)
    frac = (max(n_fast, n_slow) / n_usable) if n_usable else 0.0
    winner_margins = [r['margin'] for r in votes if r['vote'] == winner]
    med_margin = float(np.median(winner_margins)) if winner_margins else 0.0

    n_neither = sum(1 for r in per_detector if r['status'] == 'NEITHER')
    n_placeholder = sum(1 for r in per_detector if r['status'] == 'PLACEHOLDER')
    n_excluded = sum(1 for r in per_detector
                     if r['status'] in ('EXCLUDED_OVERLAP', 'NO_BANDS'))

    elevations = [r['elevation'] for r in votes if r['elevation'] is not None]
    if elevations and float(np.median(elevations)) > ELEV_WARN:
        warnings.append(
            f"stripes look illuminated (median elevation "
            f"{float(np.median(elevations)):.0f} ADU above the voted bands)")
    gradient_dets = [r['detector'] for r in votes
                     if r.get('stripe_diff', 0.0) > STRIPE_DIFF_WARN]
    if gradient_dets:
        warnings.append(
            f"stripe top/bottom gradient >{STRIPE_DIFF_WARN:.0f} ADU on "
            f"{len(gradient_dets)} detector(s): {', '.join(gradient_dets[:4])}"
            + ('...' if len(gradient_dets) > 4 else ''))
    if n_neither >= 4:
        warnings.append(f"{n_neither} detectors matched neither band "
                        f"(drift, illumination, or unmodelled frame type)")
    if n_placeholder > 2:
        warnings.append(f"{n_placeholder} placeholder/constant detector frames")
    bleed_dets = [r['detector'] for r in votes
                  if r.get('bleed_excess', 0.0) > BLEED_P95_WARN]
    if bleed_dets:
        warnings.append(f"possible bleed into bottom stripe on: "
                        f"{', '.join(bleed_dets)}")

    if n_usable < MIN_VOTERS or winner is None or frac < UNDECIDED_FRAC:
        confidence = 'UNDECIDED'
    elif frac >= 0.90 and n_usable >= 12 and med_margin >= 40.0:
        confidence = 'HIGH'
    elif frac >= 0.75 and n_usable >= 8:
        confidence = 'MEDIUM'
    else:
        confidence = 'LOW'
    if confidence in ('HIGH', 'MEDIUM') and any(
            w.startswith('stripes look illuminated') or 'neither band' in w
            for w in warnings):
        confidence = 'MEDIUM' if confidence == 'HIGH' else 'LOW'

    prior_note = None
    if exposure_type and 'sci' in str(exposure_type).lower():
        prior_note = "science frames are a-priori likely SLOW"

    return {
        'mode': winner if confidence != 'UNDECIDED' else None,
        'confidence': confidence,
        'numeric': {'votes_fast': n_fast, 'votes_slow': n_slow,
                    'n_usable': n_usable, 'n_neither': n_neither,
                    'n_excluded': n_excluded, 'n_placeholder': n_placeholder,
                    'frac': round(frac, 3), 'median_margin': round(med_margin, 1)},
        'per_detector': per_detector,
        'warnings': warnings,
        'prior_note': prior_note,
        'bands_source': ref['source'],
    }


def format_assessment(result: dict, verbose: bool = False) -> str:
    """Human-readable summary; one line, plus a per-detector table if verbose."""
    num = result['numeric']
    abstain = num['n_neither'] + num['n_excluded'] + num['n_placeholder']
    if result['mode']:
        head = (f"inferred {result['mode']} ({result['confidence']}: "
                f"{num['votes_slow']} SLOW / {num['votes_fast']} FAST / "
                f"{abstain} abstain, median margin {num['median_margin']:.0f} ADU)")
    else:
        head = (f"UNDECIDED ({num['votes_slow']} SLOW / {num['votes_fast']} FAST / "
                f"{abstain} abstain)")
    lines = [head]
    lines += [f"  warning: {w}" for w in result['warnings']]
    if result['prior_note']:
        lines.append(f"  note: {result['prior_note']}")
    if verbose:
        lines.append(f"  bands: {result['bands_source']}")
        lines.append("  detector    measured   d_FAST   d_SLOW  vote   status")
        for r in result['per_detector']:
            if 'measured' in r:
                lines.append(
                    f"  {r['detector']:<10} {r['measured']:>8.1f} "
                    f"{r.get('d_fast', float('nan')):>8.1f} "
                    f"{r.get('d_slow', float('nan')):>8.1f}  "
                    f"{r['vote'] or '-':<5}  {r['status']}")
            else:
                lines.append(f"  {r['detector']:<10} {'-':>8} {'-':>8} {'-':>8}  "
                             f"{'-':<5}  {r['status']} ({r.get('note', '')})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Infer FAST/SLOW readout mode of raw LLAMAS MEF files "
                    "from QA bias-level baselines (read-only).")
    parser.add_argument('fits_files', nargs='+', help="raw *_mef.fits files")
    parser.add_argument('--baselines', help="override QA baselines (JSON or YAML)")
    parser.add_argument('--exposure-type', help="hint, e.g. 'Science'")
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    status = 0
    for path in args.fits_files:
        try:
            result = assess_readmode(path, exposure_type=args.exposure_type,
                                     baselines_path=args.baselines)
        except (ReadmodeAssessError, OSError) as exc:
            print(f"{path}: ERROR {exc}")
            status = 3
            continue
        print(f"{Path(path).name}: {format_assessment(result, args.verbose)}")
    return status


if __name__ == '__main__':
    sys.exit(main())
