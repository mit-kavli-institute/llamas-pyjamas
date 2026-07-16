#!/usr/bin/env python3
"""Generate qa_config_cal.yaml + qa_config_science.yaml from the ROBUST derived
thresholds. Builds complete per-detector (extension.name x readout_mode) lookup
tables (color-median fallback for the always-placeholder blue detectors).

Design (post-review):
- Background/level checks consume the per-detector unilluminated-stripe band
  (edge_bg) and use the MEDIAN metric matching median-derived limits.
- Level/background/RMS are WARN (drift & warm monitoring); structure is FAIL on
  uniform frames (bias/dark) and WARN on illuminated frames; saturation uses the
  per-detector derived cap; shutter is FAIL; CCD temperature has a WARN band and a
  FAIL (shut-off) band.
- Illuminated frames (flat/sky/arc) get NO full-frame level check (illumination
  varies with exposure time); they get edge-stripe background + saturation +
  structure + shutter + temperature.
"""
import json, os
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
# derived thresholds live in baselines/ (parent of scripts/); fall back to alongside the script
_DER_PATH = next(p for p in (os.path.join(os.path.dirname(HERE), "qa_thresholds_derived.json"),
                             os.path.join(HERE, "qa_thresholds_derived.json")) if os.path.exists(p))
DER = json.load(open(_DER_PATH))
QA_DIR = "/Users/slh/Documents/Projects/Magellan_dev/LLAMAS/llamas-pyjamas/llamas_pyjamas/QA"

BENCH_SIDES = [(b, s) for b in (1, 2, 3, 4) for s in ("A", "B")]
COLORS = [("Red", "red"), ("Green", "green"), ("Blue", "blue")]
EXTENSIONS, _hdu = [], 1
for (b, s) in BENCH_SIDES:
    for (Cap, low) in COLORS:
        EXTENSIONS.append({"name": f"{b}.{s}.{Cap}", "hdu_index": _hdu,
                           "color": Cap, "bench": b, "side": s})
        _hdu += 1
NAMES = [e["name"] for e in EXTENSIONS]
NAME_COLOR = {e["name"]: e["color"].lower() for e in EXTENSIONS}


def modes_for(pc):
    return sorted(DER["per_detector"].get(pc, {}).keys())


def ent(pc, mode, name):
    return DER["per_detector"].get(pc, {}).get(mode, {}).get(name)


def fallback(pc, mode, color, getter):
    vals = [getter(e) for nm, e in DER["per_detector"].get(pc, {}).get(mode, {}).items()
            if NAME_COLOR[nm] == color and getter(e) is not None]
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def build_table(pc, leaf_fn):
    """leaf_fn(entry) -> dict of numeric fields; None-entries filled by color median
    of each field."""
    table = {}
    for name in NAMES:
        color = NAME_COLOR[name]
        for mode in modes_for(pc):
            e = ent(pc, mode, name)
            if e:
                leaf = leaf_fn(e)
            else:
                sample = next((leaf_fn(x) for x in DER["per_detector"].get(pc, {}).get(mode, {}).values()), None)
                if sample is None:
                    continue
                leaf = {}
                for f in sample:
                    leaf[f] = fallback(pc, mode, color, lambda x, f=f: leaf_fn(x)[f])
            table.setdefault(name, {})[mode] = {k: round(float(v), 4) for k, v in leaf.items()}
    return table


edge_leaf = lambda e: {"min": e["edge_bg"]["min"], "max": e["edge_bg"]["max"]}
level_leaf = lambda e: {"med_min": e["full_level"]["med_min"], "med_max": e["full_level"]["med_max"],
                        "rms_max": e["full_level"]["rms_max"]}
struct_leaf = lambda e: {"row_max": e["structure"]["row_max"], "col_max": e["structure"]["col_max"]}
sat_leaf = lambda e: {"frac_max": e["sat_frac_max"]}

KEYS = [{"from": "extension.name"}, {"from": "metadata.readout_mode"}]


def base_blocks():
    return {
        "config_version": "1.0",
        "instrument": {"name": "LLAMAS",
                       "description": "Data-driven QA rules for LLAMAS MEF frames"},
        "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
        "extensions": EXTENSIONS,
        "regions": {
            "full_frame": {"type": "full"},
            "bottom_stripe": {"type": "rectangle", "x_start": 100, "x_end": 1948,
                              "y_start": 2, "y_end": 28},
            "top_stripe": {"type": "rectangle", "x_start": 100, "x_end": 1948,
                           "y_start": 2020, "y_end": 2046},
        },
        "metrics": {
            "median": {"type": "median"},
            "rms": {"type": "std"},
            "saturated_fraction": {"type": "fraction_above", "threshold": 63000},
            "row_banding": {"type": "row_structure"},
            "column_banding": {"type": "column_structure"},
            # camera-warming: quarter-block median gradient / exposure time (ADU/s)
            "background_gradient_rate": {"type": "background_gradient_rate",
                                         "exptime_keys": ["SEXPTIME", "REXPTIME", "EXPTIME"],
                                         "min_exptime": WARM_MIN_EXPTIME},
        },
        "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
    }


def shutter_rule(pc):
    s = DER["shutter"][pc]
    return {"name": "shutter_exptime_consistency", "severity": "FAIL",
            "header_check": {"op": "abs_or_rel_diff", "source": "SEXPTIME", "other": "REXPTIME",
                             "abs_tol": s["abs_tol"], "rel_tol": s["rel_tol"]}}


# CCD temperature keyword spellings vary across file generations (underscore /
# no-separator / hyphen); the engine reads the first populated one.
TEMP_KEY_VARIANTS = ["CCDTEMP_1", "CCDTEMP1", "CCDTEMP-1"]


def build_temp_table():
    """Per-camera CCD temperature caps, keyed by extension.name only (temperature is
    readout-mode independent). Placeholder-only cameras fall back to their colour
    median of the warn/fail caps."""
    pd = DER["ccd_temp"]["per_detector"]
    cold_min = DER["ccd_temp"]["cold_min"]
    table = {}
    for name in NAMES:
        leaf = pd.get(name)
        if leaf is None:
            color = NAME_COLOR[name]
            same = [pd[n] for n in NAMES if n in pd and NAME_COLOR[n] == color]
            if not same:
                continue
            warn = sorted(e["warn_max"] for e in same)[len(same) // 2]
            fail = sorted(e["fail_max"] for e in same)[len(same) // 2]
            leaf = {"warn_max": warn, "fail_max": fail, "cold_min": cold_min}
        table[name] = {"cold_min": round(float(leaf["cold_min"]), 2),
                       "warn_max": round(float(leaf["warn_max"]), 2),
                       "fail_max": round(float(leaf["fail_max"]), 2)}
    return table


def temp_rules():
    """Per-detector camera-temperature monitor: WARN/FAIL when a camera warms above
    its OWN healthy baseline (colour-aware margins baked into the temp table -- reds
    run warmer and get extra headroom), plus an absolute TEC-limit safety FAIL. The
    value is read from whichever CCDTEMP spelling the frame uses."""
    abs_fail = DER["ccd_temp"]["abs_fail_above"]
    tkeys = lambda: [{"from": "extension.name"}]   # fresh objects -> clean YAML (no anchors)
    return [
        {"name": "ccd_temperature_warm", "severity": "WARN", "per_extension": True,
         "header_check": {"op": "range", "per_extension_key": list(TEMP_KEY_VARIANTS)},
         "expected_from_lookup": {"table": "temp", "keys": tkeys(),
                                  "min_field": "cold_min", "max_field": "warn_max"}},
        {"name": "ccd_temperature_hot", "severity": "FAIL", "per_extension": True,
         "header_check": {"op": "range", "per_extension_key": list(TEMP_KEY_VARIANTS)},
         "expected_from_lookup": {"table": "temp", "keys": tkeys(),
                                  "max_field": "fail_max"}},
        {"name": "ccd_temperature_shutoff", "severity": "FAIL", "per_extension": True,
         "header_check": {"op": "range", "per_extension_key": list(TEMP_KEY_VARIANTS),
                          "limits": {"max": abs_fail}}},
    ]


# ---------------------------------------------------------------------------
# Camera-warming (dark-current glow) check on SCIENCE frames. A warming detector
# grows a diffuse dark-current glow -- a smooth large-scale background gradient --
# BEFORE the header CCD temperature moves (the header lags the pixel-level onset).
# Science otherwise runs NO image checks, so a warming detector there is invisible.
#
# We flag it with an exposure-normalized background-gradient RATE (ADU/s): the
# quarter-block MEDIAN gradient (robust to the sparse bright fibre/target flux, so it
# measures the background not the astrophysical signal) divided by exposure time.
# Rationale (see QA_TESTS_SUMMARY.md): dark current is a RATE, so a warming detector
# accrues its gradient fast (3.A.Green on 2026-05-06 06-08 = 2.1 ADU/s) while a normal
# frame's slow sky/scattered-light gradient accrues at <=~0.06 ADU/s even over 300 s.
# This is what makes it robust to bright/long science, which absolute structure/sigma
# thresholds were NOT (real dispersed-spectrum structure false-flagged bright frames).
# Detector-independent -> a single static threshold; WARN only (advisory alert).
#
# Calibration (20260505_06 + ut20260710_11 exposure ladder): a FIXED bias-level
# background gradient (<=~8 ADU, does NOT scale with time) inflates the rate at short
# exposures, so (a) exposures below MIN_EXPTIME are SKIPPED -- dark-current warming is
# not measurable in a few seconds anyway (1 s frames gave a spurious 6-8 ADU/s), and
# (b) the threshold sits above the 10 s fixed-structure ceiling (~0.8 ADU/s) and well
# below the warming value (3.A.Green 06-08 = 2.1 ADU/s). 60 s/900 s frames are <=0.13.
WARM_RATE_WARN = 1.3    # ADU/s WARN threshold (healthy 10s <=0.8, warming ~2.1)
WARM_MIN_EXPTIME = 8.0  # s; below this the rate is not computed (rule SKIPs)


def warming_rate_rule():
    """SCIENCE camera-warming monitor (WARN): exposure-normalized background-gradient
    rate. Static threshold, per extension. SKIPs when exposure time is absent/too short
    (rate undefined) -- handled in the engine."""
    return {"name": "camera_warming_gradient", "region": "full_frame",
            "metric": "background_gradient_rate", "per_extension": True, "severity": "WARN",
            "limits": {"max": WARM_RATE_WARN}}


def edge_bg_rule(table, severity="WARN"):
    return {"name": "edge_background_level", "region": "bottom_stripe", "metric": "median",
            "per_extension": True, "severity": severity,
            "expected_from_lookup": {"table": table, "keys": KEYS,
                                     "min_field": "min", "max_field": "max"}}


def level_rules(table, severity="WARN"):
    return [
        {"name": "frame_level_median", "region": "full_frame", "metric": "median",
         "per_extension": True, "severity": severity,
         "expected_from_lookup": {"table": table, "keys": KEYS,
                                  "min_field": "med_min", "max_field": "med_max"}},
        {"name": "frame_noise_rms", "region": "full_frame", "metric": "rms",
         "per_extension": True, "severity": severity,
         "expected_from_lookup": {"table": table, "keys": KEYS, "max_field": "rms_max"}},
    ]


def struct_rules(table, severity):
    return [
        {"name": "row_structure", "region": "full_frame", "metric": "row_banding",
         "per_extension": True, "severity": severity,
         "expected_from_lookup": {"table": table, "keys": KEYS, "max_field": "row_max"}},
        {"name": "column_structure", "region": "full_frame", "metric": "column_banding",
         "per_extension": True, "severity": severity,
         "expected_from_lookup": {"table": table, "keys": KEYS, "max_field": "col_max"}},
    ]


def sat_rule(table, severity):
    return {"name": "saturation_fraction", "region": "full_frame", "metric": "saturated_fraction",
            "per_extension": True, "severity": severity,
            "expected_from_lookup": {"table": table, "keys": KEYS, "max_field": "frac_max"}}


def build_cal_config():
    cfg = base_blocks()
    lookup, rule_sets = {}, {}
    lookup["temp"] = build_temp_table()   # per-camera CCD temperature caps (shared by all cal types)
    # (PRODCATG, set_name, uniform?)  uniform frames (bias/dark) additionally get a frame-level
    # band. Row/column structure is a hard FAIL for ALL cal types: on uniform frames banding is an
    # unambiguous defect, and on illuminated frames (arc/flat/sky) the per-detector heavy-tail caps
    # sit ~1.5x above the worst normal frame, so only a gross odd-structure anomaly (e.g. the
    # 2026-07-10 commissioning arc, ~2x cap on every detector) trips it -- see QA_TESTS_SUMMARY.md.
    types = [("CAL.R-BIA", "BIAS", True), ("CAL.R-DRK", "DARK", True),
             ("CAL.R-FLT", "LDLS_FLAT", False), ("CAL.R-SKY", "SKY_FLAT", False),
             ("CAL.R-ARC", "ARC_THAR", False)]
    for pc, setname, uniform in types:
        eb, st, sa = f"edge_bg_{setname}", f"struct_{setname}", f"sat_{setname}"
        lookup[eb] = build_table(pc, edge_leaf)
        lookup[st] = build_table(pc, struct_leaf)
        lookup[sa] = build_table(pc, sat_leaf)
        rules = [shutter_rule(pc), edge_bg_rule(eb, "WARN")]
        if uniform:
            lv = f"level_{setname}"; lookup[lv] = build_table(pc, level_leaf)
            rules += level_rules(lv, "WARN")
            rules += struct_rules(st, "FAIL")
            rules += [sat_rule(sa, "FAIL" if pc == "CAL.R-BIA" else "WARN")]
        else:
            rules += struct_rules(st, "FAIL")   # odd structure is a hard fail for illuminated cals too
            rules += [sat_rule(sa, "WARN")]
        rules += temp_rules()
        rule_sets[setname] = {"applies_when": {"exposure_type": pc}, "rules": rules}
    cfg["lookup_tables"] = lookup
    cfg["rule_sets"] = rule_sets
    return cfg


def build_science_config():
    """Science QA. Per-detector level/edge-background bands are still OMITTED (science
    frames carry real astrophysical 2D structure and variable sky, with no science
    baselines to calibrate absolute levels). But a WARMING detector must be caught here
    too -- science frames are exactly where it corrupts the data -- so science runs a
    camera-warming monitor: an exposure-normalized background-gradient RATE (ADU/s) that
    isolates the dark-current glow (~2.1 ADU/s on 3.A.Green, 2026-05-06 06-08) from the
    slow sky/scattered-light gradient of normal bright/long science (<=~0.06 ADU/s) --
    WARN only. Absolute structure/sigma thresholds were rejected: real dispersed-spectrum
    structure false-flagged bright standards / long exposures. Warm/shut-off cameras are
    still also covered by the CCD temperature header check (when populated) and the
    shutter fault; gross saturation is flagged too."""
    cfg = base_blocks()
    cfg["lookup_tables"] = {"temp": build_temp_table()}   # per-camera CCD temperature caps
    s = DER["shutter"]["SCI.R-*"]
    rules = [
        {"name": "shutter_exptime_consistency", "severity": "FAIL",
         "header_check": {"op": "abs_or_rel_diff", "source": "SEXPTIME", "other": "REXPTIME",
                          "abs_tol": s["abs_tol"], "rel_tol": s["rel_tol"]}},
        {"name": "saturation_fraction", "region": "full_frame", "metric": "saturated_fraction",
         "per_extension": True, "severity": "WARN", "limits": {"max": 0.02}},
        warming_rate_rule(),
    ] + temp_rules()
    cfg["rule_sets"] = {"SCIENCE": {"applies_when": {"exposure_type": "SCI.R-*"}, "rules": rules}}
    return cfg


def dump(cfg, path, header):
    with open(path, "w") as fh:
        fh.write(header)
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False, width=100)
    print("wrote", path)


if __name__ == "__main__":
    hdr = ("# LLAMAS %s QA config -- GENERATED from robust baseline analysis "
           "(2026-04-07/05-02/06-30). See QA_TESTS_SUMMARY.md; regenerate via baselines/scripts.\n")
    dump(build_cal_config(), os.path.join(QA_DIR, "qa_config_cal.yaml"), hdr % "calibration")
    dump(build_science_config(), os.path.join(QA_DIR, "qa_config_science.yaml"), hdr % "science")
