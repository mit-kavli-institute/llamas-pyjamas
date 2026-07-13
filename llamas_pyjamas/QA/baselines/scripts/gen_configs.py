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
DER = json.load(open(os.path.join(HERE, "qa_thresholds_derived.json")))
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
        },
        "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
    }


def shutter_rule(pc):
    s = DER["shutter"][pc]
    return {"name": "shutter_exptime_consistency", "severity": "FAIL",
            "header_check": {"op": "abs_or_rel_diff", "source": "SEXPTIME", "other": "REXPTIME",
                             "abs_tol": s["abs_tol"], "rel_tol": s["rel_tol"]}}


def temp_rules():
    c = DER["ccd_temp"]
    return [
        {"name": "ccd_temperature_warm", "severity": "WARN", "per_extension": True,
         "header_check": {"op": "range", "per_extension_key": "CCDTEMP_1",
                          "limits": {"min": c["cold_min"], "max": c["warm_warn_above"]}}},
        {"name": "ccd_temperature_shutoff", "severity": "FAIL", "per_extension": True,
         "header_check": {"op": "range", "per_extension_key": "CCDTEMP_1",
                          "limits": {"min": c["cold_min"], "max": c["warm_fail_above"]}}},
    ]


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
    # (PRODCATG, set_name, uniform?)  uniform frames (bias/dark) get level + FAIL structure
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
            rules += struct_rules(st, "WARN")
            rules += [sat_rule(sa, "WARN")]
        rules += temp_rules()
        rule_sets[setname] = {"applies_when": {"exposure_type": pc}, "rules": rules}
    cfg["lookup_tables"] = lookup
    cfg["rule_sets"] = rule_sets
    return cfg


def build_science_config():
    """Science QA. Image-based background/structure checks are intentionally OMITTED:
    science frames carry real astrophysical 2D structure (dispersed spectra) and
    sky/scattered light, and there are no science baselines to calibrate per-detector
    bands, so such checks would over-flag. Warm/shut-off cameras on science are caught
    by CCD temperature (when populated) and by the shutter fault that accompanies an
    accidental shutdown; gross saturation is flagged too."""
    cfg = base_blocks()
    cfg["lookup_tables"] = {"noop": {"placeholder": 1}}  # required block; unused
    s = DER["shutter"]["SCI.R-*"]
    rules = [
        {"name": "shutter_exptime_consistency", "severity": "FAIL",
         "header_check": {"op": "abs_or_rel_diff", "source": "SEXPTIME", "other": "REXPTIME",
                          "abs_tol": s["abs_tol"], "rel_tol": s["rel_tol"]}},
        {"name": "saturation_fraction", "region": "full_frame", "metric": "saturated_fraction",
         "per_extension": True, "severity": "WARN", "limits": {"max": 0.02}},
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
