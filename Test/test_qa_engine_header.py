"""Tests for the YAML QA engine header-value (header_check) rule type.

Covers the shutter (abs_or_rel_diff) and CCD-temperature (per-extension range)
checks added to qa_engine.py / qa_config_validator.py, plus graceful skipping of
absent header keywords. Builds tiny in-memory MEF frames so no external FITS
fixtures are required.
"""
import os
import sys
import numpy as np
import pytest
from astropy.io import fits

# The QA modules use bare imports (not a package), so add their dir to sys.path.
QA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "llamas_pyjamas", "QA")
sys.path.insert(0, QA_DIR)

import qa_engine as E  # noqa: E402
from qa_config_validator import QAConfigValidator  # noqa: E402


def make_mef(tmp_path, name, *, rexp, sexp, ccdtemp, level=700.0, temp_keyword="CCDTEMP_1"):
    """Write a 1-detector MEF with the given primary/extension headers.

    ``temp_keyword`` selects the CCD-temperature keyword spelling so tests can
    exercise the cross-generation variants (CCDTEMP_1 / CCDTEMP1 / CCDTEMP-1).
    """
    ph = fits.PrimaryHDU()
    ph.header["PRODCATG"] = "CAL.R-BIA"
    ph.header["READ-MDE"] = "SLOW"
    ph.header["REXPTIME"] = rexp
    ph.header["SEXPTIME"] = sexp
    ext = fits.ImageHDU(data=np.full((40, 40), level, dtype=np.float32))
    ext.header["COLOR"] = "Green"
    ext.header["BENCH"] = 1
    ext.header["SIDE"] = "A"
    if ccdtemp is not None:
        ext.header[temp_keyword] = ccdtemp
    path = tmp_path / name
    fits.HDUList([ph, ext]).writeto(path, overwrite=True)
    return str(path)


CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [{"name": "1.A.Green", "hdu_index": 1, "color": "Green", "bench": 1, "side": "A"}],
    "regions": {"full_frame": {"type": "full"}},
    "metrics": {"mean": {"type": "mean"}},
    "lookup_tables": {"noop": {"a": 1}},  # unused but block is required + non-empty
    "rule_sets": {
        "BIAS": {
            "applies_when": {"exposure_type": "CAL.R-BIA"},
            "rules": [
                {"name": "shutter", "severity": "FAIL",
                 "header_check": {"op": "abs_or_rel_diff", "source": "SEXPTIME",
                                  "other": "REXPTIME", "abs_tol": 0.3, "rel_tol": 0.10}},
                {"name": "temp", "severity": "WARN", "per_extension": True,
                 "header_check": {"op": "range", "per_extension_key": "CCDTEMP_1",
                                  "limits": {"min": -140, "max": -60}}},
                {"name": "level", "region": "full_frame", "metric": "mean",
                 "per_extension": True, "severity": "FAIL",
                 "limits": {"min": 600, "max": 800}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def run(path):
    return E.QAEngine(CONFIG).run(__import__("pathlib").Path(path))


def result_for(report, rule):
    return next(r for r in report["results"] if r["rule"] == rule)


def test_config_is_valid():
    errors = QAConfigValidator(CONFIG).validate()
    assert errors == [], [str(e) for e in errors]


def test_shutter_ok_passes(tmp_path):
    rep = run(make_mef(tmp_path, "ok.fits", rexp=0.001, sexp=0.003, ccdtemp="-90.0"))
    assert result_for(rep, "shutter")["passed"] is True
    assert rep["overall_verdict"] == "PASS"


def test_shutter_gross_fault_fails(tmp_path):
    # long exposure that only got a fraction of requested time (warm-incident signature)
    rep = run(make_mef(tmp_path, "bad.fits", rexp=600.0, sexp=181.0, ccdtemp="-90.0"))
    sh = result_for(rep, "shutter")
    assert sh["passed"] is False
    assert sh["verdict_effect"] == "FAIL"
    assert rep["overall_verdict"] == "FAIL"


def test_short_exposure_overhead_passes(tmp_path):
    # short exposure with fixed shutter overhead: big relative diff, tiny absolute -> PASS
    rep = run(make_mef(tmp_path, "short.fits", rexp=0.05, sexp=0.12, ccdtemp="-90.0"))
    assert result_for(rep, "shutter")["passed"] is True


def test_warm_temperature_flags_warn(tmp_path):
    rep = run(make_mef(tmp_path, "warm.fits", rexp=0.001, sexp=0.003, ccdtemp="-40.0"))
    t = result_for(rep, "temp")
    assert t["passed"] is False
    assert t["verdict_effect"] == "WARN"
    assert rep["overall_verdict"] == "WARN"


def test_missing_temperature_is_skipped_not_failed(tmp_path):
    rep = run(make_mef(tmp_path, "notemp.fits", rexp=0.001, sexp=0.003, ccdtemp=None))
    t = result_for(rep, "temp")
    assert t["status"] == "SKIPPED"
    assert t["passed"] is True
    assert rep["overall_verdict"] == "PASS"


def test_string_temperature_is_parsed(tmp_path):
    # CCDTEMP_1 is stored as a string in real data; it must parse to a number.
    rep = run(make_mef(tmp_path, "strtemp.fits", rexp=0.001, sexp=0.003, ccdtemp="-90.0"))
    assert result_for(rep, "temp")["status"] == "EVALUATED"


# ------- engine robustness (per-review fixes) -------

def json_roundtrip(obj):
    import copy
    return copy.deepcopy(obj)


def _lookup_config():
    """A config with one lookup-backed level rule whose table only covers SLOW."""
    cfg = json_roundtrip(CONFIG)
    cfg["lookup_tables"] = {"lvl": {"1.A.Green": {"SLOW": {"lo": 600, "hi": 800}}}}
    cfg["rule_sets"] = {"BIAS": {"applies_when": {"exposure_type": "CAL.R-BIA"}, "rules": [
        {"name": "level", "region": "full_frame", "metric": "mean", "per_extension": True,
         "severity": "FAIL", "expected_from_lookup": {
             "table": "lvl", "keys": [{"from": "extension.name"}, {"from": "metadata.readout_mode"}],
             "min_field": "lo", "max_field": "hi"}},
        {"name": "shutter", "severity": "FAIL", "header_check": {
            "op": "abs_or_rel_diff", "source": "SEXPTIME", "other": "REXPTIME",
            "abs_tol": 0.3, "rel_tol": 0.1}},
    ]}}
    cfg["metrics"] = {"mean": {"type": "mean"}}
    return cfg


def test_off_mode_lookup_miss_skips_not_crashes(tmp_path):
    # Frame read out in FAST, but the lookup table only models SLOW -> that ONE rule
    # is SKIPPED (not an engine crash, not a silent PASS of everything).
    cfg = _lookup_config()
    path = make_mef(tmp_path, "fastbias.fits", rexp=0.001, sexp=0.003, ccdtemp=None)
    with fits.open(path, mode="update") as h:
        h[0].header["READ-MDE"] = "FAST"
    rep = E.QAEngine(cfg).run(__import__("pathlib").Path(path))
    lvl = next(r for r in rep["results"] if r["rule"] == "level")
    assert lvl["status"] == "SKIPPED"
    # the shutter rule still evaluated normally
    assert any(r["rule"] == "shutter" and r["status"] == "EVALUATED" for r in rep["results"])


def test_region_out_of_bounds_is_error_verdict(tmp_path):
    cfg = json_roundtrip(CONFIG)
    cfg["regions"] = {"full_frame": {"type": "full"},
                      "oob": {"type": "rectangle", "x_start": 0, "x_end": 5000,
                              "y_start": 0, "y_end": 5000}}
    cfg["rule_sets"]["BIAS"]["rules"] = [
        {"name": "oob_level", "region": "oob", "metric": "mean", "per_extension": True,
         "severity": "FAIL", "limits": {"min": 0, "max": 1e9}}]
    path = make_mef(tmp_path, "oob.fits", rexp=0.001, sexp=0.003, ccdtemp=None)
    rep = E.QAEngine(cfg).run(__import__("pathlib").Path(path))
    assert rep["overall_verdict"] == "ERROR"  # ERROR must not be masked as PASS
    assert any(r["status"] == "ERROR" for r in rep["results"])


def test_no_rules_matched_warns(tmp_path):
    # A frame whose PRODCATG matches no rule set is surfaced (WARN), not silent PASS.
    path = make_mef(tmp_path, "unmatched.fits", rexp=0.001, sexp=0.003, ccdtemp=None)
    with fits.open(path, mode="update") as h:
        h[0].header["PRODCATG"] = "SCI.R-XX"
    rep = E.QAEngine(CONFIG).run(__import__("pathlib").Path(path))
    assert rep["overall_verdict"] == "WARN"
    assert any(r["status"] == "NO_RULES_MATCHED" for r in rep["results"])


def test_validator_rejects_per_extension_key_without_per_extension():
    bad = json_roundtrip(CONFIG)
    bad["rule_sets"]["BIAS"]["rules"].append(
        {"name": "badtemp", "severity": "WARN",  # missing per_extension: true
         "header_check": {"op": "range", "per_extension_key": "CCDTEMP_1",
                          "limits": {"min": -140, "max": -60}}})
    errors = [str(e) for e in QAConfigValidator(bad).validate()]
    assert any("per_extension: true" in e for e in errors), errors


# ------- per-detector camera-temperature monitor (warming) -------
# The real check judges each camera against its OWN healthy baseline: WARN when it
# warms past warn_max, FAIL past fail_max (both per-camera, from a lookup table),
# plus an absolute shut-off FAIL. The keyword spelling varies across file
# generations. Baselines below: 1.A.Green cool -90, warn>-82, fail>-75.

TEMP_VARIANTS = ["CCDTEMP_1", "CCDTEMP1", "CCDTEMP-1"]


def _temp_config():
    cfg = json_roundtrip(CONFIG)
    cfg["lookup_tables"] = {"temp": {"1.A.Green": {"cold_min": -140.0, "warn_max": -82.0,
                                                   "fail_max": -75.0}}}
    cfg["rule_sets"] = {"BIAS": {"applies_when": {"exposure_type": "CAL.R-BIA"}, "rules": [
        {"name": "ccd_temperature_warm", "severity": "WARN", "per_extension": True,
         "header_check": {"op": "range", "per_extension_key": list(TEMP_VARIANTS)},
         "expected_from_lookup": {"table": "temp", "keys": [{"from": "extension.name"}],
                                  "min_field": "cold_min", "max_field": "warn_max"}},
        {"name": "ccd_temperature_hot", "severity": "FAIL", "per_extension": True,
         "header_check": {"op": "range", "per_extension_key": list(TEMP_VARIANTS)},
         "expected_from_lookup": {"table": "temp", "keys": [{"from": "extension.name"}],
                                  "max_field": "fail_max"}},
        {"name": "ccd_temperature_shutoff", "severity": "FAIL", "per_extension": True,
         "header_check": {"op": "range", "per_extension_key": list(TEMP_VARIANTS),
                          "limits": {"max": -60.0}}},
    ]}}
    return cfg


def _run(cfg, path):
    return E.QAEngine(cfg).run(__import__("pathlib").Path(path))


def test_per_detector_temp_config_is_valid():
    errors = QAConfigValidator(_temp_config()).validate()
    assert errors == [], [str(e) for e in errors]


def test_per_detector_temp_at_baseline_passes(tmp_path):
    rep = _run(_temp_config(), make_mef(tmp_path, "cool.fits", rexp=0.001, sexp=0.003,
                                        ccdtemp="-90.0"))
    warm = result_for(rep, "ccd_temperature_warm")
    assert warm["status"] == "EVALUATED"
    assert warm["passed"] is True
    assert rep["overall_verdict"] == "PASS"


def test_per_detector_temp_warm_flags_warn(tmp_path):
    # -80 is above warn_max (-82) but below fail_max (-75): WARN only.
    rep = _run(_temp_config(), make_mef(tmp_path, "warm.fits", rexp=0.001, sexp=0.003,
                                        ccdtemp="-80.0"))
    warm = result_for(rep, "ccd_temperature_warm")
    assert warm["passed"] is False and warm["verdict_effect"] == "WARN"
    assert result_for(rep, "ccd_temperature_hot")["passed"] is True
    assert rep["overall_verdict"] == "WARN"


def test_per_detector_temp_hot_flags_fail(tmp_path):
    # -70 is above fail_max (-75): FAIL.
    rep = _run(_temp_config(), make_mef(tmp_path, "hot.fits", rexp=0.001, sexp=0.003,
                                        ccdtemp="-70.0"))
    assert result_for(rep, "ccd_temperature_hot")["verdict_effect"] == "FAIL"
    assert rep["overall_verdict"] == "FAIL"


def test_temp_keyword_variant_no_separator_is_read(tmp_path):
    # Commissioning frames store the value under CCDTEMP1 (no underscore); the check
    # must read it, not silently SKIP as it did before the variant fix.
    rep = _run(_temp_config(), make_mef(tmp_path, "variant.fits", rexp=0.001, sexp=0.003,
                                        ccdtemp="-80.0", temp_keyword="CCDTEMP1"))
    warm = result_for(rep, "ccd_temperature_warm")
    assert warm["status"] == "EVALUATED"
    assert warm["passed"] is False and warm["verdict_effect"] == "WARN"


def test_temp_absent_all_variants_is_skipped(tmp_path):
    rep = _run(_temp_config(), make_mef(tmp_path, "notemp.fits", rexp=0.001, sexp=0.003,
                                        ccdtemp=None))
    warm = result_for(rep, "ccd_temperature_warm")
    assert warm["status"] == "SKIPPED"
    assert rep["overall_verdict"] == "PASS"


def test_validator_rejects_lookup_with_nonrange_op():
    bad = _temp_config()
    bad["rule_sets"]["BIAS"]["rules"].append(
        {"name": "badlookup", "severity": "WARN",
         "header_check": {"op": "abs_or_rel_diff", "source": "A", "other": "B",
                          "abs_tol": 1.0, "rel_tol": 0.1},
         "expected_from_lookup": {"table": "temp", "keys": [{"from": "extension.name"}],
                                  "max_field": "fail_max"}})
    errors = [str(e) for e in QAConfigValidator(bad).validate()]
    assert any("only valid with op 'range'" in e for e in errors), errors


def test_shipped_config_reds_get_extra_headroom():
    # Integration guard on the generated cal config: the red detectors get a wider
    # warm margin than green/blue (warn +10 vs +8), so the warn->fail band differs.
    import yaml
    import pathlib
    cal = yaml.safe_load((pathlib.Path(QA_DIR) / "qa_config_cal.yaml").read_text())
    temp = cal["lookup_tables"]["temp"]
    red, green = temp["1.A.Red"], temp["1.A.Green"]
    assert round(red["fail_max"] - red["warn_max"], 1) == 6.0     # red: +16 - +10
    assert round(green["fail_max"] - green["warn_max"], 1) == 7.0  # green: +15 - +8


# ------- science camera-warming: exposure-normalized background-gradient RATE -------
# A warming detector grows a dark-current glow -- a smooth large-scale background
# gradient -- that accrues fast (ADU/s). Normalizing the quarter-block median gradient
# by exposure time isolates that dark-current RATE, which separates a warming detector
# (3.A.Green on 2026-05-06 06-08 = ~2.1 ADU/s) from bright/long science whose slow
# sky/scattered-light gradient accrues at <=~0.06 ADU/s. WARN only (alert, don't fail).
# Structure- and sigma-based variants were rejected: no absolute floor separates a
# warming glow from real dispersed-spectrum structure on a bright/long exposure.

_N = 64
_FLAT = np.full((_N, _N), 700.0, dtype=np.float32)
# vertical background ramp (bottom quarter ~ +45 ADU over the top) -> a real gradient
_VGRAD = (700.0 + np.linspace(0.0, 60.0, _N)[:, None] * np.ones((1, _N))).astype(np.float32)
# bright fibers (sparse columns) on a FLAT background: quarter-block medians ignore the
# sparse bright flux, so the gradient (hence the rate) stays ~0 -> must not warn.
_STRIPES = np.full((_N, _N), 700.0, dtype=np.float32)
_STRIPES[:, ::8] = 6000.0


def make_rate_mef(tmp_path, name, pattern, *, sexp=10.0, rexp=None, exptime=None):
    """1-detector science MEF (PRODCATG SCI.R-SL, ext 1.A.Green) carrying ``pattern`` as
    its image data, with the given exposure-time headers (``None`` omits a keyword) so
    the gradient-rate metric can be exercised."""
    ph = fits.PrimaryHDU()
    ph.header["PRODCATG"] = "SCI.R-SL"
    ph.header["READ-MDE"] = "SLOW"
    if sexp is not None:
        ph.header["SEXPTIME"] = sexp
    if rexp is not None:
        ph.header["REXPTIME"] = rexp
    if exptime is not None:
        ph.header["EXPTIME"] = exptime
    ext = fits.ImageHDU(data=np.asarray(pattern, dtype=np.float32))
    ext.header["COLOR"] = "Green"
    ext.header["BENCH"] = 1
    ext.header["SIDE"] = "A"
    path = tmp_path / name
    fits.HDUList([ph, ext]).writeto(path, overwrite=True)
    return str(path)


def _rate_config(threshold=1.3, min_exptime=8.0):
    cfg = json_roundtrip(CONFIG)
    cfg["regions"] = {"full_frame": {"type": "full"}}
    cfg["metrics"] = {"bg_rate": {"type": "background_gradient_rate",
                                  "exptime_keys": ["SEXPTIME", "REXPTIME", "EXPTIME"],
                                  "min_exptime": min_exptime}}
    cfg["lookup_tables"] = {"noop": {"a": 1}}
    cfg["rule_sets"] = {"SCIENCE": {"applies_when": {"exposure_type": "SCI.R-SL"}, "rules": [
        {"name": "camera_warming_gradient", "region": "full_frame", "metric": "bg_rate",
         "per_extension": True, "severity": "WARN", "limits": {"max": threshold}},
    ]}}
    return cfg


def test_science_rate_config_is_valid():
    errors = QAConfigValidator(_rate_config()).validate()
    assert errors == [], [str(e) for e in errors]


def test_flat_frame_passes_rate(tmp_path):
    # No background gradient -> rate ~0 -> PASS.
    rep = _run(_rate_config(), make_rate_mef(tmp_path, "flat.fits", _FLAT, sexp=10.0))
    warm = result_for(rep, "camera_warming_gradient")
    assert warm["status"] == "EVALUATED" and warm["passed"] is True
    assert warm["measured_value"] < 0.1
    assert rep["overall_verdict"] == "PASS"


def test_short_exposure_gradient_warns(tmp_path):
    # A ~45-ADU background gradient accrued in a 10 s exposure -> ~4.5 ADU/s -> WARN.
    rep = _run(_rate_config(), make_rate_mef(tmp_path, "warm.fits", _VGRAD, sexp=10.0))
    warm = result_for(rep, "camera_warming_gradient")
    assert warm["status"] == "EVALUATED"
    assert warm["passed"] is False and warm["verdict_effect"] == "WARN"
    assert warm["measured_value"] > 1.3
    assert rep["overall_verdict"] == "WARN"


def test_same_gradient_long_exposure_passes(tmp_path):
    # The SAME gradient over a 100 s exposure is only ~0.45 ADU/s: exposure
    # normalization is what tells a slow sky gradient from a fast warming glow.
    rep = _run(_rate_config(), make_rate_mef(tmp_path, "long.fits", _VGRAD, sexp=100.0))
    warm = result_for(rep, "camera_warming_gradient")
    assert warm["status"] == "EVALUATED" and warm["passed"] is True
    assert warm["measured_value"] < 1.3
    assert rep["overall_verdict"] == "PASS"


def test_bright_fibers_flat_background_passes(tmp_path):
    # Bright standards land in fibers, not the background. Quarter-block MEDIANS ignore
    # the sparse bright flux, so a bright short exposure with a flat background PASSes.
    rep = _run(_rate_config(), make_rate_mef(tmp_path, "bright.fits", _STRIPES, sexp=10.0))
    warm = result_for(rep, "camera_warming_gradient")
    assert warm["status"] == "EVALUATED" and warm["passed"] is True
    assert warm["measured_value"] < 0.1
    assert rep["overall_verdict"] == "PASS"


def test_missing_exposure_time_skips(tmp_path):
    # A rate is undefined without an exposure time -> SKIP this rule, never crash/fail.
    rep = _run(_rate_config(), make_rate_mef(tmp_path, "noexp.fits", _VGRAD,
                                             sexp=None, rexp=None, exptime=None))
    warm = result_for(rep, "camera_warming_gradient")
    assert warm["status"] == "SKIPPED"
    assert warm["passed"] is True
    assert rep["overall_verdict"] == "PASS"


def test_below_min_exposure_skips(tmp_path):
    # A ~0 s frame's fixed bias-structure gradient would inflate the rate; below
    # min_exptime the rule SKIPs rather than false-WARNing.
    rep = _run(_rate_config(), make_rate_mef(tmp_path, "tiny.fits", _VGRAD, sexp=2.0))
    warm = result_for(rep, "camera_warming_gradient")
    assert warm["status"] == "SKIPPED"
    assert rep["overall_verdict"] == "PASS"


def test_shipped_science_config_has_warming_rate():
    # Integration guard on the generated science config: warming is a single WARN
    # gradient-rate rule; the rejected structure approach (struct_science table +
    # row/column_structure rules on science) must be gone.
    import yaml
    import pathlib
    sci = yaml.safe_load((pathlib.Path(QA_DIR) / "qa_config_science.yaml").read_text())
    rules = {r["name"]: r for r in sci["rule_sets"]["SCIENCE"]["rules"]}
    warm = rules["camera_warming_gradient"]
    assert warm["severity"] == "WARN"
    assert warm["metric"] == "background_gradient_rate"
    metric = sci["metrics"]["background_gradient_rate"]
    assert metric["type"] == "background_gradient_rate"
    assert metric["min_exptime"] > 0
    assert "row_structure" not in rules and "column_structure" not in rules
    assert "struct_science" not in sci["lookup_tables"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
