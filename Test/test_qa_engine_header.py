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


def make_mef(tmp_path, name, *, rexp, sexp, ccdtemp, level=700.0):
    """Write a 1-detector MEF with the given primary/extension headers."""
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
        ext.header["CCDTEMP_1"] = ccdtemp
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


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
