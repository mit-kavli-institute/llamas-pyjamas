#!/usr/bin/env python3
"""LLAMAS calibration QA entry point.

``check_image`` is the single function ``QA_assess.py`` calls. It:

  1. loads and validates the YAML rule definitions,
  2. runs a header preflight (can the image type be identified?) before any
     type-specific pixel test is allowed to count,
  3. delegates the type-specific metric rules to ``qa_engine.QAEngine``,
  4. collapses the engine verdict into a simple pass/warn/fail result.

Failure model:
- *System* problems (missing file/config, invalid config, astropy absent)
  raise, so the CLI reports a system error.
- *QA* problems (unidentifiable header, triggered rules) are returned as a
  "fail"/"warn" status, never raised.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .qa_config_validator import QAConfigValidator
from .qa_engine import QAEngine, QAEngineError, fits as _fits, load_yaml

DEFAULT_CONFIG_NAME = "qa_config.yaml"
# Generated per-type configs (in the QA package dir), auto-selected by PRODCATG.
CAL_CONFIG_NAME = "qa_config_cal.yaml"
SCIENCE_CONFIG_NAME = "qa_config_science.yaml"
_VERDICT_TO_STATUS = {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}


def check_image(
    input_path: str,
    suite: str = "basic_cal",
    qa_yaml: str | None = None,
    calib_root: str | None = None,
    report: str | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the QA suite on a single calibration or science FITS/MEF image.

    Returns a dict with at least ``status`` ("pass"|"warn"|"fail") and
    ``message``. Raises ``QAEngineError`` on system-level problems.
    """
    image_path = Path(input_path).expanduser()
    if not image_path.is_file():
        raise QAEngineError(f"input is not a file: {image_path}")
    if _fits is None:
        raise QAEngineError("astropy is required to read FITS files; install with "
                            "`pip install astropy`.")

    config_path = _resolve_config_path(qa_yaml, calib_root, suite, image_path)
    config = load_yaml(config_path)
    _validate_config(config, config_path)

    engine = QAEngine(config)
    try:
        engine_report = engine.run(image_path)
    except QAEngineError as exc:
        # Config is valid, so a runtime error means this image/header is unfit
        # for the rules it matched -> a QA failure, not a system error.
        return _result("fail", f"QA could not be completed: {exc}",
                       suite, image_path, report)

    # Header preflight: the image type must be identifiable from the header.
    if not engine_report["active_rule_sets"]:
        return _result("fail", _unidentified_message(engine_report, config),
                       suite, image_path, report, engine_report)

    status = _VERDICT_TO_STATUS[engine_report["overall_verdict"]]
    message = _build_message(engine_report)
    if verbose:
        _print_details(engine_report)

    return _result(status, message, suite, image_path, report, engine_report)


def _resolve_config_path(qa_yaml: str | None, calib_root: str | None, suite: str,
                         image_path: Path) -> Path:
    if qa_yaml:
        return Path(qa_yaml).expanduser()
    base = Path(calib_root).expanduser() if calib_root else Path(__file__).resolve().parent
    if suite:
        by_suite = base / f"{suite}.yaml"
        if by_suite.exists():
            return by_suite
    # Default path: auto-select the generated cal/science config from the frame's
    # PRODCATG so QA_assess runs the real per-type rules, not the stale base config.
    auto = _config_for_prodcatg(image_path, base)
    if auto is not None:
        return auto
    return base / DEFAULT_CONFIG_NAME


def _config_for_prodcatg(image_path: Path, base: Path) -> Path | None:
    """Pick the generated per-type config from the frame's primary-header PRODCATG.

    Returns the cal config for ``CAL*`` frames and the science config for ``SCI*``
    frames, or ``None`` when the type is unknown/missing, the config file is absent,
    or the FITS cannot be opened (the caller then falls back to the base config).
    """
    try:
        with _fits.open(image_path, memmap=False) as hdul:
            prodcatg = str(hdul[0].header.get("PRODCATG", "")).strip().upper()
    except Exception:
        return None
    if prodcatg.startswith("CAL"):
        candidate = base / CAL_CONFIG_NAME
    elif prodcatg.startswith("SCI"):
        candidate = base / SCIENCE_CONFIG_NAME
    else:
        return None
    return candidate if candidate.exists() else None


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    errors = QAConfigValidator(config, filename=str(config_path)).validate()
    if errors:
        details = "\n".join(f"  {error}" for error in errors)
        raise QAEngineError(f"invalid QA configuration: {config_path}\n{details}")


def _unidentified_message(engine_report: dict[str, Any], config: dict[str, Any]) -> str:
    """Explain why no rule set matched: missing keyword vs unknown type."""
    metadata = engine_report["metadata"]
    needed = sorted({key for rule_set in config["rule_sets"].values()
                     for key in rule_set["applies_when"]})
    missing = [key for key in needed if metadata.get(key) is None]
    if missing:
        return f"header missing identification keyword(s): {', '.join(missing)}"
    return f"unrecognised image type for header metadata {metadata}"


def _build_message(engine_report: dict[str, Any]) -> str:
    verdict = engine_report["overall_verdict"]
    summary = engine_report["summary"]
    if verdict == "PASS":
        return f"PASS: {summary['passed_checks']}/{summary['evaluated_checks']} checks passed"
    triggered = [result for result in engine_report["results"]
                 if result["status"] == "EVALUATED" and result["verdict_effect"] == verdict]
    shown = ", ".join(f"{result['rule']}@{result['extension']}" for result in triggered[:5])
    extra = "" if len(triggered) <= 5 else f" (+{len(triggered) - 5} more)"
    return f"{verdict}: {len(triggered)} {verdict.lower()} check(s): {shown}{extra}"


def _print_details(engine_report: dict[str, Any]) -> None:
    for result in engine_report["results"]:
        if result["status"] == "EVALUATED" and not result["passed"]:
            print(f"  {result['verdict_effect']} {result['rule']} @ "
                  f"{result['extension']}: {result['message']}", file=sys.stderr)


def _result(
    status: str,
    message: str,
    suite: str,
    image_path: Path,
    report_path: str | None,
    engine_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "message": message,
        "suite": suite,
        "fits_file": str(image_path),
    }
    if engine_report is not None:
        result["overall_verdict"] = engine_report["overall_verdict"]
        result["summary"] = engine_report["summary"]

    if report_path:
        payload = dict(result)
        if engine_report is not None:
            payload["report"] = engine_report
        out = Path(report_path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return result