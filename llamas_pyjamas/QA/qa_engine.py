#!/usr/bin/env python3
"""Minimal FITS/MEF QA engine driven by the YAML configuration.
 
This engine assumes the YAML structure validated by qa_config_validator.py.
It validates the configuration before every run, reads FITS/MEF files,
extracts configured metadata, evaluates matching rule sets, and writes JSON
QA reports.
 
Input handling:
- If the input path is a FITS file, a single-file QA report is produced.
- If the input path is a directory, all FITS files in that directory are
  processed in batch mode using ProcessPoolExecutor.
"""
 
from __future__ import annotations
 
import argparse
import concurrent.futures
import fnmatch
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
 
import yaml
 
try:
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit("ERROR: numpy is required to run the QA engine.") from exc
 
try:
    from astropy.io import fits
except ImportError:  # pragma: no cover
    fits = None
 
try:
    from qa_config_validator import QAConfigValidator
except ImportError:  # pragma: no cover
    QAConfigValidator = None
 
 
FITS_SUFFIXES = (
    ".fits",
    ".fit",
    ".fts",
)
 
 
@dataclass
class RuleResult:
    rule_set: str
    rule: str
    extension: str | None
    hdu_index: int | None
    region: str
    metric: str
    measured_value: float | None
    passed: bool
    severity: str
    verdict_effect: str
    limits: dict[str, float] | None = None
    lookup_path: list[str] | None = None
    status: str = "EVALUATED"
    message: str = ""
 
 
class QAEngineError(RuntimeError):
    """Raised when the QA engine cannot complete evaluation."""


class LookupMissError(QAEngineError):
    """Raised when a lookup table has no entry for a detector/mode combination.

    Treated as a benign SKIP for that one rule+extension (e.g. an off-mode frame
    whose readout mode is not modelled), rather than an ERROR that aborts the file.
    """
 
 
class QAEngine:
    def __init__(self, config: dict[str, Any]):
        self.config = config
 
    @staticmethod
    def normalize(value: Any) -> str:
        """Normalize metadata values and lookup keys for robust comparison."""
        return str(value).strip().lower()
 
    def run(self, fits_path: Path) -> dict[str, Any]:
        if fits is None:
            raise QAEngineError("astropy is required to read FITS files; "
                                "install it with `pip install astropy`.")
 
        # memmap=False is required for FITS files containing BZERO/BSCALE/BLANK
        # keywords, because Astropy needs to scale the image data in memory.
        with fits.open(fits_path, memmap=False) as hdul:
            metadata = self._extract_metadata(hdul)
            active_rule_sets = self._select_rule_sets(metadata)
            results: list[RuleResult] = []

            if not active_rule_sets:
                results.append(self._no_rules_matched_result(metadata))

            for rule_set_name, rule_set in active_rule_sets.items():
                for rule in rule_set["rules"]:
                    if "header_check" in rule:
                        results.extend(
                            self._evaluate_header_rule(hdul, metadata, rule_set_name, rule)
                        )
                    elif rule.get("per_extension", False):
                        for extension in self.config["extensions"]:
                            results.append(
                                self._evaluate_rule_for_extension(
                                    hdul, metadata, rule_set_name, rule, extension
                                )
                            )
                    else:
                        extension = self._default_extension_for_non_per_extension_rule(hdul)
                        results.append(
                            self._evaluate_rule_for_extension(
                                hdul, metadata, rule_set_name, rule, extension
                            )
                        )
 
        verdict = self._final_verdict(results)
        return {
            "fits_file": str(fits_path),
            "instrument": self.config.get("instrument", {}),
            "metadata": metadata,
            "active_rule_sets": list(active_rule_sets.keys()),
            "overall_verdict": verdict,
            "summary": self._summary(results),
            "results": [asdict(result) for result in results],
        }
 
    def _extract_metadata(self, hdul: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for internal_name, fits_keyword in self.config["metadata_keys"].items():
            raw_value = self._find_header_value(hdul, fits_keyword)
            if raw_value is None:
                metadata[internal_name] = None
            elif isinstance(raw_value, str):
                metadata[internal_name] = self.normalize(raw_value)
            else:
                metadata[internal_name] = raw_value
        return metadata
 
    @staticmethod
    def _find_header_value(hdul: Any, keyword: str) -> Any | None:
        """Find a FITS header value, checking primary first, then extensions.
 
        Header keyword names are intentionally not normalized. Values are
        normalized later only when they are strings.
        """
        for hdu in hdul:
            if keyword in hdu.header:
                return hdu.header[keyword]
        return None
 
    def _select_rule_sets(self, metadata: dict[str, Any]) -> dict[str, Any]:
        selected: dict[str, Any] = {}
        for name, rule_set in self.config["rule_sets"].items():
            if self._applies_when_matches(rule_set["applies_when"], metadata):
                selected[name] = rule_set
        return selected
 
    @staticmethod
    def _match(value: Any, pattern: Any) -> bool:
        value_s = str(value).strip().lower()
        pattern_s = str(pattern).strip().lower()
        return fnmatch.fnmatch(value_s, pattern_s)
 
    def _applies_when_matches(self, applies_when: dict[str, Any], metadata: dict[str, Any]) -> bool:
        for field, expected in applies_when.items():
            actual = metadata.get(field)
            if actual is None:
                return False
            if isinstance(actual, str) or isinstance(expected, str):
                if not self._match(self.normalize(actual), self.normalize(expected)):
                    return False
            elif actual != expected:
                return False
        return True
 
    def _default_extension_for_non_per_extension_rule(self, hdul: Any) -> dict[str, Any]:
        # Current YAML uses per_extension rules. This fallback keeps the engine
        # deterministic if a future rule omits per_extension or sets it false.
        for extension in self.config["extensions"]:
            hdu_index = extension["hdu_index"]
            if hdu_index >= len(hdul):
                continue
            data = getattr(hdul[hdu_index], "data", None)
            if data is not None and not self._is_placeholder_data(data):
                return extension
        raise QAEngineError("no configured extension with usable image data is available")

    # ---------------------------------------------------------------------
    # Header-value rules (shutter/EXPTIME, CCD temperature)
    # ---------------------------------------------------------------------
    def _evaluate_header_rule(
        self,
        hdul: Any,
        metadata: dict[str, Any],
        rule_set_name: str,
        rule: dict[str, Any],
    ) -> list[RuleResult]:
        """Evaluate a ``header_check`` rule.

        Header rules check FITS header values rather than image-region metrics.
        A non-``per_extension`` rule (e.g. shutter REXPTIME vs SEXPTIME) is
        evaluated once from the file's global header. A ``per_extension`` rule
        (e.g. per-detector CCD temperature) is evaluated once per configured
        extension, reading the keyword from that extension's own header. Absent
        keywords are reported with status ``SKIPPED`` and never affect the
        verdict, because keywords such as CCDTEMP_1 are only intermittently
        populated. ``metadata`` is threaded through so a ``range`` check can
        resolve per-detector limits from a lookup table (``expected_from_lookup``).
        """
        if rule.get("per_extension", False):
            return [self._eval_header_one(hdul, metadata, rule_set_name, rule, extension)
                    for extension in self.config["extensions"]]
        return [self._eval_header_one(hdul, metadata, rule_set_name, rule, None)]

    def _eval_header_one(
        self,
        hdul: Any,
        metadata: dict[str, Any],
        rule_set_name: str,
        rule: dict[str, Any],
        extension: dict[str, Any] | None,
    ) -> RuleResult:
        hc = rule["header_check"]
        op = hc["op"]
        severity = rule["severity"]
        ext_name = extension.get("name") if extension else None
        hdu_index = extension.get("hdu_index") if extension else None

        def skipped(msg: str) -> RuleResult:
            return RuleResult(
                rule_set=rule_set_name, rule=rule["name"], extension=ext_name,
                hdu_index=hdu_index, region="header", metric=f"header:{op}",
                measured_value=None, passed=True, severity=severity,
                verdict_effect="SKIPPED", limits=None, lookup_path=None,
                status="SKIPPED", message=msg,
            )

        # For per-extension keys (e.g. CCDTEMP_1), resolve this extension's header.
        ext_header = None
        if extension is not None:
            if hdu_index is None or hdu_index >= len(hdul):
                return skipped(f"extension {ext_name!r} references missing HDU index {hdu_index}")
            ext_header = hdul[hdu_index].header

        # --- single-value range check ---
        if op == "range":
            if "per_extension_key" in hc:
                # per_extension_key may be a single keyword or a list of spelling
                # variants (e.g. CCDTEMP_1 / CCDTEMP1 / CCDTEMP-1 differ across
                # file generations); use the first one populated in this header.
                keys = hc["per_extension_key"]
                keys = [keys] if isinstance(keys, str) else list(keys)
                key, raw = keys[0], None
                if ext_header is not None:
                    for candidate in keys:
                        if ext_header.get(candidate) is not None:
                            key, raw = candidate, ext_header.get(candidate)
                            break
            else:
                key = hc["source"]
                raw = self._find_header_value(hdul, key)
            value = self._to_number(raw)
            if value is None:
                return skipped(f"header keyword {key!r} absent or non-numeric")
            # Limits come either from a static block (hc["limits"]) or, for
            # per-detector checks (e.g. per-camera temperature caps), from a
            # lookup table via the rule's expected_from_lookup.
            if "expected_from_lookup" in rule:
                if extension is None:
                    return skipped("expected_from_lookup requires a per_extension rule")
                try:
                    limits, _ = self._resolve_rule_limits(rule, metadata, extension)
                except LookupMissError:
                    return skipped(f"no lookup entry for extension {ext_name!r}")
            else:
                limits = {k: float(v) for k, v in hc["limits"].items()}
            passed, message = self._check_limits(value, limits)
            return self._header_result(rule_set_name, rule, op, ext_name, hdu_index,
                                       f"header:{key}", value, passed, severity,
                                       limits, message)

        # --- difference checks between two global keywords ---
        a = self._to_number(self._find_header_value(hdul, hc["source"]))
        b = self._to_number(self._find_header_value(hdul, hc["other"]))
        if a is None or b is None:
            return skipped(f"header keyword {hc['source']!r} or {hc['other']!r} "
                           "absent or non-numeric")
        absdiff = abs(a - b)
        reldiff = absdiff / abs(b) if b != 0 else float("inf")
        abs_tol = hc.get("abs_tol")
        rel_tol = hc.get("rel_tol")
        if op == "abs_diff":
            passed = absdiff <= float(abs_tol)
            message = (f"|{hc['source']}-{hc['other']}|={absdiff:.4g} "
                       f"{'<=' if passed else '>'} abs_tol {abs_tol}")
        elif op == "rel_diff":
            passed = reldiff <= float(rel_tol)
            message = (f"rel|{hc['source']}-{hc['other']}|={reldiff:.4g} "
                       f"{'<=' if passed else '>'} rel_tol {rel_tol}")
        else:  # abs_or_rel_diff: passes if within EITHER tolerance
            passed = (absdiff <= float(abs_tol)) or (reldiff <= float(rel_tol))
            message = (f"absdiff={absdiff:.4g} (abs_tol {abs_tol}), "
                       f"reldiff={reldiff:.4g} (rel_tol {rel_tol}) -> "
                       f"{'within tolerance' if passed else 'exceeds both'}")
        limits = {k: float(v) for k, v in
                  (("abs_tol", abs_tol), ("rel_tol", rel_tol)) if v is not None}
        return self._header_result(rule_set_name, rule, op, ext_name, hdu_index,
                                   f"header:{hc['source']}-{hc['other']}", absdiff,
                                   passed, severity, limits, message)

    def _header_result(self, rule_set_name, rule, op, ext_name, hdu_index,
                       region_label, value, passed, severity, limits, message) -> RuleResult:
        return RuleResult(
            rule_set=rule_set_name, rule=rule["name"], extension=ext_name,
            hdu_index=hdu_index, region=region_label, metric=f"header:{op}",
            measured_value=round(float(value), 6), passed=passed, severity=severity,
            verdict_effect="PASS" if passed else severity, limits=limits,
            lookup_path=None, status="EVALUATED", message=message,
        )

    @staticmethod
    def _to_number(raw: Any) -> float | None:
        """Coerce a header value (number or numeric string) to float, else None.

        Empty strings, astropy Undefined cards, and non-numeric values return None
        so that intermittently-populated keywords (e.g. CCDTEMP_1) are skipped, not
        treated as failures.
        """
        try:
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                return None
            return float(raw)
        except (TypeError, ValueError):
            return None
 
    @staticmethod
    def _is_placeholder_data(data: Any) -> bool:
        """Return True for software-generated placeholder images.
 
        Current placeholder convention: the image exists but every pixel is
        exactly 1. Such extensions are reported but skipped from metric
        evaluation and do not affect the final PASS/WARN/FAIL verdict.
        """
        array = np.asarray(data)
        if array.size == 0:
            return False
 
        finite_values = array[np.isfinite(array)]
        if finite_values.size == 0:
            return False
 
        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        return min_value == 1.0 and max_value == 1.0
 
    def _skipped_rule_result(
        self,
        rule_set_name: str,
        rule: dict[str, Any],
        extension: dict[str, Any],
        status: str,
        message: str,
    ) -> RuleResult:
        return RuleResult(
            rule_set=rule_set_name,
            rule=rule["name"],
            extension=extension.get("name"),
            hdu_index=extension.get("hdu_index"),
            region=self._resolve_rule_region_name(rule, extension),
            metric=rule["metric"],
            measured_value=None,
            passed=True,
            severity=rule["severity"],
            verdict_effect=status,
            limits=None,
            lookup_path=None,
            status=status,
            message=message,
        )

    def _error_rule_result(
        self,
        rule_set_name: str,
        rule: dict[str, Any],
        extension: dict[str, Any],
        message: str,
    ) -> RuleResult:
        """A rule that could not be evaluated due to an unexpected error.

        verdict_effect='ERROR' propagates to the overall verdict (never silently
        passes), but only this one rule+extension is affected.
        """
        return RuleResult(
            rule_set=rule_set_name,
            rule=rule["name"],
            extension=extension.get("name") if extension else None,
            hdu_index=extension.get("hdu_index") if extension else None,
            region=rule.get("region", "unknown"),
            metric=rule.get("metric", "unknown"),
            measured_value=None,
            passed=False,
            severity=rule.get("severity", "FAIL"),
            verdict_effect="ERROR",
            limits=None,
            lookup_path=None,
            status="ERROR",
            message=message,
        )

    def _no_rules_matched_result(self, metadata: dict[str, Any]) -> RuleResult:
        """Surface a frame that no rule set matched (mis-routed / bad PRODCATG).

        WARN severity so it is visible in the report and verdict rather than a
        silent PASS with zero checks.
        """
        return RuleResult(
            rule_set="(none)", rule="no_rules_matched", extension=None, hdu_index=None,
            region="n/a", metric="n/a", measured_value=None, passed=False,
            severity="WARN", verdict_effect="WARN", limits=None, lookup_path=None,
            status="NO_RULES_MATCHED",
            message=f"no rule set matched this frame's metadata: {metadata}",
        )

    def _resolve_rule_region_name(
        self,
        rule: dict[str, Any],
        extension: dict[str, Any],
    ) -> str:
        """Resolve the region name for a rule and extension."""
        if "region" in rule:
            return rule["region"]
 
        region_by_extension = rule["region_by_extension"]
        key = region_by_extension["key"]
        values = region_by_extension.get("values", {})
        default_region = region_by_extension.get("default")
 
        extension_value = extension.get(key)
        if extension_value is None:
            if default_region is not None:
                return default_region
            raise QAEngineError(
                f"extension {extension.get('name')!r} has no field {key!r} "
                f"required by rule {rule.get('name')!r}"
            )
 
        normalized_extension_value = self.normalize(extension_value)
 
        for configured_value, region_name in values.items():
            if self.normalize(configured_value) == normalized_extension_value:
                return region_name
 
        if default_region is not None:
            return default_region
 
        raise QAEngineError(
            f"rule {rule.get('name')!r} has no region mapping for "
            f"extension {extension.get('name')!r} field {key!r}={extension_value!r}"
        )
 
    def _evaluate_rule_for_extension(
        self,
        hdul: Any,
        metadata: dict[str, Any],
        rule_set_name: str,
        rule: dict[str, Any],
        extension: dict[str, Any],
    ) -> RuleResult:
        hdu_index = extension["hdu_index"]
        extension_name = extension.get("name")
 
        if hdu_index >= len(hdul):
            return self._skipped_rule_result(
                rule_set_name,
                rule,
                extension,
                status="MISSING",
                message=f"extension {extension_name!r} references missing HDU index {hdu_index}",
            )
 
        data = getattr(hdul[hdu_index], "data", None)
        if data is None:
            return self._skipped_rule_result(
                rule_set_name,
                rule,
                extension,
                status="MISSING",
                message=f"extension {extension_name!r} / HDU {hdu_index} has no image data",
            )
 
        if self._is_placeholder_data(data):
            return self._skipped_rule_result(
                rule_set_name,
                rule,
                extension,
                status="PLACEHOLDER",
                message=(
                    f"extension {extension_name!r} / HDU {hdu_index} "
                    "appears to be a placeholder image"
                ),
            )
 
        region_name = self._resolve_rule_region_name(rule, extension)
        metric_name = rule["metric"]
        severity = rule["severity"]
        # Isolate per-rule failures: a lookup miss (e.g. an off-mode frame whose
        # readout mode is unmodelled) SKIPs just this rule+extension; any other
        # evaluation error becomes an ERROR result for this rule+extension only —
        # neither aborts QA for the rest of the 24 detectors.
        try:
            region = self.config["regions"][region_name]
            metric = self.config["metrics"][metric_name]
            region_data = self._extract_region(np.asarray(data), region, region_name, extension_name)
            if metric.get("type") == "background_gradient_rate":
                # rate metrics need the exposure time; absent/too-short -> SKIP this rule
                exptime = self._resolve_exptime(hdul, metric)
                measured_value = self._compute_metric(region_data, metric, exptime=exptime)
            else:
                measured_value = self._compute_metric(region_data, metric)
            limits, lookup_path = self._resolve_rule_limits(rule, metadata, extension)
        except LookupMissError as exc:
            return self._skipped_rule_result(rule_set_name, rule, extension,
                                             status="SKIPPED", message=str(exc))
        except QAEngineError as exc:
            return self._error_rule_result(rule_set_name, rule, extension, str(exc))
        passed, message = self._check_limits(measured_value, limits)
        verdict_effect = "PASS" if passed else severity
 
        return RuleResult(
            rule_set=rule_set_name,
            rule=rule["name"],
            extension=extension_name,
            hdu_index=hdu_index,
            region=region_name,
            metric=metric_name,
            measured_value=round(float(measured_value), 6),
            passed=passed,
            severity=severity,
            verdict_effect=verdict_effect,
            limits=limits,
            lookup_path=lookup_path,
            status="EVALUATED",
            message=message,
        )
 
    @staticmethod
    def _extract_region(
        data: Any,
        region: dict[str, Any],
        region_name: str,
        extension_name: str | None,
    ) -> Any:
        if data.ndim < 2:
            raise QAEngineError(f"extension {extension_name!r} does not contain a 2D image")
 
        if region["type"] == "full":
            return data
 
        if region["type"] != "rectangle":
            raise QAEngineError(f"unsupported region type {region['type']!r} "
                                f"in region {region_name!r}")
 
        x_start = region["x_start"]
        x_end = region["x_end"]
        y_start = region["y_start"]
        y_end = region["y_end"]
 
        height, width = data.shape[-2], data.shape[-1]
        if not (0 <= x_start < x_end <= width and 0 <= y_start < y_end <= height):
            raise QAEngineError(
                f"region {region_name!r} is outside image bounds for extension {extension_name!r}: "
                f"region x=[{x_start}:{x_end}], y=[{y_start}:{y_end}], "
                f"image width={width}, height={height}"
            )
        return data[..., y_start:y_end, x_start:x_end]
 
    def _resolve_exptime(self, hdul: Any, metric: dict[str, Any]) -> float:
        """Exposure time (s) for a rate metric: first populated positive value among
        ``metric['exptime_keys']`` (default SEXPTIME -> REXPTIME -> EXPTIME; SEXPTIME
        is the actual shutter-open time, the correct integration for dark current).
        Absent or below ``min_exptime`` raises LookupMissError -> the rule is SKIPPED
        (a rate is undefined/noise-dominated for a ~0 s frame), never failed."""
        keys = metric.get("exptime_keys", ["SEXPTIME", "REXPTIME", "EXPTIME"])
        min_exptime = float(metric.get("min_exptime", 1.0))
        for key in keys:
            value = self._to_number(self._find_header_value(hdul, key))
            if value is not None and value >= min_exptime:
                return float(value)
        raise LookupMissError(f"no usable exposure time ({', '.join(keys)}) "
                              f">= {min_exptime}s for gradient-rate metric")

    @staticmethod
    def _compute_metric(region_data: Any, metric: dict[str, Any],
                        exptime: float | None = None) -> float:
        values = np.asarray(region_data, dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise QAEngineError("metric cannot be computed because the selected "
                                "region contains no finite pixels")
 
        metric_type = metric["type"]

        if metric_type == "background_gradient_rate":
            # Camera-warming signal. A warming detector grows a dark-current glow --
            # a smooth large-scale background gradient -- at a rate (ADU/s) far above
            # the slow sky/scattered-light gradient of a normal frame. Measure the
            # gradient with quarter-block MEDIANS (robust to the sparse bright fiber/
            # target flux, so it tracks the background, not the astrophysical signal)
            # and divide by exposure time to isolate the dark-current RATE -- which
            # separates a warming detector (~2 ADU/s) from bright/long science (whose
            # background gradient accrues at <=~0.06 ADU/s). See QA_TESTS_SUMMARY.md.
            if values.ndim < 2:
                raise QAEngineError("background_gradient_rate requires a 2D region")
            if not exptime or exptime <= 0:
                raise QAEngineError("background_gradient_rate requires a positive exposure time")
            ny, nx = values.shape[-2], values.shape[-1]
            qy, qx = max(ny // 4, 1), max(nx // 4, 1)
            v_grad = abs(np.nanmedian(values[..., :qy, :]) - np.nanmedian(values[..., -qy:, :]))
            h_grad = abs(np.nanmedian(values[..., :, :qx]) - np.nanmedian(values[..., :, -qx:]))
            return float(max(v_grad, h_grad) / exptime)

        if metric_type == "mean":
            return float(np.mean(finite_values))
        
        if metric_type == "median":
            return float(np.median(finite_values))
        
        if metric_type == "std":
            return float(np.std(finite_values))

        if metric_type == "min":
            return float(np.min(finite_values))
        
        if metric_type == "max":
            return float(np.max(finite_values))
        
        if metric_type == "percentile":
            return float(np.percentile(finite_values, metric["percentile"]))
        
        if metric_type == "fraction_above":
            return float(np.count_nonzero(finite_values > metric["threshold"]) / finite_values.size)
        
        if metric_type == "sum":
            return float(np.sum(finite_values))
        
        if metric_type == "count_above":
            return float(np.count_nonzero(finite_values > metric["threshold"]))
        
        if metric_type in ("row_structure", "column_structure"):
            axis = -1 if metric_type == "row_structure" else -2  # collapse cols / rows
            profile = np.nanmean(values, axis=axis)
            profile = profile[np.isfinite(profile)]
            if profile.size == 0:
                raise QAEngineError("structure metric has no finite rows/columns")
            return float(np.nanstd(profile))
        
        raise QAEngineError(f"unsupported metric type {metric_type!r}")
 
    def _resolve_rule_limits(
        self,
        rule: dict[str, Any],
        metadata: dict[str, Any],
        extension: dict[str, Any],
    ) -> tuple[dict[str, float], list[str] | None]:
        if "limits" in rule:
            return {key: float(value) for key, value in rule["limits"].items()}, None
 
        lookup = rule["expected_from_lookup"]
        table = self.config["lookup_tables"][lookup["table"]]
        path_parts: list[str] = []
 
        for key_spec in lookup["keys"]:
            source = key_spec["from"]
            scope, field = source.split(".", 1)
            if scope == "extension":
                value = extension.get(field)
            elif scope == "metadata":
                value = metadata.get(field)
            else:  # Should be impossible after validation.
                raise QAEngineError(f"unsupported lookup source {source!r}")
            if value is None:
                raise QAEngineError(f"lookup source {source!r} is missing for rule {rule['name']!r}")
            path_parts.append(self.normalize(value))
 
        leaf = self._resolve_lookup_leaf(table, path_parts, lookup["table"])
        limits: dict[str, float] = {}
        if "min_field" in lookup:
            limits["min"] = float(leaf[lookup["min_field"]])
        if "max_field" in lookup:
            limits["max"] = float(leaf[lookup["max_field"]])
        return limits, path_parts
 
    def _resolve_lookup_leaf(self, table: Any, path_parts: list[str], table_name: str) -> dict[str, Any]:
        node = table
        resolved_path: list[str] = []
        for part in path_parts:
            if not isinstance(node, dict):
                raise QAEngineError(f"lookup table {table_name!r} path {resolved_path!r} "
                                    f"is not a mapping")
            matched_key = None
            for candidate in node:
                if self.normalize(candidate) == self.normalize(part):
                    matched_key = candidate
                    break
            if matched_key is None:
                raise LookupMissError(f"lookup table {table_name!r} has no entry for path {path_parts!r}")
            resolved_path.append(str(matched_key))
            node = node[matched_key]
        if not isinstance(node, dict):
            raise QAEngineError(f"lookup table {table_name!r} leaf at {resolved_path!r} "
                                f"is not a mapping")
        return node
 
    @staticmethod
    def _check_limits(value: float, limits: dict[str, float]) -> tuple[bool, str]:
        if "min" in limits and value < limits["min"]:
            return False, f"value {value:.6g} is below min {limits['min']:.6g}"
        if "max" in limits and value > limits["max"]:
            return False, f"value {value:.6g} is above max {limits['max']:.6g}"
        return True, "value is within configured limits"
 
    def _final_verdict(self, results: list[RuleResult]) -> str:
        policy = self.config["verdict_policy"]
        # An unevaluable rule (ERROR) must never be masked as PASS; it outranks FAIL.
        if any(result.verdict_effect == "ERROR" for result in results):
            return "ERROR"
        if policy.get("fail_if_any_fail", True) and any(
            result.verdict_effect == "FAIL" for result in results
        ):
            return "FAIL"
        if policy.get("warn_if_any_warn", True) and any(
            result.verdict_effect == "WARN" for result in results
        ):
            return "WARN"
        return "PASS"
 
    @staticmethod
    def _summary(results: list[RuleResult]) -> dict[str, int]:
        return {
            "total_checks": len(results),
            "evaluated_checks": sum(1 for result in results if result.status == "EVALUATED"),
            "skipped_checks": sum(1 for result in results if result.status != "EVALUATED"),
            "passed_checks": sum(1 for result in results
                                 if result.status == "EVALUATED" and result.passed),
            "failed_checks": sum(1 for result in results
                                 if result.status == "EVALUATED" and not result.passed),
            "fail_effects": sum(1 for result in results if result.verdict_effect == "FAIL"),
            "warn_effects": sum(1 for result in results if result.verdict_effect == "WARN"),
            "missing_extensions": len({
                (result.extension, result.hdu_index)
                for result in results
                if result.status == "MISSING"
            }),
            "placeholder_extensions": len({
                (result.extension, result.hdu_index)
                for result in results
                if result.status == "PLACEHOLDER"
            }),
        }
 
 
def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as fnf_exc:
        raise QAEngineError(f"configuration file not found: {path}") from fnf_exc
    except yaml.YAMLError as yaml_exc:
        raise QAEngineError(f"YAML syntax error in {path}: {yaml_exc}") from yaml_exc
 
    if not isinstance(data, dict):
        raise QAEngineError(f"configuration root must be a mapping: {path}")
    return data
 
 
def validate_config(config: dict[str, Any], config_path: Path) -> None:
    if QAConfigValidator is None:
        raise QAEngineError(
            "qa_config_validator.py could not be imported; place it next to "
            "qa_engine.py or install it on PYTHONPATH."
        )
    validator = QAConfigValidator(config, filename=str(config_path))
    errors = validator.validate()
    if errors:
        details = "\n".join(f"  ERROR: {error}" for error in errors)
        raise QAEngineError(f"invalid QA configuration: {config_path}\n{details}")
 
 
def write_report(report: dict[str, Any], output_path: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=False)
    if output_path is None:
        print(text)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
 
 
def is_fits_file(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(suffix) for suffix in FITS_SUFFIXES)
 
 
def collect_fits_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise QAEngineError(f"not a directory: {directory}")
    files = [path for path in directory.iterdir() if is_fits_file(path)]
    return sorted(files)
 
 
def run_single_file(fits_file: Path, config_path: Path, no_validate: bool = False) -> dict[str, Any]:
    config = load_yaml(config_path)
    if not no_validate:
        validate_config(config, config_path)
    engine = QAEngine(config)
    return engine.run(fits_file)
 
 
def _batch_worker(args: tuple[str, str, bool]) -> dict[str, Any]:
    fits_file_s, config_path_s, no_validate = args
    fits_file = Path(fits_file_s)
    config_path = Path(config_path_s)
    try:
        report = run_single_file(fits_file, config_path, no_validate=no_validate)
        return {
            "fits_file": str(fits_file),
            "status": "OK",
            "overall_verdict": report["overall_verdict"],
            "report": report,
        }
    except Exception as exc:  # The parent process should receive all per-file failures.
        return {
            "fits_file": str(fits_file),
            "status": "ERROR",
            "overall_verdict": "ERROR",
            "error": str(exc),
        }
 
 
def run_batch(input_dir: Path, config_path: Path, jobs: int, no_validate: bool = False) -> dict[str, Any]:
    files = collect_fits_files(input_dir)
    if not files:
        raise QAEngineError(f"no FITS files found in directory: {input_dir}")
 
    # Validate once in the parent process for fast feedback before starting workers.
    if not no_validate:
        config = load_yaml(config_path)
        validate_config(config, config_path)
 
    worker_args = [(str(path), str(config_path), no_validate) for path in files]
    max_workers = max(1, int(jobs))
 
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_batch_worker, item): item[0]
            for item in worker_args
        }
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive fallback.
                results.append({
                    "fits_file": path,
                    "status": "ERROR",
                    "overall_verdict": "ERROR",
                    "error": str(exc),
                })
 
    results.sort(key=lambda item: item["fits_file"])
 
    ok_count = sum(1 for item in results if item["status"] == "OK")
    # An ERROR overall_verdict (unevaluable rule) counts as an error even when the
    # worker itself did not throw (status == "OK").
    error_count = sum(1 for item in results
                      if item["status"] == "ERROR" or item["overall_verdict"] == "ERROR")
    fail_count = sum(1 for item in results if item["overall_verdict"] == "FAIL")
    warn_count = sum(1 for item in results if item["overall_verdict"] == "WARN")
    pass_count = sum(1 for item in results if item["overall_verdict"] == "PASS")
 
    if error_count:
        overall_status = "ERROR"
    elif fail_count:
        overall_status = "FAIL"
    elif warn_count:
        overall_status = "WARN"
    else:
        overall_status = "PASS"
 
    return {
        "mode": "batch",
        "input_directory": str(input_dir),
        "config_file": str(config_path),
        "jobs": max_workers,
        "overall_status": overall_status,
        "summary": {
            "total_files": len(results),
            "ok_files": ok_count,
            "error_files": error_count,
            "pass_files": pass_count,
            "warn_files": warn_count,
            "fail_files": fail_count,
        },
        "files": results,
    }
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description="Run YAML-driven FITS/MEF QA checks.")
    parser.add_argument("input_path", type=Path, help="Input FITS/MEF file or directory")
    parser.add_argument("--config", type=Path, required=True,
                        help="QA YAML configuration file")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Number of parallel workers in directory/batch mode")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip config validation before running")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print quick human-readable summary")
    args = parser.parse_args()
 
    try:
        input_path = args.input_path
 
        # --------------------------------------------------
        # SINGLE FILE MODE
        # --------------------------------------------------
        if input_path.is_file():
            try:
                report = run_single_file(
                    input_path,
                    args.config,
                    no_validate=args.no_validate
                )
                verdict = report["overall_verdict"]
 
                out_path = input_path.with_suffix(".qa.json")
                write_report(report, out_path)
            except Exception as exc:
                verdict = "ERROR"
                out_path = input_path.with_suffix(".qa.json")
 
                error_report = {
                    "fits_file": str(input_path),
                    "overall_verdict": "ERROR",
                    "error": str(exc),
                }
                write_report(error_report, out_path)
 
            if args.summary_only:
                print(f"{input_path.name} {verdict}")

            # PASS/WARN -> 0; FAIL -> 1; ERROR (unreadable file, bad config,
            # off-mode frame, unevaluable rule) -> 2, so a run that never actually
            # executed does not silently pass the CI gate.
            return {"PASS": 0, "WARN": 0, "FAIL": 1}.get(verdict, 2)
 
        # --------------------------------------------------
        # DIRECTORY / BATCH MODE
        # --------------------------------------------------
        if input_path.is_dir():
            report = run_batch(
                input_path,
                args.config,
                jobs=args.jobs,
                no_validate=args.no_validate
            )
 
            any_fail = False
 
            for item in report["files"]:
                fits_path = Path(item["fits_file"])
                out_path = fits_path.with_suffix(".qa.json")
 
                if item["status"] == "OK":
                    write_report(item["report"], out_path)
                    verdict = item["overall_verdict"]
                else:
                    verdict = "ERROR"
                    error_report = {
                        "fits_file": item["fits_file"],
                        "overall_verdict": "ERROR",
                        "error": item.get("error", "unknown error"),
                    }
                    write_report(error_report, out_path)
 
                if verdict == "FAIL":
                    any_fail = True
 
                if args.summary_only:
                    print(f"{fits_path.name} {verdict}")

            # Propagate ERROR (any file errored) as exit 2, FAIL as 1, else 0 —
            # so errored/off-mode files cannot slip through the gate as green.
            status = report.get("overall_status", "PASS")
            return {"PASS": 0, "WARN": 0, "FAIL": 1}.get(status, 2)
 
        # --------------------------------------------------
        # INVALID INPUT
        # --------------------------------------------------
        raise QAEngineError(f"input path does not exist: {input_path}")
 
    except QAEngineError as qa_exc:
        print(f"ERROR: {qa_exc}", file=sys.stderr)
        return 2
 
 
if __name__ == "__main__":
    raise SystemExit(main())