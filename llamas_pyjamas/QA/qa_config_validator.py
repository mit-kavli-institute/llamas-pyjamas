#!/usr/bin/env python3
"""Standalone validator for the FITS/MEF QA YAML configuration."""
 
from __future__ import annotations
 
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
 
import yaml
 
 
@dataclass
class SourceLocation:
    line: int | None = None
    column: int | None = None
    value: Any | None = None
 
 
@dataclass
class ValidationErrorItem:
    path: str
    message: str
    filename: str | None = None
    line: int | None = None
    column: int | None = None
    value: Any | None = None
 
    def __str__(self) -> str:
        location = ""
        if self.filename and self.line is not None:
            location = f"{self.filename}:{self.line}"
            if self.column is not None:
                location += f":{self.column}"
            location += ": "
        elif self.filename:
            location = f"{self.filename}: "
        value = ""
        if self.value is not None:
            value = f" (value: {self.value!r})"
        return f"{location}{self.path}: {self.message}{value}"
 
 
class QAConfigValidator:
    REQUIRED_TOP_LEVEL = {
        "config_version",
        "instrument",
        "metadata_keys",
        "extensions",
        "regions",
        "metrics",
        "lookup_tables",
        "rule_sets",
        "verdict_policy",
    }
    OPTIONAL_TOP_LEVEL: set[str] = set()
    REGION_TYPES = {"full", "rectangle"}
    METRIC_TYPES = {
        "mean",
        "median",
        "std",
        "min",
        "max",
        "percentile",
        "fraction_above",
        "sum",
        "count_above",
        "row_structure",
        "column_structure",
    }
    SEVERITIES = {"PASS", "WARN", "FAIL"}
    HEADER_OPS = {"range", "abs_diff", "rel_diff", "abs_or_rel_diff"}
 
    def __init__(
        self,
        config: dict[str, Any],
        source_map: dict[str, SourceLocation] | None = None,
        filename: str | None = None,
    ):
        self.config = config
        self.source_map = source_map or {}
        self.filename = filename
        self.errors: list[ValidationErrorItem] = []
 
    @staticmethod
    def normalize(value: Any) -> str:
        return str(value).strip().lower()
 
    def add_error(self, path: str, message: str) -> None:
        location = self.source_map.get(path) or self.source_map.get(self.parent_path(path))
        self.errors.append(
            ValidationErrorItem(
                path=path,
                message=message,
                filename=self.filename,
                line=location.line if location else None,
                column=location.column if location else None,
                value=location.value if location else None,
            )
        )
 
    @staticmethod
    def parent_path(path: str) -> str:
        if path == "$" or not path:
            return "$"
        if path.endswith("]") and "[" in path:
            return path.rsplit("[", 1)[0]
        if "." in path:
            return path.rsplit(".", 1)[0]
        return "$"
 
    def validate(self) -> list[ValidationErrorItem]:
        self.validate_top_level()
        self.validate_instrument()
        self.validate_metadata_keys()
        self.validate_extensions()
        self.validate_regions()
        self.validate_metrics()
        self.validate_lookup_tables()
        self.validate_rule_sets()
        self.validate_verdict_policy()
        return self.errors
 
    def validate_top_level(self) -> None:
        if not isinstance(self.config, dict):
            self.add_error("$", "configuration root must be a mapping")
            return
        for key in sorted(self.REQUIRED_TOP_LEVEL):
            if key not in self.config:
                self.add_error(key, "required top-level block is missing")
        allowed = self.REQUIRED_TOP_LEVEL | self.OPTIONAL_TOP_LEVEL
        for key in self.config:
            if key not in allowed:
                self.add_error(key, "unknown top-level block")
 
    def validate_instrument(self) -> None:
        instrument = self.config.get("instrument")
        if instrument is None:
            return
        if not isinstance(instrument, dict):
            self.add_error("instrument", "must be a mapping")
            return
        if not isinstance(instrument.get("name"), str) or not instrument.get("name", "").strip():
            self.add_error("instrument.name", "is required and must be a non-empty string")
        # description is optional, but if present it must be a string
        if "description" in instrument and not isinstance(instrument["description"], str):
            self.add_error("instrument.description", "must be a string")
 
    def validate_metadata_keys(self) -> None:
        metadata_keys = self.config.get("metadata_keys")
        if metadata_keys is None:
            return
        if not isinstance(metadata_keys, dict) or not metadata_keys:
            self.add_error("metadata_keys", "must be a non-empty mapping")
            return
        for key, value in metadata_keys.items():
            if not isinstance(key, str) or not key.strip():
                self.add_error("metadata_keys", "all internal metadata names must be non-empty strings")
            if not isinstance(value, str) or not value.strip():
                self.add_error(f"metadata_keys.{key}", "FITS header keyword must be a non-empty string")
 
    def validate_extensions(self) -> None:
        extensions = self.config.get("extensions")
        if extensions is None:
            return
        if not isinstance(extensions, list) or not extensions:
            self.add_error("extensions", "must be a non-empty list")
            return
        seen_names: set[str] = set()
        seen_hdu: set[int] = set()
        for idx, ext in enumerate(extensions):
            path = f"extensions[{idx}]"
            if not isinstance(ext, dict):
                self.add_error(path, "must be a mapping")
                continue
            name = ext.get("name")
            if not isinstance(name, str) or not name.strip():
                self.add_error(f"{path}.name", "is required and must be a non-empty string")
            elif name in seen_names:
                self.add_error(f"{path}.name", f"duplicate extension name '{name}'")
            else:
                seen_names.add(name)
            hdu_index = ext.get("hdu_index")
            if not isinstance(hdu_index, int) or hdu_index < 0:
                self.add_error(f"{path}.hdu_index", "is required and must be a non-negative integer")
            elif hdu_index in seen_hdu:
                self.add_error(f"{path}.hdu_index", f"duplicate HDU index {hdu_index}")
            else:
                seen_hdu.add(hdu_index)
            for field, value in ext.items():
                if value is None:
                    self.add_error(f"{path}.{field}", "must not be null")
 
    def validate_regions(self) -> None:
        regions = self.config.get("regions")
        if regions is None:
            return
        if not isinstance(regions, dict) or not regions:
            self.add_error("regions", "must be a non-empty mapping")
            return
        for name, region in regions.items():
            path = f"regions.{name}"
            if not isinstance(region, dict):
                self.add_error(path, "must be a mapping")
                continue
            rtype = region.get("type")
            if rtype not in self.REGION_TYPES:
                self.add_error(f"{path}.type", f"must be one of {sorted(self.REGION_TYPES)}")
                continue
            allowed = {"type"} if rtype == "full" else {"type", "x_start", "x_end", "y_start", "y_end"}
            for field in region:
                if field not in allowed:
                    self.add_error(f"{path}.{field}", "unknown field for this region type")
            if rtype == "rectangle":
                for coord in ["x_start", "x_end", "y_start", "y_end"]:
                    value = region.get(coord)
                    if not isinstance(value, int) or value < 0:
                        self.add_error(f"{path}.{coord}", "is required and must be a non-negative integer")
 
    def validate_metrics(self) -> None:
        metrics = self.config.get("metrics")
        if metrics is None:
            return
        if not isinstance(metrics, dict) or not metrics:
            self.add_error("metrics", "must be a non-empty mapping")
            return
        for name, metric in metrics.items():
            path = f"metrics.{name}"
            if not isinstance(metric, dict):
                self.add_error(path, "must be a mapping")
                continue
            mtype = metric.get("type")
            if mtype not in self.METRIC_TYPES:
                self.add_error(f"{path}.type", f"must be one of {sorted(self.METRIC_TYPES)}")
                continue
            allowed = {"type"}
            if mtype == "percentile":
                allowed.add("percentile")
                if not self.is_number(metric.get("percentile")):
                    self.add_error(f"{path}.percentile", "is required and must be numeric")
            if mtype in {"fraction_above", "count_above"}:
                allowed.add("threshold")
                if not self.is_number(metric.get("threshold")):
                    self.add_error(f"{path}.threshold", "is required and must be numeric")
            for field, value in metric.items():
                if field not in allowed:
                    self.add_error(f"{path}.{field}", "unknown field for this metric type")
                if value is None:
                    self.add_error(f"{path}.{field}", "must not be null")
 
    def validate_lookup_tables(self) -> None:
        lookup_tables = self.config.get("lookup_tables")
        if lookup_tables is None:
            return
        if not isinstance(lookup_tables, dict) or not lookup_tables:
            self.add_error("lookup_tables", "must be a non-empty mapping")
            return
        for table_name, table in lookup_tables.items():
            self.validate_lookup_node(table, f"lookup_tables.{table_name}")
 
    def validate_lookup_node(self, node: Any, path: str) -> str | None:
        """Return 'nested', 'leaf', or None when invalid."""
        if not isinstance(node, dict) or not node:
            self.add_error(path, "must be a non-empty mapping")
            return None
        child_kinds: set[str] = set()
        for key, value in node.items():
            if not isinstance(key, str) or not key.strip():
                self.add_error(path, "all lookup keys must be non-empty strings")
            if isinstance(value, dict):
                kind = self.validate_lookup_node(value, f"{path}.{key}")
                if kind:
                    child_kinds.add("nested")
            else:
                child_kinds.add("leaf_value")
                if not self.is_number(value):
                    self.add_error(f"{path}.{key}", "lookup leaf values must be numeric and non-null")
        if child_kinds == {"leaf_value"}:
            return "leaf"
        if child_kinds == {"nested"}:
            return "nested"
        if "leaf_value" in child_kinds and "nested" in child_kinds:
            self.add_error(path, "mixing nested levels and leaf values at the same level is not allowed")
        return None
 
    def validate_rule_sets(self) -> None:
        rule_sets = self.config.get("rule_sets")
        if rule_sets is None:
            return
        if not isinstance(rule_sets, dict) or not rule_sets:
            self.add_error("rule_sets", "must be a non-empty mapping")
            return
        regions = self.config.get("regions") if isinstance(self.config.get("regions"), dict) else {}
        metrics = self.config.get("metrics") if isinstance(self.config.get("metrics"), dict) else {}
        metadata_keys = self.config.get("metadata_keys") if isinstance(self.config.get("metadata_keys"), dict) else {}
        lookup_tables = self.config.get("lookup_tables") if isinstance(self.config.get("lookup_tables"), dict) else {}
        extensions = self.config.get("extensions") if isinstance(self.config.get("extensions"), list) else []
 
        for set_name, rule_set in rule_sets.items():
            set_path = f"rule_sets.{set_name}"
            if not isinstance(rule_set, dict):
                self.add_error(set_path, "must be a mapping")
                continue
            allowed_set_fields = {"applies_when", "rules"}
            for field in rule_set:
                if field not in allowed_set_fields:
                    self.add_error(f"{set_path}.{field}", "unknown field")
            applies_when = rule_set.get("applies_when")
            if not isinstance(applies_when, dict) or not applies_when:
                self.add_error(f"{set_path}.applies_when", "is required and must be a non-empty mapping")
            else:
                for key, value in applies_when.items():
                    if key not in metadata_keys:
                        self.add_error(f"{set_path}.applies_when.{key}", "must reference an existing metadata_keys entry")
                    if value is None:
                        self.add_error(f"{set_path}.applies_when.{key}", "must not be null")
            rules = rule_set.get("rules")
            if not isinstance(rules, list) or not rules:
                self.add_error(f"{set_path}.rules", "is required and must be a non-empty list")
                continue
            seen_rule_names: set[str] = set()
            for idx, rule in enumerate(rules):
                self.validate_rule(
                    rule,
                    f"{set_path}.rules[{idx}]",
                    seen_rule_names,
                    regions,
                    metrics,
                    metadata_keys,
                    lookup_tables,
                    extensions,
                )
 
    def validate_rule(
        self,
        rule: Any,
        path: str,
        seen_rule_names: set[str],
        regions: dict[str, Any],
        metrics: dict[str, Any],
        metadata_keys: dict[str, Any],
        lookup_tables: dict[str, Any],
        extensions: list[Any],
    ) -> None:
        if not isinstance(rule, dict) or not rule:
            self.add_error(path, "must be a non-empty mapping")
            return
        is_header = "header_check" in rule
        allowed = {"name", "severity", "per_extension"}
        allowed |= (
            {"header_check"} if is_header
            else {"region", "region_by_extension", "metric", "limits", "expected_from_lookup"}
        )
        for field in rule:
            if field not in allowed:
                self.add_error(f"{path}.{field}", "unknown field")
        name = rule.get("name")
        if not isinstance(name, str) or not name.strip():
            self.add_error(f"{path}.name", "is required and must be a non-empty string")
        elif name in seen_rule_names:
            self.add_error(f"{path}.name", f"duplicate rule name '{name}' within rule set")
        else:
            seen_rule_names.add(name)
        severity = rule.get("severity")
        if severity not in self.SEVERITIES:
            self.add_error(f"{path}.severity", f"must be one of {sorted(self.SEVERITIES)}")
        if "per_extension" in rule and not isinstance(rule["per_extension"], bool):
            self.add_error(f"{path}.per_extension", "must be boolean")

        if is_header:
            self.validate_header_check(rule.get("header_check"), f"{path}.header_check",
                                       rule.get("per_extension", False))
            return

        has_region = "region" in rule
        has_region_by_extension = "region_by_extension" in rule
 
        if has_region == has_region_by_extension:
            self.add_error(path, "must define exactly one of 'region' or 'region_by_extension'")
        elif has_region:
            region = rule.get("region")
            if not isinstance(region, str) or region not in regions:
                self.add_error(f"{path}.region", "must reference an existing region")
        else:
            self.validate_region_by_extension(
                rule.get("region_by_extension"),
                f"{path}.region_by_extension",
                regions,
                extensions,
            )
        metric = rule.get("metric")
        if not isinstance(metric, str) or metric not in metrics:
            self.add_error(f"{path}.metric", "must reference an existing metric")
 
        has_limits = "limits" in rule
        has_lookup = "expected_from_lookup" in rule
        if has_limits == has_lookup:
            self.add_error(f"{path}", "must define exactly one of 'limits' or 'expected_from_lookup'")
        if has_limits:
            self.validate_limits(rule.get("limits"), f"{path}.limits")
        if has_lookup:
            self.validate_expected_from_lookup(
                rule.get("expected_from_lookup"),
                f"{path}.expected_from_lookup",
                metadata_keys,
                lookup_tables,
                extensions,
            )
 
    def validate_header_check(self, hc: Any, path: str, per_extension: bool = False) -> None:
        """Validate a ``header_check`` rule block (shutter/temperature checks).

        ``source``/``other``/``per_extension_key`` are raw FITS header keywords
        (e.g. REXPTIME, SEXPTIME, CCDTEMP_1), not metadata_keys aliases.
        """
        if not isinstance(hc, dict) or not hc:
            self.add_error(path, "must be a non-empty mapping")
            return
        allowed = {"op", "source", "other", "per_extension_key", "limits", "abs_tol", "rel_tol"}
        for field in hc:
            if field not in allowed:
                self.add_error(f"{path}.{field}", "unknown field")
        op = hc.get("op")
        if op not in self.HEADER_OPS:
            self.add_error(f"{path}.op", f"must be one of {sorted(self.HEADER_OPS)}")
            return

        # per_extension_key is read from the extension's own header, so the rule
        # must be per_extension: true — otherwise the engine evaluates it once with
        # no extension header and silently SKIPs it (disabling the check).
        if "per_extension_key" in hc and not per_extension:
            self.add_error(path, "header_check with 'per_extension_key' requires the rule "
                                 "to set 'per_extension: true'")

        if op == "range":
            has_source = "source" in hc
            has_pek = "per_extension_key" in hc
            if has_source == has_pek:
                self.add_error(path, "op 'range' must define exactly one of "
                                     "'source' or 'per_extension_key'")
            key = hc.get("source") if has_source else hc.get("per_extension_key")
            if key is not None and (not isinstance(key, str) or not key.strip()):
                key_field = "source" if has_source else "per_extension_key"
                self.add_error(f"{path}.{key_field}", "must be a non-empty header keyword string")
            self.validate_limits(hc.get("limits"), f"{path}.limits")
            for extra in ("other", "abs_tol", "rel_tol"):
                if extra in hc:
                    self.add_error(f"{path}.{extra}", "not allowed for op 'range'")
            return

        # difference ops: abs_diff / rel_diff / abs_or_rel_diff
        for req in ("source", "other"):
            if not isinstance(hc.get(req), str) or not hc.get(req, "").strip():
                self.add_error(f"{path}.{req}", "must be a non-empty header keyword string")
        need_abs = op in ("abs_diff", "abs_or_rel_diff")
        need_rel = op in ("rel_diff", "abs_or_rel_diff")
        if need_abs and not self.is_number(hc.get("abs_tol")):
            self.add_error(f"{path}.abs_tol", "is required and must be numeric")
        if need_rel and not self.is_number(hc.get("rel_tol")):
            self.add_error(f"{path}.rel_tol", "is required and must be numeric")
        if op == "abs_diff" and "rel_tol" in hc:
            self.add_error(f"{path}.rel_tol", "not allowed for op 'abs_diff'")
        if op == "rel_diff" and "abs_tol" in hc:
            self.add_error(f"{path}.abs_tol", "not allowed for op 'rel_diff'")
        for extra in ("per_extension_key", "limits"):
            if extra in hc:
                self.add_error(f"{path}.{extra}", f"not allowed for op {op!r}")

    def validate_region_by_extension(
        self,
        spec: Any,
        path: str,
        regions: dict[str, Any],
        extensions: list[Any],
    ) -> None:
        """Validate extension-dependent region selection for a rule."""
        if not isinstance(spec, dict) or not spec:
            self.add_error(path, "must be a non-empty mapping")
            return
 
        allowed = {"key", "values", "default"}
        for field in spec:
            if field not in allowed:
                self.add_error(f"{path}.{field}", "unknown field")
 
        key = spec.get("key")
        if not isinstance(key, str) or not key.strip():
            self.add_error(f"{path}.key", "is required and must be a non-empty string")
            key = None
 
        values = spec.get("values")
        if not isinstance(values, dict) or not values:
            self.add_error(f"{path}.values", "is required and must be a non-empty mapping")
            values = {}
        else:
            for extension_value, region_name in values.items():
                if not isinstance(extension_value, str) or not extension_value.strip():
                    self.add_error(
                        f"{path}.values",
                        "all extension values must be non-empty strings",
                    )
                if not isinstance(region_name, str) or region_name not in regions:
                    self.add_error(
                        f"{path}.values.{extension_value}",
                        "must reference an existing region",
                    )
 
        default_region = spec.get("default")
        if "default" in spec and (
            not isinstance(default_region, str) or default_region not in regions
        ):
            self.add_error(f"{path}.default", "must reference an existing region")
 
        if key is None:
            return
 
        for ext_idx, ext in enumerate(extensions):
            if not isinstance(ext, dict):
                continue
            if key not in ext:
                self.add_error(
                    f"{path}.key",
                    f"extension field '{key}' is missing in extensions[{ext_idx}]",
                )
                continue
            if ext.get(key) is None:
                self.add_error(
                    f"{path}.key",
                    f"extension field '{key}' must not be null in extensions[{ext_idx}]",
                )
 
        if isinstance(values, dict) and "default" not in spec:
            configured_values = {self.normalize(value) for value in values}
            for ext_idx, ext in enumerate(extensions):
                if not isinstance(ext, dict) or key not in ext or ext.get(key) is None:
                    continue
                ext_value = self.normalize(ext.get(key))
                if ext_value not in configured_values:
                    self.add_error(
                        f"{path}.values",
                        f"missing mapping for extension field '{key}' value {ext.get(key)!r} "
                        f"used by extensions[{ext_idx}]",
                    )
 
    def validate_limits(self, limits: Any, path: str) -> None:
        if not isinstance(limits, dict) or not limits:
            self.add_error(path, "must be a non-empty mapping")
            return
        allowed = {"min", "max"}
        if not any(k in limits for k in allowed):
            self.add_error(path, "must contain at least one of 'min' or 'max'")
        for field, value in limits.items():
            if field not in allowed:
                self.add_error(f"{path}.{field}", "unknown field")
            elif not self.is_number(value):
                self.add_error(f"{path}.{field}", "must be numeric and non-null")
 
    def validate_expected_from_lookup(
        self,
        lookup: Any,
        path: str,
        metadata_keys: dict[str, Any],
        lookup_tables: dict[str, Any],
        extensions: list[Any],
    ) -> None:
        if not isinstance(lookup, dict) or not lookup:
            self.add_error(path, "must be a non-empty mapping")
            return
        allowed = {"table", "keys", "min_field", "max_field"}
        for field in lookup:
            if field not in allowed:
                self.add_error(f"{path}.{field}", "unknown field")
        table_name = lookup.get("table")
        if not isinstance(table_name, str) or table_name not in lookup_tables:
            self.add_error(f"{path}.table", "must reference an existing lookup table")
            table = None
        else:
            table = lookup_tables[table_name]
        keys = lookup.get("keys")
        if not isinstance(keys, list) or not keys:
            self.add_error(f"{path}.keys", "is required and must be a non-empty list")
            keys = []
        key_sources: list[str] = []
        for idx, key_spec in enumerate(keys):
            key_path = f"{path}.keys[{idx}]"
            if not isinstance(key_spec, dict) or set(key_spec) != {"from"}:
                self.add_error(key_path, "must contain exactly one field: 'from'")
                continue
            source = key_spec["from"]
            if not isinstance(source, str):
                self.add_error(f"{key_path}.from", "must be a string")
                continue
            if source.startswith("metadata."):
                field = source.split(".", 1)[1]
                if field not in metadata_keys:
                    self.add_error(f"{key_path}.from", "metadata field does not exist in metadata_keys")
            elif source.startswith("extension."):
                field = source.split(".", 1)[1]
                for ext_idx, ext in enumerate(extensions):
                    if not isinstance(ext, dict) or field not in ext:
                        self.add_error(f"{key_path}.from", f"extension field '{field}' is missing in extensions[{ext_idx}]")
                        break
            else:
                self.add_error(f"{key_path}.from", "must start with 'extension.' or 'metadata.'")
            key_sources.append(source)
        if not ("min_field" in lookup or "max_field" in lookup):
            self.add_error(path, "must define min_field and/or max_field")
        for field in ["min_field", "max_field"]:
            if field in lookup and (not isinstance(lookup[field], str) or not lookup[field].strip()):
                self.add_error(f"{path}.{field}", "must be a non-empty string")
 
        if table is not None and isinstance(keys, list):
            self.validate_lookup_references(table, key_sources, lookup, path, extensions)
 
    def validate_lookup_references(
        self,
        table: Any,
        key_sources: list[str],
        lookup: dict[str, Any],
        path: str,
        extensions: list[Any],
    ) -> None:
        # Validate all lookup paths that can be known from static config.
        # metadata.* values come from FITS headers at runtime, so only their field names are checked here.
        if not key_sources:
            return
        known_paths: list[tuple[list[str], str]] = [([], "")]
        for source in key_sources:
            if source.startswith("extension."):
                field = source.split(".", 1)[1]
                values = sorted({self.normalize(ext.get(field)) for ext in extensions if isinstance(ext, dict) and field in ext})
                known_paths = [(parts + [value], desc + f"/{source}={value}") for parts, desc in known_paths for value in values]
            else:
                # Runtime metadata value is unknown at static validation time.
                known_paths = [(parts, desc + f"/{source}=<runtime>") for parts, desc in known_paths]
 
        fields = [lookup.get(f) for f in ("min_field", "max_field") if isinstance(lookup.get(f), str)]
        for parts, desc in known_paths:
            node = self.resolve_lookup_path_case_insensitive(table, parts)
            if node is None:
                self.add_error(path, f"lookup table is missing static path for {desc.strip('/')}")
                continue
            if not isinstance(node, dict):
                continue
            # Validate the referenced leaf fields. When all dimensions are static the
            # node IS the leaf; when a trailing metadata dimension is runtime, check
            # every leaf mapping under this node so a typo'd min_field/max_field is
            # caught statically instead of surfacing as a runtime error.
            leaves = [node] if len(parts) == len(key_sources) else list(self._iter_leaf_mappings(node))
            for leaf in leaves:
                for field in fields:
                    if field not in leaf:
                        self.add_error(path, f"lookup leaf under {desc.strip('/')} is missing field '{field}'")
                        break

    @staticmethod
    def _iter_leaf_mappings(node: Any):
        """Yield the deepest mapping levels (those with no nested-dict children)."""
        if not isinstance(node, dict):
            return
        if not any(isinstance(v, dict) for v in node.values()):
            yield node
            return
        for value in node.values():
            if isinstance(value, dict):
                yield from QAConfigValidator._iter_leaf_mappings(value)
 
    def resolve_lookup_path_case_insensitive(self, table: Any, parts: list[str]) -> Any | None:
        node = table
        for part in parts:
            if not isinstance(node, dict):
                return None
            matched_key = None
            for key in node:
                if self.normalize(key) == self.normalize(part):
                    matched_key = key
                    break
            if matched_key is None:
                return None
            node = node[matched_key]
        return node
 
    def validate_verdict_policy(self) -> None:
        verdict_policy = self.config.get("verdict_policy")
        if verdict_policy is None:
            return
        if not isinstance(verdict_policy, dict):
            self.add_error("verdict_policy", "must be a mapping")
            return
        allowed = {"fail_if_any_fail", "warn_if_any_warn"}
        for field in allowed:
            if not isinstance(verdict_policy.get(field), bool):
                self.add_error(f"verdict_policy.{field}", "is required and must be boolean")
        for field in verdict_policy:
            if field not in allowed:
                self.add_error(f"verdict_policy.{field}", "unknown field")
 
    @staticmethod
    def is_number(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
 
 
def build_source_map(path: Path) -> dict[str, SourceLocation]:
    """Build a best-effort map from validator paths to YAML line/column/value.
 
    PyYAML's safe_load returns plain Python objects and discards source
    locations, so we compose the YAML once to inspect parser nodes and keep
    line numbers for friendly validation errors.
    """
    source_map: dict[str, SourceLocation] = {}
 
    def scalar_value(node: yaml.Node) -> Any:
        return getattr(node, "value", None)
 
    def add_location(path_key: str, node: yaml.Node, value: Any | None = None) -> None:
        source_map[path_key] = SourceLocation(
            line=node.start_mark.line + 1,
            column=node.start_mark.column + 1,
            value=value,
        )
 
    def walk(node: yaml.Node, current_path: str) -> None:
        if isinstance(node, yaml.MappingNode):
            add_location(current_path, node)
            for key_node, value_node in node.value:
                key = str(key_node.value)
                child_path = key if current_path == "$" else f"{current_path}.{key}"
                # Put the key location on the field path, then let scalar values
                # override it below with the value location and token.
                add_location(child_path, key_node, key)
                walk(value_node, child_path)
        elif isinstance(node, yaml.SequenceNode):
            add_location(current_path, node)
            for index, child in enumerate(node.value):
                child_path = f"{current_path}[{index}]"
                walk(child, child_path)
        elif isinstance(node, yaml.ScalarNode):
            add_location(current_path, node, scalar_value(node))
 
    with path.open("r", encoding="utf-8") as f:
        root = yaml.compose(f)
    if root is not None:
        walk(root, "$")
    return source_map
 
 
def load_yaml(path: Path) -> tuple[dict[str, Any], dict[str, SourceLocation]] | None:
    try:
        source_map = build_source_map(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return None
    except yaml.YAMLError as exc:
        print(f"ERROR: YAML syntax error in {path}: {exc}", file=sys.stderr)
        return None
    if data is None:
        print(f"ERROR: YAML file is empty: {path}", file=sys.stderr)
        return None
    return data, source_map
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a FITS/MEF QA YAML configuration file.")
    parser.add_argument("config", type=Path, help="Path to qa_config.yaml")
    args = parser.parse_args()
 
    loaded = load_yaml(args.config)
    if loaded is None:
        return 2
    config, source_map = loaded
 
    validator = QAConfigValidator(config, source_map=source_map, filename=str(args.config))
    errors = validator.validate()
    if errors:
        print(f"INVALID: {args.config}")
        for error in errors:
            print(f"ERROR: {error}")
        return 1
 
    print(f"VALID: {args.config}")
    return 0
 
 
if __name__ == "__main__":
    raise SystemExit(main())