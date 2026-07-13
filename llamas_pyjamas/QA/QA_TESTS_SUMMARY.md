# LLAMAS QA Test Suite — Summary

YAML-driven quality-assurance checks for LLAMAS multi-extension (MEF) frames, run by
`qa_engine.py` and validated by `qa_config_validator.py`.

| Config | Applies to | Rule sets |
|---|---|---|
| `qa_config_cal.yaml` | calibration frames (auto-selected by `PRODCATG`) | BIAS, DARK, LDLS_FLAT, SKY_FLAT, ARC_THAR |
| `qa_config_science.yaml` | science frames (`SCI.R-*`) | SCIENCE |

## How to run

```bash
cd llamas_pyjamas/QA
python qa_engine.py /path/to/night_dir --config qa_config_cal.yaml     # batch, parallel
python qa_engine.py LLAMAS_..._SCI22_mef.fits --config qa_config_science.yaml
```
Each file gets a `<name>.qa.json` report. **Exit codes:** PASS/WARN → 0, FAIL → 1,
**ERROR → 2** (unreadable file, bad config, off-mode frame, or an unevaluable rule — a run
that never actually executed does *not* pass the gate). Rule sets are auto-selected from
`PRODCATG`.

## Frame types and selectors

| Type | PRODCATG | REXPTIME | Readout modes |
|---|---|---|---|
| Bias | `CAL.R-BIA` | 0.001 s | FAST, SLOW |
| Dark | `CAL.R-DRK` | 600 s | SLOW |
| Arc (ThAr) | `CAL.R-ARC` | 0.05–1 s | FAST, SLOW |
| LDLS flat | `CAL.R-FLT` | 0.07–0.5 s | FAST |
| Sky flat | `CAL.R-SKY` | 1–60 s | SLOW, FAST |
| Science | `SCI.R-*` | varies | SLOW, FAST |

Selection keys on `PRODCATG` (always present); `OBJECT` is unreliable and unused. ARC QA is
**not** gated on the ThAr lamp state, so an arc still gets shutter/saturation/temperature QA
even if the lamp keyword is missing.

## Geometry & regions

Detectors 2048×2048; 24 image extensions (HDU 1–24) red→green→blue per bench-side
1A,1B,2A,2B,3A,3B,4A,4B. Illuminated fibre stack ≈ y[32:2004]; the **unilluminated stripes**
are the per-detector bias/background reference on any frame type.

| Region | Definition |
|---|---|
| `full_frame` | whole detector |
| `bottom_stripe` | y[2:28], x[100:1948] |
| `top_stripe` | y[2020:2046], x[100:1948] |

Missing cameras are all-ones placeholder extensions and are **skipped** (never failed).
Off-mode frames (a readout mode not in a lookup table) **skip just that rule**, never crash
the file.

## Tests per rule set

Severity: **FAIL** blocks the verdict; **WARN** flags but passes; missing header keys or
off-mode lookups → SKIPPED. Background/level/RMS are **WARN** (drift & warm monitoring) — the
doc's requirement is to *track* bias level and its drift; structure is **FAIL** on uniform
frames where banding is an unambiguous defect. All background/structure/saturation limits are
per-detector (`extension.name`) × `readout_mode`.

### BIAS (`CAL.R-BIA`)
| Rule | Region / source | Metric | Threshold | Sev | Flags |
|---|---|---|---|---|---|
| shutter_exptime_consistency | SEXPTIME vs REXPTIME | abs_or_rel_diff | abs≈0.33 s / rel 10% | FAIL | shutter stuck/aborted |
| edge_background_level | bottom_stripe | median | per-det band | WARN | bias level drift / warm |
| frame_level_median | full_frame | median | per-det band | WARN | level drift / warm |
| frame_noise_rms | full_frame | std | per-det cap | WARN | excess noise |
| row_structure / column_structure | full_frame | row/col banding | per-det cap | FAIL | banding / odd structure |
| saturation_fraction | full_frame | fraction>63000 | per-det cap (~0) | FAIL | saturation |
| ccd_temperature_warm | CCDTEMP_1 (per ext) | range | > −73.8 °C | WARN | warming camera |
| ccd_temperature_shutoff | CCDTEMP_1 (per ext) | range | > −60 °C | FAIL | shut-off / hot camera |

### DARK (`CAL.R-DRK`)
Same as BIAS (SLOW, 600 s) except saturation is **WARN** (hot pixels tolerated). Structure is
**FAIL** — this is where the June-2026 odd-structure dark trips (`column_structure` 2.B.Green
13.1 ≫ cap 2.0; `row_structure` 2.B.Green 6.5 > 2.98; `column_structure` 1.B.Green 4.1 > 2).

### LDLS_FLAT / SKY_FLAT / ARC_THAR (`CAL.R-FLT` / `-SKY` / `-ARC`)
shutter (FAIL), edge-stripe background (WARN), row/column structure (**WARN** — illuminated
frames carry real structure), per-detector saturation (WARN), temperature warm/shut-off. **No
full-frame level check** (illumination scales with exposure time, so a fixed level band is not
physical).

### SCIENCE (`SCI.R-*`)
| Rule | Source | Threshold | Sev |
|---|---|---|---|
| shutter_exptime_consistency | SEXPTIME vs REXPTIME | abs 1 s / rel 10% | FAIL |
| saturation_fraction | full_frame fraction>63000 | ≤0.02 | WARN |
| ccd_temperature_warm / _shutoff | CCDTEMP_1 per ext | > −73.8 / > −60 °C | WARN / FAIL |

Image background/structure checks are intentionally **omitted** on science: science frames
carry real dispersed-spectrum structure and sky/scattered light, and there are no science
baselines to calibrate per-detector bands, so such checks would over-flag.

## The two must-flag cases

1. **Warm / accidentally-shut-off cameras.** Signals: (a) per-detector `CCDTEMP_1` range
   (WARN > −73.8 °C ≈ median+5σ of the cold distribution; FAIL > −60 °C) — but this keyword is
   only populated ~19% of the time, so it is supplementary; (b) elevated unilluminated-stripe
   / full-frame background from dark current (WARN). **Reality check:** in the May-2026 example
   frames the cameras still read ~−91 °C and show no elevated background, so for *that*
   incident the reliable flag is the **shutter fault** (600 s requested → 181 s actual → FAIL).
   The temperature/background checks are calibrated for cases where warming is more advanced.
2. **Odd detector structure.** Per-detector `row_structure`/`column_structure` FAIL caps on
   uniform (bias/dark) frames. Being per-detector, persistently-structured detectors (e.g.
   blue 3.A row-banding ≈1424, normal) keep their own band and don't false-fail, while a
   genuine anomaly (green 2.B col ≈13 vs cap 2) trips. Calibrated against the June-2026 odd
   dark.

## Header-value checks (engine extension)

`qa_engine.py` / `qa_config_validator.py` gained a `header_check` rule type:
```yaml
header_check:
  op: range | abs_diff | rel_diff | abs_or_rel_diff
  source: SEXPTIME          # single keyword (range) or first operand (diff ops)
  other: REXPTIME           # second operand for diff ops
  per_extension_key: CCDTEMP_1   # read from THIS extension's header (requires per_extension: true)
  limits: {min: -140, max: -60}  # for range
  abs_tol: 1.0                   # for *_diff ops
  rel_tol: 0.10
```
`abs_or_rel_diff` passes if within **either** tolerance: `abs_tol` absorbs the fixed
sub-second shutter overhead on short exposures; the tight 10% `rel_tol` catches gross
proportional faults on long exposures. Absent keywords → SKIPPED, never failed.

## Threshold provenance & tracking

Derived from **153 baseline cal files** over **2026-04-07, 2026-05-02, 2026-06-30** (the
known-bad June-4 odd dark held out for validation). Methodology (robust):
- Per (type × mode × detector) samples are **MAD sigma-clipped** to reject anomalous baseline
  frames *before* deriving limits (so an outlier can't inflate its own cap).
- Background/level bands are additive: `median ± max(6σ_robust, 15 ADU)` — an **absolute** ADU
  floor, not a percentage.
- Structure/RMS/saturation caps are robust one-sided: `max(median + 8σ_robust, 3×median,
  floor)`.
- Self-check: 2.7% of normal detector-frames breach their edge-background band (WARN-level
  drift), 0 breach at FAIL severity.

Artifacts (`baselines/`): `qa_thresholds_derived.json` (limits + observed stats),
`qa_tracking_baselines.csv` (per-epoch medians for **drift monitoring**), and
`scripts/` (`sort_baselines`, `extract_stats`, `aggregate_thresholds`, `gen_configs`) — the
reproducible pipeline. Re-run to refresh thresholds as more nights arrive; the YAMLs are
generated, so edit `gen_configs.py` (or the YAML) and re-validate with `qa_config_validator.py`.

## Baseline organisation

`QA_baselines/copies/` (untouched originals) was copy-sorted into sibling type folders
`Bias/ Darks/ Arcs/ lamp_flats/ twilight_flats/`, each with a `manifest.csv`.
