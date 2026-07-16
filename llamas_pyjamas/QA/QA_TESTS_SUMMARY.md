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

### `QA_assess.py` — user-facing single-file front-end

`llamas_pyjamas/Postprocessing/QA_assess.py` is the interface that links QA results to users
(downstream consumes its exit code and `--report` JSON). It runs the same engine on **one** image
but **auto-selects the config from the frame's `PRODCATG`** — `CAL.*` → `qa_config_cal.yaml`,
`SCI.*` → `qa_config_science.yaml` — so you do not pass `--config`:

```bash
# run the script directly on a file; auto-selects cal vs science by PRODCATG:
python llamas_pyjamas/Postprocessing/QA_assess.py /path/to/LLAMAS_..._mef.fits --report out.json
# explicit override still available:
python llamas_pyjamas/Postprocessing/QA_assess.py /path/to/file.fits --qa-yaml llamas_pyjamas/QA/qa_config_cal.yaml
```
Run it as a **direct script** (`python …/QA_assess.py <file>`), which is the canonical form: the
script prepends its own source-tree root to `sys.path`, so it always imports the QA code that ships
beside it — no `PYTHONPATH` needed even though the editable install's `.pth` points at a different
(sky-framework/flux-cal) checkout that lacks this QA suite. The module form
`python -m llamas_pyjamas.Postprocessing.QA_assess …` works **only** when this checkout is already on
the path (`PYTHONPATH=<repo-root>` or `pip install -e .` here), because `-m` resolves the parent
package before the script's bootstrap runs.

It writes **no** `.qa.json` next to the input; pass `--report out.json` to save the report (which
carries `status`, `overall_verdict`, `summary`, and the full per-rule `results`, e.g. which cameras
tripped `ccd_temperature_*` / structure). Its **exit-code convention differs** from `qa_engine.py`:
**`0 = pass, 1 = warn, 2 = fail, 3 = system error`**. If `PRODCATG` is missing/unrecognised it falls
back to the base `qa_config.yaml`.

**Quiet by default — the exit code IS the interface.** A `0`/`1`/`2` verdict prints **nothing** on
stdout or stderr; callers branch on the exit code (and read `--report` JSON for detail). `--verbose`
(`-v`) opts into the human-readable summary plus per-rule detail (summary on stdout for PASS, stderr
for WARN/FAIL). One deliberate exception: a **system error still prints one line to stderr and exits
`3`** even when quiet, so a broken invocation stays diagnosable. Importing the QA code also drags in
the full pipeline (Trace/Cube/Ray), which prints a banner to stdout and a `pkg_resources`
deprecation warning at import time; `QA_assess.py` **suppresses that import chatter** too, so nothing
leaks onto either stream.

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
doc's requirement is to *track* bias level and its drift; **row/column structure is `FAIL` on
every cal type** (bias, dark, LDLS flat, sky flat, arc) — odd/anomalous structure must alert the
observer. All background/structure/saturation limits are per-detector (`extension.name`) ×
`readout_mode`.

### BIAS (`CAL.R-BIA`)
| Rule | Region / source | Metric | Threshold | Sev | Flags |
|---|---|---|---|---|---|
| shutter_exptime_consistency | SEXPTIME vs REXPTIME | abs_or_rel_diff | abs≈0.33 s / rel 10% | FAIL | shutter stuck/aborted |
| edge_background_level | bottom_stripe | median | per-det band | WARN | bias level drift / warm |
| frame_level_median | full_frame | median | per-det band | WARN | level drift / warm |
| frame_noise_rms | full_frame | std | per-det cap | WARN | excess noise |
| row_structure / column_structure | full_frame | row/col banding | per-det cap | FAIL | banding / odd structure |
| saturation_fraction | full_frame | fraction>63000 | per-det cap (~0) | FAIL | saturation |
| ccd_temperature_warm | CCDTEMP* (per ext) | range | > per-camera baseline **+8 °C** (red +10) | WARN | camera warming above its healthy baseline |
| ccd_temperature_hot | CCDTEMP* (per ext) | range | > per-camera baseline **+15 °C** (red +16) | FAIL | camera running hot |
| ccd_temperature_shutoff | CCDTEMP* (per ext) | range | > **−60 °C** absolute | FAIL | near TEC-limit / shut-off |

### DARK (`CAL.R-DRK`)
Same as BIAS (SLOW, 600 s) except saturation is **WARN** (hot pixels tolerated). Structure is
**FAIL** — this is where the June-2026 odd-structure dark trips (`column_structure` 2.B.Green
13.1 ≫ cap 2.0; `row_structure` 2.B.Green 6.5 > 2.98; `column_structure` 1.B.Green 4.1 > 2).

### LDLS_FLAT / SKY_FLAT / ARC_THAR (`CAL.R-FLT` / `-SKY` / `-ARC`)
shutter (FAIL), edge-stripe background (WARN), row/column structure (**FAIL** — see below),
per-detector saturation (WARN), per-camera temperature (warm/hot/shut-off). **No full-frame level check**
(illumination scales with exposure time, so a fixed level band is not physical).

Row/column structure is **FAIL** even though these frames carry real fibre/line structure,
because the per-detector caps are heavy-tail (≈1.5× the worst normal frame): every normal
illuminated frame sits at ≤0.67× its cap, so only a **gross odd-structure anomaly** trips it.
Validated across **115 normal illuminated frames** (39 baseline arcs + 16 normal 2026-07-10
commissioning arcs + 36 LDLS + 24 twilight) → **0 breaches**; the anomalous 2026-07-10 arcs sit
at **1.8–3.0× cap on all 22 detectors** → FAIL. Saturation is deliberately **not** the
discriminator here: normal red ThAr arcs already saturate (p99 ≈ 65535), so only structure
cleanly separates the defect.

### SCIENCE (`SCI.R-*`)
| Rule | Source | Threshold | Sev |
|---|---|---|---|
| shutter_exptime_consistency | SEXPTIME vs REXPTIME | abs 1 s / rel 10% | FAIL |
| saturation_fraction | full_frame fraction>63000 | ≤0.02 | WARN |
| **camera_warming_gradient** | full_frame background-gradient RATE (ADU/s) | ≤ 1.3 ADU/s (exp ≥ 8 s) | **WARN** |
| ccd_temperature_warm | CCDTEMP* per ext | > per-camera baseline +8 °C (red +10) | WARN |
| ccd_temperature_hot | CCDTEMP* per ext | > per-camera baseline +15 °C (red +16) | FAIL |
| ccd_temperature_shutoff | CCDTEMP* per ext | > −60 °C absolute | FAIL |

Per-detector **level/edge-background bands are still omitted** on science (absolute levels vary
with target and sky, and there are no science baselines to calibrate them). The camera-warming
signal on science is caught by **`camera_warming_gradient`**, an **exposure-normalized
background-gradient RATE**. A warming detector grows a dark-current glow — a smooth large-scale
background gradient — that accrues *fast*; the metric measures that gradient with quarter-block
**medians** (`max(|median(bottom¼)−median(top¼)|, |median(left¼)−median(right¼)|)`) and divides by
the exposure time (`SEXPTIME`→`REXPTIME`→`EXPTIME`). Because dark current is a **rate**, this cleanly
separates a warming detector (3A.green on `2026-05-06 06-08` = **2.1 ADU/s**) from *bright/long*
science, whose slow sky/scattered-light gradient accrues at **≤0.06 ADU/s** — a ~35× margin. One
static **WARN** threshold (**1.3 ADU/s**) is shared by all detectors (the rate is
detector-independent, so no lookup table); severity is **WARN only** — a warming detector alerts the
observer without failing the frame. Frames below **`min_exptime` = 8 s** **SKIP** the rule: a fixed
bias-structure gradient (≤8 ADU that does *not* scale with exposure) would inflate the rate at ~0 s.
The median blocks make it robust to the sparse bright fiber/target flux (standards land in fibers,
not the background). The per-camera temperature monitor uses the **same per-detector temperature
table** as the cals and works identically here.

Two earlier variants were **rejected**. (a) *Row/column structure with an absolute per-detector
floor* — false-flags bright/long science: `06-16-29.8` (180 s) and `10-18-12.0` (300 s) carry real
dispersed-spectrum + sky structure (row/col 5–62 on many detectors) that no absolute floor separates
from a warming glow. (b) *Full-frame robust σ (MAD)* — no transferable absolute threshold (healthy
noise ~3–15 ADU across epochs, a warming detector at ~10 sits inside a noisier dataset's healthy
spread). Both fail because they are *level/amplitude* metrics; the **rate** (per-second) is the only
one that isolates dark-current accumulation from a bright, long, or noisy but healthy exposure.

## The two must-flag cases

1. **Warm / accidentally-shut-off cameras.** Each detector is actively cooled and monitored, so
   the temperature check judges **every camera against its OWN healthy baseline** (median from
   clean baselines) and flags warming above it: **WARN at baseline + 8 °C, FAIL at + 15 °C**; the
   **red detectors run warmer and more variably, so they get +10 / +16**, plus an absolute
   **FAIL > −60 °C** near the TEC limit. Per-camera baselines span −78.5 °C (1A.red) to −105 °C
   (1A.green) — a 26 °C spread — so a single global threshold is unusable (warmer than −78 would
   false-WARN 1A.red on every frame); the per-detector caps give ~4–5× headroom over the ≤2.5 °C
   healthy scatter, so nuisance WARNs are near-zero. The value is read from whichever `CCDTEMP*`
   spelling the frame uses (keyword-variant note below). **The header temperature LAGS the pixel-
   level onset** — a camera can be warming (a spreading dark-current glow) while `CCDTEMP*` still
   reads its healthy baseline — so the temperature header check is **necessary but not sufficient**;
   the pixel-level check (below) is what catches the early onset. Real example: `2026-05-06 06-08`
   camera **3A.green is warming** (visible dark-current glow), yet its header reads −92.1 °C, right
   at its −92.9 baseline → the temperature check passes; on science the **`camera_warming_gradient`**
   rate check flags it (**2.1 ADU/s** vs a healthy ≤0.06) → WARN. The separate shutter fault on
   `05-10-56.6` (600 s requested → 181 s actual → FAIL) is a *different* incident.
2. **Odd detector structure and camera warming.** Two distinct pixel-level checks, by frame class:
   - **Cals** — per-detector `row_structure`/`column_structure` **FAIL** caps on **every cal type**
     (bias, dark, LDLS flat, sky flat, arc). Being per-detector, persistently-structured detectors
     (e.g. blue 3.A row-banding ≈1424, normal) keep their own band and don't false-fail, while a
     genuine anomaly — banding, blooming, a warming glow on an unilluminated frame — trips.
   - **Science** — the exposure-normalized **`camera_warming_gradient`** rate check (**WARN**, see
     the SCIENCE section). Structure caps are *not* used on science: real dispersed-spectrum + sky
     structure on bright/long targets is indistinguishable from a warming glow by any absolute
     structure floor. The rate metric targets the dark-current glow specifically; fixed electronic
     banding (not a smooth time-accruing gradient) is out of its scope on science and remains caught
     on the cal side.

   Real cases caught:
   - **June-2026 odd dark** — `column_structure` 2.B.Green 13.1 ≫ cap 2.0 (uniform frame). *(cal)*
   - **July-2026 odd arc** — `…/ut20260710_11/LLAMAS_2026-07-10_20-22-43.7_CAL22` (ThAr):
     anomalous vertical column-banding + red blooming on **all 22 detectors** at 1.8–3.0× cap →
     FAIL. The consecutive frame `20-22-58.8` shares the defect and also FAILs; the surrounding
     16 commissioning arcs (0.07–1.0 s) are normal and PASS. *(cal)*
   - **May-2026 warming science** — `…/20260505_06/LLAMAS_2026-05-06_06-08-21.4_SCI22` (10 s): camera
     3A.green has a dark-current glow → `camera_warming_gradient` **2.1 ADU/s** → WARN, while its
     header temperature is still at baseline. The bright/long `06-16-29.8` (180 s) and `10-18-12.0`
     (300 s) and the earlier same-night `04-43-06.4` (30 s) are all ≤0.10 ADU/s → PASS. *(science)*
   - **July-2026 commissioning science** — `…/ut20260710_11/…05-05-22.4_SCI22`: 4A.red has fixed
     horizontal banding (not a smooth dark-current gradient), so the rate metric does **not** flag it
     → PASS. Banding ≠ warming; this is by design (the earlier structure approach FAILed it). *(science)*

## Header-value checks (engine extension)

`qa_engine.py` / `qa_config_validator.py` provide a `header_check` rule type:
```yaml
header_check:
  op: range | abs_diff | rel_diff | abs_or_rel_diff
  source: SEXPTIME          # single keyword (range) or first operand (diff ops)
  other: REXPTIME           # second operand for diff ops
  per_extension_key: [CCDTEMP_1, CCDTEMP1, CCDTEMP-1]  # read from THIS ext; first populated
                                                        # spelling wins (requires per_extension)
  limits: {min: -140, max: -60}  # static limits for a range check, OR (per-detector) omit and use:
  abs_tol: 1.0                   # for *_diff ops
  rel_tol: 0.10
# per-detector range limits — a rule-level lookup replaces static `limits`:
expected_from_lookup: {table: temp, keys: [{from: extension.name}], max_field: warn_max}
```
Two features make the temperature check robust across data generations: **(a)** `per_extension_key`
accepts a **list of keyword spellings** — `CCDTEMP_1` (underscore) in baselines, `CCDTEMP1`
(no-separator) in commissioning frames, `CCDTEMP-1` (hyphen) — and the engine uses the first
populated one, so the check no longer silently SKIPs on newer files; **(b)** a `range` check may
take its limits from a rule-level `expected_from_lookup` (a per-detector table keyed on
`extension.name`) instead of a static `limits` block — this is how the per-camera temperature caps
are applied, reusing the same lookup machinery as the image-metric rules. `abs_or_rel_diff` passes
if within **either** tolerance: `abs_tol` absorbs the fixed sub-second shutter overhead on short
exposures; the tight 10% `rel_tol` catches gross proportional faults. Absent keywords → SKIPPED,
never failed.

## Threshold provenance & tracking

Derived from **153 baseline cal files** over **2026-04-07, 2026-05-02, 2026-06-30** (the
known-bad June-4 odd dark held out for validation). Methodology (robust):
- Per (type × mode × detector) samples are **MAD sigma-clipped** to reject anomalous baseline
  frames *before* deriving limits (so an outlier can't inflate its own cap).
- Background/level bands are additive: `median ± max(6σ_robust, 15 ADU)` — an **absolute** ADU
  floor, not a percentage.
- Structure/RMS/saturation caps are robust one-sided: `max(median + 8σ_robust, 3×median,
  floor)`.
- **CCD temperature is per camera**: each detector's own healthy median (within-camera
  sigma-clipped) plus a colour-aware warm margin (red +10/+16 °C, green/blue +8/+15 °C). Pooling
  all cameras would hide the warm-running detectors — 1A.red at −78 °C sits 12 °C above the −90 °C
  array median, so the legacy pooled threshold clipped it away as an "outlier."
- Self-check: 2.7% of normal detector-frames breach their edge-background band (WARN-level
  drift), 0 breach at FAIL severity.

Artifacts (`baselines/`): `qa_thresholds_derived.json` (limits + observed stats),
`qa_tracking_baselines.csv` (per-epoch medians for **drift monitoring**), and
`scripts/` (`sort_baselines`, `extract_stats`, `aggregate_thresholds`, `gen_configs`) — the
reproducible pipeline. Re-run to refresh thresholds as more nights arrive; the YAMLs are
generated, so edit `gen_configs.py` (or the YAML) and re-validate with `qa_config_validator.py`.

## Thresholds used

**Scalar / global thresholds** (identical across detectors):

| Check | Threshold | Severity | Notes |
|---|---|---|---|
| Saturation pixel value | **> 63000 ADU** | — | counted by `saturation_fraction` |
| Shutter `\|SEXPTIME − REXPTIME\|` | abs **or** rel tol (passes within either) | FAIL (cal) / FAIL (sci) | rel_tol = **0.10** all types; abs_tol per type below |
| — BIAS abs_tol | 0.214 s | | |
| — DARK abs_tol | 0.238 s | | |
| — ARC abs_tol | 0.328 s | | |
| — LDLS FLAT abs_tol | 0.252 s | | |
| — SKY FLAT abs_tol | 0.348 s | | |
| — SCIENCE abs_tol | 1.0 s | | 1 s + 10 % |
| CCD temperature shut-off (`CCDTEMP*`) | **> −60 °C** absolute | FAIL | near TEC limit; also guards any camera absent from the per-detector table; cold floor of the range = −140 °C |
| CCD temperature warm / hot | **per-camera** (see per-detector table below) | WARN / FAIL | baseline + 8/+15 °C (red +10/+16) |
| Science saturation fraction | ≤ 0.02 | WARN | fixed cap (no per-detector science baselines) |

**Per-detector caps** (keyed `extension.name` × `readout_mode`; formulas — robust MAD σ):

| Check | Cap formula | Floor | Severity |
|---|---|---|---|
| `edge_background_level` / `frame_level_median` | additive band `median ± max(6σ, 15 ADU)` | 15 ADU | WARN |
| `frame_noise_rms` | `max(median + 8σ, 3×median, 5.0)` | 5.0 ADU | WARN |
| `row_structure` / `column_structure` | heavy-tail `max(max×1.5, median + 8σ, 2.0)` | 2.0 | **FAIL (all cal types)** |
| `saturation_fraction` | heavy-tail `max(max×1.5, median + 8σ, 0.0005)` | 5e-4 | FAIL (bias) / WARN (other cals) |
| `ccd_temperature_warm` / `_hot` | per-camera `baseline_med + margin`; margin **WARN +8 / FAIL +15 °C**, **red +10 / +16**; keyed `extension.name` only (temperature is readout-mode independent) | — | WARN / FAIL |

Derivation constants: `K_LEVEL 6`, `FLOOR_LEVEL 15 ADU`, `N_CAP 8σ`, `FLOOR_STRUCT 2.0`,
`FLOOR_RMS 5.0`, `FLOOR_SAT 0.0005`. Structure caps are **heavy-tail** (the detector's own
unscreened max × 1.5, not sigma-clipped) so naturally-structured detectors (e.g. blue 3.A
row-banding ≈1424) keep loose caps and don't false-fail. The **full per-detector cap values**
(24 detectors × FAST/SLOW × type) live in the `lookup_tables:` of `qa_config_cal.yaml` and in
`baselines/qa_thresholds_derived.json` (`per_detector`); the shutter tolerances and the
**per-camera temperature baselines/caps** are in that same JSON (`shutter`, `ccd_temp.per_detector`),
surfaced as the `temp` lookup table in both `qa_config_cal.yaml` and `qa_config_science.yaml`.

## Baseline organisation

`QA_baselines/copies/` (untouched originals) was copy-sorted into sibling type folders
`Bias/ Darks/ Arcs/ lamp_flats/ twilight_flats/`, each with a `manifest.csv`.

## Tests performed (validation run, 2026-07-15)

The engine was run on representative frames of every type, the two must-flag field cases, and
the warm-incident calibrations. **Exit codes: `0` = PASS/WARN, `1` = FAIL, `2` = ERROR**; every
run also writes `<name>.qa.json` next to the input. (Frames are now overscan-trimmed to
2028×2048; the engine and the derived thresholds were both built on this geometry, so region
slices and limits stay consistent.)

### Targeted test cases — `python qa_engine.py <file> --config <cfg> --summary-only`

| # | Test case | Config | Exit | Verdict | What fired |
|---|---|---|:--:|:--:|---|
| 1 | Normal BIAS `04-07_19-23-19.3` | cal | 0 | WARN | benign drift: `edge_background_level`/`frame_level_median` on 3 green dets, ≤8 ADU over band |
| 2 | Normal DARK `04-07_21-13-34.9` | cal | 0 | PASS | 132 image-metric rules on 22 real ext, all pass; temp SKIPPED (no CCDTEMP) |
| 3 | Normal ARC ThAr `04-07_20-40-49.6` | cal | 0 | PASS | all evaluated rules pass |
| 4 | Normal LDLS FLAT `04-07_20-37-07.4` | cal | 0 | PASS | all evaluated rules pass |
| 5 | Normal SKY FLAT `05-02_22-15-33.4` | cal | 0 | PASS | all evaluated rules pass |
| 6 | **ODD-STRUCTURE dark** `06-04_20-36-33.1` | cal | **1** | **FAIL** | `column_structure` 2.B.Green **13.1** & 1.B.Green 4.1; `row_structure` 2.B.Green 6.5 (+RMS WARN) |
| 7 | **ODD-STRUCTURE arc** `07-10_20-22-43.7` | cal | **1** | **FAIL** | `column_structure`/`row_structure` FAIL on **all 22 detectors** (1.8–3.0× cap); red blooming/saturation |
| 8 | **ODD-STRUCTURE arc #2** `07-10_20-22-58.8` | cal | **1** | **FAIL** | same defect, 22 detectors (consecutive frame) |
| 9 | Normal commissioning arc `07-10_20-22-28.7` | cal | 0 | PASS | structure ≤0.67× cap — normal 2048-geometry arc passes |
| 10 | **WARM sci** `05-06_05-10-56.6` | science | **1** | **FAIL** | `shutter` REXP 600 → SEXP 181 s (Δ419) |
| 11 | Normal sci `05-06_06-08-21.4` (warm-running cams) | science | 0 | PASS | shutter nominal (10→10.06); every camera at its baseline (1A.red −78.3) → per-detector temp PASS |
| 12 | **WARM cal ARC** `05-06_05-50-23.7` | cal | **1** | **FAIL** | `shutter` REXP 1 → SEXP 122.7 s (Δ121.7) |
| 13 | WARM cal BIAS `05-06_05-56-25.8` | cal | 0 | PASS | shutter nominal (0.001→0.003) |
| 14 | **WARM cal BIAS** `05-06_05-56-30.7` | cal | **1** | **FAIL** | `shutter` REXP 0.001 → SEXP 0.352 s (stuck shutter) |
| 15 | Commissioning sci `07-11_05-05-22.4` | science | 0 | PASS | keyword-variant fix: temp rules now **EVALUATE** (`CCDTEMP1` read, were SKIPPED); all cameras healthy → PASS |

All 15 matched expectation. Normal frames PASS (bias WARN is expected drift); every must-flag
case FAILs on the intended rule (odd structure on cases 6–8, shutter on 10/12/14). The temperature
rules add no false flags — the doc example (row 11) and the commissioning frame (row 15) PASS.

### Full-baseline sweep — all 154 cal files, cal config

| Type | n | PASS | WARN | FAIL |
|---|:--:|:--:|:--:|:--:|
| Bias | 48 | 46 | 2 | 0 |
| Dark | 7 | 6 | 0 | 1 |
| Arc | 39 | 39 | 0 | 0 |
| LDLS flat | 36 | 30 | 5 | 1 |
| Sky flat | 24 | 18 | 6 | 0 |
| **Total** | **154** | **139** | **13** | **2** |

The 13 WARN are benign per-detector background/level drift (non-blocking). This sweep is
**unchanged after flipping illuminated structure to FAIL and after adding the per-detector
temperature monitor** — 0/115 normal illuminated frames breach their structure cap and 0/154
breach a temperature cap, so no new FAILs appeared. A parallel sweep of all **29 commissioning
science frames** (`ut20260710_11`) gives 0 false temperature WARN/FAIL; the 1 FAIL (shutter fault
`10-11-03.4`, Δ684 s) and 1 ERROR (truncated file `10-19-11.8` → exit 2) are unrelated real
issues, as expected for unvetted commissioning data.

#### FAIL cases found in the month-by-month baseline files
The **only two FAILs are both real anomalies — no normal baseline produced a false FAIL**:

| Month (epoch) | File | Type | FAIL rule | Detail |
|---|---|---|---|---|
| **2026-05-02** | `lamp_flats/…22-58-00.4` | LDLS flat | `shutter_exptime_consistency` | REXP 0.5 → SEXP 0.8 s (Δ0.3 > tol 0.252); only 1/36 flats |
| **2026-06-04** | `Darks/…20-36-33.1` | dark (held-out) | `row_/column_structure` | 2.B.Green col 13.1 ≫ cap 2.0; row 6.5 |
| 2026-04-07 | — | — | — | no FAILs (46 bias, arcs, flats all PASS/WARN) |
| 2026-06-30 | — | — | — | no FAILs |

Separately, the **2026-07-10 commissioning** arcs contribute 2 structure FAILs (`20-22-43.7`,
`20-22-58.8`) — these are *not* in the 154-file baseline set; they are the odd-structure ARC
must-flag case (table rows 7–8).

### Warm/shut-off cameras — per-detector detection

The temperature check monitors **each camera against its own healthy baseline** and reads the
`CCDTEMP*` keyword in whatever spelling the frame uses, so genuine warming on a science frame is
now caught: **WARN at baseline + 8 °C, FAIL at + 15 °C** (red +10/+16), plus absolute
**FAIL > −60 °C**. Validated: **0** false temperature WARN/FAIL across all 154 baseline cals and all
29 commissioning science frames — the normally-warm reds (e.g. 1A.red −78 °C) do not trip — while a
synthetic +8/+15 header injection WARNs then FAILs (`Test/test_qa_engine_header.py`). The doc's
named example `06-08-21.4_SCI22` correctly PASSES: it is a **normal** frame (nominal shutter, every
camera at its baseline), not a warming case; the separate May-2026 shutter-fault frame
(`05-10-56.6`, REXP 600 → SEXP 181) still FAILs on the shutter. Residual gap: mild dark-current
elevation with *no* temperature or shutter signal would still need a science background reference
(not yet available).

### Reproduce

`python baselines/scripts/run_qa_tests.py` re-runs the targeted 15-case matrix (prints the
exit code + fired rules per case, writes `qa_test_results.json`); `python
baselines/scripts/batch_tally.py` re-runs the full 154-file sweep in-process (no report writes);
`pytest Test/test_qa_engine_header.py` runs the header/temperature unit tests (shutter, per-detector
warm/hot, keyword variants). Thresholds regenerate via `python baselines/scripts/aggregate_thresholds.py`
then `python baselines/scripts/gen_configs.py` (re-validate with `qa_config_validator.py`). The
scripts carry the author's local baseline/warm/commissioning-folder paths at the top — edit those
for another machine.
