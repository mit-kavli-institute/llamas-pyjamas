# Sky — modelling & subtraction

Two layers, kept deliberately separate.

## 1. Base sky model (`skyLlamas.py`) — unchanged

Operates in the **extraction/pickle domain**:

- `skyModel_1d(science_extraction_file, color, ...)` — fits a per-camera 1-D
  B-spline to throughput-corrected counts (middle-third-by-brightness fibres as a
  sky proxy) and writes it into each fibre's `obj.sky`. Saved as
  `*_sky1d_extractions.pkl`.
- `refineSkyX(...)` — optional per-fibre `xshift` refinement from sky-line
  centroids.

This model flows downstream into the RSS files as the **`SKY` extension**, where
`FLUX = COUNTS - SKY`.

## 2. Sky-subtraction framework (this addition)

A **post-fibre-flat, per-colour, FITS-level** stage that *builds on* the base
model and applies standard IFU practice. It never modifies `skyLlamas.py`.

```
LLAMAS_{name}_RSS_{color}_FF.fits   ──►   LLAMAS_{name}_RSS_{color}_FF_SKYSUB.fits
```

Per colour:

| Step | Module | What it does |
|------|--------|--------------|
| Source masking | `skyMask.py` | `build_sky_fiber_mask` — pick sky-dominated fibres from `FIBER_TYPE` + white-light (`nansum` of `COUNTS`). |
| OH scaling | `skyScale.py` | `scale_sky_per_fiber` — fit a signed per-fibre coefficient against the base `SKY` line shape in OH windows; remove `alpha·line(SKY)`. Continuum untouched. |
| PCA residual | `skyResidual.py` | `clean_residuals` — ZAP-style: resample to a common wavelength grid, build an eigenbasis from masked fibres, project the leading components out of every fibre, map back. |
| Orchestrate | `skySubtract.py` | `subtract_sky_rss` / `subtract_sky_all_colors` — read FF, run the chain, write SKYSUB. |
| QA | `skyQA.py` | `sky_subtraction_qa` — before/after OH-residual RMS + figure. |
| Config | `skyConfig.py` | `SkySubtractConfig` — all tunables; `from_pipeline_config(config)`. |

### Why corrections are applied in FLUX space

`apply_fibre_flat_to_rss` divides **only `FLUX`/`ERROR`** by the fibre-flat `C_i`;
`COUNTS`/`SKY` are *not* fibre-flat corrected. Recomputing `FLUX = COUNTS - sky`
would drop `C_i`, so the framework applies its corrections directly to the FF
`FLUX` and uses `SKY` only as the OH line-*shape* template (`C_i` is smooth over
an OH line width).

### Output FITS

Input extensions copied through; `FLUX` ← cleaned flux; new `SKYRESID` =
total removed from the FF `FLUX` (`alpha·line(SKY) + PCA residual`). Header keys:
`SKYSUB2`, `SKYMETH`, `SKYPCANC`, `SKYNMASK`, `SKYNCOMP`.

## Usage

```python
from llamas_pyjamas.Sky import subtract_sky_rss, SkySubtractConfig

cfg = SkySubtractConfig(method="pca", pca_ncomp=20, qa_plots=True)
out = subtract_sky_rss("output/.../LLAMAS_<name>_RSS_green_FF.fits", config=cfg)
```

In the pipeline, set `sky_framework: True` in the config; `reduce.py` runs the
framework after the fibre-flat step and cube construction then prefers the
`_FF_SKYSUB` files.

### Key config keys (pipeline `config`)

`sky_framework` (enable), `sky_method` (`scaled`|`pca`), `sky_pca_ncomp`,
`sky_scale_window`, `sky_mask_method` (`whitelight`|`none`),
`sky_fiber_percentile`, `sky_qa_plots`.

## 3. Half-IFU mask

A frontend mask can block half of the IFU. Consecutive fibres on a benchside alternate
between the two fibre-map x halves (low-x `xpos ≤ 23.0`, high-x `xpos ≥ 23.5`). With the mask
in, every other trace is therefore dark, and a faint-first sky selection would build its model
from the dark fibres. `Utils/ifuMask.py` holds the logic; `reduce.py` drives it.

- **Pre-flight.** Every input's `HIERARCH FRONTEND IFU MASK-STATE` is read before any
  processing.
  - `TOP`/`BOTTOM` switch the handling on.
  - `EMPTY` means full field.
  - `HOME` means the whole field is covered. That is normal for biases, darks and lamp
    calibrations. The run stops, naming the files, if a science, standard, sky or twilight
    frame reports it.
  - Blank headers (older data) follow `IFU_mask`.
  - A quick check: `python -m llamas_pyjamas.Utils.ifuMask FILES...`.
- **Per frame**, after wavelength calibration, `measure_halves` compares the two halves'
  continuum and sky-line levels. This confirms the blocked half, or decides it for
  blank-header frames when `IFU_mask = true`; frames that look unmasked are left alone.
  - The result is stamped into the pickle and RSS primary headers: `IFUMSTAT`, `IFUMREQ`,
    `IFUMASK`, `IFUMBLK`, `IFUMSRC`, `IFUMDATA`, `IFUMRAT`, `IFUMLRAT`.
- **Sky.** Every sky pool is restricted to the open half, without the fibres next to the
  mask edge. That covers `skyModel_1d` (all selection methods and fallbacks, via
  `skySelect` `allowed=`), the `refineSkyX` template, the pedestal blank set and the
  framework's sky basis.
  - With only half the field open, an extended target that fills the open half leaves no blank
    sky in the frame. In that case use `sky_selection_method = frame` (a dedicated sky exposure;
    a masked sky frame is fine) or `skymap`.
- **Blocked fibres.** These have `SKY = 0` (no sky photons reach them), RSS `MASK` bit 4,
  FIBERMAP `IFU_BLOCKED = T` and `NIFUMSK` in the primary header.
  - The pedestal and framework leave them untouched.
  - Cubes NaN the blocked half (both the simple and the traditional constructor).
  - Registration's `per_fibre_flux` returns NaN for them.
  - `Combine` transparency fitting leaves half-masked exposures unscaled.
- **Hidden keys:**
  - `IFU_mask_force_off = true` disables the half-mask handling (it never bypasses the HOME
    failure);
  - `IFU_mask_half = top|bottom` is used for blank-header frames the data check cannot decide.
- **Re-runs.** Frames reduced under a different mask request stop the run with a request for
  `clobber = true`.

Future work:
- tracing, fibre flats, twilight throughput and arcs taken with the mask in (pre-flight only
  warns about them today);
- the extraction-time white-light QA images;
- confirming `BOTTOM` → low-x on a real frame (`TOP` → high-x was measured on
  2026-01-07 09:16:45);
- waveQA medians that include blocked fibres;
- the CRR cube adapter.
