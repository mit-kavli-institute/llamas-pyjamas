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
