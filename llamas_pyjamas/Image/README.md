# Image Module

This module handles image processing and reconstruction for the LLAMAS instrument, specifically the white light image creation.

## Core Functionality

The image module provides:
- White light image reconstruction from fibre spectra
- Image quality assessment tools
- FITS file manipulation and handling
- Flat field processing

## Key Files

### `imageLlamas.py`
Main image processing class containing:
- `ImageLlamas` class: Core image processing
- White light reconstruction algorithms
- Image quality metrics

### White Light Reconstruction
The white light reconstruction process:
1. Takes extracted 1D spectra from each fibre
2. Collapses the spectra along wavelength axis
3. Maps these values back to their original spatial positions
4. Reconstructs a 2D image showing what the telescope was pointing at
5. Useful for:
   - Target acquisition verification
   - Field identification
   - fibre positioning confirmation
   - Quick-look assessment of data quality

## Usage

```python
from llamas_pyjamas.Image.imageLlamas import ImageLlamas

# Create image object
imager = ImageLlamas(fitsfile)

# Generate white light image
white_light = imager.generate_white_light()

# Save reconstructed image
imager.save_white_light(output_file)
```

The white light images provide immediate visual feedback about the observation quality and pointing accuracy.

## Quick-look white light (`QuickWhiteLightCube`)

`Postprocessing/binary_whightlight.py` calls `QuickWhiteLightCube` so observers can check their
pointing. It runs on the instrument-control machine, so it is single-process by design (no Ray).

```bash
python -m llamas_pyjamas.Postprocessing.binary_whightlight <science.fits> [--hex] [--bias B] [--dark D | --no-dark]
    [--skyref R | --no-skyref] [--weighted auto|on|off] [--crclip] [--outfile F] [--plot] [-v]
```

By default a run prints only warnings and the output path. `-v`/`--verbose` adds per-detector
details: the bias, dark and sky-reference files, the residual bias levels, the fibre-label source
and each colour's sky level and scatter.

### What limits faint sources, and what the steps below do about it

Measured on 1350-1800 s SLOW frames from 2026-09-30 and 2026-10-14 (both nights, all colours):
photon plus read noise per fibre is 0.1-2 % of the sky, but the raw fibre sums carry structure
10-100x larger. In order of size: **dark current and glow** inside the fibre stack (blue 1A/4A/4B:
50-60 DN per fibre per second, invisible to the edge-row residual; 30-400 % of a dark-time blue
sky), a **stale single-frame master bias** (interior offsets of 2k-137k DN per fibre plus its baked-in
cosmic rays), and a **fixed fibre-to-fibre pattern** of 9-12 % that neither the LDLS lamp flat nor
the twilight-tie corrections reproduce but a sky-illuminated twilight flat does. The 5 s acquisition
frames are a different regime: read-noise and bias limited, with SLOW giving 4x the per-fibre S/N
of FAST. With the night's median bias, its dark and its twilight reference, the sky-fibre scatter of
an 1800 s frame falls from 11-12 % of the sky to 1.2-1.6 % in every colour, and the 5 s SLOW green
image goes from unusable (sky S/N 1.5) to S/N 10 per fibre.

### Output file

Extensions, in order: `RED`, `RED_TAB`, `GREEN`, `GREEN_TAB`, `BLUE`, `BLUE_TAB`, then
`RED_SKYSUB`, `GREEN_SKYSUB`, `BLUE_SKYSUB` and `WHITE`. A colour with no data is left out, so look
extensions up by name. The first six keep their meaning: the colour images are the raw fibre sums,
and each `*_TAB` holds `XDATA`, `YDATA` and `FLUX` for every fibre used to build the image, plus
`NORM` (`FLUX` divided by the per-fibre sky reference) and `SKYSUB` (below).

- **`{COLOR}_SKYSUB`** is `NORM` minus a robust per-bench-side sky (the 30th percentile of that
  bench-side's fibres, so an object covering part of a stripe is not removed with the sky), divided
  by that colour's sky-fibre scatter: an S/N map. Bench-side steps are gone and ds9's zscale lands
  on the faint sources. `SKYLVL` and `SKYSIG` (sky level and scatter, `NORM` units) and `SKYSIGF`
  (scatter as a fraction of the sky, the figure quoted above) are in its header.
- **`WHITE`** is the sum of the `*_SKYSUB` maps divided by the square root of their number: one
  S/N map for the pointing check. A colour with no signal (blue in a 5 s frame) only adds unit
  noise. `WHTCOLS` lists the colours used.

The primary header records, besides the keys below: `QLEXPT` (exposure time used), `DARKFILE`,
`DARKEXPT`, `DARKSCL`, `SKYREF`, `SKYREFN` (fibres without a reference value), `WEIGHTED` and
`CRCLIP` (pixels clipped, -1 when off).

### Calibrations: master dark and per-fibre sky reference

Both come from one afternoon command on the night's raw directory
(`python -m llamas_pyjamas.Scripts.update_quicklook_calibs RAW_DIR`, see `Bias/README.md`), which
also median-combines the night's biases. Each is optional: without it the quick look logs one line
and behaves as before.

- **Dark.** After the master bias, `Bias/{slow,fast}_master_dark.fits` (bias-subtracted, `DARKEXPT`
  in its header) is subtracted scaled by the frame's exposure time, then the edge-row residual is
  measured as before. The dark signature is stable from night to night (the same blue cameras at
  the same level two weeks apart), so a dark from a previous night is far better than none. A dark
  of the other read mode, or a frame with no exposure time in its header, is skipped with a
  warning, and so is any dark extension whose level is negative or whose pixel scatter exceeds
  four read noises (a bad detector that night; listed in `DARKSKIP`). The 2026-10-01 dark has
  green 1A and 2B like that. `--dark FILE` overrides, `--no-dark` disables.
- **Sky reference** (`mastercalib/LLAMAS_quicklook_skyref.npz`, built by
  `Postprocessing/build_quicklook_skyref.py`). One number per (detector, trace fibre): the fibre's
  summed flux in the night's twilight sky flats (`CAL.R-SKY`), normalised by the detector median
  and median-combined over frames; saturated detectors and detectors with no light are dropped.
  `NORM = FLUX / reference`. The file is keyed to the fibre-label cache: after the master traces
  (and the cache) change, the quick look ignores it with one warning until it is rebuilt. A
  reference from a different night still removes most of the pattern (2 % instead of 11 % in red
  and green after two weeks); sky-dominated science frames can also be used (`--science`, bright
  fibres masked). `--skyref FILE` overrides, `--no-skyref` disables.

### Short exposures: profile-weighted sums (`--weighted`)

Below 60 s the fibre sums are read-noise limited and the plain sum is not the best estimate. With
`--weighted auto` (default) such frames use, per fibre and column, the Horne (1986) estimate
`sum(P d) / sum(P^2)` with `P` the normalised spatial fibre profile, summed over columns; it returns
the plain-sum flux for a fibre that follows the profile, with 15-25 % less noise on 5 s frames and
no gain on long ones. It needs the profile cache
(`python -m llamas_pyjamas.Postprocessing.build_quicklook_fiberimg --profiles --force`, ~60 MB
in `mastercalib/`, rebuilt with the fibre-label cache); without it the quick look warns and sums
plainly. `on` / `off` force either.

### Cosmic-ray clip (`--crclip`, off by default)

Replaces pixels more than 8 sigma above a 1x5 median along the dispersion axis with that median
(sky lines survive, hits do not). It costs ~3 s per frame, and most speckles in short exposures came
from the single-frame master bias, which the median bias removes, so it is a flag rather than the
default.

### Operational note for the pointing check

Take the acquisition frame in SLOW, not FAST: for the same 5 s the per-fibre S/N is 4x higher
(read noise 3.5 vs 10 DN), and the FAST red 4A bias structure changes between consecutive frames
(hardware), which no master bias can remove. If FAST is used, take FAST biases in the afternoon.
Judge the pointing on `WHITE` or `RED_SKYSUB`; blue carries no signal in a 5 s frame.

The primary header records:
- `READMODE`: the frame's `READ-MDE`.
- `BIASFILE`: the master bias that was subtracted.
- `RB{C}{bench}{side}` (e.g. `RBR1A`, `RBG4B`, `RBB2A`): the residual bias subtracted from each
  detector, in DN. `RBMETHOD` names the estimator.
- `BIASPH` (only when there are any): the detectors whose master-bias extension is a constant
  placeholder, e.g. `blue1A,blue4A`.

### Bias subtraction (FAST and SLOW)

1. **Master bias.** Each detector gets the `READ-MDE`-matched master bias from `Bias/`, or the one
   given with `--bias`. If the chosen bias is in the other read mode and the matching master
   exists, the matching master is used instead, with a warning.
2. **Residual DC level, every read mode** (`estimate_residual_bias`). This is measured in the
   unilluminated rows outside the fibre stack, taken from the trace's `fiberimg`, then subtracted.
   - **Which rows:** every row more than 20 rows below the first or above the last fibre row,
     top and bottom, ignoring the two outermost detector rows.
   - **Why the level matters:** a stale master bias leaves a constant offset per detector. Each
     bench-side fills a contiguous stripe of the field, so the offset shows up as striations.
     Each fibre sums roughly 8000 pixels, so a 1 DN offset is about 8000 counts per fibre.
   - **Why a clipped mean:** raw and bias values are integers, so a median is quantised to 1 DN.
     The level is therefore a 3σ-clipped mean.

   This replaces the earlier FAST-only estimate, which was the median of rows 5–50. The fibre stack
   starts as low as row ~11, so on most detectors rows 5–50 contain fibre pixels and the estimate
   picked up sky, which over-subtracted the background.
3. **Diagnostics:**
   - One warning covers detectors whose residual exceeds 5 DN (`BiasCheckThresholds`), since the
     master bias may be out of date. It names up to three detectors, or else gives the count and
     the three largest; every level is in the `RB*` keys.
   - Constant (placeholder) master-bias extensions are listed in `BIASPH` and logged at INFO. They
     are left out of the stale-bias warning, because their ~1000 DN residual is the whole pedestal,
     not drift.
   - **Bias gradient.** When the clean rows at the bottom and top of a detector differ by more than
     5 DN, a second warning names the detector. A constant cannot fix that, so the bench-side will
     show a stripe until the master bias is remade. Seen on red4A and green4A in a 2026-09-26 FAST
     frame (−14 and +11 DN top-to-bottom), where the earlier frames showed none, so the bias
     structure has changed.
   - Placeholder science cameras stay at zero.

   **Why not a row-dependent background?** Measuring the level between the fibres was tried and
   rejected: the inter-fibre pixels carry fibre wings and scattered light (a per-row mean removes
   10–90% of the fibre flux), and low percentiles are biased by the FAST read noise (σ ≈ 10 DN).
   Only the rows outside the fibre stack are clean, and they give one number per end.

The residual level is removed per fibre (`offset × npix`) rather than from every pixel. The step
costs about 0.1 s per frame.

### Performance notes

- **Per-fibre sums:** `_detector_fibre_fluxes(fiberimg, nfib, benchside, data, ...)` works on
  runs of pixels that share a label. Fibres are horizontal bands, so a detector's raveled label
  image has only ~10k such runs. It sums each run with `np.add.reduceat` (float64), then bins the
  runs per fibre. This takes ~7 ms per detector, against ~31 ms for a per-pixel `np.bincount`, and
  ~7200 full-frame masks before that. Only NaNs are zeroed, so the result matches `np.nansum`.
  Fibres with no pixels are still skipped, and the dead-fibre trace→physical remapping is unchanged.
- **Fibre labels from the cache:** see below. This avoids unpickling 2.8 GB of trace objects.
- **Streamed float32 reads:** the science frame and master bias are opened memory-mapped with
  `do_not_scale_image_data=True`. `File/llamasIO.trim_and_orient` then scales, trims and orients
  one detector at a time into float32. It uses the same logic as `process_fits_by_color`, and
  float32 is exact for 16-bit data. This replaces astropy's pseudo-integer scaling of all 48 HDUs
  followed by a float64 copy.
- **No pkg_resources:** five pipeline modules imported `pkg_resources` without using it. That cost
  import time and printed a deprecation warning on every run, so the imports are removed.
- **Fibre positions:** `FiberMap_LUT` is a dictionary lookup (`FIBERMAP_XY`, built once at import
  from `LUT/LLAMAS_FiberMap_rev04.dat`) instead of an astropy Table filter per call. Misses still
  return `(-1, -1)`. Every module that uses `FiberMap_LUT` benefits.
- **One detector at a time:** each bias-subtracted frame (and trace pickle, on the fallback path)
  is released as soon as its fibre sums are done.
- **No needless writes:** nothing is written beside the science frame. `process_fits_by_color(...,
  write=False)` returns HDUs without writing `*_trimmed.fits`. The master-bias cache
  (`_load_bias_hdus_cached`) uses it, so it no longer writes a temp copy. The default
  (`write=True`) is unchanged for other callers.
- **Header-only camera scan:** `get_existing_cameras` checks `NAXIS` instead of loading each
  extension's data. The placeholder-extension scan, whose result was only logged, has been removed
  from the quick-look path.

The speed-ups on their own left the output bit-identical to the old code (images and `*_TAB`
tables, grid and `--hex`). This was checked on a 24-camera FAST frame and on a SCI22 frame with
placeholder cameras. Two later changes mean the output no longer matches the old code in either
read mode: the red-first extension order and the residual-bias step above. Fibre positions are
unchanged. The second speed pass (cache, streamed float32 reads, run-length sums) is bit-identical
to the version before it: images, `*_TAB` and `RB*` keys, on both frames, grid and `--hex`, with
and without the cache.

| `binary_whightlight.py` CLI, sci22 + sci24 (MacBook, warm disk cache) | wall | peak memory |
|---|---|---|
| before | 20–22 s | 4.4 GB |
| after | 5.3–6.1 s | 0.9–1.0 GB |
| after + residual bias in all modes | 5.5–6.4 s | 0.9–1.0 GB |
| + fibre-label cache, streamed float32 reads, run-length sums | 4.0–4.6 s | 0.8–1.0 GB |

About 2.2 s of the remaining time is package import (`llamas_pyjamas/__init__` pulls in ray,
pypeit and sklearn). Making the package `__init__` files load their submodules lazily is the next
step if more speed is needed. It would touch how the whole pipeline imports, including the Ray
pickling registration for `TraceRay`. The per-frame work is now ~1.5 s. Without the cache, a run
takes ~1 s longer. Expect larger absolute times on the instrument machine.

### Fibre-label cache (`Postprocessing/build_quicklook_fiberimg.py`)

The quick look only needs each detector's `fiberimg`, `nfibers`, `bench` and `side`, not the 24
master trace objects (~117 MB each, ~2.8 GB in total). This script writes just those fields to one
compressed `.npz` (int16 labels, ~1.4 MB, loads in ~0.2 s, builds in ~6 s):

```bash
python -m llamas_pyjamas.Postprocessing.build_quicklook_fiberimg [--calib-dir DIR] [--out PATH.npz] [--force] [--profiles]
```

`--profiles` also writes `mastercalib/LLAMAS_quicklook_profiles.npz`: each detector's spatial
fibre profile normalised per fibre and column (float16) with its per-column `sum(P^2)`, read lazily
one detector at a time by `QuickWhiteLightCube` for the profile-weighted sums of short exposures.
It shares the fibre cache's metadata, so the same freshness check applies.

- **Freshness check:** `QuickWhiteLightCube` uses `mastercalib/LLAMAS_quicklook_fiberimg.npz`
  whenever `quicklook_cache_is_fresh()` confirms that every recorded master-trace pickle still has
  the same size and mtime, and that no new pickle has appeared.
- **Fallback:** if the cache is missing or stale, the quick look unpickles the master traces, with
  identical output and ~1 s slower. It prints one warning with the rebuild command.
- **Rebuild with `--force` whenever the master traces change.** The quick look never rebuilds the
  cache itself.
- **Both trace filename forms are accepted** (`LLAMAS_master_{c}_{b}_{s}_traces.pkl` and
  `LLAMAS_{c}_{b}_{s}_traces.pkl`), through `Utils/utils.find_trace_pickle`, in the cache builder,
  the freshness check and the pickle fallback.
- **Trace pickles need the code that wrote them.** The 2026-09-29 master traces reference
  `llamas_pyjamas.Bias.BiasCameraMissingError`, which only exists from the `sensfunc-standard-match`
  branch onwards; on older code `pickle.load` fails with `AttributeError`, in the cache builder and
  in the quick look alike. Check with
  `python -c "import pickle; pickle.load(open('llamas_pyjamas/mastercalib/LLAMAS_master_red_1_A_traces.pkl','rb'))"`.
- **Build it once on each machine:** `mastercalib/` is not in git, so every machine that runs the
  quick look, including the instrument-control machine, needs its own copy.
