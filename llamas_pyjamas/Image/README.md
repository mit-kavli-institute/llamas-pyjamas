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
python -m llamas_pyjamas.Postprocessing.binary_whightlight <science.fits> [--hex] [--bias B] [--outfile F] [--plot] [-v]
```

By default a run prints only warnings and the output path. `-v`/`--verbose` adds per-detector
details: the bias file, the residual bias levels and the fibre-label source.

### Output file

Extensions, in order: `RED`, `RED_TAB`, `GREEN`, `GREEN_TAB`, `BLUE`, `BLUE_TAB`. A colour with no
data is left out, so look extensions up by name. Each `*_TAB` holds the `XDATA`, `YDATA` and `FLUX`
of every fibre used to build the image.

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
python -m llamas_pyjamas.Postprocessing.build_quicklook_fiberimg [--calib-dir DIR] [--out PATH.npz] [--force]
```

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
