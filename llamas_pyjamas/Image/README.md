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

### Performance notes

- **Per-fibre sums:** `_detector_fibre_fluxes` sums each detector with one `np.bincount` over
  `trace_obj.fiberimg`, instead of building a full-frame mask per fibre (~7200 of them). Only NaNs
  are zeroed, so the result matches `np.nansum`. Fibres with no pixels are still skipped, and the
  dead-fibre trace→physical remapping is unchanged.
- **Fibre positions:** `FiberMap_LUT` is a dictionary lookup (`FIBERMAP_XY`, built once at import
  from `LUT/LLAMAS_FiberMap_rev04.dat`) instead of an astropy Table filter per call. Misses still
  return `(-1, -1)`. Every module that uses `FiberMap_LUT` benefits.
- **One detector at a time:** each trace pickle and bias-subtracted frame is released as soon as
  its fibre sums are done, rather than keeping all 24 trace objects in memory.
- **No needless writes:** `process_fits_by_color(..., write=False)` returns the trimmed and
  oriented HDUs without writing `*_trimmed.fits` beside the science frame. The master-bias cache
  (`_load_bias_hdus_cached`) uses it too, so it no longer writes a temp copy. The default
  (`write=True`) is unchanged for other callers.
- **Header-only camera scan:** `get_existing_cameras` checks `NAXIS` instead of loading each
  extension's data. The placeholder-extension scan, whose result was only logged, has been removed
  from the quick-look path.

Output is bit-identical to the previous code (images and `*_TAB` tables, grid and `--hex`). This
was checked on a 24-camera FAST frame and on a SCI22 frame with placeholder cameras.

| sci24, `binary_whightlight.py` CLI (MacBook, warm disk cache) | wall | peak memory |
|---|---|---|
| before | 20–22 s | 4.4 GB |
| after | 5.3–6.1 s | 0.9–1.0 GB |

About 2 s of the remaining time is package import (`llamas_pyjamas/__init__` pulls in ray, pypeit
and sklearn). Expect larger absolute times on the instrument machine.

### Fibre-label cache (`Postprocessing/build_quicklook_fiberimg.py`) — not yet wired in

The quick look only needs each detector's `fiberimg`, `nfibers`, `bench` and `side`, but it
currently unpickles all 24 master traces (~117 MB each, ~2.8 GB in total). This script writes just
those fields to one compressed `.npz` (int16 labels, ~1.4 MB, loads in ~0.2 s):

```bash
python -m llamas_pyjamas.Postprocessing.build_quicklook_fiberimg [--calib-dir DIR] [--out PATH.npz] [--force]
```

`load_quicklook_fiberimg()` reads the cache. `quicklook_cache_is_fresh()` checks that every
recorded master-trace pickle still has the same size and mtime, and that no new pickle has appeared.

- **Not used by `QuickWhiteLightCube` yet:** new master traces are due, and the cache should be
  built from them.
- **When wiring it in:** use the cache only if it is fresh, and fall back to the pickles otherwise.
- **Regenerate it whenever the master traces change.** The default `--out` is
  `mastercalib/LLAMAS_quicklook_fiberimg.npz`, and an existing file is only overwritten with `--force`.
