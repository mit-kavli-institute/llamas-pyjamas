# File — FITS I/O and RSS generation

Reads raw LLAMAS multi-extension frames and writes the pipeline's primary data product, the
row-stacked-spectra (RSS) file.

| Module | Role |
|---|---|
| `llamasIO.py` | Raw MEF reading. `llamasAllCameras` / `llamasOneCamera` wrap a frame and its extensions; `process_fits_by_color()` splits a frame by channel; `getBenchSideChannel()` resolves an extension's identity from its header |
| `llamasRSS.py` | `RSSgeneration` — assembles extractions into one RSS per colour. Also `update_ra_dec_in_fits()`, `apply_fibre_astrometry()`, `patch_rss_astrometry()` for the per-fibre WCS |

`reduce.py` imports `process_fits_by_color`, `RSSgeneration` and `update_ra_dec_in_fits`.

## Raw frame structure

A raw frame has one primary header plus 24 image extensions — 8 detector positions
(1A, 1B, 2A, 2B, 3A, 3B, 4A, 4B) × 3 colours, cycling red → green → blue per position. **Never
assume a fixed index for a given colour or bench-side**; read the identity from each extension's
header, which is what `getBenchSideChannel()` is for. Frames with missing cameras are repaired by
[`../DataModel/validate.py`](../DataModel/validate.py) before processing.

## RSS layout

One row per live fibre, per colour. Extensions: `PRIMARY`, `SKYSUB`, `ERROR`, `MASK`,
`COUNTS`, `SKY`, `WAVE`, `FWHM`, `FIBERMAP`, `SKYRESID`, `FLAM`, `FLAM_ERR`,
`FIBERWCS`. `FIBERWCS` is deliberately separate from `FIBERMAP` so the astrometry can be
re-solved without touching the data.

Full layout and the bench-side row ordering:
[`docs/workflow/07-reference.md`](../../docs/workflow/07-reference.md#file-formats).

## See also

API reference: <https://mit-kavli-institute.github.io/llamas-pyjamas/sphinx/api/llamas_pyjamas.File.html>
