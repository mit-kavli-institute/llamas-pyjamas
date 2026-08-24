# LUT — lookup tables and reference data

Static reference data the pipeline reads at run time. Some of it ships with the repository; the
large calibration products do not and must be downloaded (see the repository
[`README.md`](../../README.md)).

## Fibre maps

| File | Status |
|---|---|
| `LLAMAS_FiberMap_rev04.dat` | **Current.** What the code reads — `Flat/fibre_flat.py`, `Cube/cubeConstruct.py`, `Image/WhiteLightModule.py` |
| `LLAMAS_FiberMap_rev02.dat` | Superseded. Kept for reference |

Columns: `bench`, `fiber`, `xindex`, `yindex`, `xpos`, `ypos`. `bench` is the bench-side label
(`1A`…`4B`) and `fiber` restarts from 0 within each bench-side. rev04 holds **2392** fibres:
298/300 per bench-side in the order 1A, 1B, 2A, 2B, 3A, 3B, 4A, 4B.

Note that an RSS file has one row per **live** fibre, so its row count is normally below 2392 and
varies with the number of dead fibres. Resolve rows through `FIBERMAP['BENCHSIDE']` rather than
assuming fixed offsets — see
[`docs/workflow/07-reference.md`](../../docs/workflow/07-reference.md#file-formats).

## Wavelength calibration

| File | Role |
|---|---|
| `ThAr_MagE_lines.dat` | ThAr line list |
| `{red,green,blue}_peaks.csv` | Per-channel peak catalogues used by `Arc/arcSurface.py` |
| `LLAMAS_reference_arc.pkl` | **Not in the repository** — download it (git-ignored). The reference wavelength solution that `Arc/` transfers onto science extractions |

## Trace lookup

`traceLUT.json` is the active table; `traceLUT_orig.json` and `traceLUT_template.json` are kept
alongside it. Dated and experiment-specific variants (`traceLUT_20241129.json`,
`traceLUT_jan2025.json`, `traceLUT_blueflip.json`, …) are **git-ignored** and exist only in local
working copies.

## Flux standards

[`standards/`](standards/README.md) holds the flux-standard index and spectra used by
`Flux/fluxStandards.py` and `Flux/sensFunc.py`. `sensfunc_breakpoints.dat` holds the sensitivity
function's b-spline breakpoints.

## Not a Python package

This directory contains data only — no `__init__.py`, no modules, nothing in the API reference.
