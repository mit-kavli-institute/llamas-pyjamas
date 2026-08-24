# Flat — flat fielding

Two separate corrections live here, and they are easy to confuse:

1. **Pixel-to-pixel (QE) flat** — the 2-D detector response, divided out of the science frames.
   Built from the lamp flats.
2. **Fibre-to-fibre flat** — the relative throughput of each fibre, applied in the RSS domain
   (`*_RSS_{color}.fits` → `*_RSS_{color}_FF.fits`).

LLAMAS has **no dome flats.** The inputs are lamp flats and twilight flats. The inter-bench-side
absolute scale is tied from the **lamp**, not the twilight — a twilight cannot break the
degeneracy between a real sky gradient and a bench-side throughput step.

## What runs

| Module | Role |
|---|---|
| `flatLlamas.py` | Pixel flat. `process_flat_field_complete()` (the full workflow), `process_pixel_flat_simple()`, `create_master_flat()`. Both entry points are imported by `reduce.py` |
| `flatProcessing.py` | `reduce_flat()`, `produce_flat_extractions()`, `apply_flat_field()` — flat reduction and extraction orchestration |
| `fibreFlat.py` | Fibre-to-fibre flat. `compute_fibre_flat_lamp_only()`, `compute_fibre_flat_twilight()`, `reduce_twilight_flat()`, `apply_fibre_flat_to_rss()` — all imported by `reduce.py` |
| `scattered2dLlamas.py` | `scattered2dLlamas()` — 2-D scattered-light model |
| `fibre_flat.py` | Earlier fibre-flat implementation (`FibreFlatField`, `run_fibre_flat`). Re-exported by `__init__.py` but not called by `reduce.py` |

## Running the pixel flat directly

There is no command-line interface on this module; call it from Python.

```python
from llamas_pyjamas.Flat.flatLlamas import process_flat_field_complete

results = process_flat_field_complete(
    red_flat_file="path/to/red_flat.fits",
    green_flat_file="path/to/green_flat.fits",
    blue_flat_file="path/to/blue_flat.fits",
    arc_calib_file=None,        # defaults to LUT/LLAMAS_reference_arc.pkl
    output_dir="./output",
    trace_dir="./mastercalib",
)
print(results['processing_status'], results['pixel_map_file'])
```

**Prerequisites:** master traces (`LLAMAS_master*traces.pkl`) in the calibration directory, and
a reference arc (`../LUT/LLAMAS_reference_arc.pkl`). Run tracing and arc calibration first.

**Outputs**, written to `<output_dir>/flat/`:

- `pixel_maps.fits` — one 24-extension MEF holding every detector's pixel map
- `flat_smooth_models.fits` — the smooth per-fibre models
- intermediate `*_extractions_flat.pkl` and `combined_flat_extractions*.pkl`

The pipeline is resume-aware: if these already exist it skips the stage unless `clobber = true`.

## Open reviews

Two standing reviews cover the current state of this subpackage, including known gaps:

- [`FLAT_FIELDING_REVIEW.md`](FLAT_FIELDING_REVIEW.md) — pixel-to-pixel QE path
- [`THROUGHPUT_FLAT_REVIEW.md`](THROUGHPUT_FLAT_REVIEW.md) — fibre throughput path

`lampThroughput.py` and `twilightTie.py` are prototypes addressing those reviews'
recommendations. They have tests (`Test/test_lamp_throughput.py`, `Test/test_twilight_tie.py`)
but **are not yet wired into the pipeline** — nothing imports them.

## See also

API reference: <https://mit-kavli-institute.github.io/llamas-pyjamas/sphinx/api/llamas_pyjamas.Flat.html>
