# Trace — fibre tracing

Finds where each fibre lands on the detector and follows it along the dispersion axis. Everything
downstream — extraction, flat fielding, wavelength calibration — depends on these traces.

## What runs

`reduce.py` imports `run_ray_tracing` from **`traceLlamasMaster`**. That is the production
path.

Three modules define a `TraceLlamas` class and two define `run_ray_tracing`, which is a genuine
hazard when reading this directory. The distinction:

| Module | Status |
|---|---|
| `traceLlamasMaster.py` | **Production.** `TraceLlamas`, `TraceRay` (Ray actor), `run_ray_tracing()`, plus the trace-validation helpers `check_fibre_number()`, `drop_spacing_outliers()`, `validate_trace_comb()` |
| `traceLlamasMulti.py` | An earlier parallel variant with its own `TraceLlamas`/`TraceRay`/`run_ray_tracing`. Not used by `reduce.py` |
| `traceLlamas.py` | The original serial implementation (`saveTraces`, `loadTraces`, `traceAllCameras`). Deprecated |
| `traceLlamasMaster_backup.py` | A backup copy. Excluded from the API build |

```python
from llamas_pyjamas.Trace.traceLlamasMaster import run_ray_tracing

run_ray_tracing(fitsfile, channel=None, outpath=CALIB_DIR)
```

## Method

Peak detection on a collapsed comb, then a B-spline fit along the dispersion axis, per camera.
Tracing is parallelised across cameras with Ray. A per-camera fibre count that disagrees with the
expected complement is caught by `check_fibre_number()`; dead fibres are recorded rather than
silently absorbed, and extraction later emits one row per *live* fibre.

## Outputs

`LLAMAS_{color}_{bench}_{side}_traces.pkl` in the calibration directory. Master traces shipped
with the pipeline live in `../mastercalib/`; the pipeline falls back to them when a night's own
flats cannot be traced. See the repository [`README.md`](../../README.md) for where to download
them.

## See also

API reference: <https://mit-kavli-institute.github.io/llamas-pyjamas/sphinx/api/llamas_pyjamas.Trace.html>
