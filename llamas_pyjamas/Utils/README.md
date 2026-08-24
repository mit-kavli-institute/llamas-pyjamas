# Utils — shared helpers

Cross-cutting utilities used across the pipeline: logging, header checks, astrometry, Ray
lifecycle management, and the setup GUI.

| Module | Role |
|---|---|
| `utils.py` | `configure_pipeline_logging()`, `setup_logger()`, `check_header()`, `count_trace_fibres()`, `exposure_time()`, `concat_extractions()` — the four `reduce.py` imports live here |
| `rayManager.py` | Ray lifecycle: `init_ray()`, `resolve_run_temp_dir()`, `prune_stale()`, `preflight_disk_check()`, `check_inputs_reachable()`, `cleanup_scratch()` |
| `register.py` | Gaia-anchored astrometric registration. `query_gaia()`, `detect_fibre_sources()`, `per_fibre_flux()`, `RegistrationResult` |
| `wcsLlamas.py` | `celestial_wcs()`, `pointing_from_header()`, `fit_wcs_from_stars()`, `register_pointing()` |
| `waveFrame.py` | Heliocentric / barycentric correction: `velocity_and_factor()`, `stamp_and_factor()` |
| `deadfibers.py` | `live_fibre_ids()`, `insert_dead_fibre_rows()` — converts between live-row and fibre-map indexing |
| `centroid.py` | `fibre_centroid()`, `Centroid` |
| `detectorProps.py` | Per-camera gain / read-noise from the lab CSV, by `CAMSN`: `get_props()`, `props_for_header()` |
| `reduxSetupGUI.py` | Classifies a raw night and writes the config file: `scan_directory()`, `generate_config()`, `combine_biases()`, `parse_config()` |
| `reporter.py` | `PipelineReporter` — run-level progress and summary reporting |
| `clean_ray_scratch.py` | Maintenance: clears stale Ray scratch directories |

Plotting helpers are **not** here — `plot_traces_on_image()`, `plot_ds9()` and the trace-QA
plots live in [`../QA/llamasQA.py`](../QA/llamasQA.py).

## Logging

Per-module: `logger = logging.getLogger(__name__)`, or `setup_logger(...)` when file output is
needed. Run logs land in `reduced/logs/llamas_pipeline_*.log` and record what each stage reused
versus rebuilt.

## Starting a reduction

`reduxSetupGUI` is stage 1 of the workflow — it builds the master bias and writes `config.txt`:

```bash
python -m llamas_pyjamas.Utils.reduxSetupGUI /path/to/raw_night -o config.txt
```

See [`docs/workflow/01-setup-and-config.md`](../../docs/workflow/01-setup-and-config.md).

## See also

API reference: <https://mit-kavli-institute.github.io/llamas-pyjamas/sphinx/api/llamas_pyjamas.Utils.html>
