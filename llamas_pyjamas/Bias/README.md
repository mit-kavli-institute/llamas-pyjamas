# Bias — master bias creation and subtraction

Removes the electronic offset from every frame. This is the first correction applied, before
tracing, flat fielding or extraction, and it runs on science, flat, arc and standard-star frames
alike.

LLAMAS has two detector read modes — **slow** and **fast** — and each needs its own master bias.
The pipeline picks the one matching the frame's read mode.

| Module | Role |
|---|---|
| `llamasBias.py` | `BiasLlamas` — the class `reduce.py` imports. Construct with `BiasLlamas(input_data)`, then call `master_bias()` to combine the input frames |
| `biasFirst.py` | `resolve_master_bias_file()`, `bias_correct_frame()` — read-mode resolution and per-frame subtraction |
| `biasChecking.py` | Validation: `run_bias_checks()`, `check_calibration_biases()`, `measure_edge_dc_offset()`, plus the `BiasCheckThresholds` / `BiasCheckReport` dataclasses |
| `biasPlots.py` | Diagnostics: `plot_bias_check_dashboard()`, `plot_bias_level_heatmap()`, `plot_interfibre_residuals()` |

## Required files

`slow_master_bias.fits` and `fast_master_bias.fits` must sit in this directory. They are not in
the repository — see the repository [`README.md`](../../README.md) for the download location.
Both are git-ignored.

Beyond the master bias, a small per-frame **edge DC offset** is measured from unilluminated pixels
and removed; that is what `measure_edge_dc_offset()` provides.

## See also

API reference: <https://mit-kavli-institute.github.io/llamas-pyjamas/sphinx/api/llamas_pyjamas.Bias.html>
