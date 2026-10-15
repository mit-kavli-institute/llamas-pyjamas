# Bias Module

This module handles bias frame processing and correction for the LLAMAS instrument.

## Core Functionality

The bias module performs:
- Reading and validation of bias frames
- Creation of master bias frames
- Bias subtraction from science, arc and flat data (bias-first preprocessing)
- Fallback bias estimation when a master bias or camera is unavailable
- Statistical analysis of bias levels

## Key Files

### `llamasBias.py`
- `BiasLlamas` class: takes a directory or list of raw bias MEFs; `master_bias()`
  median-combines them per (COLOR, BENCH, SIDE) and writes `combined_bias.fits`.

### `biasFirst.py`
- `bias_correct_frame()`: the first data operation for every frame type. Subtracts the
  read-mode-matched master bias, then the residual edge pedestal, and stamps
  `BIASSUB`/`BIASSRC`/`BIASLVL` keywords so later stages apply the correction once.
- `resolve_master_bias_file()`: picks `slow_master_bias.fits` or `fast_master_bias.fits`
  from the frame's `READ-MDE`.

### `__init__.py`
- `BiasNotFoundError`, `BiasReadModeError`, `BiasCameraMissingError`
- `generate_fallback_bias_hdu()`: constant-level bias HDU estimated from the frame itself.

### `biasChecking.py`, `biasPlots.py`
Diagnostics: inter-fibre masks, bias-level statistics and plots.

## Packaged master biases

`slow_master_bias.fits` and `fast_master_bias.fits` in this directory (and their copies in
`mastercalib/`) are what the pipeline subtracts by default. They are **single raw bias frames**
selected for quality, not median stacks.

### Updating them

```
python -m llamas_pyjamas.Scripts.update_bias_master RAW_DIR [--no-warn-fallback] [--no-qa]
```

The script finds every bias in `RAW_DIR` (primary header `PRODCATG = CAL.R-BIA`), runs the
[llamas-checks](https://github.com/mit-kavli-institute/llamas-checks) QA suite on each, and
installs the first frame per read mode (SLOW and FAST, sorted filename order) whose verdict is
`pass`. If a mode has no passing frame the first `warn` frame is used; `--no-warn-fallback`
disables that. It prints a verdict table for every candidate and exits 0 with
`Master BIAS files updated successfully (...)`, or 1 with `Update Failed: <reason>` on stderr.

`llamas-checks` is optional and is used only by this script; the pipeline never imports it.
On a machine without it the script refuses to install an unvetted frame unless `--no-qa` is
given, in which case it copies the first bias per read mode by header and warns.

### Median-stacked alternative

To build a true master from many frames, tag them as *Bias* in the reduction setup GUI
(`python -m llamas_pyjamas.Utils.reduxSetupGUI RAW_DIR`), which calls `BiasLlamas.master_bias()`
per read mode and writes `slow_master_bias.fits` / `fast_master_bias.fits` next to your config.

## Usage

```python
from llamas_pyjamas.Bias.llamasBias import BiasLlamas
from llamas_pyjamas.Bias.biasFirst import bias_correct_frame

# Median-combine a set of raw biases into combined_bias.fits
BiasLlamas(bias_files).master_bias()

# Bias-correct a raw frame (master bias + edge pedestal)
bias_correct_frame(raw_mef, output_dir)
```
