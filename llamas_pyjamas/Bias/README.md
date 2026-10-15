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
- `median_combine_mef(paths, out, subtract=None)`: detector-matched median of raw MEFs
  (float32, raw orientation), optionally minus another MEF; builds the packaged master
  biases and darks.

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

## Packaged master biases and darks

`slow_master_bias.fits` and `fast_master_bias.fits` in this directory (and their copies in
`mastercalib/`) are what the pipeline and the quick look subtract by default. Since 2026-10 they
are the **median of the QA-passing raw biases of a night** (float32, `NCOMBINE` and `COMBSRCn` in
the primary header); before that they were single raw frames. A single frame carries its own
cosmic-ray hits and hot pixels (20-80 pixels per detector), which then appear as dark speckles in
every bias-subtracted image; the median removes them.

`slow_master_dark.fits` / `fast_master_dark.fits` (optional) are the median of a night's darks
**minus the master bias of that mode**, with the dark exposure time in `DARKEXPT`. The quick-look
white light scales them to each frame's exposure time; the main pipeline does not use them yet.
Blue cameras 1A, 4A and 4B accumulate 50-60 DN per fibre per second of dark current and glow
inside the fibre stack, which the edge-row residual cannot see, so long blue exposures need one.

### Updating them

```
python -m llamas_pyjamas.Scripts.update_quicklook_calibs RAW_DIR [--no-warn-fallback] [--no-qa]
```

installs everything the quick look needs from one night's raw directory: the median biases, the
median darks and the twilight per-fibre sky reference (`mastercalib/LLAMAS_quicklook_skyref.npz`,
see `Image/README.md`). Each stage is skipped with a note when the night has no acceptable frame of
that kind, and the exit status is 1 only when nothing at all was installed. The biases alone:

```
python -m llamas_pyjamas.Scripts.update_bias_master RAW_DIR [--no-warn-fallback] [--no-combine] [--no-qa]
```

Both scripts find frames by primary-header `PRODCATG` (`CAL.R-BIA`, `CAL.R-DRK`, `CAL.R-SKY`), run
the [llamas-checks](https://github.com/mit-kavli-institute/llamas-checks) QA suite on each, and use
every frame per read mode whose verdict is `pass`, or every `warn` frame if there is no pass
(`--no-warn-fallback` disables that; tiers are never mixed). `--no-combine` copies the first frame
only, as before. They print a verdict table per frame type and end with
`Master BIAS files updated successfully (...)` / `Quick-look calibrations updated (...)`, or
`Update Failed: <reason>` on stderr with exit 1.

`llamas-checks` is optional and is used only by these scripts; the pipeline never imports it. On a
machine without it they refuse to install unvetted frames unless `--no-qa` is given.

### Median-stacked alternative

`BiasLlamas.master_bias()` (also behind the reduction setup GUI,
`python -m llamas_pyjamas.Utils.reduxSetupGUI RAW_DIR`) writes `combined_bias.fits` next to your
config; `median_combine_mef()` in the same module is the detector-matched combiner the scripts use.

## Usage

```python
from llamas_pyjamas.Bias.llamasBias import BiasLlamas
from llamas_pyjamas.Bias.biasFirst import bias_correct_frame

# Median-combine a set of raw biases into combined_bias.fits
BiasLlamas(bias_files).master_bias()

# Bias-correct a raw frame (master bias + edge pedestal)
bias_correct_frame(raw_mef, output_dir)
```
