# Flat Module

This module handles flat field calibration and processing for the LLAMAS instrument.

## Core Functionality

The flat module performs:
- Processing of twilight and dome flats
- Creation of master flat frames
- Flat field correction of science data
- Fiber response calibration
- Illumination correction

## Key Files

### `flatLlamas.py`
Main flat processing class containing:
- `FlatLlamas` class: Core flat handling
- Master flat creation routines
- Fiber throughput calculations
- Illumination correction methods
- Pixel-to-pixel response normalization

## Usage

```python
from llamas_pyjamas.Flat.flatLlamas import FlatLlamas

# Create flat object
flat = FlatLlamas(flat_files)

# Generate master flat
master_flat = flat.create_master_flat()

# Apply flat field correction
corrected_data = flat.apply_flat(science_data)

# Calculate fiber throughputs
throughputs = flat.calculate_throughputs()
```

The flat module is typically run after bias correction and before spectral extraction to normalize the pixel-to-pixel variations and fibre-to-fibre throughput differences.

## Fibre-to-fibre flat (`fibreFlat.py`)

`compute_fibre_flat_twilight` (or `compute_fibre_flat_lamp_only` without a twilight) writes
`fibre_flat_corrections.fits`, and `apply_fibre_flat_to_rss` divides each RSS by it to make the
`_FF` files.

### Fibre ids

**Every per-fibre table is keyed by physical fibre id.** This covers `FIBER_ID` in
`flat_smooth_models.fits`, in `fibre_flat_corrections.fits` and in the RSS FIBERMAP. The physical
id is the fibre's fibremap position.

**Extractions are indexed by live row.** Dead fibres are absent, so above a dead fibre the row
number and the physical id differ. The two are converted with `Utils/deadfibers.py`:
`live_fibre_ids` gives row → id, and `live_row_index` gives id → row.

Both flat products record the convention as `FIBIDS = 'physical'`. Files made before
2026-11-14 labelled fibres by row, so above the dead fibre on 2A (fibre 270) and 2B (fibre 49)
each fibre got its neighbour's correction. Those files are refused; rebuild them with
`clobber = true`.

### Checks reported after the fibre flat

`assess_fibre_flat` reads the header cards below and prints a summary under the wavelength and
sky stage's line, plus one line in the end-of-run report. The table shows the **sky scatter left
within each benchside after the flat**, measured on the science frames:
- per fibre, the median COUNTS over the central 80% of its wavelength range;
- per benchside, the robust scatter (1.4826 × MAD) of those levels, in %.

A frame counts only if the noise on one fibre's level is below 1%, so it can resolve the ceiling.

The checks, from categorical to per-fibre:

| Check | Fires when |
|---|---|
| Fallback | a benchside used lamp-only throughput, fibres took the benchside-median T_i, or fibres were left uncorrected (`MASK_NOT_FLAT_FIELDED`) |
| Relative outlier | a benchside's T_i scatter is more than `TI_RELATIVE_OUTLIER` (1.5) × the median of the other benchsides of its colour |
| Absolute ceiling | the sky scatter after the flat exceeds `FF_RESIDUAL_CEILING_PCT` (3%). This is **provisional**: it is set from what a working flat should leave (twilight-to-twilight repeatability is 0.3–1%), not tuned to current data. If it fails everywhere, the summary says so |
| Fibre outliers | a fibre's T_i is more than `TI_OUTLIER_SIGMA` (5) robust sigma from its benchside median |

### Header cards

| Where | Card | Meaning |
|---|---|---|
| each benchside extension of `fibre_flat_corrections.fits` | `TIMED`, `TIRMS`, `TIMAD` | median T_i; RMS and robust scatter about it, % (raw throughput, before flattening) |
| | `TINOUT`, `TINFALL` | fibres > 5 robust sigma from the median; fibres given the benchside-median T_i |
| primary header of each `_FF` RSS | `FFSB<bs>`, `FFSA<bs>` | sky scatter within benchside `<bs>` before and after the flat, % |
| | `FFSN<bs>` | noise on one fibre's sky level, % |
| | `FFNSKIP` | fibres without a correction |
