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

### One twilight reduction

**Each twilight flat is reduced once** (`reduce_twilights`), right after the flat stage:
1. the pixel flat is applied;
2. the frame is extracted with the `twilight` recipe (no cosmic-ray cleaning);
3. it is wavelength-calibrated;
4. it is saved as `<flat dir>/twilight/<stem>_twilight.pkl`.

A sidecar `.json` records the recipe hash, the frame and the pixel flat. The pickle is reused
while they all match, so a resumed run does not reduce the twilights again.

**Both throughputs come from that one pickle.** Each colour uses the twilight configured for it
(`{red,green,blue}_twilight_flat`, else `twilight_flat`):
- the per-fibre throughput the RSS is divided by (`reduce.twilight_throughputs`);
- the fibre flat's T_i (`load_twilights` → `compute_fibre_flat_twilight`).

Before 2026-11-14 the twilights were reduced twice, by two code paths. The RSS throughput came
from whichever twilight pickle was newest, for every colour.

### Throughput is applied once

The RSS writer already divides each fibre by its throughput: the `relative_throughput` from the
twilight (`twilight_throughput = true`) or carried from the arc. It records that divisor in the
FIBERMAP column `THRUPUT`, and its source in the primary card `TPSRC`. The sky model uses the
same throughput.

`C_i = T_i × W_i` holds the same fibre throughput again (`T_i`), so `apply_fibre_flat_to_rss`
divides by `C_i / THRUPUT`. Before 2026-11-14 it divided by `C_i`, which applied the throughput
twice: the sky scatter within each benchside went from about 2% to 9–19%.

The `_FF` products record this as `FFVER = 2`. An RSS without `THRUPUT` is refused. A sensfunc
must come from a standard at the same `FFVER` as the science (`Flux/fluxCalibrate.py`).

### Provenance of the flat products

`pixel_maps.fits`, `flat_smooth_models.fits` and `unillum_mask.fits` carry the following in
their primary header:
- the `lamp_flat` recipe stamp (`RECIPE`, `RCPHASH`, `PIPEVER`, `FIBIDS`; see
  `Utils/recipes.py`);
- `CRCLEAN = False`: lamp flats are never cosmic-ray cleaned;
- `FLATFILR`/`FLATFILG`/`FLATFILB`: the flat used for each colour.

`fibre_flat_corrections.fits` carries:
- the `twilight` recipe stamp, or `lamp_flat` for the lamp-only fallback;
- `TWIFILR`/`TWIFILG`/`TWIFILB`: the twilight used for each colour;
- `SMRCPHS`: the recipe hash of the smooth models it was made from.

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
The table also gives the range of two other scatters: *raw* (no throughput correction) and
*tp only* (the RSS throughput alone), which is what a flat has to beat.

Two checks look at the lamp shape `W_i` itself, because a narrow dip in `W_i` becomes an
emission-like bump in every science spectrum divided by it:
- **narrow structure:** each fibre's `W_i` is divided by its 25-pixel running median, on the
  central 80% of its range. The robust rms of that ratio is reported, and dips deeper than 5%
  are counted;
- **twilight cross-check:** each twilight fibre, on a common wavelength grid, is divided by the
  median of its ±4 neighbours, which removes the solar spectrum. A `W_i` dip deeper than 2%
  that this ratio does not show at no less than half depth was put in by the lamp flat, not
  the fibre.

The pixel flat is checked too (`Flat.flatLlamas.assess_pixel_flat`): the share of corrected
pixels that sit at the clip floor, per detector. A detector, or a whole colour, far above the
rest means the lamp flat's fibre cores were lowered (cosmic-ray cleaning) or saturated. On
20261015_16 this share was 1.8% for green, against 0.01% for red.

The checks, from categorical to per-fibre:

| Check | Fires when |
|---|---|
| Fallback | a benchside used lamp-only throughput, fibres took the benchside-median T_i, or fibres were left uncorrected (`MASK_NOT_FLAT_FIELDED`) |
| Relative outlier | a benchside's T_i scatter is more than `TI_RELATIVE_OUTLIER` (1.5) × the median of the other benchsides of its colour |
| Relative: worse than the throughput | the sky scatter after the flat exceeds the scatter with the RSS throughput alone (`FFSA` > `FFST`) |
| Relative: clip floor | a pixel-flat detector, or a colour's median, has more than 5× the floor-clipped share of the others (above 0.1%) |
| Absolute ceiling | the sky scatter after the flat exceeds `FF_RESIDUAL_CEILING_PCT` (3%). This is **provisional**: it is set from what a working flat should leave (twilight-to-twilight repeatability is 0.3–1%), not tuned to current data. If it fails everywhere, the summary says so |
| Absolute: narrow structure | a benchside's `WINRMS` exceeds `NARROW_CEILING_PCT` (1%, provisional, set from twilight repeatability) |
| Twilight cross-check | more than `TWILIGHT_UNCONFIRMED_MAX` (20%, provisional) of a benchside's `W_i` dips deeper than 2% are missing from the twilight |
| Fibre outliers | a fibre's T_i is more than `TI_OUTLIER_SIGMA` (5) robust sigma from its benchside median |

### Header cards

| Where | Card | Meaning |
|---|---|---|
| each benchside extension of `fibre_flat_corrections.fits` | `TIMED`, `TIRMS`, `TIMAD` | median T_i; RMS and robust scatter about it, % (raw throughput, before flattening) |
| | `TINOUT`, `TINFALL` | fibres > 5 robust sigma from the median; fibres given the benchside-median T_i |
| | `WINRMS`, `WINDIP` | median narrow rms of `W_i`, %; dips deeper than 5% |
| | `WIDIPN`, `WIDIPTW` | `W_i` dips deeper than 2% tested against the twilight; the fraction it does not show |
| each extension of `pixel_maps.fits` | `NCORPIX`, `FCLIPLO`, `FCLIPHI` | pixels the flat corrects; % of them at the clip floor and ceiling |
| primary header of each `_FF` RSS | `FFSB<bs>`, `FFST<bs>`, `FFSA<bs>` | sky scatter within benchside `<bs>`: raw, with the RSS throughput only, and after the flat, % |
| | `FFSN<bs>` | noise on one fibre's sky level, % |
| | `FFNSKIP` | fibres without a correction |
| | `FFVER` | fibre-flat version; 2 = divides by `C_i / THRUPUT` |
| FIBERMAP of each RSS | `THRUPUT` | the throughput the row was divided by when the RSS was written |
| primary header of each RSS | `TPSRC` | where `THRUPUT` came from: a twilight extraction, or `arc` |
