
# Extract Module

This module handles the spectral extraction functionality for the LLAMAS instrument. It extracts one-dimensional spectra from two-dimensional fibre traces.

## Core Functionality

The extraction module performs several key operations:
- Optimally extracts spectra using variance-weighted algorithms
- Handles cross-talk corrections between adjacent fibres
- Processes both science and calibration data
- Generates variance arrays for extracted spectra
- Manages metadata for extracted spectra

## Key Files

### 

extractLlamas.py


Current production version of the extraction code. Contains:
- 

ExtractLlamas

 class: Main class for spectral extraction
- Core extraction algorithms and utilities
- Optimal extraction routines

### `extractLlamasMaster.py`
Development version with planned Ray implementation for parallel processing (work in progress, not yet implemented).

## Usage

The extraction module is typically used as part of the LLAMAS data reduction pipeline:

```python
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

# Create extraction object
extractor = ExtractLlamas(fitsfile)

# Process data
extractor.process_hdu_data(hdu_data, hdu_header)

# Save extracted spectra
extractor.saveSpectra()
```

Note: The parallel processing features using Ray are currently under development and not yet available for production use.

## Horne (optimal) extraction and masked pixels

`method = 'horne'` (alias `optimal`) uses the bspline profile `P` of each fibre:
`F = Σ_valid(P·f/V) / Σ_valid(P²/V)`.

**`P` is normalised to 1 over the whole aperture in every column.** Pixels that are NaN or
otherwise invalid then drop out of both sums, and `F` stays unbiased.

Before 2026-11-14, `P` was renormalised over the valid pixels only. That returned `F` times
the valid share of the profile: a column with its peak pixel masked came out ~30% low.

**Masked pixels raise the error.** The information they carried is lost, so each column's
variance is multiplied by `Σ_all(P²/V) / Σ_valid(P²/V)`. This ratio is 1 when nothing is
masked. The error model itself (photon plus read noise over the effective aperture) is
unchanged. `Tests/test_horne_masking.py` checks both the flux and the error growth.

The tracer can also leave `P = 0` on aperture pixels whose flat value was an outlier. On
20261015_16 this removed no measurable profile share (0.000% on the green 4B, green 1A,
red 1A and blue 3B traces), so the traces are unchanged.

