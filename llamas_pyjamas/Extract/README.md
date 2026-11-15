
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

## Extraction method

`extraction_method` in the config sets the method of **every** extraction in a run (science,
lamp flat, twilight, arc and sky), because mixing methods breaks the per-fibre throughput
calibration. Each product records it as `EXTMETH` (and `EXTHW`, the boxcar half-width), and a
resumed run never mixes methods (`Utils/README.md`).

| Method | What it does |
|---|---|
| `horne` (default since 2026-11-15; alias `optimal`) | Horne (1986): the profile-weighted, variance-weighted estimate below. Masked and cosmic-ray pixels drop out without bias. |
| `boxcar` (default before) | A trace-following aperture of ±`boxcar_halfwidth` px with fractional edge weights. A masked pixel in the aperture makes the sample missing. |
| `legacy` | The old profile-weighted mean, not flux conserving; for comparison only. |

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


## Cosmic-ray pixels are missing data

Science, flux-standard and sky frames are cleaned with L.A.Cosmic before extraction
(`Masking/README.md`). The extraction is given L.A.Cosmic's flag image as `cr_mask`, and the
flagged pixels are **missing data**. L.A.Cosmic's replacement values are never measured; they
are only used for display and for the smooth detector-background estimate.

| Method | A sample whose aperture holds a flagged pixel |
|---|---|
| `boxcar` | Has no data if any flagged pixel has aperture weight > 0. Counts and error are NaN, and `dq` gets `MASK_CR_MISSING` (256). There is no profile repair. |
| `horne` | Is measured from the other aperture pixels, as for any masked pixel above, with the error inflated by the information lost. `dq` gets `MASK_CR_PARTIAL` (128, informational). If less than `MIN_VALID_FRAC` (0.5) of the column's profile is valid, the sample has no data (NaN and `MASK_CR_MISSING`). |
| `legacy` | Ignores the mask (comparison method only). |

- **Into the RSS.** `generate_rss` writes `dq` into `MASK`, and the primary header records
  `CRMISS`, `CRPART` and `CRMISSF` (`File/README.md`). The extension header of the extraction
  records `CRNMISS` and `CRNPART` per camera.
- **Downstream.** A missing sample stays NaN through the fibre flat, sky subtraction and flux
  calibration. It is never interpolated across in the cube or spectrum resamplers, so other
  fibres and other exposures fill it.
- **Error model.** For Horne, the information ratio multiplies the whole variance: photon,
  read noise and the master dark's noise.

`Tests/test_cr_mask_extraction.py` covers both methods.
