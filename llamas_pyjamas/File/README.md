# File Module

This module handles file management and I/O operations for the LLAMAS data reduction pipeline.

## Core Functionality

The file module performs:
- FITS file handling and validation
- File sorting and organization
- Header keyword management
- Raw data import/export
- Directory structure maintenance
- Data product output formatting

## Key Files

### `fileLlamas.py`
Main file handling class containing:
- `FileLlamas` class: Core file operations
- FITS file reading/writing
- Header manipulation tools
- File type identification
- Path management utilities
- Output formatting standards

## Usage

```python
from llamas_pyjamas.File.fileLlamas import FileLlamas

# Create file handler
file_handler = FileLlamas()

# Sort raw files
sorted_files = file_handler.sort_raw_files(directory)

# Read FITS file
data, header = file_handler.read_fits(filename)

# Write processed data
file_handler.write_fits(data, header, output_file)

# Validate FITS header
is_valid = file_handler.validate_header(header)
```

The file module provides consistent file handling across the entire reduction pipeline, ensuring data integrity and proper metadata management.
## RSS `MASK` bits

Each RSS channel file carries a `MASK` extension (int16, one value per fibre and wavelength sample).
The bits are defined in `llamasRSS.py`. Every reader decides whether a sample is usable with
`bad_pixel_mask`: any bit that isn't informational makes the sample bad, and so does a `COUNTS`
value that is 0 or non-finite.

| Bit | Name | Kind | Meaning |
|---|---|---|---|
| 1 | `MASK_FLAT_LAMP_ONLY` | informational | The fibre flat's T_i did not come from this fibre's twilight |
| 2 | `MASK_UNMAPPED_FIBRE_INTERP` | informational | Fibre missing from the fibre map; its twilight gradient was interpolated |
| 4 | `MASK_NOT_FLAT_FIELDED` | bad | The fibre flat had no correction for this fibre |
| 8 | `MASK_DEAD_FIBRE` | bad | Reserved (legacy) |
| 16 | `MASK_NO_COUNTS` | bad | The extraction wrote no data here (zero-fill) |
| 32, 64 | — | reserved | Per-fibre sky quality (not written to `MASK` today) |
| 128 | `MASK_CR_PARTIAL` | informational | Horne extraction measured this sample from the pixels around cosmic-ray pixels; its error is larger to match |
| 256 | `MASK_CR_MISSING` | bad | A cosmic ray in the aperture: no data. `COUNTS`, `SKYSUB`, `ERROR` and `FLAM` are NaN |

**Cosmic-ray gaps are never bridged.** A resampler may interpolate across a single bad sample of
another kind, but never across bit 256 (`MASK_NO_BRIDGE_BITS`, read with `no_bridge_mask`). Other
fibres and other exposures fill that gap instead.

**Summary cards.** The primary header records `CRMISS` (samples carrying bit 256), `CRPART`
(samples carrying bit 128) and `CRMISSF` (`CRMISS` as a percentage of the samples that have
counts).
