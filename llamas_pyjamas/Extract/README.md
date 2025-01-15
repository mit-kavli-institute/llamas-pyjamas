
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