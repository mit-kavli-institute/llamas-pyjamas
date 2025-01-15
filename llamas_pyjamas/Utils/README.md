# Utils Module

This module provides utility functions and helper routines used throughout the LLAMAS data reduction pipeline.

## Core Functionality

The utils module provides several essential functions:
- Plotting and visualization routines
- Data manipulation helpers
- Statistical analysis tools
- File handling utilities
- Configuration management

## Key Files

### 

utils.py


Main utilities file containing:
- Plotting functions for traces and spectra
- ZScale implementation for image display
- Data verification routines
- File handling helpers

Notable functions include:
- `plot_traces()`: Basic trace plotting
- 

plot_traces_on_image()

: Overlay traces on raw data
- `verify_header()`: FITS header validation
- `write_fits()`: FITS file output handler

## Usage

Common usage patterns:

```python
from llamas_pyjamas.Utils.utils import plot_traces_on_image

# Plot traces overlaid on raw data
plot_traces_on_image(trace_object, raw_data, zscale=True)

# Write output to FITS file
from llamas_pyjamas.Utils.utils import write_fits
write_fits(data, header, output_file)
```

The utils module is designed to be imported and used by other modules in the pipeline rather than run directly.