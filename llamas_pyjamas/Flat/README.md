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