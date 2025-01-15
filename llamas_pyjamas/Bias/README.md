# Bias Module

This module handles bias frame processing and correction for the LLAMAS instrument.

## Core Functionality

The bias module performs:
- Reading and validation of bias frames
- Creation of master bias frames
- Bias subtraction from science data
- Statistical analysis of bias levels
- Overscan region processing

## Key Files

### `biasLlamas.py`
Main bias processing class containing:
- `BiasLlamas` class: Core bias handling
- Master bias creation routines
- Bias statistics calculations
- Overscan correction methods

## Usage

```python
from llamas_pyjamas.Bias.biasLlamas import BiasLlamas

# Create bias object
bias = BiasLlamas(bias_files)

# Generate master bias
master_bias = bias.create_master_bias()

# Apply bias correction
corrected_data = bias.apply_bias(science_data)
```

The bias module is typically run as part of the initial data reduction steps to remove the electronic offset in the CCD readout.