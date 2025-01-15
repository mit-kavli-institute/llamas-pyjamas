# Arc Module

This module handles wavelength calibration using arc lamp spectra for the LLAMAS instrument.

## Core Functionality

The arc module performs:
- Processing of ThAr arc lamp exposures
- Wavelength solution calculation
- Line identification and fitting
- Generation of wavelength calibration maps
- Quality control of wavelength solutions

## Key Files

### 

arcLlamas.py


Main arc processing class containing:
- 

ArcLlamas

 class: Core wavelength calibration
- ThAr line identification
- Polynomial fitting routines
- Wavelength solution generation
- Solution validation tools

## Usage

```python
from llamas_pyjamas.Arc.arcLlamas import ArcLlamas

# Create arc object
arc = ArcLlamas(arc_file)

# Process arc spectrum
arc.process_arc()

# Get wavelength solution
wavelength_solution = arc.get_wavelength_solution()

# Apply solution to science data
calibrated_spectrum = arc.apply_wavelength_solution(science_spectrum)
```

The arc module is typically run after extraction to convert pixel positions to wavelengths, enabling scientific analysis of the spectra.