# LUT (Look-Up Tables) Module

This module contains essential reference data files used by the LLAMAS data reduction pipeline.

## Key Files

### 

traceLUT.json


Core reference file containing:
- Fibre trace positions for each spectrograph arm
- Reference wavelength calibration data
- Used by the tracing module to:
  - Provide initial trace positions
  - Guide trace finding algorithms
  - Validate identified traces

### `LLAMAS_FiberMap_rev02.dat`
Fibre mapping configuration file that:
- Maps fibre numbers to physical positions
- Defines the relationship between:
  - Input fibre positions at telescope focal plane
  - Output positions on the spectrograph
- Essential for:
  - Reconstructing white light images
  - Mapping extracted spectra to sky positions
  - Understanding cross-talk between adjacent fibers

## Usage

These files are automatically loaded by the pipeline:

```python
# Trace LUT is loaded during trace finding
from llamas_pyjamas.Trace.traceLlamasMaster import TraceLlamas
tracer = TraceLlamas(fitsfile)  # Automatically loads traceLUT.json

# Fiber mapping is used during image reconstruction
from llamas_pyjamas.Image.imageLlamas import ImageLlamas
imager = ImageLlamas(fitsfile)  # Automatically loads fibermap.dat
```