# Image Module

This module handles image processing and reconstruction for the LLAMAS instrument, specifically the white light image creation.

## Core Functionality

The image module provides:
- White light image reconstruction from fibre spectra
- Image quality assessment tools
- FITS file manipulation and handling
- Flat field processing

## Key Files

### `imageLlamas.py`
Main image processing class containing:
- `ImageLlamas` class: Core image processing
- White light reconstruction algorithms
- Image quality metrics

### White Light Reconstruction
The white light reconstruction process:
1. Takes extracted 1D spectra from each fibre
2. Collapses the spectra along wavelength axis
3. Maps these values back to their original spatial positions
4. Reconstructs a 2D image showing what the telescope was pointing at
5. Useful for:
   - Target acquisition verification
   - Field identification
   - fibre positioning confirmation
   - Quick-look assessment of data quality

## Usage

```python
from llamas_pyjamas.Image.imageLlamas import ImageLlamas

# Create image object
imager = ImageLlamas(fitsfile)

# Generate white light image
white_light = imager.generate_white_light()

# Save reconstructed image
imager.save_white_light(output_file)
```

The white light images provide immediate visual feedback about the observation quality and pointing accuracy.