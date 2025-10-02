# White Light Imaging Feature

## Overview
The White Light Imaging feature reconstructs 2D images from fiber spectroscopy data by collapsing extracted 1D spectra along the wavelength dimension. These reconstructed images provide immediate visual feedback about observation quality, target acquisition accuracy, and field content.

## Core Functionality
- **Spectral Integration**: Collapses 1D spectra to single intensity values per fiber
- **Spatial Reconstruction**: Maps fiber intensities back to sky coordinates
- **Multi-Channel Integration**: Combines red, green, blue channels for color imaging
- **Quick-Look Products**: Rapid image generation for real-time assessment
- **Astrometric Verification**: Validates telescope pointing and target acquisition
- **Field Identification**: Enables comparison with reference catalogs/images

## Key Files
- `Image/WhiteLightModule.py` - Core white light reconstruction algorithms
- `Image/processWhiteLight.py` - Processing workflows and batch operations
- `Postprocessing/binary_whightlight.py` - Binary processing and optimization
- `GUI/` - White light visualization and display tools

## Data Structures

### Input Requirements
- **Extracted Spectra**: 1D spectra from each fiber (ExtractLlamas objects)
- **Fiber Positions**: Spatial coordinates for each fiber
- **Wavelength Information**: For selective integration ranges
- **Observation Metadata**: Pointing information, field center, rotation

### Output Products
- **White Light Images**: 2D FITS images showing reconstructed field
- **Color Images**: Multi-channel combinations (RGB)
- **Quick-Look Products**: Rapid assessment images
- **Astrometric Solutions**: WCS information for coordinate mapping

## Processing Workflow

### Spectral Integration
1. **Wavelength Range Selection**: Choose optimal integration range per channel
2. **Background Subtraction**: Remove sky emission and instrumental artifacts
3. **Integration Method**: Sum or weighted average across wavelength
4. **Error Propagation**: Calculate uncertainties in integrated values

### Spatial Reconstruction
1. **Coordinate Mapping**: Convert fiber positions to image grid
2. **Interpolation**: Fill gaps between fiber positions
3. **Smoothing**: Apply appropriate spatial filtering
4. **Normalization**: Scale intensities for display

### Multi-Channel Combination
1. **Channel Alignment**: Register red, green, blue images
2. **Color Balance**: Adjust relative channel intensities
3. **RGB Combination**: Create natural color composite
4. **Enhancement**: Apply contrast and gamma corrections

## Usage Patterns
```python
from llamas_pyjamas.Image.WhiteLightModule import WhiteLight

# Basic white light reconstruction
white_light = WhiteLight(extraction_file)
image = white_light.generate_white_light()
white_light.save_white_light("output_image.fits")

# Multi-channel color image
color_image = white_light.create_color_composite(
    red_channel, green_channel, blue_channel
)
```

## Pipeline Integration
- **Real-Time Assessment**: Generated during extraction for immediate feedback
- **Quality Control**: Used to validate observation success
- **Target Verification**: Confirms correct target acquisition
- **Archive Products**: Standard data products for observation logs
- **Scientific Analysis**: Morphological studies and source identification

## Image Types

### Monochromatic Images
- **Single Channel**: Individual red, green, or blue channel images
- **Broadband**: Integration across full spectral range
- **Narrowband**: Integration across specific wavelength ranges
- **Quick-Look**: Rapid low-resolution versions for immediate assessment

### Color Composites
- **RGB Images**: Natural color representation using three channels
- **False Color**: Enhanced contrast using channel combinations
- **Scientific Color**: Calibrated photometric color images
- **Enhanced Images**: Processed for visibility and analysis

## Calibration and Processing
- **Flat Field Correction**: Applies fiber throughput corrections
- **Background Subtraction**: Removes sky emission and scattered light
- **Photometric Calibration**: Converts to physical flux units when possible
- **Astrometric Calibration**: Applies WCS coordinate systems
- **Distortion Correction**: Corrects for optical and instrumental distortions

## Output Specifications
- **File Format**: Standard FITS with proper headers
- **Coordinate System**: WCS information for astronomical analysis
- **Photometric Information**: Flux calibration and units
- **Quality Indicators**: Data quality flags and statistics
- **Processing History**: Complete reduction provenance

## Quality Assessment
- **Source Detection**: Automated identification of objects in field
- **Photometric Accuracy**: Comparison with reference catalogs
- **Spatial Resolution**: Assessment of image sharpness and PSF
- **Completeness**: Fraction of fibers contributing to reconstruction
- **Artifacts**: Detection and flagging of processing artifacts

## Configuration Options
- **Integration Ranges**: Wavelength limits per channel
- **Spatial Sampling**: Output image pixel scale and field size
- **Interpolation Method**: Grid reconstruction algorithms
- **Background Handling**: Sky subtraction approaches
- **Color Balance**: Multi-channel combination parameters

## Dependencies
- Astropy (FITS I/O, WCS, coordinate systems)
- NumPy/SciPy (image processing, interpolation)
- Matplotlib (visualization and analysis)
- PIL/Pillow (image format conversion)
- Photutils (source detection and photometry)

## Performance Notes
- **Generation Time**: ~30 seconds to 2 minutes per image
- **Memory Usage**: ~100-500 MB depending on field size
- **Resolution Limit**: Determined by fiber spacing and PSF
- **File Sizes**: ~10-100 MB per white light image
- **Batch Processing**: Efficient for multiple exposures

## Applications
- **Target Acquisition**: Verify correct pointing and centering
- **Field Identification**: Compare with finding charts and catalogs
- **Morphological Analysis**: Study galaxy structure and star formation
- **Photometric Analysis**: Measure object brightness and colors
- **Quality Control**: Assess data quality and completeness
- **Publication Products**: Images for papers and presentations

## Advanced Features
- **Adaptive Smoothing**: Variable smoothing based on local fiber density
- **Source-Specific Integration**: Optimized wavelength ranges per object type
- **Temporal Sequences**: Time-series white light imaging
- **Polarimetric Imaging**: Polarization-sensitive reconstructions (future)
- **Multi-Object Fields**: Handling of complex multi-target observations