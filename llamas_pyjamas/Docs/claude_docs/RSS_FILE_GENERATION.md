# RSS File Generation Feature

## Overview
The RSS (Row-Stacked Spectra) File Generation feature converts wavelength-calibrated extracted spectra into standardized FITS files suitable for 3D cube construction and scientific analysis. RSS files organize fiber spectra in a systematic format with proper metadata and coordinate information.

## Core Functionality
- **FITS File Creation**: Converts extracted spectra to standard astronomical FITS format
- **Spatial Coordinate Assignment**: Maps fiber positions to RA/Dec coordinates
- **Multi-Extension Structure**: Organizes data with proper extensions and headers
- **Metadata Preservation**: Maintains observational and processing information
- **Quality Flag Propagation**: Carries forward extraction quality indicators
- **Multi-Channel Organization**: Handles red, green, blue channel data
- **WCS Integration**: Implements World Coordinate System for spatial mapping

## Key Files
- `File/llamasRSS.py` - Main RSS generation module
  - `RSSgeneration()`: Primary RSS file creation function
  - `update_ra_dec_in_fits()`: Coordinate system updates
- `File/llamasIO.py` - I/O utilities and FITS handling
  - `process_fits_by_color()`: Color channel organization
- `File/llamasOneCamera.py` - Single channel processing utilities

## Data Structures

### Input: Wavelength-Calibrated Extractions
- **ExtractLlamas Objects**: Wavelength-calibrated extracted spectra
- **Fiber Positions**: Spatial coordinates from trace information  
- **Observation Metadata**: Telescope pointing, timing, configuration
- **Quality Information**: Extraction flags and statistics

### Output: RSS FITS Files
- **Primary HDU**: Main data array (nfibers × nwavelengths)
- **Wavelength Extension**: 1D wavelength coordinate array
- **Variance Extension**: Uncertainty information per spectrum
- **Fiber Position Extension**: Spatial coordinate table
- **Metadata Extensions**: Observational parameters and processing history
- **WCS Headers**: World coordinate system information

## Usage Patterns
```python
from llamas_pyjamas.File.llamasRSS import RSSgeneration

# Generate RSS file from calibrated extractions
rss_file = RSSgeneration(
    calibrated_extractions,
    output_dir=output_directory,
    target_name="galaxy_123"
)

# Update coordinates in existing RSS
from llamas_pyjamas.File.llamasRSS import update_ra_dec_in_fits
update_ra_dec_in_fits(rss_file, ra_center, dec_center)
```

## Pipeline Integration
Called after wavelength calibration in `reduce.py`:
- Takes wavelength-calibrated ExtractLlamas objects
- Converts to RSS format for cube construction
- Preserves all calibration and quality information
- Creates input for advanced CRR cube reconstruction
- Enables standard astronomical analysis workflows

## File Structure

### FITS Extensions
1. **Primary HDU**: 2D flux array (nfibers × nwavelengths)
2. **IVAR Extension**: Inverse variance array for uncertainties
3. **MASK Extension**: Boolean quality mask
4. **WAVELENGTH Extension**: 1D wavelength coordinate vector
5. **FIBERPOS Extension**: Fiber position table (RA, Dec, fiber_id)
6. **METADATA Extension**: Observational parameters and processing log

### Header Information
- **Observation Details**: Target coordinates, exposure time, date
- **Instrument Configuration**: Channel, detector settings, calibration files
- **Processing History**: Pipeline version, reduction parameters
- **Coordinate System**: WCS information for spatial mapping
- **Quality Metrics**: Extraction statistics and flagging information

## Coordinate Systems
- **Fiber Coordinates**: Physical positions on detector (pixels)
- **Sky Coordinates**: RA/Dec positions accounting for telescope pointing
- **Relative Coordinates**: Arcsecond offsets from field center
- **WCS Integration**: Standard astronomical coordinate transformations

## Quality Control
- **Data Validation**: Checks for NaN values, valid wavelength ranges
- **Coordinate Consistency**: Validates fiber position accuracy
- **Header Completeness**: Ensures all required metadata present
- **Format Compliance**: Follows FITS standard conventions
- **Cross-Channel Consistency**: Validates multi-channel data alignment

## Output Products
- **RSS FITS Files**: Standardized spectral data files
- **Processing Logs**: Detailed conversion and validation reports
- **Quality Reports**: Data completeness and validation statistics
- **Coordinate Maps**: Fiber position visualization products

## Configuration Options
- **Output Directory**: Configurable save location
- **File Naming**: Template-based naming conventions
- **Coordinate System**: Reference frame selection
- **Extension Configuration**: Optional extension inclusion/exclusion
- **Compression Options**: FITS file compression settings
- **Quality Thresholds**: Validation criteria and flagging levels

## Dependencies
- Astropy (FITS I/O, WCS, coordinate transformations)
- NumPy (array operations)
- SciPy (interpolation, coordinate calculations)
- Standard FITS libraries

## Performance Notes
- **File Generation**: ~2-5 minutes per RSS file
- **Memory Usage**: ~500MB-2GB depending on fiber count and wavelength range
- **Disk Space**: ~100-500MB per RSS file
- **I/O Efficiency**: Optimized FITS writing with compression
- **Validation Overhead**: ~10-20% additional time for quality checks

## Format Specifications
- **FITS Standard**: Compliant with IAU FITS standards
- **Extension Naming**: Consistent extension naming conventions  
- **Header Keywords**: Standard astronomical keywords plus pipeline-specific
- **Data Types**: Appropriate precision for scientific analysis
- **Compression**: Optional GZIP compression for storage efficiency

## Integration Notes
- **CRR Compatibility**: Direct input to advanced cube construction
- **Analysis Software**: Compatible with standard astronomical tools
- **Archive Compliance**: Meets observatory data archive requirements
- **Version Control**: Processing provenance tracking
- **Multi-Object Support**: Handles multiple targets per observation