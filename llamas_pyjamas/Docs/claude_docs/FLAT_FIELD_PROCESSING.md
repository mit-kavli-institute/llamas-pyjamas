# Flat Field Processing Feature

## Overview
The Flat Field Processing feature corrects for pixel-to-pixel sensitivity variations and fiber-to-fiber throughput differences using dome and twilight flat exposures. It creates calibration maps and applies corrections to science data.

## Core Functionality
- **Flat Field Creation**: Combines multiple flat exposures into master calibrations
- **Pixel Map Generation**: Creates pixel-level correction maps for each channel
- **Fiber Throughput Calculation**: Determines relative efficiency per fiber
- **Illumination Correction**: Corrects for non-uniform illumination patterns
- **Multi-Channel Processing**: Handles red, green, blue channels independently
- **Quality Assessment**: Validates flat field quality and completeness
- **Scattered Light Modeling**: Advanced scattered light subtraction

## Key Files
- `Flat/flatLlamas.py` - Main flat field processing
  - `process_flat_field_complete()`: Complete flat field workflow
  - `FlatLlamas` class: Core flat handling (legacy)
- `Flat/flatProcessing.py` - Advanced flat field algorithms
- `Flat/scattered2dLlamas.py` - Scattered light modeling and subtraction

## Data Structures
- **Flat Field Results Dictionary**:
  - `output_files`: List of generated pixel map FITS files
  - `master_flat`: Combined flat field data
  - `throughput_maps`: Fiber efficiency corrections
  - `quality_metrics`: Processing statistics and validation
- **Pixel Maps**: FITS files containing correction arrays
  - Per-channel pixel-level corrections (e.g., `flat_pixel_map_red1A.fits`)
  - Header metadata with processing parameters

## Usage Patterns
```python
# Complete flat field workflow
from llamas_pyjamas.Flat.flatLlamas import process_flat_field_complete

results = process_flat_field_complete(
    red_flat_file, green_flat_file, blue_flat_file,
    arc_calib_file=arc_file,
    output_dir=flat_output_dir,
    trace_dir=trace_dir,
    verbose=True
)

pixel_maps = results['output_files']
```

## Pipeline Integration
Called by `reduce.py:process_flat_field_calibration()`:
- Uses fiber traces from tracing module
- Takes flat field FITS files as input
- Generates pixel correction maps
- Applied during science data extraction
- Feeds corrected data to spectrum extraction

## Processing Workflow

### Master Flat Creation
1. Load and validate flat field exposures
2. Apply bias correction if available
3. Combine exposures using median or mean
4. Remove cosmic rays and bad pixels

### Pixel Map Generation  
1. Use fiber traces to define extraction regions
2. Calculate pixel-level corrections within fibers
3. Model inter-fiber scattered light
4. Generate per-channel correction maps
5. Save as FITS files for later application

### Throughput Calculation
1. Extract 1D profiles from master flat
2. Compare fiber-to-fiber variations
3. Calculate relative throughput corrections
4. Apply to extraction weights

## Output Products
- **Pixel Maps**: FITS files with pixel-level corrections per channel/benchside
- **Master Flats**: Combined flat field images
- **Throughput Maps**: Fiber efficiency correction arrays
- **QA Products**: Diagnostic plots and quality metrics
- **Processing Logs**: Detailed workflow documentation

## Configuration
- Flat combination method (median/mean)
- Cosmic ray rejection parameters
- Scattered light modeling options
- Output file naming conventions
- Quality control thresholds
- Processing verbosity levels

## Dependencies
- Astropy (FITS I/O, image processing)
- NumPy/SciPy (array operations, statistical functions)
- Matplotlib (diagnostic plotting)
- Pypeit (advanced flat field algorithms)

## Performance Notes
- Processing time ~10-20 minutes per flat set
- Memory usage ~1-3 GB for typical flat sizes
- Pixel map generation most computationally intensive step
- Scattered light modeling adds significant processing time
- Output file sizes ~50-200 MB per pixel map

## Quality Metrics
- Flat field uniformity statistics
- Pixel-to-pixel noise characteristics
- Fiber throughput stability
- Scattered light correction effectiveness
- Bad pixel identification and flagging
- Cross-channel consistency checks

## Advanced Features
- **Scattered Light Modeling**: Uses `scattered2dLlamas.py` for advanced background subtraction
- **Sunburst Pattern Correction**: Specialized handling of LLAMAS optical patterns
- **Multi-Flat Combination**: Combines dome and twilight flats optimally
- **Wavelength-Dependent Corrections**: Accounts for spectral variations in flat response

## Developer Fixes Needed

### Multi-Extension FITS Combination for Pixel Maps

**Issue**: Individual pixel map FITS files need to be combined into a single multi-extension FITS file with extensions ordered to match raw science frame structure.

**Current State**: 
- Flat field processing generates 24 individual pixel map files (3 channels × 4 benches × 2 sides)
- Files named as: `flat_pixel_map_red1A.fits`, `flat_pixel_map_green2B.fits`, etc.
- Each file contains a single 2D correction array with appropriate headers

**Required Implementation**:

#### New Function: `combine_pixel_maps_to_mef()`
**Status**: ✅ IMPLEMENTED
- Location: `Thresholding` class in `Flat/flatLlamas.py:1061`
- Combines individual pixel maps into single multi-extension FITS file
- Maintains proper extension ordering by color, bench, and side
- Preserves original header metadata and adds standardized keywords

#### Extension Ordering Specification:
Extensions must follow LLAMAS standard ordering to match raw science frames:
1. Primary HDU (empty, with metadata)
2. Blue channel extensions: FLAT_BLUE1A, FLAT_BLUE1B, FLAT_BLUE2A, FLAT_BLUE2B, FLAT_BLUE3A, FLAT_BLUE3B, FLAT_BLUE4A, FLAT_BLUE4B
3. Green channel extensions: FLAT_GREEN1A, FLAT_GREEN1B, etc. (same bench/side order)
4. Red channel extensions: FLAT_RED1A, FLAT_RED1B, etc. (same bench/side order)

#### Header Requirements:
Each extension must include:
- `EXTNAME`: Extension name (e.g., "FLAT_RED1A")
- `EXTVER`: Extension version number (1, 2, 3, ...)
- `CHANNEL`: Color channel ("BLUE", "GREEN", "RED")
- `BENCH`: Bench number ("1", "2", "3", "4")
- `SIDE`: Bench side ("A", "B")
- `BENCHSIDE`: Combined identifier ("1A", "2B", etc.)
- `COLOUR`: Color channel for compatibility
- `ORIGFILE`: Original individual file name
- `BUNIT`: Data units ("Counts")

#### Integration Points:
**Status**: ✅ IMPLEMENTED
- `generate_complete_pixel_maps()` modified to optionally create MEF file
- `process_flat_field_complete()` returns MEF file path in results
- Combined file created automatically during flat field processing

#### Output:
- Individual pixel map files: `flat_pixel_map_*.fits` (preserved for backward compatibility)
- Combined MEF file: `combined_flat_pixel_maps.fits`
- File size: ~400MB (24 extensions × ~16MB each)

#### Performance Considerations:
- MEF creation adds ~30 seconds to processing time
- Memory usage increases temporarily during combination
- Combined file reduces I/O operations in downstream processing
- Maintains all original precision and metadata

#### Configuration Options:
- `create_mef=True` parameter in `generate_complete_pixel_maps()`
- Can be disabled for testing or debugging individual files
- MEF creation failure does not affect individual file generation

#### Usage Example:
```python
# Automatic MEF creation during complete workflow
results = process_flat_field_complete(
    red_flat, green_flat, blue_flat,
    output_dir=output_dir
)

mef_file = results['combined_mef_file']
individual_files = results['output_files']

# Manual MEF creation from existing individual files
threshold_processor = Thresholding(red_flat, green_flat, blue_flat)
mef_file = threshold_processor.combine_pixel_maps_to_mef(individual_files)
```

This implementation ensures pixel map corrections can be efficiently applied during science data processing while maintaining the flexibility of individual channel files for specialized applications.