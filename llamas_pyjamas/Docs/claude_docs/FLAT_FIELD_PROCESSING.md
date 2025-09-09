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