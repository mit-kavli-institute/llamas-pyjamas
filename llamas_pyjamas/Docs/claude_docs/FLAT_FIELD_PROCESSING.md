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

### B-spline Fitting and Pixel Map Generation  
1. Extract flat field spectra from master flats using fiber traces
2. Apply wavelength calibration to extracted spectra
3. **B-spline Fitting**: For each fiber, fit B-spline to `xshift` (pixel position along fiber) vs `counts` (flat field intensity)
4. **Pixel Map Creation**: Generate 2D maps containing B-spline predicted flat field values at each pixel location along fiber traces
5. Save individual pixel map FITS files per channel/bench/side combination

### Normalized Flat Field Creation
1. **Pixel Map Division**: Divide original flat field data by pixel maps (removes B-spline modeled variations)
2. **Trace Normalization**: Scale corrected data so median within fiber traces ≈ 1.0
3. **Background Setting**: Set pixels outside fiber traces to exactly 1.0
4. **Multi-Extension Output**: Create 24-extension FITS file ordered by `idx_lookup` for direct application to science frames

### Throughput Calculation
1. Extract 1D profiles from corrected flat field data
2. Compare fiber-to-fiber variations after pixel map corrections
3. Calculate relative throughput corrections
4. Apply to extraction weights

## Output Products
- **Pixel Maps**: Individual FITS files containing B-spline predicted flat field values per channel/bench/side (stored in `extractions/pixel_maps/`)
  - Format: `flat_pixel_map_red1A.fits`, `flat_pixel_map_green2B.fits`, etc.
  - Content: 2D arrays with predicted values along fiber traces, zero/NaN elsewhere
- **Normalized Flat Field**: Multi-extension FITS file for science frame correction (stored in `extractions/normalized_flat_field.fits`)
  - 24 extensions ordered by `idx_lookup` from `constants.py`
  - Values within fiber traces ≈ 1.0 after pixel map and trace corrections
  - Values outside fiber traces = exactly 1.0
- **Master Flats**: Combined flat field images for intermediate processing
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

## Technical Implementation Details

### Pixel Map Structure and Usage
**Pixel Maps Contain**: B-spline predicted flat field intensities along fiber traces
- **B-spline Input**: `xshift` (pixel position along fiber) vs `counts` (flat field intensity) for each fiber
- **B-spline Output**: Predicted flat field value at each pixel position  
- **Pixel Map Values**: These predicted values placed at corresponding pixel locations along fiber traces
- **Non-trace pixels**: Set to 0 or NaN in pixel maps

### Normalized Flat Field Creation Process
```python
# Step 1: Remove B-spline modeled variation
corrected_data = original_flat_data / pixel_map_data

# Step 2: Scale to median ≈ 1.0 within traces  
normalized_data = corrected_data / median(corrected_data[trace_mask])

# Step 3: Set background to 1.0
normalized_data[~trace_mask] = 1.0
```

### Directory Structure
- **Pixel Maps**: `extractions/pixel_maps/flat_pixel_map_*.fits` (individual files)
- **Normalized Flat**: `extractions/normalized_flat_field.fits` (24-extension MEF file)
- **Processing intermediate files**: Various directories as needed

### Extension Ordering Convention
Normalized flat field extensions follow `constants.py:idx_lookup` ordering:
1. Extensions 1-24: ordered by channel (red/green/blue), then bench (1-4), then side (A/B)
2. Headers include: CHANNEL, BENCH, SIDE, BENCHSIDE for proper matching
3. Compatible with science frame extension structure

## Advanced Features
- **Scattered Light Modeling**: Uses `scattered2dLlamas.py` for advanced background subtraction
- **Sunburst Pattern Correction**: Specialized handling of LLAMAS optical patterns
- **Multi-Flat Combination**: Combines dome and twilight flats optimally
- **Wavelength-Dependent Corrections**: Accounts for spectral variations in flat response

## Recent Implementation Updates

### ✅ COMPLETED: New Isolated Flat Field Processing Workflow

**Issue Resolved**: Complete redesign of the flat field processing workflow to properly create normalized images from B-spline fits and trace data.

**New Implementation**: Added comprehensive standalone workflow with the following functions:

#### Core Functions Added to `flatLlamas.py`:

1. **`find_trace_file(color, bench, side, trace_dir, fallback_dir)`**
   - Locates correct trace files using naming convention: `LLAMAS_master_{color}_{bench}_{side}_traces.pkl`
   - Searches primary directory first, then fallback (mastercalib)
   - Returns file path, location info, and existence status

2. **`create_pixel_maps_from_bsplines(extraction_file, trace_dir, fallback_dir)`**
   - Loads B-spline fit data from `combined_flat_extractions_calibrated_fits.pkl`
   - Matches each extension to correct trace file by metadata
   - Loads trace `.fiberimg` attribute and combines with B-spline values
   - Creates 2D pixel maps where trace pixels get B-spline predicted values
   - Returns pixel maps dictionary and comprehensive matching log

3. **`generate_normalized_images(pixel_maps, raw_flat_files, matching_log, output_file)`**
   - Divides raw flat field images by pixel maps (trace regions only)
   - Normalizes trace regions so median ≈ 1.0
   - Sets non-trace regions to exactly 1.0
   - Creates 24-extension `normalised_images.fits` following `idx_lookup` ordering
   - Includes detailed metadata in each extension header

4. **`print_flat_field_matching_log(matching_log)`**
   - Comprehensive verification log showing file matching and processing results
   - Cross-check of extension → trace file → B-spline data → raw flat file mapping
   - Processing statistics and quality metrics
   - Error reporting and troubleshooting information

#### Complete Standalone Test Script: `isolated_flat_fielding.py`

**Purpose**: Complete standalone script that replicates the entire flat field processing workflow from reduce.py, starting with raw flat files

**Enhanced Features**:
- **Complete Workflow**: Processes from raw red/green/blue flat files to final normalized images
- **User-configurable paths**: Modify file paths in USER CONFIGURATION section
- **Step-by-step processing**: Detailed validation and reporting at each stage
- **Debugging options**: STOP_AT_STEP parameter for inspecting intermediate results
- **Comprehensive logging**: Verbose output and detailed error reporting
- **Organized output structure**: Separate directories for traces, extractions, finals, logs
- **Dual verification**: Both existing workflow + new verification functions
- **Usage instructions**: Built-in help and troubleshooting guide

**Complete 7-Step Workflow**:
```python
# Step 1: Generate fiber traces from flat field files (optional)
generate_traces(red_flat, green_flat, blue_flat, trace_dir, bias)

# Step 2-6: Complete flat field processing workflow
process_flat_field_complete(
    red_flat_file, green_flat_file, blue_flat_file,
    arc_calib_file, use_bias, output_dir, trace_dir
)
# Includes:
#   - Step 2: Flat field extraction (extract spectra using traces)
#   - Step 3: Wavelength calibration (arc transfer for xshift vs counts)
#   - Step 4: B-spline fitting (fit B-splines to each fiber's xshift vs counts)
#   - Step 5: Pixel map generation (combine B-splines + trace.fiberimg)
#   - Step 6: Normalized image creation (divide raw flats by pixel maps)

# Step 7: Enhanced verification and cross-checking
pixel_maps, log = create_pixel_maps_from_bsplines(calibrated_file, trace_dir)
verification_file = generate_normalized_images(pixel_maps, raw_flats, log)
print_flat_field_matching_log(log)  # Comprehensive verification report
```

**Input Requirements**:
- **Raw flat files**: red_flat.fits, green_flat.fits, blue_flat.fits (multi-extension FITS)
- **Optional bias**: bias.fits for correction
- **Arc calibration**: For wavelength solution (uses default if not provided)
- **Existing traces**: Optional, will generate new traces if not provided

**Output Structure**:
```
output_dir/
├── traces/              # Generated or used trace files (.pkl)
├── extractions/         # Complete extraction workflow outputs
│   ├── *_extractions_flat.pkl         # Individual color extractions
│   ├── combined_flat_extractions.pkl  # Combined extractions
│   ├── combined_flat_extractions_calibrated.pkl  # With wavelength cal
│   ├── combined_flat_extractions_calibrated_fits.pkl  # With B-splines
│   ├── flat_pixel_map_*.fits          # Individual pixel maps
│   ├── combined_flat_pixel_maps.fits  # Combined MEF pixel maps
│   └── normalized_flat_field.fits     # Main output from workflow
├── final/               # Verification and final outputs
│   ├── verification_normalised_images.fits  # Cross-check output
│   └── processing_summary.txt         # Summary report
└── logs/                # Detailed processing logs
```

**Key Features**:
- **Debugging Support**: STOP_AT_STEP=N to halt at specific stage for inspection
- **Flexible Input**: Use existing traces or generate new ones
- **Comprehensive Verification**: Dual processing + cross-check verification
- **Error Recovery**: Detailed error reporting and troubleshooting guidance
- **Resource Management**: Optional cleanup of intermediate files

**Status**: ✅ FULLY IMPLEMENTED AND ENHANCED

**Usage**:
```bash
# Modify file paths in script, then run:
python isolated_flat_fielding.py

# For help and detailed instructions:
python isolated_flat_fielding.py --help

# Debug specific step (stops after step 3):
# Set STOP_AT_STEP = 3 in script configuration
```

#### Key Technical Details:

**File Matching Logic**:
- Extension metadata (color, bench, side) → trace filename pattern
- Search order: primary trace directory → fallback mastercalib directory
- Robust error handling for missing files

**Pixel Map Creation**:
- Extract fiber numbers from `trace.fiberimg` (values represent fiber IDs)
- Apply B-spline fit values to corresponding trace pixels
- Set non-trace pixels to NaN for proper masking

**Normalized Image Generation**:
```python
# Step 1: Divide raw flat by pixel map (trace regions)
corrected_data = raw_flat_data / pixel_map

# Step 2: Normalize to median ≈ 1.0
normalized_data = corrected_data / median(corrected_data[trace_mask])

# Step 3: Set background to 1.0
final_data[~trace_mask] = 1.0
```

**Output Structure**:
- Primary HDU with metadata
- Extensions 1-24 following `constants.py:idx_lookup` ordering  
- Each extension includes: CHANNEL, BENCH, SIDE, BENCHSIDE, processing stats
- Trace file provenance and processing metrics in headers

**Verification Features**:
- Detailed cross-check log showing all file matches
- Processing statistics (fibers processed, median values, file locations)
- Error reporting for troubleshooting
- Output file validation and size reporting

### ✅ COMPLETED: Normalized Flat Field Creation Fix

**Legacy Implementation**: The original `create_normalized_flat_field_fits()` method has been superseded by the new workflow above.

**Status**: ✅ REPLACED with comprehensive new implementation

### ✅ COMPLETED: Multi-Extension FITS Combination for Pixel Maps

**Issue Resolved**: Individual pixel map FITS files are combined into a single multi-extension FITS file with extensions ordered to match raw science frame structure.

**Current State**: 
- Flat field processing generates 24 individual pixel map files (3 channels × 4 benches × 2 sides)  
- Files named as: `flat_pixel_map_red1A.fits`, `flat_pixel_map_green2B.fits`, etc.
- Each file contains a single 2D correction array with appropriate headers

**Implemented Solution**:

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