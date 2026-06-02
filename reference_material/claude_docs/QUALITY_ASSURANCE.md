# Quality Assurance Feature

## Overview
The Quality Assurance (QA) feature provides comprehensive visualization and analysis tools for assessing data quality throughout the LLAMAS pipeline. It generates diagnostic plots, statistical summaries, and validation reports to ensure optimal data reduction quality.

## Core Functionality
- **DS9 Integration**: Direct visualization of 2D images with SAMP protocol support
- **Fiber Trace QA**: Validation plots for trace quality and completeness
- **Template Comparison**: Analysis of arc lamp templates and master references
- **Cross-Channel Analysis**: Comparison plots between different color channels
- **PSF Analysis**: Point spread function characterization and fitting
- **Extraction Diagnostics**: Validation of spectrum extraction quality
- **Statistical Reporting**: Comprehensive quality metrics and summaries

## Key Files
- `QA/llamasQA.py` - Main QA module with comprehensive plotting functions
  - `plot_ds9()`: DS9 visualization with SAMP integration
  - `plot_trace_qa()`: Fiber tracing quality assessment
  - `plot_comb_template()`: Arc template analysis
  - `plot_master_comb()`: Master template comparison
  - `compare_combs()`: Cross-channel template comparison
  - `plot_fiber_trace()`: Individual fiber trace validation
  - `analyze_psf_width()`: PSF width analysis and fitting

## Visualization Capabilities

### DS9 Integration
- **Direct Display**: Send arrays to DS9 via SAMP protocol or subprocess
- **FITS Handling**: Convert numpy arrays to FITS for visualization
- **Interactive Analysis**: Full DS9 functionality for detailed inspection
- **Automatic Scaling**: Optimal display scaling and contrast

### Trace Quality Assessment
- **Individual Fiber Profiles**: 4×4 grid showing first 16 fiber profiles
- **Trace Overlay**: Traces overlaid on raw 2D data with color coding
- **Profile Fitting**: B-spline profile fits and residuals
- **Completeness Statistics**: Fiber detection success rates

### Template Analysis
- **Arc Line Identification**: Peak detection and line identification plots
- **Master Template Display**: Reference templates from lookup tables
- **Cross-Correlation**: Template matching quality assessment
- **Channel Comparison**: Side-by-side analysis of different channels

## Advanced QA Tools

### PSF Characterization
```python
# Analyze PSF width across detector
results = analyze_psf_width(hdu_data, trace, fiber_id, n_samples=5)
# Returns FWHM measurements, mask widths, fitting parameters
```

### Extraction Validation
- **Fiber Masks**: Visualization of extraction apertures
- **Masked Data**: What data is actually extracted per fiber
- **Residual Analysis**: What light is NOT being extracted
- **Cross-sections**: 1D cuts showing extraction profiles

### Coverage Analysis
- **Spatial Coverage**: Which areas of detector are utilized
- **Wavelength Coverage**: Spectral range validation per fiber
- **Quality Maps**: Pixel-level quality indicators

## Usage Patterns
```python
from llamas_pyjamas.QA.llamasQA import *

# Display image in DS9
plot_ds9(image_array, samp=True)

# Generate trace QA plots
plot_trace_qa(trace_object, save_dir="qa_output/")

# Compare arc templates between channels
compare_combs("red", "1A", "green", "1A")

# Analyze individual fiber
plot_fiber_trace_with_residuals(hdu_data, trace, fiber_id=10)
```

## Pipeline Integration
- **Automatic QA**: Called at key pipeline stages for validation
- **Manual Inspection**: Interactive tools for detailed analysis
- **Batch Processing**: QA generation for multiple exposures
- **Report Generation**: Summary statistics and plots
- **Configuration Driven**: QA parameters set via configuration files

## Output Products
- **Diagnostic Plots**: PNG/PDF plots for offline analysis
- **Interactive Displays**: DS9 sessions for detailed inspection  
- **Statistical Reports**: Numerical quality metrics
- **Comparison Studies**: Cross-exposure and cross-channel analysis
- **Web Reports**: HTML summaries with embedded plots (future feature)

## Quality Metrics Tracked
- **Trace Quality**: Number of fibers detected, trace RMS, profile fitting residuals
- **Extraction Efficiency**: Signal extraction completeness, cross-talk levels
- **Wavelength Accuracy**: Arc line RMS residuals, solution completeness
- **Photometric Stability**: Fiber-to-fiber variations, temporal stability
- **Data Completeness**: Pixel masking statistics, coverage maps

## Visualization Types
- **2D Images**: Raw data, processed data, difference images, masks
- **1D Spectra**: Individual fiber spectra, template comparisons
- **Profile Plots**: Spatial profiles, PSF characterization
- **Statistical Plots**: Histograms, scatter plots, correlation analysis
- **Overlay Plots**: Traces on images, apertures on data

## Configuration Options
- **Plot Styling**: Colors, markers, line styles, fonts
- **Output Formats**: PNG, PDF, EPS for publications
- **Display Options**: Figure sizes, DPI settings, axis limits
- **DS9 Integration**: SAMP vs subprocess, display options
- **Batch Settings**: Automatic vs manual QA generation

## Dependencies
- Matplotlib (comprehensive plotting)
- Astropy (FITS visualization, SAMP integration)
- NumPy/SciPy (statistical analysis, fitting)
- DS9 (external visualization tool)
- SAMPy (SAMP protocol integration)

## Performance Notes
- **Plot Generation**: ~10-60 seconds per complex plot
- **DS9 Display**: Near-instantaneous for typical image sizes
- **Memory Usage**: ~100-500 MB for plot generation
- **Batch Processing**: Scales well with parallel plot generation
- **Interactive Response**: Sub-second for most interactive operations

## Advanced Features
- **Publication Quality**: Matplotlib plots suitable for journals
- **Color Schemes**: Colorblind-friendly palettes, grayscale compatibility
- **Multi-Panel Layouts**: Complex subplot arrangements
- **Animation Support**: Time-series and wavelength animations
- **3D Visualization**: Limited 3D plotting for cube data
- **Statistical Analysis**: Robust statistics, outlier detection

## Error Handling
- **Missing Data**: Graceful handling of incomplete datasets
- **Display Issues**: Fallback options when DS9 unavailable
- **File I/O**: Robust FITS file handling and validation
- **Memory Management**: Efficient handling of large datasets
- **Format Support**: Multiple image and plot formats