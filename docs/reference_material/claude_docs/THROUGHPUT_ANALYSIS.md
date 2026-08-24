# Throughput Analysis Feature

## Overview
The Throughput Analysis feature calculates relative throughput corrections and absolute flux calibrations using standard star observations. It determines system efficiency as a function of wavelength and fiber position, enabling precise photometric measurements.

## Core Functionality
- **Standard Star Processing**: Analyzes spectrophotometric standard observations
- **Relative Throughput**: Calculates fiber-to-fiber efficiency variations
- **Absolute Flux Calibration**: Converts instrumental counts to physical flux units
- **Wavelength-Dependent Response**: System efficiency across spectral range
- **Sky Background Modeling**: Separates target flux from sky emission
- **Multi-Channel Analysis**: Handles red, green, blue channels independently
- **Temporal Stability**: Tracks throughput variations over time

## Key Files
- `Flux/calcThroughput.py` - Main throughput calculation module
  - `calcThroughput()`: Primary throughput analysis function
  - Standard star spectrum processing
  - Sky background subtraction algorithms
  - Absolute flux calibration calculations

## Data Structures

### Input Requirements
- **Standard Star Extractions**: Wavelength-calibrated ExtractLlamas objects
- **Arc Calibrations**: Wavelength solutions for standard star data  
- **Reference Catalog**: Standard star flux tables (e.g., `fgd108.dat`)
- **Instrumental Parameters**: Telescope area, exposure time, detector gain

### Output Products
- **Relative Throughput Arrays**: Per-fiber efficiency corrections
- **Absolute Response Function**: System throughput vs wavelength
- **Sky Model**: Background emission templates
- **Calibration Plots**: Throughput curves and diagnostic plots

## Processing Workflow

### Standard Star Analysis
1. **Fiber Selection**: Identify brightest fibers containing standard star
2. **Sky Subtraction**: Model and remove sky background emission
3. **Flux Summation**: Combine signal from multiple fibers
4. **Wavelength Matching**: Interpolate reference spectrum to observed grid

### Throughput Calculation  
1. **Instrumental Response**: Convert counts to photons using detector parameters
2. **Atmospheric Extinction**: Apply extinction correction using standard curves
3. **Absolute Calibration**: Compare observed to reference flux
4. **System Efficiency**: Calculate overall throughput function

### Fiber Corrections
1. **Relative Efficiency**: Compare fiber responses to average
2. **Spatial Variations**: Map throughput across field of view  
3. **Wavelength Dependence**: Model spectral response variations
4. **Quality Assessment**: Validate correction accuracy

## Usage Patterns
```python
from llamas_pyjamas.Flux.calcThroughput import calcThroughput

# Calculate throughput for specific channel
calcThroughput(
    std_star_dict,    # Extracted standard star spectra  
    arc_dict,         # Wavelength calibration
    color='red'       # Channel selection
)

# Results plotted automatically
# Relative throughput corrections saved to extraction objects
```

## Pipeline Integration
Called by `reduce.py:relative_throughput()`:
- Uses wavelength-calibrated standard star extractions
- Applies wavelength solutions from arc calibration
- Generates relative throughput corrections
- Updates ExtractLlamas objects with corrections
- Feeds corrections to RSS generation and cube construction

## Calibration Standards
- **Spectrophotometric Standards**: G-dwarf stars with known flux distributions
- **Reference Catalogs**: NIST-traceable flux standards (e.g., GD108, GD153)
- **Wavelength Coverage**: 3000-10000 Angstrom typical range
- **Flux Units**: erg/s/cm²/Angstrom absolute calibration
- **Uncertainty Propagation**: Statistical and systematic error handling

## Atmospheric Effects
- **Extinction Correction**: Standard atmospheric extinction curves
- **Airmass Dependence**: Secant(z) extinction scaling
- **Site-Specific**: Magellan Observatory extinction characteristics
- **Wavelength Dependence**: λ^(-α) Rayleigh + aerosol components
- **Temporal Variations**: Night-to-night extinction changes

## Output Products
- **Throughput Plots**: System efficiency vs wavelength per channel
- **Relative Corrections**: Fiber-to-fiber normalization factors
- **Calibration Files**: Throughput lookup tables for science reductions
- **Quality Reports**: Calibration accuracy and stability metrics
- **Sky Templates**: Background emission models for subtraction

## Quality Metrics
- **Photometric Accuracy**: RMS residuals from standard flux
- **Fiber Consistency**: Throughput uniformity across field
- **Wavelength Coverage**: Calibration completeness vs wavelength
- **Temporal Stability**: Night-to-night repeatability
- **Cross-Channel Agreement**: Consistency between color channels

## Configuration Options
- **Standard Selection**: Choice of reference stars and catalogs
- **Sky Modeling**: Background subtraction methods and parameters
- **Extinction Model**: Atmospheric correction approach
- **Fiber Weighting**: Methods for combining multi-fiber data
- **Quality Thresholds**: Acceptance criteria for calibrations

## Dependencies
- Astropy (coordinate systems, table handling)
- NumPy/SciPy (numerical processing, interpolation)
- Matplotlib (diagnostic plotting)
- Pypeit (spectral fitting algorithms)

## Performance Notes
- **Processing Time**: ~5-15 minutes per standard star observation
- **Memory Usage**: ~200-500 MB for typical datasets
- **Accuracy**: ~2-5% photometric precision achievable
- **Stability**: <1% night-to-night variations typical
- **Wavelength Resolution**: Limited by spectrograph resolution

## Error Analysis
- **Statistical Errors**: Poisson noise from photon statistics  
- **Systematic Errors**: Instrumental stability, atmospheric variations
- **Calibration Errors**: Standard star flux uncertainties
- **Model Errors**: Sky subtraction and extinction correction residuals
- **Propagation**: Full error propagation through calibration chain

## Advanced Features
- **Multi-Standard Analysis**: Combining multiple standard stars
- **Color-Dependent Corrections**: Accounting for spectral type variations
- **Temporal Modeling**: Long-term throughput evolution
- **Cross-Validation**: Independent calibration verification
- **Outlier Rejection**: Robust statistical methods for bad data removal