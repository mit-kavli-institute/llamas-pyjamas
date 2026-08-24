# Wavelength Calibration Feature

## Overview
The Wavelength Calibration feature converts pixel positions to physical wavelengths using ThAr arc lamp spectra. It provides precise wavelength solutions for each fiber, enabling scientific analysis of extracted spectra.

## Core Functionality
- **Arc Lamp Processing**: Processes ThAr (Thorium-Argon) arc exposures
- **Line Identification**: Automatically identifies emission lines from reference catalogs
- **Polynomial Fitting**: Fits wavelength solutions using polynomial models
- **Solution Transfer**: Applies calibrations from arc lamps to science data
- **Quality Assessment**: Validates wavelength solution accuracy and completeness
- **X-shift Correction**: Corrects for spectral shifts between exposures
- **Throughput Calculation**: Determines relative fiber throughput from arc data

## Key Files
- `Arc/arcLlamas.py` - Main wavelength calibration code
  - `arcTransfer()`: Transfers wavelength solutions to science data
  - `shiftArcX()`: Calculates and applies X-axis shifts
  - `fiberRelativeThroughput()`: Computes throughput corrections
- `Arc/nistArc.py` - NIST line catalog and identification utilities
- `Arc/arcReidentify.py` - Arc line reidentification for multiple exposures

## Data Structures
- **Arc Wavelength Dictionary**: Contains calibrated extraction data
  - `extractions`: List of ExtractLlamas objects with wavelength information
  - `metadata`: Observational metadata for each extension
  - `primary_header`: Primary FITS header information
- **Wavelength Arrays**: Updated in ExtractLlamas objects
  - `wave`: Wavelength array (Angstroms) replacing pixel coordinates
  - `xshift`: X-axis shift corrections per fiber

## Usage Patterns
```python
# Calculate wavelength solution from arc
arc_dict = calc_wavelength_soln(arc_file, output_dir, bias=bias_frame)

# Transfer solution to science data
science_dict = load_extractions(science_extraction_file)
calibrated_data, header = correct_wavelengths(science_extraction_file, soln=arc_dict)

# Apply relative throughput corrections
relative_throughput(arc_shift_file, flat_extraction_file)
```

## Pipeline Integration
Called by `reduce.py:correct_wavelengths()`:
- Uses extracted arc lamp spectra from spectrum extraction
- Applies solutions to extracted science spectra
- References master wavelength solutions in `LUT/LLAMAS_reference_arc.pkl`
- Feeds calibrated data to RSS generation and cube construction

## Calibration Process

### Arc Processing
1. Extract 1D arc spectra using fiber traces
2. Identify emission lines from NIST catalog
3. Fit polynomial wavelength solution per fiber
4. Validate solution quality and completeness

### Science Calibration
1. Load reference arc solution or compute new one
2. Calculate spectral shifts between arc and science
3. Apply wavelength solution with shift corrections
4. Propagate uncertainties through calibration

### Throughput Correction
1. Compare arc line strengths across fibers
2. Calculate relative throughput corrections
3. Apply corrections to flat field and science data
4. Generate throughput maps for RSS products

## Output Products
- **Wavelength Solutions**: Polynomial coefficients and lookup tables
- **Calibrated Spectra**: Science data with wavelength coordinates
- **Shift Maps**: X-axis corrections for each fiber/exposure
- **Throughput Maps**: Relative efficiency corrections
- **QA Plots**: Solution quality and line identification results

## Configuration
- Reference line catalogs (NIST ThAr)
- Polynomial order for wavelength fits
- Line identification parameters (S/N thresholds, matching tolerance)
- Shift calculation methods and parameters
- Quality control thresholds

## Dependencies
- Pypeit (wavelength solution algorithms)
- Astropy (coordinate systems, FITS handling)
- NumPy/SciPy (numerical processing, polynomial fitting)
- Matplotlib (diagnostic plotting)

## Performance Notes
- Arc processing time ~5-10 minutes per exposure
- Solution transfer ~1-2 minutes per science exposure  
- Wavelength accuracy typically ~0.1-0.5 Angstrom RMS
- Line identification success rate >95% for good S/N arcs
- Memory usage ~100-500 MB depending on fiber count

## Quality Metrics
- Wavelength solution RMS residuals
- Number of lines identified per fiber
- Cross-correlation coefficients for shift measurements
- Throughput correction stability across exposures
- Solution completeness (fraction of successful fiber solutions)