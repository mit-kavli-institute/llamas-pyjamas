# Spectrum Extraction Feature

## Overview
The Spectrum Extraction feature extracts 1D spectra from 2D detector images using fiber trace information. It supports both optimal (variance-weighted) and boxcar extraction methods, with cross-talk correction and error propagation.

## Core Functionality
- **Optimal Extraction**: Variance-weighted extraction using spatial profiles
- **Boxcar Extraction**: Simple aperture summation method
- **Cross-talk Correction**: Removes signal bleed between adjacent fibers
- **Error Propagation**: Calculates uncertainties for extracted spectra
- **Multi-Channel Processing**: Handles red, green, blue channels simultaneously
- **Batch Processing**: Processes multiple exposures efficiently
- **Ray Parallelization**: Distributed processing capability (in development)

## Key Files
- `Extract/extractLlamas.py` - Main production extraction code
  - `ExtractLlamas` class: Core extraction algorithms
  - `save_extractions()`: Batch file saving utilities
  - `load_extractions()`: Batch file loading utilities
  - `ExtractLlamasRay` class: Ray-enabled parallel processing
- `GUI/guiExtract.py` - GUI wrapper for extraction workflows

## Data Structures
- **ExtractLlamas Object**: Contains extracted spectra and metadata
  - `counts`: 2D array (nfibers × nwavelengths) of extracted flux
  - `errors`: Uncertainty array matching counts
  - `wave`: Wavelength array (initially in pixels)
  - `xshift`: X-axis shift corrections
  - `relative_throughput`: Fiber-to-fiber throughput corrections
  - `fiberid`: Fiber identification numbers
  - `trace`: Reference to TraceLlamas object used
  - `hdr`: FITS header information
  - `frame`: Original 2D data array

## Usage Patterns
```python
# Basic extraction
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
extractor = ExtractLlamas(trace_obj, hdu_data, hdr, optimal=True)
# Results stored in extractor.counts, extractor.errors

# Batch processing
extraction_list = [extractor1, extractor2, ...]
save_path = save_extractions(extraction_list, save_dir=output_dir)

# Loading batch results
extractions, metadata = load_extractions(save_path)
```

## Pipeline Integration
Called by `reduce.py:run_extraction()` via `GUI/guiExtract.py:GUI_extract()`:
- Uses TraceLlamas objects from fiber tracing
- Processes bias-corrected science FITS files
- Applies flat field corrections if available
- Saves results as pickle files for wavelength calibration

## Extraction Methods

### Optimal Extraction
- Uses spatial profile information from traces
- Weights pixels by inverse variance
- Provides optimal S/N ratio for point sources
- Default method for science observations

### Boxcar Extraction  
- Simple aperture summation
- Uniform weighting within aperture
- Faster computation but lower S/N
- Used for extended sources or debugging

## Output Products
- **Extraction Files**: Pickle files containing ExtractLlamas objects
- **Batch Files**: Combined extractions from multiple exposures  
- **Metadata**: Observation information and processing parameters
- **QA Data**: Extraction quality metrics

## Configuration
- Extraction method (optimal vs boxcar) selectable
- Aperture sizes configurable
- Cross-talk correction parameters tunable
- Output directory and naming conventions configurable
- Ray parallelization settings (when available)

## Dependencies
- NumPy/SciPy (numerical processing)
- Astropy (FITS handling)
- Pypeit (core extraction algorithms)
- Ray (parallel processing - in development)
- Cloudpickle (serialization)

## Performance Notes
- Optimal extraction ~2-5x slower than boxcar but higher quality
- Memory usage ~500MB-2GB per exposure depending on fiber count
- Processing time ~10-30 minutes per exposure
- Ray parallelization will improve multi-file processing performance
- Cross-talk correction adds ~20% processing overhead

## Quality Metrics
- Signal-to-noise ratio estimates per fiber
- Extraction completeness statistics
- Cross-talk correction effectiveness
- Profile fitting residuals
- Bad pixel flagging and masking