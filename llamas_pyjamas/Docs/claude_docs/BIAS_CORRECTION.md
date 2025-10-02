# Bias Correction Feature

## Overview
The Bias Correction feature removes electronic offset (bias level) from CCD readout, providing the fundamental first step in astronomical data reduction. It processes bias frames to create master calibrations and applies corrections to all subsequent data.

## Core Functionality
- **Master Bias Creation**: Combines multiple bias exposures into stable calibration
- **Statistical Analysis**: Computes bias level statistics and stability metrics
- **Overscan Processing**: Handles overscan region correction when available
- **Bias Subtraction**: Applies bias correction to science, flat, and arc data
- **Quality Assessment**: Validates bias frame quality and consistency
- **Multi-Channel Support**: Processes red, green, blue channels independently

## Key Files
- `Bias/llamasBias.py` - Main bias processing module
  - `BiasLlamas` class: Core bias handling functionality
  - `create_master_bias()`: Combines multiple bias frames
  - `apply_bias_correction()`: Subtracts bias from data
  - Statistical analysis and validation tools

## Data Structures
- **Master Bias Frame**: 2D array with combined bias pattern
- **Bias Statistics**: Dictionary containing:
  - `mean_level`: Average bias level across detector
  - `std_dev`: Pixel-to-pixel noise characteristics  
  - `overscan_stats`: Overscan region statistics
  - `stability_metrics`: Frame-to-frame consistency measures
- **Corrected Data**: Bias-subtracted arrays ready for further processing

## Usage Patterns
```python
from llamas_pyjamas.Bias.llamasBias import BiasLlamas

# Create bias processor
bias_processor = BiasLlamas(bias_file_list)

# Generate master bias
master_bias = bias_processor.create_master_bias()

# Apply correction to data
corrected_data = bias_processor.apply_bias_correction(raw_data)

# Get bias statistics
stats = bias_processor.get_bias_statistics()
```

## Pipeline Integration
- **First Processing Step**: Applied before all other calibrations
- **Universal Application**: Used for science, flat, arc, and standard star data
- **Configuration Support**: Bias files specified in pipeline config files
- **Fallback Handling**: Uses master calibration files when user bias unavailable
- **Multi-File Support**: Can process single bias or combine multiple exposures

## Processing Workflow

### Master Bias Creation
1. Load and validate individual bias frames
2. Check header consistency and exposure parameters
3. Combine frames using median or mean algorithm
4. Compute pixel-to-pixel statistics
5. Save master bias for pipeline use

### Bias Correction Application
1. Load appropriate master bias for detector/channel
2. Subtract bias pattern from raw data
3. Handle overscan regions if present
4. Propagate uncertainties appropriately
5. Update FITS headers with correction information

## Output Products
- **Master Bias Frames**: Combined calibration stored in `mastercalib/`
- **Corrected Data**: Bias-subtracted arrays
- **Statistics Files**: Bias level and noise characteristics
- **QA Products**: Diagnostic plots showing bias stability
- **Processing Logs**: Detailed correction history

## Configuration
- Input bias file paths in pipeline config
- Combination method selection (median/mean)  
- Quality control thresholds
- Overscan region definitions
- Output file naming conventions
- Statistical analysis parameters

## Dependencies
- Astropy (FITS I/O and header handling)
- NumPy (array operations and statistics)
- Matplotlib (diagnostic plotting)
- SciPy (robust statistical methods)

## Performance Notes
- Processing time ~1-5 minutes for typical bias sets
- Memory usage ~200-500 MB for master bias creation
- Disk space ~100-500 MB per master bias frame
- Very fast application to individual exposures (~seconds)
- Minimal computational overhead in pipeline

## Quality Metrics
- **Bias Level Stability**: Frame-to-frame consistency
- **Readout Noise**: Pixel-to-pixel variation after bias subtraction
- **Pattern Recognition**: Detection of systematic bias structures
- **Overscan Consistency**: Agreement between bias frames and overscan
- **Temporal Stability**: Bias level changes over observing runs

## Error Handling
- **Missing Bias Files**: Falls back to master calibration library
- **Header Inconsistencies**: Validates observation parameters
- **Bad Pixels**: Identifies and masks problematic pixels
- **File Format Issues**: Robust FITS file validation
- **Memory Management**: Efficient handling of large detector arrays