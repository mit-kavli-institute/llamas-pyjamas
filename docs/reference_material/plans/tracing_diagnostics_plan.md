# IFU Tracing Diagnostics Coding Plan

## Overview
Create a comprehensive suite of tools to diagnose fiber tracing issues in Integral Field Unit (IFU) spectroscopy data. Tracing problems can cause systematic spatial patterns in data cubes that appear as artifacts.

## Required Libraries
```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import scipy.ndimage as ndi
from scipy import interpolate
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns
from pathlib import Path
```

## Data Structures Expected

### Input Files:
1. **Raw 2D spectrum**: FITS file with shape (y_pixels, x_pixels) - the detector image
2. **Trace solution file**: Contains fiber positions as function of wavelength
   - Could be FITS table, JSON, or custom format
   - Should contain: fiber_id, x_pixels, y_pixels, wavelength
3. **Wavelength solution**: Maps x_pixels to wavelength
4. **Fiber bundle metadata**: Physical fiber positions, bundle geometry

### Expected Data Format for Traces:
```python
# traces = {
#     'fiber_001': {
#         'x_pixels': np.array([...]),    # x positions along dispersion
#         'y_pixels': np.array([...]),    # y positions (trace curve)
#         'wavelength': np.array([...]),  # corresponding wavelengths
#         'fiber_id': 1
#     },
#     'fiber_002': { ... }
# }
```

## Core Functions to Implement

### 1. Data Loading Functions
```python
def load_raw_2d_spectrum(filename):
    """Load raw 2D spectrum from FITS file"""
    # Handle different FITS extensions
    # Return: 2D numpy array, header info

def load_trace_solution(filename, format='auto'):
    """Load fiber trace solutions from various formats"""
    # Support FITS tables, CSV, JSON formats
    # Return: dictionary structure as shown above
    # Auto-detect format if possible

def load_wavelength_solution(filename):
    """Load wavelength calibration"""
    # Return mapping from x_pixel to wavelength

def load_fiber_bundle_metadata(filename):
    """Load fiber physical positions and bundle geometry"""
    # Return: fiber positions, bundle layout info
```

### 2. Core Diagnostic Functions

#### A. Visual Trace Inspection
```python
def plot_traces_on_2d_spectrum(raw_2d, traces, title="Fiber Traces"):
    """Overlay all fiber traces on the 2D spectrum image"""
    # Create figure with raw 2D spectrum as background
    # Plot each trace as colored line
    # Add fiber ID labels at trace ends
    # Use different colors for different regions of bundle
    # Add zoom-in subplots for detailed inspection
    
def plot_individual_trace_quality(raw_2d, traces, fiber_id):
    """Detailed analysis of a single fiber trace"""
    # Show trace overlaid on 2D spectrum
    # Show cross-sections perpendicular to trace at multiple wavelengths
    # Display trace curvature and smoothness
    # Show extracted spectrum along this trace
```

#### B. Trace Accuracy Analysis
```python
def analyze_trace_straightness(traces):
    """Check how well traces follow expected smooth curves"""
    results = {}
    for fiber_id, trace in traces.items():
        # Fit polynomial (degree 2-4) to trace
        # Calculate RMS residuals from fit
        # Identify outlier points
        # Check for systematic deviations
        
        poly_coeffs = np.polyfit(trace['x_pixels'], trace['y_pixels'], deg=3)
        poly_fit = np.polyval(poly_coeffs, trace['x_pixels'])
        residuals = trace['y_pixels'] - poly_fit
        
        results[fiber_id] = {
            'rms_residual': np.std(residuals),
            'max_deviation': np.max(np.abs(residuals)),
            'systematic_trend': np.polyfit(trace['x_pixels'], residuals, 1)[0],
            'outlier_pixels': np.where(np.abs(residuals) > 3*np.std(residuals))[0]
        }
    
    return results

def check_trace_spacing_uniformity(traces):
    """Verify that trace spacing follows expected pattern"""
    # Calculate inter-fiber distances at multiple x positions
    # Check for systematic variations in spacing
    # Identify crowded or sparse regions
    # Compare to expected hexagonal/square grid pattern
```

#### C. Wavelength-Dependent Analysis
```python
def analyze_trace_wavelength_dependence(traces):
    """Check for wavelength-dependent trace shifts (atmospheric dispersion)"""
    
    # For each fiber, analyze how y-position changes with wavelength
    trace_slopes = {}
    trace_curvature = {}
    
    for fiber_id, trace in traces.items():
        # Linear fit: y_position vs wavelength
        slope, intercept = np.polyfit(trace['wavelength'], trace['y_pixels'], 1)
        
        # Quadratic fit to check for curvature
        quad_coeffs = np.polyfit(trace['wavelength'], trace['y_pixels'], 2)
        
        trace_slopes[fiber_id] = slope
        trace_curvature[fiber_id] = quad_coeffs[0]  # quadratic term
    
    # Analyze patterns across the field
    return trace_slopes, trace_curvature

def check_differential_atmospheric_refraction(traces, fiber_positions):
    """Check if trace shifts match expected atmospheric dispersion"""
    # Calculate expected DAR based on observing conditions
    # Compare measured trace slopes to theoretical predictions
    # Identify systematic deviations that suggest tracing errors
```

#### D. Cross-Fiber Correlation Analysis
```python
def analyze_trace_correlations(traces):
    """Look for systematic errors affecting multiple fibers"""
    
    # Extract trace residuals for all fibers
    all_residuals = []
    x_common = None
    
    for fiber_id, trace in traces.items():
        # Fit smooth curve and get residuals
        poly_fit = np.polyval(np.polyfit(trace['x_pixels'], trace['y_pixels'], 3), 
                             trace['x_pixels'])
        residuals = trace['y_pixels'] - poly_fit
        
        # Interpolate to common x grid if needed
        if x_common is None:
            x_common = trace['x_pixels']
        
        residuals_interp = np.interp(x_common, trace['x_pixels'], residuals)
        all_residuals.append(residuals_interp)
    
    # Calculate correlation matrix
    correlation_matrix = np.corrcoef(all_residuals)
    
    # Look for systematic patterns in correlations
    return correlation_matrix, all_residuals

def identify_systematic_trace_errors(correlation_matrix, fiber_positions):
    """Identify patterns in trace errors that suggest systematic issues"""
    # Check if nearby fibers have correlated errors
    # Look for global shifts or rotations
    # Identify potential CCD distortion effects
```

### 3. Extraction Impact Analysis
```python
def simulate_extraction_errors(raw_2d, traces, trace_errors):
    """Simulate how tracing errors affect spectral extraction"""
    
    # Create "perfect" traces and "error" traces
    # Perform extraction with both
    # Compare resulting spectra to quantify impact
    # Calculate flux losses/gains due to trace errors

def check_fiber_crosstalk(raw_2d, traces):
    """Check if poor tracing causes flux bleeding between fibers"""
    
    # Look for cases where traces are too close
    # Simulate extraction apertures and check for overlap
    # Identify potential cross-contamination
```

### 4. Quality Metrics and Reporting
```python
def calculate_trace_quality_metrics(traces, raw_2d):
    """Calculate comprehensive quality metrics"""
    
    metrics = {
        'global_metrics': {},
        'per_fiber_metrics': {},
        'spatial_patterns': {}
    }
    
    # Global metrics
    metrics['global_metrics'] = {
        'mean_trace_rms': None,
        'max_trace_deviation': None,
        'trace_spacing_uniformity': None,
        'wavelength_shift_consistency': None
    }
    
    # Per-fiber metrics
    for fiber_id in traces.keys():
        metrics['per_fiber_metrics'][fiber_id] = {
            'trace_rms': None,
            'max_deviation': None,
            'spacing_to_neighbors': None,
            'extraction_efficiency': None
        }
    
    return metrics

def generate_trace_quality_report(metrics, output_dir):
    """Generate comprehensive HTML/PDF report"""
    # Create plots and tables summarizing all diagnostics
    # Include recommendations for trace solution improvements
    # Flag problematic fibers or wavelength ranges
```

### 5. Main Diagnostic Pipeline
```python
def run_trace_diagnostics(raw_2d_file, trace_file, output_dir, 
                         fiber_metadata_file=None, wavelength_file=None):
    """Main function to run all trace diagnostics"""
    
    # Load all data
    raw_2d = load_raw_2d_spectrum(raw_2d_file)
    traces = load_trace_solution(trace_file)
    
    if wavelength_file:
        wavelength_solution = load_wavelength_solution(wavelength_file)
        # Apply wavelength solution to traces
    
    if fiber_metadata_file:
        fiber_positions = load_fiber_bundle_metadata(fiber_metadata_file)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Run diagnostics
    print("1. Analyzing trace straightness...")
    straightness_results = analyze_trace_straightness(traces)
    
    print("2. Checking trace spacing...")
    spacing_results = check_trace_spacing_uniformity(traces)
    
    print("3. Analyzing wavelength dependence...")
    slopes, curvature = analyze_trace_wavelength_dependence(traces)
    
    print("4. Checking inter-fiber correlations...")
    corr_matrix, residuals = analyze_trace_correlations(traces)
    
    print("5. Generating visualizations...")
    plot_traces_on_2d_spectrum(raw_2d, traces)
    plt.savefig(output_path / "traces_overview.png", dpi=300, bbox_inches='tight')
    
    # More plotting functions...
    
    print("6. Calculating quality metrics...")
    metrics = calculate_trace_quality_metrics(traces, raw_2d)
    
    print("7. Generating report...")
    generate_trace_quality_report(metrics, output_path)
    
    return metrics

# Example usage
if __name__ == "__main__":
    # Example file paths - adjust for your data
    raw_2d_file = "path/to/raw_2d_spectrum.fits"
    trace_file = "path/to/trace_solution.fits"
    output_dir = "trace_diagnostics_output"
    
    metrics = run_trace_diagnostics(raw_2d_file, trace_file, output_dir)
```

## Implementation Notes

1. **Error Handling**: Add robust error handling for different file formats and missing data
2. **Performance**: For large datasets, consider chunking or parallel processing
3. **Flexibility**: Make functions work with different IFU instruments (MaNGA, MUSE, etc.)
4. **Visualization**: Use interactive plots where helpful (plotly, bokeh)
5. **Configuration**: Allow customization of quality thresholds and parameters

## Expected Outputs

1. **Diagnostic Plots**:
   - Traces overlaid on 2D spectrum
   - Trace residual maps
   - Wavelength-dependent trace shifts
   - Inter-fiber correlation heatmaps

2. **Quality Metrics**:
   - RMS trace deviations
   - Systematic error measurements
   - Fiber-to-fiber consistency metrics

3. **Problem Identification**:
   - List of problematic fibers
   - Wavelength ranges with issues
   - Recommendations for improvement

## Success Criteria

The diagnostics should be able to identify:
- Systematic tracing errors affecting multiple fibers
- Individual fibers with poor trace solutions
- Wavelength-dependent issues (DAR correction problems)
- Impact of tracing errors on spectral extraction quality
- Recommendations for trace solution improvements

This comprehensive approach will help identify the root cause of spatial patterns in IFU data cubes.