# Cube Construction Feature

## Overview
The Cube Construction feature creates 3D data cubes (x, y, wavelength) from Row-Stacked Spectra (RSS) files. It offers two reconstruction methods: traditional Shepard's method and the advanced Covariance-regularized Reconstruction (CRR) method for improved spatial resolution and uncertainty propagation.

## Core Functionality
- **CRR Reconstruction**: Advanced method following Liu et al. (2020) with optimal spatial resolution
- **Traditional Reconstruction**: Shepard's method for comparison and compatibility
- **Double Gaussian Kernels**: Atmospheric seeing convolved with fiber aperture functions
- **Wavelength-dependent Seeing**: λ^(-1/5) scaling for atmospheric turbulence
- **Ray Parallelization**: Distributed processing for large datasets
- **Quality Assessment**: Coverage, PSF, and covariance diagnostics
- **Multi-Channel Support**: Handles red, green, blue channels and combinations

## Key Files
- `Cube/crr_cube_constructor.py` - Main CRR implementation
  - `CRRCubeConstructor` class: Core reconstruction algorithms
  - `CRRCubeConfig` class: Configuration management
  - `RSSData` class: Input data structure
  - `CRRDataCube` class: Output cube structure
- `Cube/crr_kernels.py` - Kernel generation utilities
- `Cube/crr_weights.py` - Weight matrix computation and SVD solving
- `Cube/crr_parallel.py` - Ray parallelization wrapper
- `Cube/crr_cli.py` - Command-line interface
- `Cube/cubeConstruct.py` - Legacy cube construction (traditional methods)
- `Cube/rss_to_crr_adapter.py` - RSS format conversion utilities

## Data Structures

### Input: RSSData
- `flux`: 2D array (nfibers × nwavelengths) of extracted spectra
- `ivar`: Inverse variance array for uncertainty propagation
- `mask`: Boolean validity mask
- `fiber_positions`: Spatial coordinates (arcsec) for each fiber
- `wavelength`: 1D wavelength array (Angstroms)
- `seeing_fwhm`: Atmospheric seeing FWHM (arcsec)
- `metadata`: Additional observational information

### Output: CRRDataCube  
- `flux`: 3D flux cube (nx × ny × nwavelength)
- `ivar`: 3D inverse variance cube
- `mask`: 3D boolean mask
- `covariance_diagonal`: Diagonal covariance elements
- `flags`: Quality flags (NOCOV, LOWCOV)
- `psf_fwhm`: PSF FWHM per wavelength slice
- `quality_metrics`: Coverage and reconstruction statistics

## Usage Patterns
```python
# Basic CRR reconstruction
from llamas_pyjamas.Cube import CRRCubeConstructor, CRRCubeConfig

config = CRRCubeConfig(
    pixel_scale=0.75,           # arcsec/pixel
    regularization_lambda=1e-3, # SVD regularization
    kernel_radius_limit=4.0     # kernel truncation
)

constructor = CRRCubeConstructor(config)
cube = constructor.reconstruct_cube(rss_data)
cube.save_to_fits("output_cube.fits")

# Command line usage
python Cube/crr_cli.py input_rss.fits --output cube.fits --parallel --workers 8
```

## Pipeline Integration
Called by `reduce.py:construct_cube()`:
- Takes RSS files from RSS generation step
- Converts RSS format to CRR-compatible structures
- Applies CRR or traditional reconstruction
- Saves cubes as multi-extension FITS files
- Generates quality assessment reports

## Reconstruction Methods

### CRR Method (Recommended)
- **Mathematical Foundation**: Regularized least-squares with covariance propagation
- **Kernel Function**: Double Gaussian (seeing ⊗ fiber aperture)
- **Regularization**: SVD with λ parameter to handle ill-conditioned matrices
- **Flux Conservation**: Exact flux preservation across reconstruction
- **Uncertainty Propagation**: Full covariance matrix computation
- **Resolution**: Near-optimal spatial resolution recovery

### Traditional Methods
- **Shepard's Method**: Inverse distance weighting
- **Gaussian Smoothing**: Simple convolution approach
- **Linear Interpolation**: Grid-based interpolation
- **Comparison Mode**: Side-by-side quality assessment

## Configuration Options
- **Spatial Sampling**: Output pixel scale (arcsec/pixel)
- **Regularization**: λ parameter for SVD conditioning
- **Kernel Parameters**: Seeing FWHM, fiber diameter, truncation radius
- **Grid Definition**: Field size, center coordinates
- **Quality Control**: Coverage thresholds, flagging criteria
- **Parallelization**: Worker count, memory limits, batch sizes

## Output Products
- **Data Cubes**: Multi-extension FITS with flux, variance, mask, flags
  - **CRR cubes**: Named with `_crr_cube.fits` suffix (e.g., `galaxy_123_crr_cube.fits`)
  - **Parallel CRR cubes**: Named with `_crr_cube_parallel.fits` suffix  
  - **Traditional cubes**: Named with `_{channel}.fits` suffix (e.g., `galaxy_123_red.fits`)
  - **Shepard cubes**: Named with `_shepard_cube.fits` suffix
- **PSF Information**: Wavelength-dependent PSF characterization
- **Quality Metrics**: Coverage maps, reconstruction statistics
- **Covariance Data**: Uncertainty correlation information (optional)
- **Comparison Cubes**: Traditional method results for validation

### FITS Header Information
CRR cubes include specific header keywords for identification:
- `METHOD = 'CRR'` with comment 'Covariance-regularized Reconstruction'
- `CRRLAMBDA`: Regularization parameter used
- `PIXSCALE`: Output pixel scale (arcsec/pixel)
- `CUBEREF = 'Liu et al. (2020)'`: Reference for CRR method

## Dependencies
- Ray (parallel processing)
- NumPy/SciPy (numerical algorithms, linear algebra)
- Astropy (FITS I/O, WCS coordinate systems)
- Sparse matrices (for large fiber bundles)
- YAML (configuration files)

## Performance Notes
- **Memory Usage**: ~4 GB per 100 fibers × 50k pixels × 1k wavelengths
- **Compute Time**: ~1 hour for typical LLAMAS cube on 8 cores  
- **Scalability**: Ray parallelization across wavelength dimension
- **Optimization**: Sparse matrix operations for large datasets
- **Batch Processing**: Wavelength slices processed in configurable batches

## Quality Assessment
- **Coverage Analysis**: Fraction of pixels with valid data
- **PSF Characterization**: Spatial resolution per wavelength
- **Flux Conservation**: Total flux preservation validation
- **Covariance Quality**: Uncertainty correlation structure
- **Comparison Metrics**: CRR vs traditional method performance
- **Artifact Detection**: Systematic reconstruction errors

## Advanced Features
- **Wavelength-dependent Seeing**: Atmospheric turbulence scaling
- **Adaptive Regularization**: λ parameter optimization per wavelength
- **Multi-Channel Combination**: Joint reconstruction across color channels
- **Custom Kernels**: User-defined spatial response functions
- **Quality Flagging**: Automated bad pixel and low-coverage identification