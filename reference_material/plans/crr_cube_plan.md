# CRR Cube Construction Implementation Plan

## Overview
Implement the Covariance-regularized Reconstruction (CRR) method from Liu et al. (2020) to create data cubes from RSS (Row-Stacked Spectra) data, with Ray parallelization for cluster compatibility.

## File Structure
```
crr_cube_constructor.py          # Main implementation
crr_kernels.py                   # Kernel generation utilities  
crr_weights.py                   # Weight matrix computation
crr_parallel.py                  # Ray parallelization wrapper
test_crr_cube.py                 # Testing and validation
config/crr_config.yaml           # Configuration parameters
```

## Step-by-Step Implementation Plan

### Step 1: Core Data Structures and Configuration
**File: `crr_cube_constructor.py`**
- Create `CRRCubeConfig` dataclass for parameters:
  - `pixel_scale`: Output pixel scale (default 0.75 arcsec/pixel per paper)
  - `regularization_lambda`: Regularization parameter (default 1e-3)
  - `kernel_radius_limit`: Kernel truncation radius (default 4 arcsec)
  - `reconstruction_radius_limit`: Pixel inclusion limit (default 1.6 arcsec)
  - `use_sky_subtraction`: Boolean flag for sky subtraction
  - `output_wavelength_grid`: Wavelength sampling for output cube
  
- Create `RSSData` dataclass to hold:
  - `flux`: 2D array (n_fibers, n_wavelengths)
  - `ivar`: 2D inverse variance array
  - `mask`: 2D mask array
  - `fiber_positions`: (n_fibers, 2) fiber coordinates for each wavelength
  - `wavelength`: 1D wavelength array
  - `metadata`: Dictionary with observation info

### Step 2: Kernel Generation Module
**File: `crr_kernels.py`**
- Implement `double_gaussian_kernel()`:
  - Takes seeing FWHM, fiber diameter (2 arcsec), position grid
  - Uses paper's parameters: σ₂/σ₁ = 2, A₂/A₁ = 1/9
  - Convolves atmospheric seeing with fiber top-hat profile
  
- Implement `wavelength_dependent_seeing()`:
  - Scale seeing as λ^(-1/5) from reference wavelength
  
- Create `build_kernel_matrix()`:
  - For each fiber-wavelength combination, compute kernel response
  - Build matrix A[i,j] = K(x_fiber - x_pixel, y_fiber - y_pixel)
  - Truncate kernel beyond `kernel_radius_limit`

### Step 3: Weight Matrix Computation
**File: `crr_weights.py`**
- Implement `compute_crr_weights()` following Equations 10-24 from paper:
  
  1. **SVD Decomposition** (Eq. 10):
     ```python
     # Build modified variance matrix N_tilde (unity where valid, zero where masked)
     N_tilde = build_variance_matrix(mask)
     
     # SVD of N_tilde^(-1/2) * A
     U, S, Vt = np.linalg.svd(N_tilde_inv_sqrt @ A_matrix)
     ```
  
  2. **Regularization** (Eq. 20):
     ```python
     S_star = S / (S**2 + lambda**2)
     ```
  
  3. **Flux Conservation Matrix R** (Eq. 14):
     ```python
     Q = Vt.T @ np.diag(S_star) @ U.T
     R = normalize_rows_for_flux_conservation(Q)
     ```
  
  4. **Final Weight Matrix** (Eq. 23):
     ```python
     W = R @ Vt.T @ np.diag(S_star) @ U.T @ N_tilde_inv_sqrt
     ```

- Implement `compute_shepard_weights()` for comparison:
  - Gaussian weights based on distance (Eq. 2)
  - Flux conservation normalization (Eq. 3)

### Step 4: Cube Reconstruction Engine
**File: `crr_cube_constructor.py` (continued)**
- Implement `CRRCubeConstructor` class:

  ```python
  class CRRCubeConstructor:
      def __init__(self, config: CRRCubeConfig):
          self.config = config
          
      def setup_output_grid(self, fiber_positions):
          """Define rectangular output pixel grid"""
          
      def process_wavelength_slice(self, wavelength_idx, rss_data):
          """Process single wavelength slice"""
          # 1. Extract fiber positions for this wavelength
          # 2. Build kernel matrix for this seeing condition  
          # 3. Compute CRR weights
          # 4. Apply weights: G = W @ f (Eq. 22)
          # 5. Compute covariance matrix: C_G = W @ N @ W.T (Eq. 24)
          # 6. Create quality masks (LOWCOV, NOCOV)
          
      def reconstruct_cube(self, rss_data):
          """Main reconstruction method"""
  ```

### Step 5: Sky Subtraction Integration
**File: `crr_cube_constructor.py` (continued)**
- Add `apply_sky_subtraction()` method:
  - If `use_sky_subtraction=True`: Apply existing sky subtraction to RSS data
  - If `use_sky_subtraction=False`: Skip this step with warning
  - Ensure sky subtraction preserves fiber positioning metadata

### Step 6: Ray Parallelization
**File: `crr_parallel.py`**
- Implement Ray-based parallelization:

  ```python
  @ray.remote
  class CRRWorker:
      def __init__(self, config):
          self.constructor = CRRCubeConstructor(config)
          
      def process_wavelength_batch(self, wavelength_indices, rss_data):
          """Process batch of wavelength slices"""
          
  def parallel_cube_construction(rss_data, config, n_workers=None):
      """Distribute wavelength processing across Ray workers"""
      # 1. Initialize Ray workers
      # 2. Split wavelength range into batches
      # 3. Distribute work across workers
      # 4. Collect and combine results
      # 5. Handle memory management for large cubes
  ```

### Step 7: Output Data Cube Format
**File: `crr_cube_constructor.py` (continued)**
- Create `CRRDataCube` class:
  - `flux`: 3D array (n_x, n_y, n_wavelength)
  - `ivar`: 3D inverse variance array
  - `mask`: 3D mask array with NOCOV, LOWCOV flags
  - `covariance_diagonal`: Diagonal elements of covariance matrix
  - `psf_fwhm`: Wavelength-dependent PSF FWHM measurements
  - `metadata`: Reconstruction parameters and quality metrics

### Step 8: Quality Assessment Tools
**File: `crr_cube_constructor.py` (continued)**
- Implement `measure_psf_quality()`:
  - Inject simulated point sources at various positions
  - Measure FWHM and Strehl ratio across field
  - Compare with kernel expectations
  
- Implement `measure_covariance_quality()`:
  - Compute correlation coefficients between neighboring pixels
  - Verify near-diagonal covariance matrix

### Step 9: Configuration and CLI Interface
**File: `config/crr_config.yaml`**
```yaml
cube_reconstruction:
  pixel_scale: 0.75  # arcsec/pixel
  regularization_lambda: 1e-3
  kernel_radius_limit: 4.0  # arcsec
  reconstruction_radius_limit: 1.6  # arcsec
  use_sky_subtraction: false
  
parallel_processing:
  n_workers: null  # auto-detect
  wavelength_batch_size: 50
  memory_limit_gb: 8
  
output:
  format: "fits"
  compression: true
  quality_assessment: true
```

**Add CLI interface to main file:**
```python
def main():
    parser = argparse.ArgumentParser(description='CRR Cube Construction')
    parser.add_argument('rss_file', help='Input RSS FITS file')
    parser.add_argument('--config', default='config/crr_config.yaml')
    parser.add_argument('--output', help='Output cube file')
    parser.add_argument('--method', choices=['crr', 'shepard'], default='crr')
```

### Step 10: Testing and Validation
**File: `test_crr_cube.py`**
- Unit tests for each component
- Integration test with simulated data
- Comparison with Shepard's method
- Performance benchmarking
- Memory usage profiling

## Implementation Priority Order
1. **Core data structures and kernel generation** (Steps 1-2)
2. **Weight matrix computation** (Step 3) 
3. **Basic reconstruction without parallelization** (Step 4)
4. **Sky subtraction integration** (Step 5)
5. **Ray parallelization** (Step 6)
6. **Output format and quality tools** (Steps 7-8)
7. **Configuration and CLI** (Step 9)
8. **Testing and optimization** (Step 10)

## Integration with Existing Pipeline
- Modify existing RSS output to include fiber position metadata for each wavelength
- Ensure RSS format includes proper error propagation and masking
- Create adapter functions to convert existing data formats to `RSSData` structure
- Add CRR option to main pipeline configuration

## Cluster Compatibility Requirements
- Ray cluster initialization with proper resource management
- Configurable memory limits per worker
- Checkpoint/resume capability for long-running jobs
- Progress monitoring and logging
- Error handling and recovery for worker failures

## Performance Considerations
- Use sparse matrices where appropriate for large fiber bundles
- Implement memory-mapped arrays for large datasets
- Cache kernel computations when possible
- Profile SVD performance and consider alternative decompositions
- Optimize wavelength batching based on available memory

This plan provides a complete roadmap for implementing the CRR method while maintaining compatibility with your existing pipeline and cluster computing requirements.