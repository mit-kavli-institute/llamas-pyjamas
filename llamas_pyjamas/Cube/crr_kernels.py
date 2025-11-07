"""
CRR Kernel Generation Module

Implementation of kernel generation utilities for the Covariance-regularized 
Reconstruction (CRR) method following Liu et al. (2020). This module handles
the creation of spatial kernels that combine atmospheric seeing with fiber
top-hat profiles.

The kernels use a double Gaussian model to represent the convolution of
atmospheric seeing (Gaussian PSF) with the fiber aperture (top-hat function),
following the methodology in Section 3.2 of Liu et al. (2020).

Functions:
    double_gaussian_kernel: Create double Gaussian kernel model
    wavelength_dependent_seeing: Calculate wavelength-dependent seeing
    build_kernel_matrix: Build kernel response matrix A
    fiber_top_hat_profile: Generate fiber top-hat aperture function

Author: Generated for LLAMAS Pipeline  
Date: September 2025
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging
from scipy.special import j1  # Bessel function for top-hat convolution
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
import warnings

from llamas_pyjamas.Utils.utils import setup_logger


def fiber_top_hat_profile(r: np.ndarray, fiber_diameter: float) -> np.ndarray:
    """Generate fiber top-hat aperture function.
    
    Creates a top-hat (uniform) aperture function representing the fiber
    cross-section. For circular fibers, this is 1 within the fiber radius
    and 0 outside.
    
    Args:
        r: Radial distance array in arcsec
        fiber_diameter: Fiber diameter in arcsec
        
    Returns:
        Top-hat aperture values (0 or 1)
    """
    fiber_radius = fiber_diameter / 2.0
    return (r <= fiber_radius).astype(float)


def double_gaussian_kernel(x: np.ndarray, y: np.ndarray, 
                          seeing_fwhm: float, fiber_diameter: float) -> np.ndarray:
    """Create double Gaussian kernel model for CRR reconstruction.
    
    Following Liu et al. (2020) Section 3.2, this implements the double Gaussian
    model that approximates the convolution of atmospheric seeing (Gaussian PSF)
    with the fiber aperture (top-hat function).
    
    The model uses:
    - σ₂/σ₁ = 2 (width ratio between components)
    - A₂/A₁ = 1/9 (amplitude ratio between components)
    - Normalization to preserve flux
    
    Args:
        x: X coordinate grid in arcsec
        y: Y coordinate grid in arcsec  
        seeing_fwhm: Atmospheric seeing FWHM in arcsec
        fiber_diameter: Fiber diameter in arcsec
        
    Returns:
        2D kernel array normalized to unit sum
    """
    # Convert FWHM to sigma for Gaussian
    sigma_1 = seeing_fwhm / (2 * np.sqrt(2 * np.log(2)))
    
    # Double Gaussian parameters from Liu et al. (2020)
    sigma_2 = 2.0 * sigma_1  # Broader component
    A_ratio = 1.0 / 9.0      # Amplitude ratio A₂/A₁
    
    # Calculate radial distance
    r_squared = x**2 + y**2
    
    # First Gaussian component (narrow)
    gaussian_1 = np.exp(-0.5 * r_squared / sigma_1**2) / (2 * np.pi * sigma_1**2)
    
    # Second Gaussian component (broad)  
    gaussian_2 = np.exp(-0.5 * r_squared / sigma_2**2) / (2 * np.pi * sigma_2**2)
    
    # Combine with amplitude weights
    # Normalization ensures A₁ + A₂ = 1
    A_1 = 1.0 / (1.0 + A_ratio)
    A_2 = A_ratio / (1.0 + A_ratio)
    
    kernel = A_1 * gaussian_1 + A_2 * gaussian_2
    
    # Ensure unit normalization (flux conservation)
    kernel_sum = np.sum(kernel)
    if kernel_sum > 0:
        kernel /= kernel_sum
    else:
        warnings.warn("Kernel sum is zero - check input parameters")
        
    return kernel


def wavelength_dependent_seeing(wavelength: np.ndarray, 
                              reference_wavelength: float,
                              reference_seeing: float,
                              power_law_index: float = -0.2) -> np.ndarray:
    """Calculate wavelength-dependent seeing FWHM.
    
    Atmospheric seeing follows a power law with wavelength:
    FWHM(λ) = FWHM_ref * (λ / λ_ref)^α
    
    where α ≈ -1/5 = -0.2 for typical atmospheric turbulence.
    
    Args:
        wavelength: Wavelength array in Angstroms
        reference_wavelength: Reference wavelength in Angstroms  
        reference_seeing: Seeing FWHM at reference wavelength in arcsec
        power_law_index: Power law index (default -0.2)
        
    Returns:
        Seeing FWHM array at each wavelength in arcsec
    """
    wavelength_ratio = wavelength / reference_wavelength
    seeing_fwhm = reference_seeing * (wavelength_ratio ** power_law_index)
    
    return seeing_fwhm


def build_kernel_matrix(fiber_positions: np.ndarray, 
                       output_grid_x: np.ndarray,
                       output_grid_y: np.ndarray,
                       seeing_fwhm: float,
                       fiber_diameter: float,
                       kernel_radius_limit: float,
                       sparse_threshold: float = 1e-6) -> np.ndarray:
    """Build kernel response matrix A for CRR reconstruction.
    
    Creates the matrix A where A[i,j] represents the response of output pixel i
    to fiber j. This implements Equation 8 from Liu et al. (2020):
    
    A[i,j] = K(x_i - x_j, y_i - y_j)
    
    where K is the double Gaussian kernel and (x_i, y_i) are pixel coordinates
    and (x_j, y_j) are fiber positions.
    
    Args:
        fiber_positions: Fiber positions array (n_fibers, 2) in arcsec
        output_grid_x: Output pixel x coordinates in arcsec
        output_grid_y: Output pixel y coordinates in arcsec
        seeing_fwhm: Atmospheric seeing FWHM in arcsec
        fiber_diameter: Fiber diameter in arcsec
        kernel_radius_limit: Truncation radius for kernel in arcsec
        sparse_threshold: Minimum value to include in sparse matrix
        
    Returns:
        Kernel matrix A with shape (n_pixels, n_fibers)
    """
    logger = setup_logger(__name__)
    
    n_fibers = fiber_positions.shape[0]
    n_x, n_y = len(output_grid_x), len(output_grid_y)
    n_pixels = n_x * n_y
    
    logger.info(f"Building kernel matrix: {n_pixels} pixels x {n_fibers} fibers")
    logger.info(f"Seeing FWHM: {seeing_fwhm:.3f} arcsec")
    logger.info(f"Kernel truncation radius: {kernel_radius_limit:.1f} arcsec")
    
    # Create coordinate meshgrids
    X_grid, Y_grid = np.meshgrid(output_grid_x, output_grid_y, indexing='ij')
    
    # Initialize kernel matrix
    A_matrix = np.zeros((n_pixels, n_fibers), dtype=np.float32)
    
    # Pre-calculate kernel size for efficiency
    pixel_scale = np.abs(output_grid_x[1] - output_grid_x[0])  # Assume square pixels
    kernel_pixels = int(np.ceil(kernel_radius_limit / pixel_scale))
    kernel_size = 2 * kernel_pixels + 1
    
    # Create coordinate arrays for kernel evaluation
    kernel_x = np.linspace(-kernel_radius_limit, kernel_radius_limit, kernel_size)
    kernel_y = np.linspace(-kernel_radius_limit, kernel_radius_limit, kernel_size)
    K_x, K_y = np.meshgrid(kernel_x, kernel_y, indexing='ij')
    
    # Pre-compute kernel template
    kernel_template = double_gaussian_kernel(K_x, K_y, seeing_fwhm, fiber_diameter)
    
    # Process each fiber
    for fiber_idx in range(n_fibers):
        fiber_x, fiber_y = fiber_positions[fiber_idx]
        
        # Find pixels within kernel radius of this fiber
        distances_squared = (X_grid - fiber_x)**2 + (Y_grid - fiber_y)**2
        within_radius = distances_squared <= kernel_radius_limit**2
        
        if not np.any(within_radius):
            # No pixels within kernel radius - skip this fiber
            continue
            
        # Get pixel indices within radius
        pixel_indices = np.where(within_radius)
        pixel_x_coords = X_grid[pixel_indices]
        pixel_y_coords = Y_grid[pixel_indices]
        
        # Calculate kernel values for these pixels
        dx = pixel_x_coords - fiber_x
        dy = pixel_y_coords - fiber_y
        
        # Evaluate kernel at pixel positions
        kernel_values = double_gaussian_kernel(dx, dy, seeing_fwhm, fiber_diameter)
        
        # Apply sparse threshold
        significant_mask = kernel_values >= sparse_threshold
        if np.any(significant_mask):
            # Convert 2D pixel indices to 1D matrix indices
            row_indices = pixel_indices[0] * n_y + pixel_indices[1]
            
            # Store significant kernel values
            A_matrix[row_indices[significant_mask], fiber_idx] = kernel_values[significant_mask]
    
    # Log matrix properties
    n_nonzero = np.count_nonzero(A_matrix)
    sparsity = 1.0 - (n_nonzero / A_matrix.size)
    logger.info(f"Kernel matrix built: {n_nonzero} non-zero elements "
                f"({sparsity:.3%} sparse)")
    
    return A_matrix


def create_sparse_kernel_matrix(fiber_positions: np.ndarray,
                               output_grid_x: np.ndarray, 
                               output_grid_y: np.ndarray,
                               seeing_fwhm: float,
                               fiber_diameter: float,
                               kernel_radius_limit: float,
                               sparse_threshold: float = 1e-6) -> csr_matrix:
    """Create sparse version of kernel matrix for memory efficiency.
    
    Same as build_kernel_matrix but returns scipy sparse matrix for
    better memory usage with large fiber bundles.
    
    Args:
        fiber_positions: Fiber positions array (n_fibers, 2) in arcsec
        output_grid_x: Output pixel x coordinates in arcsec
        output_grid_y: Output pixel y coordinates in arcsec
        seeing_fwhm: Atmospheric seeing FWHM in arcsec
        fiber_diameter: Fiber diameter in arcsec
        kernel_radius_limit: Truncation radius for kernel in arcsec
        sparse_threshold: Minimum value to include in sparse matrix
        
    Returns:
        Sparse kernel matrix A with shape (n_pixels, n_fibers)
    """
    # Build dense matrix first
    A_dense = build_kernel_matrix(
        fiber_positions, output_grid_x, output_grid_y,
        seeing_fwhm, fiber_diameter, kernel_radius_limit, sparse_threshold
    )
    
    # Convert to sparse format
    A_sparse = csr_matrix(A_dense)
    
    # Clean up memory
    del A_dense
    
    return A_sparse


def measure_kernel_properties(kernel: np.ndarray, 
                            pixel_scale: float) -> Dict[str, float]:
    """Measure properties of a kernel for quality assessment.
    
    Calculates FWHM, Strehl ratio, and other quality metrics for
    a 2D kernel array.
    
    Args:
        kernel: 2D kernel array
        pixel_scale: Pixel scale in arcsec/pixel
        
    Returns:
        Dictionary of kernel properties
    """
    # Find peak position
    peak_idx = np.unravel_index(np.argmax(kernel), kernel.shape)
    peak_value = kernel[peak_idx]
    
    # Calculate FWHM by finding half-maximum contour
    half_max = peak_value / 2.0
    above_half_max = kernel >= half_max
    
    # Estimate FWHM from area (circular approximation)
    n_pixels_hm = np.sum(above_half_max)
    if n_pixels_hm > 0:
        # Area = π * (FWHM/2)^2, so FWHM = 2 * sqrt(Area/π)  
        fwhm_pixels = 2.0 * np.sqrt(n_pixels_hm / np.pi)
        fwhm_arcsec = fwhm_pixels * pixel_scale
    else:
        fwhm_arcsec = 0.0
    
    # Calculate moments for ellipticity measurement
    y_indices, x_indices = np.indices(kernel.shape)
    x_center, y_center = peak_idx[1], peak_idx[0]  # Note index order
    
    # Weighted second moments
    total_weight = np.sum(kernel)
    if total_weight > 0:
        Ixx = np.sum(kernel * (x_indices - x_center)**2) / total_weight
        Iyy = np.sum(kernel * (y_indices - y_center)**2) / total_weight
        Ixy = np.sum(kernel * (x_indices - x_center) * (y_indices - y_center)) / total_weight
        
        # Calculate ellipticity
        trace = Ixx + Iyy
        det = Ixx * Iyy - Ixy**2
        if det > 0 and trace > 0:
            ellipticity = (trace - 2*np.sqrt(det)) / (trace + 2*np.sqrt(det))
        else:
            ellipticity = 0.0
    else:
        ellipticity = 0.0
    
    # Strehl ratio (peak value relative to ideal Airy disk)
    # For now, just use peak value as simple metric
    strehl_ratio = peak_value / np.sum(kernel)  # Normalized peak
    
    properties = {
        'fwhm_arcsec': fwhm_arcsec,
        'fwhm_pixels': fwhm_pixels if 'fwhm_pixels' in locals() else 0.0,
        'peak_value': peak_value,
        'total_flux': np.sum(kernel),
        'ellipticity': ellipticity,
        'strehl_ratio': strehl_ratio
    }
    
    return properties