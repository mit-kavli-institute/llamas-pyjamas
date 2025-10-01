"""
CRR Weight Matrix Computation Module

Implementation of weight matrix computation for the Covariance-regularized 
Reconstruction (CRR) method following Liu et al. (2020). This module contains
the core mathematical algorithms for computing optimal reconstruction weights
using SVD decomposition and regularization.

The implementation follows Equations 10-24 from Liu et al. (2020) for:
- SVD decomposition of the kernel matrix
- Regularized pseudo-inverse computation  
- Flux conservation matrix construction
- Final weight matrix assembly

Functions:
    build_variance_matrix: Create modified variance matrix N_tilde
    compute_crr_weights: Main CRR weight computation following Liu et al.
    compute_shepard_weights: Shepard's method weights for comparison
    flux_conservation_matrix: Build flux conservation normalization
    regularized_svd_inverse: Compute regularized SVD pseudo-inverse

Author: Generated for LLAMAS Pipeline
Date: September 2025
"""

import numpy as np
from typing import Tuple, Dict, Optional
import logging
from scipy.linalg import svd
from scipy.sparse import csr_matrix, diags
from scipy.spatial.distance import cdist
import warnings

from llamas_pyjamas.Utils.utils import setup_logger


def build_variance_matrix(mask: np.ndarray, ivar: np.ndarray) -> np.ndarray:
    """Build modified variance matrix N_tilde following Liu et al. (2020).
    
    Creates the modified variance matrix where:
    - N_tilde[i,i] = 1 for valid (unmasked) data
    - N_tilde[i,i] = 0 for invalid (masked) data
    
    This formulation allows the regularization to handle missing data
    naturally without requiring explicit masking in the linear algebra.
    
    Args:
        mask: Boolean mask array (True = valid data) shape (n_fibers,)
        ivar: Inverse variance array shape (n_fibers,)
        
    Returns:
        Modified variance matrix N_tilde shape (n_fibers, n_fibers)
    """
    n_fibers = len(mask)
    
    # Create diagonal matrix with unity for valid data, zero for masked
    diagonal_values = mask.astype(float)
    
    # For very low inverse variance (high noise), also set to zero
    # This handles cases where ivar is technically > 0 but unreliable
    low_ivar_threshold = 1e-10
    diagonal_values[ivar < low_ivar_threshold] = 0.0
    
    N_tilde = np.diag(diagonal_values)
    
    return N_tilde


def regularized_svd_inverse(A: np.ndarray, 
                          regularization_lambda: float,
                          condition_threshold: float = 1e-12) -> Tuple[np.ndarray, Dict[str, float]]:
    """Compute regularized SVD pseudo-inverse following Liu et al. (2020).
    
    Implements the regularized inversion from Equation 20:
    S* = S / (S² + λ²)
    
    where S are the singular values and λ is the regularization parameter.
    
    Args:
        A: Matrix to invert shape (n_pixels, n_fibers)
        regularization_lambda: Regularization parameter λ
        condition_threshold: Threshold for singular value truncation
        
    Returns:
        Tuple of (regularized_inverse, svd_info_dict)
    """
    logger = setup_logger(__name__)
    
    # Perform SVD decomposition: A = U S V^T
    U, S, Vt = svd(A, full_matrices=False)
    
    # Apply regularization to singular values
    S_regularized = S / (S**2 + regularization_lambda**2)
    
    # Count effective degrees of freedom
    n_significant = np.sum(S > condition_threshold * S[0])
    
    # Compute regularized pseudo-inverse: A^+ = V S* U^T
    A_reg_inv = Vt.T @ np.diag(S_regularized) @ U.T
    
    # Calculate condition number and other diagnostics
    condition_number = S[0] / S[-1] if S[-1] > 0 else np.inf
    effective_rank = np.sum(S**2 / (S**2 + regularization_lambda**2))
    
    svd_info = {
        'n_singular_values': len(S),
        'n_significant': n_significant,
        'condition_number': condition_number,
        'effective_rank': effective_rank,
        'regularization_lambda': regularization_lambda,
        'largest_singular_value': S[0],
        'smallest_singular_value': S[-1]
    }
    
    logger.info(f"SVD decomposition: {len(S)} singular values, "
                f"condition number = {condition_number:.2e}")
    logger.info(f"Effective rank with regularization: {effective_rank:.1f}")
    
    return A_reg_inv, svd_info


def flux_conservation_matrix(Q: np.ndarray, 
                           fiber_positions: np.ndarray,
                           output_grid_x: np.ndarray,
                           output_grid_y: np.ndarray) -> np.ndarray:
    """Build flux conservation matrix R following Liu et al. (2020) Eq. 14.
    
    Normalizes the reconstruction weights to ensure flux conservation:
    R[i,j] = Q[i,j] / Σ_k Q[i,k]
    
    where Q is the pre-normalization weight matrix and R is the final
    flux-conserving weight matrix.
    
    Args:
        Q: Pre-normalization weight matrix shape (n_pixels, n_fibers)
        fiber_positions: Fiber positions (n_fibers, 2) in arcsec
        output_grid_x: Output pixel x coordinates in arcsec
        output_grid_y: Output pixel y coordinates in arcsec
        
    Returns:
        Flux conservation matrix R shape (n_pixels, n_fibers)
    """
    logger = setup_logger(__name__)
    
    # Calculate row sums for normalization
    row_sums = np.sum(Q, axis=1)
    
    # Handle pixels with zero weight sum
    zero_weight_pixels = (row_sums == 0)
    n_zero_pixels = np.sum(zero_weight_pixels)
    
    if n_zero_pixels > 0:
        logger.warning(f"{n_zero_pixels} pixels have zero weight sum - "
                      "these will remain zero in output")
    
    # Avoid division by zero
    safe_row_sums = row_sums.copy()
    safe_row_sums[zero_weight_pixels] = 1.0  # Will be multiplied by 0 anyway
    
    # Normalize each row to sum to 1 (flux conservation)
    R = Q / safe_row_sums[:, np.newaxis]
    
    # Ensure zero-weight pixels remain zero
    R[zero_weight_pixels, :] = 0.0
    
    # Verify flux conservation
    normalized_row_sums = np.sum(R, axis=1)
    non_zero_mask = ~zero_weight_pixels
    flux_conservation_error = np.max(np.abs(normalized_row_sums[non_zero_mask] - 1.0))
    
    logger.info(f"Flux conservation matrix built: "
                f"max normalization error = {flux_conservation_error:.2e}")
    
    return R


def compute_crr_weights(A_matrix: np.ndarray,
                       mask: np.ndarray,
                       ivar: np.ndarray, 
                       regularization_lambda: float) -> Tuple[np.ndarray, np.ndarray, Dict[str, any]]:
    """Compute CRR reconstruction weights following Liu et al. (2020).
    
    Implements the full CRR weight computation algorithm from Equations 10-24:
    
    1. Build modified variance matrix N_tilde
    2. Perform SVD of N_tilde^(-1/2) @ A
    3. Apply regularization to singular values  
    4. Compute flux conservation matrix R
    5. Build final weight matrix W
    6. Calculate covariance matrix C_G
    
    Args:
        A_matrix: Kernel matrix shape (n_pixels, n_fibers)
        mask: Data validity mask shape (n_fibers,) 
        ivar: Inverse variance array shape (n_fibers,)
        regularization_lambda: Regularization parameter λ
        
    Returns:
        Tuple of (weight_matrix, covariance_matrix, reconstruction_info)
    """
    logger = setup_logger(__name__)
    logger.info("Computing CRR reconstruction weights")
    
    n_pixels, n_fibers = A_matrix.shape
    
    # Step 1: Build modified variance matrix N_tilde (Eq. 11)
    N_tilde = build_variance_matrix(mask, ivar)
    
    # Step 2: Compute N_tilde^(-1/2) for SVD preparation
    # Since N_tilde is diagonal with 0s and 1s, N_tilde^(-1/2) is also diagonal
    N_tilde_diag = np.diag(N_tilde)
    N_tilde_inv_sqrt_diag = np.zeros_like(N_tilde_diag)
    valid_mask = N_tilde_diag > 0
    N_tilde_inv_sqrt_diag[valid_mask] = 1.0 / np.sqrt(N_tilde_diag[valid_mask])
    N_tilde_inv_sqrt = np.diag(N_tilde_inv_sqrt_diag)
    
    # Step 3: SVD decomposition of N_tilde^(-1/2) @ A (Eq. 10)
    A_modified = N_tilde_inv_sqrt @ A_matrix.T  # Note: we work with transposed A
    
    # Perform regularized SVD inversion
    Q_transpose, svd_info = regularized_svd_inverse(A_modified.T, regularization_lambda)
    Q = Q_transpose.T
    
    # Step 4: Build flux conservation matrix R (Eq. 14) 
    R = flux_conservation_matrix(Q, None, None, None)  # Grid info not needed for normalization
    
    # Step 5: Final weight matrix W (Eq. 23)
    # W = R @ V^T @ S* @ U^T @ N_tilde^(-1/2)
    # But we already computed Q = regularized_inverse, so:
    W = R @ N_tilde_inv_sqrt
    
    # Step 6: Covariance matrix computation (Eq. 24)
    # C_G = W @ N @ W^T, but since N = N_tilde for valid data:
    # C_G = W @ N_tilde @ W^T
    covariance_matrix = W @ N_tilde @ W.T
    
    # Extract diagonal elements for output (most commonly needed)
    covariance_diagonal = np.diag(covariance_matrix)
    
    # Quality metrics
    reconstruction_info = {
        'method': 'CRR',
        'regularization_lambda': regularization_lambda,
        'n_valid_fibers': np.sum(valid_mask),
        'n_fibers_total': n_fibers,
        'n_pixels': n_pixels,
        'svd_info': svd_info,
        'weight_matrix_norm': np.linalg.norm(W),
        'flux_conservation_check': np.allclose(np.sum(W, axis=1), 1.0, atol=1e-10)
    }
    
    logger.info(f"CRR weights computed: {n_pixels} pixels, {np.sum(valid_mask)} valid fibers")
    logger.info(f"Weight matrix norm: {reconstruction_info['weight_matrix_norm']:.3f}")
    
    return W, covariance_diagonal, reconstruction_info


def compute_shepard_weights(fiber_positions: np.ndarray,
                          output_grid_x: np.ndarray, 
                          output_grid_y: np.ndarray,
                          mask: np.ndarray,
                          gaussian_sigma: float = 1.0) -> np.ndarray:
    """Compute Shepard's method weights for comparison with CRR.
    
    Implements traditional Shepard's method (inverse distance weighting)
    with Gaussian weights and flux conservation normalization following
    Equations 2-3 from Liu et al. (2020).
    
    Weight formula:
    w[i,j] = exp(-(r[i,j]^2) / (2σ²))
    
    where r[i,j] is distance between pixel i and fiber j.
    
    Args:
        fiber_positions: Fiber positions (n_fibers, 2) in arcsec
        output_grid_x: Output pixel x coordinates in arcsec
        output_grid_y: Output pixel y coordinates in arcsec  
        mask: Data validity mask shape (n_fibers,)
        gaussian_sigma: Gaussian width parameter in arcsec
        
    Returns:
        Shepard weight matrix shape (n_pixels, n_fibers)
    """
    logger = setup_logger(__name__)
    
    n_x, n_y = len(output_grid_x), len(output_grid_y)
    n_pixels = n_x * n_y
    n_fibers = len(fiber_positions)
    
    logger.info(f"Computing Shepard weights: σ = {gaussian_sigma:.3f} arcsec")
    
    # Create pixel coordinate arrays
    X_grid, Y_grid = np.meshgrid(output_grid_x, output_grid_y, indexing='ij')
    pixel_coords = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Compute distances between all pixels and fibers
    distances = cdist(pixel_coords, fiber_positions)  # Shape: (n_pixels, n_fibers)
    
    # Apply Gaussian weighting
    weights = np.exp(-0.5 * (distances / gaussian_sigma)**2)
    
    # Apply mask to invalid fibers
    weights[:, ~mask] = 0.0
    
    # Flux conservation normalization
    row_sums = np.sum(weights, axis=1)
    zero_weight_pixels = (row_sums == 0)
    
    # Avoid division by zero
    safe_row_sums = row_sums.copy() 
    safe_row_sums[zero_weight_pixels] = 1.0
    
    # Normalize
    weights_normalized = weights / safe_row_sums[:, np.newaxis]
    weights_normalized[zero_weight_pixels, :] = 0.0
    
    logger.info(f"Shepard weights computed: {np.sum(~zero_weight_pixels)} pixels with coverage")
    
    return weights_normalized


def compute_weight_quality_metrics(W: np.ndarray,
                                 fiber_positions: np.ndarray,
                                 output_grid_x: np.ndarray,
                                 output_grid_y: np.ndarray) -> Dict[str, float]:
    """Compute quality metrics for reconstruction weight matrix.
    
    Calculates various quality indicators for the weight matrix including:
    - Effective resolution (weighted distance spread)
    - Localization measure (weight concentration)
    - Flux conservation accuracy
    - Condition number of active weights
    
    Args:
        W: Weight matrix shape (n_pixels, n_fibers)
        fiber_positions: Fiber positions (n_fibers, 2) in arcsec
        output_grid_x: Output pixel x coordinates in arcsec
        output_grid_y: Output pixel y coordinates in arcsec
        
    Returns:
        Dictionary of quality metrics
    """
    n_pixels, n_fibers = W.shape
    n_x, n_y = len(output_grid_x), len(output_grid_y)
    
    # Create pixel coordinate arrays
    X_grid, Y_grid = np.meshgrid(output_grid_x, output_grid_y, indexing='ij')
    pixel_coords = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Calculate effective resolution for each pixel
    effective_resolutions = []
    localization_measures = []
    
    for i in range(n_pixels):
        weights_i = W[i, :]
        
        if np.sum(weights_i) == 0:
            effective_resolutions.append(np.inf)
            localization_measures.append(0.0)
            continue
            
        # Normalize weights for this pixel
        weights_norm = weights_i / np.sum(weights_i)
        
        # Calculate weighted centroid of contributing fibers
        centroid = np.sum(weights_norm[:, np.newaxis] * fiber_positions, axis=0)
        
        # Calculate weighted RMS distance from centroid
        distances_to_centroid = np.linalg.norm(fiber_positions - centroid, axis=1)
        effective_resolution = np.sqrt(np.sum(weights_norm * distances_to_centroid**2))
        effective_resolutions.append(effective_resolution)
        
        # Localization measure (inverse of effective number of fibers)
        effective_n_fibers = 1.0 / np.sum(weights_norm**2)
        localization_measures.append(1.0 / effective_n_fibers)
    
    effective_resolutions = np.array(effective_resolutions)
    localization_measures = np.array(localization_measures)
    
    # Overall quality metrics
    flux_conservation_errors = np.abs(np.sum(W, axis=1) - 1.0)
    valid_pixels = np.isfinite(effective_resolutions) & (effective_resolutions < np.inf)
    
    quality_metrics = {
        'mean_effective_resolution': np.mean(effective_resolutions[valid_pixels]),
        'median_effective_resolution': np.median(effective_resolutions[valid_pixels]),
        'mean_localization': np.mean(localization_measures[valid_pixels]),
        'median_localization': np.median(localization_measures[valid_pixels]),
        'max_flux_conservation_error': np.max(flux_conservation_errors),
        'mean_flux_conservation_error': np.mean(flux_conservation_errors),
        'fraction_valid_pixels': np.sum(valid_pixels) / len(valid_pixels),
        'weight_matrix_condition': np.linalg.cond(W) if n_pixels <= n_fibers else np.inf
    }
    
    return quality_metrics