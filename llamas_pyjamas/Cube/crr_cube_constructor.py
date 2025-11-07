"""
Covariance-regularized Reconstruction (CRR) Cube Constructor

Implementation of the CRR method from Liu et al. (2020) for creating data cubes 
from RSS (Row-Stacked Spectra) data with improved spatial resolution and 
covariance regularization.

This module provides the core CRR cube reconstruction functionality following
the mathematical framework described in Liu et al. (2020), optimized for
LLAMAS IFU data processing.

Classes:
    CRRCubeConfig: Configuration parameters for CRR reconstruction
    RSSData: Data structure for input RSS data
    CRRDataCube: Output data cube structure with covariance information
    CRRCubeConstructor: Main reconstruction engine

Author: Generated for LLAMAS Pipeline
Date: September 2025
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import RegularGridInterpolator
import warnings

from llamas_pyjamas.Utils.utils import setup_logger
from llamas_pyjamas.Cube.crr_kernels import (
    build_kernel_matrix, wavelength_dependent_seeing, 
    double_gaussian_kernel, measure_kernel_properties
)
from llamas_pyjamas.Cube.crr_weights import (
    compute_crr_weights, compute_shepard_weights,
    compute_weight_quality_metrics
)


@dataclass
class CRRCubeConfig:
    """Configuration parameters for CRR cube reconstruction.
    
    Based on Liu et al. (2020) default parameters with adaptations
    for LLAMAS telescope specifications.
    
    Attributes:
        pixel_scale: Output pixel scale in arcsec/pixel (default 0.75 from paper)
        regularization_lambda: Regularization parameter for SVD inversion (default 1e-3)
        kernel_radius_limit: Kernel truncation radius in arcsec (default 4.0)
        reconstruction_radius_limit: Pixel inclusion limit in arcsec (default 1.6)
        use_sky_subtraction: Enable sky subtraction preprocessing
        output_wavelength_grid: Wavelength sampling for output cube (optional)
        fiber_diameter: Physical fiber diameter in arcsec (default 2.0 for LLAMAS)
        seeing_reference_wavelength: Reference wavelength for seeing scaling in Angstroms
        seeing_power_law_index: Power law index for wavelength-dependent seeing (-1/5)
    """
    pixel_scale: float = 0.75  # arcsec/pixel
    regularization_lambda: float = 1e-3
    kernel_radius_limit: float = 4.0  # arcsec
    reconstruction_radius_limit: float = 1.6  # arcsec
    use_sky_subtraction: bool = False
    output_wavelength_grid: Optional[np.ndarray] = None
    fiber_diameter: float = 2.0  # arcsec, LLAMAS fiber diameter
    seeing_reference_wavelength: float = 5500.0  # Angstroms
    seeing_power_law_index: float = -0.2  # -1/5 from atmospheric seeing theory


@dataclass
class RSSData:
    """Data structure for RSS (Row-Stacked Spectra) input data.
    
    Contains all necessary information for CRR cube reconstruction including
    flux, variance, fiber positions, and wavelength information.
    
    Attributes:
        flux: 2D flux array (n_fibers, n_wavelengths)
        ivar: 2D inverse variance array (n_fibers, n_wavelengths)
        mask: 2D boolean mask array (True = valid data)
        fiber_positions: Fiber coordinates for each wavelength (n_wavelengths, n_fibers, 2)
        wavelength: 1D wavelength array in Angstroms
        seeing_fwhm: Seeing FWHM at reference wavelength in arcsec
        metadata: Dictionary containing observation information
    """
    flux: np.ndarray
    ivar: np.ndarray
    mask: np.ndarray
    fiber_positions: np.ndarray  # Shape: (n_wavelengths, n_fibers, 2) or (n_fibers, 2)
    wavelength: np.ndarray
    seeing_fwhm: float
    metadata: Dict[str, Any]
    
    def __post_init__(self):
        """Validate RSS data structure and shapes."""
        # Check basic array shapes
        n_fibers, n_wavelengths = self.flux.shape
        
        if self.ivar.shape != (n_fibers, n_wavelengths):
            raise ValueError(f"ivar shape {self.ivar.shape} doesn't match flux shape {self.flux.shape}")
            
        if self.mask.shape != (n_fibers, n_wavelengths):
            raise ValueError(f"mask shape {self.mask.shape} doesn't match flux shape {self.flux.shape}")
            
        if len(self.wavelength) != n_wavelengths:
            raise ValueError(f"wavelength length {len(self.wavelength)} doesn't match n_wavelengths {n_wavelengths}")
        
        # Handle fiber positions - can be wavelength-dependent or static
        if self.fiber_positions.ndim == 2:
            # Static positions: (n_fibers, 2)
            if self.fiber_positions.shape[0] != n_fibers:
                raise ValueError(f"fiber_positions shape {self.fiber_positions.shape} doesn't match n_fibers {n_fibers}")
        elif self.fiber_positions.ndim == 3:
            # Wavelength-dependent positions: (n_wavelengths, n_fibers, 2)
            if self.fiber_positions.shape != (n_wavelengths, n_fibers, 2):
                raise ValueError(f"fiber_positions shape {self.fiber_positions.shape} doesn't match expected ({n_wavelengths}, {n_fibers}, 2)")
        else:
            raise ValueError(f"fiber_positions must be 2D or 3D array, got {self.fiber_positions.ndim}D")


class CRRDataCube:
    """Output data cube structure with CRR reconstruction information.
    
    Contains the reconstructed cube data along with covariance information,
    quality metrics, and reconstruction metadata following Liu et al. (2020).
    """
    
    def __init__(self, shape: Tuple[int, int, int], wcs: WCS = None):
        """Initialize empty data cube structure.
        
        Args:
            shape: Tuple of (n_x, n_y, n_wavelength) for cube dimensions
            wcs: World coordinate system for spatial coordinates
        """
        self.shape = shape
        n_x, n_y, n_wavelength = shape
        
        # Main data arrays
        self.flux = np.zeros(shape, dtype=np.float32)
        self.ivar = np.zeros(shape, dtype=np.float32)
        self.mask = np.ones(shape, dtype=bool)  # True = valid data
        
        # Covariance information (diagonal elements)
        self.covariance_diagonal = np.zeros(shape, dtype=np.float32)
        
        # Quality flags (following SDSS convention)
        self.flags = np.zeros(shape, dtype=np.int32)
        self.FLAG_NOCOV = 1  # No coverage
        self.FLAG_LOWCOV = 2  # Low coverage/high uncertainty
        
        # PSF information
        self.psf_fwhm = np.zeros(n_wavelength, dtype=np.float32)
        
        # WCS and metadata
        self.wcs = wcs
        self.metadata = {}
        
        # Quality assessment results
        self.quality_metrics = {}
    
    def set_flag(self, flag: int, condition: np.ndarray):
        """Set quality flag for pixels meeting condition.
        
        Args:
            flag: Flag value to set
            condition: Boolean array indicating which pixels to flag
        """
        self.flags[condition] |= flag
    
    def has_flag(self, flag: int) -> np.ndarray:
        """Check which pixels have a specific flag set.
        
        Args:
            flag: Flag value to check
            
        Returns:
            Boolean array indicating pixels with the flag
        """
        return (self.flags & flag) != 0
    
    def save_to_fits(self, filename: str):
        """Save cube to FITS file with all extensions.
        
        Args:
            filename: Output FITS filename
        """
        # Create HDU list
        hdu_list = []
        
        # Primary HDU with flux data
        primary_hdr = fits.Header()
        primary_hdr['OBJECT'] = self.metadata.get('object', 'Unknown')
        primary_hdr['INSTRUME'] = 'LLAMAS'
        primary_hdr['METHOD'] = ('CRR', 'Covariance-regularized Reconstruction')
        primary_hdr['CRRLAMBDA'] = (self.metadata.get('regularization_lambda', 1e-3), 'CRR regularization parameter')
        primary_hdr['PIXSCALE'] = (self.metadata.get('pixel_scale', 0.75), 'Output pixel scale (arcsec/pixel)')
        primary_hdr['CUBEREF'] = 'Liu et al. (2020)'
        
        primary_hdu = fits.PrimaryHDU(data=self.flux, header=primary_hdr)
        if self.wcs is not None:
            primary_hdu.header.update(self.wcs.to_header())
        hdu_list.append(primary_hdu)
        
        # Inverse variance extension
        ivar_hdu = fits.ImageHDU(data=self.ivar, name='IVAR')
        hdu_list.append(ivar_hdu)
        
        # Mask extension
        mask_hdu = fits.ImageHDU(data=self.mask.astype(np.uint8), name='MASK')
        hdu_list.append(mask_hdu)
        
        # Flags extension
        flags_hdu = fits.ImageHDU(data=self.flags, name='FLAGS')
        hdu_list.append(flags_hdu)
        
        # Covariance diagonal extension
        covar_hdu = fits.ImageHDU(data=self.covariance_diagonal, name='COVAR_DIAG')
        hdu_list.append(covar_hdu)
        
        # PSF FWHM table
        psf_col = fits.Column(name='PSF_FWHM', format='E', array=self.psf_fwhm)
        psf_hdu = fits.BinTableHDU.from_columns([psf_col], name='PSF_INFO')
        hdu_list.append(psf_hdu)
        
        # Write to file
        fits.HDUList(hdu_list).writeto(filename, overwrite=True)
        logging.info(f"CRR data cube saved to {filename}")


class CRRCubeConstructor:
    """Main CRR cube reconstruction engine.
    
    Implements the Covariance-regularized Reconstruction method following
    Liu et al. (2020) with optimizations for LLAMAS IFU data.
    """
    
    def __init__(self, config: CRRCubeConfig):
        """Initialize CRR cube constructor.
        
        Args:
            config: CRR configuration parameters
        """
        self.config = config
        self.logger = setup_logger(__name__)
        
        # Will be set during reconstruction
        self.output_grid_x = None
        self.output_grid_y = None
        self.grid_extent = None
        
    def setup_output_grid(self, fiber_positions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Define rectangular output pixel grid based on fiber positions.
        
        Args:
            fiber_positions: Fiber positions array (n_fibers, 2) or (n_wavelengths, n_fibers, 2)
            
        Returns:
            Tuple of (x_grid, y_grid) coordinate arrays
        """
        # Handle wavelength-dependent positions
        if fiber_positions.ndim == 3:
            # Take positions from middle wavelength as representative
            mid_idx = fiber_positions.shape[0] // 2
            positions = fiber_positions[mid_idx]
        else:
            positions = fiber_positions
            
        # Calculate grid bounds with padding
        x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
        y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
        
        # Add padding around fiber bundle
        padding = 2 * self.config.kernel_radius_limit
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        
        # Create regular grid
        n_x = int(np.ceil((x_max - x_min) / self.config.pixel_scale))
        n_y = int(np.ceil((y_max - y_min) / self.config.pixel_scale))
        
        x_grid = np.linspace(x_min, x_max, n_x)
        y_grid = np.linspace(y_min, y_max, n_y)
        
        self.output_grid_x = x_grid
        self.output_grid_y = y_grid
        self.grid_extent = (x_min, x_max, y_min, y_max)
        
        self.logger.info(f"Output grid setup: {n_x} x {n_y} pixels, "
                        f"scale = {self.config.pixel_scale} arcsec/pixel")
        
        return x_grid, y_grid
    
    
    def apply_sky_subtraction(self, rss_data: RSSData) -> RSSData:
        """Apply sky subtraction to RSS data if enabled.
        
        Args:
            rss_data: Input RSS data
            
        Returns:
            Sky-subtracted RSS data
        """
        if not self.config.use_sky_subtraction:
            self.logger.warning("Sky subtraction disabled - proceeding without sky subtraction")
            return rss_data
        
        # Sky subtraction implementation will be integrated with existing pipeline
        # For now, return data unchanged with warning
        self.logger.warning("Sky subtraction not yet integrated with existing pipeline")
        return rss_data
    
    def process_wavelength_slice(self, wavelength_idx: int, rss_data: RSSData) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process single wavelength slice using CRR method.
        
        Args:
            wavelength_idx: Index of wavelength slice to process
            rss_data: RSS data structure
            
        Returns:
            Tuple of (flux_slice, ivar_slice, covar_slice) for this wavelength
        """
        self.logger.info(f"Processing wavelength slice {wavelength_idx} "
                        f"(λ = {rss_data.wavelength[wavelength_idx]:.1f} Å)")
        
        # Extract data for this wavelength
        flux_slice = rss_data.flux[:, wavelength_idx]
        ivar_slice = rss_data.ivar[:, wavelength_idx]
        mask_slice = rss_data.mask[:, wavelength_idx]
        
        # Get fiber positions for this wavelength
        if rss_data.fiber_positions.ndim == 3:
            fiber_pos = rss_data.fiber_positions[wavelength_idx]
        else:
            fiber_pos = rss_data.fiber_positions
            
        # Calculate seeing for this wavelength
        seeing_fwhm = wavelength_dependent_seeing(
            np.array([rss_data.wavelength[wavelength_idx]]), 
            self.config.seeing_reference_wavelength,
            rss_data.seeing_fwhm,
            self.config.seeing_power_law_index
        )[0]
        
        # Build kernel matrix A for this wavelength
        A_matrix = build_kernel_matrix(
            fiber_pos,
            self.output_grid_x,
            self.output_grid_y,
            seeing_fwhm,
            self.config.fiber_diameter,
            self.config.kernel_radius_limit
        )
        
        # Compute CRR weights
        try:
            W, covariance_diagonal, reconstruction_info = compute_crr_weights(
                A_matrix,
                mask_slice,
                ivar_slice,
                self.config.regularization_lambda
            )
            
            # Apply reconstruction: G = W @ f (Equation 22 from Liu et al.)
            flux_reconstructed = W @ flux_slice
            
            # Compute inverse variance from covariance
            # For pixels with zero covariance, set ivar to 0
            ivar_reconstructed = np.zeros_like(covariance_diagonal)
            valid_covar = covariance_diagonal > 0
            ivar_reconstructed[valid_covar] = 1.0 / covariance_diagonal[valid_covar]
            
            # Reshape from 1D pixel arrays back to 2D grid
            n_x, n_y = len(self.output_grid_x), len(self.output_grid_y)
            
            flux_out = flux_reconstructed.reshape(n_x, n_y)
            ivar_out = ivar_reconstructed.reshape(n_x, n_y)
            covar_out = covariance_diagonal.reshape(n_x, n_y)
            
            # Log reconstruction quality
            n_valid_fibers = reconstruction_info['n_valid_fibers']
            effective_rank = reconstruction_info['svd_info']['effective_rank']
            self.logger.debug(f"Wavelength {wavelength_idx}: {n_valid_fibers} fibers, "
                             f"effective rank = {effective_rank:.1f}")
            
        except Exception as e:
            self.logger.error(f"CRR reconstruction failed for wavelength {wavelength_idx}: {e}")
            # Return zero arrays on failure
            n_x, n_y = len(self.output_grid_x), len(self.output_grid_y)
            flux_out = np.zeros((n_x, n_y))
            ivar_out = np.zeros((n_x, n_y))
            covar_out = np.zeros((n_x, n_y))
            
        return flux_out, ivar_out, covar_out
    
    def reconstruct_cube(self, rss_data: RSSData) -> CRRDataCube:
        """Main reconstruction method to create CRR data cube.
        
        Args:
            rss_data: Input RSS data structure
            
        Returns:
            Reconstructed CRR data cube
        """
        self.logger.info("Starting CRR cube reconstruction")
        
        # Apply sky subtraction if enabled
        rss_data = self.apply_sky_subtraction(rss_data)
        
        # Setup output grid
        x_grid, y_grid = self.setup_output_grid(rss_data.fiber_positions)
        n_x, n_y = len(x_grid), len(y_grid)
        n_wavelength = len(rss_data.wavelength)
        
        # Initialize output cube
        cube_shape = (n_x, n_y, n_wavelength)
        output_cube = CRRDataCube(cube_shape)
        
        # Store reconstruction parameters
        output_cube.metadata.update({
            'regularization_lambda': self.config.regularization_lambda,
            'pixel_scale': self.config.pixel_scale,
            'kernel_radius_limit': self.config.kernel_radius_limit,
            'reconstruction_radius_limit': self.config.reconstruction_radius_limit,
            'fiber_diameter': self.config.fiber_diameter,
            'seeing_fwhm_ref': rss_data.seeing_fwhm,
            'wavelength_range': (rss_data.wavelength.min(), rss_data.wavelength.max()),
            'n_fibers': rss_data.flux.shape[0],
            'reconstruction_method': 'CRR',
            'reference': 'Liu et al. (2020)'
        })
        
        # Process each wavelength slice
        for wave_idx in range(n_wavelength):
            try:
                flux_slice, ivar_slice, covar_slice = self.process_wavelength_slice(wave_idx, rss_data)
                
                # Store results
                output_cube.flux[:, :, wave_idx] = flux_slice
                output_cube.ivar[:, :, wave_idx] = ivar_slice
                output_cube.covariance_diagonal[:, :, wave_idx] = covar_slice
                
                # Calculate PSF FWHM for this wavelength
                seeing_fwhm = wavelength_dependent_seeing(
                    np.array([rss_data.wavelength[wave_idx]]), 
                    self.config.seeing_reference_wavelength,
                    rss_data.seeing_fwhm,
                    self.config.seeing_power_law_index
                )[0]
                output_cube.psf_fwhm[wave_idx] = seeing_fwhm
                
                # Set quality flags
                no_coverage = ivar_slice == 0
                low_coverage = (ivar_slice > 0) & (ivar_slice < 1e-6)  # Threshold TBD
                
                output_cube.set_flag(output_cube.FLAG_NOCOV, no_coverage)
                output_cube.set_flag(output_cube.FLAG_LOWCOV, low_coverage)
                output_cube.mask[:, :, wave_idx] = ~(no_coverage | low_coverage)
                
            except Exception as e:
                self.logger.error(f"Error processing wavelength {wave_idx}: {e}")
                # Set entire slice to no coverage
                output_cube.set_flag(output_cube.FLAG_NOCOV, 
                                   np.ones((n_x, n_y), dtype=bool))
        
        # Run quality assessment
        self.logger.info("Running quality assessment")
        quality_metrics = self.assess_cube_quality(output_cube, rss_data)
        output_cube.quality_metrics = quality_metrics
        
        self.logger.info(f"CRR cube reconstruction completed: {cube_shape}")
        return output_cube
    
    def assess_cube_quality(self, cube: CRRDataCube, rss_data: RSSData) -> Dict[str, Any]:
        """Assess quality of reconstructed cube.
        
        Args:
            cube: Reconstructed CRR data cube
            rss_data: Original RSS data
            
        Returns:
            Dictionary with quality assessment results
        """
        self.logger.info("Performing quality assessment")
        
        quality_metrics = {}
        
        # Coverage statistics
        total_pixels = np.prod(cube.shape[:2])  # x, y dimensions
        n_wavelengths = cube.shape[2]
        
        for wave_idx in [0, n_wavelengths//4, n_wavelengths//2, 3*n_wavelengths//4, n_wavelengths-1]:
            wave_slice = cube.mask[:, :, wave_idx]
            coverage_fraction = np.sum(wave_slice) / total_pixels
            quality_metrics[f'coverage_fraction_wave_{wave_idx}'] = coverage_fraction
        
        # Overall coverage
        any_coverage = np.any(cube.mask, axis=2)
        total_coverage_fraction = np.sum(any_coverage) / total_pixels
        quality_metrics['total_coverage_fraction'] = total_coverage_fraction
        
        # PSF quality
        psf_fwhm_range = (cube.psf_fwhm.min(), cube.psf_fwhm.max())
        psf_fwhm_median = np.median(cube.psf_fwhm)
        quality_metrics['psf_fwhm_range'] = psf_fwhm_range
        quality_metrics['psf_fwhm_median'] = psf_fwhm_median
        
        # Flux statistics (on valid pixels only)
        valid_mask = cube.mask & (cube.ivar > 0)
        if np.any(valid_mask):
            valid_flux = cube.flux[valid_mask]
            flux_percentiles = np.percentile(valid_flux, [1, 5, 25, 50, 75, 95, 99])
            quality_metrics['flux_percentiles'] = flux_percentiles
            
            # Signal-to-noise estimates
            valid_snr = np.sqrt(cube.ivar[valid_mask]) * np.abs(valid_flux)
            snr_percentiles = np.percentile(valid_snr, [10, 25, 50, 75, 90])
            quality_metrics['snr_percentiles'] = snr_percentiles
        
        # Covariance quality
        valid_covar = cube.covariance_diagonal[valid_mask] if np.any(valid_mask) else np.array([])
        if len(valid_covar) > 0:
            covar_percentiles = np.percentile(valid_covar, [1, 10, 50, 90, 99])
            quality_metrics['covariance_percentiles'] = covar_percentiles
        
        # Flag statistics
        n_nocov = np.sum(cube.has_flag(cube.FLAG_NOCOV))
        n_lowcov = np.sum(cube.has_flag(cube.FLAG_LOWCOV))
        total_flagged = n_nocov + n_lowcov
        
        quality_metrics['flag_statistics'] = {
            'n_nocov': int(n_nocov),
            'n_lowcov': int(n_lowcov),
            'n_total_flagged': int(total_flagged),
            'fraction_flagged': total_flagged / np.prod(cube.shape)
        }
        
        self.logger.info(f"Quality assessment complete: "
                        f"{total_coverage_fraction:.1%} coverage, "
                        f"median PSF = {psf_fwhm_median:.2f} arcsec")
        
        return quality_metrics
    
    def measure_psf_quality(self, cube: CRRDataCube, 
                          test_positions: Optional[np.ndarray] = None,
                          n_test_sources: int = 10) -> Dict[str, Any]:
        """Measure PSF quality by injecting simulated point sources.
        
        Args:
            cube: CRR data cube
            test_positions: Test positions (n_sources, 2) or None for random
            n_test_sources: Number of test sources if positions not provided
            
        Returns:
            Dictionary with PSF quality measurements
        """
        self.logger.info("Measuring PSF quality with simulated point sources")
        
        # This is a placeholder for full PSF quality assessment
        # In a complete implementation, this would:
        # 1. Inject simulated point sources at known positions
        # 2. Reconstruct the cube with these sources
        # 3. Measure the reconstructed PSF properties
        # 4. Compare with expected kernel properties
        
        psf_quality = {
            'method': 'simulated_point_sources',
            'n_test_sources': n_test_sources,
            'status': 'not_implemented',
            'note': 'PSF quality measurement requires simulated source injection'
        }
        
        return psf_quality
    
    def measure_covariance_quality(self, cube: CRRDataCube) -> Dict[str, Any]:
        """Measure covariance matrix quality.
        
        Args:
            cube: CRR data cube
            
        Returns:
            Dictionary with covariance quality measurements
        """
        self.logger.info("Measuring covariance quality")
        
        # Sample a subset of pixels for covariance analysis
        n_x, n_y, n_wave = cube.shape
        sample_size = min(1000, n_x * n_y // 4)  # Sample up to 1000 pixels
        
        # Get random valid pixels
        valid_pixels = np.where(cube.mask.any(axis=2))
        if len(valid_pixels[0]) == 0:
            return {'status': 'no_valid_pixels'}
            
        sample_indices = np.random.choice(len(valid_pixels[0]), 
                                        size=min(sample_size, len(valid_pixels[0])), 
                                        replace=False)
        
        sample_x = valid_pixels[0][sample_indices]
        sample_y = valid_pixels[1][sample_indices]
        
        # Analyze covariance diagonal values
        covar_samples = []
        for i in range(len(sample_x)):
            x, y = sample_x[i], sample_y[i]
            pixel_covar = cube.covariance_diagonal[x, y, :]
            valid_covar = pixel_covar[pixel_covar > 0]
            if len(valid_covar) > 0:
                covar_samples.append(valid_covar)
        
        if not covar_samples:
            return {'status': 'no_valid_covariance'}
        
        # Combine all covariance samples
        all_covar = np.concatenate(covar_samples)
        
        covariance_quality = {
            'n_sample_pixels': len(covar_samples),
            'n_covar_measurements': len(all_covar),
            'covar_mean': float(np.mean(all_covar)),
            'covar_std': float(np.std(all_covar)),
            'covar_median': float(np.median(all_covar)),
            'covar_range': (float(np.min(all_covar)), float(np.max(all_covar))),
            'status': 'completed'
        }
        
        return covariance_quality