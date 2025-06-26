"""
Module for constructing 3D IFU data cubes from extracted fiber spectra.

This module provides the CubeConstructor class which takes extracted wavelength data
from multiple fibers across different detectors and reconstructs a full 3D data cube
with spatial (x,y) and spectral (λ) dimensions.

Classes:
    CubeConstructor: Main class for IFU cube reconstruction from extraction data.

Dependencies:
    - numpy
    - scipy
    - astropy
    - matplotlib
    - llamas_pyjamas modules
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Optional, Union
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from datetime import datetime
import logging

# Import LLAMAS modules
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.Image.WhiteLightModule import FiberMap_LUT
from llamas_pyjamas.Image.processWhiteLight import quartile_bias, remove_striping
from llamas_pyjamas.config import OUTPUT_DIR, LUT_DIR
from llamas_pyjamas.Utils.utils import setup_logger


class CubeConstructor:
    """
    A class for constructing 3D IFU data cubes from extracted fiber spectral data.
    
    This class takes extraction files containing wavelength-calibrated spectra from 
    individual fibers across multiple detectors and reconstructs them into a unified
    3D data cube with spatial (x,y) and spectral (wavelength) dimensions.
    
    Attributes:
        fiber_map_path (str): Path to the fiber mapping lookup table
        fibermap_lut (Table): Loaded fiber mapping table
        logger (Logger): Logger instance for debugging and info
        cube_data (np.ndarray): The reconstructed 3D data cube [λ, y, x]
        wavelength_grid (np.ndarray): Wavelength axis of the cube
        spatial_grid_x (np.ndarray): X spatial coordinates
        spatial_grid_y (np.ndarray): Y spatial coordinates
        wcs (WCS): World coordinate system for the cube
    """
    
    def __init__(self, fiber_map_path: Optional[str] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize the CubeConstructor.
        
        Parameters:
            fiber_map_path (str, optional): Path to fiber mapping file. If None, uses default.
            logger (Logger, optional): Logger instance. If None, creates new one.
        """
        # Set up fiber mapping
        if fiber_map_path is None:
            self.fiber_map_path = os.path.join(LUT_DIR, 'LLAMAS_FiberMap_rev02.dat')
        else:
            self.fiber_map_path = fiber_map_path
            
        # Load fiber mapping table
        try:
            self.fibermap_lut = Table.read(self.fiber_map_path, format='ascii.fixed_width')
        except Exception as e:
            raise FileNotFoundError(f"Could not load fiber map from {self.fiber_map_path}: {e}")
        
        # Set up logging
        if logger is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.logger = setup_logger(__name__, f'CubeConstruct_{timestamp}.log')
        else:
            self.logger = logger
            
        # Initialize cube data attributes
        self.cube_data = None
        self.wavelength_grid = None
        self.spatial_grid_x = None
        self.spatial_grid_y = None
        self.wcs = None
        
        self.logger.info("CubeConstructor initialized successfully")
    
    def load_extraction_data(self, extraction_file: str) -> Tuple[List[ExtractLlamas], List[Dict]]:
        """
        Load extraction data from a pickle file.
        
        Parameters:
            extraction_file (str): Path to the extraction pickle file
            
        Returns:
            tuple: (extractions_list, metadata_list)
        """
        try:
            if isinstance(extraction_file, str):
                self.logger.info(f'Loading extraction data from {extraction_file}')
                extract_data = ExtractLlamas.loadExtraction(extraction_file)
                
                if isinstance(extract_data, dict):
                    extractions = extract_data.get('extractions', [])
                    metadata = extract_data.get('metadata', [])
                else:
                    # Handle case where loadExtraction returns tuple
                    extractions, metadata = extract_data
                    
            else:
                raise TypeError(f"Expected string path, got {type(extraction_file)}")
                
            self.logger.info(f'Loaded {len(extractions)} extraction objects')
            return extractions, metadata
            
        except Exception as e:
            self.logger.error(f'Failed to load extraction data: {e}')
            raise
    
    def get_fiber_coordinates(self, benchside: str, fiber_num: int) -> Tuple[float, float]:
        """
        Get the physical x,y coordinates for a given fiber.
        
        Parameters:
            benchside (str): Bench and side identifier (e.g., '1A', '2B')
            fiber_num (int): Fiber number
            
        Returns:
            tuple: (x, y) coordinates, or (-1, -1) if not found
        """
        try:
            return FiberMap_LUT(benchside, fiber_num)
        except Exception:
            # Fallback to direct table lookup
            fiber_row = self.fibermap_lut[
                np.logical_and(self.fibermap_lut['bench'] == benchside, 
                              self.fibermap_lut['fiber'] == fiber_num)
            ]
            if len(fiber_row) > 0:
                return float(fiber_row['xpos'][0]), float(fiber_row['ypos'][0])
            else:
                return -1.0, -1.0
    
    def map_pixel_to_sky(self, benchside: str, fiber_num: int, pixel_x: int, pixel_y: int, 
                        wavelength: float) -> Tuple[float, float]:
        """
        Map detector pixel coordinates to sky coordinates.
        
        This is a placeholder function for future implementation of pixel-level
        spatial mapping that accounts for:
        - Fiber trace spatial profiles across the detector
        - Wavelength-dependent fiber positions  
        - Detector distortion corrections
        - Astrometric calibration to sky coordinates
        
        Parameters:
            benchside (str): Bench and side identifier (e.g., '1A', '2B')
            fiber_num (int): Fiber number
            pixel_x (int): Detector pixel X coordinate
            pixel_y (int): Detector pixel Y coordinate
            wavelength (float): Wavelength in Angstroms
            
        Returns:
            tuple: (ra, dec) sky coordinates in degrees
            
        TODO: Implement pixel-to-sky mapping with:
            1. Load fiber trace information from trace files
            2. Calculate sub-fiber spatial offset from pixel position
            3. Apply wavelength-dependent corrections
            4. Transform IFU focal plane coordinates to sky coordinates
            5. Account for detector distortions and optical effects
        """
        # PLACEHOLDER IMPLEMENTATION
        # Currently returns fiber center position - needs full implementation
        self.logger.warning("map_pixel_to_sky is not fully implemented - returning fiber center")
        
        # Get fiber center coordinates (in IFU focal plane units)
        fiber_x, fiber_y = self.get_fiber_coordinates(benchside, fiber_num)
        
        if fiber_x == -1 or fiber_y == -1:
            return np.nan, np.nan
            
        # TODO: Replace with proper implementation
        # For now, just return fiber center as placeholder sky coordinates
        # This should be replaced with:
        # 1. spatial_offset = calculate_spatial_offset_from_pixel(pixel_x, pixel_y, fiber_traces)
        # 2. wavelength_correction = apply_wavelength_dependent_corrections(wavelength)
        # 3. focal_plane_coords = fiber_coords + spatial_offset + wavelength_correction
        # 4. ra, dec = transform_to_sky_coordinates(focal_plane_coords, astrometric_solution)
        
        placeholder_ra = fiber_x / 3600.0   # Convert to rough degree equivalent
        placeholder_dec = fiber_y / 3600.0  # Convert to rough degree equivalent
        
        return placeholder_ra, placeholder_dec
    
    def map_fiber_to_sky(self, benchside: str, fiber_num: int) -> Tuple[float, float]:
        """
        Map fiber center coordinates to sky coordinates.
        
        This provides fiber-level sky mapping as an intermediate step before
        full pixel-level mapping is implemented.
        
        Parameters:
            benchside (str): Bench and side identifier (e.g., '1A', '2B')
            fiber_num (int): Fiber number
            
        Returns:
            tuple: (ra, dec) sky coordinates in degrees for fiber center
            
        TODO: Implement proper astrometric calibration
        """
        # Get fiber coordinates in IFU focal plane
        fiber_x, fiber_y = self.get_fiber_coordinates(benchside, fiber_num)
        
        if fiber_x == -1 or fiber_y == -1:
            return np.nan, np.nan
        
        # TODO: Apply proper astrometric transformation
        # This is a placeholder - should use actual WCS transformation
        # from IFU focal plane coordinates to sky coordinates
        
        # Placeholder conversion (needs proper astrometric solution)
        ra = fiber_x / 3600.0   # Rough conversion to degrees
        dec = fiber_y / 3600.0  # Rough conversion to degrees
        
        return ra, dec
    
    def create_wavelength_grid(self, extractions: List[ExtractLlamas], 
                              wavelength_range: Optional[Tuple[float, float]] = None,
                              dispersion: Optional[float] = None) -> np.ndarray:
        """
        Create a common wavelength grid for the cube.
        
        Parameters:
            extractions (List[ExtractLlamas]): List of extraction objects
            wavelength_range (tuple, optional): (min_wave, max_wave) in Angstroms
            dispersion (float, optional): Wavelength dispersion in Angstroms/pixel
            
        Returns:
            np.ndarray: Common wavelength grid
        """
        # Collect all wavelength arrays
        all_wavelengths = []
        for extraction in extractions:
            if hasattr(extraction, 'wavelength') and extraction.wavelength is not None:
                all_wavelengths.extend(extraction.wavelength.flatten())
        
        if not all_wavelengths:
            self.logger.warning("No wavelength information found in extractions")
            # Create a default wavelength grid based on typical LLAMAS range
            if wavelength_range is None:
                wavelength_range = (3500.0, 9500.0)  # Typical LLAMAS range
            if dispersion is None:
                dispersion = 1.0  # 1 Angstrom per pixel
            
            n_pixels = int((wavelength_range[1] - wavelength_range[0]) / dispersion)
            wavelength_grid = np.linspace(wavelength_range[0], wavelength_range[1], n_pixels)
        else:
            # Determine wavelength range from data
            min_wave = np.min(all_wavelengths)
            max_wave = np.max(all_wavelengths)
            
            if wavelength_range is not None:
                min_wave = max(min_wave, wavelength_range[0])
                max_wave = min(max_wave, wavelength_range[1])
            
            # Estimate dispersion if not provided
            if dispersion is None:
                # Use median dispersion from extractions
                dispersions = []
                for extraction in extractions:
                    if hasattr(extraction, 'wavelength') and extraction.wavelength is not None:
                        wave_diff = np.diff(extraction.wavelength)
                        if len(wave_diff) > 0:
                            dispersions.extend(wave_diff.flatten())
                
                if dispersions:
                    dispersion = np.median(dispersions)
                else:
                    dispersion = 1.0  # Default fallback
            
            # Create wavelength grid
            n_pixels = int((max_wave - min_wave) / dispersion)
            wavelength_grid = np.linspace(min_wave, max_wave, n_pixels)
        
        self.logger.info(f'Created wavelength grid: {len(wavelength_grid)} pixels, '
                        f'{wavelength_grid[0]:.1f} - {wavelength_grid[-1]:.1f} Å, '
                        f'dispersion: {dispersion:.3f} Å/pixel')
        
        return wavelength_grid
    
    def create_spatial_grid(self, extractions: List[ExtractLlamas], metadata: List[Dict],
                           spatial_sampling: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spatial grids for the cube based on fiber positions.
        
        Parameters:
            extractions (List[ExtractLlamas]): List of extraction objects
            metadata (List[Dict]): Corresponding metadata
            spatial_sampling (float): Spatial sampling in units (default: 1.0)
            
        Returns:
            tuple: (x_grid, y_grid) spatial coordinate arrays
        """
        # Collect all fiber positions
        x_positions = []
        y_positions = []
        
        for extraction, meta in zip(extractions, metadata):
            bench = extraction.bench
            side = extraction.side
            benchside = f'{bench}{side}'
            
            nfibers = extraction.counts.shape[0] if extraction.counts.ndim > 1 else 1
            
            for fiber_num in range(nfibers):
                x, y = self.get_fiber_coordinates(benchside, fiber_num)
                if x != -1 and y != -1:
                    x_positions.append(x)
                    y_positions.append(y)
        
        if not x_positions:
            self.logger.warning("No valid fiber positions found, using default grid")
            x_grid = np.arange(0, 46)  # Default LLAMAS field size
            y_grid = np.arange(0, 43)
        else:
            # Create grids based on fiber positions
            x_min, x_max = np.min(x_positions), np.max(x_positions)
            y_min, y_max = np.min(y_positions), np.max(y_positions)
            
            # Add padding
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            
            x_min -= x_padding
            x_max += x_padding
            y_min -= y_padding
            y_max += y_padding
            
            # Create grids
            n_x = int((x_max - x_min) / spatial_sampling) + 1
            n_y = int((y_max - y_min) / spatial_sampling) + 1
            
            x_grid = np.linspace(x_min, x_max, n_x)
            y_grid = np.linspace(y_min, y_max, n_y)
        
        self.logger.info(f'Created spatial grids: X={len(x_grid)} pixels, Y={len(y_grid)} pixels')
        
        return x_grid, y_grid
    
    def interpolate_spectrum(self, wavelength_in: np.ndarray, flux_in: np.ndarray, 
                           wavelength_out: np.ndarray) -> np.ndarray:
        """
        Interpolate a spectrum onto a new wavelength grid.
        
        Parameters:
            wavelength_in (np.ndarray): Input wavelength array
            flux_in (np.ndarray): Input flux array
            wavelength_out (np.ndarray): Output wavelength grid
            
        Returns:
            np.ndarray: Interpolated flux array
        """
        if len(wavelength_in) != len(flux_in):
            raise ValueError("Wavelength and flux arrays must have same length")
        
        # Remove NaN values
        valid_mask = np.isfinite(wavelength_in) & np.isfinite(flux_in)
        if not np.any(valid_mask):
            return np.full_like(wavelength_out, np.nan)
        
        wave_clean = wavelength_in[valid_mask]
        flux_clean = flux_in[valid_mask]
        
        # Sort by wavelength
        sort_idx = np.argsort(wave_clean)
        wave_clean = wave_clean[sort_idx]
        flux_clean = flux_clean[sort_idx]
        
        # Interpolate
        flux_out = np.interp(wavelength_out, wave_clean, flux_clean, 
                           left=np.nan, right=np.nan)
        
        return flux_out
    
    def construct_cube(self, extraction_file: str, 
                      wavelength_range: Optional[Tuple[float, float]] = None,
                      dispersion: Optional[float] = None,
                      spatial_sampling: float = 1.0,
                      smooth_sigma: Optional[float] = None) -> np.ndarray:
        """
        Construct the 3D IFU data cube from extraction data.
        
        Parameters:
            extraction_file (str): Path to extraction data file
            wavelength_range (tuple, optional): (min_wave, max_wave) in Angstroms
            dispersion (float, optional): Wavelength dispersion in Angstroms/pixel
            spatial_sampling (float): Spatial sampling in units
            apply_bias_correction (bool): Apply quartile bias correction
            remove_stripes (bool): Remove striping patterns
            smooth_sigma (float, optional): Gaussian smoothing sigma for cube
            
        Returns:
            np.ndarray: 3D data cube with shape [wavelength, y, x]
        """
        self.logger.info("Starting cube construction...")
        
        # Load extraction data
        extractions, metadata = self.load_extraction_data(extraction_file)
        
        # Create wavelength and spatial grids
        self.wavelength_grid = self.create_wavelength_grid(extractions, wavelength_range, dispersion)
        self.spatial_grid_x, self.spatial_grid_y = self.create_spatial_grid(
            extractions, metadata, spatial_sampling)
        
        # Initialize cube
        cube_shape = (len(self.wavelength_grid), len(self.spatial_grid_y), len(self.spatial_grid_x))
        self.cube_data = np.full(cube_shape, np.nan)
        
        self.logger.info(f'Initialized cube with shape: {cube_shape}')
        
        # Process each extraction
        for extraction, meta in zip(extractions, metadata):
            self._process_extraction(extraction, meta)
        
        
        self.logger.info("Cube construction completed")
        return self.cube_data
    
    def _process_extraction(self, extraction: ExtractLlamas, metadata: Dict):
        """
        Process a single extraction object and add to the cube.
        
        Parameters:
            extraction (ExtractLlamas): Extraction object
            metadata (Dict): Extraction metadata
            apply_bias_correction (bool): Whether to apply bias correction
        """
        bench = extraction.bench
        side = extraction.side
        benchside = f'{bench}{side}'
        
        counts = extraction.counts
        wavelength = extraction.wavelength if hasattr(extraction, 'wavelength') else None
        
        if counts.ndim == 1:
            counts = counts.reshape(1, -1)
        
        nfibers, nspec = counts.shape
        
        self.logger.debug(f'Processing {benchside}: {nfibers} fibers, {nspec} spectral pixels')
        
        for fiber_num in range(nfibers):
            # Get fiber coordinates
            x, y = self.get_fiber_coordinates(benchside, fiber_num)
            if x == -1 or y == -1:
                continue
            
            # Get fiber spectrum
            fiber_spectrum = counts[fiber_num]
            
            
            # Get wavelength for this fiber
            if wavelength is not None:
                if wavelength.ndim > 1:
                    fiber_wavelength = wavelength[fiber_num]
                else:
                    fiber_wavelength = wavelength
                
                # Interpolate onto common wavelength grid
                interpolated_spectrum = self.interpolate_spectrum(
                    fiber_wavelength, fiber_spectrum, self.wavelength_grid)
            else:
                # No wavelength calibration available, assume linear mapping
                interpolated_spectrum = np.interp(
                    np.linspace(0, len(fiber_spectrum)-1, len(self.wavelength_grid)),
                    np.arange(len(fiber_spectrum)), fiber_spectrum)
            
            # Find nearest spatial grid points
            x_idx = np.argmin(np.abs(self.spatial_grid_x - x))
            y_idx = np.argmin(np.abs(self.spatial_grid_y - y))
            
            # Add to cube (replace NaN or add to existing)
            current_values = self.cube_data[:, y_idx, x_idx]
            mask_nan = np.isnan(current_values)
            
            # For NaN locations, replace with new values
            self.cube_data[mask_nan, y_idx, x_idx] = interpolated_spectrum[mask_nan]
            
            # For non-NaN locations, average with existing values
            mask_valid = ~mask_nan & np.isfinite(interpolated_spectrum)
            if np.any(mask_valid):
                self.cube_data[mask_valid, y_idx, x_idx] = (
                    self.cube_data[mask_valid, y_idx, x_idx] + interpolated_spectrum[mask_valid]) / 2
    
    def create_wcs(self, reference_coord: Optional[Tuple[float, float]] = None) -> WCS:
        """
        Create a World Coordinate System for the cube.
        
        Parameters:
            reference_coord (tuple, optional): (RA, Dec) reference coordinates in degrees
            
        Returns:
            WCS: World coordinate system object
        """
        wcs = WCS(naxis=3)
        
        # Spatial axes (assume small field, use linear approximation)
        if reference_coord is not None:
            ra_ref, dec_ref = reference_coord
        else:
            ra_ref, dec_ref = 0.0, 0.0  # Placeholder
        
        wcs.wcs.crpix = [len(self.spatial_grid_x)//2, len(self.spatial_grid_y)//2, 1]
        wcs.wcs.cdelt = [
            (self.spatial_grid_x[-1] - self.spatial_grid_x[0]) / len(self.spatial_grid_x) / 3600.0,  # Convert to degrees
            (self.spatial_grid_y[-1] - self.spatial_grid_y[0]) / len(self.spatial_grid_y) / 3600.0,
            self.wavelength_grid[1] - self.wavelength_grid[0]
        ]
        wcs.wcs.crval = [ra_ref, dec_ref, self.wavelength_grid[0]]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
        wcs.wcs.cunit = ['deg', 'deg', 'Angstrom']
        
        self.wcs = wcs
        return wcs
    
    def save_cube(self, output_path: str, reference_coord: Optional[Tuple[float, float]] = None,
                  header_info: Optional[Dict] = None) -> str:
        """
        Save the reconstructed cube to a FITS file.
        
        Parameters:
            output_path (str): Output file path
            reference_coord (tuple, optional): (RA, Dec) reference coordinates
            header_info (dict, optional): Additional header information
            
        Returns:
            str: Path to saved file
        """
        if self.cube_data is None:
            raise ValueError("No cube data to save. Run construct_cube() first.")
        
        self.logger.info(f'Saving cube to {output_path}')
        
        # Create WCS
        wcs = self.create_wcs(reference_coord)
        
        # Create primary HDU
        primary_hdu = fits.PrimaryHDU(data=self.cube_data.astype(np.float32))
        
        # Add WCS to header
        primary_hdu.header.update(wcs.to_header())
        
        # Add additional header information
        primary_hdu.header['BUNIT'] = 'counts'
        primary_hdu.header['ORIGIN'] = 'LLAMAS Cube Constructor'
        primary_hdu.header['DATE'] = datetime.now().isoformat()
        primary_hdu.header['NAXIS1'] = len(self.spatial_grid_x)
        primary_hdu.header['NAXIS2'] = len(self.spatial_grid_y) 
        primary_hdu.header['NAXIS3'] = len(self.wavelength_grid)
        primary_hdu.header['CRVAL3'] = self.wavelength_grid[0]
        primary_hdu.header['CDELT3'] = self.wavelength_grid[1] - self.wavelength_grid[0]
        primary_hdu.header['COMMENT'] = 'LLAMAS IFU Data Cube [wavelength, y, x]'
        
        if header_info:
            for key, value in header_info.items():
                primary_hdu.header[key] = value
        
        # Create HDU list and save
        hdul = fits.HDUList([primary_hdu])
        
        # Add wavelength extension
        wave_hdu = fits.ImageHDU(data=self.wavelength_grid, name='WAVELENGTH')
        wave_hdu.header['BUNIT'] = 'Angstrom'
        hdul.append(wave_hdu)
        
        # Add spatial coordinate extensions
        x_hdu = fits.ImageHDU(data=self.spatial_grid_x, name='XCOORD')
        y_hdu = fits.ImageHDU(data=self.spatial_grid_y, name='YCOORD')
        hdul.append(x_hdu)
        hdul.append(y_hdu)
        
        # Save to file
        full_path = os.path.join(OUTPUT_DIR, output_path) if not os.path.isabs(output_path) else output_path
        hdul.writeto(full_path, overwrite=True)
        
        self.logger.info(f'Cube saved successfully to {full_path}')
        return full_path
    
    def extract_spectrum(self, x: float, y: float, aperture_radius: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract a spectrum at a given spatial position.
        
        Parameters:
            x (float): X coordinate
            y (float): Y coordinate  
            aperture_radius (float): Aperture radius for extraction
            
        Returns:
            tuple: (wavelength, flux) arrays
        """
        if self.cube_data is None:
            raise ValueError("No cube data available. Run construct_cube() first.")
        
        # Find spatial indices within aperture
        x_indices = np.where(np.abs(self.spatial_grid_x - x) <= aperture_radius)[0]
        y_indices = np.where(np.abs(self.spatial_grid_y - y) <= aperture_radius)[0]
        
        if len(x_indices) == 0 or len(y_indices) == 0:
            self.logger.warning(f"No data found at position ({x}, {y})")
            return self.wavelength_grid, np.full_like(self.wavelength_grid, np.nan)
        
        # Extract and sum spectra within aperture
        spectrum = np.nansum(self.cube_data[:, np.ix_(y_indices, x_indices)], axis=(1, 2))
        
        return self.wavelength_grid, spectrum
    
    def create_white_light_image(self, wavelength_range: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Create a white light image by collapsing the spectral dimension.
        
        Parameters:
            wavelength_range (tuple, optional): (min_wave, max_wave) for integration
            
        Returns:
            np.ndarray: 2D white light image
        """
        if self.cube_data is None:
            raise ValueError("No cube data available. Run construct_cube() first.")
        
        if wavelength_range is not None:
            wave_mask = ((self.wavelength_grid >= wavelength_range[0]) & 
                        (self.wavelength_grid <= wavelength_range[1]))
            cube_subset = self.cube_data[wave_mask]
        else:
            cube_subset = self.cube_data
        
        white_light = np.nansum(cube_subset, axis=0)
        return white_light
    
    def plot_spectrum(self, x: float, y: float, aperture_radius: float = 1.0, 
                     save_path: Optional[str] = None):
        """
        Plot an extracted spectrum.
        
        Parameters:
            x (float): X coordinate
            y (float): Y coordinate
            aperture_radius (float): Aperture radius for extraction
            save_path (str, optional): Path to save the plot
        """
        wavelength, flux = self.extract_spectrum(x, y, aperture_radius)
        
        plt.figure(figsize=(12, 6))
        plt.plot(wavelength, flux, 'b-', linewidth=1)
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Flux (counts)')
        plt.title(f'Extracted Spectrum at ({x:.1f}, {y:.1f})')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f'Spectrum plot saved to {save_path}')
        
        plt.show()
    
    def get_cube_info(self) -> Dict:
        """
        Get information about the constructed cube.
        
        Returns:
            dict: Dictionary containing cube information
        """
        if self.cube_data is None:
            return {"status": "No cube data available"}
        
        info = {
            "cube_shape": self.cube_data.shape,
            "wavelength_range": (self.wavelength_grid[0], self.wavelength_grid[-1]),
            "wavelength_dispersion": self.wavelength_grid[1] - self.wavelength_grid[0],
            "spatial_range_x": (self.spatial_grid_x[0], self.spatial_grid_x[-1]),
            "spatial_range_y": (self.spatial_grid_y[0], self.spatial_grid_y[-1]),
            "spatial_sampling": self.spatial_grid_x[1] - self.spatial_grid_x[0],
            "total_pixels": self.cube_data.size,
            "valid_pixels": np.sum(np.isfinite(self.cube_data)),
            "data_range": (np.nanmin(self.cube_data), np.nanmax(self.cube_data))
        }
        
        return info