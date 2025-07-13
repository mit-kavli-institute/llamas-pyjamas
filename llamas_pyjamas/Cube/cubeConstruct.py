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
            self.fiber_map_path = os.path.join(LUT_DIR, 'LLAMAS_FiberMap_rev04.dat')
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
    
    def load_rss_data(self, rss_file: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Load extraction data from RSS FITS file format.

        Parameters:
            rss_file (str): Path to RSS FITS file

        Returns:
            Tuple[List[Dict], List[Dict]]: List of extraction data dictionaries and metadata
        """
        self.logger.info(f'Loading RSS data from {rss_file}')
        extractions = []
        metadata = []

        with fits.open(rss_file) as hdul:
            # First locate all TABLE extensions to get BENCHSIDE information
            table_data = {}
            for i, hdu in enumerate(hdul):
                extname = hdu.header.get('EXTNAME', '').upper()
                if extname.startswith('TABLE_'):
                    channel = extname[6:].lower()
                    if hdu.data is not None:
                        table_data[channel] = hdu.data

            # Now process the SCI/ERR extensions
            i = 1
            while i + 1 < len(hdul):
                sci_hdu = hdul[i]
                err_hdu = hdul[i+1] if i+1 < len(hdul) else None

                sci_extname = sci_hdu.header.get('EXTNAME', '').upper()
                if sci_extname.startswith('SCI_'):
                    channel = sci_extname[4:].lower()

                    # Get the corresponding table data for this channel
                    channel_table = table_data.get(channel)

                    if sci_hdu.data is not None:
                        extraction = {
                            'flux': sci_hdu.data,
                            'error': err_hdu.data if err_hdu is not None and err_hdu.data is not None else None,
                            'channel': channel,
                            'benchside': '',  # We'll populate this from the TABLE data
                            'wavelength': None
                        }

                        # Use BENCHSIDE from the table
                        if channel_table is not None:
                            # BENCHSIDE is available in the table data
                            benchside = channel_table['BENCHSIDE'][0]
                            # Convert bytes to string if needed
                            if isinstance(benchside, bytes):
                                benchside = benchside.decode('utf-8').strip()
                            extraction['benchside'] = benchside

                        extractions.append(extraction)
                        meta = {
                            'benchside': extraction['benchside'],
                            'channel': extraction['channel'],
                            'nfibers': extraction['flux'].shape[0] if 'flux' in extraction and extraction['flux'].ndim > 1 else 0
                        }
                        metadata.append(meta)

                i += 2  # Skip to next SCI/ERR pair

        return extractions, metadata

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
        # print(f"mapping sky using pos: {fiber_x}, {fiber_y}")  # Debug output
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
    
    def map_fiber_to_sky(self, benchside: str, fiber_num: int, reference_coord: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Map fiber center coordinates to sky coordinates.

        This provides fiber-level sky mapping using proper astrometric calibration
        with the reference RA/DEC from the primary header.

        Parameters:
            benchside (str): Bench and side identifier (e.g., '1A', '2B')
            fiber_num (int): Fiber number
            reference_coord (tuple, optional): Reference (RA, Dec) in degrees

        Returns:
            tuple: (ra, dec) sky coordinates in degrees for fiber center
        """
        # Get fiber coordinates in IFU focal plane (arcseconds)
        fiber_x, fiber_y = self.get_fiber_coordinates(benchside, fiber_num)

        if fiber_x == -1 or fiber_y == -1:
            print(f"Invalid fiber coordinates for {benchside} fiber {fiber_num}: ({fiber_x}, {fiber_y})")
            return np.nan, np.nan

        # Apply proper astrometric transformation using reference coordinates
        if reference_coord is not None:
            ra_ref, dec_ref = reference_coord

            # Check if reference coordinates are valid
            if ra_ref is None or dec_ref is None:
                self.logger.warning(f"Invalid reference coordinates: {reference_coord}")
                return np.nan, np.nan

            try:
                # Convert fiber coordinates from arcseconds to degrees
                # and apply offset from reference position
                ra = ra_ref + (fiber_x / 3600.0) / np.cos(np.radians(dec_ref))
                dec = dec_ref + (fiber_y / 3600.0)
            except (TypeError, ValueError) as e:
                self.logger.error(f"Error in sky coordinate conversion: {e}")
                self.logger.error(f"Reference coords: RA={ra_ref} ({type(ra_ref)}), DEC={dec_ref} ({type(dec_ref)})")
                self.logger.error(f"Fiber coords: X={fiber_x}, Y={fiber_y}")
                return np.nan, np.nan
        else:
            # Fallback to simple conversion if no reference provided
            ra = fiber_x / 3600.0
            dec = fiber_y / 3600.0

        return ra, dec
    
    def create_wavelength_grid(self, extractions: List[Dict], 
                              wavelength_range: Optional[Tuple[float, float]] = None,
                              dispersion: Optional[float] = None) -> np.ndarray:
        """
        Create a common wavelength grid for the cube (RSS version).
        
        Parameters:
            extractions (List[Dict]): List of RSS extraction dictionaries
            wavelength_range (tuple, optional): (min_wave, max_wave) in Angstroms
            dispersion (float, optional): Wavelength dispersion in Angstroms/pixel
        
        Returns:
            np.ndarray: Common wavelength grid
        """
        all_wavelengths = []
        for extraction in extractions:
            if extraction.get('wavelength') is not None:
                all_wavelengths.extend(np.ravel(extraction['wavelength']))
        if not all_wavelengths:
            self.logger.warning("No wavelength information found in RSS extractions")
            if wavelength_range is None:
                wavelength_range = (3500.0, 9500.0)
            if dispersion is None:
                dispersion = 1.0
            n_pixels = int((wavelength_range[1] - wavelength_range[0]) / dispersion)
            wavelength_grid = np.linspace(wavelength_range[0], wavelength_range[1], n_pixels)
        else:
            min_wave = np.min(all_wavelengths)
            max_wave = np.max(all_wavelengths)
            if wavelength_range is not None:
                min_wave = max(min_wave, wavelength_range[0])
                max_wave = min(max_wave, wavelength_range[1])
            if dispersion is None:
                dispersions = []
                for extraction in extractions:
                    wv = extraction.get('wavelength')
                    if wv is not None:
                        wave_diff = np.diff(wv)
                        if len(wave_diff) > 0:
                            dispersions.extend(wave_diff.flatten())
                if dispersions:
                    dispersion = np.median(dispersions)
                else:
                    dispersion = 1.0
            n_pixels = int((max_wave - min_wave) / dispersion)
            wavelength_grid = np.linspace(min_wave, max_wave, n_pixels)
        self.logger.info(f'Created wavelength grid: {len(wavelength_grid)} pixels, '
                        f'{wavelength_grid[0]:.1f} - {wavelength_grid[-1]:.1f} Å, '
                        f'dispersion: {dispersion:.3f} Å/pixel')
        return wavelength_grid

    def create_spatial_grid(self, extractions: List[Dict], metadata: List[Dict],
                           spatial_sampling: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create spatial grids for the cube based on fiber positions (RSS version).
        
        Parameters:
            extractions (List[Dict]): List of RSS extraction dictionaries
            metadata (List[Dict]): Corresponding metadata
            spatial_sampling (float): Spatial sampling in units (default: 1.0)
        
        Returns:
            tuple: (x_grid, y_grid) spatial coordinate arrays
        """
        x_positions = []
        y_positions = []
        for extraction, meta in zip(extractions, metadata):
            benchside = extraction.get('benchside', '')
            if not benchside and 'bench' in extraction and 'side' in extraction:
            # Fallback for backward compatibility
                benchside = f"{extraction['bench']}{extraction['side']}"
            
            flux = extraction['flux']
            nfibers = flux.shape[0] if flux.ndim > 1 else 1
            for fiber_num in range(nfibers):
                x, y = self.get_fiber_coordinates(benchside, fiber_num)
                if x != -1 and y != -1:
                    x_positions.append(x)
                    y_positions.append(y)
        if not x_positions:
            self.logger.warning("No valid fiber positions found, using default grid")
            x_grid = np.arange(0, 46)
            y_grid = np.arange(0, 43)
        else:
            x_min, x_max = np.min(x_positions), np.max(x_positions)
            y_min, y_max = np.min(y_positions), np.max(y_positions)
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            x_min -= x_padding
            x_max += x_padding
            y_min -= y_padding
            y_max += y_padding
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
    
    def construct_cube_from_rss(self, rss_file: str, wavelength_range: Optional[Tuple[float, float]] = None,
                           dispersion: float = 1.0, spatial_sampling: float = 0.75,
                           reference_coord: Optional[Tuple[float, float]] = None) -> Dict[str, np.ndarray]:
        """
        Construct IFU cubes from all channels in an RSS FITS file.

        This method loads all channels from the RSS file and creates one cube per channel.
        It uses the fiber map to position each spectrum in the 3D cube based on
        its x,y spatial coordinates.

        Parameters:
            rss_file (str): Path to RSS FITS file
            wavelength_range (tuple, optional): Min/max wavelength range
            dispersion (float): Wavelength dispersion in Angstroms/pixel
            spatial_sampling (float): Spatial sampling in arcsec/pixel (default: 0.75)
            reference_coord (tuple, optional): Reference RA/Dec for WCS

        Returns:
            Dict[str, np.ndarray]: Dictionary of channel cubes {channel: cube_data}
        """
        # Load all channels from the RSS file
        channels_data = self.load_rss_channels(rss_file)
        if not channels_data:
            self.logger.error("No channels found in RSS file")
            return None

        self.logger.info(f"Found channels: {list(channels_data.keys())}")

        # Extract RA and DEC from the primary header
        with fits.open(rss_file) as hdul:
            primary_header = hdul[0].header
            ra_ref = primary_header.get('RA')
            dec_ref = primary_header.get('DEC')

            # Check if RA and DEC are valid
            if ra_ref is None or dec_ref is None or (isinstance(ra_ref, str) and not ra_ref.strip()) or (isinstance(dec_ref, str) and not dec_ref.strip()):
                self.logger.warning("No valid RA/DEC found in primary header, using local coordinates")
                ref_coords = None
            else:
                # Convert to float if they're strings
                if isinstance(ra_ref, str):
                    ra_ref = float(ra_ref)
                if isinstance(dec_ref, str):
                    dec_ref = float(dec_ref)

                self.logger.info(f"Using header reference coordinates: RA={ra_ref}, DEC={dec_ref}")
                ref_coords = (ra_ref, dec_ref)

        # Override with provided reference_coord if given
        if reference_coord is not None:
            ref_coords = reference_coord
            self.logger.info(f"Using provided reference coordinates: RA={ref_coords[0]}, DEC={ref_coords[1]}")

        # Construct a cube for each channel
        channel_cubes = {}
        for channel in channels_data.keys():
            self.logger.info(f"Constructing cube for channel: {channel}")
            cube = self.construct_cube_from_rss_channel(
                rss_file=rss_file,
                channel=channel,
                wavelength_range=wavelength_range,
                dispersion=dispersion,
                spatial_sampling=spatial_sampling,
                reference_coord=ref_coords
            )
            if cube is not None:
                channel_cubes[channel] = cube

        if not channel_cubes:
            self.logger.error("Failed to construct any channel cubes")
            return None

        self.logger.info(f"Successfully constructed cubes for channels: {list(channel_cubes.keys())}")
        return channel_cubes

    def load_rss_channels(self, rss_file: str) -> Dict[str, Dict[str, any]]:
        """
        Load all channel data from an RSS FITS file.
        Returns a dict: {channel: {'flux':..., 'err':..., 'table':...}}
        
        Each channel has a single SCI, ERR, and TABLE extension with all fibers stacked
        into a 2D array with shape (n_fibers, n_pixels).
        """
        self.logger.info(f'Loading channels from RSS file: {rss_file}')
        channels = {}
        with fits.open(rss_file) as hdul:
            for hdu in hdul[1:]:
                extname = hdu.header.get('EXTNAME', '').upper()
                if extname.startswith('SCI_'):
                    channel = extname[4:]
                    channels.setdefault(channel, {})['flux'] = hdu.data.copy() if hdu.data is not None else None
                elif extname.startswith('ERR_'):
                    channel = extname[4:]
                    channels.setdefault(channel, {})['err'] = hdu.data.copy() if hdu.data is not None else None
                elif extname.startswith('TABLE_'):
                    channel = extname[6:]
                    # Make sure to copy the data while the file is open
                    if hdu.data is not None:
                        # Handle conversion from memoryview to structured array
                        try:
                            table_data = np.array(hdu.data)
                        except TypeError:
                            # For memoryview issues, try a different approach
                            column_names = hdu.columns.names
                            table_data = {}
                            for col in column_names:
                                table_data[col] = np.array(hdu.data[col])
                    else:
                        table_data = None
                    channels.setdefault(channel, {})['table'] = table_data
        return channels

    def construct_cube_from_rss_channel(self, rss_file: str, channel: str, wavelength_range: Optional[Tuple[float, float]] = None,
                                        dispersion: float = 1.0, spatial_sampling: float = 0.75,
                                        reference_coord: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Construct IFU cube from a single channel in an RSS FITS file.
        
        Each fiber spectrum in the stacked RSS file is placed at its correct spatial position
        in the output cube, based on the fiber map. Each spaxel has a fixed size of 0.75 arcseconds
        by default.
        
        Parameters:
            rss_file (str): Path to RSS FITS file
            channel (str): Channel identifier (e.g., 'RED', 'GREEN', 'BLUE')
            wavelength_range (tuple, optional): Min/max wavelength range
            dispersion (float): Wavelength dispersion in Angstroms/pixel
            spatial_sampling (float): Spatial sampling in arcsec/pixel (default: 0.75)
            reference_coord (tuple, optional): Reference RA/Dec for WCS
            
        Returns:
            np.ndarray: 3D cube data with shape [wavelength, y, x]
        """
        # Load the channel data from RSS file
        channels = self.load_rss_channels(rss_file)
        if channel not in channels:
            self.logger.error(f"Channel {channel} not found in RSS file.")
            return None
            
        # Get the flux data and table data for this channel
        flux_data = channels[channel]['flux']
        table_data = channels[channel].get('table')
        
        
        if flux_data is None:
            self.logger.error(f"No flux data found for channel {channel}")
            return None
            
        n_fibers, n_pixels = flux_data.shape
        
        
        # Create wavelength grid based on number of pixels
        if wavelength_range is None:
            # Default to pixel indices if no wavelength range specified
            self.wavelength_grid = np.arange(n_pixels) * dispersion
        else:
            min_wave, max_wave = wavelength_range
            self.wavelength_grid = np.linspace(min_wave, max_wave, n_pixels)
        
        # Get spatial coordinates for all fibers
        fiber_positions = []
        
        if table_data is not None:
            # Use table data to get fiber information
            for i in range(n_fibers):
                # benchside = table_data['BENCH'][i] + table_data['SIDE'][i]
                benchside = table_data['BENCHSIDE'][i]
                if isinstance(benchside, bytes):
                        benchside = benchside.decode('utf-8').strip()

                
                fiber_num = table_data['FIBER'][i]
                
                x, y = self.get_fiber_coordinates(benchside, fiber_num)
                if x != -1 and y != -1:
                    # Convert to sky coordinates if reference coordinate is provided
                    if reference_coord is not None:
                        ra, dec = self.map_fiber_to_sky(benchside, fiber_num, reference_coord)
                        if not (np.isnan(ra) or np.isnan(dec)):
                            # Convert back to angular offset from reference for spatial grid
                            ra_ref, dec_ref = reference_coord
                            x_sky = (ra - ra_ref) * 3600.0 * np.cos(np.radians(dec_ref))
                            y_sky = (dec - dec_ref) * 3600.0
                            fiber_positions.append((i, x_sky, y_sky))
                    else:
                        fiber_positions.append((i, x, y))
        
        if not fiber_positions:
            self.logger.error(f"No valid fiber positions found for channel {channel}")
            return None
        
        # Determine spatial extent from fiber positions
        x_coords = [pos[1] for pos in fiber_positions]
        y_coords = [pos[2] for pos in fiber_positions]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add buffer around edges
        buffer = spatial_sampling
        x_min -= buffer
        x_max += buffer
        y_min -= buffer
        y_max += buffer
        
        # Create spatial grid with fixed spaxel size
        n_x = int((x_max - x_min) / spatial_sampling) + 1
        n_y = int((y_max - y_min) / spatial_sampling) + 1
        
        self.spatial_grid_x = np.linspace(x_min, x_max, n_x)
        self.spatial_grid_y = np.linspace(y_min, y_max, n_y)
        
        # Initialize cube with NaNs
        cube_shape = (len(self.wavelength_grid), len(self.spatial_grid_y), len(self.spatial_grid_x))
        self.cube_data = np.full(cube_shape, np.nan)
        
        self.logger.info(f"Initialized cube for channel {channel} with shape: {self.cube_data.shape}")
        
        # Populate the cube with fiber spectra at their correct positions
        for fiber_idx, x, y in fiber_positions:
            # Find the closest grid points
            x_idx = np.argmin(np.abs(self.spatial_grid_x - x))
            y_idx = np.argmin(np.abs(self.spatial_grid_y - y))
            
            # Place the spectrum in the cube
            self.cube_data[:, y_idx, x_idx] = flux_data[fiber_idx]
        
        self.logger.info(f"Cube for channel {channel} constructed successfully")
        return self.cube_data
    
    def create_wcs(self, reference_coord: Optional[Tuple[float, float]] = None, spatial_sampling: float = 0.75) -> WCS:
        """
        Create a World Coordinate System for the cube.
        
        Parameters:
            reference_coord (tuple, optional): (RA, Dec) reference coordinates in degrees
            spatial_sampling (float): Spatial sampling in arcsec/pixel (default: 0.75)
            
        Returns:
            WCS: World coordinate system object
        """
        wcs = WCS(naxis=3)
        
        # Spatial axes using proper spaxel size
        if reference_coord is not None:
            ra_ref, dec_ref = reference_coord
        else:
            ra_ref, dec_ref = 0.0, 0.0  # Placeholder
        
        # Reference pixel at the center of the cube
        wcs.wcs.crpix = [len(self.spatial_grid_x)//2 + 1, len(self.spatial_grid_y)//2 + 1, 1]
        
        # Pixel scale in degrees/pixel (spatial_sampling is in arcsec/pixel)
        pixel_scale_deg = spatial_sampling / 3600.0
        
        wcs.wcs.cdelt = [
            -pixel_scale_deg,  # RA decreases with increasing pixel (standard convention)
            pixel_scale_deg,   # Dec increases with increasing pixel
            self.wavelength_grid[1] - self.wavelength_grid[0]  # Wavelength dispersion
        ]
        
        # Reference coordinate values
        wcs.wcs.crval = [ra_ref, dec_ref, self.wavelength_grid[0]]
        
        # Coordinate types
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
        wcs.wcs.cunit = ['deg', 'deg', 'Angstrom']
        
        self.wcs = wcs
        return wcs
    
    def save_cube(self, output_path: str, reference_coord: Optional[Tuple[float, float]] = None,
                  header_info: Optional[Dict] = None, spatial_sampling: float = 0.75) -> str:
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
        wcs = self.create_wcs(reference_coord, spatial_sampling)
        
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
        primary_hdu.header['SPAXELSC'] = spatial_sampling
        primary_hdu.header['COMMENT'] = 'LLAMAS IFU Data Cube [wavelength, y, x]'
        primary_hdu.header['COMMENT'] = f'Spaxel size: {spatial_sampling} arcsec'
        primary_hdu.header['COMMENT'] = 'Each slice can be summed to create whitelight image'
        primary_hdu.header['COMMENT'] = 'Each spaxel contains a full spectrum'
        
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
    
    def save_channel_cubes(self, channel_cubes: Dict[str, np.ndarray], output_prefix: str, 
                         reference_coord: Optional[Tuple[float, float]] = None,
                         header_info: Optional[Dict] = None, spatial_sampling: float = 0.75) -> Dict[str, str]:
        """
        Save multiple channel cubes to FITS files.
        
        Parameters:
            channel_cubes (Dict[str, np.ndarray]): Dictionary of channel cubes {channel: cube_data}
            output_prefix (str): Prefix for output file paths (channel name will be appended)
            reference_coord (tuple, optional): (RA, Dec) reference coordinates
            header_info (dict, optional): Additional header information
            
        Returns:
            Dict[str, str]: Dictionary of saved file paths {channel: file_path}
        """
        if not channel_cubes:
            raise ValueError("No channel cubes to save.")
        
        saved_paths = {}
        for channel, cube_data in channel_cubes.items():
            # Store the cube data temporarily
            original_cube_data = self.cube_data
            self.cube_data = cube_data
            
            # Create output path with channel name
            output_path = f"{output_prefix}_{channel}.fits"
            
            # Add channel info to header
            channel_header = {}
            if header_info:
                channel_header.update(header_info)
            channel_header['CHANNEL'] = channel
            
            # Save the cube
            file_path = self.save_cube(output_path, reference_coord, channel_header, spatial_sampling)
            saved_paths[channel] = file_path
            
            # Restore original cube data
            self.cube_data = original_cube_data
        
        self.logger.info(f"Saved {len(saved_paths)} channel cubes with prefix: {output_prefix}")
        return saved_paths