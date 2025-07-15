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
        
        This implementation uses the fact that each fiber corresponds to 0.75" on the sky.
        It uses the fiber’s nominal position (from the fiber map) as a base, adds an offset computed
        from the input detector pixel coordinates scaled by 0.75" per pixel, and then translates
        the result into sky coordinates using the RA and DEC from the primary header (stored in the WCS).
        
        Parameters:
            benchside (str): Bench and side identifier (e.g., '1A', '2B')
            fiber_num (int): Fiber number
            pixel_x (int): Detector pixel X offset (in pixels) from the fiber center
            pixel_y (int): Detector pixel Y offset (in pixels) from the fiber center
            wavelength (float): Wavelength in Angstroms (currently not used in the mapping)
            
        Returns:
            tuple: (ra, dec) sky coordinates in degrees
        """
        # Get the nominal fiber center (in IFU focal plane arcsec)
        fiber_x, fiber_y = self.get_fiber_coordinates(benchside, fiber_num)
        if fiber_x == -1 or fiber_y == -1:
            self.logger.warning(f"Invalid fiber coordinates for {benchside} fiber {fiber_num}")
            return np.nan, np.nan

        # Retrieve reference (RA, DEC) from the primary header via the WCS
        if self.wcs is not None and hasattr(self.wcs, 'wcs'):
            ra_ref = self.wcs.wcs.crval[0]
            dec_ref = self.wcs.wcs.crval[1]
        else:
            self.logger.warning("WCS is not available; defaulting to (0.0, 0.0) for reference coordinates")
            ra_ref, dec_ref = 0.0, 0.0

        # Convert the detector pixel offset to arcseconds using the 0.75" scale
        delta_x_arcsec = pixel_x * 0.75
        delta_y_arcsec = pixel_y * 0.75

        # Compute the effective position in the IFU focal plane (arcsec)
        effective_x = fiber_x + delta_x_arcsec
        effective_y = fiber_y + delta_y_arcsec

        # Convert the effective offset (arcsec) to degrees
        ra_offset_deg = effective_x / 3600.0 / np.cos(np.radians(dec_ref))
        dec_offset_deg = effective_y / 3600.0

        ra = ra_ref + ra_offset_deg
        dec = dec_ref + dec_offset_deg

        return ra, dec
    
    def map_fiber_to_sky(self, benchside: str, fiber_num: int, reference_coord: Optional[Tuple[float, float]] = None) -> Tuple[float, float]:
        """
        Map fiber center coordinates to sky coordinates.

        This provides fiber-level sky mapping using proper astrometric calibration
        with the reference RA/DEC from the primary header. Each fiber represents 
        0.75 arcseconds on the sky.

        Parameters:
            benchside (str): Bench and side identifier (e.g., '1A', '2B')
            fiber_num (int): Fiber number
            reference_coord (tuple, optional): Reference (RA, Dec) in degrees

        Returns:
            tuple: (ra, dec) sky coordinates in degrees for fiber center
        """
        # Get fiber coordinates in IFU focal plane coordinates
        fiber_x, fiber_y = self.get_fiber_coordinates(benchside, fiber_num)

        if fiber_x == -1 or fiber_y == -1:
            self.logger.warning(f"Invalid fiber coordinates for {benchside} fiber {fiber_num}")
            return np.nan, np.nan

        # Apply proper astrometric transformation using reference coordinates
        if reference_coord is not None:
            ra_ref, dec_ref = reference_coord

            # Check if reference coordinates are valid
            if ra_ref is None or dec_ref is None:
                self.logger.warning(f"Invalid reference coordinates: {reference_coord}")
                return np.nan, np.nan

            try:
                # Scale the fiber coordinates by 0.75" (the fiber size on sky)
                # The fiber map coordinates are in unit-less values, so we need to convert
                # them to actual angular offsets by scaling by 0.75" per fiber
                x_arcsec = fiber_x * 0.75
                y_arcsec = fiber_y * 0.75
                
                # Convert arcseconds to degrees and apply offset from reference position
                # Use cos(dec) correction for RA to account for spherical projection
                ra = ra_ref + (x_arcsec / 3600.0) / np.cos(np.radians(dec_ref))
                dec = dec_ref + (y_arcsec / 3600.0)
            except (TypeError, ValueError) as e:
                self.logger.error(f"Error in sky coordinate conversion: {e}")
                self.logger.error(f"Reference coords: RA={ra_ref} ({type(ra_ref)}), DEC={dec_ref} ({type(dec_ref)})")
                self.logger.error(f"Fiber coords: X={fiber_x}, Y={fiber_y}")
                return np.nan, np.nan
        else:
            # Fallback to simple conversion if no reference provided
            # Still apply the 0.75" scaling
            ra = (fiber_x * 0.75) / 3600.0
            dec = (fiber_y * 0.75) / 3600.0

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
                # Try to use HIERARCH TEL RA/DEC as fallback if available
                tel_ra = primary_header.get('HIERARCH TEL RA')
                tel_dec = primary_header.get('HIERARCH TEL DEC')

                if tel_ra is not None and tel_dec is not None:
                    # Convert telescope coordinates if they're in string format
                    from astropy.coordinates import SkyCoord
                    from astropy import units as u
                    try:
                        c = SkyCoord(ra=str(tel_ra), dec=str(tel_dec), unit=(u.deg, u.deg))
                        ra_ref = c.ra.deg
                        dec_ref = c.dec.deg
                        self.logger.info(f"Using telescope coordinates: RA={ra_ref}, DEC={dec_ref}")
                        ref_coords = (ra_ref, dec_ref)
                    except Exception as e:
                        self.logger.warning(f"Failed to convert telescope coordinates: {e}")
                        ref_coords = None
                else:
                    ref_coords = None
            else:
                # Convert to float if they're strings
                if isinstance(ra_ref, str):
                    try:
                        ra_ref = float(ra_ref)
                    except ValueError:
                        self.logger.warning(f"Could not convert RA value '{ra_ref}' to float")
                        ra_ref = None

                if isinstance(dec_ref, str):
                    try:
                        dec_ref = float(dec_ref)
                    except ValueError:
                        self.logger.warning(f"Could not convert DEC value '{dec_ref}' to float")
                        dec_ref = None

                if ra_ref is not None and dec_ref is not None:
                    self.logger.info(f"Using header reference coordinates: RA={ra_ref}, DEC={dec_ref}")
                    ref_coords = (ra_ref, dec_ref)
                else:
                    ref_coords = None

        # Override with provided reference_coord if given
        if reference_coord is not None:
            ref_coords = reference_coord
            self.logger.info(f"Using provided reference coordinates: RA={ref_coords[0]}, DEC={ref_coords[1]}")

        # Initialize the WCS with the reference coordinates
        if ref_coords is not None:
            # We're setting a preliminary WCS here, which will be updated for each channel
            wcs = WCS(naxis=3)
            ra_ref, dec_ref = ref_coords

            # Set the pixel scale based on the 0.75" fiber size on sky
            pixel_scale_deg = spatial_sampling / 3600.0  # Convert arcsec to degrees

            # Set up the WCS parameters
            wcs.wcs.crval = [ra_ref, dec_ref, 0.0]  # Wavelength will be updated later
            wcs.wcs.cdelt = [-pixel_scale_deg, pixel_scale_deg, dispersion]  # Negative for RA per convention
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
            wcs.wcs.cunit = ['deg', 'deg', 'Angstrom']

            self.wcs = wcs  # Store the WCS with proper RA/DEC
            self.logger.info(f"Initialized WCS with reference RA={ra_ref}, DEC={dec_ref} and pixel scale={spatial_sampling} arcsec")

        # Construct a cube for each channel
        channel_cubes = {}
        channel_wavelength_grids = {}  # NEW: Store wavelength grid for each channel
        channel_wcs = {}  # NEW: Store WCS for each channel

        # Save original attributes to restore later
        original_wavelength_grid = self.wavelength_grid
        original_wcs = self.wcs
        original_cube_data = self.cube_data

        # Process each channel
        for channel in channels_data.keys():
            self.logger.info(f"Constructing cube for channel: {channel}")

            # Reset wavelength grid for this channel
            self.wavelength_grid = None

            # Construct cube for this channel
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

                # Save the wavelength grid that was created for this channel
                if self.wavelength_grid is not None:
                    channel_wavelength_grids[channel] = self.wavelength_grid.copy()
                    self.logger.info(f"Saved wavelength grid for channel {channel}: " 
                                   f"{self.wavelength_grid[0]:.1f}-{self.wavelength_grid[-1]:.1f} Å")

                # Update the WCS for this channel with the proper wavelength grid
                if ref_coords is not None and self.wavelength_grid is not None:
                    channel_wcs[channel] = self.create_wcs(ref_coords, spatial_sampling)

        # Store the channel-specific data for use when saving
        self.channel_wavelength_grids = channel_wavelength_grids
        self.channel_wcs = channel_wcs

        # Restore original attributes
        self.wavelength_grid = original_wavelength_grid
        self.wcs = original_wcs
        self.cube_data = original_cube_data

        if not channel_cubes:
            self.logger.error("Failed to construct any channel cubes")
            return None

        self.logger.info(f"Successfully constructed cubes for channels: {list(channel_cubes.keys())}")
        return channel_cubes

    def load_rss_channels(self, rss_file: str) -> Dict[str, Dict[str, any]]:
        """
        Load all channel data from an RSS FITS file.
        Returns a dict: {channel: {'flux':..., 'err':..., 'wave':..., 'table':...}}
        
        Each channel is stored in a binary table extension (SPEC_{channel}) with columns:
        - FIBER: Fiber ID number
        - BENCHSIDE: Bench and side identifier
        - EXTNUM: Original extraction object index
        - FLUX: Flux array for each fiber
        - ERROR: Error array for each fiber
        - WAVELENGTH: Wavelength array for each fiber
        - DQ: Data quality array for each fiber
        """
        self.logger.info(f'Loading channels from RSS file: {rss_file}')
        channels = {}
        
        with fits.open(rss_file) as hdul:
            for hdu in hdul[1:]:
                extname = hdu.header.get('EXTNAME', '').upper()
                
                # The new format uses binary tables with SPEC_{CHANNEL} naming
                if extname.startswith('SPEC_'):
                    channel = extname[5:].lower()
                    self.logger.info(f"Found SPEC_{channel} binary table extension")
                    
                    if not isinstance(hdu, fits.BinTableHDU):
                        self.logger.warning(f"SPEC_{channel} is not a binary table. Skipping.")
                        continue
                    
                    # Initialize the channel dictionary if it doesn't exist
                    if channel not in channels:
                        channels[channel] = {}
                    
                    # Extract data from the binary table
                    try:
                        # Get fiber and benchside information
                        fiber_ids = np.array(hdu.data['FIBER'])
                        benchsides = np.array(hdu.data['BENCHSIDE'])
                        
                        # Convert benchsides from bytes to strings if needed
                        if benchsides.dtype.kind == 'S':
                            benchsides = np.array([bs.decode('utf-8').strip() for bs in benchsides])
                        
                        # Get the flux, error, wavelength, and data quality arrays
                        flux = np.array(hdu.data['FLUX'])
                        error = np.array(hdu.data['ERROR'])
                        wavelength = np.array(hdu.data['WAVELENGTH'])
                        dq = np.array(hdu.data['DQ'])
                        
                        # Create a table dictionary for fiber metadata
                        table_data = {
                            'FIBER': fiber_ids,
                            'BENCHSIDE': benchsides,
                            'EXTNUM': np.array(hdu.data['EXTNUM'])
                        }
                        
                        # Store all the extracted data in the channel dictionary
                        channels[channel]['flux'] = flux
                        channels[channel]['err'] = error
                        channels[channel]['wave'] = wavelength
                        channels[channel]['dq'] = dq
                        channels[channel]['table'] = table_data
                        
                        self.logger.info(f"Loaded channel {channel} data with {len(fiber_ids)} fibers")
                    except Exception as e:
                        self.logger.error(f"Error loading data from SPEC_{channel}: {str(e)}")
                
        if not channels:
            self.logger.warning("No channel data found in RSS file. Check if file format is correct.")
            
        return channels

    def construct_cube_from_rss_channel(self, rss_file: str, channel: str, wavelength_range: Optional[Tuple[float, float]] = None,
                                      dispersion: float = 1.0, spatial_sampling: float = 0.75,
                                      reference_coord: Optional[Tuple[float, float]] = None) -> np.ndarray:
        """
        Construct IFU cube from a single channel in an RSS FITS file.
    
        Each fiber spectrum in the stacked RSS file is placed at its correct spatial position
        in the output cube, based on the fiber map. Each spaxel has a fixed size of 0.75 arcseconds
        by default. The wavelength information for each fiber is read directly from the WAVELENGTH
        column in the binary table, allowing for proper wavelength calibration per fiber.
    
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
    
        # Get the flux data and table
        flux_data = channels[channel]['flux']
        table_data = channels[channel].get('table')
        
        # Get the wavelength data for each fiber
        fiber_wavelengths = channels[channel].get('wave')
    
        if flux_data is None:
            self.logger.error(f"No flux data found for channel {channel}")
            return None
    
        n_fibers, n_pixels = flux_data.shape
        self.logger.info(f"Channel {channel} data shape: {flux_data.shape} ({n_fibers} fibers)")
    
        # Determine the wavelength grid for the cube
        # Check if we have per-fiber wavelengths
        if fiber_wavelengths is not None and fiber_wavelengths.shape == flux_data.shape:
            self.logger.info(f"Using per-fiber wavelength arrays for channel {channel}, shape: {fiber_wavelengths.shape}")
            
            # DEBUG: Add diagnostics to understand why only 1794 wavelengths are valid
            total_wavelength_points = n_fibers * n_pixels
            nan_count = np.sum(np.isnan(fiber_wavelengths))
            nan_percentage = (nan_count / total_wavelength_points) * 100
            self.logger.info(f"Wavelength diagnostics: {nan_count} NaN values out of {total_wavelength_points} points ({nan_percentage:.2f}%)")
            
            # Check how many wavelength values are valid per pixel position
            valid_per_position = np.sum(~np.isnan(fiber_wavelengths), axis=0)
            valid_positions = np.sum(valid_per_position > 0)
            self.logger.info(f"Valid wavelength positions: {valid_positions} out of {n_pixels} ({(valid_positions/n_pixels)*100:.2f}%)")
            
            # Show distribution of valid wavelength points
            min_valid = np.min(valid_per_position)
            max_valid = np.max(valid_per_position)
            mean_valid = np.mean(valid_per_position)
            self.logger.info(f"Valid fibers per wavelength position: min={min_valid}, max={max_valid}, mean={mean_valid:.2f}")
            
            # Calculate average wavelength grid from all fibers (removing NaN values)
            # First, find valid wavelength arrays (non-NaN)
            valid_wavelengths = []
            for i in range(n_fibers):
                wave = fiber_wavelengths[i]
                if not np.all(np.isnan(wave)):
                    valid_wavelengths.append(wave)
            
            if valid_wavelengths:
                # If we have valid wavelength arrays, use the median
                valid_array = np.vstack(valid_wavelengths)
                self.wavelength_grid = np.nanmedian(valid_array, axis=0)
                self.logger.info(f"Created median wavelength grid from {len(valid_wavelengths)} valid fiber wavelengths: "
                               f"{self.wavelength_grid[0]:.1f}-{self.wavelength_grid[-1]:.1f} Å")
            else:
                # If no valid wavelength arrays, use a default grid
                self.logger.warning("No valid wavelength arrays found. Using default wavelength grid.")
                self._create_default_wavelength_grid(channel, n_pixels, wavelength_range)
        else:
            # Fall back to default grid if per-fiber wavelengths aren't available
            self.logger.warning("No per-fiber wavelength arrays found. Using default wavelength grid.")
            self._create_default_wavelength_grid(channel, n_pixels, wavelength_range)
    
        # Get spatial coordinates for all fibers
        fiber_positions = []
    
        if table_data is not None:
            # Use table data to get fiber information
            for i in range(n_fibers):
                # Get benchside and fiber number
                benchside = table_data['BENCHSIDE'][i]
                fiber_num = table_data['FIBER'][i]
    
                # Get physical fiber coordinates in the IFU focal plane
                x, y = self.get_fiber_coordinates(benchside, fiber_num)
    
                if x != -1 and y != -1:
                    # Apply the 0.75" scale to convert from fiber map units to arcseconds
                    x_arcsec = x * 0.75
                    y_arcsec = y * 0.75
    
                    # Convert to sky coordinates if reference coordinate is provided
                    if reference_coord is not None:
                        ra_ref, dec_ref = reference_coord
    
                        # Convert arcsec offsets to degrees and apply spherical projection correction
                        ra = ra_ref + (x_arcsec / 3600.0) / np.cos(np.radians(dec_ref))
                        dec = dec_ref + (y_arcsec / 3600.0)
    
                        # Convert back to angular offset from reference for spatial grid
                        x_sky = (ra - ra_ref) * 3600.0 * np.cos(np.radians(dec_ref))
                        y_sky = (dec - dec_ref) * 3600.0
    
                        fiber_positions.append((i, x_sky, y_sky))
    
                        # Debug information for the first few fibers
                        if i < 5:
                            self.logger.debug(f"Fiber {i}: benchside={benchside}, fiber={fiber_num}, "
                                              f"x={x}, y={y}, x_arcsec={x_arcsec}, y_arcsec={y_arcsec}, "
                                              f"ra={ra}, dec={dec}, x_sky={x_sky}, y_sky={y_sky}")
                    else:
                        # No reference coordinates, just use the arcsecond values directly
                        fiber_positions.append((i, x_arcsec, y_arcsec))
    
        if not fiber_positions:
            self.logger.error(f"No valid fiber positions found for channel {channel}")
            return None
    
        # Continue with rest of the method as before...
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
    
        # Create spatial grid with fixed spaxel size (spatial_sampling is in arcsec/pixel)
        n_x = int((x_max - x_min) / spatial_sampling) + 1
        n_y = int((y_max - y_min) / spatial_sampling) + 1
    
        self.spatial_grid_x = np.linspace(x_min, x_max, n_x)
        self.spatial_grid_y = np.linspace(y_min, y_max, n_y)
    
        # Initialize cube with NaNs
        cube_shape = (len(self.wavelength_grid), len(self.spatial_grid_y), len(self.spatial_grid_x))
        self.cube_data = np.full(cube_shape, np.nan)
    
        self.logger.info(f"Initialized cube for channel {channel} with shape: {self.cube_data.shape}")
    
        # Create a proper WCS for this cube with the RA/DEC reference and proper spatial scale
        if reference_coord is not None:
            wcs = self.create_wcs(reference_coord, spatial_sampling)
            self.wcs = wcs
            self.logger.info(f"Created WCS for channel {channel} with reference RA={reference_coord[0]}, "
                           f"DEC={reference_coord[1]}, pixel scale={spatial_sampling} arcsec")
    
        # Process each wavelength slice separately for spatial interpolation
        for w_idx in range(len(self.wavelength_grid)):
            # Extract data for this wavelength from all fibers
            fiber_x = []
            fiber_y = []
            fiber_values = []
    
            # Target wavelength for this slice
            target_wavelength = self.wavelength_grid[w_idx]
    
            # Collect valid data points for this wavelength
            for fiber_idx, x, y in fiber_positions:
                if fiber_wavelengths is not None and fiber_wavelengths.shape == flux_data.shape:
                    # Use per-fiber wavelength calibration
                    # Find the closest wavelength index in this fiber's wavelength array
                    fiber_wave = fiber_wavelengths[fiber_idx]
                    
                    # Skip fibers with NaN wavelengths
                    if np.all(np.isnan(fiber_wave)):
                        continue
                    
                    closest_idx = np.argmin(np.abs(fiber_wave - target_wavelength))
                    value = flux_data[fiber_idx, closest_idx]
                else:
                    # Use index directly if no per-fiber wavelengths
                    value = flux_data[fiber_idx, w_idx]
                
                if np.isfinite(value):  # Only use finite values
                    fiber_x.append(x)
                    fiber_y.append(y)
                    fiber_values.append(value)
    
            # Skip empty slices
            if not fiber_values:
                continue
            
            # Create grid for interpolation
            xi, yi = np.meshgrid(self.spatial_grid_x, self.spatial_grid_y)
    
            try:
                # Perform interpolation (using nearest neighbor to preserve data values)
                from scipy.interpolate import griddata
                grid_z = griddata(
                    (fiber_x, fiber_y),     # Points where we know values
                    fiber_values,           # Known values
                    (xi, yi),               # Points to interpolate
                    method='nearest',       # Use nearest neighbor interpolation
                    rescale=True            # Rescale to avoid precision issues
                )
    
                # Put interpolated data into cube
                self.cube_data[w_idx, :, :] = grid_z
            except Exception as e:
                self.logger.warning(f"Interpolation failed for wavelength {self.wavelength_grid[w_idx]}: {e}")
                # Fall back to nearest grid point method for this slice
                for fiber_idx, x, y in fiber_positions:
                    # Find the closest grid points
                    x_idx = np.argmin(np.abs(self.spatial_grid_x - x))
                    y_idx = np.argmin(np.abs(self.spatial_grid_y - y))
    
                    # Find the value based on per-fiber wavelength or fixed index
                    if fiber_wavelengths is not None and fiber_wavelengths.shape == flux_data.shape:
                        fiber_wave = fiber_wavelengths[fiber_idx]
                        
                        # Skip fibers with NaN wavelengths
                        if np.all(np.isnan(fiber_wave)):
                            continue
                        
                        closest_idx = np.argmin(np.abs(fiber_wave - target_wavelength))
                        value = flux_data[fiber_idx, closest_idx]
                    else:
                        value = flux_data[fiber_idx, w_idx]

                    # Place the spectrum point in the cube
                    self.cube_data[w_idx, y_idx, x_idx] = value
    
        self.logger.info(f"Cube for channel {channel} constructed successfully with spatial interpolation")
        return self.cube_data
        
    def _create_default_wavelength_grid(self, channel: str, n_pixels: int, wavelength_range: Optional[Tuple[float, float]] = None):
        """
        Create a default wavelength grid for a channel when wavelength information is not available.
        
        Parameters:
            channel (str): Channel identifier
            n_pixels (int): Number of pixels in the wavelength dimension
            wavelength_range (tuple, optional): User-provided wavelength range
        """
        if wavelength_range is not None:
            # Use the manually provided range
            min_wave, max_wave = wavelength_range
        else:
            # Infer reasonable wavelength range based on channel
            channel_upper = channel.upper()
            if channel_upper == "RED":
                min_wave, max_wave = 6300.0, 9000.0
            elif channel_upper == "GREEN":
                min_wave, max_wave = 4800.0, 6600.0
            elif channel_upper == "BLUE":
                min_wave, max_wave = 3700.0, 5100.0
            else:
                # Default for unknown channels
                min_wave, max_wave = 3700.0, 9000.0
            
        self.wavelength_grid = np.linspace(min_wave, max_wave, n_pixels)
        self.logger.info(f"Created default wavelength grid for {channel} channel: {min_wave:.1f}-{max_wave:.1f} Å")
    
    def save_cube(self, output_path: str, reference_coord: Optional[Tuple[float, float]] = None,
                  header_info: Optional[Dict] = None, spatial_sampling: float = 0.75) -> str:
        """
        Save the reconstructed cube to a FITS file.
        
        Parameters:
            output_path (str): Output file path
            reference_coord (tuple, optional): (RA, Dec) reference coordinates
            header_info (dict, optional): Additional header information
            spatial_sampling (float): Spatial sampling in arcsec/pixel
            
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
        
        # Add detailed wavelength calibration to primary header
        primary_hdu.header['CRVAL3'] = self.wavelength_grid[0]
        primary_hdu.header['CDELT3'] = (self.wavelength_grid[1] - self.wavelength_grid[0]) if len(self.wavelength_grid) > 1 else 1.0
        primary_hdu.header['CRPIX3'] = 1
        primary_hdu.header['CTYPE3'] = 'WAVE'
        primary_hdu.header['CUNIT3'] = 'Angstrom'
        
        # Add spaxel information
        primary_hdu.header['SPAXELSC'] = spatial_sampling
        primary_hdu.header['COMMENT'] = 'LLAMAS IFU Data Cube [wavelength, y, x]'
        primary_hdu.header['COMMENT'] = f'Spaxel size: {spatial_sampling} arcsec'
        primary_hdu.header['COMMENT'] = 'Each slice can be summed to create whitelight image'
        primary_hdu.header['COMMENT'] = 'Each spaxel contains a full spectrum'
        primary_hdu.header['COMMENT'] = f'Wavelength range: {self.wavelength_grid[0]:.2f}-{self.wavelength_grid[-1]:.2f} Angstroms'
        
        if header_info:
            for key, value in header_info.items():
                primary_hdu.header[key] = value
        
        # Create HDU list and save
        hdul = fits.HDUList([primary_hdu])
        
        # Add wavelength extension
        wave_hdu = fits.ImageHDU(data=self.wavelength_grid.astype(np.float32), name='WAVELENGTH')
        wave_hdu.header['EXTNAME'] = 'WAVELENGTH'
        wave_hdu.header['BUNIT'] = 'Angstrom'
        wave_hdu.header['COMMENT'] = 'Wavelength array for cube'
        hdul.append(wave_hdu)
        
        # Add spatial coordinate extensions
        x_hdu = fits.ImageHDU(data=self.spatial_grid_x.astype(np.float32), name='XCOORD')
        x_hdu.header['EXTNAME'] = 'XCOORD'
        x_hdu.header['BUNIT'] = 'arcsec'
        hdul.append(x_hdu)
        
        y_hdu = fits.ImageHDU(data=self.spatial_grid_y.astype(np.float32), name='YCOORD')
        y_hdu.header['EXTNAME'] = 'YCOORD'
        y_hdu.header['BUNIT'] = 'arcsec'
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
        """
        if not channel_cubes:
            raise ValueError("No channel cubes to save.")
        
        # Store original values to restore at the end
        original_cube_data = self.cube_data
        original_wavelength_grid = self.wavelength_grid
        original_wcs = self.wcs
        
        # Now save each channel with its specific wavelength grid and WCS
        saved_paths = {}
        for channel, cube_data in channel_cubes.items():
            # Set channel-specific values
            self.cube_data = cube_data
            
            # Use the channel-specific wavelength grid and WCS that were stored during construction
            if hasattr(self, 'channel_wavelength_grids') and channel in self.channel_wavelength_grids:
                self.wavelength_grid = self.channel_wavelength_grids[channel]
                self.logger.info(f"Using stored wavelength grid for channel {channel}: "
                              f"{self.wavelength_grid[0]:.1f}-{self.wavelength_grid[-1]:.1f} Å")
            
            if hasattr(self, 'channel_wcs') and channel in self.channel_wcs:
                self.wcs = self.channel_wcs[channel]
            else:
                # Create a new WCS if needed
                if reference_coord is not None:
                    self.wcs = self.create_wcs(reference_coord, spatial_sampling)
            
            # Get and log cube information
            self.logger.info(f"Cube information for channel {channel}:")
            cube_info = self.get_cube_info()
            for key, value in cube_info.items():
                self.logger.info(f"  {key}: {value}")
            
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
        
        # Restore original values
        self.cube_data = original_cube_data
        self.wavelength_grid = original_wavelength_grid
        self.wcs = original_wcs
        
        self.logger.info(f"Saved {len(saved_paths)} channel cubes with prefix: {output_prefix}")
        return saved_paths
    
    def create_wcs(self, reference_coord: Optional[Tuple[float, float]] = None, spatial_sampling: float = 0.75) -> WCS:
        """
        Create a World Coordinate System for the cube based on the RSS file's RA/DEC
        and the 0.75" fiber scale.

        Parameters:
            reference_coord (tuple, optional): (RA, Dec) reference coordinates in degrees
            spatial_sampling (float): Spatial sampling in arcsec/pixel (default: 0.75)

        Returns:
            WCS: World coordinate system object
        """
        wcs = WCS(naxis=3)

        # Use provided reference coordinates or defaults
        if reference_coord is not None:
            ra_ref, dec_ref = reference_coord
        else:
            # Default to (0, 0) if no reference coordinates provided
            ra_ref, dec_ref = 0.0, 0.0
            self.logger.warning("No reference coordinates provided for WCS, using (0, 0)")

        # Reference pixel at the center of the cube
        if self.spatial_grid_x is not None and self.spatial_grid_y is not None:
            crpix1 = len(self.spatial_grid_x) // 2
            crpix2 = len(self.spatial_grid_y) // 2
        else:
            crpix1, crpix2 = 0, 0
            self.logger.warning("No spatial grid defined for WCS, using (0, 0) for reference pixel")

        wcs.wcs.crpix = [crpix1, crpix2, 1]

        # Pixel scale in degrees/pixel based on 0.75" fiber scale
        # spatial_sampling is in arcsec/pixel
        pixel_scale_deg = spatial_sampling / 3600.0

        # Set wavelength dispersion if available
        if self.wavelength_grid is not None and len(self.wavelength_grid) > 1:
            wave_ref = self.wavelength_grid[0]
            wave_dispersion = self.wavelength_grid[1] - self.wavelength_grid[0]
        else:
            wave_ref = 5000.0  # Default wavelength reference
            wave_dispersion = 1.0  # Default dispersion
            self.logger.warning("No wavelength grid defined for WCS, using defaults")

        wcs.wcs.cdelt = [
            -pixel_scale_deg,  # Negative for RA per convention
            pixel_scale_deg,
            wave_dispersion
        ]

        # Reference coordinate values
        wcs.wcs.crval = [ra_ref, dec_ref, wave_ref]

        # Coordinate types
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
        wcs.wcs.cunit = ['deg', 'deg', 'Angstrom']

        self.logger.info(f"Created WCS with reference (RA,DEC)=({ra_ref},{dec_ref}), " 
                        f"pixel scale={spatial_sampling} arcsec, "
                        f"wavelength range={wave_ref}-{wave_ref+(len(self.wavelength_grid)-1)*wave_dispersion} Å")

        self.wcs = wcs
        return wcs