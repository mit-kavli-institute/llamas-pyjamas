"""
RSS to CRR Data Format Adapter

This module provides conversion functions to adapt existing LLAMAS RSS (Row-Stacked Spectra)
files to the CRR (Covariance-regularized Reconstruction) data format. This enables seamless
integration of CRR reconstruction into the existing LLAMAS pipeline.

Functions:
    load_rss_as_crr_data: Convert RSS FITS file to CRRData format
    extract_fiber_positions: Get fiber positions from fibermap
    estimate_seeing_from_fwhm: Estimate seeing from RSS FWHM data
    combine_channels: Combine multi-channel RSS data for CRR

Author: Generated for LLAMAS Pipeline Integration
Date: September 2025
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from astropy.io import fits
from astropy.table import Table

from llamas_pyjamas.Cube.crr_cube_constructor import RSSData
from llamas_pyjamas.Utils.utils import setup_logger


def extract_fiber_positions(fibermap: Table, 
                           wavelength: np.ndarray) -> np.ndarray:
    """Extract fiber positions from LLAMAS fibermap.
    
    Args:
        fibermap: LLAMAS fibermap table
        wavelength: Wavelength array
        
    Returns:
        Fiber positions array (n_fibers, 2) in arcsec
    """
    logger = setup_logger(__name__)
    
    # Check what columns are available in the fibermap
    available_columns = fibermap.colnames
    logger.info(f"Available fibermap columns: {available_columns}")
    
    # Look for position columns (various naming conventions)
    x_col = None
    y_col = None
    
    # Try different column name conventions
    position_names = [
        ('X', 'Y'),
        ('x', 'y'), 
        ('X_POS', 'Y_POS'),
        ('XPOS', 'YPOS'),
        ('RA', 'DEC'),  # If only sky coordinates available
        ('FIBX', 'FIBY'),
        ('FIBER_X', 'FIBER_Y')
    ]
    
    for x_name, y_name in position_names:
        if x_name in available_columns and y_name in available_columns:
            x_col, y_col = x_name, y_name
            break
    
    if x_col is None or y_col is None:
        logger.warning("No fiber position columns found in fibermap")
        logger.warning(f"Available columns: {available_columns}")
        
        # Create default hexagonal grid as fallback
        n_fibers = len(fibermap)
        logger.info(f"Creating default hexagonal grid for {n_fibers} fibers")
        
        n_side = int(np.ceil(np.sqrt(n_fibers)))
        x_pos = []
        y_pos = []
        
        fiber_spacing = 2.0  # arcsec, LLAMAS fiber spacing
        for i in range(n_side):
            for j in range(n_side):
                if len(x_pos) >= n_fibers:
                    break
                # Hexagonal packing
                x = j * fiber_spacing
                y = i * fiber_spacing * np.sqrt(3)/2
                if i % 2 == 1:
                    x += fiber_spacing / 2
                x_pos.append(x)
                y_pos.append(y)
        
        # Center the grid
        x_pos = np.array(x_pos[:n_fibers]) - np.mean(x_pos[:n_fibers])
        y_pos = np.array(y_pos[:n_fibers]) - np.mean(y_pos[:n_fibers])
        
    else:
        logger.info(f"Using fiber positions from columns: {x_col}, {y_col}")
        x_pos = np.array(fibermap[x_col])
        y_pos = np.array(fibermap[y_col])
        
        # Convert from degrees to arcsec if needed (typical for RA/DEC)
        if x_col in ['RA', 'DEC'] or np.max(np.abs(x_pos)) < 1.0:
            logger.info("Converting positions from degrees to arcsec")
            x_pos *= 3600.0  # deg to arcsec
            y_pos *= 3600.0
    
    fiber_positions = np.column_stack([x_pos, y_pos])
    logger.info(f"Extracted {len(fiber_positions)} fiber positions")
    logger.info(f"Position range: X=[{x_pos.min():.2f}, {x_pos.max():.2f}], "
                f"Y=[{y_pos.min():.2f}, {y_pos.max():.2f}] arcsec")
    
    return fiber_positions


def estimate_seeing_from_fwhm(fwhm_data: np.ndarray,
                            percentile: float = 50) -> float:
    """Estimate atmospheric seeing from RSS FWHM measurements.

    Args:
        fwhm_data: FWHM array [NFIBER x NWAVE] (n_fibers, n_wavelengths)
        percentile: Percentile to use for seeing estimate

    Returns:
        Seeing FWHM in arcsec
    """
    # Remove invalid/zero values
    valid_fwhm = fwhm_data[fwhm_data > 0]
    
    if len(valid_fwhm) == 0:
        # Default seeing if no FWHM data available
        return 1.5
    
    # Use percentile to get representative seeing
    seeing_fwhm = np.percentile(valid_fwhm, percentile)
    
    # Sanity check: seeing should be reasonable (0.5" to 5")
    if seeing_fwhm < 0.5 or seeing_fwhm > 5.0:
        seeing_fwhm = 1.5  # Default
    
    return float(seeing_fwhm)


def load_rss_as_crr_data(rss_file: str) -> RSSData:
    """Load RSS FITS file and convert to CRR data format.
    
    Args:
        rss_file: Path to RSS FITS file
        
    Returns:
        RSSData object ready for CRR reconstruction
    """
    logger = setup_logger(__name__)
    logger.info(f"Loading RSS file for CRR conversion: {rss_file}")
    
    with fits.open(rss_file) as hdul:
        # Get basic metadata from primary header
        primary_hdr = hdul[0].header
        object_name = primary_hdr.get('OBJECT', 'Unknown')
        channel = primary_hdr.get('CHANNEL', 'unknown')
        
        # Initialize arrays
        flux_data = None
        error_data = None
        mask_data = None
        wave_data = None
        fwhm_data = None
        fibermap = None
        
        # Process all extensions
        for hdu in hdul[1:]:
            extname = hdu.header.get('EXTNAME', '').upper()
            
            if extname == 'FLUX' and hdu.data is not None:
                flux_data = hdu.data.astype(np.float32)
                logger.info(f"Loaded FLUX: {flux_data.shape}")
                
            elif extname == 'ERROR' and hdu.data is not None:
                error_data = hdu.data.astype(np.float32)
                logger.info(f"Loaded ERROR: {error_data.shape}")
                
            elif extname == 'MASK' and hdu.data is not None:
                mask_data = hdu.data.astype(bool)
                logger.info(f"Loaded MASK: {mask_data.shape}")
                
            elif extname == 'WAVE' and hdu.data is not None:
                wave_data = hdu.data.astype(np.float32)
                logger.info(f"Loaded WAVE: {wave_data.shape}")
                
            elif extname == 'FWHM' and hdu.data is not None:
                fwhm_data = hdu.data.astype(np.float32)
                logger.info(f"Loaded FWHM: {fwhm_data.shape}")
                
            elif extname == 'FIBERMAP':
                fibermap = Table(hdu.data)
                logger.info(f"Loaded FIBERMAP: {len(fibermap)} fibers")
    
    # Validate required data
    if flux_data is None:
        raise ValueError("No FLUX extension found in RSS file")

    # RSS format is now [NFIBER x NWAVE] - rows are fibers, columns are wavelengths
    n_fiber, n_wave = flux_data.shape
    logger.info(f"RSS data dimensions: {n_fiber} fibers, {n_wave} wavelengths [NFIBER x NWAVE format]")

    # Create error data if missing
    if error_data is None:
        logger.warning("No ERROR extension found - creating uniform errors")
        error_data = np.ones_like(flux_data) * 0.1 * np.median(flux_data[flux_data > 0])

    # Create mask if missing
    if mask_data is None:
        logger.warning("No MASK extension found - creating mask from finite flux")
        mask_data = np.isfinite(flux_data) & (flux_data != 0)

    # Handle wavelength data
    if wave_data is None:
        logger.warning("No WAVE extension found - creating default wavelength grid")
        wavelength = np.linspace(3500, 9000, n_wave)
    else:
        if wave_data.ndim == 2:
            # Take wavelength from first fiber (row 0) in [NFIBER x NWAVE] format
            wavelength = wave_data[0, :]
        else:
            wavelength = wave_data

    # RSS format is already [n_fiber, n_wave] - no transpose needed
    flux_crr = flux_data  # Shape: (n_fiber, n_wave)
    error_crr = error_data
    mask_crr = mask_data
    
    # Convert error to inverse variance
    ivar_crr = np.zeros_like(error_crr)
    valid_error = (error_crr > 0) & np.isfinite(error_crr)
    ivar_crr[valid_error] = 1.0 / (error_crr[valid_error]**2)
    
    # Extract fiber positions
    if fibermap is not None:
        fiber_positions = extract_fiber_positions(fibermap, wavelength)
    else:
        logger.warning("No FIBERMAP found - creating default fiber positions")
        # Create default hexagonal grid
        n_side = int(np.ceil(np.sqrt(n_fiber)))
        x_pos = []
        y_pos = []
        
        fiber_spacing = 2.0  # arcsec
        for i in range(n_side):
            for j in range(n_side):
                if len(x_pos) >= n_fiber:
                    break
                x = j * fiber_spacing
                y = i * fiber_spacing * np.sqrt(3)/2
                if i % 2 == 1:
                    x += fiber_spacing / 2
                x_pos.append(x)
                y_pos.append(y)
        
        x_pos = np.array(x_pos[:n_fiber]) - np.mean(x_pos[:n_fiber])
        y_pos = np.array(y_pos[:n_fiber]) - np.mean(y_pos[:n_fiber])
        fiber_positions = np.column_stack([x_pos, y_pos])
    
    # Estimate seeing
    if fwhm_data is not None:
        seeing_fwhm = estimate_seeing_from_fwhm(fwhm_data)
    else:
        seeing_fwhm = 1.5  # Default seeing
        logger.warning(f"No FWHM data found - using default seeing: {seeing_fwhm} arcsec")
    
    logger.info(f"Estimated seeing FWHM: {seeing_fwhm:.2f} arcsec")
    
    # Create metadata
    metadata = {
        'object': object_name,
        'channel': channel,
        'original_file': rss_file,
        'n_fibers': n_fiber,
        'n_wavelengths': n_wave,
        'wavelength_range': (wavelength.min(), wavelength.max()),
        'rss_format_conversion': True
    }
    
    # Create RSSData object
    rss_data = RSSData(
        flux=flux_crr,
        ivar=ivar_crr,
        mask=mask_crr,
        fiber_positions=fiber_positions,
        wavelength=wavelength,
        seeing_fwhm=seeing_fwhm,
        metadata=metadata
    )
    
    logger.info("RSS to CRR conversion completed successfully")
    return rss_data


def combine_channels_for_crr(rss_files: List[str]) -> RSSData:
    """Combine multiple channel RSS files into single CRR dataset.
    
    Args:
        rss_files: List of RSS file paths (different channels)
        
    Returns:
        Combined RSSData object
    """
    logger = setup_logger(__name__)
    logger.info(f"Combining {len(rss_files)} RSS files for CRR")
    
    # Load all channels
    channel_data = []
    for rss_file in rss_files:
        try:
            rss_data = load_rss_as_crr_data(rss_file)
            channel_data.append(rss_data)
            logger.info(f"Loaded channel: {rss_data.metadata.get('channel', 'unknown')}")
        except Exception as e:
            logger.warning(f"Failed to load {rss_file}: {e}")
            continue
    
    if not channel_data:
        raise ValueError("No RSS files could be loaded")
    
    if len(channel_data) == 1:
        return channel_data[0]
    
    # Combine wavelength arrays
    all_wavelengths = []
    for data in channel_data:
        all_wavelengths.extend(data.wavelength)
    
    # Sort combined wavelengths
    combined_wavelength = np.array(sorted(set(all_wavelengths)))
    n_wave_combined = len(combined_wavelength)
    n_fiber = channel_data[0].flux.shape[0]
    
    # Initialize combined arrays
    combined_flux = np.zeros((n_fiber, n_wave_combined), dtype=np.float32)
    combined_ivar = np.zeros((n_fiber, n_wave_combined), dtype=np.float32)
    combined_mask = np.zeros((n_fiber, n_wave_combined), dtype=bool)
    
    # Fill combined arrays
    for data in channel_data:
        for i, wave in enumerate(data.wavelength):
            # Find index in combined wavelength array
            idx = np.searchsorted(combined_wavelength, wave)
            if idx < n_wave_combined and np.abs(combined_wavelength[idx] - wave) < 0.01:
                combined_flux[:, idx] = data.flux[:, i]
                combined_ivar[:, idx] = data.ivar[:, i]
                combined_mask[:, idx] = data.mask[:, i]
    
    # Use fiber positions from first channel (should be same for all)
    fiber_positions = channel_data[0].fiber_positions
    
    # Average seeing across channels
    seeing_fwhm = np.mean([data.seeing_fwhm for data in channel_data])
    
    # Combined metadata
    channels = [data.metadata.get('channel', 'unknown') for data in channel_data]
    metadata = {
        'object': channel_data[0].metadata.get('object', 'Unknown'),
        'channels': channels,
        'original_files': rss_files,
        'n_fibers': n_fiber,
        'n_wavelengths': n_wave_combined,
        'wavelength_range': (combined_wavelength.min(), combined_wavelength.max()),
        'rss_format_conversion': True,
        'combined_channels': True
    }
    
    logger.info(f"Combined channels: {channels}")
    logger.info(f"Combined wavelength range: {combined_wavelength.min():.1f} - {combined_wavelength.max():.1f} Å")
    
    return RSSData(
        flux=combined_flux,
        ivar=combined_ivar,
        mask=combined_mask,
        fiber_positions=fiber_positions,
        wavelength=combined_wavelength,
        seeing_fwhm=seeing_fwhm,
        metadata=metadata
    )