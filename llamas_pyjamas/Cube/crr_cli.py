#!/usr/bin/env python3
"""
CRR Cube Construction Command Line Interface

Command-line interface for the Covariance-regularized Reconstruction (CRR)
cube construction following Liu et al. (2020). This script provides a 
convenient interface for processing LLAMAS IFU data with CRR reconstruction.

Usage:
    python crr_cli.py input_rss.fits --output output_cube.fits
    python crr_cli.py input_rss.fits --config custom_config.yaml --parallel
    python crr_cli.py input_rss.fits --method shepard --output shepard_cube.fits

Author: Generated for LLAMAS Pipeline
Date: September 2025
"""

import argparse
import sys
import os
import logging
from pathlib import Path
import yaml
import time
from typing import Dict, Any, Optional

import numpy as np
from astropy.io import fits

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llamas_pyjamas.Cube.crr_cube_constructor import (
    CRRCubeConstructor, CRRCubeConfig, RSSData
)
from llamas_pyjamas.Cube.crr_parallel import parallel_cube_construction
from llamas_pyjamas.Cube.crr_weights import compute_shepard_weights
from llamas_pyjamas.Utils.utils import setup_logger


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")


def create_crr_config(config_dict: Dict[str, Any]) -> CRRCubeConfig:
    """Create CRRCubeConfig from configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary from YAML
        
    Returns:
        CRR configuration object
    """
    cube_config = config_dict.get('cube_reconstruction', {})
    seeing_config = config_dict.get('seeing', {})
    
    return CRRCubeConfig(
        pixel_scale=cube_config.get('pixel_scale', 0.75),
        regularization_lambda=cube_config.get('regularization_lambda', 1e-3),
        kernel_radius_limit=cube_config.get('kernel_radius_limit', 4.0),
        reconstruction_radius_limit=cube_config.get('reconstruction_radius_limit', 1.6),
        use_sky_subtraction=cube_config.get('use_sky_subtraction', False),
        fiber_diameter=cube_config.get('fiber_diameter', 2.0),
        seeing_reference_wavelength=seeing_config.get('reference_wavelength', 5500.0),
        seeing_power_law_index=seeing_config.get('power_law_index', -0.2)
    )


def load_rss_data(rss_file: str) -> RSSData:
    """Load RSS data from FITS file.
    
    Args:
        rss_file: Path to RSS FITS file
        
    Returns:
        RSS data structure
    """
    logger = setup_logger(__name__)
    logger.info(f"Loading RSS data from {rss_file}")
    
    with fits.open(rss_file) as hdul:
        # Extract main data arrays
        flux = hdul[0].data.astype(np.float32)  # Shape: (n_fibers, n_wavelengths)
        
        # Look for standard extensions
        ivar = None
        mask = None
        wavelength = None
        fiber_positions = None
        seeing_fwhm = 1.5  # Default seeing
        
        for hdu in hdul[1:]:  # Skip primary HDU
            if hdu.name == 'IVAR' or 'IVAR' in hdu.name:
                ivar = hdu.data.astype(np.float32)
            elif hdu.name == 'MASK' or 'MASK' in hdu.name:
                mask = hdu.data.astype(bool)
            elif hdu.name == 'WAVELENGTH' or 'WAVE' in hdu.name:
                if hdu.data.ndim == 1:
                    wavelength = hdu.data.astype(np.float32)
                else:
                    # 2D wavelength array - take first fiber as representative
                    wavelength = hdu.data[0, :].astype(np.float32)
            elif hdu.name == 'FIBERPOS' or 'FIBER' in hdu.name:
                fiber_positions = hdu.data.astype(np.float32)
        
        # Get metadata
        header = hdul[0].header
        object_name = header.get('OBJECT', 'Unknown')
        
        # Extract seeing information if available
        if 'SEEING' in header:
            seeing_fwhm = float(header['SEEING'])
        elif 'FWHM' in header:
            seeing_fwhm = float(header['FWHM'])
    
    # Validate and create default arrays if needed
    n_fibers, n_wavelengths = flux.shape
    
    if ivar is None:
        logger.warning("No IVAR extension found - creating uniform inverse variance")
        ivar = np.ones_like(flux)
        
    if mask is None:
        logger.warning("No MASK extension found - using valid data mask from flux")
        mask = np.isfinite(flux) & (flux != 0)
    
    if wavelength is None:
        logger.warning("No WAVELENGTH extension found - creating default wavelength grid")
        wavelength = np.linspace(3500, 9000, n_wavelengths)
    
    if fiber_positions is None:
        logger.warning("No fiber positions found - creating default grid positions")
        # Create a simple hexagonal grid as placeholder
        n_side = int(np.sqrt(n_fibers))
        x = np.arange(n_side) * 2.0  # 2 arcsec fiber spacing
        y = np.arange(n_side) * 2.0
        X, Y = np.meshgrid(x, y)
        fiber_positions = np.column_stack([X.ravel()[:n_fibers], Y.ravel()[:n_fibers]])
    
    # Ensure fiber positions have correct shape
    if fiber_positions.ndim == 2 and fiber_positions.shape[0] != n_fibers:
        # May need to transpose or reshape
        if fiber_positions.shape[1] == n_fibers:
            fiber_positions = fiber_positions.T
    
    metadata = {
        'object': object_name,
        'original_file': rss_file,
        'seeing_fwhm': seeing_fwhm,
        'n_fibers': n_fibers,
        'n_wavelengths': n_wavelengths
    }
    
    logger.info(f"RSS data loaded: {n_fibers} fibers, {n_wavelengths} wavelengths")
    logger.info(f"Wavelength range: {wavelength.min():.1f} - {wavelength.max():.1f} Å")
    
    return RSSData(
        flux=flux,
        ivar=ivar,
        mask=mask,
        fiber_positions=fiber_positions,
        wavelength=wavelength,
        seeing_fwhm=seeing_fwhm,
        metadata=metadata
    )


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='CRR Cube Construction for LLAMAS IFU Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic CRR reconstruction (auto-generates: input_rss_crr_cube.fits)
  %(prog)s input_rss.fits
  
  # Custom output filename  
  %(prog)s input_rss.fits --output my_galaxy_crr_cube.fits
  
  # Parallel processing (auto-generates: input_rss_crr_cube_parallel.fits)
  %(prog)s input_rss.fits --parallel --workers 8
  
  # Shepard method comparison (auto-generates: input_rss_shepard_cube.fits)
  %(prog)s input_rss.fits --method shepard
  
  # Run both methods with auto-generated filenames
  %(prog)s input_rss.fits --method both
        """
    )
    
    # Required arguments
    parser.add_argument('rss_file', 
                       help='Input RSS FITS file')
    
    # Output options
    parser.add_argument('--output', '-o',
                       help='Output cube FITS file (default: auto-generated with method suffix)')
    parser.add_argument('--shepard-output',
                       help='Output file for Shepard method (when method=both, default: auto-generated)')
    
    # Method selection
    parser.add_argument('--method', choices=['crr', 'shepard', 'both'],
                       default='crr',
                       help='Reconstruction method (default: crr)')
    
    # Configuration
    parser.add_argument('--config', '-c',
                       default='config/crr_config.yaml',
                       help='Configuration YAML file (default: config/crr_config.yaml)')
    
    # Parallel processing
    parser.add_argument('--parallel', action='store_true',
                       help='Use parallel processing with Ray')
    parser.add_argument('--workers', type=int,
                       help='Number of workers for parallel processing')
    parser.add_argument('--batch-size', type=int, default=50,
                       help='Wavelength batch size for parallel processing')
    
    # Reconstruction parameters (override config)
    parser.add_argument('--pixel-scale', type=float,
                       help='Output pixel scale in arcsec/pixel')
    parser.add_argument('--regularization-lambda', type=float,
                       help='Regularization parameter')
    parser.add_argument('--seeing-fwhm', type=float,
                       help='Seeing FWHM in arcsec (override file value)')
    
    # Quality and output options
    parser.add_argument('--no-quality', action='store_true',
                       help='Skip quality assessment')
    parser.add_argument('--save-weights', action='store_true',
                       help='Save reconstruction weight matrices (large files)')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='count', default=0,
                       help='Increase verbosity (use -vv for debug)')
    parser.add_argument('--log-file',
                       help='Log file path')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.WARNING
    if args.verbose == 1:
        log_level = logging.INFO
    elif args.verbose >= 2:
        log_level = logging.DEBUG
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=args.log_file
    )
    
    logger = setup_logger(__name__)
    logger.info("Starting CRR cube construction")
    
    try:
        # Load configuration
        if os.path.exists(args.config):
            config_dict = load_config(args.config)
            logger.info(f"Configuration loaded from {args.config}")
        else:
            logger.warning(f"Configuration file {args.config} not found - using defaults")
            config_dict = {}
        
        # Create CRR configuration
        crr_config = create_crr_config(config_dict)
        
        # Override with command line parameters
        if args.pixel_scale:
            crr_config.pixel_scale = args.pixel_scale
        if args.regularization_lambda:
            crr_config.regularization_lambda = args.regularization_lambda
        
        # Load RSS data
        rss_data = load_rss_data(args.rss_file)
        
        # Override seeing if specified
        if args.seeing_fwhm:
            rss_data.seeing_fwhm = args.seeing_fwhm
            logger.info(f"Seeing FWHM overridden to {args.seeing_fwhm} arcsec")
        
        # Generate output filename if not provided
        if not args.output:
            input_path = Path(args.rss_file)
            if args.method == 'crr':
                method_suffix = "_crr_cube"
            elif args.method == 'shepard':
                method_suffix = "_shepard_cube"
            elif args.method == 'both':
                method_suffix = "_crr_cube"  # Default for both mode
            else:
                method_suffix = f"_{args.method}_cube"
            
            # Add parallel suffix if using parallel processing
            if args.parallel:
                method_suffix += "_parallel"
                
            args.output = str(input_path.with_suffix('').name + f'{method_suffix}.fits')
        
        # Perform reconstruction based on method
        start_time = time.time()
        
        if args.method == 'crr' or args.method == 'both':
            logger.info("Starting CRR reconstruction")
            
            if args.parallel:
                logger.info("Using parallel processing")
                parallel_config = config_dict.get('parallel_processing', {})
                
                crr_cube = parallel_cube_construction(
                    rss_data,
                    crr_config,
                    n_workers=args.workers or parallel_config.get('n_workers'),
                    wavelength_batch_size=args.batch_size,
                    memory_limit_gb=parallel_config.get('memory_limit_gb')
                )
            else:
                logger.info("Using serial processing")
                constructor = CRRCubeConstructor(crr_config)
                crr_cube = constructor.reconstruct_cube(rss_data)
            
            # Save CRR cube
            logger.info(f"Saving CRR cube to {args.output}")
            crr_cube.save_to_fits(args.output)
            
            reconstruction_time = time.time() - start_time
            logger.info(f"CRR reconstruction completed in {reconstruction_time:.1f}s")
        
        if args.method == 'shepard' or args.method == 'both':
            logger.info("Starting Shepard reconstruction")
            
            # Generate Shepard output filename
            if args.method == 'both':
                if args.shepard_output:
                    shepard_output = args.shepard_output
                else:
                    # Generate shepard filename from input
                    input_path = Path(args.rss_file)
                    shepard_suffix = "_shepard_cube"
                    if args.parallel:
                        shepard_suffix += "_parallel"
                    shepard_output = str(input_path.with_suffix('').name + f'{shepard_suffix}.fits')
            else:
                shepard_output = args.output
            
            logger.warning("Shepard method reconstruction not fully implemented")
            logger.info(f"Shepard reconstruction would be saved to {shepard_output}")
        
        logger.info("All reconstructions completed successfully")
        
    except Exception as e:
        logger.error(f"Reconstruction failed: {e}")
        if args.verbose >= 2:
            import traceback
            traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())