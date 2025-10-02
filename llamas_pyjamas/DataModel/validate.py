"""FITS Extension Validation Module for LLAMAS Pipeline.

This module provides utilities to validate and fix FITS files with missing camera
extensions to ensure compatibility with the LLAMAS data reduction pipeline.

The LLAMAS instrument expects science images with a primary HDU plus 24 extensions,
representing 4 benches × 2 sides × 3 channels. When cameras fail, this module can
fill in missing extensions with placeholder data to prevent pipeline failures.

Functions:
    validate_and_fix_extensions: Main function to check and fix missing extensions.
    get_expected_camera_list: Returns the complete list of expected camera configurations.
    get_existing_cameras: Extracts camera metadata from existing FITS extensions.
    identify_missing_cameras: Determines which camera configurations are missing.
    create_placeholder_hdu: Creates a placeholder HDU with 1.0-valued arrays.
    validate_fits_structure: Checks if a FITS file needs validation.
    get_reference_dimensions: Determines appropriate array dimensions for placeholders.

Example:
    Basic usage to validate and fix a FITS file::

        from llamas_pyjamas.DataModel.validate import validate_and_fix_extensions

        # Fix missing extensions in-place
        validate_and_fix_extensions('science.fits')

        # Create a corrected copy
        validate_and_fix_extensions('science.fits', output_file='science_fixed.fits')
"""

import os
import logging
from typing import List, Dict, Tuple, Optional, Union
import numpy as np
from astropy.io import fits
from llamas_pyjamas.constants import idx_lookup
from llamas_pyjamas.File.llamasIO import llamasOneCamera, llamasAllCameras
from llamas_pyjamas.Utils.utils import setup_logger

# Set up logger
logger = setup_logger(__name__)

def get_expected_camera_list() -> List[Tuple[str, str, str]]:
    """Get the complete list of expected camera configurations.

    Returns:
        List of tuples containing (channel, bench, side) for all 24 expected cameras.

    Example:
        >>> cameras = get_expected_camera_list()
        >>> len(cameras)
        24
        >>> cameras[0]
        ('red', '1', 'A')
    """
    return list(idx_lookup.keys())

def get_existing_cameras(fits_file: str) -> List[Tuple[str, str, str, int]]:
    """Extract camera metadata from existing FITS extensions.

    Args:
        fits_file: Path to the FITS file to analyze.

    Returns:
        List of tuples containing (channel, bench, side, extension_index) for existing cameras.

    Raises:
        FileNotFoundError: If the FITS file does not exist.
        Exception: If there are issues reading the FITS file.
    """
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    existing_cameras = []

    try:
        with fits.open(fits_file) as hdul:
            for i, hdu in enumerate(hdul[1:], start=1):  # Skip primary HDU
                if hdu.data is None:
                    continue

                # Extract camera metadata from header
                try:
                    bench = str(hdu.header['BENCH'])
                    side = hdu.header['SIDE']
                    channel = hdu.header['COLOR'].lower()

                    existing_cameras.append((channel, bench, side, i))

                except KeyError as e:
                    logger.warning(f"Extension {i} missing required header key: {e}")
                    continue

    except Exception as e:
        logger.error(f"Error reading FITS file {fits_file}: {e}")
        raise

    return existing_cameras

def identify_missing_cameras(existing_cameras: List[Tuple[str, str, str, int]],
                           expected_cameras: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
    """Identify which camera configurations are missing.

    Args:
        existing_cameras: List of existing camera configurations with extension indices.
        expected_cameras: List of all expected camera configurations.

    Returns:
        List of missing camera configurations as (channel, bench, side) tuples.
    """
    # Extract just the camera configuration tuples (ignore extension indices)
    existing_configs = {(channel, bench, side) for channel, bench, side, _ in existing_cameras}
    expected_configs = set(expected_cameras)

    missing = expected_configs - existing_configs

    # Sort missing cameras by their expected index for consistent ordering
    missing_sorted = sorted(missing, key=lambda x: idx_lookup[x])

    logger.info(f"Found {len(existing_configs)} existing cameras, {len(missing_sorted)} missing")
    if missing_sorted:
        logger.info(f"Missing cameras: {missing_sorted}")

    return missing_sorted

def get_reference_dimensions(fits_file: str) -> Tuple[int, int]:
    """Determine appropriate array dimensions for placeholder extensions.

    Uses the first available extension as a reference for array dimensions.

    Args:
        fits_file: Path to the FITS file.

    Returns:
        Tuple of (height, width) for array dimensions.

    Raises:
        ValueError: If no valid extensions with data are found.
    """
    try:
        with fits.open(fits_file) as hdul:
            for hdu in hdul[1:]:  # Skip primary HDU
                if hdu.data is not None:
                    shape = hdu.data.shape
                    logger.debug(f"Using reference dimensions: {shape}")
                    return shape

        raise ValueError("No extensions with valid data found for reference dimensions")

    except Exception as e:
        logger.error(f"Error determining reference dimensions: {e}")
        raise

def is_placeholder_extension(hdu: fits.ImageHDU) -> bool:
    """Check if an HDU is a placeholder for a missing camera extension.

    Detects placeholder extensions created by the validation module by checking:
    1. Header COMMENT field for placeholder marker
    2. Data array for uniform 1.0 values (fallback detection)

    Args:
        hdu: FITS ImageHDU to check.

    Returns:
        bool: True if HDU is a placeholder, False otherwise.

    Example:
        >>> with fits.open('science.fits') as hdul:
        ...     if is_placeholder_extension(hdul[5]):
        ...         print("Extension 5 is a placeholder")
    """
    # Check for placeholder marker in header comments
    if 'COMMENT' in hdu.header:
        for comment in hdu.header['COMMENT']:
            if 'Placeholder extension created for missing camera' in str(comment):
                return True

    # Fallback: Check for uniform 1.0 data (characteristic of placeholders)
    if hdu.data is not None:
        try:
            unique_vals = np.unique(hdu.data)
            if len(unique_vals) == 1 and np.isclose(unique_vals[0], 1.0):
                return True
        except Exception:
            # If we can't analyze the data, assume not a placeholder
            pass

    return False


def get_placeholder_extension_indices(fits_file: str) -> List[int]:
    """Get list of extension indices that are placeholders.

    Scans through all extensions in a FITS file and identifies which ones
    are placeholders for missing camera data.

    Args:
        fits_file: Path to FITS file to scan.

    Returns:
        List of extension indices (1-based) that are placeholders.

    Raises:
        FileNotFoundError: If FITS file does not exist.

    Example:
        >>> placeholder_indices = get_placeholder_extension_indices('science.fits')
        >>> print(f"Found {len(placeholder_indices)} placeholder extensions")
        Found 4 placeholder extensions
    """
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    placeholder_indices = []

    try:
        with fits.open(fits_file) as hdul:
            for i in range(1, len(hdul)):  # Skip primary HDU
                if is_placeholder_extension(hdul[i]):
                    placeholder_indices.append(i)

                    # Log camera info if available
                    if logger.isEnabledFor(logging.DEBUG):
                        channel = hdul[i].header.get('COLOR', '?')
                        bench = hdul[i].header.get('BENCH', '?')
                        side = hdul[i].header.get('SIDE', '?')
                        logger.debug(f"Extension {i} ({channel}{bench}{side}) is a placeholder")

        if placeholder_indices:
            logger.info(f"Found {len(placeholder_indices)} placeholder extensions in {os.path.basename(fits_file)}")

    except Exception as e:
        logger.error(f"Error scanning for placeholder extensions: {e}")
        raise

    return placeholder_indices


def create_placeholder_hdu(camera_config: Tuple[str, str, str],
                          reference_shape: Tuple[int, int],
                          extension_name: Optional[str] = None) -> fits.ImageHDU:
    """Create a placeholder HDU with 1.0-valued arrays for a missing camera.

    Args:
        camera_config: Tuple of (channel, bench, side) for the missing camera.
        reference_shape: Shape tuple (height, width) for the array.
        extension_name: Optional name for the extension.

    Returns:
        FITS ImageHDU with placeholder data and appropriate headers.
    """
    channel, bench, side = camera_config

    # Create array filled with 1.0 values
    data = np.ones(reference_shape, dtype=np.float32)

    # Create header with camera metadata
    header = fits.Header()
    header['BENCH'] = (bench, 'Bench identifier')
    header['SIDE'] = (side, 'Side identifier (A or B)')
    header['COLOR'] = (channel.upper(), 'Color channel')
    header['EXTNAME'] = (extension_name or f"{channel.upper()}{bench}{side}", 'Extension name')
    header['COMMENT'] = 'Placeholder extension created for missing camera'
    header['COMMENT'] = 'Data filled with 1.0 values to maintain pipeline compatibility'

    # Create ImageHDU
    hdu = fits.ImageHDU(data=data, header=header)

    logger.debug(f"Created placeholder HDU for {channel}{bench}{side} with shape {reference_shape}")

    return hdu

def validate_fits_structure(fits_file: str, expected_extensions: int = 24) -> Dict[str, Union[bool, int, List]]:
    """Check if a FITS file needs validation and return diagnostic information.

    Args:
        fits_file: Path to the FITS file to check.
        expected_extensions: Number of extensions expected (default: 24).

    Returns:
        Dictionary containing validation results:
        - 'needs_validation': bool indicating if file needs fixing
        - 'current_extensions': int number of current extensions
        - 'missing_count': int number of missing extensions
        - 'existing_cameras': list of existing camera configurations
        - 'missing_cameras': list of missing camera configurations

    Raises:
        FileNotFoundError: If the FITS file does not exist.
    """
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    # Get current structure
    try:
        with fits.open(fits_file) as hdul:
            current_extensions = len(hdul) - 1  # Exclude primary HDU
    except Exception as e:
        logger.error(f"Error reading FITS file structure: {e}")
        raise

    # Get camera information
    existing_cameras = get_existing_cameras(fits_file)
    expected_cameras = get_expected_camera_list()
    missing_cameras = identify_missing_cameras(existing_cameras, expected_cameras)

    needs_validation = len(missing_cameras) > 0

    result = {
        'needs_validation': needs_validation,
        'current_extensions': current_extensions,
        'missing_count': len(missing_cameras),
        'existing_cameras': existing_cameras,
        'missing_cameras': missing_cameras
    }

    logger.info(f"Validation check: {current_extensions}/{expected_extensions} extensions present, "
                f"needs_validation={needs_validation}")

    return result

def validate_and_fix_extensions(fits_file: str,
                               expected_extensions: int = 24,
                               output_file: Optional[str] = None,
                               backup: bool = True) -> str:
    """Main function to validate and fix missing extensions in a FITS file.

    This function checks if a FITS file has the expected number of extensions,
    and if not, creates placeholder extensions filled with 1.0 values to maintain
    pipeline compatibility.

    Args:
        fits_file: Path to the input FITS file.
        expected_extensions: Number of extensions expected (default: 24).
        output_file: Optional output file path. If None, modifies input file in-place.
        backup: Whether to create a backup of the original file (default: True).

    Returns:
        Path to the output file (either input file or specified output file).

    Raises:
        FileNotFoundError: If the input FITS file does not exist.
        ValueError: If the file has more than expected extensions.
        Exception: For other file I/O or processing errors.

    Example:
        >>> # Fix file in-place with backup
        >>> result_file = validate_and_fix_extensions('science.fits')

        >>> # Create corrected copy
        >>> result_file = validate_and_fix_extensions('science.fits',
        ...                                          output_file='science_fixed.fits')
    """
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"Input FITS file not found: {fits_file}")

    logger.info(f"Validating FITS file: {fits_file}")

    # Validate current structure
    validation_info = validate_fits_structure(fits_file, expected_extensions)

    if not validation_info['needs_validation']:
        logger.info("File already has correct number of extensions - no changes needed")
        return fits_file

    current_extensions = validation_info['current_extensions']
    missing_cameras = validation_info['missing_cameras']

    if current_extensions > expected_extensions:
        raise ValueError(f"File has more extensions ({current_extensions}) than expected ({expected_extensions})")

    logger.info(f"File needs fixing: {current_extensions}/{expected_extensions} extensions present")
    logger.info(f"Will add {len(missing_cameras)} placeholder extensions")

    # Determine output file
    if output_file is None:
        output_file = fits_file
        if backup:
            backup_file = fits_file + '.backup'
            logger.info(f"Creating backup: {backup_file}")
            import shutil
            shutil.copy2(fits_file, backup_file)

    # Get reference dimensions
    reference_shape = get_reference_dimensions(fits_file)
    logger.info(f"Using reference dimensions: {reference_shape}")

    try:
        # Read original file
        with fits.open(fits_file) as original_hdul:
            # Create new HDU list starting with primary HDU
            new_hdul = fits.HDUList([original_hdul[0].copy()])

            # Track which extensions we've added
            existing_cameras = validation_info['existing_cameras']

            # Create a mapping of expected extension indices to camera configs
            expected_positions = {idx_lookup[config]: config for config in get_expected_camera_list()}

            # Build the complete HDU list with placeholders for missing cameras
            for ext_idx in range(1, expected_extensions + 1):
                expected_config = expected_positions[ext_idx]

                # Check if this extension exists in the original file
                existing_ext = None
                for channel, bench, side, orig_idx in existing_cameras:
                    if (channel, bench, side) == expected_config:
                        existing_ext = orig_idx
                        break

                if existing_ext is not None:
                    # Copy existing extension
                    new_hdul.append(original_hdul[existing_ext].copy())
                    logger.debug(f"Copied existing extension {existing_ext} for {expected_config}")
                else:
                    # Create placeholder extension
                    placeholder_hdu = create_placeholder_hdu(expected_config, reference_shape)
                    new_hdul.append(placeholder_hdu)
                    logger.info(f"Added placeholder extension for {expected_config} at position {ext_idx}")

            # Write the corrected file
            logger.info(f"Writing corrected file: {output_file}")
            new_hdul.writeto(output_file, overwrite=True)

    except Exception as e:
        logger.error(f"Error processing FITS file: {e}")
        raise

    # Verify the result
    try:
        final_validation = validate_fits_structure(output_file, expected_extensions)
        if final_validation['needs_validation']:
            logger.error("Validation failed after processing - file still needs correction")
            raise RuntimeError("File validation failed after processing")
        else:
            logger.info(f"Successfully validated and fixed FITS file: {output_file}")
            logger.info(f"Final structure: {final_validation['current_extensions']}/{expected_extensions} extensions")

    except Exception as e:
        logger.error(f"Error verifying corrected file: {e}")
        raise

    return output_file

if __name__ == "__main__":
    """Command-line interface for FITS validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate and fix FITS files with missing camera extensions")
    parser.add_argument("fits_file", help="Input FITS file to validate")
    parser.add_argument("-o", "--output", help="Output file (default: modify input file)")
    parser.add_argument("-e", "--expected", type=int, default=24,
                       help="Expected number of extensions (default: 24)")
    parser.add_argument("--no-backup", action="store_true",
                       help="Do not create backup when modifying input file")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        result_file = validate_and_fix_extensions(
            args.fits_file,
            expected_extensions=args.expected,
            output_file=args.output,
            backup=not args.no_backup
        )
        print(f"Successfully processed: {result_file}")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)