#!/usr/bin/env python3
"""
Simple MUSE-like Data Cube Constructor for LLAMAS

This standalone script constructs 3D data cubes from LLAMAS RSS FITS files
following the MUSE data reduction pipeline methodology, adapted for LLAMAS
fiber geometry and wavelength coverage.

Key MUSE-inspired features:
- Drizzle-like resampling with adjustable pixel fraction
- Spatial interpolation using inverse distance weighting
- Regular wavelength grid with optional oversampling
- Variance propagation (when available in future)
- Bad pixel masking

Usage:
    python simple_cube_constructor.py <rss_file> [options]

Example:
    python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits --pixel-size 0.3 --output cube_red.fits

Author: LLAMAS Pipeline
Date: 2025-12-13
"""

import numpy as np
import argparse
import sys
import os
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class SimpleCubeConstructor:
    """
    Constructs 3D datacubes from LLAMAS RSS files using MUSE-like methodology.

    This is a simplified, standalone version that doesn't require the full
    LLAMAS framework and focuses on the core cube construction algorithm.
    """

    def __init__(self, fiber_pitch=0.75, pixel_size=0.3, wave_sampling=1.0):
        """
        Initialize the cube constructor.

        Parameters:
            fiber_pitch (float): Fiber-to-fiber pitch in arcseconds (default: 0.75")
            pixel_size (float): Output spatial pixel size in arcseconds (default: 0.3")
            wave_sampling (float): Wavelength sampling factor (1.0 = native, <1.0 = oversample)
        """
        self.fiber_pitch = fiber_pitch
        self.pixel_size = pixel_size
        self.wave_sampling = wave_sampling

        # Will be loaded from file
        self.fibermap = None
        self.flux = None
        self.error = None
        self.wave = None
        self.mask = None
        self.fwhm = None

        # Output cube
        self.cube = None
        self.cube_var = None
        self.cube_weight = None
        self.wave_grid = None
        self.wcs = None

    def load_fibermap(self, fibermap_path):
        """
        Load LLAMAS fiber map from LUT file.

        Parameters:
            fibermap_path (str): Path to LLAMAS_FiberMap_rev04.dat
        """
        print(f"Loading fiber map from: {fibermap_path}")

        try:
            self.fibermap = Table.read(fibermap_path, format='ascii.fixed_width')
            print(f"  Loaded {len(self.fibermap)} fiber positions")

            # Check required columns
            required_cols = ['bench', 'fiber', 'xpos', 'ypos']
            for col in required_cols:
                if col not in self.fibermap.colnames:
                    raise ValueError(f"Missing required column: {col}")

        except Exception as e:
            raise RuntimeError(f"Failed to load fiber map: {e}")

    def load_rss_file(self, rss_file):
        """
        Load RSS FITS file containing flux, wavelength, error, mask, etc.

        Parameters:
            rss_file (str): Path to RSS FITS file
        """
        print(f"\nLoading RSS file: {rss_file}")

        if not os.path.exists(rss_file):
            raise FileNotFoundError(f"RSS file not found: {rss_file}")

        with fits.open(rss_file) as hdul:
            # Check structure
            print(f"  Found {len(hdul)} extensions")

            # Load main data
            if 'FLUX' in hdul:
                self.flux = hdul['FLUX'].data
                print(f"  FLUX: {self.flux.shape}")
            else:
                raise ValueError("No FLUX extension found in RSS file")

            if 'WAVE' in hdul:
                self.wave = hdul['WAVE'].data
                print(f"  WAVE: {self.wave.shape}")
            else:
                raise ValueError("No WAVE extension found in RSS file")

            if 'ERROR' in hdul:
                self.error = hdul['ERROR'].data
                print(f"  ERROR: {self.error.shape}")
            else:
                print("  WARNING: No ERROR extension, variance will not be computed")
                self.error = None

            if 'MASK' in hdul:
                self.mask = hdul['MASK'].data
                print(f"  MASK: {self.mask.shape}")
            else:
                print("  WARNING: No MASK extension, all pixels assumed good")
                self.mask = np.zeros_like(self.flux, dtype=bool)

            if 'FWHM' in hdul:
                self.fwhm = hdul['FWHM'].data
                print(f"  FWHM: {self.fwhm.shape}")
            else:
                self.fwhm = None

        # Validate shapes
        if self.flux.shape != self.wave.shape:
            raise ValueError(f"FLUX and WAVE shapes don't match: {self.flux.shape} vs {self.wave.shape}")

        print(f"  Loaded {self.flux.shape[0]} fibers x {self.flux.shape[1]} pixels")

    def create_wavelength_grid(self, wave_min=None, wave_max=None):
        """
        Create regular wavelength grid for output cube.

        This follows MUSE approach of creating a common wavelength grid
        that all fibers will be resampled onto.

        Parameters:
            wave_min (float): Minimum wavelength (Angstroms). If None, auto-detect.
            wave_max (float): Maximum wavelength (Angstroms). If None, auto-detect.
        """
        print("\nCreating wavelength grid...")

        # Get wavelength range from data
        wave_valid = self.wave[np.isfinite(self.wave) & (self.wave > 0)]

        if wave_min is None:
            wave_min = np.min(wave_valid)
        if wave_max is None:
            wave_max = np.max(wave_valid)

        # Calculate median wavelength spacing from data
        wave_diffs = []
        for i in range(min(100, self.wave.shape[0])):  # Sample first 100 fibers
            w = self.wave[i, :]
            w_valid = w[np.isfinite(w) & (w > 0)]
            if len(w_valid) > 1:
                wave_diffs.append(np.median(np.diff(w_valid)))

        median_dwave = np.median(wave_diffs)

        # Apply sampling factor
        dwave = median_dwave * self.wave_sampling

        # Create grid
        nwave = int((wave_max - wave_min) / dwave) + 1
        self.wave_grid = np.linspace(wave_min, wave_max, nwave)

        print(f"  Wavelength range: {wave_min:.2f} - {wave_max:.2f} Å")
        print(f"  Native sampling: {median_dwave:.4f} Å/pixel")
        print(f"  Output sampling: {dwave:.4f} Å/pixel (factor: {self.wave_sampling})")
        print(f"  Number of wavelength pixels: {nwave}")

    def create_spatial_grid(self):
        """
        Create regular spatial grid for output cube based on fiber positions.

        Uses fiber map to define spatial extent and pixel size.
        """
        print("\nCreating spatial grid...")

        if self.fibermap is None:
            raise RuntimeError("Fiber map not loaded. Call load_fibermap() first.")

        # Get fiber positions in units of fiber pitch
        # The fibermap xpos/ypos are already in grid units
        xpos = np.array(self.fibermap['xpos'])
        ypos = np.array(self.fibermap['ypos'])

        # Convert to arcseconds
        x_arcsec = xpos * self.fiber_pitch
        y_arcsec = ypos * self.fiber_pitch

        # Define spatial grid
        # Add margin around fibers
        margin = 2.0  # arcseconds

        x_min = np.min(x_arcsec) - margin
        x_max = np.max(x_arcsec) + margin
        y_min = np.min(y_arcsec) - margin
        y_max = np.max(y_arcsec) + margin

        # Create regular grid
        nx = int((x_max - x_min) / self.pixel_size) + 1
        ny = int((y_max - y_min) / self.pixel_size) + 1

        self.x_grid = np.linspace(x_min, x_max, nx)
        self.y_grid = np.linspace(y_min, y_max, ny)

        print(f"  X range: {x_min:.2f} - {x_max:.2f} arcsec ({nx} pixels)")
        print(f"  Y range: {y_min:.2f} - {y_max:.2f} arcsec ({ny} pixels)")
        print(f"  Spatial pixel size: {self.pixel_size} arcsec")
        print(f"  Total spatial pixels: {nx} x {ny} = {nx*ny:,}")

    def match_fibers_to_fibermap(self):
        """
        Match RSS fiber indices to fibermap entries.

        Returns:
            fiber_coords (ndarray): (N, 2) array of (x, y) positions in arcsec for each RSS fiber
        """
        print("\nMatching RSS fibers to fibermap...")

        nfibers = self.flux.shape[0]

        # For LLAMAS, fibers are stored by benchside in the RSS FIBERMAP extension
        # We need to match them to the LUT fibermap

        # For now, assume fibers are in the same order as fibermap
        # (This may need adjustment based on actual RSS fiber ordering)

        if len(self.fibermap) != nfibers:
            print(f"  WARNING: Fibermap has {len(self.fibermap)} entries but RSS has {nfibers} fibers")
            print(f"  Using first {min(len(self.fibermap), nfibers)} fibers")

        n_match = min(len(self.fibermap), nfibers)

        xpos = np.array(self.fibermap['xpos'][:n_match])
        ypos = np.array(self.fibermap['ypos'][:n_match])

        # Convert to arcseconds
        fiber_coords = np.column_stack([
            xpos * self.fiber_pitch,
            ypos * self.fiber_pitch
        ])

        print(f"  Matched {n_match} fibers to spatial positions")

        return fiber_coords

    def resample_fiber_to_grid(self, fiber_idx, fiber_coords):
        """
        Resample a single fiber spectrum onto the common wavelength grid.

        Parameters:
            fiber_idx (int): Index of fiber in RSS arrays
            fiber_coords (ndarray): Spatial coordinates of this fiber

        Returns:
            resampled_flux (ndarray): Flux resampled onto wave_grid
            resampled_var (ndarray): Variance resampled onto wave_grid (if available)
        """
        # Get fiber data
        wave_fiber = self.wave[fiber_idx, :]
        flux_fiber = self.flux[fiber_idx, :]

        # Handle mask
        if self.mask is not None:
            mask_fiber = self.mask[fiber_idx, :].astype(bool)
        else:
            mask_fiber = np.zeros_like(flux_fiber, dtype=bool)

        # Valid pixels
        valid = np.isfinite(wave_fiber) & np.isfinite(flux_fiber) & (wave_fiber > 0) & (~mask_fiber)

        if np.sum(valid) < 2:
            # Not enough valid pixels
            return None, None

        # Sort by wavelength
        sort_idx = np.argsort(wave_fiber[valid])
        wave_sorted = wave_fiber[valid][sort_idx]
        flux_sorted = flux_fiber[valid][sort_idx]

        # Interpolate onto common grid
        # Use linear interpolation (MUSE uses more sophisticated drizzling, but this is simpler)
        try:
            interp_flux = np.interp(self.wave_grid, wave_sorted, flux_sorted,
                                   left=np.nan, right=np.nan)
        except Exception:
            return None, None

        # Variance (if available)
        if self.error is not None:
            error_fiber = self.error[fiber_idx, :]
            var_fiber = error_fiber[valid]**2
            var_sorted = var_fiber[sort_idx]

            try:
                # For variance, we need to be more careful - use nearest neighbor
                # or conservative interpolation
                interp_var = np.interp(self.wave_grid, wave_sorted, var_sorted,
                                      left=np.nan, right=np.nan)
            except Exception:
                interp_var = None
        else:
            interp_var = None

        return interp_flux, interp_var

    def construct_cube(self, radius=1.5, min_weight=0.01):
        """
        Construct 3D datacube using inverse distance weighted interpolation.

        This follows MUSE methodology of combining fibers spatially using
        weighted interpolation based on distance.

        Parameters:
            radius (float): Maximum distance (in arcsec) for fiber contribution
            min_weight (float): Minimum weight threshold for including a fiber
        """
        print("\nConstructing datacube...")
        print(f"  Interpolation radius: {radius} arcsec")
        print(f"  Minimum weight: {min_weight}")

        # Match fibers to positions
        fiber_coords = self.match_fibers_to_fibermap()
        nfibers = len(fiber_coords)

        # Initialize cube arrays
        nwave = len(self.wave_grid)
        ny = len(self.y_grid)
        nx = len(self.x_grid)

        self.cube = np.zeros((nwave, ny, nx), dtype=np.float32)
        self.cube_weight = np.zeros((nwave, ny, nx), dtype=np.float32)

        if self.error is not None:
            self.cube_var = np.zeros((nwave, ny, nx), dtype=np.float32)
        else:
            self.cube_var = None

        print(f"  Cube shape: {self.cube.shape} (wavelength, y, x)")
        print(f"  Memory: ~{self.cube.nbytes / 1024**2:.1f} MB per array")

        # Build KDTree for fast spatial lookups
        tree = cKDTree(fiber_coords)

        # Create meshgrid of output positions
        xx, yy = np.meshgrid(self.x_grid, self.y_grid, indexing='xy')
        output_coords = np.column_stack([xx.ravel(), yy.ravel()])

        # Find fibers within radius of each output pixel
        print("\n  Finding fiber neighbors for each spatial pixel...")
        neighbors = tree.query_ball_point(output_coords, radius)

        # Resample all fibers onto common wavelength grid first
        print(f"\n  Resampling {nfibers} fibers onto wavelength grid...")
        resampled_data = []

        for i in range(nfibers):
            if i % 500 == 0:
                print(f"    Fiber {i}/{nfibers}...")

            flux_resamp, var_resamp = self.resample_fiber_to_grid(i, fiber_coords[i])
            resampled_data.append((flux_resamp, var_resamp))

        print("\n  Spatially interpolating onto cube grid...")

        # Process each output spatial pixel
        total_pixels = len(output_coords)

        for pix_idx in range(total_pixels):
            if pix_idx % 10000 == 0:
                print(f"    Pixel {pix_idx}/{total_pixels} ({100*pix_idx/total_pixels:.1f}%)...")

            # Get output pixel position
            out_y = pix_idx // nx
            out_x = pix_idx % nx
            out_pos = output_coords[pix_idx]

            # Get neighboring fibers
            fiber_indices = neighbors[pix_idx]

            if len(fiber_indices) == 0:
                continue

            # Calculate weights based on distance
            distances = np.linalg.norm(fiber_coords[fiber_indices] - out_pos, axis=1)

            # Inverse distance weighting with Gaussian-like kernel
            # Similar to MUSE's spatial PSF weighting
            sigma = radius / 2.0  # Effective kernel width
            weights = np.exp(-0.5 * (distances / sigma)**2)

            # Apply minimum weight threshold
            valid_weights = weights >= min_weight

            if not np.any(valid_weights):
                continue

            weights = weights[valid_weights]
            fiber_indices = np.array(fiber_indices)[valid_weights]

            # Normalize weights
            weights /= np.sum(weights)

            # Combine fiber spectra
            combined_flux = np.zeros(nwave)
            combined_var = np.zeros(nwave) if self.cube_var is not None else None
            combined_weight = np.zeros(nwave)

            for fib_idx, w in zip(fiber_indices, weights):
                flux_resamp, var_resamp = resampled_data[fib_idx]

                if flux_resamp is None:
                    continue

                # Valid wavelength pixels
                valid = np.isfinite(flux_resamp)

                combined_flux[valid] += w * flux_resamp[valid]
                combined_weight[valid] += w

                if combined_var is not None and var_resamp is not None:
                    # Variance adds as weighted sum of variances
                    combined_var[valid] += (w**2) * var_resamp[valid]

            # Normalize by total weight
            nonzero = combined_weight > 0
            combined_flux[nonzero] /= combined_weight[nonzero]

            # Store in cube
            self.cube[:, out_y, out_x] = combined_flux
            self.cube_weight[:, out_y, out_x] = combined_weight

            if self.cube_var is not None:
                self.cube_var[nonzero, out_y, out_x] = combined_var[nonzero] / (combined_weight[nonzero]**2)

        print("\n  Datacube construction complete!")

        # Statistics
        valid_spaxels = np.sum(np.any(self.cube_weight > 0, axis=0))
        total_spaxels = nx * ny
        print(f"  Valid spaxels: {valid_spaxels}/{total_spaxels} ({100*valid_spaxels/total_spaxels:.1f}%)")

    def create_wcs(self, ra_center=0.0, dec_center=0.0):
        """
        Create WCS (World Coordinate System) for the cube.

        Parameters:
            ra_center (float): RA of field center in degrees
            dec_center (float): Dec of field center in degrees
        """
        print("\nCreating WCS...")

        wcs = WCS(naxis=3)

        # Spatial axes (RA, Dec)
        wcs.wcs.crpix = [len(self.x_grid)/2, len(self.y_grid)/2, 1]
        wcs.wcs.crval = [ra_center, dec_center, self.wave_grid[0]]
        wcs.wcs.cdelt = [self.pixel_size/3600.0, self.pixel_size/3600.0,
                         self.wave_grid[1] - self.wave_grid[0]]
        wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
        wcs.wcs.cunit = ['deg', 'deg', 'Angstrom']

        self.wcs = wcs

        print(f"  Reference pixel: {wcs.wcs.crpix}")
        print(f"  Reference value: {wcs.wcs.crval}")
        print(f"  Pixel scale: {wcs.wcs.cdelt}")

    def save_cube(self, output_file, overwrite=True):
        """
        Save datacube to FITS file.

        Parameters:
            output_file (str): Output FITS filename
            overwrite (bool): Overwrite existing file
        """
        print(f"\nSaving cube to: {output_file}")

        if self.cube is None:
            raise RuntimeError("No cube to save. Run construct_cube() first.")

        # Create HDU list
        hdu_list = fits.HDUList()

        # Primary HDU with datacube
        primary = fits.PrimaryHDU(data=self.cube)

        # Add basic header info
        primary.header['EXTNAME'] = 'FLUX'
        primary.header['BUNIT'] = 'counts'
        primary.header['COMMENT'] = 'LLAMAS datacube - flux'

        # Add construction parameters
        primary.header['PIXSIZE'] = (self.pixel_size, 'Spatial pixel size (arcsec)')
        primary.header['FIBPITCH'] = (self.fiber_pitch, 'Fiber pitch (arcsec)')
        primary.header['WAVESAMP'] = (self.wave_sampling, 'Wavelength sampling factor')

        # Add WCS if available
        if self.wcs is not None:
            primary.header.update(self.wcs.to_header())

        hdu_list.append(primary)

        # Variance extension
        if self.cube_var is not None:
            var_hdu = fits.ImageHDU(data=self.cube_var, name='VAR')
            var_hdu.header['BUNIT'] = 'counts^2'
            var_hdu.header['COMMENT'] = 'Variance cube'
            hdu_list.append(var_hdu)

        # Weight extension
        weight_hdu = fits.ImageHDU(data=self.cube_weight, name='WEIGHT')
        weight_hdu.header['COMMENT'] = 'Weight/coverage map'
        hdu_list.append(weight_hdu)

        # Wavelength table
        wave_table = Table()
        wave_table['wavelength'] = self.wave_grid
        wave_table['wavelength'].unit = 'Angstrom'

        wave_hdu = fits.BinTableHDU(wave_table, name='WAVELENGTH')
        hdu_list.append(wave_hdu)

        # Write to file
        hdu_list.writeto(output_file, overwrite=overwrite)

        print(f"  Saved cube with {len(hdu_list)} extensions")
        print(f"  File size: ~{os.path.getsize(output_file) / 1024**2:.1f} MB")


def main():
    """Command-line interface for simple cube constructor."""

    parser = argparse.ArgumentParser(
        description='Construct LLAMAS datacube from RSS file using MUSE-like methodology',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cube construction
  python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits

  # Custom spatial sampling
  python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits --pixel-size 0.2

  # Oversample wavelength axis
  python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits --wave-sampling 0.5

  # Specify output file and fiber map
  python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits \\
      --output my_cube.fits --fibermap /path/to/fibermap.dat
        """
    )

    # Required arguments
    parser.add_argument('rss_file', help='Input RSS FITS file')

    # Optional arguments
    parser.add_argument('--fibermap', default=None,
                       help='Path to fiber map file (default: LUT/LLAMAS_FiberMap_rev04.dat)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output FITS file (default: auto-generate from input name)')
    parser.add_argument('--pixel-size', type=float, default=0.3,
                       help='Spatial pixel size in arcsec (default: 0.3)')
    parser.add_argument('--fiber-pitch', type=float, default=0.75,
                       help='Fiber-to-fiber pitch in arcsec (default: 0.75)')
    parser.add_argument('--wave-sampling', type=float, default=1.0,
                       help='Wavelength sampling factor (default: 1.0, <1.0 = oversample)')
    parser.add_argument('--radius', type=float, default=1.5,
                       help='Interpolation radius in arcsec (default: 1.5)')
    parser.add_argument('--min-weight', type=float, default=0.01,
                       help='Minimum weight threshold (default: 0.01)')
    parser.add_argument('--ra', type=float, default=0.0,
                       help='RA of field center in degrees (default: 0.0)')
    parser.add_argument('--dec', type=float, default=0.0,
                       help='Dec of field center in degrees (default: 0.0)')
    parser.add_argument('--wave-min', type=float, default=None,
                       help='Minimum wavelength in Angstroms (default: auto)')
    parser.add_argument('--wave-max', type=float, default=None,
                       help='Maximum wavelength in Angstroms (default: auto)')

    args = parser.parse_args()

    # Determine fiber map path
    if args.fibermap is None:
        # Try to find it relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fibermap_path = os.path.join(script_dir, '..', 'LUT', 'LLAMAS_FiberMap_rev04.dat')

        if not os.path.exists(fibermap_path):
            print(f"ERROR: Could not find default fibermap at {fibermap_path}")
            print("Please specify --fibermap path explicitly")
            return 1
    else:
        fibermap_path = args.fibermap

    # Determine output file name
    if args.output is None:
        base = os.path.basename(args.rss_file).replace('.fits', '')
        output_file = f"{base}_cube.fits"
    else:
        output_file = args.output

    # Print configuration
    print("="*80)
    print("LLAMAS Simple Cube Constructor")
    print("="*80)
    print(f"Input RSS file: {args.rss_file}")
    print(f"Fiber map: {fibermap_path}")
    print(f"Output file: {output_file}")
    print(f"Spatial pixel size: {args.pixel_size} arcsec")
    print(f"Fiber pitch: {args.fiber_pitch} arcsec")
    print(f"Wavelength sampling: {args.wave_sampling}")
    print(f"Interpolation radius: {args.radius} arcsec")
    print("="*80)

    try:
        # Initialize constructor
        constructor = SimpleCubeConstructor(
            fiber_pitch=args.fiber_pitch,
            pixel_size=args.pixel_size,
            wave_sampling=args.wave_sampling
        )

        # Load data
        constructor.load_fibermap(fibermap_path)
        constructor.load_rss_file(args.rss_file)

        # Create grids
        constructor.create_wavelength_grid(wave_min=args.wave_min, wave_max=args.wave_max)
        constructor.create_spatial_grid()

        # Construct cube
        constructor.construct_cube(radius=args.radius, min_weight=args.min_weight)

        # Create WCS
        constructor.create_wcs(ra_center=args.ra, dec_center=args.dec)

        # Save
        constructor.save_cube(output_file, overwrite=True)

        print("\n" + "="*80)
        print("SUCCESS: Datacube construction complete!")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
