#!/usr/bin/env python3
"""
Simple MUSE-like Data Cube Constructor for LLAMAS

This standalone script constructs 3D data cubes from LLAMAS RSS FITS files
following the MUSE data reduction pipeline methodology, adapted for LLAMAS
fiber geometry and wavelength coverage.

Supports three spatial grid methods:
  - oversampled:  Fine rectangular grid with IDW interpolation (default)
  - native_hex:   One spaxel per fiber in the native 52x46 hex layout
  - nearest_hex:  Rectangular grid at fiber pitch with nearest-fiber assignment

Key features:
- Proper fiber mapping via RSS FIBERMAP extension (BENCHSIDE + FIBER_ID)
- Drizzle-like resampling with adjustable pixel fraction
- Spatial interpolation using inverse distance weighting
- Regular wavelength grid with optional oversampling
- Variance propagation (when available)
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

from llamas_pyjamas.Image.WhiteLightModule import FiberMap_LUT

# LLAMAS IFU geometry constants
LLAMAS_FOV = 35.0          # Field of view in arcseconds
LLAMAS_NFIBERS = 2392      # Total spatial elements
LLAMAS_HEX_ROWS = 52       # Number of hex rows
LLAMAS_HEX_COLS = 46       # Fibers per hex row
LLAMAS_FILL_FACTOR = 0.93  # ~93% fill factor
LLAMAS_HEX_ROW_SPACING = np.sqrt(3) / 2  # Row spacing in pitch units


class SimpleCubeConstructor:
    """
    Constructs 3D datacubes from LLAMAS RSS files using MUSE-like methodology.

    Supports three spatial grid methods controlled by the grid_method parameter.
    Fiber positions are read from the RSS FIBERMAP extension and looked up via
    the FiberMap_LUT to ensure correct IFU spatial mapping.
    """

    VALID_GRID_METHODS = ('oversampled', 'native_hex', 'nearest_hex')

    def __init__(self, fiber_pitch=0.75, pixel_size=0.3, wave_sampling=1.0,
                 grid_method='oversampled'):
        """
        Initialize the cube constructor.

        Parameters:
            fiber_pitch (float): Fiber-to-fiber pitch in arcseconds (default: 0.75")
            pixel_size (float): Output spatial pixel size in arcseconds (default: 0.3")
            wave_sampling (float): Wavelength sampling factor (1.0 = native, <1.0 = oversample)
            grid_method (str): Spatial grid method - 'oversampled', 'native_hex', or 'nearest_hex'
        """
        if grid_method not in self.VALID_GRID_METHODS:
            raise ValueError(f"grid_method must be one of {self.VALID_GRID_METHODS}, got '{grid_method}'")

        self.fiber_pitch = fiber_pitch
        self.pixel_size = pixel_size
        self.wave_sampling = wave_sampling
        self.grid_method = grid_method

        # Will be loaded from RSS file
        self.flux = None
        self.error = None
        self.wave = None
        self.mask = None
        self.fwhm = None

        # Fiber identity from RSS FIBERMAP extension
        self.benchside = None
        self.fiber_id = None

        # Matched fiber coordinates (set by match_fibers_to_fibermap)
        self.fiber_coords = None
        self.fiber_indices = None  # RSS row indices for matched fibers

        # Output cube
        self.cube = None
        self.cube_var = None
        self.cube_weight = None
        self.wave_grid = None
        self.wcs = None

    def load_rss_file(self, rss_file):
        """
        Load RSS FITS file containing flux, wavelength, error, mask, and FIBERMAP.

        Parameters:
            rss_file (str): Path to RSS FITS file
        """
        print(f"\nLoading RSS file: {rss_file}")

        if not os.path.exists(rss_file):
            raise FileNotFoundError(f"RSS file not found: {rss_file}")

        with fits.open(rss_file) as hdul:
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

            # Load FIBERMAP extension (critical for correct fiber mapping)
            if 'FIBERMAP' in hdul:
                fibermap_hdu = hdul['FIBERMAP']
                raw_benchside = np.array(fibermap_hdu.data['BENCHSIDE'])
                # Handle byte-string decoding
                self.benchside = np.array([
                    val.decode('utf-8').strip() if isinstance(val, bytes) else str(val).strip()
                    for val in raw_benchside
                ])
                self.fiber_id = np.array(fibermap_hdu.data['FIBER_ID'])
                print(f"  FIBERMAP: {len(self.benchside)} entries, "
                      f"benchsides: {sorted(set(self.benchside))}")
            else:
                raise ValueError(
                    "No FIBERMAP extension found in RSS file. "
                    "The FIBERMAP extension with BENCHSIDE and FIBER_ID columns "
                    "is required for correct fiber-to-IFU mapping."
                )

        # Validate shapes
        if self.flux.shape != self.wave.shape:
            raise ValueError(f"FLUX and WAVE shapes don't match: {self.flux.shape} vs {self.wave.shape}")

        nfibers = self.flux.shape[0]
        if len(self.benchside) != nfibers:
            raise ValueError(
                f"FIBERMAP has {len(self.benchside)} entries but FLUX has {nfibers} rows"
            )

        print(f"  Loaded {nfibers} fibers x {self.flux.shape[1]} pixels")

    def match_fibers_to_fibermap(self):
        """
        Match RSS fiber indices to IFU spatial positions using FiberMap_LUT.

        Uses BENCHSIDE and FIBER_ID from the RSS FIBERMAP extension to look up
        each fiber's (x, y) position in the IFU focal plane.

        Returns:
            fiber_coords (ndarray): (N, 2) array of (x, y) positions in arcsec
        """
        print("\nMatching RSS fibers to fibermap...")

        if self.benchside is None or self.fiber_id is None:
            raise RuntimeError("FIBERMAP not loaded. Call load_rss_file() first.")

        nfibers = self.flux.shape[0]
        coords = []
        valid_indices = []

        for i in range(nfibers):
            benchside = self.benchside[i]
            fiber_id = int(self.fiber_id[i])

            x, y = FiberMap_LUT(benchside, fiber_id)

            if x == -1 and y == -1:
                print(f"  WARNING: No mapping for fiber {fiber_id} on bench {benchside}")
                continue

            # x, y from LUT are in grid units; convert to arcseconds
            coords.append([x * self.fiber_pitch, y * self.fiber_pitch])
            valid_indices.append(i)

        self.fiber_coords = np.array(coords)
        self.fiber_indices = np.array(valid_indices)

        n_matched = len(self.fiber_coords)
        n_skipped = nfibers - n_matched
        print(f"  Matched {n_matched}/{nfibers} fibers to spatial positions")
        if n_skipped > 0:
            print(f"  Skipped {n_skipped} fibers (no LUT mapping)")

        return self.fiber_coords

    def create_wavelength_grid(self, wave_min=None, wave_max=None):
        """
        Create regular wavelength grid for output cube.

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

        print(f"  Wavelength range: {wave_min:.2f} - {wave_max:.2f} Ang")
        print(f"  Native sampling: {median_dwave:.4f} Ang/pixel")
        print(f"  Output sampling: {dwave:.4f} Ang/pixel (factor: {self.wave_sampling})")
        print(f"  Number of wavelength pixels: {nwave}")

    def create_spatial_grid(self):
        """
        Create spatial grid for output cube based on fiber positions and grid_method.

        Must be called after match_fibers_to_fibermap().
        """
        print(f"\nCreating spatial grid (method: {self.grid_method})...")

        if self.fiber_coords is None:
            raise RuntimeError("Fiber coordinates not set. Call match_fibers_to_fibermap() first.")

        if self.grid_method == 'oversampled':
            self._create_oversampled_grid()
        elif self.grid_method == 'native_hex':
            self._create_native_hex_grid()
        elif self.grid_method == 'nearest_hex':
            self._create_nearest_hex_grid()

    def _create_oversampled_grid(self):
        """Create fine rectangular grid with sub-fiber-pitch pixel size."""
        x_arcsec = self.fiber_coords[:, 0]
        y_arcsec = self.fiber_coords[:, 1]

        margin = 2.0 * self.fiber_pitch  # ~1.5" margin
        x_min = np.min(x_arcsec) - margin
        x_max = np.max(x_arcsec) + margin
        y_min = np.min(y_arcsec) - margin
        y_max = np.max(y_arcsec) + margin

        nx = int((x_max - x_min) / self.pixel_size) + 1
        ny = int((y_max - y_min) / self.pixel_size) + 1

        self.x_grid = np.linspace(x_min, x_max, nx)
        self.y_grid = np.linspace(y_min, y_max, ny)

        print(f"  X range: {x_min:.2f} - {x_max:.2f} arcsec ({nx} pixels)")
        print(f"  Y range: {y_min:.2f} - {y_max:.2f} arcsec ({ny} pixels)")
        print(f"  Spatial pixel size: {self.pixel_size} arcsec")
        print(f"  Total spatial pixels: {nx} x {ny} = {nx*ny:,}")

    def _create_native_hex_grid(self):
        """Create grid matching the native IFU hex layout (52 rows x 46 cols)."""
        # In the LUT, the hex grid has:
        #   - y positions at multiples of sqrt(3)/2 in grid units
        #   - x positions at integer steps, offset by 0.5 on odd rows
        # We map each fiber to its (row, col) in the 52x46 grid.

        nrows = LLAMAS_HEX_ROWS
        ncols = LLAMAS_HEX_COLS

        # Build y_grid and x_grid in arcseconds matching the hex centers
        # Row spacing in arcsec = sqrt(3)/2 * fiber_pitch
        row_spacing = LLAMAS_HEX_ROW_SPACING * self.fiber_pitch
        # Column spacing in arcsec = fiber_pitch
        col_spacing = self.fiber_pitch

        # Grid centered on actual fiber extent
        y_arcsec = self.fiber_coords[:, 1]
        x_arcsec = self.fiber_coords[:, 0]

        y_min_fiber = np.min(y_arcsec)
        x_min_fiber = np.min(x_arcsec)

        self.y_grid = y_min_fiber + np.arange(nrows) * row_spacing
        self.x_grid = x_min_fiber + np.arange(ncols) * col_spacing

        # Build the fiber-to-grid mapping
        # For each fiber, find which (row, col) it belongs to
        self._hex_fiber_map = {}  # (row, col) -> fiber_index in fiber_coords

        for idx in range(len(self.fiber_coords)):
            fx, fy = self.fiber_coords[idx]

            # Find nearest row
            row = int(round((fy - y_min_fiber) / row_spacing))
            if row < 0 or row >= nrows:
                continue

            # For odd rows, x is offset by half a pitch
            if row % 2 == 1:
                col = int(round((fx - x_min_fiber - 0.5 * self.fiber_pitch) / col_spacing))
            else:
                col = int(round((fx - x_min_fiber) / col_spacing))

            if col < 0 or col >= ncols:
                continue

            self._hex_fiber_map[(row, col)] = idx

        print(f"  Native hex grid: {nrows} rows x {ncols} cols = {nrows * ncols} spaxels")
        print(f"  Mapped {len(self._hex_fiber_map)} fibers to hex positions")
        print(f"  Row spacing: {row_spacing:.4f} arcsec")
        print(f"  Column spacing: {col_spacing:.4f} arcsec")

    def _create_nearest_hex_grid(self):
        """Create rectangular grid at fiber pitch with nearest-fiber assignment."""
        x_arcsec = self.fiber_coords[:, 0]
        y_arcsec = self.fiber_coords[:, 1]

        margin = self.fiber_pitch
        x_min = np.min(x_arcsec) - margin
        x_max = np.max(x_arcsec) + margin
        y_min = np.min(y_arcsec) - margin
        y_max = np.max(y_arcsec) + margin

        nx = int((x_max - x_min) / self.fiber_pitch) + 1
        ny = int((y_max - y_min) / self.fiber_pitch) + 1

        self.x_grid = np.linspace(x_min, x_max, nx)
        self.y_grid = np.linspace(y_min, y_max, ny)

        print(f"  X range: {x_min:.2f} - {x_max:.2f} arcsec ({nx} pixels)")
        print(f"  Y range: {y_min:.2f} - {y_max:.2f} arcsec ({ny} pixels)")
        print(f"  Spatial pixel size: {self.fiber_pitch} arcsec (= fiber pitch)")
        print(f"  Total spatial pixels: {nx} x {ny} = {nx*ny:,}")

    def resample_fiber_to_grid(self, fiber_idx):
        """
        Resample a single fiber spectrum onto the common wavelength grid.

        Parameters:
            fiber_idx (int): Index of fiber in RSS arrays

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
            return None, None

        # Sort by wavelength
        sort_idx = np.argsort(wave_fiber[valid])
        wave_sorted = wave_fiber[valid][sort_idx]
        flux_sorted = flux_fiber[valid][sort_idx]

        # Interpolate onto common grid
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
                interp_var = np.interp(self.wave_grid, wave_sorted, var_sorted,
                                      left=np.nan, right=np.nan)
            except Exception:
                interp_var = None
        else:
            interp_var = None

        return interp_flux, interp_var

    def construct_cube(self, radius=1.5, min_weight=0.01):
        """
        Construct 3D datacube using the selected grid method.

        Parameters:
            radius (float): Maximum distance (in arcsec) for fiber contribution (oversampled only)
            min_weight (float): Minimum weight threshold for including a fiber (oversampled only)
        """
        if self.grid_method == 'oversampled':
            self._construct_cube_oversampled(radius, min_weight)
        elif self.grid_method == 'native_hex':
            self._construct_cube_native_hex()
        elif self.grid_method == 'nearest_hex':
            self._construct_cube_nearest_hex()

    def _construct_cube_oversampled(self, radius, min_weight):
        """Construct cube using IDW interpolation on oversampled rectangular grid."""
        print("\nConstructing datacube (oversampled IDW)...")
        print(f"  Interpolation radius: {radius} arcsec")
        print(f"  Minimum weight: {min_weight}")

        fiber_coords = self.fiber_coords
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
        resampled_data = self._resample_all_fibers()

        print("\n  Spatially interpolating onto cube grid...")

        # Process each output spatial pixel
        total_pixels = len(output_coords)

        for pix_idx in range(total_pixels):
            if pix_idx % 10000 == 0:
                print(f"    Pixel {pix_idx}/{total_pixels} ({100*pix_idx/total_pixels:.1f}%)...")

            out_y = pix_idx // nx
            out_x = pix_idx % nx
            out_pos = output_coords[pix_idx]

            fiber_indices = neighbors[pix_idx]

            if len(fiber_indices) == 0:
                continue

            # Calculate weights based on distance
            distances = np.linalg.norm(fiber_coords[fiber_indices] - out_pos, axis=1)

            # Gaussian kernel weighting
            sigma = radius / 2.0
            weights = np.exp(-0.5 * (distances / sigma)**2)

            valid_weights = weights >= min_weight
            if not np.any(valid_weights):
                continue

            weights = weights[valid_weights]
            fiber_indices = np.array(fiber_indices)[valid_weights]
            weights /= np.sum(weights)

            # Combine fiber spectra
            combined_flux = np.zeros(nwave)
            combined_var = np.zeros(nwave) if self.cube_var is not None else None
            combined_weight = np.zeros(nwave)

            for fib_idx, w in zip(fiber_indices, weights):
                flux_resamp, var_resamp = resampled_data[fib_idx]

                if flux_resamp is None:
                    continue

                valid = np.isfinite(flux_resamp)
                combined_flux[valid] += w * flux_resamp[valid]
                combined_weight[valid] += w

                if combined_var is not None and var_resamp is not None:
                    combined_var[valid] += (w**2) * var_resamp[valid]

            nonzero = combined_weight > 0
            combined_flux[nonzero] /= combined_weight[nonzero]

            self.cube[:, out_y, out_x] = combined_flux
            self.cube_weight[:, out_y, out_x] = combined_weight

            if self.cube_var is not None:
                self.cube_var[nonzero, out_y, out_x] = combined_var[nonzero] / (combined_weight[nonzero]**2)

        self._print_cube_stats()

    def _construct_cube_native_hex(self):
        """Construct cube by placing each fiber directly into its hex grid position."""
        print("\nConstructing datacube (native hex)...")

        nwave = len(self.wave_grid)
        nrows = LLAMAS_HEX_ROWS
        ncols = LLAMAS_HEX_COLS

        self.cube = np.full((nwave, nrows, ncols), np.nan, dtype=np.float32)
        self.cube_weight = np.zeros((nwave, nrows, ncols), dtype=np.float32)

        if self.error is not None:
            self.cube_var = np.full((nwave, nrows, ncols), np.nan, dtype=np.float32)
        else:
            self.cube_var = None

        print(f"  Cube shape: {self.cube.shape} (wavelength, row, col)")

        # Resample all fibers
        resampled_data = self._resample_all_fibers()

        # Place each fiber at its hex grid position
        for (row, col), fib_idx in self._hex_fiber_map.items():
            flux_resamp, var_resamp = resampled_data[fib_idx]

            if flux_resamp is None:
                continue

            self.cube[:, row, col] = flux_resamp
            self.cube_weight[:, row, col] = np.where(np.isfinite(flux_resamp), 1.0, 0.0)

            if self.cube_var is not None and var_resamp is not None:
                self.cube_var[:, row, col] = var_resamp

        self._print_cube_stats()

    def _construct_cube_nearest_hex(self):
        """Construct cube using nearest-fiber assignment on rectangular grid."""
        print("\nConstructing datacube (nearest hex)...")

        fiber_coords = self.fiber_coords
        nfibers = len(fiber_coords)

        nwave = len(self.wave_grid)
        ny = len(self.y_grid)
        nx = len(self.x_grid)

        self.cube = np.full((nwave, ny, nx), np.nan, dtype=np.float32)
        self.cube_weight = np.zeros((nwave, ny, nx), dtype=np.float32)

        if self.error is not None:
            self.cube_var = np.full((nwave, ny, nx), np.nan, dtype=np.float32)
        else:
            self.cube_var = None

        print(f"  Cube shape: {self.cube.shape} (wavelength, y, x)")

        # Build KDTree
        tree = cKDTree(fiber_coords)

        # Create meshgrid of output positions
        xx, yy = np.meshgrid(self.x_grid, self.y_grid, indexing='xy')
        output_coords = np.column_stack([xx.ravel(), yy.ravel()])

        # Find nearest fiber for each output pixel
        distances, nearest_idx = tree.query(output_coords)

        # Only assign if within one fiber pitch
        max_dist = self.fiber_pitch * 1.1  # Small tolerance

        # Resample all fibers
        resampled_data = self._resample_all_fibers()

        print(f"\n  Assigning nearest fibers to {len(output_coords)} pixels...")

        for pix_idx in range(len(output_coords)):
            if distances[pix_idx] > max_dist:
                continue

            out_y = pix_idx // nx
            out_x = pix_idx % nx
            fib_idx = nearest_idx[pix_idx]

            flux_resamp, var_resamp = resampled_data[fib_idx]

            if flux_resamp is None:
                continue

            self.cube[:, out_y, out_x] = flux_resamp
            self.cube_weight[:, out_y, out_x] = np.where(np.isfinite(flux_resamp), 1.0, 0.0)

            if self.cube_var is not None and var_resamp is not None:
                self.cube_var[:, out_y, out_x] = var_resamp

        self._print_cube_stats()

    def _resample_all_fibers(self):
        """Resample all matched fibers onto the common wavelength grid."""
        nfibers = len(self.fiber_coords)
        resampled_data = []

        for i in range(nfibers):
            if i % 500 == 0:
                print(f"    Fiber {i}/{nfibers}...")

            rss_idx = self.fiber_indices[i]
            flux_resamp, var_resamp = self.resample_fiber_to_grid(rss_idx)
            resampled_data.append((flux_resamp, var_resamp))

        return resampled_data

    def _print_cube_stats(self):
        """Print datacube construction statistics."""
        print("\n  Datacube construction complete!")

        ny, nx = self.cube.shape[1], self.cube.shape[2]
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

        nx = len(self.x_grid) if hasattr(self, 'x_grid') and self.x_grid is not None else self.cube.shape[2]
        ny = len(self.y_grid) if hasattr(self, 'y_grid') and self.y_grid is not None else self.cube.shape[1]

        if self.grid_method == 'native_hex':
            # Native hex: pixel scale matches hex geometry
            # Column spacing = fiber_pitch, row spacing = fiber_pitch * sqrt(3)/2
            col_scale = self.fiber_pitch / 3600.0  # arcsec -> deg
            row_scale = (LLAMAS_HEX_ROW_SPACING * self.fiber_pitch) / 3600.0

            wcs.wcs.crpix = [nx / 2.0, ny / 2.0, 1]
            wcs.wcs.crval = [ra_center, dec_center, self.wave_grid[0]]
            wcs.wcs.cdelt = [-col_scale, row_scale, self.wave_grid[1] - self.wave_grid[0]]
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
            wcs.wcs.cunit = ['deg', 'deg', 'Angstrom']

        elif self.grid_method == 'nearest_hex':
            # Nearest hex: pixel scale = fiber pitch
            pixel_scale = self.fiber_pitch / 3600.0

            wcs.wcs.crpix = [nx / 2.0, ny / 2.0, 1]
            wcs.wcs.crval = [ra_center, dec_center, self.wave_grid[0]]
            wcs.wcs.cdelt = [-pixel_scale, pixel_scale, self.wave_grid[1] - self.wave_grid[0]]
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
            wcs.wcs.cunit = ['deg', 'deg', 'Angstrom']

        else:  # oversampled
            pixel_scale = self.pixel_size / 3600.0

            wcs.wcs.crpix = [nx / 2.0, ny / 2.0, 1]
            wcs.wcs.crval = [ra_center, dec_center, self.wave_grid[0]]
            wcs.wcs.cdelt = [-pixel_scale, pixel_scale, self.wave_grid[1] - self.wave_grid[0]]
            wcs.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
            wcs.wcs.cunit = ['deg', 'deg', 'Angstrom']

        self.wcs = wcs

        print(f"  Grid method: {self.grid_method}")
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

        hdu_list = fits.HDUList()

        # Primary HDU with datacube
        primary = fits.PrimaryHDU(data=self.cube)

        primary.header['EXTNAME'] = 'FLUX'
        primary.header['BUNIT'] = 'counts'
        primary.header['COMMENT'] = 'LLAMAS datacube - flux'

        # Construction parameters
        primary.header['PIXSIZE'] = (self.pixel_size, 'Spatial pixel size (arcsec)')
        primary.header['FIBPITCH'] = (self.fiber_pitch, 'Fiber pitch (arcsec)')
        primary.header['WAVESAMP'] = (self.wave_sampling, 'Wavelength sampling factor')
        primary.header['GRIDMETH'] = (self.grid_method, 'Spatial grid method')

        # IFU geometry
        primary.header['FOV'] = (LLAMAS_FOV, 'Field of view (arcsec)')
        primary.header['NSPAXEL'] = (LLAMAS_NFIBERS, 'Total IFU spatial elements')
        primary.header['HEXROWS'] = (LLAMAS_HEX_ROWS, 'Hex grid rows')
        primary.header['HEXCOLS'] = (LLAMAS_HEX_COLS, 'Hex grid columns')

        # Add WCS
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

        # For native_hex, add a SPAXEL_MAP table with per-spaxel coordinates
        if self.grid_method == 'native_hex' and hasattr(self, '_hex_fiber_map'):
            rows_list = []
            cols_list = []
            x_list = []
            y_list = []
            for (row, col), fib_idx in sorted(self._hex_fiber_map.items()):
                rows_list.append(row)
                cols_list.append(col)
                x_list.append(self.fiber_coords[fib_idx, 0])
                y_list.append(self.fiber_coords[fib_idx, 1])

            spaxel_table = Table()
            spaxel_table['ROW'] = np.array(rows_list, dtype=np.int32)
            spaxel_table['COL'] = np.array(cols_list, dtype=np.int32)
            spaxel_table['X_ARCSEC'] = np.array(x_list)
            spaxel_table['Y_ARCSEC'] = np.array(y_list)

            spaxel_hdu = fits.BinTableHDU(spaxel_table, name='SPAXEL_MAP')
            spaxel_hdu.header['COMMENT'] = 'Hex spaxel to arcsec coordinate mapping'
            hdu_list.append(spaxel_hdu)

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
  # Basic cube construction (oversampled grid, default)
  python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits

  # Native hex grid (one spaxel per fiber)
  python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits --grid-method native_hex

  # Nearest-fiber assignment at fiber pitch
  python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits --grid-method nearest_hex

  # Custom spatial sampling (oversampled only)
  python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits --pixel-size 0.2

  # Oversample wavelength axis
  python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits --wave-sampling 0.5
        """
    )

    # Required arguments
    parser.add_argument('rss_file', help='Input RSS FITS file')

    # Optional arguments
    parser.add_argument('--output', '-o', default=None,
                       help='Output FITS file (default: auto-generate from input name)')
    parser.add_argument('--grid-method', default='oversampled',
                       choices=['oversampled', 'native_hex', 'nearest_hex'],
                       help='Spatial grid method (default: oversampled)')
    parser.add_argument('--pixel-size', type=float, default=0.3,
                       help='Spatial pixel size in arcsec (default: 0.3, oversampled only)')
    parser.add_argument('--fiber-pitch', type=float, default=0.75,
                       help='Fiber-to-fiber pitch in arcsec (default: 0.75)')
    parser.add_argument('--wave-sampling', type=float, default=1.0,
                       help='Wavelength sampling factor (default: 1.0, <1.0 = oversample)')
    parser.add_argument('--radius', type=float, default=1.5,
                       help='Interpolation radius in arcsec (default: 1.5, oversampled only)')
    parser.add_argument('--min-weight', type=float, default=0.01,
                       help='Minimum weight threshold (default: 0.01, oversampled only)')
    parser.add_argument('--ra', type=float, default=0.0,
                       help='RA of field center in degrees (default: 0.0)')
    parser.add_argument('--dec', type=float, default=0.0,
                       help='Dec of field center in degrees (default: 0.0)')
    parser.add_argument('--wave-min', type=float, default=None,
                       help='Minimum wavelength in Angstroms (default: auto)')
    parser.add_argument('--wave-max', type=float, default=None,
                       help='Maximum wavelength in Angstroms (default: auto)')

    args = parser.parse_args()

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
    print(f"Output file: {output_file}")
    print(f"Grid method: {args.grid_method}")
    print(f"Spatial pixel size: {args.pixel_size} arcsec")
    print(f"Fiber pitch: {args.fiber_pitch} arcsec")
    print(f"Wavelength sampling: {args.wave_sampling}")
    if args.grid_method == 'oversampled':
        print(f"Interpolation radius: {args.radius} arcsec")
    print("="*80)

    try:
        # Initialize constructor
        constructor = SimpleCubeConstructor(
            fiber_pitch=args.fiber_pitch,
            pixel_size=args.pixel_size,
            wave_sampling=args.wave_sampling,
            grid_method=args.grid_method
        )

        # Load RSS file (includes FIBERMAP)
        constructor.load_rss_file(args.rss_file)

        # Match fibers to IFU positions
        constructor.match_fibers_to_fibermap()

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
