"""
Multi-Detector IFU Flat-Fielding Module

Adapted from PypeIt's flat-fielding approach for a 24-detector fiber IFU system.
This implementation handles per-detector normalization while preserving fiber-to-fiber
throughput variations and pixel-to-pixel QE differences.

Key features:
- B-spline fitting for spectral response
- 2D polynomial fitting for residual structure
- Per-detector normalization relative to reference fiber
- Preserves fiber throughput and pixel QE variations

Author: Adapted from PypeIt (Prochaska et al. 2020)
Date: 2025
"""

import numpy as np
from scipy import interpolate
from scipy.linalg import cholesky_banded
from scipy.ndimage import median_filter, gaussian_filter
import warnings


class BSplineFitter:
    """
    B-spline fitting class adapted from PypeIt for IFU flat-fielding.

    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g., wavelength or pixel coordinates)
    nord : int, optional
        B-spline order (4 = cubic, default)
    bkpt : np.ndarray, optional
        Breakpoint locations. If None, computed automatically
    """

    def __init__(self, x, nord=4, bkpt=None, npoly=1):
        self.x = np.asarray(x)
        self.nord = nord
        self.npoly = npoly

        # Set up breakpoints
        if bkpt is None:
            # Automatic breakpoint spacing based on data range
            n_bkpts = max(10, len(x) // 100)  # Adjust density as needed
            self.bkpt = np.linspace(x.min(), x.max(), n_bkpts)
        else:
            self.bkpt = np.asarray(bkpt)

        # Create full breakpoint array with padding
        self.fullbkpt = self._pad_breakpoints(self.bkpt, self.nord)

        # Initialize coefficients (will be filled by fit)
        self.coeff = None

    def _pad_breakpoints(self, bkpt, nord):
        """Add padding to breakpoints for B-spline boundary conditions"""
        # Pad with repeated edge values
        pad_left = np.repeat(bkpt[0], nord - 1)
        pad_right = np.repeat(bkpt[-1], nord - 1)
        return np.concatenate([pad_left, bkpt, pad_right])

    def action_matrix(self, x=None):
        """
        Construct the B-spline action matrix.

        Parameters
        ----------
        x : np.ndarray, optional
            Points at which to evaluate. If None, uses self.x

        Returns
        -------
        action : np.ndarray
            Action matrix (n_points, nord * npoly)
        lower : np.ndarray
            Lower indices for each segment
        upper : np.ndarray
            Upper indices for each segment
        """
        if x is None:
            x = self.x
        else:
            x = np.asarray(x)

        n = len(x)
        nbkpt = len(self.fullbkpt)
        n_segments = nbkpt - self.nord

        # Initialize action matrix
        action = np.zeros((n, self.nord * self.npoly), dtype=float)
        lower = np.zeros(n_segments, dtype=int)
        upper = np.zeros(n_segments, dtype=int)

        # Find which segment each x point belongs to
        segments = np.searchsorted(self.fullbkpt, x, side='right') - 1
        segments = np.clip(segments, self.nord - 1, n_segments - 1)

        # Build action matrix using scipy's BSpline basis functions
        for i in range(self.nord):
            # Get B-spline basis of order i
            basis = interpolate.BSpline.basis_element(
                self.fullbkpt[segments - self.nord + 1 + i:segments + 2 + i],
                extrapolate=False
            )

            # Evaluate at x points
            for j, (seg, xval) in enumerate(zip(segments, x)):
                t_local = self.fullbkpt[seg - self.nord + 1 + i:seg + 2 + i]
                if len(t_local) >= 2:
                    basis_i = interpolate.BSpline(
                        t_local,
                        np.eye(len(t_local))[0],
                        self.nord - 1
                    )
                    action[j, i * self.npoly] = basis_i(xval)

        # Compute lower and upper indices
        for seg in range(n_segments):
            mask = segments == seg
            if np.any(mask):
                indices = np.where(mask)[0]
                lower[seg] = indices[0]
                upper[seg] = indices[-1]
            else:
                lower[seg] = 0
                upper[seg] = -1  # Empty segment

        return action, lower, upper

    def fit(self, ydata, invvar=None, maxiter=10, upper_reject=5.0, lower_reject=5.0):
        """
        Fit B-spline to data with iterative rejection.

        Parameters
        ----------
        ydata : np.ndarray
            Data to fit
        invvar : np.ndarray, optional
            Inverse variance weights. If None, uniform weighting
        maxiter : int, optional
            Maximum rejection iterations
        upper_reject : float, optional
            Upper sigma rejection threshold
        lower_reject : float, optional
            Lower sigma rejection threshold

        Returns
        -------
        yfit : np.ndarray
            Best-fit model evaluated at self.x
        mask : np.ndarray
            Boolean mask (True = good point)
        """
        ydata = np.asarray(ydata)

        if invvar is None:
            invvar = np.ones_like(ydata)
        else:
            invvar = np.asarray(invvar)

        # Initial mask (exclude NaN/inf and zero inverse variance)
        mask = np.isfinite(ydata) & np.isfinite(invvar) & (invvar > 0)

        # Iterative fitting with rejection
        for iteration in range(maxiter):
            # Get action matrix for good points
            x_good = self.x[mask]
            y_good = ydata[mask]
            ivar_good = invvar[mask]

            if len(x_good) < self.nord:
                warnings.warn("Too few points for B-spline fit")
                return np.full_like(ydata, np.nan), mask

            # Solve weighted least squares using scipy's splrep
            try:
                # Use scipy's B-spline fitter
                weights = np.sqrt(ivar_good)
                tck = interpolate.splrep(x_good, y_good, w=weights,
                                        k=min(self.nord - 1, 3),
                                        s=len(x_good))  # Smoothing parameter

                # Evaluate fit at all x points
                yfit = interpolate.splev(self.x, tck, ext=1)  # ext=1: return zero outside domain

            except Exception as e:
                warnings.warn(f"B-spline fit failed: {e}")
                return np.full_like(ydata, np.nan), mask

            # Calculate residuals
            residuals = ydata - yfit
            sigma = np.sqrt(1.0 / np.clip(invvar, 1e-10, None))

            # Compute robust sigma for rejection
            good_residuals = residuals[mask]
            med_resid = np.median(good_residuals)
            mad = np.median(np.abs(good_residuals - med_resid))
            robust_sigma = 1.4826 * mad  # MAD to sigma conversion

            if robust_sigma == 0:
                break  # Perfect fit or all points identical

            # Reject outliers
            normalized_residuals = (residuals - med_resid) / robust_sigma
            new_mask = (normalized_residuals > -lower_reject) & \
                      (normalized_residuals < upper_reject)
            new_mask &= mask  # Keep previously good points

            # Check convergence
            if np.array_equal(new_mask, mask):
                break

            mask = new_mask

        return yfit, mask

    def value(self, x_new):
        """
        Evaluate fitted B-spline at new x values.

        Parameters
        ----------
        x_new : np.ndarray
            Points at which to evaluate

        Returns
        -------
        y_new : np.ndarray
            Evaluated values
        """
        if self.coeff is None:
            raise ValueError("Must call fit() before value()")

        # This would require storing tck from fit()
        # For now, return interpolation
        raise NotImplementedError("Use fit() return value directly")


class TwoDPolynomialFit:
    """
    2D polynomial fitting for residual flat-field structure.

    Fits residuals after spectral and spatial corrections have been applied.
    Adapted from PypeIt's twod_fit_npoly approach.
    """

    def __init__(self, degree=3):
        """
        Parameters
        ----------
        degree : int
            Polynomial degree (default 3 for PypeIt)
        """
        self.degree = degree
        self.coeffs = None

    def fit(self, x, y, z, weights=None, mask=None):
        """
        Fit 2D polynomial to data.

        Parameters
        ----------
        x : np.ndarray
            First coordinate (e.g., spatial)
        y : np.ndarray
            Second coordinate (e.g., spectral)
        z : np.ndarray
            Data values
        weights : np.ndarray, optional
            Weights for fitting
        mask : np.ndarray, optional
            Boolean mask (True = good)

        Returns
        -------
        z_fit : np.ndarray
            Fitted surface evaluated at (x, y)
        """
        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()
        z = np.asarray(z).ravel()

        if mask is None:
            mask = np.ones(len(x), dtype=bool)
        else:
            mask = np.asarray(mask).ravel()

        if weights is None:
            weights = np.ones(len(x))
        else:
            weights = np.asarray(weights).ravel()

        # Apply mask
        x_good = x[mask]
        y_good = y[mask]
        z_good = z[mask]
        w_good = weights[mask]

        # Normalize coordinates to [-1, 1] for numerical stability
        self.x_min, self.x_max = x.min(), x.max()
        self.y_min, self.y_max = y.min(), y.max()

        x_norm = 2 * (x_good - self.x_min) / (self.x_max - self.x_min) - 1
        y_norm = 2 * (y_good - self.y_min) / (self.y_max - self.y_min) - 1

        # Build design matrix
        n_terms = (self.degree + 1) * (self.degree + 2) // 2
        A = np.zeros((len(x_good), n_terms))

        term = 0
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                A[:, term] = (x_norm ** i) * (y_norm ** j)
                term += 1

        # Weighted least squares
        W = np.diag(w_good)
        ATA = A.T @ W @ A
        ATb = A.T @ W @ z_good

        # Solve with regularization to avoid singularities
        reg = 1e-10 * np.eye(n_terms)
        self.coeffs = np.linalg.solve(ATA + reg, ATb)

        # Evaluate at all input points
        x_norm_all = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        y_norm_all = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1

        A_all = np.zeros((len(x), n_terms))
        term = 0
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                A_all[:, term] = (x_norm_all ** i) * (y_norm_all ** j)
                term += 1

        z_fit = A_all @ self.coeffs
        return z_fit.reshape(z.shape)

    def evaluate(self, x, y):
        """Evaluate fitted polynomial at new coordinates"""
        if self.coeffs is None:
            raise ValueError("Must call fit() first")

        x = np.asarray(x).ravel()
        y = np.asarray(y).ravel()

        x_norm = 2 * (x - self.x_min) / (self.x_max - self.x_min) - 1
        y_norm = 2 * (y - self.y_min) / (self.y_max - self.y_min) - 1

        n_terms = (self.degree + 1) * (self.degree + 2) // 2
        A = np.zeros((len(x), n_terms))

        term = 0
        for i in range(self.degree + 1):
            for j in range(self.degree + 1 - i):
                A[:, term] = (x_norm ** i) * (y_norm ** j)
                term += 1

        return A @ self.coeffs


class MultiDetectorFlatField:
    """
    Main flat-fielding class for 24-detector IFU system.

    Implements the full PypeIt-style pipeline:
    1. Spectral response fitting (B-spline along wavelength)
    2. Spatial illumination fitting (B-spline along slit/fiber)
    3. 2D residual fitting (polynomial)
    4. Per-detector normalization

    Parameters
    ----------
    n_detectors : int
        Number of detectors (24 for your system)
    reference_fiber : int
        Index of reference fiber for normalization (fiber 150 in your case)
    """

    def __init__(self, n_detectors=24, reference_fiber=150):
        self.n_detectors = n_detectors
        self.reference_fiber = reference_fiber

        # Storage for detector-level calibrations
        self.pixelflats = []  # Per-detector pixel flats
        self.illumflats = []  # Per-detector illumination corrections
        self.flat_models = []  # Full flat models
        self.spec_responses = []  # Spectral response functions

    def process_detector(self, flat_image, fiber_ids, wavelengths,
                        detector_idx, variance=None):
        """
        Process flat field for a single detector.

        Parameters
        ----------
        flat_image : np.ndarray
            2D flat field image (spatial x spectral)
        fiber_ids : np.ndarray
            1D array of fiber IDs for each spatial position
        wavelengths : np.ndarray
            2D wavelength solution array (same shape as flat_image)
        detector_idx : int
            Detector index (0-23)
        variance : np.ndarray, optional
            Variance array for weighting

        Returns
        -------
        pixel_flat : np.ndarray
            Normalized pixel flat (detector response / fiber throughput)
        illum_flat : np.ndarray
            Spatial illumination correction
        flat_model : np.ndarray
            Full flat model
        """
        print(f"Processing detector {detector_idx + 1}/{self.n_detectors}")

        ny, nx = flat_image.shape

        # Initialize inverse variance
        if variance is None:
            invvar = np.ones_like(flat_image)
        else:
            invvar = np.where(variance > 0, 1.0 / variance, 0.0)

        # Mask bad pixels
        mask = np.isfinite(flat_image) & (flat_image > 0) & (invvar > 0)

        # Step 1: Fit spectral response
        print("  Fitting spectral response...")
        spec_response = self._fit_spectral_response(
            flat_image, wavelengths, mask, invvar
        )

        # Normalize by spectral response
        flat_norm_spec = np.where(spec_response > 0,
                                  flat_image / spec_response,
                                  0)

        # Step 2: Fit spatial illumination profile
        print("  Fitting spatial illumination...")
        illum_profile = self._fit_spatial_illumination(
            flat_norm_spec, fiber_ids, mask, invvar
        )

        # Normalize by both spectral and spatial
        flat_norm_both = np.where(illum_profile > 0,
                                  flat_norm_spec / illum_profile,
                                  0)

        # Step 3: Fit 2D residuals
        print("  Fitting 2D residual structure...")
        residual_2d = self._fit_2d_residuals(
            flat_norm_both, mask, invvar
        )

        # Build full flat model
        flat_model = spec_response * illum_profile * residual_2d

        # Compute pixel flat (preserve fiber throughput & pixel QE)
        pixel_flat = np.where(flat_model > 0,
                             flat_image / flat_model,
                             1.0)

        # Store results
        self.pixelflats.append(pixel_flat)
        self.illumflats.append(illum_profile)
        self.flat_models.append(flat_model)
        self.spec_responses.append(spec_response)

        return pixel_flat, illum_profile, flat_model

    def _fit_spectral_response(self, flat_image, wavelengths, mask, invvar):
        """
        Fit spectral response using B-spline along wavelength direction.

        Collapses spatially and fits 1D B-spline.
        """
        ny, nx = flat_image.shape

        # Collapse spatially (weighted by inverse variance)
        weights = invvar * mask
        weights_sum = np.sum(weights, axis=0)
        spec_collapsed = np.where(weights_sum > 0,
                                 np.sum(flat_image * weights, axis=0) / weights_sum,
                                 0)

        # Get representative wavelength for each spectral pixel
        wave_1d = np.median(wavelengths, axis=0)

        # Mask bad spectral channels
        spec_mask = (spec_collapsed > 0) & (weights_sum > 0) & np.isfinite(wave_1d)

        # Fit B-spline
        bspline = BSplineFitter(wave_1d[spec_mask], nord=4)
        spec_fit_1d = np.zeros_like(spec_collapsed)

        try:
            ivar_1d = weights_sum[spec_mask] / np.sum(mask, axis=0)[spec_mask]
            spec_fit_good, fit_mask = bspline.fit(
                spec_collapsed[spec_mask],
                invvar=ivar_1d,
                maxiter=5
            )
            spec_fit_1d[spec_mask] = spec_fit_good

        except Exception as e:
            warnings.warn(f"Spectral B-spline fit failed: {e}. Using median filter.")
            spec_fit_1d[spec_mask] = median_filter(
                spec_collapsed[spec_mask], size=51, mode='nearest'
            )

        # Broadcast to 2D
        spec_response_2d = np.tile(spec_fit_1d, (ny, 1))

        # Normalize to median (preserves scale differences between fibers)
        median_response = np.median(spec_fit_1d[spec_fit_1d > 0])
        if median_response > 0:
            spec_response_2d /= median_response

        return spec_response_2d

    def _fit_spatial_illumination(self, flat_image, fiber_ids, mask, invvar):
        """
        Fit spatial illumination profile using B-spline along fiber direction.

        This preserves fiber-to-fiber throughput variations.
        """
        ny, nx = flat_image.shape

        # Collapse spectrally
        weights = invvar * mask
        weights_sum = np.sum(weights, axis=1)
        spat_collapsed = np.where(weights_sum > 0,
                                 np.sum(flat_image * weights, axis=1) / weights_sum,
                                 0)

        # Median filter to smooth
        spat_smoothed = median_filter(spat_collapsed, size=5, mode='nearest')

        # Gaussian smooth for illumination profile
        spat_illum = gaussian_filter(spat_smoothed, sigma=3)

        # Fit B-spline
        spatial_coord = np.arange(ny)
        spat_mask = (spat_illum > 0) & (weights_sum > 0)

        if np.sum(spat_mask) > 10:
            bspline = BSplineFitter(spatial_coord[spat_mask], nord=4)
            try:
                ivar_spat = weights_sum[spat_mask] / np.sum(mask, axis=1)[spat_mask]
                spat_fit, fit_mask = bspline.fit(
                    spat_illum[spat_mask],
                    invvar=ivar_spat,
                    maxiter=3
                )
                spat_illum[spat_mask] = spat_fit
            except Exception as e:
                warnings.warn(f"Spatial B-spline fit failed: {e}")

        # Broadcast to 2D
        illum_profile_2d = np.tile(spat_illum[:, np.newaxis], (1, nx))

        # Normalize to median
        median_illum = np.median(spat_illum[spat_illum > 0])
        if median_illum > 0:
            illum_profile_2d /= median_illum

        return illum_profile_2d

    def _fit_2d_residuals(self, flat_residual, mask, invvar):
        """
        Fit 2D polynomial to residual structure.

        This captures any remaining 2D patterns not captured by separable
        spectral and spatial components.
        """
        ny, nx = flat_residual.shape

        # Create coordinate grids
        yy, xx = np.mgrid[0:ny, 0:nx]

        # Normalize coordinates
        yy_norm = yy.astype(float) / ny
        xx_norm = xx.astype(float) / nx

        # Fit 2D polynomial
        poly_fitter = TwoDPolynomialFit(degree=3)

        weights = invvar * mask
        weights = np.where(weights > 0, np.sqrt(weights), 0)

        try:
            residual_fit_2d = poly_fitter.fit(
                yy_norm, xx_norm, flat_residual,
                weights=weights, mask=mask
            )
        except Exception as e:
            warnings.warn(f"2D polynomial fit failed: {e}. Using constant.")
            residual_fit_2d = np.ones_like(flat_residual)

        # Normalize to median
        median_resid = np.median(residual_fit_2d[mask])
        if median_resid > 0:
            residual_fit_2d /= median_resid

        # Clip extreme values
        residual_fit_2d = np.clip(residual_fit_2d, 0.5, 2.0)

        return residual_fit_2d

    def normalize_multi_detector(self, scale_to_reference=True):
        """
        Normalize all detectors to a common scale.

        This is critical for multi-detector systems: normalize each detector's
        response scale while preserving relative pixel-to-pixel variations.

        Parameters
        ----------
        scale_to_reference : bool
            If True, scale all detectors to match the reference fiber's response
        """
        print("\nNormalizing across detectors...")

        if len(self.pixelflats) != self.n_detectors:
            warnings.warn(f"Only {len(self.pixelflats)} detectors processed")

        if not scale_to_reference:
            return

        # Find reference detector (the one containing reference fiber)
        # For your system, you'd determine this from fiber mappings
        ref_det_idx = self.reference_fiber // (1500 // self.n_detectors)  # Approximate

        if ref_det_idx >= len(self.pixelflats):
            warnings.warn(f"Reference detector {ref_det_idx} not found")
            return

        # Get reference median
        ref_median = np.median(self.pixelflats[ref_det_idx])

        # Scale all detectors
        for i, pixelflat in enumerate(self.pixelflats):
            det_median = np.median(pixelflat)
            if det_median > 0:
                scale_factor = ref_median / det_median
                self.pixelflats[i] *= scale_factor
                self.flat_models[i] *= scale_factor

                print(f"  Detector {i}: scale factor = {scale_factor:.4f}")

    def save_calibrations(self, output_prefix):
        """Save calibration products to FITS files"""
        from astropy.io import fits

        for i, (pixflat, illum, model) in enumerate(
            zip(self.pixelflats, self.illumflats, self.flat_models)
        ):
            # Create HDU list
            hdul = fits.HDUList([
                fits.PrimaryHDU(),
                fits.ImageHDU(pixflat, name='PIXELFLAT'),
                fits.ImageHDU(illum, name='ILLUMFLAT'),
                fits.ImageHDU(model, name='FLATMODEL'),
            ])

            # Add header info
            hdul[0].header['DETECTOR'] = i
            hdul[0].header['REFFIBER'] = self.reference_fiber
            hdul[0].header['COMMENT'] = 'Multi-detector IFU flat-field calibration'

            # Save
            filename = f"{output_prefix}_detector{i:02d}_flat.fits"
            hdul.writeto(filename, overwrite=True)
            print(f"Saved {filename}")


def example_usage():
    """
    Example of how to use the MultiDetectorFlatField class.
    """
    # Initialize the flat-field processor
    ff = MultiDetectorFlatField(n_detectors=24, reference_fiber=150)

    # Process each detector
    for det in range(24):
        # Load your flat field data for this detector
        # flat_image = load_flat_for_detector(det)
        # fiber_ids = get_fiber_ids_for_detector(det)
        # wavelengths = get_wavelength_solution_for_detector(det)

        # For this example, create dummy data
        ny, nx = 2048, 4096  # Example dimensions
        flat_image = np.random.randn(ny, nx) * 0.1 + 1.0  # ~10% noise
        fiber_ids = np.arange(ny)
        wavelengths = np.tile(np.linspace(4000, 9000, nx), (ny, 1))

        # Process this detector
        pixel_flat, illum_flat, flat_model = ff.process_detector(
            flat_image, fiber_ids, wavelengths, det
        )

    # Normalize across detectors
    ff.normalize_multi_detector(scale_to_reference=True)

    # Save results
    ff.save_calibrations("llamas_flat_calibration")

    print("\nFlat-field calibration complete!")


if __name__ == "__main__":
    # Run example
    print("Multi-Detector IFU Flat-Field Calibration")
    print("=" * 50)
    example_usage()
