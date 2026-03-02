"""Wavelength calibration quality validation module.

This module provides quality checking functions for wavelength solutions
WITHOUT modifying the original arcSolve() wavelength generation algorithm.

Used by arcLlamasMulti.py to validate wavelength data during transfer.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_wavelength_solution(extraction_obj, channel, fiber_idx):
    """Validate wavelength solution quality for a single fiber.

    This function performs comprehensive validation of wavelength calibration
    data to catch common issues like missing calibration, NaN values,
    non-monotonic wavelengths, or out-of-range values.

    Args:
        extraction_obj: ExtractLlamas object containing wavelength data
        channel: Channel name ('red', 'green', 'blue')
        fiber_idx: Fiber index to validate

    Returns:
        dict: {
            'valid': bool - True if wavelength solution is valid
            'errors': list of error messages
            'warnings': list of warning messages
            'metrics': dict of quality metrics
        }
    """
    wave_data = extraction_obj.wave[fiber_idx, :]
    errors = []
    warnings = []
    metrics = {}

    # Check 1: Non-zero
    if np.all(wave_data == 0):
        errors.append("Wavelength array is all zeros (no calibration applied)")
        return {'valid': False, 'errors': errors, 'warnings': warnings, 'metrics': metrics}

    # Check 2: No NaNs or Infs
    finite_mask = np.isfinite(wave_data)
    n_bad = np.sum(~finite_mask)
    if n_bad > 0:
        errors.append(f"Wavelength array contains {n_bad} NaN/Inf values")
        if n_bad == len(wave_data):
            return {'valid': False, 'errors': errors, 'warnings': warnings, 'metrics': metrics}
        warnings.append(f"Using only {np.sum(finite_mask)} valid wavelength points")
        wave_data_clean = wave_data[finite_mask]
    else:
        wave_data_clean = wave_data

    # Check 3: Monotonically increasing
    dwave = np.diff(wave_data_clean)
    if not np.all(dwave > 0):
        n_decreasing = np.sum(dwave <= 0)
        errors.append(f"Wavelength not monotonically increasing ({n_decreasing} reversals)")
        return {'valid': False, 'errors': errors, 'warnings': warnings, 'metrics': metrics}

    # Check 4: Reasonable wavelength range for channel
    wv_ranges = {
        'red': (6500, 10100),
        'green': (4500, 7200),
        'blue': (3100, 5000)
    }

    if channel not in wv_ranges:
        warnings.append(f"Unknown channel '{channel}', skipping range check")
    else:
        expected_min, expected_max = wv_ranges[channel]
        actual_min, actual_max = wave_data_clean.min(), wave_data_clean.max()

        if actual_min < expected_min - 100 or actual_max > expected_max + 100:
            errors.append(
                f"Wavelength range [{actual_min:.1f}, {actual_max:.1f}] Å "
                f"outside expected [{expected_min}, {expected_max}] Å for {channel} channel"
            )
        elif actual_min < expected_min or actual_max > expected_max:
            warnings.append(
                f"Wavelength range [{actual_min:.1f}, {actual_max:.1f}] Å "
                f"slightly outside nominal [{expected_min}, {expected_max}] Å"
            )

    # Check 5: Reasonable dispersion (Å/pixel)
    median_dispersion = np.median(dwave)
    expected_dispersion = {
        'red': (1.5, 2.5),
        'green': (1.2, 2.0),
        'blue': (0.8, 1.5)
    }

    if channel in expected_dispersion:
        disp_min, disp_max = expected_dispersion[channel]
        if not (disp_min <= median_dispersion <= disp_max):
            warnings.append(
                f"Unusual dispersion {median_dispersion:.3f} Å/pixel "
                f"(expected {disp_min}-{disp_max} Å/pixel for {channel})"
            )

    # Calculate quality metrics
    metrics['wave_min'] = float(actual_min)
    metrics['wave_max'] = float(actual_max)
    metrics['wave_range'] = float(actual_max - actual_min)
    metrics['median_dispersion'] = float(median_dispersion)
    metrics['dispersion_std'] = float(np.std(dwave))
    metrics['n_valid_pixels'] = int(np.sum(finite_mask))
    metrics['n_total_pixels'] = int(len(wave_data))

    is_valid = len(errors) == 0

    return {
        'valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'metrics': metrics
    }
