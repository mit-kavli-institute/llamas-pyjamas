# Wavelength Calibration Issues and Solutions

**Date**: 2025-01-24
**Context**: Issues identified in `arcTransfer()` function and flat field wavelength calibration workflow
**Priority**: HIGH - These issues can cause systematic flux calibration errors in science data

---

## ⚠️ IMPORTANT: Scope of Changes

**THIS PLAN DOES NOT MODIFY `arcLlamas.py` - THE ORIGINAL FILE REMAINS COMPLETELY UNTOUCHED**

### Implementation Location:
**All changes are implemented in `Arc/arcLlamasMulti.py` to preserve `Arc/arcLlamas.py` in its original state.**

### What This Plan Changes:
- ✅ **`arcLlamasMulti.py::arcTransfer()` only** - Adds validation when transferring wavelength solutions to flat/science data
- ✅ **New validation functions** - Separate module `Arc/arcValidation.py` for quality checking (non-invasive)
- ✅ **Flat field pipeline protection** - Prevents bad wavelength data from corrupting flat fields
- ✅ **Optional feature** - Validation can be disabled with `enable_validation=False` parameter

### What This Plan Does NOT Change:
- ❌ **`Arc/arcLlamas.py`** - **ENTIRE FILE UNCHANGED** (preserved in original state)
- ❌ **`arcSolve()`** - Original wavelength solution generation algorithm (untouched)
- ❌ **`shiftArcX()`** - Cross-correlation shift calculation (untouched)
- ❌ **ThAr line matching** - Existing line identification logic (untouched)
- ❌ **Polynomial fitting** - Legendre polynomial wavelength fitting (untouched)

### Implementation Strategy:
1. ✅ **Created new file**: `Arc/arcValidation.py` for all validation code (keeps original code clean)
2. ✅ **Modified only**: `Arc/arcLlamasMulti.py::arcTransfer()` to call validation functions (optional with flag)
3. ✅ **Backwards compatible**: All validation can be disabled via `enable_validation=False` parameter
4. ✅ **Safe testing**: Original `Arc/arcLlamas.py` workflow completely unchanged

---

## Table of Contents

1. [Issue #3: No Validation of Wavelength Data Quality](#issue-3-no-validation-of-wavelength-data-quality)
2. [Issue #4: Fiber ID Correspondence Assumption](#issue-4-fiber-id-correspondence-assumption)
3. [Implementation Roadmap](#implementation-roadmap)
4. [Testing Strategy](#testing-strategy)

---

## Issue #3: No Validation of Wavelength Data Quality

### Problem Description

The `arcTransfer()` function ([Arc/arcLlamas.py:365-426](../../Arc/arcLlamas.py)) blindly copies wavelength calibration data from arc lamp extractions to science/flat field extractions without any quality validation:

```python
# Line 422-424: No validation before or after this copy
for ifiber in range(min_nfibers):
    scispec[fits_ext].wave[ifiber, :] = arcspec[arc_idx].wave[ifiber, :]
    scispec[fits_ext].xshift[ifiber, :] = arcspec[arc_idx].xshift[ifiber, :]
    scispec[fits_ext].relative_throughput[ifiber] = arcspec[arc_idx].relative_throughput[ifiber]
```

### Failure Modes

#### 3A: Line Matching Failed

**Location**: [Arc/arcLlamas.py:333-335](../../Arc/arcLlamas.py#L333-L335)

```python
if (len(final_fitx) == 0):
    print(f"No lines found in ThAr linelist for this channel {channel}")
    continue  # Skips to next channel - NO WAVELENGTH SOLUTION CREATED!
```

**What happens**:
- If no ThAr lines are matched (wrong lamp, bad spectrum, wavelength range off)
- The `final_arcfit` is never created for that channel
- The `.wave` array remains as zeros (initialized in `ExtractLlamas.__init__`)
- `arcTransfer` will blindly copy these zeros to the flat field

**Result**: All flat field fibers for that channel get `wave = [0, 0, 0, ..., 0]`

**Impact**: CRITICAL - Flat fielding completely fails for entire channel

#### 3B: Poor Fit Quality

**Location**: [Arc/arcLlamas.py:345-347](../../Arc/arcLlamas.py#L345-L347)

```python
rms = np.std(final_fitwv - final_arcfit.eval(final_fitx))
ax2.set_title(f'RMS = {rms:.2f} A')
# No threshold check - any fit quality is accepted!
```

**What happens**:
- If the fit has high RMS (e.g., 10 Å instead of <0.5 Å), wavelength solution is inaccurate
- No threshold check exists - any fit quality is accepted
- `arcTransfer` has no way to know if wavelength solution is garbage

**Result**: Inaccurate wavelengths propagate to flat field (few Å systematic errors)

**Impact**: MEDIUM - Causes spectral misalignment in pixel maps

#### 3C: Extrapolation Beyond Calibration Range

**Location**: [Arc/arcLlamas.py:358](../../Arc/arcLlamas.py#L358)

```python
arcspec_shifted[extension].wave[ifiber, :] = final_arcfit.eval(x)
# No check if x is within training range of final_fitx
```

**What happens**:
- If a fiber's `xshift` values fall outside the range of `final_fitx` (calibrated pixels)
- Legendre polynomial extrapolates (can diverge wildly outside training range)
- No warning is given for extrapolation

**Result**: Edge fibers may have completely wrong wavelengths

**Impact**: MEDIUM - Affects fibers at detector edges

#### 3D: NaN Propagation

**Location**: [Arc/arcLlamas.py:132-136](../../Arc/arcLlamas.py#L132-L136)

```python
if (success == 1):
    arcspec[fits_ext].xshift[ifiber, :] = (x*stretch + x**2*stretch2) + shift
else:
    print("....Warning: arc shift failed for this fiber!")
    arcspec[fits_ext].xshift[ifiber, :] = x  # Fallback to pixel coordinates
```

**What happens**:
- If cross-correlation fails and xshift gets bad values
- When `final_arcfit.eval(x)` is called with NaN/Inf values
- `.wave` array gets populated with NaNs
- `arcTransfer` copies these NaNs to flat field

**Result**: B-spline fitting in flat field fails (can't fit to NaN x-axis)

**Impact**: HIGH - Breaks flat field processing for affected fibers

---

### Solutions for Issue #3

**Solution Overview**: Add validation when wavelength data is transferred from arc to flat/science, WITHOUT modifying the original wavelength generation algorithm.

**Key Approach**:
- ✅ Create new `Arc/arcValidation.py` module (separate file, no risk)
- ✅ Modify only `arcTransfer()` to call validation (minimal change)
- ❌ Do NOT modify `arcSolve()` (wavelength generation stays intact)

**Priority**: Solution 3.1 (validation module) + Solution 3.3 (integrate into arcTransfer) are REQUIRED. Solutions 3.2 and 3.4 are OPTIONAL.

---

#### Solution 3.1: Add Wavelength Quality Validation Function

**⚠️ IMPORTANT**: Create new file `Arc/arcValidation.py` for all validation code (keeps `arcLlamas.py` unchanged)

**Location**: Create **NEW FILE** `Arc/arcValidation.py`

```python
"""Wavelength calibration quality validation module.

This module provides quality checking functions for wavelength solutions
WITHOUT modifying the original arcSolve() wavelength generation algorithm.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


def validate_wavelength_solution(extraction_obj, channel, fiber_idx):
    """Validate wavelength solution quality for a single fiber.

    Args:
        extraction_obj: ExtractLlamas object containing wavelength data
        channel: Channel name ('red', 'green', 'blue')
        fiber_idx: Fiber index to validate

    Returns:
        dict: {
            'valid': bool,
            'errors': list of error messages,
            'warnings': list of warning messages,
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
```

#### Solution 3.2: Add Arc Fit Quality Validation (OPTIONAL - Only if desired)

**⚠️ OPTIONAL ENHANCEMENT**: These changes to `arcSolve()` are **completely optional** and only add quality warnings. The original wavelength fitting algorithm remains unchanged.

**Recommendation**: Skip this solution initially. Only add after Solution 3.1 and 3.3 are tested and working.

**Location**: OPTIONAL modifications to `arcSolve()` in [Arc/arcLlamas.py:336](../../Arc/arcLlamas.py#L336)

**Option A: Add quality checks that store metadata but don't stop execution**

```python
# After line 345 (RMS calculation) - OPTIONAL addition
rms = np.std(final_fitwv - final_arcfit.eval(final_fitx))

# OPTIONAL: Store quality metrics in metadata for later validation
if 'wavelength_calib_metrics' not in arcdict:
    arcdict['wavelength_calib_metrics'] = {}

arcdict['wavelength_calib_metrics'][channel] = {
    'rms': float(rms),
    'n_lines': len(final_fitx),
    'fit_order': 5
}

# OPTIONAL: Print warning if quality is poor (but continue processing)
rms_threshold = 1.0  # Angstroms
if rms > rms_threshold:
    print(f"WARNING: Arc fit RMS = {rms:.2f} Å exceeds recommended threshold {rms_threshold} Å")
    print(f"  This may indicate poor wavelength calibration quality")
    print(f"  Found {len(final_fitx)} matched lines")
    print(f"  Channel: {channel}")
    # NO ERROR RAISED - just a warning

min_lines_recommended = 20
if len(final_fitx) < min_lines_recommended:
    print(f"WARNING: Only {len(final_fitx)} lines found (recommended: {min_lines_recommended})")
    print(f"  Wavelength solution may be less accurate")
    # NO ERROR RAISED - just a warning
```

**Option B: Leave `arcSolve()` completely unchanged (RECOMMENDED)**

Simply don't modify `arcSolve()` at all. All validation happens in `arcTransfer()` using the validation functions from `Arc/arcValidation.py`. This is the safest approach.

**Why Option B is recommended**:
- `arcSolve()` already creates the wavelength solution correctly
- User can visually inspect fit quality from the plots
- Adding validation at the transfer stage (Solution 3.3) catches all the same problems
- Keeps original code pristine and unchanged

#### Solution 3.3: Integrate Validation into arcTransfer

**Location**: Modify `arcTransfer()` in [Arc/arcLlamas.py:420-424](../../Arc/arcLlamas.py#L420-L424)

**This is the MAIN change** - adds validation before transferring wavelength data.

```python
# At the top of arcLlamas.py, add import:
from llamas_pyjamas.Arc.arcValidation import validate_wavelength_solution

# Then in arcTransfer(), replace the current transfer loop with validated version:

validation_results = {
    'n_fibers_total': 0,
    'n_fibers_valid': 0,
    'n_fibers_invalid': 0,
    'n_fibers_warned': 0,
    'failed_fibers': []
}

for ifiber in range(min_nfibers):
    validation_results['n_fibers_total'] += 1

    # Validate arc wavelength data before transfer
    channel = scidict['metadata'][fits_ext]['channel']
    validation = validate_wavelength_solution(arcspec[arc_idx], channel, ifiber)

    if not validation['valid']:
        validation_results['n_fibers_invalid'] += 1
        validation_results['failed_fibers'].append({
            'extension': fits_ext,
            'fiber': ifiber,
            'channel': channel,
            'bench': scidict['metadata'][fits_ext]['bench'],
            'side': scidict['metadata'][fits_ext]['side'],
            'errors': validation['errors']
        })

        # Log detailed error
        print(f"ERROR: Extension {fits_ext} ({channel}), Fiber {ifiber} failed validation:")
        for error in validation['errors']:
            print(f"  - {error}")

        # CRITICAL DECISION: Skip this fiber or fail entirely?
        # Option A: Skip fiber (set to NaN, handle downstream)
        scispec[fits_ext].wave[ifiber, :] = np.nan
        scispec[fits_ext].xshift[ifiber, :] = np.nan
        continue

        # Option B: Fail immediately (safer for production)
        # raise ValueError(f"Cannot transfer invalid wavelength calibration to fiber {ifiber}")

    if len(validation['warnings']) > 0:
        validation_results['n_fibers_warned'] += 1
        print(f"WARNING: Extension {fits_ext} ({channel}), Fiber {ifiber}:")
        for warning in validation['warnings']:
            print(f"  - {warning}")

    # Transfer wavelength data (now validated)
    scispec[fits_ext].wave[ifiber, :] = arcspec[arc_idx].wave[ifiber, :]
    scispec[fits_ext].xshift[ifiber, :] = arcspec[arc_idx].xshift[ifiber, :]
    scispec[fits_ext].relative_throughput[ifiber] = arcspec[arc_idx].relative_throughput[ifiber]

    validation_results['n_fibers_valid'] += 1

# Print summary after all fibers processed
print(f"\nWavelength transfer summary for extension {fits_ext}:")
print(f"  Total fibers: {validation_results['n_fibers_total']}")
print(f"  Valid: {validation_results['n_fibers_valid']}")
print(f"  Invalid: {validation_results['n_fibers_invalid']}")
print(f"  Warnings: {validation_results['n_fibers_warned']}")

if validation_results['n_fibers_invalid'] > 0:
    print(f"\nFailed fibers detail:")
    for failed in validation_results['failed_fibers']:
        print(f"  Extension {failed['extension']}, Fiber {failed['fiber']} "
              f"({failed['channel']} {failed['bench']}{failed['side']})")
        for error in failed['errors']:
            print(f"    - {error}")
```

#### Solution 3.4: Add Quality Metadata to Arc Dict (OPTIONAL)

**⚠️ OPTIONAL ENHANCEMENT**: Only needed if you want quality metrics stored in arc calibration files.

**Recommendation**: Skip this initially. Only add if you want historical quality tracking.

**Location**: OPTIONAL modification to `arcSolve()` around [Arc/arcLlamas.py:362](../../Arc/arcLlamas.py#L362)

```python
# OPTIONAL: Add import at top of arcLlamas.py
from llamas_pyjamas.Arc.arcValidation import validate_wavelength_solution

# Before saving arc solution (line 362) - OPTIONAL addition
print("Saving wavelength solution to disk")

# OPTIONAL: Store calibration quality metrics in metadata
for extension in range(len(arcspec_shifted)):
    channel = arcdict['metadata'][extension]['channel']

    # Calculate per-extension quality metrics
    n_valid_fibers = 0
    n_total_fibers = arcdict['metadata'][extension]['nfibers']
    wave_rms_per_fiber = []

    for ifiber in range(n_total_fibers):
        validation = validate_wavelength_solution(arcspec_shifted[extension], channel, ifiber)
        if validation['valid']:
            n_valid_fibers += 1
            if 'dispersion_std' in validation['metrics']:
                wave_rms_per_fiber.append(validation['metrics']['dispersion_std'])

    # Store in metadata
    arcdict['metadata'][extension]['wavelength_calib_valid'] = (n_valid_fibers == n_total_fibers)
    arcdict['metadata'][extension]['wavelength_n_valid_fibers'] = n_valid_fibers
    arcdict['metadata'][extension]['wavelength_n_total_fibers'] = n_total_fibers
    if len(wave_rms_per_fiber) > 0:
        arcdict['metadata'][extension]['wavelength_median_dispersion_std'] = float(np.median(wave_rms_per_fiber))

# Save with metadata
extract.save_extractions(arcspec_shifted, savefile='LLAMAS_reference_arc.pkl')
```

---

## Issue #4: Fiber ID Correspondence Assumption

### Problem Description

The `arcTransfer()` function assumes that fiber index `i` in the flat field corresponds to the same physical fiber as fiber index `i` in the arc lamp:

```python
# Line 420-424: Assumes flat fiber i = arc fiber i
for ifiber in range(min_nfibers):
    scispec[fits_ext].wave[ifiber, :] = arcspec[arc_idx].wave[ifiber, :]
```

**This assumption may be violated if**:
- Different trace files were used for flat vs. arc extractions
- Dead fibers are handled differently between flat and arc
- Trace file versions differ in fiber ordering
- Temporal instrument changes between observations

### Failure Modes

#### 4A: Different Trace Files Used

**Location**: [Flat/flatProcessing.py:144](../../Flat/flatProcessing.py#L144)

```python
# Flat field extraction uses one set of traces
_hdu_trace_pairs = match_hdu_to_traces(flat_hdus, flat_trace_files, start_idx=0)

# Arc extraction might use different traces (different directory, different version)
# No verification that the same traces were used!
```

**What happens**:
- Traces define which pixels belong to which fiber via `.fiberimg`
- If flat uses `trace_v1.pkl` and arc uses `trace_v2.pkl`
- Fiber numbering might differ between versions

**Example**:
```
trace_v1.pkl:  Fiber 0 → y-pixels [10-15]
               Fiber 1 → y-pixels [16-21]

trace_v2.pkl:  Fiber 0 → y-pixels [11-16]  # Off by 1 pixel
               Fiber 1 → y-pixels [17-22]
```

**Result**: Flat's fiber 0 gets arc's fiber 0 wavelength, but they correspond to different physical fibers

**Impact**: CRITICAL - Systematic wavelength mismatch across entire IFU

#### 4B: Dead Fiber Handling

**Location**: [Extract/extractLlamas.py:143-148](../../Extract/extractLlamas.py#L143-L148)

```python
if 'dead_fibers' in self.LUT and benchside in self.LUT['dead_fibers']:
    self.dead_fibers = self.LUT['dead_fibers'][benchside]
else:
    self.dead_fibers = []
```

**What happens**:
- If flat field extraction excludes dead fibers → 297 fibers
- If arc extraction includes dead fibers → 300 fibers
- The `min_nfibers` logic handles count mismatch but assumes same ordering

**Example**:
```
Flat (297 fibers, dead fiber 100 excluded):
  Fiber 0 → Physical fiber 0
  Fiber 1 → Physical fiber 1
  ...
  Fiber 100 → Physical fiber 101 (skipped dead fiber 100!)
  Fiber 101 → Physical fiber 102

Arc (300 fibers, includes dead):
  Fiber 0 → Physical fiber 0
  Fiber 1 → Physical fiber 1
  ...
  Fiber 100 → Physical fiber 100 (DEAD!)
  Fiber 101 → Physical fiber 101
```

**Result**: After fiber 100, all fiber indices are offset by 1 → wrong wavelengths for 200+ fibers!

**Impact**: HIGH - Systematic offset affects large fraction of fibers

#### 4C: Different Fiber Extraction Order

**What happens**:
```
Flat trace:  fibers sorted by y-position (bottom to top)
Arc trace:   fibers sorted by IFU spaxel ID (different spatial order)
```

**Result**: Fiber 0 in flat ≠ Fiber 0 in arc physically

**Impact**: CRITICAL - Complete scrambling of wavelength-fiber mapping

#### 4D: Trace Attribute Removed by Sanitization

**Location**: [Flat/flatLlamas.py:353](../../Flat/flatLlamas.py#L353)

```python
sanitized_flat_dict = sanitize_extraction_dict_for_pickling(flat_dict_calibrated)
```

**Location of sanitization**: [Flat/flatLlamas.py:62-98](../../Flat/flatLlamas.py#L62-L98)

```python
for extraction in extractions:
    if hasattr(extraction, 'trace'):
        extraction.trace = None  # ⚠️ TRACE LOST HERE!
```

**What happens**:
- After wavelength calibration, trace references are removed from extraction objects
- Can't verify trace file consistency later in pipeline
- No way to catch trace mismatches after this point

**Result**: No mechanism exists to validate trace file consistency

**Impact**: CRITICAL - Silent failures become undetectable

---

### Solutions for Issue #4

#### Solution 4.1: Add Physical Fiber ID Tracking

**Location**: Modify `ExtractLlamas.__init__()` in [Extract/extractLlamas.py:77-126](../../Extract/extractLlamas.py#L77-L126)

```python
# Add after line 126 (self.fiberid initialization)
self.fiberid = np.arange(trace.nfibers)  # Current implementation

# REPLACE with physical fiber ID tracking:
# Option A: If trace object has fiber IDs (preferred)
if hasattr(trace, 'fiber_physical_ids'):
    self.fiberid = trace.fiber_physical_ids.copy()
else:
    # Fallback: Create unique ID from detector + fiber index
    # Format: BBSSCCC where BB=bench, SS=side, CCC=fiber_index
    # Example: Bench 1, Side A, Fiber 0 → 1A000
    bench_code = int(self.bench)
    side_code = ord(self.side.upper())  # 'A' = 65, 'B' = 66
    self.fiberid = np.array([
        bench_code * 1000000 + side_code * 10000 + ifiber
        for ifiber in range(trace.nfibers)
    ])
    print(f"Warning: No physical fiber IDs in trace, using synthetic IDs")

# Also store fiber y-positions for spatial matching
if hasattr(trace, 'fiber_positions'):
    self.fiber_positions = trace.fiber_positions.copy()
else:
    # Extract from trace polynomial fits if available
    if hasattr(trace, 'poly_coeffs'):
        # Evaluate at middle of detector
        mid_x = trace.naxis1 // 2
        self.fiber_positions = np.array([
            np.polyval(trace.poly_coeffs[ifiber], mid_x)
            for ifiber in range(trace.nfibers)
        ])
    else:
        self.fiber_positions = None
        print(f"Warning: No fiber positions available for spatial matching")
```

**Add fiber ID to save/load**:

```python
# In save_extractions() at line 351-356
'metadata': [{
    'channel': ext.channel,
    'bench': ext.bench,
    'side': ext.side,
    'nfibers': ext.trace.nfibers,
    'fiberid': ext.fiberid.tolist(),  # ADD THIS
    'fiber_positions': ext.fiber_positions.tolist() if ext.fiber_positions is not None else None  # ADD THIS
} for ext in extraction_list]
```

#### Solution 4.2: Add Fiber ID Matching to arcTransfer

**Location**: Replace transfer loop in [Arc/arcLlamas.py:420-424](../../Arc/arcLlamas.py#L420-L424)

```python
# BEFORE FIBER LOOP: Check if fiber IDs are available
sci_has_fiberid = 'fiberid' in scidict['metadata'][fits_ext]
arc_has_fiberid = 'fiberid' in arcdict['metadata'][arc_idx]

use_fiberid_matching = sci_has_fiberid and arc_has_fiberid

if use_fiberid_matching:
    print(f"Using physical fiber ID matching for extension {fits_ext}")
    sci_fiberids = np.array(scidict['metadata'][fits_ext]['fiberid'])
    arc_fiberids = np.array(arcdict['metadata'][arc_idx]['fiberid'])
else:
    print(f"WARNING: Fiber IDs not available, using index-based matching (may be incorrect!)")
    print(f"  Science has fiberid: {sci_has_fiberid}")
    print(f"  Arc has fiberid: {arc_has_fiberid}")

# FIBER TRANSFER LOOP
n_matched = 0
n_unmatched = 0
unmatched_fibers = []

for sci_fiber_idx in range(sci_nfibers):
    if use_fiberid_matching:
        # Match by physical fiber ID
        sci_fiberid = sci_fiberids[sci_fiber_idx]

        # Find matching fiber in arc
        arc_fiber_match = np.where(arc_fiberids == sci_fiberid)[0]

        if len(arc_fiber_match) == 0:
            # No match found
            n_unmatched += 1
            unmatched_fibers.append((sci_fiber_idx, sci_fiberid))
            print(f"  WARNING: Science fiber {sci_fiber_idx} (ID {sci_fiberid}) has no match in arc")

            # Set to NaN to mark as invalid
            scispec[fits_ext].wave[sci_fiber_idx, :] = np.nan
            scispec[fits_ext].xshift[sci_fiber_idx, :] = np.nan
            scispec[fits_ext].relative_throughput[sci_fiber_idx] = np.nan
            continue

        elif len(arc_fiber_match) > 1:
            # Multiple matches (should not happen)
            print(f"  ERROR: Science fiber {sci_fiber_idx} (ID {sci_fiberid}) has {len(arc_fiber_match)} matches in arc!")
            print(f"    Matched arc fiber indices: {arc_fiber_match}")
            raise ValueError("Duplicate fiber IDs in arc calibration")

        arc_fiber_idx = arc_fiber_match[0]
        n_matched += 1

    else:
        # Fallback: index-based matching (UNSAFE!)
        arc_fiber_idx = sci_fiber_idx
        n_matched += 1

    # Validate wavelength before transfer (using solution 3.3)
    channel = scidict['metadata'][fits_ext]['channel']
    validation = validate_wavelength_solution(arcspec[arc_idx], channel, arc_fiber_idx)

    if not validation['valid']:
        print(f"  ERROR: Arc fiber {arc_fiber_idx} failed wavelength validation")
        for error in validation['errors']:
            print(f"    - {error}")
        scispec[fits_ext].wave[sci_fiber_idx, :] = np.nan
        scispec[fits_ext].xshift[sci_fiber_idx, :] = np.nan
        scispec[fits_ext].relative_throughput[sci_fiber_idx] = np.nan
        continue

    # Transfer calibration data from matched fiber
    scispec[fits_ext].wave[sci_fiber_idx, :] = arcspec[arc_idx].wave[arc_fiber_idx, :]
    scispec[fits_ext].xshift[sci_fiber_idx, :] = arcspec[arc_idx].xshift[arc_fiber_idx, :]
    scispec[fits_ext].relative_throughput[sci_fiber_idx] = arcspec[arc_idx].relative_throughput[arc_fiber_idx]

# Print matching summary
print(f"\nFiber matching summary for extension {fits_ext}:")
print(f"  Matched: {n_matched}/{sci_nfibers}")
print(f"  Unmatched: {n_unmatched}/{sci_nfibers}")
if n_unmatched > 0:
    print(f"  Unmatched fiber details:")
    for sci_idx, sci_id in unmatched_fibers[:10]:  # Show first 10
        print(f"    Science fiber {sci_idx} (ID {sci_id})")
    if len(unmatched_fibers) > 10:
        print(f"    ... and {len(unmatched_fibers) - 10} more")
```

#### Solution 4.3: Add Trace File Consistency Checking

**Location**: Create new validation function in `Utils/utils.py`

```python
def compute_trace_hash(trace_obj):
    """Compute hash of trace object for consistency checking.

    Args:
        trace_obj: TraceRay or TraceLlamas object

    Returns:
        str: MD5 hash of key trace attributes
    """
    import hashlib
    import json

    # Collect key identifying attributes
    trace_info = {
        'channel': trace_obj.channel,
        'bench': trace_obj.bench,
        'side': trace_obj.side,
        'nfibers': trace_obj.nfibers,
        'naxis1': trace_obj.naxis1,
        'naxis2': trace_obj.naxis2,
    }

    # Add fiberimg shape and checksum
    if hasattr(trace_obj, 'fiberimg') and trace_obj.fiberimg is not None:
        trace_info['fiberimg_shape'] = trace_obj.fiberimg.shape
        trace_info['fiberimg_checksum'] = int(np.sum(trace_obj.fiberimg))

    # Add fiber positions if available
    if hasattr(trace_obj, 'fiber_positions'):
        trace_info['fiber_pos_checksum'] = float(np.sum(trace_obj.fiber_positions))

    # Serialize to JSON and hash
    trace_json = json.dumps(trace_info, sort_keys=True)
    trace_hash = hashlib.md5(trace_json.encode()).hexdigest()

    return trace_hash


def compare_trace_consistency(extraction_dict1, extraction_dict2, label1="Dataset 1", label2="Dataset 2"):
    """Compare trace consistency between two extraction dictionaries.

    Args:
        extraction_dict1: First extraction dictionary
        extraction_dict2: Second extraction dictionary
        label1: Label for first dataset (e.g., "Flat field")
        label2: Label for second dataset (e.g., "Arc lamp")

    Returns:
        dict: {
            'consistent': bool,
            'mismatches': list of (extension_idx, details) tuples,
            'report': str
        }
    """
    mismatches = []
    report_lines = []

    ext1 = extraction_dict1['extractions']
    ext2 = extraction_dict2['extractions']
    meta1 = extraction_dict1['metadata']
    meta2 = extraction_dict2['metadata']

    # Check extension counts
    if len(ext1) != len(ext2):
        report_lines.append(f"ERROR: Extension count mismatch: {label1}={len(ext1)}, {label2}={len(ext2)}")
        return {
            'consistent': False,
            'mismatches': [(None, "Extension count mismatch")],
            'report': '\n'.join(report_lines)
        }

    report_lines.append(f"Comparing trace consistency between {label1} and {label2}")
    report_lines.append(f"Total extensions: {len(ext1)}")
    report_lines.append("")

    for ext_idx in range(len(ext1)):
        # Check metadata consistency
        channel1 = meta1[ext_idx]['channel']
        bench1 = meta1[ext_idx]['bench']
        side1 = meta1[ext_idx]['side']

        channel2 = meta2[ext_idx]['channel']
        bench2 = meta2[ext_idx]['bench']
        side2 = meta2[ext_idx]['side']

        if (channel1 != channel2) or (bench1 != bench2) or (side1 != side2):
            mismatch_detail = (
                f"Extension {ext_idx}: Metadata mismatch\n"
                f"  {label1}: {channel1} {bench1}{side1}\n"
                f"  {label2}: {channel2} {bench2}{side2}"
            )
            mismatches.append((ext_idx, mismatch_detail))
            report_lines.append(f"✗ {mismatch_detail}")
            continue

        # Check if traces are available
        trace1 = ext1[ext_idx].trace if hasattr(ext1[ext_idx], 'trace') else None
        trace2 = ext2[ext_idx].trace if hasattr(ext2[ext_idx], 'trace') else None

        if trace1 is None and trace2 is None:
            report_lines.append(
                f"⚠  Extension {ext_idx} ({channel1} {bench1}{side1}): "
                f"No trace available in either dataset (cannot verify)"
            )
            continue

        if trace1 is None or trace2 is None:
            mismatch_detail = (
                f"Extension {ext_idx} ({channel1} {bench1}{side1}): "
                f"Trace available in {label2 if trace1 is None else label1} but not {label1 if trace1 is None else label2}"
            )
            mismatches.append((ext_idx, mismatch_detail))
            report_lines.append(f"✗ {mismatch_detail}")
            continue

        # Compare trace hashes
        hash1 = compute_trace_hash(trace1)
        hash2 = compute_trace_hash(trace2)

        if hash1 != hash2:
            mismatch_detail = (
                f"Extension {ext_idx} ({channel1} {bench1}{side1}): "
                f"Trace file mismatch (hash {hash1[:8]} vs {hash2[:8]})"
            )
            mismatches.append((ext_idx, mismatch_detail))
            report_lines.append(f"✗ {mismatch_detail}")
        else:
            report_lines.append(
                f"✓ Extension {ext_idx} ({channel1} {bench1}{side1}): "
                f"Trace consistent (hash {hash1[:8]})"
            )

    report_lines.append("")
    report_lines.append(f"Summary:")
    report_lines.append(f"  Consistent: {len(ext1) - len(mismatches)}/{len(ext1)}")
    report_lines.append(f"  Mismatched: {len(mismatches)}/{len(ext1)}")

    is_consistent = len(mismatches) == 0

    return {
        'consistent': is_consistent,
        'mismatches': mismatches,
        'report': '\n'.join(report_lines)
    }
```

**Use in arcTransfer**:

```python
# At the start of arcTransfer (before line 384 loop)
def arcTransfer(scidict, arcdict):
    """Transfer wavelength calibration from arc to science spectra."""
    from llamas_pyjamas.constants import idx_lookup
    from llamas_pyjamas.Utils.utils import compare_trace_consistency

    # NEW: Check trace consistency before transferring calibration
    print("\nVerifying trace consistency between science and arc...")
    consistency_check = compare_trace_consistency(
        scidict, arcdict,
        label1="Science/Flat",
        label2="Arc"
    )

    print(consistency_check['report'])

    if not consistency_check['consistent']:
        print("\nWARNING: Trace files are not consistent between science/flat and arc!")
        print("This may cause incorrect wavelength calibration.")
        print("\nOptions:")
        print("  1. Re-extract arc using the same trace files as science/flat")
        print("  2. Continue anyway (NOT RECOMMENDED - may produce bad calibration)")

        # Could make this configurable
        allow_inconsistent_traces = False  # Set to True to allow (unsafe!)

        if not allow_inconsistent_traces:
            raise ValueError(
                "Trace file mismatch between science/flat and arc. "
                "Cannot safely transfer wavelength calibration. "
                "Please re-extract with consistent traces."
            )

    # Continue with existing arcTransfer logic...
    scispec = scidict['extractions']
    arcspec = arcdict['extractions']
    # ... rest of function
```

#### Solution 4.4: Preserve Trace Metadata Through Sanitization

**Location**: Modify `sanitize_extraction_dict_for_pickling()` in [Flat/flatLlamas.py:62-98](../../Flat/flatLlamas.py#L62-L98)

```python
def sanitize_extraction_dict_for_pickling(extraction_dict):
    """Sanitize extraction dictionary to remove problematic references for pickling.

    Modified to preserve trace metadata even when removing trace object reference.
    """
    import copy

    logger.info("Sanitizing extraction dictionary for pickling")

    # Make a deep copy to avoid modifying the original
    sanitized_dict = copy.deepcopy(extraction_dict)

    # Remove trace references from each extraction object
    extractions = sanitized_dict.get('extractions', [])
    sanitized_count = 0

    for i, extraction in enumerate(extractions):
        # BEFORE removing trace, extract key metadata
        if hasattr(extraction, 'trace') and extraction.trace is not None:
            logger.debug(f"Preserving trace metadata from extraction {i} before sanitization")

            # Store trace hash for consistency checking
            from llamas_pyjamas.Utils.utils import compute_trace_hash
            trace_hash = compute_trace_hash(extraction.trace)

            # Add to metadata if not already there
            if 'metadata' in sanitized_dict and i < len(sanitized_dict['metadata']):
                sanitized_dict['metadata'][i]['trace_hash'] = trace_hash
                sanitized_dict['metadata'][i]['trace_nfibers'] = extraction.trace.nfibers
                sanitized_dict['metadata'][i]['trace_naxis1'] = extraction.trace.naxis1
                sanitized_dict['metadata'][i]['trace_naxis2'] = extraction.trace.naxis2

                # Store fiberimg checksum for validation
                if hasattr(extraction.trace, 'fiberimg') and extraction.trace.fiberimg is not None:
                    sanitized_dict['metadata'][i]['fiberimg_checksum'] = int(np.sum(extraction.trace.fiberimg))

            # Now remove the trace reference
            logger.debug(f"Removing trace reference from extraction {i}")
            extraction.trace = None
            sanitized_count += 1

        # Also remove any other potentially problematic attributes
        for attr_name in ['LUT', 'dead_fibers']:
            if hasattr(extraction, attr_name):
                logger.debug(f"Removing {attr_name} from extraction {i}")
                setattr(extraction, attr_name, None)

    logger.info(f"Sanitized {sanitized_count} extraction objects with trace references")
    logger.info(f"Preserved trace metadata in metadata dict for consistency checking")

    return sanitized_dict
```

**Use trace hash in arcTransfer**:

```python
# In arcTransfer, after line 390 (key creation)
key = (channel, bench, side)
arc_idx = idx_lookup[key] - 1

# NEW: Compare trace hashes if available
if 'trace_hash' in scidict['metadata'][fits_ext] and 'trace_hash' in arcdict['metadata'][arc_idx]:
    sci_trace_hash = scidict['metadata'][fits_ext]['trace_hash']
    arc_trace_hash = arcdict['metadata'][arc_idx]['trace_hash']

    if sci_trace_hash != arc_trace_hash:
        print(f"WARNING: Extension {fits_ext} - Trace hash mismatch!")
        print(f"  Science/Flat trace hash: {sci_trace_hash[:8]}")
        print(f"  Arc trace hash: {arc_trace_hash[:8]}")
        print(f"  This indicates different trace files were used.")
        print(f"  Fiber-to-fiber wavelength matching may be incorrect!")

        # Could raise error here if strict checking is desired
        # raise ValueError("Trace file mismatch detected")
```

---

## Implementation Roadmap

**⚠️ CRITICAL**: This roadmap keeps `arcSolve()` completely unchanged. All changes are in separate validation module and `arcTransfer()` only.

### Phase 1: Wavelength Quality Validation (High Priority)

**Estimated effort**: 1-2 days

**Key principle**: Create new validation module WITHOUT modifying existing wavelength generation code.

1. **Create new validation module** (Solution 3.1) - **REQUIRED**
   - Create **NEW FILE**: `Arc/arcValidation.py`
   - Add `validate_wavelength_solution()` function
   - Write unit tests with synthetic data
   - **Does NOT modify `arcLlamas.py` at all**

2. **Integrate validation into `arcTransfer()`** (Solution 3.3) - **REQUIRED**
   - Modify ONLY the `arcTransfer()` function in `Arc/arcLlamas.py`
   - Add import: `from llamas_pyjamas.Arc.arcValidation import validate_wavelength_solution`
   - Add validation loop before wavelength transfer
   - Implement failure handling (skip vs. error)
   - Add summary reporting
   - **`arcSolve()` remains completely untouched**

3. **Add arc fit quality checks** (Solution 3.2) - **OPTIONAL (Skip initially)**
   - Only add if you want quality warnings in `arcSolve()`
   - Non-invasive: only adds warnings, doesn't change algorithm
   - Recommended: Skip this step initially

4. **Add quality metadata to arc dict** (Solution 3.4) - **OPTIONAL (Skip initially)**
   - Only needed for historical quality tracking
   - Can be added later if desired
   - Recommended: Skip this step initially

**Testing**:
- Test with good arc data (should pass validation)
- Test with intentionally bad arc data (zeros, NaNs, poor fits)
- Test with partial failures (some fibers bad)
- **Verify `arcSolve()` output is identical before/after changes** (critical test!)

### Phase 2: Fiber ID Tracking (High Priority)

**Estimated effort**: 3-4 days

1. Add physical fiber ID tracking to `ExtractLlamas` (Solution 4.1)
   - Modify `__init__()` to store fiber IDs
   - Update `save_extractions()` to include fiber IDs
   - Add fiber position tracking

2. Implement fiber ID matching in `arcTransfer()` (Solution 4.2)
   - Replace index-based matching with ID-based matching
   - Add unmatched fiber handling
   - Add detailed matching reports

3. Test with real data
   - Verify fiber IDs are consistent
   - Test with dead fibers excluded/included
   - Verify spatial positions match

**Testing**:
- Test with identical traces (should match 100%)
- Test with different traces (should detect mismatches)
- Test with dead fibers in different positions
- Test with fiber count mismatches

### Phase 3: Trace Consistency Checking (Medium Priority)

**Estimated effort**: 2-3 days

1. Implement trace hashing (Solution 4.3)
   - Create `compute_trace_hash()` function
   - Create `compare_trace_consistency()` function
   - Add to `Utils/utils.py`

2. Preserve trace metadata through sanitization (Solution 4.4)
   - Modify `sanitize_extraction_dict_for_pickling()`
   - Store trace hash in metadata
   - Update arcTransfer to use trace hash

3. Add trace consistency check to pipeline
   - Check before wavelength transfer
   - Option to fail or warn on mismatch

**Testing**:
- Test with same trace files (should pass)
- Test with different trace files (should detect)
- Test with missing traces (should warn)

### Phase 4: Integration and Documentation (Essential)

**Estimated effort**: 2-3 days

1. Integration testing
   - Run full flat field pipeline with validation enabled
   - Test with multiple datasets
   - Verify validation catches real errors

2. Configuration options
   - Add flags to enable/disable strict checking
   - Allow configurable thresholds
   - Add verbose logging option

3. Documentation
   - Update function docstrings
   - Create user guide for validation
   - Document error messages and fixes

4. Backwards compatibility
   - Handle old arc files without quality metadata
   - Handle old extraction files without fiber IDs
   - Provide migration path for existing data

---

## Testing Strategy

### Unit Tests

Create `tests/test_wavelength_validation.py`:

```python
import pytest
import numpy as np
from llamas_pyjamas.Arc.arcLlamas import validate_wavelength_solution, arcTransfer
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

class MockExtraction:
    """Mock extraction object for testing."""
    def __init__(self, nfibers, nwave):
        self.channel = 'red'
        self.bench = '1'
        self.side = 'A'
        self.wave = np.zeros((nfibers, nwave))
        self.xshift = np.zeros((nfibers, nwave))
        self.relative_throughput = np.zeros(nfibers)

def test_validate_wavelength_all_zeros():
    """Test validation fails for all-zero wavelength array."""
    extraction = MockExtraction(10, 2048)
    result = validate_wavelength_solution(extraction, 'red', 0)
    assert not result['valid']
    assert 'all zeros' in result['errors'][0].lower()

def test_validate_wavelength_with_nans():
    """Test validation fails for wavelength array with NaNs."""
    extraction = MockExtraction(10, 2048)
    extraction.wave[0, :] = np.linspace(6500, 10000, 2048)
    extraction.wave[0, 500:600] = np.nan  # Insert NaNs

    result = validate_wavelength_solution(extraction, 'red', 0)
    assert not result['valid']
    assert 'nan' in result['errors'][0].lower()

def test_validate_wavelength_non_monotonic():
    """Test validation fails for non-monotonic wavelength."""
    extraction = MockExtraction(10, 2048)
    extraction.wave[0, :] = np.linspace(6500, 10000, 2048)
    extraction.wave[0, 1000] = 6000  # Create reversal

    result = validate_wavelength_solution(extraction, 'red', 0)
    assert not result['valid']
    assert 'monotonic' in result['errors'][0].lower()

def test_validate_wavelength_wrong_range():
    """Test validation fails for wavelength outside expected range."""
    extraction = MockExtraction(10, 2048)
    # Blue wavelengths in red channel
    extraction.wave[0, :] = np.linspace(3500, 5000, 2048)

    result = validate_wavelength_solution(extraction, 'red', 0)
    assert not result['valid']
    assert 'range' in result['errors'][0].lower()

def test_validate_wavelength_valid():
    """Test validation passes for valid wavelength array."""
    extraction = MockExtraction(10, 2048)
    extraction.wave[0, :] = np.linspace(6500, 10000, 2048)

    result = validate_wavelength_solution(extraction, 'red', 0)
    assert result['valid']
    assert len(result['errors']) == 0
    assert 'wave_min' in result['metrics']
    assert 'median_dispersion' in result['metrics']

def test_validate_wavelength_unusual_dispersion():
    """Test validation warns for unusual dispersion."""
    extraction = MockExtraction(10, 2048)
    # Very large dispersion (5 A/pixel instead of ~2 A/pixel)
    extraction.wave[0, :] = np.linspace(6500, 16000, 2048)

    result = validate_wavelength_solution(extraction, 'red', 0)
    # Should pass but with warning
    assert len(result['warnings']) > 0
    assert 'dispersion' in result['warnings'][0].lower()
```

Create `tests/test_fiber_matching.py`:

```python
import pytest
import numpy as np
from llamas_pyjamas.Arc.arcLlamas import arcTransfer

def create_test_extraction_dict(nfibers=300, has_fiberid=True):
    """Create test extraction dictionary."""
    # ... implementation
    pass

def test_fiber_matching_with_ids():
    """Test fiber matching using physical fiber IDs."""
    # Create flat and arc dicts with same fiber IDs
    flat_dict = create_test_extraction_dict(nfibers=300, has_fiberid=True)
    arc_dict = create_test_extraction_dict(nfibers=300, has_fiberid=True)

    # Transfer should match all fibers
    result = arcTransfer(flat_dict, arc_dict)
    # Assert all fibers matched correctly
    # ...

def test_fiber_matching_dead_fiber_excluded():
    """Test fiber matching when flat excludes dead fiber."""
    # Flat: 297 fibers (dead fiber 100 excluded)
    flat_dict = create_test_extraction_dict(nfibers=297, has_fiberid=True)
    # Remove fiber ID 100

    # Arc: 300 fibers (includes dead fiber)
    arc_dict = create_test_extraction_dict(nfibers=300, has_fiberid=True)

    result = arcTransfer(flat_dict, arc_dict)
    # Assert fiber 100 in arc is not matched
    # Assert fibers after 100 match correctly despite index offset
    # ...

def test_fiber_matching_without_ids():
    """Test fallback index matching when no fiber IDs."""
    flat_dict = create_test_extraction_dict(nfibers=300, has_fiberid=False)
    arc_dict = create_test_extraction_dict(nfibers=300, has_fiberid=False)

    # Should fall back to index matching with warning
    result = arcTransfer(flat_dict, arc_dict)
    # Assert warning was issued
    # ...
```

### Integration Tests

Create `tests/test_flat_field_integration.py`:

```python
def test_full_flat_field_pipeline_with_validation():
    """Test complete flat field pipeline with validation enabled."""
    # Use test data with known good/bad characteristics
    # Run through full pipeline
    # Verify validation catches expected issues
    pass

def test_arc_transfer_with_bad_wavelength():
    """Test that bad wavelength calibration is caught."""
    # Create arc with intentionally bad wavelength solution
    # Attempt to transfer to flat
    # Verify error is raised or fibers marked invalid
    pass

def test_trace_consistency_check():
    """Test trace consistency checking."""
    # Extract flat with trace_v1
    # Extract arc with trace_v2 (different!)
    # Verify mismatch is detected
    pass
```

### Regression Tests

- Test with existing good datasets to ensure no functionality break
- Compare outputs before/after changes (wavelength calibration should be identical for good data)
- Verify performance impact is minimal

---

## Configuration Options

Add to pipeline configuration file:

```python
# Wavelength validation settings
wavelength_validation = {
    'enabled': True,  # Enable wavelength quality checks
    'rms_threshold': 1.0,  # Maximum RMS in Angstroms for arc fit
    'min_lines': 20,  # Minimum number of arc lines required
    'fail_on_invalid': True,  # Raise error if validation fails (vs. skip fiber)
    'verbose': True,  # Print detailed validation messages
}

# Fiber matching settings
fiber_matching = {
    'method': 'fiberid',  # 'fiberid', 'spatial', or 'index'
    'require_fiberid': True,  # Fail if fiber IDs not available
    'spatial_tolerance': 1.0,  # Maximum pixel distance for spatial matching
    'warn_on_index_matching': True,  # Warn when falling back to index matching
}

# Trace consistency settings
trace_consistency = {
    'check_enabled': True,  # Check trace consistency before wavelength transfer
    'fail_on_mismatch': True,  # Raise error if traces don't match
    'require_trace_hash': False,  # Require trace hash in metadata (for old files)
}
```

---

## Error Messages and Fixes

### Error: "Wavelength array is all zeros"

**Cause**: Arc calibration failed completely for this fiber/channel
**Fix**:
1. Check arc lamp exposure (too short/long?)
2. Check ThAr line list matches lamp used
3. Verify wavelength range in arc fitting code
4. Inspect arc spectrum visually

### Error: "Arc fit RMS exceeds threshold"

**Cause**: Poor wavelength fit quality
**Fix**:
1. Increase number of arc lines identified
2. Check for cosmic rays in arc spectrum
3. Adjust fitting parameters (order, rejection thresholds)
4. Use different arc lamp or longer exposure

### Warning: "Fiber ID {id} not found in arc"

**Cause**: Flat has fiber that arc doesn't (or vice versa)
**Fix**:
1. Verify same trace files used for both extractions
2. Check dead fiber handling consistency
3. Re-extract arc with same traces as flat

### Error: "Trace file mismatch detected"

**Cause**: Different trace files used for flat vs. arc
**Fix**:
1. Re-extract arc using same trace directory as flat
2. Verify trace files haven't changed between observations
3. Check trace version consistency

---

## Notes for Implementation

### Critical: Preserving Original Wavelength Generation

**THE MOST IMPORTANT RULE**: `arcSolve()` must remain completely unchanged.

**Files that are NOT modified (completely preserved)**:
- ✅ `Arc/arcLlamas.py` - **ENTIRE FILE UNCHANGED** (all functions preserved)
  - `arcSolve()` - Lines 217-363
  - `shiftArcX()` - Lines 88-137
  - `reidentifyArc()` - Lines 22-86
  - `fiberRelativeThroughput()` - Lines 169-214
  - `arcTransfer()` - Lines 365-426

**Files that WERE created**:
- ✅ `Arc/arcValidation.py` - NEW FILE (all validation logic) - **COMPLETED**

**Files that WERE modified**:
- ✅ `Arc/arcLlamasMulti.py::arcTransfer()` - Line 1064 (added validation calls with enable_validation flag) - **COMPLETED**

**Files for future phases (not yet implemented)**:
- `Extract/extractLlamas.py::__init__()` - Add fiber ID tracking (Phase 2)
- `Flat/flatLlamas.py::sanitize_extraction_dict_for_pickling()` - Preserve trace metadata (Phase 3)
- `Utils/utils.py` - Add trace hashing functions (Phase 3)

### Implementation Safeguards

1. **Create separate validation module first**: Build `Arc/arcValidation.py` completely before modifying any existing files

2. **Test validation module independently**: Ensure validation functions work correctly with synthetic data before integration

3. **Add validation to arcTransfer with feature flag**: Make validation optional and disabled by default for initial testing

4. **Verify wavelength generation unchanged**: Compare arc calibration files before/after implementation - they should be identical

5. **Use version control carefully**: Commit validation module separately from integration changes

### Other Implementation Notes

1. **Backwards Compatibility**: All validation should be optional initially, with warnings not errors, to allow testing against existing data

2. **Performance**: Validation adds minimal overhead (<1% typically), but could be made optional for production runs once validated

3. **Logging**: Use proper Python logging instead of print statements for easier debugging

4. **User Experience**: Clear, actionable error messages with suggested fixes

5. **Documentation**: Update all docstrings and create user guide explaining validation process

6. **Git Branch**: Create feature branch `wavelength-validation-improvements` for implementation

---

## Quick Start Implementation Guide

**For minimal risk implementation, follow this exact order**:

### Step 1: Create validation module (no risk)
```bash
# Create new file - doesn't touch existing code
touch Arc/arcValidation.py
# Implement validate_wavelength_solution() function
# Write unit tests
```

### Step 2: Test validation module independently (no risk)
```bash
# Test with synthetic data before integrating
python -m pytest tests/test_arc_validation.py
```

### Step 3: Add optional validation to arcTransfer (minimal risk)
```python
# In Arc/arcLlamas.py, add ONE line at top:
from llamas_pyjamas.Arc.arcValidation import validate_wavelength_solution

# In arcTransfer(), add validation with SKIP_VALIDATION flag:
SKIP_VALIDATION = True  # Set to False to enable validation

if not SKIP_VALIDATION:
    validation = validate_wavelength_solution(...)
    # Handle validation results
```

### Step 4: Test with existing data (verify no breakage)
```bash
# Run with validation DISABLED
# Verify output is identical to before

# Then enable validation
# Test with known good/bad data
```

### Step 5: Optional enhancements (only after above works)
- Add fiber ID tracking (Phase 2)
- Add trace consistency checking (Phase 3)
- Add optional quality warnings to arcSolve (only if desired)

---

## References

- Current implementation: [Arc/arcLlamas.py](../../Arc/arcLlamas.py)
- Extraction code: [Extract/extractLlamas.py](../../Extract/extractLlamas.py)
- Flat field processing: [Flat/flatLlamas.py](../../Flat/flatLlamas.py)
- Pipeline orchestration: [reduce.py](../../reduce.py)

---

## Summary: What Changes and What Doesn't

### ❌ DOES NOT CHANGE (Original file completely preserved)

| Component | Function | Lines | Status |
|-----------|----------|-------|--------|
| **`Arc/arcLlamas.py`** | **ENTIRE FILE** | **ALL** | ✅ **100% UNCHANGED** |
| `Arc/arcLlamas.py` | `arcSolve()` | 217-363 | ✅ **UNCHANGED** |
| `Arc/arcLlamas.py` | `shiftArcX()` | 88-137 | ✅ **UNCHANGED** |
| `Arc/arcLlamas.py` | `reidentifyArc()` | 22-86 | ✅ **UNCHANGED** |
| `Arc/arcLlamas.py` | `fiberRelativeThroughput()` | 169-214 | ✅ **UNCHANGED** |
| `Arc/arcLlamas.py` | `arcTransfer()` | 365-426 | ✅ **UNCHANGED** |
| ThAr line matching | Algorithm | N/A | ✅ **UNCHANGED** |
| Legendre polynomial fitting | Algorithm | N/A | ✅ **UNCHANGED** |
| Cross-correlation shifts | Algorithm | N/A | ✅ **UNCHANGED** |

### ✅ COMPLETED CHANGES (Phase 1 - Wavelength validation)

| Component | Function | Purpose | Status |
|-----------|----------|---------|--------|
| `Arc/arcValidation.py` | **NEW FILE** | Wavelength quality validation | ✅ **COMPLETED** |
| `Arc/arcLlamasMulti.py` | `arcTransfer()` | Add validation calls with enable_validation flag | ✅ **COMPLETED** |

### 🔄 FUTURE PHASES (Not yet implemented)

| Component | Function | Purpose | Phase |
|-----------|----------|---------|-------|
| `Extract/extractLlamas.py` | `__init__()` | Add fiber ID tracking | Phase 2 |
| `Arc/arcLlamasMulti.py` | `arcTransfer()` | Use fiber IDs for matching | Phase 2 |
| `Flat/flatLlamas.py` | `sanitize_extraction_dict_for_pickling()` | Preserve trace metadata | Phase 3 |
| `Utils/utils.py` | **NEW FUNCTIONS** | Trace hashing | Phase 3 |

### 🎯 Phase 1 Implementation Complete

**✅ Completed changes for Issue #3 (Wavelength validation)**:
1. ✅ Created `Arc/arcValidation.py` (new file - 127 lines)
2. ✅ Modified `Arc/arcLlamasMulti.py::arcTransfer()` (added 159 lines with validation logic)

**Total lines of code changed in existing files**: ~160 lines (all in `arcLlamasMulti.py`, NOT in `arcLlamas.py`)

### 🔒 Safety Guarantees

1. **Wavelength generation algorithm (`arcSolve()`)**: **100% unchanged**
2. **Arc shift calculation (`shiftArcX()`)**: **100% unchanged**
3. **Backwards compatibility**: All validation can be disabled via flags
4. **Testing**: Compare arc calibration files before/after - should be byte-identical when validation is disabled
5. **Rollback**: Simply delete `Arc/arcValidation.py` and revert changes to `arcTransfer()` to return to original behavior

---

**END OF DOCUMENT**
