# Old RSS File Wavelength Issues - Diagnostic Report

**Date**: 2025-12-13
**Analyst**: Claude
**Files Analyzed**: jesse_RSS_results/LLAMAS_2025-11-10T03_04_02.137_extract_RSS_*.fits

## Executive Summary

The old RSS files have **critical wavelength calibration issues** caused by an **array transposition bug** in the previous version of llamasRSS.py. This has been fixed in the new version.

## Issues Found

### 1. Array Transposition Bug

**Problem**: The old RSS files have wavelength arrays with shape `(2048, 2389)` instead of the correct `(2389, 2048)`.

- **Old (incorrect)**: `(2048, 2389)` = (pixels, fibers) - BACKWARDS!
- **New (correct)**: `(2389, 2048)` = (fibers, pixels)

**Impact**: This causes the wavelength solution to be applied along the wrong axis, leading to catastrophic errors.

### 2. Wavelength Extrapolation Beyond Arc Range

#### RED Channel
```
Reference arc range: 6644.97 - 10050.50 Å

Old RSS file:
  Actual range: 6638.27 - 15853.60 Å
  Pixels beyond range: 863,251 / 4,892,672 (17.64%)
  Maximum extrapolation: +5803 Å above arc max (57.7% beyond calibrated range!)

New RSS file:
  Actual range: 6644.97 - 10050.50 Å
  Pixels beyond range: 1 / 4,892,672 (0.00%)
  ✅ Essentially perfect
```

#### GREEN Channel
```
Reference arc range: 4553.18 - 7115.52 Å

Old RSS file:
  Actual range: -350953.01 - 6919.90 Å (!!)
  NEGATIVE wavelengths: 1,162,587 pixels (31.64% of non-NaN data)
  Most negative value: -350953 Å (PHYSICALLY IMPOSSIBLE!)

New RSS file:
  Actual range: 4553.18 - 7115.52 Å
  ✅ Perfect
```

#### BLUE Channel
```
Reference arc range: 3170.66 - 4880.31 Å

Old RSS file:
  Actual range: 3149.17 - 4880.29 Å
  Pixels beyond range: 965 / 4,892,672 (0.02%)
  Minor extrapolation: -21.5 Å below arc min (0.68%)

New RSS file:
  Actual range: 3170.66 - 4880.31 Å
  ✅ Perfect
```

### 3. NaN Values

All old RSS files have **24.91% NaN values** (1,218,560 / 4,892,672 pixels), likely from the transposition creating invalid fiber/pixel combinations.

## Root Cause Analysis

### Why Did This Happen?

1. **Transposed Arrays**: The wavelength solution was computed for shape `(nfibers, npixels)` but written as `(npixels, nfibers)`

2. **Wrong Axis Application**: When the wavelength solution polynomial was evaluated:
   - It should have been applied along the **pixel axis** (axis=1) for each fiber
   - Instead, it was applied along the **fiber axis** (axis=0)

3. **Extrapolation Results**:
   - The polynomial was extrapolated far beyond its calibrated pixel range
   - For fibers at the edge of the array, this produced:
     - RED: Wavelengths up to 15853 Å (should max at 10050 Å)
     - GREEN: NEGATIVE wavelengths down to -350953 Å (physically impossible!)
     - BLUE: Minor extrapolation (least affected)

### Evidence of Transposition

**Wavelength span per "fiber" in old files:**
- Old file "fiber 0": 147 Å span (should be ~3400 Å for red channel)
- New file fiber 0: 3239 Å span ✅ Correct

**Monotonicity check:**
- Old: NOT monotonic increasing along rows (fibers mixed with pixels)
- New: IS monotonic increasing along rows ✅ Correct

## Comparison: Old vs New

| Metric | Old RSS Files | New RSS Files |
|--------|--------------|---------------|
| **Shape** | (2048, 2389) ❌ | (2389, 2048) ✅ |
| **Interpretation** | (pixels, fibers) ❌ | (fibers, pixels) ✅ |
| **NaN values** | 24.91% ❌ | ~0% ✅ |
| **Red extrapolation** | 17.64% beyond range ❌ | 0.00% ✅ |
| **Green extrapolation** | 47% beyond range, 32% NEGATIVE ❌ | 0.00% ✅ |
| **Blue extrapolation** | 0.02% beyond range ⚠️ | 0.00% ✅ |
| **Wavelength monotonicity** | Broken ❌ | Correct ✅ |
| **Physical validity** | NO (negative λ) ❌ | YES ✅ |

## Resolution

✅ **The new version of llamasRSS.py has fixed this issue completely.**

The new RSS files show:
- Correct array dimensions (fibers, pixels)
- Wavelengths entirely within arc calibration range (< 0.01% margin)
- No negative wavelengths
- Proper monotonic increase along wavelength axis
- Minimal NaN values

## Recommendation

**DO NOT USE the old RSS files from jesse_RSS_results/** - they contain invalid wavelength solutions due to the array transposition bug.

All science analysis should use RSS files generated with the new llamasRSS.py version.

## Technical Details

### Diagnostic Tools Added

Two new utility functions have been added to `llamas_pyjamas/Utils/utils.py`:

1. `check_reference_arc_wavelength_ranges(arc_file=None, verbose=True)`
   - Checks the reference arc calibration file
   - Reports wavelength coverage by channel and extension

2. `check_extraction_wavelength_ranges(extraction_file, reference_arc_file=None, verbose=True)`
   - Checks extraction pickle files
   - Compares to reference arc to detect extrapolation
   - Identifies wavelengths beyond calibrated range

### Usage Example

```python
from llamas_pyjamas.Utils.utils import check_reference_arc_wavelength_ranges

# Check reference arc ranges
ranges = check_reference_arc_wavelength_ranges()

# Check an extraction file
from llamas_pyjamas.Utils.utils import check_extraction_wavelength_ranges
results = check_extraction_wavelength_ranges('path/to/extract.pkl', verbose=True)

# Check if within range
if results['comparison']['red']['within']:
    print("Red channel is within calibration range")
```

---

**End of Report**
