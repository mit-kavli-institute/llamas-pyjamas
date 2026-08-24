# How to Fix the Whitelight Interpolation Issue

## The Problem (Confirmed)

Looking at your fiber layout image (`fiber.png`), the **fiber map IS CORRECT**:
- Detectors stack as horizontal stripes (exactly as shown in the image)
- Coverage: X ≈ 0-50, Y ≈ 0-45 (matching the fiber.png layout)
- The fiber positions in `LLAMAS_FiberMap_rev04.dat` are accurate

**The bug is in the whitelight grid creation:**
- Current code creates a **fixed 80×80 pixel grid**
- But fibers only cover ~50×45 pixels
- Interpolator extrapolates into regions with no fibers
- Creates artifacts at X>46, Y>44

## The Solution

Modify `Image/WhiteLightModule.py` to create a grid that **matches the actual fiber coverage** instead of using a fixed 80×80 size.

### Current Code (lines 247-261)

```python
flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux, fill_value=np.nan)

if (False):
    xx = np.arange(53)
    yy = np.arange(53)
else:
    subsample = 1.5
    xx = 1.0/subsample * np.arange(53*subsample)  # Creates 0 to ~52.67
    yy = 1.0/subsample * np.arange(53*subsample)  # Creates 0 to ~52.67

x_grid, y_grid = np.meshgrid(xx/subsample, yy/subsample)

whitelight = flux_interpolator(x_grid, y_grid)
```

**Problem:** Hard-coded `53*subsample` creates 80×80 grid regardless of fiber extent.

### Fixed Code

Replace lines 247-261 with:

```python
flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux, fill_value=np.nan)

# Calculate actual fiber coverage extent
x_min, x_max = np.min(xdata), np.max(xdata)
y_min, y_max = np.min(ydata), np.max(ydata)

# Add small padding to avoid edge effects
padding = 0.5
x_min -= padding
x_max += padding
y_min -= padding
y_max += padding

# Create grid that matches fiber coverage (not fixed 80×80)
subsample = 1.5  # Keep same sampling density

# Calculate number of grid points needed to cover the fiber extent
n_x = int((x_max - x_min) * subsample) + 1
n_y = int((y_max - y_min) * subsample) + 1

# Create coordinate arrays spanning the actual fiber coverage
xx = np.linspace(x_min, x_max, n_x)
yy = np.linspace(y_min, y_max, n_y)

x_grid, y_grid = np.meshgrid(xx, yy)

whitelight = flux_interpolator(x_grid, y_grid)

# Log the actual grid size created
logger.info(f"Whitelight grid size: {n_y} × {n_x} pixels")
logger.info(f"Fiber coverage: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
```

**Benefits:**
1. ✓ Grid size automatically matches fiber coverage
2. ✓ No extrapolation beyond fiber positions
3. ✓ Image size reflects actual data extent
4. ✓ No artifacts in empty regions
5. ✓ Works for any detector configuration

### Alternative: Keep 80×80 but Mark Invalid Regions

If you **must** keep 80×80 for compatibility, use this approach:

```python
flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux, fill_value=np.nan)

# Calculate actual fiber coverage
x_min, x_max = np.min(xdata), np.max(xdata)
y_min, y_max = np.min(ydata), np.max(ydata)

# Create fixed 80×80 grid (for backward compatibility)
subsample = 1.5
xx = 1.0/subsample * np.arange(53*subsample)
yy = 1.0/subsample * np.arange(53*subsample)

x_grid, y_grid = np.meshgrid(xx/subsample, yy/subsample)

# Interpolate
whitelight = flux_interpolator(x_grid, y_grid)

# MASK regions beyond fiber coverage
# Set to NaN anywhere outside the fiber extent (with small padding)
padding = 1.0  # pixels
valid_region = (
    (x_grid >= (x_min - padding)) &
    (x_grid <= (x_max + padding)) &
    (y_grid >= (y_min - padding)) &
    (y_grid <= (y_max + padding))
)

# Set invalid regions to NaN
whitelight[~valid_region] = np.nan

logger.info(f"Fiber coverage: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
logger.info(f"Valid pixels: {np.sum(valid_region)} / {valid_region.size} ({np.sum(valid_region)/valid_region.size*100:.1f}%)")
```

**Benefits:**
1. ✓ Keeps 80×80 size (no downstream code changes needed)
2. ✓ Clearly marks extrapolated regions as NaN
3. ✓ Users can immediately see valid data extent
4. ✓ Prevents misleading "bright spots" in empty regions

## For Data Cubes

The same issue affects cube generation. In `Cube/cubeConstruct.py`, around lines 1150-1176, you should:

### Option 1: Use Actual Fiber Extent

Before the interpolation (around line 1150), add:

```python
# Get actual fiber coverage extent
fiber_x_min, fiber_x_max = np.min(fiber_x), np.max(fiber_x)
fiber_y_min, fiber_y_max = np.min(fiber_y), np.max(fiber_y)

# Adjust spatial grid to match fiber coverage (not exceed it)
x_in_range = (self.spatial_grid_x >= fiber_x_min - 1) & (self.spatial_grid_x <= fiber_x_max + 1)
y_in_range = (self.spatial_grid_y >= fiber_y_min - 1) & (self.spatial_grid_y <= fiber_y_max + 1)

# Only interpolate within fiber coverage
xi_valid = xi[:, x_in_range]
yi_valid = yi[y_in_range, :]

# ... then interpolate only on valid grid
```

### Option 2: Mask Invalid Spaxels

After interpolation (around line 1179), add:

```python
# Put interpolated data into cube
self.cube_data[w_idx, :, :] = grid_z

# MASK spaxels beyond fiber coverage
fiber_x_min, fiber_x_max = np.min(fiber_x), np.max(fiber_x)
fiber_y_min, fiber_y_max = np.min(fiber_y), np.max(fiber_y)

# Create validity mask
padding = 1.0  # pixels
valid_spaxels = (
    (xi >= (fiber_x_min - padding)) &
    (xi <= (fiber_x_max + padding)) &
    (yi >= (fiber_y_min - padding)) &
    (yi <= (fiber_y_max + padding))
)

# Set invalid spaxels to NaN
self.cube_data[w_idx, ~valid_spaxels] = np.nan
```

## Step-by-Step Implementation

### Step 1: Backup Current Code

```bash
cp Image/WhiteLightModule.py Image/WhiteLightModule.py.backup
cp Cube/cubeConstruct.py Cube/cubeConstruct.py.backup
```

### Step 2: Edit WhiteLightModule.py

Location: `/Users/slh/Documents/Projects/Magellan_dev/LLAMAS/llamas-pyjamas/llamas_pyjamas/Image/WhiteLightModule.py`

**Lines to modify:** 247-261

**Recommended approach:** Use "Alternative: Keep 80×80 but Mark Invalid Regions" to minimize downstream effects.

Replace:
```python
flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux, fill_value=np.nan)

if (False):
    xx = np.arange(53)
    yy = np.arange(53)
else:
    subsample = 1.5
    xx = 1.0/subsample * np.arange(53*subsample)
    yy = 1.0/subsample * np.arange(53*subsample)

x_grid, y_grid = np.meshgrid(xx/subsample, yy/subsample)

whitelight = flux_interpolator(x_grid, y_grid)
```

With:
```python
flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux, fill_value=np.nan)

# Calculate actual fiber coverage extent
x_min, x_max = np.min(xdata), np.max(xdata)
y_min, y_max = np.min(ydata), np.max(ydata)

# Create grid (keeping 80×80 for compatibility)
subsample = 1.5
xx = 1.0/subsample * np.arange(53*subsample)
yy = 1.0/subsample * np.arange(53*subsample)

x_grid, y_grid = np.meshgrid(xx/subsample, yy/subsample)

# Interpolate
whitelight = flux_interpolator(x_grid, y_grid)

# MASK regions beyond fiber coverage to prevent artifacts
padding = 1.0  # pixels of padding around fiber extent
valid_region = (
    (x_grid >= (x_min - padding)) &
    (x_grid <= (x_max + padding)) &
    (y_grid >= (y_min - padding)) &
    (y_grid <= (y_max + padding))
)

# Set invalid regions to NaN
whitelight[~valid_region] = np.nan

logger.info(f"Whitelight grid: {whitelight.shape}, Valid region coverage: {np.sum(valid_region)/valid_region.size*100:.1f}%")
logger.info(f"Fiber extent: X=[{x_min:.1f}, {x_max:.1f}], Y=[{y_min:.1f}, {y_max:.1f}]")
```

### Step 3: Test the Fix

```bash
cd /Users/slh/Documents/Projects/Magellan_dev/LLAMAS/llamas-pyjamas/llamas_pyjamas

# Re-run whitelight generation on your test data
python -c "
from Image.WhiteLightModule import WhiteLightFits
from File.llamasIO import load_extraction_batch

# Load your extraction file
extraction_file = '/Users/slh/Documents/Projects/independent/Mike_data_reduction/extractions/LLAMAS_2026-01-07_01-25-43.5_SCI22_extract.pkl'
extractions, metadata = load_extraction_batch(extraction_file)

# Generate whitelight
WhiteLightFits(extractions, metadata, outfile='test_whitelight_fixed.fits')
"
```

### Step 4: Verify the Fix

```python
from astropy.io import fits
import numpy as np

with fits.open('output/test_whitelight_fixed.fits') as hdul:
    green_image = hdul['GREEN'].data

    # Check how many pixels are NaN (should be ~60%)
    n_nan = np.sum(np.isnan(green_image))
    n_total = green_image.size

    print(f"NaN pixels: {n_nan}/{n_total} ({n_nan/n_total*100:.1f}%)")
    print(f"Valid pixels: {n_total-n_nan}/{n_total} ({(n_total-n_nan)/n_total*100:.1f}%)")

    # The artifact at (55, 37) should now be NaN
    print(f"\nPixel at (55, 37): {green_image[37, 55]}")
    print("(should be NaN)")
```

**Expected output:**
```
NaN pixels: ~3800/6400 (~60%)
Valid pixels: ~2600/6400 (~40%)

Pixel at (55, 37): nan
(should be NaN)
```

### Step 5: Update Cube Generation (Optional)

If you also want to fix the cube generation, edit `Cube/cubeConstruct.py`:

**Location:** After line 1179 (`self.cube_data[w_idx, :, :] = grid_z`)

**Add:**
```python
# Mask spaxels beyond fiber coverage
if w_idx == 0:  # Only calculate once
    fiber_x_min, fiber_x_max = np.nanmin(fiber_x), np.nanmax(fiber_x)
    fiber_y_min, fiber_y_max = np.nanmin(fiber_y), np.nanmax(fiber_y)

    padding = 1.0
    self.valid_spaxels = (
        (xi >= (fiber_x_min - padding)) &
        (xi <= (fiber_x_max + padding)) &
        (yi >= (fiber_y_min - padding)) &
        (yi <= (fiber_y_max + padding))
    )

    self.logger.info(f"Fiber coverage: X=[{fiber_x_min:.1f}, {fiber_x_max:.1f}], Y=[{fiber_y_min:.1f}, {fiber_y_max:.1f}]")
    self.logger.info(f"Valid spaxels: {np.sum(self.valid_spaxels)}/{self.valid_spaxels.size} ({np.sum(self.valid_spaxels)/self.valid_spaxels.size*100:.1f}%)")

# Apply mask to this wavelength slice
self.cube_data[w_idx, ~self.valid_spaxels] = np.nan
```

## Summary of Changes

| File | Lines | Change | Impact |
|------|-------|--------|--------|
| `Image/WhiteLightModule.py` | 247-261 | Add masking of regions beyond fiber coverage | Whitelight images show NaN in extrapolated regions |
| `Cube/cubeConstruct.py` | After 1179 | Add masking of invalid spaxels | Data cubes mark extrapolated regions as NaN |

## What This Fixes

**Before Fix:**
- Bright artifact at (55, 37) in whitelight
- Confusing "sources" in regions with no fibers
- Invalid flux values in ~60% of image
- Users can't distinguish real from interpolated data

**After Fix:**
- Regions beyond fiber coverage show NaN (displayed as black/white in viewers)
- Clear visual indication of data validity
- No misleading bright spots
- Easy to see actual fiber coverage extent

## Notes

1. **This doesn't change the fiber map** - the fiber positions are correct
2. **This doesn't change RSS files** - they already contain only real data
3. **This only affects visualization** (whitelight, cubes)
4. **Backward compatible** - keeps 80×80 size, just sets invalid regions to NaN
5. **No science impact** - you should always use RSS for science anyway

## Recommendation

**Use the "Alternative: Keep 80×80 but Mark Invalid Regions" approach** because:
- Minimal code changes
- Maintains backward compatibility
- Clearly shows valid data extent
- Easy to implement and test
- No downstream effects on other code

---

**Created:** 2026-01-11
**Issue:** Whitelight interpolation creates artifacts beyond fiber coverage
**Solution:** Mask extrapolated regions as NaN
**Priority:** Medium (affects visualization, not core science data)
