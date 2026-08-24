# GUI_extract Background and Bias Subtraction Documentation

This document describes all background and bias subtraction operations that occur during the `GUI_extract` extraction pipeline.

## Overview

The `GUI_extract` function in [guiExtract.py](../GUI/guiExtract.py) performs spectral extraction from LLAMAS MEF FITS files. The pipeline includes multiple stages of background/bias handling:

1. **Bias frame subtraction** (per-detector, 2D image level)
2. **Post-bias background measurement** (per-detector)
3. **Per-color detector normalization** (offset correction on extracted spectra)
4. **Placeholder camera handling** (setting uniform background level)

---

## Stage 1: Bias Frame Subtraction

**Location:** `process_trace()` function (lines 165-227 in guiExtract.py)

**What happens:**
```python
# In process_trace():
bias = _grab_bias_hdu(bench=bench, side=side, color=color, dir=bias_file)
bias_data = bias.data
bias_subtracted_data = hdu_data.astype(float) - bias_data
```

**Details:**
- **Input:** Raw science frame data (`hdu_data`) for each detector extension
- **Bias source:** Combined bias file from `BIAS_DIR/combined_bias.fits` (or user-specified)
- **Method:** Full 2D pixel-by-pixel subtraction
- **The `_grab_bias_hdu()` function:** (in [traceLlamasMaster.py](../Trace/traceLlamasMaster.py))
  - Looks up the correct bias extension using a color/bench/side index
  - Returns the matching bias HDU for that specific camera

**Output:** `bias_subtracted_data` - the science frame with the master bias subtracted

---

## Stage 2: Post-Bias Background Measurement

**Location:** `process_trace()` function (line 206) and `compute_detector_background()` (lines 291-304)

**What happens:**
```python
# In process_trace():
detector_background = compute_detector_background(bias_subtracted_data, rows=(30, 50))

# compute_detector_background():
def compute_detector_background(data, rows=(30, 50)):
    upper_det = data[rows[0]:rows[1], :]
    upper_background_value = np.median(upper_det)
    return upper_background_value
```

**Details:**
- **Input:** Bias-subtracted 2D detector image
- **Method:** Takes median of rows 30-50 across all columns
- **Purpose:** Measures residual background AFTER bias subtraction
- **Why rows 30-50?** This region is typically free of bright fiber traces

**Output:** Single scalar value representing the detector's residual background level

---

## Stage 3: Spectral Extraction (No Additional Background Subtraction)

**Location:** `ExtractLlamas` class in [extractLlamas.py](../Extract/extractLlamas.py)

**What happens:**
```python
# In process_trace():
extraction = ExtractLlamas(tracer, bias_subtracted_data, header, optimal=True)
```

**Details:**
- **Input:** Already bias-subtracted data from Stage 1
- **Method:** Optimal extraction (Horne 1986) or boxcar extraction
- **NO additional background subtraction occurs during extraction itself**
- The extraction uses profile weights from the trace to extract fiber spectra

**Key point:** The `ExtractLlamas` class does NOT perform any background subtraction. It assumes the input data has already been background-corrected.

---

## Stage 4: Per-Color Detector Normalization

**Location:** `GUI_extract()` function (lines 517-552)

**What happens:**
```python
# Group detector backgrounds by color
backgrounds_by_color = {'red': {}, 'green': {}, 'blue': {}}
for result in writable_results:
    hdu_idx = result['hdu_index']
    hdu_data = hdu[hdu_idx].data
    if not is_placeholder_camera(hdu_data):
        color = result['extraction'].channel.lower()
        backgrounds_by_color[color][hdu_idx] = result['detector_background']

# Normalize each color separately
for color, color_backgrounds in backgrounds_by_color.items():
    min_background = min(color_backgrounds.values())

    # Apply offsets to detectors of this color
    for result in writable_results:
        hdu_idx = result['hdu_index']
        if hdu_idx in color_backgrounds:
            offset = color_backgrounds[hdu_idx] - min_background
            if offset > 0:
                result['extraction'].counts -= offset  # Subtracts from extracted spectra
```

**Details:**
- **Input:** Extracted spectra (`extraction.counts`) and measured backgrounds from Stage 2
- **Method:**
  1. Group all detectors by color (red/green/blue)
  2. Find the minimum background within each color group
  3. Subtract `(detector_background - min_background)` from each detector's extracted counts
- **Purpose:** Normalize detector-to-detector background variations WITHIN each color channel
- **Why per-color?** Different colors have systematically different background levels (e.g., RED ~7-38, GREEN ~-5 to -16, BLUE ~-17 to -18 per pixel after bias subtraction)

**Output:** Modified `extraction.counts` arrays with per-pixel offset subtracted

---

## Stage 5: Placeholder Camera Handling

**Location:** `GUI_extract()` function (lines 545-552)

**What happens:**
```python
# Set placeholder cameras for this color to min_background
for result in writable_results:
    hdu_idx = result['hdu_index']
    hdu_data = hdu[hdu_idx].data
    extraction_color = result['extraction'].channel.lower()
    if is_placeholder_camera(hdu_data) and extraction_color == color:
        result['extraction'].counts[:] = min_background
```

**Details:**
- **Input:** Placeholder cameras (missing cameras filled with 1.0)
- **Method:** Set ALL counts to `min_background` (the per-color minimum)
- **Purpose:** Ensure placeholder cameras contribute uniform background flux to whitelight images
- **Detection:** `is_placeholder_camera()` checks if all pixel values are 1.0

---

## Summary Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GUI_extract Pipeline                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 1: Bias Frame Subtraction (in process_trace)                 │
│  ─────────────────────────────────────────────────────────────────  │
│  Input:  Raw science frame (2D image)                               │
│  Method: science_data - bias_frame (pixel-by-pixel)                 │
│  Source: combined_bias.fits matched by bench/side/color             │
│  Output: bias_subtracted_data (2D image)                            │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 2: Background Measurement (in process_trace)                 │
│  ─────────────────────────────────────────────────────────────────  │
│  Input:  bias_subtracted_data                                       │
│  Method: np.median(data[30:50, :])                                  │
│  Output: detector_background (single scalar per detector)           │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 3: Spectral Extraction (ExtractLlamas)                       │
│  ─────────────────────────────────────────────────────────────────  │
│  Input:  bias_subtracted_data, trace profiles                       │
│  Method: Optimal or boxcar extraction using trace weights           │
│  Output: extraction.counts (nfibers × nwavelengths array)           │
│  NOTE:   No background subtraction in this stage                    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 4: Per-Color Normalization (in GUI_extract)                  │
│  ─────────────────────────────────────────────────────────────────  │
│  Input:  extraction.counts, detector_background values              │
│  Method: For each color:                                            │
│          offset = detector_bg - min(all detector_bgs for color)     │
│          extraction.counts -= offset                                │
│  Output: Normalized extraction.counts                               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE 5: Placeholder Handling (in GUI_extract)                     │
│  ─────────────────────────────────────────────────────────────────  │
│  Input:  Placeholder camera extractions                             │
│  Method: extraction.counts[:] = min_background                      │
│  Output: Uniform background level for missing cameras               │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  WhiteLightFits (in WhiteLightModule.py)                            │
│  ─────────────────────────────────────────────────────────────────  │
│  Input:  Normalized extractions                                     │
│  Method: np.nansum(counts[ifib]) for each fiber → interpolate       │
│  Output: Whitelight FITS file with RGB images                       │
│  NOTE:   No additional background subtraction                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Alternative Functions

### `ExtractLlamasCube()` (lines 39-84)

Uses a **different** bias subtraction method:
```python
bias = np.nanmedian(hdu[hdu_index].data.astype(float))
extraction = ExtractLlamas(tracer, hdu[hdu_index].data.astype(float)-bias, hdu[hdu_index].header)
```
- Uses **scalar median** of the entire frame as bias (not a bias frame)
- No per-color normalization
- No placeholder handling

### `box_extract()` (lines 615-714)

Uses a **different** bias subtraction method:
```python
data = hdu[hdu_index].data.astype(float)
bias = np.nanmedian(data)
extraction = ExtractLlamas(tracer, data-bias, hdu[hdu_index].header, optimal=False)
```
- Uses **scalar median** of each frame as bias (not a bias frame)
- No per-color normalization
- No placeholder handling

### `QuickWhiteLightCube()` (in WhiteLightModule.py, lines 910-1075)

Uses yet another method:
```python
bias = bias_hdul[i].data
bias_data = np.median(bias[20:50])  # Scalar from rows 20-50 of bias frame
data = ext.data - bias_data
```
- Uses **scalar median** of rows 20-50 from the bias frame
- Not a full 2D bias subtraction
- No per-color normalization

---

## Key Differences Summary

| Function | Bias Method | Background Normalization | Placeholder Handling |
|----------|-------------|--------------------------|---------------------|
| `GUI_extract` | Full 2D bias frame | Yes (per-color) | Yes |
| `ExtractLlamasCube` | Scalar median of frame | No | No |
| `box_extract` | Scalar median of frame | No | No |
| `QuickWhiteLightCube` | Scalar median of bias rows 20-50 | No | No |

---

## Potential Issues

1. **Bias mismatch:** If the combined bias file was taken under different conditions than the science data, the bias subtraction will leave residuals. The per-color normalization helps compensate for this.

2. **GREEN negative values:** If the bias being subtracted is higher than the actual background in the science frame, the result will be negative. This manifests as negative GREEN flux in whitelight images.

3. **Detector-to-detector sensitivity:** The background normalization only corrects for background level differences, NOT for detector sensitivity differences. Sensitivity variations require flat field correction, which is NOT applied in this pipeline.

---

## File Locations

- **guiExtract.py:** `llamas_pyjamas/GUI/guiExtract.py`
- **extractLlamas.py:** `llamas_pyjamas/Extract/extractLlamas.py`
- **WhiteLightModule.py:** `llamas_pyjamas/Image/WhiteLightModule.py`
- **traceLlamasMaster.py:** `llamas_pyjamas/Trace/traceLlamasMaster.py`
- **Combined bias:** `llamas_pyjamas/Bias/combined_bias.fits`
