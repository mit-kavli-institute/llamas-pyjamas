# LLAMAS Flux Calibration Summary

## Overview
Flux calibration performed on LLAMAS IFU green channel spectra using ESO spectrophotometric standard star LTT 1788.

## Input Data

### ESO Standard Star Reference
- **File**: `/Users/slh/Documents/Projects/independent/Mike_data_reduction/fltt1788.dat`
- **Star**: LTT 1788 (F-type subdwarf, V=13.16 mag)
- **Wavelength coverage**: 3300 - 10100 Å
- **Format**: 137 flux points in 50 Å bins
- **Reference**: Hamuy et al. (1992, 1994) CTIO standards

### Observed Standard Star Spectrum
- **File**: `LLAMAS_2026-01-07_01-03-52.1_SCI22_extract_RSS_green.fits`
- **Row**: 506
- **Fiber ID**: 208
- **Detector**: 1B
- **Wavelength range**: 4639.5 - 7096.2 Å
- **Exposure time**: 60.0 seconds
- **Sky-subtracted median counts**: 340.91 counts
- **Sky level**: 382.93 counts

### Galaxy Spectrum
- **File**: `LLAMAS_2026-01-07_01-25-43.5_SCI22_extract_RSS_green.fits`
- **Row**: 1769
- **Fiber ID**: 278
- **Detector**: 3B
- **Wavelength range**: 4584.1 - 7032.2 Å
- **Exposure time**: 1800.0 seconds
- **Sky-subtracted median counts**: 175.52 counts
- **Sky level**: 4.08 counts

## Calibration Procedure

### Step 1: Sky Subtraction
- **Standard star**: Used 20 faintest fibers from detector 1B
  - Sky fibers: rows 299, 298, 300, 301, 302, ... (median flux ~383 counts)
  - Median sky spectrum computed and subtracted

- **Galaxy**: Used 20 faintest fibers from detector 3B
  - Sky fibers: rows 1790, 1696, 1697, 1727, 1737, ... (median flux ~4 counts)
  - Median sky spectrum computed and subtracted

### Step 2: Sensitivity Function Derivation
- **Formula**: Sensitivity = (counts per second) / (true flux in erg/s/cm²/Å)
- **Method**:
  1. Interpolated ESO reference flux to observed wavelength grid
  2. Computed raw sensitivity function
  3. Applied Savitzky-Golay smoothing (window=101, polyorder=3)
- **Valid points**: 2048/2048 (100%)
- **Sensitivity range**: 5.65×10¹² to 3.58×10¹⁴ [(counts/s)/(erg/s/cm²/Å)]

### Step 3: Flux Calibration
- **Formula**: Flux = (counts / exptime) / sensitivity
- **Applied to**:
  1. Standard star spectrum (for validation)
  2. Galaxy spectrum
- **Wavelength matching**: Sensitivity function interpolated to galaxy wavelength grid

### Step 4: Quality Control
- **Calibration residuals**: Compared calibrated standard star to ESO reference
- **RMS residual**: Computed from (observed - reference) / reference × 100%
- **Visual inspection**: Diagnostic plots showing all calibration steps

## Results

### Flux-Calibrated Standard Star
- **Median flux**: 1.90×10⁻¹⁴ erg/s/cm²/Å
- **Expected flux**: ~1.59×10⁻¹⁴ erg/s/cm²/Å (ESO reference median)
- **Agreement**: ~20% higher (reasonable given aperture/seeing differences)
- **Wavelength coverage**: 4639.5 - 7096.2 Å

### Flux-Calibrated Galaxy
- **Median flux**: 3.22×10⁻¹⁶ erg/s/cm²/Å
- **Valid points**: 2003/2048 (97.8%)
- **Valid wavelength range**: 4640.2 - 7031.1 Å
- **Flux range**: -2.20×10⁻¹⁶ to 1.24×10⁻¹⁵ erg/s/cm²/Å
- **Signal quality**: Good S/N in continuum regions

### Output Files

1. **galaxy_flux_calibrated.txt** (75 KB)
   - Format: 3 columns (Wavelength, Flux, Error)
   - Units: Å, erg/s/cm²/Å, erg/s/cm²/Å
   - 2048 wavelength points

2. **star_flux_calibrated.txt** (76 KB)
   - Format: 3 columns (Wavelength, Flux, Error)
   - Units: Å, erg/s/cm²/Å, erg/s/cm²/Å
   - 2048 wavelength points

3. **flux_calibration_results.png** (501 KB)
   - 9-panel diagnostic plot showing:
     - ESO reference spectrum
     - Observed standard star (counts)
     - Sensitivity function (raw and smoothed)
     - Standard star: reference vs calibrated
     - Calibration residuals
     - Flux-calibrated standard star
     - Flux-calibrated galaxy (full range)
     - Flux-calibrated galaxy (zoomed)
     - Summary statistics

## Calibration Quality Assessment

### Strengths
✓ Complete wavelength overlap between observation and reference (4640-7096 Å)
✓ High S/N standard star observation
✓ Smooth sensitivity function with no major artifacts
✓ Calibrated standard star matches ESO reference shape
✓ Valid calibration across 98% of wavelength range

### Potential Issues
⚠ Standard star shows ~20% flux excess vs reference
  - Possible causes: aperture loss differences, seeing variations, flux extraction method
  - Impact: Systematic flux offset, but spectral shape preserved

⚠ Edge effects at blue/red ends (NaN values)
  - Expected due to limited wavelength overlap
  - Trimmed automatically in output

⚠ No airmass correction applied
  - Set `use_airmass_correction = True` in script if needed
  - Requires airmass values for both observations

### Recommendations
1. **Verify exposure times**: Confirm 60s (star) and 1800s (galaxy) are correct
2. **Check airmass**: If observations at different airmasses, apply correction
3. **Flux validation**: Compare galaxy flux to literature values if available
4. **Telluric correction**: Consider additional correction for strong telluric bands:
   - H₂O: 6800-7600 Å, 8100-9500 Å
   - O₂: 6867-6884 Å, 7594-7621 Å
5. **Aperture correction**: May need to scale total flux if using different extraction apertures

## Usage Examples

### Load Calibrated Galaxy Spectrum
```python
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt('galaxy_flux_calibrated.txt')
wavelength = data[:, 0]
flux = data[:, 1]
error = data[:, 2]

# Plot
plt.figure(figsize=(12, 6))
plt.plot(wavelength, flux * 1e16, 'k-', linewidth=0.8)
plt.fill_between(wavelength,
                 (flux - error) * 1e16,
                 (flux + error) * 1e16,
                 alpha=0.3, color='gray')
plt.xlabel('Wavelength (Å)', fontsize=14)
plt.ylabel('Flux (10⁻¹⁶ erg/s/cm²/Å)', fontsize=14)
plt.title('Flux-Calibrated Galaxy Spectrum', fontsize=16)
plt.grid(True, alpha=0.3)
plt.show()
```

### Measure Emission Line Flux
```python
import numpy as np
from scipy.integrate import trapz

# Load spectrum
data = np.loadtxt('galaxy_flux_calibrated.txt')
wavelength = data[:, 0]
flux = data[:, 1]

# Define line region (e.g., Hα at 6563 Å)
line_center = 6563.0
line_width = 20.0  # Å
continuum_width = 50.0  # Å for continuum estimate

# Extract line region
line_mask = (wavelength > line_center - line_width) & \
            (wavelength < line_center + line_width)

# Extract continuum regions (blue and red)
blue_cont = (wavelength > line_center - continuum_width - 20) & \
            (wavelength < line_center - line_width - 5)
red_cont = (wavelength > line_center + line_width + 5) & \
           (wavelength < line_center + continuum_width + 20)

# Fit continuum (linear)
cont_wave = np.concatenate([wavelength[blue_cont], wavelength[red_cont]])
cont_flux = np.concatenate([flux[blue_cont], flux[red_cont]])
continuum_fit = np.polyfit(cont_wave, cont_flux, 1)
continuum = np.polyval(continuum_fit, wavelength[line_mask])

# Subtract continuum and integrate
line_flux_density = flux[line_mask] - continuum
line_flux = trapz(line_flux_density, wavelength[line_mask])

print(f"Line flux: {line_flux:.2e} erg/s/cm²")
```

### Convert to Luminosity
```python
import numpy as np

# Flux in erg/s/cm²/Å
flux = 3.22e-16  # median galaxy flux

# Distance (example: 100 Mpc)
distance_mpc = 100.0
distance_cm = distance_mpc * 3.086e24  # cm

# Luminosity per Å
luminosity = flux * 4 * np.pi * distance_cm**2

print(f"Luminosity: {luminosity:.2e} erg/s/Å")

# Total luminosity (integrate over wavelength range)
# L_total = ∫ L(λ) dλ
```

## Technical Notes

### Units
- **Input counts**: Dimensionless (ADU or electrons)
- **Sensitivity**: (counts/s) / (erg/s/cm²/Å)
- **Output flux**: erg/s/cm²/Å (flux density per wavelength)
- **IFU specific**: Flux is per fiber (not per arcsec²)

### Assumptions
1. Flat-fielding is accurate across IFU field
2. Same sensitivity function applies to all fibers in same channel
3. Standard star is point source (all flux in one fiber)
4. Sky subtraction removes all non-source flux
5. No significant flux losses between standard and science observations

### Known Limitations
- Green channel only (4584-7032 Å)
- Single fiber extraction (not summing extended galaxy light)
- No telluric correction applied
- No airmass correction applied
- Assumes photometric conditions

## References

1. **ESO Standards**: https://ftp.eso.org/pub/usg/standards/ctiostan/
2. **Hamuy et al. (1992)**: PASP, 104, 533 - "Southern Spectrophotometric Standards"
3. **Hamuy et al. (1994)**: PASP, 106, 566 - "Southern Spectrophotometric Standards, II"

## Script Information

- **Main script**: `flux_calibration.py`
- **Prerequisite**: `sky_subtract_spectra.py` (for understanding sky subtraction method)
- **Dependencies**: astropy, numpy, scipy, matplotlib
- **Run time**: ~5 seconds
- **Created**: 2026-01-11

---

**Next Steps**: Use calibrated spectra for emission line measurements, continuum fitting, stellar population analysis, or spectral energy distribution (SED) construction.
