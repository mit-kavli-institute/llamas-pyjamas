# LLAMAS Simple Cube Constructor

A standalone, MUSE-inspired datacube constructor for LLAMAS RSS files.

## Overview

This script constructs 3D datacubes from LLAMAS RSS FITS files using methodology inspired by the MUSE data reduction pipeline. It's designed to be simple, standalone, and independent of the main LLAMAS reduction framework.

## Key Features

### MUSE-Inspired Methodology

1. **Regular Wavelength Grid**: All fibers resampled onto common wavelength axis
2. **Inverse Distance Weighting**: Spatial interpolation using Gaussian-weighted fiber contributions
3. **Drizzle-like Resampling**: Adjustable spatial pixel size and wavelength sampling
4. **Bad Pixel Masking**: Respects MASK extension from RSS files
5. **Variance Propagation**: Computes variance cube when ERROR extension available

### Advantages Over Complex Methods

- **Standalone**: No dependencies on LLAMAS reduction framework
- **Fast**: Efficient KDTree-based spatial lookups
- **Simple**: Easy to understand and modify
- **Well-tested**: Based on proven MUSE algorithms

## Installation

No installation required - just run the script directly with Python 3:

```bash
cd /path/to/llamas-pyjamas/llamas_pyjamas/Cube
python simple_cube_constructor.py <rss_file>
```

### Dependencies

- numpy
- scipy
- astropy

## Usage

### Basic Usage

```bash
python simple_cube_constructor.py LLAMAS_2025-10-15T04_35_21.383_extract_RSS_red.fits
```

This will create `LLAMAS_2025-10-15T04_35_21.383_extract_RSS_red_cube.fits` with default parameters.

### Advanced Usage

```bash
python simple_cube_constructor.py LLAMAS_extract_RSS_red.fits \
    --pixel-size 0.2 \
    --wave-sampling 0.5 \
    --radius 2.0 \
    --output my_custom_cube.fits
```

## Command-Line Options

### Required Arguments

- `rss_file`: Path to input RSS FITS file

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--fibermap` | Auto-detect | Path to LLAMAS_FiberMap_rev04.dat |
| `--output` | Auto-generate | Output FITS filename |
| `--pixel-size` | 0.3 | Spatial pixel size in arcseconds |
| `--fiber-pitch` | 0.75 | Fiber-to-fiber pitch in arcseconds |
| `--wave-sampling` | 1.0 | Wavelength sampling factor (< 1.0 = oversample) |
| `--radius` | 1.5 | Spatial interpolation radius in arcseconds |
| `--min-weight` | 0.01 | Minimum weight threshold for fiber contribution |
| `--ra` | 0.0 | RA of field center (degrees) |
| `--dec` | 0.0 | Dec of field center (degrees) |
| `--wave-min` | Auto | Minimum wavelength (Angstroms) |
| `--wave-max` | Auto | Maximum wavelength (Angstroms) |

## Examples

### 1. High Spatial Resolution Cube

```bash
python simple_cube_constructor.py RSS_red.fits --pixel-size 0.2
```

Creates cube with 0.2" pixels (finer than default 0.3").

### 2. Oversampled Wavelength Axis

```bash
python simple_cube_constructor.py RSS_blue.fits --wave-sampling 0.5
```

Doubles the wavelength sampling (0.5x native spacing).

### 3. Larger Interpolation Kernel

```bash
python simple_cube_constructor.py RSS_green.fits --radius 2.0
```

Uses 2" radius for spatial interpolation (smoother, less noisy).

### 4. Process All Channels

```bash
for channel in red green blue; do
    python simple_cube_constructor.py LLAMAS_extract_RSS_${channel}.fits \
        --pixel-size 0.25 \
        --output cube_${channel}.fits
done
```

### 5. Custom Fiber Map and Coordinates

```bash
python simple_cube_constructor.py RSS_red.fits \
    --fibermap /custom/path/fibermap.dat \
    --ra 150.1234 \
    --dec -23.4567 \
    --output field123_red_cube.fits
```

## Output FITS Structure

The output cube FITS file contains:

| Extension | Type | Shape | Description |
|-----------|------|-------|-------------|
| `FLUX` (Primary) | ImageHDU | (nλ, ny, nx) | Flux datacube |
| `VAR` | ImageHDU | (nλ, ny, nx) | Variance cube (if ERROR available) |
| `WEIGHT` | ImageHDU | (nλ, ny, nx) | Weight/coverage map |
| `WAVELENGTH` | BinTableHDU | (nλ,) | Wavelength array |

### FITS Header Keywords

- `PIXSIZE`: Spatial pixel size (arcsec)
- `FIBPITCH`: Fiber pitch (arcsec)
- `WAVESAMP`: Wavelength sampling factor
- WCS keywords for spatial and spectral axes

## Algorithm Details

### Spatial Interpolation

For each output spatial pixel (x, y):

1. Find all fibers within `radius` arcseconds
2. Calculate distances d_i to each fiber
3. Compute Gaussian weights: w_i = exp(-0.5 * (d_i / σ)²) where σ = radius/2
4. Normalize weights: w_i / Σw_i
5. Combine fiber spectra: flux = Σ(w_i * flux_i) / Σw_i

This approach:
- Smoothly interpolates between fibers
- Naturally handles edge effects
- Preserves flux (approximately)
- Similar to MUSE's spatial resampling

### Wavelength Resampling

For each fiber:

1. Extract wavelength and flux arrays
2. Remove bad pixels (mask, NaNs)
3. Linearly interpolate onto common wavelength grid
4. Store resampled spectrum

For variance:
- Interpolate variance values
- Propagate through weighted combination: var_out = Σ(w_i² * var_i) / (Σw_i)²

### Parameter Recommendations

**Pixel Size**:
- Default (0.3"): Good balance between resolution and noise
- Small (0.2"): Higher spatial resolution, noisier
- Large (0.5"): Smoother, better S/N for faint sources

**Interpolation Radius**:
- Default (1.5"): ~2 fiber pitches
- Small (1.0"): Sharper PSF, more artifacts
- Large (2.0-3.0"): Smoother, lower resolution

**Wavelength Sampling**:
- 1.0: Native sampling from RSS
- 0.5: 2x oversampling (smoother spectra)
- 0.25: 4x oversampling (for line fitting)

## Differences from Full CubeConstructor

| Feature | SimpleCubeConstructor | Full CubeConstructor |
|---------|----------------------|---------------------|
| **Complexity** | ~600 lines | ~2000+ lines |
| **Dependencies** | Minimal (scipy, astropy) | Full LLAMAS framework |
| **Usage** | Standalone script | Integrated in pipeline |
| **Spatial Method** | Inverse distance weighting | Multiple methods available |
| **Performance** | Fast (KDTree) | Optimized but complex |
| **Variance** | Simple propagation | Full error model |
| **Flexibility** | Fixed algorithm | Customizable |

## Limitations

1. **No Error Propagation Yet**: Assumes errors not yet propagated in RSS (as per current status)
2. **Simple Interpolation**: Uses linear wavelength interpolation (not full drizzling)
3. **Fixed Algorithm**: One interpolation method (inverse distance weighting)
4. **No Sky Subtraction**: Assumes sky already subtracted in RSS
5. **No Telluric Correction**: Not included (should be done before cubing)

## Future Enhancements (Optional)

- [ ] Add variance propagation when errors available
- [ ] Implement true drizzle algorithm
- [ ] Add PSF modeling option
- [ ] Support for combining multiple exposures
- [ ] Parallel processing for large cubes
- [ ] Interactive parameter tuning

## Testing

Test on the new RSS files:

```bash
cd /Users/slh/Documents/Projects/Magellan_dev/LLAMAS/testing/jesse_test/extraction_output

python /path/to/simple_cube_constructor.py \
    LLAMAS_2025-10-15T04_35_21.383_extract_RSS_red.fits \
    --pixel-size 0.3 \
    --output test_cube_red.fits
```

Expected output:
- Cube dimensions: ~(3400, 140, 140) for red channel
- Valid spaxels: ~60-70% of total
- File size: ~500-800 MB

## Troubleshooting

**Q: "Fibermap has different number of entries than RSS"**
A: This is usually fine - the script uses the minimum of both. Check fiber ordering if results look wrong.

**Q: "Very few valid spaxels (<50%)"**
A: Try increasing `--radius` to 2.0 or 2.5 arcseconds.

**Q: "Cube is too noisy"**
A: Increase `--pixel-size` to 0.4 or 0.5, or increase `--radius`.

**Q: "Wavelength range seems wrong"**
A: Explicitly set `--wave-min` and `--wave-max` to match your channel.

## References

- MUSE Data Reduction Pipeline: [ESO MUSE DRS](https://www.eso.org/sci/software/pipelines/muse/)
- Drizzle Algorithm: Fruchter & Hook 2002, PASP, 114, 144
- Inverse Distance Weighting: Shepard 1968

## Author

LLAMAS Pipeline Development Team
Date: 2025-12-13

## License

Part of the LLAMAS-pyjamas pipeline
