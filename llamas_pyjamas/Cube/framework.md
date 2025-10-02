# LLAMAS Cube Generation Framework

This document outlines the step-by-step process of how the LLAMAS pipeline transforms RSS (Row-Stacked Spectra) FITS files into 3D IFU data cubes. The process involves reading fiber data, mapping fibers to spatial positions, and constructing a 3D cube with proper wavelength and spatial axes.

## Overview of the Pipeline

The cube generation follows these main steps:

1. Extract RA/DEC coordinates from the RSS FITS header
2. Load channel data from the RSS file 
3. Process each channel separately
4. For each channel, map fiber spectra to their spatial positions
5. Apply interpolation to create a regularly-spaced 3D data cube
6. Save the results as FITS files with proper WCS information

## Detailed Process

### 1. Reading the RSS FITS File

The process begins with the `construct_cube_from_rss` method which:

- Opens the RSS FITS file and loads all available channels using `load_rss_channels`
- Extracts the RA/DEC coordinates from the primary header
- Converts RA/DEC to proper floating-point values if needed
- Passes these coordinates to each channel's cube construction

```python
# Extract RA and DEC from the primary header
with fits.open(rss_file) as hdul:
    primary_header = hdul[0].header
    ra_ref = primary_header.get('RA')
    dec_ref = primary_header.get('DEC')
    
    # Convert to proper floating-point values if they're strings
    if isinstance(ra_ref, str):
        ra_ref = float(ra_ref)
    if isinstance(dec_ref, str):
        dec_ref = float(dec_ref)
```

### 2. Loading Channel Data

The `load_rss_channels` method:

- Scans through all extensions in the RSS FITS file
- Identifies SCI, ERR, and TABLE extensions for each channel
- Loads the flux data from SCI extensions
- Loads error data from ERR extensions
- Loads fiber metadata from TABLE extensions including the 'BENCHSIDE' and 'FIBER' information
- Returns a dictionary with all channel data organized by channel name

### 3. Mapping Fibers to Sky Coordinates

For each channel, the `construct_cube_from_rss_channel` method processes the fiber data:

1. It reads the flux data and table data for the channel
2. Creates a wavelength grid based on the flux array dimensions
3. Maps each fiber to its spatial position using two key methods:

#### Fiber to Physical Coordinates

The `get_fiber_coordinates` method:
- Takes a 'benchside' identifier (e.g., '4B') and fiber number
- Uses the LLAMAS fiber mapping table to look up the physical (x,y) coordinates in arcseconds
- Returns the fiber's position in the focal plane

#### Physical Coordinates to Sky Coordinates

The `map_fiber_to_sky` method:
- Takes the physical coordinates from the previous step
- Uses the reference RA/DEC from the FITS header
- Applies proper astrometric transformation:
  ```python
  # Convert fiber coordinates from arcseconds to degrees and apply offset
  ra = ra_ref + (fiber_x / 3600.0) / np.cos(np.radians(dec_ref))
  dec = dec_ref + (fiber_y / 3600.0)
  ```
- Returns the fiber's position in sky coordinates (RA, DEC)

### 4. Creating the 3D Data Cube

For each channel, after mapping all fibers to spatial positions:

1. Determine the spatial extent of the fiber positions
2. Create regular spatial grids with specified sampling (default: 0.75 arcsec/pixel)
3. Initialize a 3D cube with dimensions [wavelength, y, x]
4. Process each wavelength slice separately:

#### Wavelength Slice Interpolation

For each wavelength slice:
1. Collect all fiber values at this wavelength
2. Create a 2D interpolation grid
3. Use SciPy's `griddata` function with nearest-neighbor interpolation:
   ```python
   grid_z = griddata(
       (fiber_x, fiber_y),     # Points where we know values
       fiber_values,           # Known values
       (xi, yi),               # Points to interpolate
       method='nearest',       # Use nearest neighbor interpolation
       rescale=True            # Rescale to avoid precision issues
   )
   ```
4. Store the interpolated slice in the 3D cube

### 5. Creating World Coordinate System (WCS)

The `create_wcs` method:
- Creates a proper 3D WCS object
- Sets the reference pixel to the center of the cube
- Applies proper pixel scale in degrees/pixel
- Sets reference coordinates (RA, DEC, wavelength)
- Defines appropriate coordinate types and units

### 6. Saving the Cube

Each channel's cube is saved to a separate FITS file using the `save_cube` method:
- Creates a primary HDU with the 3D cube data
- Adds the WCS information to the header
- Adds additional metadata (units, origin, date, etc.)
- Adds wavelength and spatial coordinate extensions
- Writes the FITS file to disk

## Key Data Transformations

The cube construction involves several key data transformations:

1. **Fiber Selection**: Each fiber from the RSS file is identified by its 'BENCHSIDE' and 'FIBER' numbers
2. **Spatial Mapping**: Fibers are mapped from 1D sequential order to 2D spatial positions
3. **Coordinate Conversion**: Physical coordinates are converted to sky coordinates using proper astrometric equations
4. **Spatial Interpolation**: Irregular fiber positions are interpolated onto a regular grid
5. **Metadata Transfer**: Essential metadata (WCS, wavelength calibration) is preserved in the output cube

## Error Handling

The code includes robust error handling at several stages:
- Checking for valid RA/DEC values
- Validating fiber positions
- Handling byte-string encoding of 'BENCHSIDE' identifiers
- Graceful fallback to simpler methods if interpolation fails
- Comprehensive logging of the process

This framework ensures that the irregularly-spaced fiber data from the RSS files is properly transformed into regularly-gridded 3D data cubes with correct spatial and wavelength calibration, ready for scientific analysis.