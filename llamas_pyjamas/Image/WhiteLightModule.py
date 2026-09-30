"""Module for processing and analyzing white light images from LLAMAS.

This module provides functions for processing white light images from the LLAMAS 
instrument, including color channel isolation, FITS file creation, and fiber mapping.

Functions:
    color_isolation: Isolates blue, green, and red channels from extraction objects.
    WhiteLightFits: Creates a FITS file from extraction objects.
    WhiteLight: Generates a white light image from extraction objects.
    WhiteLightQuickLook: Generates a quick look white light image.
    WhiteLightHex: Creates hexagonal grid white light images.
    FiberMap: Maps a fiber to its x and y coordinates.
    FiberMap_LUT: Looks up fiber coordinates using a lookup table.
    plot_fibermap: Plots the fiber map for the LLAMAS instrument.
    fibermap_table: Generates a table of fiber mappings.
    rerun: Reruns the white light generation process.
"""

import logging
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import cKDTree
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.Utils.deadfibers import live_fibre_ids
from llamas_pyjamas.Utils.utils import find_trace_pickle
from llamas_pyjamas.QA import plot_ds9
from llamas_pyjamas.config import OUTPUT_DIR, CALIB_DIR, BIAS_DIR
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
import os
import json
from matplotlib.tri import Triangulation, LinearTriInterpolator
from datetime import datetime
import traceback
from llamas_pyjamas.config import LUT_DIR
from typing import Tuple

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from llamas_pyjamas.File.llamasIO import trim_and_orient
from llamas_pyjamas.DataModel.validate import validate_for_gui
from llamas_pyjamas.Bias.biasChecking import BiasCheckThresholds
from llamas_pyjamas.Postprocessing.build_quicklook_fiberimg import (
    DEFAULT_OUT as QUICKLOOK_CACHE, load_quicklook_fiberimg, quicklook_cache_is_fresh)

from matplotlib.patches import RegularPolygon
import matplotlib.cm as cm
from matplotlib.colors import Normalize



logger = logging.getLogger(__name__)

# Colour order of the image/table extensions in the quick-look white light file
# (extension 1 = RED, ..., BLUE last).
QUICKLOOK_COLOR_ORDER = ('red', 'green', 'blue')

orig_fibre_map_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'LLAMAS_FiberMap_revA.dat')
fibre_map_path = os.path.join(LUT_DIR, 'LLAMAS_FiberMap_rev04.dat')
logger.debug(f'Fibre map path: {fibre_map_path}')   # was print(): fires at import, cluttered startup
fibermap_lut = Table.read(fibre_map_path, format='ascii.fixed_width')

# Extent of the fibre field, taken from the (static) fibre map: x 0.5..46.0,
# y 0.0..44.2. Every white-light path samples this same field, so the grid is
# defined once here rather than re-derived (differently) in each function.
FIELD_XMAX = float(np.max(fibermap_lut['xpos']))
FIELD_YMAX = float(np.max(fibermap_lut['ypos']))
WHITELIGHT_SUBSAMPLE = 1.5


def _lattice_pitch(x, y) -> float:
    """Median nearest-neighbour distance = the fibre lattice spacing."""
    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    if len(pts) < 2:
        return 1.0
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(d[:, 1]))


# (benchside, fibre) -> (xpos, ypos). Built once: filtering the astropy Table per
# fibre (FiberMap_LUT is called ~7000 times per white light) cost over a second.
# setdefault keeps the first matching row, as the old Table lookup did.
FIBERMAP_XY = {}
for _row in fibermap_lut:
    FIBERMAP_XY.setdefault((str(_row['bench']), int(_row['fiber'])), (_row['xpos'], _row['ypos']))

# Fibre lattice spacing from the (static) map: exactly 1.0 -- every fibre has all
# six neighbours at unit distance, rows sqrt(3)/2 apart, alternate rows offset 0.5.
HEX_PITCH = _lattice_pitch(fibermap_lut['xpos'], fibermap_lut['ypos'])


def hex_header_keys(pix_per_unit: int = 10, pitch: float = None) -> dict:
    """FITS keys describing a :func:`hex_tile_image` render.

    Records that the image is hexagonal tiles rather than a resampling, and the
    pixel scale needed to map image pixels back to fibre-map coordinates.
    """
    pitch = HEX_PITCH if pitch is None else float(pitch)
    step = 1.0 / float(pix_per_unit)
    return {
        'HEXTILE': (True, 'Hexagonal fibre tiles, no interpolation'),
        'HEXPITCH': (float(pitch), 'Fibre lattice spacing (fibre-map units)'),
        'PIXUNIT': (int(pix_per_unit), 'Output pixels per fibre-map unit'),
        'CRPIX1': (1.0, 'Reference pixel (1-indexed)'),
        'CRPIX2': (1.0, 'Reference pixel (1-indexed)'),
        'CRVAL1': (0.0, 'Fibre-map x at reference pixel'),
        'CRVAL2': (0.0, 'Fibre-map y at reference pixel'),
        'CDELT1': (step, 'Fibre-map units per pixel'),
        'CDELT2': (step, 'Fibre-map units per pixel'),
    }


def whitelight_grid(subsample: float = WHITELIGHT_SUBSAMPLE):
    """Canonical white-light sampling grid, tight to the fibre field.

    Returns ``(x_grid, y_grid)`` covering ``0..FIELD_XMAX`` by ``0..FIELD_YMAX``
    in steps of ``1/subsample``.

    The old grid hard-coded 53 units on both axes while the fibres only reach
    x=46.0, y=44.2, so ~28% of every frame was NaN padding beyond the last fibre.
    Deriving the extent from the fibre map removes that dead margin and keeps the
    grid identical across frames, channels and code paths (the map is static) —
    previously WhiteLight and QuickWhiteLight built their grids separately and
    silently disagreed about the sky coverage.
    """
    step = 1.0 / float(subsample)
    nx = int(np.floor(FIELD_XMAX / step)) + 1
    ny = int(np.floor(FIELD_YMAX / step)) + 1
    xx = step * np.arange(nx)
    yy = step * np.arange(ny)
    return np.meshgrid(xx, yy)


def hex_tile_image(xdata, ydata, flux, pix_per_unit: int = 10, pitch: float = None):
    """Render each fibre's flux as a flat hexagonal tile -- no interpolation.

    The LLAMAS fibre map is an exact regular hexagonal lattice (spacing 1.0,
    rows sqrt(3)/2 apart, alternate rows offset 0.5), so each fibre's Voronoi
    cell IS its hexagon: pointy-top, inradius ``r = pitch/2``, with flat vertical
    edges shared with its x-neighbours.  Every output pixel therefore takes the
    raw, unresampled flux of the fibre whose hexagon contains it, and NaN if it
    falls outside every hexagon.

    Unlike the interpolated white light, this preserves per-fibre values exactly
    and leaves dead/missing fibres as visible empty hexagons rather than filling
    them in from their neighbours.

    Parameters
    ----------
    xdata, ydata, flux : array_like
        Per-fibre positions (fibre-map units) and scalar fluxes.
    pix_per_unit : int
        Output pixels per lattice unit; 10 gives hexagons 10 px wide (~460x442
        over the full field).
    pitch : float, optional
        Lattice spacing. Defaults to the median nearest-neighbour distance of
        the supplied fibres (1.0 for the real map), so a subset still works.

    Returns
    -------
    (image, header) : (np.ndarray, dict)
        ``image`` is ``(ny, nx)``; ``header`` carries the pixel scale so image
        pixels map back to fibre-map coordinates.
    """
    x = np.asarray(xdata, dtype=float)
    y = np.asarray(ydata, dtype=float)
    f = np.asarray(flux, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)
    x, y, f = x[good], y[good], f[good]
    if x.size == 0:
        raise ValueError("hex_tile_image: no finite fibre positions")

    pts = np.column_stack([x, y])
    if pitch is None:
        pitch = _lattice_pitch(x, y)
    r = 0.5 * float(pitch)          # hexagon inradius (half the lattice spacing)

    step = 1.0 / float(pix_per_unit)
    nx = int(np.floor(FIELD_XMAX / step)) + 1
    ny = int(np.floor(FIELD_YMAX / step)) + 1
    X, Y = np.meshgrid(step * np.arange(nx), step * np.arange(ny))

    # Voronoi cell of a regular hex lattice == the hexagon, so nearest-neighbour
    # assignment is exact inside the field; the slab test below then clips the
    # boundary fibres to true hexagons instead of smearing them outward.
    _, idx = cKDTree(pts).query(np.column_stack([X.ravel(), Y.ravel()]), k=1)
    dx = X.ravel() - x[idx]
    dy = Y.ravel() - y[idx]

    # Tolerance so pixels landing exactly on a hexagon edge are kept: the pitch is
    # derived from finite-precision map coordinates (0.9999996, not 1.0), and pixel
    # centres fall exactly on dx=+/-r, which would otherwise leave a NaN seam around
    # every tile. The nearest-neighbour step already assigns each pixel to exactly
    # one fibre, so a tolerance cannot double-paint.
    tol = 1e-6 * pitch
    s3 = np.sqrt(3.0) / 2.0
    inside = ((np.abs(dx) <= r + tol) &
              (np.abs(0.5 * dx + s3 * dy) <= r + tol) &
              (np.abs(0.5 * dx - s3 * dy) <= r + tol))

    image = np.where(inside, f[idx], np.nan).reshape(X.shape)
    return image, hex_header_keys(pix_per_unit, pitch=pitch)


def color_isolation(extractions: list, metadata: dict)-> Tuple[list, list, list]:
    """Isolate blue, green, and red channels from extraction objects.

    This function separates extraction objects by their color channels and returns 
    both the extraction objects and their corresponding metadata.

    Args:
        extractions (list): A list of extraction objects loaded from ExtractLlamas.
        metadata (dict): Dictionary containing metadata for each extraction.

    Returns:
        tuple: A tuple containing six lists:
            - blue_extractions (list): Blue channel extraction objects.
            - green_extractions (list): Green channel extraction objects.
            - red_extractions (list): Red channel extraction objects.
            - blue_meta (list): Metadata for blue channel extractions.
            - green_meta (list): Metadata for green channel extractions.
            - red_meta (list): Metadata for red channel extractions.
    """

    blue_extractions = [ext for ext in extractions if ext.channel.lower() == 'blue']
    green_extractions = [ext for ext in extractions if ext.channel.lower() == 'green']
    red_extractions = [ext for ext in extractions if ext.channel.lower() == 'red']

    blue_meta = [meta for meta in metadata if meta['channel'].lower() == 'blue']
    green_meta = [meta for meta in metadata if meta['channel'].lower() == 'green']
    red_meta = [meta for meta in metadata if meta['channel'].lower() == 'red']

    
    return blue_extractions, green_extractions, red_extractions, blue_meta, green_meta, red_meta


def _whitelight_wcs_header(x, y, primary_header, hex_tiles, pix_per_unit):
    """Celestial WCS cards for a white-light colour image, or None if no header pointing.

    Mirrors ``RSSScene.collapse``: CRVAL at the field centre (header RA/DEC), CRPIX the image
    pixel that maps there. Both render grids map pixel p -> fibre-map (p-1)*step, so the field
    centre (midpoint of the fibre extent) is at centre/step + 1. Interpolated grid step is
    1/WHITELIGHT_SUBSAMPLE; hex-tile step is 1/pix_per_unit.
    """
    from llamas_pyjamas.Utils.wcsLlamas import (ARCSEC_PER_FIBRE, celestial_wcs,
                                                pointing_from_header)
    ra, dec, pa = pointing_from_header(primary_header)
    if ra is None or dec is None or len(x) == 0:
        return None
    step = (1.0 / float(pix_per_unit)) if hex_tiles else (1.0 / float(WHITELIGHT_SUBSAMPLE))
    cx = 0.5 * (float(np.nanmin(x)) + float(np.nanmax(x)))
    cy = 0.5 * (float(np.nanmin(y)) + float(np.nanmax(y)))
    wcs = celestial_wcs(ra, dec, crpix=(cx / step + 1.0, cy / step + 1.0),
                        arcsec_per_pixel=ARCSEC_PER_FIBRE * step, pa_deg=pa)
    return wcs.to_header()


def WhiteLightFits(extraction_array: list, metadata: dict, outfile=None,
                   hex_tiles: bool = False, pix_per_unit: int = 10,
                   primary_header=None)-> str:
    """Process extraction data to create a white light FITS file.

    Set ``hex_tiles=True`` to render each fibre as a flat hexagonal tile of its
    raw flux (no interpolation; see :func:`hex_tile_image`) instead of the
    default resampling onto the rectangular grid. The per-fibre ``{COLOR}_TAB``
    extension is written either way.

    This function takes an array of extracted color data and creates a FITS file 
    containing white light images for each bench/side/channel combination.

    Args:
        extraction_array (list): A list of extracted color data arrays.
        metadata (dict): Dictionary containing metadata for each extraction.
        outfile (str, optional): The output file path for the white light FITS file. 
            If None, the output file name is generated based on the input file name. 
            Defaults to None.

    Returns:
        str: The file path of the created white light FITS file.

    Note:
        - The function assumes that all extraction objects came from the same original file.
        - The function processes blue, green, and red data if they exist in the extraction array.
        - The function creates a primary HDU and additional HDUs for each color data and their corresponding tables.
        - The function writes the created HDU list to a FITS file in the specified output directory.
    """

    
    blue, green, red, blue_meta, green_meta, red_meta = color_isolation(extraction_array, metadata)
    print(blue, green, red)
    fitsfile = None
    ###For now assuming that all extraction objects came from the same original file
    if all(not color for color in [blue, green, red]):
        logger.error('No blue, green, or red extractions found. Exiting...')
        return
    
    # Object name for the frame (from the science header) — set on the primary and every image
    # HDU so DS9 shows it whichever extension is opened.
    obj_name = str(primary_header.get('OBJECT', '')) if primary_header is not None else ''

    # Create HDU list
    hdul = fits.HDUList()
    primary_hdu = fits.PrimaryHDU()
    fitsfile = blue[0].fitsfile if blue else green[0].fitsfile if green else red[0].fitsfile
    primary_hdu.header['ORIGFILE'] = os.path.basename(fitsfile)
    # Carry object + pointing from the science header so the frame is self-describing. (This
    # PrimaryHDU used to be discarded — a fresh empty one was appended — losing all of it.)
    if primary_header is not None:
        for _k in ('OBJECT', 'RA', 'DEC', 'TEL RA', 'TEL DEC', 'TEL ROT', 'TEL PA'):
            if _k in primary_header:
                primary_hdu.header[_k] = primary_header[_k]
    hdul.append(primary_hdu)

    # Process blue data if exists
    if blue:
        
        blue_whitelight, blue_x, blue_y, blue_flux = WhiteLight(blue, blue_meta, ds9plot=False, hex_tiles=hex_tiles, pix_per_unit=pix_per_unit)
        blue_hdu = fits.ImageHDU(data=blue_whitelight.astype(float), name='BLUE')
        if hex_tiles:
            for _k, _v in hex_header_keys(pix_per_unit).items():
                blue_hdu.header[_k] = _v
        _wcs = _whitelight_wcs_header(blue_x, blue_y, primary_header, hex_tiles, pix_per_unit)
        if _wcs is not None:
            blue_hdu.header.update(_wcs)
        blue_hdu.header['OBJECT'] = obj_name
        hdul.append(blue_hdu)
        
        blue_tab = fits.BinTableHDU.from_columns([
            fits.Column(name='XDATA', format='E', array=blue_x.astype(np.float32)),
            fits.Column(name='YDATA', format='E', array=blue_y.astype(np.float32)),
            fits.Column(name='FLUX', format='E', array=blue_flux.astype(np.float32))
        ], name='BLUE_TAB')
        hdul.append(blue_tab)
    
    # Process green data if exists
    if green:
      
        green_whitelight, green_x, green_y, green_flux = WhiteLight(green, green_meta, ds9plot=False, hex_tiles=hex_tiles, pix_per_unit=pix_per_unit)
        green_hdu = fits.ImageHDU(data=green_whitelight.astype(float), name='GREEN')
        if hex_tiles:
            for _k, _v in hex_header_keys(pix_per_unit).items():
                green_hdu.header[_k] = _v
        _wcs = _whitelight_wcs_header(green_x, green_y, primary_header, hex_tiles, pix_per_unit)
        if _wcs is not None:
            green_hdu.header.update(_wcs)
        green_hdu.header['OBJECT'] = obj_name
        hdul.append(green_hdu)
        
        green_tab = fits.BinTableHDU.from_columns([
            fits.Column(name='XDATA', format='E', array=green_x.astype(float)),
            fits.Column(name='YDATA', format='E', array=green_y.astype(float)),
            fits.Column(name='FLUX', format='E', array=green_flux.astype(float))
        ], name='GREEN_TAB')
        hdul.append(green_tab)
    
    # Process red data if exists
    if red:
        red_whitelight, red_x, red_y, red_flux = WhiteLight(red, red_meta, ds9plot=False, hex_tiles=hex_tiles, pix_per_unit=pix_per_unit)
        red_hdu = fits.ImageHDU(data=red_whitelight.astype(np.float32), name='RED')
        if hex_tiles:
            for _k, _v in hex_header_keys(pix_per_unit).items():
                red_hdu.header[_k] = _v
        _wcs = _whitelight_wcs_header(red_x, red_y, primary_header, hex_tiles, pix_per_unit)
        if _wcs is not None:
            red_hdu.header.update(_wcs)
        red_hdu.header['OBJECT'] = obj_name
        hdul.append(red_hdu)
        
        red_tab = fits.BinTableHDU.from_columns([
            fits.Column(name='XDATA', format='E', array=red_x.astype(np.float32)),
            fits.Column(name='YDATA', format='E', array=red_y.astype(np.float32)),
            fits.Column(name='FLUX', format='E', array=red_flux.astype(np.float32))
        ], name='RED_TAB')
        hdul.append(red_tab)
    if not outfile:
        fitsfilebase = fitsfile.split('/')[-1]
        white_light_file = fitsfilebase.replace('.fits', '_whitelight.fits')
    
    #code added in to allow for normalisation in flat fielding process
    elif outfile == -1:
        return hdul
    else:
        white_light_file = outfile
    
    print(f'Writing white light file to {white_light_file}')
    # Write to file
    hdul.writeto(os.path.join(OUTPUT_DIR, white_light_file), overwrite=True)
    
    return white_light_file


def WhiteLight(extraction_array: list, metadata: list, ds9plot=True,
               hex_tiles: bool = False, pix_per_unit: int = 10)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a white light image from an array of extraction files or objects.
    Parameters:
    extraction_array (list): A list of extraction files (str) or ExtractLlamas objects.
    ds9plot (bool, optional): If True, plot the white light image using DS9. Default is True.
    hex_tiles (bool, optional): If True, render each fibre as a flat hexagonal
        tile of its raw flux (no interpolation, dead fibres left as holes) via
        :func:`hex_tile_image` instead of resampling onto the rectangular grid.
        Default False -- the interpolated image is unchanged.
    pix_per_unit (int, optional): Output pixels per fibre-map unit; only used
        when ``hex_tiles`` is True. Default 10 (~460x442).
    Returns:
    tuple: A tuple containing:
        - whitelight (numpy.ndarray): The interpolated white light image.
        - xdata (numpy.ndarray): The x-coordinates of the fiber positions.
        - ydata (numpy.ndarray): The y-coordinates of the fiber positions.
        - flux (numpy.ndarray): The flux values for each fiber.
    Raises:
    AssertionError: If extraction_array is not a list.
    TypeError: If an element in extraction_array is not a string or ExtractLlamas object.
    """

    
    assert type(extraction_array) == list, 'Extraction array must be a list of extraction files'
    
    xdata = np.array([])
    ydata = np.array([])
    flux  = np.array([])
    
    for extraction_obj, meta in zip(extraction_array, metadata):
        channel = meta['channel']
        side = meta['side']
        counts = extraction_obj.counts
        #Might need to put in side condition here as well it depends on the outcome
        # if channel == 'blue':
        #     counts = np.flipud(extraction_obj.counts)
    
        if isinstance(extraction_obj, str):
            extraction, _ = ExtractLlamas.loadExtraction(extraction_obj)
            logger.info(f'Loaded extraction object {extraction.bench}{extraction.side}')
        elif isinstance(extraction_obj, ExtractLlamas):
            extraction = extraction_obj
        else:
            raise TypeError(f"Unexpected type: {type(extraction_obj)}. Must be string or ExtractLlamas object")
        
        
        nfib, naxis1 = np.shape(extraction.counts)

        # counts is LIVE-indexed (dead fibres absent, since commit c8ab75f), so
        # row `ifib` is NOT the fibremap position. Map each live row to its
        # physical fibremap position before the LUT lookup, else every fibre
        # after the first dead one is placed at the wrong (x, y).
        fibmap_pos = live_fibre_ids(nfib, getattr(extraction, 'dead_fibers', []))

        for ifib in range(nfib):
            benchside = f'{extraction.bench}{extraction.side}'
            fpos = fibmap_pos[ifib]

            try:
                x, y = FiberMap_LUT(benchside, fpos)
            except Exception as e:
                logger.info(f'Fiber {fpos} (live row {ifib}) not found in fiber map for bench {benchside} for color {extraction.channel}')
                logger.error(traceback.format_exc())
                continue


            # thisflux = np.nansum(extraction.counts[ifib])
            thisflux = np.nansum(counts[ifib])
            flux = np.append(flux, thisflux)
            xdata = np.append(xdata,x)
            ydata = np.append(ydata,y)

    if hex_tiles:
        whitelight, _ = hex_tile_image(xdata, ydata, flux, pix_per_unit=pix_per_unit)
    else:
        flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux, fill_value=np.nan)

        x_grid, y_grid = whitelight_grid()

        whitelight = flux_interpolator(x_grid, y_grid)
    # whitelight = np.fliplr(whitelight)
    if (ds9plot):
        #ds9 = pyds9.DS9(target='DS9:*', start=True, wait=10, verify=True)
        #ds9.set_np2arr(whitelight)
        plot_ds9(whitelight)

    return whitelight, xdata, ydata, flux


def WhiteLightFromRSS(rss_file: str, outfile: str = None,
                      wave_min: float = None, wave_max: float = None,
                      hex_tiles: bool = False, pix_per_unit: int = 10) -> str:
    """Create a white light image by summing the sky-subtracted FLUX extension of an RSS file.

    Reads the FLUX (extension 1), WAVE (extension 5), and FIBERMAP extensions,
    collapses each fiber's spectrum to a scalar by nansum within the requested
    wavelength range, looks up the spatial position via FiberMap_LUT, interpolates
    onto a regular grid, and writes a FITS file.

    Args:
        rss_file (str): Path to an RSS FITS file produced by generate_rss().
        outfile (str, optional): Output path.  Defaults to rss_file with
            '_whitelight.fits' substituted for '.fits'.
        wave_min (float, optional): Minimum wavelength in Angstroms.  If None,
            no lower bound is applied.
        wave_max (float, optional): Maximum wavelength in Angstroms.  If None,
            no upper bound is applied.

    Returns:
        str: Path to the written white light FITS file.
    """

    if outfile is None:
        outfile = rss_file.replace('.fits', '_whitelight.fits')

    with fits.open(rss_file) as hdul:
        from llamas_pyjamas.File.llamasRSS import skysub_extname
        flux     = hdul[skysub_extname(hdul)].data   # sky-subtracted plane (SKYSUB, or FLUX pre-rename)
        wave     = hdul['WAVE'].data          # shape: (n_fibers, n_wave)
        fibermap = hdul['FIBERMAP'].data
        channel  = hdul[0].header.get('CHANNEL', 'UNKNOWN')

    fiber_ids  = fibermap['FIBER_ID']
    benchsides = fibermap['BENCHSIDE']

    xdata = np.array([])
    ydata = np.array([])
    fdata = np.array([])

    for i in range(len(fiber_ids)):
        benchside = str(benchsides[i]).strip()
        fiber_id  = int(fiber_ids[i])
        try:
            x, y = FiberMap_LUT(benchside, fiber_id)
        except Exception:
            continue

        fiber_wave = wave[i, :]
        mask = np.ones(fiber_wave.shape, dtype=bool)
        if wave_min is not None:
            mask &= fiber_wave >= wave_min
        if wave_max is not None:
            mask &= fiber_wave <= wave_max

        thisflux = np.nansum(flux[i, mask])
        xdata = np.append(xdata, x)
        ydata = np.append(ydata, y)
        fdata = np.append(fdata, thisflux)

    if len(xdata) == 0:
        logger.error(f'WhiteLightFromRSS: no fibers mapped for {rss_file}')
        return None

    hex_header = None
    if hex_tiles:
        whitelight, hex_header = hex_tile_image(xdata, ydata, fdata,
                                                pix_per_unit=pix_per_unit)
    else:
        flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), fdata,
                                                 fill_value=np.nan)
        x_grid, y_grid = whitelight_grid()
        whitelight = flux_interpolator(x_grid, y_grid)

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['ORIGFILE'] = os.path.basename(rss_file)
    primary_hdu.header['CHANNEL']  = channel
    if wave_min is not None:
        primary_hdu.header['WAVEMIN'] = (wave_min, 'Minimum wavelength (Angstroms)')
    if wave_max is not None:
        primary_hdu.header['WAVEMAX'] = (wave_max, 'Maximum wavelength (Angstroms)')

    img_hdu = fits.ImageHDU(data=whitelight.astype(np.float32), name=channel.upper())
    if hex_header:
        for key, val in hex_header.items():
            img_hdu.header[key] = val

    tab_hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='XDATA', format='E', array=xdata.astype(np.float32)),
        fits.Column(name='YDATA', format='E', array=ydata.astype(np.float32)),
        fits.Column(name='FLUX',  format='E', array=fdata.astype(np.float32)),
    ], name=f'{channel.upper()}_TAB')

    hdul_out = fits.HDUList([primary_hdu, img_hdu, tab_hdu])
    hdul_out.writeto(outfile, overwrite=True)
    print(f'White light image written to {outfile}')
    return outfile


def WhiteLightQuickLook(tracefile: str, data)-> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a quick look white light image from trace data and image data.
    Parameters:
    tracefile (str): Path to the trace file containing the trace object.
    data (numpy.ndarray): Image data array.
    Returns:
    tuple: A tuple containing:
        - whitelight (numpy.ndarray): Interpolated white light image.
        - xdata (numpy.ndarray): Array of x-coordinates for fibers.
        - ydata (numpy.ndarray): Array of y-coordinates for fibers.
        - flux (numpy.ndarray): Array of flux values for fibers.
    Notes:
    - The function reads the trace object from the provided tracefile.
    - It uses a fiber map lookup table (FiberMap_LUT) to get x and y coordinates for each fiber.
    - The flux for each fiber is calculated by summing the data values where the fiber image matches the fiber index.
    - A linear interpolator (LinearNDInterpolator) is used to create the white light image.
    - Optionally, the white light image can be plotted using DS9 (if ds9plot is set to True).
    """

    #    hdul = fits.open(data)

    # Each trace object represents one camera / side pair
    
    with open(tracefile, "rb") as fp:
        traceobj = pickle.load(fp)
    fiberimg = traceobj.fiberimg
    nfib     = traceobj.nfibers
    
    xdata = np.array([])
    ydata = np.array([])
    flux  = np.array([])
    for ifib in range(nfib):
        benchside = f'{traceobj.bench}{traceobj.side}'
        channel = f'{traceobj.channel}'
        try:
            x, y = FiberMap_LUT(benchside,ifib)
        except Exception as e:
            logger.info(f'Fiber {ifib} not found in fiber map for bench {benchside}')
            logger.error(traceback.format_exc())
            continue
        
        thisflux = np.nansum(data[fiberimg == ifib])
        flux = np.append(flux, thisflux)
        xdata = np.append(xdata,x)
        ydata = np.append(ydata,y)

    flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux, fill_value=np.nan)
        
    # Shared field grid: the old 46x43 grid stopped at y=42 and clipped the top of
    # bench 1A (which runs to y=44.2).
    x_grid, y_grid = whitelight_grid()

    whitelight = flux_interpolator(x_grid, y_grid)

    ds9plot = False
    if (ds9plot):
        plot_ds9(whitelight, samp=True)

    return whitelight, xdata, ydata, flux

        
def WhiteLightHex(extraction_array, ds9plot=True):
    pass

    ## placeholder for eventual hexagonal grid inclusion

    return

   
def FiberMap(bench: str, infiber: int)-> Tuple[float, float]:
    """
    Calculate the fiber map coordinates for a given bench and fiber number.
    Parameters:
    bench (str): The bench identifier, which can be '1A', '2A', '3A', '4A', '1B', '2B', '3B', or '4B'.
    infiber (int): The fiber number.
    Returns:
    tuple: A tuple containing the x and y coordinates of the fiber.
    """


    n_right    = 23
    n_left     = 23
    ncols = n_left + n_right
    n_vertical = 51
    dy         = 0.8660254037 # == np.sin(60), but hardwire the factor for time saving
    dx         = 1.0

    wrap = 0
    if (bench == '1A'):
        x0 = 0.0
        y0 = 0.0
        Nfib=298
    elif (bench == '2A'):
        x0 = np.floor_divide(n_right,2)
        y0 = 6.0
        wrap = 23
        Nfib=300
    elif (bench == '3A'):
        x0 = 0.0
        y0 = 13.0
        Nfib=298
    elif (bench == '4A'):
        x0 = np.floor_divide(n_right,2)
        y0 = 19
        wrap = 23
        Nfib=300

        
    elif (bench == '1B'):
        x0 = 0.0
        y0 = 45
        Nfib = 300#298
        wrap = 23
    elif (bench == '2B'):
        x0 = np.floor_divide(n_right,2)
        y0 = 39.0
        Nfib = 298#300
    elif (bench == '3B'):
        x0 = 0.0
        y0 = 32.0
        Nfib=300#298
        wrap = 23
    elif (bench == '4B'):
        x0 = np.floor_divide(n_right,2)
        y0 = 26.0
        Nfib = 298#300
        
    fiber = infiber
        
    if (fiber % 2 == 1):
        xoffset = n_left
        wrap -= 1
    else:
        xoffset = 0


    if ('A' in bench):
        y_rownum = np.floor_divide((fiber+wrap), (n_left+n_right))
        y_value  = (y0 + y_rownum) * dy

        # Account for the fact that the fibers snake back and forth within the two
        # sides of the IFU. Even rows go right to left, odd rows go left to right

        x_index = np.floor_divide(((fiber+wrap) % (n_left+n_right)),2)

        if ((bench == '1A') or (bench == '2A')):
            offset = 1
        else:
            offset = 0

        if (((y_rownum+y0+offset) % 2) == 0):
            if ((fiber % 2) == 1):
                x_fiber = (n_left+n_right) - x_index - 1
            else:
                x_fiber = n_left - x_index - 1
        else:
            if ((fiber % 2) == 1):
                x_fiber = n_left + 0.5 + x_index
            else:
                x_fiber = 0.5 + x_index

        x_value = x_fiber - 0.5
        x_fiber = ncols - x_fiber
        y_value = n_vertical*dy-y_value

    elif ('B' in bench):

        y_rownum = np.floor_divide((fiber+wrap), ncols)
        y_value  = (y0 + y_rownum) * dy

        # Account for the fact that the fibers snake back and forth within the two
        # sides of the IFU. Even rows go right to left, odd rows go left to right

        x_index = np.floor_divide(((fiber+wrap) % (n_left+n_right)),2)

        if ((bench == '2B') or (bench == '1B')):
            offset = 1
        else:
            offset = 0

        if (((y_rownum+y0+offset) % 2) == 0):
            if ((fiber % 2) == 0):
                # Even fiber number
                x_fiber = (n_left+n_right) - x_index
            else:
                # Odd fiber numbers
                x_fiber = n_left - x_index

            x_fiber -= 0.5
            # x_value = x_fiber - 0.5

            #if (bench=='2B'):
            #    x_fiber+=0.5
            
        else:
            if ((fiber % 2) == 0):
                x_fiber = n_left + 0.5 + x_index
            else:
                x_fiber = 0.5 + x_index

            x_fiber += 0.5

            #if (bench=='2B'):
            #    x_fiber-=0.5
            

        y_value = n_vertical*dy-y_value
        
    # return(x_value,y_value)

    y_final = n_vertical-int(y0+y_rownum)
    x_final = x_fiber

    if ((bench == '1B') or (bench == '2B') or (bench == '3A') or (bench == '4A')):
        if (y_final % 2 == 0):
            x_final += 0.5
        else:
            x_final -= 0.5   

    # return(x_fiber,n_vertical-int(y0+y_rownum))
    return(x_final, y_final)

def FiberMap_LUT(bench: str, fiber: int)-> Tuple[float, float]:
    """(xpos, ypos) of a physical fibre on a benchside, or (-1, -1) if not in the map."""
    return FIBERMAP_XY.get((bench, fiber), (-1, -1))

def plot_fibermap(outpath: str)-> None:
    """
    Plots the fiber map for the LLAMAS IFU, showing the mapping of fibers to different configurations.
    This function generates a plot with the fiber numbers annotated at their respective positions for 
    different configurations (1A, 2A, 3A, 4A, 1B, 2B, 3B, 4B). It also includes directional annotations 
    (N for North, E for East) and saves the plot as an image file.
    Annotations:
    - Fibers in configuration 1A, 3A, 4B, and 2B are plotted in black and red.
    - Fibers in configuration 2A, 4A, 3B, and 1B are plotted in blue and green.
    - Directional annotations for North and East are included.
    The plot is saved as "fiber.png" in the specified directory.
    Returns:
        None
    """


    # 1A - N=298
    # 2A - N=300
    # 3A - all fibers fine (N=298)
    # 4A - all fibers fine (N=300)
    # 4B - all fibers fine (N=298)
    # 3B - all fibers fine (N=300)
    # 2B - fiber 49 (zero index) is broken / dead (N=297 good + 1 dead)
    # 1B - all fibers fine (N=300)
    
    fig, ax = plt.subplots(1)
    fibernum_a = np.arange(298)
    fibernum_b = np.arange(300)

    fs = 4
    for fiber in fibernum_a:
        x, y = FiberMap_LUT('1A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='k')

        x, y = FiberMap_LUT('3A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='r')

        x, y = FiberMap_LUT('4B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='k')

        x, y = FiberMap_LUT('2B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='r')
        
        
    for fiber in fibernum_b:
        x, y = FiberMap_LUT('2A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='b')

        x, y = FiberMap_LUT('4A', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='g')

        x, y = FiberMap_LUT('3B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='b')

        x, y = FiberMap_LUT('1B', int(fiber))
        ax.text(x, y, f'{fiber}', fontsize=fs, horizontalalignment='center',verticalalignment='center', color='g')
        
        
    ax.set_xlim(-2,75)
    ax.set_ylim(-2,47)
    ax.set_aspect('equal')

    ax.text(49,2.5,"1B")
    ax.text(49,7.5,"2B")
    ax.text(49,12.5,"3B")
    ax.text(49,17.5,"4B")
    ax.text(49,24.5,"4A")
    ax.text(49,29.5,"3A")
    ax.text(49,34.5,"2A")
    ax.text(49,39.5,"1A")

    fs = 8

    ax.text(52.25, 40.5, "R-1", fontsize=fs)
    ax.text(52.25, 39.5, "G-2", fontsize=fs)
    ax.text(52.25, 38.5, "B-3", fontsize=fs)

    ax.text(52.25, 35.5, "R-7", fontsize=fs)
    ax.text(52.25, 34.5, "G-8", fontsize=fs)
    ax.text(52.25, 33.5, "B-9", fontsize=fs)

    ax.text(52.25, 30.5, "R-13", fontsize=fs)
    ax.text(52.25, 29.5, "G-14", fontsize=fs)
    ax.text(52.25, 28.5, "B-15", fontsize=fs)

    ax.text(52.25, 25.5, "R-19", fontsize=fs)
    ax.text(52.25, 24.5, "G-20", fontsize=fs)
    ax.text(52.25, 23.5, "B-21", fontsize=fs)

    ax.text(52.25, 3.5, "R-4", fontsize=fs)
    ax.text(52.25, 2.5, "G-5", fontsize=fs)
    ax.text(52.25, 1.5, "B-6", fontsize=fs)

    ax.text(52.25, 8.5, "R-10", fontsize=fs)
    ax.text(52.25, 7.5, "G-11", fontsize=fs)
    ax.text(52.25, 6.5, "B-12", fontsize=fs)

    ax.text(52.25, 13.5, "R-16", fontsize=fs)
    ax.text(52.25, 12.5, "G-17", fontsize=fs)
    ax.text(52.25, 11.5, "B-18", fontsize=fs)

    ax.text(52.25, 18.5, "R-22", fontsize=fs)
    ax.text(52.25, 17.5, "G-23", fontsize=fs)
    ax.text(52.25, 16.5, "B-24", fontsize=fs)

    ax.annotate('', xy=(73,20), xytext=(60,20),
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))

    ax.annotate('', xy=(60,25), xytext=(60,20),
            arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))
    ax.text(71, 15.5, "N", fontsize=12)
    ax.text(60, 27, "E", fontsize=12)

    plt.title("LLAMAS IFU Fiber to slit / FITS extension mapping (Rev 04)")

    plt.tight_layout()
    fig.savefig(outpath, dpi=600)
    plt.show()


def fibermap_table()-> Table:
    """
    Generates a fiber map table for the LLAMAS instrument and writes it to a file.
    The function creates a table with columns: 'bench', 'fiber', 'xindex', 'yindex', 'xpos', and 'ypos'.
    It populates the table with fiber positions for both A and B sides of the instrument, using the FiberMap function
    to get the x and y indices for each fiber. The y position is adjusted by the sine of 60 degrees.
    The table is then written to a file named 'LLAMAS_FiberMap_rev02_updated.dat' in fixed-width ASCII format.
    Returns:
        astropy.table.Table: The populated fiber map table.
    """


    fiber_table = Table(names=('bench','fiber','xindex','yindex','xpos','ypos'),\
                        dtype=('S4','i4','i4','i4','f4','f4'))

    # A sides - DONE!
    
    for ifib in range(298):
        ix, iy = FiberMap('1A',ifib)
        fiber_table.add_row(['1A',int(ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(300):
        ix, iy = FiberMap('2A',ifib)
        fiber_table.add_row(['2A',int(ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(298):
        ix, iy = FiberMap('3A',ifib)
        fiber_table.add_row(['3A',int(ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(300):
        ix, iy = FiberMap('4A',ifib)
        fiber_table.add_row(['4A',int(ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    # B Sides
        
    for ifib in range(300):
        ix, iy = FiberMap('1B',ifib)
        fiber_table.add_row(['1B',int(299-ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(298):
        ix, iy = FiberMap('2B',ifib)
        fiber_table.add_row(['2B',int(297-ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(300):
        ix, iy = FiberMap('3B',ifib)
        fiber_table.add_row(['3B',int(299-ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    for ifib in range(298):
        ix, iy = FiberMap('4B',ifib)
        fiber_table.add_row(['4B',int(297-ifib),ix,iy,ix,iy*np.sin(60*np.pi/180)])

    fiber_table.write('LLAMAS_FiberMap_rev03.dat', format='ascii.fixed_width', overwrite=True)
        
    return(fiber_table)
    
def rerun():
    """
    Reruns the WhiteLight process with a set of predefined extractions.
    This function loads several extraction files using the ExtractLlamas class and then
    runs the WhiteLight process with these extractions.
    The following extraction files are loaded:
    - Extract_1A.pkl
    - Extract_2A.pkl
    - Extract_3A.pkl
    - Extract_4A.pkl
    - Extract_1B.pkl
    - Extract_2B.pkl
    - Extract_3B.pkl
    - Extract_4B.pkl
    The loaded extractions are then passed to the WhiteLight function for processing.
    """

    extraction1a = ('Extract_1A.pkl')
    extraction2a = ExtractLlamas.loadExtraction('Extract_2A.pkl')
    extraction3a = ExtractLlamas.loadExtraction('Extract_3A.pkl')
    extraction4a = ExtractLlamas.loadExtraction('Extract_4A.pkl')

    extraction1b = ExtractLlamas.loadExtraction('Extract_1B.pkl')
    extraction2b = ExtractLlamas.loadExtraction('Extract_2B.pkl')
    extraction3b = ExtractLlamas.loadExtraction('Extract_3B.pkl')
    extraction4b = ExtractLlamas.loadExtraction('Extract_4B.pkl')
    

    WhiteLight([extraction1a, extraction2a, extraction3a, extraction4a, extraction1b, extraction2b, extraction3b, extraction4b])



######### Testing qucik whitelight

def _load_dead_fiber_lut() -> dict:
    """Dead physical fibres per benchside from traceLUT.json (same source as extractLlamas)."""
    try:
        with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'r') as f:
            return json.load(f).get('dead_fibers', {})
    except Exception as e:
        logger.warning(f'Could not load dead fiber definitions from traceLUT.json: {e}')
        return {}


def _detector_fibre_fluxes(fiberimg, nfib: int, benchside: str, data, dead_fiber_lut: dict,
                           offset: float = 0.0):
    """Summed flux and IFU position of every traced fibre on one detector.

    Sums ``data - offset`` over each fibre's pixels in ``fiberimg`` (-1 = no fibre;
    NaNs ignored, like ``np.nansum``), accumulating in float64. Fibres are
    horizontal bands, so the raveled label image is ~10k runs of one label: each
    run is summed with ``np.add.reduceat`` and the runs are binned per fibre,
    which is ~4x faster than a per-pixel ``np.bincount``. ``offset`` (e.g. the
    residual bias) is removed per fibre as ``offset * npix`` rather than from
    every pixel. Fibres with no pixels, or not in the fibre map, are skipped.

    Parameters
    ----------
    fiberimg : 2D int array, the detector's fibre-label image (trace or quick-look cache)
    nfib : number of traced fibres; labels >= nfib are ignored
    benchside : e.g. '2B', for the dead-fibre and fibre-map lookups
    data : 2D bias-subtracted frame, same shape and orientation as fiberimg
    dead_fiber_lut : dead physical fibres per benchside (_load_dead_fiber_lut)
    offset : constant level to subtract from every pixel

    Returns
    -------
    (x, y, flux) : lists, in trace-fibre order
    """
    # Get the sorted list of dead physical fiber indices for this bench
    dead_fibers = sorted(dead_fiber_lut.get(benchside, []))
    if dead_fibers:
        logger.info(f'Bench {benchside}: {nfib} traced fibers, '
                    f'dead physical fibers: {dead_fibers}')

    # Build the trace-index → physical-fiber-number mapping.
    # The trace object has nfibers entries (e.g. 297 for 2B) because dead
    # fibers were never detected during tracing.  We need to re-insert the
    # gaps so that trace index i maps to the correct physical fiber number
    # that the FiberMap LUT expects.
    #
    # Example for 2B (dead fiber 49, nfibers=297):
    #   trace 0-48  → physical 0-48
    #   trace 49-296 → physical 50-297
    trace_to_physical = []
    physical = 0
    dead_set = set(dead_fibers)
    for trace_idx in range(nfib):
        while physical in dead_set:
            physical += 1
        trace_to_physical.append(physical)
        physical += 1

    # Per-fibre pixel counts and sums. fiberimg is -1 off-fibre; labels >= nfib
    # are ignored. Only NaNs are zeroed so an inf still propagates, as np.nansum
    # does; a NaN pixel contributes nothing, so the offset is applied per pixel
    # on that (rare) path.
    labels = fiberimg.ravel()
    values = data.ravel()
    if np.isnan(values).any():
        values = np.where(np.isnan(values), 0.0, values.astype(np.float64) - offset)
        offset = 0.0
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    run_labels = labels[starts]
    run_npix = np.diff(np.r_[starts, labels.size])
    run_sums = np.add.reduceat(values, starts, dtype=np.float64)
    keep = (run_labels >= 0) & (run_labels < nfib)
    npix = np.bincount(run_labels[keep], weights=run_npix[keep], minlength=nfib)
    sums = np.bincount(run_labels[keep], weights=run_sums[keep], minlength=nfib) - offset * npix

    xdata, ydata, flux = [], [], []
    for ifib in range(nfib):
        physical_fiber = trace_to_physical[ifib]
        if npix[ifib] == 0:
            logger.debug(f'Skipping trace fiber {ifib} (physical {physical_fiber}) '
                         f'on bench {benchside}: no pixels in fiberimg')
            continue

        # Map physical fiber number to IFU position
        x, y = FiberMap_LUT(benchside, physical_fiber)
        if x == -1 and y == -1:
            continue  # Skip if fiber mapping not found

        xdata.append(x)
        ydata.append(y)
        flux.append(sums[ifib])

    return xdata, ydata, flux


def _render_whitelight(xdata, ydata, flux, hex_tiles: bool = False, pix_per_unit: int = 10):
    """White-light image from per-fibre fluxes: hexagonal tiles or the shared grid."""
    # Dead fibers are simply absent from the interpolation inputs;
    # LinearNDInterpolator will naturally fill those positions from neighbours.
    if hex_tiles:
        whitelight, _ = hex_tile_image(xdata, ydata, flux, pix_per_unit=pix_per_unit)
    else:
        flux_interpolator = LinearNDInterpolator(list(zip(xdata, ydata)), flux,
                                                 fill_value=np.nan)
        x_grid, y_grid = whitelight_grid()
        whitelight = flux_interpolator(x_grid, y_grid)
    return whitelight


def QuickWhiteLight(trace_list, data_list, metadata=None, ds9plot=False,
                    hex_tiles: bool = False, pix_per_unit: int = 10):
    """
    Generate a white light image by directly summing unmasked fiber values without extraction.
    
    Parameters:
    -----------
    trace_list : list
        A list of TraceLlamas objects containing the fiber trace information.
    data_list : list
        A list of data arrays corresponding to each trace object.
    metadata : list, optional
        Optional metadata for each trace/data pair (unused; kept for compatibility).
    ds9plot : bool, optional
        If True, display the resulting white light image using DS9. Default is False.
    
    Returns:
    --------
    tuple
        A tuple containing:
        - whitelight (numpy.ndarray): The interpolated white light image.
        - xdata (numpy.ndarray): The x-coordinates of the fiber positions.
        - ydata (numpy.ndarray): The y-coordinates of the fiber positions.
        - flux (numpy.ndarray): The flux values for each fiber.
    """
    dead_fiber_lut = _load_dead_fiber_lut()

    xdata, ydata, flux = [], [], []
    for trace_obj, data in zip(trace_list, data_list):
        x, y, f = _detector_fibre_fluxes(trace_obj.fiberimg, trace_obj.nfibers,
                                         f'{trace_obj.bench}{trace_obj.side}', data, dead_fiber_lut)
        xdata.extend(x)
        ydata.extend(y)
        flux.extend(f)
    xdata, ydata, flux = np.array(xdata, dtype=float), np.array(ydata, dtype=float), np.array(flux, dtype=float)

    whitelight = _render_whitelight(xdata, ydata, flux, hex_tiles=hex_tiles, pix_per_unit=pix_per_unit)

    # Optional DS9 plot
    if ds9plot:
        plot_ds9(whitelight)

    return whitelight, xdata, ydata, flux

def compute_residual_background(data, regions=((5, 20), (20, 50), (30, 50))):
    """Compute residual detector background from multiple regions of a bias-subtracted frame.

    Calculates the median of each candidate region, then returns the median
    of those estimates for robustness against signal contamination in any
    single region.

    Parameters:
        data: 2D numpy array (bias-subtracted frame)
        regions: tuple of (start_row, end_row) pairs to sample

    Returns:
        float: robust background estimate
    """
    medians = []
    for r_start, r_end in regions:
        region = data[r_start:r_end, :]
        medians.append(np.median(region))
    return np.median(medians)


def estimate_residual_bias(data, fiberimg, min_distance=20, edge_trim=2, min_pixels=2000):
    """Measure the residual DC bias left after master-bias subtraction.

    Uses the unilluminated rows outside the fibre stack: every row more than
    ``min_distance`` rows below the first or above the last fibre row of
    ``fiberimg`` (both ends), skipping the ``edge_trim`` outermost detector rows.
    This is the row-wise equivalent of the pipeline's edge-DC stripes
    (Bias/biasChecking.build_topbottom_stripe_mask) without its full-frame
    distance transform. Fixed rows such as 5-50 are not used because the fibre
    stack starts as low as row ~11, so they pick up sky.

    The level is a 3-sigma-clipped mean rather than a median: raw and master-bias
    values are integers, so a median is quantised to 1 DN, and 1 DN per pixel
    over the thousands of pixels in each fibre is a visible bench-side step.

    Parameters:
        data: 2D bias-subtracted frame (trimmed/oriented like ``fiberimg``)
        fiberimg: 2D fibre-label image from the trace (-1 = no fibre)
        min_distance: rows to leave between the fibre stack and the clean rows
        edge_trim: outermost detector rows to ignore at top and bottom
        min_pixels: fewer clean pixels than this -> fall back to rows 5-50

    Returns:
        tuple: (level, npix, source), where level is the DC level to subtract
        and source is 'edges', 'placeholder' (constant data; level is that
        constant) or 'rows5-50' (fallback).
    """
    nrows = data.shape[0]
    fibre_rows = np.flatnonzero((fiberimg >= 0).any(axis=1))
    if fibre_rows.size:
        clean = np.r_[edge_trim:max(fibre_rows[0] - min_distance, edge_trim),
                      min(fibre_rows[-1] + min_distance + 1, nrows - edge_trim):nrows - edge_trim]
    else:
        clean = np.arange(edge_trim, nrows - edge_trim)
    vals = data[clean].ravel().astype(np.float64)   # float64 statistics for float32 frames

    if vals.size < min_pixels:
        logger.warning(f"Only {vals.size} clean edge pixels (< {min_pixels}); "
                       f"falling back to rows 5-50 for the residual bias")
        return float(compute_residual_background(data)), 0, 'rows5-50'

    # Constant data = placeholder camera; remove the constant so it stays at zero.
    if np.nanmin(vals) == np.nanmax(vals):
        return float(vals[0]), vals.size, 'placeholder'

    mean, _, _ = sigma_clipped_stats(vals, sigma=3, maxiters=5)
    return float(mean), vals.size, 'edges'


def _detector_id(header):
    """(color, bench, side) of an extension from COLOR/BENCH/SIDE, else CAM_NAME
    (e.g. '1A_Red'); None if neither identifies the detector."""
    if 'COLOR' in header:
        return (str(header['COLOR']).lower(), str(header.get('BENCH', '')),
                str(header.get('SIDE', '')).upper())
    parts = str(header.get('CAM_NAME', '')).split('_')
    if len(parts) >= 2 and len(parts[0]) >= 2:
        return parts[1].lower(), parts[0][0], parts[0][1].upper()
    return None


def _detector_hdus(hdul):
    """{(color, bench, side): HDU} for the image extensions of an open MEF, matched
    by header rather than position (robust to missing or reordered cameras)."""
    hdus = {}
    for hdu in hdul[1:]:
        key = _detector_id(hdu.header)
        if key is not None:
            hdus[key] = hdu
    return hdus


def QuickWhiteLightCube(science_file, bias: str = None, ds9plot: bool = False,
                        outfile: str = None, use_dir: str = None,
                        hex_tiles: bool = False, pix_per_unit: int = 10) -> str:
        """
        Generates a FITS file with quick-look white light images for each color
        straight from a raw science frame, using the master traces in CALIB_DIR.
        Fibre labels come from the quick-look cache
        (mastercalib/LLAMAS_quicklook_fiberimg.npz, see
        Postprocessing/build_quicklook_fiberimg.py) when it matches the master trace
        pickles, and from the pickles otherwise.

        Each detector is bias subtracted with the READ-MDE-matched master bias, then
        its residual DC level (measured in the unilluminated rows outside the fibre
        stack, see estimate_residual_bias) is subtracted in every read mode; the
        per-detector levels are written to the primary header as RB{C}{bench}{side}
        (e.g. RBR1A) and a warning is logged if any exceed 5 DN (stale master bias).
        Detectors whose master bias is a constant placeholder are listed in BIASPH.

        Output extensions, in order: RED, RED_TAB, GREEN, GREEN_TAB, BLUE, BLUE_TAB
        (colours with no data are omitted). Each *_TAB holds the fibre XDATA, YDATA
        and FLUX used to build the image.

        Parameters:
            science_file (str): Raw LLAMAS science MEF.
            bias (str, optional): Master bias to use instead of the READ-MDE default.
            ds9plot (bool, optional): If True, display each white light image in DS9.
            outfile (str, optional): Output file name; default <science>_quickwhitelight.fits.
            use_dir (str, optional): Output directory; default OUTPUT_DIR. Ignored when
                outfile is an absolute path.
            hex_tiles (bool, optional): Render fibres as hexagonal tiles instead of
                interpolating onto the rectangular grid.
            pix_per_unit (int, optional): Hex tile resolution (pixels per fibre-map unit).

        Returns:
            str: The file path of the created quick-look white light FITS file.
        """

        # Validate and create GUI version if needed (preserves original file)
        science_file = validate_for_gui(science_file)

        primary_hdr = fits.getheader(science_file, 0)

        # Determine bias file based on READ-MDE header keyword
        read_mode = primary_hdr.get('READ-MDE', None)
        if read_mode is not None:
            read_mode = read_mode.strip().upper()
            logger.info(f"Detected READ-MDE: {read_mode}")

        # Default fallback is slow_master_bias.fits (standard readout mode)
        default_bias = os.path.join(BIAS_DIR, 'slow_master_bias.fits')

        if bias is not None:
            # User-specified bias file takes priority, but fall back to default if not found
            if os.path.isfile(bias):
                masterbiasfile = bias
            else:
                logger.warning(f"User-specified bias file {bias} not found. Falling back to default bias selection.")
                bias = None  # Fall through to auto-selection below

        if bias is None:
            # Auto-select bias based on READ-MDE
            if read_mode == 'FAST':
                candidate_bias = os.path.join(BIAS_DIR, 'fast_master_bias.fits')
                if os.path.isfile(candidate_bias):
                    masterbiasfile = candidate_bias
                    logger.info(f"Using FAST mode bias: {masterbiasfile}")
                else:
                    logger.warning(f"fast_master_bias.fits not found, falling back to slow_master_bias.fits")
                    masterbiasfile = default_bias
            elif read_mode == 'SLOW':
                candidate_bias = os.path.join(BIAS_DIR, 'slow_master_bias.fits')
                if os.path.isfile(candidate_bias):
                    masterbiasfile = candidate_bias
                    logger.info(f"Using SLOW mode bias: {masterbiasfile}")
                else:
                    raise FileNotFoundError(f"slow_master_bias.fits not found in {BIAS_DIR}")
            else:
                # Unknown or missing READ-MDE, use slow mode as default
                masterbiasfile = default_bias
                if read_mode is None:
                    logger.warning(f"READ-MDE header not found, defaulting to slow_master_bias.fits")
                else:
                    logger.warning(f"Unknown READ-MDE value '{read_mode}', defaulting to slow_master_bias.fits")

        logger.info(f"Bias file is {masterbiasfile}")

        # A bias taken in the other read mode leaves a large 2D pedestal, so prefer
        # the mode-matched master bias when one exists.
        bias_mode = str(fits.getheader(masterbiasfile, 0).get('READ-MDE', '')).strip().upper()
        if read_mode in ('FAST', 'SLOW') and bias_mode and bias_mode != read_mode:
            matched_bias = os.path.join(BIAS_DIR, f'{read_mode.lower()}_master_bias.fits')
            if os.path.isfile(matched_bias):
                logger.warning(f"Bias {masterbiasfile} is {bias_mode} mode but the frame is {read_mode}; "
                               f"using {matched_bias} instead")
                masterbiasfile = matched_bias
            else:
                logger.warning(f"Bias {masterbiasfile} is {bias_mode} mode but the frame is {read_mode}, "
                               f"and no {matched_bias} exists; using it anyway")

        # Validate bias file structure (add placeholders for missing cameras)
        masterbiasfile = validate_for_gui(masterbiasfile)

        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header['COMMENT'] = "Quick White Light Cube created from science file extensions."
        primary_hdu.header['READMODE'] = (read_mode or 'UNKNOWN', 'READ-MDE of the science frame')
        primary_hdu.header['BIASFILE'] = (os.path.basename(masterbiasfile), 'Master bias subtracted')
        primary_hdu.header['RBMETHOD'] = ('edge rows, 3-sigma clipped mean',
                                          'Residual bias (RB*) estimator')
        hdul = fits.HDUList([primary_hdu])

        # Fibre labels: the small quick-look cache if it matches the master trace
        # pickles, otherwise the pickles themselves (~1 s slower in total).
        if quicklook_cache_is_fresh(QUICKLOOK_CACHE):
            fibre_labels = load_quicklook_fiberimg(QUICKLOOK_CACHE)
        else:
            fibre_labels = None
            logger.warning("Quick-look fibre-label cache is missing or out of date; reading the master "
                           "trace pickles instead (slower). Rebuild it with: python -m "
                           "llamas_pyjamas.Postprocessing.build_quicklook_fiberimg --force")

        # Per-colour fibre (x, y, flux), filled one detector at a time so that each
        # frame (and trace object, on the pickle path) is released before the next
        # is read. None means no detector of that colour was processed.
        fibre_fluxes = {color: None for color in QUICKLOOK_COLOR_ORDER}
        dead_fiber_lut = _load_dead_fiber_lut()
        stale_limit = BiasCheckThresholds().max_residual_median
        stale_detectors = []
        placeholder_bias = []

        # Both files are memory-mapped and read one detector at a time as float32
        # (exact for 16-bit data), rather than scaling and copying all 48 HDUs up front.
        science_hdul = fits.open(science_file, do_not_scale_image_data=True)
        bias_hdul = fits.open(masterbiasfile, do_not_scale_image_data=True)
        bias_hdus = _detector_hdus(bias_hdul)

        for ext in science_hdul[1:]:
            detector = _detector_id(ext.header)
            if detector is None or ext.data is None:
                logger.warning(f"Skipping extension {ext.name}: no image or no COLOR/CAM_NAME")
                continue
            color, bench, side = detector
            benchside = f'{bench}{side}'

            if fibre_labels is not None:
                entry = fibre_labels.get(detector)
                if entry is None:
                    logger.info(f"No master trace for {benchside} {color}. Skipping extension.")
                    continue
                fiberimg, nfib = entry['fiberimg'], entry['nfibers']
            else:
                # Accept both shipped forms: LLAMAS_{c}_{b}_{s}_traces.pkl (mastercalib
                # bundle) and LLAMAS_master_{c}_{b}_{s}_traces.pkl (locally generated).
                try:
                    trace_filepath = find_trace_pickle(color, bench, side, CALIB_DIR)
                except FileNotFoundError as exc:
                    logger.info(f"{exc} for {benchside} {color}. Skipping extension.")
                    continue
                with open(trace_filepath, "rb") as f:
                    trace_obj = pickle.load(f)
                fiberimg, nfib = trace_obj.fiberimg, trace_obj.nfibers
                del trace_obj

            # Step 1: full 2D master-bias subtraction, matched by COLOR/BENCH/SIDE
            bias_ext = bias_hdus.get(detector)
            if bias_ext is None or bias_ext.data is None:
                raise ValueError(f"No master bias extension for {benchside} {color} in {masterbiasfile}")
            bias_data = trim_and_orient(bias_ext.data, bias_ext.header)
            if bias_data.min() == bias_data.max():
                placeholder_bias.append(f'{color}{benchside}')
                logger.info(f"Master bias for {benchside} {color} is a constant placeholder; "
                            f"this detector only gets the DC residual correction")
            data = trim_and_orient(ext.data, ext.header)
            data -= bias_data
            del bias_data

            # Step 2 (all read modes): the residual DC level measured in the
            # unilluminated rows outside the fibre stack. This absorbs the drift an
            # out-of-date master bias leaves, which otherwise shows up as bench-side
            # stripes in the white light. It is removed per fibre (offset * npix).
            residual_bias, npix, source = estimate_residual_bias(data, fiberimg)
            logger.info(f"{benchside} {color}: residual bias = {residual_bias:.2f} DN "
                        f"({source}, {npix} px)")
            primary_hdu.header[f'RB{color[:1].upper()}{bench}{side}'] = (
                round(residual_bias, 3), f'Residual bias {benchside} {color} (DN, {source})')
            # A placeholder bias leaves the full pedestal (~1000 DN), which says nothing
            # about staleness; those detectors are reported in BIASPH instead.
            if (source != 'placeholder' and f'{color}{benchside}' not in placeholder_bias
                    and abs(residual_bias) > stale_limit):
                stale_detectors.append((f'{color}{benchside}', residual_bias))

            if color in fibre_fluxes:
                x, y, f = _detector_fibre_fluxes(fiberimg, nfib, benchside, data, dead_fiber_lut,
                                                 offset=residual_bias)
                if fibre_fluxes[color] is None:
                    fibre_fluxes[color] = ([], [], [])
                for acc, vals in zip(fibre_fluxes[color], (x, y, f)):
                    acc.extend(vals)
            del data

        science_hdul.close()
        bias_hdul.close()

        if placeholder_bias:
            primary_hdu.header['BIASPH'] = (','.join(placeholder_bias),
                                            'Constant (placeholder) master-bias extensions')
        if stale_detectors:
            # Name up to three detectors; the full set is in the RB* header keys.
            stale_detectors.sort(key=lambda d: -abs(d[1]))
            if len(stale_detectors) <= 3:
                named = ', '.join(f'{name} ({level:+.1f})' for name, level in stale_detectors)
            else:
                largest = ', '.join(f'{name} {level:+.1f}' for name, level in stale_detectors[:3])
                named = f"{len(stale_detectors)} detectors (largest {largest}; all in RB* keys)"
            logger.warning(f"Residual bias > {stale_limit:g} DN on {named}: "
                           f"{os.path.basename(masterbiasfile)} may be out of date (corrected)")

        # After processing all science_hdul extensions, generate white light images for each color
        whitelight_results = {}
        for col in QUICKLOOK_COLOR_ORDER:
            if fibre_fluxes[col] is None:
                logger.info(f"No data found for {col} color.")
                whitelight_results[col] = (None, None, None, None)
                continue
            xdata, ydata, flux = (np.array(v, dtype=float) for v in fibre_fluxes[col])
            wl = _render_whitelight(xdata, ydata, flux, hex_tiles=hex_tiles, pix_per_unit=pix_per_unit)
            if ds9plot:
                plot_ds9(wl)
            whitelight_results[col] = (wl, xdata, ydata, flux)

        for color in QUICKLOOK_COLOR_ORDER:
            wl, xdata, ydata, flux = whitelight_results[color]
            if wl is None:
                continue
            # Create an image HDU for the white light image
            image_hdu = fits.ImageHDU(data=wl.astype(np.float32), name=color.upper())
            if hex_tiles:
                for _k, _v in hex_header_keys(pix_per_unit).items():
                    image_hdu.header[_k] = _v
            hdul.append(image_hdu)
            
            # Create a binary table HDU with the fiber x, y positions and flux data
            tab_hdu = fits.BinTableHDU.from_columns([
                fits.Column(name='XDATA', format='E', array=np.array(xdata, dtype=np.float32)),
                fits.Column(name='YDATA', format='E', array=np.array(ydata, dtype=np.float32)),
                fits.Column(name='FLUX',  format='E', array=np.array(flux, dtype=np.float32))
            ], name=f'{color.upper()}_TAB')
            hdul.append(tab_hdu)

        # Determine output file name
        if outfile is None:
            filename = os.path.basename(science_file)
            name, ext = os.path.splitext(filename)
            white_light_file = f"{name}_quickwhitelight.fits"
        else:
            white_light_file = outfile
            
        
        # Write the FITS file to disk.
        if use_dir is None:
            use_dir = OUTPUT_DIR

        outpath = os.path.join(use_dir, white_light_file)
        hdul.writeto(outpath, overwrite=True)
        
        print(f'Quick white light cube saved to {outpath}')
        return outpath


def WhiteLightHex(extraction_file, ds9plot=False, median=False, mask=None, 
                 zscale=True, scale_min=None, scale_max=None, colorbar=True, 
                 colormap='viridis', fig=None, ax=None, **kwargs):
    """
    Create a hexagonal grid white light image without interpolation between fibers.
    Each fiber is represented as a discrete hexagon with its measured value.
    
    Parameters
    ----------
    extraction_list : list
        List of ExtractLlamas objects
    metadata : list, optional
        Metadata for each extraction object, by default None
    ds9plot : bool, optional
        If True, display the image with DS9, by default False
    median : bool, optional
        If True, use median instead of mean for combining extractions, by default False
    mask : ndarray, optional
        Mask to apply to the data, by default None
    zscale : bool, optional
        If True, use zscale for display, by default True
    scale_min : float, optional
        Minimum value for display scaling, by default None
    scale_max : float, optional
        Maximum value for display scaling, by default None
    colorbar : bool, optional
        If True, display colorbar, by default True
    colormap : str, optional
        Colormap to use, by default 'viridis'
    fig : matplotlib.figure.Figure, optional
        Figure to plot on, by default None
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None
        
    Returns
    -------
    ndarray
        2D hexagonal grid image
    """


    
    # Initialize data structures
    xdata = np.array([])
    ydata = np.array([])
    flux = np.array([])
    bench_sides = np.array([])


    if isinstance(extraction_file, str):
        print(f'Type is str-> loading file {extraction_file}')
        extraction, _ = ExtractLlamas.loadExtraction(extraction_file)
        logger.info(f'Loaded extraction object from file: {extraction_file}')
    elif isinstance(extraction_obj, ExtractLlamas):
        print(f'Type is ExtractLlamas-> using object')
        extraction = extraction_obj
    else:
        raise TypeError(f"Unexpected type: {type(extraction_obj)}. Must be string or ExtractLlamas object")
    

    extract_obj = ExtractLlamas.loadExtraction(extraction_file)
    extraction_list = extract_obj['extractions']
    metadata = extract_obj['metadata'] if 'metadata' in extract_obj else None
    
    # Process extraction list
    for i, extraction_obj in enumerate(extraction_list):
        meta = metadata[i] if metadata else None
        
        channel = extraction_obj.channel
        side = extraction_obj.side
        counts = extraction_obj.counts
        
        nfib, naxis1 = np.shape(counts)

        # counts is LIVE-indexed (dead fibres absent); map each live row to its
        # physical fibremap position before the LUT lookup (see WhiteLight).
        fibmap_pos = live_fibre_ids(nfib, getattr(extraction_obj, 'dead_fibers', []))

        for ifib in range(nfib):
            benchside = f'{extraction_obj.bench}{extraction_obj.side}'
            fpos = fibmap_pos[ifib]

            try:
                x, y = FiberMap_LUT(benchside, fpos)
                if x == -1 and y == -1:
                    continue  # Skip if fiber mapping not found
            except Exception as e:
                logger.info(f'Fiber {fpos} (live row {ifib}) not found in fiber map for bench {benchside}')
                logger.error(traceback.format_exc())
                continue

            # Get fiber value
            thisflux = np.nansum(counts[ifib])
            if mask is not None and len(mask) > 0:
                thisflux = thisflux * (1 - mask[ifib])
                
            # Store data
            flux = np.append(flux, thisflux)
            xdata = np.append(xdata, x)
            ydata = np.append(ydata, y)
            bench_sides = np.append(bench_sides, benchside)
    
    # Create figure if not provided
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine colormap scaling
    if zscale:
        from astropy.visualization import ZScaleInterval
        interval = ZScaleInterval()
        vmin, vmax = interval.get_limits(flux)
    else:
        vmin = scale_min if scale_min is not None else np.nanmin(flux)
        vmax = scale_max if scale_max is not None else np.nanmax(flux)
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = cm.get_cmap(colormap)
    
    # Compute hexagon size based on fiber spacing (using median distance between adjacent fibers)
    x_sorted = np.sort(np.unique(xdata))
    if len(x_sorted) > 1:
        x_diffs = np.diff(x_sorted)
        hex_size = np.nanmedian(x_diffs) / 1.5  # Adjust to prevent overlap
    else:
        hex_size = 0.5  # Default if we can't compute
    
    # Create a grid to store hexagonal values for DS9 display
    x_range = (np.max(xdata) - np.min(xdata)) + 2*hex_size
    y_range = (np.max(ydata) - np.min(ydata)) + 2*hex_size
    x_min, y_min = np.min(xdata) - hex_size, np.min(ydata) - hex_size
    
    # Create grid with higher resolution for DS9
    grid_scale = 5  # Higher resolution for better hexagon approximation
    hex_grid = np.full(
        (int(y_range * grid_scale) + 1, int(x_range * grid_scale) + 1),
        np.nan
    )
    
    # Plot hexagons for each fiber
    for i in range(len(xdata)):
        x, y = xdata[i], ydata[i]
        value = flux[i]
        
        if np.isnan(value):
            continue
            
        color = cmap(norm(value))
        
        # Create hexagon patch for matplotlib
        hex_patch = RegularPolygon(
            (x, y), 
            numVertices=6, 
            radius=hex_size,
            orientation=np.pi/6,  # 30 degrees rotation
            facecolor=color, 
            edgecolor='black', 
            linewidth=0.5,
            alpha=1.0
        )
        ax.add_patch(hex_patch)
        
        # Fill corresponding area in hex_grid for DS9
        # Convert hexagon vertices to grid coordinates
        for phi in np.linspace(0, 2*np.pi, 60):  # 60 points around hexagon
            hx = x + hex_size * np.cos(phi)
            hy = y + hex_size * np.sin(phi)
            
            # Convert to grid indices
            ix = int((hx - x_min) * grid_scale)
            iy = int((hy - y_min) * grid_scale)
            
            # Check bounds and set value
            if (0 <= ix < hex_grid.shape[1] and 0 <= iy < hex_grid.shape[0]):
                hex_grid[iy, ix] = value
    
    # Fill in the interior of hexagons in grid (simple flood fill)
    from scipy import ndimage
    # Create a binary mask of valid points
    mask = ~np.isnan(hex_grid)
    # Label connected regions
    labels, num = ndimage.label(mask)
    # Fill holes in each labeled region
    for i in range(1, num+1):
        region = labels == i
        values = hex_grid[region]
        if len(values) > 0:
            median_value = np.nanmedian(values)
            # Fill entire region with median value
            hex_grid[region] = median_value
    
    # Set axis limits
    ax.set_xlim(np.min(xdata) - 2*hex_size, np.max(xdata) + 2*hex_size)
    ax.set_ylim(np.min(ydata) - 2*hex_size, np.max(ydata) + 2*hex_size)
    ax.set_aspect('equal')
    
    # Add colorbar if requested
    if colorbar:
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('Flux')
    
    # Set labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Hexagonal Fiber Grid (No Interpolation)')
    
    # Show the plot
    if ds9plot:
        # Display in DS9
        try:
            from llamas_pyjamas.QA import plot_ds9
            plot_ds9(hex_grid, samp=True)
        except Exception as e:
            logger.error(f"Error displaying in DS9: {e}")
            plt.tight_layout()
            plt.show()
    else:
        plt.tight_layout()
        plt.show()
    
    return hex_grid




def plot_spaxelmap(outpath: str)-> None:
    """
    Plots the fiber map for the LLAMAS IFU in arcseconds, showing the hexagonal spaxel layout.
    
    This function generates a plot with hexagonal patches representing each fiber position for 
    different configurations (1A, 2A, 3A, 4A, 1B, 2B, 3B, 4B). The positions are scaled by the 
    spatial pitch of 0.75" to create a spaxel map with physical units. It also includes directional 
    annotations (N for North, E for East) and saves the plot as an image file.
    
    Color coding:
    - Fibers in configuration 1A are plotted in black
    - Fibers in configuration 3A are plotted in red
    - Fibers in configuration 4B are plotted in purple
    - Fibers in configuration 2B are plotted in orange
    - Fibers in configuration 2A are plotted in blue
    - Fibers in configuration 4A are plotted in green
    - Fibers in configuration 3B are plotted in cyan
    - Fibers in configuration 1B are plotted in magenta
    
    The plot is saved as a high-resolution image in the specified path.
    
    Returns:
        None
    """
    import matplotlib.patches as mpatches
    
    # Spatial pitch in arcseconds
    SPAXEL_PITCH = 0.75
    
    # Size of hexagon (radius to vertex)
    HEX_SIZE = 0.4 * SPAXEL_PITCH
    
    # Configuration colors
    colors = {
        '1A': 'black',
        '3A': 'red',
        '4B': 'purple',
        '2B': 'orange',
        '2A': 'blue',
        '4A': 'green',
        '3B': 'cyan',
        '1B': 'magenta'
    }
    
    # 1A - N=298
    # 2A - N=300
    # 3A - all fibers fine (N=298)
    # 4A - all fibers fine (N=300)
    # 4B - all fibers fine (N=298)
    # 3B - all fibers fine (N=300)
    # 2B - fiber 49 (zero index) is broken / dead (N=297 good + 1 dead)
    # 1B - all fibers fine (N=300)
    
    fig, ax = plt.subplots(1, figsize=(12, 8))
    fibernum_a = np.arange(298)
    fibernum_b = np.arange(300)
    
    # Create legend handles
    legend_handles = []
    
    # Plot fibers for configurations with 298 fibers
    for config in ['1A', '3A', '4B', '2B']:
        for fiber in fibernum_a:
            x, y = FiberMap_LUT(config, int(fiber))
            # Convert to arcseconds
            x_arcsec, y_arcsec = x * SPAXEL_PITCH, y * SPAXEL_PITCH
            
            # Create and add hexagon patch
            hex_patch = mpatches.RegularPolygon(
                (x_arcsec, y_arcsec),  # center coordinates
                numVertices=6,         # hexagon
                radius=HEX_SIZE,       # size
                orientation=0,         # flat top
                facecolor='none',      # transparent fill
                edgecolor=colors[config],
                linewidth=0.5,
                alpha=0.7
            )
            ax.add_patch(hex_patch)
        
        # Add to legend
        legend_handles.append(mpatches.Patch(color=colors[config], label=config))
    
    # Plot fibers for configurations with 300 fibers
    for config in ['2A', '4A', '3B', '1B']:
        for fiber in fibernum_b:
            x, y = FiberMap_LUT(config, int(fiber))
            # Convert to arcseconds
            x_arcsec, y_arcsec = x * SPAXEL_PITCH, y * SPAXEL_PITCH
            
            # Create and add hexagon patch
            hex_patch = mpatches.RegularPolygon(
                (x_arcsec, y_arcsec),  # center coordinates
                numVertices=6,         # hexagon
                radius=HEX_SIZE,       # size
                orientation=0,         # flat top
                facecolor='none',      # transparent fill
                edgecolor=colors[config],
                linewidth=0.5,
                alpha=0.7
            )
            ax.add_patch(hex_patch)
        
        # Add to legend
        legend_handles.append(mpatches.Patch(color=colors[config], label=config))
    
    # Adjust axis limits to reflect the new scale
    ax.set_xlim(-2 * SPAXEL_PITCH, 75 * SPAXEL_PITCH)
    ax.set_ylim(-2 * SPAXEL_PITCH, 47 * SPAXEL_PITCH)
    ax.set_aspect('equal')
    
    # Add axis labels
    ax.set_xlabel('Arcseconds')
    ax.set_ylabel('Arcseconds')
    
    # Add legend
    ax.legend(handles=legend_handles, loc='upper right', framealpha=0.7)
    
    # Extension labels - scale positions by spatial pitch
    fs = 8
    ax.text(52.25 * SPAXEL_PITCH, 40.5 * SPAXEL_PITCH, "R-1", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 39.5 * SPAXEL_PITCH, "G-2", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 38.5 * SPAXEL_PITCH, "B-3", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 35.5 * SPAXEL_PITCH, "R-7", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 34.5 * SPAXEL_PITCH, "G-8", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 33.5 * SPAXEL_PITCH, "B-9", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 30.5 * SPAXEL_PITCH, "R-13", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 29.5 * SPAXEL_PITCH, "G-14", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 28.5 * SPAXEL_PITCH, "B-15", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 25.5 * SPAXEL_PITCH, "R-19", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 24.5 * SPAXEL_PITCH, "G-20", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 23.5 * SPAXEL_PITCH, "B-21", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 3.5 * SPAXEL_PITCH, "R-4", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 2.5 * SPAXEL_PITCH, "G-5", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 1.5 * SPAXEL_PITCH, "B-6", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 8.5 * SPAXEL_PITCH, "R-10", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 7.5 * SPAXEL_PITCH, "G-11", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 6.5 * SPAXEL_PITCH, "B-12", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 13.5 * SPAXEL_PITCH, "R-16", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 12.5 * SPAXEL_PITCH, "G-17", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 11.5 * SPAXEL_PITCH, "B-18", fontsize=fs)

    ax.text(52.25 * SPAXEL_PITCH, 18.5 * SPAXEL_PITCH, "R-22", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 17.5 * SPAXEL_PITCH, "G-23", fontsize=fs)
    ax.text(52.25 * SPAXEL_PITCH, 16.5 * SPAXEL_PITCH, "B-24", fontsize=fs)

    # Direction arrows - scale positions by spatial pitch
    ax.annotate('', xy=(73 * SPAXEL_PITCH, 20 * SPAXEL_PITCH), 
                xytext=(60 * SPAXEL_PITCH, 20 * SPAXEL_PITCH),
                arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))

    ax.annotate('', xy=(60 * SPAXEL_PITCH, 25 * SPAXEL_PITCH), 
                xytext=(60 * SPAXEL_PITCH, 20 * SPAXEL_PITCH),
                arrowprops=dict(facecolor='blue', edgecolor='blue', arrowstyle='->', lw=2))
    
    ax.text(71 * SPAXEL_PITCH, 15.5 * SPAXEL_PITCH, "N", fontsize=12)
    ax.text(60 * SPAXEL_PITCH, 27 * SPAXEL_PITCH, "E", fontsize=12)

    # Update title to reflect physical units
    plt.title("LLAMAS IFU Hexagonal Spaxel Map (0.75\"/spaxel)")

    plt.tight_layout()
    fig.savefig(outpath, dpi=600)
    plt.show()
