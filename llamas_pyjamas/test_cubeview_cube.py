"""Tests for the cube scene (CubeViewer/cubeViewCube.py).

Headless coverage of the SpectralScene the CubeViewer wraps around a combined cube: white-light
collapse, DS9-pixel -> spaxel mapping (respecting coverage), the per-spaxel spectrum, aperture
elements_within, region strings, and a FITS round-trip. The DS9/Qt display is verified live.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_cubeview_cube`).
"""

import os
import tempfile

import numpy as np
from astropy.wcs import WCS

from llamas_pyjamas.Combine.cube import CoaddCube
from llamas_pyjamas.CubeViewer.cubeViewCube import CoaddCubeScene


def _cube(nw=5, ny=10, nx=10, channel='green', wave0=5000.0, peak=9.0):
    wave = wave0 + np.arange(nw)
    data = np.zeros((nw, ny, nx), float)
    data[:, 4, 4] = np.array([1, 2, peak, 2, 1], float)   # a line at spaxel (iy=4, ix=4)
    var = np.ones((nw, ny, nx))
    coverage = np.full((ny, nx), 3, int)
    coverage[0, 0] = 0                                     # one uncovered corner
    nexp = np.full((ny, nx), 2, int)
    w = WCS(naxis=3)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
    w.wcs.crval = [150.0, 20.0, wave0]
    w.wcs.crpix = [5.0, 5.0, 1.0]
    w.wcs.cdelt = [-0.5 / 3600, 0.5 / 3600, 1.0]
    w.wcs.cunit = ['deg', 'deg', 'Angstrom']
    return CoaddCube(data=data, var=var, wave=wave, coverage=coverage, nexp=nexp, wcs=w,
                     bunit='erg/s/cm2/Angstrom/arcsec2',
                     meta={'CHANNEL': channel, 'FIELD': 'T', 'PIXSCALE': 0.5})


def test_scene_basics_and_channels():
    sc = CoaddCubeScene(_cube())
    assert sc.channels == ('green',) and sc.has_flam is True
    lo, hi = sc.wavelength_range()
    assert lo == 5000.0 and hi == 5004.0


def test_collapse_returns_image_and_celestial_wcs():
    sc = CoaddCubeScene(_cube())
    img, wcs, meta = sc.collapse(5000, 5004)
    assert img.shape == (10, 10)
    assert wcs.naxis == 2 and wcs.wcs.ctype[0].startswith('RA')
    assert np.isclose(img[4, 4], np.mean([1, 2, 9, 2, 1]))    # white-light = mean over lambda


def test_element_at_maps_ds9_pixel_and_respects_coverage():
    sc = CoaddCubeScene(_cube())
    assert sc.element_at(5, 5) == (4, 4)                       # DS9 1-based pixel 5 -> index 4
    assert sc.element_at(1, 1) is None                         # uncovered corner (coverage 0)
    assert sc.element_at(100, 100) is None                     # off grid


def test_spectrum_at_spaxel():
    sc = CoaddCubeScene(_cube())
    sp = sc.spectra_at(5, 5)
    assert len(sp) == 1
    assert np.allclose(sp[0].flux, [1, 2, 9, 2, 1]) and sp[0].channel == 'green'
    assert sp[0].has_flam                                      # calibrated cube


def test_elements_within_and_region():
    sc = CoaddCubeScene(_cube())
    one = sc.elements_within(5, 5, 0)
    assert one == [(4, 4)]
    many = sc.elements_within(5, 5, 1.5)
    assert (4, 4) in many and len(many) > 1
    assert 'box(5,5' in sc.region_for([(4, 4)])


def test_multichannel_scene_shows_all_channels():
    # green + red cubes on the same spatial grid -> a spaxel returns both spectra (full coverage)
    cubes = {'green': _cube(channel='green', wave0=5000.0, peak=9.0),
             'red': _cube(channel='red', wave0=6000.0, peak=5.0)}
    sc = CoaddCubeScene(cubes)
    assert sc.channels == ('green', 'red')
    sp = sc.spectra_at(5, 5)
    assert len(sp) == 2 and {s.channel for s in sp} == {'green', 'red'}
    assert np.isclose(max(s.flux.max() for s in sp if s.channel == 'green'), 9.0)
    # white-light collapse over the union spans both channels' contributions
    lo, hi = sc.wavelength_range()
    assert lo <= 5000.0 and hi >= 6004.0
    img, wcs, meta = sc.collapse(lo, hi)
    assert set(meta['contributions']) == {'green', 'red'}


def test_calibrated_cube_has_no_counts_plane():
    sc = CoaddCubeScene(_cube())                      # bunit 'erg/.../arcsec2' -> calibrated
    sp = sc.spectra_at(5, 5)[0]
    assert sp.has_flam and sp.has_counts is False     # panel greys out Counts for a FLAM cube

    cube = _cube()
    cube.bunit = 'counts/arcsec2'                      # instrumental cube -> counts, no flam
    sp2 = CoaddCubeScene(cube).spectra_at(5, 5)[0]
    assert sp2.has_counts is True and not sp2.has_flam


def test_from_fits_roundtrip():
    cube = _cube()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'F_cube_green.fits')
        cube.write(p)
        sc = CoaddCubeScene.from_fits(p)
    assert sc.cube.data.shape == cube.data.shape
    assert np.allclose(sc.cube.wave, cube.wave)
    assert np.allclose(sc.spectra_at(5, 5)[0].flux, [1, 2, 9, 2, 1])


def test_from_fits_loads_all_channel_siblings():
    with tempfile.TemporaryDirectory() as d:
        for c, w0 in (('green', 5000.0), ('red', 6000.0)):
            _cube(channel=c, wave0=w0).write(os.path.join(d, f'J_cube_{c}.fits'))
        sc = CoaddCubeScene.from_fits(os.path.join(d, 'J_cube_green.fits'))   # open one
    assert sc.channels == ('green', 'red')               # both siblings loaded
    assert len(sc.spectra_at(5, 5)) == 2                 # full spectrum across channels


def _point_cube(ny=11, nx=11, nw=6, sigma_pix=1.5):
    yy, xx = np.mgrid[0:ny, 0:nx]
    prof = np.exp(-0.5 * ((xx - 5) ** 2 + (yy - 5) ** 2) / sigma_pix ** 2)
    data = prof[None, :, :] * np.full(nw, 3.0)[:, None, None]
    w = WCS(naxis=3)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN', 'WAVE']
    w.wcs.crval = [150.0, 20.0, 5000.0]; w.wcs.crpix = [6, 6, 1]
    w.wcs.cdelt = [-0.5 / 3600, 0.5 / 3600, 1.0]; w.wcs.cunit = ['deg', 'deg', 'Angstrom']
    return CoaddCube(data, np.ones((nw, ny, nx)), 5000.0 + np.arange(nw),
                     np.full((ny, nx), 3, int), np.full((ny, nx), 2, int), w,
                     'erg/s/cm2/Angstrom/arcsec2', {'FIELD': 'T', 'CHANNEL': 'green', 'PIXSCALE': 0.5})


def test_cube_space_optimal_extraction_without_super_rss():
    sc = CoaddCubeScene(_point_cube())                   # no super-RSS -> cube-space path
    assert sc.super_rss is None
    sky = sc.cube.wcs.celestial.pixel_to_world(5, 5)     # source centre
    spectra, fit = sc.optimal_spectrum(float(sky.ra.deg), float(sky.dec.deg), radius_arcsec=3.0)
    assert fit.fitted and len(spectra) == 1
    assert abs(fit.fwhm - 2.3548 * 1.5 * 0.5) < 0.5      # ~sigma 1.5px * 0.5"/px
    assert np.all(np.isfinite(spectra[0].flux))


if __name__ == '__main__':
    import sys
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f'PASS {name}')
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
