"""Tests for the celestial WCS builder (Utils/wcsLlamas.py).

Pins the LLAMAS sky-WCS conventions: North up / East left parity, the 0.75"/fibre plate scale,
CRVAL landing on the field centre, rotation applied, and the header pointing extractor
(including the None-not-(0,0) fallback that lets callers keep a linear WCS).

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_wcs_llamas`).
"""

import numpy as np
from astropy.io import fits

from llamas_pyjamas.Utils.wcsLlamas import (
    ARCSEC_PER_FIBRE,
    IFU_PARITY,
    celestial_wcs,
    pointing_from_header,
)


def _wcs(pa_deg=0.0, arcsec_per_pixel=0.075):
    # crpix is 1-indexed; (50, 40) => 0-indexed reference pixel (49, 39)
    return celestial_wcs(180.0, 0.0, crpix=(50.0, 40.0),
                         arcsec_per_pixel=arcsec_per_pixel, pa_deg=pa_deg)


def test_crval_lands_on_crpix():
    w = _wcs()
    c = w.pixel_to_world(49.0, 39.0)          # 0-indexed reference pixel
    assert np.isclose(c.ra.deg, 180.0, atol=1e-9)
    assert np.isclose(c.dec.deg, 0.0, atol=1e-9)


def test_north_up_east_left_at_pa0():
    w = _wcs(pa_deg=0.0)
    ref = w.pixel_to_world(49.0, 39.0)
    plus_x = w.pixel_to_world(50.0, 39.0)
    plus_y = w.pixel_to_world(49.0, 40.0)
    # East is to the LEFT: increasing pixel x must DECREASE RA
    assert plus_x.ra.deg < ref.ra.deg, 'East-left: +x should decrease RA'
    # North is UP: increasing pixel y must INCREASE DEC
    assert plus_y.dec.deg > ref.dec.deg, 'North-up: +y should increase DEC'
    assert IFU_PARITY == -1


def test_plate_scale_isotropic_and_correct():
    w = _wcs(arcsec_per_pixel=0.075)
    ref = w.pixel_to_world(49.0, 39.0)
    assert np.isclose(ref.separation(w.pixel_to_world(50.0, 39.0)).arcsec, 0.075, atol=1e-4)
    assert np.isclose(ref.separation(w.pixel_to_world(49.0, 40.0)).arcsec, 0.075, atol=1e-4)
    # 10 image pixels == one fibre spacing == 0.75" (row-to-row 0.65" follows from the
    # fibre-map y-coordinates carrying the sqrt(3)/2 compression, not from the scale).
    assert np.isclose(ref.separation(w.pixel_to_world(59.0, 39.0)).arcsec,
                      10 * 0.075, atol=1e-3)
    assert ARCSEC_PER_FIBRE == 0.75


def test_rotation_changes_orientation():
    ref0 = _wcs(pa_deg=0.0).pixel_to_world(50.0, 39.0)
    ref90 = _wcs(pa_deg=90.0).pixel_to_world(50.0, 39.0)
    # a 90 deg field rotation must move where +x points on the sky
    assert not (np.isclose(ref0.ra.deg, ref90.ra.deg) and
                np.isclose(ref0.dec.deg, ref90.dec.deg))


def test_pointing_from_header_decimal_and_pa():
    h = fits.Header()
    h['RA'] = 243.2592
    h['DEC'] = 8.1349
    h['TEL PA'] = 30.0
    h['TEL ROT'] = 212.0
    ra, dec, pa = pointing_from_header(h)
    assert np.isclose(ra, 243.2592) and np.isclose(dec, 8.1349)
    assert pa == 30.0, 'TEL PA is preferred over TEL ROT'


def test_pointing_from_header_rot_fallback_and_tel_radec():
    # no decimal RA/DEC, no TEL PA -> sexagesimal TEL RA/DEC and TEL ROT
    h = fits.Header()
    h['TEL RA'] = '16:13:02.2'
    h['TEL DEC'] = '+08:08:05'
    h['TEL ROT'] = 212.0
    ra, dec, pa = pointing_from_header(h)
    assert 243.0 < ra < 243.5 and 8.0 < dec < 8.3
    assert pa == 212.0


def test_pointing_from_header_absent_is_none_not_zero():
    ra, dec, pa = pointing_from_header(fits.Header())
    assert ra is None and dec is None, 'absent pointing must be None so callers can fall back'
    assert pa == 0.0
    assert pointing_from_header(None) == (None, None, 0.0)


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
