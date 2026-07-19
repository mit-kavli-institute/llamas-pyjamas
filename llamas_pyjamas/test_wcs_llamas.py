"""Tests for the LLAMAS celestial WCS (Utils/wcsLlamas.py).

Pins the calibrated convention (mirrored field, det(CD)>0; rotation = header PA + IFU_PA_OFFSET;
0.75"/fibre) and the two solve paths -- the rough header builder and the astrometric fit /
translation-only registration. Exact catalog values are not asserted; geometry is.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_wcs_llamas`).
"""

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from llamas_pyjamas.Utils.wcsLlamas import (
    ARCSEC_PER_FIBRE,
    IFU_MIRRORED,
    IFU_PA_OFFSET,
    celestial_wcs,
    fit_wcs_from_stars,
    pointing_from_header,
    register_pointing,
)


def _wcs(pa_deg=0.0, arcsec_per_pixel=0.075, **kw):
    return celestial_wcs(180.0, 0.0, crpix=(50.0, 40.0),
                         arcsec_per_pixel=arcsec_per_pixel, pa_deg=pa_deg, **kw)


def _det(w):
    return np.linalg.det(w.pixel_scale_matrix)


def test_crval_lands_on_crpix():
    c = _wcs().pixel_to_world(49.0, 39.0)          # 0-indexed reference pixel
    assert np.isclose(c.ra.deg, 180.0, atol=1e-9) and np.isclose(c.dec.deg, 0.0, atol=1e-9)


def test_mirror_parity_default_and_override():
    assert IFU_MIRRORED is True
    assert _det(_wcs()) > 0, 'default LLAMAS field is mirrored (det>0)'
    assert _det(_wcs(mirrored=False)) < 0, 'mirrored=False gives a standard det<0 sky'


def test_plate_scale_isotropic_and_correct():
    w = _wcs(arcsec_per_pixel=0.075)
    ref = w.pixel_to_world(49.0, 39.0)
    assert np.isclose(ref.separation(w.pixel_to_world(50.0, 39.0)).arcsec, 0.075, atol=1e-4)
    assert np.isclose(ref.separation(w.pixel_to_world(49.0, 40.0)).arcsec, 0.075, atol=1e-4)
    # 10 px == one fibre spacing == 0.75"
    assert np.isclose(ref.separation(w.pixel_to_world(59.0, 39.0)).arcsec, 0.75, atol=1e-3)
    assert ARCSEC_PER_FIBRE == 0.75


def _axis_pa(w):
    c0 = w.pixel_to_world(49.0, 39.0)
    return c0.position_angle(w.pixel_to_world(50.0, 39.0)).deg


def test_pa_x_tracks_telrot_with_slope_plus_one():
    # Calibrated: PA(+x) = TEL_ROT + IFU_PA_OFFSET (slope +1). Guards against the sign flip that
    # made it slope -1 (right near 212 deg, ~90 deg off by 265). Check across a wide TEL_ROT span.
    for tel_rot in (0.0, 90.0, 212.0, 265.0, 330.0):
        got = _axis_pa(_wcs(pa_deg=tel_rot, pa_offset=0.0))
        assert np.isclose(((got - tel_rot + 180) % 360) - 180, 0.0, atol=0.5), tel_rot


def test_pa_offset_shifts_pa_x():
    pa0 = _axis_pa(_wcs(pa_deg=100.0, pa_offset=0.0))
    pa30 = _axis_pa(_wcs(pa_deg=100.0, pa_offset=30.0))
    assert np.isclose(((pa30 - pa0 + 180) % 360) - 180, 30.0, atol=0.5)
    # header PA and offset are interchangeable (both feed PA(+x) with slope +1)
    a = _axis_pa(_wcs(pa_deg=40.0, pa_offset=0.0))
    b = _axis_pa(_wcs(pa_deg=0.0, pa_offset=40.0))
    assert np.isclose(((a - b + 180) % 360) - 180, 0.0, atol=1e-6)


def test_pointing_from_header_decimal_rot_and_none():
    h = fits.Header()
    h['RA'] = 243.2592
    h['DEC'] = 8.1349
    h['TEL PA'] = 30.0
    h['TEL ROT'] = 212.0
    ra, dec, pa = pointing_from_header(h)
    assert np.isclose(ra, 243.2592) and np.isclose(dec, 8.1349) and pa == 30.0
    # sexagesimal fallback + TEL ROT when no decimal / no TEL PA
    h2 = fits.Header()
    h2['TEL RA'] = '16:13:02.2'
    h2['TEL DEC'] = '+08:08:05'
    h2['TEL ROT'] = 212.0
    ra2, dec2, pa2 = pointing_from_header(h2)
    assert 243.0 < ra2 < 243.5 and 8.0 < dec2 < 8.3 and pa2 == 212.0
    # absent -> None (so callers can skip the celestial WCS)
    assert pointing_from_header(fits.Header()) == (None, None, 0.0)
    assert pointing_from_header(None) == (None, None, 0.0)


def test_fit_wcs_from_two_stars_reproduces_them():
    # build a known WCS, project two stars to pixels, then recover it from the pairs
    truth = celestial_wcs(150.0, -20.0, crpix=(100.0, 100.0), arcsec_per_pixel=0.075, pa_deg=37.0)
    px = [(60.0, 55.0), (170.0, 130.0)]
    sky = [truth.pixel_to_world(x, y) for x, y in px]
    w = fit_wcs_from_stars(px, sky)                 # mirrored default matches truth (det>0)
    assert np.linalg.det(w.pixel_scale_matrix) > 0
    for (x, y), s in zip(px, sky):
        assert w.pixel_to_world(x, y).separation(s).arcsec < 1e-4


def test_register_pointing_fixes_translation_only():
    truth = celestial_wcs(150.0, -20.0, crpix=(100.0, 100.0), arcsec_per_pixel=0.075, pa_deg=37.0)
    px = [(60.0, 55.0), (170.0, 130.0)]
    sky = [truth.pixel_to_world(x, y) for x, y in px]
    # a rough WCS with the right orientation but a ~5" wrong pointing
    rough = celestial_wcs(150.0 + 5 / 3600.0, -20.0 - 4 / 3600.0, crpix=(100.0, 100.0),
                          arcsec_per_pixel=0.075, pa_deg=37.0)
    before = rough.pixel_to_world(*px[0]).separation(sky[0]).arcsec
    reg = register_pointing(rough, px, sky)
    after = reg.pixel_to_world(*px[0]).separation(sky[0]).arcsec
    assert before > 3.0 and after < 0.2, 'pointing shift brings stars onto catalog'
    # CD (rotation+scale) untouched -- only CRVAL moved
    assert np.allclose(reg.pixel_scale_matrix, rough.pixel_scale_matrix)


def test_whitelight_wcs_header_uses_convention():
    from llamas_pyjamas.Image.WhiteLightModule import _whitelight_wcs_header
    x = np.array([0.0, 46.0, 23.0])
    y = np.array([0.0, 44.0, 22.0])
    h = fits.Header()
    h['RA'] = 100.0
    h['DEC'] = -20.0
    cards = _whitelight_wcs_header(x, y, h, hex_tiles=True, pix_per_unit=10)
    assert cards['CTYPE1'] == 'RA---TAN' and cards['CTYPE2'] == 'DEC--TAN'
    assert np.isclose(cards['CRVAL1'], 100.0) and np.isclose(cards['CRVAL2'], -20.0)
    assert _whitelight_wcs_header(x, y, fits.Header(), hex_tiles=True, pix_per_unit=10) is None


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
