"""Tests for Flat/flatCombine.py: finding and combining a lamp-flat set."""

import os

import numpy as np
from astropy.io import fits

from llamas_pyjamas.Flat.flatCombine import (combine_lamp_flats, combined_is_current,
                                             find_lamp_flat_set, resolve_lamp_flat_sets)
from llamas_pyjamas.Utils.recipes import check_recipe

SETTING = {'OBJECT': 'LDLS flat', 'REXPTIME': 0.07, 'READ-MDE': 'FAST',
           'DATE-OBS': '2026-10-15', 'FRONTEND SELECTOR STATE': 'LDLS',
           'FRONTEND FILTER2 STATE': 'ONE'}


def raw(path, utc, **changes):
    h = fits.Header()
    for k, v in {**SETTING, 'UTC': utc, **changes}.items():
        h[k] = v
    fits.PrimaryHDU(header=h).writeto(path)
    return str(path)


def test_set_is_the_contiguous_run_with_the_same_setting(tmp_path):
    a = raw(tmp_path / 'LLAMAS_a_CAL24_mef.fits', '16:34:12')
    b = raw(tmp_path / 'LLAMAS_b_CAL24_mef.fits', '16:34:31')
    c = raw(tmp_path / 'LLAMAS_c_CAL24_mef.fits', '16:34:54')
    raw(tmp_path / 'LLAMAS_d_CAL24_mef.fits', '16:35:20', **{'FRONTEND FILTER2 STATE': 'FOUR'})
    raw(tmp_path / 'LLAMAS_e_CAL24_mef.fits', '17:20:00')          # same setting, 45 min later
    raw(tmp_path / 'LLAMAS_f_CAL24_mef.fits', '16:34:40', REXPTIME=0.15)
    assert find_lamp_flat_set(b) == [os.path.abspath(p) for p in (a, b, c)]


def test_listed_set_and_combine_off(tmp_path):
    a = raw(tmp_path / 'LLAMAS_a_CAL24_mef.fits', '16:34:12')
    b = raw(tmp_path / 'LLAMAS_b_CAL24_mef.fits', '16:34:31')
    config = {'green_flat_file': [a, b], 'blue_flat_file': a, 'lamp_flat_combine': False}
    sets = resolve_lamp_flat_sets(config)
    assert sets == {'green': [a, b], 'blue': [a]}
    assert config['green_flat_file'] == a          # single-path consumers keep working


def corrected(path, level, cosmic=None, reverse=False):
    """A bias-corrected MEF with cameras red 1A and green 1A (order reversed if asked)."""
    hdus = [fits.PrimaryHDU()]
    hdus[0].header['BIASSUB'] = True
    cams = [('red', '1', 'A'), ('green', '1', 'A')]
    for color, bench, side in (cams[::-1] if reverse else cams):
        data = np.full((50, 60), 3000.0 if color == 'red' else 6000.0) * level
        data[:, :5] = 0.0                                   # unlit columns
        if cosmic and color == 'green':
            data[cosmic] = 60000.0
        h = fits.ImageHDU(data.astype(np.float32))
        h.header.update({'COLOR': color, 'BENCH': bench, 'SIDE': side, 'BIASSUB': True})
        hdus.append(h)
    fits.HDUList(hdus).writeto(path)
    return str(path)


def test_combine_matches_cameras_by_header_scales_and_rejects_a_cosmic_ray(tmp_path):
    paths = [corrected(tmp_path / 'a.fits', 1.00),
             corrected(tmp_path / 'b.fits', 1.05, cosmic=(20, 30), reverse=True),
             corrected(tmp_path / 'c.fits', 0.95)]
    names = ['LLAMAS_a_mef.fits', 'LLAMAS_b_mef.fits', 'LLAMAS_c_mef.fits']
    out = combine_lamp_flats(paths, str(tmp_path / 'x3.fits'), raw_names=names)
    with fits.open(out) as h:
        assert check_recipe(h[0].header, 'lamp_flat')
        assert h[0].header['NCOMBINE'] == 3 and h[0].header['FLATIN2'] == 'LLAMAS_b_mef.fits'
        green = next(x for x in h[1:] if x.header['COLOR'] == 'green')
        assert green.header['BIASSUB'] and green.header['COMBMETH'] == 'median'
        assert green.header['FLSCALES'].startswith('1.0000,0.952')
        np.testing.assert_allclose(green.data[:, 5:], 6000.0, rtol=1e-5)   # no cosmic ray
    assert combined_is_current(out, names)
    assert not combined_is_current(out, names[:2])
