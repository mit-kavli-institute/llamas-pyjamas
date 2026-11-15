"""Tests for Utils/recipes.py: per-type processing recipes and their header stamps."""

import logging

import pytest
from astropy.io import fits

from llamas_pyjamas.Utils.recipes import (RECIPES, CalibrationRecipeError, Recipe, check_recipe,
                                          get_recipe, recipe_hash, recipe_problem, stamp_recipe)


def test_calibration_recipes_never_clean_cosmic_rays():
    for name in ('lamp_flat', 'twilight', 'arc'):
        assert not RECIPES[name].cosmic_rays
    assert RECIPES['science'].cosmic_rays


def test_hash_is_stable_and_changes_with_any_field():
    r = get_recipe('lamp_flat')
    assert recipe_hash(r) == recipe_hash('lamp_flat') and len(recipe_hash(r)) == 12
    bumped = Recipe(**{**r.__dict__, 'fmt': r.fmt + 1})
    assert recipe_hash(bumped) != recipe_hash(r)


def test_unknown_recipe_names_the_known_ones():
    with pytest.raises(KeyError, match='lamp_flat'):
        get_recipe('dome')


def test_stamped_header_passes_its_own_check():
    h = stamp_recipe(fits.Header(), 'twilight')
    assert h['RECIPE'] == 'twilight' and h['FIBIDS'] == 'physical'
    assert check_recipe(h, 'twilight')


def test_dict_headers_with_comment_tuples_are_read():
    h = stamp_recipe({}, 'arc')
    assert recipe_problem(h, 'arc') is None


@pytest.mark.parametrize('header, expected, match', [
    (fits.Header(), 'lamp_flat', 'no recipe stamp'),
    (stamp_recipe(fits.Header(), 'arc'), 'lamp_flat', "made with recipe 'arc'"),
])
def test_wrong_or_missing_stamp_is_refused(header, expected, match):
    with pytest.raises(CalibrationRecipeError, match=match):
        check_recipe(header, expected, what='pixel_maps.fits')


def test_changed_recipe_is_stale():
    h = stamp_recipe(fits.Header(), 'lamp_flat')
    h['RCPHASH'] = '000000000000'
    assert 'has changed' in recipe_problem(h, 'lamp_flat')


def test_cosmic_ray_cleaned_calibration_is_refused():
    h = stamp_recipe(fits.Header(), 'lamp_flat')
    h['CRCLEAN'] = True
    assert 'CRCLEAN=True' in recipe_problem(h, 'lamp_flat')


def test_non_strict_check_warns_and_returns_false(caplog):
    with caplog.at_level(logging.WARNING):
        assert check_recipe(fits.Header(), 'twilight', strict=False) is False
    assert 'clobber' in caplog.text
