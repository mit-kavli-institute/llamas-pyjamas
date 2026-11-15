"""Tests for Utils/recipes.py: per-type processing recipes and their header stamps."""

import logging
import os

import numpy as np
import pytest
from astropy.io import fits

import llamas_pyjamas.GUI.guiExtract as ge
from llamas_pyjamas.reduce import (apply_flat_field_correction, cleanup_extraction_pkls,
                                   recipe_matrix, stale_products)
from llamas_pyjamas.Utils.recipes import (RECIPES, CalibrationRecipeError, Recipe, check_recipe,
                                          cosmic_rays_for, get_recipe, read_recipe_sidecar,
                                          recipe_for_frame, recipe_hash, recipe_problem,
                                          sidecar_path, stamp_recipe, write_recipe_sidecar)


def test_calibration_recipes_never_clean_cosmic_rays():
    for name in ('lamp_flat', 'twilight', 'arc'):
        assert not RECIPES[name].cosmic_rays
    assert RECIPES['science'].cosmic_rays


def test_science_like_recipes_are_at_the_cosmic_ray_mask_format():
    # fmt 2: cosmic-ray pixels are missing data in the extraction
    for name in ('science', 'standard', 'sky_frame'):
        assert RECIPES[name].fmt == 2


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


# --- B4: enforcement -------------------------------------------------------------------

def test_every_extraction_names_its_recipe():
    with pytest.raises(TypeError, match='needs a recipe'):
        cosmic_rays_for(None)
    with pytest.raises(ValueError, match='forbids'):
        cosmic_rays_for('arc', True)
    assert cosmic_rays_for('science') is True and cosmic_rays_for('science', False) is False
    assert cosmic_rays_for('lamp_flat') is False and cosmic_rays_for('twilight', False) is False


def test_extraction_entry_points_refuse_a_missing_recipe():
    with pytest.raises(TypeError, match='needs a recipe'):
        ge.process_trace._function(np.zeros((4, 4)), {}, 'x.pkl', 1)
    with pytest.raises(TypeError, match='needs a recipe'):
        ge.GUI_extract('frame.fits')


@pytest.mark.parametrize('cards, expected', [
    ({'OBJECT': 'LDLS flat'}, 'lamp_flat'),
    ({'OBJECT': 'SKYFLAT'}, 'twilight'),
    ({'OBJECT': 'ThAr', 'FRONTEND LAMPS THAR STATE': 'ON'}, 'arc'),
    ({'OBJECT': 'SPT-CLJ0111-5637', 'FRONTEND LAMPS THAR STATE': 'OFF'}, 'science'),
])
def test_quick_look_picks_the_recipe_from_the_header(cards, expected):
    h = fits.Header()
    for k, v in cards.items():
        h[k] = v
    assert recipe_for_frame(h) == expected


def test_sidecar_round_trip_and_cleanup(tmp_path):
    pkl = str(tmp_path / 'x_extract.pkl')
    open(pkl, 'w').close()
    h = stamp_recipe(fits.Header(), 'arc')
    h['CRCLEAN'] = False
    write_recipe_sidecar(pkl, h)
    assert recipe_problem(read_recipe_sidecar(pkl), 'arc') is None
    assert cleanup_extraction_pkls([pkl], save_pkl=False) == [pkl]
    assert not os.path.exists(sidecar_path(pkl))


def test_pixel_flat_without_its_recipe_is_refused(tmp_path):
    old = str(tmp_path / 'pixel_maps.fits')
    fits.PrimaryHDU().writeto(old)
    with pytest.raises(CalibrationRecipeError, match='pixel_maps.fits'):
        apply_flat_field_correction(str(tmp_path / 'science.fits'), [old], str(tmp_path))


def test_arcs_are_checked_on_resume_and_in_the_matrix(tmp_path):
    arcs = tmp_path / 'arcs'
    arcs.mkdir()
    good, old = str(arcs / 'a_extract.pkl'), str(arcs / 'b_extract.pkl')
    for p in (good, old):
        open(p, 'w').close()
    write_recipe_sidecar(good, stamp_recipe(fits.Header(), 'arc'))
    ext = tmp_path / 'extractions'
    (ext / 'flat').mkdir(parents=True)
    stale = dict(stale_products(str(ext), str(ext / 'flat'), arc_dir=str(arcs)))
    assert set(stale) == {old}
    rows = {name: why for name, _, why in recipe_matrix(str(tmp_path), str(ext), str(ext / 'flat'))}
    assert rows['a_extract.pkl'] is None and 'no recipe stamp' in rows['b_extract.pkl']
