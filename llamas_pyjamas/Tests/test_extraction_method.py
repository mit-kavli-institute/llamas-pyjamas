"""Every product records how it was extracted (EXTMETH/EXTHW), and nothing mixes methods.

The extraction method changes every per-fibre flux, so a calibration made with one method
must not be used on science made with another: resumed runs refuse or redo such products,
and flux calibration refuses a sensfunc from a standard extracted differently.
"""

import os
import shutil
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.Bias.biasFirstResume import stale_reason
from llamas_pyjamas.Utils.recipes import (LEGACY_EXTMETH, copy_extraction_method,
                                          extraction_method_problem, recipe_problem,
                                          run_extraction_method, stamp_recipe)
from llamas_pyjamas.Tests.rss_fixtures import stage_name, write_rss


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def horne_run(monkeypatch):
    monkeypatch.setenv('LLAMAS_EXTRACT_METHOD', 'horne')


@pytest.fixture
def boxcar_run(monkeypatch):
    monkeypatch.setenv('LLAMAS_EXTRACT_METHOD', 'boxcar')
    monkeypatch.setenv('LLAMAS_BOXCAR_HALFWIDTH', '2.5')


def test_no_run_no_stamp_and_no_check(monkeypatch):
    monkeypatch.delenv('LLAMAS_EXTRACT_METHOD', raising=False)
    assert run_extraction_method() is None
    h = stamp_recipe(fits.Header(), 'twilight')
    assert 'EXTMETH' not in h
    assert recipe_problem(h, 'twilight') is None             # e.g. the GUI quick look


def test_stamps_follow_the_run(horne_run):
    h = stamp_recipe(fits.Header(), 'lamp_flat')
    assert h['EXTMETH'] == 'horne' and 'EXTHW' not in h
    assert recipe_problem(h, 'lamp_flat') is None


def test_boxcar_records_its_width(boxcar_run):
    h = stamp_recipe(fits.Header(), 'twilight')
    assert h['EXTMETH'] == 'boxcar' and h['EXTHW'] == 2.5
    assert extraction_method_problem(h, ('boxcar', 3.0)) is not None
    assert extraction_method_problem(h, ('boxcar', 2.5)) is None


def test_optimal_is_horne():
    h = stamp_recipe(fits.Header(), 'twilight', method='optimal')
    assert h['EXTMETH'] == 'horne'


def test_calibration_from_another_method_is_refused(horne_run):
    boxcar = stamp_recipe(fits.Header(), 'lamp_flat', method='boxcar', halfwidth=2.5)
    assert 'extracted with boxcar' in recipe_problem(boxcar, 'lamp_flat')
    legacy = stamp_recipe(fits.Header(), 'lamp_flat', method='horne')
    del legacy['EXTMETH']                                    # made before EXTMETH existed
    assert 'no EXTMETH stamp' in recipe_problem(legacy, 'lamp_flat')


def test_unstamped_products_count_as_the_old_default(boxcar_run):
    legacy = stamp_recipe(fits.Header(), 'lamp_flat', method='boxcar')
    del legacy['EXTMETH'], legacy['EXTHW']
    assert LEGACY_EXTMETH == 'boxcar'
    assert recipe_problem(legacy, 'lamp_flat') is None


def test_stale_products_lists_a_boxcar_flat_under_a_horne_run(tmp, monkeypatch):
    from llamas_pyjamas.reduce import stale_products
    monkeypatch.setenv('LLAMAS_EXTRACT_METHOD', 'boxcar')
    flat = os.path.join(tmp, 'flat')
    os.makedirs(flat)
    h = fits.PrimaryHDU()
    stamp_recipe(h.header, 'lamp_flat')
    h.writeto(os.path.join(flat, 'pixel_maps.fits'))
    assert stale_products(tmp, flat) == []
    monkeypatch.setenv('LLAMAS_EXTRACT_METHOD', 'horne')
    stale = stale_products(tmp, flat)
    assert len(stale) == 1 and 'extracted with boxcar' in stale[0][1]


def test_copy_into_the_rss_primary():
    src = stamp_recipe(fits.Header(), 'science', method='boxcar', halfwidth=3.0)
    dst = fits.Header()
    copy_extraction_method(src, dst)
    assert dst['EXTMETH'] == 'boxcar' and dst['EXTHW'] == 3.0
    copy_extraction_method(None, dst)                        # no source header: nothing


# ---------------------------------------------------------------- resumed science frames
BIAS = {'BFVER': 2, 'BIASID': 'b1', 'DARKID': 'none', 'DARKSCL': 0.0}


def _write(path, **cards):
    h = fits.Header()
    for k, v in cards.items():
        h[k] = v
    fits.PrimaryHDU(np.zeros(2), header=h).writeto(path, overwrite=True)
    return str(path)


def test_resume_redoes_science_extracted_with_another_method(tmp):
    bc = _write(os.path.join(tmp, 'X_SCI24_mef_bias_corrected.fits'), **BIAS)
    boxcar = _write(os.path.join(tmp, 'X_RSS_red.fits'), **BIAS, CRID='off', EXTMETH='boxcar',
                    EXTHW=2.5)
    legacy = _write(os.path.join(tmp, 'Y_RSS_red.fits'), **BIAS, CRID='off')
    horne = dict(BIAS, CRID='off', EXTMETH='horne', EXTHW=None)
    assert stale_reason(horne, [bc, boxcar]) == 'extraction method changed'
    assert stale_reason(horne, [bc, legacy]) == 'extraction method changed'
    box = dict(BIAS, CRID='off', EXTMETH='boxcar', EXTHW=2.5)
    assert stale_reason(box, [bc, boxcar]) == ''
    assert stale_reason(box, [bc, legacy]) == ''              # legacy = the old boxcar default
    assert stale_reason(dict(box, EXTHW=3.0), [bc, boxcar]) == 'extraction method changed'
    assert stale_reason(dict(BIAS), [bc, boxcar]) == ''       # calibration: nothing compared


# ---------------------------------------------------------------- flux calibration
def _sensfunc(meta):
    from llamas_pyjamas.Flux.sensFunc import SensChannel, SensFunc
    w = np.linspace(4600, 7100, 200)
    chans = {'green': SensChannel(channel='green', wave=w, sens=np.full(200, 1e-16),
                                  raw=np.full(200, 1e-16), good=np.ones(200, bool))}
    return SensFunc(channels=chans, meta=dict(meta))


def test_flux_calibration_refuses_a_sensfunc_from_another_method(tmp):
    from llamas_pyjamas.Flux.fluxCalibrate import SensfuncStageError, flux_calibrate_file
    path = write_rss(stage_name(tmp, 'green', 'base'), header={'EXTMETH': 'horne'})
    base = {'fibrflat': False, 'rssstage': 'base'}
    with pytest.raises(SensfuncStageError):
        flux_calibrate_file(path, _sensfunc(dict(base, extmeth='boxcar')), apply_extinction=False)
    with pytest.raises(SensfuncStageError):                   # an old sensfunc: boxcar
        flux_calibrate_file(path, _sensfunc(base), apply_extinction=False)
    flux_calibrate_file(path, _sensfunc(dict(base, extmeth='horne')), apply_extinction=False)
    with fits.open(path) as h:
        assert 'FLAM' in h
