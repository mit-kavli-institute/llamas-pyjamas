"""Tests for the fibre-flat fixes of 2026-11-14 (plan steps A0-A4).

A synthetic benchside 2B (dead fibre 49, as in LUT/traceLUT.json) is used throughout:
extractions are live-indexed, so above fibre 49 a row number and a physical fibre id
differ by one.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.File.llamasRSS import RSSStageError
from llamas_pyjamas.Flat.fibreFlat import (apply_fibre_flat_to_rss, compute_fibre_flat_twilight,
                                           _load_smooth_models)
from llamas_pyjamas.Flat.flatLlamas import smooth_model_table, _write_smooth_models_fits
from llamas_pyjamas.Flux.fluxCalibrate import SensfuncStageError, flux_calibrate_file
from llamas_pyjamas.Flux.sensFunc import SensChannel, SensFunc
from llamas_pyjamas.Tests.rss_fixtures import N_WAVE as RSS_NWAVE, write_rss
from llamas_pyjamas.Utils.deadfibers import live_fibre_ids, live_row_index
from llamas_pyjamas.Utils.recipes import CalibrationRecipeError

N_WAVE = 400
N_LIVE = 297
DEAD = [49]
WAVE = np.linspace(4700.0, 7000.0, N_WAVE)


def throughput_of(fid):
    """A distinct, known throughput for every physical fibre."""
    return 1.0 + 0.3 * np.sin(1.7 * fid)


def lamp_extraction():
    """Live-indexed lamp extraction of green 2B, its smooth models keyed by row."""
    physical = live_fibre_ids(N_LIVE, DEAD)
    counts = np.ones((N_LIVE, N_WAVE))
    ext = SimpleNamespace(counts=counts, wave=np.tile(WAVE, (N_LIVE, 1)), dead_fibers=DEAD)
    smooth = {row: np.full(N_WAVE, 1000.0) for row in range(N_LIVE)}
    return ext, smooth, physical


def twilight_extraction():
    """Live-indexed twilight of green 2B whose row k holds fibre live_fibre_ids()[k]."""
    physical = live_fibre_ids(N_LIVE, DEAD)
    counts = np.array([np.full(N_WAVE, 5000.0 * throughput_of(p)) for p in physical])
    ext = SimpleNamespace(counts=counts, wave=np.tile(WAVE, (N_LIVE, 1)), dead_fibers=DEAD)
    return {'extractions': [ext], 'metadata': [{'channel': 'green', 'bench': '2', 'side': 'B'}],
            'primary_header': fits.Header()}


def write_models(tmp_path):
    ext, smooth, _ = lamp_extraction()
    ids, sm, wv = smooth_model_table(ext, smooth)
    path = str(tmp_path / 'flat_smooth_models.fits')
    _write_smooth_models_fits({'green_2_B': {'fiber_ids': ids, 'smooth': sm, 'wave': wv,
                                            'channel': 'green', 'bench': '2', 'side': 'B'}},
                              path)
    return path


# --- A0: physical fibre ids ---------------------------------------------------------

def test_live_row_index_inverts_live_fibre_ids():
    rows = live_row_index(N_LIVE, DEAD)
    assert 49 not in rows and rows[48] == 48 and rows[50] == 49 and rows[297] == 296
    assert [live_fibre_ids(N_LIVE, DEAD)[r] for r in rows.values()] == list(rows)


def test_smooth_model_ids_are_physical():
    ext, smooth, physical = lamp_extraction()
    ids, sm, wv = smooth_model_table(ext, smooth)
    assert list(ids) == physical            # 0..48, 50..297
    assert 49 not in ids and ids[-1] == 297
    assert sm.shape == wv.shape == (N_LIVE, N_WAVE)


def test_rows_without_a_model_stay_aligned():
    ext, smooth, physical = lamp_extraction()
    del smooth[60]                          # row 60 is physical fibre 61
    ids, sm, _ = smooth_model_table(ext, smooth)
    assert 61 not in ids and len(ids) == len(sm) == N_LIVE - 1


def test_each_fibre_gets_its_own_twilight_throughput(tmp_path):
    corr = compute_fibre_flat_twilight(twilight_extraction(), write_models(tmp_path),
                                       str(tmp_path))
    with fits.open(corr) as h:
        assert h[0].header['FIBIDS'] == 'physical'
        ids = np.asarray(h['green_2_B'].data['FIBER_ID'])
        ti = np.nanmedian(h['green_2_B'].data['CORRECTION'], axis=1)
    truth = throughput_of(ids)
    assert np.corrcoef(ti, truth)[0, 1] > 0.98
    above = ids > 49
    # the neighbour's throughput (the old off-by-one) is a much worse match
    assert np.corrcoef(ti[above][:-1], throughput_of(ids[above][1:]))[0, 1] < 0.5


def test_row_labelled_smooth_models_are_refused(tmp_path):
    path = write_models(tmp_path)
    with fits.open(path, mode='update') as h:
        del h[0].header['FIBIDS']
    with pytest.raises(CalibrationRecipeError, match='extraction row'):
        _load_smooth_models(path)


# --- A1: throughput applied once ------------------------------------------------------

def _rss_divided_by(tmp_path, true_tp, rss_tp):
    """Base RSS whose raw sky is true_tp x sky, already divided by rss_tp (as the RSS
    writer divides by relative_throughput), with rss_tp recorded in THRUPUT."""
    path = write_rss(str(tmp_path / 'x_RSS_green.fits'), thruput=rss_tp)
    sky = 300.0 + 80.0 * np.sin(np.linspace(0, 6, RSS_NWAVE))
    with fits.open(path, mode='update') as h:
        raw = true_tp[:, None] * sky
        h['COUNTS'].data[:] = raw / rss_tp[:, None]
        h['SKY'].data[:] = 0.0
        h['SKYSUB'].data[:] = raw / rss_tp[:, None]
        h['ERROR'].data[:] = 0.5
    return path, sky


def _corrections(tmp_path, true_tp):
    hdus = [fits.PrimaryHDU()]
    hdus[0].header['METHOD'] = 'twilight'
    hdus[0].header['FIBIDS'] = 'physical'
    for k, bs in enumerate(('1A', '1B')):
        cols = [fits.Column(name='FIBER_ID', format='J', array=np.arange(30)),
                fits.Column(name='CORRECTION', format=f'{RSS_NWAVE}D',
                            array=np.tile(true_tp[30 * k:30 * (k + 1), None], RSS_NWAVE)),
                fits.Column(name='WAVE', format=f'{RSS_NWAVE}D',
                            array=np.tile(np.linspace(4700.0, 7000.0, RSS_NWAVE), (30, 1))),
                fits.Column(name='FLAGS', format='I', array=np.zeros(30, dtype=np.int16))]
        h = fits.BinTableHDU.from_columns(cols)
        h.header['EXTNAME'] = f'green_{bs[0]}_{bs[1]}'
        hdus.append(h)
    path = str(tmp_path / 'fibre_flat_corrections.fits')
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_flat_divides_out_the_throughput_once(tmp_path):
    rng = np.random.default_rng(1)
    true_tp = rng.uniform(0.75, 1.25, 60)
    rss_tp = true_tp * rng.uniform(0.98, 1.02, 60)       # the RSS's own, slightly different
    base, sky = _rss_divided_by(tmp_path, true_tp, rss_tp)
    out = apply_fibre_flat_to_rss(base, _corrections(tmp_path, true_tp),
                                  str(tmp_path / 'x_RSS_green_FF.fits'))
    with fits.open(out) as h:
        flat = h['SKYSUB'].data
        assert h[0].header['FFVER'] == 2
        assert h[0].header['FFSA1A'] < 0.5            # flat sky after the flat
    # every fibre is back to the sky itself: the throughput was divided out exactly once
    np.testing.assert_allclose(flat, np.tile(sky, (60, 1)), rtol=1e-5)


def test_rss_without_thruput_is_refused(tmp_path):
    true_tp = np.ones(60)
    base, _ = _rss_divided_by(tmp_path, true_tp, true_tp)
    with fits.open(base) as h:
        cols = [c for c in h['FIBERMAP'].columns if c.name != 'THRUPUT']
        h['FIBERMAP'] = fits.BinTableHDU.from_columns(cols, name='FIBERMAP')
        h.writeto(base, overwrite=True)
    with pytest.raises(RSSStageError, match='THRUPUT'):
        apply_fibre_flat_to_rss(base, _corrections(tmp_path, true_tp),
                                str(tmp_path / 'x_RSS_green_FF.fits'))


@pytest.mark.parametrize('sens_ffver, ok', [(2, True), (1, False)])
def test_flux_calibration_needs_the_same_ffver(tmp_path, sens_ffver, ok):
    path = write_rss(str(tmp_path / 'x_RSS_green_FF.fits'), stage='ff', header={'FFVER': 2})
    w = np.linspace(4600, 7100, 200)
    sens = SensFunc(channels={'green': SensChannel('green', w, np.full(200, 1e-16),
                                                   np.full(200, 1e-16), np.ones(200, bool))},
                    meta={'fibrflat': True, 'rssstage': 'ff', 'ffver': sens_ffver})
    if ok:
        flux_calibrate_file(path, sens, apply_extinction=False)
    else:
        with pytest.raises(SensfuncStageError, match='FFVER'):
            flux_calibrate_file(path, sens, apply_extinction=False)
