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

from llamas_pyjamas.Flat.fibreFlat import compute_fibre_flat_twilight, _load_smooth_models
from llamas_pyjamas.Flat.flatLlamas import smooth_model_table, _write_smooth_models_fits
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
