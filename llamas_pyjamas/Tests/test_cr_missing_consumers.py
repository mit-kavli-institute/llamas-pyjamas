"""Missing samples (NaN, e.g. cosmic-ray gaps) are left out, never counted as zero.

Pins Utils/maskedSums.row_total and interp_missing, and the consumers that run on extraction
pickles before the RSS exists: the sky-fibre selection in skyModel_1d (one NaN used to drop a
fibre, so whole cameras fell back to the channel-global sky) and the white-light image.
"""

import os
import pickle
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.Image.WhiteLightModule import WhiteLight
from llamas_pyjamas.Sky.skyLlamas import SKY_FALLBACK, SKY_NONE, skyModel_1d
from llamas_pyjamas.Utils.maskedSums import interp_missing, row_total


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- helpers
def test_row_total_equals_sum_without_missing_samples():
    v = np.arange(12, dtype=float).reshape(3, 4)
    np.testing.assert_allclose(row_total(v), v.sum(axis=1))


def test_row_total_leaves_missing_samples_out():
    v = np.array([[2.0, 2.0, np.nan, 2.0], [np.nan] * 4])
    t = row_total(v)
    assert t[0] == pytest.approx(8.0)                      # not 6 (nansum)
    assert np.isnan(t[1])                                  # nothing measured


def test_interp_missing_fills_gaps_for_detection_only():
    s = np.array([np.nan, 1.0, 2.0, np.nan, 4.0, np.nan])
    out = interp_missing(s)
    np.testing.assert_allclose(out, [1.0, 1.0, 2.0, 3.0, 4.0, 4.0])
    assert np.isnan(s[0])                                  # input untouched
    np.testing.assert_array_equal(interp_missing(np.full(3, np.nan)), np.zeros(3))


# ---------------------------------------------------------------- sky selection
def _sky_spectrum(x):
    s = np.full(x.shape, 100.0)
    for c, a in ((80, 900.0), (210, 400.0), (330, 1500.0), (470, 600.0)):
        s += a * np.exp(-0.5 * ((x - c) / 1.3) ** 2)
    return s


def _camera_pickle(path, nan_samples, seed=1, nfib=40, nx=600):
    """One green camera of sky-dominated fibres (a few bright 'objects'); every fibre loses
    1-3 samples to NaN when ``nan_samples``."""
    rng = np.random.default_rng(seed)
    # each fibre samples the sky at its own sub-pixel offset, as real fibres do
    xshift = np.arange(nx, dtype=float)[None, :] + rng.uniform(-0.5, 0.5, nfib)[:, None]
    tp = rng.uniform(0.85, 1.15, nfib)
    counts = tp[:, None] * _sky_spectrum(xshift) + rng.normal(0, 1.0, (nfib, nx))
    counts[:5] += 300.0                                    # bright fibres, not sky
    if nan_samples:
        for f in range(nfib):
            counts[f, rng.choice(nx, rng.integers(1, 4), replace=False)] = np.nan
    obj = SimpleNamespace(channel='green', bench='1', side='A',
                          trace=SimpleNamespace(nfibers=nfib, traces=None),
                          counts=counts, xshift=xshift,
                          wave=np.tile(np.linspace(5000, 5600, nx), (nfib, 1)),
                          relative_throughput=tp, sky=np.zeros((nfib, nx)), dead_fibers=[])
    with open(path, 'wb') as fh:
        pickle.dump({'primary_header': fits.Header(), 'extractions': [obj],
                     'metadata': [{'channel': 'green', 'bench': '1', 'side': 'A',
                                   'nfibers': nfib}]}, fh)
    return path


def _sky_of(path):
    out = skyModel_1d(path, 'green')
    obj = ExtractLlamas.loadExtraction(out)['extractions'][0]
    return obj


def test_sky_model_survives_missing_samples(tmp):
    """Every fibre with 1-3 NaN samples: the camera still gets its own sky model (no
    channel-global fallback), finite, and equal to the model without the NaN."""
    ctrl = _sky_of(_camera_pickle(os.path.join(tmp, 'ctrl_extractions.pkl'), False))
    gaps = _sky_of(_camera_pickle(os.path.join(tmp, 'gaps_extractions.pkl'), True))
    q = getattr(gaps, 'sky_quality', None)
    assert q is None or not np.any(np.isin(q, (SKY_FALLBACK, SKY_NONE)))
    model = gaps.sky / gaps.relative_throughput[:, None]
    ref = ctrl.sky / ctrl.relative_throughput[:, None]
    assert np.all(np.isfinite(model))
    np.testing.assert_allclose(model, ref, rtol=0.01, atol=1.0)


# ---------------------------------------------------------------- white light
def test_white_light_leaves_missing_samples_out():
    nfib, nx = 4, 50
    ext = ExtractLlamas(None, None, None)
    ext.bench, ext.side, ext.channel = '1', 'A', 'green'
    ext.dead_fibers = []
    ext.counts = np.full((nfib, nx), 2.0)
    ext.counts[1, 7] = np.nan
    _, _, _, flux = WhiteLight([ext], [{'channel': 'green', 'side': 'A'}], ds9plot=False,
                               hex_tiles=True)
    np.testing.assert_allclose(flux, 2.0 * nx)             # fibre 1 not 2 * (nx - 1)
