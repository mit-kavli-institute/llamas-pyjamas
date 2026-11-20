"""Horne extraction with masked pixels: unbiased flux, and errors that grow by the right amount.

A synthetic single-fibre trace: a Gaussian profile over 9 rows, the same in every column, so
the columns are independent samples of one measurement.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas

NX, NY, ROW0 = 3000, 21, 6
FLUX = 2000.0


def synthetic_trace():
    rows = np.arange(NY)[:, None]
    profile = np.exp(-0.5 * ((rows - (ROW0 + 4)) / 1.2) ** 2) * np.ones((1, NX))
    in_ap = (rows >= ROW0) & (rows < ROW0 + 9)
    profimg = np.where(in_ap, profile, 0.0)
    profimg /= profimg.sum(axis=0, keepdims=True)
    fiberimg = np.where(in_ap, 0, -1) * np.ones((1, NX), dtype=int)
    return SimpleNamespace(bench='1', side='A', channel='green', nfibers=1, naxis1=NX,
                           naxis2=NY, fiberimg=fiberimg, profimg=profimg,
                           traces=np.full((1, NX), ROW0 + 4.0), fitsfile='synthetic.fits'), profimg


def extract(frame, monkeypatch):
    monkeypatch.setenv('LLAMAS_EXTRACT_METHOD', 'horne')
    trace, _ = synthetic_trace()
    hdr = {'GAIN': 1.0, 'RDNOISE': 3.0, 'COLOR': 'green', 'BENCH': '1', 'SIDE': 'A'}
    ext = ExtractLlamas(trace, frame, hdr, method='horne')
    return ext.counts[0], ext.counts_err[0]


@pytest.mark.parametrize('masked_share', [0.0, 0.2, 0.4])
def test_masked_pixels_do_not_bias_the_flux(monkeypatch, masked_share):
    _, profimg = synthetic_trace()
    frame = FLUX * profimg
    rng = np.random.default_rng(0)
    rows, cols = np.nonzero(profimg > 0)
    pick = rng.random(rows.size) < masked_share
    frame[rows[pick], cols[pick]] = np.nan
    counts, _ = extract(frame, monkeypatch)
    measured = np.any(np.isfinite(frame) & (profimg > 0), axis=0)   # a fully masked column is 0
    np.testing.assert_allclose(counts[measured], FLUX, rtol=1e-9)
    assert np.all(counts[~measured] == 0)


def test_error_grows_with_the_information_masked(monkeypatch):
    _, profimg = synthetic_trace()
    rng = np.random.default_rng(1)
    noise_sigma = np.sqrt(3.0 ** 2 + FLUX * profimg)
    frame = FLUX * profimg + rng.normal(0, 1, profimg.shape) * noise_sigma
    clean, clean_err = extract(frame.copy(), monkeypatch)
    peak = ROW0 + 4
    masked_frame = frame.copy()
    masked_frame[peak, :] = np.nan                      # the brightest pixel of every column
    masked, masked_err = extract(masked_frame, monkeypatch)

    assert abs(np.mean(masked) - FLUX) < 4 * np.std(masked) / np.sqrt(NX)    # unbiased
    measured = np.std(masked) / np.std(clean)
    reported = np.median(masked_err) / np.median(clean_err)
    assert measured > 1.1                                # masking the peak costs information
    assert reported == pytest.approx(measured, rel=0.1)
