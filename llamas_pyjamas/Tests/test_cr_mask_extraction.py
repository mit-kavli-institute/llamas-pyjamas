"""Cosmic-ray pixels are missing data in the extraction, never L.A.Cosmic's replacement values.

A synthetic single-fibre trace (as in test_horne_masking): a Gaussian profile over 9 rows,
the same in every column. ``cr_mask`` marks cosmic-ray pixels; the frame holds a deliberately
wrong value there (a replacement), which must never reach the extracted counts.
"""

import os
import shutil
import tempfile
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.File.llamasRSS import (MASK_CR_MISSING, MASK_CR_PARTIAL, MASK_NO_COUNTS,
                                           RSSgeneration, bad_pixel_mask)
from llamas_pyjamas.Utils.maskedSums import MIN_VALID_FRAC

NX, NY, ROW0 = 400, 21, 6
PEAK = ROW0 + 4
FLUX = 2000.0
BOGUS = -5000.0                     # what a "replacement" would put in the frame
HDR = {'GAIN': 1.0, 'RDNOISE': 3.0, 'COLOR': 'green', 'BENCH': '1', 'SIDE': 'A'}


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def synthetic_trace():
    rows = np.arange(NY)[:, None]
    profile = np.exp(-0.5 * ((rows - PEAK) / 1.2) ** 2) * np.ones((1, NX))
    in_ap = (rows >= ROW0) & (rows < ROW0 + 9)
    profimg = np.where(in_ap, profile, 0.0)
    profimg /= profimg.sum(axis=0, keepdims=True)
    fiberimg = np.where(in_ap, 0, -1) * np.ones((1, NX), dtype=int)
    return SimpleNamespace(bench='1', side='A', channel='green', nfibers=1, naxis1=NX,
                           naxis2=NY, fiberimg=fiberimg, profimg=profimg,
                           traces=np.full((1, NX), float(PEAK)), fitsfile='synthetic.fits'), profimg


def extract(method, frame, cr_mask, monkeypatch, **kw):
    monkeypatch.setenv('LLAMAS_EXTRACT_METHOD', method)
    monkeypatch.setenv('LLAMAS_BOXCAR_HALFWIDTH', '2.5')
    trace, _ = synthetic_trace()
    return ExtractLlamas(trace, frame, dict(HDR), method=method, cr_mask=cr_mask, **kw)


def _frame_and_mask(rows, col):
    _, profimg = synthetic_trace()
    frame = FLUX * profimg
    cr = np.zeros(frame.shape, bool)
    for r in rows:
        cr[r, col] = True
        frame[r, col] = BOGUS
    return frame, cr


# ---------------------------------------------------------------- boxcar
def test_boxcar_sample_with_a_cosmic_ray_is_missing(monkeypatch):
    frame, cr = _frame_and_mask([PEAK + 1], 50)
    ext = extract('boxcar', frame, cr, monkeypatch)
    assert np.isnan(ext.counts[0, 50]) and np.isnan(ext.counts_err[0, 50])
    assert ext.dq[0, 50] & MASK_CR_MISSING
    assert np.all(np.isfinite(ext.counts[0, :50])) and np.all(np.isfinite(ext.counts[0, 51:]))
    assert np.count_nonzero(ext.dq) == 1


def test_boxcar_ignores_a_cosmic_ray_outside_the_aperture(monkeypatch):
    # half-width 2.5 around a trace centred on row PEAK covers rows PEAK-2 .. PEAK+2; a cosmic
    # ray at PEAK + 4 (towards the next fibre) has zero aperture weight
    _, profimg = synthetic_trace()
    plain = extract('boxcar', FLUX * profimg, None, monkeypatch)
    frame, cr = _frame_and_mask([PEAK + 4], 50)
    ext = extract('boxcar', frame, cr, monkeypatch)
    assert ext.dq[0, 50] == 0
    assert ext.counts[0, 50] == pytest.approx(plain.counts[0, 50])


def test_boxcar_without_cosmic_rays_is_unchanged(monkeypatch):
    _, profimg = synthetic_trace()
    frame = FLUX * profimg
    plain = extract('boxcar', frame, None, monkeypatch)
    empty = extract('boxcar', frame, np.zeros(frame.shape, bool), monkeypatch)
    np.testing.assert_array_equal(plain.counts, empty.counts)
    np.testing.assert_array_equal(plain.counts_err, empty.counts_err)
    assert not plain.dq.any()


# ---------------------------------------------------------------- horne
def test_horne_measures_around_a_wing_cosmic_ray(monkeypatch):
    frame, cr = _frame_and_mask([PEAK + 3], 50)               # a wing pixel: little profile
    clean = extract('horne', synthetic_trace()[1] * FLUX, None, monkeypatch)
    ext = extract('horne', frame, cr, monkeypatch)
    assert ext.counts[0, 50] == pytest.approx(FLUX, rel=1e-9)  # unbiased, bogus value unused
    assert ext.dq[0, 50] == MASK_CR_PARTIAL
    assert ext.counts_err[0, 50] > clean.counts_err[0, 50]     # less information, larger error
    assert ext.counts_err[0, 49] == pytest.approx(clean.counts_err[0, 49])


def test_horne_sample_missing_when_most_of_the_profile_is_lost(monkeypatch):
    frame, cr = _frame_and_mask([PEAK - 1, PEAK, PEAK + 1], 50)   # the core: > half the profile
    _, profimg = synthetic_trace()
    assert profimg[PEAK - 1:PEAK + 2, 50].sum() > 1 - MIN_VALID_FRAC
    ext = extract('horne', frame, cr, monkeypatch)
    assert np.isnan(ext.counts[0, 50]) and np.isnan(ext.counts_err[0, 50])
    assert ext.dq[0, 50] == MASK_CR_MISSING


def test_horne_partial_error_includes_the_dark_term(monkeypatch):
    """The information ratio inflates the whole variance: photon, read and dark noise."""
    frame, cr = _frame_and_mask([PEAK + 3], 50)
    dark = np.full(frame.shape, 50.0)
    hdr_dark = dict(HDR, DARKSCL=1.0)
    monkeypatch.setenv('LLAMAS_EXTRACT_METHOD', 'horne')
    trace, profimg = synthetic_trace()
    plain = ExtractLlamas(trace, FLUX * profimg, dict(hdr_dark), method='horne', dark=dark,
                          dark_noise=0.1)
    part = ExtractLlamas(trace, frame, dict(hdr_dark), method='horne', dark=dark,
                         dark_noise=0.1, cr_mask=cr)
    nodark = ExtractLlamas(trace, frame, dict(hdr_dark), method='horne', cr_mask=cr)
    plain_nodark = ExtractLlamas(trace, FLUX * profimg, dict(hdr_dark), method='horne')
    ratio = (nodark.counts_err[0, 50] / plain_nodark.counts_err[0, 50]) ** 2
    assert ratio > 1
    assert (part.counts_err[0, 50] / plain.counts_err[0, 50]) ** 2 == pytest.approx(ratio, rel=1e-5)


def test_cr_mask_shape_must_match(monkeypatch):
    _, profimg = synthetic_trace()
    with pytest.raises(ValueError):
        extract('boxcar', FLUX * profimg, np.zeros((3, 3), bool), monkeypatch)


# ---------------------------------------------------------------- into the RSS
def test_rss_carries_the_cosmic_ray_bits(tmp, monkeypatch):
    frame, cr = _frame_and_mask([PEAK + 3], 50)                # partial (Horne)
    frame[PEAK - 1:PEAK + 2, 80] = BOGUS
    cr[PEAK - 1:PEAK + 2, 80] = True                           # missing (Horne)
    ext = extract('horne', frame, cr, monkeypatch)
    ext.wave = np.tile(np.linspace(5000, 6000, NX), (1, 1))
    ext.relative_throughput = np.ones(1)
    ext.dead_fibers = []
    hdr = fits.Header()
    hdr['OBS TYPE'] = 'CAL24'
    pkl = os.path.join(tmp, 'x.pkl')
    with open(pkl, 'wb') as fh:
        cloudpickle.dump({'primary_header': hdr, 'extractions': [ext],
                          'metadata': [{'channel': 'green', 'bench': '1', 'side': 'A'}]}, fh)
    out = os.path.join(tmp, 'x_RSS.fits')
    RSSgeneration().generate_rss(pkl, out)
    with fits.open(out) as h:
        m = h['MASK'].data
        assert m[0, 50] & MASK_CR_PARTIAL and not bad_pixel_mask(h)[0, 50]
        assert m[0, 80] & MASK_CR_MISSING and not m[0, 80] & MASK_NO_COUNTS
        assert bad_pixel_mask(h)[0, 80] and np.isnan(h['SKYSUB'].data[0, 80])
        assert h[0].header['CRMISS'] == 1 and h[0].header['CRPART'] == 1
