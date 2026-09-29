"""Unit tests for master-bias camera lookup when the bias file lacks a camera.

The 2026-01-06 FAST master bias has 22 extensions -- blue 1A and 4A are absent.
The lookup used to fall through to an index guess, get a neighbouring camera,
and raise a bare ValueError that failed the trace and crashed bias-first. It
must instead raise BiasCameraMissingError (a BiasNotFoundError), which every
caller already knows how to fall back from.
Runnable with pytest or as a plain script
(`python -m llamas_pyjamas.Tests.test_bias_lookup`).
"""

import numpy as np
import pytest
from astropy.io import fits

import llamas_pyjamas.Trace.traceLlamasMaster as tlm
from llamas_pyjamas.Bias import BiasCameraMissingError, BiasNotFoundError

MISSING = {('1', 'A', 'blue'), ('4', 'A', 'blue')}


def fake_bias(missing=MISSING):
    hdus = [fits.PrimaryHDU()]
    for bench in '1234':
        for side in 'AB':
            for color in ('red', 'green', 'blue'):
                if (bench, side, color) in missing:
                    continue
                h = fits.ImageHDU(np.full((4, 4), 100.0 + int(bench)))
                h.header['COLOR'], h.header['BENCH'], h.header['SIDE'] = color, bench, side
                hdus.append(h)
    return hdus


@pytest.fixture
def bias22(monkeypatch):
    hdus = fake_bias()
    monkeypatch.setattr(tlm, '_load_bias_hdus_cached', lambda path: hdus)
    return hdus


def test_missing_camera_raises_the_specific_error(bias22):
    with pytest.raises(BiasCameraMissingError) as exc:
        tlm._grab_bias_hdu(bench='1', side='A', color='blue', dir='fast.fits')
    assert (exc.value.bench, exc.value.side, exc.value.color) == ('1', 'A', 'blue')


def test_missing_camera_is_a_bias_not_found(bias22):
    # Existing fallbacks catch BiasNotFoundError; the new error must reach them.
    with pytest.raises(BiasNotFoundError):
        tlm._grab_bias_hdu(bench='4', side='A', color='blue', dir='fast.fits')


def test_present_camera_still_matches_by_header(bias22):
    hdu = tlm._grab_bias_hdu(bench='1', side='B', color='blue', dir='fast.fits')
    assert (hdu.header['BENCH'], hdu.header['SIDE'], hdu.header['COLOR']) == ('1', 'B', 'blue')


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
