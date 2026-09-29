"""Tests for the quick-look white light: residual-bias estimator, run-length fibre
sums and the streamed trim/orient reader."""

import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.File.llamasIO import process_fits_by_color, trim_and_orient
from llamas_pyjamas.Image.WhiteLightModule import (
    FiberMap_LUT, _detector_fibre_fluxes, estimate_residual_bias)

SHAPE = (2048, 2048)


def _fiberimg(first_row, last_row, pitch=8, width=5):
    """Fibre-label image with horizontal fibres every `pitch` rows between first_row and last_row."""
    fiberimg = np.full(SHAPE, -1, dtype=np.int16)
    for ifib, row in enumerate(range(first_row, last_row - width + 2, pitch)):
        fiberimg[row:row + width] = ifib
    return fiberimg


def _frame(fiberimg, offset, sky=400.0, seed=0):
    """Integer-valued frame: bias noise + offset everywhere, plus bright sky in fibres."""
    rng = np.random.default_rng(seed)
    data = np.round(rng.normal(offset, 4.0, SHAPE))
    data[fiberimg >= 0] += sky
    return data


@pytest.mark.parametrize("offset", [0.0, 7.4, -3.6])
def test_recovers_offset_with_sky_in_fibres(offset):
    fiberimg = _fiberimg(40, 2000)
    level, npix, source = estimate_residual_bias(_frame(fiberimg, offset), fiberimg)
    assert source == 'edges'
    assert npix > 0
    # Sub-DN precision despite integer data; the sky in the fibres must not leak in.
    assert level == pytest.approx(offset, abs=0.1)


def test_fibres_touching_one_edge_uses_other_edge():
    fiberimg = _fiberimg(0, 1990)
    level, npix, source = estimate_residual_bias(_frame(fiberimg, 5.2), fiberimg)
    assert source == 'edges'
    last_fibre_row = np.flatnonzero((fiberimg >= 0).any(axis=1))[-1]
    assert npix == (2048 - 2 - (last_fibre_row + 20 + 1)) * 2048
    assert level == pytest.approx(5.2, abs=0.1)


def test_placeholder_is_zeroed():
    fiberimg = _fiberimg(40, 2000)
    data = np.full(SHAPE, -1.0)
    level, _, source = estimate_residual_bias(data, fiberimg)
    assert source == 'placeholder'
    assert np.all(data - level == 0)


def test_falls_back_without_clean_rows():
    fiberimg = _fiberimg(0, 2047)
    level, npix, source = estimate_residual_bias(_frame(fiberimg, 3.0), fiberimg)
    assert source == 'rows5-50'
    assert npix == 0
    assert np.isfinite(level)


def _tilted_fiberimg(nfib, pitch=8, width=5, shape=(200, 300)):
    """Tilted fibre bands (several label runs per row), two extra labels >= nfib,
    and fibre 4 with no pixels."""
    rows, cols = np.indices(shape)
    pos = rows - 10 - 0.02 * cols
    fibre = np.floor(pos / pitch).astype(int)
    fiberimg = np.where((pos >= 0) & (pos % pitch < width) & (fibre < nfib + 2), fibre, -1)
    fiberimg[fiberimg == 4] = -1
    return fiberimg.astype(np.int16)


@pytest.mark.parametrize("with_nan", [False, True])
def test_run_length_fibre_sums_match_nansum(with_nan):
    nfib, offset, benchside = 20, 3.7, '2A'
    dead_lut = {benchside: [3]}
    fiberimg = _tilted_fiberimg(nfib)
    rng = np.random.default_rng(1)
    data = rng.normal(100, 5, fiberimg.shape).astype(np.float32)
    if with_nan:
        data[rng.random(data.shape) < 0.01] = np.nan

    x, y, flux = _detector_fibre_fluxes(fiberimg, nfib, benchside, data, dead_lut, offset=offset)

    # Direct per-fibre nansum; trace fibre i >= 3 is physical fibre i + 1 (dead fibre 3).
    exp_x, exp_y, exp_flux = [], [], []
    for ifib in range(nfib):
        in_fibre = fiberimg == ifib
        if not in_fibre.any():
            continue
        fx, fy = FiberMap_LUT(benchside, ifib if ifib < 3 else ifib + 1)
        if fx == -1 and fy == -1:
            continue
        exp_x.append(fx)
        exp_y.append(fy)
        exp_flux.append(np.nansum(data[in_fibre].astype(np.float64) - offset))

    assert len(flux) == nfib - 1          # fibre 4 has no pixels
    assert x == exp_x and y == exp_y
    np.testing.assert_allclose(flux, exp_flux, rtol=1e-12)


def test_trim_and_orient_matches_process_fits_by_color(tmp_path):
    rng = np.random.default_rng(2)
    hdus = [fits.PrimaryHDU()]
    for color in ('red', 'green', 'blue'):
        hdu = fits.ImageHDU(rng.integers(0, 65535, (12, 24), dtype=np.uint16))
        hdu.header.update(COLOR=color, BENCH=1, SIDE='A', DATASEC='[1:20, 1:10]')
        hdus.append(hdu)
    path = tmp_path / 'mef.fits'
    fits.HDUList(hdus).writeto(path)

    expected, _ = process_fits_by_color(str(path), write=False)
    with fits.open(path, do_not_scale_image_data=True) as hdul:
        for ext, ref in zip(hdul[1:], expected[1:]):
            got = trim_and_orient(ext.data, ext.header)
            assert got.dtype == np.float32 and got.shape == (10, 20)
            np.testing.assert_array_equal(got, ref.data.astype(np.float32))
