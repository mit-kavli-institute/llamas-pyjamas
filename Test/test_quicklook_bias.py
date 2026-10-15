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
    level, npix, source, _ = estimate_residual_bias(_frame(fiberimg, offset), fiberimg)
    assert source == 'edges'
    assert npix > 0
    # Sub-DN precision despite integer data; the sky in the fibres must not leak in.
    assert level == pytest.approx(offset, abs=0.1)


def test_fibres_touching_one_edge_uses_other_edge():
    fiberimg = _fiberimg(0, 1990)
    level, npix, source, _ = estimate_residual_bias(_frame(fiberimg, 5.2), fiberimg)
    assert source == 'edges'
    last_fibre_row = np.flatnonzero((fiberimg >= 0).any(axis=1))[-1]
    assert npix == (2048 - 2 - (last_fibre_row + 20 + 1)) * 2048
    assert level == pytest.approx(5.2, abs=0.1)


def test_placeholder_is_zeroed():
    fiberimg = _fiberimg(40, 2000)
    data = np.full(SHAPE, -1.0)
    level, _, source, _ = estimate_residual_bias(data, fiberimg)
    assert source == 'placeholder'
    assert np.all(data - level == 0)


def test_reports_top_bottom_gradient():
    fiberimg = _fiberimg(200, 1850)
    data = _frame(fiberimg, 0.0)
    data[:200] -= 4.0          # all bottom clean rows sit 4 DN below the top ones
    level, _, source, gradient = estimate_residual_bias(data, fiberimg)
    assert source == 'edges'
    assert gradient == pytest.approx(4.0, abs=0.15)
    assert -4.0 < level < 0.0


def test_falls_back_without_clean_rows():
    fiberimg = _fiberimg(0, 2047)
    level, npix, source, _ = estimate_residual_bias(_frame(fiberimg, 3.0), fiberimg)
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


# ---------------------------------------------------------------------------
# Profile-weighted fibre sums, cosmic-ray clip and the display products
# ---------------------------------------------------------------------------

from llamas_pyjamas.Image.WhiteLightModule import (   # noqa: E402
    _resolve_master_dark, bench_sky_subtract, clip_cosmic_rays, combine_white, dark_extension_ok,
    robust_scatter)
from llamas_pyjamas.Postprocessing.build_quicklook_fiberimg import normalised_profile  # noqa: E402


def test_weighted_fibre_sums_match_horne_formula():
    nfib, offset, benchside = 20, 2.5, '2A'
    fiberimg = _tilted_fiberimg(nfib)
    rng = np.random.default_rng(3)
    # A profile that peaks in the middle row of each band, then normalised per column.
    rows = np.indices(fiberimg.shape)[0].astype(float)
    profile = np.where(fiberimg >= 0, np.exp(-0.5 * ((rows % 8) - 2.0) ** 2), 0.0)
    weights = normalised_profile(profile, fiberimg, nfib)
    data = (50.0 * profile + rng.normal(offset, 1.0, fiberimg.shape)).astype(np.float32)

    x, y, flux, index = _detector_fibre_fluxes(fiberimg, nfib, benchside, data, {},
                                               offset=offset, weights=weights, return_index=True)
    x0, y0, plain = _detector_fibre_fluxes(fiberimg, nfib, benchside, data, {}, offset=offset)

    assert x == x0 and y == y0 and len(index) == len(flux)
    for ifib, f in zip(index, flux):
        expected = 0.0
        for col in range(fiberimg.shape[1]):
            sel = fiberimg[:, col] == ifib
            w = weights[sel, col].astype(np.float64)
            if w.any():
                expected += np.sum(w * (data[sel, col].astype(np.float64) - offset)) / np.sum(w * w)
        assert f == pytest.approx(expected, rel=1e-9)
    # For data that follows the profile the weighted estimate recovers the plain sum.
    np.testing.assert_allclose(flux, plain, rtol=0.05)
    # The cache supplies the per-column denominator precomputed: same result.
    from llamas_pyjamas.Postprocessing.build_quicklook_fiberimg import profile_column_norm
    pden = profile_column_norm(weights, fiberimg, nfib)
    _, _, flux2 = _detector_fibre_fluxes(fiberimg, nfib, benchside, data, {}, offset=offset,
                                         weights=(weights, pden))
    np.testing.assert_allclose(flux2, flux, rtol=1e-5)


def test_weighted_sums_ignore_nan_and_skip_zero_weight_fibres():
    nfib = 20
    fiberimg = _tilted_fiberimg(nfib)
    weights = np.where(fiberimg >= 0, 0.2, 0.0).astype(np.float32)
    weights[fiberimg == 7] = 0.0                   # no profile for fibre 7
    data = np.full(fiberimg.shape, 10.0, dtype=np.float32)
    data[fiberimg == 2] = np.nan                   # a NaN fibre contributes nothing
    _, _, flux, index = _detector_fibre_fluxes(fiberimg, nfib, '2A', data, {}, weights=weights,
                                               return_index=True)
    assert 7 not in index
    assert flux[index.index(2)] == 0.0
    assert flux[index.index(1)] == pytest.approx(10.0 * np.sum(fiberimg == 1))


def test_clip_cosmic_rays_removes_hits_not_sky_lines():
    rng = np.random.default_rng(4)
    data = rng.normal(100.0, 3.5, (64, 256)).astype(np.float32)
    data[:, 100:103] += 2000.0                     # a sky line: survives
    data[10, 40] += 500.0                          # a cosmic ray: clipped
    data[30, 200:202] += 300.0
    clipped, count = clip_cosmic_rays(data, read_noise=3.5)
    assert count == 3
    assert clipped[10, 40] < 130 and clipped[30, 200] < 130
    assert clipped[:, 101].min() > 2000


def test_bench_sky_subtract_is_per_group():
    values = np.array([100, 110, 105, 300,   200, 210, 205, 190])
    groups = np.array(['1A'] * 4 + ['1B'] * 4)
    skysub, sky_by_group, global_sky = bench_sky_subtract(values, groups, percentile=30.0)
    assert sky_by_group['1A'] == pytest.approx(np.percentile(values[:4], 30))
    assert sky_by_group['1B'] == pytest.approx(np.percentile(values[4:], 30))
    assert global_sky == pytest.approx(np.median(list(sky_by_group.values())))
    np.testing.assert_allclose(skysub[:4], values[:4] - sky_by_group['1A'])
    np.testing.assert_allclose(skysub[4:], values[4:] - sky_by_group['1B'])


def test_robust_scatter_ignores_sources():
    rng = np.random.default_rng(5)
    sky = rng.normal(0.0, 0.02, 500)
    with_sources = np.r_[sky, np.full(20, 5.0)]
    assert robust_scatter(with_sources) == pytest.approx(0.02, rel=0.25)
    assert np.isnan(robust_scatter(np.arange(5.0)))


def test_combine_white_sums_snr_maps():
    red = np.full((4, 4), 1.0)
    green = np.full((4, 4), 3.0)
    green[0, 0] = np.nan
    white, used = combine_white({'red': red, 'green': green, 'blue': None})
    assert used == ['red', 'green']
    assert white[1, 1] == pytest.approx((1.0 + 3.0) / np.sqrt(2))
    assert white[0, 0] == pytest.approx(1.0)        # only red is finite there
    assert combine_white({'red': None}) == (None, [])


def test_resolve_master_dark_scales_and_checks_mode(tmp_path):
    dark = tmp_path / 'slow_master_dark.fits'
    hdu = fits.PrimaryHDU()
    hdu.header['DARKEXPT'] = 600.0
    hdu.header['READ-MDE'] = 'SLOW'
    fits.HDUList([hdu, fits.ImageHDU(np.zeros((4, 4), dtype=np.float32))]).writeto(dark)

    path, scale, dark_exptime = _resolve_master_dark(str(dark), True, 'SLOW', 1800.0)
    assert path == str(dark) and scale == pytest.approx(3.0) and dark_exptime == 600.0
    assert _resolve_master_dark(str(dark), True, 'FAST', 1800.0) == (None, 0.0, None)   # mode mismatch
    assert _resolve_master_dark(str(dark), True, 'SLOW', None) == (None, 0.0, None)     # no exposure time
    assert _resolve_master_dark(str(dark), False, 'SLOW', 1800.0) == (None, 0.0, None)  # disabled


def test_dark_extension_sanity():
    rng = np.random.default_rng(6)
    good = rng.normal(12.0, 4.0, (256, 256))
    negative = rng.normal(-8.0, 4.0, (256, 256))
    noisy = rng.normal(5.0, 30.0, (256, 256))
    assert dark_extension_ok(good, read_noise=3.5)[0]
    assert not dark_extension_ok(negative, read_noise=3.5)[0]
    assert not dark_extension_ok(noisy, read_noise=3.5)[0]
