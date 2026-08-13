"""Unit tests for llamas_pyjamas.Sky.skySelect (sky-fibre selection)."""

import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.Sky import skySelect


# ---------------------------------------------------------------------------
# select_sky_fibres
# ---------------------------------------------------------------------------
def test_dimmest_picks_n_faintest():
    rng = np.random.default_rng(0)
    brightness = rng.uniform(10, 1000, size=30)
    finite = np.ones(30, dtype=bool)
    mask = skySelect.select_sky_fibres(brightness, finite, method="dimmest", n_fibres=20)
    assert mask.sum() == 20
    expected = set(np.argsort(brightness)[:20])
    assert set(np.where(mask)[0]) == expected


def test_dimmest_excludes_nonfinite():
    brightness = np.arange(10, dtype=float)
    brightness[0] = np.nan          # faintest-looking but not usable
    finite = np.isfinite(brightness)
    mask = skySelect.select_sky_fibres(brightness, finite, method="dimmest", n_fibres=3)
    assert not mask[0]
    assert set(np.where(mask)[0]) == {1, 2, 3}


def test_middle_third_parity():
    # Reproduce the legacy skyModel_1d selection: central third of the
    # brightness-descending order.
    rng = np.random.default_rng(1)
    brightness = rng.uniform(0, 1, size=30)
    finite = np.ones(30, dtype=bool)
    mask = skySelect.select_sky_fibres(brightness, finite, method="middle-third")
    order = np.argsort(-brightness)
    expected = set(order[30 // 3:2 * 30 // 3])
    assert set(np.where(mask)[0]) == expected


def test_skymap_uses_region():
    brightness = np.linspace(1, 100, 20)
    finite = np.ones(20, dtype=bool)
    region = np.zeros(20, dtype=bool)
    region[5:15] = True
    mask = skySelect.select_sky_fibres(brightness, finite, method="skymap",
                                       in_sky_region=region)
    assert set(np.where(mask)[0]) == set(range(5, 15))


def test_skymap_without_region_falls_back_to_dimmest():
    brightness = np.linspace(1, 100, 20)
    finite = np.ones(20, dtype=bool)
    mask = skySelect.select_sky_fibres(brightness, finite, method="skymap",
                                       n_fibres=7, in_sky_region=None)
    assert mask.sum() == 7


def test_frame_uses_all_finite():
    brightness = np.arange(10, dtype=float)
    finite = np.ones(10, dtype=bool)
    finite[3] = False
    mask = skySelect.select_sky_fibres(brightness, finite, method="frame")
    assert set(np.where(mask)[0]) == set(i for i in range(10) if i != 3)


def test_dimmest_excludes_dead_zero_fibres():
    # Mirrors skyModel_1d's usable pool: finite & (counts > 0). Dead/zero fibres
    # (the absolute faintest) must NOT be chosen as sky.
    counts = np.array([0.0, 0.0, 5.0, 6.0, 7.0, 8.0, 9.0, 100.0])
    finite = np.isfinite(counts) & (counts > 0)
    mask = skySelect.select_sky_fibres(counts, finite, method="dimmest", n_fibres=3)
    assert set(np.where(mask)[0]) == {2, 3, 4}   # faintest three *positive* fibres
    assert not mask[0] and not mask[1]           # dead/zero fibres excluded


def test_low_signal_broadening_logic():
    # Reproduces skyModel_1d's min-fibre floor: a low-signal (e.g. blue) camera
    # with only 2 positive fibres but 30 finite ones. 'dimmest' on the positive
    # pool yields few; broadening to middle-third of finite gives a sturdier set.
    counts = np.full(30, -5.0)      # mostly negative noise
    counts[10] = 3.0                # two genuinely positive fibres
    counts[11] = 4.0
    finite_any = np.isfinite(counts)
    usable = finite_any & (counts > 0)

    dim = skySelect.select_sky_fibres(counts, usable, method="dimmest", n_fibres=20)
    assert dim.sum() == 2           # only two positive fibres available

    broadened = skySelect.select_sky_fibres(counts, finite_any, method="middle-third")
    assert broadened.sum() == 10    # n//3 : 2n//3 of 30
    assert broadened.sum() > dim.sum()                 # floor would broaden
    assert broadened.sum() >= skySelect.MIN_SKY_FIT_FIBRES


def test_empty_pool_returns_empty_mask():
    # A placeholder/missing camera: no finite-positive fibres -> empty selection,
    # which the skyModel_1d guard turns into a clean per-camera skip.
    counts = np.zeros(10)
    finite = np.isfinite(counts) & (counts > 0)   # all False
    mask = skySelect.select_sky_fibres(counts, finite, method="dimmest", n_fibres=5)
    assert mask.sum() == 0


def test_degenerate_region_falls_back():
    brightness = np.linspace(1, 100, 30)
    finite = np.ones(30, dtype=bool)
    region = np.zeros(30, dtype=bool)
    region[0] = True   # only one sky fibre -> below MIN_SKY_FIBRES
    mask = skySelect.select_sky_fibres(brightness, finite, method="skymap",
                                       n_fibres=20, in_sky_region=region)
    assert mask.sum() >= skySelect.MIN_SKY_FIBRES


# ---------------------------------------------------------------------------
# load_sky_map
# ---------------------------------------------------------------------------
def test_load_sky_map_detects_mask(tmp_path):
    p = tmp_path / "mask.fits"
    data = np.zeros((20, 20), dtype=np.int16)
    data[5:10, 5:10] = 1
    fits.PrimaryHDU(data).writeto(p)
    sm = skySelect.load_sky_map(str(p))
    assert sm.is_mask is True
    assert sm.data.shape == (20, 20)


def test_load_sky_map_detects_flux_image(tmp_path):
    p = tmp_path / "flux.fits"
    rng = np.random.default_rng(2)
    fits.PrimaryHDU(rng.uniform(0, 100, size=(20, 20)).astype(np.float32)).writeto(p)
    sm = skySelect.load_sky_map(str(p))
    assert sm.is_mask is False


def test_load_sky_map_collapses_cube(tmp_path):
    p = tmp_path / "cube.fits"
    rng = np.random.default_rng(3)
    fits.PrimaryHDU(rng.uniform(0, 1, size=(8, 20, 20)).astype(np.float32)).writeto(p)
    sm = skySelect.load_sky_map(str(p))
    assert sm.data.ndim == 2
    assert sm.data.shape == (20, 20)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
