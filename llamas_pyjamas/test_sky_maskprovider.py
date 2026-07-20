"""Tests for the SkyMask abstraction + build_sky_mask provider (Sky/skySelect.py, sky-refine Phase 1).

Behaviour-preserving guarantee: build_sky_mask must return the SAME fibres as select_sky_fibres for
every brightness-based method (it just wraps it and attaches provenance). Plus: the 'manual' provider,
SkyMask properties, and FITS round-trip.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_sky_maskprovider`).
"""

import numpy as np

from llamas_pyjamas.Sky.skySelect import (SkyMask, build_sky_mask, select_sky_fibres,
                                          VALID_METHODS)

rng = np.random.default_rng(0)
N = 120
# a realistic-ish brightness field: mostly faint sky + a few bright object fibres, some dead
BRIGHT = np.abs(rng.normal(100.0, 15.0, N))
BRIGHT[rng.choice(N, 6, replace=False)] += 5000.0          # objects
FINITE = np.ones(N, dtype=bool)
BRIGHT[rng.choice(N, 4, replace=False)] = np.nan           # dead
FINITE[~np.isfinite(BRIGHT)] = False
FIBER_Y = np.linspace(0.0, 1.0, N)                          # slit position for 'stratified'


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    raise AssertionError(f'expected {exc.__name__}')


def test_provider_reproduces_select_sky_fibres_all_methods():
    # the core behaviour-preserving contract: same mask as the underlying selector
    region = FIBER_Y > 0.8                                   # a fake sky region for 'skymap'
    for method in ("dimmest", "quantile", "middle-third", "stratified", "frame", "all"):
        sm = build_sky_mask(BRIGHT, FINITE, method=method, n_fibres=20, fiber_y=FIBER_Y)
        ref = select_sky_fibres(BRIGHT, FINITE, method=method, n_fibres=20, fiber_y=FIBER_Y)
        assert np.array_equal(sm.mask, ref), method
        assert sm.method == method
    sm = build_sky_mask(BRIGHT, FINITE, method="skymap", in_sky_region=region)
    ref = select_sky_fibres(BRIGHT, FINITE, method="skymap", in_sky_region=region)
    assert np.array_equal(sm.mask, ref)


def test_skymask_properties():
    sm = build_sky_mask(BRIGHT, FINITE, method="dimmest", n_fibres=15)
    assert sm.n_fiber == N
    assert sm.n_sky == int(sm.mask.sum())
    assert np.array_equal(sm.ids(), np.where(sm.mask)[0])
    assert sm.n_sky >= 1


def test_manual_provider_bool_mask_and_ids():
    want = np.zeros(N, dtype=bool)
    want[[3, 7, 42]] = True
    a = build_sky_mask(method="manual", explicit=want)            # boolean mask
    assert a.method == "manual" and np.array_equal(a.mask, want)
    b = build_sky_mask(method="manual", explicit=[3, 7, 42], n_fiber=N)   # integer ids
    assert np.array_equal(b.mask, want)
    # ids need n_fiber; out-of-range rejected
    _raises(ValueError, build_sky_mask, method="manual", explicit=[3, 7])
    _raises(ValueError, build_sky_mask, method="manual", explicit=[3, 999], n_fiber=N)
    _raises(ValueError, build_sky_mask, method="manual")          # explicit required


def test_provenance_carries_method_and_params():
    sm = build_sky_mask(BRIGHT, FINITE, method="quantile", q_lo=0.05, q_hi=0.12)
    assert sm.method == "quantile"
    assert sm.provenance["q_lo"] == 0.05 and sm.provenance["q_hi"] == 0.12
    assert sm.provenance["source"] == "in-exposure"


def test_fits_hdu_roundtrip():
    sm = build_sky_mask(BRIGHT, FINITE, method="stratified", n_fibres=20, fiber_y=FIBER_Y)
    hdu = sm.to_hdu()
    assert hdu.name == "SKYMASK"
    assert hdu.header["SKYNSKY"] == sm.n_sky and hdu.header["SKYNFIB"] == N
    back = SkyMask.from_hdu(hdu)
    assert np.array_equal(back.mask, sm.mask)
    assert back.method == sm.method
    assert back.provenance == sm.provenance


def test_manual_is_a_valid_method():
    assert "manual" in VALID_METHODS


if __name__ == '__main__':
    import sys
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f'PASS {name}')
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
