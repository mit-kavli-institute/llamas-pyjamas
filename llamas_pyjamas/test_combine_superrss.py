"""Tests for the super-RSS substrate (Combine/superRSS.py).

The file-reading `build_super_rss` / `load_exposure` are validated on real may26 data; these cover
the headless view logic: band collapse (native per-fibre, no resample), masking + dropping fibres
with no good pixel in the window, quadrature variance, multi-channel concatenation, and the
surface-brightness conversion.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_combine_superrss`).
"""

import numpy as np

from llamas_pyjamas.Combine.superRSS import (ChannelStack, SuperRSS, combined_dir,
                                             exposure_prefix)


def _stack(channel, wave, flux, var, mask, ra=None, dec=None, area=0.5, exp=0):
    n = flux.shape[0]
    return ChannelStack(
        channel=channel,
        ra=np.arange(n, dtype=float) if ra is None else np.asarray(ra, float),
        dec=np.zeros(n) if dec is None else np.asarray(dec, float),
        wave=np.asarray(wave, float), flux=np.asarray(flux, float), var=np.asarray(var, float),
        mask=np.asarray(mask, bool), solid_angle=np.full(n, area),
        exposure=np.full(n, exp, dtype=int))


def _one_channel_super(stack):
    return SuperRSS(field='T', plane='skysub', bunit='counts', exposures=[], channels={stack.channel: stack})


def test_collapse_band_sums_window_and_respects_mask():
    wave = np.tile([5000, 5001, 5002, 5003, 5004], (3, 1))
    flux = np.array([[1, 2, 3, 4, 5],       # fibre 0: clean
                     [10, 10, 10, 10, 10],  # fibre 1: pixel @5002 masked
                     [7, 7, 7, 7, 7]])      # fibre 2: fully masked -> dropped
    var = np.ones((3, 5))
    mask = np.zeros((3, 5), bool)
    mask[1, 2] = True
    mask[2, :] = True
    sr = _one_channel_super(_stack('green', wave, flux, var, mask))

    ft = sr.collapse_band(5001, 5003)                 # window = indices 1,2,3
    assert len(ft) == 2                                # fibre 2 dropped
    order = np.argsort(ft.ra)
    assert list(ft.value[order]) == [2 + 3 + 4, 10 + 10]      # fibre1 drops its masked @5002
    assert list(ft.var[order]) == [3.0, 2.0]                  # variance adds in quadrature (=count here)
    assert list(ft.npix[order]) == [3, 2]


def test_collapse_band_empty_when_window_outside_coverage():
    wave = np.tile([5000, 5001, 5002], (2, 1))
    sr = _one_channel_super(_stack('green', wave, np.ones((2, 3)), np.ones((2, 3)),
                                   np.zeros((2, 3), bool)))
    ft = sr.collapse_band(6000, 6100)                 # no coverage
    assert len(ft) == 0


def test_collapse_band_concatenates_channels():
    wave = np.tile([5000, 5001, 5002], (2, 1))
    g = _stack('green', wave, np.full((2, 3), 2.0), np.ones((2, 3)), np.zeros((2, 3), bool),
               ra=[10, 11])
    r = _stack('red', wave, np.full((2, 3), 5.0), np.ones((2, 3)), np.zeros((2, 3), bool),
               ra=[20, 21], exp=1)
    sr = SuperRSS(field='T', plane='skysub', bunit='counts', exposures=[],
                  channels={'green': g, 'red': r})
    ft = sr.collapse_band(5000, 5002)
    assert len(ft) == 4
    assert set(ft.channel) == {'green', 'red'}
    assert sorted(ft.value.tolist()) == [6.0, 6.0, 15.0, 15.0]   # green 3*2, red 3*5


def test_surface_brightness_divides_by_area():
    wave = np.tile([5000, 5001], (1, 1))
    sr = _one_channel_super(_stack('green', wave, np.array([[4.0, 4.0]]), np.array([[1.0, 1.0]]),
                                   np.zeros((1, 2), bool), area=0.5))
    ft = sr.collapse_band(5000, 5001)
    sb_v, sb_var = ft.surface_brightness()
    assert sb_v[0] == 8.0 / 0.5                        # value / area
    assert sb_var[0] == 2.0 / 0.5 ** 2                 # var / area^2


def test_mask_bad_fibres_flags_strong_negatives_as_no_data():
    # 12 fibres (>= the routine's min); fibre 5 is a broken/over-subtracted fibre (strong negative)
    rng = np.random.default_rng(0)
    n = 12
    wave = np.tile([5000, 5001, 5002], (n, 1))
    flux = 10.0 + rng.normal(0, 0.5, (n, 3))
    flux[5] = -500.0                                            # broken fibre
    st = _stack('green', wave, flux, np.ones((n, 3)), np.zeros((n, 3), bool))
    sr = _one_channel_super(st)
    nbad = sr.mask_bad_fibres(neg_nsigma=5.0)
    assert nbad['green'] == 1
    assert st.mask[5].all() and np.isinf(st.var[5]).all()      # broken fibre -> no-data
    assert not st.mask[0].any()                                # good fibres untouched
    ft = sr.collapse_band(5000, 5002)                          # drops out; no negative hole
    assert len(ft) == n - 1 and (ft.value > 0).all()


def test_exposure_prefix():
    assert exposure_prefix('/x/LLAMAS_2026-05-17_02-49-56.7_RSS_green.fits') \
        == 'LLAMAS_2026-05-17_02-49-56.7'


def test_combined_dir_standard_location():
    # exposures in <reduced>/extractions -> combined products in the sibling <reduced>/combined
    assert combined_dir(['/d/reduced/extractions/x_RSS_green.fits']) == '/d/reduced/combined'
    # otherwise a 'combined' subdir beside the inputs
    assert combined_dir(['/d/foo/x_RSS_green.fits']) == '/d/foo/combined'


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
