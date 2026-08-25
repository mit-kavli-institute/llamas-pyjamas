"""Tests for the Gaia absolute-flux anchor SED comparison (Combine/fluxanchor.py).

Pure/offline: given a reference SED and an extracted spectrum that is a scaled (aperture-lossy)
version of it, anchor_scale must recover the scale with low scatter; a colour-sloped extraction must
show high scatter; no overlap -> None. The gaiaxpy XP retrieval + TAP query are verified live.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.Tests.test_combine_fluxanchor`).
"""

import numpy as np

from llamas_pyjamas.Combine.fluxanchor import (anchor_scale, _phot_to_sed, GAIA_VEGA_ZP,
                                               NoXPSpectrumError)


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    raise AssertionError(f'expected {exc.__name__}')


def _ref():
    wave = np.linspace(3400.0, 9200.0, 800)
    flam = 1e-15 * (wave / 6000.0) ** -1.2               # a smooth reference SED
    return wave, flam


def _extracted(scale=1.0, colour=0.0):
    """Extracted spectrum per channel = reference / scale (aperture loss), optionally tilted."""
    rw, rf = _ref()
    out = {}
    for c, (lo, hi) in (('blue', (3500, 5000)), ('green', (4800, 7000)), ('red', (6800, 9000))):
        w = np.linspace(lo, hi, 300)
        f = np.interp(w, rw, rf) / scale
        if colour:
            f = f * (w / 6000.0) ** colour               # residual throughput slope
        out[c] = (w, f, np.ones_like(w))
    return out


def test_anchor_recovers_scale():
    rw, rf = _ref()
    res = anchor_scale(_extracted(scale=1.16), rw, rf)   # 16% aperture loss
    assert res is not None
    assert np.isclose(res['scale'], 1.16, rtol=0.02)     # recovers the scale
    assert res['scatter'] < 0.02                          # flat ratio -> pure scale


def test_colour_slope_shows_high_scatter():
    rw, rf = _ref()
    flat = anchor_scale(_extracted(scale=1.16, colour=0.0), rw, rf)
    tilt = anchor_scale(_extracted(scale=1.16, colour=0.5), rw, rf)   # throughput slope
    assert tilt['scatter'] > flat['scatter'] * 3          # slope flagged by higher scatter


def test_no_overlap_returns_none():
    rw, rf = _ref()
    far = {'green': (np.linspace(12000, 13000, 100), np.ones(100), np.ones(100))}
    assert anchor_scale(far, rw, rf) is None


def test_phot_to_sed_converts_mags_to_flux():
    # broadband fallback: Vega mag -> f_lambda = F0 * 10**(-0.4*mag) at each band pivot, wave-sorted
    mags = {'G': 15.0, 'BP': 15.3, 'RP': 14.5}
    wave, flam = _phot_to_sed(mags)
    assert list(wave) == sorted(wave)                              # BP, G, RP by wavelength
    for w, f in zip(wave, flam):
        band = min(GAIA_VEGA_ZP, key=lambda b: abs(GAIA_VEGA_ZP[b][0] - w))
        wl0, f0 = GAIA_VEGA_ZP[band]
        assert np.isclose(w, wl0)
        assert np.isclose(f, f0 * 10.0 ** (-0.4 * mags[band]), rtol=1e-6)


def test_phot_to_sed_needs_two_bands():
    _raises(NoXPSpectrumError, _phot_to_sed, {'G': 18.9})          # a lone band -> no colour baseline
    _raises(NoXPSpectrumError, _phot_to_sed, {'G': np.nan, 'BP': np.nan, 'RP': 14.0})  # NaNs excluded


def test_phot_fallback_sed_anchors_like_xp():
    # a coarse 3-point photometric SED still recovers the aperture-loss scale via anchor_scale
    F0 = {b: 1e-15 * (GAIA_VEGA_ZP[b][0] / 6000.0) ** -1.2 for b in GAIA_VEGA_ZP}
    mags = {b: -2.5 * np.log10(F0[b] / GAIA_VEGA_ZP[b][1]) for b in F0}   # invert the ZP relation
    wave, flam = _phot_to_sed(mags)                                # ~ the same smooth SED as _ref()
    res = anchor_scale(_extracted(scale=1.16), wave, flam)
    assert res is not None
    assert np.isclose(res['scale'], 1.16, rtol=0.05)              # coarse SED, looser tol than XP


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
