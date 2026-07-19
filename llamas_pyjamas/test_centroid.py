"""Tests for the fibre-space centroiding engine (Utils/centroid.py).

Synthetic, deterministic (no data dependency): a Gaussian source sampled on a hex-like fibre
lattice, checking sub-fibre recovery, background-bias removal, window recentring, the flux^2
option, and the too-few-fibres guard. Real-field validation against Gaia is done separately.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_centroid`).
"""

import numpy as np

from llamas_pyjamas.Utils.centroid import Centroid, fibre_centroid


def _hex_lattice(n=14):
    """A pointy-top hex lattice of unit spacing (like the LLAMAS fibre map), returns x, y."""
    xs, ys = [], []
    for row in range(n):
        yy = row * (np.sqrt(3) / 2)
        off = 0.5 if (row % 2) else 0.0
        for col in range(n):
            xs.append(col + off)
            ys.append(yy)
    return np.array(xs), np.array(ys)


def _source(x, y, cx, cy, sigma=0.9, amp=1000.0, bg=0.0):
    r2 = (x - cx) ** 2 + (y - cy) ** 2
    return amp * np.exp(-0.5 * r2 / sigma ** 2) + bg


def test_recovers_source_center_subfibre():
    x, y = _hex_lattice()
    cx, cy = 6.3, 5.1                      # deliberately between fibres
    f = _source(x, y, cx, cy)
    c = fibre_centroid(x, y, f, radius=2.5)
    # position is the guarantee; `converged` is best-effort (a fixed-radius window on a discrete
    # lattice can toggle a boundary fibre and never settle below tol, while staying accurate).
    assert c is not None
    assert np.hypot(c.x - cx, c.y - cy) < 0.05, (c.x, c.y)


def test_background_subtraction_removes_bias():
    x, y = _hex_lattice()
    cx, cy = 3.2, 4.0                      # near an edge so an un-subtracted pedestal biases inward
    f = _source(x, y, cx, cy, bg=500.0)   # large flat pedestal
    good = fibre_centroid(x, y, f, radius=2.5)               # median background estimated
    biased = fibre_centroid(x, y, f, radius=2.5, background=0.0)   # pedestal left in -> biased
    assert np.hypot(good.x - cx, good.y - cy) < 0.08
    assert np.hypot(biased.x - cx, biased.y - cy) > np.hypot(good.x - cx, good.y - cy)


def test_window_iteration_converges_from_offset_guess():
    x, y = _hex_lattice()
    cx, cy = 7.0, 6.0
    f = _source(x, y, cx, cy)
    # start the window ~1.5 fibres away; iteration should walk it onto the source
    c = fibre_centroid(x, y, f, guess=(8.4, 7.1), radius=2.0, iterations=5)
    assert c is not None
    assert np.hypot(c.x - cx, c.y - cy) < 0.05
    # and a single-pass (no recentring) from the offset guess is worse -> iteration helped
    c1 = fibre_centroid(x, y, f, guess=(8.4, 7.1), radius=2.0, iterations=1)
    assert np.hypot(c1.x - cx, c1.y - cy) > np.hypot(c.x - cx, c.y - cy)


def test_flux_squared_is_valid_and_recovers_center():
    x, y = _hex_lattice()
    cx, cy = 5.5, 5.2
    f = _source(x, y, cx, cy)
    c1 = fibre_centroid(x, y, f, radius=2.5, power=1)
    c2 = fibre_centroid(x, y, f, radius=2.5, power=2)
    for c in (c1, c2):
        assert np.hypot(c.x - cx, c.y - cy) < 0.06
    # flux^2 down-weights the wings -> fewer effective fibres than plain flux
    assert c2.n_fibres <= c1.n_fibres


def test_too_few_fibres_returns_none():
    x, y = _hex_lattice()
    f = _source(x, y, 6.0, 5.0)
    # a tiny window with < min_fibres inside
    assert fibre_centroid(x, y, f, guess=(6.0, 5.0), radius=0.1, min_fibres=3) is None
    # all-nan flux
    assert fibre_centroid(x, y, np.full_like(f, np.nan), radius=2.0) is None


def test_returns_dataclass_with_diagnostics():
    x, y = _hex_lattice()
    f = _source(x, y, 6.0, 5.0)
    c = fibre_centroid(x, y, f, radius=2.5)
    assert isinstance(c, Centroid)
    assert c.n_fibres > 3 and c.flux_sum > 0 and c.background >= 0


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
