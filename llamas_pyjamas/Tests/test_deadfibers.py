"""Unit tests for dead-fibre bookkeeping + per-fibre array alignment.

These pin down the invariant that motivated the 2026-07 dead-fibre refactor:
every per-fibre array is LIVE-indexed and mutually aligned, and the fibermap
expansion is a clean, round-trippable operation. Runnable with pytest or as a
plain script (`python -m llamas_pyjamas.Tests.test_deadfibers`).
"""

import numpy as np

from llamas_pyjamas.Utils.deadfibers import live_fibre_ids, insert_dead_fibre_rows


def test_live_fibre_ids_skips_dead():
    assert live_fibre_ids(5, [2]) == [0, 1, 3, 4, 5]
    assert live_fibre_ids(4, [0, 3]) == [1, 2, 4, 5]
    assert live_fibre_ids(3, []) == [0, 1, 2]


def test_live_fibre_ids_real_cameras():
    # green 2A: 298 live + dead at fibremap 270, 299
    ids = live_fibre_ids(298, [270, 299])
    assert len(ids) == 298
    assert 270 not in ids and 299 not in ids
    assert ids[0] == 0 and ids[-1] == 298          # last live sits at position 298


def test_insert_dead_rows_positions_and_order():
    live = np.array([[10], [11], [12], [13], [14]], float)   # row k -> fibre 10+k
    out = insert_dead_fibre_rows(live, [2], fill=0.0)
    assert out.shape == (6, 1)
    assert out[2, 0] == 0.0                                  # dead position filled
    assert list(out[:, 0]) == [10, 11, 0, 12, 13, 14]


def test_insert_dead_rows_nan_fill_multi():
    live = np.arange(4.0).reshape(4, 1)
    out = insert_dead_fibre_rows(live, [1, 4], fill=np.nan)  # total 6, dead at 1 and 4
    assert np.isnan(out[1, 0]) and np.isnan(out[4, 0])
    assert list(out[[0, 2, 3, 5], 0]) == [0, 1, 2, 3]


def test_insert_out_of_range_raises():
    try:
        insert_dead_fibre_rows(np.zeros((3, 1)), [10])       # dead beyond total
    except ValueError:
        return
    raise AssertionError("expected ValueError for out-of-range dead index")


def test_roundtrip_remove_recovers_live():
    live = np.random.default_rng(0).normal(size=(10, 3))
    dead = [3, 7]
    fib = insert_dead_fibre_rows(live, dead)
    keep = [i for i in range(fib.shape[0]) if i not in dead]
    np.testing.assert_array_equal(fib[keep], live)


def test_two_arrays_expanded_with_same_dead_stay_aligned():
    # counts and wave both live-indexed; each row encodes its live-fibre id.
    n_live, dead = 8, [2, 5]
    counts_live = np.arange(n_live, dtype=float).reshape(-1, 1)
    wave_live = (1000 + np.arange(n_live, dtype=float)).reshape(-1, 1)
    counts_fm = insert_dead_fibre_rows(counts_live, dead, 0.0)
    wave_fm = insert_dead_fibre_rows(wave_live, dead, np.nan)
    # at every non-dead fibermap position the two arrays describe the SAME fibre
    for live_id, pos in enumerate(live_fibre_ids(n_live, dead)):
        assert counts_fm[pos, 0] == live_id
        assert wave_fm[pos, 0] == 1000 + live_id
    for d in dead:
        assert counts_fm[d, 0] == 0.0
        assert np.isnan(wave_fm[d, 0])


def test_regression_old_padded_counts_misaligns_live_wave():
    # The bug: counts fibermap-padded (dead inserted) while wave stays LIVE.
    # Pairing counts[i] with wave[i] misaligns every fibre after the first dead.
    n_live, dead = 6, [2]
    fibre_id = np.arange(n_live, dtype=float).reshape(-1, 1)
    counts_padded = insert_dead_fibre_rows(fibre_id, dead, -1.0)   # 7 rows (fibermap)
    wave_live = fibre_id.copy()                                     # 6 rows (LIVE)
    # index 3: padded counts holds live fibre 2, live wave holds live fibre 3 -> misaligned
    assert counts_padded[3, 0] == 2.0
    assert wave_live[3, 0] == 3.0
    assert counts_padded[3, 0] != wave_live[3, 0]
    # The fix: keep BOTH live -> identical indexing -> aligned everywhere.
    counts_live = fibre_id.copy()
    np.testing.assert_array_equal(counts_live, wave_live)


if __name__ == "__main__":
    import sys
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
