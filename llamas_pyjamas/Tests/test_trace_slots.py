"""Unit tests for matching a traced fibre comb to the fibremap.

These pin down the invariant broken on 2025-03-05 and 2026-08-31: a trace's index
IS its fibremap position, so discarding the wrong trace when the detection
over-counts silently moves every fibre after it onto the neighbouring lenslet.
Benchside 2A over-counts on every dataset because dead fibre 299 leaks a faint
peak at the regular comb pitch at the end of the slit, where trimming by spacing
isolation has nothing to grab. Runnable with pytest or as a plain script
(`python -m llamas_pyjamas.Tests.test_trace_slots`).
"""

import numpy as np
import pytest

from llamas_pyjamas.Trace.traceLlamasMaster import resolve_trace_slots, TraceCombError

PITCH = 6.45
BASE = 60.0


def comb(slots, pitch=PITCH, base=BASE):
    """Detector rows for the given slit slots, with a realistic 1% pitch drift."""
    s = np.asarray(slots, dtype=float)
    return base + pitch * s * (1.0 + 0.01 * s / 300.0)


def test_clean_comb_is_a_noop():
    slots = list(range(298))
    keep, resolved = resolve_trace_slots(np.arange(298), comb(slots), 298, (), '1A')
    assert len(keep) == 298
    np.testing.assert_array_equal(resolved, slots)


def test_interior_dead_fibre_leaves_a_gap():
    # 2B: 297 live, dead at fibremap 49
    slots = [s for s in range(298) if s != 49]
    keep, resolved = resolve_trace_slots(np.arange(297), comb(slots), 297, [49], '2B')
    assert len(keep) == 297
    np.testing.assert_array_equal(resolved, slots)
    assert 49 not in resolved


def test_2a_end_of_slit_leak_is_the_trace_dropped():
    # 2A: 298 live, dead at 270 and 299. The detector yields 299 peaks because the
    # dead slot 299 leaks -- and it sits at the regular pitch, so spacing alone
    # cannot identify it. The one to drop is the leak, not a live fibre.
    detected = [s for s in range(300) if s != 270]
    assert len(detected) == 299
    keep, resolved = resolve_trace_slots(np.arange(299), comb(detected), 298,
                                         [270, 299], '2A')
    assert len(keep) == 298
    assert keep[-1] not in keep[:-1] and 298 not in keep      # last trace discarded
    np.testing.assert_array_equal(resolved,
                                  [s for s in range(300) if s not in (270, 299)])


def test_isolated_ghost_off_the_comb_is_dropped():
    slots = list(range(298))
    pos = np.append(comb(slots), comb([304]))     # ghost 6 pitches past the slit end
    keep, resolved = resolve_trace_slots(np.arange(299), pos, 298, (), '3A')
    assert len(keep) == 298
    np.testing.assert_array_equal(resolved, slots)


def test_missing_live_fibre_refuses_rather_than_guessing():
    # Exactly the 2025-03-05 failure: a live mid-slit fibre was never detected AND
    # the slot-299 leak was. The count is still 298, so a count check passes -- the
    # comb must be checked instead.
    detected = [s for s in range(300) if s not in (169, 270)]
    assert len(detected) == 298
    with pytest.raises(TraceCombError):
        resolve_trace_slots(np.arange(298), comb(detected), 298, [270, 299], '2A')


def test_short_comb_with_correct_count_refuses():
    # A live fibre missing from the middle with no compensating extra: the slots no
    # longer match the fibremap, so this must fail rather than renumber.
    detected = [s for s in range(299) if s != 100]
    with pytest.raises(TraceCombError):
        resolve_trace_slots(np.arange(298), comb(detected), 298, (), '1A')


def test_dead_fibre_at_the_end_leaves_no_gap():
    # Nothing to detect at slot 299, and no interior gap either: the resolver must
    # still land every trace on the right slot.
    detected = [s for s in range(299) if s != 270]
    keep, resolved = resolve_trace_slots(np.arange(298), comb(detected), 298,
                                         [270, 299], '2A')
    assert len(keep) == 298
    np.testing.assert_array_equal(resolved,
                                  [s for s in range(300) if s not in (270, 299)])


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
