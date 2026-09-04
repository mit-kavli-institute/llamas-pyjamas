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


def assert_dropped(keep, n_detected, expected):
    """Assert WHICH detections were discarded.

    Slot labels alone cannot express this: dropping the first trace and dropping
    the last both leave ``resolved == range(n)``, so a test that only checks the
    labels passes whether the ghost or a live fibre was thrown away. That is
    exactly how the green 3B one-lenslet shift got through.
    """
    got = sorted(set(range(n_detected)) - set(np.asarray(keep).tolist()))
    assert got == list(expected), f"dropped {got}, expected {list(expected)}"


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
    # The load-bearing assertion: the GHOST must go, not the live fibre at slot 0.
    assert_dropped(keep, 299, [298])


def test_end_gap_does_not_define_its_own_pitch():
    """Green 3B, 2026-08-31: the exact geometry that shifted a bench by one lenslet.

    The running pitch median is padded with the edge value, so without care the
    ghost's own 4.2x gap becomes the pitch it is measured against, it quantises to
    one slot, and the resolver discards a live fibre at the far end instead.
    """
    slots = list(range(300))
    pos = np.append(comb(slots, pitch=6.469, base=38.7),
                    comb([300], pitch=6.469, base=38.7) + 3.21 * 6.469)
    keep, resolved = resolve_trace_slots(np.arange(301), pos, 300, (), '3B')
    assert len(keep) == 300
    np.testing.assert_array_equal(resolved, slots)
    assert_dropped(keep, 301, [300])


def test_ghost_below_the_first_fibre_is_dropped():
    """Direction-agnostic: green 1B's spurious peak sits BELOW slot 0, not above."""
    slots = list(range(298))
    pos = np.insert(comb(slots), 0, comb([0])[0] - 4.0 * PITCH)
    keep, resolved = resolve_trace_slots(np.arange(299), pos, 298, (), '1B')
    assert len(keep) == 298
    np.testing.assert_array_equal(resolved, slots)
    assert_dropped(keep, 299, [0])


def test_ghost_one_pitch_past_the_end_is_refused():
    """A ghost exactly one pitch beyond a gapless bench is a genuine coin flip.

    Dropping the ghost and dropping the live slot-0 fibre fit the fibremap equally
    well and are rigid relabellings of each other, so nothing in the detector
    positions can choose. Refuse rather than register the bench off by one.
    """
    pos = comb(list(range(299)))                  # 299 traces on a 298-slot bench
    with pytest.raises(TraceCombError, match='coin flip'):
        resolve_trace_slots(np.arange(299), pos, 298, (), '1A')


def test_interior_dead_gap_survives_the_pitch_cleanup():
    """Cleaning the pitch denominator must not eat a real dead-fibre gap.

    2B (dead 49) plus a ghost past the slit end: the 2x gap at 49 must still count
    as two slots while the ghost alone is discarded.
    """
    slots = [s for s in range(298) if s != 49]
    pos = np.append(comb(slots), comb([303]))
    keep, resolved = resolve_trace_slots(np.arange(298), pos, 297, [49], '2B')
    assert len(keep) == 297
    np.testing.assert_array_equal(resolved, slots)
    assert 49 not in resolved
    assert_dropped(keep, 298, [297])


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
