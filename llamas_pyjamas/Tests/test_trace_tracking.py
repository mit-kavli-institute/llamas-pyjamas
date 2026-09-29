"""Unit tests for walking fibre centroids along the dispersion axis.

On 2026-09-29 blue 4A, faint fibres sitting between saturated bright neighbours
crept onto the neighbour one pitch away as the old walk re-centroided around
``int(previous)`` and accepted any step under 1.5 px. ``track_comb_step`` must
keep every fibre on its own peak over a full half-detector walk.
Runnable with pytest or as a plain script
(`python -m llamas_pyjamas.Tests.test_trace_tracking`).
"""

import numpy as np
import pytest
from scipy.ndimage import minimum_filter1d

from llamas_pyjamas.Trace.traceLlamasMaster import track_comb_step

PITCH, SIGMA, NFIB, NROW = 6.43, 1.4, 60, 450


def truth(step, slope=0.45):
    """Fibre rows after ``step`` tracing columns, tilting ``slope`` px per step."""
    return 30.0 + PITCH * np.arange(NFIB) + slope * step


def comb_at(step, sat=1.0):
    """Valley-subtracted cross-section: bright odd fibres clipped at saturation."""
    y = np.arange(NROW, dtype=float)
    amp = np.where(np.arange(NFIB) % 2, 1.7, 0.6)
    prof = np.zeros(NROW)
    for a, c in zip(amp, truth(step)):
        prof += np.minimum(a * np.exp(-0.5 * ((y - c) / SIGMA) ** 2), sat)
    prof = np.minimum(prof, sat)
    return prof - minimum_filter1d(prof, size=int(PITCH) + 1)


def walk(nstep=72):
    prev, prevprev, prev_ok, step_ok = truth(0), None, np.ones(NFIB, bool), None
    worst, n_valid = 0.0, 0
    for k in range(1, nstep + 1):
        pos, ok = track_comb_step(comb_at(k), prev, prevprev, pitch=PITCH, step_valid=step_ok)
        worst = max(worst, float(np.abs(pos - truth(k)).max()))
        n_valid += int(ok.sum())
        step_ok = ok & prev_ok
        prevprev, prev, prev_ok = prev, pos, ok
    return worst, n_valid / (nstep * NFIB)


def test_faint_fibres_between_saturated_neighbours_stay_on_their_peak():
    worst, frac_valid = walk()
    assert worst < 1.0, f"a centroid drifted {worst:.2f} px from its fibre"
    assert frac_valid > 0.9


def test_collapsed_neighbours_are_masked():
    # Two guesses on the same peak: neither may be reported as a valid centroid.
    comb = comb_at(0)
    prev = truth(0).copy()
    prev[11] = prev[10] + 0.5
    _, ok = track_comb_step(comb, prev, pitch=PITCH)
    assert not ok[10] and not ok[11]


def test_negative_comb_is_not_a_centroid():
    pos, ok = track_comb_step(-np.ones(100), np.array([50.0]), pitch=PITCH)
    assert not ok[0] and pos[0] == 50.0


def test_dark_stretch_does_not_run_away():
    """A block of fibres with no signal for 25 columns (vignetted corner) must
    re-acquire afterwards, not drift together on carried predictions -- the
    first cut of this fix ran blue 4A's bottom 40 fibres 67 px off this way."""
    prev, prevprev, prev_ok, step_ok = truth(0), None, np.ones(NFIB, bool), None
    for k in range(1, 73):
        comb = comb_at(k)
        if 20 <= k < 45:
            comb[:150] = 0.0                      # fibres 0-18 go dark
        pos, ok = track_comb_step(comb, prev, prevprev, pitch=PITCH, step_valid=step_ok)
        step_ok = ok & prev_ok
        prevprev, prev, prev_ok = prev, pos, ok
    assert np.abs(prev - truth(72)).max() < 1.0
    assert ok.all()


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
