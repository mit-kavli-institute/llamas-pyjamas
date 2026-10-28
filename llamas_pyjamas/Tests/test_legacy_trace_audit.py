"""Unit tests for auditing legacy trace pickles before they are reused.

A legacy trace predates the tracer's fibre-registration check, so it has no
``comb_ok`` attribute, and a misregistered comb can still have the right fibre
COUNT (e.g. a 2A trim that drops a live fibre but keeps dead fibre 299's leak).
These tests pin down that the audit catches that, that it leaves good legacy
traces and new-style traces alone, and that a failing one is swapped for the
mastercalib trace without losing the user's file. Runnable with pytest or as a
plain script (``python -m llamas_pyjamas.Tests.test_legacy_trace_audit``).
"""

import io
import logging
import os
import pickle

import numpy as np
import pytest

from llamas_pyjamas.Tests.test_trace_slots import comb, throughput
from llamas_pyjamas.Trace.traceLlamasMaster import (
    audit_trace_registration, normalise_trace_heights)
from llamas_pyjamas.Utils.reporter import PipelineReporter
from llamas_pyjamas.Utils.utils import (
    MasterTraceUnusableError, build_trace_fingerprints, confirm_unusable_master_traces,
    copy_mastercalib_trace, count_trace_fibres, validate_and_fix_trace_fibres)

# 2A per traceLUT.json: 298 live fibres, dead at fibremap 270 and 299.
DEAD_2A = [270, 299]
LIVE_2A = [s for s in range(300) if s not in DEAD_2A]
# A mis-trim: live slot 126 dropped, dead slot 299's leak kept. Right count.
MISTRIM_2A = [s for s in range(300) if s not in (126, 270)]


class LegacyTrace:
    """Stand-in for a trace pickle written before the registration checks."""

    def __init__(self, channel, bench, side, slots, flux, nx=64):
        rows = comb(slots)
        self.channel, self.bench, self.side = channel, bench, side
        self.naxis1 = nx
        self.nfibers = len(slots)
        self.traces = np.repeat(rows[:, None], nx, axis=1)
        # A flat whose fibre fluxes are the per-fibre throughput.
        self.data = np.zeros((int(rows.max()) + 60, nx))
        for row, f in zip(rows, flux):
            self.data[int(round(row)), :] = f


class CheckedTrace(LegacyTrace):
    """A trace from the current tracer: it carries ``comb_ok``."""

    comb_ok = True


def legacy_2a(slots, seed=0, cls=LegacyTrace):
    flux = throughput(len(slots), seed=seed)
    return cls('red', 2, 'A', slots, flux)


def save(obj, directory, name):
    path = os.path.join(str(directory), name)
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh)
    return path


def load(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# --- audit_trace_registration -------------------------------------------------

def test_clean_legacy_trace_passes():
    flux = throughput(298, seed=1)
    obj = LegacyTrace('green', 1, 'A', list(range(298)), flux)
    assert audit_trace_registration(obj, fingerprint=normalise_trace_heights(flux)) is None


def test_2a_mistrim_is_flagged():
    # Right count, so a count-only check accepts it; the slot check does not.
    obj = legacy_2a(MISTRIM_2A)
    assert obj.nfibers == 298
    assert audit_trace_registration(obj, dead_fibers=DEAD_2A) is not None


def test_correct_2a_trace_passes():
    assert audit_trace_registration(legacy_2a(LIVE_2A), dead_fibers=DEAD_2A) is None


def test_bench_one_slot_off_is_flagged_by_the_fingerprint():
    # Positions are a perfect comb, but every fibre carries its neighbour's flux.
    flux = throughput(301, seed=3)
    obj = LegacyTrace('blue', 4, 'A', list(range(300)), flux[1:])
    reason = audit_trace_registration(obj, fingerprint=normalise_trace_heights(flux[:300]))
    assert reason is not None and 'slot lag' in reason


def test_trace_without_flat_skips_the_fingerprint():
    flux = throughput(301, seed=3)
    obj = LegacyTrace('blue', 4, 'A', list(range(300)), flux[1:])
    del obj.data
    assert audit_trace_registration(obj, fingerprint=normalise_trace_heights(flux[:300])) is None


# --- validate_and_fix_trace_fibres ---------------------------------------------

@pytest.fixture
def dirs(tmp_path):
    trace_dir, master_dir = tmp_path / 'traces', tmp_path / 'mastercalib'
    trace_dir.mkdir()
    master_dir.mkdir()
    return trace_dir, master_dir


def test_failing_legacy_trace_is_backed_up_and_replaced(dirs):
    trace_dir, master_dir = dirs
    legacy = save(legacy_2a(MISTRIM_2A), trace_dir, 'LLAMAS_red_2_A_traces.pkl')
    save(legacy_2a(LIVE_2A, seed=5), master_dir, 'LLAMAS_master_red_2_A_traces.pkl')

    result = validate_and_fix_trace_fibres(str(trace_dir), mastercalib_dir=str(master_dir))

    assert [t[:3] for t in result['misregistered_traces']] == [('red', '2', 'A')]
    assert not result['all_valid']
    assert ('red', '2', 'A') in [f[:3] for f in result['fallback_used']]
    # The user's file survives under a name no *.pkl glob picks up...
    backup = load(legacy + '.legacy')
    np.testing.assert_array_equal(backup.traces[:, 0], comb(MISTRIM_2A))
    # ...and the mastercalib trace now sits under the user name.
    np.testing.assert_array_equal(load(legacy).traces[:, 0], comb(LIVE_2A))


def unusable(result):
    """Camera and own-trace path of each ``unusable_masters`` entry."""
    return [(ch + b + sd, own) for ch, b, sd, _reason, own in result['unusable_masters']]


def test_unusable_master_is_not_swapped_in_for_a_legacy_trace(dirs):
    # Swapping in a trace that fails the same check would gain nothing. Nothing is
    # moved: the user is asked first (confirm_unusable_master_traces).
    trace_dir, master_dir = dirs
    legacy = save(legacy_2a(MISTRIM_2A), trace_dir, 'LLAMAS_red_2_A_traces.pkl')
    save(legacy_2a(MISTRIM_2A, seed=5), master_dir, 'LLAMAS_red_2_A_traces.pkl')

    result = validate_and_fix_trace_fibres(str(trace_dir), mastercalib_dir=str(master_dir))

    assert [t[:3] for t in result['misregistered_traces']] == [('red', '2', 'A')]
    assert ('red', '2', 'A') not in [f[:3] for f in result['fallback_used']]
    assert unusable(result) == [('red2A', legacy)]
    assert os.path.exists(legacy) and not os.path.exists(legacy + '.legacy')


def test_wrong_count_master_is_not_copied_for_a_wrong_count_trace(dirs):
    trace_dir, master_dir = dirs
    short = save(legacy_2a(LIVE_2A[:-1]), trace_dir, 'LLAMAS_red_2_A_traces.pkl')
    save(legacy_2a(LIVE_2A[:-2]), master_dir, 'LLAMAS_master_red_2_A_traces.pkl')

    result = validate_and_fix_trace_fibres(str(trace_dir), mastercalib_dir=str(master_dir))

    assert unusable(result) == [('red2A', short)]
    assert 'needs 298' in result['unusable_masters'][0][3]
    assert load(short).nfibers == 297          # still the user's own file


def test_unusable_master_is_not_copied_for_a_missing_camera(dirs):
    trace_dir, master_dir = dirs
    save(legacy_2a(MISTRIM_2A), master_dir, 'LLAMAS_master_red_2_A_traces.pkl')

    result = validate_and_fix_trace_fibres(str(trace_dir), mastercalib_dir=str(master_dir))

    assert unusable(result) == [('red2A', None)]
    assert os.listdir(trace_dir) == []


def test_empty_trace_dir_is_filled_camera_by_camera(dirs):
    trace_dir, master_dir = dirs
    save(legacy_2a(LIVE_2A), master_dir, 'LLAMAS_master_red_2_A_traces.pkl')
    save(CheckedTrace('green', 2, 'A', LIVE_2A[:-3], throughput(295)),
         master_dir, 'LLAMAS_master_green_2_A_traces.pkl')

    result = validate_and_fix_trace_fibres(str(trace_dir), mastercalib_dir=str(master_dir))

    assert [f[:3] for f in result['fallback_used']] == [('red', '2', 'A')]
    assert unusable(result) == [('green2A', None)]
    assert os.listdir(trace_dir) == ['LLAMAS_red_2_A_traces.pkl']


def test_mastercalib_directory_is_not_renamed_before_asking(dirs):
    _, master_dir = dirs
    legacy = save(legacy_2a(MISTRIM_2A), master_dir, 'LLAMAS_master_red_2_A_traces.pkl')

    result = validate_and_fix_trace_fibres(str(master_dir), mastercalib_dir=str(master_dir))

    assert [t[:3] for t in result['misregistered_traces']] == [('red', '2', 'A')]
    assert unusable(result) == [('red2A', legacy)]
    assert os.path.exists(legacy) and not os.path.exists(legacy + '.legacy')


def test_copy_refuses_an_unreadable_master(dirs):
    trace_dir, master_dir = dirs
    with open(os.path.join(str(master_dir), 'LLAMAS_red_2_A_traces.pkl'), 'wb') as fh:
        fh.write(b'not a pickle')
    with pytest.raises(MasterTraceUnusableError, match='cannot be read'):
        copy_mastercalib_trace('red', '2', 'A', str(master_dir), str(trace_dir))


def test_good_legacy_and_new_style_traces_are_left_alone(dirs):
    trace_dir, master_dir = dirs
    good = save(legacy_2a(LIVE_2A), trace_dir, 'LLAMAS_red_2_A_traces.pkl')
    # A new-style trace was checked when it was made, so it is not re-audited.
    checked = save(CheckedTrace('green', 2, 'A', MISTRIM_2A, throughput(298)),
                   trace_dir, 'LLAMAS_green_2_A_traces.pkl')

    result = validate_and_fix_trace_fibres(str(trace_dir), mastercalib_dir=str(master_dir))

    assert result['misregistered_traces'] == []
    assert {t[3] for t in result['valid_traces']} == {good, checked}
    assert not any(name.endswith('.legacy') for name in os.listdir(trace_dir))


# --- mastercalib checks --------------------------------------------------------

def test_count_trace_fibres_fails_a_misregistered_master(dirs, caplog):
    # A fallback must have the right fibres: failing the audit fails the set just
    # as a wrong count does.
    _, master_dir = dirs
    save(legacy_2a(MISTRIM_2A), master_dir, 'LLAMAS_red_2_A_traces.pkl')
    with caplog.at_level(logging.WARNING):
        assert count_trace_fibres(str(master_dir)) is False
    assert 'red 2A fails the registration check' in caplog.text


def test_count_trace_fibres_passes_a_good_legacy_master(dirs):
    _, master_dir = dirs
    save(legacy_2a(LIVE_2A), master_dir, 'LLAMAS_red_2_A_traces.pkl')
    assert count_trace_fibres(str(master_dir)) is True


def test_count_trace_fibres_fails_an_empty_directory(dirs):
    _, master_dir = dirs
    assert count_trace_fibres(str(master_dir)) is False


# --- confirm_unusable_master_traces --------------------------------------------

def answers(*replies):
    """A scripted ``input_fn`` that also records how often it was asked."""
    replies = list(replies)
    asked = []

    def input_fn():
        asked.append(1)
        return replies.pop(0)
    input_fn.asked = asked
    return input_fn


@pytest.fixture
def bad_trace(tmp_path):
    path = tmp_path / 'LLAMAS_red_2_A_traces.pkl'
    path.write_bytes(b'bad')
    return str(path)


def entries(own):
    return [('red', '2', 'A', 'mastercalib has 297 fibres', own),
            ('blue', '1', 'A', 'mastercalib fails the registration check', None)]


def test_nothing_unusable_asks_nothing():
    input_fn = answers()
    confirm_unusable_master_traces([], interactive=True, input_fn=input_fn, stream=io.StringIO())
    assert input_fn.asked == []


def test_no_terminal_terminates_with_nothing_changed(bad_trace):
    out = io.StringIO()
    with pytest.raises(RuntimeError, match='red2A, blue1A'):
        confirm_unusable_master_traces(entries(bad_trace), interactive=False, stream=out)
    assert os.path.exists(bad_trace)
    assert 'red2A: mastercalib has 297 fibres' in out.getvalue()


def test_terminate_stops_the_run_with_nothing_changed(bad_trace):
    with pytest.raises(SystemExit, match='Terminated by user'):
        confirm_unusable_master_traces(entries(bad_trace), interactive=True,
                                       input_fn=answers('t'), stream=io.StringIO())
    assert os.path.exists(bad_trace) and not os.path.exists(bad_trace + '.legacy')


def test_proceed_sets_the_bad_trace_aside(bad_trace):
    confirm_unusable_master_traces(entries(bad_trace), interactive=True,
                                   input_fn=answers('p'), stream=io.StringIO())
    assert not os.path.exists(bad_trace) and os.path.exists(bad_trace + '.legacy')


def test_unclear_answer_asks_again(bad_trace):
    input_fn = answers('maybe', '', 'P')
    confirm_unusable_master_traces(entries(bad_trace), interactive=True,
                                   input_fn=input_fn, stream=io.StringIO())
    assert len(input_fn.asked) == 3
    assert os.path.exists(bad_trace + '.legacy')


def test_fingerprints_build_from_an_unprefixed_bundle(dirs, tmp_path):
    _, master_dir = dirs
    save(legacy_2a(LIVE_2A), master_dir, 'LLAMAS_red_2_A_traces.pkl')
    out = tmp_path / 'fingerprints.json'
    fingerprints = build_trace_fingerprints(str(master_dir), out=str(out))
    assert len(fingerprints['red']['2A']) == 298
    assert out.exists()


def test_fingerprints_refuse_to_write_an_empty_file(dirs, tmp_path):
    _, master_dir = dirs
    out = tmp_path / 'fingerprints.json'
    with pytest.raises(FileNotFoundError):
        build_trace_fingerprints(str(master_dir), out=str(out))
    assert not out.exists()



def test_reporter_holds_the_spinner_while_asking():
    # The spinner redraws its line every 0.12 s; it must not draw over the question.
    stream = io.StringIO()
    reporter = PipelineReporter(enabled=True, stream=stream)
    reporter.tty, reporter._active = True, True          # as on a real terminal
    with reporter.paused():
        before = stream.getvalue()
        reporter._redraw()
        assert stream.getvalue() == before
    assert stream.getvalue() != before                   # redrawn on resume


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
