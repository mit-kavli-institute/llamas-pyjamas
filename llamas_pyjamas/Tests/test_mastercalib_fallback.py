"""Unit tests for locating a mastercalib trace pickle.

Two filename conventions are in circulation: the mastercalib bundle ships
``LLAMAS_{channel}_{bench}_{side}_traces.pkl``, while traces written by
``run_ray_tracing(is_master_calib=True)`` carry a ``LLAMAS_master_`` prefix.
Every lookup has to accept both. Checking only the prefixed form made the
fallback unreachable against a stock bundle, so a camera that failed to trace
silently produced no trace at all rather than mastercalib data -- which is how
blue 1A and 4A became 598 blank rows in the RSS on 2026-08-31.

Runnable with pytest or as a plain script
(``python -m llamas_pyjamas.Tests.test_mastercalib_fallback``).
"""

import os

import pytest

from llamas_pyjamas.Utils.utils import copy_mastercalib_trace, find_trace_pickle


def _touch(directory, name, payload=b'trace'):
    path = os.path.join(directory, name)
    with open(path, 'wb') as fh:
        fh.write(payload)
    return path


def test_finds_the_bundle_filename(tmp_path):
    """The shipped mastercalib name, with no 'master_' prefix."""
    src = tmp_path / 'mastercalib'
    src.mkdir()
    want = _touch(str(src), 'LLAMAS_green_3_B_traces.pkl')
    assert find_trace_pickle('green', '3', 'B', str(src)) == want


def test_finds_the_master_prefixed_filename(tmp_path):
    """Traces generated locally as master calibration."""
    src = tmp_path / 'mastercalib'
    src.mkdir()
    want = _touch(str(src), 'LLAMAS_master_green_3_B_traces.pkl')
    assert find_trace_pickle('green', '3', 'B', str(src)) == want


def test_missing_trace_names_both_attempts(tmp_path):
    src = tmp_path / 'mastercalib'
    src.mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        find_trace_pickle('green', '3', 'B', str(src))
    message = str(excinfo.value)
    assert 'LLAMAS_master_green_3_B_traces.pkl' in message
    assert 'LLAMAS_green_3_B_traces.pkl' in message


def test_copy_from_a_stock_bundle_succeeds(tmp_path):
    """The regression: a bundle holding only the unprefixed name must still work.

    Against the old code this raised FileNotFoundError, the fallback was skipped,
    and the camera ended up as blank fibres downstream.
    """
    src = tmp_path / 'mastercalib'
    src.mkdir()
    dst = tmp_path / 'traces'
    dst.mkdir()
    _touch(str(src), 'LLAMAS_blue_1_A_traces.pkl', b'mastercalib-payload')

    copied = copy_mastercalib_trace('blue', '1', 'A', str(src), str(dst))

    # Always written under the USER convention, whatever the source was called.
    assert os.path.basename(copied) == 'LLAMAS_blue_1_A_traces.pkl'
    with open(copied, 'rb') as fh:
        assert fh.read() == b'mastercalib-payload'


def test_copy_from_a_master_prefixed_bundle_succeeds(tmp_path):
    src = tmp_path / 'mastercalib'
    src.mkdir()
    dst = tmp_path / 'traces'
    dst.mkdir()
    _touch(str(src), 'LLAMAS_master_blue_4_A_traces.pkl', b'prefixed-payload')

    copied = copy_mastercalib_trace('blue', '4', 'A', str(src), str(dst))

    assert os.path.basename(copied) == 'LLAMAS_blue_4_A_traces.pkl'
    with open(copied, 'rb') as fh:
        assert fh.read() == b'prefixed-payload'


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
