"""Unit tests for locating a mastercalib trace pickle.

Two filename conventions are in circulation: the mastercalib bundle ships
``LLAMAS_{channel}_{bench}_{side}_traces.pkl``, while traces written by
``run_ray_tracing(is_master_calib=True)`` carry a ``LLAMAS_master_`` prefix.
Every lookup has to accept both, or the fallback is unreachable against a stock
bundle and a camera that fails to trace silently becomes blank RSS rows.

Runnable with pytest or as a plain script
(``python -m llamas_pyjamas.Tests.test_mastercalib_fallback``).
"""

import os
import pickle

import numpy as np
import pytest

from llamas_pyjamas.Utils.utils import copy_mastercalib_trace, find_trace_pickle


class _Trace:
    """Minimal trace pickle from the current tracer (``comb_ok``): only its fibre
    count is checked before it may be copied in as a fallback."""

    comb_ok = True

    def __init__(self, nfibers, tag):
        self.traces = np.zeros((nfibers, 4))
        self.tag = tag


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
    """A bundle holding only the unprefixed name must still work, or the fallback
    is skipped and the camera ends up as blank fibres downstream."""
    src = tmp_path / 'mastercalib'
    src.mkdir()
    dst = tmp_path / 'traces'
    dst.mkdir()
    payload = pickle.dumps(_Trace(298, 'mastercalib'))
    _touch(str(src), 'LLAMAS_blue_1_A_traces.pkl', payload)

    copied = copy_mastercalib_trace('blue', '1', 'A', str(src), str(dst))

    # Always written under the USER convention, whatever the source was called.
    assert os.path.basename(copied) == 'LLAMAS_blue_1_A_traces.pkl'
    with open(copied, 'rb') as fh:
        assert fh.read() == payload


def test_copy_from_a_master_prefixed_bundle_succeeds(tmp_path):
    src = tmp_path / 'mastercalib'
    src.mkdir()
    dst = tmp_path / 'traces'
    dst.mkdir()
    payload = pickle.dumps(_Trace(300, 'prefixed'))
    _touch(str(src), 'LLAMAS_master_blue_4_A_traces.pkl', payload)

    copied = copy_mastercalib_trace('blue', '4', 'A', str(src), str(dst))

    assert os.path.basename(copied) == 'LLAMAS_blue_4_A_traces.pkl'
    with open(copied, 'rb') as fh:
        assert fh.read() == payload


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
