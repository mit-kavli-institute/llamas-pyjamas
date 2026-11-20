"""Unit tests for clearing trace pickles before they are regenerated.

A camera that fails to trace writes no pickle, so the previous run's file
survives. ``validate_and_fix_trace_fibres`` only checks the fibre COUNT, so a
stale but correctly-counted trace is accepted as valid and the mastercalib
fallback never engages -- a re-run then silently reuses a misregistered trace.

The guard that matters most here is the one protecting ``CALIB_DIR``:
``run_ray_tracing`` defaults to ``outpath=CALIB_DIR, is_master_calib=True``, so
an unguarded clear would wipe the shipped mastercalib set before tracing.

Runnable with pytest or as a plain script
(``python -m llamas_pyjamas.Tests.test_clear_stale_traces``).
"""

import os

import pytest

from llamas_pyjamas.config import CALIB_DIR
from llamas_pyjamas.Trace.traceLlamasMaster import camera_from_header, clear_stale_traces


def _touch(directory, name):
    path = os.path.join(directory, name)
    with open(path, 'wb') as fh:
        fh.write(b'x')
    return path


def test_camera_from_header_colour_convention():
    assert camera_from_header({'COLOR': 'Green', 'BENCH': 3, 'SIDE': 'B'}) == ('green', '3', 'B')


def test_camera_from_header_camname_convention():
    assert camera_from_header({'CAM_NAME': '3B_green'}) == ('green', '3', 'B')


def test_removes_only_the_cameras_being_retraced(tmp_path):
    out = tmp_path / 'traces'
    out.mkdir()
    _touch(str(out), 'LLAMAS_green_3_B_traces.pkl')
    _touch(str(out), 'LLAMAS_green_1_A_traces.pkl')
    _touch(str(out), 'LLAMAS_red_2_A_traces.pkl')      # different channel, untouched

    removed = clear_stale_traces(
        [{'COLOR': 'green', 'BENCH': '3', 'SIDE': 'B'},
         {'COLOR': 'green', 'BENCH': '1', 'SIDE': 'A'}],
        str(out), is_master_calib=False)

    assert sorted(removed) == ['LLAMAS_green_1_A_traces.pkl', 'LLAMAS_green_3_B_traces.pkl']
    assert not (out / 'LLAMAS_green_3_B_traces.pkl').exists()
    assert not (out / 'LLAMAS_green_1_A_traces.pkl').exists()
    assert (out / 'LLAMAS_red_2_A_traces.pkl').exists()


def test_master_calib_naming(tmp_path):
    out = tmp_path / 'traces'
    out.mkdir()
    _touch(str(out), 'LLAMAS_master_blue_4_B_traces.pkl')
    _touch(str(out), 'LLAMAS_blue_4_B_traces.pkl')     # user-form, not the target here

    removed = clear_stale_traces([{'COLOR': 'blue', 'BENCH': '4', 'SIDE': 'B'}],
                                 str(out), is_master_calib=True)

    assert removed == ['LLAMAS_master_blue_4_B_traces.pkl']
    assert (out / 'LLAMAS_blue_4_B_traces.pkl').exists()


def test_never_clears_the_shipped_mastercalib_directory():
    """The load-bearing guard: run_ray_tracing defaults to outpath=CALIB_DIR."""
    removed = clear_stale_traces([{'COLOR': 'green', 'BENCH': '3', 'SIDE': 'B'}],
                                 CALIB_DIR, is_master_calib=True)
    assert removed == []


def test_missing_files_and_bad_headers_are_tolerated(tmp_path):
    out = tmp_path / 'traces'
    out.mkdir()
    removed = clear_stale_traces(
        [{'COLOR': 'green', 'BENCH': '3', 'SIDE': 'B'},   # nothing to remove
         {'NOTAHEADER': 1}],                              # unparseable, skipped
        str(out), is_master_calib=False)
    assert removed == []


def test_no_outpath_is_a_noop(tmp_path):
    assert clear_stale_traces([{'COLOR': 'green', 'BENCH': '3', 'SIDE': 'B'}], None) == []


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
