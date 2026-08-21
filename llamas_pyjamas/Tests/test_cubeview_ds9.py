"""Unit tests for the CubeViewer DS9/XPA transport.

These cover the parts that can be pinned down without a running DS9: tool discovery,
reply parsing, command construction, and the in-memory FITS piping. A fake pair of
xpaget/xpaset executables stands in for the real XPA clients, which records the argv and
stdin it was handed — so the bytes CubeViewer would send to DS9 are asserted directly.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.Tests.test_cubeview_ds9`).
"""

import json
import os
import stat
import tempfile

import numpy as np
from astropy.io import fits

from llamas_pyjamas.CubeViewer.cubeViewDS9 import (
    DS9,
    DS9Error,
    find_xpa_tools,
    parse_coordinates,
    parse_xpans_line,
)

# A fake XPA client: records argv + stdin to a JSON file next to itself, then replies with
# whatever REPLY contains. Stands in for xpaget/xpaset/xpaaccess.
FAKE_TOOL = """#!/usr/bin/env python3
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
data = sys.stdin.buffer.read() if not sys.stdin.isatty() else b""
record = {"argv": sys.argv[1:], "stdin_len": len(data), "stdin_head": data[:80].decode("latin-1")}
with open(os.path.join(here, os.path.basename(sys.argv[0]) + ".call.json"), "w") as fh:
    json.dump(record, fh)
reply_path = os.path.join(here, os.path.basename(sys.argv[0]) + ".reply")
if os.path.exists(reply_path):
    sys.stdout.write(open(reply_path).read())
status_path = os.path.join(here, os.path.basename(sys.argv[0]) + ".status")
sys.exit(int(open(status_path).read().strip()) if os.path.exists(status_path) else 0)
"""


def _make_fake_xpa(directory, replies=None, statuses=None):
    """Write fake xpaset/xpaget/xpaaccess into `directory`; return that directory.

    `statuses` sets a tool's exit code, which matters because xpaaccess reports its match
    *count* through the exit status rather than success/failure.
    """
    for tool in ('xpaset', 'xpaget', 'xpaaccess'):
        path = os.path.join(directory, tool)
        with open(path, 'w') as fh:
            fh.write(FAKE_TOOL)
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    for tool, reply in (replies or {}).items():
        with open(os.path.join(directory, tool + '.reply'), 'w') as fh:
            fh.write(reply)
    for tool, status in (statuses or {}).items():
        with open(os.path.join(directory, tool + '.status'), 'w') as fh:
            fh.write(str(status))
    return directory


def _last_call(directory, tool):
    with open(os.path.join(directory, tool + '.call.json')) as fh:
        return json.load(fh)


def test_parse_xpans_line_valid():
    assert parse_xpans_line('DS9 ds9 gs 7f000001:60778 simcoe') == ('DS9', 'ds9')
    assert parse_xpans_line('DS9 ds9a gs 7f000001:1234 simcoe') == ('DS9', 'ds9a')


def test_parse_xpans_line_rejects_blank_and_short():
    assert parse_xpans_line('') is None
    assert parse_xpans_line('   ') is None
    assert parse_xpans_line('DS9') is None


def test_parse_coordinates_basic():
    assert parse_coordinates('123.5 88.0') == (123.5, 88.0)
    assert parse_coordinates('  1 2  ') == (1.0, 2.0)
    # DS9 sometimes appends the coordinate system; extra fields must not break parsing.
    assert parse_coordinates('10 20 image') == (10.0, 20.0)


def test_parse_coordinates_rejects_garbage():
    for bad in ('', 'nope', 'x y'):
        try:
            parse_coordinates(bad)
        except DS9Error:
            continue
        raise AssertionError(f'expected DS9Error for {bad!r}')


def test_find_xpa_tools_missing_raises_with_hint():
    with tempfile.TemporaryDirectory() as empty:
        os.environ['CUBEVIEWER_XPA_DIR'] = empty
        saved_path = os.environ.get('PATH', '')
        os.environ['PATH'] = empty          # hide any real tools
        try:
            find_xpa_tools()
        except DS9Error as exc:
            assert 'xpaset' in str(exc)
            assert 'github.com/ericmandel/xpa' in str(exc), 'error must be actionable'
        else:
            raise AssertionError('expected DS9Error when tools are absent')
        finally:
            os.environ['PATH'] = saved_path
            os.environ.pop('CUBEVIEWER_XPA_DIR', None)


def test_find_xpa_tools_honours_env_dir():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_xpa(tmp)
        os.environ['CUBEVIEWER_XPA_DIR'] = tmp
        try:
            tools = find_xpa_tools()
            assert tools['xpaget'] == os.path.join(tmp, 'xpaget')
            assert tools['xpaset'] == os.path.join(tmp, 'xpaset')
        finally:
            os.environ.pop('CUBEVIEWER_XPA_DIR', None)


def test_set_uses_dash_p_when_no_data():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_xpa(tmp)
        os.environ['CUBEVIEWER_XPA_DIR'] = tmp
        try:
            DS9().set('scale zscale')
            argv = _last_call(tmp, 'xpaset')['argv']
            assert argv == ['-p', 'ds9', 'scale', 'zscale']
        finally:
            os.environ.pop('CUBEVIEWER_XPA_DIR', None)


def test_get_builds_expected_argv_and_returns_reply():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_xpa(tmp, replies={'xpaget': '123.5 88.0\n'})
        os.environ['CUBEVIEWER_XPA_DIR'] = tmp
        try:
            assert DS9().get('crosshair image') == '123.5 88.0'
            assert _last_call(tmp, 'xpaget')['argv'] == ['ds9', 'crosshair', 'image']
            # and the typed accessor parses it
            assert DS9().crosshair() == (123.5, 88.0)
        finally:
            os.environ.pop('CUBEVIEWER_XPA_DIR', None)


def test_set_fits_pipes_real_fits_bytes_and_never_touches_disk():
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_xpa(tmp)
        os.environ['CUBEVIEWER_XPA_DIR'] = tmp
        try:
            image = np.arange(12, dtype=np.float32).reshape(3, 4)
            hdul = fits.HDUList([fits.PrimaryHDU(image)])
            before = set(os.listdir(tmp))
            DS9().set_fits(hdul, frame=1)

            call = _last_call(tmp, 'xpaset')
            assert call['argv'] == ['ds9', 'fits'], 'FITS load must not use -p (needs stdin)'
            assert call['stdin_head'].startswith('SIMPLE  ='), 'must pipe real FITS bytes'
            assert call['stdin_len'] % 2880 == 0, 'FITS blocks are 2880 bytes'
            # No FITS file should have been created anywhere by the transport.
            new = set(os.listdir(tmp)) - before
            assert not any(n.endswith('.fits') for n in new), f'wrote to disk: {new}'
        finally:
            os.environ.pop('CUBEVIEWER_XPA_DIR', None)


def test_is_alive_matches_real_xpaaccess_exit_semantics():
    # Regression. xpaaccess reports the MATCH COUNT via its exit status, so a running DS9
    # exits 1 (one match) and an absent one exits 0 -- inverted from the shell convention.
    # Verified against xpa 2.1.20: `xpaaccess ds9` -> "yes", exit=1;
    # `xpaaccess -n nosuchthing` -> "0", exit=0. Treating exit!=0 as failure made is_alive()
    # raise exactly when DS9 *was* running.
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_xpa(tmp, replies={'xpaaccess': '1\n'}, statuses={'xpaaccess': 1})
        os.environ['CUBEVIEWER_XPA_DIR'] = tmp
        try:
            assert DS9().is_alive() is True, 'exit=1 means one match, not an error'
            assert _last_call(tmp, 'xpaaccess')['argv'] == ['-n', 'ds9']
        finally:
            os.environ.pop('CUBEVIEWER_XPA_DIR', None)

    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_xpa(tmp, replies={'xpaaccess': '0\n'}, statuses={'xpaaccess': 0})
        os.environ['CUBEVIEWER_XPA_DIR'] = tmp
        try:
            assert DS9().is_alive() is False, 'zero matches means no DS9'
        finally:
            os.environ.pop('CUBEVIEWER_XPA_DIR', None)


def test_is_alive_falls_back_to_yes_no():
    # Older xpaaccess builds without -n answer "yes"/"no" on stdout.
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_xpa(tmp, replies={'xpaaccess': 'yes\n'}, statuses={'xpaaccess': 1})
        os.environ['CUBEVIEWER_XPA_DIR'] = tmp
        try:
            assert DS9().is_alive() is True
        finally:
            os.environ.pop('CUBEVIEWER_XPA_DIR', None)


def test_get_and_set_still_treat_nonzero_as_failure():
    # Only xpaaccess has the odd exit semantics; a genuinely failing xpaget must still raise.
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_xpa(tmp, replies={'xpaget': ''}, statuses={'xpaget': 1})
        os.environ['CUBEVIEWER_XPA_DIR'] = tmp
        try:
            DS9().get('version')
        except DS9Error:
            pass
        else:
            raise AssertionError('a failing xpaget must raise')
        finally:
            os.environ.pop('CUBEVIEWER_XPA_DIR', None)


def test_targets_parses_xpans_listing():
    listing = ('DS9 ds9 gs 7f000001:60778 simcoe\n'
               'DS9 ds9b gs 7f000001:60999 simcoe\n')
    with tempfile.TemporaryDirectory() as tmp:
        _make_fake_xpa(tmp, replies={'xpaget': listing})
        os.environ['CUBEVIEWER_XPA_DIR'] = tmp
        try:
            assert DS9().targets() == [('DS9', 'ds9'), ('DS9', 'ds9b')]
        finally:
            os.environ.pop('CUBEVIEWER_XPA_DIR', None)


def test_construction_does_not_require_ds9_or_tools():
    # The spectrum panel must work without DS9, so constructing DS9() must never probe.
    saved = os.environ.pop('CUBEVIEWER_XPA_DIR', None)
    try:
        ds9 = DS9(target='nonexistent')
        assert ds9.target == 'nonexistent'
    finally:
        if saved is not None:
            os.environ['CUBEVIEWER_XPA_DIR'] = saved


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
