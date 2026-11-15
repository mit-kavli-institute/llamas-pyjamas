"""Unit tests for the disk-space work of 2026-11-14.

- Trace pickles must not carry the master-bias cache (``Bias/biasCache.py``), and old ones that
  do are slimmed (``traceValidation.slim_bloated_traces``).
- A full disk surfaces as itself (``GUI_extract`` raises; ``diskSpace.is_disk_full``), and the
  fallbacks that tolerate a failed extraction never swallow it.
- The pre-flight estimate and check (``diskSpace``, ``reduce.check_run_disk_space``).
- No ``_trimmed.fits`` side copies; ``output_dir`` defaults to the night.
"""
import errno
import os
import pickle
import subprocess
import sys
import textwrap
from collections import namedtuple

import cloudpickle
import numpy as np
import pytest
from astropy.io import fits
from ray.exceptions import ObjectStoreFullError, OutOfDiskError, RayTaskError

import llamas_pyjamas  # noqa: F401  (registers traceLlamasMaster for pickling by value)
import llamas_pyjamas.GUI.guiExtract as ge
import llamas_pyjamas.reduce as reduce
import llamas_pyjamas.Utils.diskSpace as ds
from llamas_pyjamas.Bias import biasCache
from llamas_pyjamas.File.llamasIO import process_fits_by_color
from llamas_pyjamas.Trace import traceLlamasMaster as tlm
from llamas_pyjamas.Trace.traceValidation import frozen_bias_cache_bytes, slim_bloated_traces
from llamas_pyjamas.Utils.userPrompt import NoTerminalError, ask_proceed_or_terminate

GB = 1e9
Usage = namedtuple('Usage', 'total used free')


def answers(*replies):
    replies = list(replies)
    asked = []

    def input_fn():
        asked.append(1)
        if not replies:
            raise EOFError
        return replies.pop(0)
    input_fn.asked = asked
    return input_fn


# --- trace pickles and the bias cache ---------------------------------------------------------

def test_trace_pickle_does_not_carry_the_bias_cache(monkeypatch):
    monkeypatch.setattr(biasCache, '_BIAS_HDU_CACHE', {'bias': (0, [np.zeros((2048, 2048))])})
    trace = tlm.TraceLlamas.__new__(tlm.TraceLlamas)
    trace.__dict__.update(traces=np.zeros((10, 10)))
    blob = cloudpickle.dumps(trace)
    assert len(blob) < 1e6                         # the cache alone is 33.6 MB
    assert frozen_bias_cache_bytes(pickle.loads(blob)) == 0


# Writes, in a fresh interpreter, trace pickles made the way the old code made them: the
# module held the cache and a method reached it.
_WRITE_BLOATED = textwrap.dedent('''
    import sys
    import cloudpickle, numpy as np
    from astropy.io import fits
    import llamas_pyjamas
    from llamas_pyjamas.Trace import traceLlamasMaster as tlm
    ns = {}
    exec("def old_grab(*a, **k):\\n    return _BIAS_HDU_CACHE\\n", tlm.__dict__, ns)
    tlm._grab_bias_hdu = ns['old_grab']
    tlm._BIAS_HDU_CACHE = {'bias': (0, fits.HDUList([fits.PrimaryHDU(),
                                                     fits.ImageHDU(np.ones((1000, 1000)))]))}
    for name, current in (('LLAMAS_green_1_A_traces.pkl', True),
                          ('LLAMAS_red_1_A_traces.pkl', False)):
        t = tlm.TraceLlamas.__new__(tlm.TraceLlamas)
        t.traces = np.arange(20.0).reshape(4, 5)
        t.fiberimg = np.arange(30).reshape(5, 6)
        if current:
            t.comb_ok = True
        with open(f"{sys.argv[1]}/{name}", 'wb') as f:
            cloudpickle.dump(t, f)
''')

_CHECK_SLIM = textwrap.dedent('''
    import sys, pickle
    import numpy as np
    t = pickle.load(open(f"{sys.argv[1]}/LLAMAS_green_1_A_traces.pkl", 'rb'))
    assert type(t).__name__ == 'TraceLlamas' and t.comb_ok
    assert np.array_equal(t.traces, np.arange(20.0).reshape(4, 5))
    assert np.array_equal(t.fiberimg, np.arange(30).reshape(5, 6))
    print('ok')
''')


def test_slim_bloated_traces(tmp_path):
    traces, master = tmp_path / 'traces', tmp_path / 'mastercalib'
    traces.mkdir()
    master.mkdir()
    subprocess.run([sys.executable, '-c', _WRITE_BLOATED, str(traces)], check=True)
    green, red = traces / 'LLAMAS_green_1_A_traces.pkl', traces / 'LLAMAS_red_1_A_traces.pkl'
    assert green.stat().st_size > 8e6 and red.stat().st_size > 8e6

    slimmed = slim_bloated_traces(str(traces), str(master), min_bytes=1e6)

    assert [os.path.basename(p) for p, _old, _new in slimmed] == [green.name]
    assert green.stat().st_size < 1e6
    assert red.stat().st_size > 8e6                # old format: re-traced, never rewritten
    out = subprocess.run([sys.executable, '-c', _CHECK_SLIM, str(traces)], check=True,
                         capture_output=True, text=True)
    assert out.stdout.strip() == 'ok'
    assert slim_bloated_traces(str(traces), str(master), min_bytes=1e6) == []


def test_slim_leaves_a_master_copy_alone(tmp_path):
    traces, master = tmp_path / 'traces', tmp_path / 'mastercalib'
    traces.mkdir()
    master.mkdir()
    subprocess.run([sys.executable, '-c', _WRITE_BLOATED, str(traces)], check=True)
    green = traces / 'LLAMAS_green_1_A_traces.pkl'
    (master / 'LLAMAS_master_green_1_A_traces.pkl').write_bytes(green.read_bytes())

    assert slim_bloated_traces(str(traces), str(master), min_bytes=1e6) == []
    assert green.stat().st_size > 8e6


# --- full-disk errors -------------------------------------------------------------------------

def test_is_disk_full():
    wrapped = RayTaskError('process_trace', 'tb', OutOfDiskError('full')).as_instanceof_cause()
    for exc in (OutOfDiskError('full'), ObjectStoreFullError('full'), wrapped,
                OSError(errno.ENOSPC, 'No space left on device')):
        assert ds.is_disk_full(exc)
    for exc in (OSError(errno.EACCES, 'denied'), ValueError('x'), TypeError('x')):
        assert not ds.is_disk_full(exc)


def test_gui_extract_raises_instead_of_returning_none(monkeypatch):
    def full(file):
        raise OutOfDiskError('Local disk is full')
    monkeypatch.setattr('llamas_pyjamas.DataModel.validate.validate_for_gui', full)
    with pytest.raises(OutOfDiskError):
        ge.GUI_extract('frame.fits')


def _flat_for_rss(monkeypatch, tmp_path, error):
    monkeypatch.setattr(reduce, 'apply_flat_field_correction',
                        lambda *a, **k: (str(tmp_path / 'flat_corrected.fits'), None))

    def extract(*a, **k):
        raise error
    monkeypatch.setattr(reduce, 'run_extraction', extract)
    return reduce._process_flat_for_rss(['twilight.fits'], ['maps.fits'], str(tmp_path),
                                        str(tmp_path), None, 'ts', label='twilight')


def test_twilight_fallback_continues_on_an_ordinary_error(monkeypatch, tmp_path):
    assert _flat_for_rss(monkeypatch, tmp_path, ValueError('bad trace')) == []


def test_twilight_fallback_never_swallows_a_full_disk(monkeypatch, tmp_path):
    with pytest.raises(OutOfDiskError):
        _flat_for_rss(monkeypatch, tmp_path, OutOfDiskError('full'))


def test_disk_full_message(tmp_path):
    msg = ds.disk_full_message([str(tmp_path)], ray_folders=(str(tmp_path),))
    assert msg.startswith('out of disk space: ') and 'GB free on' in msg and 'Ray stops' in msg


# --- the prompt -------------------------------------------------------------------------------

@pytest.mark.parametrize('replies, expected, asked', [(('p',), True, 1), (('t',), False, 1),
                                                      (('maybe', 'proceed'), True, 2), ((), False, 1)])
def test_ask_proceed_or_terminate(replies, expected, asked):
    import io
    input_fn = answers(*replies)
    assert ask_proceed_or_terminate('Go on?', interactive=True, input_fn=input_fn,
                                    stream=io.StringIO()) is expected
    assert len(input_fn.asked) == asked


def test_ask_without_a_terminal_raises():
    with pytest.raises(NoTerminalError):
        ask_proceed_or_terminate('Go on?', interactive=False)


# --- the estimate and the check ---------------------------------------------------------------

def _raw(path, nbytes=1_000_000):
    with open(path, 'wb') as f:
        f.truncate(nbytes)
    return str(path)


@pytest.fixture
def run(tmp_path):
    """Folders on one volume, two science frames, a flat, a twilight and an arc."""
    folders = {k: str(tmp_path / k) for k in ('output', 'bias_corrected', 'traces', 'extractions',
                                              'flat', 'arcs', 'lut', 'ray')}
    raw = tmp_path / 'raw'
    raw.mkdir()
    frames = {_raw(raw / 'LLAMAS_2026-10-16_00-31-08.4_SCI24_mef.fits'): 'science',
              _raw(raw / 'LLAMAS_2026-10-16_00-54-54.4_SCI24_mef.fits'): 'science',
              _raw(raw / 'LLAMAS_2026-10-15_16-39-02.9_CAL24_mef.fits'): 'flat',
              _raw(raw / 'LLAMAS_2026-10-15_23-12-49.3_CAL24_mef.fits'): 'twilight',
              _raw(raw / 'LLAMAS_2026-10-15_16-41-56.2_CAL24_mef.fits'): 'arc'}
    return folders, frames


def _disk(monkeypatch, free, total=1000 * GB):
    monkeypatch.setattr(ds.shutil, 'disk_usage', lambda path: Usage(total, total - free, free))


def test_estimate_sizes(run):
    folders, frames = run
    e = ds.N_CAMERAS * ds.TRACE_PICKLE_BYTES * ds.EXTRACTION_OVER_TRACES
    items = ds.estimate_disk_use(frames, folders, lambda kind, raw: False,
                                 {'sky_subtract': True}, object_store_bytes=4 * GB)
    science = [i for i in items if i.group == 'science' and i.label.startswith('flat-corrected')]
    assert len(science) == 2
    assert science[0].nbytes == pytest.approx(2e6 + e + 9 * ds.RSS_COLOUR_BYTES + ds.CR_MASK_BYTES)
    assert any(i.label.startswith('frame being processed') and i.nbytes == pytest.approx(2 * e)
               for i in items)
    ray = [i for i in items if i.group == 'ray']
    assert ray[0].nbytes == pytest.approx(4 * GB + e) and ray[0].folder == folders['ray']
    assert sum(i.group == 'twilight' for i in items) == 3   # bias copy + two stages

    cubes = ds.estimate_disk_use(frames, folders, lambda kind, raw: False,
                                 {'generate_cubes': True, 'generate_standard_cubes': True,
                                  'generate_skysub_cubes': True})
    with_cubes = [i for i in cubes if i.label.startswith('flat-corrected')
                  and i.frame == science[0].frame][0]
    assert with_cubes.nbytes - science[0].nbytes == pytest.approx(6 * ds.CUBE_COLOUR_BYTES)
    no_master = ds.estimate_disk_use(frames, folders, lambda kind, raw: False,
                                     {'generate_standard_cubes': True}, object_store_bytes=4 * GB)
    assert [i.nbytes for i in no_master] == [i.nbytes for i in items]  # generate_cubes is off


def test_estimate_uses_the_real_trace_sizes(run):
    folders, frames = run
    os.makedirs(folders['traces'])
    for n in range(24):
        _raw(os.path.join(folders['traces'], f'LLAMAS_x_{n}_traces.pkl'), 10_000_000)
    assert ds.extraction_bytes(folders['traces']) == pytest.approx(24 * 10e6 * 1.05)


def test_done_products_cost_nothing(run):
    folders, frames = run
    sci = sorted(r for r, role in frames.items() if role == 'science')
    items = ds.estimate_disk_use(frames, folders,
                                 lambda kind, raw: kind == 'science' and raw == sci[0], {})
    done = [i for i in items if i.done]
    assert [i.frame for i in done] == [sci[0]]


def _shares(items):
    """(calibration share, whole need with the 1.2 margin) of a list of items."""
    calib = sum(i.nbytes for i in items if i.group in ds.CALIBRATION_GROUPS)
    return calib, 1.2 * sum(i.nbytes for i in items)


# usable space (free - 50 GB Ray reserve): above the need, between, or below the calibrations
@pytest.mark.parametrize('usable, verdict', [(lambda c, n: n + GB, 'ok'),
                                             (lambda c, n: (c + n) / 2, 'ask'),
                                             (lambda c, n: c / 2, 'stop')])
def test_verdicts(monkeypatch, run, usable, verdict):
    folders, frames = run
    items = ds.estimate_disk_use(frames, folders, lambda kind, raw: False, {},
                                 object_store_bytes=4 * GB)
    _disk(monkeypatch, 50 * GB + usable(*_shares(items)))
    (check,) = ds.check_disk_space(items, {})
    assert check.ray and check.reserve == pytest.approx(50 * GB)   # Ray: 5% of 1000 GB
    assert check.need == pytest.approx(1.2 * sum(i.nbytes for i in items))
    assert check.verdict == verdict


def test_ray_threshold_override_and_other_volumes(monkeypatch, run):
    folders, frames = run
    _disk(monkeypatch, 100 * GB)
    monkeypatch.setenv('RAY_local_fs_capacity_threshold', '0.99')
    items = ds.estimate_disk_use(frames, folders, lambda kind, raw: False, {})
    (check,) = ds.check_disk_space(items, {})
    assert check.reserve == pytest.approx(10 * GB)
    no_ray = [i for i in items if i.group != 'ray']
    (check,) = ds.check_disk_space(no_ray, {'min_free_gb': 7})
    assert not check.ray and check.reserve == pytest.approx(7 * GB)


def test_gate_asks_and_respects_the_answer(monkeypatch, run):
    import io
    folders, frames = run
    items = ds.estimate_disk_use(frames, folders, lambda kind, raw: False, {})
    calib, need = _shares(items)
    _disk(monkeypatch, 50 * GB + (calib + need) / 2)
    out = io.StringIO()
    ds.disk_space_gate(items, {}, interactive=True, input_fn=answers('p'), stream=out)
    assert 'NOT ENOUGH' in out.getvalue() and 'skip_disk_check' in out.getvalue()
    with pytest.raises(ds.DiskSpaceError, match='terminated by user'):
        ds.disk_space_gate(items, {}, interactive=True, input_fn=answers('t'),
                           stream=io.StringIO())
    with pytest.raises(ds.DiskSpaceError, match='no terminal to ask'):
        ds.disk_space_gate(items, {}, interactive=False, stream=io.StringIO())


def test_gate_stops_without_asking_when_calibrations_do_not_fit(monkeypatch, run):
    import io
    folders, frames = run
    items = ds.estimate_disk_use(frames, folders, lambda kind, raw: False, {})
    _disk(monkeypatch, 50 * GB + _shares(items)[0] / 2)
    input_fn = answers('p')
    with pytest.raises(ds.DiskSpaceError, match='calibrations'):
        ds.disk_space_gate(items, {}, interactive=True, input_fn=input_fn, stream=io.StringIO())
    assert not input_fn.asked


def test_gate_is_quiet_when_there_is_room(monkeypatch, run):
    import io
    folders, frames = run
    _disk(monkeypatch, 900 * GB)
    out = io.StringIO()
    items = ds.estimate_disk_use(frames, folders, lambda kind, raw: False, {})
    ds.disk_space_gate(items, {}, interactive=False, stream=out)
    assert out.getvalue() == ''


# --- reduce: products on disk, output_dir default, the whole check ----------------------------

def test_products_done(tmp_path, run):
    folders, frames = run
    sci = sorted(r for r, role in frames.items() if role == 'science')[0]
    stem = os.path.splitext(os.path.basename(sci))[0]
    for k in ('bias_corrected', 'extractions', 'flat'):
        os.makedirs(folders[k])
    open(os.path.join(folders['bias_corrected'], f'{stem}_bias_corrected.fits'), 'w').close()
    open(os.path.join(folders['extractions'], 'LLAMAS_2026-10-16_00-31-08.4_RSS_red.fits'),
         'w').close()
    for name in ('pixel_maps.fits', 'flat_smooth_models.fits'):
        open(os.path.join(folders['flat'], name), 'w').close()
    done = reduce.products_done(folders)
    assert done('bias_corrected', sci) and done('science', sci) and done('flat', None)
    other = sorted(r for r, role in frames.items() if role == 'science')[1]
    assert not done('bias_corrected', other) and not done('science', other)
    assert not done('traces', None) and not done('arc', other)


def test_default_output_dir_and_checks(tmp_path, monkeypatch):
    night = tmp_path / 'night'
    night.mkdir()
    sci = _raw(night / 'LLAMAS_2026-10-16_00-31-08.4_SCI24_mef.fits')
    config = {'science_files': [sci]}
    assert reduce.default_output_dir(config) == str(night / 'reduced')
    assert reduce.output_dir_checks({**config, 'output_dir': '/x'}) == ([], [])

    old = tmp_path / 'package_reduced'
    (old / 'bias_corrected').mkdir(parents=True)
    (old / 'bias_corrected' / 'LLAMAS_2026-10-16_00-31-08.4_SCI24_mef_bias_corrected.fits').touch()
    monkeypatch.setattr(reduce, 'OLD_DEFAULT_OUTPUT_DIR', str(old))
    errors, warnings = reduce.output_dir_checks(config)
    assert not errors and len(warnings) == 1 and str(old) in warnings[0]

    night.chmod(0o500)
    try:
        errors, _ = reduce.output_dir_checks(config)
    finally:
        night.chmod(0o700)
    assert len(errors) == 1 and 'not writable' in errors[0]


def test_check_run_disk_space(tmp_path, monkeypatch):
    night = tmp_path / 'night'
    night.mkdir()
    sci = _raw(night / 'LLAMAS_2026-10-16_00-31-08.4_SCI24_mef.fits')
    flat = _raw(night / 'LLAMAS_2026-10-15_16-39-02.9_CAL24_mef.fits')
    config = {'science_files': [sci], 'green_flat_file': flat, 'blue_flat_file': flat}
    folders = reduce.disk_folders(config, str(night / 'reduced'), str(tmp_path / 'ray'))
    _disk(monkeypatch, 900 * GB)
    (check,) = reduce.check_run_disk_space(config, folders, interactive=False)
    bias_copies = [i for i in check.items if i.label == 'bias-corrected frame']
    assert len(bias_copies) == 2                    # the flat listed twice counts once
    _disk(monkeypatch, 60 * GB)
    with pytest.raises(ds.DiskSpaceError):
        reduce.check_run_disk_space(config, folders, interactive=False)


# --- leftover files ---------------------------------------------------------------------------

def test_process_fits_by_color_writes_nothing_by_default(tmp_path):
    hdu = fits.ImageHDU(np.ones((10, 12), np.int16))
    hdu.header.update(COLOR='red', BENCH='1', SIDE='A', DATASEC='[1:10,1:8]')
    path = tmp_path / 'frame_mef.fits'
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    hdus, written = process_fits_by_color(str(path))
    assert written is None and hdus[1].data.shape == (8, 10)
    assert os.listdir(tmp_path) == ['frame_mef.fits']
    hdus, written = process_fits_by_color(str(path), output_file=str(tmp_path / 'out.fits'))
    assert written == str(tmp_path / 'out.fits') and os.path.exists(written)


def test_superseded_sky1d_pickle_is_removed(tmp_path):
    paths = [str(tmp_path / n) for n in ('x_extract.pkl', 'x_sky1d_lr_extractions.pkl',
                                         'x_sky1d_extractions.pkl')]
    for p in paths:
        open(p, 'w').close()
    assert reduce.cleanup_extraction_pkls(paths, save_pkl=True) == []
    assert sorted(reduce.cleanup_extraction_pkls(paths + [None], save_pkl=False)) == sorted(paths)
