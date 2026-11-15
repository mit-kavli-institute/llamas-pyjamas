"""Unit tests for the shifted-master fallback (``Trace/masterShift.py``).

A camera that fails to trace gets its mastercalib trace moved onto the night's flat.
These tests use synthetic flats: Gaussian fibres on the fibremap comb, with a
throughput that alternates by +-10% between neighbouring fibres. That alternation is
real (the fingerprints' lag-2 autocorrelation is up to +0.77), and it is what makes a
two-fibre alias the dangerous one.

They pin down the following:
- shifts, stretches and tilts are recovered, including blue-sized ones (-34 px);
- which fibre is which comes from the slit ends, confirmed by the fingerprint;
- every case that cannot be placed with confidence is refused, never misplaced;
- the strict fingerprint rule names its config switch, and the relaxed rule accepts;
- ``validate_and_fix_trace_fibres`` writes a shifted master only on success, and never
  falls back to an unshifted copy when the camera's flat exists.
"""

import io
import os
import pickle
import subprocess
import sys

import cloudpickle
import numpy as np
import pytest
from astropy.io import fits

import llamas_pyjamas.Trace.traceValidation as traceValidation
from llamas_pyjamas.Tests.test_trace_slots import comb as slot_rows, throughput
from llamas_pyjamas.Trace.masterShift import (
    SWITCH_HINT, MasterShiftError, build_shifted_trace, measure_master_shift,
    read_flat_cameras)
from llamas_pyjamas.Trace.traceLlamasMaster import normalise_trace_heights
from llamas_pyjamas.Trace.traceValidation import (
    confirm_unusable_master_traces, validate_and_fix_trace_fibres)

NY, NX, SIGMA = 2048, 256, 1.3
PITCH = 6.45
SLOTS_1A = list(range(298))
DEAD_2A = [270, 299]
LIVE_2A = [s for s in range(300) if s not in DEAD_2A]


def master_traces(slots, tilt=0.002):
    """Master trace positions: the fibremap comb with a slight tilt along x."""
    x = np.arange(NX)
    return slot_rows(slots)[:, None] + tilt * (x[None, :] - NX / 2)


def fibre_flux(n, seed=0):
    """Per-fibre throughput with the real odd/even alternation."""
    return throughput(n, seed=seed) * (1 + 0.1 * (-1) ** np.arange(n))


def moved(master, a=0.0, stretch=0.0, tilt=0.0, curve=0.0):
    """The master moved by a shift, a stretch about the detector centre, a tilt along x,
    and a quadratic term along the slit (``curve`` px at the slit ends)."""
    x = np.arange(NX)[None, :]
    v = (master - NY / 2) / (NY / 2)
    return master + a + stretch * v * NY / 2 + tilt * (x - NX / 2) + curve * v ** 2


def flat(traces, flux, ny=NY, extra=()):
    """A flat with a Gaussian fibre at each trace; ``extra`` adds (rows, flux) fibres."""
    y = np.arange(ny)[:, None]
    img = np.full((ny, traces.shape[1]), 100.0)
    for t, f in list(zip(traces, flux)) + list(extra):
        img += f * np.exp(-0.5 * ((y - np.asarray(t)[None, :]) / SIGMA) ** 2)
    return img


def check_placed(traces, truth, tol=0.05):
    err = np.abs(np.sort(traces, axis=0) - np.sort(truth, axis=0))
    assert err[:, 20:-20].max() < tol, f"max error {err.max():.3f} px"


@pytest.fixture(scope='module')
def red1a():
    master = master_traces(SLOTS_1A)
    flux = fibre_flux(len(SLOTS_1A))
    return master, flux, normalise_trace_heights(flux)


# --- recovering the motion ----------------------------------------------------

@pytest.mark.parametrize('a', [3.1, -3.1, 0.0])
def test_sub_pitch_shifts_are_recovered(red1a, a):
    master, flux, fp = red1a
    truth = moved(master, a)
    traces, record = measure_master_shift(master, flat(truth, flux), 'red', [], fp)
    check_placed(traces, truth)
    assert record['whole_fibres'] == 0


def test_blue_sized_motion_is_recovered(red1a):
    """Commissioning -> 2026 blue1A: -34 px, a 9 px compression, a tilt and a curve."""
    master, flux, fp = red1a
    truth = moved(master, a=-34.0, stretch=-0.0047, tilt=-0.0098, curve=0.3)
    traces, record = measure_master_shift(master, flat(truth, flux), 'blue', [], fp)
    check_placed(traces, truth)
    assert record['whole_fibres'] == -5


@pytest.mark.parametrize('n', [-2, -1, 1, 2])
def test_whole_fibre_shifts_are_followed(red1a, n):
    """One and two pitches: the aliases the odd/even pattern makes dangerous."""
    master, flux, fp = red1a
    truth = moved(master, n * PITCH + 0.4)
    traces, record = measure_master_shift(master, flat(truth, flux), 'red', [], fp)
    check_placed(traces, truth)
    assert record['whole_fibres'] == n


def test_2a_dead_fibre_leak_never_misregisters():
    master = master_traces(LIVE_2A)
    flux = fibre_flux(len(LIVE_2A))
    fp = normalise_trace_heights(flux)
    truth = moved(master, 2.0)
    leak_row = moved(master_traces([299]), 2.0)[0]
    for level in (0.12, 0.3):
        data = flat(truth, flux, extra=[(leak_row, level * np.median(flux))])
        try:
            traces, record = measure_master_shift(master, data, 'green', DEAD_2A, fp)
        except MasterShiftError:
            continue                       # refusing is acceptable; misplacing is not
        check_placed(traces, truth)
        assert record['whole_fibres'] == 0


def test_faint_end_fibre_still_counts(red1a):
    master, flux, fp = red1a
    faint = flux.copy()
    faint[0] *= 0.4
    truth = moved(master, 1.0)
    traces, record = measure_master_shift(master, flat(truth, faint), 'red', [], fp)
    check_placed(traces, truth)


def test_2049_row_flat_works(red1a):
    master, flux, fp = red1a
    truth = moved(master, -1.5)
    traces, _ = measure_master_shift(master, flat(truth, flux, ny=2049), 'red', [], fp)
    check_placed(traces, truth)


# --- refusing what cannot be placed with confidence ---------------------------

def test_slit_ends_off_the_detector_cannot_decide(red1a):
    """Both ends within a pitch of the detector edges: the odd/even pattern alone must
    not choose the fibre offset."""
    master, flux, fp = red1a
    truth = moved(master, 5.0 - master[0, NX // 2])          # first fibre at row ~5
    ny = int(truth[-1].max()) + 6                              # last fibre ~5 from the top
    with pytest.raises(MasterShiftError, match='slit ends'):
        measure_master_shift(master, flat(truth, flux, ny=ny), 'red', [], fp)


def test_masked_slit_end_is_refused(red1a):
    master, flux, fp = red1a
    dark = flux.copy()
    dark[:3] = 0.0
    with pytest.raises(MasterShiftError, match='3 trace.s. on no fibre'):
        measure_master_shift(master, flat(moved(master, 1.0), dark), 'red', [], fp)


def test_fibres_off_their_traces_are_refused(red1a):
    """Every tenth fibre 0.5 px off a smooth mapping: no surface can place them."""
    master, flux, fp = red1a
    truth = moved(master, 1.0)
    truth[::10] += 0.5
    with pytest.raises(MasterShiftError, match='fibres sit off'):
        measure_master_shift(master, flat(truth, flux), 'red', [], fp)


def test_traces_leaving_the_detector_are_refused(red1a):
    master, flux, fp = red1a
    truth = moved(master, 2.0)
    ny = int(truth[-1, 200:].min()) + 2                        # last fibre 2 px from the edge
    with pytest.raises(MasterShiftError, match='leave the detector'):
        measure_master_shift(master, flat(truth, flux, ny=ny), 'red', [], fp)


def test_wrong_column_count_is_refused(red1a):
    master, flux, fp = red1a
    data = flat(moved(master, 1.0), flux)[:, :200]
    with pytest.raises(MasterShiftError, match='columns'):
        measure_master_shift(master, data, 'red', [], fp)


def test_fingerprint_contradicting_the_slit_ends_is_refused_in_both_modes(red1a):
    master, flux, _ = red1a
    wrong = normalise_trace_heights(np.roll(flux, 2))
    data = flat(moved(master, 1.0), flux)
    for strict in (True, False):
        with pytest.raises(MasterShiftError, match='contradicts') as err:
            measure_master_shift(master, data, 'red', [], wrong, require_fingerprint=strict)
        assert not err.value.fingerprint_only


@pytest.mark.parametrize('fingerprint', ['unrelated', None])
def test_strict_rule_names_its_switch_and_relaxed_rule_accepts(red1a, fingerprint):
    master, flux, _ = red1a
    fp = (normalise_trace_heights(fibre_flux(len(flux), seed=7)) if fingerprint else None)
    truth = moved(master, 1.0)
    data = flat(truth, flux)
    with pytest.raises(MasterShiftError) as err:
        measure_master_shift(master, data, 'red', [], fp)
    assert err.value.fingerprint_only and SWITCH_HINT in str(err.value)
    traces, record = measure_master_shift(master, data, 'red', [], fp,
                                          require_fingerprint=False)
    check_placed(traces, truth)
    assert record['require_fingerprint'] is False


# --- reading the flat, building and saving the trace -------------------------

def write_flat(path, cameras):
    """A MEF with one extension per ``(colour, bench, side, data)``."""
    hdus = [fits.PrimaryHDU()]
    for colour, bench, side, data in cameras:
        hdu = fits.ImageHDU(np.asarray(data, dtype=np.float32))
        hdu.header.update(COLOR=colour.upper(), BENCH=bench, SIDE=side, BIASSUB=True)
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return str(path)


def test_read_flat_cameras_matches_the_tracer(tmp_path):
    data = np.arange(64 * 32, dtype=float).reshape(64, 32)
    path = write_flat(tmp_path / 'flat_mef.fits', [('red', 1, 'A', data),
                                                   ('blue', 1, 'A', data),
                                                   ('red', 2, 'B', np.zeros((64, 32)))])
    cams = read_flat_cameras(path)
    np.testing.assert_array_equal(cams[('red', '1', 'A')][0], data)
    np.testing.assert_array_equal(cams[('blue', '1', 'A')][0], data[::-1, ::-1])
    assert not cams[('red', '1', 'A')][2] and cams[('red', '2', 'B')][2]
    assert set(read_flat_cameras(path, 'blue')) == {('blue', '1', 'A')}


@pytest.fixture(scope='module')
def shifted_trace(red1a, tmp_path_factory):
    master, flux, fp = red1a
    truth = moved(master, -2.3, stretch=-0.002)
    tl = build_shifted_trace(master, flat(truth, flux), {'BIASSUB': True}, 'red', 1, 'A',
                             [], fp, flat_name='/night/flat_mef.fits',
                             master_path='/mc/LLAMAS_master_red_1_A_traces.pkl')
    path = tmp_path_factory.mktemp('shifted') / 'LLAMAS_red_1_A_traces.pkl'
    with open(path, 'wb') as f:
        cloudpickle.dump(tl, f)                    # as write_shifted_master saves it
    return tl, truth, str(path)


def test_built_trace_is_a_complete_current_trace(shifted_trace):
    tl, truth, _ = shifted_trace
    check_placed(tl.traces, truth)
    assert tl.comb_ok and tl.nfibers == 298
    assert tl.fiberimg.shape == tl.profimg.shape == (NY, NX)
    assert tl.fiberimg.min() == -1 and tl.fiberimg.max() == 297
    assert tl.master_shift['flat'] == 'flat_mef.fits'
    assert tl.master_shift['master'] == 'LLAMAS_master_red_1_A_traces.pkl'


def test_built_trace_loads_in_a_fresh_interpreter(shifted_trace):
    _, _, path = shifted_trace
    code = ("import pickle, sys; t = pickle.load(open(sys.argv[1], 'rb')); "
            "print(t.channel, t.nfibers, hasattr(t, 'comb_ok'), t.master_shift['whole_fibres'])")
    out = subprocess.run([sys.executable, '-c', code, path], capture_output=True, text=True)
    assert out.stdout.split() == ['red', '298', 'True', '0'], out.stderr[-500:]


# --- validate_and_fix_trace_fibres --------------------------------------------

class MasterTrace:
    """Stand-in for a current-format mastercalib trace pickle."""
    comb_ok = True

    def __init__(self, traces):
        self.channel, self.bench, self.side = 'red', 1, 'A'
        self.traces, self.naxis1 = traces, traces.shape[1]


@pytest.fixture
def night(red1a, tmp_path, monkeypatch):
    """A mastercalib dir with red 1A only, an empty trace dir, and red flats."""
    master, flux, fp = red1a
    monkeypatch.setattr(traceValidation, 'trace_references',
                        lambda: ({}, {'red': {'1A': list(fp)}}))
    mc, td = tmp_path / 'mastercalib', tmp_path / 'traces'
    mc.mkdir()
    td.mkdir()
    with open(mc / 'LLAMAS_master_red_1_A_traces.pkl', 'wb') as f:
        pickle.dump(MasterTrace(master), f)
    good = write_flat(tmp_path / 'red_flat.fits', [('red', 1, 'A', flat(moved(master, 1.7), flux))])
    weak = write_flat(tmp_path / 'weak_flat.fits',
                      [('red', 1, 'A', flat(moved(master, 1.7), fibre_flux(298, seed=7)))])
    empty = write_flat(tmp_path / 'placeholder_flat.fits', [('red', 1, 'A', np.zeros((NY, NX)))])
    return {'mc': str(mc), 'td': str(td), 'good': good, 'weak': weak, 'empty': empty}


def red1a_result(result, key):
    return [e for e in result[key] if e[:3] == ('red', '1', 'A')]


def test_missing_camera_gets_a_shifted_master(night):
    result = validate_and_fix_trace_fibres(night['td'], night['mc'], flats={'red': night['good']})
    (entry,) = red1a_result(result, 'shifted_masters')
    assert not red1a_result(result, 'fallback_used')
    with open(entry[4], 'rb') as f:
        saved = pickle.load(f)
    assert abs(saved.master_shift['shift'] - 1.7) < 0.05 and saved.comb_ok

    # On resume the shifted master is a valid trace, listed again, and not rewritten.
    mtime = os.path.getmtime(entry[4])
    again = validate_and_fix_trace_fibres(night['td'], night['mc'], flats={'red': night['good']})
    assert red1a_result(again, 'valid_traces') and red1a_result(again, 'shifted_masters')
    assert os.path.getmtime(entry[4]) == mtime


def test_unconfirmed_shift_writes_nothing_and_names_the_switch(night):
    result = validate_and_fix_trace_fibres(night['td'], night['mc'], flats={'red': night['weak']})
    (entry,) = red1a_result(result, 'unusable_masters')
    assert SWITCH_HINT in entry[3]
    assert not os.listdir(night['td'])                       # no unshifted copy either

    stream = io.StringIO()
    with pytest.raises(RuntimeError, match='shifted_master_require_fingerprint = false'):
        confirm_unusable_master_traces(result['unusable_masters'], interactive=False,
                                       stream=stream)
    assert 'Note:' in stream.getvalue()

    relaxed = validate_and_fix_trace_fibres(night['td'], night['mc'],
                                            flats={'red': night['weak']},
                                            require_fingerprint=False)
    assert red1a_result(relaxed, 'shifted_masters')


def test_without_flats_or_for_a_placeholder_the_master_is_copied(night):
    for flats in (None, {'red': night['empty']}):
        for name in os.listdir(night['td']):
            os.remove(os.path.join(night['td'], name))
        result = validate_and_fix_trace_fibres(night['td'], night['mc'], flats=flats)
        assert red1a_result(result, 'fallback_used')
        assert not red1a_result(result, 'shifted_masters')


def test_wrong_count_trace_is_replaced_only_on_success(night, red1a):
    master, _, _ = red1a
    own = os.path.join(night['td'], 'LLAMAS_red_1_A_traces.pkl')
    with open(own, 'wb') as f:
        pickle.dump(MasterTrace(master[:297]), f)

    result = validate_and_fix_trace_fibres(night['td'], night['mc'], flats={'red': night['weak']})
    assert red1a_result(result, 'unusable_masters')[0][4] == own
    with open(own, 'rb') as f:
        assert pickle.load(f).traces.shape[0] == 297         # untouched

    result = validate_and_fix_trace_fibres(night['td'], night['mc'], flats={'red': night['good']})
    assert red1a_result(result, 'shifted_masters')[0][4] == own
    with open(own, 'rb') as f:
        assert pickle.load(f).traces.shape[0] == 298
