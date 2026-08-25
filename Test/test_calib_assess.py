"""
Tests for llamas_pyjamas/Postprocessing/calib_assess.py.

Follows the bare-import pattern of test_fix_exposure_info.py: the
Postprocessing directory goes on sys.path and the module is imported without
triggering the heavy llamas_pyjamas package __init__.

Unit tests exercise the metric functions on in-memory 2048x2048 synthetic
frames (fast); a small end-to-end test writes real-shape single-channel MEFs
to tmp_path and runs main().
"""

import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

POSTPROC = Path(__file__).resolve().parents[1] / 'llamas_pyjamas' / 'Postprocessing'
sys.path.insert(0, str(POSTPROC))

import calib_assess as ca            # noqa: E402


# ------------------------------------------------------------------ helpers

NY, NX = ca.EXPECTED_SHAPE


def fibre_positions(benchside):
    n = ca.N_FIB[benchside]
    return np.linspace(30, 2010, n)


def flat_frame(benchside, amplitude=3000.0, faint_every=0, faint_amp=300.0,
               baseline=200.0, ghost=False):
    """2048x2048 flat: vertical fibre comb replicated across columns.
    Noise is seeded per call so tests are order-independent."""
    rng = np.random.default_rng(
        zlib.crc32(f"{benchside}|{amplitude}|{faint_every}|{ghost}".encode()))
    y = np.arange(NY, dtype=np.float64)
    comb = np.full(NY, baseline)
    if ghost:
        # one spurious peak outside the fibre stack (e.g. scattered light)
        comb += amplitude * np.exp(-0.5 * ((y - 24.0) / 1.2) ** 2)
    # sigma 1.2 px at the ~6.6 px fibre pitch keeps neighbour wings small
    # enough that a faint fibre still has its own prominence (as on the real
    # detector); at sigma 1.5 the wings of bright neighbours would swamp it.
    for i, pos in enumerate(fibre_positions(benchside)):
        amp = faint_amp if (faint_every and i % faint_every == 0) else amplitude
        comb += amp * np.exp(-0.5 * ((y - pos) / 1.2) ** 2)
    frame = np.tile(comb[:, None], (1, NX))
    frame += rng.normal(0, 3.0, size=frame.shape)
    return np.clip(frame, 0, 65535)


def arc_frame(channel, catalog, shift=0.0, amplitude=8000.0, baseline=150.0,
              n_lines=None):
    """2048x2048 arc: one bright fibre band (rows 1000+-2) with lines at
    catalog + shift; positions are given in pipeline orientation, so the
    frame is 'de-oriented' back to raw for the given channel."""
    x = np.arange(NX, dtype=np.float64)
    spec = np.full(NX, baseline)
    pix = catalog if n_lines is None else catalog[:n_lines]
    for pos in pix:
        p = pos + shift
        if 5 < p < NX - 5:
            spec += amplitude * np.exp(-0.5 * ((x - p) / 1.8) ** 2)
    frame = np.full((NY, NX), baseline)
    frame[998:1003, :] = spec[None, :]
    rng = np.random.default_rng(
        zlib.crc32(f"{channel}|{shift}|{n_lines}".encode()))
    frame += rng.normal(0, 4.0, size=frame.shape)
    frame = np.clip(frame, 0, 65535)
    # orient_extension will flip green/blue back; pre-apply the inverse.
    if channel == 'green':
        frame = np.fliplr(frame)
    elif channel == 'blue':
        frame = np.flipud(np.fliplr(frame))
    return frame


def write_mef(path, kind, channel, frames_by_benchside, prodcatg=True,
              object_name=None, readmde='FAST', exptime=5.0):
    primary = fits.PrimaryHDU()
    ph = primary.header
    if prodcatg:
        ph['PRODCATG'] = ca.KIND_PRODCATG[kind]
    if object_name:
        ph['OBJECT'] = object_name
    if readmde:
        ph['READ-MDE'] = readmde
    ph['SEXPTIME'] = exptime
    hdus = [primary]
    for benchside, frame in frames_by_benchside.items():
        hdu = fits.ImageHDU(frame.astype(np.float32))
        hdu.header['BENCH'] = benchside[0]
        hdu.header['SIDE'] = benchside[1]
        hdu.header['COLOR'] = channel
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture(scope='module')
def catalogs():
    return ca.load_line_catalogs()


# ------------------------------------------------------- classification

def _hdr(**cards):
    h = fits.Header()
    for k, v in cards.items():
        h[k.replace('_', '-')] = v
    return h


def test_classify_prodcatg():
    assert ca.classify_frame(_hdr(PRODCATG='CAL.R-FLT'))[0] == 'flat'
    assert ca.classify_frame(_hdr(PRODCATG='CAL.R-SKY'))[0] == 'twilight'
    assert ca.classify_frame(_hdr(PRODCATG='CAL.R-ARC'))[0] == 'arc'
    assert ca.classify_frame(_hdr(PRODCATG='CAL.R-BIA'))[0] == 'bias'
    assert ca.classify_frame(_hdr(PRODCATG='SCI.R-SL'))[0] == 'science'


def test_classify_object_heuristics():
    assert ca.classify_frame(_hdr(OBJECT='ThAr arc lamp'))[0] == 'arc'
    # 'twilight flat' must classify as twilight, not flat
    assert ca.classify_frame(_hdr(OBJECT='Twilight flat r'))[0] == 'twilight'
    assert ca.classify_frame(_hdr(OBJECT='LDLS flat'))[0] == 'flat'
    assert ca.classify_frame(_hdr(OBJECT='bias frame'))[0] == 'bias'
    kind, how = ca.classify_frame(_hdr(OBJECT='NGC 1234'))
    assert kind == 'unknown'


def test_classify_prodcatg_wins_over_object():
    kind, how = ca.classify_frame(_hdr(PRODCATG='CAL.R-ARC', OBJECT='flat'))
    assert kind == 'arc' and 'PRODCATG' in how


def test_classify_from_log():
    import obs_log as ol
    stem = 'LLAMAS_2024-11-29T10_35_13.115'
    name = stem + '_mef.fits'

    def row(obj, obs_type='calibration', mode='FAST', exp=0.1):
        return ol.LogRow(stem=stem, raw_filename=name, object_name=obj,
                         obs_type=obs_type, read_mode=mode, exp_time=exp)

    index = {stem: [row('ThAr + strong diffuser; good for Red')]}
    kind, how, mode, exp = ca.classify_from_log(name, index)
    assert (kind, mode, exp) == ('arc', 'FAST', 0.1)

    # a bare 'sky' during a calibration block is a twilight
    assert ca.classify_from_log(
        name, {stem: [row('sky')]})[0] == 'twilight'
    # conflicting duplicate rows -> unknown
    kind, how, mode, exp = ca.classify_from_log(
        name, {stem: [row('sky'), row('BR0305', obs_type='Science')]})
    assert kind == 'unknown' and 'conflicting' in how
    # not in the log at all
    assert ca.classify_from_log('LLAMAS_2024-11-29T11_00_00.000_mef.fits',
                                index)[0] == 'unknown'


# ------------------------------------------------------- flat metrics

def test_flat_metrics_good():
    m = ca.flat_extension_metrics(flat_frame('1A'), '1A')
    assert m['n_peaks'] == ca.N_FIB['1A']
    assert m['trace_pass'] is True
    assert m['n_sat_peaks'] == 0
    assert m['n_faint_missed'] == 0


def test_flat_metrics_faint_fibres_fail():
    # every 10th fibre at amplitude 300 -> below the prominence=500 cut but
    # above the 150 floor: the pipeline drops them, count comes up short.
    m = ca.flat_extension_metrics(flat_frame('1B', faint_every=10), '1B')
    assert m['n_peaks'] < ca.N_FIB['1B']
    assert m['trace_pass'] is False
    # every 10th of 300 fibres is faint: most must be seen at the floor
    assert m['n_faint_missed'] >= 20
    assert any('below the pipeline prominence cut' in r for r in m['reasons'])


def test_flat_metrics_overcount_recovered():
    # one ghost peak beyond the fibre stack: the pipeline drops spacing
    # outliers on over-count, so this still traces
    m = ca.flat_extension_metrics(flat_frame('3A', ghost=True), '3A')
    assert m['n_peaks'] == ca.N_FIB['3A'] + 1
    assert m['trace_pass'] is True
    assert any('extra peak' in r for r in m['reasons'])


def test_flat_metrics_saturated_fail():
    # bright enough to clip at the 16-bit ceiling: the peak count can still be
    # right, but saturation alone must fail the benchside.
    m = ca.flat_extension_metrics(flat_frame('2A', amplitude=70000.0), '2A')
    assert m['n_sat_peaks'] > 0
    assert m['trace_pass'] is False
    assert any('saturated' in r for r in m['reasons'])


# ------------------------------------------------------- arc metrics

def test_estimate_shift_recovers_small_offset(catalogs):
    cat = catalogs['red']
    shift, n = ca.estimate_shift(cat + 3.0, cat)
    assert n == cat.size
    assert shift == pytest.approx(3.0, abs=0.5)


def test_estimate_shift_recovers_large_negative_offset(catalogs):
    # the 2024 commissioning blue epoch shift (~ -206 px)
    cat = catalogs['blue']
    detected = cat - 206.0
    detected = detected[detected > 0]
    shift, n = ca.estimate_shift(detected, cat)
    assert shift == pytest.approx(-206.0, abs=0.5)
    assert n == detected.size


def test_estimate_shift_no_lines():
    assert ca.estimate_shift(np.array([]), np.array([100.0])) == (None, 0)


@pytest.mark.parametrize('channel', ['red', 'green', 'blue'])
def test_arc_metrics_matches_catalog(channel, catalogs):
    # arc_frame returns raw orientation; metrics expect pipeline orientation
    # (assess_file always orients first)
    cat = catalogs[channel]
    data = ca.orient_extension(arc_frame(channel, cat, shift=3.0),
                               fits.Header(), channel)
    m = ca.arc_extension_metrics(data, cat)
    assert m['n_matched'] >= int(0.9 * cat.size)
    assert m['shift'] == pytest.approx(3.0, abs=1.0)
    assert m['n_sat_lines'] == 0


def test_arc_metrics_weak_lamp(catalogs):
    cat = catalogs['red']
    m = ca.arc_extension_metrics(     # red needs no orientation change
        arc_frame('red', cat, n_lines=4, amplitude=6000.0), cat)
    assert m['n_matched'] < ca.ARC_MIN_LINES


# ------------------------------------------------------- summaries

def _flat_rows(pass_count, sat=0, faint=0):
    rows = {}
    for i, bs in enumerate(ca.BENCHSIDES):
        ok = i < pass_count
        rows[bs] = {'status': 'OK', 'trace_pass': ok,
                    'n_peaks': ca.N_FIB[bs] if ok else ca.N_FIB[bs] - 3,
                    'expected': ca.N_FIB[bs], 'n_marginal': 0,
                    'n_faint_missed': 0 if ok else faint,
                    'n_sat_peaks': 0 if ok else sat, 'sat_frac': 0.0,
                    'median_prominence': 2500.0, 'frame_median': 5000.0,
                    'reasons': []}
    return rows


def test_summarize_flat_verdicts():
    assert ca.summarize_flat_channel(_flat_rows(8), 'flat')['verdict'] == 'GOOD'
    assert ca.summarize_flat_channel(_flat_rows(6), 'flat')['verdict'] == 'MARGINAL'
    assert ca.summarize_flat_channel(_flat_rows(1), 'flat')['verdict'] == 'POOR'
    assert ca.summarize_flat_channel(_flat_rows(0), 'flat')['verdict'] == 'UNUSABLE'


def test_summarize_arc_unusable_below_min_lines():
    rows = {bs: {'status': 'OK', 'n_lines': 5, 'catalog_size': 23,
                 'n_matched': 4, 'shift': 2.0, 'n_sat_lines': 0,
                 'sat_frac': 0.0, 'peak_row': 1000}
            for bs in ca.BENCHSIDES}
    s = ca.summarize_arc_channel(rows)
    assert s['verdict'] == 'UNUSABLE'
    assert any('catalog lines' in r for r in s['reasons'])


def test_summarize_arc_flags_missing_4a():
    rows = {bs: {'status': 'OK', 'n_lines': 23, 'catalog_size': 23,
                 'n_matched': 20, 'shift': 2.0, 'n_sat_lines': 0,
                 'sat_frac': 0.0, 'peak_row': 1000}
            for bs in ca.BENCHSIDES if bs != '4A'}
    s = ca.summarize_arc_channel(rows)
    assert s['verdict'] == 'MARGINAL'
    assert any('4A' in r for r in s['reasons'])


# ------------------------------------------------------- end-to-end

@pytest.fixture(scope='module')
def night(tmp_path_factory, catalogs):
    """A tiny red-only night: good flat, faint flat, arc, twilight."""
    root = tmp_path_factory.mktemp('night')
    good = {bs: flat_frame(bs) for bs in ca.BENCHSIDES}
    faint = {bs: flat_frame(bs, faint_every=10) for bs in ca.BENCHSIDES}
    arc = {bs: arc_frame('red', catalogs['red'], shift=3.0)
           for bs in ca.BENCHSIDES}
    twi = {bs: flat_frame(bs, amplitude=2500.0) for bs in ca.BENCHSIDES}
    write_mef(root / 'goodflat_mef.fits', 'flat', 'red', good)
    write_mef(root / 'faintflat_mef.fits', 'flat', 'red', faint)
    write_mef(root / 'arc_mef.fits', 'arc', 'red', arc)
    # twilight classified via OBJECT heuristics (no PRODCATG - old data)
    write_mef(root / 'twiflat_mef.fits', 'twilight', 'red', twi,
              prodcatg=False, object_name='twilight flat')
    return root


def test_main_end_to_end(night, tmp_path):
    hashes = {p.name: sha(p) for p in night.glob('*.fits')}
    out = tmp_path / 'out'
    code = ca.main([str(night), '--output-dir', str(out),
                    '--report', 'report.json'])
    # green/blue roles have no candidates at all -> unviable exit code
    assert code == ca.EXIT_UNVIABLE

    snippet = (out / 'calib_assess_recommendations.txt').read_text()
    assert f"red_flat_file = {night / 'goodflat_mef.fits'}" in snippet
    assert f"red_arc_file = {night / 'arc_mef.fits'}" in snippet
    assert f"red_twilight_flat = {night / 'twiflat_mef.fits'}" in snippet
    assert '# green_flat_file: no viable candidate' in snippet

    report = json.loads((out / 'report.json').read_text())
    red_flat = report['roles']['red_flat']
    assert red_flat['recommended'].endswith('goodflat_mef.fits')
    assert [Path(r['file']).name for r in red_flat['ranked']] == \
        ['goodflat_mef.fits', 'faintflat_mef.fits']
    assert red_flat['ranked'][0]['summary']['verdict'] == 'GOOD'
    assert red_flat['ranked'][1]['summary']['verdict'] != 'GOOD'
    arc_summary = report['roles']['red_arc']['ranked'][0]['summary']
    assert arc_summary['min_matched'] >= ca.ARC_MIN_LINES
    assert arc_summary['median_shift'] == pytest.approx(3.0, abs=1.0)
    # no FAST bias in the directory -> bias sanity warning
    assert any('bias' in w.lower() for w in report['warnings'])

    # originals untouched
    assert {p.name: sha(p) for p in night.glob('*.fits')} == hashes


def test_main_saturated_flat_demoted(tmp_path, catalogs):
    root = tmp_path / 'sat'
    root.mkdir()
    sat = {bs: flat_frame(bs, amplitude=70000.0) for bs in ca.BENCHSIDES}
    write_mef(root / 'satflat_mef.fits', 'flat', 'red', sat)
    out = tmp_path / 'out'
    code = ca.main([str(root), '--output-dir', str(out),
                    '--report', 'report.json'])
    assert code == ca.EXIT_UNVIABLE
    report = json.loads((out / 'report.json').read_text())
    summary = report['roles']['red_flat']['ranked'][0]['summary']
    assert summary['verdict'] == 'UNUSABLE'
    assert summary['n_sat_peaks'] > 0
    assert any('saturated' in r for r in summary['reasons'])
    snippet = (out / 'calib_assess_recommendations.txt').read_text()
    assert '# red_flat_file: no viable candidate' in snippet


def test_main_bad_directory():
    assert ca.main(['/nonexistent/dir']) == ca.EXIT_ERROR


def test_placeholder_and_bad_geometry_gates(tmp_path):
    root = tmp_path / 'gates'
    root.mkdir()
    frames = {bs: flat_frame(bs) for bs in ca.BENCHSIDES}
    frames['1A'] = np.zeros(ca.EXPECTED_SHAPE)          # placeholder
    frames['1B'] = np.ones((1024, 1024)) * 500.0        # bad geometry
    write_mef(root / 'flat_mef.fits', 'flat', 'red', frames)
    out = tmp_path / 'out'
    ca.main([str(root), '--output-dir', str(out), '--report', 'report.json',
             '--no-snippet'])
    report = json.loads((out / 'report.json').read_text())
    per_bench = report['files'][0]['per_bench']['red']
    assert per_bench['1A']['status'] == 'PLACEHOLDER'
    assert per_bench['1B']['status'] == 'BAD_GEOMETRY'
    summary = report['roles']['red_flat']['ranked'][0]['summary']
    assert summary['n_pass'] == 6
    assert summary['verdict'] == 'MARGINAL'
