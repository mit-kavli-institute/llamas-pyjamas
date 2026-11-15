"""Terminal reporting for reduce.py runs: phase summaries, collapsed repeats, the final report,
and the fibre-flat checks that feed them (``apply_fibre_flat_to_rss`` header cards and
``assess_fibre_flat``)."""

import io
import logging
import os
import shutil
import tempfile
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.File.llamasRSS import RSSgeneration
from llamas_pyjamas.Flat.fibreFlat import (BENCHSIDES, FF_RESIDUAL_CEILING_PCT,
                                           apply_fibre_flat_to_rss, assess_fibre_flat)
from llamas_pyjamas.reduce import output_report_row, trace_report_row
from llamas_pyjamas.Tests.rss_fixtures import N_WAVE, stage_name, write_rss
from llamas_pyjamas.Utils.reporter import (PipelineReporter, ReporterLogHandler, logger_area,
                                           warning_pattern)


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def run():
    """A curated reporter writing to a buffer, fed by a logger under llamas_pyjamas."""
    buf = io.StringIO()
    rep = PipelineReporter(n_frames=1, enabled=True, stream=buf)
    log = logging.getLogger('llamas_pyjamas.Bias.test_reporting')
    handler = ReporterLogHandler(rep)
    log.addHandler(handler)
    log.propagate = False
    rep.start(total_phases=2)
    yield rep, log, buf
    log.removeHandler(handler)
    log.propagate = True


# ---------------------------------------------------------------- reporter
def test_summary_prints_under_its_phase_tick(run):
    rep, _, buf = run
    rep.phase("🌌", "Sky")
    rep.summary("Fibre flat", ["row one", "⚠️  a check fired"])
    assert "row one" not in buf.getvalue()           # held until the phase finishes
    rep.phase("🧊", "Cubes")
    out = buf.getvalue()
    assert out.index("✓ [1/2] 🌌 Sky") < out.index("Fibre flat") < out.index("row one")
    assert out.index("row one") < out.index("[2/2] 🧊 Cubes")


def test_repeated_messages_collapse_per_phase(run):
    rep, log, buf = run
    rep.phase("🧾", "Bias")
    for colour in ('red', 'green', 'blue'):
        for bs in ('1A', '2B', '4B'):
            log.warning(f"bias-first: no tracer for {bs} {colour} — fallback unavailable")
    log.error("traceLlamas 2A green FAILED: only 6 traces")
    log.error("traceLlamas 3B green FAILED: only 30 traces")
    log.error("Pipeline failed: disk full")               # different wording: its own line
    rep.phase("🔦", "Trace")
    log.warning("bias-first: no tracer for 1A red — fallback unavailable")   # new phase: shown
    out = buf.getvalue()
    assert out.count("bias-first: no tracer") == 3        # inline + "more like" + next phase
    assert '⚠️  ×8 more like "bias-first: no tracer for 1A red' in out
    assert '❌ ×1 more like "traceLlamas 2A green FAILED: only 6 traces"' in out
    assert "3B green FAILED" not in out and "Pipeline failed: disk full" in out
    assert rep._n_warn == 10 and rep._n_err == 3


def test_summarised_records_are_counted_not_printed(run):
    rep, log, buf = run
    rep.phase("🌌", "Sky")
    log.warning("shown in the summary instead", extra={'summarised': True})
    assert "shown in the summary" not in buf.getvalue()
    assert rep._n_warn == 1 and rep._tally['Bias'] == 1


def test_final_report_rows_and_failure(run):
    rep, log, buf = run
    rep.report('Output', 'placeholder')
    rep.report('Traces', '24 cameras, existing traces')
    rep.report('Output', '2 science frames → 6 RSS')    # replaces, keeps its place
    rep.log_path = '/data/logs/run.log'
    log.warning("one warning")
    rep.finish(ok=False, message='RuntimeError: boom')
    tail = buf.getvalue().split("❌ Failed after", 1)[1]
    assert tail.startswith(" 0s: RuntimeError: boom")
    assert tail.index('Output') < tail.index('Traces') < tail.index('Warnings') < tail.index('Log')
    assert '2 science frames → 6 RSS' in tail and 'placeholder' not in tail
    assert '1 (Bias 1) — see log' in tail


def test_report_prints_in_verbose_mode():
    buf = io.StringIO()
    rep = PipelineReporter(enabled=False, stream=buf)
    rep.start()
    rep.phase("🧾", "Bias")                              # no live display in verbose mode
    rep.summary("Fibre flat", ["table row"])            # printed straight away
    rep.report('Output', '1 science frame → 3 RSS')
    rep.finish(ok=True)
    out = buf.getvalue()
    assert "Bias" not in out and "table row" in out
    assert "✅ Done in" in out and "1 science frame → 3 RSS" in out


def test_warning_pattern_ignores_camera_file_and_numbers():
    a = warning_pattern("✗ red2A: Expected 298 fibers, found 297")
    b = warning_pattern("✗ blue4B: Expected 300 fibers, found 12")
    assert a == b
    assert (warning_pattern("LLAMAS_2026-10-15_SCI24_RSS_red.fits: 2 fibre(s) skipped")
            == warning_pattern("LLAMAS_2026-10-16_SCI23_RSS_blue.fits: 5 fibre(s) skipped"))
    assert warning_pattern("tracer missing for 1A") != warning_pattern("bias missing for 1A")
    assert logger_area('llamas_pyjamas.Flat.fibreFlat') == 'Flat'
    assert logger_area('llamas_pyjamas.rss') == 'RSS'


# ---------------------------------------------------------------- sky false alarm
def _flat_like_pickle(tmp, counts):
    n, nw = counts.shape
    obj = SimpleNamespace(counts=counts, wave=np.tile(np.linspace(5000, 6000, nw), (n, 1)),
                          errors=np.ones((n, nw)), sky=np.zeros((n, nw)), dead_fibers=[],
                          relative_throughput=np.ones(n))
    hdr = fits.Header()
    hdr['OBS TYPE'] = 'CAL24'
    pkl = os.path.join(tmp, 'flat.pkl')
    with open(pkl, 'wb') as fh:
        cloudpickle.dump({'primary_header': hdr, 'extractions': [obj],
                          'metadata': [{'channel': 'green', 'bench': '1', 'side': 'A'}]}, fh)
    return pkl


def test_flat_rss_without_sky_subtraction_is_silent(tmp, caplog):
    pkl = _flat_like_pickle(tmp, np.full((6, 50), 10.0))
    gen = RSSgeneration(logger=logging.getLogger('llamas_pyjamas.rss'))
    with caplog.at_level(logging.WARNING, logger='llamas_pyjamas.rss'):
        gen.generate_rss(pkl, os.path.join(tmp, 'a_RSS.fits'), subtract_sky=False)
    assert 'sky' not in caplog.text.lower()
    with caplog.at_level(logging.WARNING, logger='llamas_pyjamas.rss'):
        gen.generate_rss(pkl, os.path.join(tmp, 'b_RSS.fits'))     # science default
    assert 'No sky model for channel green' in caplog.text


# ---------------------------------------------------------------- fibre flat: sky check
def _science_rss(tmp, throughput, seed=1):
    """A base RSS whose COUNTS are a flat sky times a known per-fibre throughput."""
    path = write_rss(stage_name(tmp, 'green', 'base'), noflat=True)
    rng = np.random.default_rng(seed)
    with fits.open(path, mode='update') as h:
        nfib = h['COUNTS'].data.shape[0]
        sky = 200.0 + 50.0 * np.sin(np.linspace(0, 6, N_WAVE))
        h['COUNTS'].data[:] = throughput[:nfib, None] * sky + rng.normal(0, 0.5, (nfib, N_WAVE))
        h['ERROR'].data[:] = 0.5
    return path


def _corrections_per_fibre(path, factors, benchsides=('1A', '1B')):
    hdus = [fits.PrimaryHDU()]
    hdus[0].header['METHOD'] = 'twilight'
    hdus[0].header['FIBIDS'] = 'physical'
    for k, bs in enumerate(benchsides):
        f = factors[30 * k:30 * (k + 1)]
        cols = [fits.Column(name='FIBER_ID', format='J', array=np.arange(30)),
                fits.Column(name='CORRECTION', format=f'{N_WAVE}D', array=np.tile(f[:, None], N_WAVE)),
                fits.Column(name='WAVE', format=f'{N_WAVE}D',
                            array=np.tile(np.linspace(4700.0, 7000.0, N_WAVE), (30, 1))),
                fits.Column(name='FLAGS', format='I', array=np.zeros(30, dtype=np.int16))]
        h = fits.BinTableHDU.from_columns(cols)
        h.header['EXTNAME'] = f'green_{bs[0]}_{bs[1]}'
        hdus.append(h)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_sky_check_true_flat_flattens_scrambled_flat_does_not(tmp):
    throughput = np.random.default_rng(0).uniform(0.8, 1.2, 60)
    base = _science_rss(tmp, throughput)

    good = apply_fibre_flat_to_rss(base, _corrections_per_fibre(os.path.join(tmp, 'c1.fits'), throughput),
                                   os.path.join(tmp, 'good_FF.fits'))
    h = fits.getheader(good)
    assert h['FFSB1A'] > 5.0 and h['FFSA1A'] < 0.5 and h['FFSN1A'] < 0.1
    assert h['FFNSKIP'] == 0

    scrambled = np.random.default_rng(5).permutation(throughput)
    bad = apply_fibre_flat_to_rss(base, _corrections_per_fibre(os.path.join(tmp, 'c2.fits'), scrambled),
                                  os.path.join(tmp, 'bad_FF.fits'))
    h = fits.getheader(bad)
    assert h['FFSA1A'] > h['FFSB1A'] and h['FFSA1B'] > h['FFSB1B']


def test_uncorrected_fibres_are_collected(tmp):
    throughput = np.ones(60)
    base = _science_rss(tmp, throughput)
    stats = {}
    out = apply_fibre_flat_to_rss(base, _corrections_per_fibre(os.path.join(tmp, 'c.fits'), throughput,
                                                               benchsides=('1A',)),
                                  os.path.join(tmp, 'x_FF.fits'), stats=stats)
    assert fits.getheader(out)['FFNSKIP'] == 30
    assert sorted(stats['skipped'])[:2] == [('green', '1B', 0), ('green', '1B', 1)]


# ---------------------------------------------------------------- fibre flat: assessment
def _assess_inputs(tmp, timad=12.0, after=1.0, before=3.0, noise=0.2, method='twilight',
                   ext_cards=None, ff_cards=None):
    """Corrections + one FF header for the green channel, all eight benchsides."""
    hdus = [fits.PrimaryHDU()]
    hdus[0].header['METHOD'] = method
    for bs in BENCHSIDES:
        h = fits.BinTableHDU.from_columns([fits.Column(name='FIBER_ID', format='J', array=[0])])
        h.header.update({'EXTNAME': f'green_{bs[0]}_{bs[1]}', 'CHANNEL': 'green',
                         'BENCH': bs[0], 'SIDE': bs[1], 'METHOD': method,
                         'TIMAD': timad, 'TINOUT': 0, 'TINFALL': 0})
        h.header.update((ext_cards or {}).get(bs, {}))
        hdus.append(h)
    corr = os.path.join(tmp, 'fibre_flat_corrections.fits')
    fits.HDUList(hdus).writeto(corr, overwrite=True)

    p = fits.PrimaryHDU()
    p.header['CHANNEL'] = 'green'
    p.header['FFNSKIP'] = 0
    for bs in BENCHSIDES:
        p.header.update({f'FFSB{bs}': before, f'FFSA{bs}': after, f'FFSN{bs}': noise})
    p.header.update(ff_cards or {})
    ff = os.path.join(tmp, 'x_RSS_green_FF.fits')
    p.writeto(ff, overwrite=True)
    return corr, [ff]


def test_assess_quiet_when_the_flat_works(tmp):
    lines, warnings, report = assess_fibre_flat(*_assess_inputs(tmp))
    assert warnings == []
    assert lines[0].startswith("Fibre flat (twilight)")
    assert lines[2].split()[0] == 'green' and lines[2].split()[1] == '1.0'
    assert 'sky scatter after flat 1.0–1.0% (throughput only 3.0–3.0%)' in report


def test_assess_says_plainly_when_the_ceiling_fails_everywhere(tmp):
    _, warnings, report = assess_fibre_flat(*_assess_inputs(tmp, after=15.0, before=3.0))
    # relative (worse than the RSS throughput alone) and absolute (ceiling), in that order
    assert len(warnings) == 2
    assert 'more sky scatter than the RSS throughput alone on 8/8' in warnings[0]
    assert f'exceeds the {FF_RESIDUAL_CEILING_PCT:g}% ceiling (provisional) on 8/8' in warnings[1]
    assert 'not flattening the sky' in warnings[1]
    assert 'over 3% on 8/8 ⚠️' in report


def test_assess_names_the_benchsides_over_the_ceiling(tmp):
    _, warnings, _ = assess_fibre_flat(*_assess_inputs(tmp, ff_cards={'FFSA3B': 6.5}))
    assert len(warnings) == 2 and 'green 3B 3.0→6.5%' in warnings[0]
    assert 'on 1/8 benchsides: green 3B 6.5%' in warnings[1]
    assert 'not flattening' not in warnings[1]


def test_assess_relative_outlier_lamp_only_fallback_and_fibre_outliers(tmp):
    ext = {'1A': {'TIMAD': 30.0}, '2B': {'METHOD': 'lamp_only', 'TIMAD': 0.0},
           '3A': {'TINFALL': 4}, '4B': {'TINOUT': 2}}
    _, warnings, _ = assess_fibre_flat(*_assess_inputs(tmp, ext_cards=ext))
    text = '\n'.join(warnings)
    assert 'T_i scatter outlier: green 1A 30.0%' in text
    assert 'green 2B' not in text.split('outlier')[1]      # lamp-only is not a scatter outlier
    assert 'Lamp-only throughput (no twilight data) for green 2B' in text
    assert '4 fibre(s) use the benchside-median T_i' in text and 'green 3A 4' in text
    assert '2 fibre(s) with T_i > 5 robust sigma' in text and 'green 4B 2' in text


def test_assess_lamp_only_method(tmp):
    _, warnings, report = assess_fibre_flat(*_assess_inputs(tmp, method='lamp_only', timad=0.0))
    assert any(w.startswith('Lamp-only fibre flat (no twilight)') for w in warnings)
    assert report.startswith('lamp only')


def test_assess_excludes_frames_too_faint_to_measure(tmp):
    lines, warnings, report = assess_fibre_flat(*_assess_inputs(tmp, after=20.0, noise=5.0))
    assert warnings == []                                  # cannot test the ceiling
    assert lines[2].split()[1] == '—' and any('sky too faint' in ln for ln in lines)
    assert 'too faint' in report


def test_assess_lists_uncorrected_fibres(tmp):
    corr, ff = _assess_inputs(tmp, ff_cards={'FFNSKIP': 2})
    _, warnings, report = assess_fibre_flat(corr, ff, skipped=[('green', '2A', 297),
                                                               ('green', '2A', 297),
                                                               ('red', '2A', 297)])
    assert any('2 fibre(s) not flat-fielded (MASK_NOT_FLAT_FIELDED): green 2A #297, red 2A #297'
               in w for w in warnings)
    assert '2 fibres not flat-fielded' in report


# ---------------------------------------------------------------- end-of-run rows (reduce.py)
def test_trace_and_output_report_rows():
    results = {'valid_traces': [('green', '1', 'A', 'p')] * 22,
               'fallback_used': [('red', '2', 'A', 'm', 'c')],
               'unusable_masters': [('blue', '3', 'B', 'wrong count', None)]}
    assert trace_report_row('existing_validated') == 'existing traces'
    assert (trace_report_row('existing_partial_fallback', results)
            == '23 cameras, existing traces; mastercalib for red 2A; no trace for blue 3B')
    green = {'valid_traces': [0] * 16,
             'fallback_used': [('green', b, s, 'm', 'c') for b in '1234' for s in 'AB']}
    assert (trace_report_row('generated_partial_fallback', green)
            == '24 cameras, traced this run; mastercalib for all green')
    written = {'LLAMAS_2026-10-16_00-31-08.4_SCI24_mef_extract_RSS_green.fits',
               'LLAMAS_2026-10-16_00-31-08.4_SCI24_mef_extract_RSS_green_FF.fits',
               'LLAMAS_2026-10-16_00-31-08.4_RSS_red.fits',
               'LLAMAS_2026-10-16_00-31-08.4_SCI24_whitelight.fits'}
    assert (output_report_row(1, written, 2, '/out')
            == '1 science frame → 2 RSS, 2 cubes in /out')
