"""
Tests for the exposure-metadata repair tools in llamas_pyjamas/Postprocessing:
fix_exposure_info.py, obs_log.py, readmode_assess.py.

Follows the bare-import pattern of test_qa_engine_header.py: the Postprocessing
directory goes on sys.path and modules are imported without triggering the
heavy llamas_pyjamas package __init__.
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

POSTPROC = Path(__file__).resolve().parents[1] / 'llamas_pyjamas' / 'Postprocessing'
sys.path.insert(0, str(POSTPROC))

import fix_exposure_info as fei      # noqa: E402
import obs_log as ol                 # noqa: E402
import readmode_assess as rma        # noqa: E402


# ------------------------------------------------------------------ helpers

DETS = [('1', 'A', 'red'), ('1', 'A', 'green'), ('1', 'A', 'blue'),
        ('1', 'B', 'red'), ('1', 'B', 'green'), ('1', 'B', 'blue')]


def make_mef(path, n_ext=2, exptime='blank', readmde=None, tel_block=True,
             ext_exptime=0.0, shape=(40, 40), aexptime_card=True):
    """Mini raw-style MEF. exptime: 'blank' | 'rexptime' | float."""
    primary = fits.PrimaryHDU()
    ph = primary.header
    if exptime == 'blank':
        ph.append(fits.Card('EXPTIME'))         # present-but-undefined, the real
        if aexptime_card:                       # ut20241128-29 blank-card state
            ph.append(fits.Card('AEXPTIME'))
    elif exptime == 'rexptime':
        ph.append(fits.Card('EXPTIME'))
        ph['REXPTIME'] = 10.0
    elif isinstance(exptime, float):
        ph['EXPTIME'] = exptime
    if readmde:
        ph['READ-MDE'] = readmde
    if tel_block:
        ph['HIERARCH TEL UTC'] = '01:34:17.5'
        ph['HIERARCH TEL DATE-OBS'] = '2024-11-29'
        ph['HIERARCH TEL RA'] = '02:34:56.7'
        ph['HIERARCH TEL DEC'] = '-29:12:34'
        ph['HIERARCH TEL AIRMASS'] = 1.14
    hdus = [primary]
    for i in range(n_ext):
        hdu = fits.ImageHDU(np.zeros(shape, dtype=np.uint16))
        b, s, c = DETS[i % len(DETS)]
        hdu.header['BENCH'], hdu.header['SIDE'], hdu.header['COLOR'] = b, s, c
        ev = ext_exptime[i] if isinstance(ext_exptime, (list, tuple)) else ext_exptime
        hdu.header['EXPTIME'] = ev
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return Path(path)


def scan(path):
    return fei.scan_file(Path(path), fei._stem_of(Path(path)))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------- blankness / scan

def test_card_status_semantics():
    hdr = fits.Header()
    hdr.append(fits.Card('EXPTIME'))                 # undefined value
    hdr['SEXPTIME'] = 0.0
    hdr['OBJECT'] = '   '
    hdr['REXPTIME'] = 10.0
    assert fei.card_status(hdr, 'EXPTIME', True).status == 'blank'
    assert fei.card_status(hdr, 'ABSENT', True).status == 'absent'
    assert fei.card_status(hdr, 'SEXPTIME', True).status == 'suspect_zero'
    assert fei.card_status(hdr, 'OBJECT').status == 'blank'
    f = fei.card_status(hdr, 'REXPTIME', True)
    assert f.status == 'ok' and f.current == 10.0


def test_scan_broken_night_state(tmp_path):
    audit = scan(make_mef(tmp_path / 'a_mef.fits'))
    assert audit.scan_error is None
    assert audit.finding('EXPTIME').status == 'blank'
    assert audit.finding('READ-MDE').status == 'absent'
    assert audit.finding('TEL UTC').status == 'ok'
    assert audit.ext_exptime_state == 'zero'
    assert audit.stem == 'a'


# ------------------------------------------------------------------- Case A

def test_case_a_rexptime_and_tel_block(tmp_path):
    audit = scan(make_mef(tmp_path / 'a_mef.fits', exptime='rexptime'))
    fei.propose_case_a(audit)
    exp = audit.fix_for('EXPTIME')
    assert exp is not None and exp.value == 10.0 and exp.source == 'REXPTIME'
    for key, expect in [('UTC', '01:34:17.5'), ('DATE-OBS', '2024-11-29'),
                        ('DATE', '2024-11-29'), ('RA', '02:34:56.7'),
                        ('DEC', '-29:12:34'), ('AIRMASS', 1.14)]:
        fix = audit.fix_for(key)
        assert fix is not None and fix.value == expect, key
    mjd = audit.fix_for('MJD-OBS')
    assert mjd is not None and abs(mjd.value - 60643.06547454) < 1e-4


def test_case_a_extension_exptime_donor(tmp_path):
    audit = scan(make_mef(tmp_path / 'a_mef.fits', ext_exptime=25.0))
    fei.propose_case_a(audit)
    fix = audit.fix_for('EXPTIME')
    assert fix is not None and fix.value == 25.0
    assert fix.source == 'extension EXPTIME'


def test_case_a_inconsistent_extensions_no_proposal(tmp_path):
    audit = scan(make_mef(tmp_path / 'a_mef.fits', ext_exptime=[10.0, 20.0]))
    fei.propose_case_a(audit)
    assert audit.fix_for('EXPTIME') is None
    assert any('disagree' in n for n in audit.notes)


def test_sexptime_never_fabricated_aexptime_only_unblanked(tmp_path):
    audit = scan(make_mef(tmp_path / 'a_mef.fits', exptime='rexptime',
                          aexptime_card=False))
    fei.propose_case_a(audit)
    for fix in audit.fixes:
        fix.approved = True
    fei.finalize_fixes(audit)
    assert audit.fix_for('SEXPTIME') is None
    assert audit.fix_for('AEXPTIME') is None      # card absent -> never added
    ext = audit.fix_for('EXPTIME', 'all_extensions')
    assert ext is not None and ext.approved and ext.value == 10.0

    audit2 = scan(make_mef(tmp_path / 'b_mef.fits'))    # blank AEXPTIME card exists
    audit2.propose('EXPTIME', 5.0, 'obs log (x row 1)', 'B').approved = True
    fei.finalize_fixes(audit2)
    aex = audit2.fix_for('AEXPTIME')
    assert aex is not None and aex.value == 5.0


# ------------------------------------------------------------- obs_log: GUI

GUI_LOG = (
    "file name\tChekSum\tUT\tOBJECT\tEXPTIME\tREAD-MDE\tAIRMASS\n"
    "---------\t------\t------\t------\t----\t----\n"
    "LLAMAS_2026-06-30_19-33-12.8_CAL22_mef.fits\tERR\tLLAMAS_2026-06-30_19-33-12.8_CAL22"
    "\tLDLS flat\t0.07\tFAST\t1.0 / Airmass at start of exposure 1.1 / at end\n"
    "LLAMAS_2026-06-30_19-40-15.9_CAL22_mef.fits\tERR\tLLAMAS_2026-06-30_19-40-15.9_CAL22"
    "\tARC ThAr\t1.0\tslow\t\n"
)


def test_gui_log_parsing(tmp_path):
    log = tmp_path / 'obs.log'
    log.write_text(GUI_LOG)
    rows = ol.load_obs_log(log)
    assert len(rows) == 2
    r0, r1 = rows
    assert r0.stem == 'LLAMAS_2026-06-30_19-33-12.8_CAL22'
    assert r0.object_name == 'LDLS flat'          # space survives tab-splitting
    assert r0.exp_time == 0.07
    assert r0.read_mode == 'FAST'
    assert r0.airmass == 1.0                      # first float of the free text
    assert r1.read_mode == 'SLOW'                 # case-normalized
    assert r1.airmass is None


def test_unrecognized_log_header(tmp_path):
    log = tmp_path / 'obs.log'
    log.write_text("colA\tcolB\nx\ty\n")
    with pytest.raises(ol.ObsLogError):
        ol.load_obs_log(log)


# ------------------------------------------------------------ obs_log: xlsx

XLSX_ROWS = [
    ['file name', 'local time', 'read mode', 'exp. time', 'type', 'object', 'comment'],
    ['LLAMAS_2024-11-28T23_37_24.874_mef', '20:55', 'fast', 5, 'calibration', 'sky', 'half field'],
    ['LLAMAS_2024-11-28T23_37_24.874_mef', '20:56', '"', 20, '"', '"', '"'],
    [None, '20:58', 'slow', 60, '"', '"', '"'],
    [None, None, None, None, None, None, None],
    ['LLAMAS_2024-11-29T00_38_42.162_mef.fits', '21.4', 'Slow', '10', 'Flux Std', 'Feige 110', None],
    ['LLAMAS_2024-11-29T02_40_39.618', None, 'SLow', 'abc', 'Science', 'SPT CL 2311', None],
]


def _write_xlsx(tmp_path):
    openpyxl = pytest.importorskip('openpyxl')
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = '20241128'
    for row in XLSX_ROWS:
        ws.append(row)
    path = tmp_path / 'log.xlsx'
    wb.save(path)
    return path


def test_xlsx_hazards(tmp_path):
    rows = ol.load_obs_log(_write_xlsx(tmp_path))
    assert len(rows) == 5                                  # separator row dropped
    r1, r2, r3, r5, r6 = rows
    assert r2.read_mode == 'FAST' and r2.exp_time == 20.0  # ditto expanded
    assert r2.obs_type == 'calibration' and r2.object_name == 'sky'
    assert r3.is_continuation and r3.stem == r1.stem       # filename attached...
    assert r3.read_mode == 'SLOW'                          # ...own divergent mode kept
    assert r5.stem == 'LLAMAS_2024-11-29T00_38_42.162'     # .fits+_mef stripped
    assert r5.read_mode == 'SLOW' and r5.exp_time == 10.0
    assert r6.stem == 'LLAMAS_2024-11-29T02_40_39.618'     # bare stem spelling
    assert r6.read_mode == 'SLOW'                          # 'SLow' normalized
    assert r6.exp_time is None and r6.exp_time_raw == 'abc'


def test_index_keeps_conflicts_collapses_duplicates(tmp_path):
    rows = ol.load_obs_log(_write_xlsx(tmp_path))
    index = ol.build_log_index(rows)
    conflicted = index['LLAMAS_2024-11-28T23_37_24.874']
    assert len(conflicted) == 3                    # 5s fast / 20s fast / 60s slow
    assert len(index['LLAMAS_2024-11-29T00_38_42.162']) == 1
    dup = rows[1]
    index2 = ol.build_log_index(rows + [dup])      # identical dup collapses
    assert len(index2['LLAMAS_2024-11-28T23_37_24.874']) == 3


def test_csv_parity_handwritten_schema(tmp_path):
    csv_path = tmp_path / 'log.csv'
    csv_path.write_text(
        "file name,local time,read mode,exp. time,type,object,comment\n"
        'LLAMAS_2024-11-29T00_38_42.162_mef.fits,21:00,slow,10,Flux Std,Feige 110,\n')
    rows = ol.load_obs_log(csv_path)
    assert rows[0].read_mode == 'SLOW' and rows[0].exp_time == 10.0
    assert rows[0].stem == 'LLAMAS_2024-11-29T00_38_42.162'


def test_normalize_stem_spellings():
    for spelling in ('LLAMAS_2024-11-29T02_40_39.618_mef.fits',
                     'LLAMAS_2024-11-29T02_40_39.618_mef',
                     'LLAMAS_2024-11-29T02_40_39.618'):
        assert ol.normalize_stem(spelling) == 'LLAMAS_2024-11-29T02_40_39.618'
    assert ol.normalize_stem('  ') is None


def test_stem_timestamp_both_eras():
    t1 = ol.stem_timestamp('LLAMAS_2024-11-29T02_40_39.618')
    t2 = ol.stem_timestamp('LLAMAS_2026-06-30_19-33-12.8_CAL22')
    assert (t1.year, t1.hour, t1.second) == (2024, 2, 39)
    assert (t2.year, t2.hour, t2.second) == (2026, 19, 12)


# ------------------------------------------------------------ readmode_assess

def _bands_json(tmp_path, dets, fast=(1740.0, 1760.0), slow=(1640.0, 1660.0)):
    def entry(lo, hi):
        # observed is the source of truth; the padded top-level min/max are
        # deliberately wrong here so a regression to padded-first fails tests
        return {'edge_bg': {'min': lo - 500, 'max': hi + 500,
                            'observed': {'min': lo, 'max': hi,
                                         'med': (lo + hi) / 2}}}
    data = {'per_detector': {'CAL.R-BIA': {
        'FAST': {d: entry(*fast) for d in dets},
        'SLOW': {d: entry(*slow) for d in dets}}}}
    path = tmp_path / 'bands.json'
    path.write_text(json.dumps(data))
    return path


def _full_mef(path, level_by_det, shape=(2048, 2048)):
    hdus = [fits.PrimaryHDU()]
    for (b, s, c), level in level_by_det:
        data = np.full(shape, level, dtype=np.uint16)
        data[::2, :] += 1                        # break the constant-frame skip
        hdu = fits.ImageHDU(data)
        hdu.header['BENCH'], hdu.header['SIDE'], hdu.header['COLOR'] = b, s, c
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_assess_votes_fast(tmp_path):
    labels = [f"{b}.{s}.{c.capitalize()}" for b, s, c in DETS]
    bands = _bands_json(tmp_path, labels)
    mef = _full_mef(tmp_path / 'f.fits', [(d, 1750) for d in DETS])
    result = rma.assess_readmode(mef, baselines_path=bands)
    assert result['mode'] == 'FAST'
    assert result['numeric']['votes_fast'] == len(DETS)
    assert result['numeric']['votes_slow'] == 0


def test_assess_neither_abstains_and_undecided(tmp_path):
    labels = [f"{b}.{s}.{c.capitalize()}" for b, s, c in DETS]
    bands = _bands_json(tmp_path, labels)
    mef = _full_mef(tmp_path / 'n.fits', [(d, 1400) for d in DETS])  # far from both
    result = rma.assess_readmode(mef, baselines_path=bands)
    assert result['numeric']['n_usable'] == 0
    assert result['confidence'] == 'UNDECIDED' and result['mode'] is None


def test_assess_placeholder_and_min_stripe(tmp_path):
    labels = [f"{b}.{s}.{c.capitalize()}" for b, s, c in DETS]
    bands = _bands_json(tmp_path, labels)
    hdus = [fits.PrimaryHDU()]
    for i, (b, s, c) in enumerate(DETS):
        if i == 0:
            data = np.full((2048, 2048), 500, dtype=np.uint16)   # constant
        else:
            data = np.full((2048, 2048), 1650, dtype=np.uint16)
            data[::2, :] += 1
            data[:1024, :] += 800        # bottom half illuminated: top stripe clean
        hdu = fits.ImageHDU(data)
        hdu.header['BENCH'], hdu.header['SIDE'], hdu.header['COLOR'] = b, s, c
        hdus.append(hdu)
    mef = tmp_path / 'p.fits'
    fits.HDUList(hdus).writeto(mef)
    result = rma.assess_readmode(mef, baselines_path=bands)
    assert result['numeric']['n_placeholder'] == 1
    assert result['numeric']['votes_slow'] == len(DETS) - 1   # min(bot, top) wins
    assert result['mode'] == 'SLOW'


def test_assess_science_prior_is_advisory(tmp_path):
    labels = [f"{b}.{s}.{c.capitalize()}" for b, s, c in DETS]
    bands = _bands_json(tmp_path, labels)
    mef = _full_mef(tmp_path / 's.fits', [(d, 1750) for d in DETS])
    result = rma.assess_readmode(mef, exposure_type='Science', baselines_path=bands)
    assert result['mode'] == 'FAST'               # prior never flips the tally
    assert 'SLOW' in result['prior_note']


def test_map_extensions_cam_name_and_positional(tmp_path):
    hdus = [fits.PrimaryHDU()]
    h1 = fits.ImageHDU(np.zeros((4, 4), dtype=np.uint16))
    h1.header['CAM_NAME'] = '1A_red'
    h2 = fits.ImageHDU(np.zeros((4, 4), dtype=np.uint16))     # no identity cards
    hdus += [h1, h2]
    mapping, warnings = rma.map_extensions(fits.HDUList(hdus))
    assert mapping[0][0] == '1.A.Red'
    assert mapping[1][0] == '1.A.Green'           # positional fallback
    assert any('assumed' in w for w in warnings)


def test_load_bands_gap_exclusion(tmp_path):
    path = _bands_json(tmp_path, ['1.A.Red'], fast=(1000.0, 1100.0),
                       slow=(1090.0, 1190.0))     # overlapping bands
    with pytest.raises(rma.ReadmodeAssessError):
        rma.load_bands(path)


# ---------------------------------------------------------- interaction

def scripted(*answers):
    answers = list(answers)
    def input_fn(prompt=''):
        if not answers:
            raise AssertionError(f"unexpected prompt: {prompt}")
        return answers.pop(0)
    return input_fn


def test_ask_retries_and_eof():
    ask = fei.make_ask(scripted('z', 'y'))
    assert ask("go", 'ynq') == 'y'
    def eof(prompt=''):
        raise EOFError
    assert fei.make_ask(eof)("go", 'ynq') == 'q'


def test_resolve_case_a_all(tmp_path):
    audit = scan(make_mef(tmp_path / 'a_mef.fits', exptime='rexptime'))
    fei.propose_case_a(audit)
    opts = fei.parse_args([str(tmp_path)])
    fei.resolve_case_a([audit], opts, fei.make_ask(scripted('a')))
    assert all(f.approved for f in audit.fixes if f.case == 'A')


def test_resolve_case_c_stub_threshold(tmp_path, monkeypatch):
    def result(confidence, mode='FAST'):
        return {'mode': mode, 'confidence': confidence,
                'numeric': {'votes_fast': 19, 'votes_slow': 0, 'n_usable': 19,
                            'n_neither': 0, 'n_excluded': 0, 'n_placeholder': 0,
                            'frac': 1.0, 'median_margin': 100.0},
                'per_detector': [], 'warnings': [], 'prior_note': None,
                'bands_source': 'stub'}
    class Stub:
        calls = []
        @staticmethod
        def assess_readmode(path, exposure_type=None, baselines_path=None):
            return result(Stub.tier)
        @staticmethod
        def format_assessment(res, verbose=False):
            return f"stub {res['confidence']}"
    monkeypatch.setitem(sys.modules, 'readmode_assess', Stub)

    audit = scan(make_mef(tmp_path / 'a_mef.fits'))
    opts = fei.parse_args([str(tmp_path), '--yes'])
    Stub.tier = 'HIGH'
    fei.resolve_case_c([audit], opts, fei.make_ask(scripted()), scripted())
    fix = audit.fix_for('READ-MDE')
    assert fix is not None and fix.approved and fix.value == 'FAST'
    assert 'HIGH' in fix.source

    audit2 = scan(make_mef(tmp_path / 'b_mef.fits'))
    Stub.tier = 'MEDIUM'                          # below default 'high' threshold
    fei.resolve_case_c([audit2], opts, fei.make_ask(scripted()), scripted())
    assert audit2.fix_for('READ-MDE') is None
    assert any('below threshold' in n for n in audit2.notes)


# --------------------------------------------------------------- writing

def test_write_repaired_guards_and_provenance(tmp_path):
    src_dir = tmp_path / 'input'
    out_dir = tmp_path / 'out'
    src_dir.mkdir(); out_dir.mkdir()
    src = make_mef(src_dir / 'a_mef.fits', exptime='rexptime')
    before = sha(src)
    audit = scan(src)
    fei.propose_case_a(audit)
    for f in audit.fixes:
        f.approved = True
    fei.finalize_fixes(audit)
    dest = fei.write_repaired(audit, out_dir, overwrite=False, obs_log_name='log.xlsx')
    assert sha(src) == before                     # original byte-identical
    with fits.open(dest) as hdul:
        phdr = hdul[0].header
        assert phdr['EXPTIME'] == 10.0
        assert phdr['FIXEXP ORIG'] == 'a_mef.fits'
        assert phdr['FIXEXP OBSLOG'] == 'log.xlsx'
        assert 'REXPTIME' in phdr['FIXEXP SRC EXPTIME']
        assert any('fix_exposure_info' in str(h) for h in phdr['HISTORY'])
        for hdu in hdul[1:]:
            assert hdu.header['EXPTIME'] == 10.0
    assert not fei.verify_written(audit)
    with pytest.raises(FileExistsError):
        fei.write_repaired(audit, out_dir, overwrite=False, obs_log_name=None)
    with pytest.raises(RuntimeError):             # dest would be the original
        fei.write_repaired(audit, src_dir, overwrite=True, obs_log_name=None)
    assert sha(src) == before


def test_main_end_to_end_yes(tmp_path):
    src_dir = tmp_path / 'night'
    src_dir.mkdir()
    make_mef(src_dir / 'LLAMAS_2026-06-30_19-33-12.8_CAL22_mef.fits')
    make_mef(src_dir / 'LLAMAS_2026-06-30_19-40-15.9_CAL22_mef.fits')
    log = tmp_path / 'obs.log'
    log.write_text(GUI_LOG)
    out = tmp_path / 'fixed'
    code = fei.main([str(src_dir), '--obs-log', str(log), '--output-dir',
                     str(out), '--yes', '--verify'],
                    input_fn=scripted())
    written = sorted(p.name for p in out.glob('*.fits'))
    assert len(written) == 2
    with fits.open(out / written[0]) as hdul:
        phdr = hdul[0].header
        assert phdr['EXPTIME'] == 0.07
        assert phdr['READ-MDE'] == 'FAST'
        assert phdr['OBJECT'] == 'LDLS flat'
        assert phdr['UTC'] == '01:34:17.5'        # Case A TEL propagation
    assert (out / 'fix_exposure_info_summary.txt').exists()
    assert code in (fei.EXIT_OK, fei.EXIT_PARTIAL)


def test_main_refuses_output_equals_input(tmp_path):
    src_dir = tmp_path / 'night'
    src_dir.mkdir()
    make_mef(src_dir / 'LLAMAS_x_mef.fits')
    code = fei.main([str(src_dir), '--output-dir', str(src_dir), '--yes'])
    assert code == fei.EXIT_ERROR


# ------------------------------------------------- review-found regressions

def test_symlink_in_output_dir_never_reaches_original(tmp_path):
    src_dir = tmp_path / 'input'
    out_dir = tmp_path / 'out'
    src_dir.mkdir(); out_dir.mkdir()
    src_a = make_mef(src_dir / 'LLAMAS_A_mef.fits', exptime='rexptime')
    src_b = make_mef(src_dir / 'LLAMAS_B_mef.fits', exptime='rexptime')
    (out_dir / 'LLAMAS_A_mef.fits').symlink_to(src_b)      # stale staging link
    hash_b = sha(src_b)
    audit = scan(src_a)
    fei.propose_case_a(audit)
    for f in audit.fixes:
        f.approved = True
    with pytest.raises(FileExistsError):                   # no --overwrite: refuse
        fei.write_repaired(audit, out_dir, overwrite=False, obs_log_name=None)
    dest = fei.write_repaired(audit, out_dir, overwrite=True, obs_log_name=None)
    assert sha(src_b) == hash_b                            # B untouched
    assert not dest.is_symlink()
    with fits.open(dest) as hdul:
        assert hdul[0].header['FIXEXP ORIG'] == 'LLAMAS_A_mef.fits'


def test_ask_rejects_whole_words():
    ask = fei.make_ask(scripted('skip', 'k'))
    assert ask("READ-MDE", 'fskq') == 'k'                  # 'skip' must NOT mean slow


def test_finalize_derives_nothing_for_rejected_file(tmp_path):
    audit = scan(make_mef(tmp_path / 'a_mef.fits', exptime='rexptime'))
    fei.propose_case_a(audit)                              # user rejects everything
    fei.finalize_fixes(audit)
    assert not any(f.approved for f in audit.fixes)
    assert audit.fix_for('EXPTIME', 'all_extensions') is None
    assert audit.fix_for('AEXPTIME') is None


def test_mixed_extension_exptime_is_inconsistent(tmp_path):
    audit = scan(make_mef(tmp_path / 'a_mef.fits', ext_exptime=[25.0, 0.0]))
    assert audit.ext_exptime_state == 'inconsistent'
    fei.propose_case_a(audit)
    assert audit.fix_for('EXPTIME') is None                # never used as donor


def test_rexptime_prefers_filled_exptime_over_sexptime(tmp_path):
    path = tmp_path / 'a_mef.fits'
    make_mef(path, exptime=30.0)
    with fits.open(path, mode='update') as hdul:
        hdul[0].header['SEXPTIME'] = 30.02
        hdul[0].header.append(fits.Card('REXPTIME'))       # blank card
    audit = scan(path)
    fei.propose_case_a(audit)
    fix = audit.fix_for('REXPTIME')
    assert fix is not None and fix.value == 30.0 and fix.source == 'EXPTIME'


def test_csv_with_utf8_bom(tmp_path):
    csv_path = tmp_path / 'log.csv'
    csv_path.write_bytes(
        '﻿file name,local time,read mode,exp. time,type,object,comment\n'
        'LLAMAS_2024-11-29T00_38_42.162_mef.fits,21:00,slow,10,Std,Feige 110,\n'
        .encode('utf-8'))
    rows = ol.load_obs_log(csv_path)
    assert rows and rows[0].exp_time == 10.0


def test_filename_ditto_is_duplicate_not_garbage_stem(tmp_path):
    openpyxl = pytest.importorskip('openpyxl')
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(['file name', 'local time', 'read mode', 'exp. time', 'type',
               'object', 'comment'])
    ws.append(['LLAMAS_2024-11-28T23_37_24.874_mef', None, 'fast', 5, 'cal', 'sky', None])
    ws.append(['"', None, 'slow', 20, 'cal', 'sky', None])
    path = tmp_path / 'ditto.xlsx'
    wb.save(path)
    rows = ol.load_obs_log(path)
    assert all(r.stem == 'LLAMAS_2024-11-28T23_37_24.874' for r in rows)
    index = ol.build_log_index(rows)
    assert len(index['LLAMAS_2024-11-28T23_37_24.874']) == 2   # kept as conflict


def test_log_vs_infile_exptime_disagreement_noted(tmp_path):
    audit = scan(make_mef(tmp_path / 'a_mef.fits', exptime='rexptime'))
    fei.propose_case_a(audit)                              # unapproved (declined)
    audit.chosen_log_row = ol.LogRow(stem=audit.stem, raw_filename=None,
                                     exp_time=12.0, source='log', row_number=4)
    fei.propose_from_log(audit)
    assert audit.fix_for('EXPTIME').value == 10.0          # in-file donor wins
    assert any('12.0' in n and 'manual entry' in n for n in audit.notes)


def test_declined_readmode_fix_replaceable_by_log(tmp_path):
    audit = scan(make_mef(tmp_path / 'a_mef.fits'))
    audit.propose('READ-MDE', 'FAST', 'QA inference (LOW, 3S/5F)', 'C')  # declined
    audit.chosen_log_row = ol.LogRow(stem=audit.stem, raw_filename=None,
                                     read_mode='SLOW', source='log', row_number=4)
    fei.propose_from_log(audit)
    fix = audit.fix_for('READ-MDE')
    assert fix.value == 'SLOW' and fix.case == 'B'         # unapproved fix replaced


def test_no_gaps_counts_skipped_files(tmp_path):
    a1 = scan(make_mef(tmp_path / 'a_mef.fits'))
    a1.skipped = True
    assert not fei._no_gaps([a1])                          # skipped gaps still count
    a2 = scan(make_mef(tmp_path / 'b_mef.fits', exptime=10.0, readmde='SLOW'))
    with fits.open(tmp_path / 'b_mef.fits') as hdul:
        pass
    for key in ('OBJECT', 'UTC', 'DATE-OBS', 'DATE', 'RA', 'DEC', 'AIRMASS',
                'MJD-OBS', 'REXPTIME'):
        fix = a2.propose(key, 'x', 't', 'A')
        fix.approved = True
    assert fei._no_gaps([a2])


def test_report_log_marker(tmp_path, capsys):
    audit = scan(make_mef(tmp_path / 'a_mef.fits'))
    audit.log_rows = [ol.LogRow(stem=audit.stem, raw_filename=None,
                                read_mode='SLOW', exp_time=10.0,
                                source='log', row_number=2)]
    audit.chosen_log_row = audit.log_rows[0]
    fei.print_report([audit])
    out = capsys.readouterr().out
    line = [l for l in out.splitlines() if audit.stem in l][0]
    assert 'LOG' in line
