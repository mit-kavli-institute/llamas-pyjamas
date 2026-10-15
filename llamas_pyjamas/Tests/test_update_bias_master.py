"""Unit tests for the QA-gated master bias updater.

llamas-checks is NOT required: the QA step is monkeypatched, and one test
verifies the pipeline still imports with ``llamas_checks`` blocked. Runnable
with pytest or as a plain script
(`python -m llamas_pyjamas.Tests.test_update_bias_master`).
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits

import llamas_pyjamas.Scripts.update_bias_master as ubm


def _v(path, readmode, verdict):
    return {'path': path, 'readmode': readmode, 'verdict': verdict}


def write_frame(directory, name, prodcatg='CAL.R-BIA', readmode='SLOW'):
    """Write a tiny two-HDU FITS file with the headers the script inspects."""
    primary = fits.PrimaryHDU()
    if prodcatg is not None:
        primary.header['PRODCATG'] = prodcatg
    if readmode is not None:
        primary.header['READ-MDE'] = readmode
    image = fits.ImageHDU(np.full((4, 4), 500.0 + len(name)))
    path = os.path.join(directory, name)
    fits.HDUList([primary, image]).writeto(path)
    return path


# ---------------------------------------------------------------------------
# select_bias
# ---------------------------------------------------------------------------

def test_first_pass_wins_over_earlier_warn():
    verdicts = [_v('a', 'SLOW', 'warn'), _v('b', 'SLOW', 'pass'), _v('c', 'SLOW', 'pass')]
    assert ubm.select_bias(verdicts) == {'SLOW': 'b', 'FAST': None}


def test_warn_fallback_when_no_pass():
    verdicts = [_v('a', 'FAST', 'fail'), _v('b', 'FAST', 'warn'), _v('c', 'FAST', 'warn')]
    assert ubm.select_bias(verdicts)['FAST'] == 'b'


def test_warn_fallback_can_be_disabled():
    verdicts = [_v('a', 'FAST', 'warn')]
    assert ubm.select_bias(verdicts, allow_warn=False)['FAST'] is None


def test_fail_and_error_never_selected():
    verdicts = [_v('a', 'SLOW', 'fail'), _v('b', 'SLOW', 'error'), _v('c', 'FAST', 'error')]
    assert ubm.select_bias(verdicts) == {'SLOW': None, 'FAST': None}


def test_modes_are_independent():
    verdicts = [_v('s1', 'SLOW', 'pass'), _v('f1', 'FAST', 'fail'), _v('f2', 'FAST', 'pass')]
    assert ubm.select_bias(verdicts) == {'SLOW': 's1', 'FAST': 'f2'}


def test_unchecked_only_accepted_in_no_qa_mode():
    verdicts = [_v('a', 'SLOW', 'unchecked')]
    assert ubm.select_bias(verdicts)['SLOW'] is None
    assert ubm.select_bias(verdicts, accept_unchecked=True)['SLOW'] == 'a'


def test_select_bias_frames_takes_whole_tier_without_mixing():
    verdicts = [_v('a', 'SLOW', 'warn'), _v('b', 'SLOW', 'pass'), _v('c', 'SLOW', 'pass'),
                _v('d', 'FAST', 'warn'), _v('e', 'FAST', 'warn'), _v('f', 'FAST', 'fail')]
    assert ubm.select_bias_frames(verdicts) == {'SLOW': ['b', 'c'], 'FAST': ['d', 'e']}
    assert ubm.select_bias_frames(verdicts, allow_warn=False) == {'SLOW': ['b', 'c'], 'FAST': []}


def test_install_bias_medians_several_frames_and_copies_one(tmp_path):
    raw = str(tmp_path / 'raw')
    os.makedirs(raw)
    frames = [write_frame(raw, f'LLAMAS_0{i}_mef.fits', readmode='SLOW') for i in range(3)]
    single = write_frame(raw, 'LLAMAS_09_mef.fits', readmode='FAST')
    out = str(tmp_path / 'out')

    installed = ubm.install_bias({'SLOW': frames, 'FAST': [single]}, out, out)

    assert installed == ['SLOW', 'FAST']
    with fits.open(os.path.join(out, 'slow_master_bias.fits')) as hdul:
        assert hdul[0].header['NCOMBINE'] == 3
        expected = np.median(np.stack([fits.getdata(f, 1) for f in frames]), axis=0)
        np.testing.assert_allclose(hdul[1].data, expected)
    assert np.array_equal(fits.getdata(os.path.join(out, 'fast_master_bias.fits'), 1),
                          fits.getdata(single, 1))
    # --no-combine semantics: only the first frame, copied.
    installed = ubm.install_bias({'SLOW': frames, 'FAST': []}, out, out, combine=False)
    assert installed == ['SLOW']
    assert np.array_equal(fits.getdata(os.path.join(out, 'slow_master_bias.fits'), 1),
                          fits.getdata(frames[0], 1))


# ---------------------------------------------------------------------------
# find_bias_candidates
# ---------------------------------------------------------------------------

def test_find_bias_candidates_filters_by_prodcatg(tmp_path):
    raw = str(tmp_path)
    b1 = write_frame(raw, 'LLAMAS_01_mef.fits', readmode='SLOW')
    write_frame(raw, 'LLAMAS_02_mef.fits', prodcatg='CAL.R-FLAT')
    write_frame(raw, 'LLAMAS_03_mef.fits', prodcatg='SCIENCE')
    b4 = write_frame(raw, 'LLAMAS_04_mef.fits', readmode='FAST')
    write_frame(raw, 'LLAMAS_05_white.fits')          # name-filtered
    with open(os.path.join(raw, 'LLAMAS_00_broken.fits'), 'w') as fh:
        fh.write('not a fits file')                    # unreadable, skipped

    found = ubm.find_bias_candidates(raw)
    assert [c['path'] for c in found] == [b1, b4]
    assert [c['readmode'] for c in found] == ['SLOW', 'FAST']


# ---------------------------------------------------------------------------
# main() end to end, QA step stubbed
# ---------------------------------------------------------------------------

@pytest.fixture
def night(tmp_path):
    raw = tmp_path / 'raw'
    raw.mkdir()
    frames = {
        's_warn': write_frame(str(raw), 'LLAMAS_01_mef.fits', readmode='SLOW'),
        's_pass': write_frame(str(raw), 'LLAMAS_02_mef.fits', readmode='SLOW'),
        'f_fail': write_frame(str(raw), 'LLAMAS_03_mef.fits', readmode='FAST'),
        'f_warn': write_frame(str(raw), 'LLAMAS_04_mef.fits', readmode='FAST'),
    }
    out = tmp_path / 'out'
    return raw, out, frames


def fake_qa(verdict_by_path):
    def _qa(candidates, rule_set='BIAS'):
        return [{**c, 'verdict': verdict_by_path[c['path']]} for c in candidates]
    return _qa


def run_main(args):
    with pytest.raises(SystemExit) as exc:
        ubm.main(args)
    return exc.value.code


def test_main_installs_first_pass_and_warn_fallback(night, monkeypatch, capsys):
    raw, out, f = night
    monkeypatch.setattr(ubm, 'HAVE_LLAMAS_CHECKS', True)
    monkeypatch.setattr(ubm, 'qa_verdicts', fake_qa({
        f['s_warn']: 'warn', f['s_pass']: 'pass', f['f_fail']: 'fail', f['f_warn']: 'warn'}))
    bias_dir, calib_dir = str(out / 'Bias'), str(out / 'mastercalib')

    code = run_main([str(raw), '--bias-dir', bias_dir, '--calib-dir', calib_dir])
    captured = capsys.readouterr()

    assert code == 0
    assert 'Master BIAS files updated successfully (SLOW, FAST).' in captured.out
    for d in (bias_dir, calib_dir):
        slow = fits.getdata(os.path.join(d, 'slow_master_bias.fits'), 1)
        fast = fits.getdata(os.path.join(d, 'fast_master_bias.fits'), 1)
        assert np.array_equal(slow, fits.getdata(f['s_pass'], 1))
        assert np.array_equal(fast, fits.getdata(f['f_warn'], 1))


def test_main_no_warn_fallback_updates_only_slow(night, monkeypatch, capsys):
    raw, out, f = night
    monkeypatch.setattr(ubm, 'HAVE_LLAMAS_CHECKS', True)
    monkeypatch.setattr(ubm, 'qa_verdicts', fake_qa({
        f['s_warn']: 'warn', f['s_pass']: 'pass', f['f_fail']: 'fail', f['f_warn']: 'warn'}))
    bias_dir = str(out)

    code = run_main([str(raw), '--bias-dir', bias_dir, '--calib-dir', bias_dir,
                     '--no-warn-fallback'])
    captured = capsys.readouterr()

    assert code == 0
    assert 'FAST master bias not updated' in captured.out
    assert 'updated successfully (SLOW).' in captured.out
    assert os.path.exists(os.path.join(bias_dir, 'slow_master_bias.fits'))
    assert not os.path.exists(os.path.join(bias_dir, 'fast_master_bias.fits'))


def test_main_fails_when_nothing_acceptable(night, monkeypatch, capsys):
    raw, out, f = night
    monkeypatch.setattr(ubm, 'HAVE_LLAMAS_CHECKS', True)
    monkeypatch.setattr(ubm, 'qa_verdicts', fake_qa({p: 'fail' for p in f.values()}))

    code = run_main([str(raw), '--bias-dir', str(out), '--calib-dir', str(out)])
    captured = capsys.readouterr()

    assert code == 1
    assert captured.err.startswith('Update Failed')
    assert not os.path.exists(os.path.join(str(out), 'slow_master_bias.fits'))


def test_main_fails_on_missing_directory(tmp_path, capsys):
    code = run_main([str(tmp_path / 'nope')])
    assert code == 1
    assert 'Update Failed' in capsys.readouterr().err


def test_main_refuses_without_llamas_checks(night, monkeypatch, capsys):
    raw, out, _ = night
    monkeypatch.setattr(ubm, 'HAVE_LLAMAS_CHECKS', False)

    code = run_main([str(raw), '--bias-dir', str(out), '--calib-dir', str(out)])
    err = capsys.readouterr().err

    assert code == 1
    assert 'llamas-checks is not installed' in err
    assert '--no-qa' in err


def test_main_no_qa_combines_every_frame_per_mode(night, monkeypatch, capsys):
    raw, out, f = night
    monkeypatch.setattr(ubm, 'HAVE_LLAMAS_CHECKS', False)
    bias_dir = str(out)

    code = run_main([str(raw), '--bias-dir', bias_dir, '--calib-dir', bias_dir, '--no-qa'])
    captured = capsys.readouterr()

    assert code == 0
    assert 'WITHOUT QA vetting' in captured.err
    # Every unvetted frame of a mode is median-combined.
    for mode, keys in (('slow', ('s_warn', 's_pass')), ('fast', ('f_fail', 'f_warn'))):
        expected = np.median(np.stack([fits.getdata(f[k], 1) for k in keys]), axis=0)
        np.testing.assert_allclose(fits.getdata(os.path.join(bias_dir, f'{mode}_master_bias.fits'), 1),
                                   expected)


def test_main_no_combine_takes_first_per_mode(night, monkeypatch, capsys):
    raw, out, f = night
    monkeypatch.setattr(ubm, 'HAVE_LLAMAS_CHECKS', False)
    bias_dir = str(out)

    code = run_main([str(raw), '--bias-dir', bias_dir, '--calib-dir', bias_dir, '--no-qa',
                     '--no-combine'])
    captured = capsys.readouterr()

    assert code == 0
    assert np.array_equal(fits.getdata(os.path.join(bias_dir, 'slow_master_bias.fits'), 1),
                          fits.getdata(f['s_warn'], 1))
    assert np.array_equal(fits.getdata(os.path.join(bias_dir, 'fast_master_bias.fits'), 1),
                          fits.getdata(f['f_fail'], 1))


# ---------------------------------------------------------------------------
# Isolation: the pipeline must not depend on llamas_checks
# ---------------------------------------------------------------------------

def test_pipeline_imports_without_llamas_checks():
    """Block llamas_checks in a fresh interpreter and import the pipeline entry points."""
    code = (
        "import sys; sys.modules['llamas_checks'] = None\n"
        "import llamas_pyjamas.Bias\n"
        "import llamas_pyjamas.reduce\n"
        "import llamas_pyjamas.Scripts.update_bias_master as m\n"
        "import llamas_pyjamas.Scripts.update_quicklook_calibs as q\n"
        "import llamas_pyjamas.Postprocessing.build_quicklook_skyref\n"
        "assert m.HAVE_LLAMAS_CHECKS is False and q.HAVE_LLAMAS_CHECKS is False\n"
        "print('ok')\n"
    )
    proc = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert 'ok' in proc.stdout


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
