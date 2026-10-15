"""Tests for the nightly quick-look calibration installer, the raw-MEF median
combiner and the per-fibre sky reference builder.

llamas-checks is NOT required: QA is monkeypatched. Frames are tiny synthetic
MEFs with the headers the code inspects. Runnable with pytest or as a script
(`python -m llamas_pyjamas.Tests.test_update_quicklook_calibs`).
"""

import json
import os

import numpy as np
import pytest
from astropy.io import fits

import llamas_pyjamas.Scripts.update_quicklook_calibs as uqc
from llamas_pyjamas.Bias.llamasBias import median_combine_mef
from llamas_pyjamas.Postprocessing import build_quicklook_skyref as bqs

NROW, NCOL = 2048, 64          # tall enough for estimate_residual_bias's clean edge rows
NFIB = 6


def _fiberimg():
    """Six horizontal fibres, 5 rows each, between rows 300 and 1700."""
    fiberimg = np.full((NROW, NCOL), -1, dtype=np.int16)
    for ifib in range(NFIB):
        row = 300 + ifib * 250
        fiberimg[row:row + 5] = ifib
    return fiberimg


def write_fibre_cache(path):
    """A quick-look fibre-label cache with one detector, red 2A."""
    meta = {'format_version': 1, 'build_time_utc': 'test', 'calib_dir': '',
            'detectors': {'red_2_A': {'source': 'none', 'size': 0, 'mtime': 0,
                                      'nfibers': NFIB, 'shape': [NROW, NCOL]}}}
    np.savez_compressed(path, meta=np.str_(json.dumps(meta)),
                        red_2_A_fiberimg=_fiberimg(), red_2_A_nfibers=np.int32(NFIB),
                        red_2_A_bench=np.str_('2'), red_2_A_side=np.str_('A'),
                        red_2_A_channel=np.str_('red'))
    return path


def write_mef(path, prodcatg, readmode='SLOW', exptime=None, level=500.0, fibre_flux=0.0,
              fibre_scale=None, seed=0, detectors=(('red', 2, 'A'), ('green', 2, 'A'))):
    """Raw-like MEF: primary header keys + one 2048x64 uint16 extension per detector.

    Each extension is ``level`` plus noise; fibre pixels (same label image for every
    detector) get ``fibre_flux`` extra, scaled per fibre by ``fibre_scale``.
    """
    rng = np.random.default_rng(seed)
    primary = fits.PrimaryHDU()
    primary.header['PRODCATG'] = prodcatg
    primary.header['READ-MDE'] = readmode
    if exptime is not None:
        primary.header['HIERARCH OBS EXPTIME REQ'] = exptime
    hdus = [primary]
    fiberimg = _fiberimg()
    for color, bench, side in detectors:
        data = rng.normal(level, 2.0, (NROW, NCOL))
        for ifib in range(NFIB):
            scale = 1.0 if fibre_scale is None else fibre_scale[ifib]
            data[fiberimg == ifib] += fibre_flux * scale
        hdu = fits.ImageHDU(np.clip(np.round(data), 0, 65535).astype(np.uint16))
        hdu.header['COLOR'] = color
        hdu.header['BENCH'] = bench
        hdu.header['SIDE'] = side
        hdu.header['DATASEC'] = f'[1:{NCOL}, 1:{NROW}]'
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


# ---------------------------------------------------------------------------
# median_combine_mef
# ---------------------------------------------------------------------------

def test_median_combine_matches_detectors_and_subtracts(tmp_path):
    frames = [write_mef(str(tmp_path / f'b{i}.fits'), 'CAL.R-BIA', level=500 + 10 * i, seed=i)
              for i in range(3)]
    # Drop the green detector from one frame: the red median still uses 3, green only 2.
    with fits.open(frames[1], mode='update') as hdul:
        del hdul[2]
    sub = write_mef(str(tmp_path / 'sub.fits'), 'CAL.R-BIA', level=100.0, seed=9)

    out = median_combine_mef(frames, str(tmp_path / 'out.fits'), subtract=sub,
                             extra_primary={'DARKEXPT': (600.0, 'test')})

    with fits.open(out) as hdul:
        assert hdul[0].header['NCOMBINE'] == 3
        assert hdul[0].header['DARKEXPT'] == 600.0
        assert hdul[0].header['COMBSUB'] == 'sub.fits'
        red, green = hdul[1], hdul[2]
        assert red.header['NCOMBINE'] == 3 and green.header['NCOMBINE'] == 2
        assert red.data.dtype.kind == 'f' and red.data.dtype.itemsize == 4
        stack = np.stack([fits.getdata(f, 1).astype(float) for f in frames])
        expected = np.median(stack, axis=0) - fits.getdata(sub, 1).astype(float)
        np.testing.assert_allclose(red.data, expected, atol=1e-3)
        assert abs(np.median(green.data) - (510 - 100)) < 2   # frames 0 and 2 only


# ---------------------------------------------------------------------------
# dark selection and install
# ---------------------------------------------------------------------------

def _v(path, readmode, verdict, exptime):
    return {'path': path, 'readmode': readmode, 'verdict': verdict, 'exptime': exptime}


def test_select_dark_frames_keeps_longest_exposure_only():
    verdicts = [_v('d600a', 'SLOW', 'pass', 600.0), _v('d600b', 'SLOW', 'pass', 600.0),
                _v('d1350', 'SLOW', 'pass', 1350.0), _v('dwarn', 'SLOW', 'warn', 1800.0),
                _v('f', 'FAST', 'warn', 60.0), _v('fnoexp', 'FAST', 'pass', None)]
    chosen, exptimes = uqc.select_dark_frames(verdicts)
    assert chosen['SLOW'] == ['d1350'] and exptimes['SLOW'] == 1350.0
    assert chosen['FAST'] == ['f'] and exptimes['FAST'] == 60.0


def test_install_darks_is_bias_subtracted_and_stamped(tmp_path):
    bias_dir = tmp_path / 'bias'
    bias_dir.mkdir()
    write_mef(str(bias_dir / 'slow_master_bias.fits'), 'CAL.R-BIA', level=500.0, seed=1)
    darks = [write_mef(str(tmp_path / f'd{i}.fits'), 'CAL.R-DRK', exptime=600.0,
                       level=500.0 + 30.0, seed=10 + i) for i in range(2)]

    installed = uqc.install_darks({'SLOW': darks, 'FAST': []}, {'SLOW': 600.0, 'FAST': None},
                                  str(bias_dir), str(tmp_path / 'calib'))

    assert installed == ['SLOW']
    for d in (bias_dir, tmp_path / 'calib'):
        with fits.open(d / 'slow_master_dark.fits') as hdul:
            assert hdul[0].header['DARKEXPT'] == 600.0
            assert hdul[0].header['READ-MDE'] == 'SLOW'
            assert hdul[0].header['NCOMBINE'] == 2
            assert abs(np.median(hdul[1].data) - 30.0) < 1.0     # dark minus bias


# ---------------------------------------------------------------------------
# sky reference
# ---------------------------------------------------------------------------

@pytest.fixture
def skyref_inputs(tmp_path):
    cache = write_fibre_cache(str(tmp_path / 'fiberimg.npz'))
    bias = write_mef(str(tmp_path / 'slow_master_bias.fits'), 'CAL.R-BIA', level=500.0, seed=1)
    pattern = np.array([1.0, 0.8, 1.2, 1.0, 0.9, 1.1])
    twilights = [write_mef(str(tmp_path / f'sky{i}.fits'), 'CAL.R-SKY', exptime=5.0, level=500.0,
                           fibre_flux=200.0 * (i + 1), fibre_scale=pattern, seed=20 + i)
                 for i in range(3)]
    return cache, bias, twilights, pattern


def test_build_quicklook_skyref_recovers_fibre_pattern(skyref_inputs, tmp_path):
    cache, bias, twilights, pattern = skyref_inputs
    out = bqs.build_quicklook_skyref(twilights, out=str(tmp_path / 'ref.npz'), slow_bias=bias,
                                     fibre_cache=cache)
    ref = bqs.load_quicklook_skyref(out)
    assert ref['meta']['mode'] == 'twilight'
    assert ref['meta']['detectors']['red_2_A']['nframes'] == 3
    assert ('green', '2', 'A') not in ref                # not in the fibre cache
    np.testing.assert_allclose(ref[('red', '2', 'A')], pattern / np.median(pattern), rtol=0.02)
    assert bqs.skyref_is_fresh(out, fibre_cache=cache)


def test_skyref_stale_when_fibre_cache_changes(skyref_inputs, tmp_path):
    cache, bias, twilights, _ = skyref_inputs
    out = bqs.build_quicklook_skyref(twilights[:1], out=str(tmp_path / 'ref.npz'), slow_bias=bias,
                                     fibre_cache=cache)
    assert bqs.skyref_is_fresh(out, fibre_cache=cache)
    write_fibre_cache(cache)                             # rebuilt: new size/mtime
    os.utime(cache, (0, 0))
    assert not bqs.skyref_is_fresh(out, fibre_cache=cache)


def test_skyref_science_mode_masks_sources(skyref_inputs, tmp_path):
    cache, bias, _, pattern = skyref_inputs
    bright = pattern.copy()
    bright[2] = 5.0                                      # a source on fibre 2
    frame = write_mef(str(tmp_path / 'sci.fits'), 'SCI.R-DT', exptime=100.0, level=500.0,
                      fibre_flux=200.0, fibre_scale=bright, seed=7)
    out = bqs.build_quicklook_skyref([frame], out=str(tmp_path / 'ref.npz'), slow_bias=bias,
                                     science=True, fibre_cache=cache)
    ref = bqs.load_quicklook_skyref(out)[('red', '2', 'A')]
    assert np.isnan(ref[2]) and np.isfinite(ref[[0, 1, 3, 4, 5]]).all()


def test_skyref_refuses_saturated_and_dark_detectors(skyref_inputs, tmp_path):
    cache, bias, _, pattern = skyref_inputs
    saturated = write_mef(str(tmp_path / 'sat.fits'), 'CAL.R-SKY', exptime=5.0, level=500.0,
                          fibre_flux=65000.0, seed=3)
    closed = write_mef(str(tmp_path / 'closed.fits'), 'CAL.R-SKY', exptime=5.0, level=500.0,
                       fibre_flux=0.0, seed=4)
    with pytest.raises(ValueError):
        bqs.build_quicklook_skyref([saturated, closed], out=str(tmp_path / 'ref.npz'),
                                   slow_bias=bias, fibre_cache=cache)


# ---------------------------------------------------------------------------
# main() end to end, QA stubbed
# ---------------------------------------------------------------------------

def run_main(args):
    with pytest.raises(SystemExit) as exc:
        uqc.main(args)
    return exc.value.code


def test_main_installs_all_three_stages(tmp_path, monkeypatch, capsys):
    raw = tmp_path / 'raw'
    raw.mkdir()
    pattern = np.array([1.0, 0.8, 1.2, 1.0, 0.9, 1.1])
    b1 = write_mef(str(raw / 'LLAMAS_01_mef.fits'), 'CAL.R-BIA', level=500.0, seed=1)
    b2 = write_mef(str(raw / 'LLAMAS_02_mef.fits'), 'CAL.R-BIA', level=500.0, seed=2)
    d1 = write_mef(str(raw / 'LLAMAS_03_mef.fits'), 'CAL.R-DRK', exptime=600.0, level=530.0, seed=3)
    s1 = write_mef(str(raw / 'LLAMAS_04_mef.fits'), 'CAL.R-SKY', exptime=5.0, level=500.0,
                   fibre_flux=400.0, fibre_scale=pattern, seed=4)
    s2 = write_mef(str(raw / 'LLAMAS_05_mef.fits'), 'CAL.R-SKY', exptime=5.0, level=500.0,
                   fibre_flux=800.0, fibre_scale=pattern, seed=5)
    write_mef(str(raw / 'LLAMAS_06_mef.fits'), 'SCI.R-DT', exptime=100.0, level=500.0, seed=6)
    cache = write_fibre_cache(str(tmp_path / 'fiberimg.npz'))
    out = tmp_path / 'out'

    monkeypatch.setattr(uqc, 'HAVE_LLAMAS_CHECKS', True)
    monkeypatch.setattr(uqc, 'FIBRE_CACHE', cache)
    verdict = {b1: 'pass', b2: 'warn', d1: 'pass', s1: 'pass', s2: 'pass'}
    monkeypatch.setattr(uqc, 'qa_verdicts',
                        lambda cands, rule_set: [{**c, 'verdict': verdict[c['path']]} for c in cands])

    code = run_main([str(raw), '--bias-dir', str(out), '--calib-dir', str(out),
                     '--skyref-out', str(out / 'skyref.npz')])
    captured = capsys.readouterr()

    assert code == 0, captured.err
    assert 'Master BIAS files updated successfully (SLOW).' in captured.out
    assert 'Master DARK files updated successfully (SLOW: 600 s x 1).' in captured.out
    assert 'Quick-look sky reference updated successfully (2 frames)' in captured.out
    assert 'Quick-look calibrations updated (bias, dark, skyref).' in captured.out
    # Bias: only the pass frame (warn tier not mixed in) -> a plain copy of b1.
    assert np.array_equal(fits.getdata(out / 'slow_master_bias.fits', 1), fits.getdata(b1, 1))
    with fits.open(out / 'slow_master_dark.fits') as hdul:
        assert hdul[0].header['DARKEXPT'] == 600.0
        assert abs(np.median(hdul[1].data) - 30.0) < 1.0
    ref = bqs.load_quicklook_skyref(str(out / 'skyref.npz'))[('red', '2', 'A')]
    np.testing.assert_allclose(ref, pattern / np.median(pattern), rtol=0.02)


def test_main_fails_when_nothing_is_installable(tmp_path, monkeypatch, capsys):
    raw = tmp_path / 'raw'
    raw.mkdir()
    b1 = write_mef(str(raw / 'LLAMAS_01_mef.fits'), 'CAL.R-BIA')
    monkeypatch.setattr(uqc, 'HAVE_LLAMAS_CHECKS', True)
    monkeypatch.setattr(uqc, 'qa_verdicts',
                        lambda cands, rule_set: [{**c, 'verdict': 'fail'} for c in cands])
    code = run_main([str(raw), '--bias-dir', str(tmp_path / 'o'), '--calib-dir', str(tmp_path / 'o')])
    assert code == 1
    assert 'Update Failed: nothing installed' in capsys.readouterr().err
    assert not os.path.exists(tmp_path / 'o' / 'slow_master_bias.fits')


def test_main_refuses_without_llamas_checks(tmp_path, monkeypatch, capsys):
    raw = tmp_path / 'raw'
    raw.mkdir()
    write_mef(str(raw / 'LLAMAS_01_mef.fits'), 'CAL.R-BIA')
    monkeypatch.setattr(uqc, 'HAVE_LLAMAS_CHECKS', False)
    code = run_main([str(raw), '--bias-dir', str(tmp_path / 'o'), '--calib-dir', str(tmp_path / 'o')])
    assert code == 1
    assert '--no-qa' in capsys.readouterr().err


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
