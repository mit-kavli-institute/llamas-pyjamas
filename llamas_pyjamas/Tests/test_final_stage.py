"""The RSS processing-stage contract: which file of an exposure every tool uses.

Pins :mod:`llamas_pyjamas.File.llamasRSS`'s stage helpers (exposure_key, rss_stage, stage_files,
final_rss, final_siblings, wcs_quality, check_stage_consistency, write_rss_manifest) and the
producer guards (the fibre flat refuses flat-fielded input, carries FIBERWCS and the MASK bits
forward, flags fibres it cannot correct; the sky framework only takes 'ff' input).
"""

import json
import logging
import os
import shutil
import tempfile

import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.File.llamasRSS import (
    DuplicateStageError, MASK_NO_COUNTS, MASK_NOT_FLAT_FIELDED, MixedStageError,
    RegistrationMismatchError, RSSStageError, StageHeaderMismatchError,
    check_stage_consistency, exposure_key, final_rss, final_siblings, rss_stage, stage_files,
    wcs_quality, write_rss_manifest)
from llamas_pyjamas.Tests.rss_fixtures import (
    BENCHSIDES, EXP_ID, FIBER_IDS, LONG, N_WAVE, stage_name, write_exposure, write_rss)


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- identity
def test_exposure_key_same_for_long_and_consolidated_names():
    assert exposure_key(f'{LONG}_RSS_green.fits') == EXP_ID
    assert exposure_key(f'/x/{LONG}_RSS_red_FF_SKYSUB.fits') == EXP_ID
    assert exposure_key(f'{EXP_ID}_RSS_blue.fits') == EXP_ID


def test_rss_stage_from_markers(tmp):
    for stage in ('base', 'ff', 'ff_skysub'):
        p = write_rss(os.path.join(tmp, f'x_{stage}_RSS_green.fits'), stage=stage, rsstage=False)
        assert rss_stage(p) == stage


def test_rss_stage_markers_beat_inherited_rsstage(tmp, caplog):
    # e.g. an FF file whose primary header was copied from a base file that carried RSSSTAGE
    p = write_rss(os.path.join(tmp, 'x_RSS_green_FF.fits'), stage='ff', rsstage='base')
    with caplog.at_level(logging.WARNING):
        assert rss_stage(p) == 'ff'
    assert 'disagrees' in caplog.text


def test_rss_stage_header_only_file_is_base(tmp):
    p = os.path.join(tmp, 'y_RSS_green.fits')
    fits.PrimaryHDU().writeto(p)
    assert rss_stage(p) == 'base'


def test_rss_stage_rejects_residual_sky_without_flat(tmp):
    p = write_rss(os.path.join(tmp, 'x_RSS_green.fits'), stage='ff_skysub', rsstage=False)
    with fits.open(p, mode='update') as h:
        del h[0].header['FIBRFLAT']
    with pytest.raises(RSSStageError):
        rss_stage(p)


# ---------------------------------------------------------------- resolution
def test_final_rss_picks_most_corrected_stage(tmp):
    files = write_exposure(tmp)
    for ch in ('blue', 'green', 'red'):
        assert final_rss(files[('base', ch)]) == files[('ff_skysub', ch)]
        assert final_rss(files[('ff_skysub', ch)], 'base') == files[('base', ch)]
        assert final_rss(files[('base', ch)], 'ff') == files[('ff', ch)]
    assert stage_files(files[('ff', 'green')]) == {s: files[(s, 'green')]
                                                    for s in ('base', 'ff', 'ff_skysub')}


def test_final_rss_consolidated_single_file(tmp):
    p = write_rss(stage_name(tmp, 'green', None, consolidated=True), stage='ff_skysub')
    assert final_rss(p) == p
    assert stage_files(p) == {'ff_skysub': p}
    with pytest.raises(FileNotFoundError):
        final_rss(p, 'base')


def test_final_rss_lone_candidate_reads_no_header(tmp):
    p = os.path.join(tmp, f'{EXP_ID}_RSS_green.fits')
    with open(p, 'w') as fh:              # not FITS: any header read would fail
        fh.write('x')
    assert final_rss(p) == p


def test_derived_and_superseded_files_are_ignored(tmp):
    files = write_exposure(tmp, channels=('green',))
    write_rss(os.path.join(tmp, f'{LONG}_RSS_green_FF_EMPSKY.fits'), stage='ff')
    os.makedirs(os.path.join(tmp, 'superseded'))
    write_rss(os.path.join(tmp, 'superseded', f'{EXP_ID}_RSS_green.fits'), stage='ff_skysub')
    assert final_rss(files[('base', 'green')]) == files[('ff_skysub', 'green')]


def test_duplicate_stage_raises(tmp):
    files = write_exposure(tmp, channels=('green',))
    # an old consolidated survivor next to a newer long-named run
    write_rss(stage_name(tmp, 'green', None, consolidated=True), stage='ff_skysub')
    with pytest.raises(DuplicateStageError):
        final_rss(files[('base', 'green')])


def test_shared_exposure_id_stems_are_duplicates(tmp):
    a = f'LLAMAS_2026-10-16_01-18-48.8_SCI23_mef_zeroed_bias_corrected_flat_corrected_extract'
    b = f'LLAMAS_2026-10-16_01-18-48.8_SCI23_mef_bad_bias_corrected_flat_corrected_extract'
    pa = write_rss(os.path.join(tmp, f'{a}_RSS_green.fits'))
    write_rss(os.path.join(tmp, f'{b}_RSS_green.fits'))
    with pytest.raises(DuplicateStageError):
        stage_files(pa)


def test_final_siblings_mixed_stage(tmp):
    files = write_exposure(tmp, stages=('base', 'ff'))
    write_rss(stage_name(tmp, 'green', 'ff_skysub'), stage='ff_skysub', fiberwcs=False)
    with pytest.raises(MixedStageError):
        final_siblings(files[('base', 'green')])
    sib = final_siblings(files[('base', 'green')], allow_mixed=True)
    assert sib['green'].endswith('_FF_SKYSUB.fits') and sib['red'].endswith('_FF.fits')
    assert final_siblings(files[('base', 'green')], stage='ff') == {
        c: files[('ff', c)] for c in ('blue', 'green', 'red')}


# ---------------------------------------------------------------- astrometry quality
def test_wcs_quality_ranks(tmp):
    reg = write_rss(os.path.join(tmp, 'a_RSS_green.fits'), registered=True)
    rough = write_rss(os.path.join(tmp, 'b_RSS_green.fits'))
    failed_pin = write_rss(os.path.join(tmp, 'c_RSS_green.fits'), registered=True,
                           wcs_method='relative-source', wcs_refined=False)
    legacy_ff = write_rss(os.path.join(tmp, 'd_RSS_green_FF.fits'), stage='ff', fiberwcs=False)
    assert wcs_quality(reg)[0] == 2
    assert wcs_quality(rough)[0] == 1
    assert wcs_quality(failed_pin)[0] == 1
    assert wcs_quality(legacy_ff)[0] == 1


# ---------------------------------------------------------------- consistency
def test_registered_base_with_unregistered_final_raises(tmp):
    files = write_exposure(tmp, channels=('green',), per_stage={'base': {'registered': True}})
    final = files[('ff_skysub', 'green')]
    with pytest.raises(RegistrationMismatchError):
        check_stage_consistency(final)
    rep = check_stage_consistency(final, raise_errors=False)
    assert rep.better_wcs_stage == 'base'
    # the registered stage itself is fine
    check_stage_consistency(files[('base', 'green')])


def test_legacy_rough_everywhere_only_warns(tmp, caplog):
    files = write_exposure(tmp, channels=('green',))
    with caplog.at_level(logging.WARNING):
        rep = check_stage_consistency(files[('ff_skysub', 'green')])
    assert rep.better_wcs_stage is None
    assert 'no registered astrometry' in caplog.text


def test_header_mismatch_raises_and_flags_derived(tmp):
    files = write_exposure(tmp, channels=('green',),
                           per_stage={'base': {'header': {'AIRMASS': 1.17, 'SKYPED': True}}})
    final = files[('ff_skysub', 'green')]
    with pytest.raises(StageHeaderMismatchError) as exc:
        check_stage_consistency(final)
    assert 'AIRMASS' in str(exc.value) and 'SKYPED' in str(exc.value)
    rep = check_stage_consistency(final, allow_header_mismatch=True)
    assert set(rep.header_diffs['base']) == {'AIRMASS', 'SKYPED'}
    assert rep.derived_keys == ['SKYPED']


def test_stage_keywords_are_not_mismatches(tmp):
    files = write_exposure(tmp, channels=('green',))
    rep = check_stage_consistency(files[('ff_skysub', 'green')])
    assert rep.header_diffs == {}


# ---------------------------------------------------------------- producers
def _corrections(path, channel='green', benchsides=('1A',), factor=2.0, flags=None):
    hdus = [fits.PrimaryHDU()]
    hdus[0].header['METHOD'] = 'twilight'
    lo, hi = {'green': (4700.0, 7000.0)}[channel]
    for bs in benchsides:
        n = int(np.sum(BENCHSIDES == bs))
        cols = [fits.Column(name='FIBER_ID', format='J', array=np.arange(n)),
                fits.Column(name='CORRECTION', format=f'{N_WAVE}D',
                            array=np.full((n, N_WAVE), factor)),
                fits.Column(name='WAVE', format=f'{N_WAVE}D',
                            array=np.tile(np.linspace(lo, hi, N_WAVE), (n, 1))),
                fits.Column(name='FLAGS', format='I',
                            array=np.zeros(n, dtype=np.int16) if flags is None else flags)]
        h = fits.BinTableHDU.from_columns(cols)
        h.header['EXTNAME'] = f'{channel}_{bs[0]}_{bs[1]}'
        hdus.append(h)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_fibre_flat_carries_fiberwcs_mask_and_stage(tmp):
    from llamas_pyjamas.Flat.fibreFlat import apply_fibre_flat_to_rss
    base = write_rss(stage_name(tmp, 'green', 'base'), registered=True, noflat=True,
                     mask_bits={32: (slice(0, 3), slice(10, 20))},
                     zero_counts=(slice(5, 6), slice(0, 50)))
    with fits.open(base, mode='update') as h:          # as generate_rss would have flagged it
        h['MASK'].data[5, :50] |= MASK_NO_COUNTS
    corr = _corrections(os.path.join(tmp, 'corr.fits'), benchsides=('1A',))  # 1B has none
    out = apply_fibre_flat_to_rss(base, corr, stage_name(tmp, 'green', 'ff'))
    with fits.open(out) as h, fits.open(base) as b:
        assert h[0].header['FIBRFLAT'] and h[0].header['RSSSTAGE'] == 'ff'
        assert rss_stage(h) == 'ff'
        assert 'FIBERWCS' in h and 'NOFLAT' not in h
        np.testing.assert_array_equal(h['FIBERWCS'].data['RA'], b['FIBERWCS'].data['RA'])
        assert h['FIBERWCS'].header['WCSREFIN']
        m = h['MASK'].data
        assert np.all(m[0:3, 10:20] & 32) and np.all(m[5, :50] & MASK_NO_COUNTS)
        is_1b = BENCHSIDES == '1B'
        assert np.all(m[is_1b] & MASK_NOT_FLAT_FIELDED)
        assert not np.any(m[~is_1b] & MASK_NOT_FLAT_FIELDED)
        np.testing.assert_allclose(h['SKYSUB'].data[~is_1b], b['SKYSUB'].data[~is_1b] / 2.0,
                                   rtol=1e-6)
    with pytest.raises(RSSStageError):                  # never flat-field twice
        apply_fibre_flat_to_rss(out, corr, os.path.join(tmp, 'twice_RSS_green_FF.fits'))


def test_fibre_flat_refuses_misaligned_fiberwcs(tmp):
    from llamas_pyjamas.Flat.fibreFlat import apply_fibre_flat_to_rss
    base = write_rss(stage_name(tmp, 'green', 'base'))
    with fits.open(base, mode='update') as h:
        h['FIBERWCS'].data['FIBER_ID'][:] = h['FIBERWCS'].data['FIBER_ID'][::-1]
    corr = _corrections(os.path.join(tmp, 'corr.fits'), benchsides=('1A', '1B'))
    with pytest.raises(RSSStageError):
        apply_fibre_flat_to_rss(base, corr, stage_name(tmp, 'green', 'ff'))


def test_sky_framework_needs_ff_input(tmp):
    from llamas_pyjamas.Sky.skySubtract import subtract_sky_rss
    from llamas_pyjamas.Sky.skyConfig import SkySubtractConfig
    base = write_rss(os.path.join(tmp, 'x_RSS_green_FF.fits'), stage='base')  # misnamed base
    with pytest.raises(RSSStageError):
        subtract_sky_rss(base, config=SkySubtractConfig(method='scaled', qa_plots=False))
    ff = write_rss(os.path.join(tmp, 'y_RSS_green_FF.fits'), stage='ff')
    out = subtract_sky_rss(ff, config=SkySubtractConfig(method='scaled', qa_plots=False))
    with fits.open(out) as h:
        assert h[0].header['RSSSTAGE'] == 'ff_skysub' and rss_stage(h) == 'ff_skysub'


# ---------------------------------------------------------------- manifest / reduce resume
def test_manifest_records_final_and_run(tmp):
    files = write_exposure(tmp)
    run = {'written': {os.path.basename(files[('ff_skysub', 'green')])},
           'notes': {os.path.basename(files[('base', 'red')]): ['pedestal skipped']}}
    path = write_rss_manifest(tmp, run)
    m = json.load(open(path))['exposures'][EXP_ID]
    assert m['green']['stage'] == 'ff_skysub'
    assert m['green']['final'] == os.path.basename(files[('ff_skysub', 'green')])
    assert m['green']['regenerated'] == {'base': False, 'ff': False, 'ff_skysub': True}
    assert m['red']['notes'] == ['pedestal skipped']


def test_manifest_records_duplicates_as_error(tmp):
    write_exposure(tmp, channels=('green',))
    write_rss(stage_name(tmp, 'green', None, consolidated=True), stage='ff_skysub')
    m = json.load(open(write_rss_manifest(tmp)))['exposures'][EXP_ID]
    assert 'more than one file' in m['green']['error']


def test_reduce_base_selector_keeps_consolidated_final(tmp):
    from llamas_pyjamas.reduce import _base_stage_rss
    final = write_rss(stage_name(tmp, 'green', None, consolidated=True), stage='ff_skysub')
    base = write_rss(os.path.join(tmp, f'LLAMAS_2026-10-16_02-00-00.0_SCI24_mef_x_extract'
                                       f'_RSS_green.fits'))
    write_rss(os.path.join(tmp, f'{LONG}_RSS_green_EMPSKY.fits'))
    notes = {}
    got = _base_stage_rss(tmp, lambda p, m: notes.setdefault(os.path.basename(p), []).append(m))
    assert got == [base]
    assert notes == {os.path.basename(final): ['kept existing ff_skysub file; not regenerated']}


# ---------------------------------------------------------------- twilight T_i
def _twilight_inputs(tmp, zero_frac_by_fid):
    from types import SimpleNamespace
    wave = np.linspace(4700.0, 7000.0, N_WAVE)
    hdus = [fits.PrimaryHDU()]
    exts, meta = [], []
    for side in ('A', 'B'):
        cols = [fits.Column(name='FIBER_ID', format='J', array=np.arange(30)),
                fits.Column(name='SMOOTH', format=f'{N_WAVE}D', array=np.full((30, N_WAVE), 1e3)),
                fits.Column(name='WAVE', format=f'{N_WAVE}D', array=np.tile(wave, (30, 1)))]
        h = fits.BinTableHDU.from_columns(cols)
        h.header.update({'EXTNAME': f'green_1_{side}', 'CHANNEL': 'green', 'BENCH': '1',
                         'SIDE': side})
        hdus.append(h)
        counts = np.full((30, N_WAVE), 1e3)
        if side == 'A':
            for fid, frac in zero_frac_by_fid.items():
                counts[fid, :int(frac * N_WAVE)] = 0.0     # zero-filled blue end of the fibre
        exts.append(SimpleNamespace(counts=counts, wave=np.tile(wave, (30, 1)), dead_fibers=[]))
        meta.append({'channel': 'green', 'bench': '1', 'side': side})
    models = os.path.join(tmp, 'flat_smooth_models.fits')
    fits.HDUList(hdus).writeto(models)
    return {'extractions': exts, 'metadata': meta, 'primary_header': fits.Header()}, models


def test_twilight_ti_unbiased_by_zero_fill(tmp):
    from llamas_pyjamas.Flat.fibreFlat import compute_fibre_flat_twilight, MASK_FLAT_LAMP_ONLY
    # fibre 5: 40% of its pixels zero-filled (~37% of the 10-90% window) -> still measured;
    # fibre 7: 75% zero-filled -> too little data, takes the benchside median, flagged.
    twi, models = _twilight_inputs(tmp, {5: 0.40, 7: 0.75})
    corr_file = compute_fibre_flat_twilight(twi, models, tmp)
    with fits.open(corr_file) as h:
        d = h['green_1_A'].data
        c = np.nanmedian(d['CORRECTION'], axis=1)
        flags = d['FLAGS']
    ref = np.median(np.delete(c, [5, 7]))
    assert abs(c[5] / ref - 1.0) < 0.01          # nansum gave ~0.63 here
    assert abs(c[7] / ref - 1.0) < 0.01          # benchside median, not a silent 1.0
    assert flags[7] & MASK_FLAT_LAMP_ONLY and not flags[5] & MASK_FLAT_LAMP_ONLY


# ================================================================ step 3: guards
def _sensfunc(meta=None):
    from llamas_pyjamas.Flux.sensFunc import SensChannel, SensFunc
    chans = {}
    for c, (lo, hi) in {'blue': (3400, 4900), 'green': (4600, 7100), 'red': (6700, 9900)}.items():
        w = np.linspace(lo, hi, 200)
        chans[c] = SensChannel(channel=c, wave=w, sens=np.full(200, 1e-16), raw=np.full(200, 1e-16),
                               good=np.ones(200, bool))
    return SensFunc(channels=chans, meta=dict(meta or {}))


def test_missing_flam_raises_no_silent_fallback(tmp):
    from llamas_pyjamas.Combine.superRSS import MissingFluxCalibrationError, build_super_rss
    files = write_exposure(tmp, stages=('base',))
    with pytest.raises(MissingFluxCalibrationError):
        build_super_rss([files[('base', 'green')]])
    sr = build_super_rss([files[('base', 'green')]], plane='skysub')       # explicit opt-in
    assert sr.plane == 'skysub' and sr.bunit == 'counts/s' and sr.stage == 'base'


def test_partial_flam_raises(tmp):
    from llamas_pyjamas.Combine.superRSS import MissingFluxCalibrationError, build_super_rss
    files = write_exposure(tmp, stages=('base',))
    write_rss(files[('base', 'green')], flam=True)                         # green only
    with pytest.raises(MissingFluxCalibrationError) as exc:
        build_super_rss([files[('base', 'green')]])
    assert 'blue' in str(exc.value) and 'red' in str(exc.value)


def test_cube_records_plane_and_stage(tmp):
    from llamas_pyjamas.Combine.superRSS import build_super_rss
    from llamas_pyjamas.Combine.cube import combine_cube
    files = write_exposure(tmp, stages=('base',), flam=True)
    sr = build_super_rss([files[('base', 'green')]])
    cube = combine_cube(sr, 'green')
    out = cube.write(os.path.join(tmp, 'cube.fits'))
    hdr = fits.getheader(out)
    assert hdr['PLANE'] == 'flam' and hdr['RSSSTAGE'] == 'base'
    assert hdr['RSSFIL1'].endswith('_RSS_green.fits')


def test_cube_plane_inferred_for_legacy_cubes():
    from llamas_pyjamas.CubeViewer.cubeViewCube import cube_plane
    assert cube_plane(fits.Header({'PLANE': 'skysub'})) == 'skysub'
    assert cube_plane(fits.Header({'BUNIT': 'erg/s/cm2/Angstrom/arcsec2'})) == 'flam'
    assert cube_plane(fits.Header({'BUNIT': 'counts/s/arcsec2'})) == 'skysub'


def test_mixed_stage_exposures_raise_unless_allowed(tmp):
    from llamas_pyjamas.Combine.superRSS import build_super_rss
    from llamas_pyjamas.File.llamasRSS import MixedStageError
    a = write_exposure(tmp, stages=('base',), flam=True)
    other = 'LLAMAS_2026-10-16_00-54-54.4_SCI24_mef_bias_corrected_flat_corrected_extract'
    b = write_exposure(tmp, stages=('ff',), base=other, flam=True, fiberwcs=True)
    paths = [a[('base', 'green')], b[('ff', 'green')]]
    with pytest.raises(MixedStageError):
        build_super_rss(paths)
    sr = build_super_rss(paths, allow_mixed_stage=True)
    assert sr.stage == 'mixed' and sr.provenance()['OVR_MIX'] is True


def test_combiner_refuses_unregistered_final_when_base_is_registered(tmp):
    from llamas_pyjamas.Combine.superRSS import build_super_rss
    files = write_exposure(tmp, per_stage={'base': {'registered': True}}, flam=True)
    with pytest.raises(RegistrationMismatchError):
        build_super_rss([files[('ff_skysub', 'green')]])
    build_super_rss([files[('base', 'green')]])                           # the registered one


def test_combine_cli_reports_instead_of_traceback(tmp, capsys):
    from llamas_pyjamas.Combine.combineField import main
    files = write_exposure(tmp, stages=('base',))
    assert main([files[('base', 'green')], '-o', os.path.join(tmp, 'x.fits')]) == 2
    assert 'MissingFluxCalibrationError' in capsys.readouterr().err


# ---------------------------------------------------------------- sensfunc stage (C1)
def test_sensfunc_stage_rules(tmp, caplog):
    from llamas_pyjamas.Flux.fluxCalibrate import SensfuncStageError, apply_sensfunc
    ff = write_rss(os.path.join(tmp, 'a_RSS_green_FF.fits'), stage='ff')
    base = write_rss(os.path.join(tmp, 'a_RSS_green.fits'), stage='base')
    with fits.open(ff) as h:
        with pytest.raises(SensfuncStageError):                           # built at base
            apply_sensfunc(h, _sensfunc({'fibrflat': False, 'rssstage': 'base'}),
                           apply_extinction=False)
        with pytest.raises(SensfuncStageError):                           # unknown on FF science
            apply_sensfunc(h, _sensfunc(), apply_extinction=False)
        apply_sensfunc(h, _sensfunc(), apply_extinction=False, sensfunc_stage='ff_skysub')
        assert h['FLAM'].header['SENSPROV'] == 'declared'
        assert h['FLAM'].header['RSSSTAGE'] == 'ff'
        apply_sensfunc(h, _sensfunc({'fibrflat': False}), apply_extinction=False,
                       allow_stage_mismatch=True)
        assert h['FLAM'].header['OVR_STG'] is True
        with pytest.raises(SensfuncStageError):                           # declaration contradicts
            apply_sensfunc(h, _sensfunc({'fibrflat': True}), apply_extinction=False,
                           sensfunc_stage='base')
    with fits.open(base) as h, caplog.at_level(logging.WARNING):
        apply_sensfunc(h, _sensfunc(), apply_extinction=False)            # unknown on base: warns
        assert 'does not record' in caplog.text
        assert h['FLAM'].header['SENSPROV'] == 'unknown'


def test_sensfunc_stage_meta_round_trips(tmp):
    from llamas_pyjamas.Flux.sensFunc import SensFunc
    p = os.path.join(tmp, 's.fits')
    _sensfunc({'fibrflat': False, 'rssstage': 'base', 'stdfile': 'x_RSS_green.fits'}).save(p)
    m = SensFunc.load(p).meta
    assert m['fibrflat'] is False and m['rssstage'] == 'base' and m['stdfile'] == 'x_RSS_green.fits'


def test_flux_calibrate_file_is_atomic(tmp, monkeypatch):
    from llamas_pyjamas.Flux import fluxCalibrate
    p = write_rss(os.path.join(tmp, 'a_RSS_green.fits'))
    before = open(p, 'rb').read()

    def boom(*a, **k):
        raise OSError('disk full')
    monkeypatch.setattr(fluxCalibrate.os, 'replace', boom)
    with pytest.raises(OSError):
        fluxCalibrate.flux_calibrate_file(p, _sensfunc({'fibrflat': False}), apply_extinction=False)
    assert open(p, 'rb').read() == before
    assert not [f for f in os.listdir(tmp) if f.endswith('-tmp')]


# ---------------------------------------------------------------- promote_final_rss (F1)
def _legacy(tmp, **per_stage):
    """A legacy keep-intermediate reduction: base registered + FLAM, FF stages have neither."""
    ps = {'base': {'registered': True, 'flam': True}}
    ps.update(per_stage)
    return write_exposure(tmp, channels=('green',), per_stage=ps)


def test_promote_copies_registration_and_flam(tmp):
    from llamas_pyjamas.Postprocessing.promote_final_rss import main
    files = _legacy(tmp)
    sens = os.path.join(tmp, 'sens.fits')
    _sensfunc({'fibrflat': True, 'rssstage': 'ff_skysub'}).save(sens)
    assert main([tmp, '-y', '--sensfunc', sens, '--no-extinction']) == 0
    final, base = files[('ff_skysub', 'green')], files[('base', 'green')]
    with fits.open(final) as f, fits.open(base) as b:
        np.testing.assert_array_equal(f['FIBERWCS'].data['RA'], b['FIBERWCS'].data['RA'])
        np.testing.assert_array_equal(f['FIBERMAP'].data['DEC'], b['FIBERMAP'].data['DEC'])
        assert f['FIBERWCS'].header['WCSREFIN'] and 'FLAM' in f and 'FLAM_ERR' in f
        assert f['FLAM'].header['RSSSTAGE'] == 'ff_skysub'
    check_stage_consistency(final)                       # now nothing to object to
    from llamas_pyjamas.Combine.superRSS import build_super_rss
    sr = build_super_rss([final])
    assert sr.plane == 'flam' and sr.stage == 'ff_skysub'


def test_promote_dry_run_writes_nothing(tmp):
    from llamas_pyjamas.Postprocessing.promote_final_rss import main
    files = _legacy(tmp)
    final = files[('ff_skysub', 'green')]
    before = open(final, 'rb').read()
    assert main([tmp, '--dry-run']) == 1                 # FLAM still missing -> attention
    assert open(final, 'rb').read() == before


def test_promote_refuses_mismatched_fibre_rows(tmp):
    from llamas_pyjamas.Postprocessing.promote_final_rss import promote_directory
    files = _legacy(tmp)
    with fits.open(files[('ff_skysub', 'green')], mode='update') as h:
        h['FIBERMAP'].data['FIBER_ID'][:] = h['FIBERMAP'].data['FIBER_ID'][::-1]
    results, n_problem = promote_directory(tmp)
    assert n_problem == 1 and any('do not match' in p for p in results[0][4])
    with fits.open(files[('ff_skysub', 'green')]) as h:
        assert 'FIBERWCS' not in h


def test_promote_interrupted_write_leaves_original(tmp, monkeypatch):
    from llamas_pyjamas.Postprocessing import promote_final_rss as pf
    files = _legacy(tmp)
    final = files[('ff_skysub', 'green')]
    before = open(final, 'rb').read()

    def boom(*a, **k):
        raise OSError('interrupted')
    monkeypatch.setattr(pf.os, 'replace', boom)
    with pytest.raises(OSError):
        pf.promote_directory(tmp)
    assert open(final, 'rb').read() == before
    assert not [f for f in os.listdir(tmp) if f.endswith('-tmp')]


def test_promote_keep_duplicates_moves_others_aside(tmp):
    from llamas_pyjamas.Postprocessing.promote_final_rss import promote_directory
    files = _legacy(tmp)
    old = write_rss(stage_name(tmp, 'green', None, consolidated=True), stage='ff_skysub')
    results, _ = promote_directory(tmp)
    assert 'more than one file' in results[0][4][0]
    results, _ = promote_directory(tmp, keep_duplicates='long')
    assert not os.path.exists(old)
    assert os.path.exists(os.path.join(tmp, 'superseded', os.path.basename(old)))
    assert final_rss(files[('base', 'green')]) == files[('ff_skysub', 'green')]


def test_promote_header_sync_refuses_derived_keys(tmp):
    from llamas_pyjamas.Postprocessing.promote_final_rss import promote_directory
    files = _legacy(tmp, base={'registered': True, 'flam': True,
                               'header': {'AIRMASS': 1.17, 'SKYPED': True}})
    results, n_problem = promote_directory(tmp, sync_from='base')
    final = files[('ff_skysub', 'green')]
    with fits.open(final) as h:
        assert h[0].header['AIRMASS'] == 1.17            # plain key synced
        assert 'SKYPED' not in h[0].header               # data-change key refused
    assert any('re-reduce' in p for p in results[0][4]) and n_problem == 1


# ================================================================ propagation through every stage
def test_propagation_base_ff_skysub_consolidate_promote(tmp):
    """Each extension / key / mask bit of the audit table is preserved, regenerated or
    intentionally absent at base -> FF -> FF_SKYSUB -> (consolidate | register base + promote)."""
    from llamas_pyjamas.Flat.fibreFlat import apply_fibre_flat_to_rss
    from llamas_pyjamas.Sky.skySubtract import subtract_sky_rss
    from llamas_pyjamas.Sky.skyConfig import SkySubtractConfig
    from llamas_pyjamas.File.llamasRSS import apply_fibre_astrometry
    from llamas_pyjamas.Flux.fluxCalibrate import flux_calibrate_file
    from llamas_pyjamas.Postprocessing.promote_final_rss import promote_directory
    from llamas_pyjamas.reduce import consolidate_rss_files

    derived = {'SKYPED': True, 'SKYPEDSC': 'template', 'SKYLR': True, 'SKYLRTMP': False}
    base = write_rss(stage_name(tmp, 'green', 'base'), noflat=True, header=derived,
                     mask_bits={32: (slice(10, 12), slice(0, 400))},
                     zero_counts=(slice(20, 21), slice(0, 60)))
    with fits.open(base, mode='update') as h:
        h['MASK'].data[20, :60] |= MASK_NO_COUNTS      # as generate_rss flags zero-fill
    corr = _corrections(os.path.join(tmp, 'corr.fits'), benchsides=('1A', '1B'))
    ff = apply_fibre_flat_to_rss(base, corr, stage_name(tmp, 'green', 'ff'))
    sky = subtract_sky_rss(ff, config=SkySubtractConfig(method='scaled', qa_plots=False))
    assert sky == stage_name(tmp, 'green', 'ff_skysub')

    # post-processing as it used to happen: registration + FLAM on the BASE file only
    with fits.open(base, mode='update') as h:
        fw = h['FIBERWCS'].data
        apply_fibre_astrometry(h, ras=fw['RA'] + 1.0 / 3600, decs=fw['DEC'],
                               xs=fw['X_FIBERMAP'], ys=fw['Y_FIBERMAP'],
                               prov={'method': 'astrometric', 'tier': 'gaia', 'refined': True,
                                     'pa_offset': 0.0, 'catalog': 'GaiaDR3', 'rms': 0.1,
                                     'nstars': 3})
    flux_calibrate_file(base, _sensfunc({'fibrflat': False, 'rssstage': 'base'}),
                        apply_extinction=False)

    with fits.open(base) as b, fits.open(ff) as f, fits.open(sky) as s:
        nb, nf, ns = ({h.name for h in x} for x in (b, f, s))
        # primary: every key carried; only the stage keywords differ
        assert primary_header_diff_ok(b, f) and primary_header_diff_ok(f, s)
        assert [x[0].header['RSSSTAGE'] for x in (b, f, s)] == ['base', 'ff', 'ff_skysub']
        assert all(x[0].header[k] == v for x in (f, s) for k, v in derived.items())
        # data planes
        np.testing.assert_allclose(f['SKYSUB'].data, b['SKYSUB'].data / 2.0, rtol=1e-6)
        np.testing.assert_allclose(f['ERROR'].data, b['ERROR'].data / 2.0, rtol=1e-6)
        np.testing.assert_array_equal(s['ERROR'].data, f['ERROR'].data)       # copied, not grown
        assert not np.array_equal(s['SKYSUB'].data, f['SKYSUB'].data)         # regenerated
        for ext in ('COUNTS', 'SKY', 'WAVE', 'FWHM'):
            np.testing.assert_array_equal(f[ext].data, b[ext].data)
            np.testing.assert_array_equal(s[ext].data, f[ext].data)
        np.testing.assert_array_equal(s['FIBERMAP'].data['FIBER_ID'], b['FIBERMAP'].data['FIBER_ID'])
        # mask bits from base survive both stages
        for x in (f, s):
            assert np.all(x['MASK'].data[10:12] & 32)
            assert np.all(x['MASK'].data[20, :60] & MASK_NO_COUNTS)
        # extensions: NOFLAT dropped on purpose; FIBERWCS carried (rough); sky products added
        assert 'NOFLAT' in nb and 'NOFLAT' not in nf | ns
        assert 'FIBERWCS' in nf and 'FIBERWCS' in ns and not s['FIBERWCS'].header['WCSREFIN']
        assert {'SKYRESID', 'SKYMASK'} <= ns and not {'SKYRESID', 'SKYMASK'} & nf
        assert 'FLAM' in nb and 'FLAM' not in nf | ns

    # consolidation is a byte-identical rename of the most-corrected file
    cons_dir = os.path.join(tmp, 'cons')
    os.makedirs(cons_dir)
    for p in (base, ff, sky):
        shutil.copy(p, cons_dir)
    sky_bytes = open(sky, 'rb').read()
    consolidate_rss_files(cons_dir, keep_intermediate=False)
    (only,) = os.listdir(cons_dir)
    assert only == f'{EXP_ID}_RSS_green.fits' and rss_stage(os.path.join(cons_dir, only)) == 'ff_skysub'
    assert open(os.path.join(cons_dir, only), 'rb').read() == sky_bytes

    # the final stage now refuses to be combined; promote brings registration + FLAM across
    with pytest.raises(RegistrationMismatchError):
        check_stage_consistency(sky)
    sens = os.path.join(tmp, 'sens.fits')
    _sensfunc({'fibrflat': True, 'rssstage': 'ff_skysub'}).save(sens)
    os.remove(corr)
    results, n_problem = promote_directory(tmp, sensfunc_path=sens, apply_extinction=False)
    assert n_problem == 0, results
    with fits.open(base) as b, fits.open(sky) as s:
        np.testing.assert_array_equal(s['FIBERWCS'].data['RA'], b['FIBERWCS'].data['RA'])
        np.testing.assert_array_equal(s['FIBERMAP'].data['RA'], b['FIBERMAP'].data['RA'])
        assert s['FIBERWCS'].header['WCSREFIN'] and 'FLAM' in s
        assert {'SKYRESID', 'SKYMASK'} <= {h.name for h in s}
        assert np.all(s['MASK'].data[10:12] & 32)
        assert primary_header_diff_ok(b, s)
    check_stage_consistency(sky)


def primary_header_diff_ok(a, b):
    from llamas_pyjamas.File.llamasRSS import primary_header_diff
    return primary_header_diff(a[0].header, b[0].header) == {}


# ================================================================ step 4: flux calibration on the final stage
def test_fluxcal_cli_writes_flam_on_the_final_stage(tmp, capsys):
    from llamas_pyjamas.Postprocessing.apply_fluxcal import main
    files = write_exposure(tmp, channels=('green',))
    sens = os.path.join(tmp, 'std_sens.fits')
    _sensfunc({'fibrflat': True, 'rssstage': 'ff_skysub'}).save(sens)
    # every stage of the exposure named -> one target, the final file
    assert main([sens, files[('base', 'green')], files[('ff', 'green')], '--no-extinction']) == 0
    with fits.open(files[('ff_skysub', 'green')]) as f:
        assert 'FLAM' in f and f['FLAM'].header['SENSFILE'] == 'std_sens.fits'
        assert f['FLAM'].header['RSSSTAGE'] == 'ff_skysub' and f['FLAM'].header['SENSPROV'] == 'recorded'
    for s in ('base', 'ff'):
        with fits.open(files[(s, 'green')]) as h:
            assert 'FLAM' not in h
    assert 'Flux-calibrated 1/1' in capsys.readouterr().out


def test_fluxcal_cli_refuses_wrong_stage_sensfunc_and_header_mismatch(tmp, capsys):
    from llamas_pyjamas.Postprocessing.apply_fluxcal import main
    files = write_exposure(tmp, channels=('green',),
                           per_stage={'base': {'header': {'AIRMASS': 1.17}}})
    base_sens = os.path.join(tmp, 'base_sens.fits')
    _sensfunc({'fibrflat': False, 'rssstage': 'base'}).save(base_sens)
    assert main([base_sens, files[('base', 'green')], '--no-extinction',
                 '--allow-header-mismatch']) == 1                   # stage mismatch
    assert 'was built at stage' in capsys.readouterr().err
    ff_sens = os.path.join(tmp, 'ff_sens.fits')
    _sensfunc({'fibrflat': True}).save(ff_sens)
    assert main([ff_sens, files[('base', 'green')], '--no-extinction']) == 1   # AIRMASS differs
    err = capsys.readouterr().err
    assert 'AIRMASS' in err
    # an explicit stage is honoured
    assert main([base_sens, files[('ff_skysub', 'green')], '--stage', 'base', '--no-extinction',
                 '--allow-header-mismatch']) == 0
    with fits.open(files[('base', 'green')]) as h:
        assert 'FLAM' in h


def test_calibration_targets_deduplicate_exposures(tmp):
    from llamas_pyjamas.Flux.fluxCalibrate import calibration_targets
    files = write_exposure(tmp)
    got = calibration_targets([files[(s, c)] for s in ('base', 'ff', 'ff_skysub')
                               for c in ('green', 'red')])
    assert got == [files[('ff_skysub', 'green')], files[('ff_skysub', 'red')]]


# ================================================================ step 5: registration on the final stage
def test_registration_writes_the_final_stage(tmp):
    from llamas_pyjamas.Utils.register import _registration_siblings, reset_rough
    files = write_exposure(tmp, per_stage={'base': {'registered': True}})
    sib = _registration_siblings(files[('base', 'green')])
    assert sib == {c: files[('ff_skysub', c)] for c in ('blue', 'green', 'red')}
    base_bytes = open(files[('base', 'green')], 'rb').read()
    written = reset_rough(files[('base', 'green')])          # opened via the BASE file
    assert set(written) >= set(sib.values())
    assert open(files[('base', 'green')], 'rb').read() == base_bytes   # base untouched
    with fits.open(files[('ff_skysub', 'green')]) as h:
        assert h['FIBERWCS'].header['WCSMETH'] == 'rough-header'


def test_detect_common_sources_keys_stay_the_given_paths(tmp):
    from llamas_pyjamas.Utils.register import detect_common_sources
    files = write_exposure(tmp, channels=('green',))
    out = detect_common_sources([files[('base', 'green')], files[('ff', 'green')]])
    assert list(out) == [files[('base', 'green')]]           # one per exposure, keyed as given


def test_registration_refuses_header_mismatch(tmp):
    from llamas_pyjamas.Utils.register import _registration_siblings
    files = write_exposure(tmp, channels=('green',), per_stage={'base': {'header': {'TEL ROT': 90.0}}})
    with pytest.raises(StageHeaderMismatchError):
        _registration_siblings(files[('base', 'green')])


def test_apply_fibre_astrometry_updates_in_place_only_for_matching_rows(tmp):
    from llamas_pyjamas.File.llamasRSS import apply_fibre_astrometry
    prov = {'method': 'astrometric', 'tier': 'gaia', 'refined': True, 'pa_offset': 0.0,
            'catalog': 'GaiaDR3', 'rms': 0.1, 'nstars': 3}
    p = write_rss(os.path.join(tmp, 'x_RSS_green_FF_SKYSUB.fits'), stage='ff_skysub', fiberwcs=True)
    with fits.open(p) as h:
        idx = [x.name for x in h].index('FIBERWCS')
        fw = h['FIBERWCS'].data
        ras, decs, xs, ys = fw['RA'] + 1e-3, fw['DEC'].copy(), fw['X_FIBERMAP'], fw['Y_FIBERMAP']
    with fits.open(p, mode='update') as h:
        apply_fibre_astrometry(h, ras=ras, decs=decs, xs=xs, ys=ys, prov=prov)
        assert [x.name for x in h].index('FIBERWCS') == idx  # updated in place, not moved
    with fits.open(p) as h:
        np.testing.assert_allclose(h['FIBERWCS'].data['RA'], ras)
        assert h['FIBERWCS'].header['WCSREFIN']
    # same shape, permuted fibres: must be rebuilt from FIBERMAP, not overwritten row by row
    with fits.open(p, mode='update') as h:
        h['FIBERWCS'].data['FIBER_ID'][:] = h['FIBERWCS'].data['FIBER_ID'][::-1]
    with fits.open(p, mode='update') as h:
        apply_fibre_astrometry(h, ras=ras, decs=decs, xs=xs, ys=ys, prov=prov)
    with fits.open(p) as h:
        np.testing.assert_array_equal(h['FIBERWCS'].data['FIBER_ID'], h['FIBERMAP'].data['FIBER_ID'])
        assert [x.name for x in h][-1] == 'FIBERWCS'
