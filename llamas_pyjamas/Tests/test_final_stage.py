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
