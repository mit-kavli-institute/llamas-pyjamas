"""Tests for the fibre-flat fixes of 2026-11-14 (plan steps A0-A4).

A synthetic benchside 2B (dead fibre 49, as in LUT/traceLUT.json) is used throughout:
extractions are live-indexed, so above fibre 49 a row number and a physical fibre id
differ by one.
"""

import ast
import os
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.File.llamasRSS import RSSStageError
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
from llamas_pyjamas.Flat.fibreFlat import (apply_fibre_flat_to_rss, assess_fibre_flat,
                                           compute_fibre_flat_twilight, load_twilights,
                                           reduce_twilights, _load_smooth_models,
                                           _narrow_cards, _twilight_dip_cards)
from llamas_pyjamas.Flat import flatProcessing
from llamas_pyjamas.Flat.flatLlamas import (assess_pixel_flat, clip_fraction_cards,
                                            smooth_model_table, sort_and_write_pixel_maps,
                                            stamp_lamp_flat_products, _write_smooth_models_fits)
from llamas_pyjamas.Flux.fluxCalibrate import SensfuncStageError, flux_calibrate_file
from llamas_pyjamas.Flux.sensFunc import SensChannel, SensFunc
from llamas_pyjamas.Tests.rss_fixtures import N_WAVE as RSS_NWAVE, stage_name, write_rss
import llamas_pyjamas.reduce as reduce_mod
from llamas_pyjamas.reduce import (apply_twilight_throughput, stale_products,
                                   twilight_throughputs)
from llamas_pyjamas.Utils.deadfibers import live_fibre_ids, live_row_index
from llamas_pyjamas.Utils.recipes import (CalibrationRecipeError, check_recipe, recipe_hash,
                                          stamp_recipe)

N_WAVE = 400
N_LIVE = 297
DEAD = [49]
WAVE = np.linspace(4700.0, 7000.0, N_WAVE)


def throughput_of(fid):
    """A distinct, known throughput for every physical fibre."""
    return 1.0 + 0.3 * np.sin(1.7 * fid)


def lamp_extraction():
    """Live-indexed lamp extraction of green 2B, its smooth models keyed by row."""
    physical = live_fibre_ids(N_LIVE, DEAD)
    counts = np.ones((N_LIVE, N_WAVE))
    ext = SimpleNamespace(counts=counts, wave=np.tile(WAVE, (N_LIVE, 1)), dead_fibers=DEAD)
    smooth = {row: np.full(N_WAVE, 1000.0) for row in range(N_LIVE)}
    return ext, smooth, physical


def twilight_extraction():
    """Live-indexed twilight of green 2B whose row k holds fibre live_fibre_ids()[k]."""
    physical = live_fibre_ids(N_LIVE, DEAD)
    counts = np.array([np.full(N_WAVE, 5000.0 * throughput_of(p)) for p in physical])
    ext = SimpleNamespace(counts=counts, wave=np.tile(WAVE, (N_LIVE, 1)), dead_fibers=DEAD)
    return {'extractions': [ext], 'metadata': [{'channel': 'green', 'bench': '2', 'side': 'B'}],
            'primary_header': fits.Header()}


def write_models(tmp_path):
    ext, smooth, _ = lamp_extraction()
    ids, sm, wv = smooth_model_table(ext, smooth)
    path = str(tmp_path / 'flat_smooth_models.fits')
    _write_smooth_models_fits({'green_2_B': {'fiber_ids': ids, 'smooth': sm, 'wave': wv,
                                            'channel': 'green', 'bench': '2', 'side': 'B'}},
                              path)
    with fits.open(path, mode='update') as h:
        stamp_recipe(h[0].header, 'lamp_flat')
    return path


# --- A0: physical fibre ids ---------------------------------------------------------

def test_live_row_index_inverts_live_fibre_ids():
    rows = live_row_index(N_LIVE, DEAD)
    assert 49 not in rows and rows[48] == 48 and rows[50] == 49 and rows[297] == 296
    assert [live_fibre_ids(N_LIVE, DEAD)[r] for r in rows.values()] == list(rows)


def test_smooth_model_ids_are_physical():
    ext, smooth, physical = lamp_extraction()
    ids, sm, wv = smooth_model_table(ext, smooth)
    assert list(ids) == physical            # 0..48, 50..297
    assert 49 not in ids and ids[-1] == 297
    assert sm.shape == wv.shape == (N_LIVE, N_WAVE)


def test_rows_without_a_model_stay_aligned():
    ext, smooth, physical = lamp_extraction()
    del smooth[60]                          # row 60 is physical fibre 61
    ids, sm, _ = smooth_model_table(ext, smooth)
    assert 61 not in ids and len(ids) == len(sm) == N_LIVE - 1


def test_each_fibre_gets_its_own_twilight_throughput(tmp_path):
    corr = compute_fibre_flat_twilight(twilight_extraction(), write_models(tmp_path),
                                       str(tmp_path))
    with fits.open(corr) as h:
        assert h[0].header['FIBIDS'] == 'physical'
        ids = np.asarray(h['green_2_B'].data['FIBER_ID'])
        ti = np.nanmedian(h['green_2_B'].data['CORRECTION'], axis=1)
    truth = throughput_of(ids)
    assert np.corrcoef(ti, truth)[0, 1] > 0.98
    above = ids > 49
    # the neighbour's throughput (the old off-by-one) is a much worse match
    assert np.corrcoef(ti[above][:-1], throughput_of(ids[above][1:]))[0, 1] < 0.5


def test_row_labelled_smooth_models_are_refused(tmp_path):
    path = write_models(tmp_path)
    with fits.open(path, mode='update') as h:
        del h[0].header['FIBIDS']
    with pytest.raises(CalibrationRecipeError, match='extraction row'):
        _load_smooth_models(path)


# --- A1: throughput applied once ------------------------------------------------------

def _rss_divided_by(tmp_path, true_tp, rss_tp):
    """Base RSS whose raw sky is true_tp x sky, already divided by rss_tp (as the RSS
    writer divides by relative_throughput), with rss_tp recorded in THRUPUT."""
    path = write_rss(str(tmp_path / 'x_RSS_green.fits'), thruput=rss_tp)
    sky = 300.0 + 80.0 * np.sin(np.linspace(0, 6, RSS_NWAVE))
    with fits.open(path, mode='update') as h:
        raw = true_tp[:, None] * sky
        h['COUNTS'].data[:] = raw / rss_tp[:, None]
        h['SKY'].data[:] = 0.0
        h['SKYSUB'].data[:] = raw / rss_tp[:, None]
        h['ERROR'].data[:] = 0.5
    return path, sky


def _corrections(tmp_path, true_tp):
    hdus = [fits.PrimaryHDU()]
    hdus[0].header['METHOD'] = 'twilight'
    stamp_recipe(hdus[0].header, 'twilight')
    for k, bs in enumerate(('1A', '1B')):
        cols = [fits.Column(name='FIBER_ID', format='J', array=np.arange(30)),
                fits.Column(name='CORRECTION', format=f'{RSS_NWAVE}D',
                            array=np.tile(true_tp[30 * k:30 * (k + 1), None], RSS_NWAVE)),
                fits.Column(name='WAVE', format=f'{RSS_NWAVE}D',
                            array=np.tile(np.linspace(4700.0, 7000.0, RSS_NWAVE), (30, 1))),
                fits.Column(name='FLAGS', format='I', array=np.zeros(30, dtype=np.int16))]
        h = fits.BinTableHDU.from_columns(cols)
        h.header['EXTNAME'] = f'green_{bs[0]}_{bs[1]}'
        hdus.append(h)
    path = str(tmp_path / 'fibre_flat_corrections.fits')
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_flat_divides_out_the_throughput_once(tmp_path):
    rng = np.random.default_rng(1)
    true_tp = rng.uniform(0.75, 1.25, 60)
    rss_tp = true_tp * rng.uniform(0.98, 1.02, 60)       # the RSS's own, slightly different
    base, sky = _rss_divided_by(tmp_path, true_tp, rss_tp)
    out = apply_fibre_flat_to_rss(base, _corrections(tmp_path, true_tp),
                                  str(tmp_path / 'x_RSS_green_FF.fits'))
    with fits.open(out) as h:
        flat = h['SKYSUB'].data
        assert h[0].header['FFVER'] == 2
        assert h[0].header['FFSA1A'] < 0.5            # flat sky after the flat
        assert h[0].header['FFST1A'] < 2.0            # RSS throughput alone: close
        assert h[0].header['FFSB1A'] > 8.0            # raw: the full throughput spread
    # every fibre is back to the sky itself: the throughput was divided out exactly once
    np.testing.assert_allclose(flat, np.tile(sky, (60, 1)), rtol=1e-5)


def test_rss_without_thruput_is_refused(tmp_path):
    true_tp = np.ones(60)
    base, _ = _rss_divided_by(tmp_path, true_tp, true_tp)
    with fits.open(base) as h:
        cols = [c for c in h['FIBERMAP'].columns if c.name != 'THRUPUT']
        h['FIBERMAP'] = fits.BinTableHDU.from_columns(cols, name='FIBERMAP')
        h.writeto(base, overwrite=True)
    with pytest.raises(RSSStageError, match='THRUPUT'):
        apply_fibre_flat_to_rss(base, _corrections(tmp_path, true_tp),
                                str(tmp_path / 'x_RSS_green_FF.fits'))


@pytest.mark.parametrize('sens_ffver, ok', [(2, True), (1, False)])
def test_flux_calibration_needs_the_same_ffver(tmp_path, sens_ffver, ok):
    path = write_rss(str(tmp_path / 'x_RSS_green_FF.fits'), stage='ff', header={'FFVER': 2})
    w = np.linspace(4600, 7100, 200)
    sens = SensFunc(channels={'green': SensChannel('green', w, np.full(200, 1e-16),
                                                   np.full(200, 1e-16), np.ones(200, bool))},
                    meta={'fibrflat': True, 'rssstage': 'ff', 'ffver': sens_ffver})
    if ok:
        flux_calibrate_file(path, sens, apply_extinction=False)
    else:
        with pytest.raises(SensfuncStageError, match='FFVER'):
            flux_calibrate_file(path, sens, apply_extinction=False)


# --- A2: no cosmic-ray cleaning on lamp flats ------------------------------------------

def test_lamp_flat_extraction_never_cleans_cosmic_rays():
    """Every process_trace call in Flat/flatProcessing.py passes remove_cosmic_rays=False."""
    tree = ast.parse(open(flatProcessing.__file__).read())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'remote'
             and getattr(n.func.value, 'id', None) == 'process_trace']
    assert calls
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert 'remove_cosmic_rays' in kw and kw['remove_cosmic_rays'].value is False


def test_flat_products_record_the_lamp_flat_recipe(tmp_path):
    path = write_models(tmp_path)
    stamp_lamp_flat_products([path], {'red': '/x/red.fits', 'green': '/x/g.fits',
                                      'blue': None})
    header = fits.getheader(path, 0)
    assert check_recipe(header, 'lamp_flat')
    assert header['CRCLEAN'] is False and header['FLATFILG'] == 'g.fits'
    assert header['FLATFILB'] == 'none'


def test_corrections_record_their_twilight_and_smooth_models(tmp_path):
    models = write_models(tmp_path)
    stamp_lamp_flat_products([models], {'green': 'flat.fits'})
    corr = compute_fibre_flat_twilight(twilight_extraction(), models, str(tmp_path),
                                       header_extra={'TWIFILG': 'twi.fits'})
    header = fits.getheader(corr, 0)
    assert check_recipe(header, 'twilight')
    assert header['SMRCPHS'] == recipe_hash('lamp_flat') and header['TWIFILG'] == 'twi.fits'


# --- A3: QA ----------------------------------------------------------------------------

def lamp_shapes(n=40, dips_at=(), depth=0.10, width=8, seed=3):
    """Smooth W_i rows on a common grid, with narrow dips at the given pixel positions."""
    rng = np.random.default_rng(seed)
    wave = np.tile(WAVE, (n, 1))
    w = np.tile(1.0 + 0.2 * np.sin(np.linspace(0, 3, N_WAVE)), (n, 1))
    w *= 1 + rng.normal(0, 0.001, w.shape)
    for f, x in dips_at:
        w[f, x:x + width] *= 1 - depth
    return w, wave


def test_narrow_cards_count_dips_and_stay_small_on_a_clean_shape():
    clean = _narrow_cards(*lamp_shapes())
    assert clean['WINDIP'][0] == 0 and clean['WINRMS'][0] < 0.5
    dipped = _narrow_cards(*lamp_shapes(dips_at=[(3, 150), (7, 200), (7, 260)]))
    assert dipped['WINDIP'][0] == 3


def twilight_for(w, wave, fibre_dips=()):
    """Twilight of the same fibres: a solar-like spectrum common to all, plus narrow
    fibre-specific dips at ``fibre_dips`` (real features of those fibres)."""
    solar = 1.0 - 0.3 * np.exp(-0.5 * ((WAVE - 5890.0) / 3.0) ** 2)
    counts = np.tile(20000.0 * solar, (w.shape[0], 1))
    for f, x in fibre_dips:
        counts[f, x:x + 8] *= 0.9
    return SimpleNamespace(counts=counts, wave=wave.copy(), dead_fibers=[])


def test_twilight_check_flags_lamp_only_dips_and_passes_real_ones():
    dips = [(5, 150), (12, 220), (20, 300)]
    w, wave = lamp_shapes(dips_at=dips)
    ids = np.arange(w.shape[0])
    fake = _twilight_dip_cards(w, wave, ids, twilight_for(w, wave))
    assert fake['WIDIPN'][0] == 3 and fake['WIDIPTW'][0] == 1.0
    real = _twilight_dip_cards(w, wave, ids, twilight_for(w, wave, fibre_dips=dips))
    assert real['WIDIPN'][0] == 3 and real['WIDIPTW'][0] == 0.0


def assess_inputs(tmp_path, ext_cards, after=1.0, tp_only=2.0):
    """A corrections file and one FF header, green, all eight benchsides."""
    hdus = [fits.PrimaryHDU()]
    hdus[0].header['METHOD'] = 'twilight'
    for bs in ('1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B'):
        h = fits.BinTableHDU.from_columns([fits.Column(name='FIBER_ID', format='J',
                                                       array=np.arange(3))])
        h.header.update({'EXTNAME': f'green_{bs[0]}_{bs[1]}', 'CHANNEL': 'green',
                         'BENCH': bs[0], 'SIDE': bs[1], 'TIMAD': 10.0,
                         'WINRMS': 0.4, 'WINDIP': 0, 'WIDIPN': 10, 'WIDIPTW': 0.0})
        h.header.update(ext_cards.get(bs, {}))
        hdus.append(h)
    corr = str(tmp_path / 'fibre_flat_corrections.fits')
    fits.HDUList(hdus).writeto(corr, overwrite=True)
    p = fits.PrimaryHDU()
    p.header['CHANNEL'] = 'green'
    for bs in ('1A', '1B', '2A', '2B', '3A', '3B', '4A', '4B'):
        p.header.update({f'FFSB{bs}': 12.0, f'FFST{bs}': tp_only, f'FFSA{bs}': after,
                         f'FFSN{bs}': 0.2})
    ff = str(tmp_path / 'x_RSS_green_FF.fits')
    p.writeto(ff, overwrite=True)
    return corr, [ff]


def test_assess_quiet_on_a_good_flat(tmp_path):
    lines, warnings, report = assess_fibre_flat(*assess_inputs(tmp_path, {}))
    assert warnings == []
    assert any(line.startswith('Lamp shape W_i: narrow rms 0.40–0.40%') for line in lines)
    assert any('0/80 W_i dips' in line for line in lines)


def test_assess_warns_on_narrow_structure_and_unconfirmed_dips(tmp_path):
    cards = {'3B': {'WINRMS': 3.5, 'WINDIP': 40}, '4A': {'WIDIPTW': 0.9}}
    _, warnings, _ = assess_fibre_flat(*assess_inputs(tmp_path, cards))
    text = '\n'.join(warnings)
    assert 'Narrow structure in the lamp shape W_i above 1% (provisional) on 1/8' in text
    assert 'green 3B 3.50%' in text
    assert 'twilight does not show' in text and 'green 4A 90% of 10' in text


def test_assess_warns_when_the_flat_is_worse_than_the_throughput(tmp_path):
    _, warnings, _ = assess_fibre_flat(*assess_inputs(tmp_path, {}, after=12.0, tp_only=2.0))
    assert 'more sky scatter than the RSS throughput alone on 8/8' in warnings[0]


def test_clip_fraction_cards():
    m = np.ones((10, 10))
    m[:2] = 0.9            # 20 pixels at the floor
    m[2:4] = 1.05          # 20 corrected, not clipped
    cards = clip_fraction_cards(m, (0.9, 1.1))
    assert cards['NCORPIX'][0] == 40 and cards['FCLIPLO'][0] == 50.0
    assert cards['FCLIPHI'][0] == 0.0


def test_assess_pixel_flat_flags_a_colour_with_lowered_cores(tmp_path):
    maps, cards = {}, {}
    for ch, frac in (('red', 0.0001), ('green', 0.02), ('blue', 0.0002)):
        for b in '1234':
            for side in 'AB':
                m = np.full((100, 100), 1.02)
                m.flat[:int(frac * m.size)] = 0.9
                name = f'{ch}_{b}_{side}'
                maps[name] = m
                cards[name] = clip_fraction_cards(m, (0.9, 1.1))
    path = str(tmp_path / 'pixel_maps.fits')
    sort_and_write_pixel_maps(maps, path, ext_cards=cards)
    lines, warnings, report = assess_pixel_flat(path)
    assert lines[0].startswith('Pixel flat: corrected pixels at the clip floor')
    assert len(warnings) == 1 and 'Pixel flat green: median 2.00%' in warnings[0]
    assert 'clip floor red 0.01%, green 2.00%, blue 0.02%' == report


# --- A4: stale products are refused on resume -------------------------------------------

def test_stale_products_are_listed_and_current_ones_pass(tmp_path):
    ext = tmp_path / 'extractions'
    flat = ext / 'flat'
    flat.mkdir(parents=True)
    old_map = str(flat / 'pixel_maps.fits')
    fits.PrimaryHDU().writeto(old_map)                       # before recipe stamps
    models = write_models(flat)                              # stamped below
    stamp_lamp_flat_products([models], {'green': 'flat.fits'})
    write_rss(stage_name(str(ext), 'green', 'base'))         # current: has THRUPUT
    old_base = write_rss(stage_name(str(ext), 'red', 'base'))
    with fits.open(old_base) as h:                           # an RSS from before THRUPUT
        cols = [c for c in h['FIBERMAP'].columns if c.name != 'THRUPUT']
        h['FIBERMAP'] = fits.BinTableHDU.from_columns(cols, name='FIBERMAP')
        h.writeto(old_base, overwrite=True)
    write_rss(stage_name(str(ext), 'green', 'ff'), stage='ff', header={'FFVER': 2})
    old_ff = write_rss(stage_name(str(ext), 'blue', 'ff'), stage='ff')   # no FFVER

    stale = dict(stale_products(str(ext), str(flat)))
    assert set(stale) == {old_map, old_base, old_ff}
    assert 'no recipe stamp' in stale[old_map]
    assert 'THRUPUT' in stale[old_base] and 'divided out twice' in stale[old_ff]


def test_nothing_is_stale_in_an_empty_folder(tmp_path):
    assert stale_products(str(tmp_path), str(tmp_path / 'flat')) == []


# --- B1: one twilight reduction for the throughput and T_i ------------------------------

def cameras(level_by_channel, n=40):
    """Extraction objects for 1A of each channel, counts = level x (1 + fibre pattern)."""
    exts = []
    for ch, level in level_by_channel.items():
        counts = level * np.tile((1 + 0.1 * np.sin(np.arange(n)))[:, None], (1, 2048))
        exts.append(SimpleNamespace(counts=counts, wave=np.tile(np.linspace(4700, 7000, 2048), (n, 1)),
                                    channel=ch, bench='1', side='A', dead_fibers=[],
                                    trace=SimpleNamespace(nfibers=n)))
    return exts


@pytest.fixture
def fake_pipeline(monkeypatch, tmp_path):
    """Stand-ins for the pixel flat, extraction and wavelength steps; counts the calls."""
    calls = []
    levels = {'twi_a.fits': {'red': 100.0, 'green': 100.0, 'blue': 100.0},
              'twi_b.fits': {'red': 900.0, 'green': 900.0, 'blue': 900.0}}

    def flat(path, maps, out_dir):
        calls.append(os.path.basename(path))
        out = os.path.join(out_dir, os.path.basename(path) + '_flat.fits')
        open(out, 'w').close()
        return out, None

    def extract(path, out_dir, **kw):
        assert kw['remove_cosmic_rays'] is False
        out = os.path.join(out_dir, os.path.basename(path) + '_extract.pkl')
        open(out, 'w').close()
        return out

    def wavelengths(path, soln=None):
        name = os.path.basename(path).split('_flat')[0]
        exts = cameras(levels[name])
        return ({'extractions': exts, 'metadata': [{'channel': e.channel, 'bench': '1',
                                                    'side': 'A'} for e in exts]}, fits.Header())

    monkeypatch.setattr(reduce_mod, 'apply_flat_field_correction', flat)
    monkeypatch.setattr(reduce_mod, 'run_extraction', extract)
    monkeypatch.setattr(reduce_mod, 'correct_wavelengths', wavelengths)
    pixmap = str(tmp_path / 'pixel_maps.fits')
    fits.PrimaryHDU().writeto(pixmap)
    for name in levels:
        open(tmp_path / name, 'w').close()
    return calls, pixmap


def run_twilights(tmp_path, pixmap, by_colour):
    return reduce_twilights({c: str(tmp_path / f) if f else None for c, f in by_colour.items()},
                            pixmap, 'traces', None, None, str(tmp_path / 'flat'))


def test_each_twilight_is_reduced_once_and_reused(tmp_path, fake_pipeline):
    calls, pixmap = fake_pipeline
    pkls = run_twilights(tmp_path, pixmap, {'red': 'twi_a.fits', 'green': 'twi_b.fits',
                                            'blue': 'twi_b.fits'})
    assert sorted(calls) == ['twi_a.fits', 'twi_b.fits']        # twi_b once for two colours
    assert pkls['green'] == pkls['blue'] != pkls['red']
    assert not any(f.endswith('_extract.pkl') or f.endswith('_flat.fits')
                   for f in os.listdir(tmp_path / 'flat' / 'twilight'))   # intermediates gone
    header = ExtractLlamas.loadExtraction(pkls['red'])['primary_header']
    assert check_recipe(header, 'twilight') and header['TWIFILE'] == 'twi_a.fits'

    run_twilights(tmp_path, pixmap, {'red': 'twi_a.fits', 'green': 'twi_b.fits'})
    assert len(calls) == 2                                        # cached: nothing re-reduced
    os.utime(pixmap, (1, 1))                                      # a different pixel flat
    run_twilights(tmp_path, pixmap, {'red': 'twi_a.fits'})
    assert calls[-1] == 'twi_a.fits' and len(calls) == 3


def test_throughput_and_t_i_come_from_each_colours_twilight(tmp_path, fake_pipeline):
    _, pixmap = fake_pipeline
    pkls = run_twilights(tmp_path, pixmap, {'red': 'twi_a.fits', 'green': 'twi_b.fits'})
    tp = twilight_throughputs(pkls)
    assert set(tp) == {('red', '1', 'A'), ('green', '1', 'A')}
    assert tp[('red', '1', 'A')][1] == 'twi_a.fits' and tp[('green', '1', 'A')][1] == 'twi_b.fits'
    merged = load_twilights(pkls)
    by_channel = {m['channel']: e for e, m in zip(merged['extractions'], merged['metadata'])}
    assert set(by_channel) == {'red', 'green'}
    assert np.median(by_channel['red'].counts) < 200 < np.median(by_channel['green'].counts)

    science = {'extractions': cameras({'red': 1.0, 'green': 1.0, 'blue': 1.0}),
               'metadata': [{'channel': c, 'bench': '1', 'side': 'A'}
                            for c in ('red', 'green', 'blue')]}
    assert apply_twilight_throughput(science, tp) == 2            # blue has no twilight
    red = science['extractions'][0]
    assert red.throughput_source == 'twi_a.fits'
    np.testing.assert_allclose(red.relative_throughput, tp[('red', '1', 'A')][0])
