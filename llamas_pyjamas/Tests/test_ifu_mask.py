"""Tests for half-IFU mask handling (llamas_pyjamas.Utils.ifuMask and the stages that use it).

Self-contained: synthetic extraction pickles and RSS files only (no raw data, no Ray). A synthetic
camera has the fibre-map high-x half "behind the mask" -- its fibres carry ~3% of the open-half
signal, mimicking the profile wings of their lit detector neighbours.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.Tests.test_ifu_mask`).
"""

import glob
import os
from types import SimpleNamespace

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import numpy as np
import pytest
from astropy.io import fits

from llamas_pyjamas.Utils import ifuMask as im
from llamas_pyjamas.Sky import skySelect


NPIX = 400
WAVE = np.linspace(5400.0, 5800.0, NPIX)          # covers [OI] 5577


def _n_fibres(bs):
    return sum(1 for (b, _f) in im._xpos_table() if b == bs)


X_OI = (5577.34 - WAVE[0]) / (WAVE[1] - WAVE[0])  # [OI] 5577 in pixel (xshift) units


def _sky_fn(x, cont=100.0, line_amp=400.0):
    """Sky in xshift (pixel) space: continuum + [OI] 5577 + two more lines."""
    s = cont + line_amp * np.exp(-0.5 * ((x - X_OI) / 1.5) ** 2)
    for c, a in ((60, 150.0), (300, 250.0)):
        s = s + a * np.exp(-0.5 * ((x - c) / 1.3) ** 2)
    return s


def _camera(bs, hi=(0.03, 0.0), lo=(1.0, 0.0), channel='green', noise=1.0, seed=0,
            cont=100.0, line_amp=400.0):
    """ExtractLlamas-like camera and its true sky. High-x / low-x fibres carry
    ``factor * sky + offset`` for ``hi`` / ``lo`` = (factor, offset). Each fibre gets its own
    sub-pixel xshift offset, as real fibres do (identical grids make the b-spline singular)."""
    rng = np.random.default_rng(seed)
    n = _n_fibres(bs)
    xshift = np.arange(NPIX, dtype=float)[None, :] + ((np.arange(n) * 0.618034) % 1.0)[:, None]
    truth = _sky_fn(xshift, cont, line_amp)
    is_hi = (im.fibre_halves(bs, np.arange(n)) == im.HIGHX)[:, None]
    counts = (np.where(is_hi, hi[0], lo[0]) * truth + np.where(is_hi, hi[1], lo[1])
              + rng.normal(0.0, noise, size=(n, NPIX)))
    cam = SimpleNamespace(
        counts=counts, wave=np.tile(WAVE, (n, 1)), xshift=xshift,
        sky=np.zeros((n, NPIX)), relative_throughput=np.ones(n), dead_fibers=[],
        channel=channel, bench=bs[0], side=bs[1],
        trace=SimpleNamespace(nfibers=n, traces=30.0 + 6.5 * np.arange(n)[:, None]
                              + np.zeros((1, NPIX))))
    return cam, truth


def _masked_cameras(benchsides=('1A', '3A', '1B'), dark_frac=0.03, **kw):
    return [_camera(bs, hi=(dark_frac, 0.0), **kw)[0] for bs in benchsides]


def _stamp(dec_blocked):
    hdr = fits.Header()
    dec = im.IFUMaskDecision(state='TOP', request=im.REQ_HEADER, masked=True,
                             blocked=dec_blocked, source='TABLE', verdict='MASKED-HIGHX')
    return im.stamp_header(hdr, dec)


# ---------------------------------------------------------------------------
# Header state, requests, decisions
# ---------------------------------------------------------------------------
def test_read_mask_state_variants():
    h = fits.Header()
    assert im.read_mask_state(h) == ''                                   # missing
    h['HIERARCH FRONTEND IFU MASK-STATE'] = 'TOP     '
    assert im.read_mask_state(h) == 'TOP'                                # padded
    h['HIERARCH FRONTEND IFU MASK-STATE'] = 'bottom'
    assert im.read_mask_state(h) == 'BOTTOM'                             # case
    h['HIERARCH FRONTEND IFU MASK-STATE'] = ''
    assert im.read_mask_state(h) == ''                                   # blank
    u = fits.Header()
    u.append(fits.Card('HIERARCH FRONTEND IFU MASK-STATE'))              # undefined (2025 data)
    assert im.read_mask_state(u) == ''
    assert im.read_mask_state({'FRONTEND IFU MASK-STATE': 'EMPTY'}) == 'EMPTY'
    assert im.read_mask_state(None) == ''


def test_request_table():
    rf = im.request_for
    assert rf('TOP') == im.REQ_HEADER
    assert rf('BOTTOM', ifu_mask=False) == im.REQ_HEADER                 # header beats config
    assert rf('EMPTY', ifu_mask=True) == im.REQ_NONE
    assert rf('', ifu_mask=True) == im.REQ_CONFIG
    assert rf('', ifu_mask=False) == im.REQ_NONE
    assert rf('HALFFIELD', ifu_mask=True) == im.REQ_CONFIG               # unrecognised = blank
    assert rf('TOP', force_off=True) == im.REQ_FORCEOFF
    # HOME: fatal for sky-light roles (even with force_off), normal for lamp calibrations
    for role in im.SKY_LIGHT_ROLES:
        assert rf('HOME', role=role) == im.REQ_CLOSED
        assert rf('HOME', force_off=True, role=role) == im.REQ_CLOSED
    for role in im.LAMP_ROLES:
        assert rf('HOME', role=role) == im.REQ_NONE


def test_config_switches():
    assert im.config_switches({}) == (False, False, None)
    assert im.config_switches({'IFU_mask': True, 'IFU_mask_force_off': 'true',
                               'IFU_mask_half': 'Top'}) == (True, True, 'TOP')
    assert im.config_switches({'IFU_mask': 'false', 'IFU_mask_half': 'left'}) == (False, False, None)


def test_finalise_rules():
    masked_lo = im.HalfMeasurement(verdict='MASKED-LOWX', dark=im.LOWX, cont_ratio=0.03)
    masked_hi = im.HalfMeasurement(verdict='MASKED-HIGHX', dark=im.HIGHX, cont_ratio=0.03)
    unmasked = im.HalfMeasurement(verdict='UNMASKED', cont_ratio=0.9)
    inconcl = im.HalfMeasurement(verdict='INCONCLUSIVE')

    d = im.finalise(im.REQ_HEADER, 'TOP', inconcl)
    assert (d.masked, d.blocked, d.source) == (True, im.HIGHX, 'TABLE')
    d = im.finalise(im.REQ_HEADER, 'TOP', masked_lo)                     # data disagree -> data
    assert (d.masked, d.blocked, d.source) == (True, im.LOWX, 'DATA') and d.notes
    d = im.finalise(im.REQ_HEADER, 'BOTTOM', unmasked)                   # header authoritative
    assert (d.masked, d.blocked) == (True, im.LOWX) and d.notes
    d = im.finalise(im.REQ_CONFIG, '', masked_hi)
    assert (d.masked, d.blocked, d.source) == (True, im.HIGHX, 'DATA')
    d = im.finalise(im.REQ_CONFIG, '', unmasked)                         # mixed nights
    assert not d.masked
    d = im.finalise(im.REQ_CONFIG, '', inconcl, half_override='BOTTOM')
    assert (d.masked, d.blocked, d.source) == (True, im.LOWX, 'OVERRIDE')
    d = im.finalise(im.REQ_CONFIG, '', inconcl)
    assert not d.masked and d.notes
    d = im.finalise(im.REQ_NONE, 'EMPTY', masked_hi)                     # warn, never act
    assert not d.masked and d.notes
    d = im.finalise(im.REQ_FORCEOFF, 'TOP', masked_hi)
    assert not d.masked
    with pytest.raises(ValueError):
        im.finalise(im.REQ_CLOSED, 'HOME', inconcl)


def test_stamp_round_trip():
    hdr = fits.Header()
    assert im.decision_from_header(hdr) is None                          # unstamped -> no action
    dec = im.finalise(im.REQ_HEADER, 'TOP', im.HalfMeasurement(verdict='MASKED-HIGHX',
                                                               dark=im.HIGHX, cont_ratio=0.04))
    im.stamp_header(hdr, dec)
    back = im.decision_from_header(hdr)
    assert back.masked and back.blocked == im.HIGHX and back.request == im.REQ_HEADER
    assert 'IFUMRAT' in hdr and 'IFUMLRAT' not in hdr                    # NaN ratio omitted
    assert set(im.stamp_items(hdr)) >= {'IFUMASK', 'IFUMBLK', 'IFUMREQ'}
    im.stamp_header(hdr, im.finalise(im.REQ_NONE, 'EMPTY'))
    assert not im.decision_from_header(hdr).masked


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
def test_halves_match_parity_rule_for_every_fibre():
    lut = im._xpos_table()
    bs = np.array([b for b, _ in lut])
    fid = np.array([f for _, f in lut])
    halves = im.fibre_halves(bs, fid)
    high_by_parity = (fid + np.char.endswith(bs, 'B').astype(int)) % 2 == 0
    assert len(fid) == 2392
    assert np.array_equal(halves == im.HIGHX, high_by_parity)


def test_live_split_and_pools():
    from llamas_pyjamas.Utils.deadfibers import live_fibre_ids
    dead = {'2A': [270, 299], '2B': [49]}
    lo = hi = 0
    for bs in ('1A', '2A', '3A', '4A', '1B', '2B', '3B', '4B'):
        n = _n_fibres(bs) - len(dead.get(bs, []))
        h = im.fibre_halves(bs, live_fibre_ids(n, dead.get(bs, [])))
        lo += int((h == im.LOWX).sum())
        hi += int((h == im.HIGHX).sum())
    assert (lo, hi) == (1195, 1194)
    n = _n_fibres('1A')
    pool = im.sky_pool_mask('1A', np.arange(n), im.HIGHX)
    x = im.fibre_xpos('1A', np.arange(n))
    assert np.all(x[pool] <= 22.0)                                       # open half, guard band out
    assert not np.any(pool & im.blocked_mask('1A', np.arange(n), im.HIGHX))


# ---------------------------------------------------------------------------
# skySelect allowed= invariant
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('method', ['stratified', 'quantile', 'dimmest', 'middle-third',
                                    'frame', 'all', 'skymap'])
def test_selection_stays_in_allowed(method):
    rng = np.random.default_rng(1)
    n = 300
    allowed = np.arange(n) % 2 == 0
    brightness = np.where(allowed, rng.uniform(50, 100, n), rng.uniform(1, 3, n))  # blocked faintest
    finite = np.ones(n, dtype=bool)
    kw = dict(method=method, n_fibres=20, allowed=allowed)
    if method == 'stratified':
        kw['fiber_y'] = np.arange(n, dtype=float)
    if method == 'skymap':
        kw['in_sky_region'] = np.ones(n, dtype=bool)
    mask = skySelect.select_sky_fibres(brightness, finite, **kw)
    assert mask.any() and not np.any(mask & ~allowed)


def test_selection_fallback_stays_in_allowed():
    n = 40
    allowed = np.zeros(n, dtype=bool)
    allowed[:4] = True
    finite = np.ones(n, dtype=bool)
    mask = skySelect.select_sky_fibres(np.arange(n, dtype=float), finite, method='quantile',
                                       allowed=allowed)
    assert mask.sum() >= 1 and not np.any(mask & ~allowed)
    manual = skySelect.build_sky_mask(method='manual', explicit=np.ones(n, dtype=bool),
                                      allowed=allowed).mask
    assert np.array_equal(manual, allowed)


def test_allowed_none_is_unchanged():
    rng = np.random.default_rng(2)
    b = rng.uniform(1, 100, 120)
    f = np.isfinite(b)
    y = np.arange(120, dtype=float)
    for m in ('stratified', 'quantile', 'dimmest', 'middle-third'):
        a = skySelect.select_sky_fibres(b, f, method=m, fiber_y=y)
        c = skySelect.select_sky_fibres(b, f, method=m, fiber_y=y, allowed=None)
        assert np.array_equal(a, c)


# ---------------------------------------------------------------------------
# Data check
# ---------------------------------------------------------------------------
def test_measure_masked_twilight_like():
    cams = _masked_cameras(dark_frac=0.03)
    for c in cams:                       # no sky lines: pure continuum (twilight)
        hi = im.fibre_halves(f"{c.bench}{c.side}", np.arange(c.counts.shape[0])) == im.HIGHX
        c.counts = np.where(hi[:, None], 3.0, 100.0) + 0 * c.counts
    m = im.measure_halves(cams)
    assert m.verdict == 'MASKED-HIGHX' and m.dark == im.HIGHX
    assert m.cont_ratio < 0.1


def test_measure_masked_line_free_noisy_frame():
    # 1 s twilight-like: weak continuum, no emission lines, read-noise dominated. Noise must not
    # pass for "equal sky lines in both halves" (regression: real TOP frame 2026-01-07 09:16:45).
    cams = [_camera(bs, hi=(0.03, 0.0), lo=(1.0, 0.0), cont=10.0, line_amp=0.0, noise=3.0, seed=i)[0]
            for i, bs in enumerate(('1A', '3A', '1B'))]
    for c in cams:                       # remove the two pixel-space lines as well
        hi = im.fibre_halves(f"{c.bench}{c.side}", np.arange(c.counts.shape[0])) == im.HIGHX
        rng = np.random.default_rng(3)
        c.counts = (np.where(hi[:, None], 0.3, 10.0)
                    + rng.normal(0.0, 3.0, size=c.counts.shape))
    m = im.measure_halves(cams)
    assert m.verdict == 'MASKED-HIGHX', (m.verdict, m.cont_ratio, m.line_ratio)


def test_measure_masked_with_sky_lines():
    m = im.measure_halves(_masked_cameras(dark_frac=0.03))
    assert m.verdict == 'MASKED-HIGHX'
    assert np.isfinite(m.line_ratio) and m.line_ratio < im.LINE_MASKED


def test_measure_bright_half_object_is_not_a_mask():
    # bright extended object filling the high-x half, no mask
    cams = [_camera(bs, hi=(1.0, 2000.0), lo=(1.0, 0.0), cont=20.0)[0]
            for bs in ('1A', '3A', '1B')]
    m = im.measure_halves(cams)
    assert m.verdict == 'UNMASKED'            # sky lines are equal in both halves


def test_measure_noise_is_inconclusive():
    cams = [_camera(bs, hi=(0.0, 0.0), lo=(0.0, 0.0), noise=5.0, seed=i)[0]
            for i, bs in enumerate(('1A', '3A', '1B'))]
    assert im.measure_halves(cams).verdict == 'INCONCLUSIVE'


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------
def _mef(path, state=None, frontend=None, undefined=False):
    ph = fits.PrimaryHDU()
    if undefined:
        ph.header.append(fits.Card('HIERARCH FRONTEND IFU MASK-STATE'))
    elif state is not None:
        ph.header['HIERARCH FRONTEND IFU MASK-STATE'] = state
    if frontend is not None:
        ph.header['HIERARCH FRONTEND STATE'] = frontend
    fits.HDUList([ph]).writeto(path, overwrite=True)
    return str(path)


def test_preflight_home_fails_only_for_sky_light_inputs(tmp_path):
    cfg = {
        'science_files': [_mef(tmp_path / 's1.fits', 'TOP'), _mef(tmp_path / 's2.fits', 'HOME'),
                          _mef(tmp_path / 'std.fits', 'HOME')],
        'flux_standard_files': [str(tmp_path / 'std.fits')],
        'red_flat_file': _mef(tmp_path / 'flat.fits', 'HOME', frontend='CALIBRATION'),
        'arc_file': _mef(tmp_path / 'arc.fits', 'HOME', frontend='CALIBRATION'),
        'green_twilight_flat': _mef(tmp_path / 'twi.fits', 'HOME'),
    }
    rows = im.assess_inputs(cfg, force_off=True)          # force_off never bypasses HOME
    failed = {(r.role, os.path.basename(r.path)) for r in im.closed_mask_failures(rows)}
    assert failed == {('science', 's2.fits'), ('standard', 'std.fits'),
                      ('green_twilight', 'twi.fits')}
    msg = im.format_closed_failure(im.closed_mask_failures(rows))
    assert 's2.fits' in msg and 'twi.fits' in msg and 'flat.fits' not in msg
    _summary, warnings, _table = im.preflight_report(rows, force_off=True)
    assert not any('flat.fits' in w or 'arc.fits' in w for w in warnings)   # lamp cals normal


def test_preflight_warnings(tmp_path):
    cfg = {
        'science_files': [_mef(tmp_path / 'a.fits', 'EMPTY'), _mef(tmp_path / 'b.fits', 'BOTTOM'),
                          _mef(tmp_path / 'c.fits', 'WEIRD'), _mef(tmp_path / 'std.fits', 'TOP')],
        'flux_standard_files': [str(tmp_path / 'std.fits')],
        'blue_flat_file': _mef(tmp_path / 'f.fits', 'TOP'),
        'red_arc_file': _mef(tmp_path / 'arc.fits', 'HOME', frontend='ADC'),
    }
    rows = im.assess_inputs(cfg, ifu_mask=True)
    summary, warnings, table = im.preflight_report(rows, ifu_mask=True)
    text = '\n'.join(warnings)
    assert 'f.fits' in text and 'not mask-aware' in text                 # masked calibration
    assert 'flux standard std.fits' in text                              # masked standard
    assert "'WEIRD'" in text                                             # unrecognised
    assert 'arc.fits' in text and 'FRONTEND STATE = ADC' in text         # HOME lamp, wrong path
    assert 'ignored for 1 frame' in text                                 # EMPTY + IFU_mask
    assert 'BOTTOM 1' in summary and len(table) == len(rows)
    assert not im.closed_mask_failures(rows)


def test_preflight_ifu_mask_without_blank_headers_warns(tmp_path):
    rows = im.assess_inputs({'science_files': [_mef(tmp_path / 'a.fits', 'EMPTY')]}, ifu_mask=True)
    _s, warnings, _t = im.preflight_report(rows, ifu_mask=True)
    assert any('has no effect' in w for w in warnings)
    rows = im.assess_inputs({'science_files': [_mef(tmp_path / 'u.fits', undefined=True)]},
                            ifu_mask=True)
    assert rows[0].request == im.REQ_CONFIG


def test_cli_exit_codes(tmp_path, capsys):
    ok = _mef(tmp_path / 'ok.fits', 'TOP')
    bias = tmp_path / 'bias.fits'
    ph = fits.PrimaryHDU()
    ph.header['PRODCATG'] = 'CAL.R-BIA'
    ph.header['HIERARCH FRONTEND IFU MASK-STATE'] = 'HOME'
    ph.writeto(bias)
    assert im.main([ok, str(bias)]) == 0
    bad = _mef(tmp_path / 'bad.fits', 'HOME')                            # science (no PRODCATG)
    assert im.main([ok, bad]) == 1
    assert 'bad.fits' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Sky model, RSS, downstream
# ---------------------------------------------------------------------------
def _write_pickle(path, cams, hdr):
    from llamas_pyjamas.Extract.extractLlamas import save_extractions
    save_extractions(cams, primary_header=hdr, savefile=str(path), save_dir=os.path.dirname(path))
    return str(path)


def _sky_of(pkl):
    from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas
    return ExtractLlamas.loadExtraction(pkl)['extractions'][0].sky


@pytest.mark.parametrize('method', ['stratified', 'dimmest'])
def test_sky_model_uses_open_half_only(tmp_path, method):
    from llamas_pyjamas.Sky.skyLlamas import skyModel_1d
    cam, truth = _camera('1A')
    open_rows = im.fibre_halves('1A', np.arange(cam.counts.shape[0])) == im.LOWX

    plain = _write_pickle(tmp_path / 'plain_extractions.pkl', [cam], fits.Header())
    stamped = _write_pickle(tmp_path / 'stamped_extractions.pkl', [cam], _stamp(im.HIGHX))
    sky_plain = _sky_of(skyModel_1d(plain, None, selection_method=method))
    sky_mask = _sky_of(skyModel_1d(stamped, None, selection_method=method))

    ratio_mask = sky_mask[open_rows] / truth[open_rows]
    ratio_plain = sky_plain[open_rows] / truth[open_rows]
    assert np.median(np.abs(ratio_mask - 1.0)) < 0.02     # open-half sky recovered
    assert np.percentile(np.abs(ratio_mask - 1.0), 99) < 0.1
    assert np.median(ratio_plain) < 0.5                   # the bug: no usable sky from dark fibres


def _rss_from_masked_pickle(tmp_path, stamped=True):
    from llamas_pyjamas.File.llamasRSS import RSSgeneration
    cam, truth = _camera('1A')
    cam.sky = truth.copy()                                   # what skyModel_1d would write
    hdr = _stamp(im.HIGHX) if stamped else fits.Header()
    pkl = _write_pickle(tmp_path / f"x{'s' if stamped else 'p'}_sky1d_extractions.pkl", [cam], hdr)
    out = tmp_path / f"x{'s' if stamped else 'p'}_RSS.fits"
    RSSgeneration().generate_rss(pkl, str(out), subtract_sky=True, wave_frame='none')
    files = sorted(glob.glob(str(tmp_path / f"x{'s' if stamped else 'p'}_RSS*.fits")))
    assert files
    return files[0], cam


def test_generate_rss_flags_blocked_fibres(tmp_path):
    path, cam = _rss_from_masked_pickle(tmp_path, stamped=True)
    with fits.open(path) as h:
        blk = np.asarray(h['FIBERMAP'].data[im.FIBERMAP_COLUMN], dtype=bool)
        expected = im.blocked_mask('1A', h['FIBERMAP'].data['FIBER_ID'], im.HIGHX)
        assert np.array_equal(blk, expected) and blk.sum() == 149
        assert np.all((h['MASK'].data[blk] & 4) != 0) and not np.any(h['MASK'].data[~blk])
        assert np.all(h['SKY'].data[blk] == 0) and np.all(h['SKY'].data[~blk] != 0)
        assert np.allclose(h['SKYSUB'].data[blk], h['COUNTS'].data[blk])
        assert h[0].header['NIFUMSK'] == 149 and h[0].header['IFUMASK'] is True
        assert np.array_equal(im.rss_blocked_rows(h), blk)


def test_generate_rss_unstamped_is_unchanged(tmp_path):
    path, _cam = _rss_from_masked_pickle(tmp_path, stamped=False)
    with fits.open(path) as h:
        assert not np.any(h['FIBERMAP'].data[im.FIBERMAP_COLUMN])
        assert not np.any(h['MASK'].data) and h[0].header['NIFUMSK'] == 0
        assert np.all(h['SKY'].data != 0)


def test_framework_and_pedestal_pools_exclude_blocked(tmp_path):
    from llamas_pyjamas.Sky.skyConfig import SkySubtractConfig
    from llamas_pyjamas.Sky.skyMask import build_sky_fiber_mask
    from llamas_pyjamas.Sky.skyPedestal import _rss_blank_mask
    path, _cam = _rss_from_masked_pickle(tmp_path, stamped=True)
    with fits.open(path) as h:
        counts = np.asarray(h['COUNTS'].data, float)
        fibermap = h['FIBERMAP'].data
        blk = im.rss_blocked_rows(h)
    for sel in ('dimmest', 'middle-third'):
        cfg = SkySubtractConfig()
        cfg.selection_method = sel
        assert not np.any(build_sky_fiber_mask(counts, fibermap, cfg).mask & blk)
    assert np.any(_rss_blank_mask(counts, 60.0) & blk)                   # the faintest = blocked
    assert not np.any(_rss_blank_mask(counts, 60.0, exclude=blk) & blk)


def test_pedestal_leaves_blocked_rows_alone(tmp_path):
    from llamas_pyjamas.Sky.skyPedestal import apply_pedestal_rss
    src, _cam = _rss_from_masked_pickle(tmp_path, stamped=True)
    path = str(tmp_path / 'p_RSS_green.fits')
    os.replace(src, path)
    with fits.open(path) as h:
        sky0, skysub0 = h['SKY'].data.copy(), h['SKYSUB'].data.copy()
        blk = im.rss_blocked_rows(h)
    apply_pedestal_rss(path, {'sky_pedestal_scope': 'camera'})
    with fits.open(path) as h:
        assert h[0].header.get('SKYPED') is True
        assert np.array_equal(h['SKY'].data[blk], sky0[blk])
        assert np.array_equal(h['SKYSUB'].data[blk], skysub0[blk])


def test_framework_leaves_blocked_rows_alone(tmp_path):
    from llamas_pyjamas.Sky.skyConfig import SkySubtractConfig
    from llamas_pyjamas.Sky.skySubtract import subtract_sky_rss
    src, _cam = _rss_from_masked_pickle(tmp_path, stamped=True)
    path = str(tmp_path / 'f_RSS_green_FF.fits')
    os.replace(src, path)
    with fits.open(path) as h:
        skysub0 = h['SKYSUB'].data.copy()
        blk = im.rss_blocked_rows(h)
    out = subtract_sky_rss(path, config=SkySubtractConfig())
    with fits.open(out) as h:
        # the framework writes float32; blocked rows are the input values at that precision
        assert np.array_equal(h['SKYSUB'].data[blk], skysub0[blk].astype(np.float32))
        assert not np.array_equal(h['SKYSUB'].data[~blk], skysub0[~blk].astype(np.float32))


def _band_columns(x_grid, y_grid, cube, x_lo, x_hi):
    cols = (x_grid > x_lo) & (x_grid < x_hi)
    return cube[:, :, cols]


def test_simple_cube_blanks_blocked_half(tmp_path):
    from llamas_pyjamas.Cube.simple_cube_constructor import SimpleCubeConstructor
    path, _cam = _rss_from_masked_pickle(tmp_path, stamped=True)
    c = SimpleCubeConstructor(grid_method='oversampled')
    c.load_rss_file(path)
    c.match_fibers_to_fibermap()
    c.create_wavelength_grid()
    c.create_spatial_grid()
    c.construct_cube(radius=1.5, min_weight=0.01)
    yb = c.fiber_coords[:, 1]
    rows = (c.y_grid > yb.min() + 0.5) & (c.y_grid < yb.max() - 0.5)
    cube = c.cube[:, rows, :]
    pitch = c.fiber_pitch
    blocked_side = _band_columns(c.x_grid, None, cube, (im.X_SPLIT + 1) * pitch, 45 * pitch)
    open_side = _band_columns(c.x_grid, None, cube, 2 * pitch, (im.X_SPLIT - 1) * pitch)
    assert np.all(np.isnan(blocked_side))
    assert np.all(np.isfinite(open_side))


def test_traditional_cube_blanks_blocked_half(tmp_path):
    from llamas_pyjamas.Cube.cubeConstruct import CubeConstructor
    path, _cam = _rss_from_masked_pickle(tmp_path, stamped=True)
    with fits.open(path) as h:
        h[0].header['CHANNEL'] = 'green'
        h.writeto(path, overwrite=True)
    c = CubeConstructor()
    cube = c.construct_cube_from_rss_channel(path, 'green', spatial_sampling=0.75)
    x = c.spatial_grid_x
    good_wave = np.isfinite(cube).any(axis=(1, 2))
    cube = cube[good_wave]
    blocked_side = cube[:, :, x > (im.X_SPLIT + 1) * 0.75]
    open_side = cube[:, :, (x > 2 * 0.75) & (x < (im.X_SPLIT - 1) * 0.75)]
    assert np.all(np.isnan(blocked_side))
    assert np.isfinite(open_side).mean() > 0.9


def test_per_fibre_flux_blocked_rows_are_nan(tmp_path):
    from llamas_pyjamas.Utils.register import per_fibre_flux
    path, _cam = _rss_from_masked_pickle(tmp_path, stamped=True)
    with fits.open(path) as h:
        flux = per_fibre_flux(h)
        blk = im.rss_blocked_rows(h)
    assert np.all(np.isnan(flux[blk])) and np.all(np.isfinite(flux[~blk]))
    assert im.file_is_half_masked(path)


# ---------------------------------------------------------------------------
# Config / GUI / resume guard
# ---------------------------------------------------------------------------
def test_gui_option_round_trip(tmp_path):
    from llamas_pyjamas.Utils.reduxSetupGUI import CONFIG_CATALOG, generate_config, parse_config
    sky_keys = [k[0] for cat, items in CONFIG_CATALOG if cat == 'Sky' for k in items]
    assert 'IFU_mask' in sky_keys
    text = generate_config({}, output_dir=str(tmp_path), overrides={'IFU_mask': 'true'})
    cfg_path = tmp_path / 'cfg.txt'
    cfg_path.write_text(text)
    config, _bad = parse_config(str(cfg_path))
    assert str(config.get('IFU_mask')).lower() == 'true'
    assert im.config_switches(config)[0] is True


def test_resume_guard_detects_changed_requests(tmp_path):
    from llamas_pyjamas.reduce import _ifu_mask_resume_conflicts

    def rss(stem, req=None):
        ph = fits.PrimaryHDU()
        if req is not None:
            ph.header['IFUMREQ'] = req
        ph.writeto(tmp_path / f'{stem}_SCI22_mef_extract_RSS_green.fits')

    rss('LLAMAS_2026-01-07_09-16-45.2')                     # pre-feature product (no stamp)
    rss('LLAMAS_2026-01-07_08-42-44.3', 'HEADER')
    rss('LLAMAS_2026-01-07_08-30-11.6')
    rows = [im.PreflightRow('science', '/raw/LLAMAS_2026-01-07_09-16-45.2_SCI22_mef.fits',
                            'TOP', im.REQ_HEADER),
            im.PreflightRow('science', '/raw/LLAMAS_2026-01-07_08-42-44.3_SCI22_mef.fits',
                            'TOP', im.REQ_HEADER),
            im.PreflightRow('science', '/raw/LLAMAS_2026-01-07_08-30-11.6_SCI22_mef.fits',
                            'EMPTY', im.REQ_NONE),
            im.PreflightRow('red_flat', '/raw/LLAMAS_2026-01-07_08-30-11.6_CAL22_mef.fits',
                            'TOP', im.REQ_HEADER)]
    stale = _ifu_mask_resume_conflicts(rows, str(tmp_path))
    assert stale == [('LLAMAS_2026-01-07_09-16-45.2_SCI22_mef.fits', 'NONE', 'HEADER')]


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
