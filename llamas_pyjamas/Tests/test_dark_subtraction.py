"""Tests for master-dark subtraction in the bias-first step.

Synthetic 24-camera MEFs (69x40 imaging area plus a metadata row, uint16 with BZERO)
with known per-camera bias, dark and signal. Every array is laid out in pipeline
orientation and flipped to raw orientation for writing, so a step applied in the
wrong orientation shows up as a mismatch.

Runnable with pytest or as a plain script
(`python -m llamas_pyjamas.Tests.test_dark_subtraction`).
"""

import os
import time
import types

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import cloudpickle
import numpy as np
import pytest
from astropy.io import fits

import llamas_pyjamas.Bias.darkLlamas as dl
from llamas_pyjamas.Bias import camera_of_header, match_camera_hdu, verify_camera_hdu
from llamas_pyjamas.Bias.biasChecking import _MASK_CACHE, _cache_mask, _cached_mask
from llamas_pyjamas.Bias.biasFirst import (bias_correct_frame, dark_summary,
                                           expected_provenance, provenance_of)
from llamas_pyjamas.Bias.biasFirstResume import is_product_of, redo_stale_frames
from llamas_pyjamas.Extract.extractLlamas import ExtractLlamas, aperture_sum, dark_variance
from llamas_pyjamas.File.llamasIO import (orient_by_color, process_fits_by_color,
                                          to_raw, trim_and_orient)
import llamas_pyjamas.Masking.cosmicLlamas as cosmic
from llamas_pyjamas.Utils.reduxSetupGUI import (ReduxSetupWindow, generate_config,
                                                parse_config)

NY, NX = 69, 40                      # imaging area; the raw frame has one more row
DATASEC = f'[1:{NX}, 1:{NY}]'
CAMS = dl.CAMERAS
TEMPLATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        'example_config.txt')


def label(cam):
    return f"{cam[0]}{cam[1]} {cam[2]}"


def camsn(cam):
    return f"SN{CAMS.index(cam):03d}"


def arrays(seed=0):
    """Per camera: integer bias, dark and signal in pipeline orientation (asymmetric)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:NY, 0:NX]
    out = {}
    for i, cam in enumerate(CAMS):
        bias = 500 + 3 * yy + 2 * xx + rng.integers(0, 4, (NY, NX)) + i
        dark = 20 + (10 * xx) // NX + (yy > 50) * 7
        dark[5 + i % 7, 3 + i % 11] += 400                     # a hot pixel
        signal = rng.integers(0, 100, (NY, NX))
        out[cam] = (bias.astype(np.int64), dark.astype(np.int64), signal.astype(np.int64))
    return out


def ext_header(cam, color_case=str.upper):
    h = fits.Header()
    h['COLOR'], h['BENCH'], h['SIDE'] = color_case(cam[2]), int(cam[0]), cam[1]
    h['CAMSN'] = camsn(cam)
    h['DATASEC'] = DATASEC
    return h


def write_mef(path, images, primary=None, order=None, extra=None):
    """Raw-orientation MEF: images[cam] is the pipeline-orientation imaging area (ints)."""
    hdus = [fits.PrimaryHDU(header=primary or fits.Header())]
    for cam in (order or list(images)):
        h = ext_header(cam)
        if extra:
            extra(cam, h)
        raw = np.full((NY + 1, NX), 7, dtype=np.int64)        # metadata row = 7
        raw[:NY] = orient_by_color(images[cam], cam[2])
        hdus.append(fits.ImageHDU(raw.astype(np.uint16), header=h))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return str(path)


def primary(cat, mode='SLOW', rexp=1350.0, dexp=1351.0, obj=''):
    h = fits.Header()
    h['PRODCATG'], h['READ-MDE'], h['REXPTIME'], h['OBJECT'] = cat, mode, rexp, obj
    if dexp is not None:
        h['DEXPTIME'] = dexp
    return h


def write_bias(path, data, constant=()):
    hdus = [fits.PrimaryHDU(header=primary('CAL.R-BIA', rexp=0.001, dexp=1.0))]
    for cam, (bias, _d, _s) in data.items():
        img = np.ones((NY, NX)) if cam in constant else orient_by_color(bias, cam[2])
        hdus.append(fits.ImageHDU(img.astype(np.float32), header=ext_header(cam)))
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return str(path)


@pytest.fixture
def night(tmp_path):
    """Bias file, 3 darks (one with a cosmic ray) and a science frame."""
    data = arrays()
    bias = write_bias(tmp_path / 'bias.fits', data)
    darks = []
    for k in range(3):
        imgs = {cam: b + d for cam, (b, d, _s) in data.items()}
        if k == 1:
            imgs = {cam: img.copy() for cam, img in imgs.items()}
            for img in imgs.values():
                img[40, 20] += 3000                            # cosmic ray in one dark
        darks.append(write_mef(tmp_path / f'LLAMAS_2026-10-15_17-0{k}-00.0_CAL24_mef.fits', imgs,
                               primary('CAL.R-DRK', obj='DARK')))
    sci_imgs = {cam: b + d + s for cam, (b, d, s) in data.items()}
    sci = write_mef(tmp_path / 'LLAMAS_2026-10-16_00-31-08.4_SCI24_mef.fits', sci_imgs,
                    primary('SCI.R-SL', obj='FIELD'))
    empty = tmp_path / 'no_traces'
    empty.mkdir()
    return types.SimpleNamespace(tmp=tmp_path, data=data, bias=bias, darks=darks, sci=sci,
                                 empty=str(empty))


def build(night, **kw):
    group = dl.group_darks(night.darks)[0]
    return dl.build_master_dark(group, night.bias, str(night.tmp / 'darks'), **kw)


def correct(night, master, sci=None, **kw):
    return bias_correct_frame(sci or night.sci, str(night.tmp / 'bc'), slow_bias=night.bias,
                              trace_dir=None, mastercalib_trace_dir=night.empty,
                              dark=master, **kw)


def pipeline_view(path):
    hdus, _ = process_fits_by_color(path, output_file=path + '.trim.fits')
    os.remove(path + '.trim.fits')
    return {(str(h.header['BENCH']), h.header['SIDE'], h.header['COLOR'].lower()): h
            for h in hdus[1:]}


# ---------------------------------------------------------------------------
# Orientation helpers
# ---------------------------------------------------------------------------

def test_orient_is_its_own_inverse_and_matches_process_fits_by_color(tmp_path):
    img = np.arange(NY * NX).reshape(NY, NX)
    for color in ('red', 'green', 'blue'):
        assert np.array_equal(orient_by_color(orient_by_color(img, color), color), img)
    raw = np.arange((NY + 1) * NX).reshape(NY + 1, NX)
    h = ext_header(('1', 'A', 'blue'))
    p = trim_and_orient(raw, h)
    assert p.shape == (NY, NX) and np.array_equal(p, np.flipud(np.fliplr(raw[:NY])))
    back = to_raw(p, h, raw)
    assert np.array_equal(back, raw.astype(np.float32))


# ---------------------------------------------------------------------------
# Matching darks to frames
# ---------------------------------------------------------------------------

def test_match_dark_rules():
    def g(mode, rexp, n=3, dexp=None):
        grp = dl.DarkGroup(mode=mode, rexptime=rexp)
        grp.files = [f'{mode}{rexp}_{i}' for i in range(n)]
        grp.dexptimes = [dexp if dexp else rexp + 1.0] * n
        return grp
    groups = [g('SLOW', 1350.0), g('SLOW', 1300.0), g('SLOW', 10.0, 2), g('FAST', 1350.0)]

    def frame(rexp, mode='SLOW', dexp=None):
        h = {'REXPTIME': rexp}
        if mode:
            h['READ-MDE'] = mode
        h['DEXPTIME'] = dexp if dexp is not None else rexp + 1.0
        return h

    grp, scale, note = dl.match_dark(frame(1340.0), groups)
    assert grp.label == 'SLOW 1350 s'                      # nearest of 1300 and 1350
    assert scale == pytest.approx(1341.0 / 1351.0) and note == ''
    assert dl.match_dark(frame(1350.0, mode='FAST'), groups)[0].mode == 'FAST'
    assert dl.match_dark(frame(1350.0, mode=None), groups)[0].mode == 'SLOW'
    grp, _s, note = dl.match_dark(frame(1150.0), groups)  # 1300 is 13% away
    assert grp is None and note == 'no SLOW 1150 s dark'
    # the tolerance is a fraction of the frame's exposure, as in the reference notebook
    assert dl.match_dark(frame(1182.0), groups)[0].label == 'SLOW 1300 s'   # 118 <= 118.2
    assert dl.match_dark(frame(1181.0), groups)[0] is None                  # 119 > 118.1
    grp, scale, note = dl.match_dark(frame(1350.0, dexp=622.5), groups)     # cut short
    assert grp is not None and scale == pytest.approx(622.5 / 1351.0) and 'cut short' in note
    nodex = {'READ-MDE': 'SLOW', 'REXPTIME': 1350.0}
    assert dl.match_dark(nodex, groups)[1] == pytest.approx(1.0)            # REXPTIME ratio


def test_check_dark_files_refuses_non_darks(night):
    sci_err, _w = dl.check_dark_files([night.sci])
    assert sci_err and 'not a dark' in sci_err[0]
    assert dl.check_dark_files(night.darks) == ([], [])


# ---------------------------------------------------------------------------
# Master dark
# ---------------------------------------------------------------------------

def test_master_dark_equals_median_minus_bias_in_true_orientation(night):
    master = build(night)
    assert master.ncombine == 3 and len(master.cameras) == 24
    with fits.open(master.path) as hdul:
        assert hdul[0].header['DARKID'] == master.darkid
        for h in hdul[1:]:
            cam = camera_of_header(h.header)
            # stored raw orientation; the cosmic ray in one dark is rejected by the median
            assert np.array_equal(h.data, orient_by_color(night.data[cam][1], cam[2]))
            assert h.header['NCOMBINE'] == 3 and h.header['CAMSN'] == camsn(cam)
    again = build(night)
    assert again.path == master.path and again.darkid == master.darkid      # reused


def test_master_dark_id_changes_with_its_darks(night):
    first = build(night)
    st = os.stat(night.darks[0])
    os.utime(night.darks[0], (st.st_atime, st.st_mtime + 10))
    second = build(night)
    assert second.darkid != first.darkid and os.path.exists(first.path)


def test_two_darks_reject_a_cosmic_ray(night):
    group = dl.group_darks(night.darks[:2])[0]                 # dark 1 has the cosmic ray
    master = dl.build_master_dark(group, night.bias, str(night.tmp / 'd2'))
    with fits.open(master.path) as hdul:
        for h in hdul[1:]:
            cam = camera_of_header(h.header)
            assert h.header['DARKMETH'] == 'pair_min'
            assert np.allclose(h.data, orient_by_color(night.data[cam][1], cam[2]))


def test_single_dark_is_filtered(night):
    group = dl.group_darks(night.darks[1:2])[0]
    master = dl.build_master_dark(group, night.bias, str(night.tmp / 'd1'))
    with fits.open(master.path) as hdul:
        assert all(h.header['DARKMETH'] == 'filter5' for h in hdul[1:])
        assert max(float(h.data.max()) for h in hdul[1:]) < 400      # cosmic ray filtered out


def test_constant_bias_and_placeholder_cameras_are_left_out(night):
    data = night.data
    bias = write_bias(night.tmp / 'bias_fill.fits', data, constant=[('1', 'A', 'blue')])
    # 4A blue is a zero placeholder in every dark
    darks = [write_mef(night.tmp / f'ph_{k}_CAL24_mef.fits',
                       {cam: (np.zeros((NY, NX), np.int64) if cam == ('4', 'A', 'blue') else b + d)
                        for cam, (b, d, _s) in data.items()},
                       primary('CAL.R-DRK')) for k in range(3)]
    master = dl.build_master_dark(dl.group_darks(darks)[0], bias, str(night.tmp / 'dph'))
    assert master.cameras['1A blue']['note'] == 'the master bias is a constant fill for this camera'
    assert master.cameras['4A blue']['note'] == 'no usable dark frames for this camera'
    with fits.open(master.path) as hdul:
        names = {camera_of_header(h.header) for h in hdul[1:]}
    assert ('1', 'A', 'blue') not in names and ('4', 'A', 'blue') not in names and len(names) == 22
    reread = dl._report_from_file(master.path)
    assert reread.cameras['1A blue']['note'].startswith('the master bias is a constant fill')


# ---------------------------------------------------------------------------
# Bias-first with the dark
# ---------------------------------------------------------------------------

def test_bias_first_recovers_signal_exactly_for_every_colour(night):
    master = build(night)
    out = correct(night, master)
    views = pipeline_view(out)
    with fits.open(out) as hdul, fits.open(night.sci) as raw:
        assert hdul[0].header['BFVER'] == 2 and hdul[0].header['DARKID'] == master.darkid
        for x, y in zip(raw[1:], hdul[1:]):
            assert np.array_equal(y.data[NY:], x.data[NY:].astype(np.float32))  # metadata row
    for cam, (_b, dark, signal) in night.data.items():
        h = views[cam]
        assert h.header['DARKSUB'] and h.header['DARKSCL'] == pytest.approx(1.0)
        assert np.array_equal(h.data, signal.astype(np.float32)), label(cam)
    assert all(applied for _c, applied, _f in dark_summary(out))


def test_mirrored_bias_would_not_recover_the_signal(night):
    # The bug this replaces: a flipped bias subtracted from the raw frame.
    b, _d, _s = night.data[('2', 'A', 'blue')]
    assert not np.array_equal(orient_by_color(b, 'blue'), b)


def test_extensions_out_of_order_are_paired_by_header(night):
    data = night.data
    order = list(reversed(CAMS))
    darks = [write_mef(night.tmp / f'rev_{k}_CAL24_mef.fits',
                       {cam: b + d for cam, (b, d, _s) in data.items()},
                       primary('CAL.R-DRK'), order=order) for k in range(3)]
    sci = write_mef(night.tmp / 'LLAMAS_2026-10-16_01-00-00.0_SCI24_mef.fits',
                    {cam: b + d + s for cam, (b, d, s) in data.items()},
                    primary('SCI.R-SL'), order=order[3:] + order[:3])
    master = dl.build_master_dark(dl.group_darks(darks)[0], night.bias, str(night.tmp / 'drev'))
    views = pipeline_view(correct(night, master, sci=sci))
    for cam, (_b, _d, signal) in data.items():
        assert np.array_equal(views[cam].data, signal.astype(np.float32)), label(cam)


def _relabel_master(master, path, edit):
    with fits.open(master.path) as hdul:
        hdus = [h.copy() for h in hdul]
    edit(hdus)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return dl.MasterDark(path=str(path), darkid=master.darkid, mode=master.mode,
                         rexptime=master.rexptime, dexptime=master.dexptime,
                         ncombine=master.ncombine)


def test_dark_extension_failures_give_no_dark_never_another_camera(night):
    master = build(night)

    def edit(hdus):
        for h in hdus[1:]:
            cam = camera_of_header(h.header)
            if cam == ('1', 'A', 'red'):
                h.header['BENCH'] = 2                     # mislabelled: now claims 2A red
            if cam == ('1', 'B', 'green'):
                h.header['CAMSN'] = 'OTHER'               # a different detector
            if cam == ('3', 'A', 'red'):
                h.data = h.data[:-1]                      # wrong shape
        del hdus[[i for i, h in enumerate(hdus[1:], 1)
                  if camera_of_header(h.header) == ('4', 'B', 'blue')][0]]   # missing camera
    bad = _relabel_master(master, night.tmp / 'bad_master.fits', edit)
    out = correct(night, bad, sci=night.sci)
    views = pipeline_view(out)
    why = {c: d for c, applied, d in dark_summary(out) if not applied}
    assert why['1A red'] == 'no extension for this camera'
    assert why['1B green'].startswith('detector serial differs')
    assert why['3A red'].startswith('shape')
    assert why['4B blue'] == 'no extension for this camera'
    # 2A red now has two candidate extensions; the first (1A red's data, relabelled)
    # carries 1A red's serial, so it is refused rather than used.
    assert why['2A red'].startswith('detector serial differs')
    for cam in [('1', 'A', 'red'), ('2', 'A', 'red'), ('1', 'B', 'green')]:
        b, d, s = night.data[cam]
        assert np.array_equal(views[cam].data, (d + s).astype(np.float32))   # bias only
    assert len(why) == 5


def test_read_mode_and_identity_checks_on_load(night):
    master = build(night)
    hdr = ext_header(('1', 'A', 'red'))
    assert dl.load_master_dark_camera(master.path, hdr, master.darkid, mode='FAST')[1]['reason'] \
        == 'master dark is SLOW, frame is FAST'
    assert 'identity changed' in dl.load_master_dark_camera(master.path, hdr, 'nope')[1]['reason']
    data, info = dl.load_master_dark_camera(master.path, hdr, master.darkid, mode='SLOW')
    assert info['reason'] == '' and np.array_equal(data, night.data[('1', 'A', 'red')][1])


def test_shared_camera_helpers():
    hdus = [fits.PrimaryHDU()] + [fits.ImageHDU(np.zeros((2, 2)), header=ext_header(c))
                                  for c in CAMS[:6]]
    assert camera_of_header(match_camera_hdu(hdus, '1', 'b', 'Green').header) == ('1', 'B', 'green')
    assert match_camera_hdu(hdus, '4', 'B', 'blue') is None           # all named, absent
    assert verify_camera_hdu(ext_header(('1', 'A', 'red')), '1', 'A', 'red')
    assert not verify_camera_hdu(ext_header(('1', 'A', 'red')), '2', 'A', 'red')
    assert verify_camera_hdu(fits.Header(), '1', 'A', 'red')          # nothing to contradict
    anon = [fits.PrimaryHDU()] + [fits.ImageHDU(np.zeros((2, 2))) for _ in range(24)]
    assert match_camera_hdu(anon, '1', 'A', 'red') is None            # index hit names no camera


def test_placeholders_pass_through(night):
    data = night.data
    imgs = {cam: b + d + s for cam, (b, d, s) in data.items()}
    imgs[('2', 'B', 'red')] = np.zeros((NY, NX), np.int64)
    sci = write_mef(night.tmp / 'LLAMAS_2026-10-16_02-00-00.0_SCI24_mef.fits', imgs,
                    primary('SCI.R-SL'))
    out = correct(night, build(night), sci=sci)
    with fits.open(out) as hdul:
        h = next(x for x in hdul[1:] if camera_of_header(x.header) == ('2', 'B', 'red'))
        assert h.header['BIASSUB'] is False and h.header['DARKWHY'] == 'dead camera in this frame'
        assert not np.any(h.data[:NY])


def test_scaled_dark(night):
    data = night.data
    sci = write_mef(night.tmp / 'LLAMAS_2026-10-16_03-00-00.0_SCI24_mef.fits',
                    {cam: b + 2 * d + s for cam, (b, d, s) in data.items()},
                    primary('SCI.R-SL', rexp=1350.0, dexp=2702.0))
    master = build(night)
    views = pipeline_view(correct(night, master, sci=sci))
    for cam, (_b, _d, signal) in data.items():
        assert views[cam].header['DARKSCL'] == pytest.approx(2.0)
        assert np.allclose(views[cam].data, signal, atol=1e-3)


def test_stripe_mask_and_edge_dc_in_true_orientation(night):
    """Fibres in the top rows (pipeline orientation): the stripes are the bottom rows."""
    data = night.data
    traces = night.tmp / 'traces'
    traces.mkdir()
    fiberimg = -np.ones((NY, NX), int)
    fiberimg[45:63] = 0
    for cam in CAMS:
        with open(traces / f'LLAMAS_{cam[2]}_{cam[0]}_{cam[1]}_traces.pkl', 'wb') as f:
            cloudpickle.dump(types.SimpleNamespace(fiberimg=fiberimg), f)
    rng = np.random.default_rng(3)
    signal = {}
    for cam in CAMS:
        s = 5 + rng.integers(-1, 2, (NY, NX))                 # unlit level 5 DN
        s[45:63] += 1000
        signal[cam] = s
    sci = write_mef(night.tmp / 'LLAMAS_2026-10-16_04-00-00.0_SCI24_mef.fits',
                    {cam: data[cam][0] + data[cam][1] + signal[cam] for cam in CAMS},
                    primary('SCI.R-SL'))
    out = bias_correct_frame(sci, str(night.tmp / 'bc_tr'), slow_bias=night.bias,
                             trace_dir=str(traces), mastercalib_trace_dir=night.empty,
                             dark=build(night), edge_bias={'min_distance': 20, 'min_pixels': 500})
    views = pipeline_view(out)
    for cam in CAMS:
        h = views[cam]
        assert h.header['EDGEDC'] == pytest.approx(5.0), label(cam)
        assert h.header['DARKRES'] == pytest.approx(5.0) and h.header['EDGETRC'] == 'night'
        assert np.allclose(h.data, signal[cam] - 5.0)


class FakeTracer:
    """Like a loaded TraceLlamas for the mask cache: weakly referenceable."""
    def __init__(self, fiberimg):
        self.fiberimg = fiberimg


def test_mask_cache_never_crosses_tracers():
    a = FakeTracer(np.zeros((4, 4)))
    b = FakeTracer(np.zeros((4, 4)))
    mask = np.ones((4, 4), bool)
    _cache_mask(('k',), a, mask)
    assert _cached_mask(('k',), a) is mask
    assert _cached_mask(('k',), b) is None
    del a
    _cache_mask(('j',), b, mask)                             # prunes the dead entry
    assert ('k',) not in _MASK_CACHE
    _MASK_CACHE.clear()


# ---------------------------------------------------------------------------
# Extraction: errors and cosmic rays
# ---------------------------------------------------------------------------

def _boxcar_loop(image, trace_y, half):
    out = np.zeros(trace_y.shape)
    for f in range(trace_y.shape[0]):
        for i in range(trace_y.shape[1]):
            yc = trace_y[f, i]
            lo, hi = yc - half, yc + half
            for jj in range(int(np.floor(lo + 0.5)), int(np.floor(hi + 0.5 - 1e-9)) + 1):
                w = min(hi, jj + 0.5) - max(lo, jj - 0.5)
                if w > 0 and 0 <= jj < image.shape[0]:
                    out[f, i] += image[jj, i] * min(w, 1.0)
    return out


def test_aperture_sum_matches_the_boxcar_loop():
    rng = np.random.default_rng(1)
    img = rng.random((NY, NX))
    traces = np.vstack([20.3 + 0.01 * np.arange(NX), 40.5 + 0.0 * np.arange(NX)])
    assert np.allclose(aperture_sum(img, traces, 2.5), _boxcar_loop(img, traces, 2.5))


def test_extraction_errors_include_the_dark(monkeypatch):
    monkeypatch.setenv('LLAMAS_EXTRACT_METHOD', 'boxcar')
    monkeypatch.setenv('LLAMAS_BOXCAR_HALFWIDTH', '2.5')
    traces = np.vstack([20.0 + 0 * np.arange(NX), 40.0 + 0 * np.arange(NX)])
    trace = types.SimpleNamespace(bench='1', side='A', channel='red', fitsfile='x.fits',
                                  nfibers=2, naxis1=NX, naxis2=NY, traces=traces)
    hdr = fits.Header({'EGAIN': 2.0, 'RDNOISE': 3.0, 'DARKSCL': 0.9})
    frame = np.full((NY, NX), 40.0)
    dark = np.full((NY, NX), 30.0)
    plain = ExtractLlamas(trace, frame, hdr, method='boxcar')
    with_dark = ExtractLlamas(trace, frame, hdr, method='boxcar', dark=dark, dark_noise=0.4)
    assert np.array_equal(plain.counts, with_dark.counts)
    added = with_dark.counts_err.astype(float) ** 2 - plain.counts_err.astype(float) ** 2
    S = 5 * 30.0                                                # 5 px aperture
    expect = S / 2.0 + 0.9 * 0.4 * S / 2.0 + 0.9 ** 2 * 0.4 * 5 * (3.0 / 2.0) ** 2
    assert np.allclose(added, expect, rtol=1e-4)
    assert np.allclose(dark_variance(dark, traces, 2.5, 5, 2.0, 3.0, 0.9, 0.4), expect)


def test_cosmic_ray_inputs_and_passthrough(monkeypatch):
    dark = np.full((NY, NX), 30.0)
    dark[10, 10] = 900.0
    smooth, hot = dl.cosmic_ray_inputs(dark)
    assert smooth.max() < 100 and hot[10, 10] and hot.sum() == 1
    seen = {}

    def fake(data, **kw):
        seen.update(kw)
        return data, np.zeros(data.shape, bool)
    monkeypatch.setattr(cosmic, 'remove_cosmics', fake)
    cosmic.clean_cosmic_rays(np.zeros((NY, NX)), color='blue', background=smooth, mask=hot)
    assert seen['background'] is smooth and seen['mask'] is hot


# ---------------------------------------------------------------------------
# Resume: out-of-date frames
# ---------------------------------------------------------------------------

def _product(path, prov=None):
    h = fits.Header()
    for k, v in (prov or {}).items():
        h[k] = v
    fits.PrimaryHDU(np.zeros(2), header=h).writeto(path, overwrite=True)


def _layout(tmp, sci_prov, cal_prov):
    out, ext, cubes = tmp / 'out', tmp / 'out' / 'extractions', tmp / 'out' / 'cubes'
    for d in (out / 'bias_corrected', ext / 'masks', ext / 'flat', cubes, out / 'traces',
              out / 'arcs'):
        d.mkdir(parents=True, exist_ok=True)
    sci = 'LLAMAS_2026-10-16_00-31-08.4_SCI24_mef'
    other = 'LLAMAS_2026-10-16_00-54-54.4_SCI24_mef'
    flat = 'LLAMAS_2026-10-15_16-39-02.9_CAL24_mef'
    for stem, prov in ((sci, sci_prov), (other, cal_prov), (flat, cal_prov)):
        _product(out / 'bias_corrected' / f'{stem}_bias_corrected.fits', prov)
    for stem, prov in ((sci, sci_prov), (other, cal_prov)):
        _product(out / f'{stem}_bias_corrected_flat_corrected.fits', prov)
        _product(ext / f'{stem}_bias_corrected_flat_corrected_RSS_red.fits', prov)
        (ext / f'{stem}_bias_corrected_flat_corrected_extract.pkl').write_bytes(b'x')
        (ext / 'masks' / f'cosmic_ray_mask_{stem}_bias_corrected_flat_corrected.fits').write_bytes(b'x')
        (cubes / f'{stem}_bias_corrected_cube_red.fits').write_bytes(b'x')
    raw_sci = out / f'{sci}.fits'                          # a raw frame inside output_dir
    raw_sci.write_bytes(b'raw')
    (out / f'{sci}_whitelight.fits').write_bytes(b'quicklook')
    (out / 'traces' / 'LLAMAS_red_1_A_traces.pkl').write_bytes(b't')
    (ext / 'flat' / 'pixel_maps.fits').write_bytes(b'f')
    (out / 'arcs' / 'arc_extract.pkl').write_bytes(b'a')
    (ext / 'floor_template_cps_red.fits').write_bytes(b'p')
    return out, ext, cubes, raw_sci, sci, other, flat


def _redo(tmp, frames, out, ext, cubes, raw_sci, since=None):
    return redo_stale_frames(
        frames, dirs=[out, out / 'bias_corrected', ext, ext / 'masks', cubes],
        calib_dirs={'trace_dir': out / 'traces', 'extraction_dir': ext,
                    'flat_dir': ext / 'flat', 'qa_dir': out / 'QA'},
        output_dir=str(out), protected=[str(raw_sci)],
        since=time.time() + 1 if since is None else since)


def test_redo_moves_only_the_stale_science_exposure(tmp_path):
    cur = {'BFVER': 2, 'BIASID': 'b1', 'DARKID': 'none', 'DARKSCL': 0.0}
    want = {'BFVER': 2, 'BIASID': 'b1', 'DARKID': 'd1', 'DARKSCL': 1.0}
    out, ext, cubes, raw_sci, sci, other, flat = _layout(tmp_path, cur, cur)
    frames = {str(raw_sci): ('science', want), f'/raw/{other}.fits': ('science', cur),
              f'/raw/{flat}.fits': ('flat', cur)}
    summary = _redo(tmp_path, frames, out, ext, cubes, raw_sci)
    assert summary['scope'] == 'frames' and summary['stale'][str(raw_sci)][1] == 'dark now available'
    assert summary['moved'] == 6
    left = {p.name for p in out.rglob('*') if p.is_file() and 'stale' not in p.parts}
    assert not any(sci + '_bias' in n for n in left)
    assert f'{sci}.fits' in left and f'{sci}_whitelight.fits' in left    # raw + quick look kept
    assert sum(other in n for n in left) == 6 and 'LLAMAS_red_1_A_traces.pkl' in left
    stale = os.path.join(summary['stale_dir'], 'bias_corrected', f'{sci}_bias_corrected.fits')
    assert os.path.exists(stale)
    # a second pass finds nothing to do
    assert not _redo(tmp_path, frames, out, ext, cubes, raw_sci)['stale']


def test_redo_of_a_calibration_moves_everything(tmp_path):
    old = {'BFVER': 1}
    out, ext, cubes, raw_sci, sci, other, flat = _layout(tmp_path, old, old)
    want = {'BFVER': 2, 'BIASID': 'b1', 'DARKID': 'none', 'DARKSCL': 0.0}
    frames = {str(raw_sci): ('science', want), f'/raw/{other}.fits': ('science', want),
              f'/raw/{flat}.fits': ('flat', want)}
    summary = _redo(tmp_path, frames, out, ext, cubes, raw_sci)
    assert summary['scope'] == 'all'
    assert summary['stale'][f'/raw/{flat}.fits'][1] == 'made by the older bias step'
    left = {p.name for p in out.rglob('*') if p.is_file() and 'stale' not in p.parts}
    assert left == {f'{sci}.fits', f'{sci}_whitelight.fits'}


def test_redo_catches_an_old_rss_under_a_deleted_bias_corrected_folder(tmp_path):
    want = {'BFVER': 2, 'BIASID': 'b1', 'DARKID': 'd1', 'DARKSCL': 1.0}
    out, ext, cubes, raw_sci, sci, other, flat = _layout(tmp_path, want, want)
    _product(ext / f'{sci}_bias_corrected_flat_corrected_RSS_red.fits', {})      # unstamped
    os.remove(out / 'bias_corrected' / f'{sci}_bias_corrected.fits')
    frames = {str(raw_sci): ('science', want)}
    summary = _redo(tmp_path, frames, out, ext, cubes, raw_sci)
    assert summary['stale'][str(raw_sci)][1] == 'made by the older bias step'


def test_redo_never_moves_files_from_this_run(tmp_path):
    cur = {'BFVER': 2, 'BIASID': 'b1', 'DARKID': 'none', 'DARKSCL': 0.0}
    out, ext, cubes, raw_sci, sci, other, flat = _layout(tmp_path, cur, cur)
    frames = {str(raw_sci): ('science', dict(cur, DARKID='d1', DARKSCL=1.0))}
    summary = _redo(tmp_path, frames, out, ext, cubes, raw_sci, since=0.0)
    assert not summary['stale']


def test_interrupted_redo_finishes_next_time(tmp_path):
    cur = {'BFVER': 2, 'BIASID': 'b1', 'DARKID': 'none', 'DARKSCL': 0.0}
    want = dict(cur, DARKID='d1', DARKSCL=1.0)
    out, ext, cubes, raw_sci, sci, other, flat = _layout(tmp_path, cur, cur)
    # an interruption after the later products moved but before the bias-corrected frame
    for p in list(ext.glob(f'{sci}*')) + [out / f'{sci}_bias_corrected_flat_corrected.fits']:
        os.remove(p)
    summary = _redo(tmp_path, {str(raw_sci): ('science', want)}, out, ext, cubes, raw_sci)
    assert summary['stale'] and not (out / 'bias_corrected' / f'{sci}_bias_corrected.fits').exists()


def test_product_names():
    eid = 'LLAMAS_2026-10-16_00-31-08.4'
    assert is_product_of(f'{eid}_SCI24_mef_bias_corrected.fits', eid)
    assert is_product_of(f'{eid}_RSS_red.fits', eid)
    assert is_product_of(f'{eid}_whitelight_fullpipeline.fits', eid)
    assert not is_product_of(f'{eid}_SCI24_mef.fits', eid)
    assert not is_product_of(f'{eid}_SCI24_mef_whitelight.fits', eid)
    assert not is_product_of(f'{eid}5_SCI24_mef_bias_corrected.fits', eid)


def test_provenance_round_trip(night):
    master = build(night)
    out = correct(night, master)
    with fits.open(out) as hdul:
        found = provenance_of(hdul[0].header)
    assert found == expected_provenance(fits.getheader(night.sci, 0), night.bias, None, master, 1.0)


# ---------------------------------------------------------------------------
# Setup GUI
# ---------------------------------------------------------------------------

def test_gui_writes_and_reads_dark_files(tmp_path):
    darks = ['/data/d1_mef.fits', '/data/d2_mef.fits']
    text = generate_config({'science': ['/data/s_mef.fits'], 'dark': darks}, output_dir='/tmp/o',
                           template_path=TEMPLATE)
    cfg = tmp_path / 'cfg.txt'
    cfg.write_text(text)
    config, _bad = parse_config(str(cfg))
    assert config['dark_files'] == ', '.join(darks)
    old = tmp_path / 'old.txt'                              # a template without the key
    old.write_text('science_files = /x.fits\n')
    text = generate_config({'science': ['/x.fits'], 'dark': darks}, output_dir='/tmp/o',
                           template_path=str(old))
    cfg.write_text(text)
    assert parse_config(str(cfg))[0]['dark_files'] == ', '.join(darks)


def test_gui_assigns_darks_without_error():
    fake = types.SimpleNamespace(roles={'/d1.fits': {'dark'}, '/s.fits': {'science'}})
    a = ReduxSetupWindow.current_assignments(fake)
    assert a['dark'] == ['/d1.fits'] and a['science'] == ['/s.fits']


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
