"""A masked sample never counts as zero in a total.

Pins Utils/maskedSums (complete_sum, band_integral, renormalised_mean, resample_bad) and the
consumers rewired onto it and onto the shared bad_pixel_mask: the viewer aperture sum that feeds
the sensitivity function, transparency photometry, registration / viewer fibre flux, cube
resampling and coverage, and the zero-count flag written at RSS generation.
"""

import os
import shutil
import tempfile
from types import SimpleNamespace

import cloudpickle
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.io import fits

from llamas_pyjamas.Utils.maskedSums import (band_integral, common_coverage, complete_sum,
                                             renormalised_mean, resample_bad)
from llamas_pyjamas.File.llamasRSS import (MASK_FLAT_LAMP_ONLY, MASK_NO_COUNTS,
                                           MASK_NOT_FLAT_FIELDED, bad_pixel_mask)
from llamas_pyjamas.Tests.rss_fixtures import write_rss


@pytest.fixture
def tmp():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- the rules
def test_complete_sum_masks_instead_of_undercounting():
    vals = np.array([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    bad = np.array([[False, True, False, False], [False, False, False, False]])
    total, mask = complete_sum(vals, bad)
    assert list(mask) == [False, True, False, False]
    assert total[0] == 3.0 and np.isnan(total[1])          # never 2.0 (a partial total)
    total_w, _ = complete_sum(vals, bad, weights=[2.0, 1.0])
    assert total_w[0] == 4.0


def test_renormalised_mean_uses_valid_weight_only():
    vals = np.array([[1.0, 1.0], [3.0, 3.0]])
    bad = np.array([[False, True], [False, False]])
    mean, mask = renormalised_mean(vals, bad)
    assert mean[0] == 2.0 and mean[1] == 3.0 and not mask.any()


def test_band_integral_unbiased_by_masked_pixels():
    wave = np.linspace(5000, 6000, 101)
    flux = np.full(101, 2.0)
    bad = np.zeros(101, bool)
    bad[::3] = True                                        # ~1/3 masked
    full = band_integral(flux, wave, np.zeros(101, bool), 5000, 6000, var=np.ones(101))
    part = band_integral(flux, wave, bad, 5000, 6000, var=np.ones(101))
    assert np.isclose(part.value[0], full.value[0])        # nansum would give ~2/3 of it
    n, big_n = part.n[0], 101
    assert np.isclose(part.var[0], (big_n / n) ** 2 * n)
    mostly = np.ones(101, bool)
    mostly[:30] = False                                    # 30% valid < min_frac
    assert np.isnan(band_integral(flux, wave, mostly, 5000, 6000).value[0])


def test_band_integral_clips_to_common_coverage():
    wave = np.tile(np.linspace(5000, 6000, 101), (3, 1))
    flux = np.ones((3, 101))
    bad = np.zeros((3, 101), bool)
    bad[1, 90:] = True                                     # one fibre's coverage ends early
    bad[2, 95:] = True
    common = common_coverage(wave, ~bad, lo_pct=0, hi_pct=100)
    assert common == (5000.0, 5890.0)
    bi = band_integral(flux, wave, bad, 5000, 6000, common=common)
    assert np.all(bi.value == 90.0)                        # every fibre over the same window


def test_resample_bad_flags_neighbours_and_range():
    src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    bad = np.array([False, False, True, False, False])
    ref = np.array([0.5, 1.5, 2.5, 3.0, 3.5, 4.5, 5.5])
    assert list(resample_bad(ref, src, bad)) == [True, False, True, True, True, False, True]


# ---------------------------------------------------------------- shared bad-pixel rule
def test_bad_pixel_mask_rule(tmp):
    p = write_rss(os.path.join(tmp, 'x_RSS_green.fits'),
                  mask_bits={int(MASK_FLAT_LAMP_ONLY): (slice(0, 2), slice(None)),
                             int(MASK_NOT_FLAT_FIELDED): (slice(2, 3), slice(None))},
                  zero_counts=(slice(3, 4), slice(0, 10)))
    with fits.open(p) as h:
        bad = bad_pixel_mask(h)
    assert not bad[0:2].any()                              # informational bit: usable
    assert bad[2].all()                                    # not flat-fielded: bad
    assert bad[3, :10].all() and not bad[3, 10:].any()     # zero-count fill: bad (legacy too)


def test_readers_agree_on_legacy_zero_count_pixels(tmp):
    """A legacy file (MASK all zero) with zero-filled pixels: every reader masks the same."""
    from llamas_pyjamas.Combine.superRSS import _read_channel
    from llamas_pyjamas.CubeViewer.cubeViewRSS import RSSChannel
    from llamas_pyjamas.Cube.simple_cube_constructor import SimpleCubeConstructor
    p = write_rss(os.path.join(tmp, 'x_RSS_green.fits'), zero_counts=(slice(4, 7), slice(100, 160)))
    with fits.open(p) as h:
        expect = bad_pixel_mask(h)
        _, _, _, m_super, _, _, _ = _read_channel(h, 'SKYSUB', 'ERROR')
    assert expect[4:7, 100:160].all() and expect.sum() == 3 * 60
    assert np.array_equal(m_super, expect)
    assert np.array_equal(RSSChannel(p).mask, expect)
    scc = SimpleCubeConstructor()
    scc.load_rss_file(p)
    assert np.array_equal(np.asarray(scc.mask, bool), expect)


def test_generate_rss_flags_zero_counts(tmp):
    from llamas_pyjamas.File.llamasRSS import RSSgeneration
    n, nw = 6, 50
    counts = np.full((n, nw), 10.0)
    counts[2, :20] = 0.0                                   # extraction zero-fill
    obj = SimpleNamespace(counts=counts, wave=np.tile(np.linspace(5000, 6000, nw), (n, 1)),
                          errors=np.ones((n, nw)), sky=np.full((n, nw), 1.0), dead_fibers=[],
                          relative_throughput=np.ones(n))
    hdr = fits.Header()
    hdr['OBS TYPE'] = 'CAL24'                              # no heliocentric scaling
    pkl = os.path.join(tmp, 'x.pkl')
    with open(pkl, 'wb') as fh:
        cloudpickle.dump({'primary_header': hdr, 'extractions': [obj],
                          'metadata': [{'channel': 'green', 'bench': '1', 'side': 'A'}]}, fh)
    out = os.path.join(tmp, 'x_RSS.fits')
    RSSgeneration().generate_rss(pkl, out)
    with fits.open(out) as h:
        m = h['MASK'].data
        assert np.all(m[2, :20] & MASK_NO_COUNTS) and not np.any(m[2, 20:] & MASK_NO_COUNTS)
        assert not np.any(m[[0, 1, 3, 4, 5]] & MASK_NO_COUNTS)
        assert h[0].header['RSSSTAGE'] == 'base'


# ---------------------------------------------------------------- sensfunc aperture
def _aperture(gap=(5500.0, 5700.0), gap_member=3, full_bad_member=None):
    from llamas_pyjamas.CubeViewer.cubeViewScene import Spectrum
    shares = np.array([0.4, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05])
    s_true = lambda w: 1e-16 * (1.0 + 0.3 * np.sin((w - 4700.0) / 400.0))  # noqa: E731
    f_ref = 1e-14
    exptime = 30.0
    members = []
    for i, share in enumerate(shares):
        w = np.linspace(4700.0, 7000.0, 1200) + (0.0 if i == 0 else 0.7 * (i - 3))
        counts = share * f_ref / s_true(w) * exptime
        mask = np.zeros(w.size, bool)
        if i == gap_member:
            in_gap = (w >= gap[0]) & (w <= gap[1])
            counts = np.where(in_gap, 0.0, counts)         # zero-filled span of one fibre
            mask |= in_gap
        if i == full_bad_member:
            mask[:] = True
        members.append(Spectrum(wave=w, flux=counts, channel='green', label=f'fibre {i}',
                                mask=mask))
    return members, s_true, f_ref, exptime


def test_aperture_sum_is_complete_coverage():
    from llamas_pyjamas.CubeViewer.cubeViewScene import combine
    members, s_true, f_ref, exptime = _aperture()
    (spec,) = combine(members, mode='sum')
    w = spec.wave
    in_gap = (w >= 5490) & (w <= 5710)
    assert spec.mask[(w > 5520) & (w < 5680)].all()        # the gap is masked, not summed low
    ok = ~spec.mask
    expect = f_ref / s_true(w) * exptime
    assert np.allclose(spec.flux[ok], expect[ok], rtol=2e-3)
    assert not spec.mask[~in_gap & (w > 4710) & (w < 6990)].any()
    gw, gf = spec.good()
    assert not ((gw > 5520) & (gw < 5680)).any()


def test_sensfunc_from_aperture_is_unbiased():
    from llamas_pyjamas.CubeViewer.cubeViewScene import combine
    from llamas_pyjamas.Flux.sensFunc import build_sensfunc
    members, s_true, f_ref, exptime = _aperture()
    (spec,) = combine(members, mode='sum')
    ref_w = np.linspace(4000, 8000, 4001)
    sf = build_sensfunc({'green': spec.good()}, exptime, ref_w, np.full(ref_w.size, f_ref),
                        regions=[], refine_regions=[], weighted=False)
    w = np.linspace(4900, 6800, 400)
    ratio = sf.channels['green'].value(w) / s_true(w)
    assert np.nanmax(np.abs(ratio - 1.0)) < 0.01            # a 15% partial-sum dip is gone


def test_fully_bad_member_is_named():
    from llamas_pyjamas.CubeViewer.cubeViewScene import combine
    members, *_ = _aperture(full_bad_member=5)
    (spec,) = combine(members, mode='sum')
    assert spec.empty_members == ('fibre 5',)
    assert spec.mask.all()


# ---------------------------------------------------------------- transparency
def _sr_two_exposures(mask_fibre_in_exp1=False, n_src=1):
    from llamas_pyjamas.Combine.superRSS import ChannelStack, ExposureMeta, SuperRSS
    rows_ra, rows_ex, flux = [], [], []
    srcs = [150.0, 150.01][:n_src]
    for e in (0, 1):
        for ra0 in srcs:
            for k in range(3):                             # 3 fibres on each source
                rows_ra.append(ra0 + k * 0.5 / 3600.0)
                rows_ex.append(e)
                flux.append(np.full(20, 100.0))
    flux = np.array(flux)
    mask = np.zeros(flux.shape, bool)
    if mask_fibre_in_exp1:
        first_exp1 = 3 * len(srcs)
        mask[first_exp1, :] = True                         # one aperture fibre of source 0 lost
    n = flux.shape[0]
    st = ChannelStack('green', np.array(rows_ra), np.full(n, 2.0),
                      np.tile(np.linspace(5000, 5100, 20), (n, 1)), flux, np.ones(flux.shape),
                      mask, np.full(n, 0.44), np.array(rows_ex))
    exps = [ExposureMeta('e0', 'p0', 100.0, 1.0, 1.0), ExposureMeta('e1', 'p1', 100.0, 1.0, 2.0)]
    return SuperRSS('T', 'skysub', 'counts', exps, {'green': st}), srcs


def test_transparency_drops_incomplete_source_instead_of_scaling_up():
    from llamas_pyjamas.Combine.transparency import transparency_scales
    sr, srcs = _sr_two_exposures(mask_fibre_in_exp1=True, n_src=2)
    sources = [SkyCoord(r * u.deg, 2.0 * u.deg) for r in srcs]
    scales = transparency_scales(sr, sources=sources, radius_arcsec=2.0, channels=['green'])
    # a partial sum (2 of 3 fibres of source 0 in e1) would scale e1 up by 1.5
    assert np.isclose(scales['e0'], 1.0) and np.isclose(scales['e1'], 1.0)


def test_transparency_raises_when_an_exposure_is_unmeasurable():
    from llamas_pyjamas.Combine.transparency import TransparencyError, transparency_scales
    sr, srcs = _sr_two_exposures(mask_fibre_in_exp1=True, n_src=1)
    with pytest.raises(TransparencyError):
        transparency_scales(sr, sources=[SkyCoord(srcs[0] * u.deg, 2.0 * u.deg)],
                            radius_arcsec=2.0, channels=['green'])


# ---------------------------------------------------------------- registration vs viewer
def test_registration_and_viewer_fibre_flux_agree(tmp):
    from llamas_pyjamas.Utils.register import per_fibre_flux
    from llamas_pyjamas.CubeViewer.cubeViewRSS import RSSChannel
    p = write_rss(os.path.join(tmp, 'x_RSS_green.fits'),
                  zero_counts=(slice(0, 3), slice(150, 250)),
                  mask_bits={int(MASK_NOT_FLAT_FIELDED): (slice(9, 10), slice(None))})
    with fits.open(p) as h:
        reg = per_fibre_flux(h, band=(5000, 6500))
    view = RSSChannel(p).collapse_fibres(5000, 6500)
    np.testing.assert_allclose(reg, view, equal_nan=True)
    assert np.isnan(reg[9])                                 # a bad fibre is no data, not 0
    # the zero-filled bright fibres are not read faint
    assert abs(reg[0] / reg[3] - 1.0) < 0.02


# ---------------------------------------------------------------- cube
def _stack_for_cube(mask):
    from llamas_pyjamas.Combine.superRSS import ChannelStack
    n, nw = mask.shape
    return ChannelStack('green', np.full(n, 150.0), np.full(n, 2.0),
                        np.tile(np.linspace(5000, 5100, nw), (n, 1)), np.ones((n, nw)),
                        np.ones((n, nw)), mask, np.full(n, 0.44), np.zeros(n, int))


def test_resample_bridges_one_pixel_not_wide_gaps():
    from llamas_pyjamas.Combine.cube import _resample_fibres
    mask = np.zeros((2, 101), bool)
    mask[0, 50] = True                                     # one-pixel gap
    mask[1, 40:45] = True                                  # five-pixel gap
    st = _stack_for_cube(mask)
    wl = np.linspace(5000, 5100, 101)
    _, _, bad = _resample_fibres(st, wl, 'flux')
    assert not bad[0].any()
    assert bad[1, 40:45].all() and not bad[1, :38].any() and not bad[1, 47:].any()


def test_cube_coverage_ignores_fibres_without_data():
    from llamas_pyjamas.Combine.cube import combine_cube
    from llamas_pyjamas.Combine.superRSS import ExposureMeta, SuperRSS
    mask = np.zeros((2, 50), bool)
    mask[1, :] = True                                      # second fibre: no data at all
    st = _stack_for_cube(mask)
    sr = SuperRSS('T', 'skysub', 'counts', [ExposureMeta('e0', 'p0', 1.0, 1.0, 1.0)],
                  {'green': st})
    cube = combine_cube(sr, 'green')
    assert cube.coverage.max() == 1                        # not 2: the empty fibre must not count
