"""Unit + integration tests for the sensitivity-function core.

The unit tests use synthetic data with a known answer: the key property is that masking a
stellar-line region lets the fit recover the smooth continuum response *through* the line
rather than diving into it. The integration test builds a real sensfunc from the bundled
GD108 extraction if the may26 data is present, and is skipped otherwise.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.Tests.test_sensfunc`).
"""

import os

import numpy as np

from llamas_pyjamas.Flux.sensFunc import (
    SensChannel,
    SensFunc,
    build_good_mask,
    build_sensfunc,
    default_masks,
    fit_sensitivity,
    sensitivity_ratio,
)


def test_default_masks_are_stellar_only():
    # Stellar lines are hard-masked by default; tellurics are handled by S/N weighting so they
    # are NOT hard-masked unless explicitly requested (hard-masking the red band truncates
    # coverage).
    regions = default_masks()
    def masked(w, regs=regions):
        return any(lo <= w <= hi for lo, hi in regs)
    assert masked(6562.8), 'H-alpha must be masked'
    assert masked(4861.3), 'H-beta must be masked'
    assert not masked(7620.0), 'O2 A-band must NOT be masked by default'
    assert not masked(5500.0), 'clean continuum must not be masked'
    # opt-in tellurics
    with_tell = default_masks(include_telluric=True)
    assert masked(7620.0, with_tell), 'A-band masked when include_telluric=True'


def test_build_good_mask_excludes_regions():
    wave = np.arange(4000.0, 5000.0, 1.0)
    good = build_good_mask(wave, [(4400.0, 4500.0)])
    assert not good[(wave >= 4400) & (wave <= 4500)].any()
    assert good[(wave < 4400) | (wave > 4500)].all()


def test_sensitivity_ratio_basic():
    obs_wave = np.linspace(5000, 6000, 101)
    obs_flux = np.full_like(obs_wave, 200.0)       # counts
    ref_wave = np.linspace(4000, 7000, 301)
    ref_flux = np.full_like(ref_wave, 1e-15)       # erg/s/cm2/A
    wave, sens, valid = sensitivity_ratio(obs_wave, obs_flux, exptime=10.0,
                                          ref_wave=ref_wave, ref_flux=ref_flux)
    # S = F_ref / (counts/s) = 1e-15 / (200/10) = 1e-15 / 20 = 5e-17
    assert valid.all()
    assert np.allclose(sens, 5e-17)


def test_sensitivity_ratio_flags_zero_and_out_of_range():
    obs_wave = np.linspace(3000, 8000, 201)        # extends beyond the reference
    obs_flux = np.full_like(obs_wave, 100.0)
    obs_flux[50] = 0.0                              # a zero-count sample
    ref_wave = np.linspace(4000, 7000, 301)
    ref_flux = np.full_like(ref_wave, 1e-15)
    wave, sens, valid = sensitivity_ratio(obs_wave, obs_flux, 10.0, ref_wave, ref_flux)
    assert not valid[obs_wave < 4000].any() and not valid[obs_wave > 7000].any()
    assert not valid[50]
    assert np.isnan(sens[50])


def test_exptime_must_be_positive():
    try:
        sensitivity_ratio(np.array([5000.]), np.array([1.]), 0.0,
                          np.array([5000.]), np.array([1e-15]))
    except ValueError:
        return
    raise AssertionError('expected ValueError for exptime <= 0')


def _synthetic_ratio():
    """A smooth log-linear S(lambda) with a sharp absorption-like dip at 5000-5100 A."""
    wave = np.arange(4000.0, 7000.0, 2.0)
    true = 10.0 ** (-16.0 + 0.3 * (wave - 4000.0) / 3000.0)   # smooth, ~1e-16
    sens = true.copy()
    dip = (wave >= 5000) & (wave <= 5100)
    sens[dip] *= 0.3                                          # a "stellar line"
    return wave, sens, true, dip


def test_fit_recovers_continuum_through_masked_line():
    wave, sens, true, dip = _synthetic_ratio()
    good = build_good_mask(wave, [(4990.0, 5110.0)])          # mask the dip
    fit, used = fit_sensitivity(wave, sens, good, bkspace=200.0)
    at_line = np.argmin(np.abs(wave - 5050.0))
    # the fit at the line centre should track the true continuum, not the 0.3x dip
    assert abs(fit[at_line] - true[at_line]) / true[at_line] < 0.1, \
        'fit should ignore the masked line and follow the continuum'


def test_fit_without_mask_is_dragged_by_line():
    # Control: without masking, the fit is pulled toward the dip -- shows the mask matters.
    wave, sens, true, dip = _synthetic_ratio()
    good = np.isfinite(sens)
    fit, _ = fit_sensitivity(wave, sens, good, bkspace=200.0)
    at_line = np.argmin(np.abs(wave - 5050.0))
    assert fit[at_line] < true[at_line] * 0.95, 'unmasked fit should be dragged below continuum'


def test_throughput_floor_drops_faint_edges():
    # Counts high in the middle, collapsing to ~1% at both ends (dichroic rolloff). The floor
    # should drop those ends so the fit is not defined out in the noise.
    from llamas_pyjamas.Flux.sensFunc import build_sensfunc
    wave = np.linspace(4600.0, 7000.0, 400)
    peak = 5000.0
    counts = peak * np.exp(-0.5 * ((wave - 5800.0) / 350.0) ** 2)   # bell; ~0 at the ends
    ref_wave = np.linspace(4000.0, 7500.0, 400)
    ref_flux = np.full_like(ref_wave, 1e-15)
    sf = build_sensfunc({'green': (wave, counts)}, exptime=10.0,
                        ref_wave=ref_wave, ref_flux=ref_flux, regions=[],
                        throughput_floor=0.05, meta={'standard': 'T'})
    ch = sf.channels['green']
    defined = np.isfinite(ch.sens)
    # the faint ends (well below 5% of peak) must be undefined; the bright middle defined
    assert defined[np.argmin(np.abs(wave - 5800.0))]
    assert not defined[0] and not defined[-1]


def test_weighting_mitigates_low_count_spike():
    # A low-count dip makes S spike up. With S/N weighting the unmasked fit should follow the
    # continuum more closely than without it — the mechanism that stabilises real edges.
    from llamas_pyjamas.Flux.sensFunc import fit_channel_sens
    wave = np.linspace(4600.0, 7000.0, 400)
    counts = np.full_like(wave, 400.0)
    dip = (wave >= 5500) & (wave <= 5600)
    counts[dip] = 40.0
    ref_wave = np.linspace(4000.0, 7500.0, 400)
    ref_flux = np.full_like(ref_wave, 1e-15)
    at = np.argmin(np.abs(wave - 5550.0))

    _, _, fit_w, _ = fit_channel_sens(wave, counts, 10.0, ref_wave, ref_flux, regions=[],
                                      bkspace=150.0, throughput_floor=0.0, weighted=True)
    _, _, fit_u, _ = fit_channel_sens(wave, counts, 10.0, ref_wave, ref_flux, regions=[],
                                      bkspace=150.0, throughput_floor=0.0, weighted=False)
    continuum = 1e-15 / (400.0 / 10.0)      # true S off the dip
    assert abs(fit_w[at] - continuum) < abs(fit_u[at] - continuum), \
        'S/N weighting should pull the fit at the spike closer to the continuum'


def test_build_breakpoints_densifies_refine_region():
    from llamas_pyjamas.Flux.sensFunc import build_breakpoints
    assert build_breakpoints(4000, 5000, 100.0, []) is None      # no refine -> uniform bkspace
    bk = build_breakpoints(4000.0, 5000.0, 100.0, [(4400.0, 4600.0)], refine_factor=4)
    in_band = np.diff(bk[(bk >= 4400) & (bk <= 4600)])
    out_band = np.diff(bk[(bk < 4400) | (bk > 4600)])
    assert np.median(in_band) < 40.0, 'knots inside the refine region are ~base/4'
    assert np.median(out_band) > 80.0, 'knots outside stay at the base spacing'


def test_refine_regions_roundtrip(tmp_path):
    from llamas_pyjamas.Flux.sensFunc import save_refine_regions, load_refine_regions
    path = os.path.join(str(tmp_path), 'bk.dat')
    save_refine_regions([(4250.0, 4650.0), (7000.0, 7100.0)], path=path)
    assert load_refine_regions(path, use_cache=False) == [(4250.0, 4650.0), (7000.0, 7100.0)]
    assert load_refine_regions(os.path.join(str(tmp_path), 'absent.dat'), use_cache=False) == []


def test_refine_region_changes_the_fit():
    # A localised wiggle the base spacing smooths over; a refine region there should let the
    # fit follow it, changing the fitted value at the feature.
    from llamas_pyjamas.Flux.sensFunc import fit_channel_sens
    wave = np.linspace(4000.0, 5600.0, 800)
    counts = np.full_like(wave, 400.0)
    bump = np.exp(-0.5 * ((wave - 4450.0) / 25.0) ** 2)     # narrow throughput bump
    counts *= (1 + 0.4 * bump)
    ref_wave = np.linspace(3500.0, 6000.0, 400)
    ref_flux = np.full_like(ref_wave, 1e-15)
    at = np.argmin(np.abs(wave - 4450.0))
    _, _, coarse, _ = fit_channel_sens(wave, counts, 10.0, ref_wave, ref_flux, regions=[],
                                       bkspace=200.0, throughput_floor=0.0, refine_regions=[])
    _, _, fine, _ = fit_channel_sens(wave, counts, 10.0, ref_wave, ref_flux, regions=[],
                                     bkspace=200.0, throughput_floor=0.0,
                                     refine_regions=[(4350.0, 4550.0)])
    true = 1e-15 / (400.0 * (1 + 0.4) / 10.0)               # S at the bump peak
    assert abs(fine[at] - true) < abs(coarse[at] - true), \
        'a refine region should track the local feature better than base spacing'


def test_build_and_roundtrip(tmp_path):
    wave = np.arange(4000.0, 7000.0, 2.0)
    obs_flux = np.full_like(wave, 500.0)
    ref_wave = np.arange(3500.0, 7500.0, 5.0)
    ref_flux = np.full_like(ref_wave, 2e-15)
    sf = build_sensfunc({'green': (wave, obs_flux)}, exptime=20.0,
                        ref_wave=ref_wave, ref_flux=ref_flux,
                        regions=[], meta={'standard': 'TEST'})
    assert 'green' in sf.channels
    path = os.path.join(str(tmp_path), 'sens.fits')
    sf.save(path)
    loaded = SensFunc.load(path)
    probe = np.array([4500.0, 5500.0, 6500.0])
    assert np.allclose(sf.value(probe, 'green'), loaded.value(probe, 'green'), equal_nan=True)
    assert loaded.meta.get('standard') == 'TEST'


# --- integration on real data, skipped when absent ------------------------------------------

_GD108 = os.path.expanduser(
    '~/data/LLAMAS/may26/ut20260516_17/reduced/extractions/'
    'LLAMAS_2026-05-16_23-17-22.7_SCI22_mef_bias_corrected_flat_corrected_extract_RSS_green_FF.fits')


def test_real_gd108_sensfunc_is_smooth():
    if not os.path.exists(_GD108):
        print('SKIP real GD108 test (data absent)')
        return
    from astropy.io import fits
    from llamas_pyjamas.CubeViewer.cubeViewRSS import RSSScene
    from llamas_pyjamas.CubeViewer.cubeViewScene import combine
    from llamas_pyjamas.Flux.fluxStandards import load_catalog

    scene = RSSScene.open(_GD108)
    hdr = fits.getheader(_GD108, 0)
    exptime = float(hdr.get('SEXPTIME', hdr.get('REXPTIME')))
    std = load_catalog().match_header(hdr).standard
    ref_w, ref_f = std.load_spectrum()

    img, _, _ = scene.collapse(5000, 6500, channels=['green'])
    pk = np.unravel_index(np.nanargmax(img), img.shape)
    members = scene.elements_within(pk[1] + 1, pk[0] + 1, 2 * scene.pitch / scene._step())
    spectra = {s.channel: s.good() for s in combine(scene.spectra_of(members), mode='sum')}

    sf = build_sensfunc(spectra, exptime, ref_w, ref_f, meta={'standard': std.name})
    assert 'green' in sf.channels
    c = sf.channels['green']
    ok = np.isfinite(c.sens)
    # smooth: consecutive log steps are tiny
    roughness = np.median(np.abs(np.diff(np.log(c.sens[ok]))))
    assert roughness < 0.01, f'green sensfunc should be smooth, roughness={roughness}'


if __name__ == '__main__':
    import sys
    import tempfile
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            if 'tmp_path' in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                with tempfile.TemporaryDirectory() as td:
                    fn(td)
            else:
                fn()
            print(f'PASS {name}')
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
