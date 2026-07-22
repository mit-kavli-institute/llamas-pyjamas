"""Unit + integration tests for applying a sensitivity function (Phase III).

The unit tests use a synthetic single-channel RSS with a constant sensitivity so FLAM has a
known value, and check the extension writing, idempotency, the SKYSUB/FLUX fallback, and the
differential-extinction factor. The integration test does a real self-application on GD108 and
checks the published flux is recovered; it is skipped when the may26 data is absent.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.Tests.test_flux_calibrate`).
"""

import os

import numpy as np
from astropy.io import fits

from llamas_pyjamas.Flux.fluxCalibrate import (
    apply_sensfunc,
    skysub_extname,
    _differential_extinction,
)
from llamas_pyjamas.Flux.sensFunc import SensChannel, SensFunc


def _synthetic_rss(channel='green', skysub_name='SKYSUB', exptime=10.0, airmass=1.2,
                   nfib=5, nwave=100, counts=200.0):
    wave = np.tile(np.linspace(5000.0, 6000.0, nwave), (nfib, 1))
    primary = fits.PrimaryHDU()
    primary.header['CHANNEL'] = channel
    primary.header['SEXPTIME'] = exptime
    primary.header['AIRMASS'] = airmass
    hdus = [primary,
            fits.ImageHDU(np.full((nfib, nwave), counts, dtype=float), name=skysub_name),
            fits.ImageHDU(np.full((nfib, nwave), 10.0, dtype=float), name='ERROR'),
            fits.ImageHDU(wave, name='WAVE')]
    return fits.HDUList(hdus)


def _flat_sensfunc(channel='green', s_value=5e-17, airmass=1.2):
    wave = np.linspace(4500.0, 6500.0, 50)
    ch = SensChannel(channel=channel, wave=wave, sens=np.full_like(wave, s_value),
                     raw=np.full_like(wave, s_value), good=np.ones_like(wave, dtype=bool))
    return SensFunc(channels={channel: ch}, meta={'standard': 'TEST', 'airmass': airmass})


def test_skysub_extname_prefers_skysub_then_flux():
    assert skysub_extname(_synthetic_rss(skysub_name='SKYSUB')) == 'SKYSUB'
    assert skysub_extname(_synthetic_rss(skysub_name='FLUX')) == 'FLUX'


def test_skysub_extname_raises_when_neither():
    hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((2, 2)), name='COUNTS')])
    try:
        skysub_extname(hdul)
    except KeyError:
        return
    raise AssertionError('expected KeyError when no sky-subtracted plane')


def test_apply_adds_flam_with_expected_value_no_extinction():
    # Equal airmass (sci == std) => extinction factor 1, so FLAM = counts/exptime * S exactly.
    hdul = _synthetic_rss(exptime=10.0, airmass=1.2, counts=200.0)
    sf = _flat_sensfunc(s_value=5e-17, airmass=1.2)
    apply_sensfunc(hdul, sf)
    assert 'FLAM' in {h.name for h in hdul} and 'FLAM_ERR' in {h.name for h in hdul}
    flam = hdul['FLAM'].data
    # 200/10 * 5e-17 = 1e-15
    assert np.allclose(flam, 1e-15, rtol=1e-4)
    assert hdul['FLAM'].header['BUNIT'] == 'erg/s/cm2/Angstrom'
    assert hdul['FLAM'].header['FLUXCAL'] is True


def test_apply_is_idempotent():
    hdul = _synthetic_rss()
    sf = _flat_sensfunc()
    apply_sensfunc(hdul, sf)
    apply_sensfunc(hdul, sf)
    assert sum(1 for h in hdul if h.name == 'FLAM') == 1, 'FLAM must be replaced, not duplicated'
    assert sum(1 for h in hdul if h.name == 'FLAM_ERR') == 1


def test_apply_works_on_flux_named_plane():
    # Forward/backward compat: a file still using FLUX must calibrate the same way.
    hdul = _synthetic_rss(skysub_name='FLUX', exptime=10.0, airmass=1.2, counts=200.0)
    apply_sensfunc(hdul, _flat_sensfunc(s_value=5e-17, airmass=1.2))
    assert np.allclose(hdul['FLAM'].data, 1e-15, rtol=1e-4)


def test_apply_raises_for_missing_channel():
    hdul = _synthetic_rss(channel='red')
    try:
        apply_sensfunc(hdul, _flat_sensfunc(channel='green'))
    except ValueError:
        return
    raise AssertionError('expected ValueError when sensfunc lacks the channel')


def test_differential_extinction_unity_without_standard_airmass():
    wave = np.linspace(4000.0, 9000.0, 50)
    e = _differential_extinction(wave, x_sci=1.5, x_std=None, extinct=None)
    assert np.allclose(e, 1.0)


def test_differential_extinction_direction_and_magnitude():
    from llamas_pyjamas.Flux.fluxCalibrate import load_lco_extinction
    ext = load_lco_extinction()
    wave = np.array([4000.0, 7000.0])
    # science at higher airmass than standard => must scale flux UP (E > 1), more so in the blue
    e = _differential_extinction(wave, x_sci=2.0, x_std=1.0, extinct=ext)
    assert (e > 1.0).all()
    assert e[0] > e[1], 'blue extinction correction exceeds red'
    # equal airmass => unity
    assert np.allclose(_differential_extinction(wave, 1.3, 1.3, ext), 1.0)


def test_extinction_applied_flag_reflects_state():
    hdul = _synthetic_rss(airmass=1.5)
    sf = _flat_sensfunc(airmass=1.1)
    apply_sensfunc(hdul, sf, apply_extinction=True)
    assert hdul['FLAM'].header['EXTCORR'] is True
    hdul2 = _synthetic_rss(airmass=1.5)
    sf2 = SensFunc(channels=sf.channels, meta={'standard': 'T'})   # no airmass
    apply_sensfunc(hdul2, sf2, apply_extinction=True)
    assert hdul2['FLAM'].header['EXTCORR'] is False, 'no std airmass => extinction skipped'


# --- integration on real data ---------------------------------------------------------------

_GD108 = os.path.expanduser(
    '~/data/LLAMAS/may26/ut20260516_17/reduced/extractions/'
    'LLAMAS_2026-05-16_23-17-22.7_SCI22_mef_bias_corrected_flat_corrected_extract_RSS_green_FF.fits')


def test_real_self_application_recovers_published_flux():
    if not os.path.exists(_GD108):
        print('SKIP real flux-cal test (data absent)')
        return
    from llamas_pyjamas.CubeViewer.cubeViewRSS import RSSScene
    from llamas_pyjamas.CubeViewer.cubeViewScene import combine
    from llamas_pyjamas.Flux.fluxStandards import load_catalog
    from llamas_pyjamas.Flux.sensFunc import build_sensfunc

    scene = RSSScene.open(_GD108)
    hdr = fits.getheader(_GD108, 0)
    exptime = float(hdr.get('SEXPTIME', hdr.get('REXPTIME')))
    airmass = float(hdr.get('AIRMASS', hdr.get('TEL AIRMASS')))
    std = load_catalog().match_header(hdr).standard
    ref_w, ref_f = std.load_spectrum()

    img, _, _ = scene.collapse(5000, 6500, channels=['green'])
    pk = np.unravel_index(np.nanargmax(img), img.shape)
    members = scene.elements_within(pk[1] + 1, pk[0] + 1, 2 * scene.pitch / scene._step())
    spectra = {s.channel: s.good() for s in combine(scene.spectra_of(members), mode='sum')}
    sf = build_sensfunc(spectra, exptime, ref_w, ref_f, airmass=airmass,
                        meta={'standard': std.name})

    # apply the sensfunc back to the same aperture spectrum
    wave, counts = spectra['green']
    s_grid = sf.channels['green'].value(wave)
    flam = (counts / exptime) * s_grid                 # self => equal airmass, E == 1
    ref = np.interp(wave, ref_w, ref_f, left=np.nan, right=np.nan)
    ratio = flam / ref
    ok = np.isfinite(ratio) & (ratio > 0)
    assert abs(np.nanmedian(ratio[ok]) - 1.0) < 0.1, \
        f'self-application should recover published flux, got {np.nanmedian(ratio[ok]):.3f}'


if __name__ == '__main__':
    import sys
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f'PASS {name}')
        except Exception as e:                       # noqa: BLE001
            failed += 1
            print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
