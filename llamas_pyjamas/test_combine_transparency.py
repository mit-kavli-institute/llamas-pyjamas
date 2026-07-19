"""Tests for transparency scaling (Combine/transparency.py) + SuperRSS.apply_scales.

Synthetic super-RSS with a bright reference source seen at different throughputs in two exposures;
checks that the per-exposure scales bring the source flux to a common level, that variance scales
by scale^2, and that apply_scales is applied as a ratio (composes/idempotent). find_reference_sources
and the peak-finder are validated on real may26 data.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_combine_transparency`).
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from llamas_pyjamas.Combine.superRSS import ChannelStack, SuperRSS, ExposureMeta
from llamas_pyjamas.Combine.transparency import transparency_scales, measure_reference_flux

RAS, DECS = 150.0, 20.0            # source position
FAR = 0.02                          # deg (~72") -> outside a 2" aperture


def _two_exposure_super(src_flux0, src_flux1):
    # rows: [exp0 source, exp0 sky, exp1 source, exp1 sky]; nw=1 so collapse_band == the value
    wave = np.full((4, 1), 5000.0)
    flux = np.array([[src_flux0], [1.0], [src_flux1], [1.0]])
    var = np.ones((4, 1))
    st = ChannelStack(
        channel='green',
        ra=np.array([RAS, RAS + FAR, RAS, RAS + FAR]),
        dec=np.array([DECS, DECS, DECS, DECS]),
        wave=wave, flux=flux, var=var, mask=np.zeros((4, 1), bool),
        solid_angle=np.full(4, 0.5), exposure=np.array([0, 0, 1, 1]))
    exps = [ExposureMeta('exp0', 'p0', 100.0, 1.0, 1.0), ExposureMeta('exp1', 'p1', 100.0, 1.0, 2.0)]
    for e in exps:
        e.scale = 1.0
    return SuperRSS(field='T', plane='skysub', bunit='counts', exposures=exps,
                    channels={'green': st})


def test_scales_equalize_source_flux():
    sr = _two_exposure_super(100.0, 50.0)              # exp1 at half throughput
    src = SkyCoord(RAS * u.deg, DECS * u.deg)
    scales = transparency_scales(sr, sources=[src], radius_arcsec=2.0, channels=['green'],
                                 band=(4999, 5001), reference='median')
    # ref = median(100,50) = 75 -> scale0=0.75, scale1=1.5, ratio 2
    assert np.isclose(scales['exp0'], 0.75) and np.isclose(scales['exp1'], 1.5)
    assert np.isclose(scales['exp1'] / scales['exp0'], 2.0)


def test_apply_scales_scales_flux_and_variance():
    sr = _two_exposure_super(100.0, 50.0)
    sr.apply_scales({'exp0': 0.75, 'exp1': 1.5})
    g = sr.channels['green']
    # source fibres (rows 0 and 2) now equal at 75
    assert np.isclose(g.flux[0, 0], 75.0) and np.isclose(g.flux[2, 0], 75.0)
    # variance scaled by scale^2
    assert np.isclose(g.var[0, 0], 0.75 ** 2) and np.isclose(g.var[2, 0], 1.5 ** 2)
    assert sr.exposures[0].scale == 0.75 and sr.exposures[1].scale == 1.5


def test_apply_scales_is_ratio_idempotent():
    sr = _two_exposure_super(100.0, 50.0)
    sr.apply_scales({'exp0': 0.75, 'exp1': 1.5})
    sr.apply_scales({'exp0': 0.75, 'exp1': 1.5})       # same target again -> no further change
    g = sr.channels['green']
    assert np.isclose(g.flux[0, 0], 75.0) and np.isclose(g.flux[2, 0], 75.0)


def test_measure_reference_flux_ignores_out_of_aperture():
    sr = _two_exposure_super(100.0, 50.0)
    src = SkyCoord(RAS * u.deg, DECS * u.deg)
    fx = measure_reference_flux(sr, [src], radius_arcsec=2.0, channels=['green'], band=(4999, 5001))
    assert np.isclose(fx[0][0], 100.0) and np.isclose(fx[1][0], 50.0)   # sky fibre excluded


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
