"""Tests for the cube co-add (Combine/cube.py).

A synthetic super-RSS with one source fibre carrying an emission-line spectrum, seen in two
exposures, run through combine_cube; checks the spectrum is recovered at the source spaxel (line
position + inverse-variance combined amplitude), coverage/exposure depth, surface-brightness units,
and the wavelength axis. The full pipeline is validated on real may26 data.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_combine_cube`).
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from llamas_pyjamas.Combine.superRSS import ChannelStack, SuperRSS, ExposureMeta
from llamas_pyjamas.Combine.cube import combine_cube

RAS, DECS = 150.0, 20.0
FAR = 0.02                                       # deg, outside the kernel


def _line_super(area=1.0):
    wl = np.linspace(5000.0, 5020.0, 21)         # native grid (dwave 1 A)
    line = 2.0 + 10.0 * np.exp(-0.5 * ((wl - 5010.0) / 1.5) ** 2)   # continuum + emission line
    # rows: [exp0 source, exp0 sky, exp1 source, exp1 sky]
    flux = np.array([line, np.ones_like(wl), line, np.ones_like(wl)])
    wave = np.tile(wl, (4, 1))
    var = np.ones((4, wl.size))
    st = ChannelStack(
        channel='green',
        ra=np.array([RAS, RAS + FAR, RAS, RAS + FAR]),
        dec=np.array([DECS, DECS, DECS, DECS]),
        wave=wave, flux=flux, var=var, mask=np.zeros((4, wl.size), bool),
        solid_angle=np.full(4, area), exposure=np.array([0, 0, 1, 1]))
    exps = [ExposureMeta('e0', 'p0', 1.0, 1.0, 0.0), ExposureMeta('e1', 'p1', 1.0, 1.0, 0.0)]
    return SuperRSS(field='T', plane='skysub', bunit='counts', exposures=exps,
                    channels={'green': st}), wl


def _spaxel(cube, ra, dec):
    px, py = cube.wcs.celestial.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
    return int(round(float(py))), int(round(float(px)))


def test_cube_recovers_line_spectrum_at_source():
    sr, wl = _line_super(area=1.0)
    cube = combine_cube(sr, 'green', pixscale=1.0, kernel='tophat', kernel_fwhm=1.0, units='sb')
    iy, ix = _spaxel(cube, RAS, DECS)
    sp = cube.data[:, iy, ix]
    assert np.isfinite(sp).any()
    # line peak near 5010 A, amplitude ~ continuum(2)+line(10) = 12 (ivar combine of 2 equal expos)
    kpk = int(np.nanargmax(sp))
    assert abs(cube.wave[kpk] - 5010.0) <= cube.meta['DWAVE'] * 1.5
    assert np.isclose(np.nanmax(sp), 12.0, rtol=0.1)
    assert np.isclose(np.nanmedian(sp[sp < 5]), 2.0, rtol=0.1)     # continuum level


def test_cube_coverage_and_units():
    sr, wl = _line_super(area=2.0)
    cube = combine_cube(sr, 'green', pixscale=1.0, kernel='tophat', kernel_fwhm=1.0, units='sb')
    iy, ix = _spaxel(cube, RAS, DECS)
    assert cube.coverage[iy, ix] == 2 and cube.nexp[iy, ix] == 2
    # SB = flux/area: continuum 2/2 = 1
    sp = cube.data[:, iy, ix]
    assert np.isclose(np.nanmedian(sp[sp < 2.5]), 1.0, rtol=0.1)
    assert cube.bunit.endswith('/arcsec2')


def test_cube_wave_axis_spans_range():
    sr, wl = _line_super()
    cube = combine_cube(sr, 'green', pixscale=1.0, kernel='tophat', kernel_fwhm=1.0)
    assert cube.wave[0] >= 5000.0 - 1 and cube.wave[-1] <= 5020.0 + 1
    assert cube.data.shape[0] == cube.wave.size
    assert cube.data.shape[1:] == cube.coverage.shape


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
