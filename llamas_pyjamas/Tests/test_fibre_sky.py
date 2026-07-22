"""Tests for per-fibre plate-solved sky coordinates (llamasRSS._fibre_sky_table, Phase 2).

Uses the bundled fibre map (FiberMap_LUT) with a synthetic pointing header. Checks that the rough
header WCS fills RA/DEC for real fibres, that the fibre-map scale round-trips to 0.75"/unit, and
that a header without pointing leaves RA/DEC NaN while still returning fibre-map x/y (so a later
astrometric solve can run).

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.Tests.test_fibre_sky`).
"""

import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

from llamas_pyjamas.File.llamasRSS import _fibre_sky_table
from llamas_pyjamas.Image.WhiteLightModule import FiberMap_LUT
from llamas_pyjamas.Utils.wcsLlamas import ARCSEC_PER_FIBRE, IFU_PA_OFFSET


def _header():
    h = fits.Header()
    h['RA'] = 243.2592
    h['DEC'] = 8.1349
    h['TEL ROT'] = 212.0
    return h


def _valid_fibres(n=40):
    out = []
    for bench in ('1A', '2A', '3A', '4A', '1B', '2B', '3B', '4B'):
        for fid in range(1, 60):
            x, y = FiberMap_LUT(bench, fid)
            if not (x < 0 and y < 0):
                out.append((bench, fid))
                if len(out) >= n:
                    return out
    return out


def test_rough_header_fills_radec_and_xy():
    fibres = _valid_fibres()
    benchsides = [b for b, _ in fibres]
    fiber_ids = [f for _, f in fibres]
    ras, decs, xs, ys, prov = _fibre_sky_table(fiber_ids, benchsides, _header())
    assert prov['method'] == 'rough-header' and prov['refined'] is False
    assert prov['pa_offset'] == IFU_PA_OFFSET
    assert np.isfinite(ras).all() and np.isfinite(decs).all()
    assert np.isfinite(xs).all() and np.isfinite(ys).all()


def test_scale_roundtrips_to_fibre_pitch():
    fibres = _valid_fibres()
    benchsides = [b for b, _ in fibres]
    fiber_ids = [f for _, f in fibres]
    ras, decs, xs, ys, _ = _fibre_sky_table(fiber_ids, benchsides, _header())
    # two fibres well separated on the map: sky sep / fibre-map sep == 0.75"/unit
    i, j = 0, len(xs) - 1
    map_sep = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
    sky_sep = SkyCoord(ras[i] * u.deg, decs[i] * u.deg).separation(
        SkyCoord(ras[j] * u.deg, decs[j] * u.deg)).arcsec
    assert np.isclose(sky_sep / map_sep, ARCSEC_PER_FIBRE, rtol=1e-3)


def test_no_pointing_leaves_radec_nan_but_keeps_xy():
    fibres = _valid_fibres(10)
    benchsides = [b for b, _ in fibres]
    fiber_ids = [f for _, f in fibres]
    ras, decs, xs, ys, prov = _fibre_sky_table(fiber_ids, benchsides, fits.Header())
    assert prov['method'] == 'none'
    assert np.isnan(ras).all() and np.isnan(decs).all()
    assert np.isfinite(xs).all() and np.isfinite(ys).all()   # still re-solvable later


def test_lut_miss_gives_nan_xy():
    # a nonsense fibre id -> LUT miss -> NaN x/y and RA/DEC, no crash
    ras, decs, xs, ys, _ = _fibre_sky_table([99999], ['9Z'], _header())
    assert np.isnan(xs[0]) and np.isnan(ras[0])


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
