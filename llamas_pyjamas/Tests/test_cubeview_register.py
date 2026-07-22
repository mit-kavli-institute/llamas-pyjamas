"""Tests for the interactive-WCS session model (CubeViewer/cubeViewRegister.py).

The Qt dialog and DS9 interaction are not exercised here (they need a display + DS9); this covers
the headless :class:`InteractiveWCSSession` -- the click->centroid snap, the force-raw escape, the
nearest-Gaia pairing, and the incremental solve (translation, translation+rotation, held rotation)
that the dialog is a thin shell around.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.Tests.test_cubeview_register`).
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from llamas_pyjamas.CubeViewer.cubeViewRegister import InteractiveWCSSession
from llamas_pyjamas.Utils.register import _rough_wcs
from llamas_pyjamas.Image.WhiteLightModule import FIELD_XMAX, FIELD_YMAX

RA, DEC, PA = 180.0, 0.0, 100.0
CX, CY = FIELD_XMAX / 2.0, FIELD_YMAX / 2.0


def _lattice(half=6):
    xs, ys = [], []
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            xs.append(CX + i)
            ys.append(CY + j)
    return np.array(xs, float), np.array(ys, float)


def _shifted(wcs, dra_arcsec=0.0, ddec_arcsec=0.0):
    w = wcs.deepcopy()
    w.wcs.crval = [wcs.wcs.crval[0] + dra_arcsec / 3600.0, wcs.wcs.crval[1] + ddec_arcsec / 3600.0]
    return w


def test_grab_snaps_to_centroid_and_force_uses_raw():
    x, y = _lattice()
    star = (CX + 2.0, CY + 1.0)
    flux = 1000.0 * np.exp(-((x - star[0]) ** 2 + (y - star[1]) ** 2) / (2 * 0.8 ** 2)) + 1.0
    s = InteractiveWCSSession(np.c_[x, y], flux, RA, DEC, PA)

    (gx, gy), forced = s.grab(star[0] + 0.3, star[1] - 0.2)   # guess a bit off the star
    assert not forced
    assert abs(gx - star[0]) < 0.25 and abs(gy - star[1]) < 0.25

    (rx, ry), forced2 = s.grab(star[0] + 0.3, star[1] - 0.2, force=True)  # last-resort raw
    assert forced2 and rx == star[0] + 0.3 and ry == star[1] - 0.2


def test_nearest_gaia_pairs_the_right_source():
    x, y = _lattice()
    s = InteractiveWCSSession(np.c_[x, y], np.ones_like(x), RA, DEC, PA)
    here = s.rough.pixel_to_world(CX + 2.0, CY)
    far = s.rough.pixel_to_world(CX - 5.0, CY + 5.0)
    s.gaia = SkyCoord([here.ra.deg, far.ra.deg] * u.deg, [here.dec.deg, far.dec.deg] * u.deg)
    near = s.nearest_gaia((CX + 2.0, CY))
    assert near is not None and near[1] < 0.05                # matched the on-position source


def test_solve_recovers_shift_and_rotation():
    x, y = _lattice()
    s = InteractiveWCSSession(np.c_[x, y], np.ones_like(x), RA, DEC, PA)
    truth = _shifted(_rough_wcs(RA, DEC, PA, 1.5), dra_arcsec=3.0, ddec_arcsec=-2.0)
    for xy in [(CX - 3.0, CY - 2.0), (CX + 4.0, CY + 1.0), (CX, CY + 4.0)]:
        s.add_pair(xy, truth.pixel_to_world(xy[0], xy[1]), source='gaia')
    out = s.solve()
    assert out['n'] == 3 and out['refined'] is True
    assert out['rms'] < 0.1
    assert abs(s.rotation_offset() - 1.5) < 0.3
    for p in s.pairs:                                          # each star lands on its Gaia
        assert s.wcs.pixel_to_world(*p['xy']).separation(p['sky']).arcsec < 0.1


def test_held_rotation_is_translation_only():
    x, y = _lattice()
    s = InteractiveWCSSession(np.c_[x, y], np.ones_like(x), RA, DEC, PA)
    s.held_rotation = 1.5                                      # rotation solved earlier in the block
    truth = _shifted(_rough_wcs(RA, DEC, PA, 1.5), dra_arcsec=3.0)
    xy = (CX + 3.0, CY + 2.0)
    s.add_pair(xy, truth.pixel_to_world(xy[0], xy[1]))
    out = s.solve()
    assert out['refined'] is False                            # rotation held, not re-fitted
    assert abs(s.rotation_offset() - 1.5) < 0.1
    assert s.wcs.pixel_to_world(*xy).separation(truth.pixel_to_world(*xy)).arcsec < 0.1


def test_provenance_flags_manual_vs_gaia():
    x, y = _lattice()
    s = InteractiveWCSSession(np.c_[x, y], np.ones_like(x), RA, DEC, PA)
    sky = s.rough.pixel_to_world(CX, CY)
    s.add_pair((CX, CY), sky, source='gaia')
    assert s.provenance()['catalog'] == 'GaiaDR3' and s.provenance()['method'] == 'interactive'
    s.add_pair((CX + 1, CY), sky, source='manual')
    assert 'manual' in s.provenance()['catalog'] and s.provenance()['tier'] == 'manual'


if __name__ == '__main__':
    import sys
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f'PASS {name}')
        except Exception as e:                              # noqa: BLE001
            failed += 1
            print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
