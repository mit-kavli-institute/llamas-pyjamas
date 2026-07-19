"""Tests for the astrometric registration engine (Utils/register.py, Phase 3).

Synthetic and network-free: source detection on a fibre lattice, the common-offset matcher
(robust to the pointing error, rejecting decoys), and the solve with its rotation cap. The full
register_exposure orchestrator + live Gaia is validated on real fields separately.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_register`).
"""

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

from llamas_pyjamas.Utils.register import (detect_fibre_sources, _match_common_offset, solve_wcs,
                                           _image_wcs_from_fibremap)
from llamas_pyjamas.Utils.wcsLlamas import celestial_wcs


def _hex_lattice(n=16):
    xs, ys = [], []
    for row in range(n):
        yy = row * (np.sqrt(3) / 2)
        off = 0.5 if (row % 2) else 0.0
        for col in range(n):
            xs.append(col + off)
            ys.append(yy)
    return np.array(xs), np.array(ys)


def _two_sources(x, y):
    def g(cx, cy, a):
        return a * np.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / 0.9 ** 2)
    rng = np.random.default_rng(0)
    # pedestal + realistic noise so the MAD-based threshold behaves like on real data
    return g(4.0, 5.0, 1000.0) + g(11.0, 9.0, 600.0) + 50.0 + rng.normal(0, 20.0, size=x.shape)


def test_detect_finds_the_two_sources():
    x, y = _hex_lattice()
    f = _two_sources(x, y)
    srcs = detect_fibre_sources(x, y, f, nsigma=5, min_sep=2.0)
    assert len(srcs) == 2
    found = sorted([(s.x, s.y) for s in srcs])
    assert np.allclose(found[0], (4.0, 5.0), atol=0.1)
    assert np.allclose(found[1], (11.0, 9.0), atol=0.1)


def test_detect_returns_brightest_first():
    x, y = _hex_lattice()
    f = _two_sources(x, y)
    srcs = detect_fibre_sources(x, y, f)
    assert srcs[0].flux_sum >= srcs[1].flux_sum
    assert np.allclose((srcs[0].x, srcs[0].y), (4.0, 5.0), atol=0.1)   # the amp=1000 one


def test_match_common_offset_matches_true_and_rejects_decoy():
    # three detected sources sharing a common ~2" offset from Gaia, plus a decoy far away
    truth = SkyCoord([100.0, 100.003, 99.997] * u.deg, [20.0, 20.002, 19.996] * u.deg)
    off_ra, off_dec = 2.0 / 3600.0, -1.0 / 3600.0     # common pointing error (within coarse_tol)
    det = SkyCoord((truth.ra.deg - np.array([off_ra / np.cos(np.deg2rad(20)), off_ra /
                    np.cos(np.deg2rad(20)), off_ra / np.cos(np.deg2rad(20))])) * u.deg,
                   (truth.dec.deg - off_dec) * u.deg)
    det = SkyCoord(list(det.ra.deg) + [100.05], list(det.dec.deg) + [20.05], unit='deg')  # +decoy
    gaia = truth
    di, gi = _match_common_offset(det, gaia)
    assert len(di) == 3                                # three real, decoy excluded
    assert 3 not in di                                 # the decoy (index 3) is not matched


def test_match_common_offset_cluster_beats_median():
    # 6 Gaia; 2 detections share the TRUE common offset, 4 are spurious with scattered offsets
    # whose inclusion drags a MEDIAN-based seed off the real pair. The densest-cluster seed must
    # still recover the 2 real matches (regression for the J1613 06-22 banding failure).
    gra = 100.0 + np.array([0.000, 0.004, 0.008, 0.012, 0.016, 0.020])
    gdec = 20.0 + np.array([0.000, 0.003, -0.002, 0.005, -0.004, 0.001])
    gaia = SkyCoord(gra * u.deg, gdec * u.deg)
    off_ra, off_dec = 2.7 / 3600.0, -5.4 / 3600.0        # true common offset (~6" pointing error)
    cosd = np.cos(np.deg2rad(20.0))
    dra = [gra[0] - off_ra / cosd, gra[1] - off_ra / cosd]
    ddec = [gdec[0] - off_dec, gdec[1] - off_dec]
    for j, (a, b) in enumerate([(3.5, 1.0), (-4.0, -1.0), (1.0, 4.5), (-2.0, 3.0)]):   # spurious
        dra.append(gra[2 + j] - a / 3600.0 / cosd)
        ddec.append(gdec[2 + j] - b / 3600.0)
    det = SkyCoord(np.array(dra) * u.deg, np.array(ddec) * u.deg)
    di, gi = _match_common_offset(det, gaia)
    assert 0 in di and 1 in di                            # both real detections recovered
    assert set(gi) >= {0, 1}


def test_solve_one_star_is_translation_only():
    rough = celestial_wcs(180.0, 0.0, crpix=(50.0, 50.0), arcsec_per_pixel=0.75, pa_deg=100.0)
    xy = [(40.0, 45.0)]
    # true position = rough prediction shifted by a small translation (5")
    p = rough.pixel_to_world(40.0, 45.0)
    gaia = SkyCoord((p.ra.deg + 2 / 3600.0 / np.cos(np.deg2rad(0))) * u.deg,
                    (p.dec.deg + 1 / 3600.0) * u.deg)
    wcs, rms, rot, refined, n = solve_wcs(xy, gaia, rough)
    assert refined is False and n == 1                 # translation only (rotation held)
    assert wcs.pixel_to_world(40.0, 45.0).separation(gaia).arcsec < 0.05


def test_solve_rejects_gross_mismatch():
    rough = celestial_wcs(180.0, 0.0, crpix=(50.0, 50.0), arcsec_per_pixel=0.75, pa_deg=100.0)
    p = rough.pixel_to_world(40.0, 45.0)
    far = SkyCoord((p.ra.deg + 40 / 3600.0) * u.deg, p.dec.deg * u.deg)   # 40" mismatch
    assert solve_wcs([(40.0, 45.0)], far, rough, max_shift_arcsec=15.0) is None


def test_solve_drops_outlier_star():
    rough = celestial_wcs(180.0, 0.0, crpix=(50.0, 50.0), arcsec_per_pixel=0.75, pa_deg=100.0)
    xy = [(40.0, 45.0), (60.0, 55.0), (50.0, 60.0)]
    preds = rough.pixel_to_world(np.array([p[0] for p in xy]), np.array([p[1] for p in xy]))
    # all shifted +5"; the 3rd star is a bad match (extra 20" off) -> should be dropped
    ra = preds.ra.deg + np.array([2, 2, 22]) / 3600.0
    dec = preds.dec.deg + np.array([0, 0, 0]) / 3600.0
    wcs, rms, rot, refined, n = solve_wcs(xy, SkyCoord(ra * u.deg, dec * u.deg), rough)
    assert n == 2 and rms < 0.3                         # outlier rejected, clean fit on the 2 good


def test_solve_rotation_cap_gate():
    rough = celestial_wcs(180.0, 0.0, crpix=(50.0, 50.0), arcsec_per_pixel=0.75, pa_deg=100.0)
    xy = [(40.0, 45.0), (60.0, 55.0)]
    gaia = rough.pixel_to_world(np.array([40.0, 60.0]), np.array([45.0, 55.0]))
    # cap 0 -> even a tiny fitted rotation is rejected -> translation only
    _, _, _, refined0, _ = solve_wcs(xy, gaia, rough, refine_rotation=True, max_rot_deg=0.0)
    assert refined0 is False
    # generous cap -> a two-star rotation solve is accepted
    _, _, _, refined1, _ = solve_wcs(xy, gaia, rough, refine_rotation=True, max_rot_deg=30.0)
    assert refined1 is True


def test_image_wcs_from_fibremap_matches():
    # image pixel p (1-idx) -> fibre-map (p-1)*step, so the image WCS must reproduce the fibre-map
    # WCS evaluated at that fibre-map coordinate.
    fm = celestial_wcs(150.0, -20.0, crpix=(23.0, 22.0), arcsec_per_pixel=0.75, pa_deg=100.0)
    step = 0.1                                          # hex render: 10 px per fibre unit
    img = _image_wcs_from_fibremap(fm, step)
    for px, py in [(0.0, 0.0), (231.0, 221.0), (100.0, 50.0)]:
        c_img = img.pixel_to_world(px, py)
        c_fm = fm.pixel_to_world(px * step, py * step)
        assert c_img.separation(c_fm).arcsec < 1e-4, (px, py)


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
