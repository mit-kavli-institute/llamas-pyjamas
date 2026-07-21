"""Tests for the per-camera additive continuum pedestal (Sky/skyPedestal.py, sky-refine Phase 3a).

Synthetic camera: base sky already subtracted (sci.sky = spline*throughput), leaving a smooth
additive continuum FLOOR on every fibre plus object flux on the bright fibres. The pedestal must:
recover the floor from blank fibres, remove it, PRESERVE narrow line emission (the Lyα guard),
preserve real object continuum, select only blank fibres, and no-op when there is no data.

Runnable with pytest or as a plain script (`python -m llamas_pyjamas.test_sky_pedestal`).
"""

import numpy as np

from llamas_pyjamas.Sky.skyPedestal import estimate_pedestal, apply_continuum_pedestal

NFIB, NWAVE = 60, 400
rng = np.random.default_rng(0)


def _gauss(x, x0, amp, sig=1.2):
    return amp * np.exp(-0.5 * ((x - x0) / sig) ** 2)


def _camera(line_fibres=(), line_amp=40.0, floor_amp=5.0):
    """Return (counts, base_sky, tp, floor). Fibres 0-19 bright (object), 20-59 blank."""
    x = np.arange(NWAVE)
    oh = sum(_gauss(x, c, 500.0) for c in (80, 160, 240, 320))      # OH lines (in base sky)
    sky_model = 100.0 + oh
    tp = np.clip(1.0 + 0.05 * rng.normal(size=NFIB), 0.5, None)
    floor = floor_amp + 0.01 * x                                    # smooth additive continuum floor
    base_sky = sky_model[None, :] * tp[:, None]
    counts = base_sky + floor[None, :] + rng.normal(0, 0.3, size=(NFIB, NWAVE))
    counts[:20] += 2000.0                                           # bright object continuum
    line = _gauss(x, 200, line_amp, sig=1.0)                        # narrow "Lyα" emission
    for i in line_fibres:
        counts[i] += line
    return counts, base_sky, tp, floor


class _Cam:
    def __init__(self, counts, sky, tp, dead=()):
        self.counts = counts; self.sky = sky
        self.relative_throughput = tp; self.dead_fibers = list(dead)


def _raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    raise AssertionError(f'expected {exc.__name__}')


def test_estimate_recovers_the_floor():
    counts, base_sky, tp, floor = _camera()
    blank = np.zeros(NFIB, bool); blank[20:] = True
    ped = estimate_pedestal(counts, base_sky, blank)
    # recovers the smooth additive floor (away from the median-filter edges)
    assert np.allclose(ped[50:-50], floor[50:-50], atol=1.0)


def test_narrow_line_not_absorbed_into_pedestal():
    # a narrow line in some BLANK fibres must NOT enter the continuum pedestal (the Lyα guard)
    counts, base_sky, tp, floor = _camera(line_fibres=(30, 35, 40), line_amp=60.0)
    blank = np.zeros(NFIB, bool); blank[20:] = True
    ped = estimate_pedestal(counts, base_sky, blank)
    assert abs(ped[200] - floor[200]) < 2.0            # pedestal at the line ~ floor, not floor+line


def test_apply_removes_floor_preserves_line_and_object():
    counts, base_sky, tp, floor = _camera(line_fibres=(30,), line_amp=60.0)
    cam = _Cam(counts.copy(), base_sky.copy(), tp)
    apply_continuum_pedestal([cam], {'sky_pedestal_nfibres': 40, 'sky_pedestal_scope': 'camera'})
    resid = cam.counts - cam.sky                        # SKYSUB after the pedestal
    # blank, line-free fibre: floor removed -> ~0
    assert abs(np.median(resid[50, 50:-50])) < 1.5
    # the narrow line survives in the line fibre
    assert resid[30, 200] > 0.5 * 60.0
    # real object continuum preserved on a bright fibre (only the ~5-count floor removed)
    assert abs(np.median(resid[5, 50:-50]) - 2000.0) < 5.0
    assert cam.sky_pedestal.shape == (NWAVE,)


def test_blank_selection_excludes_bright_fibres():
    from llamas_pyjamas.Sky.skyPedestal import _blank_mask
    counts, base_sky, tp, floor = _camera()
    cam = _Cam(counts, base_sky, tp)
    blank = _blank_mask(cam, 40)
    assert not blank[:20].any()                          # no bright object fibres
    assert blank[20:].sum() >= 35                        # essentially the blank set


def test_clip_negative():
    counts, base_sky, tp, floor = _camera(floor_amp=-8.0)   # a (non-physical) negative floor
    blank = np.zeros(NFIB, bool); blank[20:] = True
    ped = estimate_pedestal(counts, base_sky, blank, clip_negative=True)
    assert np.all(ped >= 0)


def test_too_few_blank_is_noop():
    counts, base_sky, tp, floor = _camera()
    blank = np.zeros(NFIB, bool); blank[:2] = True         # <MIN_BLANK
    ped = estimate_pedestal(counts, base_sky, blank)
    assert np.allclose(ped, 0.0)


def _camera_slitfloor(line_fibres=(), line_amp=40.0):
    """Like _camera but the floor VARIES along the slit (arch, the diagnosed shape)."""
    x = np.arange(NWAVE)
    oh = sum(_gauss(x, c, 500.0) for c in (80, 160, 240, 320))
    sky_model = 100.0 + oh
    tp = np.clip(1.0 + 0.05 * rng.normal(size=NFIB), 0.5, None)
    fib = np.arange(NFIB)
    arch = 8.0 * np.sin(np.pi * fib / (NFIB - 1)) - 3.0        # -3 at slit ends, +5 mid: an arch
    base_sky = sky_model[None, :] * tp[:, None]
    counts = base_sky + arch[:, None] + rng.normal(0, 0.3, size=(NFIB, NWAVE))
    counts[25:35] += 2000.0                                     # an object BLOCK mid-slit
    line = _gauss(x, 200, line_amp, sig=1.0)
    for i in line_fibres:
        counts[i] += line
    return counts, base_sky, tp, arch


def test_slit_scope_recovers_along_slit_floor():
    counts, base_sky, tp, arch = _camera_slitfloor(line_fibres=(50,))
    cam = _Cam(counts.copy(), base_sky.copy(), tp)
    apply_continuum_pedestal([cam], {'sky_pedestal_scope': 'slit'})
    resid = cam.counts - cam.sky
    # blank fibres at slit END and MIDDLE both go to ~0 (a per-camera constant cannot do this)
    for i in (2, 15, 45, 55):
        assert abs(np.median(resid[i, 60:-60])) < 1.5, i
    # the pedestal tracked the arch shape
    assert cam.sky_pedestal.shape == (NFIB, NWAVE)
    prof = np.median(cam.sky_pedestal[:, 60:-60], axis=1)
    assert np.corrcoef(prof, arch)[0, 1] > 0.95
    # narrow line in a blank fibre survives
    assert resid[50, 200] > 0.5 * 40.0


def test_slit_scope_object_continuum_preserved():
    # the object block's pedestal is interpolated from blank NEIGHBOURS -> its own 2000-count
    # continuum is untouched; only the underlying arch floor is removed
    counts, base_sky, tp, arch = _camera_slitfloor()
    cam = _Cam(counts.copy(), base_sky.copy(), tp)
    apply_continuum_pedestal([cam], {'sky_pedestal_scope': 'slit'})
    resid = cam.counts - cam.sky
    med_obj = np.median(resid[30, 60:-60])
    assert abs(med_obj - 2000.0) < 4.0                       # object continuum preserved (0.2%)
    # and the pedestal under the object is close to the true local floor, not floor+object.
    # (2.5 tolerance: with only 60 toy fibres the linear interpolation across the object gap
    # slightly undershoots the arch peak; real cameras have ~300 fibres -> far less curvature.)
    assert abs(np.median(cam.sky_pedestal[30, 60:-60]) - arch[30]) < 2.5


def test_placeholder_camera_skipped():
    cam = _Cam(np.zeros((NFIB, NWAVE)), np.zeros((NFIB, NWAVE)), np.ones(NFIB))
    apply_continuum_pedestal([cam], {})
    assert not hasattr(cam, 'sky_pedestal') or cam.sky_pedestal is None


if __name__ == '__main__':
    import sys
    fns = [(k, v) for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            fn(); print(f'PASS {name}')
        except Exception as e:                       # noqa: BLE001
            failed += 1; print(f'FAIL {name}: {type(e).__name__}: {e}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
