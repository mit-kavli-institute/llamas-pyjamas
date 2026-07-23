"""Unit tests for skyLineRefine.refine_fibre (pkl-domain per-line OH refinement)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import numpy as np
from llamas_pyjamas.Sky.skyLineRefine import refine_fibre, refine_sky_lines_pkl

rng = np.random.default_rng(0)


def _gauss(x, c, s, a):
    return a * np.exp(-0.5 * ((x - c) / s) ** 2)


def _sky_with_lines(n=400):
    x = np.arange(n, dtype=float)
    sky = 40.0 + _gauss(x, 100, 2.2, 5000) + _gauss(x, 250, 2.2, 3000) + _gauss(x, 330, 2.0, 1500)
    return x, sky


def test_amplitude_residual_removed():
    x, sky = _sky_with_lines()
    line = sky - 40.0
    counts = sky + 0.15 * line                       # base under-subtracted lines by 15%
    corr = refine_fibre(counts, sky)
    before = counts - sky
    after = counts - (sky + corr)
    core = line > 0.3 * line.max()
    assert np.std(after[core]) < 0.25 * np.std(before[core])   # >75% of the line residual removed


def test_shift_residual_removed():
    x, sky = _sky_with_lines()
    # data lines shifted +0.4 px vs the model -> a derivative (shift) residual
    counts = 40.0 + _gauss(x, 100.4, 2.2, 5000) + _gauss(x, 250.4, 2.2, 3000) + _gauss(x, 330.4, 2.0, 1500)
    corr = refine_fibre(counts, sky)
    before = counts - sky
    after = counts - (sky + corr)
    core = (sky - 40.0) > 0.3 * (sky - 40.0).max()
    assert np.std(after[core]) < 0.4 * np.std(before[core])


def test_continuum_untouched():
    x, sky = _sky_with_lines()
    counts = sky + 0.1 * (sky - 40.0)
    corr = refine_fibre(counts, sky)
    offline = (sky - 40.0) < 1.0                      # far from any line
    assert np.allclose(corr[offline], 0.0, atol=1e-6)  # correction is line-localised


def test_no_lines_is_noop():
    x = np.arange(300, dtype=float)
    sky = 40.0 + rng.normal(0, 0.3, x.size)           # flat, no OH lines
    counts = sky + rng.normal(0, 0.3, x.size)
    corr = refine_fibre(counts, sky)
    assert np.allclose(corr, 0.0)


def test_camera_driver_folds_into_sky():
    class Cam:
        pass
    x, sky = _sky_with_lines()
    S = np.vstack([sky, sky]); C = S + 0.12 * np.vstack([sky - 40, sky - 40])
    cam = Cam(); cam.sky = S.copy(); cam.counts = C
    refine_sky_lines_pkl([cam], {}, metadata=[{"channel": "green"}])
    assert hasattr(cam, "sky_line_refine") and cam.sky_line_refine.shape == S.shape
    line = sky - 40.0; core = line > 0.3 * line.max()
    assert np.std((C - cam.sky)[0][core]) < np.std((C - S)[0][core])


def test_template_term_removes_shape():
    from llamas_pyjamas.Sky.skyLineTemplate import OFF
    x, sky = _sky_with_lines()
    line = sky - 40.0
    xshift = np.arange(x.size, dtype=float)
    tprof = 0.04 * (OFF / 3.0) * np.exp(-((OFF / 3.0) ** 2))   # antisymmetric wing, ~few %
    resid = np.zeros(x.size)
    for c0, p0 in [(100, 5000.0), (250, 3000.0), (330, 1500.0)]:
        resid += p0 * np.interp(xshift - c0, OFF, tprof, left=0.0, right=0.0)
    counts = sky + resid                                        # residual is a pure template shape
    corr = refine_fibre(counts, sky, xshift_1d=xshift, tprof=tprof, offgrid=OFF)
    m = line > 0.05 * line.max()
    assert np.std((resid - corr)[m]) < 0.35 * np.std(resid[m])  # template component removed


def test_template_save_load_roundtrip(tmp_path=None):
    import tempfile, os
    from llamas_pyjamas.Sky.skyLineTemplate import save_template, load_template, OFF, N_SLITBIN
    tmpl = {"1A": rng.normal(0, 0.02, (N_SLITBIN, OFF.size)),
            "2B": rng.normal(0, 0.02, (N_SLITBIN, OFF.size))}
    d = tmp_path or tempfile.mkdtemp()
    p = os.path.join(str(d), "line_template_green.fits")
    save_template(p, tmpl, OFF, "green")
    t2, o2 = load_template(p, "green")
    assert sorted(t2) == sorted(tmpl)
    assert np.allclose(o2, OFF)
    assert np.allclose(t2["1A"], tmpl["1A"], atol=1e-5)


if __name__ == "__main__":
    p = f = 0
    for name in list(globals()):
        if name.startswith("test_"):
            try:
                globals()[name](); print("PASS", name); p += 1
            except AssertionError as e:
                print("FAIL", name, e); f += 1
    print(f"{p}/{p+f} passed")
