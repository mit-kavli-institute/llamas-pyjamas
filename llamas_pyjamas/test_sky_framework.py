"""
Unit / integration tests for the sky-subtraction framework (llamas_pyjamas.Sky).

Self-contained: builds a small synthetic per-colour RSS FITS file (no external
data, no Ray, no pypeit) and exercises masking, OH scaling, PCA residual
cleaning, and the full orchestrator + FITS contract.

Run:
    pytest llamas_pyjamas/test_sky_framework.py -v
    python -m unittest llamas_pyjamas.test_sky_framework
"""

import os
import tempfile
import unittest

import numpy as np
from astropy.io import fits

from llamas_pyjamas.Sky.skyConfig import SkySubtractConfig
from llamas_pyjamas.Sky.skyMask import build_sky_fiber_mask, white_light
from llamas_pyjamas.Sky.skyScale import scale_sky_per_fiber, _continuum, _line_mask
from llamas_pyjamas.Sky.skyResidual import clean_residuals
from llamas_pyjamas.Sky.skySubtract import subtract_sky_rss


N_WAVE = 400
N_OBJ = 8
N_SKY = 52
N_FIBER = N_OBJ + N_SKY          # object fibres first, then sky fibres
OH_CENTERS = [60, 130, 200, 275, 340]
OH_AMP = 100.0


def _gaussian(x, c, amp, sigma=2.0):
    return amp * np.exp(-0.5 * ((x - c) / sigma) ** 2)


def _sky_template(rng):
    """1-D OH sky-line template on the pixel grid."""
    x = np.arange(N_WAVE)
    s = np.zeros(N_WAVE)
    for c in OH_CENTERS:
        s += _gaussian(x, c, OH_AMP)
    return s


def build_synthetic_rss(path, rng, with_common=True):
    """Write a synthetic FF RSS file and return the ground-truth arrays.

    Construction
    ------------
    SKY[f]   = OH template (the base model / line shape).
    COUNTS[f]= continuum (bright for object fibres) + SKY  -> drives white-light.
    FLUX[f]  = object_continuum + per-fibre OH residual (alpha_f * SKY)
               + a common correlated residual (PCA target) + noise.
    """
    x = np.arange(N_WAVE)
    wave = np.tile(np.linspace(6000.0, 7000.0, N_WAVE), (N_FIBER, 1))
    template = _sky_template(rng)
    sky = np.tile(template, (N_FIBER, 1)).astype(float)

    # White-light separation: object fibres bright, sky fibres faint.
    obj_cont = np.zeros((N_FIBER, N_WAVE))
    for f in range(N_OBJ):
        obj_cont[f] = 500.0 + 50.0 * np.sin(x / 40.0)      # bright smooth source
    for f in range(N_OBJ, N_FIBER):
        obj_cont[f] = 5.0                                   # faint sky fibre

    counts = obj_cont + sky

    # Per-fibre OH residual (imperfect base subtraction): alpha_f * SKY.
    alpha = rng.normal(0.0, 0.12, size=N_FIBER)
    oh_resid = alpha[:, None] * sky

    # Common correlated residual across fibres (the ZAP/PCA target).
    common = np.zeros((N_FIBER, N_WAVE))
    if with_common:
        pattern = 8.0 * np.sin(x / 15.0) + 6.0 * np.cos(x / 7.0)
        amp = rng.uniform(0.8, 1.2, size=N_FIBER)
        common = amp[:, None] * pattern[None, :]

    noise = rng.normal(0.0, 1.0, size=(N_FIBER, N_WAVE))
    flux = obj_cont * 0.0 + oh_resid + common + noise      # base-subtracted residual
    # Object fibres also retain a faint continuum in FLUX (post fibre-flat).
    for f in range(N_OBJ):
        flux[f] += 20.0

    # Build FITS.
    phdu = fits.PrimaryHDU()
    phdu.header["CHANNEL"] = "green"

    def img(data, name):
        h = fits.ImageHDU(data.astype(np.float32))
        h.header["EXTNAME"] = name
        return h

    fiber_ids = np.arange(N_FIBER, dtype=np.int32)
    benchsides = np.array(["1A"] * N_FIBER)
    fiber_types = np.array(["UNKNOWN"] * N_FIBER)   # forces white-light masking path
    cols = [
        fits.Column(name="FIBER_ID", format="J", array=fiber_ids),
        fits.Column(name="BENCHSIDE", format="10A", array=benchsides),
        fits.Column(name="FIBER_TYPE", format="10A", array=fiber_types),
        fits.Column(name="RA", format="D", array=np.zeros(N_FIBER)),
        fits.Column(name="DEC", format="D", array=np.zeros(N_FIBER)),
    ]
    fibermap = fits.BinTableHDU.from_columns(cols)
    fibermap.header["EXTNAME"] = "FIBERMAP"

    hdul = fits.HDUList([phdu, img(flux, "FLUX"), img(counts, "COUNTS"),
                         img(sky, "SKY"), img(wave, "WAVE"), fibermap])
    hdul.writeto(path, overwrite=True)

    return dict(flux=flux, counts=counts, sky=sky, wave=wave,
                alpha=alpha, common=common, template=template)


def _oh_pixel_mask(template, config):
    cont = _continuum(template, config.scale_window_pix)
    return _line_mask(template - cont, config.oh_sigdetect)


def _line_rms(flux, line_px, rows):
    vals = [np.sqrt(np.mean(flux[r, line_px] ** 2)) for r in rows
            if np.all(np.isfinite(flux[r, line_px]))]
    return float(np.median(vals)) if vals else np.nan


class TestSkyMask(unittest.TestCase):
    def test_excludes_bright_object_fibers(self):
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as d:
            truth = build_synthetic_rss(os.path.join(d, "x_RSS_green_FF.fits"), rng)
        cfg = SkySubtractConfig()
        mask = build_sky_fiber_mask(truth["counts"], None, cfg)
        self.assertEqual(mask.dtype, np.bool_)
        # The bright object fibres (first N_OBJ) should be excluded.
        self.assertFalse(mask[:N_OBJ].any(),
                         "bright object fibres must not be selected as sky")
        # Plenty of genuine sky fibres should remain.
        self.assertGreater(mask.sum(), N_SKY // 2)

    def test_white_light_orders_fibers(self):
        rng = np.random.default_rng(1)
        with tempfile.TemporaryDirectory() as d:
            truth = build_synthetic_rss(os.path.join(d, "x_RSS_green_FF.fits"), rng)
        wl = white_light(truth["counts"])
        self.assertTrue(np.all(wl[:N_OBJ] > wl[N_OBJ:].max()))


class TestSkyScale(unittest.TestCase):
    def test_reduces_per_fiber_oh_residual(self):
        rng = np.random.default_rng(2)
        with tempfile.TemporaryDirectory() as d:
            truth = build_synthetic_rss(os.path.join(d, "x_RSS_green_FF.fits"),
                                        rng, with_common=False)
        cfg = SkySubtractConfig(method="scaled")
        scale, corr, flux1 = scale_sky_per_fiber(truth["flux"], truth["sky"], cfg)
        self.assertEqual(scale.shape, (N_FIBER,))
        line_px = _oh_pixel_mask(truth["template"], cfg)
        rows = np.arange(N_OBJ, N_FIBER)            # sky fibres
        rms_before = _line_rms(truth["flux"], line_px, rows)
        rms_after = _line_rms(flux1, line_px, rows)
        self.assertLess(rms_after, rms_before,
                        "OH-line residual RMS should drop after scaling")


class TestSkyResidual(unittest.TestCase):
    def test_removes_common_pattern(self):
        rng = np.random.default_rng(3)
        with tempfile.TemporaryDirectory() as d:
            truth = build_synthetic_rss(os.path.join(d, "x_RSS_green_FF.fits"), rng)
        cfg = SkySubtractConfig(method="pca", pca_ncomp=10)
        sky_mask = build_sky_fiber_mask(truth["counts"], None, cfg)
        model, info = clean_residuals(truth["flux"], truth["wave"], sky_mask, cfg)
        self.assertEqual(model.shape, truth["flux"].shape)
        self.assertGreaterEqual(info["ncomp"], 1)
        cleaned = truth["flux"] - model
        rows = np.where(sky_mask)[0]
        # Overall residual variance on sky fibres should fall substantially.
        var_before = np.var(truth["flux"][rows])
        var_after = np.var(cleaned[rows])
        self.assertLess(var_after, var_before)


class TestOrchestrator(unittest.TestCase):
    def test_end_to_end_contract(self):
        rng = np.random.default_rng(4)
        with tempfile.TemporaryDirectory() as d:
            ff = os.path.join(d, "LLAMAS_test_RSS_green_FF.fits")
            truth = build_synthetic_rss(ff, rng)
            cfg = SkySubtractConfig(method="pca", pca_ncomp=10, qa_plots=False)
            out = subtract_sky_rss(ff, config=cfg)

            # Naming contract.
            self.assertTrue(out.endswith("_FF_SKYSUB.fits"))
            self.assertTrue(os.path.exists(out))

            with fits.open(out) as h:
                self.assertIn("FLUX", h)
                self.assertIn("SKYRESID", h)
                self.assertIn("COUNTS", h)   # copied through
                self.assertIn("SKY", h)
                flux_out = h["FLUX"].data
                self.assertEqual(flux_out.shape, truth["flux"].shape)
                self.assertTrue(np.all(np.isfinite(flux_out)))
                self.assertTrue(h["FLUX"].header.get("SKYSUB2"))

            # OH residual on sky fibres should improve end-to-end.
            cfg2 = SkySubtractConfig()
            line_px = _oh_pixel_mask(truth["template"], cfg2)
            sky_rows = np.arange(N_OBJ, N_FIBER)
            rms_before = _line_rms(truth["flux"], line_px, sky_rows)
            rms_after = _line_rms(flux_out, line_px, sky_rows)
            self.assertLess(rms_after, rms_before)

    def test_scaled_only_skips_pca(self):
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as d:
            ff = os.path.join(d, "LLAMAS_test_RSS_red_FF.fits")
            build_synthetic_rss(ff, rng)
            cfg = SkySubtractConfig(method="scaled")
            out = subtract_sky_rss(ff, config=cfg)
            with fits.open(out) as h:
                self.assertEqual(h["FLUX"].header.get("SKYNCOMP", 0), 0)


if __name__ == "__main__":
    unittest.main()
