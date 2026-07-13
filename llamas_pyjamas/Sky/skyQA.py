"""
llamas_pyjamas.Sky.skyQA
=======================
Quality-assurance diagnostics for the sky-subtraction framework.

Quantifies how much OH-line residual was removed by comparing the FF ``FLUX``
(before) against the SKYSUB ``FLUX`` (after) inside OH-line windows, on
sky-dominated fibres, and writes a summary figure.

Public API
----------
sky_subtraction_qa(ff_fits, skysub_fits, sky_mask, config) -> dict
    Returns a small stats dict and (when possible) writes ``*_skyQA.png``.
"""

import logging
import os

import numpy as np
from astropy.io import fits

from llamas_pyjamas.Sky.skyScale import _continuum, _line_mask

logger = logging.getLogger(__name__)


def _oh_line_pixels(sky, sky_mask, config):
    """Boolean OH-line pixel mask (1-D, len n_wave) from the median sky template."""
    rows = np.where(sky_mask)[0]
    if rows.size == 0:
        rows = np.arange(sky.shape[0])
    template = np.nanmedian(sky[rows], axis=0)
    cont = _continuum(template, config.scale_window_pix)
    line_resid = np.nan_to_num(template) - cont
    return _line_mask(line_resid, config.oh_sigdetect)


def _line_rms(flux, line_px, sky_mask):
    """Median over sky fibres of the RMS of FLUX within OH-line pixels."""
    rows = np.where(sky_mask)[0]
    if rows.size == 0 or not line_px.any():
        return np.nan
    vals = []
    for r in rows:
        seg = flux[r, line_px]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            vals.append(np.sqrt(np.mean(seg ** 2)))
    return float(np.median(vals)) if vals else np.nan


def sky_subtraction_qa(ff_fits, skysub_fits, sky_mask, config):
    """Compute before/after OH residual RMS and write a QA figure.

    Returns a dict with ``rms_before``, ``rms_after``, ``improvement`` (ratio),
    and ``figure`` (path or None).
    """
    with fits.open(ff_fits) as h:
        flux_before = np.array(h["FLUX"].data, dtype=float)
        sky = np.array(h["SKY"].data, dtype=float) if "SKY" in h else np.zeros_like(flux_before)
        wave = np.array(h["WAVE"].data, dtype=float) if "WAVE" in h else None
    with fits.open(skysub_fits) as h:
        flux_after = np.array(h["FLUX"].data, dtype=float)

    line_px = _oh_line_pixels(sky, sky_mask, config)
    rms_before = _line_rms(flux_before, line_px, sky_mask)
    rms_after = _line_rms(flux_after, line_px, sky_mask)
    improvement = (rms_before / rms_after) if (rms_after and np.isfinite(rms_after)
                                               and rms_after > 0) else np.nan

    stats = {"rms_before": rms_before, "rms_after": rms_after,
             "improvement": improvement, "n_oh_pixels": int(line_px.sum()),
             "figure": None}
    logger.info("skyQA: OH-residual RMS before=%.4g after=%.4g (x%.2f better) "
                "over %d OH pixels", rms_before, rms_after, improvement,
                int(line_px.sum()))

    try:
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt

        qa_dir = config.qa_dir or os.path.dirname(os.path.abspath(skysub_fits))
        os.makedirs(qa_dir, exist_ok=True)
        fig_path = os.path.join(
            qa_dir,
            os.path.basename(skysub_fits).replace(".fits", "_skyQA.png"))

        rows = np.where(sky_mask)[0]
        example = rows[len(rows) // 2] if rows.size else 0

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        ax = axes[0]
        ax.bar(["before", "after"], [rms_before, rms_after],
               color=["0.6", "tab:green"])
        ax.set_ylabel("Median OH-line RMS (sky fibres)")
        ax.set_title(f"{os.path.basename(skysub_fits)}\n"
                     f"x{improvement:.2f} residual reduction, "
                     f"{int(line_px.sum())} OH pixels")

        ax = axes[1]
        xax = wave[example] if wave is not None else np.arange(flux_before.shape[1])
        ax.plot(xax, flux_before[example], color="0.6", lw=0.6, label="FF FLUX (before)")
        ax.plot(xax, flux_after[example], color="tab:green", lw=0.6, label="SKYSUB FLUX (after)")
        ax.set_xlabel("Wavelength (Å)" if wave is not None else "Pixel")
        ax.set_ylabel("Flux")
        ax.set_title(f"Example sky fibre #{example}")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)
        stats["figure"] = fig_path
        logger.info("skyQA: wrote %s", fig_path)
    except Exception as e:
        logger.warning("skyQA: could not write figure (%s)", e)

    return stats
