"""
llamas_pyjamas.Sky.skySubtract
==============================
Orchestrator for the LLAMAS sky-subtraction framework.

Runs *after* the fibre-to-fibre flat as a per-colour, FITS-level stage.  It
consumes a fibre-flat-corrected RSS file (``..._RSS_{color}_FF.fits``), builds on
the base sky model already in its ``SKY`` extension, and writes a refined
product ``..._RSS_{color}_FF_SKYSUB.fits``.

Pipeline (per colour)::

    F0  = FF FLUX                       # (COUNTS - SKY)/C_i, base-subtracted, fibre-flat
    mask = build_sky_fiber_mask(COUNTS, FIBERMAP)
    F1  = F0 - alpha[f]*line(SKY[f])    # skyScale  (per-fibre OH residual)
    R   = PCA(F1 over masked fibres)    # skyResidual (ZAP; skipped if method='scaled')
    FLUX_out = F1 - R

The fibre-flat correction lives only in FF ``FLUX``; ``COUNTS``/``SKY`` are not
fibre-flat corrected, so we never recompute from ``COUNTS`` — all corrections are
applied directly to ``FLUX`` using ``SKY`` purely as the OH line-shape template.

Public API
----------
subtract_sky_rss(ff_fits, output_file=None, config=None) -> str
subtract_sky_all_colors(ff_files, config=None) -> list[str]
"""

import logging
import os

import numpy as np
from astropy.io import fits

from llamas_pyjamas.Sky.skyConfig import SkySubtractConfig
from llamas_pyjamas.Sky.skyMask import build_sky_fiber_mask
from llamas_pyjamas.Sky.skyScale import scale_sky_per_fiber
from llamas_pyjamas.Sky.skyResidual import clean_residuals

logger = logging.getLogger(__name__)


def _get_data(hdul, extname):
    """Return a float64 copy of an extension's data, or None if absent."""
    try:
        data = hdul[extname].data
    except KeyError:
        return None
    return None if data is None else np.array(data, dtype=float)


def subtract_sky_rss(ff_fits, output_file=None, config=None):
    """Refine sky subtraction on one fibre-flat-corrected RSS file.

    Parameters
    ----------
    ff_fits : str
        Path to ``..._RSS_{color}_FF.fits``.
    output_file : str, optional
        Output path; defaults to ``ff_fits`` with ``_SKYSUB`` appended.
    config : SkySubtractConfig, optional
        Defaults to :class:`SkySubtractConfig` defaults.

    Returns
    -------
    str
        Path to the written ``..._FF_SKYSUB.fits`` file.
    """
    config = config or SkySubtractConfig()
    if output_file is None:
        base, ext = os.path.splitext(ff_fits)
        output_file = f"{base}_SKYSUB{ext}"

    logger.info("skySubtract: %s -> %s",
                os.path.basename(ff_fits), os.path.basename(output_file))

    with fits.open(ff_fits) as hdul:
        flux = _get_data(hdul, "FLUX")
        if flux is None:
            raise ValueError(f"{ff_fits}: no FLUX extension")
        sky = _get_data(hdul, "SKY")
        counts = _get_data(hdul, "COUNTS")
        wave = _get_data(hdul, "WAVE")
        try:
            fibermap = hdul["FIBERMAP"].data
        except KeyError:
            fibermap = None

        if sky is None or np.all(sky == 0):
            logger.warning("%s: SKY extension empty — OH scaling will be a no-op",
                           os.path.basename(ff_fits))
            sky = np.zeros_like(flux)
        if wave is None:
            raise ValueError(f"{ff_fits}: no WAVE extension (needed for PCA)")
        wl_source = counts if counts is not None else flux

        # 1. Source masking.
        sky_mask = build_sky_fiber_mask(wl_source, fibermap, config)

        # 2. Per-fibre OH scaling (in FLUX space).
        scale, scale_corr, flux1 = scale_sky_per_fiber(flux, sky, config,
                                                       sky_mask=sky_mask)

        # 3. PCA residual cleaning (optional).
        if config.run_pca:
            residual_model, pca_info = clean_residuals(flux1, wave, sky_mask, config)
        else:
            residual_model = np.zeros_like(flux1)
            pca_info = {"ncomp": 0, "n_basis": int(sky_mask.sum())}

        flux_out = (flux1 - residual_model).astype(np.float32)
        total_removed = (scale_corr + residual_model).astype(np.float32)

        # --- assemble output, copying every input extension ---
        out = fits.HDUList([h.copy() for h in hdul])
        out["FLUX"].data = flux_out
        out["FLUX"].header["SKYSUB2"] = (True, "Sky framework refinement applied")
        for key, val in config.to_header_dict().items():
            out["FLUX"].header[key] = val
        out["FLUX"].header["SKYNMASK"] = (int(sky_mask.sum()),
                                          "N sky fibres used")
        out["FLUX"].header["SKYNBAS"] = (int(pca_info.get("n_basis", 0)),
                                         "N PCA basis fibres")
        out["FLUX"].header["SKYNCOMP"] = (int(pca_info.get("ncomp", 0)),
                                          "N PCA components removed")

        # Traceability: total residual removed from the FF FLUX.
        resid_hdu = fits.ImageHDU(total_removed, header=out["FLUX"].header.copy())
        resid_hdu.header["EXTNAME"] = "SKYRESID"
        resid_hdu.header["COMMENT"] = ("Total removed from FF FLUX: "
                                       "OH scaling + PCA residual")
        out.append(resid_hdu)

        out.writeto(output_file, overwrite=True)

    logger.info("skySubtract: wrote %s", os.path.basename(output_file))

    if config.qa_plots:
        try:
            from llamas_pyjamas.Sky.skyQA import sky_subtraction_qa
            sky_subtraction_qa(ff_fits, output_file, sky_mask, config)
        except Exception as e:  # QA must never break the pipeline
            logger.warning("skySubtract: QA failed (%s)", e)

    return output_file


def subtract_sky_all_colors(ff_files, config=None):
    """Run :func:`subtract_sky_rss` over a list of per-colour FF files.

    Files that are not ``..._FF.fits`` (e.g. an already-processed ``_SKYSUB``)
    are skipped.  Failures on one colour are logged and do not abort the rest.

    Returns
    -------
    list[str]
        Paths to the SKYSUB files successfully written.
    """
    config = config or SkySubtractConfig()
    outputs = []
    for ff in ff_files:
        if not ff.endswith("_FF.fits"):
            logger.debug("skySubtract: skipping non-FF file %s",
                         os.path.basename(ff))
            continue
        try:
            outputs.append(subtract_sky_rss(ff, config=config))
        except Exception as e:
            logger.error("skySubtract: failed on %s (%s)",
                         os.path.basename(ff), e, exc_info=True)
    return outputs
