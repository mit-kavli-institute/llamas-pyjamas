"""
llamas_pyjamas.Bias.biasFirst
=============================
Bias-first frame preprocessing: make bias subtraction the FIRST data operation
of the pipeline, for every frame type (science, arc, flat).

For each real extension of a raw MEF this module:

1. subtracts the mode-appropriate (FAST/SLOW, via the READ-MDE header) 2D
   master bias, using the same selection/verification/fallback logic as
   extraction (``guiExtract.select_bias_for_extension``), and
2. measures and subtracts the residual per-frame DC pedestal from the
   unilluminated stripes above the topmost / below the bottommost fibre
   (``guiExtract._apply_edge_bias``), which tracks the nightly drift the
   static master bias misses,

then writes ``{base}_bias_corrected.fits`` with ``BIASSUB``/``BIASSRC``/
``BIASLVL`` and ``EDGE*`` keywords stamped in each corrected extension header.
Downstream stages (trace generation, extraction) detect those keywords and
skip their internal bias handling, so the correction is applied exactly once.

Why this matters: the flat-field division previously operated on frames that
still carried their ~500 DN bias pedestal.  Percent-level spectral structure
in the pixel flat was thereby multiplied by the pedestal instead of the
signal, imprinting tens-of-DN fake emission lines (discovered as spurious
blue "sky lines" that no sky model could subtract).  With the pedestal
removed first, the same flat structure perturbs the signal by <1 DN.
"""

import os
import logging

import numpy as np
from astropy.io import fits

from llamas_pyjamas.config import CALIB_DIR, BIAS_DIR

logger = logging.getLogger(__name__)


def resolve_master_bias_file(primary_header, slow_bias=None, fast_bias=None):
    """Pick the master-bias file matching the frame's READ-MDE.

    Mirrors the selection logic in ``GUI_extract`` (caller-supplied paths take
    priority; falls back to the package BIAS_DIR defaults). Returns a path or
    None (extension-level fallback bias will then be used).
    """
    read_mode = (primary_header.get('READ-MDE') or '').strip().upper() or None
    if slow_bias is not None or fast_bias is not None:
        if read_mode == 'FAST' and fast_bias is not None:
            return fast_bias
        if slow_bias is not None:
            if read_mode == 'FAST':
                logger.warning(
                    "READ-MDE=FAST but no fast_bias supplied; using slow bias")
            return slow_bias
        return fast_bias

    candidate = os.path.join(
        BIAS_DIR,
        'fast_master_bias.fits' if read_mode == 'FAST' else 'slow_master_bias.fits')
    return candidate if os.path.exists(candidate) else None


def bias_correct_frame(fits_file, output_dir, slow_bias=None, fast_bias=None,
                       trace_dir=None, mastercalib_trace_dir=None,
                       edge_bias=None, suffix='_bias_corrected'):
    """Write a bias-corrected copy of ``fits_file`` (master bias + edge DC).

    This must be the FIRST data operation applied to any raw frame (science,
    arc or flat), before flat-field division, tracing or extraction.

    Parameters
    ----------
    fits_file : str
        Raw MEF path.
    output_dir : str
        Directory for the ``{base}{suffix}.fits`` output.
    slow_bias, fast_bias : str, optional
        Master bias FITS paths; selected per frame via READ-MDE.
    trace_dir : str, optional
        Directory with the night's trace pickles (for the edge/fallback
        masks). Extensions without a matching trace fall back to
        ``mastercalib_trace_dir`` (default CALIB_DIR) — the stripe mask only
        needs the approximate fibre-bundle boundary, so master traces are
        adequate before the night's own traces exist.
    edge_bias : dict, optional
        Edge-DC configuration forwarded to ``_apply_edge_bias``
        (None => Tier 1 stripe mask, enabled).
    suffix : str
        Output filename suffix.

    Returns
    -------
    str : path to the bias-corrected MEF.
    """
    # Imported lazily: guiExtract imports the Bias package at module scope.
    from llamas_pyjamas.GUI.guiExtract import (select_bias_for_extension,
                                               _apply_edge_bias,
                                               get_trace_file,
                                               is_placeholder_camera)
    import cloudpickle

    if mastercalib_trace_dir is None:
        mastercalib_trace_dir = CALIB_DIR
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(fits_file))[0]
    output_file = os.path.join(output_dir, f"{base}{suffix}.fits")

    with fits.open(fits_file) as hdul:
        masterbiasfile = resolve_master_bias_file(hdul[0].header,
                                                  slow_bias=slow_bias,
                                                  fast_bias=fast_bias)
        logger.info(f"bias-first: {os.path.basename(fits_file)} -> "
                    f"master bias = {masterbiasfile}")

        out_hdus = [hdul[0].copy()]
        n_done = n_skipped = 0
        tracer_cache = {}

        for ext in range(1, len(hdul)):
            hdu = hdul[ext]
            if hdu.data is None:
                out_hdus.append(hdu.copy())
                continue

            header = hdu.header.copy()

            # Parse camera identifiers (COLOR keywords with CAM_NAME fallback)
            if 'COLOR' in header:
                color = header['COLOR'].lower()
                bench = header['BENCH']
                side = header['SIDE']
            else:
                camname = header.get('CAM_NAME', '')
                parts = camname.split('_')
                color = parts[1].lower() if len(parts) >= 2 else ''
                bench = parts[0][0] if parts else ''
                side = parts[0][1] if parts and len(parts[0]) >= 2 else ''

            # Placeholder cameras (constant fill) are passed through untouched
            if is_placeholder_camera(hdu.data):
                header['BIASSUB'] = (False, 'Placeholder extension - no bias applied')
                out_hdus.append(fits.ImageHDU(data=hdu.data, header=header,
                                              name=hdu.name))
                n_skipped += 1
                continue

            # Idempotence: never correct twice
            if header.get('BIASSUB', False):
                logger.info(f"bias-first: ext {ext} ({bench}{side} {color}) "
                            f"already BIASSUB — passing through")
                out_hdus.append(hdu.copy())
                n_skipped += 1
                continue

            # Resolve the tracer (night traces preferred, mastercalib fallback)
            tracer = None
            key = (color, str(bench), str(side))
            if key in tracer_cache:
                tracer = tracer_cache[key]
            else:
                for tdir in (trace_dir, mastercalib_trace_dir):
                    if not tdir:
                        continue
                    try:
                        tfile = get_trace_file(color, bench, side, tdir)
                        with open(tfile, 'rb') as tf:
                            tracer = cloudpickle.load(tf)
                        break
                    except (FileNotFoundError, Exception) as exc:
                        logger.debug(f"bias-first: no trace in {tdir} for "
                                     f"{bench}{side} {color}: {exc}")
                tracer_cache[key] = tracer
            if tracer is None:
                logger.warning(f"bias-first: no tracer for {bench}{side} {color} "
                               f"— edge-DC and inter-fibre fallback unavailable")

            data = hdu.data.astype(float)

            # 1) master bias (with verification + inter-fibre/test-region fallback)
            bias = select_bias_for_extension(data, header, tracer,
                                             masterbiasfile, bench, side, color)
            bias_data = bias.data
            # Align shapes (raw frames may carry an extra row vs the master bias)
            ny = min(data.shape[0], bias_data.shape[0])
            nx = min(data.shape[1], bias_data.shape[1])
            corrected = data.copy()
            corrected[:ny, :nx] = data[:ny, :nx] - bias_data[:ny, :nx]
            if data.shape != bias_data.shape:
                logger.info(f"bias-first: shape mismatch ext {ext} "
                            f"(frame {data.shape} vs bias {bias_data.shape}); "
                            f"corrected overlap only")

            header['BIASSUB'] = (True, 'True = bias was subtracted (bias-first)')
            header['BIASSRC'] = (bias.header.get('BIASSRC', 'master_bias'),
                                 'Source of bias subtracted')
            header['BIASLVL'] = (float(np.nanmedian(bias_data)),
                                 'Median bias level subtracted (DN)')

            # 2) per-frame edge DC from the unilluminated stripes.
            # The stripe mask is built at the tracer's detector geometry
            # (2048x2048); raw frames may carry an extra row. The DC is a
            # scalar, so measure it on the tracer-compatible sub-frame and
            # subtract it from the full frame.
            if tracer is not None:
                fib = getattr(tracer, 'fiberimg', None)
                tshape = fib.shape if fib is not None else (2048, 2048)
                ny2 = min(corrected.shape[0], tshape[0])
                nx2 = min(corrected.shape[1], tshape[1])
                _sub, edge_stats = _apply_edge_bias(
                    corrected[:ny2, :nx2].copy(), tracer, header,
                    bench, side, color, edge_bias)
                corrected = corrected - float(edge_stats.get('edge_dc', 0.0))

            out_hdus.append(fits.ImageHDU(data=corrected.astype(np.float32),
                                          header=header, name=hdu.name))
            n_done += 1

        fits.HDUList(out_hdus).writeto(output_file, overwrite=True)

    logger.info(f"bias-first: corrected {n_done} extensions "
                f"({n_skipped} passed through) -> {output_file}")
    print(f"Bias-first correction: {n_done} extensions corrected, "
          f"{n_skipped} passed through -> {os.path.basename(output_file)}")
    return output_file
