"""
llamas_pyjamas.Bias.biasFirst
=============================
Bias-first frame preprocessing: make bias (and dark) subtraction the FIRST data
operation of the pipeline, for every frame type (science, arc, flat).

For each real extension of a raw MEF this module, working in pipeline
orientation (the DATASEC trim and colour flip of ``process_fits_by_color``):

1. subtracts the mode-appropriate (FAST/SLOW, via the READ-MDE header) 2D
   master bias, using the same selection/verification/fallback logic as
   extraction (``guiExtract.select_bias_for_extension``);
2. for science, flux-standard and sky frames with a matched master dark,
   subtracts ``scale x master dark`` (``Bias.darkLlamas``), the dark extension
   matched and confirmed by header exactly like the bias;
3. measures and subtracts the residual per-frame DC pedestal from the
   unilluminated stripes above the topmost / below the bottommost fibre
   (``guiExtract._apply_edge_bias``). Without a dark this is mostly dark
   current; after a matched dark it is a small residual (bias drift since the
   darks, and what a scaled dark leaves) and a check on the dark;

then flips the result back and writes ``{base}_bias_corrected.fits`` in raw
orientation, so downstream stages read it exactly as they read a raw frame.
``BIASSUB``/``BIASSRC``/``BIASLVL``, ``DARK*`` and ``EDGE*`` keywords are stamped
in each corrected extension header, and the primary header records which bias
and dark were used (``BFVER``, ``BIASID``, ``DARKID``, ``DARKSCL``) so a resumed
run can tell when a frame is out of date. Downstream stages detect ``BIASSUB``
and skip their internal bias handling, so the correction is applied exactly once.

Why this matters: dividing the pixel flat into a frame that still carries its
~500 DN bias pedestal multiplies percent-level flat structure by the pedestal
instead of the signal, imprinting tens-of-DN fake emission lines that no sky
model can subtract. With the pedestal removed first, the same structure
perturbs the signal by <1 DN.

Orientation: the master bias, the master dark and the tracer masks all come in
pipeline orientation, so the frame is brought into that orientation first.
(Before BFVER 2 the raw frame was corrected directly, which mirrored the bias
and the stripe mask on green and blue cameras.)
"""

import os
import logging

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

from llamas_pyjamas.config import CALIB_DIR, BIAS_DIR
from llamas_pyjamas.Bias.biasChecking import build_topbottom_stripe_mask
from llamas_pyjamas.Bias.darkLlamas import (bias_id, dark_scale, is_placeholder,
                                            load_master_dark_camera, noise_factor,
                                            read_mode)
from llamas_pyjamas.File.llamasIO import to_raw, trim_and_orient

logger = logging.getLogger(__name__)

#: Bias-first version, stamped as BFVER. 2: pipeline orientation + master dark.
BFVER = 2

# Dark checks on the unlit stripes (the reference notebook's, see Bias/README.md).
STRIPE_LEVEL_FLAG = 5.0     # DN: residual stripe level after bias + dark
NOISE_RATIO_FLAG = 1.1      # x sqrt(1 + c/N), the rise expected from an N-frame master
LOW_PIXELS_FLAG = 500       # rise in pixels below level - 6 sigma
FALLBACK_STRIPE_ROWS = 15   # without traces: this many rows at the bottom and top


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


def expected_provenance(primary_header, slow_bias=None, fast_bias=None, dark=None, scale=None):
    """The provenance stamps bias_correct_frame will write for a frame (dict)."""
    bias_file = resolve_master_bias_file(primary_header, slow_bias, fast_bias)
    return {'BFVER': BFVER, 'BIASID': bias_id(bias_file),
            'DARKID': dark.darkid if dark is not None else 'none',
            'DARKSCL': round(float(scale), 4) if dark is not None else 0.0}


def provenance_of(primary_header):
    """The provenance stamps in a product's primary header (missing keys -> None)."""
    return {'BFVER': primary_header.get('BFVER'), 'BIASID': primary_header.get('BIASID'),
            'DARKID': primary_header.get('DARKID'),
            'DARKSCL': (round(float(primary_header['DARKSCL']), 4)
                        if primary_header.get('DARKSCL') is not None else None)}


def _stripe_stats(img, mask):
    """(level, noise, low-pixel count) on the unlit stripes, as the reference notebook.

    level: median of the stripe pixels; noise: 4-sigma-clipped std there; low
    pixels: pixels anywhere in the image below level - 6 noise.
    """
    if mask is None:
        ny = img.shape[0]
        mask = np.zeros(img.shape, bool)
        mask[np.r_[0:FALLBACK_STRIPE_ROWS, ny - FALLBACK_STRIPE_ROWS:ny]] = True
    vals = img[mask]
    level = float(np.median(vals))
    noise = float(sigma_clipped_stats(vals, sigma=4)[2])
    return level, noise, int(np.count_nonzero(img < level - 6.0 * noise))


def bias_correct_frame(fits_file, output_dir, slow_bias=None, fast_bias=None,
                       trace_dir=None, mastercalib_trace_dir=None,
                       edge_bias=None, suffix='_bias_corrected',
                       dark=None, dark_note=''):
    """Write a bias-corrected copy of ``fits_file`` (master bias, dark, edge DC).

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
    dark : darkLlamas.MasterDark, optional
        The matched master dark (science, flux-standard and sky frames only).
        The scale is the frame's DEXPTIME over the darks'.
    dark_note : str
        Why there is no dark, stamped in DARKSUB's comment when ``dark`` is None.

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
    edge_cfg = edge_bias or {}
    dmin = float(edge_cfg.get('min_distance', 20.0))

    base = os.path.splitext(os.path.basename(fits_file))[0]
    output_file = os.path.join(output_dir, f"{base}{suffix}.fits")

    with fits.open(fits_file) as hdul:
        primary_hdr = hdul[0].header
        masterbiasfile = resolve_master_bias_file(primary_hdr,
                                                  slow_bias=slow_bias,
                                                  fast_bias=fast_bias)
        frame_mode = read_mode(primary_hdr)
        scale = dark_scale(primary_hdr, dark.dexptime, dark.rexptime) if dark else None
        logger.info(f"bias-first: {os.path.basename(fits_file)} -> "
                    f"master bias = {masterbiasfile}"
                    + (f", master dark = {os.path.basename(dark.path)} x {scale:.4f}"
                       if dark else ""))

        out_hdus = [hdul[0].copy()]
        prov = expected_provenance(primary_hdr, slow_bias, fast_bias, dark, scale)
        prim = out_hdus[0].header
        prim['BFVER'] = (prov['BFVER'], 'Bias-first version (2: true orientation + dark)')
        prim['BIASFILE'] = (os.path.abspath(masterbiasfile) if masterbiasfile else 'none',
                            'Master bias file')
        prim['BIASID'] = (prov['BIASID'], 'Master bias identity (name, size, mtime)')
        prim['DARKID'] = (prov['DARKID'], 'Master dark identity (none = no dark)')
        prim['DARKSCL'] = (prov['DARKSCL'], 'Scale applied to the master dark')
        n_done = n_skipped = n_dark = 0
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

            # Placeholder cameras (constant fill: 1.0, or 0.0 / COMMENT marker)
            # are passed through untouched.
            if is_placeholder_camera(hdu.data) or is_placeholder(hdu):
                header['BIASSUB'] = (False, 'Placeholder extension - no bias applied')
                header['DARKSUB'] = (False, 'Placeholder extension')
                header['DARKWHY'] = ('dead camera in this frame', 'Why no dark')
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
            tracer, tracer_src = None, 'none'
            key = (color, str(bench), str(side))
            if key in tracer_cache:
                tracer, tracer_src = tracer_cache[key]
            else:
                for tdir, src in ((trace_dir, 'night'), (mastercalib_trace_dir, 'mastercalib')):
                    if not tdir:
                        continue
                    try:
                        tfile = get_trace_file(color, bench, side, tdir)
                        with open(tfile, 'rb') as tf:
                            tracer = cloudpickle.load(tf)
                        tracer_src = src
                        break
                    except (FileNotFoundError, Exception) as exc:
                        logger.debug(f"bias-first: no trace in {tdir} for "
                                     f"{bench}{side} {color}: {exc}")
                tracer_cache[key] = (tracer, tracer_src)
            if tracer is None:
                logger.warning(f"bias-first: no tracer for {bench}{side} {color} "
                               f"— edge-DC and inter-fibre fallback unavailable")
            header['EDGETRC'] = (tracer_src, 'Traces used for the bias-first stripe mask')

            # Work in pipeline orientation: the bias, dark and tracer masks are.
            image = np.asarray(trim_and_orient(hdu.data, header), dtype=np.float64)

            # 1) master bias (with verification + inter-fibre/test-region fallback)
            bias = select_bias_for_extension(image, header, tracer,
                                             masterbiasfile, bench, side, color)
            bias_data = np.asarray(bias.data, dtype=np.float64)
            # Align shapes (an extension without DATASEC keeps any extra row)
            ny = min(image.shape[0], bias_data.shape[0])
            nx = min(image.shape[1], bias_data.shape[1])
            image[:ny, :nx] -= bias_data[:ny, :nx]
            if image.shape != bias_data.shape:
                logger.info(f"bias-first: shape mismatch ext {ext} "
                            f"(frame {image.shape} vs bias {bias_data.shape}); "
                            f"corrected overlap only")

            header['BIASSUB'] = (True, 'True = bias was subtracted (bias-first)')
            header['BIASSRC'] = (bias.header.get('BIASSRC', 'master_bias'),
                                 'Source of bias subtracted')
            header['BIASLVL'] = (float(np.nanmedian(bias_data)),
                                 'Median bias level subtracted (DN)')

            stripes = None
            if tracer is not None and getattr(tracer, 'fiberimg', None) is not None \
                    and tracer.fiberimg.shape == image.shape:
                stripes = build_topbottom_stripe_mask(tracer, image.shape, min_distance=dmin)
                if int(np.count_nonzero(stripes)) < int(edge_cfg.get('min_pixels', 500)):
                    stripes = None

            # 2) master dark, matched and confirmed by header like the bias
            if dark is None:
                header['DARKSUB'] = (False, 'No dark subtracted (see DARKWHY)')
                header['DARKWHY'] = (dark_note or 'no matched dark', 'Why no dark')
            else:
                dark_data, info = load_master_dark_camera(dark.path, header,
                                                          darkid=dark.darkid, mode=frame_mode)
                if dark_data is not None and dark_data.shape != image.shape:
                    dark_data, info = None, {'reason': f'shape {dark_data.shape} vs '
                                                       f'frame {image.shape}'}
                if dark_data is None:
                    logger.error(f"bias-first: no dark for {bench}{side} {color} in "
                                 f"{os.path.basename(fits_file)}: {info['reason']}")
                    header['DARKSUB'] = (False, 'No dark subtracted (see DARKWHY)')
                    header['DARKWHY'] = (info['reason'], 'Why no dark')
                else:
                    before = _stripe_stats(image, stripes)
                    image -= scale * dark_data
                    after = _stripe_stats(image, stripes)
                    c_over_n = noise_factor(info['ncombine'], info['method'])
                    flags = []
                    if abs(after[0]) > STRIPE_LEVEL_FLAG:
                        flags.append('STRIPE-LEVEL')
                    ratio = after[1] / before[1] if before[1] > 0 else 1.0
                    if ratio > NOISE_RATIO_FLAG * np.sqrt(1.0 + c_over_n):
                        flags.append('NOISE-UP')
                    if after[2] - before[2] > LOW_PIXELS_FLAG:
                        flags.append('NEGATIVE-PIXELS')
                    header['DARKSUB'] = (True, 'Master dark subtracted (bias-first)')
                    header['DARKFILE'] = (os.path.abspath(dark.path), 'Master dark file')
                    header['DARKID'] = (dark.darkid, 'Master dark identity')
                    header['DARKSCL'] = (float(scale), 'Scale applied to the master dark')
                    header['DARKNCMB'] = (int(info['ncombine']), 'Dark frames in the master')
                    header['DARKMETH'] = (str(info['method']), 'How the darks were combined')
                    header['DARKLVL'] = (float(np.median(scale * dark_data)),
                                         'Median dark subtracted (DN)')
                    header['EDGEDC0'] = (before[0], 'Stripe level before the dark (DN)')
                    header['DARKRES'] = (after[0], 'Stripe level after the dark (DN)')
                    header['DARKNRAT'] = (ratio, 'Stripe noise after/before the dark')
                    header['DARKNLOW'] = (after[2] - before[2], 'Rise in pixels < level-6sig')
                    header['DARKFLAG'] = (','.join(flags) or 'none', 'Dark checks flagged')
                    if flags:
                        logger.warning(f"bias-first: dark check {','.join(flags)} on "
                                       f"{bench}{side} {color} in {os.path.basename(fits_file)} "
                                       f"(stripe level {after[0]:.1f} DN, noise ratio "
                                       f"{ratio:.2f}, low pixels {after[2] - before[2]:+d})")
                    n_dark += 1

            # 3) per-frame edge DC from the unilluminated stripes (now a residual
            # where a dark was subtracted). Without a tracer the EDGE* keywords are
            # not written, so extraction measures it with the night's traces.
            if tracer is not None:
                image, _edge_stats = _apply_edge_bias(image, tracer, header,
                                                      bench, side, color, edge_bias)

            out_hdus.append(fits.ImageHDU(data=to_raw(image, header, hdu.data),
                                          header=header, name=hdu.name))
            n_done += 1

        tmp = output_file + '.tmp'
        fits.HDUList(out_hdus).writeto(tmp, overwrite=True)
        os.replace(tmp, output_file)

    logger.info(f"bias-first: corrected {n_done} extensions "
                f"({n_skipped} passed through, dark on {n_dark}) -> {output_file}")
    print(f"Bias-first correction: {n_done} extensions corrected"
          + (f" (dark on {n_dark})" if dark else "")
          + f", {n_skipped} passed through -> {os.path.basename(output_file)}")
    return output_file


def dark_summary(path):
    """Per-extension dark outcome of a bias-corrected file, for the report.

    Returns a list of ``(camera, applied, detail)``: detail is the flags when
    applied, else the reason.
    """
    rows = []
    with fits.open(path, memmap=True) as hdul:
        for hdu in hdul[1:]:
            h = hdu.header
            cam = f"{h.get('BENCH', '?')}{h.get('SIDE', '?')} {str(h.get('COLOR', '?')).lower()}"
            if 'DARKSUB' not in h:
                continue
            if h['DARKSUB']:
                rows.append((cam, True, h.get('DARKFLAG', 'none')))
            else:
                rows.append((cam, False, h.get('DARKWHY', 'no dark')))
    return rows
