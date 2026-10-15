"""
Build the per-fibre sky reference used by the quick-look white light.

The quick look sums every fibre's pixels and maps the sums onto the IFU. Even
with a fresh bias and dark those sums carry a fixed fibre-to-fibre pattern of
~10 % (fibre throughput as seen by the night sky, which the LDLS lamp flat does
not reproduce). Dividing each fibre by its value in a sky-illuminated reference
removes that pattern (to 1-2 %), which is what lets faint sources show.

The reference is one number per (detector, trace fibre): the fibre's summed
flux in a sky-illuminated frame, normalised by the median over the detector, and
median-combined over frames. By default the frames are the night's twilight sky
flats (``PRODCATG = CAL.R-SKY``); ``--science`` accepts sky-dominated science
frames instead, with bright fibres (sources) masked out before the median.
Frames are bias-subtracted with the read-mode-matched master bias and the same
edge-row residual as the quick look; saturated detectors are dropped.

The file is keyed to the quick-look fibre-label cache
(``mastercalib/LLAMAS_quicklook_fiberimg.npz``): the quick look ignores a
reference built against a different cache (new master traces) with one warning.

Usage::

    python -m llamas_pyjamas.Postprocessing.build_quicklook_skyref FRAME.fits [...]
        [--out PATH] [--slow-bias F] [--fast-bias F] [--science] [--force]
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
from astropy.io import fits

from llamas_pyjamas.config import BIAS_DIR, CALIB_DIR
from llamas_pyjamas.File.llamasIO import trim_and_orient
from llamas_pyjamas.Postprocessing.build_quicklook_fiberimg import (
    DEFAULT_OUT as QUICKLOOK_CACHE, load_quicklook_fiberimg)

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
DEFAULT_SKYREF = os.path.join(CALIB_DIR, 'LLAMAS_quicklook_skyref.npz')
# A detector is dropped from a frame when more than this fraction of its fibre
# pixels is at or above SATURATION_DN (raw, before bias subtraction).
SATURATION_DN = 60000.0
MAX_SATURATED_FRACTION = 0.005
# ... or when its median fibre sum is below this (placeholder camera, closed
# shutter, or far too faint to define a reference).
MIN_MEDIAN_FLUX_DN = 2000.0
# --science: fibres this far from the detector median are sources, not sky.
SOURCE_HIGH = 1.3
SOURCE_LOW = 0.3


def _file_stamp(path):
    st = os.stat(path)
    return {'path': os.path.abspath(path), 'size': st.st_size, 'mtime': st.st_mtime}


def _detector_name(key):
    return '_'.join(key)


def _bias_file_for(primary_header, slow_bias=None, fast_bias=None):
    """Master bias for a frame's READ-MDE: caller-supplied paths first, else the
    packaged ``Bias/{slow,fast}_master_bias.fits``."""
    read_mode = str(primary_header.get('READ-MDE', '') or '').strip().upper()
    if read_mode == 'FAST':
        return fast_bias or os.path.join(BIAS_DIR, 'fast_master_bias.fits')
    return slow_bias or os.path.join(BIAS_DIR, 'slow_master_bias.fits')


def _frame_fibre_ratios(frame, bias_file, fibre_labels, dead_fiber_lut, science):
    """{(color, bench, side): ratio array (nfib, NaN where undefined)} for one frame."""
    # Imported here, not at module top: WhiteLightModule imports this module's
    # loader, so a top-level import would be circular.
    from llamas_pyjamas.Image.WhiteLightModule import (
        _detector_fibre_fluxes, _detector_hdus, _detector_id, estimate_residual_bias)

    ratios = {}
    with fits.open(frame, do_not_scale_image_data=True) as sci, \
            fits.open(bias_file, do_not_scale_image_data=True) as bias:
        bias_hdus = _detector_hdus(bias)
        for ext in sci[1:]:
            key = _detector_id(ext.header)
            if key is None or ext.data is None or key not in fibre_labels:
                continue
            bias_ext = bias_hdus.get(key)
            if bias_ext is None or bias_ext.data is None:
                logger.warning('%s: no master bias extension for %s; detector skipped',
                               os.path.basename(frame), _detector_name(key))
                continue
            fiberimg = fibre_labels[key]['fiberimg']
            nfib = fibre_labels[key]['nfibers']
            data = trim_and_orient(ext.data, ext.header)
            in_fibre = fiberimg >= 0
            saturated = np.mean(data[in_fibre] >= SATURATION_DN) if in_fibre.any() else 0.0
            if saturated > MAX_SATURATED_FRACTION:
                logger.warning('%s: %s has %.1f%% saturated fibre pixels; detector skipped',
                               os.path.basename(frame), _detector_name(key), 100 * saturated)
                continue
            data -= trim_and_orient(bias_ext.data, bias_ext.header)
            level = estimate_residual_bias(data, fiberimg)[0]
            _, _, flux, index = _detector_fibre_fluxes(
                fiberimg, nfib, f'{key[1]}{key[2]}', data, dead_fiber_lut,
                offset=level, return_index=True)
            flux = np.asarray(flux, dtype=float)
            if flux.size == 0 or np.median(flux) < MIN_MEDIAN_FLUX_DN:
                logger.warning('%s: %s median fibre flux %.0f DN too low; detector skipped',
                               os.path.basename(frame), _detector_name(key),
                               np.median(flux) if flux.size else 0.0)
                continue
            ratio = flux / np.median(flux)
            if science:
                ratio[(ratio > SOURCE_HIGH) | (ratio < SOURCE_LOW)] = np.nan
            full = np.full(nfib, np.nan)
            full[np.asarray(index, dtype=int)] = ratio
            ratios[key] = full
    return ratios


def build_quicklook_skyref(frames, out=DEFAULT_SKYREF, slow_bias=None, fast_bias=None,
                           science=False, force=False, fibre_cache=QUICKLOOK_CACHE):
    """Build the reference from *frames* and write it to *out*; returns *out*.

    Raises ``FileExistsError`` if *out* exists and not *force*, and
    ``ValueError`` when no detector of any frame yields a usable reference.
    """
    # See _frame_fibre_ratios for why this import is not at module top.
    from llamas_pyjamas.Image.WhiteLightModule import _load_dead_fiber_lut

    if not out.endswith('.npz'):
        out += '.npz'
    if os.path.exists(out) and not force:
        raise FileExistsError(f'{out} exists; use --force to overwrite')
    if not frames:
        raise ValueError('build_quicklook_skyref: no input frames')

    fibre_labels = load_quicklook_fiberimg(fibre_cache)
    dead_fiber_lut = _load_dead_fiber_lut()

    per_detector = {}
    bias_files = set()
    for frame in frames:
        bias_file = _bias_file_for(fits.getheader(frame, 0), slow_bias, fast_bias)
        if not os.path.isfile(bias_file):
            logger.warning('%s: master bias %s not found; frame skipped',
                           os.path.basename(frame), bias_file)
            continue
        bias_files.add(os.path.basename(bias_file))
        for key, ratio in _frame_fibre_ratios(frame, bias_file, fibre_labels,
                                              dead_fiber_lut, science).items():
            per_detector.setdefault(key, []).append(ratio)
        logger.info('%s: %d detectors used', os.path.basename(frame), len(per_detector))

    if not per_detector:
        raise ValueError('no usable detector in any input frame')

    arrays = {}
    detectors = {}
    for key, stack in per_detector.items():
        ref = np.nanmedian(np.stack(stack), axis=0)
        name = _detector_name(key)
        arrays[f'{name}_ref'] = ref.astype(np.float32)
        detectors[name] = {'nfibers': int(ref.size), 'nframes': len(stack),
                           'nvalid': int(np.isfinite(ref).sum())}
    meta = {
        'format_version': FORMAT_VERSION,
        'build_time_utc': datetime.now(timezone.utc).isoformat(),
        'mode': 'science' if science else 'twilight',
        'frames': [os.path.basename(f) for f in frames],
        'bias_files': sorted(bias_files),
        'fibre_cache': _file_stamp(fibre_cache),
        'detectors': detectors,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, meta=np.str_(json.dumps(meta)), **arrays)
    logger.info('Wrote reference for %d detectors to %s', len(detectors), out)
    return out


def _read_meta(npz):
    return json.loads(str(npz['meta']))


def load_quicklook_skyref(path):
    """``{(color, bench, side): ref}`` with ``ref`` a float array of length
    nfibers (NaN where the reference is undefined), plus the metadata under
    the key ``'meta'``."""
    out = {}
    with np.load(path) as npz:
        meta = _read_meta(npz)
        if meta.get('format_version') != FORMAT_VERSION:
            raise ValueError(f'{path}: unsupported format version {meta.get("format_version")}')
        for name in meta['detectors']:
            color, bench, side = name.split('_')
            out[(color, bench, side)] = npz[f'{name}_ref'].astype(float)
    out['meta'] = meta
    return out


def skyref_is_fresh(path, fibre_cache=QUICKLOOK_CACHE):
    """True if *path* exists and was built against the current fibre-label cache
    (same path, size and mtime). The reason is logged at INFO when not."""
    if not os.path.exists(path):
        logger.info('Quick-look sky reference %s not found', path)
        return False
    if not os.path.exists(fibre_cache):
        logger.info('Fibre-label cache %s not found; sky reference cannot be checked', fibre_cache)
        return False
    with np.load(path) as npz:
        recorded = _read_meta(npz).get('fibre_cache', {})
    current = _file_stamp(fibre_cache)
    if (recorded.get('size'), recorded.get('mtime')) != (current['size'], current['mtime']):
        logger.info('Quick-look sky reference stale: fibre-label cache has changed since it was built')
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Build the quick-look per-fibre sky reference from twilight sky flats '
                    '(or, with --science, sky-dominated science frames)')
    parser.add_argument('frames', nargs='+', help='Raw LLAMAS MEF frames')
    parser.add_argument('--out', default=DEFAULT_SKYREF, help=f'Output .npz (default: {DEFAULT_SKYREF})')
    parser.add_argument('--slow-bias', default=None, help='SLOW master bias (default: packaged)')
    parser.add_argument('--fast-bias', default=None, help='FAST master bias (default: packaged)')
    parser.add_argument('--science', action='store_true',
                        help='Frames are science exposures: mask bright fibres before combining')
    parser.add_argument('--force', action='store_true', help='Overwrite an existing reference')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show INFO logging')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(levelname)s: %(message)s')
    out = build_quicklook_skyref(args.frames, out=args.out, slow_bias=args.slow_bias,
                                 fast_bias=args.fast_bias, science=args.science, force=args.force)
    print(f'Quick-look sky reference written to {out}')


if __name__ == '__main__':
    main()
