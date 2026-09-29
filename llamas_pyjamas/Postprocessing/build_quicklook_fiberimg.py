"""
Build the quick-look fibre-image cache used for observer pointing checks.

QuickWhiteLightCube (Image/WhiteLightModule.py) currently unpickles all 24
master trace objects (mastercalib/LLAMAS_master_{color}_{bench}_{side}_traces.pkl,
~117 MB each, ~2.8 GB total) but only needs each detector's fibre-label image
(`fiberimg`, 2048x2048, -1 = no fibre, 0..nfibers-1 = fibre index) plus
nfibers, bench, side and channel. This script extracts just those into one
small compressed .npz file (fiberimg stored as int16), so a quick-look run can
skip the pickles entirely.

This cache is intentionally NOT yet wired into QuickWhiteLightCube: new master
traces are about to be delivered, and the cache should be generated from those.
Rerun this script whenever the master traces change; use
quicklook_cache_is_fresh() to detect a stale cache.

Usage:
    python -m llamas_pyjamas.Postprocessing.build_quicklook_fiberimg \
        [--calib-dir DIR] [--out PATH] [--force]
"""

import argparse
import gc
import json
import logging
import os
import pickle
from datetime import datetime, timezone

import numpy as np

from llamas_pyjamas.config import CALIB_DIR

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
DEFAULT_OUT = os.path.join(CALIB_DIR, 'LLAMAS_quicklook_fiberimg.npz')
COLORS = ('red', 'green', 'blue')
BENCHES = ('1', '2', '3', '4')
SIDES = ('A', 'B')
INT16_MAX = np.iinfo(np.int16).max


def trace_filename(color, bench, side):
    return f"LLAMAS_master_{color}_{bench}_{side}_traces.pkl"


def detector_key(color, bench, side):
    """Key prefix used inside the .npz for one detector, e.g. 'red_1_A'."""
    return f"{color}_{bench}_{side}"


def _file_stamp(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime


def _load_one(pkl_path, color, bench, side):
    """Unpickle one master trace and return only the quick-look fields."""
    with open(pkl_path, 'rb') as fp:
        traceobj = pickle.load(fp)

    obj_channel = str(traceobj.channel).lower()
    obj_bench = str(traceobj.bench)
    obj_side = str(traceobj.side).upper()
    if (obj_channel, obj_bench, obj_side) != (color, bench, side):
        logger.warning(f"{os.path.basename(pkl_path)}: trace object says "
                       f"{obj_channel} {obj_bench}{obj_side}, filename says "
                       f"{color} {bench}{side}; keyed by filename, but the object's "
                       f"bench/side are stored and used for the fibre-map lookup")

    fiberimg = np.asarray(traceobj.fiberimg)
    lo, hi = int(fiberimg.min()), int(fiberimg.max())
    if lo < -1 or hi >= INT16_MAX:
        raise ValueError(f"{os.path.basename(pkl_path)}: fiberimg labels [{lo}, {hi}] "
                         f"do not fit int16")

    entry = {
        'fiberimg': fiberimg.astype(np.int16),
        'nfibers': int(traceobj.nfibers),
        'bench': obj_bench,
        'side': obj_side,
        'channel': obj_channel,
    }
    del traceobj, fiberimg
    return entry


def build_quicklook_fiberimg(calib_dir=CALIB_DIR, out=DEFAULT_OUT, force=False):
    """Build the cache from the master trace pickles in calib_dir; returns out."""
    if not out.endswith('.npz'):
        out += '.npz'  # np.savez_compressed would append it anyway
    if os.path.exists(out) and not force:
        raise FileExistsError(f"{out} exists; use --force to overwrite")

    arrays = {}
    detectors = {}
    for color in COLORS:
        for bench in BENCHES:
            for side in SIDES:
                fname = trace_filename(color, bench, side)
                pkl_path = os.path.join(calib_dir, fname)
                if not os.path.exists(pkl_path):
                    logger.warning(f"Missing master trace {fname}; skipping")
                    continue

                size, mtime = _file_stamp(pkl_path)
                entry = _load_one(pkl_path, color, bench, side)
                key = detector_key(color, bench, side)
                arrays[f"{key}_fiberimg"] = entry['fiberimg']
                arrays[f"{key}_nfibers"] = np.int32(entry['nfibers'])
                arrays[f"{key}_bench"] = np.str_(entry['bench'])
                arrays[f"{key}_side"] = np.str_(entry['side'])
                arrays[f"{key}_channel"] = np.str_(entry['channel'])
                detectors[key] = {
                    'source': fname,
                    'size': size,
                    'mtime': mtime,
                    'nfibers': entry['nfibers'],
                    'shape': list(entry['fiberimg'].shape),
                }
                logger.info(f"{key}: {entry['nfibers']} fibres")
                del entry
                gc.collect()  # release the ~117 MB trace object before the next one

    if not detectors:
        raise FileNotFoundError(f"No master trace pickles found in {calib_dir}")

    meta = {
        'format_version': FORMAT_VERSION,
        'build_time_utc': datetime.now(timezone.utc).isoformat(),
        'calib_dir': os.path.abspath(calib_dir),
        'detectors': detectors,
    }
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(out, meta=np.str_(json.dumps(meta)), **arrays)
    logger.info(f"Wrote {len(detectors)} detectors to {out}")
    return out


def _read_meta(npz):
    return json.loads(str(npz['meta']))


def load_quicklook_fiberimg(path):
    """
    Load the cache. Returns {(color, bench, side): {'fiberimg', 'nfibers',
    'bench', 'side', 'channel'}}, with fiberimg as an int16 2D array.
    """
    cache = {}
    with np.load(path) as npz:
        meta = _read_meta(npz)
        if meta.get('format_version') != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported format version {meta.get('format_version')}")
        for key in meta['detectors']:
            color, bench, side = key.split('_')
            cache[(color, bench, side)] = {
                'fiberimg': npz[f"{key}_fiberimg"],
                'nfibers': int(npz[f"{key}_nfibers"]),
                'bench': str(npz[f"{key}_bench"]),
                'side': str(npz[f"{key}_side"]),
                'channel': str(npz[f"{key}_channel"]),
            }
    return cache


def quicklook_cache_is_fresh(path, calib_dir=CALIB_DIR):
    """
    True if the cache exists and matches the master trace pickles in calib_dir:
    every recorded pickle still exists with the same size and mtime, and no
    pickle has appeared that the cache does not cover.
    """
    if not os.path.exists(path):
        logger.warning(f"Quick-look cache {path} not found")
        return False
    with np.load(path) as npz:
        recorded = _read_meta(npz)['detectors']

    fresh = True
    for key, info in recorded.items():
        pkl_path = os.path.join(calib_dir, info['source'])
        if not os.path.exists(pkl_path):
            logger.warning(f"Quick-look cache stale: {info['source']} no longer exists")
            fresh = False
        elif _file_stamp(pkl_path) != (info['size'], info['mtime']):
            logger.warning(f"Quick-look cache stale: {info['source']} has changed")
            fresh = False

    recorded_sources = {info['source'] for info in recorded.values()}
    for color in COLORS:
        for bench in BENCHES:
            for side in SIDES:
                fname = trace_filename(color, bench, side)
                if fname not in recorded_sources and os.path.exists(os.path.join(calib_dir, fname)):
                    logger.warning(f"Quick-look cache stale: {fname} is not in the cache")
                    fresh = False
    return fresh


def main():
    parser = argparse.ArgumentParser(
        description="Build the quick-look fibre-image cache from the master trace pickles"
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=CALIB_DIR,
        help=f"Directory holding the master trace pickles (default: {CALIB_DIR})"
    )
    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT,
        help=f"Output .npz path (default: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Overwrite an existing cache file (default: False)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    build_quicklook_fiberimg(calib_dir=args.calib_dir, out=args.out, force=args.force)


if __name__ == "__main__":
    main()
