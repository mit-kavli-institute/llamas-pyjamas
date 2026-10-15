"""
Build the quick-look fibre-image cache used for observer pointing checks.

QuickWhiteLightCube (Image/WhiteLightModule.py) only needs each detector's
fibre-label image (`fiberimg`, 2048x2048, -1 = no fibre, 0..nfibers-1 = fibre
index) plus nfibers, bench, side and channel, not the full master trace objects
(mastercalib/LLAMAS_master_{color}_{bench}_{side}_traces.pkl, ~117 MB each,
~2.8 GB total). This script extracts just those into one small compressed .npz
file (fiberimg stored as int16), so a quick-look run can skip the pickles.

QuickWhiteLightCube uses the cache whenever quicklook_cache_is_fresh() says it
matches the pickles, and falls back to the pickles (with one warning) otherwise.
Rerun this script with --force whenever the master traces change. mastercalib/ is
not in git, so each machine builds its own copy (~6 s).

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
from llamas_pyjamas.Utils.utils import find_trace_pickle

logger = logging.getLogger(__name__)

FORMAT_VERSION = 1
DEFAULT_OUT = os.path.join(CALIB_DIR, 'LLAMAS_quicklook_fiberimg.npz')
# Optional companion cache (--profiles): each detector's spatial fibre profile,
# normalised to unit sum per fibre and column, as float16. QuickWhiteLightCube
# uses it to profile-weight the fibre sums of short exposures.
DEFAULT_PROFILES = os.path.join(CALIB_DIR, 'LLAMAS_quicklook_profiles.npz')
COLORS = ('red', 'green', 'blue')
BENCHES = ('1', '2', '3', '4')
SIDES = ('A', 'B')
INT16_MAX = np.iinfo(np.int16).max


def trace_path(color, bench, side, calib_dir):
    """Path of a detector's master trace pickle under either shipped filename form
    (LLAMAS_master_... or LLAMAS_...), or None if there is none."""
    try:
        return find_trace_pickle(color, bench, side, calib_dir)
    except FileNotFoundError:
        return None


def detector_key(color, bench, side):
    """Key prefix used inside the .npz for one detector, e.g. 'red_1_A'."""
    return f"{color}_{bench}_{side}"


def _file_stamp(path):
    st = os.stat(path)
    return st.st_size, st.st_mtime


def normalised_profile(profimg, fiberimg, nfib):
    """Spatial fibre profile normalised to unit sum over each fibre's pixels in
    each column (float32, zero off-fibre and where the column has no profile).

    With this normalisation a profile-weighted fibre sum
    ``sum(P * d) / sum(P * P)`` (Horne weights for constant variance) returns the
    column-summed flux for a fibre whose pixels follow the profile, so it can
    replace the plain sum in the read-noise-limited regime.
    """
    prof = np.clip(np.asarray(profimg, dtype=np.float64), 0.0, None)
    nrow, ncol = prof.shape
    labels = np.asarray(fiberimg).ravel().astype(np.int64)
    keep = (labels >= 0) & (labels < nfib)
    cols = np.tile(np.arange(ncol, dtype=np.int64), nrow)[keep]
    idx = labels[keep] * ncol + cols
    colsum = np.bincount(idx, weights=prof.ravel()[keep], minlength=nfib * ncol)
    norm = np.zeros(prof.size)
    denom = colsum[idx]
    good = denom > 0
    norm[np.flatnonzero(keep)[good]] = prof.ravel()[keep][good] / denom[good]
    return norm.reshape(prof.shape).astype(np.float32)


def profile_column_norm(profile, fiberimg, nfib):
    """``sum(P^2)`` of the normalised profile over each fibre's pixels in each
    column, shape ``(nfib, ncol)`` float32: the denominator of the per-column
    Horne estimate, precomputed once because it depends only on the profile."""
    prof = np.asarray(profile, dtype=np.float64).ravel()
    labels = np.asarray(fiberimg).ravel().astype(np.int64)
    ncol = np.asarray(fiberimg).shape[1]
    px = np.flatnonzero((labels >= 0) & (labels < nfib))
    idx = labels[px] * ncol + (px % ncol)
    den = np.bincount(idx, weights=prof[px] * prof[px], minlength=nfib * ncol)
    return den.reshape(nfib, ncol).astype(np.float32)


def _load_one(pkl_path, color, bench, side, profiles=False):
    """Unpickle one master trace and return only the quick-look fields
    (plus the normalised profile when *profiles*)."""
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
    if profiles:
        profile = normalised_profile(traceobj.profimg, fiberimg, entry['nfibers'])
        entry['profile'] = profile.astype(np.float16)
        entry['pden'] = profile_column_norm(profile, fiberimg, entry['nfibers'])
    del traceobj, fiberimg
    return entry


def build_quicklook_fiberimg(calib_dir=CALIB_DIR, out=DEFAULT_OUT, force=False,
                             profiles=False, profiles_out=DEFAULT_PROFILES):
    """Build the cache from the master trace pickles in calib_dir; returns out.

    With *profiles* a second file, *profiles_out*, holds each detector's
    normalised spatial profile under the same metadata, so
    ``quicklook_cache_is_fresh(profiles_out)`` checks it against the pickles too.
    """
    if not out.endswith('.npz'):
        out += '.npz'  # np.savez_compressed would append it anyway
    if os.path.exists(out) and not force:
        raise FileExistsError(f"{out} exists; use --force to overwrite")
    if profiles and os.path.exists(profiles_out) and not force:
        raise FileExistsError(f"{profiles_out} exists; use --force to overwrite")

    arrays = {}
    profile_arrays = {}
    detectors = {}
    for color in COLORS:
        for bench in BENCHES:
            for side in SIDES:
                pkl_path = trace_path(color, bench, side, calib_dir)
                if pkl_path is None:
                    logger.warning(f"Missing master trace for {color} {bench}{side}; skipping")
                    continue
                fname = os.path.basename(pkl_path)

                size, mtime = _file_stamp(pkl_path)
                entry = _load_one(pkl_path, color, bench, side, profiles=profiles)
                key = detector_key(color, bench, side)
                if profiles:
                    profile_arrays[f"{key}_profile"] = entry['profile']
                    profile_arrays[f"{key}_pden"] = entry['pden']
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
    if profiles:
        np.savez_compressed(profiles_out, meta=np.str_(json.dumps(meta)), **profile_arrays)
        logger.info(f"Wrote {len(profile_arrays)} normalised profiles to {profiles_out}")
    return out


def _read_meta(npz):
    return json.loads(str(npz['meta']))


class QuicklookProfiles:
    """Lazy reader for the normalised-profile cache (``--profiles``).

    ``get((color, bench, side))`` returns that detector's ``(profile, pden)``
    as float32 arrays (the normalised profile and its per-fibre, per-column
    ``sum(P^2)``), or None; arrays are decompressed on demand so only the
    detector being summed is in memory. Use as a context manager or call
    ``close()``.
    """

    def __init__(self, path):
        self.path = path
        self._npz = np.load(path)
        meta = _read_meta(self._npz)
        if meta.get('format_version') != FORMAT_VERSION:
            self._npz.close()
            raise ValueError(f"{path}: unsupported format version {meta.get('format_version')}")
        self.keys = {tuple(k.split('_')) for k in meta['detectors']}

    def get(self, key):
        if key not in self.keys:
            return None
        name = detector_key(*key)
        return (self._npz[f"{name}_profile"].astype(np.float32),
                self._npz[f"{name}_pden"].astype(np.float32))

    def close(self):
        self._npz.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


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

    The reasons are logged at INFO (up to one line per pickle); callers report a
    stale cache once.
    """
    if not os.path.exists(path):
        logger.info(f"Quick-look cache {path} not found")
        return False
    with np.load(path) as npz:
        recorded = _read_meta(npz)['detectors']

    fresh = True
    for key, info in recorded.items():
        pkl_path = os.path.join(calib_dir, info['source'])
        if not os.path.exists(pkl_path):
            logger.info(f"Quick-look cache stale: {info['source']} no longer exists")
            fresh = False
        elif _file_stamp(pkl_path) != (info['size'], info['mtime']):
            logger.info(f"Quick-look cache stale: {info['source']} has changed")
            fresh = False

    recorded_sources = {info['source'] for info in recorded.values()}
    for color in COLORS:
        for bench in BENCHES:
            for side in SIDES:
                pkl_path = trace_path(color, bench, side, calib_dir)
                if pkl_path is not None and os.path.basename(pkl_path) not in recorded_sources:
                    logger.info(f"Quick-look cache stale: {os.path.basename(pkl_path)} is not in the cache")
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
    parser.add_argument(
        "--profiles",
        action="store_true",
        default=False,
        help="Also write the normalised fibre-profile cache used to profile-weight "
             f"short exposures (default: False; file {DEFAULT_PROFILES})"
    )
    parser.add_argument(
        "--profiles-out",
        type=str,
        default=DEFAULT_PROFILES,
        help=f"Output .npz path for --profiles (default: {DEFAULT_PROFILES})"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    build_quicklook_fiberimg(calib_dir=args.calib_dir, out=args.out, force=args.force,
                             profiles=args.profiles, profiles_out=args.profiles_out)


if __name__ == "__main__":
    main()
