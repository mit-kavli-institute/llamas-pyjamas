"""Combine the exposures of one LDLS lamp-flat set before tracing and flat fielding.

Each lamp-flat setting is taken as a short set of identical exposures (three on
20261015_16). Combining them rejects cosmic rays without L.A.Cosmic, which must never run
on a lamp flat: it takes bright, sharp fibre cores for cosmic rays (Masking/README.md). A
median of three or more frames rejects any hit in one frame; two frames use the 5-sigma
pair-minimum of the master dark (Bias/darkLlamas.combine_frames).

Combining does not fix saturated pixels, which stay saturated in every frame.

The set of a configured flat is found from the raw headers: the same OBJECT, requested
exposure (REXPTIME), read mode, date and lamp, selector and filter-wheel states, taken
within ``window_s`` of one another. A comma-separated list in ``{colour}_flat_file`` gives
the set explicitly; ``lamp_flat_combine = false`` uses the single configured frame.
"""

import glob
import logging
import os
from datetime import datetime

import numpy as np
from astropy.io import fits

from llamas_pyjamas.Bias import camera_of_header, match_camera_hdu
from llamas_pyjamas.Bias.darkLlamas import combine_frames
from llamas_pyjamas.Utils.detectorProps import props_for_header
from llamas_pyjamas.Utils.recipes import stamp_recipe

logger = logging.getLogger(__name__)

FLAT_KEYS = {'red': 'red_flat_file', 'green': 'green_flat_file', 'blue': 'blue_flat_file'}
#: Header keys that must agree for two exposures to belong to one set (plus every *STATE).
SET_KEYS = ('OBJECT', 'REXPTIME', 'READ-MDE', 'DATE-OBS')
#: Pixels above this (DN, bias-corrected) set each frame's scale to the first.
LIT_DN = 1000.0
DEFAULT_GAIN = 1.0
DEFAULT_READNOISE = 2.5


def _setting(header):
    """The header values that define a lamp-flat setting."""
    keys = list(SET_KEYS) + sorted(k for k in header if 'STATE' in k)
    return tuple((k, str(header.get(k, '')).strip()) for k in keys)


def _start_time(header):
    """Exposure start from DATE-OBS and UTC, or None."""
    try:
        return datetime.strptime(f"{header['DATE-OBS']} {header['UTC']}", '%Y-%m-%d %H:%M:%S')
    except (KeyError, ValueError):
        return None


def find_lamp_flat_set(path, window_s=600.0):
    """The raw exposures of ``path``'s lamp-flat set, in time order (``path`` included).

    Candidates are the ``LLAMAS_*_mef.fits`` files beside ``path`` with the same setting
    (:data:`SET_KEYS` and every ``*STATE`` key). The set is the run of them containing
    ``path`` whose consecutive start times are no more than ``window_s`` apart.
    """
    target = fits.getheader(path, 0)
    setting = _setting(target)
    members = []
    for cand in glob.glob(os.path.join(os.path.dirname(os.path.abspath(path)), 'LLAMAS_*_mef.fits')):
        try:
            header = fits.getheader(cand, 0)
        except OSError:
            continue
        start = _start_time(header)
        if start is not None and _setting(header) == setting:
            members.append((start, os.path.abspath(cand)))
    members.sort()
    me = os.path.abspath(path)
    if me not in [p for _, p in members]:
        return [path]
    i = [p for _, p in members].index(me)
    lo = hi = i
    while lo > 0 and (members[lo][0] - members[lo - 1][0]).total_seconds() <= window_s:
        lo -= 1
    while hi < len(members) - 1 and (members[hi + 1][0] - members[hi][0]).total_seconds() <= window_s:
        hi += 1
    return [p for _, p in members[lo:hi + 1]]


def resolve_lamp_flat_sets(config):
    """``{colour: [raw paths]}`` of the lamp-flat set for each configured colour.

    A list in ``{colour}_flat_file`` is the set as given; it is replaced by its first frame
    so single-path consumers keep working until the combined frame replaces it. A single
    path is expanded with :func:`find_lamp_flat_set` unless ``lamp_flat_combine`` is false.
    """
    combine = config.get('lamp_flat_combine', True)
    if isinstance(combine, str):
        combine = combine.strip().lower() not in ('false', '0', 'no')
    sets = {}
    for colour, key in FLAT_KEYS.items():
        value = config.get(key)
        if isinstance(value, (list, tuple)):
            files = [str(v) for v in value if str(v).strip()]
            if files:
                config[key] = files[0]
                sets[colour] = files
        elif isinstance(value, str) and value and os.path.exists(value):
            sets[colour] = find_lamp_flat_set(value) if combine else [value]
    return sets


def combined_flat_name(raw_paths):
    """File name of the combined frame of a set: the first frame's stem, ``_x<N>``."""
    stem = os.path.splitext(os.path.basename(raw_paths[0]))[0]
    return f"{stem}_x{len(raw_paths)}_bias_corrected.fits"


def _scale_to(reference, data):
    """Factor that brings ``data`` to ``reference``'s level, from their lit pixels (1 if none)."""
    lit = np.isfinite(reference) & np.isfinite(data) & (reference > LIT_DN) & (data > LIT_DN)
    if lit.sum() < 1000:
        return 1.0
    return float(np.median(reference[lit] / data[lit]))


def combine_lamp_flats(corrected_paths, out_path, raw_names=None):
    """Combine bias-corrected exposures of one lamp-flat set into ``out_path``.

    Each camera is matched across the frames by its header, never by extension index. The
    frames are scaled to the first one's level over their lit pixels, then combined with
    ``Bias.darkLlamas.combine_frames`` (median of 3 or more, 5-sigma pair-minimum for 2). A
    camera missing from any frame keeps the first frame's data. The output keeps the
    first frame's headers (bias-first stamps included) and adds NCOMBINE, COMBMETH,
    FLSCALES and, on the primary, FLATIN<n> and the ``lamp_flat`` recipe stamp.
    """
    raw_names = raw_names or [os.path.basename(p) for p in corrected_paths]
    hduls = [fits.open(p) for p in corrected_paths]
    try:
        primary = hduls[0][0].copy()
        out = fits.HDUList([primary])
        methods = set()
        for hdu in hduls[0][1:]:
            if hdu.data is None:
                out.append(hdu.copy())
                continue
            bench, side, color = camera_of_header(hdu.header)
            others = [match_camera_hdu(h, bench, side, color, source=os.path.basename(p))
                      for h, p in zip(hduls[1:], corrected_paths[1:])]
            ref = np.asarray(hdu.data, dtype=np.float32)
            new = fits.ImageHDU(data=ref, header=hdu.header.copy())
            if any(o is None or o.data is None or o.data.shape != ref.shape for o in others):
                logger.warning(f"Lamp-flat combine: {bench}{side} {color} is missing or differs "
                               f"in a frame; keeping {raw_names[0]} alone")
                new.header['NCOMBINE'] = (1, 'Lamp-flat exposures combined')
                out.append(new)
                continue
            frames = [ref] + [np.asarray(o.data, dtype=np.float32) for o in others]
            scales = [1.0] + [_scale_to(ref, f) for f in frames[1:]]
            gain, readnoise, _ = props_for_header(hdu.header, DEFAULT_GAIN, DEFAULT_READNOISE)
            combined, method = combine_frames([f * s for f, s in zip(frames, scales)],
                                              gain, readnoise, 0.0)
            methods.add(method)
            new.data = combined
            new.header['NCOMBINE'] = (len(frames), 'Lamp-flat exposures combined')
            new.header['COMBMETH'] = (method, 'Combine method (Bias/darkLlamas)')
            new.header['FLSCALES'] = (','.join(f'{s:.4f}' for s in scales),
                                      'Scale applied to each exposure')
            out.append(new)
        header = out[0].header
        stamp_recipe(header, 'lamp_flat')
        header['CRCLEAN'] = (False, 'Lamp flat not cosmic-ray cleaned')
        header['NCOMBINE'] = (len(corrected_paths), 'Lamp-flat exposures combined')
        header['COMBMETH'] = (','.join(sorted(methods)) or 'none', 'Combine method')
        for k, name in enumerate(raw_names, start=1):
            header[f'FLATIN{k}'] = (name, 'Lamp-flat exposure')
        out.writeto(out_path, overwrite=True)
    finally:
        for h in hduls:
            h.close()
    logger.info(f"Combined {len(corrected_paths)} lamp-flat exposures into "
                f"{os.path.basename(out_path)}")
    return out_path


def combined_is_current(out_path, raw_names):
    """True when ``out_path`` exists and was combined from exactly ``raw_names``."""
    if not os.path.exists(out_path):
        return False
    header = fits.getheader(out_path, 0)
    found = [header.get(f'FLATIN{k}') for k in range(1, len(raw_names) + 2)]
    return found[:len(raw_names)] == list(raw_names) and found[len(raw_names)] is None


def describe_sets(sets):
    """One line per distinct set, for the terminal."""
    lines, seen = [], {}
    for colour, files in sets.items():
        seen.setdefault(tuple(files), []).append(colour)
    for files, colours in seen.items():
        names = ', '.join(os.path.basename(f) for f in files)
        verb = f"combined ({len(files)} exposures)" if len(files) > 1 else "single exposure"
        lines.append(f"Lamp flat {'/'.join(colours)}: {verb}: {names}")
    return lines
