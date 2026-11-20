"""Trace-set validation and the per-camera mastercalib fallback.

Checks a directory of trace pickles before the pipeline uses it. Supplies a
camera's mastercalib trace when that camera's own trace is missing or has the wrong
fibre count. Every master trace that stands in for a camera goes through
``find_usable_master_trace`` first. That covers the per-camera fallback, the
placeholder cameras in ``reduce.generate_traces``, and the mastercalib directory
used in place.

When the night's flat for that camera is available, the master is not copied: it is
shifted onto the flat first (``write_shifted_master``, see ``Trace/masterShift.py``), and a
master that cannot be placed with confidence leaves the camera unusable rather than
misplaced.

Old-format traces (no ``comb_ok``) were written before the tracer checked which
fibre is which. How one is treated depends on what it is:

- **a mastercalib trace**, or a byte-identical copy of one in a trace directory: it is
  audited (``audit_trace_registration``) and used if it passes;
- **any other old trace** (an old night's own trace) is never used. It is reported in
  ``legacy_traces`` with nothing on disk changed. The caller re-traces the night from
  its flats with the current tracer, after setting the old files aside
  (``set_aside_legacy_traces``).

This module lives under ``Trace/`` so it can import the tracer at module top.
``Utils/utils.py`` cannot: the tracer imports ``Utils.rayManager``, which runs
``Utils/__init__`` and so imports ``utils.py``.
"""
import filecmp
import glob
import json
import logging
import os
import pickle
import re
import shutil
import sys

import cloudpickle
import numpy as np

from llamas_pyjamas.config import CALIB_DIR, LUT_DIR
from llamas_pyjamas.Trace.masterShift import (
    SWITCH_HINT, MasterShiftError, build_shifted_trace, describe_shift, read_flat_cameras)
from llamas_pyjamas.Trace.traceLlamasMaster import (
    TRACE_FINGERPRINT_FILE, TraceLlamas, audit_trace_registration, load_trace_fingerprints,
    measure_trace_heights, normalise_trace_heights)
from llamas_pyjamas.Utils.userPrompt import NoTerminalError, ask_proceed_or_terminate
from llamas_pyjamas.Utils.utils import find_trace_pickle

logger = logging.getLogger(__name__)

# Live fibres per benchside: every trace set must have exactly this many traces.
TRACE_FIBRE_COUNTS = {'1A': 298, '1B': 300, '2A': 298, '2B': 297,
                      '3A': 298, '3B': 300, '4A': 300, '4B': 298}

CAMERAS = [(channel, bench, side) for channel in ('red', 'green', 'blue')
           for bench in ('1', '2', '3', '4') for side in ('A', 'B')]

# A healthy trace pickle is ~120 MB (night traces) to ~300 MB (packaged masters). Larger
# ones are checked for a frozen master-bias cache (see slim_bloated_traces).
BLOATED_TRACE_BYTES = 400 * 1024 ** 2
_TRACE_NAME = re.compile(r'LLAMAS_(red|green|blue)_([1-4])_([AB])_traces\.pkl$')


class MasterTraceUnusableError(FileNotFoundError):
    """A camera's mastercalib trace exists but cannot stand in for it.

    Subclasses FileNotFoundError so every caller that already treats a missing
    mastercalib trace as "no fallback for this camera" does the same here.
    """


def is_legacy_trace(traceobj) -> bool:
    """True for a trace written before the tracer checked fibre registration.

    Only the current tracer sets ``comb_ok``, so its absence marks the old format.
    """
    return not hasattr(traceobj, 'comb_ok')


def frozen_bias_cache_bytes(traceobj) -> int:
    """Bytes of master-bias data frozen into a trace's pickled-by-value class (0 if none).

    ``traceLlamasMaster`` is pickled by value, so a trace saved while that module still held
    the master-bias cache (before ``Bias/biasCache.py``) carries the whole cache in its class's
    function globals, e.g. 1.6 GB per green shifted-master trace.
    """
    total, seen = 0, set()
    for klass in type(traceobj).__mro__:
        for value in vars(klass).values():
            globals_ = getattr(getattr(value, '__func__', value), '__globals__', None)
            if globals_ is None or id(globals_) in seen:
                continue
            seen.add(id(globals_))
            for _mtime, hdus in (globals_.get('_BIAS_HDU_CACHE') or {}).values():
                total += sum(h.data.nbytes for h in hdus if getattr(h, 'data', None) is not None)
    return total


def slim_bloated_traces(trace_dir: str, mastercalib_dir: str = CALIB_DIR,
                        min_bytes: int = BLOATED_TRACE_BYTES) -> list:
    """Re-save trace pickles that carry a frozen master-bias cache, without it.

    Only current-format traces are rewritten. An old-format trace is re-traced anyway, and a
    byte copy of a packaged master must stay identical to it (``is_master_copy``). The trace
    is given the current ``TraceLlamas`` class, whose module no longer holds the cache, and is
    written through a temporary file. Its data are unchanged.

    Returns:
        list of ``(path, old_bytes, new_bytes)`` for every file rewritten.
    """
    slimmed = []
    for path in sorted(glob.glob(os.path.join(trace_dir, '*_traces.pkl'))):
        old_bytes = os.path.getsize(path)
        if old_bytes < min_bytes:
            continue
        name = _TRACE_NAME.search(os.path.basename(path))
        if name and is_master_copy(path, *name.groups(), mastercalib_dir):
            continue
        with open(path, 'rb') as f:
            traceobj = pickle.load(f)
        if is_legacy_trace(traceobj) or not frozen_bias_cache_bytes(traceobj):
            continue
        traceobj.__class__ = TraceLlamas
        tmp = f"{path}.tmp"
        with open(tmp, 'wb') as f:
            cloudpickle.dump(traceobj, f)
        os.replace(tmp, path)
        new_bytes = os.path.getsize(path)
        logger.info(f"Slimmed {os.path.basename(path)}: {old_bytes / 1e9:.2f} GB -> "
                    f"{new_bytes / 1e9:.2f} GB (removed a frozen copy of the master bias)")
        slimmed.append((path, old_bytes, new_bytes))
    return slimmed


def trace_references():
    """``(dead_fibers, fingerprints)``: dead fibremap slots per benchside (traceLUT.json)
    and throughput fingerprints per channel and benchside (LUT/trace_fingerprints.json)."""
    with open(os.path.join(LUT_DIR, 'traceLUT.json'), 'r') as f:
        dead_fibers = json.load(f).get('dead_fibers', {})
    return dead_fibers, load_trace_fingerprints()


def make_trace_auditor():
    """Build ``audit(traceobj) -> reason or None`` for an already-saved trace.

    Wraps ``audit_trace_registration`` with this camera's dead fibres and throughput
    fingerprint (``trace_references``), loaded once.

    Returns:
        callable: ``audit(traceobj)``, None if the trace passes, else the reason.
    """
    dead_fibers, fingerprints = trace_references()

    def audit(traceobj):
        benchside = f"{traceobj.bench}{traceobj.side}"
        fingerprint = fingerprints.get(traceobj.channel.lower(), {}).get(benchside)
        return audit_trace_registration(traceobj, dead_fibers=dead_fibers.get(benchside, []),
                                        fingerprint=fingerprint)
    return audit


def find_usable_master_trace(channel: str, bench: str, side: str,
                             mastercalib_dir: str, audit=None) -> str:
    """Locate a camera's mastercalib trace and check it can stand in for the camera.

    A fallback is only better than no trace if its fibres are right: it must have
    the benchside's fibre count and, if it is old-format, pass
    ``audit_trace_registration``. Otherwise its flux lands on the wrong lenslets.

    Args:
        channel, bench, side: camera identity.
        mastercalib_dir: directory holding the fallback traces.
        audit: callable from ``make_trace_auditor``; built here if None.

    Returns:
        str: path to the mastercalib trace.

    Raises:
        FileNotFoundError: no mastercalib trace for this camera.
        MasterTraceUnusableError: one exists but is unusable; the message says why.
    """
    path = find_trace_pickle(channel, bench, side, mastercalib_dir)
    name, benchside = os.path.basename(path), f"{bench}{side}"
    try:
        with open(path, 'rb') as f:
            traceobj = pickle.load(f)
        nfib = traceobj.traces.shape[0]
    except Exception as e:                                        # noqa: BLE001
        raise MasterTraceUnusableError(f"mastercalib {name} cannot be read: {e}") from e

    expected = TRACE_FIBRE_COUNTS.get(benchside)
    if nfib != expected:
        raise MasterTraceUnusableError(
            f"mastercalib {name} has {nfib} fibres, {benchside} needs {expected}")
    if is_legacy_trace(traceobj):
        reason = (audit or make_trace_auditor())(traceobj)
        if reason is not None:
            raise MasterTraceUnusableError(
                f"mastercalib {name} fails the registration check: {reason}")
    return path


def copy_mastercalib_trace(channel: str, bench: str, side: str,
                           mastercalib_dir: str, target_dir: str, audit=None) -> str:
    """Copy a camera's mastercalib trace into a trace directory, if it is usable.

    The master is checked first (``find_usable_master_trace``): a fallback with the
    wrong fibres is never copied in. The copy is written under the user naming
    convention, replacing any user trace of that name.

    Args:
        channel, bench, side: camera identity.
        mastercalib_dir: source mastercalib directory.
        target_dir: destination trace directory.
        audit: callable from ``make_trace_auditor``; built if None.

    Returns:
        str: path to the copied trace file.

    Raises:
        FileNotFoundError: no mastercalib trace for this camera.
        MasterTraceUnusableError: one exists but is unusable as a fallback.

    Example:
        >>> copy_mastercalib_trace('green', '4', 'B', '/calib', '/user/traces')
        '/user/traces/LLAMAS_green_4_B_traces.pkl'
    """
    mastercalib_path = find_usable_master_trace(channel, bench, side, mastercalib_dir, audit)

    user_filename = f'LLAMAS_{channel.lower()}_{bench}_{side}_traces.pkl'
    target_path = os.path.join(target_dir, user_filename)
    shutil.copy2(mastercalib_path, target_path)
    logger.info(f"Copied mastercalib trace in as {user_filename}")
    return target_path


def write_shifted_master(channel: str, bench: str, side: str, mastercalib_dir: str,
                         flat_camera, target_dir: str, audit=None,
                         require_fingerprint: bool = True, own: str = None,
                         flat_name: str = '') -> tuple:
    """Shift a camera's mastercalib trace onto the night's flat and save it.

    The master is checked first (``find_usable_master_trace``), then placed on the flat
    and gated (``masterShift.build_shifted_trace``). The result is written over the
    camera's own trace if it has one (``own``), else as
    ``LLAMAS_{channel}_{bench}_{side}_traces.pkl``. Nothing is written on failure.

    Args:
        channel, bench, side: camera identity.
        mastercalib_dir: source mastercalib directory.
        flat_camera: ``(data, header, is_placeholder)`` from ``read_flat_cameras``.
        target_dir: destination trace directory.
        audit: callable from ``make_trace_auditor``; built if None.
        require_fingerprint: strict fingerprint confirmation (config
            ``shifted_master_require_fingerprint``).
        own: path of the camera's own (wrong-count) trace, replaced on success.
        flat_name: the flat's file name, recorded in the trace.

    Returns:
        (master_path, written_path, record): ``record`` is the trace's ``master_shift``.

    Raises:
        FileNotFoundError: no mastercalib trace for this camera.
        MasterTraceUnusableError: one exists but is unusable as a fallback.
        MasterShiftError: it cannot be placed on the flat with confidence.
    """
    master_path = find_usable_master_trace(channel, bench, side, mastercalib_dir, audit)
    with open(master_path, 'rb') as f:
        master = pickle.load(f)
    data, header, _ = flat_camera
    benchside = f"{bench}{side}"
    dead_fibers, fingerprints = trace_references()
    traceobj = build_shifted_trace(
        master.traces, data, header, channel, bench, side,
        dead_fibers=dead_fibers.get(benchside, []),
        fingerprint=fingerprints.get(channel.lower(), {}).get(benchside),
        require_fingerprint=require_fingerprint, flat_name=flat_name,
        master_path=master_path, expected_count=TRACE_FIBRE_COUNTS[benchside])

    target = own or os.path.join(target_dir, f'LLAMAS_{channel.lower()}_{bench}_{side}_traces.pkl')
    tmp = f"{target}.tmp"
    with open(tmp, 'wb') as f:
        cloudpickle.dump(traceobj, f)
    os.replace(tmp, target)
    logger.info(f"Shifted mastercalib {os.path.basename(master_path)} onto "
                f"{flat_name or 'the flat'} as {os.path.basename(target)}")
    return master_path, target, traceobj.master_shift


def is_master_copy(pkl_file: str, channel: str, bench: str, side: str,
                   mastercalib_dir: str) -> bool:
    """True if ``pkl_file`` is byte-identical to that camera's mastercalib trace.

    A mastercalib fallback is copied into the trace directory, so on a later resumed
    run it is found there. An old-format copy must be judged as the master it is, not
    as an old night's own trace, or every resumed run would re-trace.
    """
    try:
        master = find_trace_pickle(channel, bench, side, mastercalib_dir)
    except FileNotFoundError:
        return False
    return filecmp.cmp(pkl_file, master, shallow=False)


def legacy_backup_path(path: str) -> str:
    """First free name of ``<path>.legacy``, ``<path>.legacy.1``, ``.2``, ...

    All of these are outside every ``*.pkl`` glob, so the pipeline never reads them.
    An existing backup is never overwritten.
    """
    backup = f"{path}.legacy"
    n = 1
    while os.path.exists(backup):
        backup = f"{path}.legacy.{n}"
        n += 1
    return backup


def set_aside_legacy_traces(paths) -> list:
    """Rename trace pickles out of the pipeline's way, keeping them as backups.

    Each file becomes ``legacy_backup_path(path)``. Paths that no longer exist are
    skipped.

    Args:
        paths: trace pickle paths.

    Returns:
        list: ``(original, backup)`` pairs for the files moved.
    """
    moved = []
    for path in paths:
        if not os.path.exists(path):
            continue
        backup = legacy_backup_path(path)
        os.replace(path, backup)
        logger.info(f"Set aside {os.path.basename(path)} as {os.path.basename(backup)}")
        moved.append((path, backup))
    return moved


def validate_and_fix_trace_fibres(trace_dir: str, mastercalib_dir: str = CALIB_DIR,
                                  flats: dict = None, require_fingerprint: bool = True) -> dict:
    """Check every camera's trace in ``trace_dir``, and fall back to mastercalib per camera.

    First every pickle is classified, with nothing on disk changed:

    - **valid**: the right fibre count and the current format (``comb_ok``), or an
      old-format master (or byte-identical copy of one) that passes
      ``audit_trace_registration``;
    - **wrong count**: falls back to the mastercalib trace;
    - **old-format master that fails the audit**: listed in ``unusable_masters``;
    - **any other old-format trace** (an old night's own trace): listed in
      ``legacy_traces``.

    If there are any ``legacy_traces`` the function returns at once, having copied
    and renamed nothing. The caller re-traces the night from its flats.

    Otherwise wrong-count and missing cameras (failed tracing writes no pickle) get the
    mastercalib trace, and only if ``find_usable_master_trace`` accepts it:

    - **with the camera's flat** (``flats``, not a placeholder): the master is shifted
      onto that flat (``write_shifted_master``). If it cannot be placed with confidence
      the camera is unusable; an unshifted master is never used when a flat exists;
    - **without one**: the master is copied in as it is.

    A camera left with no usable trace goes into ``unusable_masters``, untouched, and
    the caller decides whether to go on without it (``confirm_unusable_master_traces``).
    ``trace_dir`` must therefore be a COMPLETE trace set; an empty one is filled from
    mastercalib camera by camera.

    Args:
        trace_dir: trace directory to validate.
        mastercalib_dir: mastercalib directory for fallback traces.
        flats: ``{channel: flat path}`` of the night's flats; None keeps every fallback
            an unshifted copy.
        require_fingerprint: strict fingerprint confirmation for shifted masters.

    Returns:
        dict: {
            'valid_traces': [(channel, bench, side, filepath), ...],
            'invalid_traces': [(channel, bench, side, expected, actual), ...],
            'missing_traces': [(channel, bench, side), ...],
            'legacy_traces': [(channel, bench, side, filepath), ...],
            'unusable_masters': [(channel, bench, side, reason, own_trace_path or None), ...],
            'fallback_used': [(channel, bench, side, mastercalib_path, copied_path), ...],
            'shifted_masters': [(channel, bench, side, mastercalib_path, path, record), ...],
            'all_valid': bool
        }

        ``shifted_masters`` lists the masters shifted onto the flat by this call, and
        those found from an earlier run (also in ``valid_traces``).
    """
    files = sorted(glob.glob(os.path.join(trace_dir, '*.pkl')))
    if not files:
        logger.warning(f"No trace files found in {trace_dir}; every camera falls back "
                       f"to mastercalib")

    valid_traces = []
    invalid_traces = []
    legacy_traces = []
    unusable_masters = []
    fallback_used = []
    shifted_masters = []
    own_paths = {}
    audit = make_trace_auditor()
    trace_dir_is_mastercalib = os.path.realpath(trace_dir) == os.path.realpath(mastercalib_dir)

    # 1. Classify. Nothing on disk changes here.
    for pkl_file in files:
        try:
            with open(pkl_file, "rb") as f:
                traceobj = pickle.load(f)
            nfib = traceobj.traces.shape[0]
            channel = traceobj.channel.lower()
            bench = str(traceobj.bench)
            side = traceobj.side
        except Exception as e:                                    # noqa: BLE001
            logger.error(f"Error reading trace file {pkl_file}: {e}")
            continue

        cam = f"{channel}{bench}{side}"
        expected = TRACE_FIBRE_COUNTS.get(f"{bench}{side}")
        if expected is None:
            logger.warning(f"Unknown benchside {bench}{side} in {os.path.basename(pkl_file)}")
            continue
        own_paths[(channel, bench, side)] = pkl_file

        if is_legacy_trace(traceobj):
            is_master = trace_dir_is_mastercalib or is_master_copy(
                pkl_file, channel, bench, side, mastercalib_dir)
            if not is_master:
                legacy_traces.append((channel, bench, side, pkl_file))
                logger.warning(f"✗ {cam}: {os.path.basename(pkl_file)} was written by an "
                               f"older pipeline version and is not reused")
                continue
            reason = (f"{nfib} fibres, {expected} needed" if nfib != expected
                      else audit(traceobj))
            if reason is not None:
                unusable_masters.append((channel, bench, side,
                                         f"mastercalib trace fails the registration check: {reason}",
                                         pkl_file))
                logger.error(f"✗ {cam}: old-format mastercalib trace fails the "
                             f"registration check: {reason}")
                continue
            valid_traces.append((channel, bench, side, pkl_file))
            logger.debug(f"✓ {cam}: old-format mastercalib trace passes the registration check")
        elif nfib == expected:
            valid_traces.append((channel, bench, side, pkl_file))
            record = getattr(traceobj, 'master_shift', None)
            if record is not None:
                shifted_masters.append((channel, bench, side, record.get('master'),
                                        pkl_file, record))
            logger.debug(f"✓ {cam}: {nfib}/{expected} fibers (valid)")
        else:
            invalid_traces.append((channel, bench, side, expected, nfib))
            logger.warning(f"✗ {cam}: {nfib}/{expected} fibers (invalid)")

    if legacy_traces:
        # The caller re-traces this night; nothing has been copied or renamed.
        return {
            'valid_traces': valid_traces,
            'invalid_traces': invalid_traces,
            'missing_traces': [],
            'legacy_traces': legacy_traces,
            'unusable_masters': unusable_masters,
            'fallback_used': [],
            'shifted_masters': shifted_masters,
            'all_valid': False,
        }

    flat_cameras = {}

    def night_flat(channel):
        """(cameras, error) for this channel's flat; (None, None) if there is none."""
        path = (flats or {}).get(channel)
        if not path or not os.path.exists(path):
            return None, None
        if channel not in flat_cameras:
            try:
                flat_cameras[channel] = (read_flat_cameras(path, channel), None)
            except MasterShiftError as e:
                flat_cameras[channel] = (None, str(e))
        return flat_cameras[channel]

    def shift_back(channel, bench, side, own, camera):
        try:
            master_path, path, record = write_shifted_master(
                channel, bench, side, mastercalib_dir, camera, trace_dir, audit,
                require_fingerprint, own, os.path.basename(flats[channel]))
        except MasterTraceUnusableError as e:
            unusable_masters.append((channel, bench, side, str(e), own))
            logger.error(f"✗ {channel}{bench}{side}: no usable mastercalib fallback: {e}")
            return
        except FileNotFoundError as e:
            if own is not None:
                unusable_masters.append((channel, bench, side, str(e), own))
            logger.error(f"✗ {channel}{bench}{side}: no mastercalib fallback: {e}")
            return
        except MasterShiftError as e:
            unusable_masters.append((channel, bench, side,
                                     f"mastercalib trace cannot be placed on this night's "
                                     f"flat: {e}", own))
            logger.error(f"✗ {channel}{bench}{side}: mastercalib trace cannot be placed on "
                         f"this night's flat: {e}")
            return
        shifted_masters.append((channel, bench, side, master_path, path, record))
        logger.info(f"✓ {channel}{bench}{side}: mastercalib trace shifted onto this night's "
                    f"flat ({describe_shift(record)})")

    def fall_back(channel, bench, side, own):
        cameras, error = night_flat(channel)
        if error is not None:
            unusable_masters.append((channel, bench, side,
                                     f"the night's flat cannot be read to place the "
                                     f"mastercalib trace: {error}", own))
            logger.error(f"✗ {channel}{bench}{side}: {error}")
            return
        camera = (cameras or {}).get((channel, bench, side))
        if camera is not None and not camera[2]:
            shift_back(channel, bench, side, own, camera)
            return
        # No flat for this camera (none configured, or a placeholder): copy as it is.
        try:
            copied_path = copy_mastercalib_trace(channel, bench, side, mastercalib_dir,
                                                 trace_dir, audit)
        except MasterTraceUnusableError as e:
            unusable_masters.append((channel, bench, side, str(e), own))
            logger.error(f"✗ {channel}{bench}{side}: no usable mastercalib fallback: {e}")
            return
        except FileNotFoundError as e:
            if own is not None:
                # No stand-in, and this wrong-count trace must not be used either.
                unusable_masters.append((channel, bench, side, str(e), own))
            logger.error(f"✗ {channel}{bench}{side}: no mastercalib fallback: {e}")
            return
        fallback_used.append((channel, bench, side,
                              find_trace_pickle(channel, bench, side, mastercalib_dir),
                              copied_path))
        logger.info(f"✓ {channel}{bench}{side}: mastercalib fallback copied in")

    # 2. Act: per-camera fallback for wrong-count traces...
    for channel, bench, side, _expected, _nfib in invalid_traces:
        fall_back(channel, bench, side, own_paths[(channel, bench, side)])

    # ...and for cameras with no pickle at all (failed tracing writes none), which
    # would otherwise become blank fibres in the RSS.
    present = {cam for cam in own_paths}
    missing_traces = [cam for cam in CAMERAS if cam not in present]
    for channel, bench, side in missing_traces:
        logger.warning(f"✗ {channel}{bench}{side}: no trace file")
        fall_back(channel, bench, side, None)

    return {
        'valid_traces': valid_traces,
        'invalid_traces': invalid_traces,
        'missing_traces': missing_traces,
        'legacy_traces': [],
        'unusable_masters': unusable_masters,
        'fallback_used': fallback_used,
        'shifted_masters': shifted_masters,
        'all_valid': not (invalid_traces or missing_traces or unusable_masters),
    }


def count_trace_fibres(mastercalib_dir: str = CALIB_DIR) -> bool:
    """Check whether a mastercalib directory can be used in place, as it stands.

    Every trace must have its benchside's fibre count, and an old-format trace must
    also pass ``audit_trace_registration``: these are the fallback traces, so their
    fibres must be right. On False the caller falls back camera by camera
    (``validate_and_fix_trace_fibres``).

    Args:
        mastercalib_dir (str): directory containing the trace pickles.

    Returns:
        bool: True if there are trace files and every one passes, False otherwise.
    """
    files = glob.glob(os.path.join(mastercalib_dir, '*.pkl'))
    if not files:
        logger.warning(f"No trace files in {mastercalib_dir}")
        return False

    all_match = True
    audit = make_trace_auditor()
    for pkl in files:
        with open(pkl, "rb") as f:
            traceobj = pickle.load(f)
        nfib = traceobj.traces.shape[0]
        channel = traceobj.channel
        benchside = f"{traceobj.bench}{traceobj.side}"
        req = TRACE_FIBRE_COUNTS[benchside]

        if nfib != req:
            all_match = False
        elif is_legacy_trace(traceobj):
            reason = audit(traceobj)
            if reason is not None:
                all_match = False
                logger.warning(f"Mastercalib {channel} {benchside} fails the registration "
                               f"check ({reason}); it cannot be used as a fallback")

        print(f'Channel {channel} Benchside {benchside} trace has {nfib} fibres and requires {req}')

    return all_match


def confirm_unusable_master_traces(unusable, interactive=None, input_fn=input,
                                   stream=None) -> None:
    """Ask whether to go on without the cameras that have no usable mastercalib fallback.

    Each entry (``validate_and_fix_trace_fibres()['unusable_masters']``) is a camera
    whose own trace is missing or bad and whose mastercalib trace cannot stand in for
    it. Proceeding skips those cameras: a bad trace of their own is set aside
    (``set_aside_legacy_traces``), so extraction finds no trace and leaves their
    fibres blank rather than putting flux on the wrong lenslets. Terminating stops
    the run with nothing changed. Without a terminal there is nobody to ask, so the
    run stops.

    The question is written to the real terminal (``sys.__stdout__``): during a
    curated pipeline run ``sys.stdout`` is redirected to the log file.

    Args:
        unusable: ``[(channel, bench, side, reason, own_trace_path or None), ...]``.
        interactive: whether someone can answer; defaults to ``sys.stdin.isatty()``.
        input_fn: reads the answer (for tests).
        stream: where the question is written; defaults to ``sys.__stdout__``.

    Raises:
        RuntimeError: not interactive.
        SystemExit: the user chose to terminate.
    """
    if not unusable:
        return
    out = stream or sys.__stdout__
    cameras = ', '.join(f"{ch}{b}{sd}" for ch, b, sd, *_ in unusable)
    lines = ["", "=" * 60, "NO USABLE MASTERCALIB FALLBACK", "=" * 60]
    lines += [f"  ✗ {ch}{b}{sd}: {reason}" for ch, b, sd, reason, _ in unusable]
    lines += ["Proceeding skips these cameras: they get no trace, so their fibres are",
              "blank in the RSS. Terminating stops the whole run.", ""]
    out.write("\n".join(lines) + "\n")
    out.flush()
    logger.warning(f"No usable mastercalib fallback for {cameras}")

    # Cameras the strict fingerprint rule alone refused: say how to relax it.
    strict_only = [f"{ch}{b}{sd}" for ch, b, sd, reason, _ in unusable if SWITCH_HINT in reason]
    note = (f" The fingerprint did not confirm the shifted master for "
            f"{', '.join(strict_only)}: to accept the slit-end match alone, set "
            f"shifted_master_require_fingerprint = false in the config."
            if strict_only else "")

    try:
        proceed = ask_proceed_or_terminate("Proceed without these cameras, or terminate the run?",
                                           interactive=interactive, input_fn=input_fn, stream=out)
    except NoTerminalError:
        proceed = None
    if not proceed and note:
        out.write(f"Note:{note}\n")
        out.flush()
    if proceed is None:
        raise RuntimeError(
            f"No usable mastercalib trace for {cameras}, and no terminal to ask whether to "
            f"proceed without them, so the run is terminated. Rerun interactively, or replace "
            f"those cameras' traces in the mastercalib directory.{note}")
    if not proceed:
        raise SystemExit(f"Terminated by user: no usable mastercalib trace for "
                         f"{cameras}.{note}")

    own = [path for *_, path in unusable if path]
    for original, backup in set_aside_legacy_traces(own):
        out.write(f"  set aside {os.path.basename(original)} as {os.path.basename(backup)}\n")
    out.flush()
    logger.warning(f"Proceeding without {cameras} at the user's request")


def build_trace_fingerprints(mastercalib_dir: str = CALIB_DIR, out: str = None) -> dict:
    """Build the per-camera fibre throughput fingerprints used to register traces.

    For every trace pickle in ``mastercalib_dir`` (either name form,
    ``LLAMAS_master_{channel}_{bench}_{side}_traces.pkl`` or
    ``LLAMAS_{channel}_{bench}_{side}_traces.pkl``), measures the flat flux of each
    trace at the centre column. It stores the normalised fibre-to-fibre pattern in
    detector (= live-slot) order. traceLlamasMaster uses it to break slot-offset ties
    and to reject a comb numbered whole slots off. Only regenerate from a trace set
    whose fibre registration has been verified: every future trace is checked against
    it.

    Args:
        mastercalib_dir (str): Directory holding the verified master trace pickles.
        out (str): Output JSON path. Defaults to LUT/trace_fingerprints.json.

    Returns:
        dict: ``{channel: {benchside: [pattern per live slot]}}``.

    Raises:
        FileNotFoundError: if ``mastercalib_dir`` holds no trace pickles. Writing an
            empty file would silently switch off every fingerprint check.
    """
    out = out or TRACE_FINGERPRINT_FILE
    fingerprints = {}
    # Matches both name forms; sorted, so LLAMAS_master_* wins if a camera has both.
    for pkl in sorted(glob.glob(os.path.join(mastercalib_dir, 'LLAMAS_*_traces.pkl'))):
        with open(pkl, 'rb') as f:
            traceobj = pickle.load(f)
        col = traceobj.naxis1 // 2
        rows = np.sort(traceobj.traces[:, col])
        pattern = normalise_trace_heights(measure_trace_heights(traceobj.data, rows, col))
        fingerprints.setdefault(traceobj.channel.lower(), {})[
            f"{traceobj.bench}{traceobj.side}"] = [round(float(v), 4) for v in pattern]
        print(f"{os.path.basename(pkl)}: {len(rows)} fibres")

    if not fingerprints:
        raise FileNotFoundError(f"No trace pickles in {mastercalib_dir}; not overwriting {out}")

    with open(out, 'w') as f:
        json.dump(fingerprints, f, indent=1)
    print(f"Wrote {sum(len(v) for v in fingerprints.values())} camera fingerprints to {out}")
    return fingerprints
