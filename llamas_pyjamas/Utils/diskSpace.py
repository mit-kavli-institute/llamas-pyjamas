"""Disk space: the pre-flight estimate and check, and recognising a full disk.

A run fails when a volume it writes to fills up, and Ray fails earlier: it refuses to create or
spill objects once the volume holding its temp or spill folder is more than 95% full
(``RAY_local_fs_capacity_threshold``). On a 1 TB laptop disk that is about 50 GB still free.

``estimate_disk_use`` lists what a run will write, from the frames in the config and the
products that already exist. ``disk_space_gate`` compares it with the real free space of each
volume. It goes on if there is room, asks the user if there may not be, and stops if the run
cannot even finish its calibrations.

Sizes are the peak, not the final, disk use. They were measured on the 20261014_15 and
20261015_16 reductions (2026-10/11):

- a bias-corrected or flat-corrected frame is float32: 2 × the 16-bit raw file (403 MB);
- a trace pickle is about 117 MB per camera;
- an extraction pickle carries all 24 cameras' traces: their total size × 1.05 (4.43 GB with
  4.29 GB of traces);
- an RSS file is 205–245 MB per colour, and three stages (``_RSS``, ``_FF``, ``_FF_SKYSUB``) are
  kept per frame until they are consolidated at the end of the run;
- all science extraction pickles exist together after the extraction loop; each is deleted when
  its frame's RSS is written, and the frame being processed briefly has two more pickles
  (wavelength-corrected, then sky-subtracted / line-refined);
- a twilight flat leaves 3 pickles, 2 flat-corrected copies and a 3-colour RSS;
- the flat products (pixel maps, smooth models, flat extractions, fibre-flat corrections) total
  about 5 GB, and the 2d refined arc (``LUT/``) 2.4 GB.
"""
import errno
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

from ray.exceptions import ObjectStoreFullError, OutOfDiskError

from llamas_pyjamas.Utils.userPrompt import NoTerminalError, ask_proceed_or_terminate

logger = logging.getLogger(__name__)

GB = 1e9
N_CAMERAS = 24
TRACE_PICKLE_BYTES = 120e6
EXTRACTION_OVER_TRACES = 1.05
RSS_COLOUR_BYTES = 0.25e9
RSS_STAGES = 3
CR_MASK_BYTES = 0.1e9
CUBE_COLOUR_BYTES = 0.12e9
TWILIGHT_RSS_BYTES = 0.6e9
FLAT_PRODUCTS_BYTES = 5e9
REFINED_ARC_BYTES = 2.5e9
FLOAT_COPY = 2                   # float32 product / 16-bit raw

DEFAULT_MARGIN = 1.2
DEFAULT_MIN_FREE_GB = 5.0
DEFAULT_RAY_THRESHOLD = 0.95

GROUPS = (('science', 'science and standard frames'),
          ('sky', 'sky frames'),
          ('twilight', 'twilight flats'),
          ('arc', 'arcs'),
          ('calibration', 'traces, flat products, darks, calibration copies'),
          ('ray', 'Ray scratch (object store and spill)'))
CALIBRATION_GROUPS = ('twilight', 'arc', 'calibration')


class DiskSpaceError(RuntimeError):
    """The pre-flight check found too little disk space and the run was not started."""


# ---------------------------------------------------------------------------
# recognising a full disk
# ---------------------------------------------------------------------------
def is_disk_full(exc) -> bool:
    """True if ``exc`` means a disk filled up: Ray's object-store and spill errors, or ENOSPC."""
    if isinstance(exc, (OutOfDiskError, ObjectStoreFullError)):
        return True
    return isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT)


def disk_full_message(folders, ray_folders=()) -> str:
    """One line for the end-of-run report: the free space where the run was writing."""
    parts = []
    for check in _volumes({f: f in ray_folders for f in folders if f}, _ray_threshold()):
        limit = " (Ray stops at 95% use)" if check.ray else ""
        parts.append(f"{check.free / GB:.0f} GB free on {check.mount}{limit}")
    return "out of disk space: " + ("; ".join(parts) or "see the log")


# ---------------------------------------------------------------------------
# the estimate
# ---------------------------------------------------------------------------
@dataclass
class Item:
    """Something the run writes: ``nbytes`` in ``folder``, for ``frame`` (or None)."""
    group: str
    label: str
    nbytes: float
    folder: str
    frame: str = None
    done: bool = False


def extraction_bytes(trace_dir) -> float:
    """Size of one frame's extraction pickle: the night's trace pickles × 1.05."""
    sizes = []
    if trace_dir and os.path.isdir(trace_dir):
        sizes = [os.path.getsize(os.path.join(trace_dir, f)) for f in os.listdir(trace_dir)
                 if f.endswith('_traces.pkl')]
    traces = sum(sizes) if len(sizes) >= N_CAMERAS else N_CAMERAS * TRACE_PICKLE_BYTES
    return traces * EXTRACTION_OVER_TRACES


def estimate_disk_use(frames, folders, done, config=None, dark_groups=0,
                      object_store_bytes=0.0) -> list:
    """Everything the run will write, as ``Item``s marked done where the product exists.

    Args:
        frames: ``{raw path: role}`` (``reduce.bias_first_frames``): science, sky, flat, arc,
            twilight.
        folders: resolved folders: output, bias_corrected, traces, extractions, flat, masks,
            cubes, arcs, lut, ray.
        done: ``done(kind, raw)`` -> bool, from the pipeline's own resume checks. Kinds:
            bias_corrected, science, sky, twilight_flat, twilight_fibre, arc, traces, flat,
            darks, refined_arc.
        config: the run config (cube, sky and arc switches).
        dark_groups: number of master darks to build.
        object_store_bytes: Ray's object store size.
    """
    cfg = config or {}
    e = extraction_bytes(folders.get('traces'))
    items = []

    def add(group, label, nbytes, folder, frame=None, kind=None):
        items.append(Item(group, label, nbytes, folder, frame,
                          bool(kind) and bool(done(kind, frame))))

    # Per-exposure cubes only with generate_cubes; the other two switches pick the _FF and
    # _FF_SKYSUB sets.
    cubes = 0.0
    if cfg.get('generate_cubes', False):
        sets = (bool(cfg.get('generate_standard_cubes', True))
                + bool(cfg.get('sky_subtract', True) and cfg.get('generate_skysub_cubes', True)))
        cubes = 3 * sets * CUBE_COLOUR_BYTES
    science_todo = False
    for raw, role in frames.items():
        copy = FLOAT_COPY * os.path.getsize(raw)
        group = role if role in ('science', 'sky', 'twilight', 'arc') else 'calibration'
        add(group, 'bias-corrected frame', copy, folders['bias_corrected'], raw, 'bias_corrected')
        if role == 'science':
            per_frame = copy + e + RSS_STAGES * 3 * RSS_COLOUR_BYTES + CR_MASK_BYTES + cubes
            add('science', 'flat-corrected frame, extraction, RSS', per_frame,
                folders['extractions'], raw, 'science')
            science_todo = science_todo or not items[-1].done
        elif role == 'sky':
            add('sky', 'flat-corrected frame and extractions', copy + 2 * e,
                folders['extractions'], raw, 'sky')
        elif role == 'twilight':
            add('twilight', 'flat stage: extractions and RSS', copy + 2 * e + TWILIGHT_RSS_BYTES,
                folders['flat'], raw, 'twilight_flat')
            add('twilight', 'fibre-flat stage: extraction', copy + e,
                folders['extractions'], raw, 'twilight_fibre')
        elif role == 'arc':
            add('arc', 'night-arc extraction', e, folders['arcs'], raw, 'arc')

    if science_todo:
        add('science', 'frame being processed (2 extra pickles)', 2 * e, folders['extractions'])
    add('calibration', 'traces', N_CAMERAS * TRACE_PICKLE_BYTES, folders['traces'], kind='traces')
    add('calibration', 'flat products', FLAT_PRODUCTS_BYTES, folders['flat'], kind='flat')
    if dark_groups:
        darks = [r for r in cfg.get('dark_files') or [] if isinstance(r, str) and os.path.exists(r)]
        if darks:
            add('calibration', 'master darks',
                dark_groups * FLOAT_COPY * os.path.getsize(darks[0]), folders['output'],
                kind='darks')
    if cfg.get('refine_arc', False) and cfg.get('refine_arc_method', '2d') == '2d':
        add('calibration', 'refined arc', REFINED_ARC_BYTES, folders['lut'], kind='refined_arc')
    add('ray', 'object store and spill', object_store_bytes + e, folders['ray'])
    return items


# ---------------------------------------------------------------------------
# the check
# ---------------------------------------------------------------------------
@dataclass
class VolumeCheck:
    mount: str
    free: float
    total: float
    reserve: float
    ray: bool
    folders: list = field(default_factory=list)
    items: list = field(default_factory=list)
    need: float = 0.0
    calibration_need: float = 0.0
    verdict: str = 'ok'

    @property
    def usable(self):
        return self.free - self.reserve


def _ray_threshold() -> float:
    try:
        return float(os.environ.get('RAY_local_fs_capacity_threshold', DEFAULT_RAY_THRESHOLD))
    except ValueError:
        return DEFAULT_RAY_THRESHOLD


def _existing(path):
    """``path`` or its nearest existing parent."""
    path = os.path.abspath(path)
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def _mount_point(path):
    """Where the volume holding ``path`` is mounted, as ``df`` reports it."""
    path = _existing(path)
    try:
        out = subprocess.run(['df', '-P', path], capture_output=True, text=True, timeout=10)
        return out.stdout.splitlines()[1].split(None, 5)[5]
    except Exception:
        pass
    path = os.path.realpath(path)
    while not os.path.ismount(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    return path


def _volumes(folders, threshold, min_free=DEFAULT_MIN_FREE_GB * GB):
    """``VolumeCheck`` per volume for ``{folder: holds Ray scratch}``."""
    checks = {}
    for folder, is_ray in folders.items():
        probe = _existing(folder)
        try:
            dev = os.stat(probe).st_dev
            usage = shutil.disk_usage(probe)
        except OSError:
            continue
        check = checks.get(dev)
        if check is None:
            check = checks[dev] = VolumeCheck(_mount_point(probe), usage.free, usage.total,
                                              min_free, False)
        check.folders.append(folder)
        if is_ray and not check.ray:
            check.ray = True
            check.reserve = max(min_free, (1 - threshold) * check.total)
    return list(checks.values())


def check_disk_space(items, config=None):
    """Group ``items`` by volume and set each volume's verdict: ok, ask or stop."""
    cfg = config or {}
    margin = float(cfg.get('disk_margin', DEFAULT_MARGIN))
    min_free = float(cfg.get('min_free_gb', DEFAULT_MIN_FREE_GB)) * GB
    folders = {}
    for item in items:
        folders[item.folder] = folders.get(item.folder, False) or item.group == 'ray'
    checks = _volumes(folders, _ray_threshold(), min_free)
    for check in checks:
        check.items = [i for i in items if i.folder in check.folders]
        todo = [i for i in check.items if not i.done]
        check.need = margin * sum(i.nbytes for i in todo)
        check.calibration_need = sum(i.nbytes for i in todo if i.group in CALIBRATION_GROUPS)
        if check.usable <= 0 or check.usable < check.calibration_need:
            check.verdict = 'stop'
        elif check.usable < check.need:
            check.verdict = 'ask'
    return checks


def format_report(checks, folder_names=None, margin=DEFAULT_MARGIN) -> str:
    """The pre-flight disk report: per volume, free space, the limit, and the need by group."""
    names = folder_names or {}
    lines = ["", "=" * 60, "DISK SPACE", "=" * 60]
    for check in checks:
        where = ', '.join(sorted({names.get(f, os.path.basename(f) or f) for f in check.folders}))
        lines.append(f"{check.mount}  ({where})")
        limit = (f"Ray stops below {check.reserve / GB:.0f} GB free "
                 f"({100 * _ray_threshold():.0f}% full)" if check.ray
                 else f"keeping {check.reserve / GB:.0f} GB free (min_free_gb)")
        lines.append(f"  {check.free / GB:.0f} GB free of {check.total / GB:.0f} GB; {limit} "
                     f"-> {max(check.usable, 0) / GB:.0f} GB usable")
        lines.append(f"  this run needs about {check.need / GB:.0f} GB "
                     f"(estimate x {margin:g}):")
        for group, label in GROUPS:
            items = [i for i in check.items if i.group == group]
            todo = sum(i.nbytes for i in items if not i.done)
            if not items or (not todo and group != 'science'):
                continue
            frames = {i.frame for i in items if i.frame}
            open_frames = {i.frame for i in items if i.frame and not i.done}
            count = f"{len(frames)} ({len(open_frames)} to do)" if frames else ""
            lines.append(f"    {label:<52s} {count:>13s} {todo / GB:6.1f} GB")
        if check.verdict == 'ok':
            lines.append("  OK")
        else:
            short = max(check.need, check.calibration_need) - max(check.usable, 0)
            lines.append(f"  NOT ENOUGH: about {short / GB:.0f} GB short"
                         + (" -- not even the calibrations fit" if check.verdict == 'stop' else ""))
    if any(c.verdict != 'ok' for c in checks):
        if sys.platform == 'darwin':
            lines += ["Finder's 'available' space can be much larger: it counts purgeable space "
                      "(cloud-drive copies, snapshots) that macOS frees only on demand. Ray and "
                      "this check count only the space that is really free (df)."]
        lines += ["To make room: free space on that disk; set output_dir (and scratch_dir for "
                  "Ray) to a larger disk; reduce fewer frames per run; or set "
                  "skip_disk_check = true to run anyway."]
    return "\n".join(lines) + "\n"


def disk_space_gate(items, config=None, folder_names=None, interactive=None, input_fn=input,
                    stream=None):
    """Report disk space for ``items`` and decide whether the run may start.

    Every volume fits: returns. A volume may run out: asks whether to proceed, and terminates
    if there is no terminal. A volume cannot hold even the calibrations:
    terminates without asking.

    Returns:
        the ``VolumeCheck`` list.

    Raises:
        DiskSpaceError: the run must not start, or the user chose to terminate.
    """
    cfg = config or {}
    out = stream or sys.__stdout__
    checks = check_disk_space(items, cfg)
    report = format_report(checks, folder_names, float(cfg.get('disk_margin', DEFAULT_MARGIN)))
    verdicts = {c.verdict for c in checks}
    for line in report.strip('\n').splitlines():
        logger.info(line)
    if verdicts == {'ok'} or not verdicts:
        return checks
    out.write(report)
    out.flush()
    if 'stop' in verdicts:
        raise DiskSpaceError("not enough disk space even for the calibrations "
                             "(see DISK SPACE above)")
    try:
        proceed = ask_proceed_or_terminate("Disk space may run out. Proceed anyway, or "
                                           "terminate the run?", interactive=interactive,
                                           input_fn=input_fn, stream=out)
    except NoTerminalError:
        raise DiskSpaceError("disk space may run out and there is no terminal to ask; "
                             "skip_disk_check = true runs anyway") from None
    if not proceed:
        raise DiskSpaceError("terminated by user: disk space may run out")
    logger.warning("Proceeding with too little disk space at the user's request")
    return checks
