"""Central Ray scratch / temp-directory lifecycle management for the pipeline.

Ray, left to its defaults, writes session logs, the object-store spill files, and
the ``runtime_env`` package uploads into ``/tmp/ray/session_<...>``. The LLAMAS
pipeline initialises Ray several times per run and never cleans any of it up, so
``/tmp`` fills and the pipeline crashes with "No space left on device" — and the
junk survives across runs and users.

This module gives Ray **one short, per-run, owned scratch directory** (via the
``_temp_dir`` argument every ``ray.init`` should pass) so all of that data can be
bounded and reliably removed — during the run, on normal exit, and on early
failure. It also provides:

* a pre-flight disk-space gate (:func:`preflight_disk_check`) that warns when space
  is tight and aborts *before any work* when it is guaranteed insufficient;
* shared discovery / prune helpers used by the standalone ``clean_ray_scratch`` CLI
  so existing users can reclaim already-accumulated sessions.

Design notes
------------
* Ray binds an ``AF_UNIX`` socket at
  ``<_temp_dir>/session_<ts>_<pid>/sockets/plasma_store``; the OS limit is 104
  bytes (macOS) / 108 (Linux). Ray appends ~70 chars below ``_temp_dir``, so the
  temp dir **must be short** — hence the default ``/tmp/llamas_ray`` and the
  socket-length validation with fall-back. ``$TMPDIR`` is deliberately *not* used
  for the socket dir because the default macOS ``$TMPDIR`` (~49 chars) overflows.
* Cleanup is guaranteed by ``atexit`` (covers normal exit, unhandled exceptions and
  ``sys.exit``) plus a ``SIGTERM``/``SIGHUP`` handler (scheduler kills). ``SIGKILL``
  and power loss cannot be trapped and are recovered by :func:`prune_stale` on the
  next run.

This is Phase 1 of the fix: it does not consolidate the multiple Ray sessions into
one — a future ``init_ray()`` entry point will. It only controls *where* Ray writes
and *guarantees* the cleanup.
"""

import os
import sys
import glob
import json
import time
import shutil
import atexit
import signal
import logging
import threading

logger = logging.getLogger(__name__)

# Owned base directory name (kept short so the socket path stays within the limit).
_PREFIX = "llamas_ray"
_DEFAULT_BASE = os.path.join("/tmp", _PREFIX)   # "/tmp/llamas_ray"

_ENV_TEMP = "LLAMAS_RAY_TEMP_DIR"       # exported = the resolved per-run dir
_ENV_SCRATCH = "LLAMAS_SCRATCH_DIR"     # optional user override (base dir)

# Ray appends "session_<ts>_<pid>/sockets/<leaf>" (~70 chars) below _temp_dir.
_SOCKET_RESERVE = 80
_SOCKET_USABLE = (104 if sys.platform == "darwin" else 108) - 1

# ---------------------------------------------------------------------------
# module state (single pipeline process)
# ---------------------------------------------------------------------------
_run_temp_dir = None          # resolved per-run dir, "" = resolved-but-unusable, None = unresolved
_backstops_installed = False
_cleaned = False
_keep_scratch = False         # True ⇒ cleanup_scratch=false: shut Ray down but keep files
_ray_env_applied = False      # process-env knobs (mac large store, expiration) set once


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _as_bool(value, default):
    """Coerce a config value (bool or 'true'/'false' string) to bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


def _socket_safe(path):
    """True if Ray's socket path under *path* fits the platform AF_UNIX limit."""
    return len(path) + _SOCKET_RESERVE <= _SOCKET_USABLE


def _dir_size(path):
    """Total size in bytes of a file or directory tree (best effort)."""
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _pid_from_name(name):
    """Extract a pid from an owned run dir (``<pid>``) or a Ray ``session_<...>_<pid>``."""
    if name.isdigit():
        return int(name)
    if name.startswith("session_"):
        tail = name.rsplit("_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return None


def _pid_alive(pid):
    """True if a process with *pid* currently exists (protects concurrent runs)."""
    if pid is None:
        return False
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        # Fall back to os.kill(pid, 0); assume alive on ambiguity to be safe.
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return True


# ---------------------------------------------------------------------------
# temp-dir resolution
# ---------------------------------------------------------------------------
def resolve_run_temp_dir(config=None):
    """Resolve, create, and record the per-run Ray temp dir; register cleanup backstops.

    Precedence for the base directory:
    ``config['ray_temp_dir']`` → ``$LLAMAS_RAY_TEMP_DIR`` (user override) →
    ``config['scratch_dir']`` → ``$LLAMAS_SCRATCH_DIR`` → ``/tmp/llamas_ray``.

    A per-run ``<base>/<pid>`` subdirectory is used so concurrent runs never collide
    and cleanup is a single ``rmtree``. Returns the absolute path, or ``None`` if no
    socket-length-safe location exists (callers then pass ``_temp_dir=None`` and Ray
    uses its own default; cleanup is limited to the stale-prune).

    Idempotent: only the first call per process resolves; later calls return the same
    value. Also logs the resolved location and free disk space (warns if low).
    """
    global _run_temp_dir, _keep_scratch
    if _run_temp_dir is not None:
        return _run_temp_dir or None

    cfg = config or {}
    _keep_scratch = not _as_bool(cfg.get('cleanup_scratch'), True)
    base = (cfg.get('ray_temp_dir') or os.environ.get(_ENV_TEMP)
            or cfg.get('scratch_dir') or os.environ.get(_ENV_SCRATCH)
            or _DEFAULT_BASE)
    base = os.path.abspath(os.path.expanduser(str(base)))
    run = os.path.join(base, str(os.getpid()))

    if not _socket_safe(run) and base != _DEFAULT_BASE:
        logger.warning(
            "Ray temp base %r is too long for AF_UNIX sockets (base must be "
            "<= ~%d chars); falling back to %s",
            base, _SOCKET_USABLE - _SOCKET_RESERVE - 8, _DEFAULT_BASE)
        run = os.path.join(_DEFAULT_BASE, str(os.getpid()))

    if not _socket_safe(run):
        logger.warning("No socket-safe Ray temp dir available; using Ray's default "
                       "location. Cleanup will rely on the stale-prune only.")
        _run_temp_dir = ""            # resolved, but unusable
        _install_backstops()
        return None

    os.makedirs(run, exist_ok=True)
    os.environ[_ENV_TEMP] = run
    _run_temp_dir = run
    _install_backstops()

    try:
        free_gb = shutil.disk_usage(run).free / (1024 ** 3)
        msg = f"Ray scratch dir: {run} ({free_gb:.1f} GB free on that volume)"
        logger.warning(msg + " — LOW disk space") if free_gb < 5 else logger.info(msg)
    except OSError:
        logger.info("Ray scratch dir: %s", run)
    return run


def get_ray_temp_dir():
    """Return the per-run Ray temp dir for a ``ray.init(_temp_dir=...)`` call.

    Used by every stage's Ray init site. When called inside a pipeline run the
    driver has already resolved the dir; standalone callers resolve lazily (and get
    their own cleanup backstops). May return ``None`` (→ Ray default), which is a
    valid value for ``_temp_dir``.
    """
    if _run_temp_dir is not None:
        return _run_temp_dir or None
    if _ENV_TEMP in os.environ:
        return os.environ[_ENV_TEMP]
    return resolve_run_temp_dir(config=None)


def scratch_file(name):
    """Absolute path for a short-lived working file inside the run scratch dir.

    Falls back to the system temp dir if no owned scratch dir is available. Files
    placed here are swept by :func:`cleanup_scratch` even if a caller forgets to
    remove them.
    """
    d = get_ray_temp_dir()
    if not d:
        import tempfile
        d = tempfile.gettempdir()
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, name)


# ---------------------------------------------------------------------------
# in-run and exit cleanup
# ---------------------------------------------------------------------------
def _active_session_dir():
    """Path of the currently live Ray session dir, or ``None`` if Ray is down."""
    try:
        import ray
        if not ray.is_initialized():
            return None
        node = ray._private.worker._global_node
        if node is not None:
            return node.get_session_dir_path()
    except Exception:
        return None
    return None


def prune_run_sessions():
    """Remove finished Ray session dirs under our run dir (bounds mid-run growth).

    Safe to call between stages: it skips the currently-active session so a live Ray
    instance is never disturbed. Replaces the old ad-hoc ``py_modules_files`` globs.
    """
    if _keep_scratch:
        return
    d = _run_temp_dir or os.environ.get(_ENV_TEMP)
    if not d or not os.path.isdir(d):
        return
    active = _active_session_dir()
    active = os.path.abspath(active) if active else None
    for sub in glob.glob(os.path.join(d, "session_*")):
        if active and os.path.abspath(sub) == active:
            continue
        shutil.rmtree(sub, ignore_errors=True)


def cleanup_scratch():
    """Shut Ray down and remove the run scratch dir. Idempotent; safe on any exit."""
    global _cleaned
    if _cleaned:
        return
    _cleaned = True
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    if _keep_scratch:
        logger.info("cleanup_scratch=false — Ray shut down but scratch left in place: %s",
                    _run_temp_dir or os.environ.get(_ENV_TEMP))
        return
    d = _run_temp_dir or os.environ.get(_ENV_TEMP)
    if d and os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# exit / signal backstops
# ---------------------------------------------------------------------------
def _install_backstops():
    """Register ``atexit`` + ``SIGTERM``/``SIGHUP`` cleanup once (main thread only)."""
    global _backstops_installed
    if _backstops_installed:
        return
    _backstops_installed = True
    atexit.register(cleanup_scratch)
    for sig in (signal.SIGTERM, getattr(signal, "SIGHUP", None)):
        if sig is None:
            continue
        try:
            prev = signal.getsignal(sig)
        except (ValueError, OSError):
            continue

        def _handler(signum, frame, _prev=prev):
            cleanup_scratch()
            if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                _prev(signum, frame)
            else:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)   # re-raise so the exit code reflects the signal

        try:
            signal.signal(sig, _handler)
        except (ValueError, OSError):
            pass   # not on the main thread (e.g. a GUI worker) — atexit still covers us


# ---------------------------------------------------------------------------
# pre-flight input-reachability gate
# ---------------------------------------------------------------------------
def check_inputs_reachable(paths, timeout_s=15):
    """Fail fast when an input file's filesystem is unresponsive.

    Cloud-storage mounts (Box / iCloud / Dropbox) and dead network shares can
    make a plain ``os.stat`` block *indefinitely* when a file is offline or has
    not been downloaded locally. The pipeline stats every input at startup (in
    ``validate_pipeline_config``) and would otherwise hang there in an
    uninterruptible-looking ``stat`` with no message. This probes each path in a
    daemon thread against a shared wall-clock deadline and raises a clear,
    actionable :class:`RuntimeError` if any path does not respond in time,
    instead of hanging forever.

    A path whose stat returns "does not exist" is **not** reported here — that is
    a genuine missing-file case, left to the normal existence checks. Only paths
    whose stat never returns (unresponsive filesystem) are flagged.

    Parameters
    ----------
    paths : iterable of str
        Input file paths to probe (duplicates and empty values are ignored).
    timeout_s : float
        Wall-clock budget shared across all probes (min 1 s).
    """
    seen = list(dict.fromkeys(p for p in (paths or []) if p))
    if not seen:
        return

    budget = max(1.0, float(timeout_s))
    # os.path.exists swallows OSError → the thread finishes quickly for both
    # present and genuinely-missing files, and only stays alive when the
    # underlying stat blocks (unresponsive filesystem).
    threads = {p: threading.Thread(target=os.path.exists, args=(p,), daemon=True)
               for p in seen}
    for t in threads.values():
        t.start()

    deadline = time.monotonic() + budget
    for t in threads.values():
        t.join(max(0.0, deadline - time.monotonic()))

    unreachable = [p for p, t in threads.items() if t.is_alive()]
    if unreachable:
        listing = "\n  ".join(unreachable)
        raise RuntimeError(
            f"Input file(s) did not respond within {budget:.0f}s — the filesystem "
            f"holding them is unresponsive. This usually means a cloud-storage "
            f"mount (Box / iCloud / Dropbox) is offline or the file has not been "
            f"downloaded locally, or a network share is unavailable. Make the "
            f"files available offline (or copy them to a local disk and update the "
            f"config paths), then re-run.\nUnresponsive:\n  {listing}")


# ---------------------------------------------------------------------------
# pre-flight disk-space gate
# ---------------------------------------------------------------------------
def preflight_disk_check(config, output_dir, temp_dir, input_files):
    """Fail fast when disk space is guaranteed insufficient; warn when it is tight.

    Two tiers, evaluated per filesystem (temp and output on the same volume are
    checked once against the summed requirement):

    * **hard stop** — free space below the amount certain to be needed to even start
      (Ray object store + runtime-env bundle on the temp volume, one copy of the
      inputs on the output volume) or below ``min_free_gb`` → raise ``RuntimeError``
      *before any work*.
    * **warn** — free space above the floor but below the estimated full requirement
      (``disk_estimate_factor`` × input size) → log a warning and continue.

    Honours ``skip_disk_check`` as an escape hatch.
    """
    cfg = config or {}
    if cfg.get('skip_disk_check'):
        logger.info("Pre-flight disk-space check skipped (skip_disk_check=True).")
        return

    min_free_gb = float(cfg.get('min_free_gb', 5))
    factor = float(cfg.get('disk_estimate_factor', 5))

    # Ray object store, clamped to 30% RAM (mirrors the per-stage clamps).
    try:
        import psutil
        total_mb = psutil.virtual_memory().total // (1024 * 1024)
    except Exception:
        total_mb = 0
    store_mb = int(os.environ.get('LLAMAS_RAY_OBJECT_STORE_MB',
                                  cfg.get('ray_object_store_mb', 8192)))
    if total_mb:
        store_mb = min(store_mb, int(total_mb * 0.30))
    store_bytes = store_mb * 1024 * 1024
    bundle_bytes = 200 * 1024 * 1024

    input_bytes = 0
    for f in (input_files or []):
        try:
            input_bytes += os.path.getsize(f)
        except OSError:
            pass

    temp_need = store_bytes * 2 + bundle_bytes        # store + spill headroom + bundle
    temp_hard = store_bytes + bundle_bytes
    out_need = int(factor * input_bytes)
    floor = int(min_free_gb * (1024 ** 3))

    # Group requirements by filesystem device so a shared volume is checked once.
    needs = {}    # dev -> [need, hard, sample_path]

    def _add(path, need, hard):
        if not path:
            return
        probe = path
        while probe and not os.path.exists(probe):
            probe = os.path.dirname(probe)
        probe = probe or "/"
        try:
            dev = os.stat(probe).st_dev
        except OSError:
            return
        cur = needs.setdefault(dev, [0, 0, probe])
        cur[0] += need
        cur[1] = max(cur[1], hard)

    _add(temp_dir or _DEFAULT_BASE, temp_need, temp_hard)
    _add(output_dir, out_need, input_bytes)

    gib = 1024 ** 3
    for _dev, (need, hard, sample) in needs.items():
        try:
            free = shutil.disk_usage(sample).free
        except OSError:
            continue
        hard_min = max(hard, floor)
        if free < hard_min:
            raise RuntimeError(
                f"Insufficient disk space on the volume containing {sample}: "
                f"{free / gib:.1f} GB free, but at least {hard_min / gib:.1f} GB is "
                f"required to start. Aborting before any files are written. Free up "
                f"space, point scratch_dir/output_dir at a larger volume, or set "
                f"skip_disk_check=True to override.")
        if free < need:
            logger.warning(
                "Low disk space on the volume containing %s: %.1f GB free, estimated "
                "~%.1f GB needed for this run — it may fail if the volume fills up.",
                sample, free / gib, need / gib)
        else:
            logger.info("Disk check OK for %s: %.1f GB free (estimated need ~%.1f GB).",
                        sample, free / gib, need / gib)


# ---------------------------------------------------------------------------
# discovery + stale prune (shared with the clean_ray_scratch CLI)
# ---------------------------------------------------------------------------
def _candidate_bases(config=None):
    """Owned base dirs that may hold LLAMAS run scratch (deduplicated, absolute)."""
    cfg = config or {}
    bases = {_DEFAULT_BASE}
    for b in (cfg.get('ray_temp_dir'), cfg.get('scratch_dir'),
              os.environ.get(_ENV_TEMP), os.environ.get(_ENV_SCRATCH)):
        if b:
            bases.add(os.path.abspath(os.path.expanduser(str(b))))
    # The env-exported temp dir is "<base>/<pid>"; include its parent too.
    live = os.environ.get(_ENV_TEMP)
    if live:
        bases.add(os.path.dirname(os.path.abspath(live)))
    return bases


def discover_scratch(config=None, include_generic=False):
    """List accumulated Ray scratch as dicts (path, size, mtime, pid, alive, kind).

    Covers our owned ``<base>/<pid>`` run dirs. With *include_generic*, also lists
    generic Ray sessions (``/tmp/ray/session_*`` and ``$TMPDIR/ray/session_*``) that
    may have been left by crashed runs (or other Ray apps — hence opt-in).
    """
    keep = os.path.abspath(_run_temp_dir) if _run_temp_dir else None
    seen = set()
    entries = []

    def _record(path, kind):
        ap = os.path.abspath(path)
        if ap in seen or not os.path.exists(ap):
            return
        seen.add(ap)
        pid = _pid_from_name(os.path.basename(ap))
        try:
            mtime = os.path.getmtime(ap)
        except OSError:
            mtime = 0
        entries.append({
            "path": ap,
            "size": _dir_size(ap),
            "mtime": mtime,
            "pid": pid,
            "alive": _pid_alive(pid),
            "kind": kind,
            "is_current_run": keep is not None and ap == keep,
        })

    for base in _candidate_bases(config):
        if not os.path.isdir(base):
            continue
        for p in glob.glob(os.path.join(base, "*")):
            name = os.path.basename(p)
            if os.path.isdir(p) and (name.isdigit() or name.startswith("session_")):
                _record(p, "owned")

    if include_generic:
        ray_bases = ["/tmp/ray"]
        tmp = os.environ.get("TMPDIR")
        if tmp:
            ray_bases.append(os.path.join(tmp, "ray"))
        for rb in ray_bases:
            for p in glob.glob(os.path.join(rb, "session_*")):
                if os.path.isdir(p):
                    _record(p, "generic")

    return entries


def prune_stale(config=None, hours=6, include_generic=False):
    """Delete scratch left by crashed/killed prior runs (SIGKILL / power-loss safety net).

    Removes owned run dirs (and, if *include_generic*, generic Ray sessions) that are
    older than *hours* and whose owning pid is no longer alive, never touching the
    current run's dir or a live concurrent run's dir. Returns bytes reclaimed.
    """
    cfg = config or {}
    if include_generic is False:
        include_generic = bool(cfg.get('prune_ray_sessions'))
    cutoff = time.time() - hours * 3600
    reclaimed = 0
    for e in discover_scratch(config, include_generic=include_generic):
        if e["is_current_run"] or e["alive"] or e["mtime"] > cutoff:
            continue
        shutil.rmtree(e["path"], ignore_errors=True)
        reclaimed += e["size"]
    if reclaimed:
        logger.info("Reclaimed %.1f GB of stale Ray scratch.", reclaimed / (1024 ** 3))
    return reclaimed


# ---------------------------------------------------------------------------
# Phase 2: single-session consolidation
# ---------------------------------------------------------------------------
def build_runtime_env():
    """The one canonical Ray runtime_env for the whole pipeline.

    Uses ``py_modules`` (not ``working_dir``: the repo root exceeds Ray's 512 MB
    working_dir limit, and py_modules doesn't change the worker cwd). Shipping the
    package ensures workers import *this* checkout regardless of any editable
    install. Excludes the heavy/unneeded trees but **keeps ``LUT/``** — trace reads
    ``LUT_DIR/traceLUT.json`` on the worker (``**/Docs/**`` already covers DATA_DIR).
    """
    import pkg_resources   # heavy + only needed on the driver at init time
    package_root = os.path.dirname(pkg_resources.resource_filename("llamas_pyjamas", ""))
    return {
        "py_modules": [package_root],
        "env_vars": {"PYTHONPATH": f"{package_root}:{os.environ.get('PYTHONPATH', '')}"},
        "excludes": [
            "**/*.fits", "**/*.pkl", "**/.git/**",
            "**/*.zip", "**/*.zip/**", "**/*.tar.gz", "**/*.tar.gz/**",
            "**/mastercalib*/**",
            "**/reduced/**", "**/extractions/**", "**/cubes/**", "**/traces/**",
            "**/testing/**", "**/Test/**", "**/Docs/**", "**/arc_testing/**",
            "**/__pycache__/**", "**/*.pyc",
        ],
    }


def _num_cpus(num_cpus=None):
    if num_cpus is not None:
        return int(num_cpus)
    import multiprocessing
    return int(os.environ.get("LLAMAS_RAY_CPUS", multiprocessing.cpu_count()))


def _object_store_bytes(object_store_mb=None, config=None):
    """Object-store size in bytes, clamped to ≤30% RAM (mirrors the old per-stage clamp)."""
    cfg = config or {}
    if object_store_mb is None:
        object_store_mb = int(os.environ.get("LLAMAS_RAY_OBJECT_STORE_MB",
                                             cfg.get("ray_object_store_mb", 8192)))
    else:
        object_store_mb = int(object_store_mb)
    try:
        import psutil
        total_mb = psutil.virtual_memory().total // (1024 * 1024)
        cap_mb = int(total_mb * 0.30)
        if object_store_mb > cap_mb:
            logger.warning("Clamping Ray object store %d → %d MB (30%% of %d MB RAM)",
                           object_store_mb, cap_mb, total_mb)
            object_store_mb = cap_mb
    except Exception:
        pass
    return object_store_mb * 1024 * 1024


def _spill_dir(config, temp_dir):
    """Optional separate object-spilling directory (a big volume for small-/tmp machines).

    Returns a spill path only when the user set ``scratch_dir``/``$LLAMAS_SCRATCH_DIR``;
    otherwise ``None`` → Ray spills into ``_temp_dir`` (already owned + cleaned). The
    spill dir has no socket-length limit, so it may be a long path on a large volume.
    """
    cfg = config or {}
    spill = cfg.get("scratch_dir") or os.environ.get(_ENV_SCRATCH)
    if not spill:
        return None
    spill = os.path.join(os.path.abspath(os.path.expanduser(str(spill))), "ray_spill")
    try:
        os.makedirs(spill, exist_ok=True)
    except OSError:
        return None
    return spill


def init_ray(config=None, num_cpus=None, object_store_mb=None, runtime_env=None,
             reuse=True, force_refresh=False, address=None):
    """Start, or attach to, the single Ray session for the whole run.

    Consolidation: one session per run means the package uploads once and ``/tmp``
    churn is minimized. ``reuse=True`` (default) attaches to a live session;
    ``force_refresh=True`` tears the current session down and re-inits (re-bundling
    source — used by the GUI per extraction click). ``address='auto'`` attaches to an
    external cluster and owns none of the temp/store/spill config. Returns
    ``ray.cluster_resources()`` (``None`` for the external-attach path).
    """
    global _ray_env_applied
    import ray

    # Attach to an external cluster: the head owns temp dir / store / spill.
    if address is not None:
        if not ray.is_initialized():
            ray.init(address=address, ignore_reinit_error=True)
        return None

    # Ensure the owned temp dir + atexit/SIGTERM backstops exist (idempotent).
    temp_dir = resolve_run_temp_dir(config)

    if ray.is_initialized():
        if reuse and not force_refresh:
            return ray.cluster_resources()
        ray.shutdown()   # force_refresh: re-bundle updated source into a fresh session

    # A shared session honours only its first init's process env — set these once.
    if not _ray_env_applied:
        os.environ["RAY_ENABLE_MAC_LARGE_OBJECT_STORE"] = "1"
        os.environ.setdefault("RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S", "1200")
        _ray_env_applied = True

    kwargs = dict(
        num_cpus=_num_cpus(num_cpus),
        object_store_memory=_object_store_bytes(object_store_mb, config),
        runtime_env=runtime_env or build_runtime_env(),
        include_dashboard=False,
        log_to_driver=False,
        ignore_reinit_error=True,
    )
    if temp_dir:
        kwargs["_temp_dir"] = temp_dir
        spill = _spill_dir(config, temp_dir)
        if spill:
            kwargs["_system_config"] = {
                "object_spilling_config": json.dumps(
                    {"type": "filesystem", "params": {"directory_path": [spill]}}),
                "local_fs_capacity_threshold": 0.95,
            }
    ray.init(**kwargs)
    logger.info("Ray session started (consolidated): temp=%s num_cpus=%s object_store<=30%%RAM",
                temp_dir or "(Ray default)", kwargs["num_cpus"])
    return ray.cluster_resources()


def shutdown_ray():
    """Shut Ray down without removing the scratch dir (atexit/cleanup_scratch remove it).

    For explicit teardown in standalone entrypoints; the pipeline relies on the
    ``finally: cleanup_scratch()`` + atexit backstops instead.
    """
    try:
        import ray
        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
