"""Standalone cleanup tool for accumulated Ray scratch directories.

Existing users often have a large pile of ``/tmp/ray/session_*`` (and, after the
temp-dir fix, ``/tmp/llamas_ray/<pid>``) directories left behind by many pipeline
runs. This script finds and removes them safely, independently of running the
pipeline.

Usage
-----
    # show what would be removed (default: dry run, deletes nothing)
    python -m llamas_pyjamas.Utils.clean_ray_scratch

    # actually delete owned LLAMAS scratch dirs
    python -m llamas_pyjamas.Utils.clean_ray_scratch --yes

    # only remove dirs older than 12 h, and also sweep generic /tmp/ray sessions
    python -m llamas_pyjamas.Utils.clean_ray_scratch --yes --older-than 12 --include-generic

    # read scratch_dir/ray_temp_dir from a pipeline config file
    python -m llamas_pyjamas.Utils.clean_ray_scratch --config myrun.txt --yes

Safety
------
* Dry run is the default; nothing is deleted without ``--yes``.
* A session whose owning PID is still alive, or that was modified within
  ``--min-age`` minutes, is **skipped** — this protects a pipeline that is running
  concurrently on the same machine.
* Generic ``/tmp/ray/session_*`` (which may belong to other Ray applications or
  users) is only touched with ``--include-generic``.
"""

import argparse
import os
import shutil
import time

from llamas_pyjamas.Utils.rayManager import discover_scratch


def _parse_config(path):
    """Minimal key=value config reader (mirrors reduce.py's parser for our keys)."""
    cfg = {}
    if not path:
        return cfg
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            cfg[key.strip()] = value.strip()
    return cfg


def _fmt_size(nbytes):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if nbytes < 1024 or unit == 'TB':
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024


def _fmt_age(mtime):
    if not mtime:
        return "unknown"
    hours = (time.time() - mtime) / 3600
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} d"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Clean up accumulated Ray scratch/session directories.")
    parser.add_argument('--config', type=str, default=None,
                        help='Pipeline config file to read scratch_dir/ray_temp_dir from')
    parser.add_argument('--older-than', type=float, default=0.0, metavar='HOURS',
                        help='Only remove dirs older than this many hours (default: all)')
    parser.add_argument('--min-age', type=float, default=10.0, metavar='MINUTES',
                        help='Never touch dirs modified within this many minutes (default: 10)')
    parser.add_argument('--include-generic', action='store_true',
                        help='Also sweep generic /tmp/ray/session_* (may belong to other apps)')
    parser.add_argument('--logs', action='store_true',
                        help='Also prune old llamas_pipeline_*.log files (keeps newest 10)')
    parser.add_argument('--yes', '--force', dest='yes', action='store_true',
                        help='Actually delete (default is a dry run)')
    parser.add_argument('--verbose', action='store_true', help='List skipped dirs too')
    args = parser.parse_args(argv)

    cfg = _parse_config(args.config)
    entries = discover_scratch(cfg, include_generic=args.include_generic)

    now = time.time()
    age_cutoff = now - args.older_than * 3600
    min_age_cutoff = now - args.min_age * 60

    to_remove, skipped = [], []
    for e in entries:
        reason = None
        if e['is_current_run']:
            reason = "current run"
        elif e['alive']:
            reason = f"pid {e['pid']} alive"
        elif e['mtime'] > min_age_cutoff:
            reason = "recently modified"
        elif args.older_than and e['mtime'] > age_cutoff:
            reason = "newer than --older-than"
        (skipped if reason else to_remove).append((e, reason))

    total = sum(e['size'] for e, _ in to_remove)
    print(f"Found {len(entries)} scratch dir(s); "
          f"{len(to_remove)} removable, {len(skipped)} skipped.\n")

    for e, _ in sorted(to_remove, key=lambda x: -x[0]['size']):
        action = "DELETE" if args.yes else "would delete"
        print(f"  [{action}] {e['path']}  ({_fmt_size(e['size'])}, age {_fmt_age(e['mtime'])}, {e['kind']})")
    if args.verbose:
        for e, reason in skipped:
            print(f"  [skip: {reason}] {e['path']}  ({_fmt_size(e['size'])})")

    print(f"\nReclaimable: {_fmt_size(total)}")

    removed = 0
    if args.yes:
        for e, _ in to_remove:
            shutil.rmtree(e['path'], ignore_errors=True)
            removed += e['size']
        print(f"Removed {_fmt_size(removed)}.")
    elif to_remove:
        print("Dry run — nothing deleted. Re-run with --yes to remove the above.")

    if args.logs:
        _prune_logs(cfg)

    return 0


def _prune_logs(cfg, keep=10):
    """Keep only the newest *keep* llamas_pipeline_*.log files in the log dir."""
    import glob
    log_dir = cfg.get('log_output_dir')
    if not log_dir or not os.path.isdir(log_dir):
        return
    logs = sorted(glob.glob(os.path.join(log_dir, 'llamas_pipeline_*.log')),
                  key=os.path.getmtime, reverse=True)
    for old in logs[keep:]:
        try:
            os.remove(old)
        except OSError:
            pass
    if len(logs) > keep:
        print(f"Pruned {len(logs) - keep} old log file(s) in {log_dir}.")


if __name__ == '__main__':
    raise SystemExit(main())
