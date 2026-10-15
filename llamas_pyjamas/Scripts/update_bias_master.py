#!/usr/bin/env python3
"""
llamas_pyjamas.Scripts.update_bias_master
=========================================
Install QA-vetted raw bias frames as the packaged master biases.

Scans a raw-data directory for bias frames (primary-header ``PRODCATG`` of
``CAL.R-BIA``), runs the ``llamas-checks`` QA suite on each one, and installs
one master per readout mode (SLOW / FAST) as ``{slow,fast}_master_bias.fits``
in both the package ``Bias/`` directory and ``mastercalib/``:

* every frame whose verdict is ``pass`` is **median-combined** (detector by
  detector, see ``Bias.llamasBias.median_combine_mef``), which removes the
  cosmic-ray hits and hot pixels a single frame would bake into every image;
* if a mode has no passing frame, the ``warn`` frames are combined instead;
  ``--no-warn-fallback`` turns that off;
* with ``--no-combine`` only the first acceptable frame (sorted filename
  order) is copied, as the script did before 2026-10.

``llamas-checks`` (package ``llamas_checks``) is an *optional* dependency of
this script only. It is found through a normal import, never a hard-coded
path, so the script runs unchanged on any machine where both packages are
installed in the same environment. Where it is absent the script refuses to
install unvetted frames unless ``--no-qa`` is given.

Exit status and messages are the contract the observing GUI relies on:
``0`` and ``Master BIAS files updated successfully (SLOW, FAST).`` on
success; ``1`` and ``Update Failed: <reason>`` on stderr otherwise.

The frame-finding, QA and selection helpers here are shared with
``update_quicklook_calibs``, which also installs master darks and the
twilight sky reference for the quick-look white light.

Usage::

    python -m llamas_pyjamas.Scripts.update_bias_master RAW_DIR
    python -m llamas_pyjamas.Scripts.update_bias_master RAW_DIR --no-warn-fallback
    python -m llamas_pyjamas.Scripts.update_bias_master RAW_DIR --no-combine
    python -m llamas_pyjamas.Scripts.update_bias_master RAW_DIR --no-qa
"""

import argparse
import logging
import os
import shutil
import sys
from glob import glob
from pathlib import Path

from astropy.io import fits

from llamas_pyjamas.config import BIAS_DIR, CALIB_DIR
from llamas_pyjamas.Bias.llamasBias import median_combine_mef
from llamas_pyjamas.File.llamasIO import exposure_time

# llamas-checks is an OPTIONAL dependency used by this script alone, never by
# the reduction pipeline. The guarded import is deliberate: machines without
# it must still import this module and can run it with --no-qa.
try:
    from llamas_checks import qa_engine
    from llamas_checks.paths import CAL_CONFIG_NAME as _QA_CAL_CONFIG_NAME
    from llamas_checks.paths import CONFIG_DIR as _QA_CONFIG_DIR
    HAVE_LLAMAS_CHECKS = True
except ImportError:
    qa_engine = None
    _QA_CAL_CONFIG_NAME = None
    _QA_CONFIG_DIR = None
    HAVE_LLAMAS_CHECKS = False

logger = logging.getLogger(__name__)

MODES = ('SLOW', 'FAST')
BIAS_PRODCATG = 'CAL.R-BIA'
# llamas-checks rule set activated by each calibration PRODCATG.
RULE_SET_BY_PRODCATG = {'CAL.R-BIA': 'BIAS', 'CAL.R-DRK': 'DARK', 'CAL.R-SKY': 'SKY_FLAT'}
FAIL_MSG = 'Update Failed'
# Verdicts that may be installed, in order of preference. 'unchecked' is only
# produced in --no-qa mode and only accepted there.
PREFERENCE_WITH_WARN = ('pass', 'warn')
PREFERENCE_PASS_ONLY = ('pass',)


def find_frames(directory, prodcatg):
    """Return ``[{'path', 'readmode', 'exptime'}]`` for every frame in *directory*
    whose primary-header ``PRODCATG`` starts with *prodcatg*.

    Files named like white-light or difference products are skipped, as are
    files whose header cannot be read (logged, not raised). Results are in
    sorted filename order, which is the tie-break order downstream.
    """
    files = sorted(glob(os.path.join(directory, '*.fits')))
    candidates = []
    for path in files:
        name = os.path.basename(path).lower()
        if 'white' in name or 'diff' in name:
            continue
        try:
            hdr = fits.getheader(path, 0)
        except Exception as exc:
            logger.warning('Skipping %s: header read failed (%s)', os.path.basename(path), exc)
            continue
        found = str(hdr.get('PRODCATG', '') or '').strip().upper()
        if not found.startswith(prodcatg):
            continue
        readmode = str(hdr.get('READ-MDE', '') or '').strip().upper()
        candidates.append({'path': path, 'readmode': readmode, 'exptime': exposure_time(hdr)})
    return candidates


def find_bias_candidates(directory):
    """Bias frames in *directory*: ``PRODCATG`` starting ``CAL.R-BIA``, the same
    key ``llamas-checks`` uses to activate its BIAS rule set."""
    return find_frames(directory, BIAS_PRODCATG)


def qa_verdicts(candidates, rule_set='BIAS'):
    """Run the llamas-checks calibration suite on each candidate.

    Cameras allowed to be down come from llamas-checks' own configuration
    (``camera_status.yaml``, or its environment override), as for its CLI.

    Returns the candidates with a ``verdict`` of ``pass`` / ``warn`` / ``fail``
    / ``error`` (lower-case) and ``readmode`` ``SLOW`` / ``FAST`` as reported
    by the engine. Frames for which the engine does not activate *rule_set*
    (``BIAS``, ``DARK`` or ``SKY_FLAT``) are dropped. A per-frame engine error
    is recorded as ``error`` rather than aborting the run; a problem loading
    the bundled config raises ``QAEngineError`` to the caller.
    """
    config_path = _QA_CONFIG_DIR / _QA_CAL_CONFIG_NAME
    config = qa_engine.load_yaml(config_path)
    qa_engine.validate_config(config, config_path)
    down_names, down_source = qa_engine.resolve_cameras_down(None)
    if down_names:
        logger.info('Cameras allowed down (%s): %s', down_source, ', '.join(down_names))
    engine = qa_engine.QAEngine(config, cameras_down=down_names)

    verdicts = []
    for cand in candidates:
        name = os.path.basename(cand['path'])
        try:
            report = engine.run(Path(cand['path']))
        except qa_engine.QAEngineError as exc:
            logger.warning('QA error on %s: %s', name, exc)
            verdicts.append({**cand, 'verdict': 'error'})
            continue
        if rule_set not in (report.get('active_rule_sets') or []):
            logger.info('Skipping %s: llamas-checks did not classify it as %s', name, rule_set)
            continue
        verdict = str(report.get('overall_verdict', 'ERROR')).lower()
        readmode = report.get('metadata', {}).get('readout_mode') or cand['readmode']
        verdicts.append({**cand, 'readmode': str(readmode).upper(), 'verdict': verdict})
    return verdicts


def select_bias_frames(verdicts, allow_warn=True, accept_unchecked=False):
    """All acceptable frames per readout mode. Pure function, no I/O.

    For each mode, every ``pass`` frame (in list order) is selected. If there is
    none and *allow_warn*, every ``warn`` frame is taken instead; tiers are
    never mixed. ``fail`` / ``error`` are never chosen. ``unchecked`` (from
    ``--no-qa``) is chosen only when *accept_unchecked*.
    Returns ``{'SLOW': [paths], 'FAST': [paths]}`` (lists may be empty).
    """
    preference = PREFERENCE_WITH_WARN if allow_warn else PREFERENCE_PASS_ONLY
    if accept_unchecked:
        preference = preference + ('unchecked',)

    selection = {}
    for mode in MODES:
        in_mode = [v for v in verdicts if v['readmode'] == mode]
        chosen = []
        for wanted in preference:
            chosen = [v['path'] for v in in_mode if v['verdict'] == wanted]
            if chosen:
                break
        selection[mode] = chosen
    return selection


def select_bias(verdicts, allow_warn=True, accept_unchecked=False):
    """First acceptable frame per readout mode (``select_bias_frames`` tie-broken
    by list order). Returns ``{'SLOW': path | None, 'FAST': path | None}``."""
    frames = select_bias_frames(verdicts, allow_warn=allow_warn, accept_unchecked=accept_unchecked)
    return {mode: (paths[0] if paths else None) for mode, paths in frames.items()}


def install_master(paths, target_name, bias_dir, calib_dir, subtract=None, extra_primary=None):
    """Install one master calibration file as *target_name* in both directories.

    A single frame with nothing to subtract is copied unchanged (the raw frame
    is the master, as before). Otherwise the frames are median-combined with
    ``median_combine_mef`` (optionally minus *subtract*, for darks) into
    ``bias_dir/target_name`` and that file is copied to *calib_dir*. Returns the
    two installed paths. Raises ``OSError`` on copy failure.
    """
    paths = list(paths)
    if not paths:
        raise ValueError('install_master: no frames to install')
    os.makedirs(bias_dir, exist_ok=True)
    os.makedirs(calib_dir, exist_ok=True)
    primary_target = os.path.join(bias_dir, target_name)
    if len(paths) == 1 and subtract is None and not extra_primary:
        shutil.copy2(paths[0], primary_target)
        logger.info('Installed %s as %s', os.path.basename(paths[0]), target_name)
    else:
        median_combine_mef(paths, primary_target, subtract=subtract, extra_primary=extra_primary)
        logger.info('Installed median of %d frames as %s', len(paths), target_name)
    secondary_target = os.path.join(calib_dir, target_name)
    if os.path.abspath(secondary_target) != os.path.abspath(primary_target):
        shutil.copy2(primary_target, secondary_target)
    return [primary_target, secondary_target]


def install_bias(selection, bias_dir, calib_dir, combine=True):
    """Install the selected bias frames as ``{mode}_master_bias.fits`` in both dirs.

    *selection* maps each mode to a list of paths (``select_bias_frames``) or a
    single path / None (``select_bias``). With *combine* every listed frame is
    median-combined; otherwise only the first is copied. Both locations are
    read by the pipeline (``config.SLOW_BIAS_FILE`` / ``FAST_BIAS_FILE`` and
    ``Bias.biasFirst.resolve_master_bias_file``). Returns the modes installed.
    """
    installed = []
    for mode in MODES:
        chosen = selection.get(mode)
        if not chosen:
            continue
        paths = [chosen] if isinstance(chosen, str) else list(chosen)
        if not combine:
            paths = paths[:1]
        install_master(paths, f'{mode.lower()}_master_bias.fits', bias_dir, calib_dir)
        installed.append(mode)
    return installed


def _fail(reason):
    sys.stdout.flush()
    print(f'{FAIL_MSG}: {reason}', file=sys.stderr)
    sys.exit(1)


def print_verdict_table(verdicts, selection):
    """One line per candidate: file, mode, verdict, and ``*`` if it will be used."""
    chosen = set()
    for value in selection.values():
        if isinstance(value, str):
            chosen.add(value)
        elif value:
            chosen.update(value)
    print(f"{'file':<48} {'mode':<5} {'verdict':<9} chosen")
    for v in verdicts:
        mark = '*' if v['path'] in chosen else ''
        print(f"{os.path.basename(v['path']):<48} {v['readmode'] or '?':<5} "
              f"{v['verdict']:<9} {mark}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Install the QA-passing SLOW and FAST bias frames from a raw '
                    'directory (median-combined) as the packaged master biases.')
    parser.add_argument('directory', help='Directory containing raw LLAMAS MEF files')
    parser.add_argument('--bias-dir', default=BIAS_DIR,
                        help='Directory for the package master bias files')
    parser.add_argument('--calib-dir', default=CALIB_DIR,
                        help='Directory for the master calibration copies')
    parser.add_argument('--no-warn-fallback', action='store_true',
                        help="Only accept a QA 'pass'; do not fall back to 'warn' frames")
    parser.add_argument('--no-combine', action='store_true',
                        help='Copy the first acceptable frame per mode instead of the median '
                             'of all acceptable frames')
    parser.add_argument('--no-qa', action='store_true',
                        help='Skip llamas-checks and take every bias per mode by header '
                             '(unvetted; use only when llamas-checks is unavailable)')
    parser.add_argument('-v', '--verbose', action='store_true', help='Show INFO logging')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='%(levelname)s: %(message)s')

    directory = os.path.abspath(args.directory)
    if not os.path.isdir(directory):
        _fail(f'directory not found: {directory}')

    if not args.no_qa and not HAVE_LLAMAS_CHECKS:
        _fail('llamas-checks is not installed in this environment; '
              'install it or rerun with --no-qa')

    candidates = find_bias_candidates(directory)
    if not candidates:
        _fail(f'no bias frames (PRODCATG={BIAS_PRODCATG}) found in {directory}')

    if args.no_qa:
        print('WARNING: --no-qa: installing biases WITHOUT QA vetting', file=sys.stderr)
        verdicts = [{**c, 'verdict': 'unchecked'} for c in candidates]
    else:
        try:
            verdicts = qa_verdicts(candidates, 'BIAS')
        except qa_engine.QAEngineError as exc:
            _fail(f'llamas-checks could not run: {exc}')

    selection = select_bias_frames(verdicts,
                                   allow_warn=not args.no_warn_fallback,
                                   accept_unchecked=args.no_qa)
    if args.no_combine:
        selection = {mode: paths[:1] for mode, paths in selection.items()}
    print_verdict_table(verdicts, selection)

    if not any(selection.values()):
        accepted = 'pass' if args.no_warn_fallback else 'pass or warn'
        _fail(f'no bias frame with QA verdict {accepted} found for either read mode')

    try:
        installed = install_bias(selection, args.bias_dir, args.calib_dir,
                                 combine=not args.no_combine)
    except OSError as exc:
        _fail(f'copy failed: {exc}')

    for mode in MODES:
        if mode not in installed:
            print(f'Note: {mode} master bias not updated (no acceptable frame).')
        else:
            print(f'{mode}: median of {len(selection[mode])} frame(s).')
    print(f"Master BIAS files updated successfully ({', '.join(installed)}).")
    sys.exit(0)


if __name__ == '__main__':
    main()
