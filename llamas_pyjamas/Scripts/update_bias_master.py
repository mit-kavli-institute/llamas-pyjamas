#!/usr/bin/env python3
"""
llamas_pyjamas.Scripts.update_bias_master
=========================================
Install QA-vetted raw bias frames as the packaged master biases.

Scans a raw-data directory for bias frames (primary-header ``PRODCATG`` of
``CAL.R-BIA``), runs the ``llamas-checks`` QA suite on each one, and copies
the **first** frame per readout mode (SLOW / FAST, in sorted filename order)
whose verdict is ``pass`` to ``{slow,fast}_master_bias.fits`` in both the
package ``Bias/`` directory and ``mastercalib/``. If a mode has no passing
frame the first ``warn`` frame is used instead; ``--no-warn-fallback`` turns
that off.

``llamas-checks`` (package ``llamas_checks``) is an *optional* dependency of
this script only. It is found through a normal import, never a hard-coded
path, so the script runs unchanged on any machine where both packages are
installed in the same environment. Where it is absent the script refuses to
install an unvetted frame unless ``--no-qa`` is given.

Exit status and messages are the contract the observing GUI relies on:
``0`` and ``Master BIAS files updated successfully (SLOW, FAST).`` on
success; ``1`` and ``Update Failed: <reason>`` on stderr otherwise.

Usage::

    python -m llamas_pyjamas.Scripts.update_bias_master RAW_DIR
    python -m llamas_pyjamas.Scripts.update_bias_master RAW_DIR --no-warn-fallback
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
FAIL_MSG = 'Update Failed'
# Verdicts that may be installed, in order of preference. 'unchecked' is only
# produced in --no-qa mode and only accepted there.
PREFERENCE_WITH_WARN = ('pass', 'warn')
PREFERENCE_PASS_ONLY = ('pass',)


def find_bias_candidates(directory):
    """Return ``[{'path', 'readmode'}]`` for every bias frame in *directory*.

    A bias is any ``.fits`` whose primary header ``PRODCATG`` starts with
    ``CAL.R-BIA`` -- the same key ``llamas-checks`` uses to activate its BIAS
    rule set. Files named like white-light or difference products are skipped,
    as are files whose header cannot be read (logged, not raised). Results are
    in sorted filename order, which is the tie-break order downstream.
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
        prodcatg = str(hdr.get('PRODCATG', '') or '').strip().upper()
        if not prodcatg.startswith(BIAS_PRODCATG):
            continue
        readmode = str(hdr.get('READ-MDE', '') or '').strip().upper()
        candidates.append({'path': path, 'readmode': readmode})
    return candidates


def qa_verdicts(candidates):
    """Run the llamas-checks calibration suite on each candidate.

    Cameras allowed to be down come from llamas-checks' own configuration
    (``camera_status.yaml``, or its environment override), as for its CLI.

    Returns ``[{'path', 'readmode', 'verdict'}]`` with ``verdict`` one of
    ``pass`` / ``warn`` / ``fail`` / ``error`` (lower-case) and ``readmode``
    ``SLOW`` / ``FAST`` as reported by the engine. Frames for which the engine
    does not activate the BIAS rule set are dropped. A per-frame engine error
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
        if 'BIAS' not in (report.get('active_rule_sets') or []):
            logger.info('Skipping %s: llamas-checks did not classify it as a bias', name)
            continue
        verdict = str(report.get('overall_verdict', 'ERROR')).lower()
        readmode = report.get('metadata', {}).get('readout_mode') or cand['readmode']
        verdicts.append({'path': cand['path'],
                         'readmode': str(readmode).upper(),
                         'verdict': verdict})
    return verdicts


def select_bias(verdicts, allow_warn=True, accept_unchecked=False):
    """Pick one frame per readout mode. Pure function, no I/O.

    For each mode the first ``pass`` (in list order) wins. If none and
    *allow_warn*, the first ``warn`` is taken. ``fail`` / ``error`` are never
    chosen. ``unchecked`` (from ``--no-qa``) is chosen only when
    *accept_unchecked*. Returns ``{'SLOW': path | None, 'FAST': path | None}``.
    """
    preference = PREFERENCE_WITH_WARN if allow_warn else PREFERENCE_PASS_ONLY
    if accept_unchecked:
        preference = preference + ('unchecked',)

    selection = {}
    for mode in MODES:
        in_mode = [v for v in verdicts if v['readmode'] == mode]
        chosen = None
        for wanted in preference:
            chosen = next((v['path'] for v in in_mode if v['verdict'] == wanted), None)
            if chosen is not None:
                break
        selection[mode] = chosen
    return selection


def install_bias(selection, bias_dir, calib_dir):
    """Copy each selected frame to ``{mode}_master_bias.fits`` in both dirs.

    Both locations are read by the pipeline (``config.SLOW_BIAS_FILE`` /
    ``FAST_BIAS_FILE`` and ``Bias.biasFirst.resolve_master_bias_file``).
    Returns the list of modes installed. Raises ``OSError`` on copy failure.
    """
    installed = []
    for mode in MODES:
        src = selection.get(mode)
        if src is None:
            continue
        target_name = f'{mode.lower()}_master_bias.fits'
        for dest_dir in (bias_dir, calib_dir):
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dest_dir, target_name))
        logger.info('Installed %s as %s', os.path.basename(src), target_name)
        installed.append(mode)
    return installed


def _fail(reason):
    sys.stdout.flush()
    print(f'{FAIL_MSG}: {reason}', file=sys.stderr)
    sys.exit(1)


def _print_table(verdicts, selection):
    chosen = {p for p in selection.values() if p}
    print(f"{'file':<48} {'mode':<5} {'verdict':<9} chosen")
    for v in verdicts:
        mark = '*' if v['path'] in chosen else ''
        print(f"{os.path.basename(v['path']):<48} {v['readmode'] or '?':<5} "
              f"{v['verdict']:<9} {mark}")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Install the first QA-passing SLOW and FAST bias frame from a raw '
                    'directory as the packaged master biases.')
    parser.add_argument('directory', help='Directory containing raw LLAMAS MEF files')
    parser.add_argument('--bias-dir', default=BIAS_DIR,
                        help='Directory for the package master bias files')
    parser.add_argument('--calib-dir', default=CALIB_DIR,
                        help='Directory for the master calibration copies')
    parser.add_argument('--no-warn-fallback', action='store_true',
                        help="Only accept a QA 'pass'; do not fall back to a 'warn' frame")
    parser.add_argument('--no-qa', action='store_true',
                        help='Skip llamas-checks and take the first bias per mode by header '
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
        print('WARNING: --no-qa: installing the first bias per mode WITHOUT QA vetting',
              file=sys.stderr)
        verdicts = [{**c, 'verdict': 'unchecked'} for c in candidates]
    else:
        try:
            verdicts = qa_verdicts(candidates)
        except qa_engine.QAEngineError as exc:
            _fail(f'llamas-checks could not run: {exc}')

    selection = select_bias(verdicts,
                            allow_warn=not args.no_warn_fallback,
                            accept_unchecked=args.no_qa)
    _print_table(verdicts, selection)

    if not any(selection.values()):
        accepted = 'pass' if args.no_warn_fallback else 'pass or warn'
        _fail(f'no bias frame with QA verdict {accepted} found for either read mode')

    try:
        installed = install_bias(selection, args.bias_dir, args.calib_dir)
    except OSError as exc:
        _fail(f'copy failed: {exc}')

    for mode in MODES:
        if mode not in installed:
            print(f'Note: {mode} master bias not updated (no acceptable frame).')
    print(f"Master BIAS files updated successfully ({', '.join(installed)}).")
    sys.exit(0)


if __name__ == '__main__':
    main()
