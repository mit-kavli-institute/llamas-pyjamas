#!/usr/bin/env python3
"""
llamas_pyjamas.Scripts.update_quicklook_calibs
==============================================
Install the nightly calibrations the quick-look white light relies on.

Run once in the afternoon (or on the previous night's directory) and the
observer quick look (``Postprocessing/binary_whightlight.py``) picks the
results up automatically. Three stages, each from the QA-vetted frames of one
raw directory and each optional:

1. **Master biases** -- ``update_bias_master``: the median of the QA-passing
   biases per read mode, as ``{slow,fast}_master_bias.fits``.
2. **Master darks** -- the median of the QA-passing darks per read mode, minus
   the master bias of that mode, as ``{slow,fast}_master_dark.fits`` with the
   dark exposure time in ``DARKEXPT``. The quick look scales it to each frame's
   exposure time. Where several exposure times were taken, the longest is used.
3. **Per-fibre sky reference** -- ``build_quicklook_skyref`` on the QA-passing
   twilight sky flats (``CAL.R-SKY``), as ``mastercalib/LLAMAS_quicklook_skyref.npz``.

Each stage that finds nothing acceptable prints a note and the others still
run; the exit status is 1 (``Update Failed: ...`` on stderr) only when no
stage installed anything. ``llamas-checks`` is optional, exactly as for
``update_bias_master``: without it the script refuses unless ``--no-qa``.

Usage::

    python -m llamas_pyjamas.Scripts.update_quicklook_calibs RAW_DIR
    python -m llamas_pyjamas.Scripts.update_quicklook_calibs RAW_DIR --no-warn-fallback
    python -m llamas_pyjamas.Scripts.update_quicklook_calibs RAW_DIR --no-qa
"""

import argparse
import logging
import os
import sys

from llamas_pyjamas.config import BIAS_DIR, CALIB_DIR
from llamas_pyjamas.Postprocessing.build_quicklook_fiberimg import DEFAULT_OUT as QUICKLOOK_CACHE
from llamas_pyjamas.Postprocessing.build_quicklook_skyref import (
    DEFAULT_SKYREF, build_quicklook_skyref)
from llamas_pyjamas.Scripts.update_bias_master import (
    FAIL_MSG, HAVE_LLAMAS_CHECKS, MODES, RULE_SET_BY_PRODCATG, find_frames, install_bias,
    install_master, print_verdict_table, qa_engine, qa_verdicts, select_bias_frames)

logger = logging.getLogger(__name__)


class _NoQAError(Exception):
    """Never raised; stands in for QAEngineError when llamas-checks is absent."""


QA_ERROR = qa_engine.QAEngineError if qa_engine is not None else _NoQAError

# Fibre-label cache the sky reference is keyed to (tests point this elsewhere).
FIBRE_CACHE = QUICKLOOK_CACHE
DARK_PRODCATG = 'CAL.R-DRK'
SKY_PRODCATG = 'CAL.R-SKY'
BIAS_PRODCATG = 'CAL.R-BIA'


def _fail(reason):
    sys.stdout.flush()
    print(f'{FAIL_MSG}: {reason}', file=sys.stderr)
    sys.exit(1)


def _verdicts(directory, prodcatg, no_qa):
    """Candidates of one PRODCATG with their QA verdicts (or 'unchecked' with --no-qa)."""
    candidates = find_frames(directory, prodcatg)
    if not candidates or no_qa:
        return [{**c, 'verdict': 'unchecked'} for c in candidates]
    return qa_verdicts(candidates, RULE_SET_BY_PRODCATG[prodcatg])


def select_dark_frames(verdicts, allow_warn=True, accept_unchecked=False):
    """Acceptable darks per read mode, restricted to one exposure time.

    Applies ``select_bias_frames`` then, within each mode, keeps only the frames
    of the longest exposure time present (a master dark is scaled to the
    science exposure, and the longest dark has the best signal to noise).
    Returns ``{'SLOW': [paths], 'FAST': [paths]}`` and ``{'SLOW': exptime, ...}``.
    """
    verdicts = [v for v in verdicts if v.get('exptime')]   # a dark without exposure time is useless
    selection = select_bias_frames(verdicts, allow_warn=allow_warn, accept_unchecked=accept_unchecked)
    exptime_of = {v['path']: v['exptime'] for v in verdicts}
    chosen, exptimes = {}, {}
    for mode in MODES:
        paths = list(selection[mode])
        if not paths:
            chosen[mode], exptimes[mode] = [], None
            continue
        longest = max(exptime_of[p] for p in paths)
        chosen[mode] = [p for p in paths if exptime_of[p] == longest]
        exptimes[mode] = longest
    return chosen, exptimes


def master_bias_path(mode, bias_dir):
    """``{mode}_master_bias.fits`` in *bias_dir* if present, else the packaged one
    in ``BIAS_DIR`` (so darks and the sky reference can still be built on a night
    whose biases were all rejected), else None."""
    for directory in (bias_dir, BIAS_DIR):
        candidate = os.path.join(directory, f'{mode.lower()}_master_bias.fits')
        if os.path.isfile(candidate):
            return candidate
    return None


def install_darks(selection, exptimes, bias_dir, calib_dir):
    """Median-combine the selected darks per mode, subtract that mode's master
    bias (``master_bias_path``) and install the result as
    ``{mode}_master_dark.fits`` in both directories. Returns the modes installed."""
    installed = []
    for mode in MODES:
        paths = selection.get(mode) or []
        if not paths:
            continue
        bias_file = master_bias_path(mode, bias_dir)
        if bias_file is None:
            logger.warning('No %s master bias in %s or %s; %s dark not installed',
                           mode, bias_dir, BIAS_DIR, mode)
            continue
        if os.path.dirname(os.path.abspath(bias_file)) != os.path.abspath(bias_dir):
            logger.warning('%s dark uses the packaged master bias %s', mode, bias_file)
        extra = {'DARKEXPT': (float(exptimes[mode]), 'Dark exposure time (s)'),
                 'PRODCATG': ('CAL.R-DRK', 'Type of data product'),
                 'READ-MDE': (mode, 'Readout mode')}
        install_master(paths, f'{mode.lower()}_master_dark.fits', bias_dir, calib_dir,
                       subtract=bias_file, extra_primary=extra)
        installed.append(mode)
    return installed


def main(argv=None):
    parser = argparse.ArgumentParser(
        description='Install the nightly quick-look calibrations from a raw directory: '
                    'median master biases and darks, and the twilight per-fibre sky reference.')
    parser.add_argument('directory', help='Directory containing raw LLAMAS MEF files')
    parser.add_argument('--bias-dir', default=BIAS_DIR,
                        help='Directory for the package master bias/dark files')
    parser.add_argument('--calib-dir', default=CALIB_DIR,
                        help='Directory for the master calibration copies')
    parser.add_argument('--skyref-out', default=DEFAULT_SKYREF,
                        help='Output path of the per-fibre sky reference')
    parser.add_argument('--no-warn-fallback', action='store_true',
                        help="Only accept a QA 'pass'; do not fall back to 'warn' frames")
    parser.add_argument('--no-qa', action='store_true',
                        help='Skip llamas-checks and take every frame by header (unvetted)')
    parser.add_argument('--skip-bias', action='store_true', help='Leave the master biases as they are')
    parser.add_argument('--skip-dark', action='store_true', help='Do not install master darks')
    parser.add_argument('--skip-skyref', action='store_true', help='Do not build the sky reference')
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
    if args.no_qa:
        print('WARNING: --no-qa: installing calibrations WITHOUT QA vetting', file=sys.stderr)

    allow_warn = not args.no_warn_fallback
    installed = []

    try:
        # 1. Master biases
        if not args.skip_bias:
            verdicts = _verdicts(directory, BIAS_PRODCATG, args.no_qa)
            selection = select_bias_frames(verdicts, allow_warn=allow_warn, accept_unchecked=args.no_qa)
            print('Biases:')
            print_verdict_table(verdicts, selection)
            modes = install_bias(selection, args.bias_dir, args.calib_dir)
            if modes:
                print(f"Master BIAS files updated successfully ({', '.join(modes)}).")
                installed.append('bias')
            else:
                print('Note: no acceptable bias frame; master biases unchanged.')

        # 2. Master darks
        if not args.skip_dark:
            verdicts = _verdicts(directory, DARK_PRODCATG, args.no_qa)
            selection, exptimes = select_dark_frames(verdicts, allow_warn=allow_warn,
                                                     accept_unchecked=args.no_qa)
            print('Darks:')
            print_verdict_table(verdicts, selection)
            modes = install_darks(selection, exptimes, args.bias_dir, args.calib_dir)
            if modes:
                print('Master DARK files updated successfully ('
                      + ', '.join(f'{m}: {exptimes[m]:g} s x {len(selection[m])}' for m in modes) + ').')
                installed.append('dark')
            else:
                print('Note: no acceptable dark frame; master darks unchanged.')

        # 3. Twilight per-fibre sky reference
        if not args.skip_skyref:
            verdicts = _verdicts(directory, SKY_PRODCATG, args.no_qa)
            selection = select_bias_frames(verdicts, allow_warn=allow_warn, accept_unchecked=args.no_qa)
            frames = selection['SLOW'] + selection['FAST']
            print('Twilight sky flats:')
            print_verdict_table(verdicts, selection)
            if frames:
                try:
                    out = build_quicklook_skyref(
                        frames, out=args.skyref_out, force=True, fibre_cache=FIBRE_CACHE,
                        slow_bias=master_bias_path('SLOW', args.bias_dir),
                        fast_bias=master_bias_path('FAST', args.bias_dir))
                    print(f'Quick-look sky reference updated successfully ({len(frames)} frames) -> {out}')
                    installed.append('skyref')
                except (ValueError, FileNotFoundError) as exc:
                    print(f'Note: sky reference not built: {exc}')
            else:
                print('Note: no acceptable twilight sky flat; sky reference unchanged.')
    except QA_ERROR as exc:
        _fail(f'llamas-checks could not run: {exc}')
    except OSError as exc:
        _fail(f'copy failed: {exc}')

    if not installed:
        _fail('nothing installed: no acceptable bias, dark or twilight frame found')
    print(f"Quick-look calibrations updated ({', '.join(installed)}).")
    sys.exit(0)


if __name__ == '__main__':
    main()
