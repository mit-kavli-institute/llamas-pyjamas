"""Bring an existing reduction forward onto its final RSS stage.

Reductions made with ``keep_intermediate_rss = true`` hold three files per exposure and channel
(base, ``_FF``, ``_FF_SKYSUB``). Registration and flux calibration used to land in whichever file
was opened -- usually the base one -- so the final stage, the one that should be combined, has
rough header positions and no FLAM. The combiner now refuses that (RegistrationMismatchError,
MissingFluxCalibrationError). This command, for every exposure/channel in an extraction directory:

1. reports duplicate same-stage files (an old consolidated file next to new long names), and with
   ``--keep-duplicates`` moves the unwanted ones into ``superseded/``;
2. compares the primary headers of the stages, and with ``--sync-header-from STAGE`` copies the
   differing keys from that stage (keys recording data changes -- pedestal, heliocentric scaling --
   are refused: those need a re-reduction);
3. copies a registered solution (FIBERWCS + FIBERMAP RA/DEC) from another stage onto the final
   file when that stage is better registered and the fibre rows match;
4. with ``--sensfunc``, writes FLAM/FLAM_ERR on a final file that lacks them, under the
   sensfunc-stage rules (a legacy sensfunc that records no stage must be declared with
   ``--sensfunc-stage``).

Every changed file is rebuilt in memory, written to a temporary file beside it, re-opened and
checked, then renamed into place: an interrupted run never leaves a half-written RSS.

Usage::

    python -m llamas_pyjamas.Postprocessing.promote_final_rss EXTRACTIONS_DIR --dry-run
    python -m llamas_pyjamas.Postprocessing.promote_final_rss EXTRACTIONS_DIR -y \\
        --sensfunc EG21_sensfunc.fits

Exit codes: 0 every exposure ready; 1 something still needs attention (FLAM missing, re-reduction
required, unresolved duplicates, a stage-mismatched sensfunc); 3 error.
"""

import argparse
import os
import shutil
import sys

import numpy as np
from astropy.io import fits

from llamas_pyjamas.File.llamasRSS import (
    RSS_STAGE_RE, RSSStageError, STAGE_HEADER_DERIVED, _STAGE_RANK, _stage_candidates,
    check_stage_consistency, exposure_key, primary_header_diff, rss_stage, stage_files)

EXIT_OK, EXIT_PARTIAL, EXIT_ERROR = 0, 1, 3


def _groups(extraction_dir):
    """One representative path per (exposure_key, channel) in the directory (top level)."""
    seen = {}
    for name in sorted(os.listdir(extraction_dir)):
        m = RSS_STAGE_RE.match(name)
        path = os.path.join(extraction_dir, name)
        if m and os.path.isfile(path):
            seen.setdefault((exposure_key(name), m.group(2)), path)
    return seen


def _is_consolidated(path):
    m = RSS_STAGE_RE.match(os.path.basename(path))
    return bool(m) and not m.group(3) and exposure_key(path) == m.group(1)


def resolve_duplicates(path, keep, dry_run=False):
    """Move same-stage duplicates of ``path``'s exposure/channel into ``superseded/``.

    ``keep='long'`` keeps the pipeline's long-named file, ``'consolidated'`` the short
    ``{exposure}_RSS_{channel}.fits``. Returns ``[(moved_from, moved_to)]``."""
    by_stage = {}
    for full, stage in _stage_candidates(path):
        by_stage.setdefault(stage or rss_stage(full), []).append(full)
    moves = []
    for stage, files in by_stage.items():
        if len(files) < 2:
            continue
        want_cons = keep == 'consolidated'
        keepers = [f for f in files if _is_consolidated(f) == want_cons]
        if len(keepers) != 1:
            raise RSSStageError(f'{exposure_key(path)} {stage}: cannot pick one {keep} file among '
                                f'{[os.path.basename(f) for f in files]}')
        dest_dir = os.path.join(os.path.dirname(path), 'superseded')
        for f in files:
            if f == keepers[0]:
                continue
            dest = os.path.join(dest_dir, os.path.basename(f))
            if not dry_run:
                os.makedirs(dest_dir, exist_ok=True)
                if os.path.exists(dest):
                    raise RSSStageError(f'{dest} already exists; not overwriting it')
                shutil.move(f, dest)
            moves.append((f, dest))
    return moves


def _rows_match(a, b):
    """FIBER_ID and BENCHSIDE equal row for row (not just the same shape)."""
    if len(a) != len(b):
        return False
    bs = lambda t: np.char.strip(np.asarray(t['BENCHSIDE']).astype(str))  # noqa: E731
    return (np.array_equal(np.asarray(a['FIBER_ID']), np.asarray(b['FIBER_ID']))
            and np.array_equal(bs(a), bs(b)))


def promote_one(final, report, *, sensfunc=None, sensfile='', sensfunc_stage=None,
                allow_stage_mismatch=False, sync_from=None, extinct=None):
    """Build the promoted version of ``final`` in memory. Returns ``(hdul_or_None, actions,
    problems)``: ``hdul`` is None when nothing changes; ``problems`` need user attention."""
    actions, problems = [], []
    with fits.open(final) as src:
        hdul = fits.HDUList([h.copy() for h in src])
    changed = False

    # header: report or sync
    for other_stage, diff in report.header_diffs.items():
        derived = sorted(k for k in diff if k in STAGE_HEADER_DERIVED)
        plain = sorted(k for k in diff if k not in STAGE_HEADER_DERIVED)
        if derived:
            problems.append(f'header differs from {other_stage} in {derived}: those record data '
                            f'changes -- re-reduce this exposure')
        if plain and sync_from == other_stage:
            with fits.open(report.files[other_stage]) as o:
                ohdr = o[0].header
                for k in plain:
                    if k in ohdr:
                        hdul[0].header[k] = (ohdr[k], ohdr.comments[k])
                    elif k in hdul[0].header:
                        del hdul[0].header[k]
            hdul[0].header['HISTORY'] = (f'promote_final_rss: primary keys {plain} synced from '
                                         f'{os.path.basename(report.files[other_stage])}')
            actions.append(f'synced header keys {plain} from {other_stage}')
            changed = True
        elif plain:
            problems.append(f'header differs from {other_stage} in {plain} (use '
                            f'--sync-header-from {other_stage} if that stage is the corrected one)')

    # registration
    if report.better_wcs_stage is not None:
        source = report.files[report.better_wcs_stage]
        with fits.open(source) as s:
            ok = ('FIBERWCS' in s and _rows_match(s['FIBERMAP'].data, hdul['FIBERMAP'].data)
                  and _rows_match(s['FIBERWCS'].data, hdul['FIBERMAP'].data))
            if not ok:
                problems.append(f'{report.better_wcs_stage} is registered but its fibre rows do not '
                                f'match {os.path.basename(final)}; not copied')
            else:
                names = [h.name for h in hdul]
                if 'FIBERWCS' in names:
                    del hdul[names.index('FIBERWCS')]
                hdul.append(s['FIBERWCS'].copy())
                hdul['FIBERMAP'].data['RA'][:] = s['FIBERMAP'].data['RA']
                hdul['FIBERMAP'].data['DEC'][:] = s['FIBERMAP'].data['DEC']
                if 'FIBAREA' in s[0].header:
                    hdul[0].header['FIBAREA'] = s[0].header['FIBAREA']
                meth = s['FIBERWCS'].header.get('WCSMETH', '?')
                hdul[0].header['HISTORY'] = (f'promote_final_rss: FIBERWCS + FIBERMAP RA/DEC from '
                                             f'{os.path.basename(source)} ({meth})')
                actions.append(f'registration ({meth}) copied from {report.better_wcs_stage}')
                changed = True

    # flux calibration
    if 'FLAM' not in hdul:
        if sensfunc is None:
            problems.append('no FLAM (give --sensfunc to calibrate it)')
        else:
            from llamas_pyjamas.Flux.fluxCalibrate import apply_sensfunc
            try:
                apply_sensfunc(hdul, sensfunc, extinct=extinct, sensfile=sensfile,
                               sensfunc_stage=sensfunc_stage,
                               allow_stage_mismatch=allow_stage_mismatch)
                actions.append(f'FLAM written ({os.path.basename(sensfile)})')
                changed = True
            except (RSSStageError, ValueError) as exc:
                problems.append(f'FLAM not written: {exc}')

    if changed and 'RSSSTAGE' not in hdul[0].header:
        hdul[0].header['RSSSTAGE'] = (rss_stage(hdul), 'RSS processing stage (see rss_stage)')
    return (hdul if changed else None), actions, problems


def write_atomically(hdul, final, expect_flam=False, expect_wcs=False):
    """Write ``hdul`` over ``final`` via a temporary file that is re-opened and checked first."""
    tmp = final + '.promote-tmp'
    try:
        hdul.writeto(tmp, overwrite=True)
        with fits.open(tmp) as chk:
            names = {h.name for h in chk}
            if expect_flam and 'FLAM' not in names:
                raise RSSStageError(f'{tmp}: FLAM missing after write')
            if expect_wcs and 'FIBERWCS' not in names:
                raise RSSStageError(f'{tmp}: FIBERWCS missing after write')
            _ = chk[1].data.shape                      # the data actually reads back
        os.replace(tmp, final)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def promote_directory(extraction_dir, *, sensfunc_path=None, sensfunc_stage=None,
                      allow_stage_mismatch=False, keep_duplicates=None, sync_from=None,
                      dry_run=False, apply_extinction=True, log=print):
    """Promote every exposure/channel in ``extraction_dir``. Returns ``(results, n_problem)``
    where ``results`` is a list of ``(exposure, channel, final_basename, actions, problems)``."""
    sensfunc, extinct = None, None
    if sensfunc_path:
        from llamas_pyjamas.Flux.sensFunc import SensFunc
        sensfunc = SensFunc.load(sensfunc_path)
        if apply_extinction:
            from llamas_pyjamas.Flux.fluxCalibrate import load_lco_extinction
            extinct = load_lco_extinction()
    results, n_problem = [], 0
    for (key, chan), path in sorted(_groups(extraction_dir).items()):
        actions, problems = [], []
        try:
            files = stage_files(path)
        except RSSStageError as exc:
            if keep_duplicates is None:
                results.append((key, chan, '', [], [f'{exc}']))
                n_problem += 1
                continue
            moves = resolve_duplicates(path, keep_duplicates, dry_run=dry_run)
            actions += [f'moved {os.path.basename(a)} -> superseded/' for a, _ in moves]
            if dry_run:
                results.append((key, chan, '', actions, ['(dry run: duplicates still on disk)']))
                continue
            files = stage_files([f for f, _ in _stage_candidates(path)][0])
        final = files[max(files, key=_STAGE_RANK.get)]
        report = check_stage_consistency(final, raise_errors=False)
        hdul, act, prob = promote_one(final, report, sensfunc=sensfunc,
                                      sensfile=sensfunc_path or '', sensfunc_stage=sensfunc_stage,
                                      allow_stage_mismatch=allow_stage_mismatch,
                                      sync_from=sync_from, extinct=extinct)
        actions += act
        problems += prob
        if hdul is not None and not dry_run:
            write_atomically(hdul, final, expect_flam='FLAM' in hdul,
                             expect_wcs=report.better_wcs_stage is not None)
        if problems:
            n_problem += 1
        results.append((key, chan, os.path.basename(final), actions, problems))
    return results, n_problem


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('extraction_dir', help='reduction extractions/ directory')
    parser.add_argument('--sensfunc', help='sensitivity function to write FLAM on final files '
                                           'that lack it')
    parser.add_argument('--sensfunc-stage', choices=('base', 'ff', 'ff_skysub'),
                        help='declare the stage of a legacy sensfunc that records none')
    parser.add_argument('--allow-stage-mismatch', action='store_true',
                        help='calibrate even if the sensfunc stage differs from the science '
                             '(recorded in the FLAM header)')
    parser.add_argument('--keep-duplicates', choices=('long', 'consolidated'),
                        help='resolve same-stage duplicates by keeping this name form; the others '
                             'move to superseded/')
    parser.add_argument('--sync-header-from', choices=('base', 'ff', 'ff_skysub'),
                        help='copy differing primary-header keys from this stage onto the final '
                             'file (data-change keys are refused)')
    parser.add_argument('--no-extinction', action='store_true',
                        help='skip the differential extinction correction when writing FLAM')
    parser.add_argument('--dry-run', action='store_true', help='report only; write nothing')
    parser.add_argument('-y', '--yes', action='store_true', help='do not ask for confirmation')
    args = parser.parse_args(argv)

    if not os.path.isdir(args.extraction_dir):
        print(f'Not a directory: {args.extraction_dir}', file=sys.stderr)
        return EXIT_ERROR
    try:
        if not args.dry_run and not args.yes:
            preview, _ = promote_directory(
                args.extraction_dir, sensfunc_path=args.sensfunc,
                sensfunc_stage=args.sensfunc_stage, allow_stage_mismatch=args.allow_stage_mismatch,
                keep_duplicates=args.keep_duplicates, sync_from=args.sync_header_from,
                dry_run=True, apply_extinction=not args.no_extinction)
            n_change = sum(1 for r in preview if r[3])
            _print(preview, dry_run=True)
            try:
                answer = input(f'Apply changes to {n_change} file(s)? [y/N] ')
            except EOFError:
                answer = ''
            if answer.strip().lower() not in ('y', 'yes'):
                print('Nothing written.')
                return EXIT_OK
        results, n_problem = promote_directory(
            args.extraction_dir, sensfunc_path=args.sensfunc, sensfunc_stage=args.sensfunc_stage,
            allow_stage_mismatch=args.allow_stage_mismatch, keep_duplicates=args.keep_duplicates,
            sync_from=args.sync_header_from, dry_run=args.dry_run,
            apply_extinction=not args.no_extinction)
    except Exception as exc:                            # noqa: BLE001
        print(f'promote_final_rss: error: {exc}', file=sys.stderr)
        return EXIT_ERROR
    _print(results, dry_run=args.dry_run)
    return EXIT_PARTIAL if n_problem else EXIT_OK


def _print(results, dry_run=False):
    tag = 'would ' if dry_run else ''
    for key, chan, final, actions, problems in results:
        state = 'ATTENTION' if problems else ('updated' if actions else 'ok')
        print(f'{key} {chan:5s} {final or "-"}: {state}')
        for a in actions:
            print(f'    {tag}{a}')
        for p in problems:
            print(f'    ! {p}')
    n_upd = sum(1 for r in results if r[3])
    n_att = sum(1 for r in results if r[4])
    print(f'{len(results)} exposure/channel(s): {n_upd} {tag}updated, {n_att} need attention.')


if __name__ == '__main__':
    sys.exit(main())
