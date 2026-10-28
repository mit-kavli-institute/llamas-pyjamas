"""Apply a sensitivity function to science RSS files, writing FLAM/FLAM_ERR in place.

Takes a sensitivity function (built from a standard in CubeViewer) and one or
more science RSS files, and adds flux-calibrated ``FLAM`` / ``FLAM_ERR`` extensions to each —
in place by default, so no new files are created. The differential atmospheric-extinction
correction between the science and standard airmasses is applied unless disabled.

Usage::

    python -m llamas_pyjamas.Postprocessing.apply_fluxcal SENSFUNC.fits RSS1.fits [RSS2.fits ...]
    python -m llamas_pyjamas.Postprocessing.apply_fluxcal SENSFUNC.fits RSS.fits --no-extinction
    python -m llamas_pyjamas.Postprocessing.apply_fluxcal SENSFUNC.fits RSS.fits --outdir calibrated/

Each science RSS must be a single channel present in the sensitivity function; the channel is
read from the file's ``CHANNEL`` header, so blue/green/red files are each calibrated with the
matching channel of the sensfunc.

Any stage of an exposure may be named: FLAM is written on the exposure's FINAL stage
(``_FF_SKYSUB`` > ``_FF`` > base, or a consolidated file), the one the combiner uses -- or on
``--stage`` explicitly. The sensfunc must be at the same fibre-flat state as that file
(``--sensfunc-stage`` declares it for a legacy sensfunc that does not record it).
"""

import argparse
import os
import sys

from llamas_pyjamas.File.llamasRSS import RSSStageError
from llamas_pyjamas.Flux.fluxCalibrate import (calibration_targets, check_calibration_target,
                                               flux_calibrate_file, load_lco_extinction)
from llamas_pyjamas.Flux.sensFunc import SensFunc


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sensfunc', help='Sensitivity-function FITS (from CubeViewer)')
    parser.add_argument('rss', nargs='+', help='Science RSS file(s) to flux-calibrate')
    parser.add_argument('--no-extinction', action='store_true',
                        help='Skip the differential atmospheric-extinction correction')
    parser.add_argument('--outdir', default=None,
                        help='Write calibrated copies here instead of editing in place')
    parser.add_argument('--stage', choices=('auto', 'base', 'ff', 'ff_skysub'), default='auto',
                        help='which stage of each exposure to calibrate (default: the final one)')
    parser.add_argument('--sensfunc-stage', choices=('base', 'ff', 'ff_skysub'),
                        help='declare the stage of a legacy sensfunc that records none')
    parser.add_argument('--allow-stage-mismatch', action='store_true',
                        help='calibrate even if the sensfunc and science stages differ '
                             '(recorded as OVR_STG in the FLAM header)')
    parser.add_argument('--allow-header-mismatch', action='store_true',
                        help="calibrate a file whose primary header differs from its exposure's "
                             'other stages')
    args = parser.parse_args(argv)

    if not os.path.exists(args.sensfunc):
        print(f'Sensitivity function not found: {args.sensfunc}', file=sys.stderr)
        return 1
    sensfunc = SensFunc.load(args.sensfunc)
    print(f'Loaded sensfunc: standard={sensfunc.meta.get("standard", "?")}, '
          f'channels={",".join(sensfunc.channels)}, '
          f'airmass={sensfunc.meta.get("airmass", "unknown")}')

    # Load the extinction table once and reuse across files.
    extinct = None if args.no_extinction else load_lco_extinction()

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    present = [r for r in args.rss if os.path.exists(r)]
    for r in sorted(set(args.rss) - set(present)):
        print(f'  SKIP (not found): {r}', file=sys.stderr)
    try:
        targets = calibration_targets(present, stage=args.stage)
    except (RSSStageError, FileNotFoundError) as exc:
        print(f'Cannot resolve the files to calibrate: {exc}', file=sys.stderr)
        return 1
    for rss in present:
        resolved = calibration_targets([rss], stage=args.stage)[0]
        if os.path.abspath(rss) != resolved:
            print(f'  {os.path.basename(rss)} -> {os.path.basename(resolved)} '
                  f'({args.stage} stage of that exposure)')

    n_ok = 0
    for rss in targets:
        out_path = (os.path.join(args.outdir, os.path.basename(rss)) if args.outdir else None)
        try:
            check_calibration_target(rss, allow_header_mismatch=args.allow_header_mismatch)
            written = flux_calibrate_file(rss, sensfunc,
                                          apply_extinction=not args.no_extinction,
                                          out_path=out_path, extinct=extinct,
                                          sensfile=args.sensfunc,
                                          sensfunc_stage=args.sensfunc_stage,
                                          allow_stage_mismatch=args.allow_stage_mismatch)
        except Exception as exc:                       # noqa: BLE001
            print(f'  FAILED {os.path.basename(rss)}: {exc}', file=sys.stderr)
            continue
        print(f'  calibrated {os.path.basename(written)}')
        n_ok += 1

    print(f'Flux-calibrated {n_ok}/{len(targets)} file(s).')
    return 0 if n_ok == len(targets) and targets else 1


if __name__ == '__main__':
    sys.exit(main())
