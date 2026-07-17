"""Apply a sensitivity function to science RSS files, writing FLAM/FLAM_ERR in place.

Phase III CLI. Takes a sensitivity function (built from a standard in CubeViewer) and one or
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
"""

import argparse
import os
import sys

from llamas_pyjamas.Flux.fluxCalibrate import flux_calibrate_file, load_lco_extinction
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

    n_ok = 0
    for rss in args.rss:
        if not os.path.exists(rss):
            print(f'  SKIP (not found): {rss}', file=sys.stderr)
            continue
        out_path = (os.path.join(args.outdir, os.path.basename(rss)) if args.outdir else None)
        try:
            written = flux_calibrate_file(rss, sensfunc,
                                          apply_extinction=not args.no_extinction,
                                          out_path=out_path, extinct=extinct)
        except Exception as exc:                       # noqa: BLE001
            print(f'  FAILED {os.path.basename(rss)}: {exc}', file=sys.stderr)
            continue
        print(f'  calibrated {os.path.basename(written)}')
        n_ok += 1

    print(f'Flux-calibrated {n_ok}/{len(args.rss)} file(s).')
    return 0 if n_ok else 1


if __name__ == '__main__':
    sys.exit(main())
