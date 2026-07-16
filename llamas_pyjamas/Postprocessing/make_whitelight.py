"""CLI: build a white-light image from a pipeline product.

Handles both pipeline inputs and dispatches on the file type:

  * an RSS FITS (``..._RSS_{color}[_FF][_SKYSUB].fits``) -> :func:`WhiteLightFromRSS`
  * an extraction batch pickle (``..._extractions.pkl``) -> :func:`WhiteLightFits`
    (all three colours in one file)

Pass ``--hex`` to render each fibre as a flat hexagonal tile of its raw,
un-resampled flux instead of interpolating onto the rectangular grid. Dead
fibres stay visible as empty hexagons rather than being filled in from their
neighbours.

The telescope-style quick-look (from a raw MEF) has its own CLI:
``python -m llamas_pyjamas.Postprocessing.binary_whightlight <mef> [--hex]``.

Examples
--------
    python -m llamas_pyjamas.Postprocessing.make_whitelight \\
        /path/LLAMAS_..._RSS_green_FF.fits --hex

    python -m llamas_pyjamas.Postprocessing.make_whitelight \\
        /path/LLAMAS_..._extractions.pkl --hex --outfile /path/wl_hex.fits
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Build a white-light image from an RSS file or an extraction "
                    "batch pickle (use --hex for un-interpolated hexagonal fibre tiles)."
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to an RSS FITS (..._RSS_{color}[_FF][_SKYSUB].fits) or an "
             "extraction batch pickle (..._extractions.pkl)"
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Output path. Default: input name with '_whitelight.fits'. A bare "
             "filename (no directory) is written under OUTPUT_DIR for the "
             "extraction-pickle path."
    )
    parser.add_argument(
        "--hex",
        dest="hex_tiles",
        action="store_true",
        default=False,
        help="Render each fibre as a flat hexagonal tile of its raw flux (no "
             "interpolation; dead fibres left as holes) instead of resampling "
             "onto the rectangular grid (default: False)"
    )
    parser.add_argument(
        "--pix-per-unit",
        type=int,
        default=10,
        help="Output pixels per fibre-map unit for --hex; 10 gives hexagons "
             "10 px wide (~460x442). Ignored without --hex (default: 10)"
    )
    parser.add_argument(
        "--wave-min",
        type=float,
        default=None,
        help="Minimum wavelength in Angstroms (RSS input only)"
    )
    parser.add_argument(
        "--wave-max",
        type=float,
        default=None,
        help="Maximum wavelength in Angstroms (RSS input only)"
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        parser.error(f"input not found: {args.input}")

    # Imported here so --help stays fast (the module pulls in the fibre map, ray, ...)
    from llamas_pyjamas.Image.WhiteLightModule import WhiteLightFromRSS, WhiteLightFits

    if args.input.endswith('.pkl'):
        from llamas_pyjamas.Extract.extractLlamas import load_extractions
        obj, metadata = load_extractions(args.input)
        outfile = args.outfile or args.input.replace('.pkl', '_whitelight.fits')
        result = WhiteLightFits(obj, metadata, outfile=outfile,
                                hex_tiles=args.hex_tiles,
                                pix_per_unit=args.pix_per_unit)
    elif args.input.endswith('.fits'):
        result = WhiteLightFromRSS(args.input, outfile=args.outfile,
                                   wave_min=args.wave_min, wave_max=args.wave_max,
                                   hex_tiles=args.hex_tiles,
                                   pix_per_unit=args.pix_per_unit)
    else:
        parser.error("input must be a .fits RSS file or a .pkl extraction batch")
        return 2

    if not result:
        print("white light generation failed (see log)", file=sys.stderr)
        return 1

    print(f"white light written: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
