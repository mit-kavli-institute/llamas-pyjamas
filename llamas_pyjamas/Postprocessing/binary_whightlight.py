import argparse
import logging

from llamas_pyjamas.Image.WhiteLightModule import QuickWhiteLightCube


def main():
    parser = argparse.ArgumentParser(
        description="Run QuickWhiteLightCube on an input science file"
    )
    parser.add_argument(
        "science_file",
        type=str,
        help="Path to the science file to process using QuickWhiteLightCube"
    )

    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Whether to plot the results in ds9 (default: False)"
    )

    parser.add_argument(
        "--bias",
        type=str,
        default=None,
        help="Path to an optional bias file to use during processing"
    )

    parser.add_argument(
        "--outfile",
        type=str,
        default=None,
        help="Path to save the output white light cube (default: None, saves to input file name with '_wlc' suffix)"
    )

    parser.add_argument(
        "--hex",
        dest="hex_tiles",
        action="store_true",
        default=False,
        help="Render each fibre as a flat hexagonal tile of its raw flux "
             "(no interpolation; dead fibres left as holes) instead of "
             "resampling onto the rectangular grid (default: False)"
    )

    parser.add_argument(
        "--pix-per-unit",
        type=int,
        default=10,
        help="Output pixels per fibre-map unit for --hex; 10 gives hexagons "
             "10 px wide (~460x442). Ignored without --hex (default: 10)"
    )

    parser.add_argument(
        "--dark",
        type=str,
        default=None,
        help="Master dark to subtract (bias-subtracted, DARKEXPT in its header) instead of "
             "Bias/{slow,fast}_master_dark.fits; scaled by the frame's exposure time"
    )

    parser.add_argument(
        "--no-dark",
        action="store_true",
        default=False,
        help="Do not subtract a master dark even if one is installed (default: subtract)"
    )

    parser.add_argument(
        "--skyref",
        type=str,
        default=None,
        help="Per-fibre sky reference (.npz from build_quicklook_skyref) instead of "
             "mastercalib/LLAMAS_quicklook_skyref.npz"
    )

    parser.add_argument(
        "--no-skyref",
        action="store_true",
        default=False,
        help="Do not divide by the per-fibre sky reference (default: divide when one exists)"
    )

    parser.add_argument(
        "--weighted",
        choices=("auto", "on", "off"),
        default="auto",
        help="Profile-weighted fibre sums: 'auto' for exposures shorter than 60 s, when the "
             "profile cache exists (default); 'on' always; 'off' never"
    )

    parser.add_argument(
        "--crclip",
        action="store_true",
        default=False,
        help="Clip cosmic rays with a 1x5 median along the dispersion axis (+~3 s; default: off)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Also print per-detector details (bias file, residual bias levels, "
             "fibre-label source). By default only warnings and the output path are shown."
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Call the function QuickWhiteLightCube with the provided science file
    QuickWhiteLightCube(args.science_file, bias=args.bias, ds9plot=args.plot,
                        outfile=args.outfile, hex_tiles=args.hex_tiles,
                        pix_per_unit=args.pix_per_unit,
                        dark=args.dark, use_dark=not args.no_dark,
                        skyref=args.skyref, use_skyref=not args.no_skyref,
                        weighted=args.weighted, crclip=args.crclip)

if __name__ == "__main__":
    main()
