import argparse
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

    args = parser.parse_args()

    # Call the function QuickWhiteLightCube with the provided science file
    QuickWhiteLightCube(args.science_file, bias=args.bias, ds9plot=args.plot, outfile=args.outfile)

if __name__ == "__main__":
    main()