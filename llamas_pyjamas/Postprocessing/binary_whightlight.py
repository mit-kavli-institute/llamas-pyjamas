import argparse
from llamas_pyjamas.Image.WhiteLight import QuickWhiteLightCube


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
        "plot",
        type=bool,
        default=False,
        help="Whether to plot the results in ds9 (default: False)")
    
    parser.add_argument(
        "--bias",
        type=str,
        default=None,
        help="Path to an optional bias file to use during processing"
    )

    args = parser.parse_args()

    # Call the function QuickWhiteLightCube with the provided science file
    QuickWhiteLightCube(args.science_file, bias=args.bias, ds9plot=args.plot)

if __name__ == "__main__":
    main()