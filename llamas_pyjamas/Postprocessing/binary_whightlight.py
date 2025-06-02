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
        "--plot",
        action="store_true",
        help="Whether to plot the results in ds9 (default: True)"
    )

    args = parser.parse_args()

    # Call the function QuickWhiteLightCube with the provided science file
    QuickWhiteLightCube(args.science_file, ds9plot=args.plot)

if __name__ == "__main__":
    main()