# binary_check_image.py
import argparse
from llamas_pyjamas.QA.llamasQATests import check_image  # update path if needed


def main():
    parser = argparse.ArgumentParser(
        description="Run predefined QA checks on calibration/science image(s)."
    )

    parser.add_argument(
        "input_path",
        type=str,
        help="Input FITS file, directory, or glob pattern to check."
    )

    parser.add_argument(
        "--suite",
        type=str,
        default="basic_cal",
        help="QA suite name to run (default: basic_cal)."
    )

    parser.add_argument(
        "--qa-yaml",
        type=str,
        default=None,
        help="Optional path to a QA YAML file overriding suite defaults."
    )

    parser.add_argument(
        "--calib-root",
        type=str,
        default=None,
        help="Optional path to calibration/reference data root."
    )

    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Path to save QA report (e.g., .json/.yaml/.txt)."
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit non-zero if any QA test fails."
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose logging."
    )

    args = parser.parse_args()

    # check_image should determine file type internally
    # and execute predefined YAML-based QA tests.
    results = check_image(
        input_path=args.input_path,
        suite=args.suite,
        qa_yaml=args.qa_yaml,
        calib_root=args.calib_root,
        report=args.report,
        strict=args.strict,
        verbose=args.verbose,
    )

    # Optional: respect strict mode if check_image returns result object/dict
    # with a boolean pass/fail. Adjust key/attribute as needed.
    if args.strict:
        if isinstance(results, dict) and not results.get("passed", True):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
