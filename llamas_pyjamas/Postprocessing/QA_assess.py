import argparse
import sys

from llamas_pyjamas.QA.llamasQATests import check_image

# Exit codes: 0 = pass, 1 = warn, 2 = fail, 3 = system error


def main():
    parser = argparse.ArgumentParser(description="Run QA checks on image(s).")
    parser.add_argument("input_path")
    parser.add_argument("--suite", default="basic_cal")
    parser.add_argument("--qa-yaml", default=None)
    parser.add_argument("--calib-root", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        results = check_image(
            input_path=args.input_path,
            suite=args.suite,
            qa_yaml=args.qa_yaml,
            calib_root=args.calib_root,
            report=args.report,
            verbose=args.verbose,
        )
    except Exception as exc:
        print(f"system error: {exc}", file=sys.stderr)
        return 3

    status = results.get("status", "pass")  # "pass" | "warn" | "fail"
    message = results.get("message", status)

    if status == "pass":
        print(message)
        return 0
    if status == "warn":
        print(message, file=sys.stderr)
        return 1
    print(message, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
