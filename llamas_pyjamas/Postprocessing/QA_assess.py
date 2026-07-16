import argparse
import contextlib
import io
import sys
import warnings
from pathlib import Path

# QA_assess ships inside the llamas_pyjamas source tree. Make that tree importable
# first, so we run the QA code beside this script regardless of where the editable
# install's .pth points (it currently targets the flux-cal sky-framework checkout).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Importing llamas_pyjamas pulls in the full reduction pipeline (Trace/Cube/Ray),
# which prints a banner to stdout and emits a pkg_resources deprecation warning at
# import time. This tool's interface is its exit code (+ optional --report JSON), so
# swallow that import-time chatter to keep stdout/stderr clean for callers.
with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()), \
        contextlib.redirect_stderr(io.StringIO()):
    warnings.simplefilter("ignore")
    from llamas_pyjamas.QA.llamasQATests import check_image

# Exit codes: 0 = pass, 1 = warn, 2 = fail, 3 = system error


def main():
    parser = argparse.ArgumentParser(description="Run QA checks on image(s).")
    parser.add_argument("input_path")
    parser.add_argument("--suite", default="basic_cal")
    parser.add_argument("--qa-yaml", default=None)
    parser.add_argument("--calib-root", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="print the result summary and failing-rule detail; by default the tool "
             "is quiet and signals only through its exit code "
             "(0=pass, 1=warn, 2=fail, 3=system error).",
    )
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

    # Quiet by default: the exit code IS the interface (0=pass, 1=warn, 2=fail).
    # --verbose opts into the human-readable summary (and the per-rule detail that
    # check_image prints). A system error still reported above (exit 3), regardless.
    if args.verbose:
        message = results.get("message", status)
        print(message, file=sys.stdout if status == "pass" else sys.stderr)

    return {"pass": 0, "warn": 1, "fail": 2}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
