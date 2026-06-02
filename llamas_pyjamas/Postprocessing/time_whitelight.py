"""
Timing benchmark for binary_whightlight.py.

Invokes binary_whightlight.py as a real subprocess (identical to running it
from the terminal) and measures wall time per run.

Usage:
    python time_whitelight.py <science_file> [--bias PATH] [--runs N] [--profile]
"""

import argparse
import os
import subprocess
import sys
import time

SCRIPT = os.path.join(os.path.dirname(__file__), "binary_whightlight.py")


def build_cmd(science_file, bias=None, profile=False):
    if profile:
        cmd = [sys.executable, "-m", "cProfile", "-s", "cumulative", SCRIPT, science_file]
    else:
        cmd = [sys.executable, SCRIPT, science_file]
    if bias:
        cmd += ["--bias", bias]
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description="Timing benchmark for binary_whightlight.py (subprocess)"
    )
    parser.add_argument("science_file", type=str, help="Path to the science FITS file")
    parser.add_argument("--bias", type=str, default=None, help="Optional bias file path")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs (default: 3)")
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Run cProfile on the first subprocess call and print results",
    )
    args = parser.parse_args()

    print("\n=== WhiteLight Timing Test ===")
    print(f"Mode:         subprocess (binary_whightlight.py)")
    print(f"Science file: {args.science_file}")
    print(f"Bias:         {'auto' if args.bias is None else args.bias}")
    print(f"Runs:         {args.runs}")
    if args.profile:
        print("Profiling:    enabled (first run)")
    print()

    times = []

    for i in range(args.runs):
        run_label = f"Run {i + 1}/{args.runs}"
        use_profile = args.profile and i == 0
        cmd = build_cmd(args.science_file, bias=args.bias, profile=use_profile)

        print(f"--- {run_label} ---")
        t0 = time.perf_counter()
        subprocess.run(cmd)
        elapsed = time.perf_counter() - t0

        times.append(elapsed)
        print(f"  -> {elapsed:.2f} s\n")

    print("Summary:")
    print(f"  Mean:  {sum(times) / len(times):.2f} s")
    print(f"  Min:   {min(times):.2f} s")
    print(f"  Max:   {max(times):.2f} s")


if __name__ == "__main__":
    main()
