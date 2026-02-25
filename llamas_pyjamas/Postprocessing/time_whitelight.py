"""
Timing benchmark for QuickWhiteLightCube.

Usage:
    python time_whitelight.py <science_file> [--bias PATH] [--runs N] [--profile]

Does NOT modify binary_whightlight.py or any functions it uses.
"""

import argparse
import cProfile
import io
import pstats
import time

from llamas_pyjamas.Image.WhiteLightModule import QuickWhiteLightCube


def run_once(science_file, bias=None):
    t0 = time.perf_counter()
    QuickWhiteLightCube(science_file, bias=bias, ds9plot=False, outfile=None)
    return time.perf_counter() - t0


def main():
    parser = argparse.ArgumentParser(
        description="Timing benchmark for QuickWhiteLightCube"
    )
    parser.add_argument("science_file", type=str, help="Path to the science FITS file")
    parser.add_argument("--bias", type=str, default=None, help="Optional bias file path")
    parser.add_argument("--runs", type=int, default=3, help="Number of timed runs (default: 3)")
    parser.add_argument(
        "--profile",
        action="store_true",
        default=False,
        help="Run cProfile on the first call and print top-20 functions by cumulative time",
    )
    args = parser.parse_args()

    print("\n=== WhiteLight Timing Test ===")
    print(f"Science file: {args.science_file}")
    print(f"Bias:         {'auto' if args.bias is None else args.bias}")
    print(f"Runs:         {args.runs}")
    if args.profile:
        print("Profiling:    enabled (first run)")
    print()

    times = []

    for i in range(args.runs):
        run_label = f"Run {i + 1}/{args.runs}"

        if i == 0 and args.profile:
            # Profile the first run
            pr = cProfile.Profile()
            pr.enable()
            t0 = time.perf_counter()
            QuickWhiteLightCube(args.science_file, bias=args.bias, ds9plot=False, outfile=None)
            elapsed = time.perf_counter() - t0
            pr.disable()
        else:
            t0 = time.perf_counter()
            QuickWhiteLightCube(args.science_file, bias=args.bias, ds9plot=False, outfile=None)
            elapsed = time.perf_counter() - t0

        times.append(elapsed)
        print(f"--- {run_label} ---  {elapsed:.2f} s")

    print()
    print("Summary:")
    print(f"  Mean:  {sum(times) / len(times):.2f} s")
    print(f"  Min:   {min(times):.2f} s")
    print(f"  Max:   {max(times):.2f} s")

    if args.profile:
        print()
        print("--- cProfile top-20 (cumulative time) ---")
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats(pstats.SortKey.CUMULATIVE)
        ps.print_stats(20)
        print(s.getvalue())


if __name__ == "__main__":
    main()
