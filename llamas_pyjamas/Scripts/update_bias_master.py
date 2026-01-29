#!/usr/bin/env python3
import argparse
import os
import subprocess
import shutil
import sys
from llamas_pyjamas.config import BIAS_DIR, CALIB_DIR

def main():
    parser = argparse.ArgumentParser(description="Find first SLOW BIAS file in a directory")
    parser.add_argument("directory", help="Directory containing *mef.fits files")
    parser.add_argument("--bias-dir", default=BIAS_DIR, help="Directory to store bias master files")
    parser.add_argument("--calib-dir", default=CALIB_DIR, help="Directory to store master calibration files")
    args = parser.parse_args()

    directory = os.path.abspath(args.directory)
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LIST_EXPTIME_READOUT.sh")

    subprocess.run(["bash", script], cwd=directory, check=True)

    obs_log = os.path.join(directory, "obs.log")
    with open(obs_log) as f:
        lines = f.readlines()

    slow_bias = None
    fast_bias = None

    # Skip header and separator lines
    for line in lines[2:]:
        cols = line.strip().split("\t")
        if len(cols) < 6:
            continue
        filename, obj, readmode = cols[0], cols[3], cols[5]
        if "BIAS" in obj:
            if "SLOW" in readmode and slow_bias is None:
                slow_bias = filename
            elif "FAST" in readmode and fast_bias is None:
                fast_bias = filename
        if slow_bias and fast_bias:
            break

    if slow_bias:
        print(slow_bias)
    else:
        print("No SLOW BIAS file found.", file=sys.stderr)

    if fast_bias:
        print(fast_bias)
    else:
        print("No FAST BIAS file found.", file=sys.stderr)

    if slow_bias and fast_bias:
        print("Found both SLOW and FAST BIAS files.")
        shutil.copy2(os.path.join(directory, slow_bias), os.path.join(args.bias_dir, "slow_master_bias.fits"))
        shutil.copy2(os.path.join(directory, fast_bias), os.path.join(args.bias_dir, "fast_master_bias.fits"))

        shutil.copy2(os.path.join(directory, slow_bias), os.path.join(args.calib_dir, "slow_master_bias.fits"))
        shutil.copy2(os.path.join(directory, fast_bias), os.path.join(args.calib_dir, "fast_master_bias.fits"))

        print("Both Master BIAS files updated.")

    elif slow_bias and not fast_bias:
        print("Found SLOW BIAS file only.")
        shutil.copy2(os.path.join(directory, slow_bias), os.path.join(args.bias_dir, "slow_master_bias.fits"))
        shutil.copy2(os.path.join(directory, slow_bias), os.path.join(args.calib_dir, "slow_master_bias.fits"))
        print("SLOW Master BIAS file updated.")
    elif fast_bias and not slow_bias:
        print("Found FAST BIAS file only.")
        shutil.copy2(os.path.join(directory, fast_bias), os.path.join(args.bias_dir, "fast_master_bias.fits"))
        shutil.copy2(os.path.join(directory, fast_bias), os.path.join(args.calib_dir, "fast_master_bias.fits"))
        print("FAST Master BIAS file updated.")
    else:
        print("No files copied as no BIAS files were found.")

if __name__ == "__main__":
    main()
