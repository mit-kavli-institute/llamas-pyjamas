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
    if not os.path.isdir(directory):
        print(f"Directory not found: {directory}", file=sys.stderr)
        sys.exit(1)
    
    
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "LIST_EXPTIME_READOUT.sh")

    try:
        subprocess.run(["bash", script], cwd=directory, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed to run observation log script: {e.stderr.decode().strip()}", file=sys.stderr)
        sys.exit(1)

    obs_log = os.path.join(directory, "obs.log")
    try:
        with open(obs_log) as f:
            lines = f.readlines()
    except OSError as e:
        print(f"Could not read observation log: {e}", file=sys.stderr)
        sys.exit(1)

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

    if not slow_bias and not fast_bias:
        print("No SLOW or FAST BIAS files found in the observation log.", file=sys.stderr)
        sys.exit(1)

    try:
        if slow_bias:
            shutil.copy2(os.path.join(directory, slow_bias), os.path.join(args.bias_dir, "slow_master_bias.fits"))
            shutil.copy2(os.path.join(directory, slow_bias), os.path.join(args.calib_dir, "slow_master_bias.fits"))

        if fast_bias:
            shutil.copy2(os.path.join(directory, fast_bias), os.path.join(args.bias_dir, "fast_master_bias.fits"))
            shutil.copy2(os.path.join(directory, fast_bias), os.path.join(args.calib_dir, "fast_master_bias.fits"))
    except OSError as e:
        print(f"Failed to copy BIAS file: {e}", file=sys.stderr)
        sys.exit(1)
        
    found = ", ".join(filter(None, [
        "SLOW" if slow_bias else None,
        "FAST" if fast_bias else None
    ]))
    print(f"Master BIAS files updated successfully ({found}).")
    sys.exit(0)

if __name__ == "__main__":
    main()