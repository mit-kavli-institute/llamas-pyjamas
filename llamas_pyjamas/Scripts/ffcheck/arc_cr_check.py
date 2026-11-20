"""Were the arcs cosmic-ray cleaned? (read-only; plan step 3 of 2026-11-14)

Usage:
    python arc_cr_check.py <output_dir> [--load] [--reference]

Reads the recipe sidecar (<pickle>.recipe.json) of each night-arc extraction in
<output_dir>/arcs. With --load, also opens each pickle and reports CRCLEAN/CRWHY per camera
(each is about 3 GB). With --reference, loads the packaged reference arc
(LUT/LLAMAS_reference_arc.pkl, about 4.8 GB) and reports any cosmic-ray stamps it carries.
An arc should never be cleaned: arc lines are bright and narrow, exactly what L.A.Cosmic
removes.
"""

import argparse
import collections
import glob
import os
import pickle

from llamas_pyjamas.config import LUT_DIR
from llamas_pyjamas.Utils.recipes import read_recipe_sidecar, recipe_problem


def cr_stamps(pkl):
    """Counter of (CRCLEAN, CRWHY) over the cameras of an extraction pickle."""
    with open(pkl, 'rb') as fh:
        data = pickle.load(fh)
    stamps = collections.Counter()
    for ext in data.get('extractions', []):
        hdr = getattr(ext, 'hdr', None) or {}
        stamps[(hdr.get('CRCLEAN'), hdr.get('CRWHY'))] += 1
    primary = data.get('primary_header') or {}
    return stamps, {k: primary.get(k) for k in ('CRID', 'CRCHAN', 'RECIPE')}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output_dir')
    parser.add_argument('--load', action='store_true', help='open each arc pickle')
    parser.add_argument('--reference', action='store_true', help='check the reference arc')
    args = parser.parse_args()
    for pkl in sorted(glob.glob(os.path.join(args.output_dir, 'arcs', '*_extract.pkl'))):
        side = read_recipe_sidecar(pkl)
        print(f"{os.path.basename(pkl)}: sidecar {side or 'missing'}; "
              f"{recipe_problem(side, 'arc') or 'arc recipe OK'}")
        if args.load:
            print(f"    cameras: {dict(cr_stamps(pkl)[0])}")
    if args.reference:
        ref = os.path.join(LUT_DIR, 'LLAMAS_reference_arc.pkl')
        stamps, primary = cr_stamps(ref)
        print(f"reference arc: primary {primary}; cameras {dict(stamps)} "
              f"(None = made before cosmic-ray stamps existed)")


if __name__ == '__main__':
    main()
