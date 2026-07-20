"""Compare two RSS FITS files plane-by-plane — a bit-identical / regression check.

Reduce the SAME exposure on two branches (e.g. the pre-change baseline vs the sky-refine branch),
then compare the delivered ``*_RSS_{color}.fits`` files. For a behaviour-preserving refactor every
shared image plane (SKYSUB / ERROR / FLAM / SKY / SKYRESID / COUNTS / ...) should match exactly;
extensions present in only one file (e.g. a newly-added SKYMASK) are reported, not counted as a diff.

NaN-aware: two pixels that are both NaN count as equal.

Usage
-----
    python -m llamas_pyjamas.Scripts.compare_rss A_RSS_green.fits B_RSS_green.fits
    python -m llamas_pyjamas.Scripts.compare_rss A_RSS_green.fits B_RSS_green.fits --rtol 1e-6
    # restrict to the sky planes (e.g. when the two runs used different flux
    # standards, so FLAM/FLAM_ERR legitimately differ via the sensfunc):
    python -m llamas_pyjamas.Scripts.compare_rss A.fits B.fits \
        --planes SKYSUB SKY SKYRESID COUNTS ERROR

Exit code 0 if every compared plane is identical (within tolerance), 1 otherwise.
"""

import argparse
import sys

import numpy as np
from astropy.io import fits


def _image_hdus(hdul):
    """Map EXTNAME -> ndarray for every image extension with 2-D data."""
    out = {}
    for i, h in enumerate(hdul):
        data = getattr(h, "data", None)
        if data is None or np.ndim(data) < 2:
            continue
        name = h.name or f"HDU{i}"
        out[name] = np.asarray(data, dtype=float)
    return out


def _plane_diff(a, b, rtol, atol):
    """Return (identical, n_diff, max_abs, detail) for two arrays (NaN==NaN)."""
    if a.shape != b.shape:
        return False, -1, float("nan"), f"shape {a.shape} vs {b.shape}"
    both_nan = np.isnan(a) & np.isnan(b)
    close = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True) | both_nan
    n_diff = int((~close).sum())
    with np.errstate(invalid="ignore"):
        diff = np.abs(a - b)
    diff[both_nan] = 0.0
    max_abs = float(np.nanmax(diff)) if diff.size else 0.0
    return n_diff == 0, n_diff, max_abs, ""


def compare(path_a, path_b, rtol=0.0, atol=0.0, planes=None):
    """Compare two RSS files. Returns True if all compared planes are identical.

    ``planes`` (optional) restricts the comparison to the named EXTNAMEs — use it
    to check only the sky planes when e.g. the two runs used different flux
    standards (so FLAM/FLAM_ERR differ via the sensfunc, unrelated to sky).
    """
    with fits.open(path_a) as ha, fits.open(path_b) as hb:
        A, B = _image_hdus(ha), _image_hdus(hb)
    shared = [k for k in A if k in B]
    only_a = sorted(set(A) - set(B))
    only_b = sorted(set(B) - set(A))

    if planes:
        want = {p.upper() for p in planes}
        missing = sorted(want - {k.upper() for k in shared})
        shared = [k for k in shared if k.upper() in want]
        if missing:
            print(f"WARNING: requested planes not shared/present: {missing}")

    print(f"A: {path_a}")
    print(f"B: {path_b}")
    print(f"shared image planes: {len(shared)}   only-in-A: {only_a or '-'}   "
          f"only-in-B: {only_b or '-'}")
    print(f"{'plane':<12} {'identical':<10} {'n_diff':>10} {'max|A-B|':>14}")
    all_same = True
    for k in sorted(shared):
        same, n_diff, max_abs, detail = _plane_diff(A[k], B[k], rtol, atol)
        all_same &= same
        flag = "yes" if same else "NO"
        extra = f"  {detail}" if detail else ""
        print(f"{k:<12} {flag:<10} {n_diff:>10} {max_abs:>14.6g}{extra}")
    verdict = "IDENTICAL" if all_same else "DIFFERENCES FOUND"
    print(f"\n{verdict} across {len(shared)} compared plane(s)"
          + (" (only-in-A/B extensions are informational)" if (only_a or only_b) else ""))
    return all_same


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare two RSS FITS files plane-by-plane.")
    p.add_argument("file_a")
    p.add_argument("file_b")
    p.add_argument("--rtol", type=float, default=0.0, help="relative tolerance (default 0 = exact)")
    p.add_argument("--atol", type=float, default=0.0, help="absolute tolerance (default 0 = exact)")
    p.add_argument("--planes", nargs="+", metavar="EXTNAME",
                   help="only compare these planes (e.g. SKYSUB SKY SKYRESID COUNTS ERROR)")
    args = p.parse_args(argv)
    ok = compare(args.file_a, args.file_b, rtol=args.rtol, atol=args.atol, planes=args.planes)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
