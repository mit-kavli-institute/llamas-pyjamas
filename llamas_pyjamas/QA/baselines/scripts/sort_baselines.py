#!/usr/bin/env python3
"""Copy-sort LLAMAS baseline cal MEF files from copies/ into sibling type folders
by PRODCATG, and write a per-folder obs manifest. Idempotent: skips files already
present in the destination. Non-destructive: copies/ is left intact.
"""
import os, glob, shutil, csv, sys
from astropy.io import fits

BASE = "/Users/slh/Library/CloudStorage/Box-Box/slhughes/LLAMAS_analysis/QA_baselines"
COPIES = os.path.join(BASE, "copies")

# PRODCATG -> sibling folder name
FOLDER = {
    "CAL.R-BIA": "Bias",
    "CAL.R-DRK": "Darks",
    "CAL.R-ARC": "Arcs",
    "CAL.R-FLT": "lamp_flats",
    "CAL.R-SKY": "twilight_flats",
}

def primary_info(path):
    ph = fits.getheader(path, 0)
    return {
        "PRODCATG": str(ph.get("PRODCATG", "?")).strip(),
        "OBJECT": str(ph.get("OBJECT", "")).strip(),
        "REXPTIME": ph.get("REXPTIME"),
        "SEXPTIME": ph.get("SEXPTIME"),
        "DEXPTIME": ph.get("DEXPTIME"),
        "READ-MDE": str(ph.get("READ-MDE", "")).strip(),
        "UTC": str(ph.get("UTC", "")).strip(),
        "DATE-OBS": str(ph.get("DATE-OBS", "")).strip(),
    }

def main():
    files = sorted(glob.glob(os.path.join(COPIES, "*_mef.fits")))
    print(f"[sort] {len(files)} files in copies/", flush=True)
    manifests = {name: [] for name in FOLDER.values()}
    unknown = []
    copied = skipped = errors = 0
    for i, p in enumerate(files, 1):
        try:
            info = primary_info(p)
        except Exception as e:
            print(f"[sort] ERR header {os.path.basename(p)}: {e}", flush=True)
            errors += 1
            continue
        pc = info["PRODCATG"]
        folder = FOLDER.get(pc)
        if folder is None:
            unknown.append((os.path.basename(p), pc))
            continue
        dest_dir = os.path.join(BASE, folder)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, os.path.basename(p))
        rec = {"filename": os.path.basename(p), **info}
        manifests[folder].append(rec)
        if os.path.exists(dest) and os.path.getsize(dest) == os.path.getsize(p):
            skipped += 1
        else:
            shutil.copy2(p, dest)
            copied += 1
            if copied % 10 == 0:
                print(f"[sort] copied {copied} (at {i}/{len(files)})", flush=True)
    # Write manifests
    cols = ["filename", "PRODCATG", "OBJECT", "REXPTIME", "SEXPTIME", "DEXPTIME",
            "READ-MDE", "UTC", "DATE-OBS"]
    for folder, recs in manifests.items():
        recs.sort(key=lambda r: r["filename"])
        mpath = os.path.join(BASE, folder, "manifest.csv")
        with open(mpath, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(recs)
        print(f"[sort] {folder:16s} {len(recs):3d} files -> {mpath}", flush=True)
    if unknown:
        print(f"[sort] UNKNOWN PRODCATG ({len(unknown)}): {unknown[:10]}", flush=True)
    print(f"[sort] DONE copied={copied} skipped(existing)={skipped} errors={errors}", flush=True)

if __name__ == "__main__":
    sys.exit(main())
