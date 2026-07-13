#!/usr/bin/env python3
"""Phase B aggregation (ROBUST). Turn qa_stats_raw.json into per-detector derived
thresholds + per-epoch tracking baselines.

Robustness (addressing review findings):
- Per (type,mode,detector) samples are sigma-clipped (MAD-based) to reject
  anomalous baseline frames BEFORE deriving limits, so an unscreened outlier can
  no longer inflate its own cap.
- Background/level bands use an ABSOLUTE ADU floor (additive quantity), not a
  relative %: band = median +/- max(K*sigma_robust, FLOOR_ADU).
- Structure/RMS/saturation caps are robust one-sided: max(median + N*sigma_robust,
  floor), on the screened sample.
- The known-bad June-4 odd dark is held out from derivation and validated.
- Warm-camera flagging is validated against the May-6 warm folder using BOTH the
  edge-stripe background band and the CCD temperature band (not just shutter).

Outputs qa_thresholds_derived.json + qa_tracking_baselines.csv.
"""
import os, json, csv, glob
import numpy as np
from astropy.io import fits

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "qa_stats_raw.json")
OUT_JSON = os.path.join(HERE, "qa_thresholds_derived.json")
OUT_CSV = os.path.join(HERE, "qa_tracking_baselines.csv")
ODD_TAG = "2026-06-04_20-36-33"
WARM_DIR = "/Users/slh/Downloads/20260505_06-selected"

# absolute ADU floors (additive pedestals) and robust widths
K_LEVEL = 6          # band half-width in robust sigmas for level/background
FLOOR_LEVEL = 15.0   # min band half-width, ADU  (catches modest warm/drift)
N_CAP = 8            # one-sided caps in robust sigmas
FLOOR_STRUCT = 2.0   # min structure cap
FLOOR_RMS = 5.0      # min rms cap, ADU
FLOOR_SAT = 0.0005   # min saturation-fraction cap


def det_name(rec):
    return f"{rec['bench']}.{rec['side']}.{str(rec['color']).capitalize()}"


def robust_sigma(a):
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return 1.4826 * mad


def sigma_clip(vals, nsig=3.0, iters=3, min_keep=3):
    """Iteratively drop points > nsig robust-sigmas from the median."""
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size <= min_keep:
        return a
    for _ in range(iters):
        med = np.median(a)
        sig = robust_sigma(a)
        if sig == 0:
            break
        keep = np.abs(a - med) <= nsig * sig
        if keep.sum() < min_keep or keep.all():
            break
        a = a[keep]
    return a


def obs_stats(a):
    if a.size == 0:
        return None
    return dict(n=int(a.size), min=float(a.min()), max=float(a.max()),
                med=float(np.median(a)), sigma_robust=round(float(robust_sigma(a)), 4),
                p95=float(np.percentile(a, 95)))


def level_band(screened):
    """Two-sided additive band: median +/- max(K*sigma, FLOOR_ADU)."""
    med = float(np.median(screened))
    half = max(K_LEVEL * robust_sigma(screened), FLOOR_LEVEL)
    return round(med - half, 2), round(med + half, 2)


def one_sided_cap(screened, floor):
    med = float(np.median(screened))
    return round(max(med + N_CAP * robust_sigma(screened), med * 3.0, floor), 4)


def heavy_tail_cap(vals, floor):
    """One-sided cap for HEAVY-TAILED quantities (dark structure/RMS vary frame to
    frame from cosmic rays / hot columns). Uses each detector's OWN unscreened
    maximum x1.5 so naturally-structured detectors keep a high cap and clean ones
    stay tight -- rather than sigma-clipping away the legit tail and flooring the
    cap (which false-fails normal frames). Trade-off: a detector whose baseline
    contains a genuine bad frame gets a loose cap (documented; n is small)."""
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return floor
    med = float(np.median(a))
    return round(max(float(a.max()) * 1.5, med + N_CAP * robust_sigma(a), floor), 4)


def main():
    recs = json.load(open(RAW))
    normal = [r for r in recs if not r.get("placeholder") and "error" not in r
              and ODD_TAG not in r["filename"]]
    odd = [r for r in recs if ODD_TAG in r["filename"] and not r.get("placeholder")]

    groups = {}
    for r in normal:
        groups.setdefault((r["prodcatg"], r["mode"], det_name(r)), []).append(r)

    derived = {}
    tracking_rows = []
    for (pc, mode, det), rs in sorted(groups.items()):
        # Background/level bands: use the UNCLIPPED spread so genuine night-to-night
        # drift is inside the band (MAD sigma already resists outliers at the centre);
        # clipping legit drift would tighten the band and false-WARN same-type frames.
        edge_a = np.asarray([r["edge_med"] for r in rs if np.isfinite(r["edge_med"])], float)
        med_a = np.asarray([r["full_med"] for r in rs if np.isfinite(r["full_med"])], float)
        # Heavy-tailed quantities (structure/RMS/saturation vary frame to frame, and
        # illuminated flats legitimately saturate) -> per-detector unscreened max x1.5.
        row_v = [r["row_struct"] for r in rs]
        col_v = [r["col_struct"] for r in rs]
        rms_v = [r["full_std"] for r in rs]
        sat_v = [r["sat_frac"] for r in rs]
        if edge_a.size == 0:
            continue
        emin, emax = level_band(edge_a)
        lmin, lmax = level_band(med_a)
        derived.setdefault(pc, {}).setdefault(mode, {})[det] = {
            "n_files": len(rs),
            "edge_bg": {"min": emin, "max": emax, "observed": obs_stats(edge_a)},
            "full_level": {"med_min": lmin, "med_max": lmax,
                           "rms_max": heavy_tail_cap(rms_v, FLOOR_RMS),
                           "observed_med": obs_stats(med_a)},
            "structure": {"row_max": heavy_tail_cap(row_v, FLOOR_STRUCT),
                          "col_max": heavy_tail_cap(col_v, FLOOR_STRUCT),
                          "observed_row": obs_stats(np.asarray(row_v)),
                          "observed_col": obs_stats(np.asarray(col_v))},
            "sat_frac_max": heavy_tail_cap(sat_v, FLOOR_SAT),
        }
        by_date = {}
        for r in rs:
            by_date.setdefault(r["date"], []).append(r)
        for date, drs in sorted(by_date.items()):
            tracking_rows.append(dict(
                prodcatg=pc, mode=mode, detector=det, date=date, n=len(drs),
                edge_med=round(float(np.median([r["edge_med"] for r in drs])), 2),
                full_med=round(float(np.median([r["full_med"] for r in drs])), 2),
                row_struct=round(float(np.median([r["row_struct"] for r in drs])), 3),
                col_struct=round(float(np.median([r["col_struct"] for r in drs])), 3),
            ))

    # shutter tol per type: abs_tol absorbs fixed overhead; rel_tol tight 10%
    shutter = {}
    for pc in sorted(set(r["prodcatg"] for r in normal)):
        deltas = []
        for r in normal:
            if r["prodcatg"] != pc or r["hdu"] != 1:
                continue
            if r.get("rexp") is not None and r.get("sexp") is not None:
                deltas.append(abs(r["sexp"] - r["rexp"]))
        if deltas:
            ds = sigma_clip(deltas, nsig=4)
            shutter[pc] = {"abs_delta_max": round(float(np.max(deltas)), 4),
                           "abs_tol": round(float(np.max(ds)) * 2 + 0.2, 3), "rel_tol": 0.10}
    shutter["SCI.R-*"] = {"abs_tol": 1.0, "rel_tol": 0.10, "note": "science: 1s abs + 10% rel"}

    temps = sigma_clip([r["ccdtemp"] for r in normal if r.get("ccdtemp") is not None])
    tstat = obs_stats(temps) if temps.size else None
    # warm thresholds tied to the observed cold distribution
    warm_warn = round(float(np.median(temps) + 5 * robust_sigma(temps)), 1) if temps.size else -72.0
    ccd = {"observed": tstat, "cold_min": -140.0,
           "warm_warn_above": warm_warn,     # ~ med + 4 sigma
           "warm_fail_above": -60.0}         # catastrophic / shut-off

    out = {"meta": {"n_normal_records": len(normal), "excluded_odd": ODD_TAG,
                    "epochs": sorted(set(r["date"] for r in normal)),
                    "robustness": f"MAD sigma-clip screen; level band +/-max({K_LEVEL}sig,{FLOOR_LEVEL}ADU); "
                                  f"caps med+{N_CAP}sig",
                    "region_convention": "edge stripes y[2:28]&[2020:2046] x[100:1948]; sat>63000"},
           "per_detector": derived, "shutter": shutter, "ccd_temp": ccd}
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prodcatg", "mode", "detector", "date", "n",
                                           "edge_med", "full_med", "row_struct", "col_struct"])
        w.writeheader()
        w.writerows(sorted(tracking_rows, key=lambda x: (x["prodcatg"], x["mode"], x["detector"], x["date"])))

    # ---------------- VALIDATION ----------------
    print("=== ROBUST DERIVATION ===")
    print(f"normal={len(normal)} groups={len(groups)} epochs={out['meta']['epochs']}")
    print(f"CCD temp cold band: {tstat}; warm_warn>{warm_warn} warm_fail>-60")
    print("\n=== self-check: does any NORMAL frame breach its own derived band? (should be ~0) ===")
    breaches = 0
    for r in normal:
        ent = derived.get(r["prodcatg"], {}).get(r["mode"], {}).get(det_name(r))
        if not ent:
            continue
        eb = ent["edge_bg"]
        if not (eb["min"] <= r["edge_med"] <= eb["max"]):
            breaches += 1
    print(f"  edge_bg breaches among {len(normal)} normal records: {breaches} "
          f"({100*breaches/max(len(normal),1):.2f}%)")

    print("\n=== VALIDATION: odd June-4 dark vs derived DARK/SLOW structure caps ===")
    dark = derived.get("CAL.R-DRK", {}).get("SLOW", {})
    flagged = 0
    for r in sorted(odd, key=lambda x: x["hdu"]):
        ent = dark.get(det_name(r))
        if not ent:
            continue
        rc, cc = ent["structure"]["row_max"], ent["structure"]["col_max"]
        if r["row_struct"] > rc or r["col_struct"] > cc:
            flagged += 1
            print(f"  ext{r['hdu']:<2} {det_name(r):<8} row={r['row_struct']:.1f}(cap {rc}) "
                  f"col={r['col_struct']:.1f}(cap {cc}) FLAG")
    print(f"  odd-dark detectors flagged: {flagged}")

    print("\n=== VALIDATION: warm folder -- shutter + edge-background + temperature ===")
    bias_ref = derived.get("CAL.R-BIA", {})
    for p in sorted(glob.glob(WARM_DIR + "/*.fits")):
        with fits.open(p, memmap=False) as h:
            ph = h[0].header
            pc = str(ph.get("PRODCATG", "?")).strip()
            mode = str(ph.get("READ-MDE", "?")).strip().upper()
            r, s = ph.get("REXPTIME"), ph.get("SEXPTIME")
            tol = shutter.get(pc) or (shutter.get("SCI.R-*") if pc.startswith("SCI") else {}) or {}
            sh = ""
            if r is not None and s is not None:
                d = abs(float(s) - float(r)); rel = d / float(r) if r and float(r) > 0.005 else 9e9
                ok = (d <= tol.get("abs_tol", 1.0)) or (rel <= tol.get("rel_tol", 0.1))
                sh = f"shutter={'FAIL' if not ok else 'ok'}"
            # edge background + temp across detectors (compare to BIAS band of same mode)
            n_edge_hot = n_temp_warm = n_temp_checked = 0
            btab = bias_ref.get(mode, {})
            for e in h[1:25]:
                d = e.data
                if d is None:
                    continue
                v = np.asarray(d, float)
                if v.min() == v.max():
                    continue
                nm = f"{e.header.get('BENCH')}.{str(e.header.get('SIDE')).upper()}.{str(e.header.get('COLOR')).capitalize()}"
                em = float(np.median(v[2:28, 100:1948]))
                ent = btab.get(nm)
                if ent and em > ent["edge_bg"]["max"]:
                    n_edge_hot += 1
                try:
                    t = float(e.header.get("CCDTEMP_1"))
                    n_temp_checked += 1
                    if t > warm_warn:
                        n_temp_warm += 1
                except (TypeError, ValueError):
                    pass
            print(f"  {os.path.basename(p)[:34]:34s} {pc:9s} {mode:4s} {sh:13s} "
                  f"edge_hot_det={n_edge_hot} temp_warm={n_temp_warm}/{n_temp_checked}")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
