#!/usr/bin/env python3
"""Aggregate verdict tally over ALL baseline files per type, WITHOUT writing reports.

Uses run_single_file() directly (no CLI, no .qa.json side effects) so the Box
baseline folders stay clean. Reports PASS/WARN/FAIL counts per type and lists any
FAIL (which would be a real anomaly among the 'normal' baselines).
"""
import sys, os, glob, json
from pathlib import Path
from collections import Counter

QA = "/Users/slh/Documents/Projects/Magellan_dev/LLAMAS/llamas-pyjamas/llamas_pyjamas/QA"
sys.path.insert(0, QA)
import qa_engine as E  # noqa

CAL = Path(f"{QA}/qa_config_cal.yaml")
BASE = "/Users/slh/Library/CloudStorage/Box-Box/slhughes/LLAMAS_analysis/QA_baselines"
FOLDERS = ["Bias", "Darks", "Arcs", "lamp_flats", "twilight_flats"]

# validate once, then run with no_validate for speed
cfg = E.load_yaml(CAL)
E.validate_config(cfg, CAL)
eng = E.QAEngine(cfg)

grand = {}
fails = []
for fold in FOLDERS:
    files = sorted(glob.glob(f"{BASE}/{fold}/*.fits"))
    tally = Counter()
    for f in files:
        try:
            rep = eng.run(Path(f))
            v = rep["overall_verdict"]
        except Exception as exc:
            v = "ERROR"
        tally[v] += 1
        if v == "FAIL":
            fails.append((fold, os.path.basename(f)))
    grand[fold] = dict(tally)
    print(f"{fold:16s} n={len(files):3d}  " +
          "  ".join(f"{k}={tally[k]}" for k in ("PASS", "WARN", "FAIL", "ERROR") if tally[k]))

print("\nFAIL files (real anomalies among baselines):")
for fold, name in fails:
    print(f"  {fold}: {name}")
if not fails:
    print("  (none)")

json.dump({"per_type": grand, "fails": fails},
          open(f"{os.path.dirname(os.path.abspath(__file__))}/batch_tally.json", "w"), indent=2)
print("\nwrote batch_tally.json")
