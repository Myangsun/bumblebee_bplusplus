#!/usr/bin/env python3
"""Build RESULTS/failure_analysis/subset_ablation_recovery_f1ckpt.csv under
present thesis labels D3/D4/D5/D6 using the f1-checkpoint evaluations.

Each (variant, dropped_species) cell yields three rows (one per measured
species) with the Δ = F1(ablated) − F1(full) on that measured species.
"""
from __future__ import annotations

import csv, glob, json
from pathlib import Path

ROOT = Path("/home/msun14/bumblebee_bplusplus")
OUT = ROOT / "RESULTS/failure_analysis/subset_ablation_recovery_f1ckpt.csv"

# thesis label -> code key
CODE = {"D3": "d4_synthetic",
        "D4": "d5_llm_filtered",
        "D5": "d2_centroid",
        "D6": "d6_probe"}
RARE = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]


def latest(pat):
    m = sorted(glob.glob(str(ROOT / pat)))
    return Path(m[-1]) if m else None


def load_json(pat_list):
    for pat in pat_list:
        p = latest(pat)
        if p: return json.loads(p.read_text())
    return None


def f1_sp(d, sp): return d["species_metrics"].get(sp, {}).get("f1", float("nan"))


rows_out = []
for label, code in CODE.items():
    base = load_json([f"RESULTS_seeds/{code}_seed42@f1_seed_test_results_*.json",
                      f"RESULTS/{code}_seed42@f1_seed_test_results_*.json"])
    if not base:
        print(f"MISSING base for {label}"); continue
    for dropped in RARE:
        short = dropped.replace("Bombus_", "")
        abl = load_json([f"RESULTS/{code}_seed42_no-{short}@f1_seed_test_results_*.json",
                         f"RESULTS_seeds/{code}_seed42_no-{short}@f1_seed_test_results_*.json"])
        if not abl:
            print(f"MISSING {label} no-{short}"); continue
        for measured in RARE:
            f1_full = f1_sp(base, measured)
            f1_abl = f1_sp(abl, measured)
            rec = f1_abl - f1_full
            rows_out.append({
                "variant": label,
                "dropped": dropped,
                "measured_species": measured,
                "f1_full": round(f1_full, 4),
                "f1_ablated": round(f1_abl, 4),
                "recovery": round(rec, 4),
                "target_species": dropped == measured,
            })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
    w.writeheader(); w.writerows(rows_out)
print(f"-> {OUT} ({len(rows_out)} rows)")
