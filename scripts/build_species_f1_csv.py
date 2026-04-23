#!/usr/bin/env python3
"""Rebuild RESULTS/failure_analysis/species_f1_{multiseed,kfold}.csv with
D5 (d2_centroid) and D6 (d6_probe) columns added alongside the existing D1-D4.

Columns per variant: {variant}_f1_mean, {variant}_f1_std, {variant}_f1_delta
where delta = (variant mean − baseline mean).
"""
from __future__ import annotations

import csv, glob, json
from pathlib import Path

import numpy as np

ROOT = Path("/home/msun14/bumblebee_bplusplus")
OUT_DIR = ROOT / "RESULTS/failure_analysis"

# All 16 species + their train counts + tier bucket (from §3.3).
TRAIN_N = {
    "Bombus_ashtoni": 22, "Bombus_sandersoni": 40, "Bombus_flavidus": 162,
    "Bombus_affinis": 268, "Bombus_citrinus": 395, "Bombus_vagans_Smith": 443,
    "Bombus_borealis": 471, "Bombus_terricola": 479, "Bombus_fervidus": 639,
    "Bombus_perplexus": 683, "Bombus_rufocinctus": 963,
    "Bombus_impatiens": 1227, "Bombus_ternarius_Say": 1247,
    "Bombus_bimaculatus": 1250, "Bombus_griseocollis": 1274,
    "Bombus_pensylvanicus": 1283,
}

def tier(n):
    if n >= 900: return "common"
    if n >= 200: return "moderate"
    return "rare"

# Variants to aggregate.
VARIANTS = ["baseline", "d3_cnp", "d4_synthetic", "d5_llm_filtered", "d2_centroid", "d6_probe"]


def latest(pat):
    m = sorted(glob.glob(str(ROOT / pat)))
    return Path(m[-1]) if m else None


def load_json(patterns):
    for pat in patterns:
        p = latest(pat)
        if p: return json.loads(p.read_text())
    return None


def load_variant_runs(variant, protocol):
    """Returns list of f1-eval dicts for the variant under the requested protocol."""
    out = []
    if protocol == "multiseed":
        for s in (42, 43, 44, 45, 46):
            d = load_json([f"RESULTS_seeds/{variant}_seed{s}@f1_seed_test_results_*.json",
                           f"RESULTS/{variant}_seed{s}@f1_seed_test_results_*.json"])
            if d: out.append(d)
    elif protocol == "kfold":
        for k in range(5):
            d = load_json([f"RESULTS_kfold/{variant}_fold{k}@f1_kfold_test_results_*.json",
                           f"RESULTS/{variant}_fold{k}@f1_kfold_test_results_*.json"])
            if d: out.append(d)
    return out


def species_f1(d, sp):
    return d["species_metrics"].get(sp, {}).get("f1", float("nan"))


def build(protocol):
    # species_f1_vals[variant][species] = list of F1 across seeds/folds
    species_f1_vals = {v: {sp: [] for sp in TRAIN_N} for v in VARIANTS}
    for v in VARIANTS:
        runs = load_variant_runs(v, protocol)
        if not runs:
            print(f"  WARN: no runs for {v} protocol={protocol}")
            continue
        for d in runs:
            for sp in TRAIN_N:
                species_f1_vals[v][sp].append(species_f1(d, sp))

    # Build rows per species
    rows = []
    for sp, n in TRAIN_N.items():
        row = {"species": sp, "train_n": n, "tier": tier(n)}
        for v in VARIANTS:
            vals = np.array([x for x in species_f1_vals[v][sp] if not np.isnan(x)])
            if len(vals) == 0:
                row[f"{v}_f1_mean"] = float("nan"); row[f"{v}_f1_std"] = float("nan")
            else:
                row[f"{v}_f1_mean"] = float(vals.mean())
                row[f"{v}_f1_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        base_mean = row["baseline_f1_mean"]
        for v in VARIANTS:
            if v == "baseline":
                continue
            if np.isnan(base_mean) or np.isnan(row[f"{v}_f1_mean"]):
                row[f"{v}_f1_delta"] = float("nan")
            else:
                row[f"{v}_f1_delta"] = round(row[f"{v}_f1_mean"] - base_mean, 4)
            row[f"{v}_f1_mean"] = round(row[f"{v}_f1_mean"], 4)
            row[f"{v}_f1_std"] = round(row[f"{v}_f1_std"], 4)
        row["baseline_f1_mean"] = round(row["baseline_f1_mean"], 4)
        row["baseline_f1_std"] = round(row["baseline_f1_std"], 4)
        rows.append(row)

    # Stable column order
    cols = ["species", "train_n", "tier", "baseline_f1_mean", "baseline_f1_std"]
    for v in VARIANTS:
        if v == "baseline":
            continue
        cols += [f"{v}_f1_mean", f"{v}_f1_std"]
    for v in VARIANTS:
        if v == "baseline":
            continue
        cols.append(f"{v}_f1_delta")

    out = OUT_DIR / f"species_f1_{protocol}.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    print(f"-> {out} ({len(rows)} rows, {len(cols)} cols)")


if __name__ == "__main__":
    for proto in ("multiseed", "kfold"):
        print(f"== {proto} ==")
        build(proto)
