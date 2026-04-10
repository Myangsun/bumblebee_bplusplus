#!/usr/bin/env python3
"""
Pooled bootstrap confidence intervals for k-fold CV results.

Pools predictions across all folds for each config (since each image appears
in exactly one test fold), then runs bootstrap resampling for 95% CIs.

Usage:
    python scripts/kfold_bootstrap_ci.py --checkpoint f1
    python scripts/kfold_bootstrap_ci.py --checkpoint f1 --n-bootstrap 10000
    python scripts/kfold_bootstrap_ci.py --checkpoint focus
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import RESULTS_DIR
from scripts.bootstrap_ci import bootstrap_per_species_f1, print_results, plot_comparison

CONFIGS = ["baseline", "d3_cnp", "d4_synthetic", "d5_llm_filtered"]
N_FOLDS = 5
FOCUS_SPECIES = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]


def pool_fold_predictions(config: str, checkpoint: str) -> list[dict]:
    """Pool test predictions across all folds for one config."""
    pooled = []
    for fold in range(N_FOLDS):
        results_dir = RESULTS_DIR / f"{config}_fold{fold}_gbif"
        results_path = results_dir / f"test_results_{checkpoint}.json"
        if not results_path.exists() and checkpoint == "multitask":
            results_path = results_dir / "test_results.json"
        if not results_path.exists():
            print(f"  WARNING: {results_path} not found, skipping fold {fold}")
            continue
        with open(results_path) as f:
            data = json.load(f)
        pooled.extend(data["detailed_predictions"])
    return pooled


def main():
    parser = argparse.ArgumentParser(
        description="Pooled bootstrap CIs for k-fold CV results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="f1",
                        choices=["multitask", "f1", "focus"],
                        help="Checkpoint type to analyze (default: f1)")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Number of bootstrap iterations (default: 10000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON results to file")
    parser.add_argument("--plot", type=str, default=None,
                        help="Save comparison plot to file")
    parser.add_argument("--focus-species", nargs="+", default=FOCUS_SPECIES,
                        help="Species to highlight in plots")
    args = parser.parse_args()

    cp = args.checkpoint
    output_path = args.output or str(RESULTS_DIR / f"kfold_bootstrap_ci_{cp}.json")
    plot_path = args.plot or str(RESULTS_DIR / f"kfold_bootstrap_ci_{cp}.png")

    all_results = {}
    for config in CONFIGS:
        print(f"\nPooling {N_FOLDS} folds for {config} (checkpoint: {cp})...")
        pooled = pool_fold_predictions(config, cp)
        if not pooled:
            print(f"  No predictions found for {config}, skipping")
            continue
        print(f"  Pooled {len(pooled)} predictions across {N_FOLDS} folds")

        results = bootstrap_per_species_f1(pooled, n_bootstrap=args.n_bootstrap)
        all_results[config] = results
        print_results(config, results)

    # CI overlap analysis
    if len(all_results) >= 2:
        config_names = list(all_results.keys())
        print(f"\n{'=' * 80}")
        print("CI OVERLAP ANALYSIS (pooled k-fold bootstrap)")
        print(f"{'=' * 80}")

        for sp in args.focus_species + ["__macro_f1__"]:
            label = sp.replace("Bombus_", "B. ") if not sp.startswith("__") else "Macro F1"
            print(f"\n  {label}:")
            for i in range(len(config_names)):
                for j in range(i + 1, len(config_names)):
                    a = all_results[config_names[i]].get(sp, {})
                    b = all_results[config_names[j]].get(sp, {})
                    if not a or not b:
                        continue
                    overlap_lo = max(a["ci_lower"], b["ci_lower"])
                    overlap_hi = min(a["ci_upper"], b["ci_upper"])
                    overlaps = overlap_lo < overlap_hi
                    diff = b["observed"] - a["observed"]
                    sign = "+" if diff >= 0 else ""
                    status = "OVERLAPPING" if overlaps else "NON-OVERLAPPING"
                    print(f"    {config_names[i]} vs {config_names[j]}: "
                          f"diff={sign}{diff:.4f}  {status}")

    # Save
    Path(output_path).write_text(json.dumps(all_results, indent=2))
    print(f"\nJSON saved: {output_path}")

    plot_comparison(all_results, args.focus_species, Path(plot_path))


if __name__ == "__main__":
    main()
