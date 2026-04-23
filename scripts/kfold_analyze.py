#!/usr/bin/env python3
"""
Aggregate 5-fold CV results and run paired statistical tests.

Reads test_results.json from each fold × config, computes per-fold metrics,
then runs paired t-tests to determine significance of differences.

Usage:
    python scripts/kfold_analyze.py
    python scripts/kfold_analyze.py --output RESULTS/kfold_analysis.json --plot RESULTS/kfold_analysis.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import RESULTS_DIR

CONFIGS = ["baseline", "d3_cnp", "d4_synthetic", "d5_llm_filtered"]
N_FOLDS = 5
FOCUS_SPECIES = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]


def _compute_f1(y_true: list, y_pred: list, species: str) -> float:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == species and p == species)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != species and p == species)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == species and p != species)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0


def _compute_macro_f1(y_true: list, y_pred: list) -> float:
    species = sorted(set(y_true))
    return float(np.mean([_compute_f1(y_true, y_pred, sp) for sp in species]))


def load_fold_results(checkpoint: str = "multitask") -> dict[str, dict[int, dict]]:
    """Load test results for each config x fold.

    Args:
        checkpoint: Which checkpoint to analyze (multitask, f1, focus).
                    Tries test_results_{checkpoint}.json first, falls back
                    to test_results.json for backward compatibility.

    Returns: {config: {fold_idx: {"macro_f1": ..., "accuracy": ..., "species": {...}}}}
    """
    results = {}
    for config in CONFIGS:
        results[config] = {}
        for fold in range(N_FOLDS):
            dataset = f"{config}_fold{fold}"
            results_dir = RESULTS_DIR / f"{dataset}_gbif"
            # Prefer checkpoint-specific file, fall back to legacy only for multitask
            results_path = results_dir / f"test_results_{checkpoint}.json"
            if not results_path.exists() and checkpoint == "multitask":
                results_path = results_dir / "test_results.json"

            if not results_path.exists():
                print(f"  WARNING: {results_path} not found")
                continue

            with open(results_path) as f:
                data = json.load(f)

            preds = data["detailed_predictions"]
            y_true = [p["ground_truth"] for p in preds]
            y_pred = [p["prediction"] for p in preds]

            macro_f1 = _compute_macro_f1(y_true, y_pred)
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

            species_f1 = {}
            for sp in sorted(set(y_true)):
                f1 = _compute_f1(y_true, y_pred, sp)
                n = sum(1 for t in y_true if t == sp)
                species_f1[sp] = {"f1": f1, "support": n}

            results[config][fold] = {
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "n_test": len(y_true),
                "species": species_f1,
            }

    return results


def aggregate_and_test(results: dict) -> dict:
    """Compute per-config summaries and pairwise paired t-tests."""
    analysis = {"configs": {}, "pairwise_tests": {}}

    # Per-config summary
    for config in CONFIGS:
        folds = results.get(config, {})
        if len(folds) < N_FOLDS:
            print(f"  WARNING: {config} has only {len(folds)}/{N_FOLDS} folds")
            continue

        macro_f1s = [folds[f]["macro_f1"] for f in range(N_FOLDS)]
        accs = [folds[f]["accuracy"] for f in range(N_FOLDS)]

        config_summary = {
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
            "macro_f1_per_fold": macro_f1s,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1)),
        }

        # Per-species across folds
        species_summary = {}
        for sp in FOCUS_SPECIES:
            sp_f1s = []
            sp_supports = []
            for f in range(N_FOLDS):
                sp_data = folds[f]["species"].get(sp, {"f1": 0, "support": 0})
                sp_f1s.append(sp_data["f1"])
                sp_supports.append(sp_data["support"])
            species_summary[sp] = {
                "f1_mean": float(np.mean(sp_f1s)),
                "f1_std": float(np.std(sp_f1s, ddof=1)),
                "f1_per_fold": sp_f1s,
                "total_test": sum(sp_supports),
            }
        config_summary["species"] = species_summary
        analysis["configs"][config] = config_summary

    # Pairwise paired t-tests on macro F1
    config_list = [c for c in CONFIGS if c in analysis["configs"]]
    for i in range(len(config_list)):
        for j in range(i + 1, len(config_list)):
            a, b = config_list[i], config_list[j]
            a_scores = analysis["configs"][a]["macro_f1_per_fold"]
            b_scores = analysis["configs"][b]["macro_f1_per_fold"]

            t_stat, p_value = stats.ttest_rel(b_scores, a_scores)
            diff = np.mean(b_scores) - np.mean(a_scores)
            key = f"{a}_vs_{b}"
            analysis["pairwise_tests"][key] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "mean_diff": float(diff),
                "significant_at_005": bool(p_value < 0.05),
            }

            # Also per focus species
            species_tests = {}
            for sp in FOCUS_SPECIES:
                a_sp = analysis["configs"][a]["species"][sp]["f1_per_fold"]
                b_sp = analysis["configs"][b]["species"][sp]["f1_per_fold"]
                t_sp, p_sp = stats.ttest_rel(b_sp, a_sp)
                species_tests[sp] = {
                    "t_statistic": float(t_sp),
                    "p_value": float(p_sp),
                    "mean_diff": float(np.mean(b_sp) - np.mean(a_sp)),
                    "significant_at_005": bool(p_sp < 0.05),
                }
            analysis["pairwise_tests"][key]["species"] = species_tests

    return analysis


def print_analysis(analysis: dict):
    """Pretty-print the analysis."""
    print("\n" + "=" * 80)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 80)

    # Summary table
    print(f"\n{'Config':<20} {'Macro F1':>12} {'Accuracy':>12}")
    print("-" * 48)
    for config in CONFIGS:
        if config not in analysis["configs"]:
            continue
        c = analysis["configs"][config]
        print(f"{config:<20} {c['macro_f1_mean']:.4f}±{c['macro_f1_std']:.4f} "
              f"{c['accuracy_mean']:.4f}±{c['accuracy_std']:.4f}")

    # Per-species
    print(f"\n{'Config':<20}", end="")
    for sp in FOCUS_SPECIES:
        label = sp.replace("Bombus_", "B. ")
        print(f" {label:>16}", end="")
    print()
    print("-" * (20 + 16 * len(FOCUS_SPECIES)))
    for config in CONFIGS:
        if config not in analysis["configs"]:
            continue
        c = analysis["configs"][config]
        print(f"{config:<20}", end="")
        for sp in FOCUS_SPECIES:
            s = c["species"][sp]
            print(f" {s['f1_mean']:.3f}±{s['f1_std']:.3f}", end="")
            print(f"({s['total_test']})", end="")
        print()

    # Paired tests
    print(f"\n{'Comparison':<35} {'Diff':>8} {'t-stat':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 70)
    for key, test in analysis["pairwise_tests"].items():
        sig = "YES" if test["significant_at_005"] else "no"
        print(f"{key:<35} {test['mean_diff']:>+8.4f} {test['t_statistic']:>8.3f} "
              f"{test['p_value']:>10.4f} {sig:>6}")


def plot_analysis(analysis: dict, output_path: Path):
    """Box plots of per-fold macro F1 and per-species F1."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs_present = [c for c in CONFIGS if c in analysis["configs"]]
    n_configs = len(configs_present)

    fig, axes = plt.subplots(1, 1 + len(FOCUS_SPECIES), figsize=(5 * (1 + len(FOCUS_SPECIES)), 5))

    colors = {"baseline": "#9E9E9E", "d3_cnp": "#4CAF50",
              "d4_synthetic": "#2196F3", "d5_llm_filtered": "#FF9800"}

    # Macro F1 box plot
    ax = axes[0]
    data = [analysis["configs"][c]["macro_f1_per_fold"] for c in configs_present]
    bp = ax.boxplot(data, labels=[c.replace("_", "\n") for c in configs_present],
                    patch_artist=True, widths=0.6)
    for patch, config in zip(bp["boxes"], configs_present):
        patch.set_facecolor(colors.get(config, "#CCCCCC"))
        patch.set_alpha(0.7)
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 (5-fold CV)")
    ax.grid(True, alpha=0.3, axis="y")

    # Per-species box plots
    for sp_idx, sp in enumerate(FOCUS_SPECIES):
        ax = axes[1 + sp_idx]
        data = [analysis["configs"][c]["species"][sp]["f1_per_fold"] for c in configs_present]
        total_n = analysis["configs"][configs_present[0]]["species"][sp]["total_test"]
        bp = ax.boxplot(data, labels=[c.replace("_", "\n") for c in configs_present],
                        patch_artist=True, widths=0.6)
        for patch, config in zip(bp["boxes"], configs_present):
            patch.set_facecolor(colors.get(config, "#CCCCCC"))
            patch.set_alpha(0.7)
        label = sp.replace("Bombus_", "B. ")
        ax.set_ylabel("F1")
        ax.set_title(f"{label} (n={total_n})")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("5-Fold Cross-Validation: Augmentation Strategy Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze k-fold CV results")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--checkpoint", default="multitask",
                        choices=["multitask", "f1", "focus"],
                        help="Which checkpoint type to analyze (default: multitask)")
    args = parser.parse_args()

    # Default output paths include checkpoint type for clarity
    cp = args.checkpoint
    output_path = args.output or str(RESULTS_DIR / f"kfold_analysis_{cp}.json")
    plot_path = args.plot or str(RESULTS_DIR / f"kfold_analysis_{cp}.png")

    print(f"Loading fold results (checkpoint: {cp})...")
    results = load_fold_results(checkpoint=cp)

    print("Computing analysis...")
    analysis = aggregate_and_test(results)

    print_analysis(analysis)

    Path(output_path).write_text(json.dumps(analysis, indent=2))
    print(f"\nJSON saved: {output_path}")

    plot_analysis(analysis, Path(plot_path))


if __name__ == "__main__":
    main()
