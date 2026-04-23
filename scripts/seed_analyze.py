#!/usr/bin/env python3
"""
Aggregate 5-seed training results and run paired statistical tests.

Reads test_results.json from each seed x config, computes per-seed metrics,
then runs paired t-tests to determine significance of differences.

Usage:
    python scripts/seed_analyze.py
    python scripts/seed_analyze.py --checkpoint f1
    python scripts/seed_analyze.py --output RESULTS/seed_analysis.json --plot RESULTS/seed_analysis.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import RESULTS_DIR

CONFIGS = ["baseline", "d3_cnp", "d4_synthetic", "d5_llm_filtered"]
SEEDS = [42, 43, 44, 45, 46]
N_SEEDS = len(SEEDS)
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


def load_seed_results(checkpoint: str = "f1") -> dict[str, dict[int, dict]]:
    """Load test results for each config x seed.

    Returns: {config: {seed_idx: {"macro_f1": ..., "accuracy": ..., "species": {...}}}}
    """
    results = {}
    for config in CONFIGS:
        results[config] = {}
        for i, seed in enumerate(SEEDS):
            results_dir = RESULTS_DIR / f"{config}_seed{seed}_gbif"
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

            results[config][i] = {
                "seed": seed,
                "macro_f1": macro_f1,
                "accuracy": accuracy,
                "n_test": len(y_true),
                "species": species_f1,
            }

    return results


def aggregate_and_test(results: dict) -> dict:
    """Compute per-config summaries and pairwise paired t-tests."""
    analysis = {"configs": {}, "pairwise_tests": {}}

    for config in CONFIGS:
        seeds = results.get(config, {})
        if len(seeds) < N_SEEDS:
            print(f"  WARNING: {config} has only {len(seeds)}/{N_SEEDS} seeds")
            continue

        macro_f1s = [seeds[i]["macro_f1"] for i in range(N_SEEDS)]
        accs = [seeds[i]["accuracy"] for i in range(N_SEEDS)]

        config_summary = {
            "macro_f1_mean": float(np.mean(macro_f1s)),
            "macro_f1_std": float(np.std(macro_f1s, ddof=1)),
            "macro_f1_per_seed": macro_f1s,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs, ddof=1)),
        }

        species_summary = {}
        for sp in FOCUS_SPECIES:
            sp_f1s = []
            sp_supports = []
            for i in range(N_SEEDS):
                sp_data = seeds[i]["species"].get(sp, {"f1": 0, "support": 0})
                sp_f1s.append(sp_data["f1"])
                sp_supports.append(sp_data["support"])
            species_summary[sp] = {
                "f1_mean": float(np.mean(sp_f1s)),
                "f1_std": float(np.std(sp_f1s, ddof=1)),
                "f1_per_seed": sp_f1s,
                "total_test": sp_supports[0] if sp_supports else 0,
            }
        config_summary["species"] = species_summary
        analysis["configs"][config] = config_summary

    # Pairwise paired t-tests on macro F1 (paired by seed index)
    config_list = [c for c in CONFIGS if c in analysis["configs"]]
    for i in range(len(config_list)):
        for j in range(i + 1, len(config_list)):
            a, b = config_list[i], config_list[j]
            a_scores = analysis["configs"][a]["macro_f1_per_seed"]
            b_scores = analysis["configs"][b]["macro_f1_per_seed"]

            t_stat, p_value = stats.ttest_rel(b_scores, a_scores)
            diff = np.mean(b_scores) - np.mean(a_scores)
            key = f"{a}_vs_{b}"
            analysis["pairwise_tests"][key] = {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "mean_diff": float(diff),
                "significant_at_005": bool(p_value < 0.05),
            }

            species_tests = {}
            for sp in FOCUS_SPECIES:
                a_sp = analysis["configs"][a]["species"][sp]["f1_per_seed"]
                b_sp = analysis["configs"][b]["species"][sp]["f1_per_seed"]
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
    print("\n" + "=" * 80)
    print("5-SEED TRAINING VARIANCE RESULTS")
    print("=" * 80)

    print(f"\n{'Config':<20} {'Macro F1':>12} {'Accuracy':>12}")
    print("-" * 48)
    for config in CONFIGS:
        if config not in analysis["configs"]:
            continue
        c = analysis["configs"][config]
        print(f"{config:<20} {c['macro_f1_mean']:.4f}\u00b1{c['macro_f1_std']:.4f} "
              f"{c['accuracy_mean']:.4f}\u00b1{c['accuracy_std']:.4f}")

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
            print(f" {s['f1_mean']:.3f}\u00b1{s['f1_std']:.3f}", end="")
            print(f"(n={s['total_test']})", end="")
        print()

    print(f"\n{'Comparison':<35} {'Diff':>8} {'t-stat':>8} {'p-value':>10} {'Sig?':>6}")
    print("-" * 70)
    for key, test in analysis["pairwise_tests"].items():
        sig = "YES" if test["significant_at_005"] else "no"
        print(f"{key:<35} {test['mean_diff']:>+8.4f} {test['t_statistic']:>8.3f} "
              f"{test['p_value']:>10.4f} {sig:>6}")


def plot_analysis(analysis: dict, output_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs_present = [c for c in CONFIGS if c in analysis["configs"]]

    fig, axes = plt.subplots(1, 1 + len(FOCUS_SPECIES), figsize=(5 * (1 + len(FOCUS_SPECIES)), 5))

    colors = {"baseline": "#9E9E9E", "d3_cnp": "#4CAF50",
              "d4_synthetic": "#2196F3", "d5_llm_filtered": "#FF9800"}

    ax = axes[0]
    data = [analysis["configs"][c]["macro_f1_per_seed"] for c in configs_present]
    bp = ax.boxplot(data, tick_labels=[c.replace("_", "\n") for c in configs_present],
                    patch_artist=True, widths=0.6)
    for patch, config in zip(bp["boxes"], configs_present):
        patch.set_facecolor(colors.get(config, "#CCCCCC"))
        patch.set_alpha(0.7)
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 (5 seeds)")
    ax.grid(True, alpha=0.3, axis="y")

    for sp_idx, sp in enumerate(FOCUS_SPECIES):
        ax = axes[1 + sp_idx]
        data = [analysis["configs"][c]["species"][sp]["f1_per_seed"] for c in configs_present]
        total_n = analysis["configs"][configs_present[0]]["species"][sp]["total_test"]
        bp = ax.boxplot(data, tick_labels=[c.replace("_", "\n") for c in configs_present],
                        patch_artist=True, widths=0.6)
        for patch, config in zip(bp["boxes"], configs_present):
            patch.set_facecolor(colors.get(config, "#CCCCCC"))
            patch.set_alpha(0.7)
        label = sp.replace("Bombus_", "B. ")
        ax.set_ylabel("F1")
        ax.set_title(f"{label} (n={total_n})")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("5-Seed Training: Augmentation Strategy Comparison",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze 5-seed training results")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--plot", type=str, default=None)
    parser.add_argument("--checkpoint", default="f1",
                        choices=["multitask", "f1", "focus"],
                        help="Which checkpoint type to analyze (default: f1)")
    args = parser.parse_args()

    cp = args.checkpoint
    output_path = args.output or str(RESULTS_DIR / f"seed_analysis_{cp}.json")
    plot_path = args.plot or str(RESULTS_DIR / f"seed_analysis_{cp}.png")

    print(f"Loading seed results (checkpoint: {cp})...")
    results = load_seed_results(checkpoint=cp)

    print("Computing analysis...")
    analysis = aggregate_and_test(results)

    print_analysis(analysis)

    Path(output_path).write_text(json.dumps(analysis, indent=2))
    print(f"\nJSON saved: {output_path}")

    plot_analysis(analysis, Path(plot_path))


if __name__ == "__main__":
    main()
