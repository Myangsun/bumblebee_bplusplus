#!/usr/bin/env python3
"""
Bootstrap confidence intervals for bumblebee classifier evaluation.

Resamples test set predictions with replacement to compute 95% CIs
for per-species F1, macro F1, and overall accuracy.

Usage:
    # Single model
    python scripts/bootstrap_ci.py --results RESULTS/baseline_gbif/test_results.json

    # Compare two models
    python scripts/bootstrap_ci.py --results RESULTS/baseline_gbif/test_results.json \
                                              RESULTS/d5_llm_filtered_200_gbif/test_results.json

    # All ablation models
    python scripts/bootstrap_ci.py --results RESULTS/baseline_gbif/test_results.json \
                                              RESULTS/d5_llm_filtered_50_gbif/test_results.json \
                                              RESULTS/d5_llm_filtered_100_gbif/test_results.json \
                                              RESULTS/d5_llm_filtered_200_gbif/test_results.json \
                                              RESULTS/d5_llm_filtered_300_gbif/test_results.json

    # Save output
    python scripts/bootstrap_ci.py --results RESULTS/baseline_gbif/test_results.json \
                                   --output RESULTS/bootstrap_ci_baseline.json

    # With plot
    python scripts/bootstrap_ci.py --results RESULTS/baseline_gbif/test_results.json \
                                              RESULTS/d5_llm_filtered_200_gbif/test_results.json \
                                   --plot RESULTS/bootstrap_ci_comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import RESULTS_DIR


def _compute_f1_from_counts(tp: int, fp: int, fn: int) -> float:
    """Compute F1 from confusion matrix counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def bootstrap_per_species_f1(
    predictions: List[Dict],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Bootstrap CI for per-species F1 by resampling the full test set.

    Each iteration resamples all predictions with replacement, then computes
    F1 for each species from the resampled confusion matrix. This correctly
    accounts for the interaction between species (false positives from one
    species are false negatives for another).

    Args:
        predictions: List of dicts with 'ground_truth' and 'prediction'.
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence interval width (default 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Dict mapping species -> {observed, mean, std, ci_lower, ci_upper, support}.
    """
    rng = np.random.RandomState(seed)

    gt_arr = np.array([p["ground_truth"] for p in predictions])
    pr_arr = np.array([p["prediction"] for p in predictions])
    n = len(predictions)

    # Get all species
    species_list = sorted(set(gt_arr) | set(pr_arr))
    species_idx = {sp: i for i, sp in enumerate(species_list)}

    # Observed F1 per species
    observed_f1 = {}
    support = {}
    for sp in species_list:
        tp = int(np.sum((gt_arr == sp) & (pr_arr == sp)))
        fp = int(np.sum((gt_arr != sp) & (pr_arr == sp)))
        fn = int(np.sum((gt_arr == sp) & (pr_arr != sp)))
        observed_f1[sp] = _compute_f1_from_counts(tp, fp, fn)
        support[sp] = int(np.sum(gt_arr == sp))

    # Bootstrap
    f1_samples = {sp: np.zeros(n_bootstrap) for sp in species_list}
    macro_f1_samples = np.zeros(n_bootstrap)
    accuracy_samples = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        gt_b = gt_arr[idx]
        pr_b = pr_arr[idx]

        species_f1s = []
        for sp in species_list:
            tp = np.sum((gt_b == sp) & (pr_b == sp))
            fp = np.sum((gt_b != sp) & (pr_b == sp))
            fn = np.sum((gt_b == sp) & (pr_b != sp))
            f1 = _compute_f1_from_counts(int(tp), int(fp), int(fn))
            f1_samples[sp][b] = f1
            species_f1s.append(f1)

        macro_f1_samples[b] = np.mean(species_f1s)
        accuracy_samples[b] = np.mean(gt_b == pr_b)

    alpha = (1 - ci) / 2
    results = {}
    for sp in species_list:
        samples = f1_samples[sp]
        results[sp] = {
            "observed": observed_f1[sp],
            "mean": float(np.mean(samples)),
            "std": float(np.std(samples)),
            "ci_lower": float(np.percentile(samples, alpha * 100)),
            "ci_upper": float(np.percentile(samples, (1 - alpha) * 100)),
            "support": support[sp],
        }

    results["__macro_f1__"] = {
        "observed": float(np.mean(list(observed_f1.values()))),
        "mean": float(np.mean(macro_f1_samples)),
        "std": float(np.std(macro_f1_samples)),
        "ci_lower": float(np.percentile(macro_f1_samples, alpha * 100)),
        "ci_upper": float(np.percentile(macro_f1_samples, (1 - alpha) * 100)),
    }

    results["__accuracy__"] = {
        "observed": float(np.mean(gt_arr == pr_arr)),
        "mean": float(np.mean(accuracy_samples)),
        "std": float(np.std(accuracy_samples)),
        "ci_lower": float(np.percentile(accuracy_samples, alpha * 100)),
        "ci_upper": float(np.percentile(accuracy_samples, (1 - alpha) * 100)),
    }

    return results


def print_results(model_name: str, results: Dict[str, Dict]):
    """Pretty-print bootstrap CI results."""
    print(f"\n{'=' * 80}")
    print(f"Bootstrap 95% CI: {model_name}")
    print(f"{'=' * 80}")

    # Overall metrics
    acc = results["__accuracy__"]
    mf1 = results["__macro_f1__"]
    print(f"\n  Accuracy:  {acc['observed']:.4f}  [{acc['ci_lower']:.4f}, {acc['ci_upper']:.4f}]")
    print(f"  Macro F1:  {mf1['observed']:.4f}  [{mf1['ci_lower']:.4f}, {mf1['ci_upper']:.4f}]")

    # Per-species
    species = sorted(
        [(k, v) for k, v in results.items() if not k.startswith("__")],
        key=lambda x: x[1]["observed"],
    )
    print(f"\n  {'Species':<25} {'Support':>7} {'F1':>7} {'95% CI':>20} {'Width':>7}")
    print(f"  {'-' * 68}")
    for sp, stats in species:
        ci_str = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
        width = stats["ci_upper"] - stats["ci_lower"]
        print(f"  {sp:<25} {stats['support']:>7} {stats['observed']:>7.3f} {ci_str:>20} {width:>7.3f}")


def plot_comparison(
    all_results: Dict[str, Dict[str, Dict]],
    focus_species: List[str],
    output_path: Path,
):
    """Plot bootstrap CI comparison across models for focus species."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = list(all_results.keys())
    n_models = len(model_names)

    # Include macro_f1 as an extra "species"
    plot_species = focus_species + ["__macro_f1__"]
    plot_labels = [sp.replace("Bombus_", "B. ") for sp in focus_species] + ["Macro F1"]
    n_plots = len(plot_species)

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#F44336",
              "#795548", "#607D8B", "#E91E63"]

    for ax_idx, (sp, label) in enumerate(zip(plot_species, plot_labels)):
        ax = axes[ax_idx]
        y_positions = np.arange(n_models)

        for i, model_name in enumerate(model_names):
            stats = all_results[model_name].get(sp, {})
            if not stats:
                continue
            observed = stats["observed"]
            ci_lo = stats["ci_lower"]
            ci_hi = stats["ci_upper"]
            color = colors[i % len(colors)]

            ax.errorbar(
                observed, i,
                xerr=[[observed - ci_lo], [ci_hi - observed]],
                fmt="o", color=color, capsize=5, capthick=2,
                markersize=8, linewidth=2, label=model_name,
            )

        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_xlabel("F1 Score")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        ax.set_xlim(0, 1.05)

    plt.suptitle("Bootstrap 95% Confidence Intervals", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap confidence intervals for classifier evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--results", nargs="+", required=True,
                        help="Path(s) to test_results.json file(s)")
    parser.add_argument("--n-bootstrap", type=int, default=10000,
                        help="Number of bootstrap iterations (default: 10000)")
    parser.add_argument("--output", type=str,
                        help="Save JSON results to file")
    parser.add_argument("--plot", type=str,
                        help="Save comparison plot to file")
    parser.add_argument("--focus-species", nargs="+",
                        default=["Bombus_ashtoni", "Bombus_sandersoni"],
                        help="Species to highlight in plots")

    args = parser.parse_args()

    all_results = {}
    for results_path in args.results:
        path = Path(results_path)
        with open(path) as f:
            data = json.load(f)
        preds = data["detailed_predictions"]

        # Derive model name from directory
        model_name = path.parent.name.replace("_gbif", "")
        print(f"Running bootstrap ({args.n_bootstrap} iterations) on {model_name} "
              f"({len(preds)} predictions)...")

        results = bootstrap_per_species_f1(preds, n_bootstrap=args.n_bootstrap)
        all_results[model_name] = results
        print_results(model_name, results)

    # Print overlap analysis if multiple models
    if len(all_results) >= 2:
        model_names = list(all_results.keys())
        print(f"\n{'=' * 80}")
        print("CI OVERLAP ANALYSIS")
        print(f"{'=' * 80}")

        for sp in args.focus_species + ["__macro_f1__"]:
            label = sp.replace("Bombus_", "B. ") if not sp.startswith("__") else "Macro F1"
            print(f"\n  {label}:")
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    a = all_results[model_names[i]].get(sp, {})
                    b = all_results[model_names[j]].get(sp, {})
                    if not a or not b:
                        continue
                    overlap_lo = max(a["ci_lower"], b["ci_lower"])
                    overlap_hi = min(a["ci_upper"], b["ci_upper"])
                    overlaps = overlap_lo < overlap_hi
                    diff = b["observed"] - a["observed"]
                    sign = "+" if diff >= 0 else ""
                    status = "OVERLAPPING (not significant)" if overlaps else "NON-OVERLAPPING (significant)"
                    print(f"    {model_names[i]} vs {model_names[j]}: "
                          f"diff={sign}{diff:.3f}  {status}")

    if args.output:
        output = {name: {k: v for k, v in res.items()} for name, res in all_results.items()}
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nJSON saved: {args.output}")

    if args.plot:
        plot_comparison(all_results, args.focus_species, Path(args.plot))


if __name__ == "__main__":
    main()
