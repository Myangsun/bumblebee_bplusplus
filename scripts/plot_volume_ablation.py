#!/usr/bin/env python3
"""
Plot volume ablation trends: performance vs. synthetic augmentation volume.

Generates a 2x2 grid:
  - Overall Macro F1 vs volume
  - Overall Accuracy vs volume
  - B. ashtoni F1 vs volume
  - B. sandersoni F1 vs volume

Usage:
    python scripts/plot_volume_ablation.py
    python scripts/plot_volume_ablation.py --ci RESULTS/bootstrap_ci_full_ablation.json
    python scripts/plot_volume_ablation.py --output RESULTS/my_plot.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pipeline.config import RESULTS_DIR


ABLATION_FILES = {
    "baseline": ("baseline_gbif/test_results.json", 0),
    "d4_50": ("d4_synthetic_50_gbif/test_results.json", 50),
    "d4_100": ("d4_synthetic_100_gbif/test_results.json", 100),
    "d4_200": ("d4_synthetic_200_gbif/test_results.json", 200),
    "d4_300": ("d4_synthetic_300_gbif/test_results.json", 300),
    "d5_50": ("d5_llm_filtered_50_gbif/test_results.json", 50),
    "d5_100": ("d5_llm_filtered_100_gbif/test_results.json", 100),
    "d5_200": ("d5_llm_filtered_200_gbif/test_results.json", 200),
    "d5_300": ("d5_llm_filtered_300_gbif/test_results.json", 300),
}

# Maps ablation key -> bootstrap CI JSON key
CI_KEY_MAP = {
    "baseline": "baseline",
    "d4_50": "d4_synthetic_50",
    "d4_100": "d4_synthetic_100",
    "d4_200": "d4_synthetic_200",
    "d4_300": "d4_synthetic_300",
    "d5_50": "d5_llm_filtered_50",
    "d5_100": "d5_llm_filtered_100",
    "d5_200": "d5_llm_filtered_200",
    "d5_300": "d5_llm_filtered_300",
}

FOCUS_SPECIES = ["Bombus_ashtoni", "Bombus_sandersoni"]


def load_ablation_data(results_dir: Path) -> dict:
    data = {}
    for name, (rel_path, vol) in ABLATION_FILES.items():
        path = results_dir / rel_path
        if not path.exists():
            print(f"WARNING: {path} not found, skipping {name}")
            continue
        with open(path) as f:
            d = json.load(f)
        sm = d["species_metrics"]
        f1_vals = [sm[sp]["f1"] for sp in sm]
        supports = [sm[sp]["support"] for sp in sm]
        macro_f1 = np.mean(f1_vals)
        weighted_f1 = np.average(f1_vals, weights=supports) if sum(supports) > 0 else 0
        data[name] = {
            "vol": vol,
            "acc": d["overall_accuracy"],
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        }
        for sp in FOCUS_SPECIES:
            key = sp.split("_")[1] + "_f1"  # e.g. "ashtoni_f1"
            data[name][key] = sm.get(sp, {}).get("f1", 0)
            data[name][key.replace("_f1", "_support")] = sm.get(sp, {}).get("support", 0)
    return data


def _plot_series(ax, vols, keys, data, metric, color, label, marker, ci_data=None, ci_metric=None):
    """Plot a single series (D4 or D5) with optional CI band."""
    vals = [data[k][metric] for k in keys]
    ax.plot(vols, vals, f"{marker}-", color=color, label=label, linewidth=2, markersize=7)

    if ci_data and ci_metric:
        lo, hi = [], []
        for k in keys:
            ci_key = CI_KEY_MAP.get(k, k)
            if ci_key in ci_data and ci_metric in ci_data[ci_key]:
                lo.append(ci_data[ci_key][ci_metric]["ci_lower"])
                hi.append(ci_data[ci_key][ci_metric]["ci_upper"])
            else:
                lo.append(vals[len(lo)])
                hi.append(vals[len(hi) - 1])
        ax.fill_between(vols, lo, hi, color=color, alpha=0.15)


def plot_trends(data: dict, output_path: Path, ci_data: dict | None = None):
    d4_keys = [k for k in ["baseline", "d4_50", "d4_100", "d4_200", "d4_300"] if k in data]
    d5_keys = [k for k in ["baseline", "d5_50", "d5_100", "d5_200", "d5_300"] if k in data]
    d4_vols = [data[k]["vol"] for k in d4_keys]
    d5_vols = [data[k]["vol"] for k in d5_keys]
    vols = [0, 50, 100, 200, 300]

    baseline = data.get("baseline", {})

    # Metric configs: (row, col, data_key, ci_metric_key, title)
    plots = [
        (0, 0, "macro_f1", "__macro_f1__", "Macro F1"),
        (0, 1, "acc", "__accuracy__", "Overall Accuracy"),
        (1, 0, "ashtoni_f1", "Bombus_ashtoni",
         f"B. ashtoni F1 (n={baseline.get('ashtoni_support', '?')} test)"),
        (1, 1, "sandersoni_f1", "Bombus_sandersoni",
         f"B. sandersoni F1 (n={baseline.get('sandersoni_support', '?')} test)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for row, col, metric, ci_metric, title in plots:
        ax = axes[row, col]

        _plot_series(ax, d4_vols, d4_keys, data, metric, "#2196F3",
                     "D4 (unfiltered)", "o", ci_data, ci_metric)
        _plot_series(ax, d5_vols, d5_keys, data, metric, "#FF9800",
                     "D5 (LLM-filtered)", "s", ci_data, ci_metric)

        if baseline and metric in baseline:
            ax.axhline(y=baseline[metric], color="gray", linestyle="--",
                       alpha=0.5, label="Baseline")
            # Baseline CI band
            if ci_data and "baseline" in ci_data and ci_metric in ci_data["baseline"]:
                bl_ci = ci_data["baseline"][ci_metric]
                ax.axhspan(bl_ci["ci_lower"], bl_ci["ci_upper"],
                           color="gray", alpha=0.1)

        ax.set_xlabel("Synthetic images per species")
        ax.set_ylabel("Score")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.set_xticks(vols)
        ax.grid(True, alpha=0.3)
        if row == 1:
            ax.set_ylim(0.0, 1.05)

    suptitle = "Volume Ablation with Bootstrap 95% CI" if ci_data else \
               "Volume Ablation: Synthetic Data Augmentation Impact"
    plt.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot volume ablation trends")
    parser.add_argument("--output", type=str,
                        default=str(RESULTS_DIR / "volume_ablation_trends.png"),
                        help="Output image path")
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR),
                        help="Results directory")
    parser.add_argument("--ci", type=str, default=None,
                        help="Path to bootstrap CI JSON (from bootstrap_ci.py --output) "
                             "to add 95%% CI shaded bands")
    args = parser.parse_args()

    data = load_ablation_data(Path(args.results_dir))
    if not data:
        print("No ablation data found.")
        sys.exit(1)

    ci_data = None
    if args.ci:
        with open(args.ci) as f:
            ci_data = json.load(f)
        print(f"Loaded bootstrap CIs from {args.ci}")

    plot_trends(data, Path(args.output), ci_data)


if __name__ == "__main__":
    main()
