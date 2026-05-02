#!/usr/bin/env python3
"""Additive single-species ablation bar plot.

For each "kept species" k ∈ {ashtoni, sandersoni, flavidus}, train on the
D1 baseline plus only k's synthetics (other two rare species unaugmented).
This is the additive counterpart to the subtractive ablation in
plot_subset_ablation.py.

The bar plot shows Δ from D1 baseline for four metrics — Macro F1, B.
ashtoni F1, B. sandersoni F1, B. flavidus F1 — grouped on the x-axis by
kept species. Bar colour encodes harmful (red, Δ < −0.02) / helpful
(green, Δ > +0.02) / neutral (grey).

Output: docs/plots/failure/additive_ablation_bars.{png,pdf}
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT  # noqa: E402

ROOT = PROJECT_ROOT
OUT = ROOT / "docs/plots/failure"

KEPT = ("ashtoni", "sandersoni", "flavidus")
RARE_FULL = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")
METRIC_LABELS = ("Macro F1", "B. ash", "B. san", "B. fla")
NEUTRAL = 0.02


def _latest(pattern: str) -> Path:
    matches = sorted(glob.glob(str(ROOT / pattern)))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def _row(record: dict) -> tuple[float, float, float, float]:
    sm = record["species_metrics"]
    return (
        float(record["macro_f1"]),
        float(sm["Bombus_ashtoni"]["f1"]),
        float(sm["Bombus_sandersoni"]["f1"]),
        float(sm["Bombus_flavidus"]["f1"]),
    )


def colour(delta: float) -> str:
    if delta > NEUTRAL:
        return "#2ca02c"   # green = helpful (additive synthetics improved)
    if delta < -NEUTRAL:
        return "#d62728"   # red   = harmful
    return "#9a9a9a"        # grey  = neutral


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=OUT / "additive_ablation_bars.png")
    args = parser.parse_args()

    base = json.load(open(_latest("RESULTS_seeds/baseline_seed42@f1_seed_test_results_*.json")))
    base_row = _row(base)
    deltas = {}
    for k in KEPT:
        rec = json.load(open(_latest(f"RESULTS/d4_synthetic_seed42_only-{k}@f1_seed_test_results_*.json")))
        deltas[k] = tuple(a - b for a, b in zip(_row(rec), base_row))

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor="white")
    x = np.arange(len(KEPT))
    n_metrics = len(METRIC_LABELS)
    width = 0.20
    offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2) * width

    for mi, mlabel in enumerate(METRIC_LABELS):
        vals = [deltas[k][mi] for k in KEPT]
        cols = [colour(v) for v in vals]
        bars = ax.bar(x + offsets[mi], vals, width, color=cols,
                      edgecolor="none", alpha=0.94)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.006 if v >= 0 else -0.012),
                    f"{v:+.3f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=7.5, rotation=90)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, -0.293,
                    mlabel, ha="center", va="top",
                    fontsize=7, fontstyle="italic", color="#444")

    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(NEUTRAL, color="gray", linestyle=":", linewidth=0.7)
    ax.axhline(-NEUTRAL, color="gray", linestyle=":", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"only-{k}" for k in KEPT], fontsize=11, color="black")
    ax.set_ylabel("ΔF1 vs D1 baseline\n"
                  "(positive = additive synthetics helpful; negative = harmful)")
    ax.set_title("Additive single-species ablation — D1 + only-{species} synthetics "
                 "(seed 42, f1 checkpoint)",
                 fontsize=11, loc="left")
    ax.set_ylim(-0.32, 0.26)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    legend = [
        Patch(facecolor="#2ca02c", label="helpful (Δ > +0.02)"),
        Patch(facecolor="#d62728", label="harmful (Δ < −0.02)"),
        Patch(facecolor="#9a9a9a", label="neutral (|Δ| ≤ 0.02)"),
    ]
    ax.legend(handles=legend, fontsize=8.5, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1.0),
              title="bar fill")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")
    print(f"Saved {args.output.with_suffix('.pdf')}")
    # Echo numbers for sanity.
    print("\nDelta vs D1 baseline (Macro / B.ash / B.san / B.fla):")
    for k in KEPT:
        d = deltas[k]
        print(f"  only-{k}: {d[0]:+.3f}  {d[1]:+.3f}  {d[2]:+.3f}  {d[3]:+.3f}")


if __name__ == "__main__":
    main()
