#!/usr/bin/env python3
"""
Task 1 T1.12 visualisation — subset-ablation recovery bar chart.

Reads `RESULTS/failure_analysis/subset_ablation_recovery.csv` and renders:

1. **Per-variant recovery bars.** For each rare species dropped, bar of
   F1 recovery on the target species (own recovery). Positive = that
   species' synthetics were collectively harmful; negative = they were
   collectively helpful. Bars coloured green/red by sign, grey if within
   the ±0.02 neutral band.

2. **Full recovery heatmap.** 3 × 3 matrix per variant (rows = dropped
   species, cols = measured species). Diagonal shows own-species recovery;
   off-diagonal shows cross-species collateral. Diverging colour scale.

Output:
    docs/plots/failure/subset_ablation_recovery.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from pipeline.config import PROJECT_ROOT, RESULTS_DIR

RARE_SHORT = ("ashtoni", "sandersoni", "flavidus")
VARIANTS = ("D4", "D5")
DEFAULT_CSV = RESULTS_DIR / "failure_analysis" / "subset_ablation_recovery.csv"
DEFAULT_OUT = PROJECT_ROOT / "docs" / "plots" / "failure" / "subset_ablation_recovery.png"
NEUTRAL = 0.02


def _load(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "variant": r["variant"],
                "dropped": r["dropped"].replace("Bombus_", ""),
                "measured_species": r["measured_species"].replace("Bombus_", ""),
                "f1_full": float(r["f1_full"]),
                "f1_ablated": float(r["f1_ablated"]),
                "recovery": float(r["recovery"]),
                "target_species": r["target_species"].lower() == "true",
            })
    return rows


def _matrix(rows: List[dict], variant: str) -> np.ndarray:
    M = np.zeros((len(RARE_SHORT), len(RARE_SHORT)), dtype=float)
    for r in rows:
        if r["variant"] != variant:
            continue
        i = RARE_SHORT.index(r["dropped"])
        j = RARE_SHORT.index(r["measured_species"])
        M[i, j] = r["recovery"]
    return M


def _colour(recovery: float) -> str:
    if recovery > NEUTRAL:
        return "#d62728"   # red = dropping recovered F1 = synthetics were harmful
    if recovery < -NEUTRAL:
        return "#2ca02c"   # green = dropping hurt F1 = synthetics were helpful
    return "#9a9a9a"        # grey = neutral


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = _load(args.csv)

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)

    # Panel 1: bar chart of own-species recovery, D4 and D5 side-by-side.
    ax_bars = fig.add_subplot(gs[0, 0])
    x = np.arange(len(RARE_SHORT))
    width = 0.36
    for offset, variant in zip((-width / 2, width / 2), VARIANTS):
        vals = []
        cols = []
        for sp in RARE_SHORT:
            match = next(r for r in rows if r["variant"] == variant
                         and r["dropped"] == sp and r["measured_species"] == sp)
            vals.append(match["recovery"])
            cols.append(_colour(match["recovery"]))
        bars = ax_bars.bar(x + offset, vals, width, color=cols,
                            edgecolor="black", linewidth=0.6,
                            label=variant, alpha=0.9)
        for b, v in zip(bars, vals):
            ax_bars.text(b.get_x() + b.get_width() / 2,
                         v + (0.005 if v >= 0 else -0.013),
                         f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                         fontsize=8)
    ax_bars.axhline(0, color="black", linewidth=0.5)
    ax_bars.axhline(NEUTRAL, color="gray", linestyle=":", linewidth=0.7)
    ax_bars.axhline(-NEUTRAL, color="gray", linestyle=":", linewidth=0.7)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([f"B. {sp}" for sp in RARE_SHORT])
    ax_bars.set_ylabel("F1 recovery when species' synthetics are dropped")
    ax_bars.set_title("Own-species F1 recovery (seed 42)\n"
                     "positive = synthetics harmful, negative = helpful",
                     fontsize=10)
    # Custom legend with variant offsets
    from matplotlib.patches import Patch
    legend_items = [
        Patch(facecolor="#d62728", edgecolor="black", label="harmful (>+0.02)"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="helpful (<−0.02)"),
        Patch(facecolor="#9a9a9a", edgecolor="black", label="neutral (|·|≤0.02)"),
    ]
    ax_bars.legend(handles=legend_items, fontsize=8, frameon=False, loc="best")
    ax_bars.grid(axis="y", linestyle=":", alpha=0.35)

    # Panel 2: 3x3 heatmap per variant (stacked vertically within this axis).
    ax_hm = fig.add_subplot(gs[0, 1])
    both = np.vstack([_matrix(rows, "D4"), _matrix(rows, "D5")])
    lim = float(np.abs(both).max())
    im = ax_hm.imshow(both, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    yticks = [f"D4 drop {sp}" for sp in RARE_SHORT] + [f"D5 drop {sp}" for sp in RARE_SHORT]
    ax_hm.set_yticks(range(len(yticks)))
    ax_hm.set_yticklabels(yticks, fontsize=9)
    ax_hm.set_xticks(range(len(RARE_SHORT)))
    ax_hm.set_xticklabels([f"F1 {sp}" for sp in RARE_SHORT], rotation=25, ha="right")
    ax_hm.set_title("Recovery matrix\n(diagonal = own, off-diagonal = collateral)",
                    fontsize=10)
    ax_hm.axhline(2.5, color="black", linewidth=1.0)
    for i in range(len(yticks)):
        for j in range(len(RARE_SHORT)):
            v = both[i, j]
            tc = "white" if abs(v) > 0.55 * lim else "black"
            ax_hm.text(j, i, f"{v:+.3f}", ha="center", va="center",
                        fontsize=8, color=tc)

    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("F1 recovery")

    fig.suptitle("Subset ablation — causal attribution of rare-species synthetic harm",
                 fontsize=12, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
