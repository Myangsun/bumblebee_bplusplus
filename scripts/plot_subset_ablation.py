#!/usr/bin/env python3
"""Figure 5.20 — subtractive-ablation recovery for D3/D4/D5/D6 (f1 checkpoint).

Reads RESULTS/failure_analysis/subset_ablation_recovery_f1ckpt.csv (thesis
labels D3/D4/D5/D6) and renders two panels:

1. Grouped bars of own-species recovery: 3 rare species × 4 variants (D3-D6).
   Positive = synthetics were harmful; negative = helpful; grey = |Δ| ≤ 0.02.
2. 12 × 3 recovery heatmap (rows = variant × dropped species, cols = measured),
   diverging RdBu colour scale.

Output: docs/plots/failure/subset_ablation_recovery.png
"""
from __future__ import annotations

import argparse, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path("/home/msun14/bumblebee_bplusplus")
DEFAULT_CSV = ROOT / "RESULTS/failure_analysis/subset_ablation_recovery_f1ckpt.csv"
DEFAULT_OUT = ROOT / "docs/plots/failure/subset_ablation_recovery.png"

RARE_SHORT = ("ashtoni", "sandersoni", "flavidus")
VARIANTS = ("D3", "D4", "D5", "D6")
NEUTRAL = 0.02
# Okabe-Ito-compatible variant palette; D6 is the only expert-probe filter so
# it gets the distinctive rare-species blue.
VARIANT_COLOR = {"D3": "#7a7a7a",   # mid-grey
                 "D4": "#E69F00",   # orange (LLM)
                 "D5": "#009E73",   # green (centroid)
                 "D6": "#0072B2"}   # blue  (expert probe)


def load(csv_path):
    rows = []
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "variant": r["variant"],
                "dropped": r["dropped"].replace("Bombus_", ""),
                "measured": r["measured_species"].replace("Bombus_", ""),
                "recovery": float(r["recovery"]),
                "f1_full": float(r["f1_full"]),
                "f1_ablated": float(r["f1_ablated"]),
            })
    return rows


def colour(rec):
    if rec > NEUTRAL: return "#d62728"    # red, harmful
    if rec < -NEUTRAL: return "#2ca02c"   # green, helpful
    return "#9a9a9a"                       # grey, neutral


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = load(args.csv)

    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.3, 1.25, 0.05], wspace=0.25)

    # ── Panel 1: grouped own-species recovery bars ──────────────────────────
    ax_bars = fig.add_subplot(gs[0, 0])
    x = np.arange(len(RARE_SHORT))
    width = 0.20
    offsets = np.linspace(-1.5, 1.5, len(VARIANTS)) * width

    for offset, variant in zip(offsets, VARIANTS):
        vals = []
        cols = []
        for sp in RARE_SHORT:
            m = next((r for r in rows
                      if r["variant"] == variant and r["dropped"] == sp
                      and r["measured"] == sp), None)
            if m is None: vals.append(0.0); cols.append("#eeeeee"); continue
            vals.append(m["recovery"])
            cols.append(colour(m["recovery"]))
        bars = ax_bars.bar(x + offset, vals, width, color=cols,
                           edgecolor="black", linewidth=0.6,
                           label=variant, alpha=0.92)
        # variant label above the bar
        for b, v in zip(bars, vals):
            ax_bars.text(b.get_x() + b.get_width() / 2,
                         v + (0.006 if v >= 0 else -0.012),
                         f"{v:+.3f}",
                         ha="center",
                         va="bottom" if v >= 0 else "top",
                         fontsize=7, rotation=90)
        # variant label under x-axis per group-group
        for xi, b in zip(x, bars):
            ax_bars.text(b.get_x() + b.get_width() / 2, -0.27,
                         variant, ha="center", va="top",
                         fontsize=7.5, color=VARIANT_COLOR[variant],
                         fontweight="bold")

    ax_bars.axhline(0, color="black", linewidth=0.5)
    ax_bars.axhline(NEUTRAL, color="gray", linestyle=":", linewidth=0.7)
    ax_bars.axhline(-NEUTRAL, color="gray", linestyle=":", linewidth=0.7)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels([f"B. {sp}" for sp in RARE_SHORT], fontstyle="italic", fontsize=9)
    ax_bars.set_ylabel("F1 recovery on dropped species\n"
                       "(positive = synthetics harmful; negative = helpful)")
    ax_bars.set_title("Own-species F1 recovery under single-species drop (seed 42)",
                      fontsize=10, loc="left")
    ax_bars.set_ylim(-0.3, 0.26)
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)

    legend_items = [
        Patch(facecolor="#d62728", edgecolor="black", label="harmful (Δ > +0.02)"),
        Patch(facecolor="#2ca02c", edgecolor="black", label="helpful (Δ < −0.02)"),
        Patch(facecolor="#9a9a9a", edgecolor="black", label="neutral (|Δ| ≤ 0.02)"),
    ]
    ax_bars.legend(handles=legend_items, fontsize=8, frameon=False, loc="upper right")
    ax_bars.grid(axis="y", linestyle=":", alpha=0.3)

    # ── Panel 2: recovery heatmap (12 rows × 3 cols) ────────────────────────
    ax_hm = fig.add_subplot(gs[0, 1])
    M = np.zeros((len(VARIANTS) * len(RARE_SHORT), len(RARE_SHORT)))
    ylabels = []
    for vi, variant in enumerate(VARIANTS):
        for si, dropped in enumerate(RARE_SHORT):
            row = vi * len(RARE_SHORT) + si
            ylabels.append(f"{variant} drop {dropped}")
            for mi, measured in enumerate(RARE_SHORT):
                m = next((r for r in rows
                          if r["variant"] == variant and r["dropped"] == dropped
                          and r["measured"] == measured), None)
                if m is not None:
                    M[row, mi] = m["recovery"]

    lim = float(np.abs(M).max())
    im = ax_hm.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax_hm.set_yticks(range(len(ylabels)))
    ax_hm.set_yticklabels(ylabels, fontsize=8.5)
    # Colour the variant prefix in each ytick label
    for tick, lbl in zip(ax_hm.get_yticklabels(), ylabels):
        variant = lbl.split()[0]
        tick.set_color(VARIANT_COLOR[variant])
    ax_hm.set_xticks(range(len(RARE_SHORT)))
    ax_hm.set_xticklabels([f"measured B. {sp}" for sp in RARE_SHORT],
                          rotation=25, ha="right", fontsize=9)
    ax_hm.set_title("3 × 3 recovery matrix per variant\n(diagonal = own, off-diagonal = collateral)",
                    fontsize=10, loc="left")
    for v in range(1, len(VARIANTS)):
        ax_hm.axhline(v * 3 - 0.5, color="black", linewidth=0.9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            tc = "white" if abs(v) > 0.55 * lim else "black"
            ax_hm.text(j, i, f"{v:+.3f}", ha="center", va="center",
                       fontsize=7.5, color=tc)

    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("F1 recovery")

    fig.suptitle("Subtractive ablation — causal attribution of rare-species synthetic harm "
                 "across D3 / D4 / D5 / D6 (f1 checkpoint, seed 42)",
                 fontsize=11, y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
