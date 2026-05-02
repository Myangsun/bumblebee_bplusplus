#!/usr/bin/env python3
"""Figure 5.20 — subtractive-ablation recovery for D3/D4/D5/D6 (f1 checkpoint).

Reads RESULTS/failure_analysis/subset_ablation_recovery_f1ckpt.csv (thesis
labels D3/D4/D5/D6) and renders two SEPARATE figures:

1. subset_ablation_recovery_bars.{png,pdf}
     Grouped bars of own-species recovery, x-axis = variant (D3/D4/D5/D6),
     within-group = the three rare species. Positive = harmful; negative =
     helpful; grey = |Δ| ≤ 0.02.

2. subset_ablation_recovery_heatmap.{png,pdf}
     12 × 3 recovery heatmap (rows = variant × dropped species, cols =
     measured), diverging RdBu colour scale.
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
DEFAULT_OUT_DIR = ROOT / "docs/plots/failure"

RARE_SHORT = ("ashtoni", "sandersoni", "flavidus")
VARIANTS = ("D3", "D4", "D5", "D6")
NEUTRAL = 0.02
# Okabe-Ito-compatible variant palette; D6 is the only expert-probe filter so
# it gets the distinctive rare-species blue.
VARIANT_COLOR = {"D3": "#7a7a7a",   # mid-grey
                 "D4": "#E69F00",   # orange (LLM)
                 "D5": "#009E73",   # green (centroid)
                 "D6": "#0072B2"}   # blue  (expert probe)
SPECIES_COLOR = {"ashtoni":    "#0072B2",
                 "sandersoni": "#E69F00",
                 "flavidus":   "#009E73"}


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


def _save(fig, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_bars(rows, out_path: Path):
    """Grouped own-species recovery bars, x = variant, within = species."""
    fig, ax_bars = plt.subplots(figsize=(9, 5.2), facecolor="white")
    x = np.arange(len(VARIANTS))
    width = 0.25
    offsets = np.linspace(-1, 1, len(RARE_SHORT)) * width

    for offset, sp in zip(offsets, RARE_SHORT):
        vals = []
        cols = []
        for variant in VARIANTS:
            m = next((r for r in rows
                      if r["variant"] == variant and r["dropped"] == sp
                      and r["measured"] == sp), None)
            if m is None:
                vals.append(0.0); cols.append("#eeeeee"); continue
            vals.append(m["recovery"])
            cols.append(colour(m["recovery"]))
        bars = ax_bars.bar(x + offset, vals, width, color=cols,
                           edgecolor="none", linewidth=0,
                           label=f"B. {sp}", alpha=0.94)
        for b, v in zip(bars, vals):
            ax_bars.text(b.get_x() + b.get_width() / 2,
                         v + (0.006 if v >= 0 else -0.012),
                         f"{v:+.3f}",
                         ha="center",
                         va="bottom" if v >= 0 else "top",
                         fontsize=7.5, rotation=90)
        # tiny species marker under each bar
        for b in bars:
            ax_bars.text(b.get_x() + b.get_width() / 2, -0.293,
                         f"B.{sp[:3]}", ha="center", va="top",
                         fontsize=6.8, color=SPECIES_COLOR[sp],
                         fontstyle="italic")

    ax_bars.axhline(0, color="black", linewidth=0.6)
    ax_bars.axhline(NEUTRAL, color="gray", linestyle=":", linewidth=0.7)
    ax_bars.axhline(-NEUTRAL, color="gray", linestyle=":", linewidth=0.7)
    ax_bars.set_xticks(x)
    ax_bars.set_xticklabels(VARIANTS, fontsize=11, color="black")
    ax_bars.set_ylabel("F1 recovery on dropped species\n"
                       "(positive = synthetics harmful; negative = helpful)")
    ax_bars.set_title("Own-species F1 recovery under single-species drop (seed 42, f1 checkpoint)",
                      fontsize=11, loc="left")
    ax_bars.set_ylim(-0.32, 0.26)
    ax_bars.spines["top"].set_visible(False)
    ax_bars.spines["right"].set_visible(False)
    ax_bars.grid(axis="y", linestyle=":", alpha=0.3)

    fill_legend = [
        Patch(facecolor="#d62728", label="harmful (Δ > +0.02)"),
        Patch(facecolor="#2ca02c", label="helpful (Δ < −0.02)"),
        Patch(facecolor="#9a9a9a", label="neutral (|Δ| ≤ 0.02)"),
    ]
    ax_bars.legend(handles=fill_legend, fontsize=8.5, frameon=False,
                   loc="upper left", bbox_to_anchor=(1.01, 1.0),
                   title="bar fill")

    _save(fig, out_path)


def plot_heatmap(rows, out_path: Path):
    """12 × 3 recovery heatmap (rows = variant × dropped, cols = measured)."""
    fig, ax_hm = plt.subplots(figsize=(7.5, 8), facecolor="white")
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
    ax_hm.set_yticklabels(ylabels, fontsize=9)
    for tick, lbl in zip(ax_hm.get_yticklabels(), ylabels):
        variant = lbl.split()[0]
        tick.set_color(VARIANT_COLOR[variant])
    ax_hm.set_xticks(range(len(RARE_SHORT)))
    ax_hm.set_xticklabels([f"measured B. {sp}" for sp in RARE_SHORT],
                          rotation=25, ha="right", fontsize=10)
    ax_hm.set_title("3 × 3 recovery matrix per variant\n"
                    "(diagonal = own, off-diagonal = collateral)",
                    fontsize=11, loc="left")
    for v in range(1, len(VARIANTS)):
        ax_hm.axhline(v * 3 - 0.5, color="black", linewidth=0.9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            tc = "white" if abs(v) > 0.55 * lim else "black"
            ax_hm.text(j, i, f"{v:+.3f}", ha="center", va="center",
                       fontsize=8, color=tc)
    cb = fig.colorbar(im, ax=ax_hm, fraction=0.045, pad=0.03)
    cb.set_label("F1 recovery")
    _save(fig, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    rows = load(args.csv)
    plot_bars(rows, args.output_dir / "subset_ablation_recovery_bars.png")
    plot_heatmap(rows, args.output_dir / "subset_ablation_recovery_heatmap.png")


if __name__ == "__main__":
    main()
