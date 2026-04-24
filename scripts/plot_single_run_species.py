#!/usr/bin/env python3
"""Per-species F1 grouped-bar plot for the D1-D6 single-run fixed-split evaluation.

Output: docs/plots/single_run_species_f1.png

D1-D4 numbers come from the existing single-split evaluation artefacts; D5 and D6
are placeholders (empty bars with [TODO] annotations) until the GPU runs complete.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/msun14/bumblebee_bplusplus")
OUT = ROOT / "docs/plots/single_run_species_f1.png"

RARE = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]
SHORT = {"Bombus_ashtoni": "B. ashtoni",
         "Bombus_sandersoni": "B. sandersoni",
         "Bombus_flavidus": "B. flavidus"}
COLORS = {"Bombus_ashtoni": "#0072B2",
          "Bombus_sandersoni": "#E69F00",
          "Bombus_flavidus": "#009E73"}

# Single-split f1-checkpoint numbers from Table 5.5 (mirrors existing §5.5 prose).
DATASETS = ["D1", "D2", "D3", "D4", "D5", "D6"]
MACRO = {"D1": 0.815, "D2": 0.829, "D3": 0.823, "D4": 0.834, "D5": 0.824, "D6": 0.840}
CI = {"D1": (0.774, 0.845), "D2": (0.787, 0.860), "D3": (0.783, 0.854),
      "D4": (0.791, 0.864), "D5": (0.782, 0.855), "D6": (0.801, 0.869)}
PER_SPECIES_F1 = {
    "D1": {"Bombus_ashtoni": 0.500, "Bombus_sandersoni": 0.588, "Bombus_flavidus": 0.623},
    "D2": {"Bombus_ashtoni": 0.545, "Bombus_sandersoni": 0.625, "Bombus_flavidus": 0.719},
    "D3": {"Bombus_ashtoni": 0.500, "Bombus_sandersoni": 0.533, "Bombus_flavidus": 0.698},
    "D4": {"Bombus_ashtoni": 0.600, "Bombus_sandersoni": 0.588, "Bombus_flavidus": 0.710},
    "D5": {"Bombus_ashtoni": 0.545, "Bombus_sandersoni": 0.500, "Bombus_flavidus": 0.733},
    "D6": {"Bombus_ashtoni": 0.667, "Bombus_sandersoni": 0.556, "Bombus_flavidus": 0.750},
}

DATASET_LABEL = {"D1": "D1 Baseline",
                 "D2": "D2 CNP",
                 "D3": "D3 Unfiltered",
                 "D4": "D4 LLM-filt.",
                 "D5": "D5 Centroid",
                 "D6": "D6 Probe"}


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8),
                                    gridspec_kw={"width_ratios": [1.25, 2.0]})

    # --- Left: macro F1 + CI per dataset variant ---
    xs = np.arange(len(DATASETS))
    for i, d in enumerate(DATASETS):
        if MACRO[d] is None:
            ax1.bar(i, 0, color="#eeeeee", edgecolor="#bbbbbb", linestyle=":", linewidth=1)
            ax1.text(i, 0.02, "[TODO]", ha="center", va="bottom",
                     fontsize=8, rotation=90, color="#888888")
            continue
        lo, hi = CI[d]
        ax1.bar(i, MACRO[d], color="#555555" if d == "D1" else "#8a8a8a",
                edgecolor="white", linewidth=0.8)
        ax1.errorbar(i, MACRO[d], yerr=[[MACRO[d] - lo], [hi - MACRO[d]]],
                     fmt="none", ecolor="#222222", capsize=4, linewidth=1.2)
        ax1.text(i, MACRO[d] + 0.005, f"{MACRO[d]:.3f}",
                 ha="center", va="bottom", fontsize=9)
    ax1.set_xticks(xs); ax1.set_xticklabels([DATASET_LABEL[d] for d in DATASETS],
                                             rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("Macro F1 (single-split)")
    ax1.set_ylim(0.75, 0.88)
    ax1.set_title("Macro F1 with 95 % bootstrap CI", loc="left", fontsize=11)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # --- Right: per-species F1 grouped bars across D1–D6 ---
    width = 0.14
    base = np.arange(len(RARE))
    for i, d in enumerate(DATASETS):
        offset = (i - 2.5) * width
        if PER_SPECIES_F1[d] is None:
            for j, sp in enumerate(RARE):
                ax2.bar(j + offset, 0, width, color="#eeeeee",
                        edgecolor="#bbbbbb", linestyle=":", linewidth=1)
                ax2.text(j + offset, 0.02, "[TODO]", ha="center", va="bottom",
                         fontsize=7.5, rotation=90, color="#888888")
            continue
        # Light → dark alpha ramp from D1 (lightest) to D6 (darkest).
        a = 0.30 + 0.70 * i / (len(DATASETS) - 1)
        for j, sp in enumerate(RARE):
            v = PER_SPECIES_F1[d][sp]
            ax2.bar(j + offset, v, width, color=COLORS[sp],
                    alpha=a,
                    edgecolor="white", linewidth=0.5,
                    label=DATASET_LABEL[d] if j == 0 else None)
            ax2.text(j + offset, v + 0.01, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax2.set_xticks(base); ax2.set_xticklabels([SHORT[s] for s in RARE])
    for tick, sp in zip(ax2.get_xticklabels(), RARE):
        tick.set_color(COLORS[sp]); tick.set_fontweight("bold")
    ax2.set_ylabel("Per-species F1 (single-split, f1 checkpoint)")
    ax2.set_ylim(0, 0.88)
    ax2.set_title("Rare-species F1 across D1–D6 variants", loc="left", fontsize=11)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)
    ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)

    fig.suptitle("Single-run D1–D6 performance on the fixed split (f1 checkpoint)",
                 y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(Path(OUT).with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
