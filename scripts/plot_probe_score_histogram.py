#!/usr/bin/env python3
"""Generate per-species probe score histogram split by expert strict label.
Output: docs/plots/filters/probe_score_by_expert_label.png"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/home/msun14/bumblebee_bplusplus")
OUT = ROOT / "docs/plots/filters/probe_score_by_expert_label.png"

RARE = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]
SHORT = {"Bombus_ashtoni": "B. ashtoni",
         "Bombus_sandersoni": "B. sandersoni",
         "Bombus_flavidus": "B. flavidus"}
COLORS = {"Bombus_ashtoni": "#0072B2",
          "Bombus_sandersoni": "#E69F00",
          "Bombus_flavidus": "#009E73"}


def main():
    probe = pd.read_csv(ROOT / "RESULTS/filters/probe_scores.csv")
    expert = pd.read_csv(ROOT / "RESULTS/expert_validation_results/jessie_all_150.csv")
    expert["basename"] = expert["image_path"].str.split("/").str[-1]
    expert["ground_truth_species"] = expert["ground_truth_species"].str.replace(" ", "_", regex=False)
    expert["blind_id_species"] = expert["blind_id_species"].fillna("").str.replace(" ", "_", regex=False)
    morph_cols = ["morph_legs_appendages", "morph_wing_venation_texture",
                  "morph_head_antennae", "morph_abdomen_banding", "morph_thorax_coloration"]
    expert["expert_morph_mean"] = expert[morph_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    expert["expert_strict"] = (
        (expert["blind_id_species"] == expert["ground_truth_species"])
        & (expert["diagnostic_level"] == "species")
        & (expert["expert_morph_mean"] >= 4.0)
    )
    m = probe.merge(expert[["basename", "expert_strict"]], on="basename")

    meta = json.loads((ROOT / "RESULTS/filters/probe_scores.json").read_text())["meta"]
    tau = meta["per_species_threshold_strict"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
    bins = np.linspace(0, 1, 21)
    for ax, sp in zip(axes, RARE):
        sub = m[m.species == sp]
        pos = sub[sub.expert_strict].score
        neg = sub[~sub.expert_strict].score
        ax.hist(neg, bins=bins, color="#cccccc", edgecolor="white",
                label=f"expert fail (n={len(neg)})")
        ax.hist(pos, bins=bins, color=COLORS[sp], edgecolor="white",
                alpha=0.85, label=f"expert strict pass (n={len(pos)})")
        ax.axvline(tau[sp], color="#a94442", linestyle="--", linewidth=1.4,
                   label=f"τ = {tau[sp]:.3f}")
        ax.set_xlabel("Probe pass-probability")
        ax.set_title(SHORT[sp], color=COLORS[sp], fontweight="bold")
        ax.legend(frameon=False, fontsize=8.5, loc="upper center")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Image count (n = 50 per species)")
    fig.suptitle("Probe pass-probability distribution split by expert strict label",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    fig.savefig(Path(OUT).with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
