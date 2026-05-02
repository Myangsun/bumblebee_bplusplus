#!/usr/bin/env python3
"""Additive ablation as a directed kept-→-measured effect matrix.

A 3×3 RdBu heatmap showing ΔF1 (vs D1 baseline) when training on the
baseline plus the synthetics of exactly one rare species. Rows = kept
species, columns = measured species. A side panel shows Δ macro F1 per
kept-species variant.

The combined view exposes that the synthetic-real gap is species-specific:
the off-diagonal cells are asymmetric (sandersoni → ashtoni transfer is
strongly positive, while flavidus → sandersoni is strongly negative),
and the diagonal is non-uniform (only-sandersoni hurts its own species,
only-flavidus mildly helps its own).

Output: docs/plots/failure/additive_ablation_matrix.{png,pdf}
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT  # noqa: E402

ROOT = PROJECT_ROOT
OUT = ROOT / "docs/plots/failure"
KEPT = ("ashtoni", "sandersoni", "flavidus")
RARE_FULL = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")


def _latest(pattern: str) -> Path:
    matches = sorted(glob.glob(str(ROOT / pattern)))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def _per_species_f1(record: dict) -> tuple[float, float, float]:
    sm = record["species_metrics"]
    return tuple(float(sm[s]["f1"]) for s in RARE_FULL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=OUT / "additive_ablation_matrix.png")
    args = parser.parse_args()

    base = json.load(open(_latest("RESULTS_seeds/baseline_seed42@f1_seed_test_results_*.json")))
    base_per_sp = _per_species_f1(base)
    base_macro = float(base["macro_f1"])

    abs_F1 = np.zeros((3, 3))
    delta_F1 = np.zeros((3, 3))
    delta_macro = np.zeros(3)
    abs_macro = np.zeros(3)
    for i, k in enumerate(KEPT):
        rec = json.load(open(_latest(f"RESULTS/d4_synthetic_seed42_only-{k}@f1_seed_test_results_*.json")))
        per_sp = _per_species_f1(rec)
        abs_F1[i] = per_sp
        delta_F1[i] = tuple(a - b for a, b in zip(per_sp, base_per_sp))
        abs_macro[i] = float(rec["macro_f1"])
        delta_macro[i] = abs_macro[i] - base_macro

    # Lock row order to match column order: ashtoni / sandersoni / flavidus.
    KEPT_ord = list(KEPT)

    # 4-column matrix: 3 rare species + macro F1.
    M = np.zeros((3, 4))
    A = np.zeros((3, 4))
    M[:, :3] = delta_F1
    A[:, :3] = abs_F1
    M[:, 3] = delta_macro
    A[:, 3] = abs_macro

    fig, ax_hm = plt.subplots(figsize=(8.5, 4.6), facecolor="white")

    # Diverging colormap: green (high / positive Δ = improvement) → red (low / negative Δ = harm).
    lim = float(np.abs(M).max())
    im = ax_hm.imshow(M, cmap="RdYlGn", vmin=-lim, vmax=lim, aspect="auto")

    col_labels = [f"B. {s.replace('Bombus_', '')}" for s in RARE_FULL] + ["Macro F1"]
    ax_hm.set_xticks(range(4))
    ax_hm.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=10)
    ax_hm.set_yticks(range(3))
    ax_hm.set_yticklabels([f"D1 + only-{k}" for k in KEPT_ord], fontsize=10)
    ax_hm.set_title("Additive single-species ablation — ΔF1 vs D1 baseline  "
                    "(seed 42, f1 checkpoint)\n"
                    "Green = improve, red = harm. Final column = macro F1; "
                    f"D1 baseline macro F1 = {base_macro:.3f}.",
                    fontsize=10.5, loc="left")
    # Visually separate the macro column from the per-species block.
    ax_hm.axvline(2.5, color="black", linewidth=1.0)
    for i in range(3):
        for j in range(4):
            v = M[i, j]
            tc = "white" if abs(v) > 0.55 * lim else "black"
            f = A[i, j]
            ax_hm.text(j, i - 0.10, f"{f:.3f}",
                       ha="center", va="center", fontsize=9, color=tc, fontweight="bold")
            ax_hm.text(j, i + 0.18, f"({v:+.3f})",
                       ha="center", va="center", fontsize=8.5, color=tc)

    cb = fig.colorbar(im, ax=ax_hm, fraction=0.045, pad=0.03)
    cb.set_label("ΔF1 vs D1 baseline\n(green = improve, red = harm)")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {args.output}")
    print(f"Saved {args.output.with_suffix('.pdf')}")
    print("\nRows (best Δ macro at top):")
    for i, k in enumerate(KEPT_ord):
        print(f"  D1 + only-{k}: macro={abs_macro[i]:.3f} (Δ {delta_macro[i]:+.3f})  "
              f"per-sp Δ = {delta_F1[i].tolist()}")


if __name__ == "__main__":
    main()
