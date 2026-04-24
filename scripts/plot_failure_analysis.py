#!/usr/bin/env python3
"""
Task 1 Phase 1a plots: species-level F1 deltas, flip-category heatmap, and
per-species correct-rate trajectories.

Modes
-----
    species_delta   → docs/plots/failure/species_f1_delta.png
    flip_heatmap    → docs/plots/failure/flip_category_heatmap.png
    trajectory      → docs/plots/failure/correct_rate_trajectory.png
    all             → run all three

Inputs
------
    RESULTS/failure_analysis/species_f1.csv       (from build_tier_f1_table.py)
    RESULTS/failure_analysis/flip_analysis.csv    (from analyze_flips.py)
    RESULTS/failure_analysis/flip_summary.json    (from analyze_flips.py)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from pipeline.config import PROJECT_ROOT, RESULTS_DIR

AUG_CONFIGS = ("d3_cnp", "d4_synthetic", "d5_llm_filtered", "d2_centroid", "d6_probe")
AUG_LABELS = {"d3_cnp": "D2 CNP",
              "d4_synthetic": "D3 Unfiltered",
              "d5_llm_filtered": "D4 LLM-filtered",
              "d2_centroid": "D5 Centroid",
              "d6_probe": "D6 Expert-probe"}
CATEGORIES = ("stable-correct", "stable-wrong", "improved", "harmed")
CATEGORY_LABELS = {
    "stable-correct": "stable-correct",
    "stable-wrong":   "stable-wrong",
    "improved":       "improved (aug fixed)",
    "harmed":         "harmed (aug broke)",
}

DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "plots" / "failure"


# ── Shared loaders ───────────────────────────────────────────────────────────


def _load_species_f1(path: Path) -> List[dict]:
    with path.open() as fh:
        reader = csv.DictReader(fh)
        rows = [{k: (float(v) if k not in {"species", "tier"} else v)
                 for k, v in r.items()} for r in reader]
    for r in rows:
        r["train_n"] = int(r["train_n"])
    return rows


def _load_flip_summary(path: Path) -> dict:
    return json.loads(path.read_text())


# ── Plot: per-species F1 delta heatmap ───────────────────────────────────────


def plot_species_delta(species_rows: List[dict], output_path: Path,
                       protocol: str = "multiseed") -> None:
    species_rows = sorted(species_rows, key=lambda r: r["train_n"])
    species_labels = [r["species"].replace("Bombus_", "B. ") +
                      f"\n(n={r['train_n']}, {r['tier']})"
                      for r in species_rows]
    deltas = np.array([
        [r[f"{c}_f1_delta"] for c in AUG_CONFIGS] for r in species_rows
    ])
    limit = float(np.max(np.abs(deltas)))
    limit = max(limit, 0.10)  # keep scale readable even for tiny deltas

    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.imshow(deltas, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(range(len(AUG_CONFIGS)))
    ax.set_xticklabels([AUG_LABELS[c] for c in AUG_CONFIGS], rotation=25, ha="right")
    ax.set_yticks(range(len(species_labels)))
    ax.set_yticklabels(species_labels, fontsize=9)
    subtitle = "mean across 5 seeds (fixed split)" if protocol == "multiseed" else "mean across 5 CV folds"
    ax.set_title(f"Per-species F1 delta vs baseline\n({subtitle})", fontsize=11)
    for i, row in enumerate(deltas):
        for j, val in enumerate(row):
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center",
                    fontsize=8, color=("white" if abs(val) > limit * 0.55 else "black"))
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Δ macro-F1 (aug − baseline)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(Path(output_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── Plot: flip-category heatmap (species × aug × category) ───────────────────


def plot_flip_heatmap(summary: dict, species_rows: List[dict],
                      output_path: Path, normalize: str = "both") -> None:
    """Heatmap of flip categories per species × per aug config.

    ``normalize``:
        "counts" — absolute flip counts
        "rates"  — counts / n_test per species (percent)
        "both"   — two stacked rows (counts on top, rates on bottom)
    """
    per_sp = summary["per_species_categories"]
    species_order = [r["species"] for r in sorted(species_rows, key=lambda r: r["train_n"])]
    species_order = [sp for sp in species_order if sp in per_sp]

    n_aug = len(AUG_CONFIGS)
    net_matrix = np.zeros((len(species_order), n_aug))
    harmed_matrix = np.zeros_like(net_matrix)
    improved_matrix = np.zeros_like(net_matrix)
    supports: List[int] = []
    for i, sp in enumerate(species_order):
        counts = per_sp[sp]
        total = sum(counts[AUG_CONFIGS[0]].values())
        supports.append(total)
        for j, aug in enumerate(AUG_CONFIGS):
            c = counts[aug]
            improved_matrix[i, j] = c["improved"]
            harmed_matrix[i, j] = c["harmed"]
            net_matrix[i, j] = c["improved"] - c["harmed"]

    support_col = np.array(supports, dtype=float).reshape(-1, 1)
    improved_rate = 100.0 * improved_matrix / support_col
    harmed_rate = 100.0 * harmed_matrix / support_col
    net_rate = 100.0 * net_matrix / support_col

    species_labels = [sp.replace("Bombus_", "B. ") + f"\n(n={supports[i]})"
                      for i, sp in enumerate(species_order)]
    aug_labels = [AUG_LABELS[c] for c in AUG_CONFIGS]

    rows_cfg: List[tuple] = []
    if normalize in ("counts", "both"):
        rows_cfg.append(("counts", "# images",
                         [improved_matrix, harmed_matrix, net_matrix],
                         ["# improved", "# harmed", "net"],
                         ["Greens", "Reds", "RdBu"],
                         [0, 0, None]))  # vmin for non-diverging
    if normalize in ("rates", "both"):
        rows_cfg.append(("rates", "% of test set",
                         [improved_rate, harmed_rate, net_rate],
                         ["% improved", "% harmed", "net %"],
                         ["Greens", "Reds", "RdBu"],
                         [0, 0, None]))

    n_rows = len(rows_cfg)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 7 * n_rows),
                             gridspec_kw={"wspace": 0.25})
    if n_rows == 1:
        axes = np.array([axes])

    fmt_counts = "{:.0f}".format
    fmt_rates = "{:.1f}".format

    for row_i, (kind, ylabel, mats, titles, cmaps, vmins) in enumerate(rows_cfg):
        fmt = fmt_counts if kind == "counts" else fmt_rates
        for col_i, (M, title, cmap, vmin) in enumerate(zip(mats, titles, cmaps, vmins)):
            ax = axes[row_i, col_i] if n_rows > 1 else axes[col_i]
            if cmap == "RdBu":
                lim = max(1.0, float(np.max(np.abs(M))))
                im = ax.imshow(M, cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
            else:
                im = ax.imshow(M, cmap=cmap, vmin=0,
                               vmax=max(1.0, float(M.max())), aspect="auto")
            ax.set_title(f"{title} ({ylabel})", fontsize=10)
            ax.set_xticks(range(n_aug))
            ax.set_xticklabels(aug_labels, rotation=25, ha="right")
            if col_i == 0:
                ax.set_yticks(range(len(species_labels)))
                ax.set_yticklabels(species_labels, fontsize=8)
            else:
                ax.set_yticks(range(len(species_labels)))
                ax.set_yticklabels([""] * len(species_labels))
            max_abs = max(1.0, float(np.abs(M).max()))
            for i, r in enumerate(M):
                for j, val in enumerate(r):
                    ax.text(j, i, fmt(val), ha="center", va="center",
                            fontsize=7,
                            color=("white" if abs(val) > 0.55 * max_abs else "black"))
            plt.colorbar(im, ax=ax, shrink=0.6)

    fig.suptitle("Flip-category heatmap: species × augmentation\n"
                 "(5-seed majority vote vs baseline)",
                 fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(Path(output_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── Plot: per-species correct-rate trajectory ────────────────────────────────


def plot_trajectory(species_rows: List[dict], summary: dict,
                    output_path: Path, protocol: str = "multiseed") -> None:
    """Trajectory of per-species mean correctness rate across configs.

    Focus group = 3 rare species + top-5 most-changed common/moderate species.
    """
    rates = summary["per_species_mean_correct_rate"]
    configs = ("baseline", "d3_cnp", "d4_synthetic", "d5_llm_filtered")

    rare = [r["species"] for r in species_rows if r["tier"] == "rare"]
    # Most-changed (max |delta| among d3/d4/d5)
    def max_abs_delta(r):
        return max(abs(r[f"{c}_f1_delta"]) for c in AUG_CONFIGS)
    ranked = sorted([r for r in species_rows if r["tier"] != "rare"],
                    key=max_abs_delta, reverse=True)
    top_changed = [r["species"] for r in ranked[:5]]
    focus = rare + top_changed

    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(configs))
    for sp in focus:
        stats = rates[sp]
        y = [stats[c] for c in configs]
        is_rare = sp in rare
        ax.plot(x, y, marker="o",
                linewidth=2.5 if is_rare else 1.3,
                alpha=1.0 if is_rare else 0.7,
                label=sp.replace("Bombus_", "B. ") + (" (rare)" if is_rare else ""))
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").replace("baseline", "D1 Baseline")
                        for c in configs])
    ax.set_ylabel("Mean per-image correct rate (5 seeds)")
    ax.set_title("Per-species correct-rate trajectory — rare species + top-5 most-affected")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, frameon=False, loc="center left",
              bbox_to_anchor=(1.01, 0.5))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.savefig(Path(output_path).with_suffix(".pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("species_delta", "flip_heatmap",
                                            "trajectory", "all"), default="all")
    parser.add_argument("--protocol", choices=("multiseed", "kfold"),
                        default="multiseed",
                        help="Which species F1 CSV to read (default: multiseed).")
    parser.add_argument("--normalize", choices=("counts", "rates", "both"),
                        default="both",
                        help="flip_heatmap: show absolute counts, rates (%% of test), or both rows.")
    parser.add_argument("--species-f1", type=Path, default=None,
                        help="Override species F1 CSV path (default auto from --protocol).")
    parser.add_argument("--flip-summary",
                        type=Path,
                        default=RESULTS_DIR / "failure_analysis" / "flip_summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.species_f1 is None:
        args.species_f1 = RESULTS_DIR / "failure_analysis" / f"species_f1_{args.protocol}.csv"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    species_rows = _load_species_f1(args.species_f1)
    summary = _load_flip_summary(args.flip_summary)

    suffix = f"_{args.protocol}"
    modes = ("species_delta", "flip_heatmap", "trajectory") if args.mode == "all" else (args.mode,)
    for mode in modes:
        if mode == "species_delta":
            plot_species_delta(species_rows,
                               args.output_dir / f"species_f1_delta{suffix}.png",
                               protocol=args.protocol)
        elif mode == "flip_heatmap":
            # flip_heatmap is multi-seed only (needs per-image prediction flips).
            if args.protocol != "multiseed":
                print("  [skip] flip_heatmap is multi-seed only")
                continue
            plot_flip_heatmap(summary, species_rows,
                              args.output_dir / "flip_category_heatmap.png",
                              normalize=args.normalize)
        elif mode == "trajectory":
            if args.protocol != "multiseed":
                print("  [skip] trajectory is multi-seed only")
                continue
            plot_trajectory(species_rows, summary,
                            args.output_dir / f"correct_rate_trajectory{suffix}.png",
                            protocol=args.protocol)
    print("Done.")


if __name__ == "__main__":
    main()
