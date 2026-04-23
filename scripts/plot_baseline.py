#!/usr/bin/env python3
"""Regenerate §5.1 baseline plots from the current f1-checkpoint test results.

Inputs:
  RESULTS_kfold/baseline@f1_gbif_test_results_*.json (latest)

Outputs (original-style, no per-species colour coding):
  docs/plots/baseline_f1_ci.png          horizontal per-species F1 + CI bars, sorted
                                         by F1, with test-support annotations and a
                                         dashed macro-F1 reference line.
  docs/plots/baseline_species_metrics.png precision / recall / F1 vertical bars per
                                         species, tier-coloured (Common / Moderate /
                                         Rare) with matching legend.
  docs/plots/baseline_confusion_matrix.png 16 x 16 row-normalised confusion matrix
                                         with numeric cell values (original style).
"""
from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path("/home/msun14/bumblebee_bplusplus")
OUT = ROOT / "docs/plots"
OUT.mkdir(parents=True, exist_ok=True)

RARE = {"Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"}
RNG = np.random.default_rng(42)

# Tier definitions from §3.3 (training-set size). These match the legend used in
# gbif_raw_counts.pdf / species_distribution.pdf so all §3-§5 plots read from the
# same three-tone grey palette.
TRAIN_N = {
    "Bombus_ashtoni": 22, "Bombus_sandersoni": 40, "Bombus_flavidus": 162,
    "Bombus_affinis": 268, "Bombus_citrinus": 395, "Bombus_vagans_Smith": 443,
    "Bombus_borealis": 471, "Bombus_terricola": 479, "Bombus_fervidus": 639,
    "Bombus_perplexus": 683, "Bombus_rufocinctus": 963,
    "Bombus_impatiens": 1227, "Bombus_ternarius_Say": 1247,
    "Bombus_bimaculatus": 1250, "Bombus_griseocollis": 1274,
    "Bombus_pensylvanicus": 1283,
}
TIER_COLOR_COMMON   = "#c0c0c0"
TIER_COLOR_MODERATE = "#6a6a6a"
TIER_COLOR_RARE     = "#111111"


def tier_color(species: str) -> str:
    n = TRAIN_N.get(species, 200)
    if n >= 900:
        return TIER_COLOR_COMMON
    if n >= 200:
        return TIER_COLOR_MODERATE
    return TIER_COLOR_RARE


def short_name(species: str) -> str:
    return "B. " + (species.replace("Bombus_", "")
                          .replace("_Smith", "")
                          .replace("_Say", "")
                          .replace("_", " "))


def latest_json(pattern: str) -> Path:
    matches = sorted(glob.glob(str(ROOT / pattern)))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def bootstrap_per_species_f1(y_true, y_pred, species_list, n_boot: int = 10_000):
    """Vectorised bootstrap per species: one resample of indices per iter, all
    species' F1s computed in closed form. ~100x faster than 16x sklearn.f1_score."""
    n = len(y_true)
    idx_map = {sp: i for i, sp in enumerate(species_list)}
    yt_i = np.array([idx_map[s] for s in y_true])
    yp_i = np.array([idx_map[s] for s in y_pred])
    K = len(species_list)

    def f1_per(yt, yp):
        tp = np.zeros(K); fp = np.zeros(K); fn = np.zeros(K)
        match = (yt == yp)
        np.add.at(tp, yt[match], 1)
        np.add.at(fn, yt[~match], 1)
        np.add.at(fp, yp[~match], 1)
        denom = 2 * tp + fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(denom > 0, 2 * tp / denom, 0.0)

    point = f1_per(yt_i, yp_i)
    boot = np.empty((n_boot, K))
    for b in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        boot[b] = f1_per(yt_i[idx], yp_i[idx])
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    out = {sp: (float(point[i]), float(lo[i]), float(hi[i])) for i, sp in enumerate(species_list)}
    return out


def plot_f1_ci(f1ci, support, macro_f1, out_path: Path):
    """Original style: horizontal bars sorted by F1 descending, with F1 label at
    the bar tip, test-support in the y-tick, dashed macro-F1 vertical line."""
    species_by_f1 = sorted(f1ci, key=lambda s: f1ci[s][0], reverse=True)
    ytick_labels = [f"{short_name(s)}  (n={support[s]})" for s in species_by_f1]
    y = np.arange(len(species_by_f1))[::-1]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for yi, sp in zip(y, species_by_f1):
        f1p, lo, hi = f1ci[sp]
        ax.barh(yi, f1p, height=0.7, color="#f4f1ea", edgecolor="#3d3d3d", linewidth=0.8)
        ax.errorbar(f1p, yi, xerr=[[f1p - lo], [hi - f1p]], fmt="none",
                    ecolor="#222222", elinewidth=1.1, capsize=4)
        ax.text(hi + 0.012, yi, f"{f1p:.3f}", va="center", fontsize=9, color="#222222")
        if sp in RARE:
            ytl = ax.get_yticklabels()

    ax.set_yticks(y)
    ax.set_yticklabels(ytick_labels, fontsize=9)
    for tick, sp in zip(ax.get_yticklabels(), species_by_f1):
        if sp in RARE:
            tick.set_fontweight("bold")

    ax.axvline(macro_f1, linestyle="--", color="#888888", linewidth=0.9)
    ax.text(macro_f1 + 0.005, -0.85, f"Macro F1 = {macro_f1:.3f}",
            fontsize=9, color="#888888", va="center")
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("F1 Score")
    ax.set_title("Baseline Per-Species F1 with 95 % Bootstrap CI", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_f1_ci_by_tier(f1ci, support, macro_f1, out_path: Path):
    """Vertical per-species F1 + 95 % bootstrap CI bars, x-axis ordered and styled
    to match species_distribution.pdf: tier-grouped (Common / Moderate / Rare),
    descending training count within each group, three-tone grey palette, dotted
    vertical separators between tiers and italic tier headers along the top."""
    order = sorted(TRAIN_N, key=lambda s: -TRAIN_N[s])
    order = [s for s in order if s in f1ci]
    tiers = [("Common (>900)", [s for s in order if TRAIN_N[s] > 900],  TIER_COLOR_COMMON),
             ("Moderate (200-900)", [s for s in order if 200 <= TRAIN_N[s] <= 900], TIER_COLOR_MODERATE),
             ("Rare (<200)", [s for s in order if TRAIN_N[s] < 200], TIER_COLOR_RARE)]

    xs = np.arange(len(order))
    f_vals = np.array([f1ci[s][0] for s in order])
    lo = np.array([f1ci[s][1] for s in order])
    hi = np.array([f1ci[s][2] for s in order])
    colors = [tier_color(s) for s in order]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(xs, f_vals, 0.7, color=colors, edgecolor="#3d3d3d", linewidth=0.6)
    ax.errorbar(xs, f_vals, yerr=[f_vals - lo, hi - f_vals], fmt="none",
                ecolor="#222222", elinewidth=1.0, capsize=3)
    for i, (f, h) in enumerate(zip(f_vals, hi)):
        ax.text(i, h + 0.012, f"{f:.3f}", ha="center", va="bottom",
                fontsize=7.5, color="#222222")

    # Dotted tier separators + italic headers along the top.
    common_n  = len(tiers[0][1])
    mod_n     = len(tiers[1][1])
    sep_positions = [common_n - 0.5, common_n + mod_n - 0.5]
    for xp in sep_positions:
        ax.axvline(xp, linestyle=":", color="#9e9e9e", linewidth=0.9)
    midpoints = [(-0.5 + common_n - 0.5) / 2,
                 (common_n - 0.5 + common_n + mod_n - 0.5) / 2,
                 (common_n + mod_n - 0.5 + len(order) - 0.5) / 2]
    for (label, _sp, _c), xm in zip(tiers, midpoints):
        ax.text(xm, 1.14, label, ha="center", va="bottom",
                fontsize=10, style="italic", color="#555555")

    ax.axhline(macro_f1, linestyle="--", color="#888888", linewidth=0.9)
    ax.text(common_n + mod_n - 0.55, macro_f1 + 0.02,
            f"Macro F1 = {macro_f1:.3f}",
            fontsize=8.5, color="#888888", ha="right", va="bottom")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{short_name(s)}  (n={support[s]})" for s in order],
                       rotation=40, ha="right", fontsize=9)
    for tick, sp in zip(ax.get_xticklabels(), order):
        if sp in RARE:
            tick.set_fontweight("bold")

    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xlim(-0.7, len(order) - 0.3)
    ax.margins(x=0.01)
    ax.set_title("Baseline Per-Species F1 with 95 % Bootstrap CI",
                 fontsize=12, pad=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_species_metrics(metrics: dict, out_path: Path):
    """Per-species precision / recall / F1 vertical grouped bars, using the
    Common / Moderate / Rare three-tone grey palette from species_distribution.pdf
    on the F1 bar; precision/recall are light / medium grey so the tier signal
    reads only on F1. Legend is placed outside the axes to avoid overlap."""
    # Order: by tier (Common, Moderate, Rare) matching gbif_raw_counts.pdf left→right.
    order = sorted(metrics.keys(), key=lambda s: (-TRAIN_N.get(s, 0)))
    n = len(order)
    xs = np.arange(n)
    width = 0.27

    p_vals = [metrics[s]["precision"] for s in order]
    r_vals = [metrics[s]["recall"] for s in order]
    f_vals = [metrics[s]["f1"] for s in order]
    tier_colors = [tier_color(s) for s in order]

    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.bar(xs - width, p_vals, width, color="#dcdcdc", edgecolor="#3d3d3d",
           linewidth=0.6, label="Precision")
    ax.bar(xs, r_vals, width, color="#9e9e9e", edgecolor="#3d3d3d",
           linewidth=0.6, label="Recall")
    ax.bar(xs + width, f_vals, width, color=tier_colors, edgecolor="#3d3d3d",
           linewidth=0.6, label="F1  (tier-coloured)")

    for i, f in enumerate(f_vals):
        ax.text(i + width, f + 0.012, f"{f:.2f}",
                ha="center", va="bottom", fontsize=7.5, color="#222222")

    ax.set_xticks(xs)
    ax.set_xticklabels([short_name(s) for s in order], rotation=40,
                       ha="right", fontsize=9)
    for tick, sp in zip(ax.get_xticklabels(), order):
        if sp in RARE:
            tick.set_fontweight("bold")

    ax.set_ylabel("Per-species precision / recall / F1")
    ax.set_ylim(0, 1.05)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Baseline ResNet-50 per-species precision, recall and F1",
                 loc="left", fontsize=12)

    # Legend combining metric + tier cues; placed above the plot so it does not
    # overlap bars or tick labels.
    from matplotlib.patches import Patch
    metric_handles = [
        Patch(facecolor="#dcdcdc", edgecolor="#3d3d3d", label="Precision"),
        Patch(facecolor="#9e9e9e", edgecolor="#3d3d3d", label="Recall"),
    ]
    tier_handles = [
        Patch(facecolor=TIER_COLOR_COMMON,   edgecolor="#3d3d3d", label="F1 (Common, n > 900)"),
        Patch(facecolor=TIER_COLOR_MODERATE, edgecolor="#3d3d3d", label="F1 (Moderate, 200 ≤ n ≤ 900)"),
        Patch(facecolor=TIER_COLOR_RARE,     edgecolor="#3d3d3d", label="F1 (Rare, n < 200)"),
    ]
    ax.legend(handles=metric_handles + tier_handles, loc="lower center",
              bbox_to_anchor=(0.5, 1.04), ncol=5, frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_confusion(y_true, y_pred, species_order, out_path: Path):
    """Row-normalised confusion matrix with numeric cell values (original style).
    Ordering follows the original artefact: alphabetical by species name."""
    species_alpha = sorted(species_order)
    cm = confusion_matrix(y_true, y_pred, labels=species_alpha)
    rs = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, rs, out=np.zeros_like(cm, dtype=float), where=rs > 0)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    for i in range(len(species_alpha)):
        for j in range(len(species_alpha)):
            v = cm_norm[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if v > 0.55 else "#1f1f1f")

    ax.set_xticks(np.arange(len(species_alpha)))
    ax.set_yticks(np.arange(len(species_alpha)))
    short = [short_name(s) for s in species_alpha]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    for tick, sp in zip(ax.get_xticklabels(), species_alpha):
        if sp in RARE:
            tick.set_fontweight("bold")
    for tick, sp in zip(ax.get_yticklabels(), species_alpha):
        if sp in RARE:
            tick.set_fontweight("bold")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Baseline Confusion Matrix (Row-Normalized)", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.82, label="Proportion")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    js = latest_json("RESULTS_kfold/baseline@f1_gbif_test_results_*.json")
    print(f"Loading {js}")
    d = json.loads(js.read_text())
    species_list = d["species_list"]
    preds = d["detailed_predictions"]
    y_true = [p["ground_truth"] for p in preds]
    y_pred = [p["prediction"] for p in preds]
    support = Counter(y_true)
    macro_f1 = float(d["macro_f1"])

    print(f"Macro F1 = {macro_f1:.3f}")
    print("Bootstrapping per-species F1 (10 000 resamples)…")
    f1ci = bootstrap_per_species_f1(y_true, y_pred, species_list, n_boot=10_000)
    for sp in ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"):
        p, lo, hi = f1ci[sp]
        print(f"  {short_name(sp):<18}  F1 {p:.3f}  CI [{lo:.3f}, {hi:.3f}]  n={support[sp]}")

    plot_f1_ci(f1ci, support, macro_f1, OUT / "baseline_f1_ci.png")
    print(f"-> {OUT/'baseline_f1_ci.png'}")
    plot_f1_ci_by_tier(f1ci, support, macro_f1, OUT / "baseline_f1_ci_by_tier.png")
    print(f"-> {OUT/'baseline_f1_ci_by_tier.png'}")
    plot_species_metrics(d["species_metrics"], OUT / "baseline_species_metrics.png")
    print(f"-> {OUT/'baseline_species_metrics.png'}")
    plot_confusion(y_true, y_pred, species_list, OUT / "baseline_confusion_matrix.png")
    print(f"-> {OUT/'baseline_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
