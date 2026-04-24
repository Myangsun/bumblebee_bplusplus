#!/usr/bin/env python3
"""Generate §5.3 LLM-as-judge plots that mirror the §5.4 expert-calibration plots.

Outputs (all under docs/plots/llm_judge/):
  llm_outcomes_1500.png          per-species stacked tier bars (strict/border/soft/hard)
  llm_funnel_per_species.png     500 -> blind-ID -> diag-species -> morph >= 4 funnel
  llm_blind_id_breakdown.png     per-target species, where blind-ID is when wrong
  llm_per_feature_heatmap.png    3 x 5 (species x feature) mean morph scores
  llm_failure_modes.png          gate-failure counts per species (no blind-ID / diag below
                                 species / per-feature < 3)

Uses the canonical Okabe-Ito palette for the three rare focal species.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path("/home/msun14/bumblebee_bplusplus")
OUT = ROOT / "docs/plots/llm_judge"
OUT.mkdir(parents=True, exist_ok=True)

RARE = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]
SHORT = {"Bombus_ashtoni": "B. ashtoni",
         "Bombus_sandersoni": "B. sandersoni",
         "Bombus_flavidus": "B. flavidus"}
COLORS = {"Bombus_ashtoni": "#0072B2",
          "Bombus_sandersoni": "#E69F00",
          "Bombus_flavidus": "#009E73"}

FEATURES = ["legs_appendages", "wing_venation_texture",
            "head_antennae", "abdomen_banding", "thorax_coloration"]
FEATURE_LBL = {"legs_appendages": "Legs &\nappendages",
               "wing_venation_texture": "Wing\nvenation",
               "head_antennae": "Head &\nantennae",
               "abdomen_banding": "Abdomen\nbanding",
               "thorax_coloration": "Thorax\ncoloration"}

# Tier colours mirror plot_150_outcomes.py outcome palette (green / grey / red),
# extended to four LLM tiers by adding a lighter green for the borderline tier.
TIER_COLOR = {"strict_pass":  "#86aa98",
              "borderline":   "#b8cdc1",
              "soft_fail":    "#c9c9c4",
              "hard_fail":    "#d86a6a"}


def _style():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })


def load() -> list[dict]:
    """Return one flat dict per image with derived columns."""
    d = json.loads((ROOT / "RESULTS_kfold/llm_judge_eval/results.json").read_text())
    rows = []
    for r in d["results"]:
        mf = r.get("morphological_fidelity", {}) or {}
        feats = {k: (mf.get(k, {}) or {}).get("score") for k in FEATURES}
        morph_vals = [v for v in feats.values() if v is not None]
        morph_mean = float(np.mean(morph_vals)) if morph_vals else np.nan
        bid = r.get("blind_identification", {}) or {}
        matches = bool(bid.get("matches_target", False))
        pred_species = bid.get("species")
        diag_level = (r.get("diagnostic_completeness", {}) or {}).get("level")
        strict = matches and (diag_level == "species") and (morph_mean >= 4.0)
        if strict:
            tier = "strict_pass"
        elif matches and diag_level == "species":
            tier = "borderline"
        elif matches:
            tier = "soft_fail"
        else:
            tier = "hard_fail"
        rows.append({
            "species": r["species"],
            "file": r["file"],
            "blind_species": pred_species,
            "matches_target": matches,
            "diag_level": diag_level,
            "morph_mean": morph_mean,
            **{f"m_{k}": feats[k] for k in FEATURES},
            "strict": strict,
            "tier": tier,
        })
    return rows


def plot_outcomes(rows, out):
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    x = np.arange(len(RARE))
    order = ["strict_pass", "borderline", "soft_fail", "hard_fail"]
    labels = {"strict_pass": "Strict pass", "borderline": "Borderline",
              "soft_fail": "Soft fail", "hard_fail": "Hard fail"}
    n_per = 500
    bottoms = np.zeros(len(RARE))
    for tier in order:
        vals = np.array([sum(1 for r in rows if r["species"] == sp and r["tier"] == tier)
                         for sp in RARE])
        ax.bar(x, vals, bottom=bottoms, label=labels[tier],
               color=TIER_COLOR[tier], edgecolor="white", linewidth=0.8)
        for xi, v, b in zip(x, vals, bottoms):
            if v >= 25:
                ax.text(xi, b + v/2, f"{v}\n({100*v/n_per:.0f}%)",
                        ha="center", va="center", color="white", fontsize=9)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[s] for s in RARE])
    for tick, sp in zip(ax.get_xticklabels(), RARE):
        tick.set_color(COLORS[sp]); tick.set_fontweight("bold")
    ax.set_ylabel("Images per species (n = 500)")
    ax.set_ylim(0, 500)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    ax.set_title("LLM judge tier outcomes on the 1,500-image pool",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(Path(out).with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_funnel(rows, out):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))
    for ax, sp in zip(axes, RARE):
        R = [r for r in rows if r["species"] == sp]
        gates = {
            "Generated":       500,
            "Blind-ID match":  sum(1 for r in R if r["matches_target"]),
            "Diag = species":  sum(1 for r in R if r["matches_target"] and r["diag_level"] == "species"),
            "Morph ≥ 4":       sum(1 for r in R if r["matches_target"] and r["diag_level"] == "species"
                                   and r["morph_mean"] >= 4.0),
        }
        names = list(gates.keys())
        vals = list(gates.values())
        # width proportional to count
        widths = [v/500 for v in vals]
        ys = np.arange(len(names))[::-1]
        for yi, w, v, n in zip(ys, widths, vals, names):
            ax.barh(yi, w, height=0.7, color=COLORS[sp], alpha=0.6 + 0.1*(yi/3),
                    edgecolor="white", linewidth=0.8)
            ax.text(w + 0.01, yi, f"{v}  ({100*v/500:.1f}%)", va="center", fontsize=9)
        ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=9)
        ax.set_xlim(0, 1.25); ax.set_xticks([])
        ax.set_title(SHORT[sp], color=COLORS[sp], fontweight="bold")
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("LLM strict-pass funnel per species on the 1,500-image pool",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(Path(out).with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_blind_id_breakdown(rows, out):
    """Per target species: stacked bars of (blind-ID correct, blind-ID top-k wrong guesses, no-species)."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))
    for ax, sp in zip(axes, RARE):
        R = [r for r in rows if r["species"] == sp]
        correct = sum(1 for r in R if r["matches_target"])
        wrong = [r["blind_species"] for r in R if not r["matches_target"] and r["blind_species"]]
        no_id = sum(1 for r in R if not r["matches_target"] and not r["blind_species"])
        guesses = Counter(wrong).most_common(6)
        labels = ["Correct"] + [g[0].replace("Bombus ", "B. ") if g[0] else "other"
                                 for g in guesses] + (["(no species ID)"] if no_id else [])
        vals = [correct] + [g[1] for g in guesses] + ([no_id] if no_id else [])
        palette = [COLORS[sp]] + ["#999999", "#c2c2c2", "#a08b7a", "#7f6d92",
                                  "#7aa0a0", "#bfa080"][:len(guesses)] + (["#d9534f"] if no_id else [])
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, height=0.7, color=palette, edgecolor="white", linewidth=0.8)
        for yi, v in zip(y, vals):
            ax.text(v + 6, yi, f"{v}  ({100*v/500:.0f}%)", va="center", fontsize=9)
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlim(0, 520); ax.set_xticks([])
        ax.set_title(SHORT[sp], color=COLORS[sp], fontweight="bold")
        ax.spines["bottom"].set_visible(False)
    fig.suptitle("LLM blind-ID outcomes per target species (n = 500 each)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(Path(out).with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_feature_heatmap(rows, out):
    mat = np.zeros((len(RARE), len(FEATURES)))
    for i, sp in enumerate(RARE):
        R = [r for r in rows if r["species"] == sp]
        for j, f in enumerate(FEATURES):
            vals = [r[f"m_{f}"] for r in R if r[f"m_{f}"] is not None]
            mat[i, j] = float(np.mean(vals)) if vals else np.nan
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=1, vmax=5, aspect="auto")
    for i in range(len(RARE)):
        for j in range(len(FEATURES)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=11, color="white" if v < 3.2 else "#222222",
                    fontweight="bold")
    ax.set_xticks(np.arange(len(FEATURES)))
    ax.set_xticklabels([FEATURE_LBL[f] for f in FEATURES], fontsize=9)
    ax.set_yticks(np.arange(len(RARE)))
    ax.set_yticklabels([SHORT[s] for s in RARE], fontsize=10)
    for tick, sp in zip(ax.get_yticklabels(), RARE):
        tick.set_color(COLORS[sp]); tick.set_fontweight("bold")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04, label="LLM mean morph score (1–5)")
    ax.set_title("LLM per-feature mean morphological score on the 1,500-image pool",
                 loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(Path(out).with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_failure_modes(rows, out):
    """Per species: counts of gate failures at each stage.
    LLM has no structural-failure flags (all 0), so instead we show:
      (a) no blind-ID match           → hard_fail tier red
      (b) diag below species           → soft_fail tier grey
      (c) per-feature mean < 4         → borderline tier light green

    Right panel: per-feature severe failures (score < 3) per species,
    coloured along a red gradient (all bars are failures by construction).
    """
    GATE_COLOR = {"no_id":   TIER_COLOR["hard_fail"],
                  "diag":    TIER_COLOR["soft_fail"],
                  "morph":   TIER_COLOR["borderline"]}
    SEVERE_RED = {"Bombus_ashtoni":    "#d86a6a",
                  "Bombus_sandersoni": "#bf5a5a",
                  "Bombus_flavidus":   "#a64a4a"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.3))
    x = np.arange(len(RARE))
    gate_no_id  = [sum(1 for r in rows if r["species"] == sp and not r["matches_target"]) for sp in RARE]
    gate_diag   = [sum(1 for r in rows if r["species"] == sp and r["matches_target"]
                        and r["diag_level"] != "species") for sp in RARE]
    gate_morph  = [sum(1 for r in rows if r["species"] == sp and r["matches_target"]
                        and r["diag_level"] == "species" and r["morph_mean"] < 4.0) for sp in RARE]
    bottom = np.zeros(len(RARE))
    for vals, key, lbl in [(gate_no_id, "no_id", "No blind-ID match (hard fail)"),
                           (gate_diag,  "diag",  "Diag below species (soft fail)"),
                           (gate_morph, "morph", "Morph mean < 4.0 (borderline)")]:
        ax1.bar(x, vals, bottom=bottom, color=GATE_COLOR[key], label=lbl,
                edgecolor="white", linewidth=0.8)
        for xi, v, b in zip(x, vals, bottom):
            if v >= 15:
                ax1.text(xi, b + v/2, str(v), ha="center", va="center",
                         color="#1f1f1f" if key != "no_id" else "white", fontsize=9)
        bottom += np.asarray(vals)
    ax1.set_xticks(x); ax1.set_xticklabels([SHORT[s] for s in RARE])
    for tick in ax1.get_xticklabels():
        tick.set_fontweight("bold")
    ax1.set_ylabel("Images failing each gate (n = 500 per species)")
    ax1.set_title("Gate-failure decomposition", loc="left", fontsize=11)
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Right: per-feature score-<3 counts; all bars are failures, coloured red.
    mat = np.zeros((len(RARE), len(FEATURES)))
    for i, sp in enumerate(RARE):
        R = [r for r in rows if r["species"] == sp]
        for j, f in enumerate(FEATURES):
            mat[i, j] = sum(1 for r in R if r[f"m_{f}"] is not None and r[f"m_{f}"] < 3)
    width = 0.27
    xf = np.arange(len(FEATURES))
    for i, sp in enumerate(RARE):
        ax2.bar(xf + (i - 1) * width, mat[i], width,
                label=SHORT[sp], color=SEVERE_RED[sp],
                edgecolor="white", linewidth=0.8)
        for xi, v in zip(xf + (i - 1) * width, mat[i]):
            if v > 0:
                ax2.text(xi, v + 1, f"{int(v)}", ha="center", va="bottom",
                         fontsize=7.5, color="#1f1f1f")
    ax2.set_xticks(xf)
    ax2.set_xticklabels([FEATURE_LBL[f] for f in FEATURES], fontsize=9)
    ax2.set_ylabel("Images with feature score < 3 (severe failure)")
    ax2.set_title("Per-feature severe failures (score < 3)", loc="left", fontsize=11)
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle("LLM-flagged failure modes on the 1,500-image pool",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(Path(out).with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    _style()
    rows = load()
    print(f"Loaded {len(rows)} LLM records")
    plot_outcomes(rows, OUT / "llm_outcomes_1500.png")
    print("-> outcomes")
    plot_funnel(rows, OUT / "llm_funnel_per_species.png")
    print("-> funnel")
    plot_blind_id_breakdown(rows, OUT / "llm_blind_id_breakdown.png")
    print("-> blind_id breakdown")
    plot_per_feature_heatmap(rows, OUT / "llm_per_feature_heatmap.png")
    print("-> per-feature heatmap")
    plot_failure_modes(rows, OUT / "llm_failure_modes.png")
    print("-> failure modes")
    print("DONE")


if __name__ == "__main__":
    main()
