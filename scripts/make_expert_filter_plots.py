#!/usr/bin/env python3
"""Generate §5.4 expert-calibration and filter-comparison plots.

Tier 1 (must-have, 14) + Tier 2 (nice-to-have, 8) = 22 figures.
All inputs on disk; no GPU, no new training runs required.

Output: docs/plots/filters/ (new figures) and docs/plots/filters/grids/ (image mosaics).
"""
from __future__ import annotations

import ast
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib_venn import venn3
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc as sk_auc
from sklearn.metrics import f1_score, roc_curve
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/msun14/bumblebee_bplusplus")
OUT = ROOT / "docs/plots/filters"
GRID_OUT = OUT / "grids"
OUT.mkdir(parents=True, exist_ok=True)
GRID_OUT.mkdir(parents=True, exist_ok=True)

RARE = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]
SHORT = {"Bombus_ashtoni": "B. ashtoni", "Bombus_sandersoni": "B. sandersoni", "Bombus_flavidus": "B. flavidus"}
FSLUG = {"Bombus_ashtoni": "ashtoni", "Bombus_sandersoni": "sandersoni", "Bombus_flavidus": "flavidus"}
# Canonical Okabe-Ito palette (deuteranopia-safe); used consistently across all thesis plots.
COLORS = {"Bombus_ashtoni": "#0072B2", "Bombus_sandersoni": "#E69F00", "Bombus_flavidus": "#009E73"}
SYNTH_DIR = ROOT / "RESULTS_kfold/synthetic_generation"
rng = random.Random(42)


# ---------- data loading ----------

def load_centroid() -> pd.DataFrame:
    return pd.read_csv(ROOT / "RESULTS/filters/centroid_scores.csv")


def load_probe() -> pd.DataFrame:
    return pd.read_csv(ROOT / "RESULTS/filters/probe_scores.csv")


def load_probe_meta() -> dict:
    return json.loads((ROOT / "RESULTS/filters/probe_scores.json").read_text())["meta"]


def load_llm() -> pd.DataFrame:
    """One row per synthetic. Adds: overall_pass (lenient bool), strict_pass (bool), morph_mean, per-feature morphs, matches_target, diag_level."""
    d = json.loads((ROOT / "RESULTS_kfold/llm_judge_eval/results.json").read_text())
    rows = []
    for r in d["results"]:
        mf = r.get("morphological_fidelity", {}) or {}
        feats = {k: (mf.get(k, {}) or {}).get("score") for k in
                 ("legs_appendages", "wing_venation_texture", "head_antennae", "abdomen_banding", "thorax_coloration")}
        morph_vals = [v for v in feats.values() if v is not None]
        mean_morph = float(np.mean(morph_vals)) if morph_vals else np.nan
        matches = bool((r.get("blind_identification", {}) or {}).get("matches_target", False))
        diag = (r.get("diagnostic_completeness", {}) or {}).get("level", None)
        rows.append({
            "basename": r["file"],
            "species": r["species"],
            "llm_lenient": bool(r.get("overall_pass", False)),
            "matches_target": matches,
            "diag_level": diag,
            "llm_morph_mean": mean_morph,
            **{f"morph_{k}": feats[k] for k in feats},
        })
    df = pd.DataFrame(rows)
    df["llm_strict"] = df["matches_target"] & (df["diag_level"] == "species") & (df["llm_morph_mean"] >= 4.0)
    return df


STRUCTURAL = {
    "extra_limbs", "missing_limbs", "impossible_geometry",
    "visible_artifact", "visible_artifacts", "blurry_artifacts", "repetitive_patterns",
}


def load_expert() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "RESULTS/expert_validation_results/jessie_all_150.csv")
    df["basename"] = df["image_path"].str.split("/").str[-1]
    df["ground_truth_species"] = df["ground_truth_species"].str.replace(" ", "_", regex=False)
    df["blind_id_species"] = df["blind_id_species"].fillna("").str.replace(" ", "_", regex=False)
    morph_cols = ["morph_legs_appendages", "morph_wing_venation_texture", "morph_head_antennae",
                  "morph_abdomen_banding", "morph_thorax_coloration"]
    df["expert_morph_mean"] = df[morph_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    def parse_modes(s):
        try:
            return json.loads(s) if isinstance(s, str) else {}
        except Exception:
            return {}

    df["failure_dict"] = df["failure_modes"].apply(parse_modes)

    def has_struct(d):
        all_modes = set(d.get("all") or [])
        return bool(all_modes & STRUCTURAL)

    df["has_structural"] = df["failure_dict"].apply(has_struct)
    df["expert_lenient"] = (
        (~df["has_structural"])
        & df["diagnostic_level"].isin(["genus", "species"])
        & (df["expert_morph_mean"] >= 3.0)
    )
    df["expert_strict"] = (
        (df["blind_id_species"] == df["ground_truth_species"])
        & (df["diagnostic_level"] == "species")
        & (df["expert_morph_mean"] >= 4.0)
    )
    return df


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


# =====================================================================
# Tier 1 + Tier 2 plot functions
# =====================================================================

def plot_A1_venn(cent, probe, llm):
    """Per-species 3-set Venn of pass sets on 1,500 pool."""
    tau_c = {"Bombus_ashtoni": 0.6945, "Bombus_sandersoni": 0.7468, "Bombus_flavidus": 0.6817}
    meta = load_probe_meta()
    tau_p = {k: v for k, v in meta["per_species_threshold_strict"].items()}
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, sp in zip(axes, RARE):
        c = set(cent[(cent.species == sp) & (cent.score >= tau_c[sp])].basename)
        p = set(probe[(probe.species == sp) & (probe.score >= tau_p[sp])].basename)
        l = set(llm[(llm.species == sp) & (llm.llm_strict)].basename)
        venn3([l, c, p], set_labels=("LLM-strict", "Centroid", "Probe"), ax=ax,
              set_colors=("#a6cee3", "#b2df8a", "#fb9a99"))
        ax.set_title(f"{SHORT[sp]}  |LLM|={len(l)} |Cent|={len(c)} |Probe|={len(p)}")
    fig.suptitle("Per-species pass-set Venn diagrams on the 1,500 synthetic pool", y=1.02)
    fig.savefig(OUT / "venn_llm_centroid_probe.png")
    fig.savefig(OUT / "venn_llm_centroid_probe.pdf")
    plt.close(fig)


def plot_A2_scatter(cent, probe, llm, expert):
    """Per-synth centroid × probe scatter, two-panel layout: pool view + expert-overlay view.

    Panel A: all 1,500 synthetics, coloured by species, with per-species τ guides.
              Reader can see (i) within-species spread on both axes and (ii) the joint
              region each filter selects.
    Panel B: 150 expert-labelled subset, same axes, with marker shape encoding
              expert-strict pass / fail. Reader can see whether expert-strict images
              cluster with high probe-y (D6 axis) or with high centroid-x (D5 axis).
    """
    merged = cent.rename(columns={"score": "centroid"}).merge(
        probe.rename(columns={"score": "probe"}), on=["basename", "species"]
    ).merge(llm[["basename", "llm_strict"]], on="basename", how="left")
    merged = merged.merge(
        expert[["basename", "expert_strict", "expert_lenient"]], on="basename", how="left"
    )
    meta = load_probe_meta()
    tau_p_all = meta["per_species_threshold_strict"]
    tau_c_all = {"Bombus_ashtoni": 0.6945, "Bombus_sandersoni": 0.7468, "Bombus_flavidus": 0.6817}

    # --- Panel A: full 1,500-image pool, coloured by species ---
    figA, axA = plt.subplots(1, 1, figsize=(7, 6))
    for sp in RARE:
        sub = merged[merged.species == sp]
        axA.scatter(sub.centroid, sub.probe, s=14, c=COLORS[sp], alpha=0.45,
                    edgecolor="white", linewidth=0.3, label=SHORT[sp])
    axA.set_xlabel("D5 centroid cosine  (real-image-similarity score)")
    axA.set_ylabel("D6 probe pass-probability  (expert-similarity score)")
    axA.set_title("Filter scores per synthetic, full 1,500-image pool",
                  loc="left", fontsize=11)
    axA.legend(frameon=False, loc="lower right")
    axA.spines["top"].set_visible(False); axA.spines["right"].set_visible(False)
    figA.tight_layout()
    figA.savefig(OUT / "centroid_vs_probe_scatter_pool.png", dpi=200, bbox_inches="tight")
    figA.savefig(OUT / "centroid_vs_probe_scatter_pool.pdf", dpi=200, bbox_inches="tight")
    plt.close(figA)

    # --- Panel B: 150-image expert-labelled subset, marker = expert label ---
    figB, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sp in zip(axes, RARE):
        sub = merged[(merged.species == sp) & merged.expert_strict.notna()]
        ep = sub[sub.expert_strict == True]   # noqa: E712
        ef = sub[sub.expert_strict == False]  # noqa: E712
        ax.scatter(ef.centroid, ef.probe, s=70, marker="x", c="#666666",
                   linewidth=1.5, label=f"expert fail (n={len(ef)})")
        ax.scatter(ep.centroid, ep.probe, s=80, marker="o",
                   facecolor=COLORS[sp], edgecolor="black", linewidth=0.8,
                   label=f"expert strict pass (n={len(ep)})")
        ax.axvline(tau_c_all[sp], color="#666666", ls="--", lw=0.9, alpha=0.7,
                   label=f"D5 τ = {tau_c_all[sp]:.3f}")
        ax.axhline(tau_p_all[sp], color="#a94442", ls="--", lw=0.9, alpha=0.7,
                   label=f"D6 τ = {tau_p_all[sp]:.3f}")
        ax.set_xlabel("D5 centroid cosine")
        if sp == RARE[0]:
            ax.set_ylabel("D6 probe pass-probability")
        ax.set_title(SHORT[sp], color=COLORS[sp], fontweight="bold")
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    figB.suptitle("Filter scores on 150 expert-labelled synthetics — does the probe-axis "
                  "rank expert quality better than the centroid-axis?",
                  y=1.02, fontsize=11)
    figB.tight_layout()
    figB.savefig(OUT / "centroid_vs_probe_scatter_expert.png", dpi=200, bbox_inches="tight")
    figB.savefig(OUT / "centroid_vs_probe_scatter_expert.pdf", dpi=200, bbox_inches="tight")
    plt.close(figB)

    # Also keep the original combined figure name for backward compatibility (overwrites with Panel B).
    figB2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
    for ax, sp in zip(axes2, RARE):
        sub = merged[(merged.species == sp) & merged.expert_strict.notna()]
        ep = sub[sub.expert_strict == True]   # noqa: E712
        ef = sub[sub.expert_strict == False]  # noqa: E712
        ax.scatter(ef.centroid, ef.probe, s=70, marker="x", c="#666666",
                   linewidth=1.5, label=f"expert fail (n={len(ef)})")
        ax.scatter(ep.centroid, ep.probe, s=80, marker="o",
                   facecolor=COLORS[sp], edgecolor="black", linewidth=0.8,
                   label=f"expert strict pass (n={len(ep)})")
        ax.axvline(tau_c_all[sp], color="#666666", ls="--", lw=0.9, alpha=0.7)
        ax.axhline(tau_p_all[sp], color="#a94442", ls="--", lw=0.9, alpha=0.7)
        ax.set_xlabel("D5 centroid cosine"); ax.set_title(SHORT[sp])
        if sp == RARE[0]:
            ax.set_ylabel("D6 probe pass-probability"); ax.legend(fontsize=8, frameon=False)
    figB2.savefig(OUT / "centroid_vs_probe_scatter.png", dpi=200, bbox_inches="tight")
    figB2.savefig(OUT / "centroid_vs_probe_scatter.pdf", dpi=200, bbox_inches="tight")
    plt.close(figB2)


def _selection_basenames_llm(llm, cap=200):
    out = {}
    for sp in RARE:
        sub = llm[(llm.species == sp) & (llm.llm_strict)]
        out[sp] = set(sub.basename[:cap])
    return out


def _selection_basenames_cent(cent, cap=200):
    taus = {"Bombus_ashtoni": 0.6945, "Bombus_sandersoni": 0.7468, "Bombus_flavidus": 0.6817}
    out = {}
    for sp in RARE:
        sub = cent[(cent.species == sp) & (cent.score >= taus[sp])].sort_values("score", ascending=False)
        out[sp] = set(sub.basename.iloc[:cap])
    return out


def _selection_basenames_probe(probe, cap=200):
    meta = load_probe_meta()
    taus = meta["per_species_threshold_strict"]
    out = {}
    for sp in RARE:
        sub = probe[(probe.species == sp) & (probe.score >= taus[sp])].sort_values("score", ascending=False)
        out[sp] = set(sub.basename.iloc[:cap])
    return out


def plot_A4_expert_coverage(cent, probe, llm, expert):
    """For each filter's selected 200, count expert-strict / lenient-only / fail / unlabelled."""
    sel = {"D4 LLM": _selection_basenames_llm(llm),
           "D5 Centroid": _selection_basenames_cent(cent),
           "D6 Probe": _selection_basenames_probe(probe)}
    rows = []
    for fname, per_sp in sel.items():
        for sp in RARE:
            bs = per_sp[sp]
            ex = expert[expert.basename.isin(bs)]
            n_strict = int(ex.expert_strict.sum())
            n_lenient_only = int(((~ex.expert_strict) & ex.expert_lenient).sum())
            n_fail = int((~ex.expert_lenient).sum())
            n_unlab = len(bs) - len(ex)
            rows.append({"filter": fname, "species": sp,
                         "strict": n_strict, "lenient_only": n_lenient_only,
                         "fail": n_fail, "unlabelled": n_unlab})
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    cats = ["strict", "lenient_only", "fail", "unlabelled"]
    cols = ["#2ca02c", "#ffbb78", "#d62728", "#cccccc"]
    for ax, sp in zip(axes, RARE):
        sub = df[df.species == sp].set_index("filter").loc[list(sel.keys())]
        bottom = np.zeros(len(sub))
        for cat, col in zip(cats, cols):
            ax.bar(sub.index, sub[cat].values, bottom=bottom, color=col, label=cat)
            bottom += sub[cat].values
        ax.set_title(SHORT[sp])
        ax.set_ylabel("images in 200-selection")
        ax.set_xticklabels(sub.index, rotation=20, ha="right")
    axes[-1].legend(loc="center left", bbox_to_anchor=(1, 0.5))
    fig.suptitle("Expert labelling coverage of each filter's 200-image selection", y=1.02)
    fig.savefig(OUT / "expert_coverage_of_selected200.png")
    fig.savefig(OUT / "expert_coverage_of_selected200.pdf")
    plt.close(fig)


def plot_A5_feature_heatmap():
    df = pd.read_csv(ROOT / "RESULTS/filters/label_analysis_150/per_feature_disagreement.csv")
    features = df["feature"].drop_duplicates().tolist()
    species = df["species"].drop_duplicates().tolist()
    over = df.pivot(index="species", columns="feature", values="llm_over_strict_rate").reindex(index=species, columns=features)
    blind = df.pivot(index="species", columns="feature", values="llm_blind_spot_rate").reindex(index=species, columns=features)
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.8))
    for ax, mat, title in [(axes[0], over, "LLM over-strict rate (expert≥4, LLM<4)"),
                           (axes[1], blind, "LLM blind-spot rate (LLM≥4, expert<4)")]:
        im = ax.imshow(mat.values, cmap="Reds", vmin=0, vmax=0.6, aspect="auto")
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([f.replace("_", "\n") for f in features], fontsize=9)
        ax.set_yticks(range(len(species)))
        ax.set_yticklabels([SHORT.get(s, s) for s in species])
        for i, sp in enumerate(species):
            for j, f in enumerate(features):
                v = mat.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v > 0.3 else "black", fontsize=8)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(OUT / "llm_vs_expert_feature_heatmap.png")
    fig.savefig(OUT / "llm_vs_expert_feature_heatmap.pdf")
    plt.close(fig)


def plot_A6_llm_vs_expert_2x2(llm, expert):
    """Overall + per-species 2×2 confusion LLM-strict × expert-strict."""
    m = expert.merge(llm[["basename", "llm_strict"]], on="basename")
    panels = [("Overall (n=150)", m)] + [(SHORT[sp] + f" (n={(m.ground_truth_species == sp).sum()})",
                                          m[m.ground_truth_species == sp]) for sp in RARE]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, (title, sub) in zip(axes, panels):
        tp = int(((sub.llm_strict == True) & (sub.expert_strict == True)).sum())  # noqa
        fp = int(((sub.llm_strict == True) & (sub.expert_strict == False)).sum())  # noqa
        fn = int(((sub.llm_strict == False) & (sub.expert_strict == True)).sum())  # noqa
        tn = int(((sub.llm_strict == False) & (sub.expert_strict == False)).sum())  # noqa
        mat = np.array([[tp, fn], [fp, tn]])
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() if mat.max() else 1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                        color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["expert-strict ✓", "expert-strict ✗"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["LLM-strict ✓", "LLM-strict ✗"])
        total = tp + fp + tn + fn
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        ax.set_title(f"{title}\nprec={prec:.2f} rec={rec:.2f}")
    fig.suptitle("LLM-strict × expert-strict confusion on 150-image sample", y=1.04)
    fig.savefig(OUT / "llm_vs_expert_strict_2x2.png")
    fig.savefig(OUT / "llm_vs_expert_strict_2x2.pdf")
    plt.close(fig)


def plot_A7_roc_loocv(expert):
    """LOOCV ROC per species at bioclip config."""
    from pipeline.evaluate.filters import align_synthetic_cache
    synth_cache = ROOT / "RESULTS/embeddings/bioclip_synthetic.npz"
    feats_all, basenames_all, species_all = align_synthetic_cache(synth_cache)
    basename_to_idx = {b: i for i, b in enumerate(basenames_all)}
    exp_mask = [b in basename_to_idx for b in expert.basename]
    exp_sub = expert[exp_mask].reset_index(drop=True)
    X = np.stack([feats_all[basename_to_idx[b]] for b in exp_sub.basename])
    y = exp_sub.expert_strict.astype(int).values
    chosen_C = 0.01
    loo = LeaveOneOut()
    probs = np.zeros(len(y))
    for tr, te in loo.split(X):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=chosen_C, class_weight="balanced",
                                                   solver="lbfgs", max_iter=4000, random_state=42))])
        pipe.fit(X[tr], y[tr])
        probs[te] = pipe.predict_proba(X[te])[:, 1]
    meta = load_probe_meta()
    taus = meta["per_species_threshold_strict"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, sp in zip(axes, RARE):
        mask = exp_sub.ground_truth_species.values == sp
        if mask.sum() < 2 or len(set(y[mask])) < 2:
            ax.set_visible(False); continue
        fpr, tpr, thr = roc_curve(y[mask], probs[mask])
        auc = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, c=COLORS[sp])
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        tau = taus[sp]
        idx = int(np.argmin(np.abs(thr - tau)))
        if idx < len(fpr):
            ax.scatter([fpr[idx]], [tpr[idx]], color="red", s=60, zorder=5, label=f"τ={tau:.2f}")
        ax.set_title(f"{SHORT[sp]}  AUC={auc:.3f}")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
        ax.legend(loc="lower right")
    fig.suptitle("LOOCV ROC per species (bioclip config, strict rule)", y=1.02)
    fig.savefig(OUT / "probe_roc_loocv.png")
    fig.savefig(OUT / "probe_roc_loocv.pdf")
    plt.close(fig)


def plot_A9_probe_config_ablation():
    d = json.loads((ROOT / "RESULTS/filters/probe_config_ablation.json").read_text())
    configs = list(d["configs"].keys()) if "configs" in d else [c["config"] for c in d["per_config"]]
    # Support both possible schemas.
    if "configs" in d:
        auc_s = [d["configs"][c]["loocv_auc_strict"] for c in configs]
        auc_l = [d["configs"][c]["loocv_auc_lenient"] for c in configs]
    else:
        auc_s = [c["loocv_auc_strict"] for c in d["per_config"]]
        auc_l = [c["loocv_auc_lenient"] for c in d["per_config"]]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(configs)); w = 0.35
    ax.bar(x - w / 2, auc_s, w, color="#1f77b4", label="LOOCV AUC (strict)")
    ax.bar(x + w / 2, auc_l, w, color="#ff7f0e", label="LOOCV AUC (lenient)")
    for i, (s, l) in enumerate(zip(auc_s, auc_l)):
        ax.text(i - w / 2, s + 0.005, f"{s:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, l + 0.005, f"{l:.3f}", ha="center", fontsize=8)
    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.3)
    ax.set_xticks(x); ax.set_xticklabels(configs, rotation=15, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_ylim(0.45, 0.85)
    ax.set_title("Probe feature-config ablation (150-image LOOCV)")
    ax.legend()
    fig.savefig(OUT / "probe_feature_config_ablation.png")
    fig.savefig(OUT / "probe_feature_config_ablation.pdf")
    plt.close(fig)


def plot_A15_filter_funnel(cent, probe, llm):
    """500 → LLM-strict → centroid-pass → probe-pass → cap200, per species."""
    taus_c = {"Bombus_ashtoni": 0.6945, "Bombus_sandersoni": 0.7468, "Bombus_flavidus": 0.6817}
    meta = load_probe_meta()
    taus_p = meta["per_species_threshold_strict"]
    stages = ["Generated (500)", "LLM-strict", "Centroid-pass", "Probe-pass", "After cap 200"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    for ax, sp in zip(axes, RARE):
        total = 500
        llm_n = int(llm[(llm.species == sp) & (llm.llm_strict)].shape[0])
        cent_n = int(cent[(cent.species == sp) & (cent.score >= taus_c[sp])].shape[0])
        probe_n = int(probe[(probe.species == sp) & (probe.score >= taus_p[sp])].shape[0])
        cap_n = [200, min(200, llm_n), min(200, cent_n), min(200, probe_n)]
        # Show the three filter columns side-by-side.
        bars = [total, llm_n, cent_n, probe_n, 200]
        ax.bar(stages, bars, color=["#4c4c4c", "#a6cee3", "#b2df8a", "#fb9a99", "#444444"])
        for i, v in enumerate(bars):
            ax.text(i, v + 5, str(v), ha="center", fontsize=9)
        ax.set_title(SHORT[sp])
        ax.set_ylabel("images")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Per-species selection funnel: 500 → pass sets → cap-at-200", y=1.04)
    fig.savefig(OUT / "filter_funnel_per_species.png")
    fig.savefig(OUT / "filter_funnel_per_species.pdf")
    plt.close(fig)


def plot_A20_failure_modes(expert):
    modes = {}
    for d in expert.failure_dict:
        for code in (d.get("all") or []):
            modes[code] = modes.get(code, 0) + 1
    # Per-species breakdown.
    keys = sorted(modes, key=modes.get, reverse=True)[:10]
    mat = np.zeros((3, len(keys)))
    for i, sp in enumerate(RARE):
        sub = expert[expert.ground_truth_species == sp]
        for d in sub.failure_dict:
            for code in (d.get("all") or []):
                if code in keys:
                    mat[i, keys.index(code)] += 1
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bottom = np.zeros(len(keys))
    for i, sp in enumerate(RARE):
        ax.bar(keys, mat[i], bottom=bottom, color=COLORS[sp], label=SHORT[sp])
        bottom += mat[i]
    ax.set_ylabel("image count")
    ax.set_title("Expert-flagged failure modes on 150-image sample (stacked by species)")
    ax.tick_params(axis="x", rotation=25)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend()
    fig.savefig(OUT / "expert_failure_mode_frequency.png")
    fig.savefig(OUT / "expert_failure_mode_frequency.pdf")
    plt.close(fig)


# ---------- image-grid helpers ----------

def _resolve_img(basename: str, species: str) -> Path | None:
    p = SYNTH_DIR / species / basename
    if p.exists():
        return p
    return None


def _render_grid(rows_cells, col_titles, row_titles, save_path, row_captions=None, figsize_per_cell=(1.8, 1.8)):
    n_rows = len(rows_cells); n_cols = max(len(r) for r in rows_cells) if rows_cells else 0
    if n_rows == 0 or n_cols == 0:
        return
    fig = plt.figure(figsize=(figsize_per_cell[0] * n_cols + 1.0, figsize_per_cell[1] * n_rows + 0.6))
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0.05, hspace=0.12)
    for i, cells in enumerate(rows_cells):
        for j, cell in enumerate(cells):
            ax = fig.add_subplot(gs[i, j])
            img_path, caption = cell
            if img_path and Path(img_path).exists():
                ax.imshow(Image.open(img_path).convert("RGB"))
            else:
                ax.text(0.5, 0.5, "—", ha="center", va="center")
            ax.axis("off")
            if caption:
                ax.set_title(caption, fontsize=7)
        if row_titles and i < len(row_titles):
            fig.text(0.01, 1 - (i + 0.5) / n_rows, row_titles[i], rotation=90, va="center", fontsize=10, weight="bold")
    if col_titles:
        for j, t in enumerate(col_titles):
            pass  # captions carry identifiers
    fig.savefig(save_path)
    plt.close(fig)


def plot_B1_per_species_4x4(cent, probe, llm):
    """4 rows (LLM, Centroid, Probe, All-three) × 4 random samples per rare species."""
    sel_llm = _selection_basenames_llm(llm)
    sel_c = _selection_basenames_cent(cent)
    sel_p = _selection_basenames_probe(probe)
    for sp in RARE:
        l, c, p = sel_llm[sp], sel_c[sp], sel_p[sp]
        all_three = l & c & p
        rows = [("LLM-strict", list(l)), ("Centroid", list(c)), ("Probe", list(p)),
                ("All three", list(all_three))]
        cells = []
        for name, pool in rows:
            samp = rng.sample(pool, min(4, len(pool)))
            cells.append([(_resolve_img(b, sp), b.split("::")[1]) for b in samp])
        _render_grid(cells, col_titles=None, row_titles=[r[0] for r in rows],
                     save_path=GRID_OUT / f"grid_{FSLUG[sp]}_4x4_by_filter.png")


def plot_B5_failure_mode_gallery(expert):
    modes = ["species_other", "wrong_coloration", "impossible_geometry", "wrong_scale",
             "extra_missing_limbs", "repetitive_pattern"]
    for sp in RARE:
        sub = expert[expert.ground_truth_species == sp]
        rows = []
        for mode in modes:
            imgs = sub[sub.failure_dict.apply(lambda d: mode in (d.get("all") or []))]
            samp = imgs.head(4)
            cells = [(_resolve_img(b, sp), f"m={m:.1f}" if not pd.isna(m) else "")
                     for b, m in zip(samp.basename, samp.expert_morph_mean)]
            rows.append(cells if cells else [(None, "")])
        _render_grid(rows, col_titles=None, row_titles=[m.replace("_", " ") for m in modes],
                     save_path=GRID_OUT / f"grid_{FSLUG[sp]}_failure_modes.png")


def plot_B6_probe_boundary(probe):
    meta = load_probe_meta()
    taus = meta["per_species_threshold_strict"]
    for sp in RARE:
        sub = probe[probe.species == sp].copy()
        tau = taus[sp]
        sub["dist"] = np.abs(sub.score - tau)
        low = sub[sub.score < tau].sort_values("dist").head(4)
        high = sub[sub.score >= tau].sort_values("dist").head(4)
        rows = [
            [(_resolve_img(b, sp), f"τ-0.0{int((tau-s)*100):02d}  ({s:.2f})") for b, s in zip(low.basename, low.score)],
            [(_resolve_img(b, sp), f"τ+0.0{int((s-tau)*100):02d}  ({s:.2f})") for b, s in zip(high.basename, high.score)],
        ]
        _render_grid(rows, col_titles=None, row_titles=[f"just BELOW τ={tau:.2f}", f"just ABOVE τ={tau:.2f}"],
                     save_path=GRID_OUT / f"grid_{FSLUG[sp]}_probe_boundary.png")


def plot_B14_B15_disagree_llm_expert(expert, llm):
    m = expert.merge(llm[["basename", "llm_strict"]], on="basename")
    for sp in RARE:
        sub = m[m.ground_truth_species == sp]
        pos_neg = sub[(sub.llm_strict == True) & (sub.expert_strict == False)].head(6)  # noqa
        neg_pos = sub[(sub.llm_strict == False) & (sub.expert_strict == True)].head(6)  # noqa
        rows = [
            [(_resolve_img(b, sp), f"m={m:.1f}") for b, m in zip(pos_neg.basename, pos_neg.expert_morph_mean)],
            [(_resolve_img(b, sp), f"m={m:.1f}") for b, m in zip(neg_pos.basename, neg_pos.expert_morph_mean)],
        ]
        _render_grid(rows, col_titles=None,
                     row_titles=["LLM pass, expert FAIL", "LLM fail, expert PASS"],
                     save_path=GRID_OUT / f"grid_{FSLUG[sp]}_disagree_llm_expert.png")


# ---------- Tier 2 ----------

def plot_A3_score_violins_by_tier(cent, probe, expert):
    exp = expert[["basename", "ground_truth_species", "tier"]]
    tiers = ["strict_pass", "borderline", "soft_fail", "hard_fail"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex="col")
    for j, sp in enumerate(RARE):
        for i, (df, name) in enumerate([(cent, "Centroid"), (probe, "Probe")]):
            ax = axes[i, j]
            data = []
            labels = []
            for t in tiers:
                bs = exp[(exp.ground_truth_species == sp) & (exp.tier == t)].basename
                s = df[df.basename.isin(bs)].score.values
                if len(s) >= 2:
                    data.append(s); labels.append(f"{t}\nn={len(s)}")
            if data:
                ax.violinplot(data, showmeans=True, showmedians=False)
            ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
            if j == 0: ax.set_ylabel(f"{name} score")
            if i == 0: ax.set_title(SHORT[sp])
    fig.suptitle("Filter score by LLM tier, on 150-image expert sample", y=1.02)
    fig.savefig(OUT / "score_violins_by_expert_tier.png")
    fig.savefig(OUT / "score_violins_by_expert_tier.pdf")
    plt.close(fig)


def plot_A14_probe_calibration(probe, expert):
    m = expert.merge(probe[["basename", "score"]], on="basename")
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), sharey=True)
    for ax, sp in zip(axes, RARE):
        sub = m[m.ground_truth_species == sp]
        binid = np.digitize(sub.score, bins) - 1
        rows = []
        for b in range(10):
            subb = sub[binid == b]
            if len(subb) == 0: continue
            rows.append((bins[b] + 0.05, subb.expert_strict.mean(), len(subb)))
        if not rows: continue
        xs, ys, ns = zip(*rows)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.scatter(xs, ys, s=[max(20, n * 15) for n in ns], c=COLORS[sp])
        ax.set_xlabel("Predicted probe prob"); ax.set_ylabel("Observed expert-strict rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_title(SHORT[sp])
    fig.suptitle("Probe calibration reliability (150-image sample, size ∝ bin count)", y=1.02)
    fig.savefig(OUT / "probe_calibration_reliability.png")
    fig.savefig(OUT / "probe_calibration_reliability.pdf")
    plt.close(fig)


def plot_A19_llm_vs_expert_morph(expert):
    fig, ax = plt.subplots(figsize=(6, 6))
    for sp in RARE:
        sub = expert[expert.ground_truth_species == sp]
        ax.scatter(sub.llm_morph_mean, sub.expert_morph_mean, s=30, alpha=0.7, c=COLORS[sp], label=SHORT[sp])
    ax.plot([1, 5], [1, 5], "k--", alpha=0.5)
    ax.axvline(4, color="red", ls=":", lw=0.8, alpha=0.5)
    ax.axhline(4, color="red", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlabel("LLM morph-mean (0–5)")
    ax.set_ylabel("Expert morph-mean (1–5)")
    ax.set_xlim(0, 5.2); ax.set_ylim(0, 5.2)
    mae = (expert.llm_morph_mean - expert.expert_morph_mean).abs().mean()
    ax.set_title(f"LLM vs expert morph-mean on 150-image sample (MAE={mae:.2f})")
    ax.legend(); ax.set_aspect("equal")
    fig.savefig(OUT / "llm_morph_vs_expert_morph.png")
    fig.savefig(OUT / "llm_morph_vs_expert_morph.pdf")
    plt.close(fig)


def plot_B2_B3_disagree_filter_pairs(cent, probe, llm):
    sel_c = _selection_basenames_cent(cent)
    sel_p = _selection_basenames_probe(probe)
    sel_l = _selection_basenames_llm(llm)
    pairs = [("LLM", sel_l, "Probe", sel_p, "llm_vs_probe"),
             ("Centroid", sel_c, "Probe", sel_p, "centroid_vs_probe")]
    for sp in RARE:
        for an, asel, bn, bsel, tag in pairs:
            a_not_b = list((asel[sp] - bsel[sp]))[:6]
            b_not_a = list((bsel[sp] - asel[sp]))[:6]
            rows = [[(_resolve_img(b, sp), b.split("::")[1]) for b in a_not_b],
                    [(_resolve_img(b, sp), b.split("::")[1]) for b in b_not_a]]
            _render_grid(rows, col_titles=None,
                         row_titles=[f"{an} pass, {bn} fail", f"{bn} pass, {an} fail"],
                         save_path=GRID_OUT / f"grid_{FSLUG[sp]}_disagree_{tag}.png")


def plot_B11_B12_top_bottom(cent, probe):
    for df, tag in [(cent, "centroid"), (probe, "probe")]:
        for sp in RARE:
            sub = df[df.species == sp].sort_values("score", ascending=False)
            top = sub.head(8); bot = sub.tail(8)
            rows = [[(_resolve_img(b, sp), f"{s:.2f}") for b, s in zip(top.basename, top.score)],
                    [(_resolve_img(b, sp), f"{s:.2f}") for b, s in zip(bot.basename, bot.score)]]
            _render_grid(rows, col_titles=None,
                         row_titles=[f"TOP-8 {tag}", f"BOTTOM-8 {tag}"],
                         save_path=GRID_OUT / f"grid_{FSLUG[sp]}_top_bottom_{tag}.png")


# =====================================================================

def main():
    import sys
    sys.path.insert(0, str(ROOT))
    _style()
    cent = load_centroid()
    probe = load_probe()
    llm = load_llm()
    expert = load_expert()
    print(f"Loaded: centroid={len(cent)} probe={len(probe)} llm={len(llm)} expert={len(expert)}")

    # Tier 1
    print("-> A1 venn"); plot_A1_venn(cent, probe, llm)
    print("-> A2 scatter"); plot_A2_scatter(cent, probe, llm, expert)
    print("-> A4 coverage"); plot_A4_expert_coverage(cent, probe, llm, expert)
    print("-> A5 feature heatmap"); plot_A5_feature_heatmap()
    print("-> A6 2x2"); plot_A6_llm_vs_expert_2x2(llm, expert)
    print("-> A7 ROC LOOCV"); plot_A7_roc_loocv(expert)
    print("-> A9 config ablation"); plot_A9_probe_config_ablation()
    print("-> A15 funnel"); plot_A15_filter_funnel(cent, probe, llm)
    print("-> A20 failure modes"); plot_A20_failure_modes(expert)
    print("-> B1 4x4 grid"); plot_B1_per_species_4x4(cent, probe, llm)
    print("-> B5 failure galleries"); plot_B5_failure_mode_gallery(expert)
    print("-> B6 probe boundary"); plot_B6_probe_boundary(probe)
    print("-> B14/B15 disagree LLM↔expert"); plot_B14_B15_disagree_llm_expert(expert, llm)

    # Tier 2
    print("-> A3 violins"); plot_A3_score_violins_by_tier(cent, probe, expert)
    print("-> A14 calibration"); plot_A14_probe_calibration(probe, expert)
    print("-> A19 llm vs expert morph"); plot_A19_llm_vs_expert_morph(expert)
    print("-> B2/B3 filter pair disagreements"); plot_B2_B3_disagree_filter_pairs(cent, probe, llm)
    print("-> B11/B12 top/bottom"); plot_B11_B12_top_bottom(cent, probe)

    print("DONE")


if __name__ == "__main__":
    main()
