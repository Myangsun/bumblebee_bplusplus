#!/usr/bin/env python3
"""ROC for the LLM morph-mean as a continuous ranker against expert strict
labels on the 150-image sample (Section 5.4.2, AUC 0.56).

For each image the LLM judge produces five 1-to-5 morphological scores; the
"morph-mean" is their average over visible features. Treating that mean as
a continuous score and the expert strict label (blind-ID match ∧
diag = species ∧ expert morph-mean ≥ 4) as binary truth, we sweep the
threshold from 5 down to 1 and trace (FPR, TPR). AUC = sklearn's standard
ranker-quality metric: probability that a random expert-pass image scores
above a random expert-fail image. AUC 0.5 = random, 1.0 = perfect.

For visual context we also recompute the BioCLIP linear-probe LOOCV AUC
(Section 5.4.5). The two curves on the same axes show why the language
rubric is a near-random ranker while a feature-space probe is not.

Output: docs/plots/filters/llm_morph_mean_roc.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT  # noqa: E402

ROOT = PROJECT_ROOT
OUT = ROOT / "docs/plots/filters/llm_morph_mean_roc.png"

RARE = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")
COL = {"Bombus_ashtoni": "#0072B2",
       "Bombus_sandersoni": "#E69F00",
       "Bombus_flavidus": "#009E73"}
SHORT = {s: "B. " + s.replace("Bombus_", "") for s in RARE}


def _load() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "RESULTS/expert_validation_results/jessie_all_150.csv")
    df["ground_truth_species"] = df["ground_truth_species"].str.replace(" ", "_", regex=False)
    df["blind_id_species"]     = df["blind_id_species"].fillna("").str.replace(" ", "_", regex=False)
    M = ["morph_legs_appendages", "morph_wing_venation_texture",
         "morph_head_antennae", "morph_abdomen_banding", "morph_thorax_coloration"]
    df["expert_morph_mean"] = df[M].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df["blind_id_match"] = df["blind_id_species"] == df["ground_truth_species"]
    df["diag_species"]  = df["diagnostic_level"] == "species"
    df["expert_strict"] = (df["blind_id_match"] & df["diag_species"]
                           & (df["expert_morph_mean"] >= 4.0))
    df["basename"] = df["image_path"].str.split("/").str[-1]

    # LLM morph-mean per image.
    rows = json.load(open(ROOT / "RESULTS_kfold/llm_judge_eval/results.json"))["results"]
    llm = {}
    for r in rows:
        feats = (r.get("morphological_fidelity") or {}).values()
        s = [f["score"] for f in feats
             if isinstance(f, dict) and not f.get("not_visible", False)
             and f.get("score") is not None]
        llm[r["file"]] = sum(s) / len(s) if s else float("nan")
    df["llm_morph_mean"] = df["basename"].map(llm)
    return df.dropna(subset=["llm_morph_mean"]).reset_index(drop=True)


def _probe_loocv_probs(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Re-run the LOOCV probe on the 150 expert-labelled synthetics."""
    cache = np.load(ROOT / "RESULTS/embeddings/bioclip_synthetic.npz",
                    allow_pickle=False)
    feats_all = cache["features"]
    paths_all = cache["image_paths"]
    base_to_idx = {Path(p).name: i for i, p in enumerate(paths_all)}
    keep = df["basename"].apply(lambda b: b in base_to_idx)
    df2 = df[keep].reset_index(drop=True)
    X = np.stack([feats_all[base_to_idx[b]] for b in df2.basename])
    y = df2.expert_strict.astype(int).values
    probs = np.zeros(len(y))
    for tr, te in LeaveOneOut().split(X):
        pipe = Pipeline([("sc", StandardScaler()),
                         ("lr", LogisticRegression(C=0.01, class_weight="balanced",
                                                   solver="lbfgs", max_iter=4000,
                                                   random_state=42))])
        pipe.fit(X[tr], y[tr])
        probs[te] = pipe.predict_proba(X[te])[:, 1]
    return y, probs


def main() -> None:
    df = _load()
    fig, ax = plt.subplots(figsize=(9.5, 6.4), facecolor="white")

    # Diagonal random reference.
    ax.plot([0, 1], [0, 1], "k--", alpha=0.45, lw=0.9, label="random  (AUC 0.50)")

    # Overall LLM ROC.
    y_all = df.expert_strict.astype(int).values
    s_all = df.llm_morph_mean.values
    fpr, tpr, _ = roc_curve(y_all, s_all)
    auc_all = roc_auc_score(y_all, s_all)
    ax.plot(fpr, tpr, color="#7a4ea3", lw=2.4,
            label=f"LLM morph-mean (overall, n={len(df)})  AUC {auc_all:.3f}")

    # Per-species LLM ROC.
    for sp in RARE:
        sub = df[df.ground_truth_species == sp]
        y = sub.expert_strict.astype(int).values
        s = sub.llm_morph_mean.values
        if len(set(y)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        a = roc_auc_score(y, s)
        ax.plot(fpr, tpr, color=COL[sp], lw=1.4, ls="--", alpha=0.85,
                label=f"  {SHORT[sp]} (n={len(sub)}, pos={y.sum()})  AUC {a:.3f}")

    # BioCLIP probe LOOCV ROC for context.
    try:
        y_p, prob_p = _probe_loocv_probs(df)
        fpr_p, tpr_p, _ = roc_curve(y_p, prob_p)
        auc_p = roc_auc_score(y_p, prob_p)
        ax.plot(fpr_p, tpr_p, color="#2b7a78", lw=2.4,
                label=f"BioCLIP probe LOOCV (n={len(y_p)})  AUC {auc_p:.3f}")
    except Exception as exc:
        print(f"probe ROC unavailable: {exc}")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.set_title("LLM morph-mean as a continuous ranker of expert strict pass/fail\n"
                 "(150-image sample; AUC 0.50 = random, 1.0 = perfect)",
                 fontsize=11, loc="left")
    ax.legend(loc="center left", bbox_to_anchor=(1.03, 0.5),
              fontsize=9, frameon=False)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
