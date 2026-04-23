#!/usr/bin/env python3
"""
Stage A' — rigorous analysis of the 150 expert-annotated synthetic images
before the probe is implemented.

Outputs (under ``RESULTS/filters/label_analysis_150/``):

  summary.md                 — prose + tables for the thesis
  summary.json               — machine-readable version of the same
  per_feature_disagreement.csv
  bioclip_2d_lenient.png     — UMAP/PCA of 150 BioCLIP embeddings, coloured
  bioclip_2d_strict.png        by expert lenient / strict pass respectively
  feature_correlations.png   — correlation of each LLM feature with each
                                expert feature score and with expert pass
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.filters import (RARE_SPECIES, align_synthetic_cache,
                                        load_expert_labels, load_llm_judge)

FEATURES = (
    "legs_appendages",
    "wing_venation_texture",
    "head_antennae",
    "abdomen_banding",
    "thorax_coloration",
)


def _per_species_tier_pass(csv_path: Path) -> dict:
    """Per-species × LLM tier × expert-pass cross-tab."""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))

    expert_lab = load_expert_labels(csv_path)
    tiers = ("strict_pass", "borderline", "soft_fail", "hard_fail")

    out: dict = {sp: {t: {"n": 0, "lenient_pass": 0, "strict_pass": 0} for t in tiers}
                 for sp in RARE_SPECIES}
    for r in rows:
        sp = r["ground_truth_species"].replace(" ", "_")
        tier = r["tier"]
        if sp not in out or tier not in out[sp]:
            continue
        basename = Path(r["image_path"]).name
        out[sp][tier]["n"] += 1
        if expert_lab.basename_to_lenient.get(basename):
            out[sp][tier]["lenient_pass"] += 1
        if expert_lab.basename_to_strict.get(basename):
            out[sp][tier]["strict_pass"] += 1
    return out


def _per_feature_agreement(csv_path: Path, judge) -> dict:
    """2x2 agreement between LLM feature score and expert feature score
    (both binarised at >= 4). Per species × per feature."""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))

    # map basename -> LLM feature scores
    per_image_llm_feats: dict[str, dict[str, float]] = {}
    with open(PROJECT_ROOT / "RESULTS_kfold" / "llm_judge_eval" / "results.json") as fh:
        raw = json.load(fh)
    for r in raw.get("results", []):
        mf = r.get("morphological_fidelity", {}) or {}
        per_image_llm_feats[r["file"]] = {
            f: mf.get(f, {}).get("score") for f in FEATURES
        }

    result: dict = {sp: {f: {"TP": 0, "FP": 0, "TN": 0, "FN": 0,
                              "expert_missing": 0, "llm_missing": 0}
                          for f in FEATURES}
                     for sp in RARE_SPECIES}
    for r in rows:
        sp = r["ground_truth_species"].replace(" ", "_")
        if sp not in result:
            continue
        basename = Path(r["image_path"]).name
        llm_feats = per_image_llm_feats.get(basename, {})
        for f in FEATURES:
            expert_col = f"morph_{f}"
            expert_v_raw = (r.get(expert_col, "") or "").strip()
            llm_v = llm_feats.get(f)
            if expert_v_raw == "":
                result[sp][f]["expert_missing"] += 1
                continue
            try:
                expert_v = float(expert_v_raw)
            except ValueError:
                result[sp][f]["expert_missing"] += 1
                continue
            if not isinstance(llm_v, (int, float)):
                result[sp][f]["llm_missing"] += 1
                continue
            llm_pos = llm_v >= 4
            expert_pos = expert_v >= 4
            if llm_pos and expert_pos:
                result[sp][f]["TP"] += 1
            elif llm_pos and not expert_pos:
                result[sp][f]["FP"] += 1
            elif not llm_pos and expert_pos:
                result[sp][f]["FN"] += 1
            else:
                result[sp][f]["TN"] += 1
    return result


def _failure_mode_distribution(csv_path: Path) -> dict:
    """Count expert-flagged failure modes per species."""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    per_species: dict = {sp: Counter() for sp in RARE_SPECIES}
    total = Counter()
    for r in rows:
        sp = r["ground_truth_species"].replace(" ", "_")
        if sp not in per_species:
            continue
        try:
            fm = json.loads(r.get("failure_modes", "") or "{}")
        except Exception:
            continue
        modes = fm.get("all", []) if isinstance(fm, dict) else []
        for m in modes:
            if m in ("species_no_failure", "quality_no_failure"):
                continue
            per_species[sp][m] += 1
            total[m] += 1
    return {"per_species": {k: dict(v) for k, v in per_species.items()},
            "total": dict(total)}


def _bioclip_2d(csv_path: Path, synth_cache_path: Path, output_dir: Path) -> dict:
    """PCA of the 150 expert-labelled BioCLIP embeddings, coloured by
    expert lenient and strict pass respectively. Returns separation
    diagnostics (mean within-class cosine, mean cross-class cosine)."""
    expert = load_expert_labels(csv_path)
    feats_all, basenames_all, species_all = align_synthetic_cache(synth_cache_path)

    mask = np.array([b in expert.basename_to_strict for b in basenames_all])
    X = feats_all[mask]
    basenames = [b for b, m in zip(basenames_all, mask) if m]
    species = [s for s, m in zip(species_all, mask) if m]
    y_lenient = np.array([expert.basename_to_lenient[b] for b in basenames], dtype=int)
    y_strict = np.array([expert.basename_to_strict[b] for b in basenames], dtype=int)

    coords = PCA(n_components=2, random_state=42).fit_transform(X)

    def _plot(coords, y, label_name, path):
        fig, ax = plt.subplots(figsize=(7, 6))
        for lbl, marker, color in ((0, "x", "#d62728"), (1, "o", "#2ca02c")):
            mask = y == lbl
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       marker=marker, c=color, s=50, alpha=0.75,
                       edgecolors="white", linewidths=0.3,
                       label=f"expert {label_name} = {bool(lbl)} (n={int(mask.sum())})")
        ax.set_title(f"BioCLIP PCA of 150 expert-labelled synthetics\n"
                     f"coloured by expert {label_name} pass")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(loc="best", fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot(coords, y_lenient, "lenient", output_dir / "bioclip_2d_lenient.png")
    _plot(coords, y_strict, "strict", output_dir / "bioclip_2d_strict.png")

    # Separation diagnostic: mean cosine similarity within pass group and
    # cross-group, at the BioCLIP level (not projected).
    def _mean_cosine(a, b):
        # features are L2-normalised already
        return float((a @ b.T).mean())

    def _sep(y):
        pos = X[y == 1]; neg = X[y == 0]
        if len(pos) < 2 or len(neg) < 2:
            return {"within_pos": float("nan"), "within_neg": float("nan"),
                    "across": float("nan")}
        return {
            "within_pos": _mean_cosine(pos, pos),
            "within_neg": _mean_cosine(neg, neg),
            "across": _mean_cosine(pos, neg),
        }

    return {
        "pca_variance_explained": PCA(n_components=2).fit(X).explained_variance_ratio_.tolist(),
        "separation_lenient": _sep(y_lenient),
        "separation_strict": _sep(y_strict),
        "n_train": int(len(X)),
    }


def _write_markdown(tier_pass, agreement, failure_modes, bioclip_stats,
                    csv_path, output_path):
    lines: list[str] = []
    lines.append("# Stage A′ — 150-label expert validation analysis\n")
    lines.append(
        f"Source: `{csv_path}` (150 synthetics annotated by one expert).\n"
        "All tables and figures in this directory are derived from this CSV "
        "without running any classifier or probe.\n"
    )

    # 1. Tier × species × expert
    lines.append("## 1. Per-species × LLM tier × expert-pass breakdown\n")
    lines.append("Rows are LLM-assigned tiers (used for stratified sampling). "
                 "Columns are how many of those images an expert accepted "
                 "under each rule.\n")
    for sp in RARE_SPECIES:
        lines.append(f"\n### {sp.replace('Bombus_', 'B. ')}\n")
        lines.append("| LLM tier | n | expert lenient pass | expert strict pass |")
        lines.append("|---|---:|---:|---:|")
        total_n = total_len = total_str = 0
        for tier in ("strict_pass", "borderline", "soft_fail", "hard_fail"):
            row = tier_pass[sp][tier]
            n = row["n"]; lp = row["lenient_pass"]; sp_ = row["strict_pass"]
            rate_l = f"{lp}/{n} ({100 * lp / n:.0f}%)" if n else "—"
            rate_s = f"{sp_}/{n} ({100 * sp_ / n:.0f}%)" if n else "—"
            lines.append(f"| {tier} | {n} | {rate_l} | {rate_s} |")
            total_n += n; total_len += lp; total_str += sp_
        lines.append(f"| **total** | **{total_n}** | "
                     f"**{total_len}/{total_n} ({100 * total_len / total_n:.0f}%)** | "
                     f"**{total_str}/{total_n} ({100 * total_str / total_n:.0f}%)** |")

    # 2. Per-feature 2x2 agreement
    lines.append("\n## 2. Per-feature LLM vs expert 2 × 2 agreement\n")
    lines.append("Each cell binarises the score at >= 4. "
                 "FN = expert ≥ 4 but LLM < 4 (LLM over-strictness); "
                 "FP = LLM ≥ 4 but expert < 4 (LLM blind spot).\n")
    for sp in RARE_SPECIES:
        lines.append(f"\n### {sp.replace('Bombus_', 'B. ')}\n")
        lines.append("| Feature | TP | FP | TN | FN | LLM blind spot | LLM over-strict |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for f in FEATURES:
            a = agreement[sp][f]
            tp, fp, tn, fn = a["TP"], a["FP"], a["TN"], a["FN"]
            bs = fp / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0
            os = fn / (tp + fp + tn + fn) if (tp + fp + tn + fn) else 0
            lines.append(f"| {f} | {tp} | {fp} | {tn} | {fn} | "
                         f"{bs:.2f} | {os:.2f} |")

    # 3. Failure modes
    lines.append("\n## 3. Expert-flagged failure modes\n")
    lines.append("| Failure mode | total | ashtoni | sandersoni | flavidus |")
    lines.append("|---|---:|---:|---:|---:|")
    for mode, total in sorted(failure_modes["total"].items(), key=lambda kv: -kv[1]):
        a = failure_modes["per_species"]["Bombus_ashtoni"].get(mode, 0)
        s = failure_modes["per_species"]["Bombus_sandersoni"].get(mode, 0)
        f_ = failure_modes["per_species"]["Bombus_flavidus"].get(mode, 0)
        lines.append(f"| {mode} | {total} | {a} | {s} | {f_} |")

    # 4. BioCLIP separation
    lines.append("\n## 4. BioCLIP feature-space separation of expert-pass vs expert-fail\n")
    lines.append("Mean cosine similarity (higher = more similar in feature space):\n")
    lines.append("| Expert rule | within pass | within fail | across |")
    lines.append("|---|---:|---:|---:|")
    for rule in ("lenient", "strict"):
        s = bioclip_stats[f"separation_{rule}"]
        lines.append(f"| {rule} | {s['within_pos']:.3f} | {s['within_neg']:.3f} | "
                     f"{s['across']:.3f} |")
    ve = bioclip_stats["pca_variance_explained"]
    lines.append(f"\n2-component PCA variance explained: {ve[0]:.2%} + {ve[1]:.2%} = "
                 f"{ve[0] + ve[1]:.2%}.\n")
    lines.append("See `bioclip_2d_lenient.png` and `bioclip_2d_strict.png` for the "
                 "2-D projection coloured by expert pass. If classes separate "
                 "visibly in PCA, a BioCLIP-input linear probe has a chance of "
                 "succeeding; if they don't, we rely on non-linear features or "
                 "on BioCLIP + LLM concatenation.\n")

    output_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {output_path}")


def main():
    expert_csv = RESULTS_DIR / "expert_validation_results" / "jessie_all_150.csv"
    judge_path = PROJECT_ROOT / "RESULTS_kfold" / "llm_judge_eval" / "results.json"
    synth_cache = RESULTS_DIR / "embeddings" / "bioclip_synthetic.npz"
    out_dir = RESULTS_DIR / "filters" / "label_analysis_150"
    out_dir.mkdir(parents=True, exist_ok=True)

    judge = load_llm_judge(judge_path)
    tier_pass = _per_species_tier_pass(expert_csv)
    agreement = _per_feature_agreement(expert_csv, judge)
    failures = _failure_mode_distribution(expert_csv)
    bioclip_stats = _bioclip_2d(expert_csv, synth_cache, out_dir)

    # Write summary.json
    payload = {
        "tier_pass_cross_tab": tier_pass,
        "per_feature_agreement": agreement,
        "failure_modes": failures,
        "bioclip_separation": bioclip_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))

    # Write per_feature_disagreement.csv (convenient for the thesis)
    with open(out_dir / "per_feature_disagreement.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "feature", "TP", "FP", "TN", "FN",
                    "llm_blind_spot_rate", "llm_over_strict_rate"])
        for sp in RARE_SPECIES:
            for f in FEATURES:
                a = agreement[sp][f]
                n = a["TP"] + a["FP"] + a["TN"] + a["FN"]
                bs = a["FP"] / n if n else 0
                os = a["FN"] / n if n else 0
                w.writerow([sp, f, a["TP"], a["FP"], a["TN"], a["FN"],
                            f"{bs:.3f}", f"{os:.3f}"])

    _write_markdown(tier_pass, agreement, failures, bioclip_stats,
                    expert_csv, out_dir / "summary.md")
    print(f"All outputs under {out_dir}")


if __name__ == "__main__":
    main()
