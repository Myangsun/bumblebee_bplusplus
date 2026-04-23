#!/usr/bin/env python3
"""
Stage B' — feature-input ablation for the expert-calibrated probe.

Trains the LinearProbeFilter under four feature configurations:

    bioclip               (512 dim)
    llm                   (8 dim)
    bioclip+llm           (520 dim)
    bioclip+llm+species   (523 dim)

For each config, reports:
  - LOOCV AUC-ROC under lenient and strict expert labels
  - chosen C (regularisation)
  - per-species F1-maximising threshold (strict rule)
  - LOOCV precision/recall/F1 at that threshold, per species

Writes a markdown summary to
    RESULTS/filters/probe_config_ablation.md
and a JSON version to
    RESULTS/filters/probe_config_ablation.json

The config with the highest mean per-species F1 under the strict rule
is marked as the recommended probe; this recommendation feeds Stage C'.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.filters import (RARE_SPECIES, LinearProbeFilter,
                                        align_synthetic_cache,
                                        build_feature_matrix,
                                        load_expert_labels)

CONFIGS = ("bioclip", "llm", "bioclip+llm", "bioclip+llm+species")


def _fit_one(bioclip_all, basenames_all, species_all, expert, judge_json, cfg,
              train_mask):
    """Fit one LinearProbeFilter under ``cfg`` and return its state."""
    X_all, labels = build_feature_matrix(
        bioclip_all, basenames_all, species_all, judge_json, cfg,
    )
    X_train = X_all[train_mask]
    train_basenames = [b for b, m in zip(basenames_all, train_mask) if m]
    train_species = [s for s, m in zip(species_all, train_mask) if m]

    y_lenient = np.array([expert.basename_to_lenient[b] for b in train_basenames], dtype=int)
    y_strict = np.array([expert.basename_to_strict[b] for b in train_basenames], dtype=int)

    probe = LinearProbeFilter(rule="strict", feature_config=cfg)
    probe.fit(X_train, y_lenient, y_strict,
              basenames=train_basenames, species=train_species)
    return probe, X_all, labels


def main():
    expert_csv = RESULTS_DIR / "expert_validation_results" / "jessie_all_150.csv"
    judge_json = PROJECT_ROOT / "RESULTS_kfold" / "llm_judge_eval" / "results.json"
    synth_cache = RESULTS_DIR / "embeddings" / "bioclip_synthetic.npz"
    out_dir = RESULTS_DIR / "filters"
    out_dir.mkdir(parents=True, exist_ok=True)

    expert = load_expert_labels(expert_csv)
    bioclip_all, basenames_all, species_all = align_synthetic_cache(synth_cache)
    train_mask = np.array([b in expert.basename_to_strict for b in basenames_all])

    results: dict = {"configs": {}}
    for cfg in CONFIGS:
        print(f"Fitting config: {cfg}")
        probe, X_all, labels = _fit_one(bioclip_all, basenames_all, species_all,
                                         expert, judge_json, cfg, train_mask)
        entry = {
            "feature_dim": X_all.shape[1],
            "chosen_C": probe.chosen_c,
            "loocv_auc_lenient": probe.loocv_auc_lenient,
            "loocv_auc_strict": probe.loocv_auc_strict,
            "per_species_threshold_strict": probe.per_species_threshold_strict,
            "per_species_f1_strict": probe.per_species_f1_strict,
            "per_species_threshold_lenient": probe.per_species_threshold_lenient,
            "per_species_f1_lenient": probe.per_species_f1_lenient,
        }
        results["configs"][cfg] = entry

    # Pick recommended config: mean per-species F1 under strict rule (highest wins)
    def _mean_f1(entry):
        vals = [v for v in entry["per_species_f1_strict"].values()
                if isinstance(v, (int, float)) and not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    ranked = sorted(results["configs"].items(),
                    key=lambda kv: -_mean_f1(kv[1]))
    best_cfg = ranked[0][0]
    results["recommended_config"] = best_cfg
    results["ranking_by_mean_f1_strict"] = [
        {"config": k, "mean_f1_strict": _mean_f1(v),
         "loocv_auc_strict": v["loocv_auc_strict"]}
        for k, v in ranked
    ]

    (out_dir / "probe_config_ablation.json").write_text(json.dumps(results, indent=2))

    md: list[str] = []
    md.append("# Stage B′ — Probe feature-input ablation\n")
    md.append("Comparison of four probe feature configurations, each trained on the "
              "150 expert-annotated synthetics with nested 5-fold stratified CV "
              "over C ∈ {0.001, 0.01, 0.1, 1, 10}. LOOCV AUC is reported for both "
              "lenient and strict expert labels. Per-species F1-maximising "
              "thresholds are learned from the LOOCV predictions under the strict "
              "rule.\n")

    md.append("## Summary — LOOCV AUC-ROC\n")
    md.append("| Config | dim | chosen C | AUC lenient | AUC strict |")
    md.append("|---|---:|---:|---:|---:|")
    for cfg in CONFIGS:
        e = results["configs"][cfg]
        md.append(
            f"| {cfg} | {e['feature_dim']} | {e['chosen_C']} | "
            f"{e['loocv_auc_lenient']:.3f} | {e['loocv_auc_strict']:.3f} |"
        )

    md.append("\n## Per-species F1 at F1-max threshold (strict rule)\n")
    md.append("| Config | ashtoni τ (F1) | sandersoni τ (F1) | flavidus τ (F1) | mean F1 |")
    md.append("|---|---|---|---|---:|")
    for cfg in CONFIGS:
        e = results["configs"][cfg]
        row = f"| {cfg} |"
        for sp in RARE_SPECIES:
            tau = e["per_species_threshold_strict"].get(sp, 0.5)
            f1 = e["per_species_f1_strict"].get(sp, float("nan"))
            row += f" {tau:.2f} ({f1:.2f}) |"
        row += f" {_mean_f1(e):.3f} |"
        md.append(row)

    md.append("\n## Per-species F1 at F1-max threshold (lenient rule)\n")
    md.append("| Config | ashtoni τ (F1) | sandersoni τ (F1) | flavidus τ (F1) |")
    md.append("|---|---|---|---|")
    for cfg in CONFIGS:
        e = results["configs"][cfg]
        row = f"| {cfg} |"
        for sp in RARE_SPECIES:
            tau = e["per_species_threshold_lenient"].get(sp, 0.5)
            f1 = e["per_species_f1_lenient"].get(sp, float("nan"))
            row += f" {tau:.2f} ({f1:.2f}) |"
        md.append(row)

    md.append("\n## Recommendation\n")
    md.append(
        f"The configuration with the highest mean per-species F1 under the "
        f"strict rule is **`{best_cfg}`** "
        f"(mean F1 = {_mean_f1(results['configs'][best_cfg]):.3f}, LOOCV AUC "
        f"strict = {results['configs'][best_cfg]['loocv_auc_strict']:.3f}).\n"
    )
    md.append(
        "Ranked mean F1 (strict) across configurations:\n"
    )
    md.append("| Rank | Config | mean F1 (strict) | LOOCV AUC (strict) |")
    md.append("|---:|---|---:|---:|")
    for i, row in enumerate(results["ranking_by_mean_f1_strict"], 1):
        md.append(f"| {i} | {row['config']} | {row['mean_f1_strict']:.3f} | "
                  f"{row['loocv_auc_strict']:.3f} |")

    (out_dir / "probe_config_ablation.md").write_text("\n".join(md) + "\n")
    print(f"\nRecommended: {best_cfg}")
    print(f"Wrote {out_dir/'probe_config_ablation.md'}")


if __name__ == "__main__":
    main()
