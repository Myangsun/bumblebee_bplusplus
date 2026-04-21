#!/usr/bin/env python3
"""
Task 2 Phases 2a/2c — score all 1,500 synthetic images using a chosen filter.

Supported filters
-----------------
    centroid   : unsupervised, cosine similarity to real-image species
                 centroid in BioCLIP feature space (D2 variant).
    probe      : expert-supervised LogisticRegression on BioCLIP
                 embeddings, trained on the 150 expert-annotated images
                 (D6 variant).

Outputs
-------
    RESULTS/filters/{centroid,probe}_scores.json   (per-image scores)
    RESULTS/filters/{centroid,probe}_scores.csv    (flat, easy to grep)

The JSON also carries LOOCV AUC-ROC and the chosen regularisation C
for the probe filter.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.config import PROJECT_ROOT, RESULTS_DIR
from pipeline.evaluate.embeddings import load_cache
from pipeline.evaluate.filters import (RARE_SPECIES, CentroidFilter,
                                        LinearProbeFilter,
                                        align_synthetic_cache,
                                        build_feature_matrix,
                                        load_expert_labels)


def _centroid_scores(real_cache_path: Path, synth_cache_path: Path) -> tuple[np.ndarray, list[str], list[str], dict]:
    real = load_cache(real_cache_path)
    filt = CentroidFilter().fit(real["features"], real["species"], species_list=RARE_SPECIES)
    feats, basenames, species = align_synthetic_cache(synth_cache_path)
    scores = filt.score(feats, species)
    return scores, basenames, species, {"centroids_fit_on": str(real_cache_path)}


def _probe_scores(expert_csv: Path, synth_cache_path: Path, judge_json: Path,
                  config: str, rule: str = "strict",
                  ) -> tuple[np.ndarray, list[str], list[str], dict]:
    expert = load_expert_labels(expert_csv)
    bioclip_all, basenames_all, species_all = align_synthetic_cache(synth_cache_path)

    X_all, feature_labels = build_feature_matrix(
        bioclip_all, basenames_all, species_all, judge_json, config,
    )

    train_mask = np.array([b in expert.basename_to_strict for b in basenames_all])
    X_train = X_all[train_mask]
    train_basenames = [b for b, m in zip(basenames_all, train_mask) if m]
    train_species = [s for s, m in zip(species_all, train_mask) if m]
    y_lenient = np.array([expert.basename_to_lenient[b] for b in train_basenames], dtype=int)
    y_strict = np.array([expert.basename_to_strict[b] for b in train_basenames], dtype=int)

    probe = LinearProbeFilter(rule=rule, feature_config=config)
    probe.fit(X_train, y_lenient, y_strict,
              basenames=train_basenames, species=train_species)
    scores = probe.score(X_all)

    # Pass/fail mask under per-species thresholds
    pass_flags = probe.pass_mask(X_all, species_all)

    meta = {
        "rule": rule,
        "feature_config": config,
        "feature_dim": X_all.shape[1],
        "n_train": int(len(train_basenames)),
        "chosen_C": probe.chosen_c,
        "loocv_auc_lenient": probe.loocv_auc_lenient,
        "loocv_auc_strict": probe.loocv_auc_strict,
        "per_species_threshold_strict": probe.per_species_threshold_strict,
        "per_species_threshold_lenient": probe.per_species_threshold_lenient,
        "per_species_f1_strict": probe.per_species_f1_strict,
        "per_species_f1_lenient": probe.per_species_f1_lenient,
        "pass_flags_by_basename": {
            bn: bool(pass_flags[i]) for i, bn in enumerate(basenames_all)
        },
        "expert_csv": str(expert_csv),
        "train_basenames": train_basenames,
    }
    return scores, basenames_all, species_all, meta


def _write_outputs(filter_name: str, scores: np.ndarray, basenames: list[str],
                   species: list[str], meta: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_species_rank: dict[str, list[int]] = {}
    for sp in sorted(set(species)):
        idx = [i for i, s in enumerate(species) if s == sp]
        idx.sort(key=lambda i: -float(scores[i]))
        per_species_rank[sp] = idx

    payload = {
        "filter": filter_name,
        "n_scored": int(len(scores)),
        "score_summary": {
            "overall_mean": float(scores.mean()),
            "overall_std": float(scores.std()),
            "per_species": {
                sp: {
                    "n": int(sum(1 for s in species if s == sp)),
                    "mean": float(np.mean([scores[i] for i, s in enumerate(species) if s == sp])),
                    "std": float(np.std([scores[i] for i, s in enumerate(species) if s == sp])),
                    "min": float(min(scores[i] for i, s in enumerate(species) if s == sp)),
                    "max": float(max(scores[i] for i, s in enumerate(species) if s == sp)),
                    "median": float(np.median([scores[i] for i, s in enumerate(species) if s == sp])),
                }
                for sp in sorted(set(species))
            },
        },
        "meta": meta,
        "scores": [
            {"basename": basenames[i], "species": species[i], "score": float(scores[i])}
            for i in range(len(scores))
        ],
    }

    json_path = out_dir / f"{filter_name}_scores.json"
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {json_path}")

    csv_path = out_dir / f"{filter_name}_scores.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["basename", "species", "score"])
        for i in range(len(scores)):
            w.writerow([basenames[i], species[i], f"{scores[i]:.6f}"])
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filter", choices=("centroid", "probe"), required=True)
    parser.add_argument("--real-cache", type=Path,
                        default=RESULTS_DIR / "embeddings" / "bioclip_real_train.npz")
    parser.add_argument("--synthetic-cache", type=Path,
                        default=RESULTS_DIR / "embeddings" / "bioclip_synthetic.npz")
    parser.add_argument("--expert-csv", type=Path,
                        default=RESULTS_DIR / "expert_validation_results" / "jessie_all_150.csv")
    parser.add_argument("--rule", choices=("lenient", "strict"), default="strict",
                        help="Probe training label rule (ignored for centroid filter).")
    parser.add_argument("--config",
                        default="bioclip+llm+species",
                        choices=("bioclip", "llm", "bioclip+llm", "bioclip+llm+species"),
                        help="Probe feature configuration (ignored for centroid filter).")
    parser.add_argument("--judge-json", type=Path,
                        default=PROJECT_ROOT / "RESULTS_kfold" / "llm_judge_eval" / "results.json")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "filters")
    args = parser.parse_args()

    if args.filter == "centroid":
        scores, basenames, species, meta = _centroid_scores(args.real_cache, args.synthetic_cache)
    else:  # probe
        scores, basenames, species, meta = _probe_scores(
            args.expert_csv, args.synthetic_cache, args.judge_json,
            config=args.config, rule=args.rule,
        )

    _write_outputs(args.filter, scores, basenames, species, meta, args.output_dir)


if __name__ == "__main__":
    main()
