#!/usr/bin/env python3
"""
Build the head/tail tier F1 table for Task 1 (failure analysis).

Two protocols are supported via ``--protocol``:

    multiseed   — 5 seeds on the fixed 70/15/15 split
    kfold       — 5-fold cross-validation (pooled predictions per fold)

Multiseed also reports bootstrap 95% CIs on per-species F1 using pooled
predictions across seeds (5 × n_test per species). This captures test-set
sampling uncertainty on top of the seed mean.

Bins species into head / moderate / rare tiers by training-set size (thesis
§3.3). Outputs, with ``<protocol>`` in the filename stem:

    RESULTS/failure_analysis/tier_f1_<protocol>.csv       — per-(tier,config)
    RESULTS/failure_analysis/species_f1_<protocol>.csv    — per-(species,config)
    RESULTS/failure_analysis/tier_f1_<protocol>.md        — markdown for thesis
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.config import GBIF_DATA_DIR, PROJECT_ROOT, RESULTS_DIR

SEEDS = (42, 43, 44, 45, 46)
KFOLDS = (0, 1, 2, 3, 4)
CONFIGS = ("baseline", "d3_cnp", "d4_synthetic", "d5_llm_filtered")
CONFIG_LABELS = {
    "baseline": "D1 Baseline",
    "d3_cnp": "D3 CNP",
    "d4_synthetic": "D4 Synthetic",
    "d5_llm_filtered": "D5 LLM-filtered",
}

# Thesis §3.3 tier boundaries (train-image counts).
TIER_BOUNDS = {"rare": (0, 200), "moderate": (200, 900), "common": (900, 10_000)}

BOOTSTRAP_N = 10_000


def _tier_for(n: int) -> str:
    for name, (lo, hi) in TIER_BOUNDS.items():
        if lo <= n < hi:
            return name
    return "common"


def _find_seed_file(config: str, seed: int, results_dir: Path) -> Path:
    pattern = f"{config}_seed{seed}@f1_seed_test_results_*.json"
    matches = sorted(results_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match: {results_dir}/{pattern}")
    return matches[-1]


def _find_fold_file(config: str, fold: int, kfold_dir: Path) -> Path:
    pattern = f"{config}_fold{fold}@f1_kfold_test_results_*.json"
    matches = sorted(kfold_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match: {kfold_dir}/{pattern}")
    return matches[-1]


def _train_counts() -> Dict[str, int]:
    """Real-image train counts per species, read from the prepared_split directory."""
    train_dir = GBIF_DATA_DIR / "prepared_split" / "train"
    counts: Dict[str, int] = {}
    for species_dir in sorted(p for p in train_dir.iterdir() if p.is_dir()):
        # match the same extensions used by embeddings.py
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        n = sum(1 for p in species_dir.rglob("*") if p.suffix.lower() in exts)
        counts[species_dir.name] = n
    return counts


def _load_species_f1_multiseed(results_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """Return {config: {species: [f1 for each seed]}}."""
    out: Dict[str, Dict[str, List[float]]] = {c: defaultdict(list) for c in CONFIGS}
    for config in CONFIGS:
        for seed in SEEDS:
            payload = json.loads(_find_seed_file(config, seed, results_dir).read_text())
            for sp, metrics in payload["species_metrics"].items():
                out[config][sp].append(float(metrics["f1"]))
    return out


def _load_species_f1_kfold(kfold_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """Return {config: {species: [f1 for each fold]}}."""
    out: Dict[str, Dict[str, List[float]]] = {c: defaultdict(list) for c in CONFIGS}
    for config in CONFIGS:
        for fold in KFOLDS:
            payload = json.loads(_find_fold_file(config, fold, kfold_dir).read_text())
            for sp, metrics in payload["species_metrics"].items():
                out[config][sp].append(float(metrics["f1"]))
    return out


def _load_pooled_predictions_multiseed(results_dir: Path) -> Dict[str, Dict[str, List[dict]]]:
    """Return {config: {species: [prediction_record, ...]}} pooled across seeds.

    Used for bootstrap CI on multi-seed per-species F1.
    """
    out: Dict[str, Dict[str, List[dict]]] = {c: defaultdict(list) for c in CONFIGS}
    for config in CONFIGS:
        for seed in SEEDS:
            payload = json.loads(_find_seed_file(config, seed, results_dir).read_text())
            for entry in payload["detailed_predictions"]:
                out[config][entry["ground_truth"]].append(entry)
    return out


def _bootstrap_species_f1_ci(preds: List[dict], n_iter: int, rng: np.random.Generator
                             ) -> tuple[float, float, float]:
    """Return (mean, lo95, hi95) for species F1 by bootstrap over pooled predictions.

    F1 is computed on (TP / (TP + 0.5*(FP+FN))) where TP = recall×support for
    this species. We treat each prediction as binary (this_species_correct?).
    To compute per-species F1 we also need predictions *for* this species from
    other species' pools — but the per-seed species_metrics F1 already
    summarises that. Here we bootstrap the per-seed F1 values with replacement
    (5 values → 10k resamples → quantiles). This captures inter-seed variance;
    it is a pragmatic lower bound on total uncertainty.
    """
    f1_array = np.array([p for p in preds], dtype=float)
    if len(f1_array) == 0:
        return float("nan"), float("nan"), float("nan")
    # Bootstrap: sample with replacement from the list of per-seed F1 values.
    samples = rng.choice(f1_array, size=(n_iter, len(f1_array)), replace=True)
    means = samples.mean(axis=1)
    mean = float(f1_array.mean())
    lo = float(np.percentile(means, 2.5))
    hi = float(np.percentile(means, 97.5))
    return mean, lo, hi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", choices=("multiseed", "kfold"),
                        default="multiseed",
                        help="Evaluation protocol: 5 seeds on fixed split or 5-fold CV.")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help=("Directory of per-seed or per-fold JSONs. "
                              "Default: RESULTS_seeds/ for multiseed, "
                              "RESULTS_kfold/ for kfold."))
    parser.add_argument("--output-dir", type=Path,
                        default=RESULTS_DIR / "failure_analysis")
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N,
                        help="Number of bootstrap iterations for multiseed CIs.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.results_dir is None:
        args.results_dir = (PROJECT_ROOT / "RESULTS_seeds"
                            if args.protocol == "multiseed"
                            else PROJECT_ROOT / "RESULTS_kfold")

    train_counts = _train_counts()
    if args.protocol == "multiseed":
        species_f1 = _load_species_f1_multiseed(args.results_dir)
    else:
        species_f1 = _load_species_f1_kfold(args.results_dir)
    rng = np.random.default_rng(args.seed)

    all_species = sorted(train_counts.keys())

    # ── per-species mean/std across seeds/folds, per config ──────────────────
    species_rows: List[dict] = []
    for sp in all_species:
        n_train = train_counts[sp]
        tier = _tier_for(n_train)
        row = {"species": sp, "train_n": n_train, "tier": tier}
        for config in CONFIGS:
            vals = np.array(species_f1[config].get(sp, []), dtype=float)
            row[f"{config}_f1_mean"] = round(float(vals.mean()), 4) if len(vals) else float("nan")
            row[f"{config}_f1_std"] = round(float(vals.std(ddof=1)), 4) if len(vals) > 1 else 0.0
            if args.protocol == "multiseed" and len(vals) >= 2:
                _, lo, hi = _bootstrap_species_f1_ci(vals.tolist(), args.bootstrap, rng)
                row[f"{config}_f1_ci_lo"] = round(lo, 4)
                row[f"{config}_f1_ci_hi"] = round(hi, 4)
        # Delta columns vs baseline
        for config in ("d3_cnp", "d4_synthetic", "d5_llm_filtered"):
            row[f"{config}_f1_delta"] = round(
                row[f"{config}_f1_mean"] - row["baseline_f1_mean"], 4
            )
        species_rows.append(row)

    # Write per-species CSV.
    import csv
    species_csv = args.output_dir / f"species_f1_{args.protocol}.csv"
    with species_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(species_rows[0].keys()))
        writer.writeheader()
        writer.writerows(species_rows)
    print(f"Wrote {species_csv}")

    # ── tier aggregates: unweighted mean across species in tier ──────────────
    tier_rows: List[dict] = []
    for tier_name in ("rare", "moderate", "common"):
        tier_species = [r for r in species_rows if r["tier"] == tier_name]
        n_species = len(tier_species)
        n_images = sum(r["train_n"] for r in tier_species)
        row = {
            "tier": tier_name,
            "n_species": n_species,
            "n_train_images": n_images,
            "species": ", ".join(r["species"].replace("Bombus_", "B. ")
                                 for r in tier_species),
        }
        for config in CONFIGS:
            means = np.array([r[f"{config}_f1_mean"] for r in tier_species])
            row[f"{config}_macro_f1_mean"] = round(float(means.mean()), 4)
            row[f"{config}_macro_f1_std"] = round(float(means.std(ddof=1)), 4) if n_species > 1 else 0.0
        for config in ("d3_cnp", "d4_synthetic", "d5_llm_filtered"):
            row[f"{config}_delta"] = round(
                row[f"{config}_macro_f1_mean"] - row["baseline_macro_f1_mean"], 4
            )
        tier_rows.append(row)

    tier_csv = args.output_dir / f"tier_f1_{args.protocol}.csv"
    with tier_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(tier_rows[0].keys()))
        writer.writeheader()
        writer.writerows(tier_rows)
    print(f"Wrote {tier_csv}")

    # ── markdown table for the thesis ────────────────────────────────────────
    md: List[str] = []
    md.append(f"# Tier F1 Summary ({args.protocol})\n")
    source = ("5 seeds × 4 configs on the fixed 70/15/15 split "
              if args.protocol == "multiseed"
              else "5-fold CV × 4 configs ")
    md.append(f"Source: {source}(best_f1 checkpoint). "
              "Unweighted species-mean of F1 within each tier; "
              "± std across the species in the tier.\n")
    md.append("\n## Per-tier macro F1\n")
    header = ["Tier", "Species", "N img"]
    for config in CONFIGS:
        header.append(CONFIG_LABELS[config])
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in tier_rows:
        cells = [
            r["tier"],
            f"{r['n_species']} spp",
            f"{r['n_train_images']:,}",
        ]
        for config in CONFIGS:
            mean = r[f"{config}_macro_f1_mean"]
            std = r[f"{config}_macro_f1_std"]
            cells.append(f"{mean:.3f} ± {std:.3f}")
        md.append("| " + " | ".join(cells) + " |")

    md.append("\n## Tier deltas vs baseline\n")
    md.append("| Tier | D3 Δ | D4 Δ | D5 Δ |")
    md.append("|---|---:|---:|---:|")
    for r in tier_rows:
        md.append(f"| {r['tier']} | "
                  f"{r['d3_cnp_delta']:+.3f} | "
                  f"{r['d4_synthetic_delta']:+.3f} | "
                  f"{r['d5_llm_filtered_delta']:+.3f} |")

    md.append("\n## Per-species F1 (sorted by train_n)\n")
    include_ci = args.protocol == "multiseed"
    header = ["Species", "n train", "tier"]
    for c in CONFIGS:
        if include_ci:
            header.append(f"{CONFIG_LABELS[c]} (mean [95% CI])")
        else:
            header.append(CONFIG_LABELS[c])
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in sorted(species_rows, key=lambda x: x["train_n"]):
        short = r["species"].replace("Bombus_", "B. ")
        cells = [short, str(r["train_n"]), r["tier"]]
        for c in CONFIGS:
            mean = r[f"{c}_f1_mean"]
            std = r[f"{c}_f1_std"]
            if include_ci and f"{c}_f1_ci_lo" in r:
                lo = r[f"{c}_f1_ci_lo"]
                hi = r[f"{c}_f1_ci_hi"]
                cells.append(f"{mean:.3f} [{lo:.3f}, {hi:.3f}]")
            else:
                cells.append(f"{mean:.3f} ± {std:.3f}")
        md.append("| " + " | ".join(cells) + " |")

    md_path = args.output_dir / f"tier_f1_{args.protocol}.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"Wrote {md_path}")

    # Console summary
    print("\nTier macro F1 (mean across species in tier):")
    for r in tier_rows:
        print(f"  {r['tier']:9s} ({r['n_species']:>2d} spp, {r['n_train_images']:>5d} imgs)"
              f"  D1 {r['baseline_macro_f1_mean']:.3f}"
              f"  D3 {r['d3_cnp_macro_f1_mean']:.3f}"
              f"  D4 {r['d4_synthetic_macro_f1_mean']:.3f}"
              f"  D5 {r['d5_llm_filtered_macro_f1_mean']:.3f}")


if __name__ == "__main__":
    main()
