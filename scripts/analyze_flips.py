#!/usr/bin/env python3
"""
Per-image prediction-flip analysis across multi-seed training runs.

For each test image, track whether each of the 5 seeds of each configuration
(baseline / d3_cnp / d4_synthetic / d5_llm_filtered) classifies it correctly.
Derive a ``flip category`` for each augmented config relative to the baseline:

    stable-correct   : baseline majority-correct AND aug majority-correct
    stable-wrong     : baseline majority-wrong   AND aug majority-wrong
    improved         : baseline majority-wrong   AND aug majority-correct
    harmed           : baseline majority-correct AND aug majority-wrong

Writes two artefacts to ``RESULTS/failure_analysis/``:

    flip_analysis.csv       -- one row per test image with per-seed booleans,
                               majority verdicts, correctness rates, most
                               common prediction per config, and flip
                               categories for d3/d4/d5.
    flip_summary.json       -- aggregate counts (total and per-species) for
                               each flip category.

No new training is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import PROJECT_ROOT, RESULTS_DIR

SEEDS = (42, 43, 44, 45, 46)
CONFIGS = ("baseline", "d3_cnp", "d4_synthetic", "d5_llm_filtered", "d2_centroid", "d6_probe")
AUG_CONFIGS = ("d3_cnp", "d4_synthetic", "d5_llm_filtered", "d2_centroid", "d6_probe")
CATEGORIES = ("stable-correct", "stable-wrong", "improved", "harmed")

RESULTS_SEEDS_DIR = PROJECT_ROOT / "RESULTS_seeds"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "failure_analysis"


# ── File discovery ────────────────────────────────────────────────────────────


def _find_test_result(config: str, seed: int) -> Path:
    """Locate the per-seed test-result JSON (pattern {config}_seed{seed}@f1_*.json).

    Searches RESULTS_seeds/ first (canonical for D1-D4) then RESULTS/ (used by
    the newer D5 d2_centroid and D6 d6_probe evaluations)."""
    pattern = f"{config}_seed{seed}@f1_seed_test_results_*.json"
    matches = sorted(RESULTS_SEEDS_DIR.glob(pattern))
    if not matches:
        matches = sorted((PROJECT_ROOT / "RESULTS").glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No match for {pattern} in RESULTS_seeds/ or RESULTS/")
    return matches[-1]  # newest timestamp wins when multiple exist


def _normalize_path(p: str) -> str:
    """Return a stable identifier for a test image, independent of the
    prepared_* root. Each config tests on its own prepared directory, but the
    test split (test/<species>/<filename>) is identical across them."""
    path = Path(p)
    parts = path.parts
    if "test" in parts:
        idx = parts.index("test")
        return "/".join(parts[idx:])  # "test/<species>/<filename>"
    return "/".join(parts[-3:])


# ── Loading ───────────────────────────────────────────────────────────────────


def load_seed_predictions(config: str, seed: int) -> Dict[str, dict]:
    """Return {normalized_path: prediction_record} for one (config, seed)."""
    path = _find_test_result(config, seed)
    with open(path, "r") as fh:
        payload = json.load(fh)
    out: Dict[str, dict] = {}
    for entry in payload["detailed_predictions"]:
        key = _normalize_path(entry["image_path"])
        out[key] = entry
    return out


def load_all_predictions() -> Dict[Tuple[str, int], Dict[str, dict]]:
    """Load every (config, seed) result. Keyed by (config, seed) → path → record."""
    table: Dict[Tuple[str, int], Dict[str, dict]] = {}
    for config in CONFIGS:
        for seed in SEEDS:
            table[(config, seed)] = load_seed_predictions(config, seed)
    return table


# ── Per-image aggregation ─────────────────────────────────────────────────────


def _majority(booleans: Sequence[bool]) -> bool:
    return sum(booleans) > len(booleans) / 2


def _most_common(values: Iterable[str]) -> str:
    counter = Counter(values)
    return counter.most_common(1)[0][0]


def _rate(booleans: Sequence[bool]) -> float:
    return sum(booleans) / len(booleans) if booleans else float("nan")


def _classify_flip(baseline_majority: bool, aug_majority: bool) -> str:
    if baseline_majority and aug_majority:
        return "stable-correct"
    if not baseline_majority and not aug_majority:
        return "stable-wrong"
    if not baseline_majority and aug_majority:
        return "improved"
    return "harmed"


def build_flip_rows(table: Dict[Tuple[str, int], Dict[str, dict]]) -> List[dict]:
    """Produce one output row per test image."""
    # Use baseline seed 42 as the canonical image list.
    canonical = table[("baseline", 42)]

    rows: List[dict] = []
    for path, rec in sorted(canonical.items()):
        true_species = rec["ground_truth"]

        row: Dict[str, object] = {"image_path": path, "true_species": true_species}

        # Per-seed correctness and per-config correctness rates / majority / mode prediction.
        config_correctness: Dict[str, List[bool]] = {}
        config_predictions: Dict[str, List[str]] = {}
        for config in CONFIGS:
            bools: List[bool] = []
            preds: List[str] = []
            for seed in SEEDS:
                entry = table[(config, seed)].get(path)
                if entry is None:
                    raise KeyError(f"Missing prediction for {path} in {config} seed {seed}")
                bools.append(bool(entry["correct"]))
                preds.append(entry["prediction"])
                row[f"{config}_seed{seed}_correct"] = int(entry["correct"])
            config_correctness[config] = bools
            config_predictions[config] = preds
            row[f"{config}_correct_rate"] = round(_rate(bools), 4)
            row[f"{config}_majority_correct"] = int(_majority(bools))
            row[f"{config}_mode_pred"] = _most_common(preds)

        baseline_maj = _majority(config_correctness["baseline"])
        for aug in AUG_CONFIGS:
            aug_maj = _majority(config_correctness[aug])
            row[f"category_{aug}"] = _classify_flip(baseline_maj, aug_maj)

        rows.append(row)

    return rows


# ── Summaries ────────────────────────────────────────────────────────────────


def summarize(rows: List[dict]) -> dict:
    summary: Dict[str, object] = {
        "n_test_images": len(rows),
        "configs": list(CONFIGS),
        "aug_configs": list(AUG_CONFIGS),
        "seeds": list(SEEDS),
    }

    # Overall category counts.
    overall: Dict[str, Dict[str, int]] = {}
    for aug in AUG_CONFIGS:
        counts = Counter(r[f"category_{aug}"] for r in rows)
        overall[aug] = {cat: int(counts.get(cat, 0)) for cat in CATEGORIES}
    summary["overall_categories"] = overall

    # Per-species breakdown.
    per_species: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: {aug: {cat: 0 for cat in CATEGORIES} for aug in AUG_CONFIGS}
    )
    for r in rows:
        species = r["true_species"]
        for aug in AUG_CONFIGS:
            per_species[species][aug][r[f"category_{aug}"]] += 1
    summary["per_species_categories"] = {
        sp: counts for sp, counts in sorted(per_species.items())
    }

    # Mean correctness rates overall / per species.
    def _mean(values: Sequence[float]) -> float:
        return sum(values) / len(values) if values else float("nan")

    overall_rates = {
        config: round(_mean([r[f"{config}_correct_rate"] for r in rows]), 4)
        for config in CONFIGS
    }
    summary["overall_mean_correct_rate"] = overall_rates

    per_species_rates: Dict[str, Dict[str, float]] = {}
    species_groups: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        species_groups[r["true_species"]].append(r)
    for sp, group in sorted(species_groups.items()):
        per_species_rates[sp] = {
            "n": len(group),
            **{config: round(_mean([r[f"{config}_correct_rate"] for r in group]), 4)
               for config in CONFIGS},
        }
    summary["per_species_mean_correct_rate"] = per_species_rates

    return summary


# ── Output ────────────────────────────────────────────────────────────────────


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        raise RuntimeError("No rows to write")
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-image prediction-flip analysis.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    table = load_all_predictions()

    # Sanity: every config must cover the same image set.
    reference_paths = set(table[("baseline", 42)].keys())
    for (config, seed), entries in table.items():
        missing = reference_paths - entries.keys()
        extra = entries.keys() - reference_paths
        if missing or extra:
            raise RuntimeError(
                f"Mismatched image set for ({config}, {seed}): "
                f"{len(missing)} missing, {len(extra)} extra"
            )
    print(f"Loaded {len(reference_paths)} test images "
          f"× {len(CONFIGS)} configs × {len(SEEDS)} seeds")

    rows = build_flip_rows(table)
    summary = summarize(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "flip_analysis.csv"
    json_path = args.output_dir / "flip_summary.json"
    write_csv(rows, csv_path)
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print("\nOverall categories:")
    for aug, cats in summary["overall_categories"].items():
        pieces = "  ".join(f"{k}={v}" for k, v in cats.items())
        print(f"  {aug}: {pieces}")


if __name__ == "__main__":
    main()
