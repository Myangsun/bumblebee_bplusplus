#!/usr/bin/env python3
"""
Task 2 Phase 2a / 2c — assemble a prepared dataset directory for a
filter-driven D2 or D6 variant.

For a given filter, the top-``per-species`` scoring synthetics are
symlinked into a ``GBIF_MA_BUMBLEBEES/prepared_<variant>/train/<species>``
tree alongside the real training images. Validation and test splits are
symlinked from ``prepared_split`` unchanged, matching the layout used
by D4 / D5.

Variants
--------
    centroid -> prepared_d2_centroid
    probe    -> prepared_d6_probe

Usage
-----
    python scripts/assemble_d6.py --variant centroid --per-species 200
    python scripts/assemble_d6.py --variant probe    --per-species 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR
from pipeline.evaluate.filters import RARE_SPECIES

VARIANT_TO_DIR = {
    "centroid": "prepared_d2_centroid",
    "probe": "prepared_d6_probe",
}
VARIANT_TO_SCORES = {
    "centroid": "centroid_scores.json",
    "probe": "probe_scores.json",
}

SYNTHETIC_SOURCE_DIR = Path("/orcd/home/002/msun14/bumblebee_bplusplus/RESULTS_kfold/synthetic_generation")


def _top_n_basenames(scores_json: Path, species: str, n: int) -> list[str]:
    payload = json.loads(scores_json.read_text())
    rows = [r for r in payload["scores"] if r["species"] == species]
    rows.sort(key=lambda r: -float(r["score"]))
    return [r["basename"] for r in rows[:n]]


def _symlink_tree(src: Path, dst: Path) -> int:
    """Recursively symlink every file under ``src`` into ``dst``. Returns count."""
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in src.rglob("*"):
        if p.is_file():
            rel = p.relative_to(src)
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(p.resolve())
            n += 1
    return n


def _symlink_filtered_synthetics(scores_json: Path, variant_dir: Path,
                                  per_species: int) -> dict[str, int]:
    """Under variant_dir/train/<species>/, symlink the top-``per_species``
    synthetics per rare species."""
    counts: dict[str, int] = {}
    for species in RARE_SPECIES:
        dst = variant_dir / "train" / species
        dst.mkdir(parents=True, exist_ok=True)
        basenames = _top_n_basenames(scores_json, species, per_species)
        missing: list[str] = []
        linked = 0
        for bn in basenames:
            src = SYNTHETIC_SOURCE_DIR / species / bn
            if not src.exists():
                missing.append(bn)
                continue
            target = dst / bn
            if target.exists() or target.is_symlink():
                continue
            target.symlink_to(src.resolve())
            linked += 1
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} synthetics missing from {SYNTHETIC_SOURCE_DIR}/{species}: "
                f"first {missing[:3]}"
            )
        counts[species] = linked
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANT_TO_DIR), required=True)
    parser.add_argument("--per-species", type=int, default=200)
    parser.add_argument("--scores-dir", type=Path, default=RESULTS_DIR / "filters")
    parser.add_argument("--output-root", type=Path, default=GBIF_DATA_DIR)
    args = parser.parse_args()

    variant_dir = args.output_root / VARIANT_TO_DIR[args.variant]
    scores_json = args.scores_dir / VARIANT_TO_SCORES[args.variant]
    if not scores_json.exists():
        raise SystemExit(f"Scores file missing: {scores_json} -- run scripts/run_filter.py first")

    if variant_dir.exists():
        raise SystemExit(
            f"{variant_dir} already exists. Remove or rename before re-assembling."
        )

    # 1. Real train/valid/test trees symlinked from prepared_split
    prepared_split = args.output_root / "prepared_split"
    if not prepared_split.exists():
        raise SystemExit(f"{prepared_split} not found (needed as real-image source)")

    real_counts = {
        split: _symlink_tree(prepared_split / split, variant_dir / split)
        for split in ("train", "valid", "test")
        if (prepared_split / split).exists()
    }

    # 2. Filtered synthetics → train/<species>/ (layered on top of real train)
    syn_counts = _symlink_filtered_synthetics(scores_json, variant_dir, args.per_species)

    # 3. Write a small manifest for reproducibility
    manifest = {
        "variant": args.variant,
        "output_dir": str(variant_dir),
        "scores_json": str(scores_json),
        "per_species": args.per_species,
        "real_image_counts": real_counts,
        "synthetic_counts_per_species": syn_counts,
    }
    manifest_path = variant_dir / "assembly_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nWrote {manifest_path}")


if __name__ == "__main__":
    main()
