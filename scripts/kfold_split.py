#!/usr/bin/env python3
"""
Create stratified 5-fold cross-validation splits from pooled real images.

Pools train+valid+test from prepared_split, then creates 5 stratified folds.
Each fold has train/valid/test directories compatible with the training pipeline.

Within each fold:
  - test  = held-out fold (20% of all real images)
  - valid = 15% of the remaining 80% (used for early stopping)
  - train = 85% of the remaining 80% (augmented later by assemble_kfold.py)

Output structure:
    GBIF_MA_BUMBLEBEES/kfold_splits/
        fold_0/train/{species}/*.jpg
        fold_0/valid/{species}/*.jpg
        fold_0/test/{species}/*.jpg
        fold_1/...
        ...
        fold_4/...
        splits.json  # manifest with all file assignments

Usage:
    python scripts/kfold_split.py
    python scripts/kfold_split.py --n-folds 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import GBIF_DATA_DIR

BASELINE_DIR = GBIF_DATA_DIR / "prepared_split"
OUTPUT_DIR = GBIF_DATA_DIR / "kfold_splits"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VALID_FRACTION = 0.15  # fraction of training portion used for validation


def pool_all_images(baseline_dir: Path) -> dict[str, list[Path]]:
    """Pool all images from train+valid+test into per-species lists."""
    by_species: dict[str, list[Path]] = defaultdict(list)
    for split in ["train", "valid", "test"]:
        split_dir = baseline_dir / split
        if not split_dir.exists():
            continue
        for species_dir in sorted(split_dir.iterdir()):
            if not species_dir.is_dir():
                continue
            species = species_dir.name
            for img in sorted(species_dir.iterdir()):
                if img.suffix.lower() in IMAGE_EXTENSIONS:
                    by_species[species].append(img)
    return dict(by_species)


def create_folds(
    by_species: dict[str, list[Path]],
    n_folds: int,
    seed: int,
) -> list[dict[str, dict[str, list[Path]]]]:
    """
    Create stratified k-fold splits.

    Returns list of dicts, one per fold:
        [{"train": {species: [paths]}, "valid": {...}, "test": {...}}, ...]
    """
    rng = random.Random(seed)

    # Shuffle each species independently
    shuffled = {}
    for species, paths in by_species.items():
        paths_copy = list(paths)
        rng.shuffle(paths_copy)
        shuffled[species] = paths_copy

    # Create fold assignments
    folds = []
    for fold_idx in range(n_folds):
        fold = {"train": defaultdict(list), "valid": defaultdict(list), "test": defaultdict(list)}
        folds.append(fold)

    for species, paths in shuffled.items():
        n = len(paths)
        # Assign each image to exactly one test fold
        for i, path in enumerate(paths):
            fold_idx = i % n_folds
            folds[fold_idx]["test"][species].append(path)

    # For each fold, everything NOT in test is available for train+valid
    all_paths_by_species = shuffled
    for fold_idx in range(n_folds):
        test_set = set()
        for species, paths in folds[fold_idx]["test"].items():
            test_set.update(str(p) for p in paths)

        for species, all_paths in all_paths_by_species.items():
            non_test = [p for p in all_paths if str(p) not in test_set]
            rng.shuffle(non_test)

            # Split non-test into train and valid
            n_valid = max(1, int(len(non_test) * VALID_FRACTION))
            valid_paths = non_test[:n_valid]
            train_paths = non_test[n_valid:]

            folds[fold_idx]["valid"][species] = valid_paths
            folds[fold_idx]["train"][species] = train_paths

    return folds


def write_folds(
    folds: list[dict[str, dict[str, list[Path]]]],
    output_dir: Path,
    force: bool = False,
) -> dict:
    """Write fold directories and return manifest."""
    if output_dir.exists():
        if force:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"{output_dir} exists. Use --force to overwrite.")

    output_dir.mkdir(parents=True)
    manifest = {"n_folds": len(folds), "folds": []}

    for fold_idx, fold in enumerate(folds):
        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_manifest = {}

        for split_name in ["train", "valid", "test"]:
            split_data = fold[split_name]
            split_manifest = {}

            for species in sorted(split_data.keys()):
                paths = split_data[species]
                dest_dir = fold_dir / split_name / species
                dest_dir.mkdir(parents=True, exist_ok=True)

                filenames = []
                for src_path in paths:
                    dst_path = dest_dir / src_path.name
                    # Use hard link to save disk space (same filesystem)
                    try:
                        dst_path.hardlink_to(src_path)
                    except OSError:
                        shutil.copy2(src_path, dst_path)
                    filenames.append(src_path.name)

                split_manifest[species] = {
                    "count": len(filenames),
                    "files": filenames,
                }

            fold_manifest[split_name] = split_manifest

        manifest["folds"].append(fold_manifest)

        # Print summary
        train_total = sum(len(fold["train"][sp]) for sp in fold["train"])
        valid_total = sum(len(fold["valid"][sp]) for sp in fold["valid"])
        test_total = sum(len(fold["test"][sp]) for sp in fold["test"])
        print(f"  Fold {fold_idx}: train={train_total}, valid={valid_total}, test={test_total}")

    # Save manifest
    manifest_path = output_dir / "splits.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nManifest: {manifest_path}")

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Create stratified k-fold CV splits")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing splits")
    args = parser.parse_args()

    print(f"Pooling images from {BASELINE_DIR}...")
    by_species = pool_all_images(BASELINE_DIR)

    total = sum(len(v) for v in by_species.values())
    print(f"Pooled {total} images across {len(by_species)} species\n")

    print("Per-species counts:")
    for sp in sorted(by_species):
        print(f"  {sp}: {len(by_species[sp])}")

    print(f"\nCreating {args.n_folds}-fold stratified splits (seed={args.seed})...")
    folds = create_folds(by_species, args.n_folds, args.seed)

    print(f"\nWriting to {OUTPUT_DIR}...")
    write_folds(folds, OUTPUT_DIR, force=args.force)

    # Print target species detail
    print("\nTarget species per fold (test set):")
    for fold_idx, fold in enumerate(folds):
        parts = []
        for sp in ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]:
            n = len(fold["test"].get(sp, []))
            parts.append(f"{sp.split('_')[1]}={n}")
        print(f"  Fold {fold_idx}: {', '.join(parts)}")

    print("\nDone. Next: run scripts/assemble_kfold.py to add synthetic augmentation.")


if __name__ == "__main__":
    main()
