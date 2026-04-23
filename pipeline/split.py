#!/usr/bin/env python3
"""
Reorganize the prepared dataset into 70/15/15 train/valid/test splits.

Takes a prepared dataset (with train/ and valid/) and copies images into
a new directory with three-way splits (train/valid/test).

Importable API
--------------
    from pipeline.split import run
    run(input_dir="GBIF_MA_BUMBLEBEES/prepared",
        output_dir="GBIF_MA_BUMBLEBEES/prepared_split")

CLI
---
    python pipeline/split.py
    python pipeline/split.py --input-dir GBIF_MA_BUMBLEBEES/prepared --train 0.70
"""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import GBIF_DATA_DIR

DEFAULT_INPUT_DIR = GBIF_DATA_DIR / "prepared"
DEFAULT_OUTPUT_DIR = GBIF_DATA_DIR / "prepared_split"

DEFAULT_RATIOS = {"train": 0.70, "valid": 0.15, "test": 0.15}


def run(
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    train_ratio: float = 0.70,
    valid_ratio: float = 0.15,
    seed: int = 42,
) -> bool:
    """
    Split a prepared dataset into train/valid/test.

    Args:
        input_dir: Prepared dataset directory containing train/ and valid/.
        output_dir: Output directory for the three-way split.
        train_ratio: Fraction for training (default: 0.70).
        valid_ratio: Fraction for validation (default: 0.15).
            Test fraction is inferred as 1 - train_ratio - valid_ratio.
        seed: Random seed for reproducibility.

    Returns:
        True on success, False on error.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    test_ratio = 1.0 - train_ratio - valid_ratio

    print("=" * 70)
    print("DATASET SPLIT: train/valid/test")
    print("=" * 70)
    print(f"\nSplit ratios:")
    print(f"  Train: {train_ratio * 100:.0f}%")
    print(f"  Valid: {valid_ratio * 100:.0f}%")
    print(f"  Test:  {test_ratio * 100:.0f}%")

    if not input_dir.exists():
        print(f"\nError: {input_dir} does not exist!")
        print("  Please run: python run.py prepare")
        return False

    train_dir = input_dir / "train"
    valid_dir = input_dir / "valid"

    if not train_dir.exists() or not valid_dir.exists():
        print(f"\nError: train/ or valid/ directory not found in {input_dir}!")
        return False

    output_train = output_dir / "train"
    output_valid = output_dir / "valid"
    output_test = output_dir / "test"

    for d in [output_train, output_valid, output_test]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nInput directory:  {input_dir}")
    print(f"Output directory: {output_dir}")

    # Collect all species (sorted for reproducibility)
    species_set = set()
    for sp_dir in train_dir.iterdir():
        if sp_dir.is_dir():
            species_set.add(sp_dir.name)
    for sp_dir in valid_dir.iterdir():
        if sp_dir.is_dir():
            species_set.add(sp_dir.name)
    species_dirs = sorted(species_set)

    print(f"\nFound {len(species_dirs)} species")

    rng = random.Random(seed)
    total_stats: dict[str, int] = defaultdict(int)
    species_stats: dict[str, dict] = {}

    for species in species_dirs:
        print(f"\n  Processing {species}...")

        all_images = []
        for src_dir in [train_dir / species, valid_dir / species]:
            if src_dir.exists():
                all_images.extend(src_dir.glob("*.jpg"))
                all_images.extend(src_dir.glob("*.png"))

        if not all_images:
            print(f"    No images found for {species}")
            continue

        rng.shuffle(all_images)
        total = len(all_images)
        train_idx = int(total * train_ratio)
        valid_idx = train_idx + int(total * valid_ratio)

        splits = {
            "train": all_images[:train_idx],
            "valid": all_images[train_idx:valid_idx],
            "test": all_images[valid_idx:],
        }

        for split_name, imgs in splits.items():
            dest_dir = output_dir / split_name / species
            dest_dir.mkdir(exist_ok=True)
            for img in imgs:
                shutil.copy2(img, dest_dir / img.name)

        species_stats[species] = {k: len(v) for k, v in splits.items()}
        species_stats[species]["total"] = total
        for k, v in splits.items():
            total_stats[k] += len(v)
        total_stats["total"] += total

        print(
            f"    {species}: "
            f"{len(splits['train'])} train, "
            f"{len(splits['valid'])} valid, "
            f"{len(splits['test'])} test"
        )

    print("\n" + "=" * 70)
    print("SPLIT COMPLETE")
    print("=" * 70)
    print(f"\n  Total images: {total_stats['total']:,}")
    if total_stats["total"] > 0:
        for split in ("train", "valid", "test"):
            pct = total_stats[split] / total_stats["total"] * 100
            print(f"  {split.capitalize()}: {total_stats[split]:,} ({pct:.1f}%)")

    # Per-species breakdown
    print("\nPer-species breakdown:")
    print(f"{'Species':<30} {'Total':>8} {'Train':>8} {'Valid':>8} {'Test':>8}")
    print("-" * 70)
    for sp in sorted(species_stats):
        s = species_stats[sp]
        print(f"{sp:<30} {s['total']:>8} {s['train']:>8} {s['valid']:>8} {s['test']:>8}")

    print(f"\nOutput saved to: {output_dir}/")
    print("\nNext step: python run.py train --type simple --data-dir", output_dir)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Reorganize prepared dataset into 70/15/15 train/valid/test split"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Prepared dataset directory (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--train",
        type=float,
        default=0.70,
        dest="train_ratio",
        help="Train fraction (default: 0.70)",
    )
    parser.add_argument(
        "--valid",
        type=float,
        default=0.15,
        dest="valid_ratio",
        help="Validation fraction (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()
    success = run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
