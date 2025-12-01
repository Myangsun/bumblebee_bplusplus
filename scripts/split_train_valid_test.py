"""
Split prepared dataset into train/valid/test sets

Takes the prepared dataset (train/ and valid/) and reorganizes it into:
train/ (60% or 70%)
valid/ (20% or 15%)
test/  (20% or 15%)

Usage:
    python split_train_valid_test.py
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

# Configuration
PREPARED_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES/prepared")
OUTPUT_DIR = Path("./GBIF_MA_BUMBLEBEES/prepared_split")

# Split ratios
SPLIT_RATIOS = {
    "train": 0.70,  # 70% training
    "valid": 0.15,  # 15% validation
    "test": 0.15    # 15% testing
}

# Alternative ratios (comment out the above and uncomment below for different splits)
# SPLIT_RATIOS = {
#     "train": 0.60,  # 60% training
#     "valid": 0.20,  # 20% validation
#     "test": 0.20    # 20% testing
# }


def create_split_dataset():
    """Split dataset into train/valid/test"""

    print("="*70)
    print("DATASET SPLIT: train/valid/test")
    print("="*70)
    print(f"\nSplit ratios:")
    print(f"  Train: {SPLIT_RATIOS['train']*100:.0f}%")
    print(f"  Valid: {SPLIT_RATIOS['valid']*100:.0f}%")
    print(f"  Test:  {SPLIT_RATIOS['test']*100:.0f}%")

    # Check if prepared data exists
    if not PREPARED_DATA_DIR.exists():
        print(f"\n✗ Error: {PREPARED_DATA_DIR} does not exist!")
        print("   Please run pipeline_collect_analyze.py first.")
        return False

    # Get train and valid directories
    train_dir = PREPARED_DATA_DIR / "train"
    valid_dir = PREPARED_DATA_DIR / "valid"

    if not train_dir.exists() or not valid_dir.exists():
        print(f"\n✗ Error: train/ or valid/ directory not found!")
        return False

    # Create output directory structure
    output_train = OUTPUT_DIR / "train"
    output_valid = OUTPUT_DIR / "valid"
    output_test = OUTPUT_DIR / "test"

    for out_dir in [output_train, output_valid, output_test]:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nInput directory: {PREPARED_DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Get all species with deterministic ordering
    species_dirs = set()
    for species_dir in train_dir.iterdir():
        if species_dir.is_dir():
            species_dirs.add(species_dir.name)
    for species_dir in valid_dir.iterdir():
        if species_dir.is_dir():
            species_dirs.add(species_dir.name)

    # CRITICAL: Sort alphabetically for reproducibility
    # This ensures consistent species ordering regardless of filesystem order
    species_dirs = sorted(list(species_dirs))
    print(f"\nFound {len(species_dirs)} species")

    # Process each species
    total_stats = defaultdict(int)
    species_stats = {}

    for species in species_dirs:
        print(f"\n  Processing {species}...")

        # Collect all images for this species from train and valid
        all_images = []

        train_species_dir = train_dir / species
        if train_species_dir.exists():
            all_images.extend([
                (img, "train") for img in train_species_dir.glob("*.jpg")
            ])
            all_images.extend([
                (img, "train") for img in train_species_dir.glob("*.png")
            ])

        valid_species_dir = valid_dir / species
        if valid_species_dir.exists():
            all_images.extend([
                (img, "valid") for img in valid_species_dir.glob("*.jpg")
            ])
            all_images.extend([
                (img, "valid") for img in valid_species_dir.glob("*.png")
            ])

        if not all_images:
            print(f"    ⚠️  No images found for {species}")
            continue

        # Shuffle images
        random.shuffle(all_images)

        # Calculate split indices
        total = len(all_images)
        train_idx = int(total * SPLIT_RATIOS["train"])
        valid_idx = train_idx + int(total * SPLIT_RATIOS["valid"])

        # Split images
        train_images = all_images[:train_idx]
        valid_images = all_images[train_idx:valid_idx]
        test_images = all_images[valid_idx:]

        # Create species directories in output
        (output_train / species).mkdir(exist_ok=True)
        (output_valid / species).mkdir(exist_ok=True)
        (output_test / species).mkdir(exist_ok=True)

        # Copy images to split directories
        for img_path, src_split in train_images:
            dest = output_train / species / img_path.name
            shutil.copy2(img_path, dest)

        for img_path, src_split in valid_images:
            dest = output_valid / species / img_path.name
            shutil.copy2(img_path, dest)

        for img_path, src_split in test_images:
            dest = output_test / species / img_path.name
            shutil.copy2(img_path, dest)

        # Record stats
        species_stats[species] = {
            "total": total,
            "train": len(train_images),
            "valid": len(valid_images),
            "test": len(test_images)
        }

        total_stats["total"] += total
        total_stats["train"] += len(train_images)
        total_stats["valid"] += len(valid_images)
        total_stats["test"] += len(test_images)

        print(f"    ✓ {species}: {len(train_images)} train, "
              f"{len(valid_images)} valid, {len(test_images)} test")

    # Print summary
    print("\n" + "="*70)
    print("SPLIT COMPLETE")
    print("="*70)

    print("\nOverall statistics:")
    print(f"  Total images: {total_stats['total']:,}")
    if total_stats['total'] > 0:
        print(f"  Train: {total_stats['train']:,} ({total_stats['train']/total_stats['total']*100:.1f}%)")
        print(f"  Valid: {total_stats['valid']:,} ({total_stats['valid']/total_stats['total']*100:.1f}%)")
        print(f"  Test:  {total_stats['test']:,} ({total_stats['test']/total_stats['total']*100:.1f}%)")
    else:
        print(f"  Train: 0 (0.0%)")
        print(f"  Valid: 0 (0.0%)")
        print(f"  Test:  0 (0.0%)")
        print("\n⚠️  WARNING: No images found to split!")

    # Print per-species details
    print("\nPer-species breakdown:")
    print(f"{'Species':<30} {'Total':>8} {'Train':>8} {'Valid':>8} {'Test':>8}")
    print("-" * 70)

    for species in sorted(species_stats.keys()):
        stats = species_stats[species]
        print(f"{species:<30} {stats['total']:>8} {stats['train']:>8} "
              f"{stats['valid']:>8} {stats['test']:>8}")

    # Print output structure
    print("\n" + "="*70)
    print("OUTPUT STRUCTURE")
    print("="*70)
    print(f"\nDataset saved to: {OUTPUT_DIR}/")
    print(f"\nStructure:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  ├── train/")
    print(f"  │   ├── Bombus_terricola/")
    print(f"  │   ├── Bombus_fervidus/")
    print(f"  │   └── ... (other species)")
    print(f"  ├── valid/")
    print(f"  │   ├── Bombus_terricola/")
    print(f"  │   ├── Bombus_fervidus/")
    print(f"  │   └── ... (other species)")
    print(f"  └── test/")
    print(f"      ├── Bombus_terricola/")
    print(f"      ├── Bombus_fervidus/")
    print(f"      └── ... (other species)")

    # Print rare species details
    print("\n" + "="*70)
    print("YOUR RARE TARGET SPECIES")
    print("="*70)

    for species in ["Bombus_terricola", "Bombus_fervidus"]:
        if species in species_stats:
            stats = species_stats[species]
            print(f"\n{species}:")
            print(f"  Total: {stats['total']:,}")
            print(f"  Train: {stats['train']:,} ({stats['train']/stats['total']*100:.1f}%)")
            print(f"  Valid: {stats['valid']:,} ({stats['valid']/stats['total']*100:.1f}%)")
            print(f"  Test:  {stats['test']:,} ({stats['test']/stats['total']*100:.1f}%)")

    print("\n" + "="*70)
    print("✓ DATASET SPLIT COMPLETE")
    print("="*70)
    print(f"\nReady for training!")
    print(f"\nNext steps:")
    print(f"1. Update pipeline_train_baseline.py to use {OUTPUT_DIR}")
    print(f"2. OR use the new split dataset as:")
    print(f"   - Training: {OUTPUT_DIR}/train/")
    print(f"   - Validation: {OUTPUT_DIR}/valid/")
    print(f"   - Testing: {OUTPUT_DIR}/test/")

    return True


if __name__ == "__main__":
    create_split_dataset()
