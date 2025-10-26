#!/usr/bin/env python
"""
Quick script to run only the prepare step with img_size=640
"""

import bplusplus
from pathlib import Path
from collections import defaultdict
import json

# Configuration
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")
PREPARED_DATA_DIR = GBIF_DATA_DIR / "prepared"
RESULTS_DIR = Path("./RESULTS")

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)

print("="*70)
print("STEP 3: PREPARING DATA (img_size=640)")
print("="*70)
print("\nProcessing images with YOLO detection and preparing train/valid splits...")

# Check if GBIF data exists
if not GBIF_DATA_DIR.exists() or not list(GBIF_DATA_DIR.glob("Bombus*")):
    print(f"\n✗ Error: No species directories found in {GBIF_DATA_DIR}")
    print("   Please download GBIF data first.")
    exit(1)

# Count images before preparation
species_counts = defaultdict(int)
for species_dir in GBIF_DATA_DIR.iterdir():
    if species_dir.is_dir() and species_dir.name.startswith("Bombus"):
        image_count = len(list(species_dir.glob("*.jpg"))) + len(list(species_dir.glob("*.png")))
        species_counts[species_dir.name] = image_count

print(f"\nInput directory: {GBIF_DATA_DIR}")
print(f"Output directory: {PREPARED_DATA_DIR}")
print(f"Image size: 640")
print(f"\nSpecies found: {len(species_counts)}")
print("Sample counts:")
for species, count in sorted(species_counts.items())[:5]:
    print(f"  - {species}: {count} images")

print("\nPreparing data (this may take several minutes)...")
print("  - Detecting bumblebees with YOLO")
print("  - Cropping detected regions to 640x640")
print("  - Filtering corrupted/low-quality images")
print("  - Creating train/valid splits (80/20)")

try:
    # Prepare data with correct image size
    bplusplus.prepare(
        input_directory=str(GBIF_DATA_DIR),
        output_directory=str(PREPARED_DATA_DIR),
        img_size=640  # Updated to 640
    )

    print("\n✓ Data preparation complete!")
    print(f"  ✓ Images detected and cropped with YOLO (size: 640x640)")
    print(f"  ✓ Train/Valid splits created (80/20)")
    print(f"  ✓ Output directory: {PREPARED_DATA_DIR}")
    print(f"\n  Structure:")
    print(f"    {PREPARED_DATA_DIR}/train/")
    print(f"    {PREPARED_DATA_DIR}/valid/")

    # Count prepared images
    train_images = len(list(PREPARED_DATA_DIR.glob("train/**/*.jpg"))) + len(list(PREPARED_DATA_DIR.glob("train/**/*.png")))
    valid_images = len(list(PREPARED_DATA_DIR.glob("valid/**/*.jpg"))) + len(list(PREPARED_DATA_DIR.glob("valid/**/*.png")))

    print(f"\n  Images after preparation:")
    print(f"    Train: {train_images}")
    print(f"    Valid: {valid_images}")
    print(f"    Total: {train_images + valid_images}")

    # Save metadata
    metadata = {
        "total_species": len(species_counts),
        "species_counts": dict(species_counts),
        "total_images": sum(species_counts.values()),
        "preparation_method": "YOLO-based detection and cropping",
        "split_type": "train/valid (80/20)",
        "img_size": 640,
        "prepared_images": {
            "train": train_images,
            "valid": valid_images,
            "total": train_images + valid_images
        },
        "note": "Use prepared/train/ for training and prepared/valid/ for validation"
    }
    metadata_file = RESULTS_DIR / "data_preparation_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved to: {metadata_file}")

    print("\n✓ Next step: Run split_train_valid_test.py to split into 70/15/15")

except Exception as e:
    print(f"\n✗ Error during data preparation: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
