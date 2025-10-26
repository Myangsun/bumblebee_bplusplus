"""
Pipeline 1: COLLECT & ANALYZE
Step 1: Collect GBIF data
Step 2: Analyze dataset distribution
Step 3: Prepare data (bplusplus.prepare)

This pipeline focuses on data collection and initial analysis.
Output: Prepared training/validation/test splits
"""

import bplusplus
from pathlib import Path
import subprocess
import json
from collections import defaultdict

# Configuration
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")
PREPARED_DATA_DIR = GBIF_DATA_DIR / "prepared"
RESULTS_DIR = Path("./RESULTS")

# Create directories
RESULTS_DIR.mkdir(exist_ok=True)
GBIF_DATA_DIR.mkdir(exist_ok=True)


def step1_collect_data():
    """Step 1: Collect GBIF data for Massachusetts bumblebees"""
    print("\n" + "="*70)
    print("STEP 1: COLLECTING GBIF DATA")
    print("="*70)
    print("\nDownloading GBIF images for Massachusetts bumblebee species...")
    print("Target species: Bombus terricola, Bombus fervidus, and other MA species")

    try:
        subprocess.run(["python", "collect_ma_bumblebees.py"], check=True)
        print("\n✓ Data collection complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during data collection: {e}")
        return False


def step2_analyze_dataset():
    """Step 2: Analyze dataset distribution"""
    print("\n" + "="*70)
    print("STEP 2: ANALYZING DATASET DISTRIBUTION")
    print("="*70)
    print("\nAnalyzing species distribution and identifying class imbalance...")

    try:
        subprocess.run(["python", "analyze_bumblebee_dataset.py"], check=True)
        print("\n✓ Dataset analysis complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error during dataset analysis: {e}")
        return False


def step3_prepare_data():
    """Step 3: Prepare data using bplusplus"""
    print("\n" + "="*70)
    print("STEP 3: PREPARING DATA (Cropping, filtering, and splitting)")
    print("="*70)
    print("\nProcessing images with YOLO detection and preparing train/valid splits...")
    print("(This step will automatically split data into train and valid sets)")

    # Check if GBIF data exists
    if not GBIF_DATA_DIR.exists() or not list(GBIF_DATA_DIR.glob("Bombus*")):
        print(f"\n✗ Error: No species directories found in {GBIF_DATA_DIR}")
        print("   Please run Step 1 first to collect data.")
        return False

    try:
        print(f"\nInput directory: {GBIF_DATA_DIR}")
        print(f"Output directory: {PREPARED_DATA_DIR}")

        # Count images before preparation
        species_counts = defaultdict(int)
        for species_dir in GBIF_DATA_DIR.iterdir():
            if species_dir.is_dir() and species_dir.name.startswith("Bombus"):
                image_count = len(list(species_dir.glob("*.jpg"))) + len(list(species_dir.glob("*.png")))
                species_counts[species_dir.name] = image_count

        print(f"\nSpecies found: {len(species_counts)}")
        print("Sample counts:")
        for species, count in sorted(species_counts.items())[:5]:
            print(f"  - {species}: {count} images")

        print("\nPreparing data (this may take a few minutes)...")
        print("  - Detecting bumblebees with YOLO")
        print("  - Cropping detected regions")
        print("  - Filtering corrupted images")
        print("  - Creating train/valid splits")

        # Prepare data
        # Note: bplusplus.prepare() automatically:
        # 1. Detects objects using YOLO
        # 2. Crops detected regions
        # 3. Filters corrupted/low-quality images
        # 4. Splits into train/valid (not train/val/test)
        # 5. Creates classification folder structure
        bplusplus.prepare(
            input_directory=str(GBIF_DATA_DIR),
            output_directory=str(PREPARED_DATA_DIR),
            img_size=640  # Target size for cropped images
        )

        print("\n✓ Data preparation complete")
        print(f"  ✓ Images detected and cropped with YOLO")
        print(f"  ✓ Train/Valid splits created")
        print(f"  ✓ Output directory: {PREPARED_DATA_DIR}")
        print(f"\n  Structure:")
        print(f"    {PREPARED_DATA_DIR}/train/")
        print(f"    {PREPARED_DATA_DIR}/valid/")

        # Save metadata
        metadata = {
            "total_species": len(species_counts),
            "species_counts": dict(species_counts),
            "total_images": sum(species_counts.values()),
            "preparation_method": "YOLO-based detection and cropping",
            "split_type": "train/valid (not train/val/test)",
            "img_size": 640,
            "note": "Use prepared/train/ for training and prepared/valid/ for validation"
        }
        metadata_file = RESULTS_DIR / "data_preparation_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Metadata saved to: {metadata_file}")

        return True
    except Exception as e:
        print(f"\n✗ Error during data preparation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_collect_analyze_pipeline():
    """Run the collect and analyze pipeline"""
    print("="*70)
    print("PIPELINE 1: COLLECT & ANALYZE")
    print("="*70)
    print("Steps: 1 (Collect), 2 (Analyze), 3 (Prepare Data)")
    print("="*70)

    steps = [
        ("Data Collection", step1_collect_data),
        ("Dataset Analysis", step2_analyze_dataset),
        ("Data Preparation", step3_prepare_data),
    ]

    completed_steps = []
    failed_steps = []

    for i, (name, func) in enumerate(steps, 1):
        print(f"\n\n{'='*70}")
        print(f"STEP {i}/{len(steps)}: {name}")
        print(f"{'='*70}")

        try:
            success = func()
            if success:
                completed_steps.append(name)
            else:
                failed_steps.append(name)
                print(f"\n⚠️  Pipeline stopped at Step {i}: {name}")
                print("Fix the error above and try again.")
                break
        except KeyboardInterrupt:
            print("\n\n⚠️  Pipeline interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error in {name}: {str(e)}")
            failed_steps.append(name)
            break

    # Summary
    print("\n\n" + "="*70)
    print("PIPELINE 1 EXECUTION SUMMARY")
    print("="*70)
    print(f"\nCompleted steps ({len(completed_steps)}/{len(steps)}):")
    for step in completed_steps:
        print(f"  ✓ {step}")

    if failed_steps:
        print(f"\nFailed/Incomplete steps:")
        for step in failed_steps:
            print(f"  ✗ {step}")
    else:
        print("\n✓ PIPELINE 1 COMPLETE!")
        print("\nOutput files created:")
        print(f"  - {GBIF_DATA_DIR} (GBIF images by species)")
        print(f"  - {PREPARED_DATA_DIR} (Train/Val/Test splits)")
        print(f"  - {RESULTS_DIR}/data_preparation_metadata.json")
        print("\nNext steps:")
        print("1. Review dataset analysis results")
        print("2. Run 'pipeline_generate_synthetic.py' to generate synthetic images")
        print("3. Or run 'pipeline_train_baseline.py' to train without synthetic augmentation")


if __name__ == "__main__":
    run_collect_analyze_pipeline()
