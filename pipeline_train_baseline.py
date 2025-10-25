"""
Pipeline 2: TRAIN BASELINE
Step 5: Train baseline model (GBIF only)
Step 7: Test baseline model

This pipeline trains a baseline model on GBIF data only (without synthetic augmentation).
Establishes performance baseline before synthetic augmentation.

Requirements:
- Must run pipeline_collect_analyze.py first
"""

import bplusplus
from pathlib import Path
import json
import os

# Configuration
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")
PREPARED_DATA_DIR = GBIF_DATA_DIR / "prepared"
PREPARED_SPLIT_DIR = GBIF_DATA_DIR / "prepared_split"  # With train/valid/test
RESULTS_DIR = Path("./RESULTS")

# Use split dataset if it exists, otherwise use original prepared dataset
# The split dataset has train/valid/test, the original has only train/valid
if PREPARED_SPLIT_DIR.exists():
    TRAINING_DATA_DIR = PREPARED_SPLIT_DIR
    TRAINING_DATA_TYPE = "split (train/valid/test)"
    TEST_DATA_DIR = PREPARED_SPLIT_DIR / "test"
else:
    TRAINING_DATA_DIR = PREPARED_DATA_DIR
    TRAINING_DATA_TYPE = "original (train/valid only)"
    TEST_DATA_DIR = PREPARED_DATA_DIR / "valid"

# Create results directory
RESULTS_DIR.mkdir(exist_ok=True)


def step5_train_baseline():
    """Step 5: Train baseline model (GBIF only)"""
    print("\n" + "="*70)
    print("STEP 5: TRAINING BASELINE MODEL (GBIF ONLY)")
    print("="*70)
    print("\nTraining classification model on GBIF data only...")
    print("This establishes performance baseline before synthetic augmentation")

    # Check if prepared data exists
    if not TRAINING_DATA_DIR.exists():
        print(f"\n✗ Error: {TRAINING_DATA_DIR} does not exist!")
        print("   Please run 'pipeline_collect_analyze.py' first.")
        return False

    try:
        output_dir = RESULTS_DIR / "baseline_gbif"
        print(f"\nTraining parameters:")
        print(f"  Model architecture: ResNet50")
        print(f"  Dataset type: {TRAINING_DATA_TYPE}")
        print(f"  Input data: {TRAINING_DATA_DIR}")
        print(f"  Output directory: {output_dir}")
        print(f"  Epochs: 10 (initial run - increase for full training)")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train baseline model
        bplusplus.train(
            data_directory=str(TRAINING_DATA_DIR),
            output_directory=str(output_dir),
            model_name="resnet50",
            epochs=10  # Reduced for testing; increase for full training (50-100)
        )

        print("\n✓ Baseline model training complete")
        print(f"  ✓ Model saved to: {output_dir}")

        # Save training metadata
        metadata = {
            "model_type": "baseline",
            "model_architecture": "resnet50",
            "dataset_type": TRAINING_DATA_TYPE,
            "training_data": str(TRAINING_DATA_DIR),
            "epochs": 10,
            "augmentation": "none (GBIF only)",
            "description": f"Baseline model trained on GBIF data ({TRAINING_DATA_TYPE}) without synthetic augmentation"
        }
        metadata_file = output_dir / "training_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Metadata saved to: {metadata_file}")

        return True
    except Exception as e:
        print(f"\n✗ Error during baseline training: {e}")
        import traceback
        traceback.print_exc()
        return False


def step7_test_baseline():
    """Step 7: Test baseline model on test set"""
    print("\n" + "="*70)
    print("STEP 7: TESTING BASELINE MODEL")
    print("="*70)

    test_set_type = "test set (70/15/15 split)" if PREPARED_SPLIT_DIR.exists() else "validation set"
    print(f"\nTesting baseline model on held-out {test_set_type}...")

    # Check if model exists
    model_dir = RESULTS_DIR / "baseline_gbif"
    if not model_dir.exists():
        print(f"\n✗ Error: {model_dir} does not exist!")
        print("   Please run Step 5 first to train the model.")
        return False

    # Check if test data exists
    if not TEST_DATA_DIR.exists():
        print(f"\n✗ Error: {TEST_DATA_DIR} does not exist!")
        print("   Please run 'pipeline_collect_analyze.py' first to prepare data.")
        return False

    try:
        output_file = RESULTS_DIR / "baseline_results.json"
        print(f"\nTesting parameters:")
        print(f"  Model directory: {model_dir}")
        print(f"  Test data: {TEST_DATA_DIR}")
        print(f"  Test set type: {test_set_type}")
        print(f"  Results file: {output_file}")

        # Test model
        bplusplus.test(
            model_directory=str(model_dir),
            test_directory=str(TEST_DATA_DIR),
            output_file=str(output_file)
        )

        print("\n✓ Baseline model testing complete")
        print(f"  ✓ Results saved to: {output_file}")

        # Display summary if results exist
        if output_file.exists():
            with open(output_file, 'r') as f:
                results = json.load(f)
            print("\n" + "="*70)
            print("BASELINE MODEL RESULTS SUMMARY")
            print("="*70)

            print(f"\nTest Set Type: {test_set_type}")
            print(f"Total test images: {len([d for d in TEST_DATA_DIR.rglob('*') if d.is_file()])}")

            # Pretty print results
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, dict):
                        print(f"\n{key}:")
                        for k, v in value.items():
                            print(f"  {k}: {v}")
                    else:
                        print(f"{key}: {value}")
            else:
                print(json.dumps(results, indent=2))

            # Highlight rare species performance
            print("\n" + "="*70)
            print("RARE SPECIES PERFORMANCE")
            print("="*70)
            if "per_species_accuracy" in results or "per_class_accuracy" in results:
                per_species = results.get("per_species_accuracy") or results.get("per_class_accuracy")
                if per_species:
                    for species in ["Bombus_terricola", "Bombus_fervidus"]:
                        if species in per_species:
                            acc = per_species[species]
                            print(f"\n{species}: {acc*100:.1f}%" if isinstance(acc, float) else f"\n{species}: {acc}")

        return True
    except Exception as e:
        print(f"\n✗ Error during model testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_train_baseline_pipeline():
    """Run the baseline training pipeline"""
    print("="*70)
    print("PIPELINE 2: TRAIN BASELINE")
    print("="*70)
    print("Steps: 5 (Train Baseline), 7 (Test Baseline)")
    print("="*70)

    steps = [
        ("Train Baseline Model", step5_train_baseline),
        ("Test Baseline Model", step7_test_baseline),
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
    print("PIPELINE 2 EXECUTION SUMMARY")
    print("="*70)
    print(f"\nCompleted steps ({len(completed_steps)}/{len(steps)}):")
    for step in completed_steps:
        print(f"  ✓ {step}")

    if failed_steps:
        print(f"\nFailed/Incomplete steps:")
        for step in failed_steps:
            print(f"  ✗ {step}")
    else:
        print("\n✓ PIPELINE 2 COMPLETE!")
        print("\nOutput files created:")
        print(f"  - {RESULTS_DIR}/baseline_gbif/ (trained model)")
        print(f"  - {RESULTS_DIR}/baseline_results.json (test results)")
        print("\nNext steps:")
        print("1. Review baseline_results.json for baseline performance metrics")
        print("2. Run 'pipeline_generate_synthetic.py' to generate synthetic images")
        print("3. Then run 'pipeline_train_augmented.py' to train with synthetic augmentation")
        print("4. Compare results to evaluate synthetic augmentation impact")


if __name__ == "__main__":
    run_train_baseline_pipeline()
