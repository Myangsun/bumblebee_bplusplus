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
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

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
    print("\nTraining hierarchical classification model on GBIF data only...")
    print("This establishes performance baseline before synthetic augmentation")

    # Check if prepared data exists
    if not TRAINING_DATA_DIR.exists():
        print(f"\n✗ Error: {TRAINING_DATA_DIR} does not exist!")
        print("   Please run 'pipeline_collect_analyze.py' first.")
        return False

    try:
        output_dir = RESULTS_DIR / "baseline_gbif"

        # Get species list from training directory
        species_list = []
        train_dir = TRAINING_DATA_DIR / "train"
        if train_dir.exists():
            species_list = [d.name for d in train_dir.iterdir() if d.is_dir()]

        if not species_list:
            print(f"\n✗ Error: No species directories found in {train_dir}")
            return False

        print(f"\nTraining parameters:")
        print(f"  Dataset type: {TRAINING_DATA_TYPE}")
        print(f"  Input data: {TRAINING_DATA_DIR}")
        print(f"  Output directory: {output_dir}")
        print(f"  Batch size: 4")
        print(f"  Epochs: 10 (initial run - increase for full training)")
        print(f"  Image size: 640")
        print(f"  Species: {len(species_list)} species")
        print(f"    {', '.join(species_list[:3])}{'...' if len(species_list) > 3 else ''}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Train baseline model using correct bplusplus API
        bplusplus.train(
            batch_size=4,
            epochs=10,  # Reduced for testing; increase for full training (30-50)
            patience=3,
            img_size=640,
            data_dir=str(TRAINING_DATA_DIR),
            output_dir=str(output_dir),
            species_list=species_list,
            num_workers=0  # Set to 0 for most stable, single-process loading
        )

        print("\n✓ Baseline model training complete")
        print(f"  ✓ Model saved to: {output_dir}")

        # Save training metadata
        metadata = {
            "model_type": "baseline",
            "model_architecture": "hierarchical (family, genus, species)",
            "dataset_type": TRAINING_DATA_TYPE,
            "training_data": str(TRAINING_DATA_DIR),
            "batch_size": 4,
            "epochs": 10,
            "patience": 3,
            "img_size": 640,
            "species_count": len(species_list),
            "species_list": species_list,
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


def _run_inference(model, device, test_images, species_list_unique):
    """Helper: Run inference on test images"""
    predictions = []
    ground_truth = []
    image_paths = []

    for idx, img_path in enumerate(test_images):
        if (idx + 1) % 100 == 0:
            print(f"  Processed {idx + 1}/{len(test_images)} images...")

        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = torch.tensor(np.array(img)).float().permute(2, 0, 1).unsqueeze(0) / 255.0

            with torch.no_grad():
                output = model(img_tensor.to(device))

            if isinstance(output, torch.Tensor):
                pred_idx = output.argmax(dim=1).item()
            elif isinstance(output, (list, tuple)):
                pred_idx = output[-1].argmax(dim=1).item() if len(output) > 0 else 0
            else:
                pred_idx = 0

            predictions.append(species_list_unique[pred_idx])
            ground_truth.append(img_path.parent.name)
            image_paths.append(str(img_path))

        except Exception as e:
            print(f"  Warning: Failed to process {img_path}: {str(e)}")

    return predictions, ground_truth, image_paths


def _compute_metrics(ground_truth, predictions, species_list_unique):
    """Helper: Compute and display metrics"""
    overall_accuracy = accuracy_score(ground_truth, predictions)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

    print("\n" + "-"*70)
    print("PER-SPECIES PERFORMANCE")
    print("-"*70)

    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, labels=species_list_unique, zero_division=0
    )

    species_metrics = {}
    for i, species in enumerate(species_list_unique):
        count = sum(1 for x in ground_truth if x == species)
        correct = sum(1 for j in range(len(ground_truth)) if ground_truth[j] == species and predictions[j] == species)
        species_metrics[species] = {
            "accuracy": correct / max(count, 1),
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i])
        }
        print(f"\n{species}:")
        print(f"  Test samples: {count}")
        print(f"  Accuracy: {species_metrics[species]['accuracy']:.4f}")
        print(f"  Precision: {species_metrics[species]['precision']:.4f}")
        print(f"  Recall: {species_metrics[species]['recall']:.4f}")
        print(f"  F1-Score: {species_metrics[species]['f1']:.4f}")

    return overall_accuracy, species_metrics


def _print_rare_species_results(species_metrics):
    """Helper: Print rare species performance"""
    print("\n" + "-"*70)
    print("RARE SPECIES PERFORMANCE")
    print("-"*70)

    for species in ["Bombus_terricola", "Bombus_fervidus"]:
        if species in species_metrics:
            m = species_metrics[species]
            print(f"\n{species}:")
            print(f"  Test samples: {m['support']}")
            print(f"  Accuracy: {m['accuracy']:.4f}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall: {m['recall']:.4f}")
            print(f"  F1-Score: {m['f1']:.4f}")
        else:
            print(f"\n{species}: Not in test set")


def step7_test_baseline():
    """Step 7: Test baseline model on test set"""
    print("\n" + "="*70)
    print("STEP 7: TESTING BASELINE MODEL")
    print("="*70)

    test_set_type = "test set (70/15/15 split)" if PREPARED_SPLIT_DIR.exists() else "validation set"
    print(f"\nTesting model on {test_set_type}...")

    model_dir = RESULTS_DIR / "baseline_gbif"
    model_path = model_dir / "best_multitask.pt"

    if not model_path.exists():
        print(f"\n✗ Error: {model_path} does not exist!")
        print("   Please run Step 5 first to train the model.")
        return False

    if not TEST_DATA_DIR.exists():
        print(f"\n✗ Error: {TEST_DATA_DIR} does not exist!")
        print("   Please run 'pipeline_collect_analyze.py' first to prepare data.")
        return False

    try:
        print(f"\nLoading trained model from: {model_path}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        model.eval()

        test_images = list(TEST_DATA_DIR.rglob('*.jpg')) + list(TEST_DATA_DIR.rglob('*.png'))
        species_list_unique = sorted({img.parent.name for img in test_images})

        print(f"Device: {device}")
        print(f"Total test images: {len(test_images)}")
        print(f"Species: {len(species_list_unique)}")

        print("\nRunning inference on test images...")
        predictions, ground_truth, image_paths = _run_inference(model, device, test_images, species_list_unique)
        print(f"\n✓ Inference complete on {len(predictions)} images")

        print("\n" + "="*70)
        print("EVALUATION RESULTS")
        print("="*70)

        overall_accuracy, species_metrics = _compute_metrics(ground_truth, predictions, species_list_unique)
        _print_rare_species_results(species_metrics)

        print("\n" + "-"*70)
        print("SAVING RESULTS")
        print("-"*70)

        results = {
            "test_set_type": test_set_type,
            "test_directory": str(TEST_DATA_DIR),
            "model_path": str(model_path),
            "total_test_images": len(predictions),
            "overall_accuracy": float(overall_accuracy),
            "species_count": len(species_list_unique),
            "species_metrics": species_metrics,
            "detailed_predictions": [
                {
                    "image_path": image_paths[i],
                    "ground_truth": ground_truth[i],
                    "prediction": predictions[i],
                    "correct": ground_truth[i] == predictions[i]
                }
                for i in range(len(predictions))
            ]
        }

        results_file = RESULTS_DIR / "baseline_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n✓ Detailed results saved to: {results_file}")

        print("\n" + "-"*70)
        print("CLASSIFICATION REPORT")
        print("-"*70)
        print("\n" + classification_report(ground_truth, predictions, labels=species_list_unique, zero_division=0))

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
