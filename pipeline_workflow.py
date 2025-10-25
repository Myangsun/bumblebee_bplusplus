"""
Complete Bumblebee Classification Pipeline
Massachusetts Rare Species Focus: B. terricola and B. fervidus

Pipeline Steps:
1. Collect GBIF data
2. Analyze dataset distribution
3. Prepare data (bplusplus.prepare)
4. Generate synthetic images for rare species (GPT-4o)
5. Train baseline model (GBIF only)
6. Train augmented models (GBIF + 10%, 20%, ..., 100% synthetic)
7. Test and compare performance
8. Validate synthetic images with entomologists
"""

import bplusplus
from pathlib import Path
import json
import subprocess

# Configuration
GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")
SYNTHETIC_DATA_DIR = Path("./SYNTHETIC_BUMBLEBEES")
RESULTS_DIR = Path("./RESULTS")

# Create directories
SYNTHETIC_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def step1_collect_data():
    """Step 1: Collect GBIF data for Massachusetts bumblebees"""
    print("\n" + "="*70)
    print("STEP 1: Collecting GBIF Data")
    print("="*70)
    
    subprocess.run(["python", "collect_ma_bumblebees.py"])
    print("\n✓ Data collection complete")


def step2_analyze_dataset():
    """Step 2: Analyze dataset distribution"""
    print("\n" + "="*70)
    print("STEP 2: Analyzing Dataset Distribution")
    print("="*70)
    
    subprocess.run(["python", "analyze_bumblebee_dataset.py"])
    print("\n✓ Dataset analysis complete")


def step3_prepare_data():
    """Step 3: Prepare data using bplusplus"""
    print("\n" + "="*70)
    print("STEP 3: Preparing Data (train/val/test splits)")
    print("="*70)
    
    # This organizes the data into train/val/test splits
    # You may need to adjust parameters based on your needs
    bplusplus.prepare(
        input_directory=GBIF_DATA_DIR,
        output_directory=GBIF_DATA_DIR / "prepared",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print("\n✓ Data preparation complete")
    print(f"Prepared data saved to: {GBIF_DATA_DIR / 'prepared'}")


def step4_generate_synthetic():
    """Step 4: Generate synthetic images for rare species"""
    print("\n" + "="*70)
    print("STEP 4: Generating Synthetic Images")
    print("="*70)
    print("Using GPT-4o to upsample rare species images")
    print("Target species: B. terricola and B. fervidus")
    print("\n⚠️  Note: This requires OpenAI API access")
    print("See synthetic_augmentation_gpt4o.py for implementation")
    print("\n✓ Ready to generate synthetic images")


def step5_train_baseline():
    """Step 5: Train baseline model (GBIF only)"""
    print("\n" + "="*70)
    print("STEP 5: Training Baseline Model (GBIF only)")
    print("="*70)
    
    # Train on GBIF data only
    bplusplus.train(
        data_directory=GBIF_DATA_DIR / "prepared",
        output_directory=RESULTS_DIR / "baseline_gbif",
        model_name="resnet50",  # or other architecture
        epochs=50
    )
    
    print("\n✓ Baseline model training complete")


def step6_train_augmented_models():
    """Step 6: Train models with different synthetic augmentation ratios"""
    print("\n" + "="*70)
    print("STEP 6: Training Augmented Models")
    print("="*70)
    
    # Define augmentation percentages as per your proposal
    augmentation_ratios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    for ratio in augmentation_ratios:
        print(f"\nTraining with {ratio}% synthetic augmentation...")
        
        # You'll need to merge GBIF + synthetic data at this ratio
        # This is a placeholder - implement based on your data structure
        bplusplus.train(
            data_directory=GBIF_DATA_DIR / f"prepared_with_{ratio}pct_synthetic",
            output_directory=RESULTS_DIR / f"augmented_{ratio}pct",
            model_name="resnet50",
            epochs=50
        )
    
    print("\n✓ All augmented models trained")


def step7_test_models():
    """Step 7: Test all models on GBIF test set"""
    print("\n" + "="*70)
    print("STEP 7: Testing Models")
    print("="*70)
    
    # Test baseline
    print("\nTesting baseline model...")
    bplusplus.test(
        model_directory=RESULTS_DIR / "baseline_gbif",
        test_directory=GBIF_DATA_DIR / "prepared" / "test",
        output_file=RESULTS_DIR / "baseline_results.json"
    )
    
    # Test augmented models
    augmentation_ratios = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    for ratio in augmentation_ratios:
        print(f"\nTesting model with {ratio}% synthetic augmentation...")
        bplusplus.test(
            model_directory=RESULTS_DIR / f"augmented_{ratio}pct",
            test_directory=GBIF_DATA_DIR / "prepared" / "test",
            output_file=RESULTS_DIR / f"augmented_{ratio}pct_results.json"
        )
    
    print("\n✓ All model testing complete")


def step8_analyze_results():
    """Step 8: Analyze results for rare species performance"""
    print("\n" + "="*70)
    print("STEP 8: Analyzing Results")
    print("="*70)
    
    print("\nFocus areas:")
    print("1. B. terricola classification accuracy")
    print("2. B. fervidus classification accuracy")
    print("3. Impact of synthetic augmentation on rare species")
    print("4. Confusion matrices for rare vs common species")
    
    # Load and compare results
    # This is a placeholder - implement based on your results format
    results_summary = {
        "baseline": {},
        "augmented": {}
    }
    
    # Save summary
    with open(RESULTS_DIR / "results_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✓ Results summary saved to: {RESULTS_DIR / 'results_summary.json'}")


def run_full_pipeline():
    """Run the complete pipeline"""
    print("="*70)
    print("MASSACHUSETTS BUMBLEBEE CLASSIFICATION PIPELINE")
    print("Rare Species Focus: B. terricola & B. fervidus")
    print("="*70)
    
    steps = [
        ("Data Collection", step1_collect_data),
        ("Dataset Analysis", step2_analyze_dataset),
        ("Data Preparation", step3_prepare_data),
        ("Synthetic Generation", step4_generate_synthetic),
        ("Baseline Training", step5_train_baseline),
        ("Augmented Training", step6_train_augmented_models),
        ("Model Testing", step7_test_models),
        ("Results Analysis", step8_analyze_results)
    ]
    
    for i, (name, func) in enumerate(steps, 1):
        try:
            print(f"\n\n{'='*70}")
            print(f"PIPELINE STEP {i}/{len(steps)}: {name}")
            print(f"{'='*70}")
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {str(e)}")
            print("Pipeline stopped. Fix the error and resume from this step.")
            return
    
    print("\n\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("1. Review results in ./RESULTS directory")
    print("2. Validate synthetic images with entomologists")
    print("3. Analyze rare species classification performance")
    print("4. Prepare for deployment on edge devices")


if __name__ == "__main__":
    # You can run individual steps or the full pipeline
    
    # Option 1: Run full pipeline
    # run_full_pipeline()
    
    # Option 2: Run individual steps
    print("Run individual steps by uncommenting the desired function calls:")
    print("  step1_collect_data()")
    print("  step2_analyze_dataset()")
    print("  step3_prepare_data()")
    print("  # ... etc")
    
    # Start with data collection and analysis
    step1_collect_data()
    step2_analyze_dataset()
