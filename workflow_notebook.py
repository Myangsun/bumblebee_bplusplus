#!/usr/bin/env python3
"""
BUMBLEBEE CLASSIFICATION PIPELINE - WORKFLOW NOTEBOOK
=====================================================

Complete workflow from data collection to model training and testing.
This notebook-style script uses existing scripts in the repository.

Pipeline Steps:
1. Data Collection - Download GBIF images
2. Data Analysis - Analyze species distribution and class imbalance
3. Data Preparation - YOLO detection and cropping
4. Data Splitting - Create train/valid/test splits
5. Data Augmentation - Copy-paste or synthetic image generation
6. Model Training - Train hierarchical classification model
7. Model Testing - Evaluate on test set

Usage:
    # Run full pipeline
    python workflow_notebook.py --all

    # Run specific sections
    python workflow_notebook.py --section 1  # Data collection only
    python workflow_notebook.py --section 2-4  # Steps 2 through 4

    # Interactive mode
    python workflow_notebook.py --interactive
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path
from typing import List, Optional
import json
from datetime import datetime


# ============================================================================
# CONFIGURATION
# ============================================================================

class WorkflowConfig:
    """Central configuration for the workflow"""

    # Directories
    GBIF_DATA_DIR = Path("./GBIF_MA_BUMBLEBEES")
    SYNTHETIC_DATA_DIR = Path("./SYNTHETIC_BUMBLEBEES")
    RESULTS_DIR = Path("./RESULTS")
    CACHE_DIR = Path("./CACHE_CNP")
    FLOWERS_DIR = Path("./Flowers")

    # Virtual environment
    VENV_DIR = Path("./venv")

    # Critical species to augment (most underrepresented in dataset)
    # Bombus_ashtoni: 23 train samples, Bombus_sandersoni: 39 train samples
    RARE_SPECIES = ["Bombus_ashtoni", "Bombus_sandersoni"]

    @staticmethod
    def get_python_command():
        """Get the Python command to use (venv if available, otherwise system python3)"""
        venv_python = WorkflowConfig.VENV_DIR / "bin" / "python"
        if venv_python.exists():
            return str(venv_python)
        return "python3"

    @staticmethod
    def run_script(script_name, args=None, **kwargs):
        """Run a Python script using the appropriate Python interpreter"""
        python_cmd = WorkflowConfig.get_python_command()
        cmd = [python_cmd, script_name]
        if args:
            cmd.extend(args)
        return subprocess.run(cmd, cwd=Path.cwd(), **kwargs)

    # Augmentation settings
    AUGMENTATION_RATIOS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Target ~680 samples per species (dataset average)
    # Bombus_ashtoni: 23 → needs +657, Bombus_sandersoni: 39 → needs +641
    SYNTHETIC_COUNT_PER_SPECIES = 100  # GPT-4o synthetic images per species
    CNP_COUNT_PER_SPECIES = 600        # Copy-paste augmented images per species

    # Training settings
    DATASET_TYPES = ["raw", "cnp", "synthetic"]  # raw = no augmentation


# ============================================================================
# SECTION 1: DATA COLLECTION
# ============================================================================

def section_1_data_collection():
    """
    SECTION 1: DATA COLLECTION
    --------------------------
    Download GBIF observation images for Massachusetts bumblebee species.

    Script: collect_ma_bumblebees.py

    - Downloads 16 species (focus on rare: B. terricola, B. fervidus)
    - Geographic filtering: Massachusetts, USA
    - Downloads up to 2000 images per species
    - Uses bplusplus library for GBIF API integration

    Output: GBIF_MA_BUMBLEBEES/[species_folders]/
    """
    print("\n" + "="*80)
    print("SECTION 1: DATA COLLECTION")
    print("="*80)
    print("\nDownloading GBIF observation images for Massachusetts bumblebees...")
    print("Target species: 16 species including B. terricola and B. fervidus")
    print("\nThis may take 15-30 minutes depending on network speed.")

    try:
        result = WorkflowConfig.run_script("collect_ma_bumblebees.py", check=True)
        print("\n[SUCCESS] SECTION 1 COMPLETE: Data collection successful")
        print(f"   Data saved to: {WorkflowConfig.GBIF_DATA_DIR}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] SECTION 1 FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] SECTION 1 ERROR: {e}")
        return False


# ============================================================================
# SECTION 2: DATA ANALYSIS
# ============================================================================

def section_2_data_analysis():
    """
    SECTION 2: DATA ANALYSIS
    ------------------------
    Analyze species distribution and identify class imbalance.

    Script: analyze_bumblebee_dataset.py

    - Counts images per species
    - Identifies severe class imbalance
    - Highlights rare species representation
    - Calculates imbalance metrics
    - Generates recommendations for augmentation

    Output: dataset_analysis.json
    """
    print("\n" + "="*80)
    print("SECTION 2: DATA ANALYSIS")
    print("="*80)
    print("\nAnalyzing dataset distribution and class imbalance...")

    if not WorkflowConfig.GBIF_DATA_DIR.exists():
        print("\n[WARNING]  WARNING: GBIF data directory not found!")
        print("   Please run Section 1 (Data Collection) first.")
        return False

    try:
        result = WorkflowConfig.run_script("analyze_bumblebee_dataset.py", check=True)
        print("\n[SUCCESS] SECTION 2 COMPLETE: Dataset analysis successful")
        print("   Analysis saved to: dataset_analysis.json")

        # Display summary if analysis file exists
        analysis_file = Path("dataset_analysis.json")
        if analysis_file.exists():
            with open(analysis_file, 'r') as f:
                analysis = json.load(f)
                print("\n   Summary:")
                print(f"   - Total species: {analysis.get('total_species', 'N/A')}")
                print(f"   - Total images: {analysis.get('total_images', 'N/A')}")
                if 'rare_species' in analysis:
                    print(f"   - Rare species identified: {len(analysis['rare_species'])}")

        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] SECTION 2 FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[FAILED] SECTION 2 ERROR: {e}")
        return False


# ============================================================================
# SECTION 3: DATA PREPARATION
# ============================================================================

def section_3_data_preparation():
    """
    SECTION 3: DATA PREPARATION
    ---------------------------
    Prepare data using YOLO detection and cropping.

    Script: pipeline_collect_analyze.py (Step 3 only) or prepare_data_only.py

    - YOLO-based bumblebee detection
    - Crops detected regions to 640×640
    - Filters corrupted/low-quality images
    - Creates train/valid splits (90/10)

    Output: GBIF_MA_BUMBLEBEES/prepared/
    """
    print("\n" + "="*80)
    print("SECTION 3: DATA PREPARATION")
    print("="*80)
    print("\nPreparing data with YOLO detection and cropping...")
    print("- Detecting bumblebees in images")
    print("- Cropping to 640×640")
    print("- Creating train/valid splits (90/10)")
    print("\nThis may take 20-40 minutes depending on dataset size.")

    if not WorkflowConfig.GBIF_DATA_DIR.exists():
        print("\n[WARNING]  WARNING: GBIF data directory not found!")
        print("   Please run Section 1 (Data Collection) first.")
        return False

    try:
        # Use the standalone preparation script
        result = WorkflowConfig.run_script("prepare_data_only.py", check=True)
        print("\n[SUCCESS] SECTION 3 COMPLETE: Data preparation successful")
        print(f"   Prepared data saved to: {WorkflowConfig.GBIF_DATA_DIR / 'prepared'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] SECTION 3 FAILED: {e}")
        print("\n   Trying alternative preparation method...")
        try:
            result = WorkflowConfig.run_script(
                "pipeline_collect_analyze.py",
                args=["--skip-collection", "--skip-analysis"],
                check=True
            )
            print("\n[SUCCESS] SECTION 3 COMPLETE: Data preparation successful")
            return True
        except:
            print(f"\n[FAILED] SECTION 3 FAILED: Both preparation methods failed")
            return False
    except Exception as e:
        print(f"\n[FAILED] SECTION 3 ERROR: {e}")
        return False


# ============================================================================
# SECTION 4: DATA SPLITTING
# ============================================================================

def section_4_data_splitting():
    """
    SECTION 4: DATA SPLITTING
    -------------------------
    Create train/valid/test splits (70/15/15).

    Script: split_train_valid_test.py

    - Reorganizes prepared data into train/valid/test splits
    - Configurable split ratios (default: 70/15/15)
    - Shuffles images randomly but reproducibly
    - Preserves species-level statistics

    Output: GBIF_MA_BUMBLEBEES/prepared_split/
    """
    print("\n" + "="*80)
    print("SECTION 4: DATA SPLITTING")
    print("="*80)
    print("\nCreating train/valid/test splits (70/15/15)...")

    prepared_dir = WorkflowConfig.GBIF_DATA_DIR / "prepared"
    if not prepared_dir.exists():
        print("\n[WARNING]  WARNING: Prepared data directory not found!")
        print("   Please run Section 3 (Data Preparation) first.")
        return False

    try:
        result = WorkflowConfig.run_script("split_train_valid_test.py", check=True)
        print("\n[SUCCESS] SECTION 4 COMPLETE: Data splitting successful")
        print(f"   Split data saved to: {WorkflowConfig.GBIF_DATA_DIR / 'prepared_split'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] SECTION 4 FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[FAILED] SECTION 4 ERROR: {e}")
        return False


# ============================================================================
# SECTION 5A: DATA AUGMENTATION - COPY-PASTE
# ============================================================================

def section_5a_augmentation_copy_paste():
    """
    SECTION 5A: DATA AUGMENTATION - COPY-PASTE (SAM + Flower Backgrounds)
    ---------------------------------------------------------------------
    Generate augmented images using 2-step copy-paste with SAM segmentation
    and flower backgrounds.

    Scripts:
    - scripts/extract_cutouts.py (Step 1: Extract cutouts with SAM)
    - scripts/paste_cutouts.py (Step 2: Paste onto flower backgrounds)

    Process:
    1. Extract high-quality RGBA cutouts using SAM segmentation
    2. Paste cutouts onto flower backgrounds with random rotation/scaling
    - Flower backgrounds are enlarged 5x and center-cropped to 640x640
    - Generates configurable number of images per rare species

    Output: GBIF_MA_BUMBLEBEES/prepared_cnp/
    """
    print("\n" + "="*80)
    print("SECTION 5A: DATA AUGMENTATION - COPY-PASTE (2-Step with Flowers)")
    print("="*80)
    print("\nGenerating augmented images using SAM cutouts + flower backgrounds...")
    print(f"Target species: {', '.join(WorkflowConfig.RARE_SPECIES)}")
    print(f"Images per species: {WorkflowConfig.CNP_COUNT_PER_SPECIES}")
    print("\nStep 0: Setup prepared_cnp from prepared_split")
    print("Step 1: Extract cutouts with SAM")
    print("Step 2: Paste onto flower backgrounds")
    print("\nThis may take 30-60 minutes depending on SAM model and GPU availability.")

    # Check source data exists (prepared_split from section 4)
    prepared_split_dir = WorkflowConfig.GBIF_DATA_DIR / "prepared_split"
    if not prepared_split_dir.exists():
        print("\n[WARNING] WARNING: prepared_split directory not found!")
        print("   Please run Section 4 (Data Splitting) first.")
        return False

    # Setup prepared_cnp by copying from prepared_split if needed
    prepared_cnp_dir = WorkflowConfig.GBIF_DATA_DIR / "prepared_cnp"
    if not prepared_cnp_dir.exists():
        print("\n" + "-"*60)
        print("STEP 0: SETTING UP prepared_cnp DIRECTORY")
        print("-"*60)
        print(f"   Copying {prepared_split_dir} → {prepared_cnp_dir}...")
        import shutil
        shutil.copytree(prepared_split_dir, prepared_cnp_dir)
        print("   ✓ prepared_cnp directory created")

    # Check flower backgrounds
    flower_dir = WorkflowConfig.FLOWERS_DIR / "Images"
    if not flower_dir.exists():
        # Try alternative path
        flower_dir = WorkflowConfig.FLOWERS_DIR
    if not flower_dir.exists() or not list(flower_dir.glob("*")):
        print(f"\n[WARNING] WARNING: Flower backgrounds not found at {flower_dir}")
        print("   Please add flower images for backgrounds.")
        return False

    flower_count = len(list(flower_dir.rglob("*.jpg")) + list(flower_dir.rglob("*.png")))
    print(f"\n   Found {flower_count} flower background images in {flower_dir}")

    try:
        # ==== STEP 1: Extract cutouts with SAM ====
        # print("\n" + "-"*60)
        # print("STEP 1: EXTRACTING CUTOUTS WITH SAM")
        # print("-"*60)

        # species_list = " ".join(WorkflowConfig.RARE_SPECIES)
        # result = WorkflowConfig.run_script(
        #     "scripts/extract_cutouts.py",
        #     args=[
        #         "--targets", *WorkflowConfig.RARE_SPECIES,
        #         "--dataset-root", str(WorkflowConfig.GBIF_DATA_DIR),
        #         "--sam-checkpoint", "checkpoints/sam_vit_h.pth",
        #     ],
        #     check=True
        # )
        # print("\n   ✓ Cutout extraction complete")
        # print("   Cutouts saved to: CACHE_CNP/cutouts/<species>/")

        # ==== STEP 2: Paste onto flower backgrounds ====
        print("\n" + "-"*60)
        print("STEP 2: PASTING CUTOUTS ONTO FLOWER BACKGROUNDS")
        print("-"*60)

        result = WorkflowConfig.run_script(
            "scripts/paste_cutouts.py",
            args=[
                "--cutout-species", *WorkflowConfig.RARE_SPECIES,
                "--flower-dir", str(flower_dir),
                "--dataset-root", str(WorkflowConfig.GBIF_DATA_DIR),
                "--per-class-count", str(WorkflowConfig.CNP_COUNT_PER_SPECIES),
                "--size-ratio-range", "0.8", "1.0",
                "--rotation-range", "-180", "180",
                "--paste-position", "center",
            ],
            check=True
        )
        print("\n   ✓ Paste composites complete")

        print("\n[SUCCESS] SECTION 5A COMPLETE: Copy-paste augmentation successful")
        print(f"   Augmented data saved to: {WorkflowConfig.GBIF_DATA_DIR / 'prepared_cnp'}")
        print(f"   Generation log: RESULTS/paste_composites/generation_log.json")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] SECTION 5A FAILED: {e}")
        print("\n   Troubleshooting:")
        print("   1. Make sure SAM checkpoint exists at checkpoints/sam_vit_h.pth")
        print("   2. Download from: https://github.com/facebookresearch/segment-anything")
        print("   3. Ensure flower images exist in Flowers/Images/")
        return False
    except Exception as e:
        print(f"\n[FAILED] SECTION 5A ERROR: {e}")
        return False


# ============================================================================
# SECTION 5B: DATA AUGMENTATION - SYNTHETIC (GPT-4o)
# ============================================================================

def section_5b_augmentation_synthetic():
    """
    SECTION 5B: DATA AUGMENTATION - SYNTHETIC (GPT-4o)
    --------------------------------------------------
    Generate morphologically accurate synthetic images using GPT-4o.

    Script: pipeline_generate_synthetic.py

    - Chain-of-thought prompting for anatomical accuracy
    - Variations: angles (dorsal/lateral/frontal), genders (male/female)
    - Multiple environments and host plants
    - Parallel generation with rate limiting
    - Generates 50 images per rare species

    Output: GBIF_MA_BUMBLEBEES/prepared_synthetic/
    """
    print("\n" + "="*80)
    print("SECTION 5B: DATA AUGMENTATION - SYNTHETIC (GPT-4o)")
    print("="*80)
    print("\nGenerating synthetic images using OpenAI GPT-4o...")
    print(f"Target species: {', '.join(WorkflowConfig.RARE_SPECIES)}")
    print(f"Images per species: {WorkflowConfig.SYNTHETIC_COUNT_PER_SPECIES}")
    print("\n[WARNING]  REQUIREMENTS:")
    print("   - OpenAI API key in .env file")
    print("   - API credits for GPT-4o image generation")
    print("\nThis may take 1-2 hours depending on API rate limits.")

    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("\n[WARNING]  WARNING: .env file not found!")
        print("   Please create .env file with OPENAI_API_KEY")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False

    try:
        for species in WorkflowConfig.RARE_SPECIES:
            print(f"\n   Generating synthetic images for {species}...")
            result = WorkflowConfig.run_script(
                "pipeline_generate_synthetic.py",
                args=[
                    "--species", species,
                    "--count", str(WorkflowConfig.SYNTHETIC_COUNT_PER_SPECIES),
                    "--num-workers", "3",  # Parallel generation
                ],
                check=True
            )

        print("\n[SUCCESS] SECTION 5B COMPLETE: Synthetic generation successful")
        print(f"   Synthetic data saved to: {WorkflowConfig.GBIF_DATA_DIR / 'prepared_synthetic'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] SECTION 5B FAILED: {e}")
        print("\n   Check that OpenAI API key is valid and has credits.")
        return False
    except Exception as e:
        print(f"\n[FAILED] SECTION 5B ERROR: {e}")
        return False


# ============================================================================
# SECTION 6: MODEL TRAINING
# ============================================================================

def section_6_model_training(dataset_type: str = "auto"):
    """
    SECTION 6: MODEL TRAINING
    ------------------------
    Train hierarchical classification model.

    Script: pipeline_train_baseline.py

    Dataset types:
    - raw: prepared_split (70/15/15, no augmentation)
    - cnp: prepared_cnp (copy-paste augmented)
    - synthetic: prepared_synthetic (GPT-4o augmented)
    - auto: Auto-detect (prefers synthetic > cnp > raw)

    - Hierarchical classification: Family -> Genus -> Species
    - Uses ResNet50 backbone with multi-task heads
    - Early stopping with patience
    - Saves best and final model checkpoints

    Output: RESULTS/[dataset_type]_gbif/
    """
    print("\n" + "="*80)
    print("SECTION 6: MODEL TRAINING")
    print("="*80)
    print(f"\nTraining hierarchical classification model...")
    print(f"Dataset type: {dataset_type}")
    print("\nModel architecture:")
    print("   - Backbone: ResNet50")
    print("   - Multi-task heads: Family -> Genus -> Species")
    print("   - Early stopping: patience=15")
    print("\nThis may take 2-6 hours depending on GPU and dataset size.")

    try:
        result = WorkflowConfig.run_script(
            "pipeline_train_baseline.py",
            args=["--dataset", dataset_type],
            check=True
        )
        print("\n[SUCCESS] SECTION 6 COMPLETE: Model training successful")
        print(f"   Model saved to: {WorkflowConfig.RESULTS_DIR / f'{dataset_type}_gbif'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] SECTION 6 FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[FAILED] SECTION 6 ERROR: {e}")
        return False


# ============================================================================
# SECTION 7: MODEL TESTING
# ============================================================================

def section_7_model_testing():
    """
    SECTION 7: MODEL TESTING
    -----------------------
    Evaluate trained models on hold-out test set.

    Included in: pipeline_train_baseline.py (Step 7)

    - Loads best checkpoint
    - Inference on test images
    - Calculates metrics:
      * Overall accuracy
      * Per-species: accuracy, precision, recall, F1-score
      * Special focus on rare species
    - Generates classification report

    Output: RESULTS/[dataset_type]_test_results.json
    """
    print("\n" + "="*80)
    print("SECTION 7: MODEL TESTING")
    print("="*80)
    print("\nModel testing is automatically performed in Section 6 (Model Training).")
    print("The testing step (Step 7) runs after training completes.")
    print("\nTest results are saved to:")
    print("   RESULTS/[dataset_type]_test_results.json")
    print("\nTo view results, check the RESULTS directory.")

    # List available test results
    results_dir = WorkflowConfig.RESULTS_DIR
    if results_dir.exists():
        test_results = list(results_dir.glob("*_test_results.json"))
        if test_results:
            print("\n   Available test results:")
            for result_file in test_results:
                print(f"   - {result_file.name}")
        else:
            print("\n   No test results found yet.")

    return True


# ============================================================================
# WORKFLOW ORCHESTRATION
# ============================================================================

def print_workflow_menu():
    """Print the workflow menu"""
    print("\n" + "="*80)
    print("BUMBLEBEE CLASSIFICATION PIPELINE - WORKFLOW NOTEBOOK")
    print("="*80)
    python_cmd = WorkflowConfig.get_python_command()
    venv_status = "venv" if "venv" in python_cmd else "system"
    print(f"Python: {python_cmd} ({venv_status})")
    print("\nAvailable sections:")
    print("\n[1] Data Collection")
    print("    Download GBIF images for Massachusetts bumblebees")
    print("\n[2] Data Analysis")
    print("    Analyze species distribution and class imbalance")
    print("\n[3] Data Preparation")
    print("    YOLO detection, cropping, and train/valid splits")
    print("\n[4] Data Splitting")
    print("    Create train/valid/test splits (70/15/15)")
    print("\n[5a] Data Augmentation - Copy-Paste (Flower Backgrounds)")
    print("     Extract SAM cutouts → paste onto flower backgrounds")
    print("\n[5b] Data Augmentation - Synthetic")
    print("     Generate synthetic images using GPT-4o")
    print("\n[6] Model Training")
    print("    Train hierarchical classification model")
    print("\n[7] Model Testing")
    print("    Evaluate model on test set (included in Section 6)")
    print("\n" + "="*80)


def run_workflow_section(section: str) -> bool:
    """Run a specific workflow section"""
    sections = {
        "1": section_1_data_collection,
        "2": section_2_data_analysis,
        "3": section_3_data_preparation,
        "4": section_4_data_splitting,
        "5a": section_5a_augmentation_copy_paste,
        "5b": section_5b_augmentation_synthetic,
        "6": lambda: section_6_model_training("auto"),
        "7": section_7_model_testing,
    }

    if section not in sections:
        print(f"\n[FAILED] Invalid section: {section}")
        return False

    return sections[section]()


def run_full_workflow(augmentation_type: str = "synthetic"):
    """
    Run the complete workflow from data collection to model testing.

    Args:
        augmentation_type: "copy_paste", "synthetic", or "both"
    """
    print("\n" + "="*80)
    print("RUNNING FULL WORKFLOW")
    print("="*80)
    print(f"Augmentation type: {augmentation_type}")
    print("\nThis will run all sections sequentially.")
    print("Estimated total time: 4-10 hours")

    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Workflow cancelled.")
        return

    # Core pipeline: 1-4
    core_sections = [
        ("1", "Data Collection"),
        ("2", "Data Analysis"),
        ("3", "Data Preparation"),
        ("4", "Data Splitting"),
    ]

    # Augmentation sections based on type
    aug_sections = []
    if augmentation_type in ["copy_paste", "both"]:
        aug_sections.append(("5a", "Copy-Paste Augmentation"))
    if augmentation_type in ["synthetic", "both"]:
        aug_sections.append(("5b", "Synthetic Augmentation"))

    # Training section
    training_sections = [("6", "Model Training")]

    all_sections = core_sections + aug_sections + training_sections

    # Track progress
    completed = []
    failed = []

    for section_id, section_name in all_sections:
        print(f"\n\n{'='*80}")
        print(f"RUNNING SECTION {section_id}: {section_name}")
        print(f"Progress: {len(completed)}/{len(all_sections)} sections completed")
        print(f"{'='*80}")

        if run_workflow_section(section_id):
            completed.append((section_id, section_name))
            print(f"\n[SUCCESS] Section {section_id} completed successfully")
        else:
            failed.append((section_id, section_name))
            print(f"\n[FAILED] Section {section_id} failed")
            print("\n[WARNING]  Workflow stopped due to error.")
            print("Fix the error and resume from this section.")
            break

    # Print summary
    print("\n\n" + "="*80)
    print("WORKFLOW SUMMARY")
    print("="*80)
    print(f"\nCompleted: {len(completed)}/{len(all_sections)} sections")

    if completed:
        print("\n[SUCCESS] Completed sections:")
        for section_id, section_name in completed:
            print(f"   [{section_id}] {section_name}")

    if failed:
        print("\n[FAILED] Failed sections:")
        for section_id, section_name in failed:
            print(f"   [{section_id}] {section_name}")

    if len(completed) == len(all_sections):
        print("\n" + "="*80)
        print("[SUCCESS] WORKFLOW COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Review training results in RESULTS/ directory")
        print("2. Analyze test performance, especially for rare species")
        print("3. Compare different augmentation methods")
        print("4. Validate synthetic images with entomologists")


def interactive_mode():
    """Run workflow in interactive mode"""
    while True:
        print_workflow_menu()
        print("\nOptions:")
        print("  [1-7]   Run a specific section")
        print("  [all]   Run full workflow")
        print("  [help]  Show this menu")
        print("  [quit]  Exit")

        choice = input("\nEnter your choice: ").strip().lower()

        if choice == "quit":
            print("\n[SUCCESS] Exiting workflow notebook")
            break
        elif choice == "help":
            continue
        elif choice == "all":
            aug_type = input("\nAugmentation type (copy_paste/synthetic/both): ").strip().lower()
            if aug_type not in ["copy_paste", "synthetic", "both"]:
                print("Invalid augmentation type. Using 'synthetic'.")
                aug_type = "synthetic"
            run_full_workflow(aug_type)
        elif choice in ["1", "2", "3", "4", "5a", "5b", "6", "7"]:
            run_workflow_section(choice)
        else:
            print(f"\n[FAILED] Invalid choice: {choice}")

        input("\nPress Enter to continue...")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Bumblebee Classification Pipeline - Workflow Notebook",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python workflow_notebook.py --interactive
  python workflow_notebook.py --all --augmentation synthetic
  python workflow_notebook.py --section 1
  python workflow_notebook.py --section 5a
  python workflow_notebook.py --train --dataset cnp
        """
    )

    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full workflow"
    )

    parser.add_argument(
        "--section", "-s",
        type=str,
        help="Run specific section (1, 2, 3, 4, 5a, 5b, 6, 7)"
    )

    parser.add_argument(
        "--augmentation", "-a",
        type=str,
        choices=["copy_paste", "synthetic", "both"],
        default="synthetic",
        help="Augmentation type for full workflow (default: synthetic)"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Run model training only"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["raw", "cnp", "synthetic", "auto"],
        default="auto",
        help="Dataset type for training (default: auto)"
    )

    args = parser.parse_args()

    # Handle different modes
    if args.interactive:
        interactive_mode()
    elif args.all:
        run_full_workflow(args.augmentation)
    elif args.section:
        run_workflow_section(args.section)
    elif args.train:
        section_6_model_training(args.dataset)
    else:
        # Default to interactive mode
        print("No options specified. Starting interactive mode...")
        interactive_mode()


if __name__ == "__main__":
    main()
