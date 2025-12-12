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
    SYNTHETIC_COUNT_PER_SPECIES = 50  # GPT-4o synthetic images per species
    CNP_COUNT_PER_SPECIES = 100        # Copy-paste augmented images per species

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

def section_5a_augmentation_copy_paste(cnp_count: int = None):
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

    Args:
        cnp_count: Number of CNP images per species. Creates versioned
                   directory like prepared_cnp_50/, prepared_cnp_100/

    Output: GBIF_MA_BUMBLEBEES/prepared_cnp_{count}/
    """
    # Use provided count or fall back to config default
    count = cnp_count if cnp_count is not None else WorkflowConfig.CNP_COUNT_PER_SPECIES

    print("\n" + "="*80)
    print("SECTION 5A: DATA AUGMENTATION - COPY-PASTE (2-Step with Flowers)")
    print("="*80)
    print("\nGenerating augmented images using SAM cutouts + flower backgrounds...")
    print(f"Target species: {', '.join(WorkflowConfig.RARE_SPECIES)}")
    print(f"Images per species: {count}")
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

    # Setup versioned prepared_cnp directory by copying from prepared_split if needed
    # E.g., prepared_cnp_50/, prepared_cnp_100/
    prepared_cnp_dir = WorkflowConfig.GBIF_DATA_DIR / f"prepared_cnp_{count}"
    print(f"\nOutput directory: {prepared_cnp_dir}")

    if not prepared_cnp_dir.exists():
        print("\n" + "-"*60)
        print(f"STEP 0: SETTING UP {prepared_cnp_dir.name} DIRECTORY")
        print("-"*60)
        print(f"   Copying {prepared_split_dir} → {prepared_cnp_dir}...")
        import shutil
        shutil.copytree(prepared_split_dir, prepared_cnp_dir)
        print(f"   ✓ {prepared_cnp_dir.name} directory created")
    else:
        print(f"\n   Using existing directory: {prepared_cnp_dir}")

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
                "--output-subdir", f"prepared_cnp_{count}",
                "--per-class-count", str(count),
                "--size-ratio-range", "0.5", "0.7",
                "--rotation-range", "-180", "180",
                "--paste-position", "center",
            ],
            check=True
        )
        print("\n   ✓ Paste composites complete")

        print("\n[SUCCESS] SECTION 5A COMPLETE: Copy-paste augmentation successful")
        print(f"   Augmented data saved to: {prepared_cnp_dir}")
        print(f"   Generation log: RESULTS/paste_composites/generation_log.json")
        print(f"   Use --dataset cnp_{count} for training")
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
# SECTION 5B: DATA AUGMENTATION - SYNTHETIC (gpt-image-1)
# ============================================================================

def section_5b_augmentation_synthetic(synthetic_count: int = None):
    """
    SECTION 5B: DATA AUGMENTATION - SYNTHETIC (gpt-image-1)
    -------------------------------------------------------
    Generate morphologically accurate synthetic images using gpt-image-1.

    Script: pipeline_generate_synthetic.py

    - Chain-of-thought prompting for anatomical accuracy
    - Variations: angles (dorsal/lateral/frontal), genders (male/female)
    - Multiple environments and host plants
    - Parallel generation with rate limiting
    - Generates configurable images per rare species

    Args:
        synthetic_count: Number of synthetic images per species. Creates versioned
                        directory like prepared_synthetic_50/, prepared_synthetic_100/

    Output: GBIF_MA_BUMBLEBEES/prepared_synthetic_{count}/
    """
    # Use provided count or fall back to config default
    count = synthetic_count if synthetic_count is not None else WorkflowConfig.SYNTHETIC_COUNT_PER_SPECIES

    print("\n" + "="*80)
    print("SECTION 5B: DATA AUGMENTATION - SYNTHETIC (gpt-image-1)")
    print("="*80)
    print("\nGenerating synthetic images using OpenAI gpt-image-1...")
    print(f"Target species: {', '.join(WorkflowConfig.RARE_SPECIES)}")
    print(f"Images per species: {count}")
    print("\n[WARNING]  REQUIREMENTS:")
    print("   - OpenAI API key in .env file")
    print("   - API credits for gpt-image-1 image generation")
    print("\nThis may take 1-2 hours depending on API rate limits.")

    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        print("\n[WARNING]  WARNING: .env file not found!")
        print("   Please create .env file with OPENAI_API_KEY")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False

    # Check source data exists (prepared_split from section 4)
    prepared_split_dir = WorkflowConfig.GBIF_DATA_DIR / "prepared_split"
    if not prepared_split_dir.exists():
        print("\n[WARNING] WARNING: prepared_split directory not found!")
        print("   Please run Section 4 (Data Splitting) first.")
        return False

    # Setup versioned prepared_synthetic directory by copying from prepared_split if needed
    # E.g., prepared_synthetic_50/, prepared_synthetic_100/
    prepared_synthetic_dir = WorkflowConfig.GBIF_DATA_DIR / f"prepared_synthetic_{count}"
    print(f"\nOutput directory: {prepared_synthetic_dir}")

    if not prepared_synthetic_dir.exists():
        print("\n" + "-"*60)
        print(f"STEP 0: SETTING UP {prepared_synthetic_dir.name} DIRECTORY")
        print("-"*60)
        print(f"   Copying {prepared_split_dir} → {prepared_synthetic_dir}...")
        import shutil
        shutil.copytree(prepared_split_dir, prepared_synthetic_dir)
        print(f"   ✓ {prepared_synthetic_dir.name} directory created")
    else:
        print(f"\n   Using existing directory: {prepared_synthetic_dir}")

    try:
        for species in WorkflowConfig.RARE_SPECIES:
            print(f"\n   Generating synthetic images for {species}...")
            result = WorkflowConfig.run_script(
                "pipeline_generate_synthetic.py",
                args=[
                    "--species", species,
                    "--count", str(count),
                    "--num-workers", "5",  # Parallel generation
                    "--output-dir", str(prepared_synthetic_dir),
                ],
                check=True
            )

        print("\n[SUCCESS] SECTION 5B COMPLETE: Synthetic generation successful")
        print(f"   Synthetic data saved to: {prepared_synthetic_dir}")
        print(f"   Use --dataset synthetic_{count} for training")
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
            args=["--dataset", dataset_type, "--train-only"],
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
# SECTION 6B: MODEL TRAINING - SIMPLIFIED (train_simple.py)
# ============================================================================

def section_6b_model_training_simple(dataset_type: str = "auto", backbone: str = "resnet50",
                                     epochs: int = 100, batch_size: int = 8):
    """
    SECTION 6B: MODEL TRAINING - SIMPLIFIED
    ---------------------------------------
    Train simple classification model (no hierarchical branches).

    Script: train_simple.py

    Dataset types:
    - raw: prepared_split (70/15/15, no augmentation)
    - cnp: prepared_cnp (copy-paste augmented)
    - cnp_100: prepared_cnp_100 (versioned CNP)
    - synthetic: prepared_synthetic (GPT-4o augmented)
    - synthetic_100: prepared_synthetic_100 (versioned synthetic)
    - auto: Auto-detect (prefers synthetic > cnp > raw)

    Model architecture:
    - Simple ResNet50 (or other backbone) + single FC layer
    - No hierarchical branches (Family/Genus/Species)
    - Configurable backbone: resnet18/50/101, efficientnet_b0/b4
    - Early stopping with patience
    - Saves best model checkpoint

    Args:
        dataset_type: Dataset to train on
        backbone: Model backbone (resnet18/50/101, efficientnet_b0/b4)
        epochs: Maximum number of epochs
        batch_size: Batch size for training

    Output: RESULTS/simple_[dataset_type]/
    """
    print("\n" + "="*80)
    print("SECTION 6B: MODEL TRAINING - SIMPLIFIED")
    print("="*80)
    print(f"\nTraining simple classification model (no hierarchical branches)...")
    print(f"Dataset type: {dataset_type}")
    print(f"Backbone: {backbone}")
    print("\nModel architecture:")
    print(f"   - Backbone: {backbone}")
    print(f"   - Classification: Single FC layer (direct species classification)")
    print("   - No hierarchical branches")
    print(f"   - Early stopping: patience=15")
    print("\nThis may take 2-6 hours depending on GPU and dataset size.")

    # Determine data directory based on dataset type
    data_dir = None
    result_key = None

    if dataset_type == "raw":
        data_dir = WorkflowConfig.GBIF_DATA_DIR / "prepared_split"
        result_key = "baseline"
    elif dataset_type == "cnp":
        data_dir = WorkflowConfig.GBIF_DATA_DIR / "prepared_cnp"
        result_key = "cnp"
    elif dataset_type.startswith("cnp_"):
        data_dir = WorkflowConfig.GBIF_DATA_DIR / f"prepared_{dataset_type}"
        result_key = dataset_type
    elif dataset_type == "synthetic":
        data_dir = WorkflowConfig.GBIF_DATA_DIR / "prepared_synthetic"
        result_key = "synthetic"
    elif dataset_type.startswith("synthetic_"):
        data_dir = WorkflowConfig.GBIF_DATA_DIR / f"prepared_{dataset_type}"
        result_key = dataset_type
    elif dataset_type == "auto":
        # Auto-detect: prefer synthetic > cnp > raw
        if (WorkflowConfig.GBIF_DATA_DIR / "prepared_synthetic").exists():
            data_dir = WorkflowConfig.GBIF_DATA_DIR / "prepared_synthetic"
            result_key = "synthetic"
        elif (WorkflowConfig.GBIF_DATA_DIR / "prepared_cnp").exists():
            data_dir = WorkflowConfig.GBIF_DATA_DIR / "prepared_cnp"
            result_key = "cnp"
        else:
            data_dir = WorkflowConfig.GBIF_DATA_DIR / "prepared_split"
            result_key = "baseline"
    else:
        print(f"\n[FAILED] Unknown dataset type: {dataset_type}")
        return False

    if data_dir is None or not data_dir.exists():
        print(f"\n[FAILED] Data directory not found: {data_dir}")
        return False

    output_dir = WorkflowConfig.RESULTS_DIR / f"{result_key}_gbif"
    print(f"Output directory (pipeline-compatible): {output_dir}")

    try:
        result = WorkflowConfig.run_script(
            "train_simple.py",
            args=[
                "--data-dir", str(data_dir),
                "--output-dir", str(output_dir),
                "--backbone", backbone,
                "--epochs", str(epochs),
                "--batch-size", str(batch_size)
            ],
            check=True
        )
        print("\n[SUCCESS] SECTION 6B COMPLETE: Simplified model training successful")
        print(f"   Model saved to: {output_dir}")
        print(f"   Training log: {output_dir}/training.log")
        print(f"   Metadata: {output_dir}/training_metadata.json")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] SECTION 6B FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[FAILED] SECTION 6B ERROR: {e}")
        return False


# ============================================================================
# SECTION 7: MODEL TESTING
# ============================================================================

def section_7_model_testing(models: list = None, test_dir: str = None, suffix: str = "gbif"):
    """
    SECTION 7: MODEL TESTING
    -----------------------
    Evaluate trained models on test sets with detailed output.

    Script: scripts/test_all_models.py

    Features:
    - Auto-detects available models including versioned synthetic (synthetic_50, synthetic_100)
    - Loads species list from checkpoint (ensures correct order)
    - Direct PyTorch inference with detailed per-image predictions
    - Supports single model or batch testing
    - Custom test directory override

    Args:
        models: List of models to test (e.g., ['baseline', 'synthetic_100']).
                If None, tests all available models.
        test_dir: Override test directory for all models (e.g., external dataset)
        suffix: Suffix for output files (default: 'gbif')

    Output:
    - RESULTS/{model}_{suffix}_test_results.json (detailed predictions)
    - RESULTS/test_comparison_report_{suffix}_*.txt (comparison report)
    """
    print("\n" + "="*80)
    print("SECTION 7: MODEL TESTING")
    print("="*80)

    # Build command arguments
    args = ["scripts/test_all_models.py"]

    if models:
        if len(models) == 1:
            args.extend(["--model", models[0]])
            print(f"\nTesting single model: {models[0]}")
        else:
            args.extend(["--models"] + models)
            print(f"\nTesting models: {', '.join(models)}")
    else:
        args.append("--all")
        print("\nTesting all available models...")

    if test_dir:
        args.extend(["--test-dir", test_dir])
        print(f"Test directory override: {test_dir}")

    args.extend(["--suffix", suffix])

    print("\nThis will evaluate models and generate detailed results.")
    print("Output includes per-image predictions for analysis.")

    try:
        result = WorkflowConfig.run_script(
            args[0],
            args=args[1:],
            check=True
        )
        print("\n[SUCCESS] SECTION 7 COMPLETE: Model testing successful")
        print(f"   Results saved to: {WorkflowConfig.RESULTS_DIR}")
        print(f"   JSON files: *_{suffix}_test_results.json")
        print(f"   Report: test_comparison_report_{suffix}_*.txt")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAILED] SECTION 7 FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n[FAILED] SECTION 7 ERROR: {e}")
        return False


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
    print("\n[6] Model Training - Hierarchical (bplusplus)")
    print("    Train hierarchical classification model")
    print("\n[6b] Model Training - Simplified (train_simple.py)")
    print("     Train simple classifier (no hierarchical branches)")
    print("     Configurable backbone: resnet18/50/101, efficientnet_b0/b4")
    print("\n[7] Model Testing")
    print("    Test all available models with detailed output")
    print("    Supports: baseline, cnp, synthetic, synthetic_50, synthetic_100, etc.")
    print("\n" + "="*80)


def run_workflow_section(section: str, synthetic_count: int = None, cnp_count: int = None,
                         test_models: list = None, test_dir: str = None) -> bool:
    """Run a specific workflow section

    Args:
        section: Section identifier (1, 2, 3, 4, 5a, 5b, 6, 7)
        synthetic_count: For section 5b, the number of synthetic images per species
        cnp_count: For section 5a, the number of CNP images per species
        test_models: For section 7, list of models to test
        test_dir: For section 7, override test directory
    """
    # Use default counts if not specified
    synthetic_cnt = synthetic_count if synthetic_count is not None else WorkflowConfig.SYNTHETIC_COUNT_PER_SPECIES
    cnp_cnt = cnp_count if cnp_count is not None else WorkflowConfig.CNP_COUNT_PER_SPECIES

    sections = {
        "1": section_1_data_collection,
        "2": section_2_data_analysis,
        "3": section_3_data_preparation,
        "4": section_4_data_splitting,
        "5a": lambda: section_5a_augmentation_copy_paste(cnp_cnt),
        "5b": lambda: section_5b_augmentation_synthetic(synthetic_cnt),
        "6": lambda: section_6_model_training("auto"),
        "6b": lambda: section_6b_model_training_simple("auto", "resnet50"),
        "7": lambda: section_7_model_testing(models=test_models, test_dir=test_dir),
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
        elif choice in ["1", "2", "3", "4", "5a", "5b", "6", "6b", "7"]:
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

  # Generate versioned CNP datasets
  python workflow_notebook.py --section 5a --cnp-count 50        # Creates prepared_cnp_50/
  python workflow_notebook.py --section 5a --cnp-count 100       # Creates prepared_cnp_100/

  # Generate versioned synthetic datasets
  python workflow_notebook.py --section 5b --synthetic-count 50  # Creates prepared_synthetic_50/
  python workflow_notebook.py --section 5b --synthetic-count 100 # Creates prepared_synthetic_100/

  # Train hierarchical model (section 6)
  python workflow_notebook.py --train --dataset cnp_100
  python workflow_notebook.py --train --dataset synthetic_50
  python workflow_notebook.py --train --dataset synthetic_100

  # Train simplified model (section 6b)
  python workflow_notebook.py --section 6b                       # Auto-detect dataset, resnet50
  python workflow_notebook.py --train-simple --dataset cnp_100   # Train simple model on CNP-100
  python workflow_notebook.py --train-simple --dataset cnp_100 --backbone resnet101

  # Test models
  python workflow_notebook.py --test                             # Test all available models
  python workflow_notebook.py --test --test-models baseline cnp_100 synthetic_100
  python workflow_notebook.py --test --test-dir ./hf_bees_data   # Test on external dataset
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
        help="Run hierarchical model training only (section 6)"
    )

    parser.add_argument(
        "--train-simple",
        action="store_true",
        help="Run simplified model training only (section 6b)"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="auto",
        help="Dataset type for training: raw, cnp, synthetic_50, synthetic_100, etc. (default: auto)"
    )

    parser.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet18", "resnet50", "resnet101", "efficientnet_b0", "efficientnet_b4"],
        help="Backbone for simplified training (default: resnet50)"
    )

    parser.add_argument(
        "--synthetic-count",
        type=int,
        default=50,
        help="Number of synthetic images per species (default: 50). Creates versioned directory like prepared_synthetic_100/"
    )

    parser.add_argument(
        "--cnp-count",
        type=int,
        default=100,
        help="Number of CNP augmented images per species (default: 100). Creates versioned directory like prepared_cnp_100/"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run model testing (section 7)"
    )

    parser.add_argument(
        "--test-models",
        type=str,
        nargs="+",
        help="Models to test (e.g., baseline cnp synthetic_100). If not specified, tests all available."
    )

    parser.add_argument(
        "--test-dir",
        type=str,
        help="Override test directory for model testing (e.g., ./hf_bees_data)"
    )

    args = parser.parse_args()

    # Handle different modes
    if args.interactive:
        interactive_mode()
    elif args.all:
        run_full_workflow(args.augmentation)
    elif args.section:
        run_workflow_section(
            args.section,
            synthetic_count=args.synthetic_count,
            cnp_count=args.cnp_count,
            test_models=args.test_models,
            test_dir=args.test_dir
        )
    elif args.train:
        section_6_model_training(args.dataset)
    elif args.train_simple:
        section_6b_model_training_simple(args.dataset, args.backbone)
    elif args.test:
        section_7_model_testing(models=args.test_models, test_dir=args.test_dir)
    else:
        # Default to interactive mode
        print("No options specified. Starting interactive mode...")
        interactive_mode()


if __name__ == "__main__":
    main()
