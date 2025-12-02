# Bumblebee Classification Pipeline - Complete Workflow Guide

## Quick Start

### TL;DR - Run Everything

```bash
# Interactive mode (recommended for first time)
python workflow_notebook.py --interactive

# Or run everything at once with synthetic augmentation
python workflow_notebook.py --all --augmentation synthetic
```

**Note:** The workflow automatically uses the virtual environment (`venv/bin/python`) if available, otherwise falls back to system `python3`.

### Quick Reference Commands

| Command | What it does | Time |
|---------|--------------|------|
| `python workflow_notebook.py --section 1` | Download GBIF data | 15-30 min |
| `python workflow_notebook.py --section 2` | Analyze dataset | 1-2 min |
| `python workflow_notebook.py --section 3` | Prepare data (YOLO, crop) | 20-40 min |
| `python workflow_notebook.py --section 4` | Split train/valid/test | 2-5 min |
| `python workflow_notebook.py --section 5a` | Copy-paste augmentation | 30-60 min |
| `python workflow_notebook.py --section 5b --synthetic-count 100` | Synthetic augmentation | 1-2 hours |
| `python workflow_notebook.py --train --dataset synthetic_100` | Train model | 2-6 hours |
| `python workflow_notebook.py --test` | Test all models | 10-30 min |

---

## Overview

This guide explains how to use `workflow_notebook.py` - a comprehensive notebook-style script that orchestrates the complete data pipeline from collection to model testing, using all existing scripts in the repository.

### Key Features

- **Automatic Virtual Environment Detection**: Uses `venv/bin/python` if available
- **Section-by-Section Execution**: Run individual sections or the complete workflow
- **Versioned Synthetic Datasets**: Create multiple versions (synthetic_50, synthetic_100, etc.)
- **Multi-Model Testing**: Test all available models with detailed per-image predictions
- **Progress Tracking**: Clear status messages and error handling
- **Flexible Configuration**: Customize augmentation counts, species targets, and dataset types

---

## Pipeline Sections

The workflow is organized into 7 main sections:

### Section 1: Data Collection
- **Script**: `collect_ma_bumblebees.py`
- **Purpose**: Download GBIF observation images for Massachusetts bumblebees
- **Output**: `GBIF_MA_BUMBLEBEES/[species]/`
- **Time**: 15-30 minutes
- **Target**: 16 species including rare B. ashtoni and B. sandersoni

### Section 2: Data Analysis
- **Script**: `analyze_bumblebee_dataset.py`
- **Purpose**: Analyze species distribution and class imbalance
- **Output**: `dataset_analysis.json`
- **Time**: 1-2 minutes

### Section 3: Data Preparation
- **Script**: `prepare_data_only.py`
- **Purpose**: YOLO detection, cropping to 640×640, train/valid splits (90/10)
- **Output**: `GBIF_MA_BUMBLEBEES/prepared/`
- **Time**: 20-40 minutes
- **Process**: Detects bees with YOLO, crops detected regions, filters corrupted images

### Section 4: Data Splitting
- **Script**: `split_train_valid_test.py`
- **Purpose**: Create train/valid/test splits (70/15/15)
- **Output**: `GBIF_MA_BUMBLEBEES/prepared_split/`
- **Time**: 2-5 minutes

### Section 5a: Copy-Paste Augmentation
- **Scripts**: `scripts/extract_cutouts.py` + `scripts/paste_cutouts.py`
- **Purpose**: Generate augmented images using SAM segmentation + flower backgrounds
- **Output**: `GBIF_MA_BUMBLEBEES/prepared_cnp/`
- **Time**: 30-60 minutes
- **Requirements**: SAM checkpoint, flower background images in `Flowers/`
- **Target**: Augment rare species (B. ashtoni, B. sandersoni)

### Section 5b: Synthetic Augmentation (Versioned)
- **Script**: `pipeline_generate_synthetic.py`
- **Purpose**: Generate synthetic images using gpt-image-1
- **Output**: `GBIF_MA_BUMBLEBEES/prepared_synthetic_{count}/` (versioned)
- **Time**: 1-2 hours
- **Requirements**: OpenAI API key with gpt-image-1 access
- **Default**: 50 synthetic images per rare species

```bash
# Create different versions
python workflow_notebook.py --section 5b --synthetic-count 50   # prepared_synthetic_50/
python workflow_notebook.py --section 5b --synthetic-count 100  # prepared_synthetic_100/
python workflow_notebook.py --section 5b --synthetic-count 200  # prepared_synthetic_200/
```

### Section 6: Model Training
- **Script**: `pipeline_train_baseline.py`
- **Purpose**: Train hierarchical classification model (Family → Genus → Species)
- **Output**: `RESULTS/{dataset_type}_gbif/`
- **Time**: 2-6 hours (GPU dependent)
- **Architecture**: ResNet50 backbone with multi-task heads

```bash
# Train on different datasets
python workflow_notebook.py --train --dataset raw           # Baseline (no augmentation)
python workflow_notebook.py --train --dataset cnp           # Copy-paste augmented
python workflow_notebook.py --train --dataset synthetic_50  # 50 synthetic images/species
python workflow_notebook.py --train --dataset synthetic_100 # 100 synthetic images/species
```

### Section 7: Model Testing
- **Script**: `scripts/test_all_models.py`
- **Purpose**: Test all trained models with detailed output
- **Output**:
  - `RESULTS/{model}_gbif_test_results.json` (per-image predictions)
  - `RESULTS/test_comparison_report_*.txt` (comparison report)
- **Time**: 10-30 minutes

**Features:**
- Auto-detects all available models (baseline, cnp, synthetic_50, synthetic_100, etc.)
- Loads species list from checkpoint (ensures correct order)
- Saves detailed per-image predictions for analysis
- Supports custom test directory for external datasets

```bash
# Test all available models
python workflow_notebook.py --test

# Test specific models
python workflow_notebook.py --test --test-models baseline synthetic_100

# Test on external dataset
python workflow_notebook.py --test --test-dir ./hf_bees_data

# List available models
python scripts/test_all_models.py --list-models
```

---

## Three Main Workflows

### 1. Baseline (No Augmentation)
```bash
python workflow_notebook.py --section 1  # Collect
python workflow_notebook.py --section 2  # Analyze
python workflow_notebook.py --section 3  # Prepare
python workflow_notebook.py --section 4  # Split
python workflow_notebook.py --train --dataset raw
python workflow_notebook.py --test --test-models baseline
```
**Total time: ~3-5 hours**

### 2. Copy-Paste Augmentation
```bash
python workflow_notebook.py --all --augmentation copy_paste
python workflow_notebook.py --test --test-models cnp
```
**Requirements**: SAM checkpoint, flower images
**Total time: ~4-6 hours**

### 3. Synthetic Augmentation (Recommended)
```bash
# Generate synthetic data with specific count
python workflow_notebook.py --section 5b --synthetic-count 100

# Train on the versioned dataset
python workflow_notebook.py --train --dataset synthetic_100

# Test the model
python workflow_notebook.py --test --test-models synthetic_100
```
**Requirements**: OpenAI API key with credits
**Total time: ~5-10 hours**

---

## Comparing Multiple Synthetic Counts

Run experiments with different augmentation amounts:

```bash
# Generate multiple versions
python workflow_notebook.py --section 5b --synthetic-count 50
python workflow_notebook.py --section 5b --synthetic-count 100
python workflow_notebook.py --section 5b --synthetic-count 200

# Train models (can run in parallel on different GPUs)
CUDA_VISIBLE_DEVICES=0 python workflow_notebook.py --train --dataset synthetic_50
CUDA_VISIBLE_DEVICES=1 python workflow_notebook.py --train --dataset synthetic_100
CUDA_VISIBLE_DEVICES=2 python workflow_notebook.py --train --dataset synthetic_200

# Test all models at once
python workflow_notebook.py --test
```

---

## Usage Examples

### Interactive Mode (Recommended)

```bash
python workflow_notebook.py --interactive
```

This shows a menu where you can select which sections to run.

### Run Full Workflow

```bash
# Full workflow with synthetic augmentation
python workflow_notebook.py --all --augmentation synthetic

# Full workflow with copy-paste augmentation
python workflow_notebook.py --all --augmentation copy_paste

# Full workflow with both augmentation methods
python workflow_notebook.py --all --augmentation both
```

### Run Specific Sections

```bash
# Data collection and analysis
python workflow_notebook.py --section 1
python workflow_notebook.py --section 2

# Data preparation and splitting
python workflow_notebook.py --section 3
python workflow_notebook.py --section 4

# Augmentation (choose one or multiple)
python workflow_notebook.py --section 5a                      # Copy-paste with SAM
python workflow_notebook.py --section 5b --synthetic-count 50  # 50 synthetic/species
python workflow_notebook.py --section 5b --synthetic-count 100 # 100 synthetic/species
```

### Training with Different Datasets

```bash
# Train with raw data (no augmentation)
python workflow_notebook.py --train --dataset raw

# Train with copy-paste augmented data
python workflow_notebook.py --train --dataset cnp

# Train with versioned synthetic data
python workflow_notebook.py --train --dataset synthetic_50
python workflow_notebook.py --train --dataset synthetic_100

# Auto-detect best available dataset
python workflow_notebook.py --train --dataset auto
```

### Testing Models

```bash
# Test all available models
python workflow_notebook.py --test

# Test specific models
python workflow_notebook.py --test --test-models baseline cnp synthetic_100

# Test single model
python workflow_notebook.py --test --test-models synthetic_100

# Test on external dataset
python workflow_notebook.py --test --test-dir ./hf_bees_data --test-models baseline

# Direct script usage with more options
python scripts/test_all_models.py --list-models
python scripts/test_all_models.py --model synthetic_100
python scripts/test_all_models.py --test-dir ./external_data --suffix external
```

---

## Prerequisites

### Required for All Sections
- Python 3.8+
- Virtual environment activated or `venv/` directory present
- Required packages installed (see `requirements.txt`)
- Internet connection for GBIF data download

### Required for Section 5a (Copy-Paste)
- SAM (Segment Anything Model) checkpoint at `checkpoints/sam_vit_h.pth`
- Download: https://github.com/facebookresearch/segment-anything
- Flower background images in `Flowers/` directory

### Required for Section 5b (Synthetic)
- OpenAI API key with gpt-image-1 access
- Create `.env` file with:
  ```
  OPENAI_API_KEY=your_api_key_here
  ```

### Recommended for Training (Section 6)
- CUDA-compatible GPU with 8GB+ memory
- 16GB+ system RAM

---

## Directory Structure

After running the workflow:

```
bumblebee_bplusplus/
├── workflow_notebook.py           # Main workflow orchestrator
├── venv/                          # Virtual environment (auto-detected)
├── GBIF_MA_BUMBLEBEES/           # Data directories
│   ├── [species_folders]/         # Section 1: Raw GBIF downloads
│   ├── prepared/                  # Section 3: YOLO-cropped, 90/10 split
│   ├── prepared_split/            # Section 4: 70/15/15 split
│   ├── prepared_cnp/              # Section 5a: Copy-paste augmented
│   ├── prepared_synthetic_50/     # Section 5b: 50 synthetic/species
│   ├── prepared_synthetic_100/    # Section 5b: 100 synthetic/species
│   └── prepared_synthetic_200/    # Section 5b: 200 synthetic/species
├── RESULTS/                       # Training and testing results
│   ├── baseline_gbif/             # Baseline model
│   ├── cnp_gbif/                  # Copy-paste model
│   ├── synthetic_50_gbif/         # Synthetic 50 model
│   ├── synthetic_100_gbif/        # Synthetic 100 model
│   ├── *_gbif_test_results.json   # Test results with per-image predictions
│   └── test_comparison_report_*.txt  # Comparison reports
├── CACHE_CNP/                     # Copy-paste cache (cutouts)
└── Flowers/                       # Flower backgrounds for augmentation
```

---

## Configuration

Edit `workflow_notebook.py` to customize:

```python
class WorkflowConfig:
    # Directories
    VENV_DIR = Path("./venv")  # Virtual environment location

    # Target species for augmentation (most underrepresented)
    RARE_SPECIES = ["Bombus_ashtoni", "Bombus_sandersoni"]

    # Augmentation settings
    SYNTHETIC_COUNT_PER_SPECIES = 50  # Default GPT-4o images per species
    CNP_COUNT_PER_SPECIES = 600       # Copy-paste images per species
```

---

## Monitoring Progress

### During Training
```bash
# In a separate terminal
python monitor_training.py

# Or tail the log file
tail -f RESULTS/synthetic_100_gbif/training.log
```

### Check Results
```bash
# View test results
cat RESULTS/synthetic_100_gbif_test_results.json | python -m json.tool

# Quick accuracy check
grep "overall_accuracy" RESULTS/*_test_results.json
```

---

## Troubleshooting

### Virtual Environment Not Detected
```bash
# Create venv if it doesn't exist
python3 -m venv venv

# Activate and install dependencies
source venv/bin/activate
pip install -r requirements.txt

# Run workflow (will auto-detect venv)
python workflow_notebook.py --interactive
```

### Section Failed
- Each section checks for required inputs from previous sections
- Run sections in order: 1 → 2 → 3 → 4 → (5a or 5b) → 6 → 7
- Error messages indicate which section to run first

### "Prepared data directory not found"
- Make sure you've run Sections 1-3 before running Section 4 or later
- Check that `GBIF_MA_BUMBLEBEES/prepared/` exists

### "OpenAI API key not found"
- Create `.env` file in repository root
- Add `OPENAI_API_KEY=your_key` to the file
- Make sure you have API credits

### "Model weights not found" during testing
- Train the model first using `--train --dataset <type>`
- Check that `RESULTS/{dataset}_gbif/best_multitask.pt` exists

### Out of Memory During Training
- Reduce batch size in `training_config.yaml`
- Use smaller image size (512 instead of 640)
- Close other GPU applications

---

## Tips and Best Practices

1. **Start with Interactive Mode**: If you're new to the pipeline, use `--interactive`
2. **Run Sections Separately**: For debugging, run sections one at a time
3. **Version Your Synthetic Data**: Use `--synthetic-count` to create comparable experiments
4. **Test on External Data**: Use `--test-dir` to evaluate on new datasets
5. **Check Detailed Predictions**: JSON output includes per-image results for error analysis
6. **Monitor Resources**: Keep an eye on disk space and GPU memory

---

## Example: Complete Experiment

```bash
# Step 1: Prepare data (run once)
python workflow_notebook.py --section 1
python workflow_notebook.py --section 2
python workflow_notebook.py --section 3
python workflow_notebook.py --section 4

# Step 2: Create synthetic versions
python workflow_notebook.py --section 5b --synthetic-count 50
python workflow_notebook.py --section 5b --synthetic-count 100

# Step 3: Train models (can run in parallel)
python workflow_notebook.py --train --dataset raw
python workflow_notebook.py --train --dataset synthetic_50
python workflow_notebook.py --train --dataset synthetic_100

# Step 4: Test all models
python workflow_notebook.py --test

# Step 5: Check results
cat RESULTS/test_comparison_report_gbif_*.txt
```

---

## Existing Scripts Used

This workflow notebook orchestrates the following scripts:

| Script | Section | Purpose |
|--------|---------|---------|
| `collect_ma_bumblebees.py` | 1 | GBIF data collection |
| `analyze_bumblebee_dataset.py` | 2 | Dataset analysis |
| `prepare_data_only.py` | 3 | Data preparation with YOLO |
| `split_train_valid_test.py` | 4 | Train/valid/test splitting |
| `scripts/paste_cutouts.py` | 5a | Copy-paste augmentation |
| `pipeline_generate_synthetic.py` | 5b | gpt-image-1 synthetic generation |
| `pipeline_train_baseline.py` | 6 | Model training |
| `scripts/test_all_models.py` | 7 | Multi-model testing |

---

## Running on Multiple GPUs

```bash
# Use screen or tmux to prevent interruption
screen -S train

# Run different datasets on different GPUs
CUDA_VISIBLE_DEVICES=0 python workflow_notebook.py --train --dataset raw &
CUDA_VISIBLE_DEVICES=1 python workflow_notebook.py --train --dataset cnp &
CUDA_VISIBLE_DEVICES=2 python workflow_notebook.py --train --dataset synthetic_50 &
CUDA_VISIBLE_DEVICES=3 python workflow_notebook.py --train --dataset synthetic_100 &

# Press Ctrl+A, D to detach
# Reconnect later: screen -r train
```

---

## Getting Help

```bash
# Workflow help
python workflow_notebook.py --help

# Testing script help
python scripts/test_all_models.py --help

# List available models
python scripts/test_all_models.py --list-models
```

For issues or questions, see the main repository README or open an issue.
