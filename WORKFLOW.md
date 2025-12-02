# Bumblebee Classification Pipeline - Complete Workflow Guide

## Quick Start

### TL;DR - Run Everything

```bash
# Interactive mode (recommended for first time)
python3 workflow_notebook.py --interactive

# Or run everything at once with synthetic augmentation
python3 workflow_notebook.py --all --augmentation synthetic
```

**Note:** The workflow automatically uses the virtual environment (`venv/bin/python`) if available, otherwise falls back to system `python3`.

### Quick Reference Commands

| Command | What it does | Time |
|---------|--------------|------|
| `python3 workflow_notebook.py --section 1` | Download GBIF data | 15-30 min |
| `python3 workflow_notebook.py --section 2` | Analyze dataset | 1-2 min |
| `python3 workflow_notebook.py --section 3` | Prepare data (YOLO, crop) | 20-40 min |
| `python3 workflow_notebook.py --section 4` | Split train/valid/test | 2-5 min |
| `python3 workflow_notebook.py --section 5a` | Copy-paste augmentation | 30-60 min |
| `python3 workflow_notebook.py --section 5b` | Synthetic augmentation (GPT-4o) | 1-2 hours |
| `python3 workflow_notebook.py --train` | Train model | 2-6 hours |

---

## Overview

This guide explains how to use `workflow_notebook.py` - a comprehensive notebook-style script that orchestrates the complete data pipeline from collection to model testing, using all existing scripts in the repository.

### Key Features

- **Automatic Virtual Environment Detection**: Uses `venv/bin/python` if available, otherwise falls back to system `python3`
- **Section-by-Section Execution**: Run individual sections or the complete workflow
- **Multiple Augmentation Methods**: Choose between copy-paste (SAM) or synthetic (GPT-4o) augmentation
- **Progress Tracking**: Clear status messages and error handling
- **Flexible Configuration**: Customize augmentation ratios, species targets, and dataset types

---

## Pipeline Sections

The workflow is organized into 7 main sections that use existing scripts:

### Section 1: Data Collection
- **Script**: `collect_ma_bumblebees.py`
- **Purpose**: Download GBIF observation images for Massachusetts bumblebees
- **Output**: `GBIF_MA_BUMBLEBEES/[species]/`
- **Time**: 15-30 minutes
- **Target**: 16 species including rare B. terricola and B. fervidus

### Section 2: Data Analysis
- **Script**: `analyze_bumblebee_dataset.py`
- **Purpose**: Analyze species distribution and class imbalance
- **Output**: `dataset_analysis.json`
- **Time**: 1-2 minutes

### Section 3: Data Preparation
- **Script**: `prepare_data_only.py`
- **Purpose**: YOLO detection, cropping to 640×640, train/valid splits (80/20)
- **Output**: `GBIF_MA_BUMBLEBEES/prepared/`
- **Time**: 20-40 minutes
- **Process**: Detects bees with YOLO, crops detected regions, filters corrupted images

### Section 4: Data Splitting
- **Script**: `split_train_valid_test.py`
- **Purpose**: Create train/valid/test splits (70/15/15)
- **Output**: `GBIF_MA_BUMBLEBEES/prepared_split/`
- **Time**: 2-5 minutes

### Section 5a: Copy-Paste Augmentation
- **Script**: `scripts/copy_paste_augment.py`
- **Purpose**: Generate augmented images using SAM segmentation
- **Output**: `GBIF_MA_BUMBLEBEES/prepared_cnp/`
- **Time**: 30-60 minutes
- **Requirements**: SAM checkpoint, flower background images
- **Target**: Augment rare species (B. terricola, B. fervidus)

### Section 5b: Synthetic Augmentation
- **Script**: `pipeline_generate_synthetic.py`
- **Purpose**: Generate synthetic images using GPT-4o
- **Output**: `GBIF_MA_BUMBLEBEES/prepared_synthetic/`
- **Time**: 1-2 hours
- **Requirements**: OpenAI API key with GPT-4o access
- **Default**: 50 synthetic images per rare species

### Section 6: Model Training
- **Script**: `pipeline_train_baseline.py`
- **Purpose**: Train hierarchical classification model (Family → Genus → Species)
- **Output**: `RESULTS/[dataset_type]_gbif/`
- **Time**: 2-6 hours (GPU dependent)
- **Architecture**: ResNet50 backbone with multi-task heads

### Section 7: Model Testing
- **Included in**: Section 6 (automatic)
- **Purpose**: Evaluate model on test set
- **Output**: `RESULTS/[dataset_type]_test_results.json`

---

## Three Main Workflows

### 1. Baseline (No Augmentation)
```bash
python3 workflow_notebook.py --section 1  # Collect
python3 workflow_notebook.py --section 2  # Analyze
python3 workflow_notebook.py --section 3  # Prepare
python3 workflow_notebook.py --section 4  # Split
python3 workflow_notebook.py --train --dataset raw
```
**Total time: ~3-5 hours**

### 2. Copy-Paste Augmentation
```bash
python3 workflow_notebook.py --all --augmentation copy_paste
```
**Requirements**: SAM checkpoint, flower images
**Total time: ~4-6 hours**

### 3. Synthetic Augmentation (Recommended)
```bash
python3 workflow_notebook.py --all --augmentation synthetic
```
**Requirements**: OpenAI API key with credits
**Total time: ~5-10 hours**

---

## Usage Examples

### Interactive Mode (Recommended)

```bash
python3 workflow_notebook.py --interactive
```

This shows a menu where you can select which sections to run.

### Run Full Workflow

```bash
# Full workflow with synthetic augmentation
python3 workflow_notebook.py --all --augmentation synthetic

# Full workflow with copy-paste augmentation
python3 workflow_notebook.py --all --augmentation copy_paste

# Full workflow with both augmentation methods
python3 workflow_notebook.py --all --augmentation both
```

### Run Specific Sections

```bash
# Data collection and analysis
python3 workflow_notebook.py --section 1
python3 workflow_notebook.py --section 2

# Data preparation and splitting
python3 workflow_notebook.py --section 3
python3 workflow_notebook.py --section 4

# Augmentation (choose one or both)
python3 workflow_notebook.py --section 5a  # Copy-paste with SAM
python3 workflow_notebook.py --section 5b  # Synthetic with GPT-4o
```

### Training with Different Datasets

```bash
# Train with raw data (no augmentation)
python3 workflow_notebook.py --train --dataset raw

# Train with copy-paste augmented data
python3 workflow_notebook.py --train --dataset cnp

# Train with synthetic augmented data
python3 workflow_notebook.py --train --dataset synthetic

# Auto-detect best available dataset
python3 workflow_notebook.py --train --dataset auto
```

---

## Prerequisites

### Required for All Sections
- Python 3.8+
- Virtual environment activated or `venv/` directory present
- Required packages installed (see `requirements.txt`)
- Internet connection for GBIF data download

### Required for Section 5a (Copy-Paste)
- SAM (Segment Anything Model) checkpoint
- Download: https://github.com/facebookresearch/segment-anything
- Flower background images in `Flowers/` directory

### Required for Section 5b (Synthetic)
- OpenAI API key with GPT-4o access
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
│   ├── prepared/                  # Section 3: YOLO-cropped, 80/20 split
│   ├── prepared_split/            # Section 4: 70/15/15 split
│   ├── prepared_cnp/              # Section 5a: Copy-paste augmented
│   └── prepared_synthetic/        # Section 5b: Synthetic augmented
├── RESULTS/                       # Training results
│   ├── raw_gbif/                  # Baseline model
│   ├── cnp_gbif/                  # Copy-paste model
│   ├── synthetic_gbif/            # Synthetic model
│   └── *_test_results.json        # Test results for each model
└── CACHE_CNP/                     # Copy-paste cache
```

---

## Configuration

Edit `workflow_notebook.py` to customize:

```python
class WorkflowConfig:
    # Directories
    VENV_DIR = Path("./venv")  # Virtual environment location

    # Target species for augmentation
    RARE_SPECIES = ["Bombus_terricola", "Bombus_fervidus"]

    # Augmentation settings
    SYNTHETIC_COUNT_PER_SPECIES = 50  # GPT-4o images per species
    CNP_COUNT_PER_SPECIES = 300       # Copy-paste images per species
```

---

## Monitoring Progress

### During Training
```bash
# In a separate terminal
python3 monitor_training.py

# Or tail the log file
tail -f RESULTS/synthetic_gbif/training.log
```

### Check Results
```bash
# View test results
cat RESULTS/synthetic_gbif_test_results.json

# Compare all methods
cat RESULTS/raw_gbif_test_results.json
cat RESULTS/cnp_gbif_test_results.json
cat RESULTS/synthetic_gbif_test_results.json
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
python3 workflow_notebook.py --interactive
```

### Section Failed
- Each section checks for required inputs from previous sections
- Run sections in order: 1 → 2 → 3 → 4 → (5a or 5b) → 6
- Error messages indicate which section to run first

### "Prepared data directory not found"
- Make sure you've run Sections 1-3 before running Section 4 or later
- Check that `GBIF_MA_BUMBLEBEES/prepared/` exists

### "OpenAI API key not found"
- Create `.env` file in repository root
- Add `OPENAI_API_KEY=your_key` to the file
- Make sure you have API credits

### "SAM checkpoint not found"
- Download SAM checkpoint from GitHub
- Place it in the repository root or update the path
- Model: `sam_vit_h_4b8939.pth`

### Out of Memory During Training
- Reduce batch size in the training script
- Use smaller image size (512 instead of 640)
- Close other GPU applications

### Training is Slow
- Ensure you're using GPU (check CUDA availability)
- Reduce batch size if running out of memory
- Consider using a smaller model architecture

---

## Tips and Best Practices

1. **Start with Interactive Mode**: If you're new to the pipeline, use `--interactive` to understand each section
2. **Run Sections Separately**: For debugging, run sections one at a time
3. **Check Outputs**: After each section, verify the output directory was created
4. **Use Auto Dataset**: When training, `--dataset auto` automatically selects the best available dataset
5. **Monitor Resources**: Keep an eye on disk space (datasets can be large) and GPU memory during training

---

## Example: Complete First Run

```bash
# Step 1: Start interactive mode
python3 workflow_notebook.py --interactive

# Follow the menu:
# - Choose option "all"
# - Select augmentation type: "synthetic"
# - Confirm to start

# The script will run all sections sequentially
# Total time: ~4-10 hours

# Step 2: After completion, check results
ls -la RESULTS/
cat RESULTS/synthetic_gbif_test_results.json
```

---

## Comparison Workflow

Compare all augmentation methods:

```bash
# Run all augmentation methods
python3 workflow_notebook.py --all --augmentation both

# Then train models with each dataset type
python3 workflow_notebook.py --train --dataset raw
python3 workflow_notebook.py --train --dataset cnp
python3 workflow_notebook.py --train --dataset synthetic
```

---

## Existing Scripts Used

This workflow notebook orchestrates the following existing scripts:

- `collect_ma_bumblebees.py` - GBIF data collection
- `analyze_bumblebee_dataset.py` - Dataset analysis
- `prepare_data_only.py` - Data preparation with YOLO
- `split_train_valid_test.py` - Train/valid/test splitting
- `scripts/copy_paste_augment.py` - SAM-based augmentation
- `pipeline_generate_synthetic.py` - GPT-4o synthetic generation
- `pipeline_train_baseline.py` - Model training and testing

No new code is written - this notebook simply provides a convenient interface to run the existing pipeline scripts.

---

## Next Steps After Workflow

1. **Analyze Results**: Compare test performance across different augmentation methods
2. **Validate Synthetic Images**: Run expert validation using `pipeline_validate_synthetic.py`
3. **Experiment with Ratios**: Try different synthetic augmentation ratios using `merge_datasets.py`
4. **Deploy Model**: Use best performing model for deployment

---

## Getting Help

For detailed information:
```bash
python3 workflow_notebook.py --help
```

For issues or questions, see the main repository README or open an issue.

---

## License

See repository LICENSE file.

Option 1: Prevent interruption (recommended)
Run in screen or tmux so it won't stop if terminal disconnects:
screen -S train
python3 workflow_notebook.py --train --dataset raw
# Press Ctrl+A, D to detach
# Reconnect later: screen -r train

CUDA_VISIBLE_DEVICES=1  python3 workflow_notebook.py --train --dataset cnp
CUDA_VISIBLE_DEVICES=2  python3 workflow_notebook.py --train --dataset synthetic