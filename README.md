# Bumblebee Classification Pipeline

Quick start guide for collecting GBIF data, analyzing dataset imbalance, preparing images, and training a baseline classification model for Massachusetts bumblebee species.

## Prerequisites

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Requires: Python 3.9+, PyTorch with CUDA support (optional but recommended for training)

---

## Quick Start (Recommended)

### Step 1: Collect, Analyze & Prepare Data

```bash
python pipeline_collect_analyze.py
```

Automatically:
- Downloads GBIF images for Massachusetts bumblebee species
- Analyzes class imbalance (species counts and distribution)
- Detects with YOLO, crops to 640×640, filters corrupted images
- Creates train/valid splits (80%/20%)

**Output:**
```
GBIF_MA_BUMBLEBEES/
├── [raw species folders]/
└── prepared/
    ├── train/
    └── valid/
```

---

### Step 2 (Optional): Create Test Set

```bash
python split_train_valid_test.py
```

Reorganizes data into proper train/valid/test splits (70%/15%/15%) if you need a separate evaluation set.

**Output:**
```
GBIF_MA_BUMBLEBEES/prepared_split/
├── train/    (70%)
├── valid/    (15%)
└── test/     (15%)
```

---

### Step 3: Train Baseline Model

```bash
python pipeline_train_baseline.py
```

Trains hierarchical classifier (Family → Genus → Species) on GBIF data only.

**Training parameters:**
- batch_size: 16
- epochs: 50
- patience: 10 (early stopping)
- num_workers: 1
- img_size: 640

**Output:**
```
RESULTS/baseline_gbif/
├── best_multitask.pt      # Trained model weights
├── final_multitask.pt     # Final epoch weights
├── training_log.json      # Training history
├── metrics.json           # Accuracy/F1 scores
└── training_metadata.json # Parameters used
```

---

## Workflow Summary

**Minimal (80/20 train/valid only):**
```bash
python pipeline_collect_analyze.py
python pipeline_train_baseline.py
```

**Full (with separate test set):**
```bash
python pipeline_collect_analyze.py
python split_train_valid_test.py
python pipeline_train_baseline.py
```

**Note:** Training script auto-detects data structure:
- If `prepared_split/` exists → uses train/valid/test (70/15/15)
- Otherwise → uses prepared/train and prepared/valid (80/20)

---

## Expected Results

| Metric | Value |
|--------|-------|
| Training time | ~1-2 hours (V100 GPU) |
| Common species accuracy | 85-95% |
| Rare species accuracy | 20-40% (baseline) |
| Test set size | ~15% of total |

---

## Troubleshooting

**GPU out of memory:** Reduce `batch_size` to 8 or 4 in `pipeline_train_baseline.py`

**YOLO detection fails:** Check image format (JPG/PNG). Run `python analyze_bumblebee_dataset.py` to verify data.

**Missing bplusplus:** Ensure all dependencies in `requirements.txt` are installed.

**No GBIF data:** Run `python collect_ma_bumblebees.py` first (may take 30+ minutes).

---

## Next Steps

1. After baseline training completes, run validation on test set
2. Generate synthetic images for augmentation (see `synthetic_augmentation_gpt4o.py`)
3. Train augmented models with varying synthetic ratios
4. Compare rare species performance across models

See `docs/README.md` for detailed research background and advanced usage.
