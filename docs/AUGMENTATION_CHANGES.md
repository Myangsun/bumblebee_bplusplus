# Augmentation Script Updates - Output Folder Change

## Summary
Updated the two-step augmentation pipeline to save outputs directly to the training dataset folder instead of a separate `SYNTHETIC_BUMBLEBEES` directory.

## Changes Made

### 1. `scripts/paste_cutouts.py`
- **Changed argument:** `--output-dir` → `--dataset-root`
- **Default:** `GBIF_MA_BUMBLEBEES`
- **Behavior:** Augmented images are now saved to `GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/`
- **Benefit:** Augmented images are immediately in the training dataset, mixed with original images

### 2. Documentation Updates
Updated all references in:
- `QUICK_START_AUGMENTATION.md`
- `docs/TWO_STEP_CUTOUT_PASTE.md`

Changed all examples from:
```bash
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni \
  --flower-dir flowers \
  --output-dir SYNTHETIC_BUMBLEBEES \
  --per-class-count 100
```

To:
```bash
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --per-class-count 100
```

## Output Structure

### Before
```
project/
├── GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/ (original images only)
└── SYNTHETIC_BUMBLEBEES/<species>/ (augmented images)
```

### After
```
project/
├── GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/
│   ├── original_image_1.jpg
│   ├── original_image_2.jpg
│   ├── aug_00000.png (augmented)
│   ├── aug_00001.png (augmented)
│   └── ... (more augmented)
└── CACHE_CNP/cutouts/ (for manual QA only)
```

## Workflow

```
Step 1: Extract Cutouts
GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/ (input)
        ↓
CACHE_CNP/cutouts/<species>/ (temporary)

Step 2: Manual Review
Delete low-quality cutouts from CACHE_CNP/cutouts/<species>/

Step 3: Paste onto Flowers
CACHE_CNP/cutouts/<species>/ + flowers/
        ↓
GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/ (output - augmented images added)
        ↓
Ready for training with mixed original + augmented images
```

## Quick Start

```bash
# Step 1: Extract
python scripts/extract_cutouts.py \
  --targets Bombus_sandersoni

# Step 2: Review & Delete low-quality cutouts manually
# Open: CACHE_CNP/cutouts/Bombus_sandersoni/

# Step 3: Paste
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --per-class-count 100

# Done! Check results in:
# GBIF_MA_BUMBLEBEES/prepared_cnp/train/Bombus_sandersoni/
```

## Files Modified
- `scripts/paste_cutouts.py` (function calls + docstring + argument)
- `QUICK_START_AUGMENTATION.md` (examples + output directory)
- `docs/TWO_STEP_CUTOUT_PASTE.md` (examples + workflow + file locations)

## No Changes Needed
- `scripts/extract_cutouts.py` (already saves to CACHE_CNP/cutouts/)
- Flower images location (still uses `flowers/` directory)
- SAM checkpoint location (still uses `checkpoints/sam_vit_h.pth`)
