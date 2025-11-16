# Two-Step Copy-Paste Augmentation with Manual QA

This guide explains the new two-step copy-paste augmentation workflow that separates cutout extraction from pasting, allowing manual quality control.

## Overview

The workflow consists of two independent steps:

1. **Extract Cutouts** (`scripts/extract_cutouts.py`)
   - Use SAM to segment bumblebees from training images
   - Save extracted bumblebees as high-quality RGBA PNGs
   - Store in `CACHE_CNP/cutouts/<species>/`

2. **Paste Cutouts** (`scripts/paste_cutouts.py`)
   - Load pre-extracted cutouts (after manual QA)
   - Paste onto flower backgrounds with transformations
   - Apply random rotation and resizing
   - Save augmented images with metadata

## Step 1: Extract Cutouts

### Prerequisites

- SAM checkpoint: `checkpoints/sam_vit_h.pth` (download from SAM repo)
- Training images in: `GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/`
- Python dependencies installed

### Run Extraction

```bash
python scripts/extract_cutouts.py \
  --targets Bombus_sandersoni Bombus_bohemicus \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --sam-checkpoint checkpoints/sam_vit_h.pth
```

### Optional Arguments

- `--extract-all`: Re-extract all cutouts (clears previous cache)
- `--dataset-root`: Path to dataset root (default: `GBIF_MA_BUMBLEBEES`)
- `--sam-checkpoint`: Path to SAM checkpoint (default: `checkpoints/sam_vit_h.pth`)

### Output

Extracted cutouts are saved to:
```
CACHE_CNP/cutouts/<species>/cutout_00000_original_filename.png
CACHE_CNP/cutouts/<species>/cutout_00001_another_image.png
...
```

**Filenames include the original source image name** for easy traceability:
- `cutout_00000_bee_photo_1.png` → came from `bee_photo_1.jpg`
- `cutout_00001_DSC_1234.png` → came from `DSC_1234.jpg`

Each file is a PNG with alpha channel (RGBA) for transparency.

**Extraction log:** `RESULTS/cutout_extraction/extraction_log.json`

## Manual QA: Review and Filter Cutouts

After extraction, review the cutouts in `CACHE_CNP/cutouts/<species>/`:

1. Open each directory in your file manager or image viewer
2. Look for and **DELETE**:
   - Blurry or out-of-focus cutouts
   - Cutouts where the bee is partially cut off
   - Cutouts with poor segmentation
   - Cutouts with wrong subject (not a bee)

3. Keep only **high-quality cutouts** where:
   - The bee is clearly visible and well-segmented
   - The bee occupies ~30-70% of the image
   - Edges are sharp and details are visible
   - The background is relatively clean

**Traceability:** Cutout filenames include the original source image name (e.g., `cutout_00000_my_bee.png` came from `my_bee.jpg`). This makes it easy to trace problematic cutouts back to their sources if needed.

**Tip:** Keep at least 5-10 high-quality cutouts per species for diversity.

## Step 2: Paste Cutouts onto Flower Backgrounds

### Prerequisites

- Completed Step 1 (extracted and reviewed cutouts)
- Flower images in: `flowers/` directory
- Both RGBA cutouts and flower images can be in any image format (JPG, PNG)

### Run Pasting

**Basic usage:**
```bash
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni Bombus_bohemicus \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --per-class-count 100
```

**With custom transformations:**
```bash
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --per-class-count 200 \
  --size-ratio-range 0.15 0.40 \
  --rotation-range -45 45 \
  --paste-position random
```

### Arguments Explained

| Argument | Default | Description |
|----------|---------|-------------|
| `--cutout-species` | Required | Species with cutouts to paste |
| `--flower-dir` | `flowers/` | Directory with flower background images |
| `--dataset-root` | `GBIF_MA_BUMBLEBEES/` | Root dataset directory |
| `--per-class-count` | 100 | Number of images to generate per species |
| `--size-ratio-range` | `0.15 0.35` | Cutout size as fraction of background short side (min max) |
| `--rotation-range` | `-30 30` | Random rotation in degrees (min max) |
| `--paste-position` | `center` | Where to place cutout: `center` or `random` |

**Note:** Augmented images are automatically saved to `DATASET_ROOT/prepared_cnp/train/<species>/`

### Parameters Guide

**Size Ratio Range:**
- `0.15 0.35`: Small cutouts (15-35% of background)
- `0.20 0.50`: Medium cutouts (20-50% of background)
- `0.10 0.40`: Large variation (10-40% of background)

**Rotation Range:**
- `-30 30`: Slight rotation (±30°)
- `-45 45`: Moderate rotation (±45°)
- `-180 180`: Full rotation (any orientation)

**Paste Position:**
- `center`: Always paste in middle (good for consistent placement)
- `random`: Random position with 15-85% margins (more natural variation)

### Output

Augmented images are saved directly to your training dataset:
```
GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/aug_00000.png
GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/aug_00001.png
...
```

Each species directory receives its augmented images. They are immediately available for training alongside original images.

**Generation log with metadata:** `RESULTS/paste_composites/generation_log.json`

Metadata includes:
- Output file path
- Source flower background
- Cutout size and paste position
- Applied rotation angle
- Size ratio used

## Examples

### Example 1: Small cutouts on centered flowers
```bash
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --per-class-count 300 \
  --size-ratio-range 0.10 0.25 \
  --rotation-range -15 15 \
  --paste-position center
```

### Example 2: Large varied cutouts with random placement
```bash
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni Bombus_bohemicus Bombus_ternarius \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --per-class-count 150 \
  --size-ratio-range 0.25 0.50 \
  --rotation-range -60 60 \
  --paste-position random
```

### Example 3: Re-extract cutouts and regenerate (fresh start)
```bash
# Re-extract with --extract-all to clear cache
python scripts/extract_cutouts.py \
  --targets Bombus_sandersoni \
  --extract-all

# Then review cutouts and paste
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES
```

## Workflow Summary

```
GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/ (Raw Training Images)
        ↓
[Step 1] extract_cutouts.py
        ↓
CACHE_CNP/cutouts/<species>/ (RGBA cutouts)
        ↓
[Manual QA] Delete low-quality cutouts
        ↓
CACHE_CNP/cutouts/<species>/ (High-quality only)
        ↓
[Step 2] paste_cutouts.py + flowers/
        ↓
GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/ (Augmented images added)
        ↓
Ready for Training
```

## Troubleshooting

### "No images found in <flower-dir>"
- Check that `flowers/` directory exists
- Verify it contains JPG, PNG, or JPEG images
- Check file permissions

### "No cutouts found for <species>"
- Run Step 1 (extract_cutouts.py) first
- Check that species name matches training directory name
- Verify cutouts were actually extracted (check `CACHE_CNP/cutouts/`)

### Poor augmentation quality
- In Step 1, review and delete low-quality cutouts
- Adjust size-ratio-range in Step 2 for better proportions
- Try different rotation ranges for variety
- Use `--paste-position random` for more natural placement

### Out of memory errors
- Reduce `--per-class-count` to process fewer images
- Use smaller flower images as backgrounds
- Process fewer species at once

## Tips for Best Results

1. **Cutout Quality:** Spend time manually curating cutouts. A few high-quality cutouts produce better augmentations than many poor-quality ones.

2. **Flower Diversity:** Use diverse flower backgrounds (different colors, sizes, poses) for better generalization.

3. **Size Ratios:** Experiment with size ratios to match your target domain:
   - Small bees on large flowers: `0.10 0.25`
   - Medium placement: `0.20 0.40`
   - Large dominant bees: `0.30 0.60`

4. **Rotation:** Use moderate rotations (±30-45°) for realism. Extreme angles may look unnatural.

5. **Position Variation:** Use `random` position for diverse training, `center` for controlled datasets.

6. **Multiple Runs:** Generate multiple times with different parameters to create a diverse synthetic dataset.

## File Locations Reference

| File/Directory | Purpose |
|---|---|
| `scripts/extract_cutouts.py` | Step 1: Extract cutouts |
| `scripts/paste_cutouts.py` | Step 2: Paste cutouts |
| `GBIF_MA_BUMBLEBEES/prepared_cnp/train/` | Input + Output training images (augmented saved here) |
| `CACHE_CNP/cutouts/` | Extracted cutouts (step 1 output, for manual QA) |
| `flowers/` | Flower background images |
| `RESULTS/cutout_extraction/` | Extraction logs and stats |
| `RESULTS/paste_composites/` | Pasting logs and metadata |
| `checkpoints/sam_vit_h.pth` | SAM model checkpoint |

## Dependencies

- `torch` (for SAM)
- `segment-anything` (SAM library)
- `opencv-python` (cv2)
- `Pillow` (PIL)
- `numpy`

Install with:
```bash
pip install torch torchvision opencv-python pillow numpy
pip install git+https://github.com/facebookresearch/segment-anything.git
```

## License & Attribution

- SAM model: [Meta Research](https://github.com/facebookresearch/segment-anything)
- Augmentation technique inspired by copy-paste augmentation for instance segmentation
