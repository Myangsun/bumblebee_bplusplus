# Quick Start: Two-Step Augmentation Workflow

## What Was Created

Two new Python scripts that split the copy-paste augmentation into two independent steps:

1. **`scripts/extract_cutouts.py`** - Extract bumblebee cutouts using SAM
2. **`scripts/paste_cutouts.py`** - Paste cutouts onto flower backgrounds with rotation/resize

Plus comprehensive documentation in `docs/TWO_STEP_CUTOUT_PASTE.md`

## Quick Commands

### Step 1: Extract Cutouts from Training Images

```bash
python scripts/extract_cutouts.py \
  --targets Bombus_sandersoni Bombus_bohemicus \
  --sam-checkpoint checkpoints/sam_vit_h.pth
```

**Output:** High-quality RGBA cutouts in `CACHE_CNP/cutouts/<species>/`
- Filenames include original source image names for traceability
- Example: `cutout_00000_bee_photo_1.png` (from `bee_photo_1.jpg`)

### Step 2: Manually Review Cutouts

Open `CACHE_CNP/cutouts/<species>/` and **DELETE low-quality ones**:
- Blurry or out-of-focus
- Partially cut-off bee
- Poor segmentation
- Keep only the best ones (5-10+ per species)

### Step 3: Paste Cutouts onto Flower Backgrounds

```bash
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni Bombus_bohemicus \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --per-class-count 100
```

**Output:** Augmented images in `GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/`

## Key Features

### Extract Cutouts (`extract_cutouts.py`)
- ✅ Uses SAM (Segment Anything Model) for automatic segmentation
- ✅ Applies Gaussian feathering for soft alpha edges
- ✅ Caches cutouts for reuse and manual inspection
- ✅ Provides extraction statistics and logs

### Paste Cutouts (`paste_cutouts.py`)
- ✅ **Rotation:** Random rotation (configurable range, e.g., -30 to +30°)
- ✅ **Resizing:** Smart resizing based on background size (size ratio range)
- ✅ **Position Control:** Center or random placement
- ✅ **Alpha Blending:** High-quality alpha compositing
- ✅ **Flower Backgrounds:** Uses natural flower images instead of synthetic backgrounds
- ✅ **Metadata Logging:** Records all transformations applied
- ✅ **Dataset Integration:** Saves directly to training dataset

## Configuration Examples

### Conservative (small, well-centered)
```bash
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --per-class-count 100 \
  --size-ratio-range 0.15 0.25 \
  --rotation-range -15 15 \
  --paste-position center
```

### Aggressive (large, varied placement)
```bash
python scripts/paste_cutouts.py \
  --cutout-species Bombus_sandersoni \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --per-class-count 200 \
  --size-ratio-range 0.30 0.50 \
  --rotation-range -60 60 \
  --paste-position random
```

## Directory Structure

```
project/
├── scripts/
│   ├── extract_cutouts.py          ← NEW (Step 1)
│   └── paste_cutouts.py             ← NEW (Step 2)
├── docs/
│   └── TWO_STEP_CUTOUT_PASTE.md     ← NEW (Full guide)
├── checkpoints/
│   └── sam_vit_h.pth                (SAM model, ~2.4GB)
├── GBIF_MA_BUMBLEBEES/
│   └── prepared_cnp/
│       └── train/                   (Training images + AUGMENTED OUTPUT)
│           ├── Bombus_sandersoni/
│           ├── Bombus_bohemicus/
│           └── ...
├── flowers/                          (Flower backgrounds)
├── CACHE_CNP/
│   └── cutouts/                     (Extracted cutouts - for QA)
└── RESULTS/
    ├── cutout_extraction/           (Extraction logs)
    └── paste_composites/            (Pasting logs with metadata)
```

## Important Notes

### Requirements
- SAM checkpoint: `checkpoints/sam_vit_h.pth` (already downloaded ✓)
- Training images: `GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/`
- Flower images: `flowers/` directory with your background images
- Python: `torch`, `segment-anything`, `opencv-python`, `pillow`, `numpy`

### Manual QA is Key
The middle step (manually reviewing cutouts) is crucial! Only keep high-quality cutouts:
- Sharp, well-focused bees
- Good segmentation without artifacts
- Reasonable composition (bee not too small/large in frame)

### Performance Tips
- Keep 5-10 high-quality cutouts per species (quality > quantity)
- Use diverse flower backgrounds (different colors, poses)
- Start with conservative size ratios and rotation (increase for more variation)
- Use `--paste-position center` for clean, consistent results
- Use `--paste-position random` for more naturalistic training diversity

## Parameter Guide

| Parameter | Default | Range | Notes |
|-----------|---------|-------|-------|
| `--per-class-count` | 100 | 1-∞ | Number of augmented images per species |
| `--size-ratio-range` | 0.15 0.35 | 0-1 | Cutout size as % of background short side |
| `--rotation-range` | -30 30 | -180 to 180 | Rotation angle in degrees |
| `--paste-position` | center | center/random | Where to place cutout |

## Comparison with Old Method

### Old (`copy_paste_augment.py`)
- Single script: extract → paste all in one run
- Used SAM-inpainted backgrounds (modified original images)
- Manual review of cutouts was not easy
- Generated and pasted in one go

### New (2-step method)
- **Step 1:** Extract and save cutouts independently
- **Step 2:** Use flower images as backgrounds (clean, natural)
- **Manual QA:** Review and filter cutouts between steps
- **Flexible:** Adjust parameters without re-extracting
- **Reusable:** Use same cutouts multiple times with different settings
- **Integrated:** Saves directly to training dataset

## Next Steps

1. **Verify flower directory exists:**
   ```bash
   ls flowers/ | head
   ```

2. **Run Step 1 to extract cutouts:**
   ```bash
   python scripts/extract_cutouts.py --targets Bombus_sandersoni
   ```

3. **Manually review:** Open `CACHE_CNP/cutouts/Bombus_sandersoni/` and delete low-quality ones

4. **Run Step 2 to generate augmentations:**
   ```bash
   python scripts/paste_cutouts.py \
     --cutout-species Bombus_sandersoni \
     --flower-dir flowers \
     --dataset-root GBIF_MA_BUMBLEBEES
   ```

5. **Check results:** Review generated images in `GBIF_MA_BUMBLEBEES/prepared_cnp/train/Bombus_sandersoni/`

6. **Iterate:** Adjust parameters and run Step 2 again (Step 1 cutouts are cached)

## For Detailed Documentation

See `docs/TWO_STEP_CUTOUT_PASTE.md` for:
- Detailed workflow explanation
- Complete parameter documentation
- Troubleshooting guide
- Advanced examples
- Tips for best results
