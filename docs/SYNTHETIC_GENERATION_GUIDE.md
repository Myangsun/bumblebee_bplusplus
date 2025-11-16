# Synthetic Image Generation Guide

This guide explains the refactored synthetic image generation pipeline for copy-paste augmentation.

## Overview

The new pipeline allows you to generate synthetic bumblebee images for rare/minority species using DALL-E 3. The system uses:

- **Species Configuration** (`species_config.json`): Anatomical details and reference images for each species
- **Simpler Prompts**: Structured but concise prompts that focus on key identifying features
- **Output Location**: `GBIF_MA_BUMBLEBEES/prepared_synthetic/train/<species>/`

## Setup

### 1. Update Reference Image Paths in `species_config.json`

The config file needs actual paths to reference images. Update the `reference_images` field for each species:

```json
{
  "species": {
    "Bombus_ashtoni": {
      "reference_images": [
        "GBIF_MA_BUMBLEBEES/Bombus_ashtoni/image1.jpg",
        "GBIF_MA_BUMBLEBEES/Bombus_ashtoni/image2.jpg",
        "GBIF_MA_BUMBLEBEES/Bombus_ashtoni/image3.jpg"
      ]
    }
  }
}
```

**Note**: The script will automatically find images in `GBIF_MA_BUMBLEBEES/<species>/` directory. You can leave these as-is and the script will auto-discover them.

### 2. Set OpenAI API Key

```bash
export OPENAI_API_KEY='sk-...'
```

Or provide it when running the script.

## Usage

### Basic: Generate 50 images for one species

```bash
python pipeline_generate_synthetic.py \
  --species Bombus_ashtoni \
  --count 50
```

### Generate for multiple species

```bash
python pipeline_generate_synthetic.py \
  --species Bombus_ashtoni Bombus_sandersoni Bombus_ternarius_Say \
  --count 30
```

### Generate for all configured species

```bash
python pipeline_generate_synthetic.py \
  --all \
  --count 100
```

### Specify API key on command line

```bash
python pipeline_generate_synthetic.py \
  --species Bombus_ashtoni \
  --count 50 \
  --api-key sk-...
```

## Output Structure

Generated images are saved to:

```
GBIF_MA_BUMBLEBEES/prepared_synthetic/train/
├── Bombus_ashtoni/
│   ├── synthetic_00001_natural.png
│   ├── synthetic_00002_dorsal.png
│   ├── synthetic_00003_lateral.png
│   ├── synthetic_00004_frontal.png
│   ├── generation_metadata.json
│   └── ... (more images)
├── Bombus_sandersoni/
│   ├── synthetic_00001_natural.png
│   └── ...
└── Bombus_ternarius_Say/
    ├── synthetic_00001_natural.png
    └── ...
```

Each image filename includes:
- `synthetic_`: Prefix indicating synthetic generation
- `XXXXX`: Sequential number (00001, 00002, etc.)
- `angle`: Photographic angle (natural, dorsal, lateral, frontal)

## Configuration File Format

The `species_config.json` defines:

### Per-Species Configuration

```json
{
  "Bombus_ashtoni": {
    "common_name": "Ashton's Bumblebee",
    "status": "Rare/At-risk species",

    "reference_images": [
      "GBIF_MA_BUMBLEBEES/Bombus_ashtoni/image1.jpg",
      ...
    ],

    "key_features": [
      "Black head with minimal yellow markings",
      "Yellow thorax with some black hairs",
      ...
    ],

    "anatomical_notes": {
      "head": "Mostly black face...",
      "thorax": "Yellow hair with some black...",
      "abdomen": "Yellow on T1-T2, Black on T3-T6...",
      "wings": "Dark and translucent",
      "legs": "Yellow with pollen baskets..."
    },

    "typical_backgrounds": [
      "mixed forest understory",
      "woodland edge with flowering plants",
      ...
    ],

    "host_plants": [
      "Solidago (Goldenrod)",
      "Liatris (Blazing star)",
      ...
    ]
  }
}
```

## Customizing Species Configuration

To add or modify a species:

1. Edit `species_config.json`
2. Ensure reference images actually exist in `GBIF_MA_BUMBLEBEES/<species>/`
3. Update `key_features` and `anatomical_notes` with accurate species information
4. Add typical backgrounds and host plants for ecological context

## Prompt Generation

The script creates structured prompts automatically:

```
Generate a realistic, high-quality photograph of a [COMMON_NAME] ([species]).

CRITICAL IDENTIFYING FEATURES - MUST INCLUDE ALL:
• [key_feature_1]
• [key_feature_2]
...

ANATOMICAL DETAILS:
• Head: [description]
• Thorax: [description]
• Abdomen: [description]
• Wings: [description]
• Legs: [description]

ENVIRONMENTAL CONTEXT:
• Habitat: [random habitat from config]
• Visiting flower/plant: [random plant from config]
• Photographic angle: [natural/dorsal/lateral/frontal]
• Lighting: Natural daylight, clear visibility of all features

IMPORTANT:
- Photograph should be realistic and scientifically accurate
- All identifying features MUST be clearly visible
- Image will be used for training AI classification models
- Morphological accuracy is critical
```

**Key Points**:
- Anatomically simple but accurate
- Photographic angle varies (natural, dorsal, lateral, frontal)
- Environmental context randomized for diversity
- Emphasizes importance of accuracy for AI training

## Workflow

### Step 1: Review & Update Configuration

```bash
# Edit species_config.json
# Verify reference images exist:
ls GBIF_MA_BUMBLEBEES/Bombus_ashtoni/ | head
```

### Step 2: Generate Synthetic Images

```bash
# Generate 50 images per species
python pipeline_generate_synthetic.py \
  --species Bombus_ashtoni \
  --count 50
```

### Step 3: Review Generated Images

```bash
# Check output
ls GBIF_MA_BUMBLEBEES/prepared_synthetic/train/Bombus_ashtoni/

# View metadata
cat GBIF_MA_BUMBLEBEES/prepared_synthetic/train/Bombus_ashtoni/generation_metadata.json
```

### Step 4: Use for Copy-Paste Augmentation

Extract cutouts from the synthetic images and use them for pasting:

```bash
# Extract cutouts from synthetic images
python scripts/extract_cutouts.py \
  --targets Bombus_ashtoni \
  --dataset-root GBIF_MA_BUMBLEBEES

# Manual QA: Review and delete low-quality cutouts
# Then paste onto flower backgrounds:
python scripts/paste_cutouts.py \
  --cutout-species Bombus_ashtoni \
  --flower-dir flowers \
  --dataset-root GBIF_MA_BUMBLEBEES \
  --per-class-count 100
```

### Step 5: Train with Augmented Data

```bash
python pipeline_train_baseline.py
```

The pipeline auto-detects `prepared_synthetic/` and trains with augmented data.

## Customization Examples

### Example 1: Different Angle Distributions

Edit `pipeline_generate_synthetic.py` line 319 to change angle distribution:

```python
angles = ["natural", "natural", "natural", "dorsal", "lateral", "frontal"]  # More natural views
```

### Example 2: Generate Only for Specific Species

```bash
python pipeline_generate_synthetic.py \
  --species Bombus_ashtoni \
  --count 100
```

### Example 3: Generate Large Dataset

```bash
python pipeline_generate_synthetic.py \
  --all \
  --count 500  # 500 images per species
```

## Troubleshooting

### "Species not in config"

```bash
✗ Species not in config: Bombus_xyz
  Available: Bombus_ashtoni, Bombus_sandersoni, Bombus_ternarius_Say
```

**Solution**: Add the species to `species_config.json`

### "No reference images found"

```
Warning: No reference images found in GBIF_MA_BUMBLEBEES/Bombus_ashtoni/
```

**Solution**: Ensure reference images exist:

```bash
ls GBIF_MA_BUMBLEBEES/Bombus_ashtoni/
```

If empty, run data collection first:

```bash
python pipeline_collect_analyze.py
```

### API Errors

If you get OpenAI API errors:

1. Check API key is valid: `echo $OPENAI_API_KEY`
2. Verify DALL-E 3 access on your account
3. Check rate limits (DALL-E 3 is rate-limited)
4. For rate limit issues, increase `time.sleep()` in the script (line 343)

### Rate Limiting

DALL-E 3 has rate limits. If you hit them:

1. Wait and try again later
2. Generate fewer images at once
3. Reduce `--count` parameter

## Quality Assurance

Generated images should be reviewed for:

1. **Morphological Accuracy**: All key features present
2. **Anatomical Correctness**: Coloration patterns correct
3. **Realism**: Looks like actual photograph
4. **Clarity**: All features clearly visible (not blurry/obscured)

Delete images that fail these checks before using for training:

```bash
rm GBIF_MA_BUMBLEBEES/prepared_synthetic/Bombus_ashtoni/synthetic_*.png
```

## Cost Estimation

DALL-E 3 pricing (as of 2024):
- Standard quality: $0.040 per image (1024x1024)
- HD quality: $0.080 per image (1024x1024)

This script uses HD quality.

**Examples**:
- 50 images × $0.080 = $4.00
- 500 images × $0.080 = $40.00
- 1000 images × $0.080 = $80.00

## Next Steps

After generating synthetic images:

1. Extract cutouts: `python scripts/extract_cutouts.py`
2. Manual QA: Review cutouts
3. Paste onto flowers: `python scripts/paste_cutouts.py`
4. Train model: `python pipeline_train_baseline.py`

## Files Reference

| File | Purpose |
|------|---------|
| `species_config.json` | Species configuration (anatomical features, reference images) |
| `pipeline_generate_synthetic.py` | Main generation script |
| `GBIF_MA_BUMBLEBEES/prepared_synthetic/` | Output directory for synthetic images |
| `RESULTS/` | Generation metadata and logs |

