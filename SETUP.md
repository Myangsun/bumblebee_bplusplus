# Setup Guide for BioGen Bumblebee B++ Project

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

#### Option A: Using .env file (Recommended)

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```bash
   # .env file
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```

3. Make sure `.env` is in `.gitignore` to protect your secrets:
   ```bash
   echo ".env" >> .gitignore
   ```

4. The script will automatically load from `.env` when you run it.

#### Option B: Using Environment Variables (Manual)

```bash
# Linux/Mac
export OPENAI_API_KEY='sk-your-actual-api-key-here'

# Then run the script
python pipeline_generate_synthetic.py
```

### 3. Required API Keys

#### OpenAI API
- **Get your API key**: https://platform.openai.com/api-keys
- **Required permissions**:
  - GPT-4 Vision access (for chain-of-thought prompting)
  - DALL-E 3 access (for synthetic image generation)
- **Estimated cost**: ~$0.02-0.10 per image for initial testing

## Pipeline Overview

### Step 1: Collect and Analyze Data
```bash
python pipeline_collect_analyze.py
# Downloads reference images from GBIF for Bombus fervidus
```

### Step 2: Train Baseline Model
```bash
python pipeline_train_baseline.py
# Trains on GBIF data using training_config.yaml settings
```

### Step 3: Generate Synthetic Images (NEW)
```bash
python pipeline_generate_synthetic.py
# Generates 5 test synthetic images with anatomical variations:
# - 3 different photographic angles (dorsal, lateral, frontal)
# - 2 different genders (female/worker, male/drone)
# - Varied host plants and backgrounds
```

**Test Mode**: Currently generates 5 images for evaluation. Edit `generate_synthetic_dataset()` call in `run_generate_synthetic_pipeline()` to scale up to full dataset.

### Output Structure
```
SYNTHETIC_BUMBLEBEES/
└── Bombus_fervidus/
    ├── generation_log.json          # Metadata for all generated images
    └── [generated images here]      # PNG/JPG files (after API processing)
```

## Configuration Files

### training_config.yaml
Centralized configuration for all training hyperparameters:
- Model architecture and backbone
- Training parameters (epochs, batch size, learning rate)
- Optimizer settings (Adam, learning rate schedule)
- Data augmentation strategy

### .env.example
Environment variables template:
- OpenAI API key
- Optional: API organization, timeouts, custom directories

## File Structure

```
bumblebee_bplusplus/
├── pipeline_collect_analyze.py          # Step 1: Data collection
├── pipeline_train_baseline.py           # Step 2: Training
├── pipeline_generate_synthetic.py       # Step 3: Synthetic generation (NEW)
├── pipeline_merge_datasets.py           # Step 4: Merge GBIF + synthetic
├── pipeline_train_augmented.py          # Step 5: Train with augmented data
├── training_config.yaml                 # Training hyperparameters
├── .env.example                         # Environment config template
├── requirements.txt                     # Python dependencies
├── plots/                               # Visualization scripts
├── GBIF_MA_BUMBLEBEES/                  # Reference images (from Step 1)
├── SYNTHETIC_BUMBLEBEES/                # Generated synthetic images
└── RESULTS/                             # Training results and logs
```

## Synthetic Image Generation Details

### Species Focus
**Bombus fervidus** (Golden Northern Bumble Bee)
- Status: Species at Risk
- Habitat: Large grasslands in broad valleys (April-October)
- Size: ~15mm average length

### Anatomical Accuracy
Images are generated with strict morphological requirements:

**Key Identifying Features (Y-Y-Y-Y-B pattern)**:
- Head: Black face with long face structure
- Thorax: Yellow/golden coloration with yellow wing pits
- Abdomen: Yellow on segments T1-T4, Black on T5
- Wings: Dark/dusky translucent wings
- Legs: Yellow with visible pollen baskets

**References**:
- https://www.bumblebeewatch.org/anatomy/
- https://www.bumblebeewatch.org/field-guide/14/

### Variation Combinations (5 test images)

| Image | Gender | Angle | Host Plant | Background |
|-------|--------|-------|-----------|-----------|
| 1 | Female (Worker) | Dorsal (top-down) | Monarda | Open grassland meadow |
| 2 | Female (Worker) | Lateral (side) | Asclepias | Prairie meadow |
| 3 | Female (Worker) | Frontal (face) | Solidago | Broad valley field |
| 4 | Male (Drone) | Dorsal (top-down) | Echinacea | Sunny prairie |
| 5 | Male (Drone) | Lateral (side) | Rudbeckia | Native wildflower field |

## Testing the Setup

```bash
# Test 1: Check dependencies
python -c "import torch, openai, yaml; print('✓ All dependencies loaded')"

# Test 2: Check OpenAI API key
python -c "import os; print('API Key set:', 'OPENAI_API_KEY' in os.environ)"

# Test 3: Run synthetic generation
python pipeline_generate_synthetic.py
```

## Troubleshooting

### ModuleNotFoundError: No module named 'openai'
```bash
pip install --upgrade openai>=1.0.0
```

### OpenAI API Error: "Invalid API key"
1. Verify API key in `.env` file
2. Check API key is active at https://platform.openai.com/api-keys
3. Ensure API key is from your active organization

### GBIF data not found
```bash
# Run data collection first
python pipeline_collect_analyze.py
# Ensure Bombus_fervidus directory exists in GBIF_MA_BUMBLEBEES/
```

### Generation rate limiting (429 error)
- Wait 60 seconds before retrying
- Reduce num_images parameter temporarily for testing
- Check OpenAI API usage at https://platform.openai.com/account/usage

## Next Steps

1. **Generate test images**: Run `python pipeline_generate_synthetic.py`
2. **Review quality**: Check generated images in `SYNTHETIC_BUMBLEBEES/Bombus_fervidus/`
3. **Verify accuracy**: Confirm abdominal color pattern (Y-Y-Y-Y-B) is correct
4. **Scale up**: Increase `num_images` parameter when satisfied with quality
5. **Merge datasets**: Run `pipeline_merge_datasets.py` to combine GBIF + synthetic
6. **Train augmented model**: Run `pipeline_train_augmented.py` with synthetic data

## References

- OpenAI Documentation: https://platform.openai.com/docs/
- DALL-E 3 Guide: https://platform.openai.com/docs/guides/images/generations
- GPT-4 Vision: https://platform.openai.com/docs/guides/vision
- Bumblebee Watch: https://www.bumblebeewatch.org/
