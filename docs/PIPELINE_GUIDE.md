# Modular Pipeline Guide

This guide explains the new modular pipeline structure for the bumblebee classification project.

## Overview

The pipeline is now divided into **4 independent workflows**, each handling a distinct phase of the project:

1. **Pipeline 1: COLLECT & ANALYZE** - Data acquisition and preparation
2. **Pipeline 2: TRAIN BASELINE** - Baseline model training and evaluation
3. **Pipeline 3: GENERATE SYNTHETIC** - AI-based image synthesis
4. **Pipeline 4: VALIDATE SYNTHETIC** - Expert validation materials

## Quick Start

### Option 1: Interactive Menu (Recommended)
```bash
source venv/bin/activate
python main_orchestrator.py
```

This opens an interactive menu where you can:
- Select individual pipelines
- Run preset workflow sequences
- View all available options

### Option 2: Run Specific Pipeline
```bash
source venv/bin/activate
python pipeline_collect_analyze.py
```

Replace `pipeline_collect_analyze.py` with the desired pipeline script.

### Option 3: Command-line Workflow
```bash
source venv/bin/activate
python main_orchestrator.py --workflow basic
```

Available workflows: `basic`, `full`, `synthetic`

---

## Pipeline 1: COLLECT & ANALYZE

**Purpose**: Download GBIF data and prepare train/val/test splits

**Steps**:
1. Collect GBIF images for all MA bumblebee species
2. Analyze dataset distribution (identify class imbalance)
3. Prepare data using YOLO detection and create train/valid splits

**Run**:
```bash
python pipeline_collect_analyze.py
```

**Outputs**:
- `./GBIF_MA_BUMBLEBEES/` - Raw GBIF images organized by species
- `./GBIF_MA_BUMBLEBEES/prepared/` - YOLO-processed train/valid splits
  - `prepared/train/` - Training dataset (80%)
  - `prepared/valid/` - Validation dataset (20%)
- `./RESULTS/data_preparation_metadata.json` - Dataset statistics

**Expected Duration**: 30-60 minutes (depends on internet speed and GBIF availability)

**Key Metrics**:
```
Bombus terricola: ~50 images (0.3% of dataset) ⚠️ CRITICAL
Bombus fervidus: ~70 images (0.5% of dataset) ⚠️ CRITICAL
```

---

## Pipeline 2: TRAIN BASELINE

**Purpose**: Establish baseline model performance on GBIF data only

**Steps**:
1. Train ResNet50 classification model on prepared GBIF training data
2. Test model on held-out validation set
3. Generate performance metrics

**Run**:
```bash
python pipeline_train_baseline.py
```

**Requirements**:
- ✓ Must run Pipeline 1 first
- GPU recommended but not required

**Outputs**:
- `./RESULTS/baseline_gbif/` - Trained model weights and architecture
- `./RESULTS/baseline_results.json` - Evaluation metrics

**Expected Training Time**:
- With GPU: 10-30 minutes per epoch
- Without GPU: 1-3 hours per epoch
- Default: 10 epochs (reduce to 3-5 for testing)

**Key Metrics to Check**:
```
Overall Accuracy: ~70-80% (common species)
B. terricola Accuracy: ~20-30% (rare species - poor performance expected)
B. fervidus Accuracy: ~25-35% (rare species - poor performance expected)
```

---

## Pipeline 3: GENERATE SYNTHETIC

**Purpose**: Create synthetic training images using GPT-4o/DALL-E 3

**Steps**:
1. Load reference GBIF images for rare species
2. Generate synthetic images with morphological accuracy
3. Create diverse variants with different environmental contexts
4. Log all generation attempts and results

**Run**:
```bash
python pipeline_generate_synthetic.py
```

**Requirements**:
- ✓ Must run Pipeline 1 first
- ✓ OpenAI API key with GPT-4o + DALL-E 3 access
- ✓ Sufficient API credits (~$0.10-0.20 per image)

**Setup API Key**:
```bash
# Option 1: Set environment variable
export OPENAI_API_KEY='sk-...'

# Option 2: Provide when prompted
python pipeline_generate_synthetic.py
# Enter API key when asked
```

**Outputs**:
- `./SYNTHETIC_BUMBLEBEES/Bombus_terricola/` - Generated images
- `./SYNTHETIC_BUMBLEBEES/Bombus_fervidus/` - Generated images
- `./SYNTHETIC_BUMBLEBEES/*/generation_log.json` - Generation metadata

**Prompting Strategy**:
- **Chain-of-Thought**: Detailed morphological requirements
- **Few-Shot Learning**: Reference images for guidance
- **Environmental Context**: Ecological realism (host plants, habitats)
- **Diversity**: Vary backgrounds and plant associations

**Example Generated Features**:
```
Bombus terricola:
- Abdominal pattern: B-Y-Y-B-B-B (critical)
- Thorax: Black rear 2/3
- Contexts: Wetland edges, forest clearings, on Goldenrod

Bombus fervidus:
- Wing pits: Yellow
- Face: Yellow hairs
- Contexts: Open grasslands, on Bee balm
```

---

## Pipeline 4: VALIDATE SYNTHETIC

**Purpose**: Prepare materials for expert (entomologist) validation

**Outputs**:
- `./RESULTS/validation/VALIDATION_INSTRUCTIONS.md` - Expert guidelines
- `./RESULTS/validation/Bombus_*/sample/` - Sample images for review
- `./RESULTS/validation/*_validation_form.json` - Response templates
- `./RESULTS/validation/*_validation_report_template.json` - Results summary template

**Run**:
```bash
python pipeline_validate_synthetic.py
```

**Requirements**:
- ✓ Must run Pipeline 3 first

**Validation Process**:
1. Share `./RESULTS/validation/` with entomologists
2. Experts review VALIDATION_INSTRUCTIONS.md
3. Experts examine sample images in `Bombus_*/sample/`
4. Experts complete validation form with ratings
5. Return completed `*_validation_report_*.json` files

**Key Validation Questions**:
- Is morphology accurate for the species?
- Could it be confused with another species?
- Is host plant and habitat ecologically appropriate?
- Is photographic quality realistic?
- Would you use this for training an AI model?

**Validation Rating Scale**:
- ✓ "Yes, definitely" → Use for training (high confidence)
- ~ "Mostly accurate" → Use with caution
- ? "Somewhat accurate" → Review before use
- ✗ "Not accurate" → Do not use

---

## Workflow Sequences

### BASIC WORKFLOW
For quick testing and baseline establishment:

```bash
python main_orchestrator.py --workflow basic
```

**Pipelines**: 1 → 2

**Duration**: ~1-2 hours
**Output**: Baseline model performance on GBIF data

### FULL WORKFLOW
Complete pipeline including synthetic augmentation:

```bash
python main_orchestrator.py --workflow full
```

**Pipelines**: 1 → 2 → 3 → 4

**Duration**: 2-3 hours + API time + expert validation
**Output**: Synthetic images and validation materials

### SYNTHETIC WORKFLOW
From baseline through synthetic generation and validation:

```bash
python main_orchestrator.py --workflow synthetic
```

**Pipelines**: 1 → 2 → 3 → 4

---

## File Structure

```
bumblebee_bplusplus/
├── GBIF_MA_BUMBLEBEES/              # GBIF raw data
│   ├── Bombus_terricola/
│   ├── Bombus_fervidus/
│   ├── Bombus_impatiens/
│   └── prepared/                    # Train/Val/Test splits
│
├── SYNTHETIC_BUMBLEBEES/            # Generated synthetic images
│   ├── Bombus_terricola/
│   │   ├── synthetic_001.jpg
│   │   └── generation_log.json
│   └── Bombus_fervidus/
│
├── RESULTS/                         # All results and outputs
│   ├── baseline_gbif/               # Baseline model
│   ├── baseline_results.json        # Baseline metrics
│   ├── data_preparation_metadata.json
│   └── validation/                  # Expert validation materials
│       ├── VALIDATION_INSTRUCTIONS.md
│       ├── Bombus_terricola_sample/
│       ├── Bombus_fervidus_sample/
│       ├── *_validation_form.json
│       └── *_validation_report_template.json
│
├── pipeline_collect_analyze.py      # Pipeline 1
├── pipeline_train_baseline.py       # Pipeline 2
├── pipeline_generate_synthetic.py   # Pipeline 3
├── pipeline_validate_synthetic.py   # Pipeline 4
├── main_orchestrator.py             # Menu system
│
├── collect_ma_bumblebees.py         # Data collection helper
├── analyze_bumblebee_dataset.py     # Analysis helper
├── merge_datasets.py                # Merge GBIF + synthetic
├── synthetic_augmentation_gpt4o.py  # Generation helper
└── PIPELINE_GUIDE.md                # This file
```

---

## Key Parameters to Adjust

### Data Collection
File: `collect_ma_bumblebees.py`
- `MAX_IMAGES_PER_SPECIES`: Default 2000 (adjust for bandwidth)
- `TARGET_SPECIES`: List of species to download

### Model Training
File: `pipeline_train_baseline.py`
- `epochs`: Default 10 (increase to 50-100 for production)
- `model_name`: ResNet50 (alternative: "yolov8n", "resnet18")

### Synthetic Generation
File: `pipeline_generate_synthetic.py`
- `num_images`: Default 10 per species (increase to 100-200)
- `api_key`: Requires OpenAI API key

### Data Preparation
File: `pipeline_collect_analyze.py`
- **Note**: `bplusplus.prepare()` automatically detects bumblebees with YOLO and creates train/valid splits (not configurable)
- Splits are approximately 80% train / 20% valid
- Images are cropped to focus on the detected bumblebee region

---

## Troubleshooting

### Pipeline 1: Data Collection Issues
```
Error: "Maximum retries exceeded"
→ Check internet connection
→ GBIF API may be rate-limiting
→ Reduce MAX_IMAGES_PER_SPECIES and retry

Error: "No species directories found"
→ GBIF download failed
→ Check GBIF_MA_BUMBLEBEES/ directory
→ Try running collect_ma_bumblebees.py directly
```

### Pipeline 2: Training Issues
```
Error: "CUDA out of memory"
→ Reduce batch size in pipeline_train_baseline.py
→ Use CPU instead (slower but works)
→ Reduce number of epochs

Error: "Module bplusplus not found"
→ Activate virtual environment: source venv/bin/activate
→ Reinstall: pip install bplusplus
```

### Pipeline 3: Synthetic Generation Issues
```
Error: "Invalid API key"
→ Check OPENAI_API_KEY environment variable
→ Verify API key has GPT-4o + DALL-E 3 access
→ Check account has sufficient credits

Error: "Rate limit exceeded"
→ API is limiting requests
→ Reduce num_images or add more delay between calls
→ Try again later
```

### Pipeline 4: Validation Issues
```
Error: "No synthetic images found"
→ Pipeline 3 must be run first
→ Check SYNTHETIC_BUMBLEBEES/ directory exists
→ Verify images were successfully generated
```

---

## Next Steps After Each Pipeline

### After Pipeline 1:
- Review `RESULTS/data_preparation_metadata.json`
- Check class imbalance statistics
- Verify train/val/test splits were created

### After Pipeline 2:
- Review `RESULTS/baseline_results.json`
- Check accuracy for B. terricola and B. fervidus
- Note the performance gap (baseline should be poor for rare species)

### After Pipeline 3:
- Manually inspect sample images in `SYNTHETIC_BUMBLEBEES/*/`
- Check `*/generation_log.json` for generation details
- Run Pipeline 4 to prepare validation materials

### After Pipeline 4:
- Share validation materials with entomology experts
- Collect expert feedback in validation reports
- Use feedback to refine prompts and regenerate if needed

---

## Advanced: Custom Pipelines

You can also run individual pipeline steps or combine them differently:

```bash
# Run only step 1
python pipeline_collect_analyze.py

# Run custom workflow by editing main_orchestrator.py
# Add new workflow definition to WORKFLOWS dict
```

---

## References

- **bplusplus**: https://pypi.org/project/bplusplus/
- **OpenAI API**: https://platform.openai.com/docs/
- **GBIF API**: https://www.gbif.org/developer
- **YOLOv8**: https://docs.ultralytics.com/

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review log files in `RESULTS/` directories
3. Check bplusplus documentation
4. Contact research team

**Good luck with your bumblebee research! 🐝**
