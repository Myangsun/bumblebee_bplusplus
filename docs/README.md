# Massachusetts Bumblebee Classification Pipeline
## Rare Species Focus: *Bombus terricola* and *Bombus fervidus*

### 📋 Project Overview

This pipeline addresses the challenge of classifying rare bumblebee species in Massachusetts using:
1. **GBIF biodiversity data** - Real field observations
2. **GPT-4o synthetic augmentation** - AI-generated training images
3. **Computer vision models** - Species classification

### 🎯 Target Rare Species

Both species are designated as **Species at Risk (SP)** in Massachusetts:

#### *Bombus terricola* (Yellow-banded Bumble Bee)
- **Status**: SP, HE (Species at Risk, High Elevation habitat)
- **Key Features**: Abdominal pattern BYYBBB, thorax black on rear 2/3
- **Habitat**: Cool and wet locations
- **Historical decline**: Sharp population decrease since 1920s-30s

#### *Bombus fervidus* (Golden Northern Bumble Bee)
- **Status**: SP, LH (Species at Risk, Low elevation Habitat)
- **Key Features**: Yellow wing pits, thinner black bar, golden coloration
- **Habitat**: Large grasslands in broad valleys
- **Historical decline**: Persists only in fragmented populations

### 📊 Research Questions

Based on your research proposal (Section 2.1 and 2.2):

1. **Effectiveness**: Can GPT-4o generate high-fidelity synthetic data that improves rare species classification?
2. **Biological Fidelity**: Do synthetic images preserve morphological accuracy?
3. **Scaling/Saturation**: What is the optimal ratio of synthetic to real data?
4. **Generalization/Robustness**: Do models trained with synthetic data generalize to real field conditions?

---

## 🚀 Pipeline Workflow

### Step 1: Data Collection
```bash
python collect_ma_bumblebees.py
```

**What it does:**
- Downloads GBIF images for all Massachusetts bumblebee species
- Focuses on 16 species including 2 target rare species
- Downloads 2000 images per species (adjust as needed)
- Filters by geographic location (Massachusetts only)

**Expected output:**
```
./GBIF_MA_BUMBLEBEES/
├── Bombus_terricola/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── Bombus_fervidus/
│   ├── image_001.jpg
│   └── ...
└── [other species]/
```

---

### Step 2: Dataset Analysis
```bash
python analyze_bumblebee_dataset.py
```

**What it does:**
- Counts images per species
- Identifies class imbalance
- Calculates percentage representation of rare species
- Provides recommendations for augmentation

**Key metrics to look for:**
- How many B. terricola images? (Expected: Very low, likely < 100)
- How many B. fervidus images? (Expected: Very low, likely < 100)
- Percentage of dataset? (Expected: < 1% each = severe imbalance)

**Example output:**
```
TARGET RARE SPECIES:
Bombus_terricola: 47 images (0.3% of dataset) ⚠️ CRITICAL
Bombus_fervidus: 62 images (0.4% of dataset) ⚠️ CRITICAL

RECOMMENDATIONS:
- Use synthetic augmentation to upsample rare species
- Consider class weighting in training
- Validate synthetic images with entomologists
```

---

### Step 3: Data Preparation
```bash
# Using bplusplus library
python -c "import bplusplus; bplusplus.prepare(...)"
```

**What it does:**
- Splits data into train/val/test sets (70%/15%/15%)
- Organizes directory structure for training
- Preserves species labels

---

### Step 4: Synthetic Image Generation
```bash
python synthetic_augmentation_gpt4o.py
```

**⚠️ Requirements:**
- OpenAI API key with GPT-4o access
- Sufficient API credits (estimate: ~$0.10-0.20 per image)

**What it does:**
- Loads reference GBIF images for each rare species
- Uses Chain-of-Thought prompting for morphological accuracy
- Generates images with ecological context (host plants, habitats)
- Creates 100+ synthetic images per rare species
- Varies environmental contexts for robustness

**Prompting strategy (from your proposal):**
1. **GPT-4o Multimodal Generation**: Vision-language capabilities
2. **Chain-of-Thought Prompting**: Structured morphological guidance
3. **Few-Shot Learning**: Uses 3 exemplar reference images
4. **Environmental Context Matching**: Habitat-specific backgrounds and host plants

**Example prompt structure:**
```
I need you to generate a Yellow-banded Bumble Bee with:
1. Thorax: BLACK on rear 2/3
2. Abdomen: Pattern B-Y-Y-B-B-B (CRITICAL for identification)
3. Wing pits: YELLOW
4. Face: BLACK hairs
5. Background: Wetland edge, on Goldenrod (Solidago)
6. Photographic quality: Sharp focus, natural lighting
```

---

### Step 5: Baseline Training (GBIF Only)
```bash
python -c "import bplusplus; bplusplus.train(...)"
```

**What it does:**
- Trains classification model on GBIF data only
- Establishes baseline performance
- Tests on held-out GBIF test set

**Expected result:**
- Likely poor performance on B. terricola and B. fervidus
- High accuracy on common species (B. impatiens, B. griseocollis)
- Confusion between rare species and common look-alikes

---

### Step 6: Augmented Training (GBIF + Synthetic)

Train multiple models with varying synthetic augmentation ratios:

```bash
# 10% synthetic, 90% GBIF
# 20% synthetic, 80% GBIF
# ...
# 100% synthetic (equal ratio)
```

**Experimental design (from your proposal):**
- Train with GBIF + Synthetic at ratios: 10%, 20%, 30%, ..., 100%
- Test all models on same GBIF test set
- Compare rare species accuracy across ratios

**Research question**: What is the optimal augmentation ratio?

---

### Step 7: Model Testing & Evaluation

```bash
python -c "import bplusplus; bplusplus.test(...)"
```

**Key metrics to evaluate:**
1. **Overall accuracy** (all species)
2. **Per-species accuracy** (especially B. terricola and B. fervidus)
3. **Confusion matrices** (are rare species confused with common ones?)
4. **F1-scores** for rare species (accounts for class imbalance)

**Expected findings:**
- Baseline model: Poor rare species performance (~20-30% accuracy)
- Augmented models: Improved rare species performance (~60-80% accuracy)
- Saturation point: Diminishing returns after certain augmentation ratio

---

### Step 8: Expert Validation

**Critical step for biological fidelity:**

1. Select a subset of synthetic images (e.g., 50 per species)
2. Present to entomologists for validation
3. Questions to ask:
   - Is this morphologically accurate?
   - Could this be confused with another species?
   - Are the host plant and habitat ecologically appropriate?
4. Calculate validation accuracy

**Validation workflow:**
```python
# Create validation set
synthetic_validation_set = sample_synthetic_images(n=50)

# Expert review
expert_scores = entomologist_review(synthetic_validation_set)

# Calculate metrics
validation_accuracy = sum(expert_scores) / len(expert_scores)
```

---

## 📁 Project Structure

```
.
├── GBIF_MA_BUMBLEBEES/          # Raw GBIF data
│   ├── Bombus_terricola/
│   ├── Bombus_fervidus/
│   └── [other species]/
│
├── SYNTHETIC_BUMBLEBEES/        # Generated synthetic images
│   ├── Bombus_terricola/
│   │   ├── synthetic_001.jpg
│   │   └── generation_log.json
│   └── Bombus_fervidus/
│
├── RESULTS/                     # Model outputs
│   ├── baseline_gbif/
│   ├── augmented_10pct/
│   ├── augmented_20pct/
│   └── ...
│
├── collect_ma_bumblebees.py
├── analyze_bumblebee_dataset.py
├── synthetic_augmentation_gpt4o.py
├── pipeline_workflow.py
└── README.md (this file)
```

---

## 🔬 Key Research Contributions

Based on your proposal, this pipeline addresses:

### 1. Synthetic Augmentation (Section 2.1)
- **LLM-based Generation**: GPT-4o multimodal capabilities
- **Chain-of-Thought Prompting**: Structured morphological guidance
- **Few-Shot Learning**: Exemplar-based generation
- **Domain Alignment**: Ensuring synthetic→real transferability

### 2. Environmental Context Matching (Section 2.2)
- **Context-Aware Generation**: Habitat-specific backgrounds
- **Host-Plant Associations**: Ecologically appropriate pairings
- **Counterfactual Contexts**: Cross-environment variants for robustness

### 3. Conservation Impact
- **MESA Support**: Informs Massachusetts Endangered Species Act mapping
- **Continuous Monitoring**: Edge-AI deployment for long-term tracking
- **Rare Species Detection**: Improved classification for conservation decisions

---

## 📊 Expected Outcomes

### Dataset Statistics (Estimated)
```
GBIF Baseline:
- Total bumblebee images: ~15,000
- B. terricola: ~50 images (0.3%)
- B. fervidus: ~70 images (0.5%)

After Synthetic Augmentation (100% ratio):
- Total: ~15,200 images
- B. terricola: ~150 images (1.0%)
- B. fervidus: ~170 images (1.1%)
```

### Classification Performance (Projected)
```
Baseline (GBIF only):
- B. terricola accuracy: 25-35%
- B. fervidus accuracy: 30-40%

Augmented (Optimal ratio):
- B. terricola accuracy: 65-80%
- B. fervidus accuracy: 70-85%
```

---

## ⚠️ Important Considerations

### 1. Data Quality
- **GBIF images vary in quality** - some may be mislabeled or low resolution
- **Clean your dataset** - manually review rare species images
- **Use iNaturalist's "Research Grade"** observations when possible

### 2. Synthetic Data Risks
- **Morphological errors** - GPT-4o may generate incorrect features
- **Expert validation is CRITICAL** - always validate with entomologists
- **Overfitting to synthetic patterns** - test on real field data

### 3. Conservation Ethics
- **False positives** - Misidentifying common species as rare can waste resources
- **False negatives** - Missing rare species can lead to habitat destruction
- **Validation with field surveys** - AI should complement, not replace, expert observation

### 4. Deployment Considerations
- **Edge devices** need lightweight models (consider MobileNet, EfficientNet)
- **Real-time inference** requires optimization
- **Battery life** for continuous monitoring

---

## 📚 References

From your project knowledge:

1. **Richardson et al. (2019)** - Bumble bee distribution and diversity in Vermont
2. **Jacobson et al. (2018)** - Decline of bumble bees in northeastern North America
3. **Plath, O. (1934)** - Bumblebees and Their Ways (historical baseline)
4. **Field Guides** - Bumble Bees of New England, Gegear Bumblebee ID

---

## 🎯 Next Steps

1. **Run data collection**: `python collect_ma_bumblebees.py`
2. **Analyze dataset**: `python analyze_bumblebee_dataset.py`
3. **Review rare species counts** - Are there enough images?
4. **If counts are very low (<50)**: Proceed with synthetic generation
5. **Generate synthetic images**: Set up OpenAI API and run GPT-4o script
6. **Validate samples**: Send to entomologists for review
7. **Train baseline model**: Establish performance benchmark
8. **Train augmented models**: Test different augmentation ratios
9. **Analyze results**: Focus on rare species performance
10. **Deploy edge devices**: Continuous monitoring at field sites

---

## 📧 Contact & Collaboration

This pipeline supports conservation efforts in Massachusetts, particularly:
- Cambridge Water Department (Fresh Pond Reservation)
- City of Boston Urban Wilds Program
- MassWildlife/NHESP biologists
- Community pollinator stewardship programs

---

## 🙏 Acknowledgments

- **Sarah K. de Coizart Article TENTH Perpetual Charitable Trust** - Funding
- **MIT Senseable City Lab** - Research home
- **Field guides creators** - Abbie Castriotta, Vermont Center for Ecostudies
- **GBIF community** - Open biodiversity data

---

**Good luck with your research! 🐝**

*Remember: The goal is to help conserve these declining species. Every accurate identification contributes to their protection.*
