# Documentation Index

Welcome! This folder contains all documentation for the Massachusetts Bumblebee Classification Pipeline project.

---

## 📚 Quick Navigation

### Getting Started
- **[README.md](README.md)** - Project overview, objectives, and high-level architecture
- **[SETUP_SUMMARY.md](SETUP_SUMMARY.md)** - Environment setup, what was created, and quick start guide

### Pipeline Documentation
- **[PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)** - **START HERE** - Comprehensive guide to all 4 pipelines and workflows
  - Detailed instructions for each pipeline
  - Parameter tuning
  - Troubleshooting

### Data Management
- **[DATASET_ANALYSIS_REPORT.md](DATASET_ANALYSIS_REPORT.md)** - Original GBIF dataset analysis
  - Species distribution
  - Class imbalance analysis
  - Data quality overview

- **[DATA_LOSS_EXPLANATION.md](DATA_LOSS_EXPLANATION.md)** - Explanation of data loss through bplusplus.prepare()
  - Why 16,672 → 6,821 images
  - Quality filtering analysis
  - Is this expected? (YES!)

- **[DATA_LOSS_SUMMARY.txt](DATA_LOSS_SUMMARY.txt)** - Quick reference for data loss
  - Quick answer format
  - Key takeaways
  - What to do next

- **[PIPELINE_FLOW_VISUAL.txt](PIPELINE_FLOW_VISUAL.txt)** - Visual diagram of data flow
  - ASCII diagram of bplusplus.prepare() pipeline
  - Loss analysis visualization
  - Quality improvement overview

### Train/Valid/Test Split
- **[TRAIN_VALID_TEST_SPLIT_COMPLETE.md](TRAIN_VALID_TEST_SPLIT_COMPLETE.md)** - 70/15/15 split documentation
  - Dataset statistics
  - Distribution tables
  - Usage instructions
  - Per-species breakdown

- **[SPLIT_COMPLETE_SUMMARY.txt](SPLIT_COMPLETE_SUMMARY.txt)** - Quick summary of train/valid/test split
  - What was done
  - Dataset statistics
  - Next steps
  - Ready to train checklist

---

## 🗂️ File Organization

```
docs/
├── INDEX.md (this file)
├── README.md (project overview)
├── SETUP_SUMMARY.md (environment setup)
├── PIPELINE_GUIDE.md (comprehensive pipeline guide)
├── DATASET_ANALYSIS_REPORT.md (original dataset analysis)
├── DATA_LOSS_EXPLANATION.md (bplusplus.prepare() data flow)
├── DATA_LOSS_SUMMARY.txt (quick data loss reference)
├── PIPELINE_FLOW_VISUAL.txt (ASCII diagram)
├── TRAIN_VALID_TEST_SPLIT_COMPLETE.md (train/valid/test split)
└── SPLIT_COMPLETE_SUMMARY.txt (split summary)
```

---

## 🎯 Common Tasks

### I want to understand the project
1. Read: [README.md](README.md)
2. Read: [SETUP_SUMMARY.md](SETUP_SUMMARY.md)
3. Read: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)

### I want to train a baseline model
1. Read: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Pipeline 2 section
2. Run: `python pipeline_train_baseline.py`

### I want to generate synthetic images
1. Read: [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Pipeline 3 section
2. Run: `python pipeline_generate_synthetic.py`

### I'm concerned about data loss
1. Read: [DATA_LOSS_SUMMARY.txt](DATA_LOSS_SUMMARY.txt) (quick answer)
2. Read: [DATA_LOSS_EXPLANATION.md](DATA_LOSS_EXPLANATION.md) (detailed explanation)
3. Read: [PIPELINE_FLOW_VISUAL.txt](PIPELINE_FLOW_VISUAL.txt) (visual diagram)

### I want to understand the train/valid/test split
1. Read: [SPLIT_COMPLETE_SUMMARY.txt](SPLIT_COMPLETE_SUMMARY.txt) (quick summary)
2. Read: [TRAIN_VALID_TEST_SPLIT_COMPLETE.md](TRAIN_VALID_TEST_SPLIT_COMPLETE.md) (detailed info)

### I want to understand the original dataset
1. Read: [DATASET_ANALYSIS_REPORT.md](DATASET_ANALYSIS_REPORT.md)

---

## 📊 Key Statistics at a Glance

### Original Dataset (from GBIF)
```
Total images: 16,672
Rare species:
  - Bombus_terricola: 1,033 images (7.2%)
  - Bombus_fervidus: 259 images (1.8%)
```

### After bplusplus.prepare()
```
Total images: 6,821 (59% loss due to quality filtering)
Rare species:
  - Bombus_terricola: 503 images
  - Bombus_fervidus: 155 images
```

### After Train/Valid/Test Split (70/15/15)
```
Total: 6,821 images
├── Train: 4,768 (69.9%)
├── Valid: 1,015 (14.9%)
└── Test:  1,038 (15.2%)

Rare species in training:
  - Bombus_terricola: 352 training images
  - Bombus_fervidus: 108 training images
```

---

## 🚀 Quick Start

```bash
# 1. Data preparation (if not done yet)
python pipeline_collect_analyze.py

# 2. Create train/valid/test split (if not done yet)
python split_train_valid_test.py

# 3. Train baseline model
python pipeline_train_baseline.py

# 4. (Optional) Generate synthetic images
python pipeline_generate_synthetic.py

# 5. (Optional) Validate synthetic images
python pipeline_validate_synthetic.py
```

---

## 📖 Document Descriptions

### README.md (11 KB)
Project overview including objectives, target species, pipeline architecture, expected outcomes, and acknowledgments.

### SETUP_SUMMARY.md (10 KB)
Complete summary of environment setup, what was created, key improvements, and recommended next steps.

### PIPELINE_GUIDE.md (11 KB)
**Most comprehensive reference**. Contains detailed instructions for:
- Pipeline 1: Collect & Analyze
- Pipeline 2: Train Baseline
- Pipeline 3: Generate Synthetic
- Pipeline 4: Validate Synthetic
- Workflow sequences
- Troubleshooting

### DATASET_ANALYSIS_REPORT.md (7.1 KB)
Analysis of original GBIF dataset including:
- Species distribution
- Why 2 species have 0 images
- Rare species coverage
- Recommendations

### DATA_LOSS_EXPLANATION.md (6.7 KB)
Detailed explanation of data loss through bplusplus.prepare():
- Where the 16,672 → 6,821 image loss comes from
- Is it good or bad? (YES, it's good!)
- Quality filtering benefits
- FAQ

### DATA_LOSS_SUMMARY.txt (4.8 KB)
Quick reference version of data loss explanation:
- Your exact numbers
- Where data went
- Why this is good
- Key takeaway

### PIPELINE_FLOW_VISUAL.txt (6.3 KB)
Visual ASCII diagram showing:
- Data flow through bplusplus.prepare()
- Loss at each stage
- Quality improvement
- Final output structure

### TRAIN_VALID_TEST_SPLIT_COMPLETE.md (6.1 KB)
Complete documentation of train/valid/test split:
- Dataset statistics
- Per-species distribution
- Why this is better
- How to use

### SPLIT_COMPLETE_SUMMARY.txt (5.7 KB)
Quick summary of train/valid/test split:
- What was done
- Statistics
- Next steps
- Ready for training checklist

---

## 🔑 Key Concepts

### Rare Species (Your Research Focus)
- **Bombus_terricola** (Yellow-banded Bumble Bee)
  - Status: Species at Risk (SP, HE)
  - Original: 1,033 images
  - After split: 352 training images ✓

- **Bombus_fervidus** (Golden Northern Bumble Bee)
  - Status: Species at Risk (SP, LH)
  - Original: 259 images
  - After split: 108 training images ✓

### Pipelines
1. **Collect & Analyze** - Download and prepare data
2. **Train Baseline** - Establish baseline performance
3. **Generate Synthetic** - Create synthetic images with GPT-4o
4. **Validate Synthetic** - Expert validation materials

### Data Processing Steps
1. GBIF collection → 16,672 raw images
2. bplusplus.prepare() → 6,821 quality-filtered images (59% loss due to filtering)
3. Train/Valid/Test split → 4,768 / 1,015 / 1,038 images
4. Training, validation, and testing

---

## ❓ FAQ

**Q: Where do I start?**
A: Read [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) for comprehensive overview.

**Q: Why were 59% of images lost?**
A: Read [DATA_LOSS_EXPLANATION.md](DATA_LOSS_EXPLANATION.md) - it's expected quality filtering!

**Q: Should I re-download data?**
A: No! The loss is intentional and improves data quality. See [DATA_LOSS_SUMMARY.txt](DATA_LOSS_SUMMARY.txt).

**Q: How do I train the model?**
A: Read [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md) - Pipeline 2 section, then run `python pipeline_train_baseline.py`.

**Q: What's the train/valid/test split?**
A: Read [TRAIN_VALID_TEST_SPLIT_COMPLETE.md](TRAIN_VALID_TEST_SPLIT_COMPLETE.md) for details.

---

## 📝 Document Status

All documents are current as of **October 25, 2024**.

Last updated:
- README.md - Oct 24
- PIPELINE_GUIDE.md - Oct 25
- SETUP_SUMMARY.md - Oct 25
- DATASET_ANALYSIS_REPORT.md - Oct 25
- DATA_LOSS_EXPLANATION.md - Oct 25
- TRAIN_VALID_TEST_SPLIT_COMPLETE.md - Oct 25

---

## 🔗 Related Files (Not in Docs/)

Python Scripts:
- `pipeline_collect_analyze.py` - Pipeline 1
- `pipeline_train_baseline.py` - Pipeline 2
- `pipeline_generate_synthetic.py` - Pipeline 3
- `pipeline_validate_synthetic.py` - Pipeline 4
- `split_train_valid_test.py` - Train/valid/test splitting
- `main_orchestrator.py` - Interactive menu system

Data Directories:
- `GBIF_MA_BUMBLEBEES/` - Raw GBIF data
- `GBIF_MA_BUMBLEBEES/prepared/` - After bplusplus.prepare() (train/valid)
- `GBIF_MA_BUMBLEBEES/prepared_split/` - After custom split (train/valid/test)
- `RESULTS/` - Model outputs and results

---

**Happy researching! 🐝**

For questions, check the troubleshooting section in [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md).
