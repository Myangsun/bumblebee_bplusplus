# Experiment Plan: Improving Classifier Performance on Long-Tailed Bumblebee Dataset via Generative Image Augmentation and Quality Filtering

**Author:** Mia Sun | **Date:** March 2026 | **Version:** 1.0
**Repository:** `bumblebee_bplusplus` | **Compute:** Server Cluster

---

## 1. Executive Summary

This document defines a structured experiment plan for systematically evaluating parameters that improve fine-grained species classification of Massachusetts bumblebee images, with special focus on long-tailed (rare) species. The core hypothesis is that generative image augmentation, combined with rigorous quality filtering, can significantly improve classifier recall and F1 on underrepresented species without degrading overall accuracy.

The experiments span six dimensions: target species selection, augmentation volume, augmentation method (copy-and-paste vs. generative), filtering method (embedding-based, LLM-as-judge, expert evaluation), classification method (ResNet-50 vs. MLLM structured output), and prompt engineering for iterative improvement. All experiments will be executed on a server cluster, with results reported back to the project folder.

---

## 2. Background and Motivation

### 2.1 Problem Statement

The Massachusetts bumblebee dataset (GBIF) contains 16 species with severe class imbalance. Two critical species, B. ashtoni (33 images) and B. sandersoni (56 images), are likely extirpated and have fewer than 100 training samples. Several additional species including B. flavidus (224), B. affinis (382), and B. citrinus (575) also fall in the long tail. Baseline ResNet-50 achieves 85.19% overall accuracy but completely fails on B. sandersoni (0.0 F1) and achieves only 0.67 F1 on B. ashtoni.

### 2.2 Prior Results Summary

Preliminary experiments (documented in BioGen.pdf) established four training conditions:

| Method               | Train Size | Accuracy | Precision | Recall | Avg F1 |
| -------------------- | ---------- | -------- | --------- | ------ | ------ |
| RAW (baseline)       | 10,889     | 85.19%   | 78.94%    | 79.57% | 75.87% |
| CNP (+1,200)         | 12,089     | 85.06%   | 81.02%    | 80.63% | 80.36% |
| SYNTHETIC (+100)     | 10,989     | 84.34%   | 77.21%    | 75.34% | 75.58% |
| SYNTHETIC_100 (+200) | 11,089     | 83.91%   | 80.98%    | 79.60% | 75.09% |

Key observations: Copy-and-paste augmentation (CNP) improved average F1 by 4.49 percentage points, primarily by raising B. sandersoni from 0.0 to 0.50 F1 and B. ashtoni from 0.67 to 0.80 F1. Synthetic generation (GPT-image-1) at small volumes (+50 or +100 per species) did not improve and sometimes hurt performance, suggesting generated image quality and/or volume may be insufficient.

### 2.3 Research Questions

- RQ1: Which species should receive augmentation, and how many images per species optimize the accuracy-diversity tradeoff?
- RQ2: Does generative augmentation (GPT-image-1) outperform copy-and-paste when quality filtering is applied?
- RQ3: Which filtering method (embedding distance, LLM judge, expert evaluation) best selects high-quality synthetic images for training?
- RQ4: Can MLLM zero-shot structured-output classification serve as a viable alternative or complement to ResNet-50 for rare species?
- RQ5: How do iterative prompt refinements improve generated image quality and downstream classifier performance?

### 2.4 Existing Pipeline Architecture

The pipeline consists of seven stages, each implemented as a standalone module under `bumblebee_bplusplus/pipeline/`:

1. Data Collection (`collect.py`) — download GBIF images for MA species
2. Data Analysis (`analyze.py`) — species distribution and class imbalance report
3. Data Preparation (`prepare.py`) — YOLO detection, crop, initial 80/20 split

   Input directory: /home/msun14/bumblebee_bplusplus/GBIF_MA_BUMBLEBEES
   Output directory: /home/msun14/bumblebee_bplusplus/GBIF_MA_BUMBLEBEES/prepared
   Target image size: 640px (smallest dimension)
   YOLO confidence threshold: 0.35
   Validation split: 10% validation, 90% training
   Gaussian blur: disabled

   I want to analyze species counts and class imbalance after python run.py split. I want to report  
   detailed counts and also visualize the distribution. and based on the concept of long tail, figure  
   out what species to aug. It could build on exisitng python run.py analyze or have a new script.  
   please use the best coding practice. and review the code you changed to make sure it works

4. Data Splitting (`split.py`) — reorganize into 70/15/15 train/valid/test
5. Augmentation
   - 5a: Copy-Paste (`augment/copy_paste.py`) — SAM-based cutout composites
   - 5b: Synthetic (`augment/synthetic.py`) — GPT-image-1 generation
6. Model Training (`train/simple.py`, `train/hierarchical.py`) — ResNet-50 classifier
7. Evaluation (`evaluate/metrics.py`, `evaluate/llm_judge.py`, `evaluate/bioclip.py`)

---

## 3. Dataset Overview

### 3.1 Species Distribution

Total images: 15,564. Split: Train 10,889 (70.0%), Valid 2,325 (14.9%), Test 2,350 (15.1%).

| Species          | Total | Train | Valid | Test | Category                  |
| ---------------- | ----- | ----- | ----- | ---- | ------------------------- |
| B. ashtoni       | 33    | 23    | 4     | 6    | **CRITICAL (Extirpated)** |
| B. sandersoni    | 56    | 39    | 8     | 9    | **CRITICAL (Extirpated)** |
| B. flavidus      | 224   | 156   | 33    | 35   | Rare                      |
| B. affinis       | 382   | 267   | 57    | 58   | Rare (Extirpated)         |
| B. citrinus      | 575   | 402   | 86    | 87   | Uncommon                  |
| B. vagans_Smith  | 632   | 442   | 94    | 96   | Uncommon                  |
| B. borealis      | 679   | 475   | 101   | 103  | Moderate                  |
| B. terricola     | 687   | 480   | 103   | 104  | Moderate (Rare Target)    |
| B. fervidus      | 932   | 652   | 139   | 141  | Moderate (Rare Target)    |
| B. perplexus     | 1,016 | 711   | 152   | 153  | Moderate                  |
| B. rufocinctus   | 1,373 | 961   | 205   | 207  | Common                    |
| B. impatiens     | 1,753 | 1,227 | 262   | 264  | Common                    |
| B. ternarius_Say | 1,782 | 1,247 | 267   | 268  | Common                    |
| B. griseocollis  | 1,821 | 1,274 | 273   | 274  | Common                    |
| B. pensylvanicus | 1,833 | 1,283 | 274   | 276  | Common                    |
| B. bimaculatus   | 1,786 | 1,250 | 267   | 269  | Common                    |

---

## 4. Experiment Design

### 4.1 Experiment 1: Species Selection for Augmentation

**Objective:** Determine which species benefit most from augmentation and define target groups for controlled experiments.

**Species Tiers:**

| Tier             | Species                                              | Train Count | Augmentation Priority     |
| ---------------- | ---------------------------------------------------- | ----------- | ------------------------- |
| Tier 1: Critical | B. ashtoni, B. sandersoni                            | 23, 39      | Highest — primary targets |
| Tier 2: Rare     | B. flavidus, B. affinis                              | 156, 267    | High — secondary targets  |
| Tier 3: Uncommon | B. citrinus, B. vagans_Smith                         | 402, 442    | Medium — optional         |
| Tier 4: Moderate | B. borealis, B. terricola, B. fervidus, B. perplexus | 475–711     | Low — control group       |
| Tier 5: Common   | All others (>1000 images)                            | >1000       | None — baseline           |

**Experimental Conditions:**

- Condition A: Augment only Tier 1 (B. ashtoni + B. sandersoni) — matches prior work
- Condition B: Augment Tiers 1 + 2 (add B. flavidus + B. affinis)
- Condition C: Augment Tiers 1 + 2 + 3 (add B. citrinus + B. vagans_Smith)

**Evaluation Criteria:**

- Per-species F1 change vs. baseline
- Overall accuracy change (must not degrade by more than 1%)
- Macro-average F1 across all 16 species

---

### 4.2 Experiment 2: Augmentation Volume

Your Dataset Recap  
 ┌───────────────────┬───────────────┬───────────────────────────────────┐  
 │ Species │ Current Count │ After 70/15/15 Split (train est.) │
├───────────────────┼───────────────┼───────────────────────────────────┤
│ Bombus_ashtoni │ 36 │ ~25 │
├───────────────────┼───────────────┼───────────────────────────────────┤
│ Bombus_sandersoni │ 80 │ ~56 │
├───────────────────┼───────────────┼───────────────────────────────────┤
│ Bombus_flavidus │ 314 │ ~220 │
└───────────────────┴───────────────┴───────────────────────────────────┘

Max class: 3,000 images (5 species tied at cap)

Common Augmentation Target Strategies from Literature

1. Fixed threshold (DiffuLT, NeurIPS 2024)
   Set a target N_t per dataset, generate N_t - |c_j| images for any class below the threshold. Simple and
   effective. They used dataset-specific thresholds (e.g., 300 for ImageNet-LT, 500 for CIFAR100-LT).

2. Upsample tail to median
   Bring all tail classes up to the median class count. Your median is ~1,267. This is a moderate strategy
   but generates a lot of synthetic data for your smallest classes.

3. Upsample tail to the smallest "head" class
   Use your Pareto boundary (1,506) as the target. Similar logic — bring tails up to the boundary between
   head and tail.

4. Progressive / square-root rebalancing
   Instead of full balancing, use a dampened target like sqrt(N_max \* N_i) (geometric mean of max and
   current count). This partially closes the gap without flooding with synthetic data.

5. Beery et al. (WACV 2020) — Synthetic Examples Improve Generalization for Rare Classes
   Key finding for species classification: synthetic data helps rare classes, but high variation in
   synthetic data matters more than sheer quantity. Diminishing returns kick in — doubling synthetic data
   doesn't double accuracy.

My Recommendation

Given your extreme imbalance (36 vs 3000 = 83x ratio) and that you're using GPT image generation
(expensive per image), I'd suggest a practical hybrid approach:

Target: bring each tail species to ~300–500 training images (after split)

┌───────────────────┬──────────────┬────────────┬─────────────┐
│ Species │ Train (est.) │ Aug Target │ Generate │
├───────────────────┼──────────────┼────────────┼─────────────┤
│ Bombus_ashtoni │ ~25 │ 300 │ ~275 │
├───────────────────┼──────────────┼────────────┼─────────────┤
│ Bombus_sandersoni │ ~56 │ 300 │ ~244 │
├───────────────────┼──────────────┼────────────┼─────────────┤
│ Bombus_flavidus │ ~220 │ 300 │ ~80 │
├───────────────────┼──────────────┼────────────┼─────────────┤
│ Total │ │ │ ~600 images │
└───────────────────┴──────────────┴────────────┴─────────────┘

Rationale:

- 300 is a well-studied threshold — DiffuLT used 300 for ImageNet-LT; it's enough for a CNN to learn
  meaningful features
- Don't try to fully balance to 3,000 — generating ~9,000 synthetic images would be expensive and
  research shows diminishing returns + distribution shift risk from too much synthetic data
- Focus on diversity over quantity — Beery et al. found variation in synthetic data (different poses,
  backgrounds, lighting) matters more than volume. With GPT image gen, prompt diversity is key
- ~600 images is feasible with GPT image API cost-wise
- This brings your smallest class from 83x imbalance to ~10x, which is much more manageable for standard
  techniques (weighted loss, etc.) to handle the remaining gap

If you want to be more aggressive, you could target 500 instead of 300 (~900 total generations), which
aligns with the CIFAR100-LT threshold from DiffuLT.

Key References

- DiffuLT: How to Make Diffusion Model Useful for Long-tail Recognition (NeurIPS 2024) — fixed threshold
  augmentation
- Synthetic Examples Improve Generalization for Rare Classes (Beery et al., WACV 2020) — synthetic data
  for species classification
- Tradeoffs Between Richness and Bias of Augmented Data in Long-Tail Recognition (2025) — EIG framework
  on balancing richness vs distribution shift
- Improving Long-tailed Pest Classification Using Diffusion Model-based Augmentation (2025) —
  domain-specific (insects/pests)
- Awesome Long-Tailed Learning — comprehensive list of methods

───────────────

**Objective:** Determine the optimal number of augmented images per species to maximize rare-species recall without overfitting or distribution shift.

**Experimental Conditions:** For each augmentation method, test the following volumes per target species:

| Volume Label | Images per Species | Rationale                                                |
| ------------ | ------------------ | -------------------------------------------------------- |
| V_50         | 50                 | Minimal augmentation; matches prior SYNTHETIC experiment |
| V_100        | 100                | Matches prior SYNTHETIC_100 experiment                   |
| V_200        | 200                | Moderate augmentation                                    |
| V_500        | 500                | Aggressive; approaching parity with mid-tier species     |
| V_600        | 600                | Matches prior CNP experiment (600 per critical species)  |
| V_1000       | 1,000              | Maximum; near parity with common species                 |

**Evaluation Criteria:**

- Per-species F1 curve as a function of augmentation volume
- Diminishing returns analysis: identify the elbow point
- Monitor for overfitting: valid loss divergence from train loss
- Distribution shift detection: compare augmented class embedding centroids to original

---

### 4.3 Experiment 3: Augmentation Method

**Objective:** Compare copy-and-paste augmentation vs. generative augmentation (with and without filtering) to determine which method produces higher-quality training data for rare species.

**Methods:**

**Method A: Copy-and-Paste (CNP)** — Uses SAM-based segmentation to extract bumblebee cutouts from existing training images, then composites them onto varied natural backgrounds. Advantages: preserves real morphology. Disadvantages: limited diversity, potential background artifacts.

**Method B: Generative (GPT-image-1.5)** — Uses OpenAI's gpt-image-1.5 API (image generation endpoint) with species-specific prompts including morphological descriptions, view angles (dorsal, lateral, frontal, three-quarter), gender variations, and host plant backgrounds. Prompt templates are stored in `species_config.json`.

**Method C: Generative + Image Editing** — Uses GPT-image-1 image editing endpoint with reference images to produce more morphologically accurate results. Reference images are uploaded and used as visual guides.

**Controlled Variables:**

- Same target species (Tier 1: B. ashtoni + B. sandersoni)
- Same volume (V_600, matching best CNP result)
- Same training hyperparameters (ResNet-50, epochs=100, batch=8, lr=0.001, patience=15)
- Same train/valid/test split (70/15/15, `split.py` output)

---

### 4.4 Experiment 4: Filtering Method

**Objective:** Evaluate three filtering approaches for curating generated images before they enter the training set. Filtering is expected to remove low-quality, morphologically inaccurate, or artifact-laden images that degrade classifier performance.

**Filter Methods:**

**Filter F1: Embedding-based / Quality Filtering** — Use BioCLIP or CLIP embeddings to compute cosine distance between each generated image and the real-image class centroid. Reject images whose distance exceeds a threshold (e.g., top-k closest to centroid). This is purely visual-similarity-based and does not require domain expertise.

**Filter F2: LLM-as-Judge** — Use GPT-4o with structured outputs to evaluate each generated image on a 5-dimension rubric: plausibility (realistic photo appearance), morphology (species-specific trait accuracy), environment (natural background), artifact-free (no generation glitches), and pose consistency (matches requested view). Images scoring below a threshold on any dimension are rejected. Implementation exists in `pipeline/evaluate/llm_judge.py`.

**Filter F3: Expert Evaluation** — Domain experts (entomologists) manually review generated images and label them as keep/reject based on morphological accuracy and overall quality. This is the gold standard but is expensive and slow. Use the expert evaluation pipeline in `expert_eval_pipeline/`.

**Experimental Setup:** Generate a large pool of synthetic images (e.g., 2x the target volume) and apply each filter to select the final training set:

| Dataset              | Source           | Filter                | Expected Train Addition         |
| -------------------- | ---------------- | --------------------- | ------------------------------- |
| D1: Baseline         | Real images only | None                  | +0                              |
| D2: CNP              | Copy-and-paste   | None                  | +1,200 (600 per species)        |
| D3: GEN (unfiltered) | GPT-image-1      | None                  | +1,200 (600 per species)        |
| D4: GEN + Embedding  | GPT-image-1      | F1: Embedding/quality | +1,200 (select from 2,400 pool) |
| D5: GEN + LLM Judge  | GPT-image-1      | F2: LLM-as-judge      | +1,200 (select from 2,400 pool) |
| D6: GEN + Expert     | GPT-image-1      | F3: Expert evaluation | +1,200 (select from 2,400 pool) |

**LLM Judge Configuration:** The LLM judge uses the following rubric dimensions (scored 0.0–1.0 each):

- Plausibility: Does the image look like a real camera-trap photograph?
- Morphology: Are species-specific traits (hair color, banding, body shape) correct?
- Environment: Is the background a natural outdoor habitat with flowers?
- Artifact-free: No cloned patterns, distorted geometry, or extra limbs?
- Pose consistency: Does the insect pose match the requested view angle?

Threshold: `overall_pass = True` only if ALL dimensions pass. Score threshold: 0.7 minimum.

**Evaluation Criteria:**

- Filter pass rate: what fraction of generated images survive each filter?
- Filter agreement: pairwise Cohen's kappa between F1, F2, F3
- Downstream classifier performance: per-species F1, macro-F1, overall accuracy
- Cost analysis: API cost (LLM judge) vs. time cost (expert) vs. compute cost (embeddings)

---

### 4.5 Experiment 5: Classification Method

**Objective:** Compare traditional CNN-based classification (ResNet-50) with MLLM zero-shot structured-output classification (GPT-4o) across all six datasets. Evaluate whether LLM-based classification can serve as a complementary approach for rare species where training data is limited.

**Method C1: ResNet-50 (Trained Classifier)**

Architecture: ResNet-50 backbone with hierarchical multi-task heads (family/genus/species). Training: ImageNet-pretrained, fine-tuned with cross-entropy loss, Adam optimizer (lr=0.001), early stopping (patience=15), image size 640x640.

Model selection strategy — two approaches for picking the best checkpoint:

- C1a: Best overall validation loss — standard approach, optimizes for all species equally
- C1b: Best validation loss on augmented species only — prioritizes rare species, may sacrifice common-species accuracy

**Method C2: MLLM Structured Output (Zero-Shot)**

Use GPT-4o with structured outputs (Pydantic schema) for species classification. Each image is encoded as base64, sent with a system prompt describing all 16 species and their morphological traits. The model returns: predicted species (constrained enum), confidence (0.0–1.0), and morphological reasoning.

The structured output classification currently covers only B. ashtoni and B. sandersoni (2-class). For the full experiment, extend to all 16 species. Implementation: `scripts/structured_output_eval.py`.

**Experimental Matrix:**

| Dataset              | C1a: ResNet (overall loss) | C1b: ResNet (augmented loss) | C2: MLLM Structured Output |
| -------------------- | -------------------------- | ---------------------------- | -------------------------- |
| D1: Baseline         | Run                        | N/A                          | Run                        |
| D2: CNP              | Run                        | Run                          | N/A                        |
| D3: GEN (unfiltered) | Run                        | Run                          | N/A                        |
| D4: GEN + Embedding  | Run                        | Run                          | N/A                        |
| D5: GEN + LLM Judge  | Run                        | Run                          | N/A                        |
| D6: GEN + Expert     | Run                        | Run                          | N/A                        |

Note: MLLM structured output (C2) is zero-shot and does not use training data, so it is only run on D1 (baseline test set). The comparison assesses whether C2 can match or exceed C1 on rare species without any training.

**Evaluation Metrics:**

- Per-species: Accuracy, Precision, Recall, F1, Support
- Aggregate: Overall accuracy, Macro-F1, Weighted-F1
- Rare-species focus: Mean F1 on Tier 1+2 species only
- Statistical significance: McNemar's test or paired bootstrap between conditions
- Confusion matrices: full 16x16 heatmaps per condition

---

### 4.6 Experiment 6: Prompt Engineering and Iterative Improvement

**Objective:** Document all prompts used for image generation and LLM-as-judge evaluation, track iterative refinements, and measure the impact of prompt changes on generated image quality and downstream classifier performance.

**Prompt Categories:**

**P1: Image Generation Prompts** — Prompts sent to GPT-image-1 for creating synthetic bumblebee images. Include species morphological descriptions, view angles, background contexts, and negative constraints. Stored in `species_config.json` and `prompt.json`.

**P2: Image Editing Prompts** — Prompts for the GPT-image-1 editing endpoint that modify reference images. Include instructions for pose changes, background replacement, and species-specific adjustments.

**P3: LLM Judge Prompts** — System and user prompts for the GPT-4o judge. Define the rubric, scoring criteria, and structured output schema. Currently in `pipeline/evaluate/llm_judge.py`.

**P4: Structured Output Classification Prompts** — System prompt for GPT-4o classification. Defines species labels, morphological trait descriptions, and confidence calibration instructions. Currently in `scripts/structured_output_eval.py`.

**Prompt Versioning:** Each prompt version will be logged with:

- Version number (P1v1, P1v2, etc.)
- Full prompt text
- Rationale for changes from previous version
- Quality metrics on generated images (LLM judge pass rate, embedding distance)
- Downstream classifier performance delta

**Prompt Log:**

| Version | Date       | Change Description                                  | Judge Pass Rate | Classifier F1 Delta | Notes                       |
| ------- | ---------- | --------------------------------------------------- | --------------- | ------------------- | --------------------------- |
| P1v1    | 2026-02-17 | Initial generation prompt from species_config.json  | TBD             | Baseline            | `batch_generate.py`         |
| P1v2    | TBD        | Add negative constraints (no extra legs, no studio) | TBD             | TBD                 |                             |
| P1v3    | TBD        | Add reference image guidance with edit endpoint     | TBD             | TBD                 |                             |
| P3v1    | 2026-02-24 | Initial LLM judge rubric (5 dimensions)             | TBD             | N/A                 | `llm_judge_eval.py`         |
| P4v1    | 2026-02-24 | Initial structured output classifier (2 species)    | N/A             | Baseline            | `structured_output_eval.py` |
| P4v2    | TBD        | Extend to all 16 species                            | N/A             | TBD                 |                             |

---

## 5. Immediate Next Experiments (Priority)

The following two experiments are the immediate priority, designed as controlled comparisons with minimal variable changes.

### 5.1 Priority Experiment A: LLM-as-Judge Filter vs. Baseline

**Hypothesis:** Filtering synthetic images with an LLM judge (GPT-4o rubric) before training will produce a higher-quality augmented dataset that improves classifier F1 on rare species compared to both the unfiltered synthetic dataset and the baseline.

**Controlled Variables (Held Constant):**

- Target species: B. ashtoni + B. sandersoni (Tier 1 only)
- Augmentation method: GPT-image-1 generation
- Volume: 600 images per species (1,200 total added)
- Classifier: ResNet-50 hierarchical, same hyperparameters
- Model selection: Best overall validation loss
- Train/Valid/Test split: Same as baseline (70/15/15)

**Independent Variable:** Filtering method: None (D3) vs. LLM-as-Judge with threshold=0.7 (D5).

**Procedure:**

1. Generate synthetic images (600 per species) to create the candidate pool
2. Run LLM judge (`pipeline/evaluate/llm_judge.py`) on all 1200 images
3. Select top 600 images (300 per species) that pass the judge threshold
4. Create dataset D3 (unfiltered random 600) and D5 (judge-filtered 600)
5. Train ResNet-50 on D1 (baseline), D3, and D5 using identical hyperparameters
6. Evaluate all three models on the same held-out test set
7. Compare per-species F1, macro-F1, and confusion matrices

**Expected Output:**

- `judge_results.json`: per-image rubric scores and pass/fail
- Three trained model checkpoints (baseline, unfiltered, filtered)
- Comparison table: per-species F1 across D1, D3, D5
- Statistical significance test (McNemar's or bootstrap)

---

### 5.2 Priority Experiment B: Structured Output Classification vs. ResNet-50

**Hypothesis:** GPT-4o structured-output classification can achieve comparable or better F1 on rare species (B. ashtoni, B. sandersoni) than ResNet-50, despite being zero-shot with no training data.

**Controlled Variables:**

- Test set: Same 2,350 images used for ResNet-50 evaluation
- Species labels: All 16 Massachusetts bumblebee species
- Evaluation metrics: Per-species accuracy, precision, recall, F1

**Independent Variable:** Classification method: C1a (ResNet-50, overall loss) vs. C2 (GPT-4o structured output, 16-class).

**Procedure:**

1. Extend `structured_output_eval.py` to all 16 species (currently 2-class only)
2. Run GPT-4o classification on the full test set (2,350 images)
3. Collect predictions with confidence scores and morphological reasoning
4. Compute per-species metrics and confusion matrix
5. Compare with ResNet-50 baseline metrics side-by-side
6. Analyze where MLLM excels vs. fails (especially on rare species)

**Expected Output:**

- `classification_results_16class.json`: per-image predictions with reasoning
- Confusion matrix: 16x16 heatmap for GPT-4o classifier
- Comparison table: ResNet-50 vs. GPT-4o per-species F1
- Cost analysis: API cost per image for structured output classification
- Error analysis: common misclassification patterns for each method

---

## 6. Execution Plan and Server Configuration

### 6.1 Compute Resources

All training experiments will be executed on the server cluster. Each ResNet-50 training run takes approximately 2–4 hours on a single GPU (depending on dataset size and early stopping). MLLM experiments run via OpenAI API and are limited by rate limits and cost.

### 6.2 Execution Order

| Phase    | Experiment                                 | Dependencies             | Est. Duration | Status       |
| -------- | ------------------------------------------ | ------------------------ | ------------- | ------------ |
| Phase 1  | Generate 2,400 synthetic images            | API key + species config | 4–6 hours     | Pending      |
| Phase 2  | Run LLM judge on 2,400 images              | Phase 1 complete         | 2–3 hours     | Pending      |
| Phase 3a | Train ResNet-50 on D1 (baseline)           | Existing split           | 2–4 hours     | Done (prior) |
| Phase 3b | Train ResNet-50 on D3 (unfiltered GEN)     | Phase 1                  | 2–4 hours     | Pending      |
| Phase 3c | Train ResNet-50 on D5 (judge-filtered GEN) | Phase 2                  | 2–4 hours     | Pending      |
| Phase 4  | Extend structured output to 16-class       | Code update only         | 1–2 hours     | Pending      |
| Phase 5  | Run GPT-4o classification on test set      | Phase 4                  | 3–5 hours     | Pending      |
| Phase 6  | Compile results and statistical tests      | Phases 3–5               | 1 day         | Pending      |

### 6.3 File Organization

Results will be stored in the following structure:

- `bumblebee_bplusplus/RESULTS/<experiment_name>/` — model checkpoints and training logs
- `bumblebee_bplusplus/pipeline/evaluate/judge_results.json` — LLM judge output
- `scripts/classification_results.json` — structured output classification results
- Each experiment run named with convention: `<dataset>_<classifier>_<date>`

### 6.4 Training Configuration (Held Constant)

| Parameter               | Value     | Notes                    |
| ----------------------- | --------- | ------------------------ |
| Backbone                | ResNet-50 | ImageNet pretrained      |
| Epochs                  | 100       | With early stopping      |
| Batch size              | 8         | Limited by GPU memory    |
| Learning rate           | 0.001     | Adam optimizer           |
| Early stopping patience | 15        | Based on validation loss |
| Image size              | 640 x 640 | Cropped by YOLO          |
| Dropout                 | 0.5       | Classification head      |
| Weight decay            | 0.0       | No L2 regularization     |

---

## 7. Results Recording Template

Use the following template for recording results from each experiment run. Copy this section for each new experiment.

### 7.1 Experiment Run Record

| Field               | Value                             |
| ------------------- | --------------------------------- |
| Experiment ID       | [EXP-XXX]                         |
| Date                | [YYYY-MM-DD]                      |
| Dataset             | [D1/D2/D3/D4/D5/D6]               |
| Classifier          | [C1a/C1b/C2]                      |
| Augmented Species   | [List species]                    |
| Augmentation Volume | [N per species]                   |
| Filter Method       | [None/Embedding/LLM Judge/Expert] |
| Model Selection     | [Overall loss / Augmented loss]   |
| Training Duration   | [hours]                           |
| GPU / Compute       | [server details]                  |
| Best Epoch          | [N]                               |
| Final Train Loss    | [value]                           |
| Final Valid Loss    | [value]                           |

### 7.2 Per-Species Results Template

| Species              | Accuracy | Precision | Recall | F1  | Support | Delta F1 vs. Baseline |
| -------------------- | -------- | --------- | ------ | --- | ------- | --------------------- |
| B. ashtoni           |          |           |        |     | 6       |                       |
| B. sandersoni        |          |           |        |     | 9       |                       |
| B. flavidus          |          |           |        |     | 35      |                       |
| B. affinis           |          |           |        |     | 58      |                       |
| B. citrinus          |          |           |        |     | 87      |                       |
| B. vagans_Smith      |          |           |        |     | 96      |                       |
| B. borealis          |          |           |        |     | 103     |                       |
| B. terricola         |          |           |        |     | 104     |                       |
| B. fervidus          |          |           |        |     | 141     |                       |
| B. perplexus         |          |           |        |     | 153     |                       |
| B. rufocinctus       |          |           |        |     | 207     |                       |
| B. impatiens         |          |           |        |     | 264     |                       |
| B. ternarius_Say     |          |           |        |     | 268     |                       |
| B. griseocollis      |          |           |        |     | 274     |                       |
| B. pensylvanicus     |          |           |        |     | 276     |                       |
| B. bimaculatus       |          |           |        |     | 269     |                       |
| **MACRO AVERAGE**    |          |           |        |     |         |                       |
| **OVERALL ACCURACY** |          |           |        |     | 2,350   |                       |

---

## 8. Additional Considerations

### 8.1 Statistical Rigor

- Run each training condition with at least 3 random seeds to report mean and standard deviation of F1
- Use McNemar's test for pairwise significance testing between classifiers on the same test set
- Report 95% confidence intervals on per-species metrics using bootstrap resampling

### 8.2 Cost Tracking

Track API costs for each experiment:

- GPT-image-1 generation: ~$0.04–0.08 per image (1024x1024)
- GPT-4o LLM judge: ~$0.01–0.03 per image evaluation
- GPT-4o structured output classification: ~$0.01–0.03 per image
- Expert evaluation: estimated hours per batch

### 8.3 Potential Confounds

- **Test set contamination:** Ensure no generated image leaks into the test set. Verified by `split.py` (test set is carved from real data only).
- **Prompt overfitting:** Different prompt versions may overfit to judge criteria rather than genuine quality. Mitigate by using expert evaluation (F3) as ground truth.
- **Model selection bias:** Choosing checkpoints by augmented-species loss (C1b) may overfit to augmented distribution. Always compare with overall-loss selection (C1a).
- **Class weighting:** Currently no class weighting or focal loss is used. Consider adding weighted cross-entropy as a separate ablation if initial results are inconclusive.

### 8.4 Future Extensions

- BioCLIP embedding analysis: t-SNE/PCA visualization of real vs. generated image distributions per species
- Ensemble methods: Combine ResNet-50 predictions with MLLM structured output for hybrid classifier
- Active learning: Use MLLM confidence scores to select the most informative unlabeled images for expert annotation
- Other generative models: Compare GPT-image-1 with Stable Diffusion, DALL-E 3, or domain-specific models
- Hierarchical classification: Evaluate the hierarchical (family/genus/species) multi-task model separately from the simple single-head model

---

## update

### 0303

- generateion: SIZE!!!
- llm as judge filter!!! need to understand which filtered out first, then training use filtered data
- fix copy and paste
- I have baseline/copy and paste/unfiltered trained.
- change hieracachy to simple

Key dimensions to filter on:

┌─────────────────────────────────────┬──────────────────────────────────────────────────┐
│ Criterion │ What it checks │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ diagnostic_completeness = species │ Judge could ID to species level (not just genus) │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ blind_identification.matches_target │ Judge correctly ID'd the target species │
├─────────────────────────────────────┼──────────────────────────────────────────────────┤
│ Mean morphological score │ Average of 5 feature scores (1-5 scale) │
└─────────────────────────────────────┴──────────────────────────────────────────────────┘

Stricter filter options:

┌─────────────────────────────────────┬───────┬─────────┬────────────┐
│ Filter │ Total │ Ashtoni │ Sandersoni │
├─────────────────────────────────────┼───────┼─────────┼────────────┤
│ Current (overall_pass) │ 831 │ 384 │ 447 │
├─────────────────────────────────────┼───────┼─────────┼────────────┤
│ + diagnostic=species + score >= 3.5 │ 671 │ 239 │ 432 │
├─────────────────────────────────────┼───────┼─────────┼────────────┤
│ + diagnostic=species + score >= 4.0 │ 652 │ 232 │ 420 │
├─────────────────────────────────────┼───────┼─────────┼────────────┤
│ + diagnostic=species + score >= 4.5 │ 104 │ 16 │ 88 │
└─────────────────────────────────────┴───────┴─────────┴────────────┘

The sweet spot looks like diagnostic=species + score >= 4.0: drops to 652
