# Expert Evaluation Strategy: Research-Backed Plan

**Author:** Mia Sun | **Date:** 2026-03-25

---

## 1. Why Expert Evaluation Now

The LLM-as-judge (GPT-4o) plateaus at ~50% pass rate for B. ashtoni. Background removal tests confirmed the bottleneck is **borderline coloration accuracy in synthesis**, not judge error or background confusion. The LLM judge reasons semantically ("does this look like B. ashtoni?") while experts reason morphologically ("is T4 tergite banding too orange?"). This gap is where expert-calibrated filtering can outperform prompted VLMs.

**Key literature support:**
- LLM judges systematically fail on inputs outside their reliable visual knowledge (["No Free Labels," 2025](https://arxiv.org/abs/2503.05061)). GPT-4o cannot reliably detect species-specific banding patterns.
- **LLM-Rubric (ACL 2024)** showed that per-dimension LLM scoring followed by a learned calibration model achieves **2x improvement** in correlation with human ground truth over holistic LLM scoring. The holistic single question is structurally inferior because the LLM's quality signal is more faithfully captured through disaggregated rubric scores.

---

## 2. Core Insight: Rich Structured Signals, Not Just Binary

Our LLM judge already produces **rich structured output** per image (matching the expert evaluation rubric):

| Signal | Type | What it captures |
|---|---|---|
| Blind species ID | Categorical (16 species + "Unknown") | Can the image be identified without being told the target? |
| 5 morphological scores | Ordinal 1–5 each | legs, wings, head, abdomen banding, thorax coloration |
| Diagnostic completeness | Ordinal (none/family/genus/species) | Taxonomic level the image supports |
| Species fidelity failures | Multi-label checkboxes | wrong_coloration, extra_limbs, impossible_geometry |
| Image quality failures | Multi-label checkboxes | blurry, background_bleed, repetitive_pattern |

**If the expert fills out the same rubric**, we get per-feature ground truth — not just pass/fail. This enables:

1. **Per-feature trust mapping** — discover which LLM scores to trust and which to override
2. **Weighted feature-level filtering** — learn that abdomen_banding alone drives 80% of expert rejections for B. ashtoni
3. **Generation-vs-judge diagnosis** — pinpoint whether failures are in synthesis or evaluation
4. **Targeted prompt refinement** — feed specific feature failures back into generation prompts

---

## 3. Annotation Protocol

### 3.1 Expert Annotation Rubric (Mirrors LLM Judge)

Experts fill out the **same structured form** as the LLM judge:

**Stage 1 — Blind Identification:**
- Given only the image (no target species label), identify: Family / Genus / Species (or "Unknown")

**Stage 2 — Detailed Evaluation (after revealing target species):**
- 5 morphological feature scores (1–5 scale, with "not visible" option):
  - Legs/Appendages
  - Wing Venation/Texture
  - Head/Antennae
  - **Abdomen Banding** (critical)
  - **Thorax Coloration** (critical)
- Diagnostic completeness: none / family / genus / species
- Species fidelity failures: checkboxes (no failure, wrong coloration, extra/missing limbs, impossible geometry)
- Image quality failures: checkboxes (no failure, blurry, background bleed, unrealistic flowers, repetitive patterns)
- **Overall: PASS / FAIL / UNCERTAIN**
- Optional: free-text notes on failure reason

**Why mirror the LLM rubric:** Enables direct per-feature comparison (expert score vs LLM score on the same dimension for the same image). This is what makes feature-level calibration possible.

### 3.2 Sample Selection: Stratified by LLM Judge Score

| Stratum | Per species | Selection logic |
|---|---|---|
| High-confidence pass (mean morph score >= 4.0) | 20 | Cluster centroids from LLM-passed images |
| Boundary (mean morph score 3.0–4.0) | 20 | Maximum information gain zone |
| Failed (mean morph score < 3.0) | 10 | Likely true negatives, validates judge |
| **Total** | **50/species x 3 species = 150** | |

**Why this allocation:** Boundary cases have the highest marginal information value. Active learning literature ([NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/8443219a991f068c34d9491ad68ffa94-Paper-Conference.pdf)) shows that at 150/1100 = 13.6% of the pool, stratified sampling by score distribution is near-optimal for a single annotation round.

### 3.3 Annotator Protocol

**Number of annotators:** 2 independent entomologists labeling all 150 images.

**Calibration session (critical):**
1. Both annotators jointly review 15 images (5/species) and discuss disagreements
2. This surfaces **criteria drift** — annotators discover what they actually care about by seeing examples ([EvalGen, UIST 2024](https://arxiv.org/abs/2404.12272))
3. Session produces a **written criteria card** (3–5 bullets per species: T4 banding color, thorax pile pattern, facial hair color)
4. Medical annotation literature shows calibration raises kappa from ~0.55 to ~0.75

**Annotation interface:**
- One image at a time, full resolution
- Species reference images (3 museum-quality photos) shown side-by-side
- Criteria card visible at all times
- Target: 3–4 min/image (richer rubric than binary) → 150 images = 7.5–10 hours per annotator

### 3.4 Quality Gate: Cohen's Kappa

Compute **per feature** and **overall** before proceeding:

| Kappa | Interpretation | Action |
|---|---|---|
| < 0.40 | Poor — task too ambiguous | Revise criteria card, re-calibrate, re-annotate disagreements |
| 0.40–0.60 | Moderate — usable with majority vote | Proceed, note limitation |
| 0.60–0.80 | Substantial — publication quality | Proceed confidently |
| > 0.80 | Near-perfect | Consider single-annotator for future batches |

**Compute kappa separately for:** overall pass/fail, abdomen_banding, thorax_coloration, blind_id_match. If a specific feature has low kappa, that feature's expert labels are unreliable and should be downweighted or excluded from the calibration model.

---

## 4. Learning from Expert Labels: Feature-Level Calibration

### 4.1 Step 1: Per-Feature Disagreement Analysis (Before Any Modeling)

For each of the 5 morphological features, build a 2x2 matrix:

|  | Expert score >= 4 | Expert score < 4 |
|---|---|---|
| **LLM score >= 4** | Agreement: trustworthy | **LLM blind spot** (can't see the error) |
| **LLM score < 4** | **LLM too strict** (false rejection) | Agreement: correctly flagged |

This tells you:
- **Which features the LLM is miscalibrated on** (large off-diagonal counts)
- **Which features you can trust** (large diagonal counts)
- **The direction of bias** (LLM too generous vs. too strict per feature)

Expected finding (based on prior analysis): LLM overscores abdomen_banding and thorax_coloration for B. ashtoni (the "blind spot" cell will be large for these features).

### 4.2 Step 2: Weighted Feature-Level Filter (Primary Experiment)

**Literature validation:** This approach is directly supported by three strong papers:

1. **LLM-Rubric (ACL 2024, Microsoft):** Per-dimension LLM scoring + learned calibration network achieves 2x improvement over holistic scoring. Trained on N~741 annotated examples with a 2-layer feed-forward network. Our N=150 with logistic regression (fewer parameters) is in a comparable data regime.

2. **VisionReward (AAAI 2026):** Decomposes image quality into multiple dimensions, learns weights via logistic regression on binary-valued dimension scores with L2 regularization and weight masking for monotonicity. This is the closest algorithmic precedent — our method is essentially VisionReward Algorithm 2 simplified for a binary outcome.

3. **RichHF-18K (CVPR 2024 Best Paper):** Uses fine-grained multi-dimensional human annotations to filter synthetic image data for downstream model training. Validates that per-feature annotation at scale improves data selection over holistic scoring.

**Method:** L2-regularized logistic regression on the per-feature LLM scores, predicting expert pass/fail:

```python
import numpy as np
from sklearn.linear_model import LogisticRegressionCV

# Features: the 5 LLM morphological scores + blind ID match + diagnostic level
X = np.column_stack([
    llm_legs_score,           # 1-5
    llm_wings_score,          # 1-5
    llm_head_score,           # 1-5
    llm_abdomen_banding,      # 1-5 (critical)
    llm_thorax_coloration,    # 1-5 (critical)
    llm_blind_id_matches,     # 0/1
    llm_diagnostic_level,     # 0/1/2/3 (none/family/genus/species)
])  # shape: (150, 7)

y = expert_pass_fail  # binary, shape: (150,)

# 7 features from N=150 → ~21:1 ratio, reliable with L2 regularization
clf = LogisticRegressionCV(
    cv=5, class_weight='balanced', solver='lbfgs',
    Cs=[0.01, 0.1, 1.0, 10.0]
)
clf.fit(X, y)

# Interpret: which features drive expert rejection?
for name, coef in zip(feature_names, clf.coef_[0]):
    print(f"{name}: {coef:+.3f}")
# Expected: abdomen_banding and thorax_coloration will have largest positive coefficients

# Score all 1,100 images using their LLM judge scores
all_scores = clf.predict_proba(X_all)[:, 1]
```

**Why this works:** The LLM judge scores ARE the features. We're not replacing the LLM — we're learning which of its dimension scores to trust and how much. The logistic regression learns the expert's implicit weighting over the LLM's rubric dimensions.

**Expected outcome (from LLM-Rubric):** 1.5–2x improvement in correlation with expert judgment vs. using the LLM's holistic pass/fail rule directly.

### 4.3 Step 3: Composite Scorer (LLM Scores + Visual Embeddings)

Add BioCLIP/DINOv2 embeddings as additional features to capture what the LLM scores miss entirely:

```python
# Combine LLM rubric scores with visual embedding features
X_composite = np.column_stack([
    X_llm_scores,                    # 7 features from LLM rubric
    bioclip_embedding_PCs[:, :20],   # Top 20 PCs of BioCLIP embeddings
])  # shape: (150, 27)

# 27 features from N=150 → ~5.5:1 ratio — needs strong regularization
clf_composite = LogisticRegressionCV(
    cv=5, class_weight='balanced', solver='lbfgs',
    Cs=[0.001, 0.01, 0.1, 1.0],  # bias toward stronger regularization
    penalty='l2'
)
clf_composite.fit(X_composite, y)
```

**Why add embeddings:** The LLM scores capture what GPT-4o can articulate. BioCLIP embeddings capture fine-grained visual patterns (coloration gradients, texture) that GPT-4o sees but cannot score accurately. This is the Bridge framework approach ([arXiv 2508.12792](https://arxiv.org/html/2508.12792v2)):

```
expert_label ~ β₁·llm_scores + β₂·embedding_features
```

**Compare three models on 5-fold CV:**

| Model | Features | N/p ratio | Expected AUC |
|---|---|---|---|
| A: LLM holistic rule | Current pass/fail logic | — | Baseline |
| B: Weighted LLM scores | 7 LLM rubric features | 21:1 | 0.75–0.85 |
| C: Composite (LLM + embeddings) | 7 + 20 PCs = 27 | 5.5:1 | 0.80–0.90 |

If Model B already achieves high AUC, the embeddings add little — meaning the LLM's per-feature scores contain the signal, they just need reweighting. If Model C >> Model B, the LLM is missing visual information entirely.

### 4.4 Step 4: LLM Judge Prompt Calibration (Cheapest Intervention)

Use the per-feature disagreement analysis (Step 1) to improve GPT-4o's rubric text:

- If experts consistently give low abdomen_banding scores where LLM gives high scores → add explicit visual criteria: "B. ashtoni MUST have pale yellow T2 with sharply demarcated black T3. FAIL if T4 has any orange tinge."
- Use AutoCalibrate ([LREC-COLING 2024](https://arxiv.org/abs/2309.13308)): iteratively refine rubric text using expert labels as ground truth, without any model training.

This costs nothing (just a prompt edit) and may raise LLM judge precision by 10–15% on its own.

### 4.5 Step 5: Targeted Prompt Refinement for Generation

Per-feature expert scores feed back into synthetic generation (RQ5):

| Expert finding | Action on generation prompt |
|---|---|
| Abdomen banding consistently scored low | Add: "T4 tergite must be entirely black with no orange" |
| Thorax coloration OK but legs wrong | No change to coloration prompts; add leg detail |
| Dorsal views score lower than lateral | Generate more lateral views; reduce dorsal |
| Blind ID often returns "Unknown" | Image lacks species-diagnostic gestalt; add reference image conditioning |

This closes the loop: expert evaluation → better filter → better generation → better data.

---

## 5. Downstream Evaluation

### 5.1 Filter → Dataset → Classifier Pipeline

```
Expert labels (150, structured rubric)
    → Step 1: Per-feature disagreement analysis
    → Step 2: Train weighted feature-level filter (logistic regression on LLM scores)
    → Step 3: Train composite scorer (LLM scores + BioCLIP PCs)
    → Step 4: Calibrate LLM judge prompts
    → Score all 1,100 synthetic images with best filter
    → Threshold at optimal operating point (from CV on 150 labels)
    → Assemble D6 dataset (baseline + expert-filtered synthetic)
    → Train ResNet-50 classifier (5 seeds)
    → Evaluate on test set (2,350 images)
```

### 5.2 Statistical Validation

| Method | What it controls for | Implementation |
|---|---|---|
| **Bootstrap CI** (10K resamples) | Small test set variance (n=6 for ashtoni) | `scripts/bootstrap_ci.py` |
| **McNemar's test** | Paired model comparison on full 2,350 test set | 2x2 contingency table, chi-squared |
| **Multi-seed** (5 seeds) | Training randomness | Seeds 42–46, report mean +/- std |

### 5.3 Metrics

- **Filter quality:** AUC-ROC and Average Precision on held-out fold of 150 expert labels
- **Feature importance:** Logistic regression coefficients — which LLM dimensions drive expert pass/fail?
- **Downstream:** Macro F1 across all 16 species
- **Focus species:** Per-species F1 for B. ashtoni and B. sandersoni with bootstrap CIs

---

## 6. What NOT To Do

| Temptation | Why to avoid |
|---|---|
| Collect only binary pass/fail from experts | Wastes the opportunity for per-feature calibration; the structured rubric is the key differentiator |
| Fine-tune ViT backbone on 150 labels | Will overfit; frozen LP is the correct regime |
| Train MLP (>1 hidden layer) | Overfits with N=150; logistic regression is the complexity ceiling (VisionReward validates this) |
| Discard LLM judge entirely | Its per-feature scores are the input features; the expert labels teach you how to weight them |
| Skip calibration session | Risks kappa < 0.50 and unusable labels; criteria drift is real ([EvalGen, UIST 2024](https://arxiv.org/abs/2404.12272)) |
| Use pairwise comparisons | Binary with clear morphological criteria is faster and equally informative for domain-gated decisions |
| Use >20 PCs in composite model | N/p ratio drops below 5:1; overfitting risk. Keep p < 30 total features |

---

## 7. Timeline

| Step | Time | Dependency |
|---|---|---|
| 1. Stratified sample selection (embed + cluster) | 0.5 day | BioCLIP/DINOv2 embeddings |
| 2. Build annotation interface (mirrors LLM rubric) | 1 day | Species reference images |
| 3. Calibration session (15 images, both annotators) | 1 hour | Both experts available |
| 4. Independent annotation (150 images x 2 annotators) | 2 days | After calibration |
| 5. Compute per-feature kappa, resolve disagreements | 0.5 day | Annotations complete |
| 6. Per-feature disagreement analysis (Step 1) | 0.5 day | Expert + LLM scores aligned |
| 7. Train weighted filter + composite scorer (Steps 2–3) | 0.5 day | Analysis complete |
| 8. Calibrate LLM judge prompts (Step 4) | 0.5 day | Disagreement patterns known |
| 9. Assemble D6, train classifier (5 seeds) | 2 days | Filtered dataset |
| 10. Statistical evaluation + write-up | 1 day | All results |
| **Total** | **~9 days** | |

---

## 8. Key References

| Paper | Key finding for our use case | Year |
|---|---|---|
| [LLM-Rubric](https://arxiv.org/abs/2501.00274) | Per-dimension LLM scoring + learned calibration = 2x improvement over holistic scoring (N~741) | ACL 2024 |
| [VisionReward](https://arxiv.org/abs/2412.21059) | Logistic regression on binary dimension scores is the SOTA algorithm for image preference from multi-dimensional rubrics | AAAI 2026 |
| [RichHF-18K](https://arxiv.org/abs/2312.10240) | Fine-grained multi-dimensional human annotations improve synthetic data filtering for downstream training | CVPR 2024 |
| [AutoCalibrate](https://arxiv.org/abs/2309.13308) | Iteratively refine LLM rubric text using human labels without model training | LREC-COLING 2024 |
| [EvalGen](https://arxiv.org/abs/2404.12272) | Criteria drift: annotators discover what they care about by seeing examples; calibration session is essential | UIST 2024 |
| [BioCLIP](https://arxiv.org/abs/2311.18803) | Taxonomy-aware embeddings for biological images; trained on TreeOfLife-10M | CVPR 2024 |
| [No Free Labels](https://arxiv.org/abs/2503.05061) | LLM judges fail systematically on inputs outside their visual knowledge | 2025 |
| [CHARM](https://arxiv.org/abs/2504.10045) | LLM judges have systematic generator biases; human labels expose and correct them | 2025 |
| [Bridge](https://arxiv.org/html/2508.12792v2) | Unified statistical framework: expert_label ~ β₁·llm_scores + β₂·features | 2025 |
| [LP++](https://arxiv.org/abs/2404.02285) | Linear probe on frozen CLIP features rivals prompt-tuning in few-shot regime | CVPR 2024 |
| [Post-hoc Reward Calibration](https://arxiv.org/abs/2409.17407) | Bias correction on reward models without labeled data; 3.11 avg gain on RewardBench | 2024 |
| [DINOv2](https://arxiv.org/abs/2304.07193) | Self-supervised features capture fine-grained texture/color better than CLIP | 2023 |
