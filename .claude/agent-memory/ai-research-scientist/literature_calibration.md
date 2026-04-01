---
name: literature on multi-dimensional automated evaluator calibration
description: Key papers on feature-level calibration of automated judges against human annotations, directly relevant to expert-label-guided filter design
type: reference
---

# Key Papers: Multi-Dimensional Evaluator Calibration

## Core Reference: LLM-Rubric (ACL 2024 / arXiv 2501.00274)
- **Authors:** Hashemi, Eisner, Rosset, Van Durme, Kedzie (Microsoft Research)
- **Method:** Multi-dimensional rubric questions scored by LLM → small feed-forward calibration net (shared + judge-specific params) → predicts each human judge's rating on all dimensions including summary question
- **N for calibration:** 741 synthetic dialogues with 3 judges each (~2000 annotation points)
- **Result:** Per-dimension rubric approach achieves 0.35+ Pearson with human satisfaction vs. 0.18 for direct holistic LLM question — **2× improvement**
- **Key finding:** "The LLM cannot help assess overall satisfaction until asked about finer-grained dimensions"
- **Output:** Continuous regression (real-valued), not ordinal classification
- **Code:** github.com/microsoft/LLM-Rubric

## Core Reference: VisionReward (AAAI 2026 / arXiv 2412.21059)
- **Method:** Decompose human preferences into dimensions, each represented as checklist (yes/no) questions. Linear weights learned via logistic regression on preference pairs: score = sum(x_i * w_i) where x_i ∈ {1,-1}.
- **Training objective:** Minimize binary cross-entropy on preference pairs using feature differences (Δx = x_i - x_j)
- **Key finding (Table 13):** Per-dimension reward objective (4.573) > total weighted score (4.515) > per-sub-dimension (4.514)
- **Annotation scale:** 48k images, 3M checklist questions — large scale, not small-N
- **Ordinal → binary:** Ordinal Likert → converted to yes/no binary checklist questions for each sub-dimension
- **Relevance:** Confirms linear weighting of dimension scores is a principled approach; binarizing ordinal Likert is a reasonable preprocessing step

## Core Reference: AutoCalibrate / Calibrating LLM-Based Evaluator (LREC-COLING 2024 / arXiv 2309.13308)
- **Method:** Gradient-free, prompt-based calibration. Uses small human annotation set as implicit preference signal to generate/refine scoring criteria via LLM self-refinement. No gradient access needed.
- **Result:** Significant correlation improvement on text summarization, data-to-text, hallucination datasets
- **Relevance:** When you have ~150 labels but no ability to fine-tune, this approach (refining rubric criteria via LLM) is complementary to weight learning

## Supporting Reference: MPS (CVPR 2024 / arXiv 2405.14705)
- **Method:** Multi-dimensional Preference Score for T2I. Separate dimension scores via CLIP + cross-attention with condition mask. Equal treatment across dimensions (no learned combining weights across dims).
- **N:** 918K human preference choices across 4 dims on 607K images — very large scale
- **Key finding:** Joint training slightly outperforms separate dimension models in ablation
- **Relevance:** Shows per-dimension scoring at scale is standard practice; does NOT show that learned weighting outperforms equal weighting at this scale

## Supporting Reference: Rich Human Feedback for T2I (CVPR 2024 Best Paper / arXiv 2312.10240)
- **Method:** Multi-modal transformer predicts per-attribute quality (plausibility, alignment, aesthetics, overall) plus spatial heatmaps of problem regions
- **Relevance:** Shows fine-grained annotation on multiple quality dimensions used to train quality predictors for filtering/improving generation

## Supporting Reference: Post-hoc Reward Calibration (arXiv 2409.17407)
- **Method:** Estimate bias term in reward model (e.g., length preference) and remove it using Locally Weighted Regression. Computationally efficient, generalizable to other bias types.
- **Result:** 3.11 avg performance gain across 33 reward models on RewardBench
- **Relevance:** If LLM judge has systematic bias (e.g., passes images that look "complete" regardless of coloration accuracy), this method could correct it without re-training

## Methodological Support for Learned Weighting
- VisionReward uses logistic regression with L1/L2 regularization to learn dimension weights from preference pairs
- LLM-Rubric uses a 2-layer feed-forward network trained on ~741 labeled examples
- Both show that learned combination outperforms uncalibrated LLM output
- For N=150 with binary outcome, logistic regression on 5 features is well within the regime where regularized LR is reliable (30:1 samples:features ratio)

## Relevance to Bumblebee Filter (N=150, 5 features, binary pass/fail)
- **Well-supported approach:** Yes. VisionReward and LLM-Rubric directly validate the paradigm.
- **Recommended method:** L2-regularized logistic regression on [legs, wings, head, abdomen_banding, thorax_coloration] scores, with expert binary label as target. Cross-validate C hyperparameter.
- **Ordinal handling:** Either keep as ordinal integers (1-5) or binarize per-dimension at threshold (e.g., <3 = fail). VisionReward binarization approach is a clean precedent.
- **Per-dimension vs. holistic:** Per-dimension is well-supported; LLM-Rubric shows 2× improvement over single holistic score.
