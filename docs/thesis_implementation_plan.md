# Thesis Implementation Plan

**Date**: 2026-04-16
**Scope**: Two remaining thesis tasks — (1) fine-grained failure analysis of augmentation results, (2) expert-calibrated quality filter implementation.
**Timeline**: ~2-3 weeks. GPU jobs submitted via SLURM shell scripts in `jobs/`; user submits.
**Review cadence**: Code + writing review after each numbered phase.

---

## 1. Goals

**Task 1 — Fine-grained failure analysis.** Current results (D3/D4/D5 vs baseline) show overlapping CIs; aggregate macro F1 reveals nothing. The thesis contribution is *showing why* through image-level visualizations, not through more aggregate metrics.

**Task 2 — Expert-calibrated filter.** Implement and evaluate three filters: LLM-as-judge rule (existing D5), DINOv2 centroid distance (unsupervised baseline), DINOv2 linear probe (expert-supervised). Baselines first; supervised filter only after baselines have results.

**Shared dependency.** Both tasks require DINOv2 (and optionally BioCLIP) embeddings for all real and synthetic images. Extract once, cache, reuse.

---

## 2. Conventions and Integration Points

### Repository layout (existing)

| Path | Role |
|------|------|
| `pipeline/` | Core modular pipeline (train/, augment/, evaluate/, config.py) |
| `scripts/` | Standalone analysis / assembly scripts (Python, argparse) |
| `jobs/` | SLURM shell scripts; outputs to `jobs/logs/*.out,*.err` |
| `configs/` | `training_config.yaml`, `species_config.json`, `prompt_template.txt` |
| `RESULTS/` | Primary outputs (synthetic generation, LLM judge, embeddings cache, filter scores) |
| `RESULTS_seeds/` | Multi-seed training runs (5 seeds per config: baseline, d3_cnp, d4_synthetic, d5_llm_filtered); per-seed test JSONs + confusion matrices |
| `RESULTS_kfold/` | K-fold CV runs + pooled bootstrap analysis (`kfold_analysis_f1.json`); also contains `expert_validation/` |
| `RESULTS_0312/`, `RESULTS_count_ablation/` | Single-run results + volume ablation |
| `docs/` | Thesis draft, analysis notes, this plan |

### Where new code lives

| New module | Location | Justification |
|---|---|---|
| `pipeline/evaluate/embeddings.py` | Parallels existing `pipeline/evaluate/bioclip.py`; DINOv2 + BioCLIP CLS extraction, L2-normalization, caching | Module, not script — reused by both tasks |
| `pipeline/evaluate/filters.py` | Centroid distance filter + linear probe (LOOCV + nested CV), scoring API | Module — called by scripts |
| `pipeline/evaluate/cknna.py` | CKNNA metric (kernel-based, k-NN, centered) | Module |
| `scripts/extract_embeddings.py` | CLI wrapper around `pipeline/evaluate/embeddings.py` | Script for job scripts to call |
| `scripts/run_filter.py` | Applies a filter (centroid / probe) to score synthetic images, writes selection JSON | Script |
| `scripts/assemble_d6.py` | Builds D6-centroid and D6-probe prepared dataset dirs (like existing `assemble_dataset.py`) | Script |
| `scripts/compute_cknna.py` | Computes CKNNA for each filtered set vs real, including ceiling from random real splits | Script |
| `scripts/analyze_flips.py` | Parses per-image predictions across conditions, produces flip categorization CSV | Script |
| `scripts/plot_failure_analysis.py` | All Task 1 visualizations | Script |
| `scripts/plot_embeddings.py` | UMAP / t-SNE of DINOv2 embeddings (real + synthetic) | Script |
| `jobs/extract_embeddings.sh` | SLURM job for embedding extraction | Job |
| `jobs/train_subset_ablation.sh` | 6 subset ablation runs | Job |
| `jobs/train_d6.sh` | Train D6-centroid and D6-probe (5 seeds each) | Job |

### Follow existing conventions

- Import from `pipeline.config` for paths (`GBIF_DATA_DIR`, `RESULTS_DIR`) — no hardcoded paths.
- Dataset resolution via `pipeline.config.resolve_dataset()` (add `d6_centroid`, `d6_probe` as named variants).
- Training invocation goes through `python run.py train --type simple --dataset d6_centroid` (add dataset names to the registry).
- Multi-seed training results land in `RESULTS_seeds/{dataset}_seed{seed}_*` — same convention as existing D1/D3/D4/D5.
- Single-run / aggregate outputs (embeddings cache, filter scores, CKNNA, failure analysis) land in `RESULTS/`.
- Shell scripts: SLURM headers, `cd` + `source venv/bin/activate`, outputs to `jobs/logs/`.
- Config loading: YAML via `load_training_config()`. Do not invent a new config system.

---

## 3. Phase Order and Dependencies

```
Phase 0: Shared embedding extraction  ────┐
                                          │
Phase 1: No-compute analyses (parallel)   │
  - LLM rule AUC-ROC on 150 labels  ──────┤
  - Per-image flip parsing  ──────────────┤
                                          │
Phase 2: Centroid filter + D6-centroid  ──┤
                                          │
Phase 3: Subset ablation (GPU) + flip ────┤
         analysis visualizations          │
                                          ├── Code review checkpoint
Phase 4: Head/tail tier F1 table          │
         + embedding UMAP visualizations  │
                                          │
Phase 5: Linear probe (LOOCV)             │
         + D6-probe (GPU)                 │
                                          │
Phase 6: BioCLIP supplementary check      │
         + CKNNA  ────────────────────────┤
                                          │
Phase 7: Image-level failure galleries    │
         (requires Phase 3 labels)        │
                                          │
Phase 8: Thesis writeup (§5.5, §5.6,      │
         failure analysis section) ───────┘
```

Review checkpoint after each phase. GPU work is isolated to phases 2, 3, 5, 6.

---

## 4. Phase 0 — Shared: DINOv2 Embedding Extraction

**Goal.** Cache L2-normalized DINOv2 ViT-L/14 CLS embeddings for: (a) all real training+test images (16 species), (b) all 1,500 synthetic images, (c) as an option flag, BioCLIP embeddings for the same images.

**Model choices (from ai-research-scientist review).**
- DINOv2 ViT-L/14 (1024-d CLS), input 518×518 native resolution (not 224).
- CLS token only (not mean-pooled patches).
- L2-normalize before caching.
- BioCLIP as a supplementary check — same API, flag-toggled.

**Files.**
- `pipeline/evaluate/embeddings.py` — core extractor class (DINOv2 + BioCLIP); uses `torch.hub` for DINOv2, existing BioCLIP setup from `pipeline/evaluate/bioclip.py` for BioCLIP.
- `scripts/extract_embeddings.py` — CLI: `--model {dinov2,bioclip} --images-dir ... --output ...`.
- `jobs/extract_embeddings.sh` — SLURM, single GPU, ~1 hour.

**Output paths.**
- `RESULTS/embeddings/dinov2_real.npz` — dict of {image_path: embedding}.
- `RESULTS/embeddings/dinov2_synthetic.npz`.
- `RESULTS/embeddings/bioclip_*.npz` (optional Phase 6).

**Review checkpoint.** Before Phase 1 starts:
- Verify the cached embeddings match expected shapes.
- Spot-check 2-3 images: recompute manually, confirm L2-normalization.
- Sanity: 5-NN species accuracy on real images should be high (>0.7) — tests that the embedding space separates species.

---

## 5. Phase 1 — No-Compute Analyses (Parallel)

### 5.1 LLM rule AUC-ROC baseline (Task 2)

Purpose: Establish the accuracy the learned filters must beat. No new compute.

- Input: `RESULTS_kfold/expert_validation/expert_validation_manifest.json` (expert PASS/FAIL) + LLM `morph_mean` per image.
- Ground truth mapping: decide now — UNCERTAIN → FAIL (conservative) or excluded. Document decision in `docs/@0415.md`.
- Report AUC-ROC using LLM morph_mean as score + binary expert label.
- File: `scripts/compute_llm_filter_auc.py`. No job needed.

### 5.2 Per-image prediction flip analysis (Task 1)

Purpose: For each test image, track prediction across baseline/D3/D4/D5. Categorize stable-correct / stable-wrong / improved / harmed.

- Parse existing per-seed test JSONs: `RESULTS_seeds/{config}_seed{42-46}@f1_seed_test_results_*.json` across baseline / d3_cnp / d4_synthetic / d5_llm_filtered (4 configs × 5 seeds = 20 files).
- K-fold predictions also available in `RESULTS_kfold/` if per-fold flip analysis is wanted.
- File: `scripts/analyze_flips.py` — outputs CSV with columns `[image_path, true_species, pred_baseline, pred_d3, pred_d4, pred_d5, category_d3, category_d4, category_d5]`.
- Output: `RESULTS/failure_analysis/flip_analysis.csv`.
- No job needed.

**Review checkpoint.** Confirm flip CSV is correct by spot-checking 5 known confusion cases from existing confusion matrices.

---

## 6. Phase 2 — Centroid Distance Filter + D6-Centroid

**Specifications (from ai-research-scientist review).**
- Per-species mean centroid on L2-normalized DINOv2 embeddings (real training images only, not test).
- Cosine distance (equivalent to Euclidean on L2-normalized vectors).
- No Mahalanobis (underdetermined at n=22 for B. ashtoni).
- Selection: bottom-40th-percentile per species (not fixed top-k — holds selection rate constant).
- For D6-centroid dataset assembly at +200 per species: take 200 closest per species, regardless of LLM tier. Also run centroid ∩ LLM-strict-pass as a combined variant.

**Files.**
- `pipeline/evaluate/filters.py` — `CentroidFilter` class (fit on real embeddings, score on synthetic).
- `scripts/run_filter.py --filter centroid --output RESULTS/filters/centroid_scores.json`.
- `scripts/assemble_d6.py --variant centroid --per-species 200 --output GBIF_MA_BUMBLEBEES/prepared_d6_centroid`.
- `jobs/train_d6.sh` — trains `d6_centroid` variant with 5 seeds.

**Pipeline integration.**
- Add `d6_centroid`, `d6_probe` dataset names to `pipeline/config.py` resolver.
- Symlink structure matches existing `prepared_d4_synthetic` / `prepared_d5_llm_filtered`.

**Review checkpoint.**
- Code review on filter + assembly before submitting job.
- Verify assembled dataset has correct per-species counts.

---

## 7. Phase 3 — Subset Ablation (GPU) + Flip Analysis

**Purpose.** Causal attribution: which species' synthetics harm classification?

**6 training runs.**
| Run | Training data |
|---|---|
| D4-no-ashtoni | D4 with B. ashtoni synthetics removed |
| D4-no-sandersoni | D4 with B. sandersoni synthetics removed |
| D4-no-flavidus | D4 with B. flavidus synthetics removed |
| D5-no-ashtoni | Same for D5 |
| D5-no-sandersoni | Same for D5 |
| D5-no-flavidus | Same for D5 |

**Implementation.**
- Each run uses existing D4 or D5 prepared dataset with synthetic images for one species filtered out at dataloader level.
- Prefer a new `--exclude-synthetic-species` flag on `pipeline/train/simple.py` over building 6 new prepared dirs.
- Output: `RESULTS_seeds/subset_ablation/{d4,d5}_no_{species}_seed42_*` (or `RESULTS_seeds/` directly with a naming convention like `d4_synthetic_no_sandersoni_seed42_*` for consistency with existing multi-seed naming).
- `jobs/train_subset_ablation.sh` — runs all 6 sequentially (or batched).

**Analysis output.**
- Label each synthetic image as `helpful` / `neutral` / `harmful` based on F1 delta when its species' synthetics are removed.
- File: `scripts/label_synthetic_effect.py` → `RESULTS/failure_analysis/synthetic_labels.csv`.

**Review checkpoint.** Code review on `--exclude-synthetic-species` implementation before GPU job submission.

---

## 8. Phase 4 — Head/Tail F1 Table + Embedding UMAP

### 8.1 Head/tail tier F1 table (Task 1)

- Use existing §3.3 tier partition: rare (n<200), moderate (200–900), common (>900).
- Aggregate per-species F1 into tier-level macro F1 and accuracy for each condition (baseline/D3/D4/D5/D6-centroid).
- Output: Markdown table + heatmap figure (`RESULTS/failure_analysis/tier_f1_heatmap.png`).
- File: `scripts/plot_failure_analysis.py --mode tier_table`.

### 8.2 Embedding visualizations (both tasks)

- UMAP and/or t-SNE of DINOv2 embeddings: real + synthetic images, colored by species, marker-shape by real/synthetic.
- Additional version: color-coded by helpful/neutral/harmful (from Phase 3 labels).
- Additional version: thumbnail overlay at sampled points (Task 1 image-driven).
- File: `scripts/plot_embeddings.py` — UMAP primary, t-SNE secondary.
- Output: `docs/plots/umap_real_vs_synthetic.png`, `docs/plots/umap_helpful_harmful.png`, `docs/plots/umap_thumbnails.png`.

**Review checkpoint.** Writing review — do these plots actually support the thesis narrative? Adjust colors, labels, and layout for committee readability.

---

## 9. Phase 5 — Expert-Calibrated Linear Probe + D6-Probe

**Specifications (from ai-research-scientist review).**
- Pooled probe across 3 species (not per-species — 50 samples each is too few).
- sklearn `LogisticRegression(class_weight='balanced', solver='lbfgs')`.
- L2 regularization: nested LOOCV. Outer: LOOCV over 150 images. Inner: stratified 5-fold over 149 remaining to tune C ∈ {0.001, 0.01, 0.1, 1.0, 10.0}.
- Evaluation: AUC-ROC (primary), precision@90%recall (secondary), balanced accuracy.
- Post-hoc stratification: separate AUC-ROC for {strict_pass, borderline} vs {soft_fail, hard_fail}.

**Files.**
- `pipeline/evaluate/filters.py` — add `LinearProbeFilter` class with fit / score / LOOCV.
- `scripts/run_filter.py --filter probe` (runs LOOCV + final fit on all 150 + scores all 1,500).
- `scripts/assemble_d6.py --variant probe --per-species 200`.
- `jobs/train_d6.sh` — extend to cover `d6_probe`.

**Volume ablation for D6.** Run D6 at +100, +200, +300 per species (not full grid — flat beyond +200 per D4/D5 results).

**Review checkpoint.** Code review on probe (especially nested LOOCV boundary — no C tuning in outer fold) before committing results.

---

## 10. Phase 6 — BioCLIP Supplementary Check + CKNNA

### 10.1 BioCLIP check

- Extract BioCLIP embeddings once.
- Run LOOCV probe on BioCLIP embeddings with same protocol.
- If BioCLIP LOOCV AUC-ROC > DINOv2 by >0.03, switch primary to BioCLIP and note in thesis. Otherwise keep DINOv2 as primary, report BioCLIP as footnote.

### 10.2 CKNNA

- k=5 for rare species (n~22 real), k=10 for B. flavidus (n=162 real).
- Linear kernel on L2-normalized embeddings.
- Ceiling: CKNNA between two random 50% splits of real images, averaged over 100 random splits per species. Report mean ± std.
- Report filter CKNNA as ratio to ceiling.
- Comparison: D4 / D5 / D6-centroid / D6-probe vs real.
- File: `pipeline/evaluate/cknna.py` module + `scripts/compute_cknna.py`.
- Output: `RESULTS/failure_analysis/cknna_results.json`.

**Review checkpoint.** Sanity — CKNNA of real-vs-real (disjoint split) should be clearly > CKNNA of synthetic-vs-real. If not, something is wrong with the implementation.

---

## 11. Phase 7 — Image-Level Failure Galleries (Task 1 Core Deliverable)

**Requires.** Phase 3 synthetic labels + Phase 4 embedding UMAP + flip CSV from Phase 1.

**Visualizations (in priority order).**

### 11.1 Failure chains (top priority)
For each rare species test image that flipped correct→incorrect under D4/D5:
- Center: the test image (labeled true species + wrong prediction).
- Row: top-5 nearest synthetic training neighbors in DINOv2 space, each labeled with generated-species and LLM tier.
- Expected finding: synthetic sandersoni images sitting near real vagans in embedding space.
- File: `scripts/plot_failure_analysis.py --mode failure_chains`.
- Output: `docs/plots/failure_chains_{species}.png` (one per rare species).

### 11.2 Per-species 4-column galleries
For each rare species:
- Column A: real training images (reference, sampled).
- Column B: helpful synthetics (subset-ablation removal hurts F1).
- Column C: harmful synthetics (removal recovers F1).
- Column D: LLM-passed-but-harmful (the thesis-critical case).
- Output: `docs/plots/gallery_{species}.png`.

### 11.3 LLM-pass-but-harmful annotated gallery
- Grid of 12-20 harmful synthetics that LLM strict-passed.
- For each image, annotate (manually or via feature crop) the wrong diagnostic feature: abdomen banding pattern, thorax coloration.
- Output: `docs/plots/llm_pass_but_harmful_annotated.png`.
- This is the single most thesis-valuable image — the "smoking gun" for why automated filtering fails.

### 11.4 Confusion pair side-by-side
- For the dominant rare-species confusion (identified from confusion matrices): real target | harmful synthetic of target | real confused-with.
- Output: `docs/plots/confusion_pair_{species_pair}.png`.

### 11.5 LLM-score × classifier-relevance quadrant scatter
- Each synthetic image plotted on: x-axis LLM morph_mean, y-axis cosine distance to correct-species centroid in DINOv2 space.
- Color by helpful/harmful label.
- Quadrants reveal: high-LLM / close-to-species (top-left, expected pass, mostly helpful); high-LLM / far-from-species (bottom-left, thesis-critical — LLM thinks good but classifier disagrees).
- Output: `docs/plots/llm_vs_classifier_quadrant.png`.

**Review checkpoint.** Writing review of each figure caption and panel layout. Do the visualizations support the thesis argument directly? Are species labels readable? Are the "smoking gun" images truly diagnostic, not cherry-picked?

---

## 12. Phase 8 — Thesis Writeup

**Sections to fill.**
- §5.5 Expert Calibration Results (Tables 5.11, 5.12 — fill LOOCV AUC-ROCs and CKNNA numbers).
- §5.5.4 D6 Classifier Results (add to Table 5.6).
- §5.6 Latent Space Analysis (UMAP figures, centroid distance distributions).
- New section: §5.7 (or §6.1 reworked) Failure Analysis — image-level galleries, flip analysis, tier F1 heatmap.
- §6.4 Discussion update: the "LLM judge passes but classifier disagrees" finding.

**Review.**
- Citation audit (still have 4 missing refs + 2 unused from previous audit).
- Disambiguate Lin et al. (2018) — bilinear CNN vs focal loss.

---

## 13. Implementation Order Summary

1. **Phase 0** (shared): Extract DINOv2 embeddings. [review]
2. **Phase 1** (parallel): LLM rule AUC + flip CSV. [review]
3. **Phase 2** (GPU): Centroid filter → D6-centroid training. [review]
4. **Phase 3** (GPU): 6 subset ablation runs → synthetic labels. [review]
5. **Phase 4** (analysis): Tier F1 table + embedding UMAPs. [review]
6. **Phase 5** (GPU): Linear probe LOOCV + D6-probe training. [review]
7. **Phase 6** (analysis + small GPU): BioCLIP check + CKNNA. [review]
8. **Phase 7** (analysis): Image-level failure galleries. [review]
9. **Phase 8** (writing): Thesis writeup. [review]

Between phases: code review (or writing review for analysis/figure phases) before proceeding.

---

## 14. Resolved Decisions (2026-04-16)

1. **UNCERTAIN expert label mapping.** Deferred — expert validation not yet complete. Make the ground-truth mapping configurable via `--uncertain-policy {fail, exclude}` in `scripts/compute_llm_filter_auc.py` with default `fail` (conservative). When expert validation completes, verify actual UNCERTAIN count and decide based on frequency (if <5% of 150, excluding is fine; otherwise report both policies).
2. **Embedding cache format.** NPZ (dict-of-arrays). ~5,000 images × 1,024 dims × float32 = ~20 MB. Fast numpy load for sklearn probe and umap-learn.
3. **D6 per-species volume.** +200 only (matches D4/D5). No additional volume ablation.
4. **Subset ablation seeds.** 1 seed (42) for initial causal attribution. The sandersoni drop is −0.128 to −0.145 across protocols (large, consistent) — 1 seed should cleanly show recovery or not. Extend to 5 seeds only if signal is ambiguous.
5. **Path conventions for new outputs:**
   - Multi-seed training results (subset ablation, D6-centroid, D6-probe): `RESULTS_seeds/{dataset}_seed{seed}_*`
   - Aggregate outputs (embeddings cache, filter scores, CKNNA, failure analysis, flip CSV): `RESULTS/`
   - Figures for thesis: `docs/plots/`

### Premise verification for subset ablation (Phase 3)

Existing multi-seed results (`docs/experimental_results.md` Table 5.11, 5 seeds on fixed split) confirm the sandersoni drop that motivates subset ablation:

| Config | Macro F1 | Sandersoni F1 (n=10) | Δ vs baseline |
|---|---|---|---|
| D1 Baseline | 0.839 ± 0.006 | 0.622 ± 0.070 | — |
| D3 CNP | 0.822 ± 0.014 | 0.477 ± 0.156 | −0.145 |
| D4 Synthetic | 0.828 ± 0.009 | 0.494 ± 0.059 | −0.128 |
| D5 LLM-filtered | 0.831 ± 0.008 | 0.533 ± 0.063 | −0.089 |

K-fold (Table 5.8): D4 reduces sandersoni F1 by −0.140 vs D1 (p=0.052, n=58). Effect is large enough that 1-seed subset ablation should resolve the causal question.

---

## 15. Success Criteria

- All GPU runs submitted as shell scripts in `jobs/`; no interactive training.
- New code imports from `pipeline.config` — no hardcoded paths.
- Dataset variants (`d6_centroid`, `d6_probe`) invokable via `python run.py train --dataset ...`.
- All figures produced at 300 DPI for thesis inclusion.
- At least one "smoking gun" image-level figure supports each of the two key findings: (a) LLM passes but classifier harms, (b) synthetic sandersoni drifts toward real vagans.
- Code review notes recorded in `docs/` for each phase.
