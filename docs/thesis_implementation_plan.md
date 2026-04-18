# Thesis Implementation Plan

**Last updated:** 2026-04-17
**Scope:** Two remaining thesis tracks, reorganised as independent pipelines.

- **Task 1 — Fine-grained failure analysis.** Goal: explain why augmentation (D3/D4/D5) does not improve classification; produce image-level visualisations + per-species and head/tail tier tables. **Needs no expert labels.**
- **Task 2 — Expert-calibrated quality filtering.** Goal: compare three synthetic filters (LLM rule / BioCLIP centroid distance / expert-supervised BioCLIP linear probe) by downstream classification impact. **Partially blocked on expert validation.**

---

## 1. Current State

### Done

- Shared DINOv2 + BioCLIP extraction, k-NN diagnostic, and t-SNE/UMAP/PCA figures. See §6.1.
- Backbone selected: **BioCLIP** (overall 5-NN accuracy 0.657 vs DINOv2 0.295). DINOv2 retained only as appendix comparison.
- Per-image prediction-flip analysis across baseline/D3/D4/D5 (5 seeds × 4 configs). See §6.2.

### Blocked

- Any filter component that requires expert pass/fail labels (Task 2 §5.3–§5.5).

### Next up

- Task 1 §4: failure analysis (image-level + tier tables) — all unblocked.
- Task 2 §5.1–§5.2: BioCLIP centroid filter + D6-centroid training — unblocked.

---

## 2. Shared Conventions (applies to both tasks)

### 2.1 Repository layout

| Path | Role |
|---|---|
| `pipeline/evaluate/` | Reusable modules (`embeddings.py`, upcoming `filters.py`, `cknna.py`). |
| `scripts/` | CLI entry points with argparse (`analyze_flips.py`, `plot_embeddings.py`, `diagnose_embeddings.py`, upcoming `run_filter.py`, `assemble_d6.py`, `plot_failure_analysis.py`). |
| `jobs/` | SLURM shell scripts. Logs under `jobs/logs/`. |
| `RESULTS/` | Aggregate outputs (embeddings cache, filter scores, failure analysis CSVs). |
| `RESULTS_seeds/` | Per-seed training runs (existing baseline/D3/D4/D5 + new subset ablation / D6 variants). |
| `RESULTS_kfold/` | K-fold results + `expert_validation/` annotation pool. |
| `docs/plots/` | Thesis figures. |

### 2.2 Code conventions

- Import paths via `pipeline.config` (`GBIF_DATA_DIR`, `RESULTS_DIR`, `resolve_dataset`); no hardcoded roots.
- Dataset resolver entries for new variants (`d6_centroid`, `d6_probe`) added to `pipeline/config.resolve_dataset`.
- Training invoked via `python run.py train --type simple --dataset <name>` — same interface as D4/D5.
- New modules follow `pipeline/evaluate/bioclip.py` as a template (module + CLI main in one file).

### 2.3 Review cadence

- **Code review** after each new module or script, before any job submission.
- **Writing review** after each analysis / figure phase — verify thesis narrative is supported.

---

## 3. Resolved Decisions (from prior discussion)

| Decision | Choice | Rationale |
|---|---|---|
| Primary backbone | **BioCLIP** 512-d CLS, 224×224 | 2.2× higher 5-NN accuracy than DINOv2 |
| Embedding cache format | NPZ compressed | ~20 MB total; simple numpy load |
| D6 per-species volume | +200 | Match D4/D5; no additional volume ablation |
| Subset ablation seeds | 1 (seed 42) to start | Sandersoni drop is large; 1 seed should resolve causation |
| UNCERTAIN expert label | Defer — expert validation not yet run | Configurable policy when labels arrive |

---

## 4. Task 1 — Fine-Grained Failure Analysis

### 4.1 Goal

Produce a clear, visually-driven account of **why** D3/D4/D5 fail to improve classification. Primary deliverables are image-level figures backed by compact quantitative tables. The narrative must not rest on aggregate F1 bars.

### 4.2 Data foundation (already in place)

- `RESULTS/embeddings/bioclip_real_{train,valid,test}.npz` — real image features.
- `RESULTS/embeddings/bioclip_synthetic.npz` — 1,500 synthetic features.
- `RESULTS/failure_analysis/flip_analysis.csv` — per-image × per-config prediction history.
- `RESULTS_seeds/{config}_seed{42-46}@f1_seed_test_results_*.json` — per-seed per-image predictions.
- `RESULTS_kfold/kfold_analysis_f1.json`, `kfold_bootstrap_ci_f1.json` — pooled k-fold stats.
- `RESULTS_kfold/llm_judge_eval/results.json` — per-synthetic LLM scores.

### 4.3 Phase 1a — Quantitative/tabular analyses (no compute)

| ID | Deliverable | File | Output |
|---|---|---|---|
| T1.1 | Head/tail tier F1 table (rare / moderate / common × D1/D3/D4/D5) with mean ± std across seeds | new `scripts/build_tier_f1_table.py` | `RESULTS/failure_analysis/tier_f1.csv` + markdown table |
| T1.2 | Per-species F1 delta heatmap (16 species × 4 configs, signed colour scale) | new `scripts/plot_failure_analysis.py --mode species_delta` | `docs/plots/failure/species_f1_delta.png` |
| T1.3 | Flip category × species heatmap (stable-correct/stable-wrong/improved/harmed counts per species per aug config) | new `scripts/plot_failure_analysis.py --mode flip_heatmap` | `docs/plots/failure/flip_category_heatmap.png` |
| T1.4 | Per-species correct-rate trajectory (D1 → D3 → D4 → D5) focused on rare species + top-10 most-affected common species | new `scripts/plot_failure_analysis.py --mode trajectory` | `docs/plots/failure/correct_rate_trajectory.png` |

**Review checkpoint after Phase 1a** — confirm the tables/heatmaps reproduce the k-fold narrative (p=0.041 D5 vs baseline, flavidus gains under D3, sandersoni drops under D4) before investing in image-level figures.

### 4.4 Phase 1b — Image-level visualisations (no compute, needs BioCLIP cache)

Each figure is driven by the existing flip CSV + BioCLIP embeddings + on-disk image files. No subset ablation required for this phase — synthetic images are labelled by their *generated target species*; the helpful/harmful distinction is added only in Phase 1c.

| ID | Deliverable | Purpose |
|---|---|---|
| T1.5 | **Failure chains (priority).** For every rare-species test image flagged `harmed` under D4 or D5: show the test image; its 5 nearest synthetic training neighbours in BioCLIP space; label each neighbour with (generated species, LLM morph mean, LLM tier). | Shows whether harmful flips correspond to synthetic neighbours that are close in embedding space — the core mechanism for why augmentation hurts. |
| T1.5b | **Failure chains on t-SNE.** Same failure chain rendered *directly on the BioCLIP t-SNE*: the harmed test image and its 5 synthetic neighbours drawn as thumbnails at their true embedding coordinates, with arrows from the test image to each neighbour. | Lets the reader see the failure mechanism *and* the embedding neighbourhood on one figure. |
| T1.6 | **Per-species 3-column galleries.** For each rare species: column A sampled real training images; column B sampled synthetic images; column C the harmed test images from D4/D5. | Direct visual comparison of real vs synthetic morphology + the test cases augmentation could not rescue. |
| T1.7 | **Confusion-pair triplets.** For each rare species, pick the dominant confusion target from the baseline confusion matrix (ashtoni→citrinus/vagans, sandersoni→vagans, flavidus→citrinus). Show real target | synthetic target | real confuser, side-by-side, labelled with LLM scores. | Visualises whether synthetic images drift toward the real confuser species. |
| T1.8 | **LLM-score × BioCLIP-centroid-distance quadrant scatter.** Every synthetic image plotted as (x = LLM morph mean, y = cosine distance to correct-species BioCLIP centroid). One panel per rare species. | Four quadrants let the reader see "LLM says good but classifier-space says wrong" and "LLM says bad but classifier-space says right" populations. |
| T1.8b | **Rare-species embedding atlas.** BioCLIP t-SNE/UMAP restricted to rare species + their primary confusers, with ~80–120 thumbnails sampled by 2-D grid binning so they do not overlap. Real and synthetic distinguished by thumbnail border colour/thickness. | Combines cluster structure with actual image morphology — reveals whether synthetic clusters are pose-driven or species-driven. |
| T1.8c | **All-species embedding atlas.** 16-species overview t-SNE with ~200 thumbnails sampled uniformly across the 2-D plane. | Lets the reader see the kind of images that each species cluster contains; companion to T1.2. |
| T1.9 | **Embedding plots with helpful/harmful overlay.** BioCLIP t-SNE/UMAP recoloured by Phase 1c labels. | Only produced after Phase 1c. |

**Implementation notes (image-level figures).**
- All thumbnail/gallery figures share infrastructure: `scripts/plot_failure_analysis.py` (grids, callouts, captions) and a new `scripts/plot_embedding_atlas.py` (t-SNE/UMAP with thumbnail overlay using `matplotlib.offsetbox.OffsetImage` + grid-based sampling to avoid overlap).
- Thumbnail borders encode the source: real = thin grey, synthetic = thicker orange, harmed test image = black dashed.
- Figures write to `docs/plots/failure/`.

All image-level outputs go to `docs/plots/failure/`.

Common infrastructure in `scripts/plot_failure_analysis.py`:
- Load `flip_analysis.csv` + `bioclip_real_*.npz` + `bioclip_synthetic.npz` + LLM judge results JSON.
- Helper to open an image by path and render on a matplotlib axes with a caption strip.
- Helper for grid layouts (n_rows × n_cols thumbnails).

**Review checkpoint after Phase 1b** — writing review: do the images support the narrative? Are captions informative without being redundant?

### 4.5 Phase 1c — Subset ablation (GPU, optional for full gallery)

Adds causal attribution: which specific synthetic images were helpful / neutral / harmful.

| ID | Deliverable | Dependencies |
|---|---|---|
| T1.10 | `--exclude-synthetic-species` flag on `pipeline/train/simple.py` to drop one rare species' synthetics from the loader. | Code review before use. |
| T1.11 | 6 training runs (seed 42): D4-no-ashtoni, D4-no-sandersoni, D4-no-flavidus, D5-no-ashtoni, D5-no-sandersoni, D5-no-flavidus. | `jobs/train_subset_ablation.sh` (chained or batched). Outputs to `RESULTS_seeds/subset_ablation/`. |
| T1.12 | Synthetic-level labels: compare each D4/D5 run's per-species F1 to its *no-{species}* variant; if F1 recovers when {species} synthetics are removed → that species' synthetics are harmful collectively. Per-image labels then assigned via nearest-neighbour agreement with harmed/improved real images. | new `scripts/label_synthetic_effect.py` → `RESULTS/failure_analysis/synthetic_labels.csv`. |
| T1.13 | Updated T1.6 galleries: split the synthetic column into helpful / neutral / harmful subpanels using the labels from T1.12. | Only after T1.12. |
| T1.14 | Updated T1.9 embedding plots: colour synthetics by helpful / neutral / harmful label. | Only after T1.12. |

**Review checkpoint after Phase 1c** — code review on `--exclude-synthetic-species` before submitting GPU jobs; sanity-check that the subset-ablation run has the expected reduced training-set size (D4 − 200 ashtoni synthetics, etc.).

### 4.6 Task 1 order of operations

1. Phase 1a (T1.1–T1.4): tables and heatmaps. 1–2 days, no new code beyond scripts.
2. Writing review.
3. Phase 1b (T1.5–T1.8): image-level galleries. 2–3 days.
4. Writing review.
5. **User decision point**: is Phase 1c (subset ablation) needed for the thesis narrative, or are the Phase 1b galleries sufficient?
6. If yes → Phase 1c (T1.10–T1.14). 3–5 days including GPU runs + updated figures.
7. Phase 1d — Task 1 write-up in thesis (§5 and §6).

---

## 5. Task 2 — Expert-Calibrated Quality Filter

### 5.1 Goal

Compare three filters that select synthetic images for augmentation, by downstream ResNet-50 macro F1:

- **LLM rule filter (D5, existing).** Strict threshold on LLM judge scores. No expert data.
- **BioCLIP centroid distance filter (D6-centroid).** Unsupervised; requires no expert data.
- **BioCLIP linear probe filter (D6-probe).** Expert-supervised; requires 150 expert labels.

Evaluated by (1) LOOCV filter accuracy on 150 expert images (for filters that have an expert-alignment interpretation), (2) downstream ResNet-50 macro F1 on D6 vs D4/D5, (3) CKNNA set-level alignment with real images.

### 5.2 Phase 2a — BioCLIP centroid distance filter (unblocked)

| ID | Deliverable | Notes |
|---|---|---|
| T2.1 | `pipeline/evaluate/filters.py::CentroidFilter` — fit per-species mean centroid on real training BioCLIP embeddings; score synthetic images by cosine distance. | L2-normalised embeddings; cosine = 1 − dot product. |
| T2.2 | `scripts/run_filter.py --filter centroid --output RESULTS/filters/centroid_scores.json` — rank all 1,500 synthetics. | Reuses `pipeline/evaluate/embeddings.load_cache`. |
| T2.3 | `scripts/assemble_d6.py --variant centroid --per-species 200 --output GBIF_MA_BUMBLEBEES/prepared_d6_centroid` — construct the prepared directory by symlinking top-200 closest synthetics per rare species + all real training images. | Structure mirrors `prepared_d4_synthetic` / `prepared_d5_llm_filtered`. |
| T2.4 | Register `d6_centroid` in `pipeline/config.resolve_dataset`. | |
| T2.5 | `jobs/train_d6_centroid.sh` — 5 seeds (42–46), same ResNet-50 protocol as D4/D5. Outputs to `RESULTS_seeds/d6_centroid_seed{42-46}_*`. | Code review before submission. |

**Review checkpoint after T2.5** — assembled dataset structure verified (counts per species) and config resolver tested before running.

### 5.3 Phase 2b — LLM rule AUC-ROC baseline (BLOCKED on expert labels)

| ID | Deliverable |
|---|---|
| T2.6 | `scripts/compute_llm_filter_auc.py --uncertain-policy {fail,exclude}` — read LLM morph scores + expert labels; compute AUC-ROC + precision@90%recall. Writes `RESULTS/filters/llm_rule_auc.json`. |

Script ready to run when expert labels arrive.

### 5.4 Phase 2c — Expert-supervised linear probe (BLOCKED on expert labels)

| ID | Deliverable | Notes |
|---|---|---|
| T2.7 | `pipeline/evaluate/filters.py::LinearProbeFilter` — sklearn LogisticRegression (class_weight=balanced, solver=lbfgs). Nested LOOCV: outer LOOCV over 150 images; inner stratified 5-fold over C ∈ {0.001, 0.01, 0.1, 1.0, 10.0}. Returns AUC-ROC, precision@90%recall, balanced accuracy. | Pooled probe across 3 species. |
| T2.8 | `scripts/run_filter.py --filter probe --output RESULTS/filters/probe_scores.json` — train on all 150 expert images, score all 1,500 synthetics. | |
| T2.9 | `scripts/assemble_d6.py --variant probe --per-species 200 --output GBIF_MA_BUMBLEBEES/prepared_d6_probe`. | |
| T2.10 | Register `d6_probe` in resolver. | |
| T2.11 | `jobs/train_d6_probe.sh` — 5 seeds. | |

### 5.5 Phase 2d — CKNNA evaluation

| ID | Deliverable | Notes |
|---|---|---|
| T2.12 | `pipeline/evaluate/cknna.py` — Centered Kernel Nearest-Neighbor Alignment (Huh et al., ICML 2024). Linear kernel on L2-normalised embeddings; k=5 for rare species (n≈22 real), k=10 for B. flavidus (n=162 real). | Random-split ceiling: CKNNA between two 50% splits of real images, averaged over 100 random splits. |
| T2.13 | `scripts/compute_cknna.py --embeddings-dir RESULTS/embeddings --output RESULTS/failure_analysis/cknna_results.json` — comparison: D4 / D5 / D6-centroid / D6-probe vs real. | |

CKNNA is computed for all available D6 variants — the centroid result (Phase 2a) can be reported before the probe result arrives.

### 5.6 Task 2 order of operations

1. Phase 2a (T2.1–T2.5): centroid filter + D6-centroid training. **Unblocked; next concrete step.**
2. Code review + writing review on centroid results.
3. Phase 2d partial (T2.12–T2.13 for D4/D5/D6-centroid): CKNNA for available variants.
4. **Wait for expert labels.**
5. Phase 2b (T2.6): LLM rule AUC-ROC baseline.
6. Phase 2c (T2.7–T2.11): expert-supervised linear probe + D6-probe.
7. Phase 2d rerun including D6-probe.
8. Phase 2e — thesis write-up in §4.4, §5.5.

---

## 6. Shared Infrastructure — Closed Work

### 6.1 Embedding extraction and backbone selection (closed 2026-04-16)

**Decision: BioCLIP primary, DINOv2 appendix only.**

5-NN leave-one-out diagnostic on real training images:

| Metric | DINOv2 ViT-L/14 (518×518) | BioCLIP (224×224) | Δ |
|---|---:|---:|---:|
| Overall 5-NN accuracy | 0.295 | **0.657** | +0.362 |
| Rare tier (3 spp., 224 imgs) mean purity | 0.072 | **0.125** | +0.053 |
| Moderate tier (7 spp., 3,378 imgs) mean purity | 0.133 | **0.468** | +0.335 |
| Common tier (6 spp., 7,331 imgs) mean purity | 0.273 | **0.614** | +0.341 |

Per-species purity and interpretation: see `RESULTS/embeddings/{dinov2,bioclip}_real_train_knn_diagnostic.json`.

Caches:
- `RESULTS/embeddings/bioclip_real_{train,valid,test}.npz` — (10,933 / 2,335 / 2,362) × 512.
- `RESULTS/embeddings/bioclip_synthetic.npz` — 1,500 × 512.
- `RESULTS/embeddings/dinov2_*.npz` — corresponding 1,024-d caches.

Figures:
- `docs/plots/embeddings/bioclip_{tsne,umap,pca}/` — thesis primary.
- `docs/plots/embeddings/dinov2_{tsne,umap,pca}/` — appendix.

All four figures per backbone × method (overview 16-species, real-vs-synthetic, rare-species zoom, centroid-distance histogram) use the unified 16-species HUSL palette defined in `scripts/plot_embeddings.py::SPECIES_PALETTE`.

### 6.2 Per-image flip analysis (closed 2026-04-17)

Script: `scripts/analyze_flips.py`. Output: `RESULTS/failure_analysis/flip_analysis.csv` (2,362 rows × 23 columns) + `flip_summary.json`.

Overall flip counts (baseline 5 seeds vs aug 5 seeds, majority rule):

| Config | stable-correct | stable-wrong | improved | harmed |
|---|---:|---:|---:|---:|
| D3 CNP | 2,070 | 198 | 40 | 54 |
| D4 Synthetic | 2,075 | 186 | 52 | 49 |
| D5 LLM-filtered | 2,075 | 189 | 49 | 49 |

Rare-species flip counts:

| Species | D3 improved/harmed | D4 improved/harmed | D5 improved/harmed |
|---|---|---|---|
| B. ashtoni (n=6) | 0 / 1 | 0 / 0 | 0 / 1 |
| B. sandersoni (n=10) | 0 / 1 | 0 / 1 | 0 / 1 |
| B. flavidus (n=36) | 0 / 5 | 0 / 8 | 0 / 6 |

**No rare-species test image is rescued by any augmentation.** Harm is the only directional effect.

---

## 7. Open Decisions

1. **Task 1 Phase 1c trigger.** After Phase 1a + 1b are complete, decide whether subset ablation is needed. If Phase 1b image-level figures already tell the story clearly, Phase 1c can be skipped or deferred.
2. **UNCERTAIN expert label policy.** Revisit when expert validation is under way.
3. **BioCLIP linear probe pooling.** Current plan is pooled across 3 species. Revisit if per-species performance diverges at Phase 2c evaluation.

---

## 8. Success Criteria

- Task 1 produces at least one image-level "smoking gun" figure per rare species (failure chain or confusion-pair triplet) that makes the augmentation-failure story obvious to the committee.
- Task 1 head/tail tier F1 table is reproducible from `RESULTS_seeds/` without new compute.
- Task 2 delivers D6-centroid downstream results before the expert-label blocker lifts.
- All new code reviewed; all GPU work via SLURM shell scripts in `jobs/`.
- Figures at 300 DPI with captions sufficient for direct thesis inclusion.
