# Thesis § 5 and § 6 — Results, Failure Analysis, and Discussion

This document is a standalone draft of the Experiments and Results (§ 5) and
Discussion (§ 6) sections, rewritten to integrate the Task-1 failure analysis.
Supporting detail that is not essential to the argument is deferred to the
appendix at the end of this document.

Cross-reference conventions:
- Figures under `docs/plots/` are referenced by relative path.
- Tables are either inline or referenced by path in `RESULTS/failure_analysis/`.
- "Multi-seed" = 5 seeds on the fixed 70/15/15 split. "K-fold" = 5-fold CV.
  Multi-seed is the primary protocol for per-image analysis; k-fold is reported
  where effects are sensitive to test-set composition.

---

## § 5 Experiments and Results

Section 5 develops the empirical case in three connected stages. Section 5.1 establishes baseline classifier behaviour and the visual confusion directions that frame every subsequent analysis. Section 5.2 characterises the BioCLIP feature-space geometry of real and synthetic images and identifies a per-species offset between them that pre-exists any classifier training. Sections 5.3 and 5.4 then report downstream classifier performance under three augmentation methods (copy-and-paste D3, unfiltered synthetic D4, LLM-filtered synthetic D5) and the LLM-judge quality signals that drove the D5 filter. Section 5.5 closes the argument by tracing the harm to a specific mechanism — synthetic images draw harmed test images toward wrong-species predictions in feature space — and confirming the mechanism causally with single-species ablations.

Two evaluation protocols are used throughout. **Five-fold cross-validation** is the primary protocol for aggregate and per-species claims, because it pools predictions across folds to reach rare-species effective test sizes of n = 32 / 58 / 232, following Shipard et al. (2023) and Picek et al. (2022) on small biological datasets. **Multi-seed training** (five seeds on the fixed 70/15/15 split) is used whenever a per-image analysis is required, because each seed evaluates the same 2 362 test images. Single-split results are reported alongside as an honest record of the early experiments. With df = 4 in the paired t-tests, non-significant comparisons should be read as underpowered rather than as evidence of equivalence. All results use the best-validation-macro-F1 checkpoint (best_f1.pt), matching the primary reporting metric.

### 5.1 Baseline

#### 5.1.1 Single-run classifier performance

Table 5.1 reports the ResNet-50 baseline on the fixed 70/15/15 split with 10 000-iteration bootstrap 95 % confidence intervals. Overall accuracy reaches 87.9 % and macro F1 reaches 0.810, figures driven by the eleven head-and-moderate species with n ≥ 200 training images. The three rare targets fall substantially below this aggregate: B. ashtoni reaches F1 0.545 (n = 6 test, 95 % CI [0.000, 0.857]), B. sandersoni 0.471 (n = 10, [0.133, 0.737]), and B. flavidus 0.667 (n = 36, [0.517, 0.794]). The B. ashtoni interval spans 0.86 of the F1 range — an honest reflection of evaluation variance at n = 6 that no single-run comparison can narrow.

*Table 5.1 — Baseline ResNet-50 classifier on the fixed split, f1 checkpoint. Rare species in bold.*

| Species | Train n | Test n | Precision | Recall | F1 | 95 % CI |
|---|---:|---:|---:|---:|---:|---|
| **B. ashtoni** | 22 | 6 | 0.750 | 0.500 | **0.545** | [0.000, 0.857] |
| **B. sandersoni** | 40 | 10 | 0.714 | 0.500 | **0.471** | [0.133, 0.737] |
| **B. flavidus** | 162 | 36 | 0.828 | 0.667 | **0.667** | [0.517, 0.794] |
| Macro average | — | 2 362 | — | — | 0.810 | [0.767, 0.841] |
| Overall accuracy | — | 2 362 | — | — | 0.879 | — |

Per-species F1 correlates cleanly with training-set size: every species with n ≥ 200 exceeds 0.75 F1, while every species below n = 200 falls under 0.70. The three rare species therefore define the augmentation target and determine every subsequent comparison.

#### 5.1.2 Rare-species confusion structure

The row-normalised confusion matrix (Figure 5.1) reveals three recurring confusion directions that frame the analyses to follow. B. ashtoni is misclassified primarily as B. citrinus and B. vagans — all three are dark-thorax bees with overlapping body size and pale tergite markings. B. sandersoni is confused with B. vagans in the majority of error cases, reflecting their shared yellow-anterior / black-posterior body pattern. B. flavidus errors distribute across B. citrinus, B. terricola and B. ternarius — the yellow-tergite species its pale coloration most resembles. These confusion directions provide the visual priors against which any synthetic augmentation must succeed; they reappear as the anchor for the failure-chain retrievals in § 5.5.

![Baseline confusion matrix](plots/baseline_confusion_matrix.png)
*Figure 5.1 — Row-normalised baseline confusion matrix on the fixed split. Bold labels mark the three rare augmentation targets.*

### 5.2 Latent-Space Analysis

This section characterises the real-image feature geometry that any augmentation strategy must respect, then shows that synthetics for the three rare species occupy an embedding-space region systematically offset from the corresponding real clusters. The analysis uses BioCLIP ViT-B/16 rather than DINOv2 ViT-L/14 because BioCLIP's biology-specific pre-training produces substantially better species-level separation on the training data, as established by a nearest-neighbour purity diagnostic in § 5.2.1.

#### 5.2.1 Backbone selection

To choose the diagnostic embedding backbone, I compute 5-NN leave-one-out species classification accuracy on all 10 933 real training images under both backbones. Table 5.2 reports overall and per-tier accuracy. BioCLIP achieves 0.657 overall and 0.125 on the rare tier; DINOv2 achieves 0.295 and 0.072. The gap is large and consistent across tiers — BioCLIP's representation space is more aligned with species identity than general-purpose vision features, and the choice is made on these data rather than on prior literature. While BioCLIP's feature space is not identical to the ResNet-50 classifier's, its superior species-level structure makes it the most informative available proxy for diagnostic quality. A ResNet-50 penultimate-layer probe is listed in § 7.2 as future work.

*Table 5.2 — 5-NN leave-one-out species classification accuracy on real training images (10 933 images, 16 species, cosine metric).*

| Backbone | Dim | Overall | Rare (3 spp) | Moderate (7 spp) | Common (6 spp) |
|---|---:|---:|---:|---:|---:|
| DINOv2 ViT-L/14 (518²) | 1 024 | 0.295 | 0.072 | 0.133 | 0.273 |
| BioCLIP ViT-B/16 (224²) | 512 | **0.657** | **0.125** | **0.468** | **0.614** |

#### 5.2.2 Real-image feature geometry

Figure 5.2a projects all 10 933 real training images into a BioCLIP t-SNE. Common species form compact, well-separated clusters; B. impatiens, B. ternarius and B. griseocollis are clearly resolved. The rare species do not enjoy this separation. Figure 5.2b isolates the three rare targets together with their four recurring confusers from § 5.1.2: B. sandersoni and B. vagans occupy overlapping regions, B. ashtoni sits along a boundary with B. citrinus, and B. flavidus scatters through territory shared with B. citrinus and B. terricola. This indistinguishability pre-exists any synthetic-augmentation question — it is the native problem augmentation must address.

![16-species real-image t-SNE](plots/embeddings/bioclip_tsne/embeddings_overview.png)
*Figure 5.2a — BioCLIP t-SNE of 10 933 real training images, 16 species. Common species form compact clusters; rare species overlap with their visual confusers.*

![Rare-species real-image t-SNE](plots/embeddings/bioclip_tsne/embeddings_rare_real_only.png)
*Figure 5.2b — Rare species real images and their four recurring confusers. Baseline indistinguishability is apparent before any augmentation.*

#### 5.2.3 The synthetic–real embedding gap

Projecting real and synthetic images for the three rare species into a shared t-SNE space (Figure 5.3a) reveals the central structural finding of § 5.2: synthetic images of each rare species form their own tight cluster, well-separated from the corresponding real cluster in the same projection. The gap is not a generic "synthetic ≠ real" artefact; it is a per-species manifold offset.

Figure 5.3b quantifies this. For each synthetic image I compute its cosine distance to the centroid of its target species' real training embeddings. The median synthetic-to-centroid distance is 0.31 for B. ashtoni, 0.25 for B. sandersoni, and 0.32 for B. flavidus. The corresponding real-to-centroid distances within each rare species fall in the 0.10–0.20 range — roughly half. The synthetic sub-manifold is therefore systematically carved out at a few tenths of cosine-space removed from where real rare bees actually sit. A representative confusion-pair triplet (Appendix E, B. flavidus vs B. citrinus) and the embedding atlas with thumbnails at true t-SNE coordinates (also Appendix E) confirm visually that the clusters are pose- and coloration-coherent rather than projection artefacts.

![Rare real + synthetic t-SNE](plots/embeddings/bioclip_tsne/embeddings_rare_real_synth.png)
*Figure 5.3a — BioCLIP t-SNE of rare-species real and synthetic images. Synthetic clusters sit offset from the corresponding real clusters for each species.*

![Synthetic-to-centroid cosine distance](plots/embeddings/bioclip_tsne/embeddings_centroid_distance.png)
*Figure 5.3b — Per-synthetic cosine distance to the species' real-image centroid. Dashed lines mark real-to-centroid medians for comparison.*

This feature-space offset has a direct downstream prediction: training on these synthetics teaches the classifier a set of species-discriminative features that are offset from the feature subspace real test images of the same species occupy. § 5.3 tests that prediction on macro F1, and § 5.5 traces the per-image consequences.

### 5.3 Augmentation Method Comparison

#### 5.3.1 Aggregate and tier-level effects

Table 5.3 reports macro F1 across three protocols, the rare-tier F1 under the two multi-image protocols, and the moderate- and common-tier F1 under 5-fold CV. Two facts dominate the table. First, augmentation effects are concentrated in the rare tier: moderate and common tiers move by ≤ 0.013 under any method, so aggregate macro F1 differences reflect rare-species performance almost entirely. Second, the three protocols produce different aggregate rankings — D5 is best on single-split (0.834), D3 is best on 5-fold (0.837), and D1 is best on multi-seed (0.839). The rankings are not contradictory: multi-seed and single-split share the same 6 / 10 / 36 rare test images, so flipping one or two correctly-classified rare images is enough to swap the aggregate ranking; 5-fold pools roughly five times more rare test predictions and is the more reliable reading.

*Table 5.3 — Macro F1 across augmentation strategies and three protocols, with tier-mean F1 under the multi-image protocols (f1 checkpoint). Rare-tier figures are unweighted species-means within tier; bracketed values are per-tier deltas vs. baseline. Best per row in bold.*

| Quantity | Protocol | D1 Baseline | D3 CNP | D4 Synthetic | D5 LLM-filt. |
|---|---|---:|---:|---:|---:|
| Macro F1 | Single-split | 0.815 | 0.829 | 0.823 | **0.834** |
| Macro F1 | 5-fold CV | 0.832 ± 0.013 | **0.837 ± 0.013** | 0.820 ± 0.024 | 0.821 ± 0.019 |
| Macro F1 | Multi-seed | **0.839 ± 0.006** | 0.822 ± 0.014 | 0.828 ± 0.009 | 0.831 ± 0.008 |
| Rare-tier F1 | 5-fold CV | 0.611 | **0.641** (+0.030) | 0.555 (−0.056) | 0.570 (−0.041) |
| Rare-tier F1 | Multi-seed | **0.665** | 0.593 (−0.073) | 0.590 (−0.075) | 0.617 (−0.048) |
| Moderate-tier F1 | 5-fold CV | 0.861 | 0.862 | 0.860 | 0.855 |
| Common-tier F1 | 5-fold CV | 0.908 | 0.906 | 0.906 | 0.907 |

Under 5-fold CV, copy-and-paste augmentation lifts rare-tier F1 by 0.030 above baseline, while both synthetic variants reduce it (D4 by 0.056, D5 by 0.041). Under multi-seed, all three augmentation methods reduce rare-tier F1, with synthetic variants again worst. The two protocols disagree on whether D3 helps but agree on the signed effect of D4 and D5: they harm the rare tier under every protocol considered.

#### 5.3.2 Per-species effects and statistical significance

Pairwise paired t-tests on fold-level macro F1 (Table 5.4) identify two comparisons that clear the high-power bar imposed by df = 4. D5 is significantly worse than the baseline (p = 0.041) and significantly worse than D3 (p = 0.030). D4 and D5 are statistically indistinguishable (p = 0.777): the strict LLM filter, applied to the D4 pool, produces no measurable improvement over no filter at all. This non-result is the central empirical motivation for the failure-mode analysis in § 5.5.

*Table 5.4 — Pairwise paired t-tests on fold-level macro F1 (5-fold CV, df = 4). Significant results in bold.*

| Comparison | Mean Δ | t | p | Significant |
|---|---:|---:|---:|---|
| D1 vs D3 CNP | +0.005 | 1.72 | 0.161 | No |
| D1 vs D4 Synthetic | −0.012 | −2.17 | 0.096 | No |
| **D1 vs D5 LLM-filtered** | **−0.011** | **−2.98** | **0.041** | **Yes — D5 worse** |
| D3 vs D4 Synthetic | −0.017 | −2.59 | 0.061 | No |
| **D3 vs D5 LLM-filtered** | **−0.016** | **−3.29** | **0.030** | **Yes — D5 worse** |
| D4 vs D5 LLM-filtered | +0.001 | 0.30 | 0.777 | No (filter no benefit) |

Per-species analysis locates the signal. D3 significantly improves B. flavidus F1 over baseline under 5-fold (+0.059, p = 0.005), the only statistically significant per-species gain from any augmentation method. D4 reduces B. sandersoni F1 by 0.140 under 5-fold (from 0.466 to 0.326, marginally significant at p = 0.052) and D5 reduces it by 0.068. B. ashtoni produces large numerical swings (D3 +0.064, D4 −0.045, D5 −0.044 under 5-fold) but with 95 % CIs of width 0.27–0.30 due to n = 32 pooled test images, none of these per-species comparisons reach significance. Figure 5.4 plots per-species Δ F1 for both protocols side-by-side: rare-species rows are coloured saturated under D4 and D5, while moderate and common rows are essentially flat.

![Per-species delta F1, 5-fold and multi-seed](plots/failure/species_f1_delta_kfold.png)
*Figure 5.4 — Per-species F1 change relative to D1 baseline under 5-fold CV (primary) and multi-seed (fixed split). Rare species highlighted; moderate and common tiers show negligible effects under any method.*

#### 5.3.3 Volume ablation

A natural concern is whether the D4 and D5 rare-species harm would resolve at higher synthetic volumes. Figure 5.5 reports macro F1 and rare-species F1 under D4 and D5 at volumes of +50, +100, +200, +300 and +500 images per rare species (single-split evaluation). Neither variant shows a coherent volume–performance trend: D4 macro F1 fluctuates between 0.820 and 0.834 across volumes without a monotone direction, and D5 peaks at +200 but does not maintain the gain at +300 or +500. At +500, both variants regress on B. sandersoni (D4 0.471, D5 0.556) as the synthetic-to-real ratio reaches 12.5 : 1 for that species. The absence of a volume-dependent improvement establishes that the bottleneck is generation fidelity, not quantity — adding more synthetics of the same quality does not close the embedding-space gap that § 5.2 identified.

![Volume ablation](../RESULTS_count_ablation/volume_ablation_trends_with_ci.png)
*Figure 5.5 — Volume ablation for D4 and D5 at +50 to +500 synthetic images per rare species. No consistent improvement at any volume for either variant.*

### 5.4 LLM-as-Judge Results

The LLM-as-judge (§ 4.3) evaluates every generated image on species-level morphology. Section 5.4 characterises what the judge sees. Section 5.5 returns to whether what the judge sees matches what the classifier needs.

#### 5.4.1 Pass rates and filter funnel

Table 5.5 reports the two-stage judge's output over all 1 500 generated images (500 per rare species). Blind-identification match rates are high for B. sandersoni (96.4 %) and B. flavidus (96.0 %) but much lower for B. ashtoni (76.0 %), reflecting the inverse-phenotype challenge: ashtoni's predominantly black thorax inverts the dominant Bombus prior, and the judge — like the generation model — sometimes defaults to yellow-thorax interpretations. The mean morphological score follows the same pattern (ashtoni 3.82, flavidus 4.06, sandersoni 4.37), as does the strict pass rate: 44.4 % ashtoni, 57.6 % flavidus, 91.2 % sandersoni. The strict funnel reduces the 1 500 image pool through three sequential gates — blind-ID match (1 342, 89.5 %), diagnostic = species (1 060, 70.7 %), and morph mean ≥ 4.0 (966, 64.4 %).

*Table 5.5 — LLM-as-judge evaluation of 1 500 generated images (500 per species). Strict pass requires blind-ID match AND diagnostic = species AND mean morph ≥ 4.0.*

| Species | Blind ID | Mean morph | Lenient pass | Strict pass |
|---|---:|---:|---:|---:|
| B. ashtoni | 76.0 % | 3.82 | 92.0 % | 222 / 500 (44.4 %) |
| B. flavidus | 96.0 % | 4.06 | 99.6 % | 288 / 500 (57.6 %) |
| B. sandersoni | 96.4 % | 4.37 | 100 % | 456 / 500 (91.2 %) |

#### 5.4.2 Per-feature diagnostics

The bottleneck is narrow and localised. The mean per-feature morphological scores hold above 4.0 for fourteen of the fifteen (species × feature) cells; the lone exception is B. ashtoni's thorax-coloration mean of 2.98 — far below every other cell. Wrong-coloration is the dominant failure mode overall (27.1 % of all 1 500 images), concentrated in ashtoni. Structural failure modes — extra or missing limbs, impossible geometry, visible artefacts, repetitive patterns — register at exactly 0 across all 1 500 images, confirming that the structured-prompting framework of § 4.2 has eliminated this failure class. The residual gap is a colour-fidelity gap, and it is concentrated in the species that deviates most from the genus-typical phenotype. Per-angle and per-caste breakdowns — including the frontal-view paradox in which abdomen-banding occlusion drives a high mean morph but low strict pass, and the male-caste deficit in B. ashtoni — are reported in Appendix E. They confirm the judge's scoring behaves coherently across view conditions and do not change the filter-calibration argument that follows.

#### 5.4.3 What the judge measures — and what it does not

The judge measures species-level morphological fidelity as assessed by a vision–language model on a human-visual rubric, and Table 5.5's funnel shows the measurement is informative: the judge correctly identifies the ashtoni generation bottleneck, passes sandersoni at near-ceiling, and scales morph scores with generation difficulty. What the judge does not measure is whether a synthetic image is useful for the downstream classifier. § 5.3 has already established that D4 and D5 are statistically indistinguishable on macro F1 (p = 0.777): removing the 27.1 % wrong-coloration images the judge flags does not rescue rare-species F1. § 5.5 traces the reason — the judge's pass set still contains many images that the classifier's BioCLIP feature space places beyond the real species distribution.

### 5.5 Failure Mode Analysis

Sections 5.3 and 5.4 establish *that* synthetic augmentation harms rare species and *that* the LLM filter does not fix the harm. Section 5.5 addresses *why*, combining per-image prediction tracking, embedding-space failure-chain retrieval, judge–classifier disagreement, and a causal ablation of each species' synthetic contribution. The analysis here uses the multi-seed protocol (§ 5.3.1) because per-image flip and chain analyses require every seed to evaluate the same images.

#### 5.5.1 Per-image prediction flips

Each of the 2 362 test images produces 20 predictions across the four configs and five seeds. Collapsing within each config by majority vote yields one verdict per (image, config) pair, and comparing each augmented config against the baseline partitions test images into four cells: stable-correct (both right), stable-wrong (both wrong), improved (augmentation rescued a baseline error), and harmed (augmentation broke a baseline-correct image).

*Table 5.6 — Rare-species flip counts under each augmentation method (multi-seed majority vote, fixed split). No rare image is improved by any method.*

| Species | n test | D3 (impr / harm) | D4 (impr / harm) | D5 (impr / harm) |
|---|---:|---|---|---|
| B. ashtoni | 6 | 0 / 1 | 0 / 0 | 0 / 1 |
| B. sandersoni | 10 | 0 / 1 | 0 / 1 | 0 / 1 |
| B. flavidus | 36 | 0 / 5 | 0 / 8 | 0 / 6 |

The pattern is stark: no rare-species test image is improved by any augmentation method. The effect is not a mixture of improvements and harms that averages out poorly — it is one-directional harm. Expressed as a rate, B. flavidus is harmed on 22.2 % of its D4 test images, the largest cell in any (species × method) pair, while no common species exceeds a 3 % harm rate. Figure 5.6 visualises the species-by-method harm rates; the rare tier carries the signal and the remaining tiers are near zero.

![Flip-category heatmap](plots/failure/flip_category_heatmap.png)
*Figure 5.6 — Flip-category rates by species and augmentation method (multi-seed majority vote, fixed split). Rare rows carry the signal; moderate and common rows are near zero.*

#### 5.5.2 Embedding-space failure chains

For every rare-species test image harmed under D4 or D5, I retrieve the five nearest training synthetics of the corresponding variant by BioCLIP cosine similarity, restricted to that variant's actual training pool (600 synthetics per variant). Each retrieved synthetic carries its generated-species label and LLM tier. A representative harmed B. flavidus test image under D4 (Figure 5.7) returns five nearest training synthetics that are all B. sandersoni strict-pass images at cosine similarity 0.60 — closer to the harmed image in BioCLIP space than any D4 B. flavidus synthetic. At test time the classifier predicts B. sandersoni; under the baseline the same image is classified correctly. Across all 49 D4 harmed chains, the median test-to-5-NN cosine similarity is 0.56, well below the 0.7+ range at which two real images of the same species typically match. The "nearest training synthetic" is therefore not close to the harmed test image in absolute terms; it is close only relative to the rest of the synthetic pool.

![Representative D4 failure chain](plots/failure/chains_d4_harmed/gallery/flavidus__Bombus_flavidus4512075898.png)
*Figure 5.7 — Representative D4 failure chain: a harmed B. flavidus test image and its five nearest D4 training synthetics by BioCLIP cosine similarity. The full 49-chain galleries (D4 harmed, D4 improved, D5 harmed, D5 improved) and t-SNE projections are in Appendix C.*

#### 5.5.3 Judge–classifier disagreement

The failure chains identify which synthetics the classifier latches onto; Figure 5.8 shows that many of these synthetics passed the LLM filter. For every synthetic I plot LLM mean morphological score against BioCLIP cosine distance to the correct species' real centroid, and partition the plane at per-species medians on both axes. The upper-right quadrant — "LLM passes above median morph *and* far from the real centroid" — contains 138 B. ashtoni, 49 B. sandersoni, and 108 B. flavidus synthetics. These are exactly the images the judge cannot reject but the classifier's feature space rejects. They pass through the D4 → D5 filter unchanged, providing a mechanistic explanation for the D4 vs D5 p = 0.777 result in § 5.3.2.

![Judge versus centroid distance](plots/failure/llm_vs_centroid_quadrant.png)
*Figure 5.8 — Per-synthetic LLM mean morphological score versus BioCLIP cosine distance to the correct species' real centroid. Dashed lines mark per-species medians. The upper-right quadrant counts disagreement between judge-relevant and classifier-relevant quality.*

#### 5.5.4 Causal attribution via subset ablation

To confirm the synthetic sub-manifold is causally responsible for the harm — rather than merely correlated with it — I run six additional training jobs (seed 42 only) that each drop all synthetic images of exactly one rare species from D4 or D5. Recovery is defined as F1 under ablation minus F1 under the full variant. Positive recovery implies the removed synthetics were collectively harming the target species; negative recovery implies they were collectively helping.

*Table 5.7 — Own-species F1 recovery under single-species subset ablation (seed 42). Threshold |Δ| > 0.02 for a directional label; otherwise neutral.*

| Variant | Dropped species | F1 full → ablated | Recovery | Label |
|---|---|---:|---:|---|
| D4 | ashtoni | 0.545 → 0.727 | **+0.182** | harmful |
| D4 | sandersoni | 0.571 → 0.476 | −0.095 | helpful |
| D4 | flavidus | 0.645 → 0.708 | +0.062 | harmful |
| D5 | ashtoni | 0.727 → 0.727 | +0.000 | neutral |
| D5 | sandersoni | 0.625 → 0.706 | **+0.081** | harmful |
| D5 | flavidus | 0.725 → 0.719 | −0.006 | neutral |

Two patterns dominate. First, the LLM filter neutralises the large B. ashtoni harm seen in D4 (+0.182 recovery converts to +0.000 in D5): the filter does remove genuinely bad ashtoni synthetics. Second, **the filter reverses the B. sandersoni effect**. Unfiltered sandersoni synthetics were collectively helpful in D4 (removing them lost 0.095 F1); filtered sandersoni synthetics are collectively harmful in D5 (removing them gains 0.081 F1). Because B. sandersoni has the highest LLM strict-pass rate of the three species (91.2 %), the filter retains nearly all images and the 8.8 % it discards includes the signal that was compensating for the harmful subset. This is the clearest empirical demonstration of LLM-judge miscalibration in the dataset: for the species the filter passes most easily, it discards exactly the wrong subset.

Cross-species collateral effects are substantial and reinforce the embedding-space picture. Dropping B. ashtoni synthetics in D4 simultaneously reduces B. sandersoni F1 by 0.150 and increases B. flavidus F1 by 0.124 — the same image set provides conflicting gradient signal across class decisions, consistent with the failure-chain finding that retrieved nearest synthetics frequently belong to the wrong species. Figure 5.9 shows the full 3 × 3 recovery matrix and the own-species recovery bars.

![Subset ablation recovery](plots/failure/subset_ablation_recovery.png)
*Figure 5.9 — Subset ablation recovery. Left: own-species F1 recovery under D4 and D5 by dropped species. Right: full dropped-versus-measured recovery matrix. Single seed (42); the ablation establishes direction, not magnitude.*

Each ablation cell is a single seed-42 run with rare-species test n between 6 and 36, so a 0.05 F1 change corresponds to flipping 0.3–1.8 images. The analysis is designed to establish direction of effect, not statistically-powered magnitude; the B. sandersoni D4 → D5 sign reversal is a qualitative signal that stochastic noise cannot easily produce, while the cross-species collateral magnitudes should be read as directional only. Per-synthetic labels — propagating the own-species verdicts to every generated image in each variant — are written to `RESULTS/failure_analysis/synthetic_labels.csv` for use in § 5.6.

### 5.6 Expert Calibration Results

Section 4.4 specified a learned filter trained on entomologist annotations over a stratified 150-image sample, compared against the LLM-rule and BioCLIP-centroid-distance baselines. Downstream results for the D6-probe and D6-centroid variants, leave-one-out AUC-ROC for each filter on the expert sample, CKNNA alignment between each variant and the real-image distribution, and the D6 row added to Table 5.3 depend on completed expert validation of the annotation sample. This section is held pending those annotations.

---

## § 6 Discussion

### 6.1 Augmentation harm is rare-tier-specific

The most load-bearing finding of § 5.3 is that augmentation-method effects
are not distributed uniformly across species but are concentrated in the
tier with the fewest real training images. Moderate and common tiers move
by less than a single per-species standard deviation under any method; the
rare tier moves by ~ 0.05 – 0.08 macro F1 (multi-seed) in the *negative*
direction. This is the opposite of the hypothesis that motivated
augmentation in the first place — synthetic data is the intervention least
likely to help the tier that most needs it.

The mechanism in § 5.4 is specific enough to predict which failure modes
will persist under tighter filtering. D5 is a strict LLM filter on D4,
yet D5 remains in the same harm regime (multi-seed Δ D5 = −0.048 on the
rare tier, vs D4 Δ = −0.075). The synthetic-real gap in BioCLIP space is
larger than any plausible filtering criterion can close without filtering
so aggressively that the augmentation volume collapses below usefulness.

### 6.2 Why rare-species augmentation degrades performance — a mechanistic account

§ 5.4 assembles four pieces of evidence that jointly explain the harm:

1. **No rare-species test image is improved by any augmentation method**
   (§ 5.4.1). The effect is not "some helped, some harmed, on average worse";
   it is "only harmed".
2. **Synthetic images of each rare species occupy a sub-manifold offset from
   the corresponding real cluster** (§ 5.4.2). Median synthetic → real
   cosine distance is 0.3 for rare species, vs 0.1 – 0.2 for real → real.
3. **Nearest-neighbour retrieval from harmed test images returns synthetics
   at cos 0.56** — far below the 0.7 + range at which real → real of the same
   species peaks (§ 5.4.4). The classifier sees these as out-of-distribution
   at inference time despite having been trained on them.
4. **The LLM judge cannot tell when its "pass" synthetics are still far from
   real feature space** (§ 5.4.5). The upper-right disagreement quadrant in
   Figure 5.7 contains hundreds of synthetics per species that pass the LLM
   but sit beyond the real manifold.

Taken together, the claim is: *generative augmentation trains the
classifier on a set of features that look correct to a language-mediated
rubric but that the classifier itself cannot unify with its real-image
features, because the feature-space gap exceeds the model's local
generalisation budget during fine-tuning.* The subset ablation results
in § 5.4.6 convert this from a correlational claim to a causal one:
removing B. ashtoni's D4 synthetics recovers ashtoni F1 by +0.182, and
removing D5 sandersoni synthetics recovers sandersoni F1 by +0.081.
Even after LLM filtering — which does neutralise the ashtoni harm — the
filtered sandersoni subset is still pulling classification quality down.

A second finding from § 5.4.6 complicates the picture: *augmentation
effects are cross-species*. Dropping ashtoni synthetics simultaneously
hurts sandersoni (−0.150) and helps flavidus (+0.124), meaning the same
image set is a training signal for multiple class decisions. Single-
species filtering strategies (per-species LLM thresholds, per-species
centroid distance) may therefore produce counter-intuitive interactions
when applied together, and a well-calibrated global filter needs to
weigh these cross-species gradients, not just per-image quality.

### 6.3 Copy-Paste vs Generative: fidelity and diversity

CNP (D3) preserves real-image texture by segmenting real bees and compositing
them onto real flower backgrounds. Under k-fold, CNP is the only method
that improves a rare species significantly (B. flavidus, p = 0.005). In the
Task-1 embedding views (§ 5.4.2), CNP is not shown because CNP images were
not separately embedded in this pass — but the hypothesis the data supports
is that CNP stays inside the real-image manifold because its source pixels
are real, while generative augmentation lives in its own sub-manifold.
Future work (§ 7.2) should confirm this by running the same BioCLIP
analysis on D3 images.

### 6.4 LLM judges: useful signals, wrong calibration

The LLM judge's per-feature scores are genuinely informative — the strict
pass rate correlates with species generation difficulty (ashtoni 44.4 %,
sandersoni 91.2 %). But § 5.4.5 shows the judge and classifier-relevant
quality disagree systematically in one specific quadrant: LLM passes +
classifier-space far. This is exactly the regime an automated filter will
pass through unchanged while a human taxonomist, looking at the image
alongside a real reference, would reject.

The subset ablation in § 5.4.6 gives a concrete illustration of this
calibration gap. For B. sandersoni the LLM strict-pass rate is 91.2 %,
so the D4 → D5 filter retains nearly all its sandersoni synthetics. The
ablation shows that this retained subset reverses sandersoni's effect:
*unfiltered* sandersoni synthetics help (D4 recovery −0.095), but
*filtered* sandersoni synthetics harm (D5 recovery +0.081). The filter
is discarding exactly the synthetics that were compensating for the
harmful ones. A good filter for sandersoni must therefore use signals
beyond the LLM's per-feature score — plausibly the BioCLIP centroid
distance of § 5.4.5, or the expert labels of Task 2.

The thesis case for expert calibration (Task 2) is that per-feature LLM
scores are the right *information*, but the rule that converts them to
a filter decision needs to be learned from expert labels rather than
taken from a language-model threshold.

### 6.5 Statistical challenges with rare species

Small-n rare-species test sets make multi-seed F1 estimates brittle: a
single flipped B. ashtoni test image is 0.10 F1. Multi-seed reduces training
variance but preserves this evaluation variance. K-fold pools predictions
across 5 × more test images per rare species and is the right protocol for
aggregate macro F1 claims. However, k-fold cannot support per-image flip
analysis because each image appears in exactly one fold. Multi-seed on the
fixed split is the correct protocol for § 5.4.1, even though its aggregate
F1 numbers are the least reliable.

### 6.6 Implications for urban biodiversity monitoring

The practical consequence is a false negative for rare Bombus species. If
the classifier misclassifies B. ashtoni as B. citrinus or B. vagans, a
monitoring deployment reports "ashtoni absent"; planners then may approve
development on habitat where the species still persists. § 5.3 and § 5.4
together show that current generative augmentation pipelines *increase* the
false-negative rate for exactly the species most at risk of detection
failure. Until the feature-space gap is closed, deployment should prefer
either real-texture augmentation (D3 CNP) or volume-based upweighting
strategies that do not require generative priors.

### 6.7 Limitations

- Per-seed variance for rare species is large on the fixed split; k-fold is
  used as the stability check. Both protocols are reported side-by-side.
- BioCLIP embeddings provide a strong species prior (see Appendix D) but
  are not a downstream classifier's feature space. A ResNet-50 penultimate
  extraction is outlined in the implementation plan as a future diagnostic.
- The subset-ablation causal attribution (§ 5.4.6) is run at a single seed
  and reports directions, not statistically-powered magnitudes.
- Expert calibration results (§ 5.5) are deferred until annotation is
  complete; D6 numbers will be added.

---

## Appendix

### A. Multi-protocol reconciliation

Full paired t-tests, pairwise p-values, bootstrap confidence intervals, and
fold-level variance are reproduced from `docs/experimental_results.md`
Tables 5.8–5.12. The key reconciliation is that single-split / multi-seed
and k-fold disagree on which method "wins" aggregate macro F1, but agree on
the signed rare-species effect (D4 and D5 negative under both protocols;
D3 positive under k-fold only, ambiguous under multi-seed).

### B. Per-species flip counts and correctness rates

Generated by `scripts/analyze_flips.py`. Full CSV at
`RESULTS/failure_analysis/flip_analysis.csv`. Per-species summary at
`RESULTS/failure_analysis/flip_summary.json`. Selected numbers:

- Absolute D4 harmed counts (all 16 species): max flavidus (8), sandersoni
  (1), ashtoni (0); common species max B. bimaculatus (4), which is 1.4 %
  of its test support.
- Common-species flip rates (harmed / n_test under D4) never exceed 3 %;
  rare-tier D4 harmed rates are 10 % (sandersoni), 22.2 % (flavidus), 0 %
  (ashtoni, on only n = 6).

### C. Failure-chain gallery

216 per-chain PNGs split by variant and direction:

- `docs/plots/failure/chains_d4_harmed/{gallery,tsne}/` (49 + 9)
- `docs/plots/failure/chains_d4_improved/{gallery,tsne}/` (52 + 0)
- `docs/plots/failure/chains_d5_harmed/{gallery,tsne}/` (49 + 8)
- `docs/plots/failure/chains_d5_improved/{gallery,tsne}/` (49 + 0)

t-SNE chains are produced only for rare-species test images (the shared
projection is rare-real + variant-synthetic); common-species improved
chains render galleries only. Full inventory at
`scripts/build_failure_chains.py`.

### C'. Per-synthetic helpful / neutral / harmful labels

Generated by `scripts/label_synthetic_effect.py`. Labels are propagated
at the species level from the subset ablation own-recovery (§ 5.4.6) with
a ±0.02 F1-delta neutral band:

| Species | D4 label | D5 label |
|---|---|---|
| ashtoni | harmful | neutral |
| sandersoni | helpful | harmful |
| flavidus | harmful | neutral |

Full 1,500-row CSV at `RESULTS/failure_analysis/synthetic_labels.csv`.
Per-image labels are not attempted here because a 1-seed run cannot
distinguish per-image contribution from training stochasticity at the
granularity of individual synthetics.

### D. Embedding diagnostic details

5-NN leave-one-out accuracy on `prepared_split/train/` (10 933 images, 16
species). Cosine metric.

| Backbone | Dim | Overall 5-NN accuracy | Rare tier mean purity | Moderate | Common |
|---|---:|---:|---:|---:|---:|
| DINOv2 ViT-L/14 (518²) | 1 024 | 0.295 | 0.072 | 0.133 | 0.273 |
| BioCLIP ViT-B/16 (224²) | 512 | **0.657** | **0.125** | **0.468** | **0.614** |

Per-species purity and full diagnostic JSONs at
`RESULTS/embeddings/{dinov2,bioclip}_real_train_knn_diagnostic.json`.
BioCLIP is used for all embedding figures in § 5 except the appendix
comparison.

### E. LLM-centroid quadrant counts

Per-species, split at species-median LLM morph mean and species-median
BioCLIP centroid distance. Generated by
`scripts/plot_llm_centroid_quadrant.py`.

| Species | n | Med morph | Med dist | LLM>med & far | LLM>med & close | LLM≤med & far | LLM≤med & close |
|---|---:|---:|---:|---:|---:|---:|---:|
| B. ashtoni | 500 | 3.80 | 0.305 | 138 | 97 | 112 | 153 |
| B. sandersoni | 500 | 4.40 | 0.253 | 49 | 25 | 201 | 225 |
| B. flavidus | 500 | 4.00 | 0.318 | 108 | 107 | 142 | 143 |

The "LLM>med & far" column is the disagreement quadrant discussed in
§ 5.4.5.
