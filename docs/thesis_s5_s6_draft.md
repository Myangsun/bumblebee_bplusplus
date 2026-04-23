
## 5. Experiments and Results

Section 5 develops the empirical case in three connected stages. Section 5.1 establishes baseline classifier behaviour and the visual confusion directions that frame every subsequent analysis. Section 5.2 characterises the BioCLIP feature-space geometry of real and synthetic images and identifies a per-species offset between them that pre-exists any classifier training. Sections 5.3 and 5.4 report downstream classifier performance under three augmentation methods -- copy-and-paste (D3), unfiltered generative synthetic (D4), and LLM-filtered synthetic (D5), each adding 200 images per rare species -- and the LLM-judge quality signals that drove the D5 filter. Section 5.5 closes the argument by tracing the harm to a specific mechanism -- synthetic images pull harmed test images toward wrong-species predictions in feature space -- and confirms the mechanism causally via single-species ablations. Section 5.6 is reserved for expert-calibrated filter results, which are held pending completed expert validation.

Throughout Section 5, five-fold cross-validation is the primary protocol for aggregate and per-species claims, following the rationale in Section 3.5. Multi-seed training on the fixed split is used for per-image analyses in Section 5.5 because each seed evaluates the same 2,362 test images. Single-split results are reported alongside the primary protocols as an honest record of the early experiments. All results use the best-validation-macro-F1 checkpoint.

### 5.1 Baseline

#### 5.1.1 Single-run classifier performance

Table 5.1 reports the ResNet-50 baseline on the fixed 70/15/15 split with 10,000-iteration bootstrap 95% confidence intervals. Overall accuracy reaches 88.2% and macro F1 reaches 0.815, figures driven by the eleven head-and-moderate species with n >= 200 training images. The three rare targets fall substantially below this aggregate: B. ashtoni reaches F1 0.500 (n = 6 test, 95% CI [0.000, 0.818]), B. sandersoni 0.588 (n = 10, [0.222, 0.833]), and B. flavidus 0.623 (n = 36, [0.462, 0.754]). The B. ashtoni interval spans more than 0.8 of the F1 range -- an honest reflection of evaluation variance at n = 6 that no single-run comparison can narrow.

*Table 5.1: Baseline ResNet-50 classifier on the fixed split, f1 checkpoint (10,000-iteration bootstrap 95% CI). Rare species in bold.*

| Species | Train n | Test n | Precision | Recall | F1 | 95% CI |
|---------|---------|--------|-----------|--------|-----|--------|
| **B. ashtoni** | **22** | **6** | 0.500 | 0.500 | **0.500** | [0.000, 0.818] |
| **B. sandersoni** | **40** | **10** | 0.714 | 0.500 | **0.588** | [0.222, 0.833] |
| **B. flavidus** | **162** | **36** | 0.760 | 0.528 | **0.623** | [0.462, 0.754] |
| Macro average | -- | 2,362 | -- | -- | 0.815 | [0.774, 0.845] |
| Overall accuracy | -- | 2,362 | -- | -- | 0.882 | -- |

Per-species F1 correlates cleanly with training-set size: every species with n >= 200 exceeds 0.75 F1, while every species below n = 200 falls under 0.65. The three rare species therefore define the augmentation target and determine every subsequent comparison.

#### 5.1.2 Rare-species confusion structure

The row-normalized baseline confusion matrix (Figure 5.1) shows three qualitatively different rare-species error patterns. B. sandersoni produces five errors over its ten test images, four of which are predicted as B. vagans (40% of test support, 80% of errors); the remaining error is B. rufocinctus. The vagans confusion direction is therefore singular and dominant for sandersoni, and reflects their shared yellow-anterior / black-posterior body pattern. B. flavidus produces seventeen errors over thirty-six test images and these spread across seven distinct species: B. rufocinctus (5 errors, 13.9%), B. ternarius (3, 8.3%), B. citrinus (3, 8.3%), B. ashtoni (2), B. bimaculatus (2), B. griseocollis (1), and B. terricola (1). No single confuser dominates; the misclassifications cluster among medium-sized yellow-and-black species in the genus, consistent with the diffuse-yellow phenotype that motivated the prompt-engineering work in Section 4.2. B. ashtoni produces three errors over six test images, each going to a different species: B. pensylvanicus, B. terricola, and B. flavidus. With n = 6 the confusion structure for ashtoni is statistically uninformative -- a single image flipping to a different predicted class would change the apparent dominant confuser entirely. These three patterns -- a single dominant confuser (sandersoni vs. vagans), diffuse confusion across many species (flavidus), and statistically-undetermined confusion (ashtoni) -- provide the visual priors against which any synthetic augmentation must succeed and reappear as the anchor for the failure-chain retrievals in Section 5.5.

![Baseline confusion matrix](../RESULTS_kfold/baseline@f1_confusion_matrix.png)
*Figure 5.1: Row-normalized baseline confusion matrix on the fixed split (RESULTS_kfold/baseline@f1, macro F1 0.815). Bold row labels mark the three rare augmentation targets.*

### 5.2 Latent-Space Analysis

This section characterises the real-image feature geometry that any augmentation strategy must respect, then shows that synthetics for the three rare species occupy an embedding-space region systematically offset from the corresponding real clusters. The analysis uses BioCLIP ViT-B/16 (Stevens et al., 2024) rather than DINOv2 ViT-L/14 (Oquab et al., 2024) because BioCLIP's biology-specific pre-training produces substantially better species-level separation on the training data, as established by a nearest-neighbour purity diagnostic in Section 5.2.1.

#### 5.2.1 Backbone selection

To choose the diagnostic embedding backbone, I compute 5-nearest-neighbour leave-one-out species classification accuracy on all 10,933 real training images under both backbones, using cosine similarity in L2-normalized CLS-token space. Table 5.2 reports overall and per-tier accuracy. BioCLIP achieves 0.657 overall and 0.125 on the rare tier; DINOv2 achieves 0.295 and 0.072. The gap is large and consistent across tiers -- BioCLIP's representation space is more aligned with species identity than general-purpose vision features, and the choice is made on these data rather than on prior literature. While BioCLIP's feature space is not identical to the ResNet-50 classifier's, its superior species-level structure makes it the most informative available proxy for diagnostic quality. A ResNet-50 penultimate-layer probe is listed in Section 7.2 as future work.

*Table 5.2: 5-NN leave-one-out species classification accuracy on real training images (10,933 images, 16 species, cosine metric).*

| Backbone | Dim | Overall | Rare (3 spp) | Moderate (7 spp) | Common (6 spp) |
|----------|----:|--------:|-------------:|-----------------:|---------------:|
| DINOv2 ViT-L/14 (518^2) | 1,024 | 0.295 | 0.072 | 0.133 | 0.273 |
| BioCLIP ViT-B/16 (224^2) | 512 | **0.657** | **0.125** | **0.468** | **0.614** |

#### 5.2.2 Real-image feature geometry

Figure 5.2a projects all 10,933 real training images into a BioCLIP t-SNE. Common species form compact, well-separated clusters; B. impatiens, B. ternarius, and B. griseocollis are clearly resolved. The rare species do not enjoy this separation. Figure 5.2b isolates the three rare targets together with two confusers chosen from Section 5.1.2 -- B. vagans (sandersoni's dominant confuser) and B. citrinus (one of flavidus's principal confusers). The plot shows that B. sandersoni and B. vagans occupy overlapping regions, consistent with the four-of-five sandersoni errors that go to vagans, and that the three rare species themselves occupy partially overlapping territory rather than three crisply separated clusters. This indistinguishability pre-exists any synthetic-augmentation question -- it is the native problem augmentation must address.

![16-species real-image t-SNE](plots/embeddings/bioclip_tsne/embeddings_overview.png)
*Figure 5.2a: BioCLIP t-SNE of 10,933 real training images, 16 species. Common species form compact clusters; rare species overlap with their visual confusers.*

![Rare-species real-image t-SNE](plots/embeddings/bioclip_tsne/embeddings_rare_real_only.png)
*Figure 5.2b: Rare-species real images alongside two visual confusers identified in Section 5.1.2 (B. vagans, B. citrinus). Baseline embedding-space overlap is apparent before any augmentation.*

#### 5.2.3 The synthetic-real embedding gap

Projecting real and synthetic images for the three rare species into a shared t-SNE space (Figure 5.3a) reveals the central structural finding of Section 5.2: synthetic images of each rare species form their own tight cluster, well-separated from the corresponding real cluster in the same projection. The gap is not a generic "synthetic != real" artefact; it is a per-species manifold offset.

Figure 5.3b quantifies this. For each synthetic image I compute its cosine distance to the centroid of its target species' real training embeddings. The median synthetic-to-centroid distance is 0.31 for B. ashtoni, 0.25 for B. sandersoni, and 0.32 for B. flavidus. The corresponding real-to-centroid distances within each rare species fall in the 0.10--0.20 range -- roughly half. The synthetic sub-manifold is therefore systematically carved out at a few tenths of cosine-space removed from where real rare bees actually sit. A representative confusion-pair triplet and the embedding atlas with thumbnails at true t-SNE coordinates (Appendix F) confirm visually that the clusters are pose- and coloration-coherent rather than projection artefacts.

![Rare real + synthetic t-SNE](plots/embeddings/bioclip_tsne/embeddings_rare_real_synth.png)
*Figure 5.3a: BioCLIP t-SNE of rare-species real and synthetic images. Synthetic clusters sit offset from the corresponding real clusters for each species.*

![Synthetic-to-centroid cosine distance](plots/embeddings/bioclip_tsne/embeddings_centroid_distance.png)
*Figure 5.3b: Per-synthetic cosine distance to the species' real-image centroid. Dashed lines mark real-to-centroid medians for comparison.*

This feature-space offset has a direct downstream prediction: training on these synthetics teaches the classifier a set of species-discriminative features that are offset from the feature subspace real test images of the same species occupy. Section 5.3 tests that prediction on macro F1, and Section 5.5 traces the per-image consequences.

### 5.3 Augmentation Method Comparison

#### 5.3.1 Aggregate and tier-level effects

Table 5.3 reports macro F1 across the three protocols defined in Section 3.5, the rare-tier F1 under the two multi-image protocols, and the moderate- and common-tier F1 under 5-fold CV. Two facts dominate the table. First, augmentation effects are concentrated in the rare tier: moderate and common tiers move by <= 0.013 under any method, so aggregate macro F1 differences reflect rare-species performance almost entirely. Second, the three protocols produce different aggregate rankings -- D5 is best on single-split (0.834), D3 is best on 5-fold CV (0.837), and D1 is best on multi-seed (0.839). The rankings are not contradictory: multi-seed and single-split share the same 6 / 10 / 36 rare test images, so flipping one or two correctly-classified rare images is enough to swap the aggregate ranking; 5-fold CV pools roughly five times more rare test predictions per species and is the more reliable reading.

*Table 5.3: Macro F1 across augmentation strategies and three protocols, with tier-mean F1 under the multi-image protocols (f1 checkpoint). Rare-tier figures are unweighted species-means within tier; bracketed values are per-tier deltas vs. baseline. Best per row in bold.*

| Quantity | Protocol | D1 Baseline | D3 CNP | D4 Synthetic | D5 LLM-filt. |
|----------|----------|------------:|-------:|-------------:|-------------:|
| Macro F1 | Single-split | 0.815 | 0.829 | 0.823 | **0.834** |
| Macro F1 | 5-fold CV | 0.832 +/- 0.013 | **0.837 +/- 0.013** | 0.820 +/- 0.024 | 0.821 +/- 0.019 |
| Macro F1 | Multi-seed | **0.839 +/- 0.006** | 0.822 +/- 0.014 | 0.828 +/- 0.009 | 0.831 +/- 0.008 |
| Rare-tier F1 | 5-fold CV | 0.611 | **0.641** (+0.030) | 0.555 (-0.056) | 0.570 (-0.041) |
| Rare-tier F1 | Multi-seed | **0.665** | 0.593 (-0.073) | 0.590 (-0.075) | 0.617 (-0.048) |
| Moderate-tier F1 | 5-fold CV | 0.861 | 0.862 | 0.860 | 0.855 |
| Common-tier F1 | 5-fold CV | 0.908 | 0.906 | 0.906 | 0.907 |

Under 5-fold CV, copy-and-paste augmentation lifts rare-tier F1 by 0.030 above baseline, while both synthetic variants reduce it (D4 by 0.056, D5 by 0.041). Under multi-seed, all three augmentation methods reduce rare-tier F1, with synthetic variants again worst. The two protocols disagree on whether D3 helps but agree on the signed effect of D4 and D5: they harm the rare tier under every protocol considered.

#### 5.3.2 Per-species effects and statistical significance

Pairwise paired t-tests on fold-level macro F1 (Table 5.4) identify two comparisons that clear the high-power bar imposed by df = 4. D5 is significantly worse than the baseline (p = 0.041) and significantly worse than D3 (p = 0.030). D4 and D5 are statistically indistinguishable (p = 0.777): the strict LLM filter, applied to the D4 pool, produces no measurable improvement over no filter at all. This non-result is the central empirical motivation for the failure-mode analysis in Section 5.5.

*Table 5.4: Pairwise paired t-tests on fold-level macro F1 (5-fold CV, df = 4). Significant results in bold.*

| Comparison | Mean delta | t | p | Significant |
|------------|-----------:|----:|----:|-------------|
| D1 vs D3 CNP | +0.005 | 1.72 | 0.161 | No |
| D1 vs D4 Synthetic | -0.012 | -2.17 | 0.096 | No |
| **D1 vs D5 LLM-filtered** | **-0.011** | **-2.98** | **0.041** | **Yes -- D5 worse** |
| D3 vs D4 Synthetic | -0.017 | -2.59 | 0.061 | No |
| **D3 vs D5 LLM-filtered** | **-0.016** | **-3.29** | **0.030** | **Yes -- D5 worse** |
| D4 vs D5 LLM-filtered | +0.001 | 0.30 | 0.777 | No (filter no benefit) |

Per-species analysis locates the signal. D3 significantly improves B. flavidus F1 over baseline under 5-fold CV (+0.059, p = 0.005), the only statistically significant per-species gain from any augmentation method. D4 reduces B. sandersoni F1 by 0.140 under 5-fold CV (from 0.466 to 0.326, marginally significant at p = 0.052), and D5 reduces it by 0.068. B. ashtoni produces large numerical swings (D3 +0.064, D4 -0.045, D5 -0.044 under 5-fold CV) but with 95% CIs of width 0.27--0.30 due to n = 32 pooled test images, none of these per-species comparisons reach significance. Figure 5.4 plots per-species delta F1 for both protocols side-by-side: rare-species rows are coloured saturated under D4 and D5, while moderate and common rows are essentially flat.

![Per-species delta F1, 5-fold and multi-seed](plots/failure/species_f1_delta_kfold.png)
*Figure 5.4: Per-species F1 change relative to D1 baseline under 5-fold CV (primary) and multi-seed (fixed split). Rare species highlighted; moderate and common tiers show negligible effects under any method.*

#### 5.3.3 Volume ablation

A natural concern is whether the D4 and D5 rare-species harm would resolve at higher synthetic volumes. Figure 5.5 reports macro F1 and rare-species F1 under D4 and D5 at volumes of +50, +100, +200, +300, and +500 images per rare species (single-split evaluation). Neither variant shows a coherent volume--performance trend: D4 macro F1 fluctuates between 0.820 and 0.834 across volumes without a monotone direction, and D5 peaks at +200 but does not maintain the gain at +300 or +500. At +500, both variants regress on B. sandersoni (D4 0.471, D5 0.556) as the synthetic-to-real ratio reaches 12.5:1 for that species. The absence of a volume-dependent improvement establishes that the bottleneck is generation fidelity, not quantity -- adding more synthetics of the same quality does not close the embedding-space gap that Section 5.2 identified. A companion background-removal diagnostic (synthetic images evaluated on white backgrounds; full details in Appendix E) confirms the same conclusion from the opposite direction: removing the background does not alter the strict pass rate but increases the wrong-coloration rate by 55%, pinpointing specimen coloration rather than background confusion as the generation bottleneck.

![Volume ablation](../RESULTS_count_ablation/volume_ablation_trends_with_ci.png)
*Figure 5.5: Volume ablation for D4 and D5 at +50 to +500 synthetic images per rare species (single-split evaluation). No consistent improvement at any volume for either variant.*

### 5.4 LLM-as-Judge Results

The LLM-as-judge (Section 4.3) evaluates every generated image on species-level morphology. Section 5.4 characterises what the judge sees. Section 5.5 returns to whether what the judge sees matches what the classifier needs.

#### 5.4.1 Pass rates and filter funnel

Table 5.5 reports the two-stage judge's output over all 1,500 generated images (500 per rare species). Blind-identification match rates are high for B. sandersoni (96.4%) and B. flavidus (96.0%) but much lower for B. ashtoni (76.0%), reflecting the inverse-phenotype challenge: ashtoni's predominantly black thorax inverts the dominant Bombus prior, and the judge -- like the generation model -- sometimes defaults to yellow-thorax interpretations. The mean morphological score follows the same pattern (ashtoni 3.82, flavidus 4.06, sandersoni 4.37), as does the strict pass rate: 44.4% ashtoni, 57.6% flavidus, 91.2% sandersoni. The strict funnel reduces the 1,500-image pool through three sequential gates -- blind-ID match (1,342 images, 89.5%), diagnostic completeness at species level (1,060, 70.7%), and mean morphological score >= 4.0 (966, 64.4%).

*Table 5.5: LLM-as-judge evaluation of 1,500 generated images (500 per species). Strict pass requires blind-ID match AND diagnostic = species AND mean morphological score >= 4.0.*

| Species | Blind ID | Mean morph | Lenient pass | Strict pass |
|---------|---------:|-----------:|-------------:|------------:|
| B. ashtoni | 76.0% | 3.82 | 92.0% | 222 / 500 (44.4%) |
| B. flavidus | 96.0% | 4.06 | 99.6% | 288 / 500 (57.6%) |
| B. sandersoni | 96.4% | 4.37 | 100% | 456 / 500 (91.2%) |

#### 5.4.2 Per-feature diagnostics

The bottleneck is narrow and localised. The mean per-feature morphological scores hold above 4.0 for fourteen of the fifteen (species x feature) cells; the lone exception is B. ashtoni's thorax coloration mean of 2.98 -- far below every other cell. Wrong-coloration is the dominant failure mode overall (27.1% of all 1,500 images), concentrated in B. ashtoni. Structural failure modes -- extra or missing limbs, impossible geometry, visible artefacts, repetitive patterns -- register at exactly 0 across all 1,500 images, confirming that the structured-prompting framework of Section 4.2 has eliminated this failure class. The residual gap is a colour-fidelity gap, and it is concentrated in the species that deviates most from the genus-typical phenotype. Per-angle and per-caste breakdowns -- including the frontal-view paradox (high morph mean but low strict-pass rate driven by abdomen-banding occlusion) and the male-caste deficit in B. ashtoni (30.9% caste-correct vs. 84.3% female) -- are reported in Appendix E. They confirm the judge's scoring behaves coherently across view conditions and do not change the filter-calibration argument that follows.

#### 5.4.3 What the judge measures -- and what it does not

The judge measures species-level morphological fidelity as assessed by a vision-language model on a human-visual rubric, and Table 5.5's funnel shows the measurement is informative: the judge correctly identifies the ashtoni generation bottleneck, passes sandersoni at near-ceiling, and scales morph scores with generation difficulty. What the judge does not measure is whether a synthetic image is useful for the downstream classifier. Section 5.3 has already established that D4 and D5 are statistically indistinguishable on macro F1 (p = 0.777): removing the 27.1% wrong-coloration images the judge flags does not rescue rare-species F1. Section 5.5 traces the reason -- the judge's pass set still contains many images that the classifier's BioCLIP feature space places beyond the real species distribution.

### 5.5 Failure Mode Analysis

Sections 5.3 and 5.4 establish *that* synthetic augmentation harms rare species and *that* the LLM filter does not fix the harm. Section 5.5 addresses *why*, combining per-image prediction tracking, embedding-space failure-chain retrieval, judge--classifier disagreement, and a causal ablation of each species' synthetic contribution. The analysis here uses the multi-seed protocol because per-image flip and chain analyses require every seed to evaluate the same images.

#### 5.5.1 Per-image prediction flips

Each of the 2,362 test images produces 20 predictions across the four configs and five seeds. Collapsing within each config by majority vote yields one verdict per (image, config) pair, and comparing each augmented config against the baseline partitions test images into four cells: stable-correct (both right), stable-wrong (both wrong), improved (augmentation rescued a baseline error), and harmed (augmentation broke a baseline-correct image).

*Table 5.6: Rare-species flip counts under each augmentation method (multi-seed majority vote, fixed split). No rare image is improved by any method.*

| Species | n test | D3 (impr / harm) | D4 (impr / harm) | D5 (impr / harm) |
|---------|-------:|-------------------|-------------------|-------------------|
| B. ashtoni | 6 | 0 / 1 | 0 / 0 | 0 / 1 |
| B. sandersoni | 10 | 0 / 1 | 0 / 1 | 0 / 1 |
| B. flavidus | 36 | 0 / 5 | 0 / 8 | 0 / 6 |

The pattern is stark: no rare-species test image is improved by any augmentation method. The effect is not a mixture of improvements and harms that averages out poorly -- it is one-directional harm. Expressed as a rate, B. flavidus is harmed on 22.2% of its D4 test images, the largest cell in any (species x method) pair, while no common species exceeds a 3% harm rate. Figure 5.6 visualises the species-by-method harm rates; the rare tier carries the signal and the remaining tiers are near zero.

![Flip-category heatmap](plots/failure/flip_category_heatmap.png)
*Figure 5.6: Flip-category rates by species and augmentation method (multi-seed majority vote, fixed split). Rare rows carry the signal; moderate and common rows are near zero.*

#### 5.5.2 Embedding-space failure chains

For every rare-species test image harmed under D4 or D5, I retrieve the five nearest training synthetics of the corresponding variant by BioCLIP cosine similarity, restricted to that variant's actual training pool (600 synthetics per variant). Each retrieved synthetic carries its generated-species label and LLM tier. The aggregate retrieval statistic is informative: across all 49 D4 harmed chains, the median test-to-5-NN cosine similarity is 0.56, well below the 0.7+ range typical of two real images of the same species. The "nearest training synthetic" is therefore not close to the harmed test image in absolute terms; it is close only relative to the rest of the synthetic pool. Figure 5.7 shows one such D4 chain for a harmed B. flavidus test image (`Bombus_flavidus4512075898`, baseline correct, D4 prediction B. griseocollis); the chain's nearest training neighbours include synthetics generated for species other than flavidus, consistent with the per-image prediction patterns of Section 5.5.1: the eight D4-harmed flavidus test images are predicted as B. citrinus (3), B. rufocinctus (2), B. ashtoni (2), and B. griseocollis (1) -- never as the correct B. flavidus, and never as a single dominant alternative. Full harmed and improved chain galleries, plus their t-SNE projections, are in Appendix F.

![Representative D4 failure chain](plots/failure/chains_d4_harmed/gallery/flavidus__Bombus_flavidus4512075898.png)
*Figure 5.7: Representative D4 failure chain. Top row: a harmed B. flavidus test image (baseline correct, D4 misclassified as B. griseocollis) followed by its five nearest D4 training synthetics, ranked by BioCLIP cosine similarity.*

#### 5.5.3 Judge-classifier disagreement

The failure chains identify which synthetics the classifier latches onto; Figure 5.8 shows that many of these synthetics passed the LLM filter. For every synthetic I plot LLM mean morphological score against BioCLIP cosine distance to the correct species' real centroid, and partition the plane at per-species medians on both axes. The upper-right quadrant -- "LLM passes above median morph AND far from the real centroid" -- contains 138 B. ashtoni, 49 B. sandersoni, and 108 B. flavidus synthetics. These are exactly the images the judge cannot reject but the classifier's feature space rejects. They pass through the D4 -> D5 filter unchanged, providing a mechanistic explanation for the D4 vs. D5 p = 0.777 result in Section 5.3.2.

![Judge versus centroid distance](plots/failure/llm_vs_centroid_quadrant.png)
*Figure 5.8: Per-synthetic LLM mean morphological score versus BioCLIP cosine distance to the correct species' real centroid. Dashed lines mark per-species medians. The upper-right quadrant counts disagreement between judge-relevant and classifier-relevant quality.*

#### 5.5.4 Causal attribution via subset ablation

To confirm the synthetic sub-manifold is causally responsible for the harm -- rather than merely correlated with it -- I run six additional training jobs (seed 42 only) that each drop all synthetic images of exactly one rare species from D4 or D5. Recovery is defined as F1 under ablation minus F1 under the full variant: positive recovery implies the removed synthetics were collectively harming the target species; negative recovery implies they were collectively helping.

*Table 5.7: Own-species F1 recovery under single-species subset ablation (seed 42). Threshold |delta| > 0.02 for a directional label; otherwise neutral.*

| Variant | Dropped species | F1 full -> ablated | Recovery | Label |
|---------|-----------------|---------------------:|---------:|-------|
| D4 | ashtoni | 0.545 -> 0.727 | **+0.182** | harmful |
| D4 | sandersoni | 0.571 -> 0.476 | -0.095 | helpful |
| D4 | flavidus | 0.645 -> 0.708 | +0.062 | harmful |
| D5 | ashtoni | 0.727 -> 0.727 | +0.000 | neutral |
| D5 | sandersoni | 0.625 -> 0.706 | **+0.081** | harmful |
| D5 | flavidus | 0.725 -> 0.719 | -0.006 | neutral |

Two patterns dominate. First, the LLM filter neutralises the large B. ashtoni harm seen in D4 (+0.182 recovery converts to +0.000 in D5): the filter does remove genuinely bad ashtoni synthetics. Second, *the filter reverses the B. sandersoni effect.* Unfiltered sandersoni synthetics were collectively helpful in D4 (removing them lost 0.095 F1); filtered sandersoni synthetics are collectively harmful in D5 (removing them gains 0.081 F1). Because B. sandersoni has the highest LLM strict-pass rate of the three species (91.2%), the filter retains nearly all images and the 8.8% it discards includes the signal that was compensating for the harmful subset. This is the clearest empirical demonstration of LLM-judge miscalibration in the dataset: for the species the filter passes most easily, it discards exactly the wrong subset.

Cross-species collateral effects are substantial and reinforce the embedding-space picture. Dropping B. ashtoni synthetics in D4 simultaneously reduces B. sandersoni F1 by 0.150 and increases B. flavidus F1 by 0.124 -- the same image set provides conflicting gradient signal across class decisions, consistent with the failure-chain finding that retrieved nearest synthetics frequently belong to the wrong species. Figure 5.9 shows the full 3 x 3 recovery matrix alongside the own-species recovery bars.

![Subset ablation recovery](plots/failure/subset_ablation_recovery.png)
*Figure 5.9: Subset ablation recovery. Left: own-species F1 recovery under D4 and D5 by dropped species. Right: full dropped-versus-measured recovery matrix. Single seed (42); the ablation establishes direction, not magnitude.*

Each ablation cell is a single seed-42 run with rare-species test n between 6 and 36, so a 0.05 F1 change corresponds to flipping 0.3--1.8 images. The analysis is designed to establish direction of effect, not statistically-powered magnitude; the B. sandersoni D4 -> D5 sign reversal is a qualitative signal that stochastic noise cannot easily produce, while the cross-species collateral magnitudes should be read as directional only. Per-synthetic labels -- propagating the own-species verdicts to every generated image in each variant -- are written to `RESULTS/failure_analysis/synthetic_labels.csv` for use in Section 5.6.

### 5.6 Expert Calibration Results

Section 4.4 specified a learned filter trained on entomologist annotations over a stratified 150-image sample, compared against the LLM-rule and BioCLIP-centroid-distance baselines. Downstream results for the D6-probe and D6-centroid variants, leave-one-out AUC-ROC for each filter on the expert sample, CKNNA alignment between each variant and the real-image distribution, and the D6 row added to Table 5.3 depend on completed expert validation of the annotation sample. This section is held pending those annotations.


## 6. Discussion

Section 6 interprets the empirical findings of Section 5 along five lines. The augmentation effect is concentrated almost entirely in the rare tier (Section 6.1). The harm in that tier is mechanistically attributable to a feature-space gap between synthetic and real images that pre-exists training and whose causal contribution is confirmed by single-species ablation (Section 6.2). Copy-and-paste augmentation outperforms generative augmentation precisely because it preserves the real-image manifold (Section 6.3). The LLM judge's per-feature scores carry useful signal but its filter rule is miscalibrated against classifier-relevant quality, with the B. sandersoni D4 -> D5 sign reversal the clearest empirical demonstration (Section 6.4). The remaining limitations -- evaluation power for rare species, single-seed ablation, BioCLIP as a feature-space proxy -- are honest constraints that do not invalidate the core findings (Sections 6.5 and 6.7).

### 6.1 Augmentation harm is concentrated in the rare tier

The most load-bearing structural finding of Section 5 is that augmentation effects are not distributed uniformly across species -- they are concentrated in the tier with the fewest real training images. Moderate-tier (n in [200, 900]) and common-tier (n > 900) F1 move by no more than 0.013 under any augmentation method or evaluation protocol (Table 5.3); aggregate macro F1 differences therefore reflect rare-tier performance almost entirely. Within the rare tier, both synthetic variants reduce F1 under every protocol considered: -0.075 for D4 and -0.048 for D5 under multi-seed; -0.056 and -0.041 under 5-fold CV. Copy-and-paste (D3) is the only method with a positive rare-tier signal under any protocol (+0.030 under 5-fold CV; ambiguous under multi-seed), and it is also the only method with a statistically significant per-species gain (B. flavidus +0.059, p = 0.005).

This is the opposite of the hypothesis that motivated synthetic augmentation in the first place: the intervention designed to redress class imbalance turns out to be the intervention least likely to help the tier that most needs help. The non-effect on common species is itself diagnostic. If synthetic augmentation introduced a generic distribution shift, common species would also degrade. The fact that they do not means the harm is specific to species whose real-image support is small enough that the synthetic distribution can outweigh it during fine-tuning.

### 6.2 A mechanistic account of synthetic-augmentation harm

Section 5 assembles four pieces of evidence that jointly explain why synthetic augmentation harms rare species rather than helping them.

First, no rare-species test image is improved by any augmentation method under multi-seed majority vote (Table 5.6). The effect is not a mixture of improvements and harms that averages to a small negative; it is one-directional harm. B. flavidus is harmed on 22% of its D4 test images, the largest cell in any (species x method) pair, while no common species exceeds a 3% harm rate.

Second, synthetic images of each rare species occupy a sub-manifold systematically offset from the corresponding real cluster in BioCLIP feature space (Section 5.2.3). Median synthetic-to-real-centroid cosine distance is 0.25--0.32 for the three species, against real-to-real-centroid distances of 0.10--0.20. The offset is per-species rather than a generic synthetic-to-real artefact: the synthetic clusters do not collapse into a single "synthetic" region of feature space but carve out their own per-species sub-manifolds, each a few tenths of cosine-space removed from where real images of the same species sit.

Third, when a harmed test image is matched to its five nearest training synthetics in BioCLIP space (Section 5.5.2), the median test-to-neighbour cosine similarity is 0.56 -- well below the 0.7+ range typical of two real images of the same species. The classifier is therefore trained on synthetics that, while close in absolute terms to the harmed test image relative to the rest of the synthetic pool, are not in fact close to the real test distribution. The classifier learns species-discriminative features off-manifold and applies them on-manifold at test time.

Fourth, single-species subset ablation (Section 5.5.4, Table 5.7) converts these correlational claims into causal ones. Removing B. ashtoni's synthetics from D4 recovers ashtoni F1 by 0.182; removing B. flavidus's recovers flavidus by 0.062; both interventions confirm that the synthetics in question were collectively harming the target species. Cross-species collateral magnifies the effect: dropping ashtoni's synthetics also reduces sandersoni F1 by 0.150 and increases flavidus F1 by 0.124 in the same run, indicating that the offset synthetic sub-manifold affects multiple class decisions simultaneously through shared features at the classifier's penultimate layer.

The four pieces of evidence are mutually reinforcing rather than redundant. The embedding-space gap predicts the failure-chain pattern; the failure-chain pattern predicts the per-image flip pattern; the subset ablation confirms that removing the offset synthetic mass recovers performance. Volume ablation (Section 5.3.3) closes the loop by ruling out the most natural alternative explanation: more synthetics of the same quality do not close the gap, because the bottleneck is fidelity, not quantity.

### 6.3 Copy-and-paste vs. generative: manifold preservation as the operative variable

CNP (D3) is the only augmentation method in the study that produces a statistically significant per-species F1 gain (B. flavidus +0.059, p = 0.005) and the only method that lifts rare-tier F1 above baseline under 5-fold CV (+0.030). The mechanism is consistent with Section 6.2: CNP preserves real morphological texture by segmenting real bees and compositing them onto real flower backgrounds, so its outputs sit inside the real-image feature manifold rather than on an offset synthetic sub-manifold. The cost is a diversity ceiling: with only 22 real training images of B. ashtoni, CNP can produce new compositions but cannot introduce intra-class variation in pose, lighting, or morphology beyond what the source images contain. This explains why CNP's largest gain is for the rare species with the most source images (B. flavidus, n = 162) and is essentially flat for the species with the fewest (B. ashtoni, n = 22).

The trade-off is not "fidelity vs. diversity" in the abstract -- it is specifically that generative augmentation introduces a feature-space manifold gap that exceeds the model's local generalisation budget during fine-tuning, while CNP does not. A future generative pipeline that preserves the real manifold (for instance, fine-tuning a diffusion model on real specimens via LoRA, or composite generation conditioned on real-image patches) could in principle deliver both diversity and fidelity. The current closed-source pipeline does not.

### 6.4 LLM judges: useful per-feature signals, miscalibrated filter rule

The LLM judge's per-feature scores are genuinely informative. The strict pass rate scales with generation difficulty (ashtoni 44.4%, flavidus 57.6%, sandersoni 91.2%) and the per-feature decomposition pinpoints the residual coloration-fidelity bottleneck (B. ashtoni thorax coloration mean 2.98, every other (species x feature) cell >= 4.0). This is precisely the diagnostic granularity that motivates expert-calibrated filtering: the raw signal needed to learn a quality model exists; what is missing is the calibration between language-mediated morphology scoring and classifier-relevant feature space.

The miscalibration manifests in two complementary ways. First, the judge-versus-centroid disagreement quadrant (Section 5.5.3, Figure 5.8) contains 138 ashtoni, 49 sandersoni, and 108 flavidus synthetics that the judge passes confidently above its median morph score yet the classifier's BioCLIP feature space places beyond the real species distribution. These images survive the D4 -> D5 filter unchanged and provide a mechanistic explanation for the D4-vs-D5 statistical equivalence (p = 0.777, Section 5.3.2): the filter is blind to exactly the dimension on which downstream quality varies.

Second, the subset-ablation reversal for B. sandersoni (Section 5.5.4) sharpens this point into a single empirical demonstration. In D4, B. sandersoni's full synthetic set is collectively helpful: removing it loses 0.095 F1. In D5, the LLM-filtered B. sandersoni subset is collectively harmful: removing it gains 0.081 F1. Because B. sandersoni has the highest LLM strict-pass rate (91.2%), the filter retains nearly every synthetic and the small fraction it discards turns out to include exactly the gradient signal that was compensating for the harmful subset. The per-species LLM rule discards the wrong subset for the species it passes most easily -- a result that no holistic morph-mean threshold could anticipate. A defensible quality filter must therefore use signals beyond the LLM's per-feature scores, plausibly the BioCLIP centroid distance of Section 5.5.3 or the expert labels developed in Section 4.4.

### 6.5 Statistical challenges with rare species

Evaluating augmentation strategies for rare species creates a fundamental tension: the species that most need augmentation are precisely those for which evaluation is least reliable. With n = 6 test images for B. ashtoni on the fixed split, a single flipped prediction changes F1 by up to 0.17 and reorders the aggregate ranking across augmentation methods. The dual-protocol design adopted in Section 3.5 partially addresses this: 5-fold cross-validation pools predictions to reach rare-species effective n = 32 / 58 / 232, supporting the only statistically significant pairwise comparisons in the study (D1 vs. D5 p = 0.041; D3 vs. D5 p = 0.030; D3 flavidus +0.059 p = 0.005); multi-seed training on the fixed split keeps the same test images across seeds, supporting the per-image flip and chain analyses that the cross-fold protocol cannot. The protocols are reconciled in Section 5.3.1: their disagreement on aggregate ranking reflects the fixed-split test-set composition, not a methodological inconsistency.

The remaining residual is power. With df = 4, paired t-tests demand large effect sizes for 80% power at alpha = 0.05; non-significant comparisons such as D1 vs. D3 (p = 0.161) and D1 vs. D4 (p = 0.096) are underpowered rather than null. Bootstrap confidence intervals -- reported alongside every point estimate -- are the more informative uncertainty quantification for rare-species F1, and the prose in Section 5 follows that priority.

### 6.6 Implications for urban biodiversity monitoring

The practical goal of automated pollinator monitoring is to reduce false-absence errors for rare species: a classifier that misclassifies B. ashtoni or B. sandersoni produces a false negative for that species, and if the species is reported as absent, planners may approve development on habitat where it still persists. Section 5 shows that current generative augmentation pipelines move this risk in the wrong direction: D4 and D5 both reduce rare-species F1 below baseline, and the LLM filter does not rescue the harm. The asymmetry matters operationally because the cost of a false negative for a declining species exceeds the cost of a false positive: an erroneous "ashtoni present" record can be corrected by a follow-up survey, but an erroneous "ashtoni absent" record removes the species from a planning decision entirely.

Until the synthetic-real feature-space gap is closed, the deployment recommendation is to prefer copy-and-paste augmentation where source images permit (Section 6.3) and to treat generative augmentation as conditional on a quality filter that uses classifier-relevant signals (centroid distance, learned expert filter), not the LLM judge's per-feature scores alone. The expert-calibration pipeline of Section 4.4 -- whose results are deferred to Section 5.6 pending annotation -- is the natural next step toward such a filter.

### 6.7 Limitations

Several limitations constrain the scope of these conclusions and should be read alongside the corresponding sections of Section 5.

- **BioCLIP feature space as a diagnostic proxy.** The embedding analyses in Section 5.2 and the failure-chain retrievals in Section 5.5.2 use BioCLIP rather than the ResNet-50 classifier's penultimate-layer features. BioCLIP was selected on a 5-NN species-purity diagnostic that places it well above DINOv2 on this dataset (Table 5.2), but its representation is not the classifier's. A ResNet-50 penultimate-layer probe is listed as future work in Section 7.2 to test whether the same feature-space gap is observed in the classifier's own representation.

- **Single-seed subset ablation.** The causal attribution in Section 5.5.4 is run at seed 42 only. With rare-species test n between 6 and 36, a 0.05 F1 change corresponds to flipping 0.3--1.8 images, so the analysis is designed to establish direction of effect rather than statistically powered magnitude. The cross-species collateral magnitudes in particular should be read as directional. The B. sandersoni D4 -> D5 sign reversal is robust to this caveat because direction reversal is qualitative.

- **Statistical power for rare-species t-tests.** With df = 4 for the 5-fold paired t-tests and df = 4 for the multi-seed paired t-tests, only large effect sizes reach significance. Non-significant comparisons should be read as underpowered rather than as evidence of equivalence (Section 6.5).

- **Single classifier architecture.** All experiments use ResNet-50. Architectures with stronger biology-specific pre-training (a BioCLIP or DINOv2 backbone) might respond differently to synthetic augmentation, potentially because their feature spaces tolerate distribution shift better.

- **Single generative model.** GPT-image-1.5 via the `images.edit` endpoint is closed and non-fine-tunable. The observed synthetic-real gap may be specific to general-purpose models without domain adaptation; open-source diffusion models with LoRA fine-tuning on real specimens could in principle close the gap.

- **Three target species.** The mechanistic account is demonstrated for B. ashtoni, B. sandersoni, and B. flavidus. Generalisation to other long-tailed taxa is hypothesised but not tested. The morphological-atypicality argument (rare classes deviate most from the model's prior) suggests the same pattern in other fine-grained domains, but extending to additional species is future work.

- **Fixed augmentation volume of +200 per rare species.** All three rare species received the same number of synthetics rather than counts proportional to their training-set size or their generation difficulty. The volume ablation in Section 5.3.3 provides partial evidence that quality dominates volume, but a controlled ratio experiment would strengthen this conclusion.

- **Expert calibration deferred.** Section 5.6's filter-comparison results, including the D6 row of Table 5.3, depend on completed expert validation of the 150-image stratified annotation sample. The mechanistic argument in Section 6.2 stands without these results, but the central forward-looking claim of the thesis -- that an expert-calibrated filter can close the LLM-judge calibration gap identified in Section 6.4 -- awaits empirical confirmation.

- **No comparison with loss re-weighting.** This thesis does not compare against focal loss, class-balanced loss, or decoupled training. These approaches address class imbalance through the training objective rather than the data, and are complementary rather than competing; they are not expected to resolve the feature-space gap that is the focus of this work.


## 7. Conclusion and Future Work

### 7.1 Summary

This thesis investigated the synthetic-real gap for fine-grained biodiversity classification under extreme class imbalance. Four contributions were made.

First, a **structured morphological prompting framework** using negative constraints, tergite-level colour maps, and reference-guided generation eliminated all structural failures across 1,500 generated images and isolated coloration fidelity as the sole remaining generation bottleneck. Strict pass rates range from 44.4% (B. ashtoni) to 91.2% (B. sandersoni), tracking morphological deviation from the genus-typical phenotype.

Second, a **two-stage LLM-as-judge evaluation protocol** combining blind taxonomic identification with per-feature morphological scoring provided interpretable diagnostic signals that identify *where* generated images fail. The per-feature decomposition pinpoints the residual coloration-fidelity bottleneck (B. ashtoni thorax coloration mean 2.98, every other (species x feature) cell >= 4.0).

Third, an **expert-calibrated quality filtering pipeline** was designed to learn feature-level weights from entomologist annotations on a stratified 150-image subset, comparing the learned filter against an LLM-rule baseline and a BioCLIP centroid-distance baseline. Downstream D6 results are deferred to Section 5.6 pending completed expert validation.

Fourth, **systematic empirical comparison under five-fold cross-validation, multi-seed training, and per-image failure analysis** established that copy-and-paste augmentation (D3) produces the only statistically significant rare-species F1 gain in the study (B. flavidus +0.059, p = 0.005), while both unfiltered (D4) and LLM-filtered (D5) generative augmentation reduce rare-tier F1 below baseline; D5 is significantly worse than baseline (p = 0.041) and statistically indistinguishable from D4 (p = 0.777). The harm is mechanistically attributable to a per-species feature-space offset between synthetic and real images (BioCLIP synthetic-to-centroid cosine median 0.25--0.32 vs. real-to-centroid 0.10--0.20) and is confirmed causally by single-species subset ablation, including a B. sandersoni D4 -> D5 sign reversal that demonstrates the LLM filter discards the wrong subset for the species it passes most easily. These findings motivate expert-calibrated filtering -- not as an incremental refinement of the LLM rule, but as a replacement that uses classifier-relevant signals the LLM judge cannot access.

### 7.2 Future Work

- **Complete expert-calibrated filter (D6).** Train the learned filter on expert annotations and evaluate whether feature-level reweighting closes the remaining gap between D5 and baseline.
- **Latent-space filtering.** Evaluate whether DINOv2/BioCLIP embeddings capture quality signals that LLM scores miss, and whether incorporating visual features into the filter (Model C) outperforms LLM-score-only approaches (Model B).
- **Alternative classifier backbones.** BioCLIP and DINOv2 as classifier backbones (not just filter features) may respond differently to synthetic augmentation due to stronger pretrained representations for biological imagery.
- **Open-source generation.** Replace GPT-image-1.5 with fine-tunable open-source diffusion models (SDXL, Stable Diffusion 3) that can be adapted to specific species with LoRA, potentially closing the coloration fidelity gap.
- **Extended species coverage.** Apply the pipeline to additional rare taxa beyond the three target species, testing whether the morphology-specific gap generalizes.
- **Deployment integration.** Connect the augmentation pipeline to the Beecology Project's Sensing Garden system for real-time urban pollinator monitoring.


## Appendices

### Appendix A: Bumblebee Morphology and Field Guide
Full tergite-level color maps, caste variation tables, and reference photographs for B. ashtoni, B. sandersoni, and B. flavidus. Morphological comparison table.

### Appendix B: Full Prompt Template and Iteration History
Final prompt template (v10), prompt evolution table (v1--v10), environmental and viewpoint configuration.

### Appendix C: LLM Judge Prompt and Pydantic Schema
Complete system prompt, structured output schema, strict filter rules.

### Appendix D: Expert Calibration Results -- Detailed Analysis
Inter-annotator agreement tables, per-feature disagreement matrices, learned filter coefficients, ROC analysis, latent space visualizations.

### Appendix E: Per-Species Detailed Results
Confusion matrices per dataset version, volume ablation full table, per-seed breakdowns, complete 16-species results table, per-angle strict-pass-rate table, caste-fidelity breakdown (referenced in Section 5.4.2), background-removal diagnostic (referenced in Section 5.3.3).

### Appendix F: Failure-Mode Analysis Assets
Embedding atlases at true t-SNE coordinates (Section 5.2.3); confusion-pair triplets for ashtoni x {citrinus, vagans}, sandersoni x vagans, flavidus x citrinus (Section 5.2.3); full per-species galleries of real, synthetic, and harmed-test images (Section 5.5); 49 D4-harmed, 52 D4-improved, 49 D5-harmed, and 49 D5-improved failure chains with t-SNE projections (Section 5.5.2); full LLM-vs-centroid 4-quadrant counts per species (Section 5.5.3); full dropped-versus-measured 3 x 3 subset-ablation recovery matrix for D4 and D5 (Section 5.5.4).


---

## REFERENCES TO ADD (cited in text but missing from reference list)

1. Bjerge, K., et al. 2024. [48 camera traps, 10M insect images] — cited in Section 2.1
2. Bjerge, K., et al. 2023. [hierarchical classification for insects] — cited in Section 2.1
3. Colla, S.R., Richardson, L., and Williams, P. 2011. [bumblebee morphology guide] — cited in Section 3.3
4. Yu, K., et al. 2025. "GPT-ImgEval: A Comprehensive Benchmark for Diagnosing GPT4o in Image Generation." — cited in Section 4.2.3
5. Huh, M., Cheung, B., Wang, T., and Isola, P. 2024. "The Platonic Representation Hypothesis." ICML 2024. — cited in Section 4.4.3 (CKNNA metric)
6. Shipard, J., Wiliem, A., Thanh, K.N., Xiang, W., and Fookes, C. 2023. "Diversity is Definitely Needed: Improving Model-Agnostic Zero-shot Classification via Stable Diffusion." CVPR Workshops 2023. — cited in Section 3.5 (k-fold CV on small biological image datasets)
7. Picek, L., Sulc, M., Matas, J., et al. 2022. "Danish Fungi 2020 -- Not Just Another Image Recognition Dataset." WACV 2022. — cited in Section 3.5 (k-fold CV protocol for fine-grained biological classification)

## REFERENCES TO REMOVE (in list but never cited)

1. Beery et al. 2020 — replaced by Ghiasi et al. 2021
2. Richardson & Colla 2025 — never cited in text
