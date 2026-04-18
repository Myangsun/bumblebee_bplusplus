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

### 5.1 Baseline Analysis

Baseline classifier numbers and the confusion structure between rare species
and their visual confusers are described in Appendix A (unchanged from the
single-split analysis in earlier drafts). The three rare targets are
B. ashtoni (22 train / 6 test), B. sandersoni (40 / 10), and B. flavidus (162
/ 36); their baseline macro F1 values are 0.614, 0.622, and 0.760
respectively (5-seed mean, 95 % bootstrap CI in Appendix B). Baseline
confusion matrices establish three recurring confusion directions:
ashtoni → {citrinus, vagans}, sandersoni → vagans, and flavidus → citrinus.
These confuser identities anchor the Task-1 triplet analysis in § 5.4.

### 5.2 Generation Quality

Synthetic-image generation, prompt-engineering iterations, and the LLM
judge's scoring rubric are described in prior chapters. For the failure
analysis below, the only per-image generation metadata used is: (i) the LLM
morphological mean (1–5 scale, five features averaged), (ii) the LLM tier
(`strict_pass` / `borderline` / `soft_fail` / `hard_fail`), and (iii) the
filename's generated-species label. Aggregate LLM pass rates by species
(ashtoni 44.4 %, flavidus 57.6 %, sandersoni 91.2 %) motivate the downstream
D5 dataset composition.

### 5.3 Augmentation Method Comparison

#### 5.3.1 Aggregate effect across protocols

Three protocols (single-split, 5-seed on fixed split, 5-fold cross-validation
using the f1 checkpoint) give numerically different but qualitatively
consistent rankings for macro F1 (Table 5.1). Common and moderate tier
species show no signed effect under any protocol (Δ ≤ 0.013 F1). All
interesting behaviour is concentrated in the rare tier and is reported in
dedicated sub-sections below.

*Table 5.1 — Macro F1 across augmentation strategies, three evaluation
protocols (f1 checkpoint).*

| Protocol | D1 Baseline | D3 CNP | D4 Synthetic | D5 LLM-filtered |
|---|---:|---:|---:|---:|
| Single-split (1 run) | 0.815 | **0.829** | 0.823 | 0.834 |
| 5-fold CV | 0.832 ± 0.013 | **0.837 ± 0.013** | 0.820 ± 0.024 | 0.821 ± 0.019 |
| Multi-seed (5 × fixed split) | **0.839 ± 0.006** | 0.822 ± 0.014 | 0.828 ± 0.009 | 0.831 ± 0.008 |

The multi-seed ranking inverts single-split (D1 best rather than D5 best).
The inversion is mechanical: multi-seed and single-split both evaluate on
the identical 2 362-image test set, so small changes in which rare-species
test images are correctly classified dominate macro F1. K-fold pools
predictions across folds (rare-species effective n = 32 / 58 / 232) and
produces the most stable aggregate estimate. D5 is significantly worse than
both D1 (p = 0.041) and D3 (p = 0.030) under 5-fold; no pairwise test reaches
significance under multi-seed (closest: D1 vs D3, p = 0.054). Appendix A
gives the full paired t-test table.

#### 5.3.2 Head / tail tier breakdown

Because augmentation effects are concentrated in the rare tier, Table 5.2
reports tier-mean F1 under multi-seed. Moderate (n ∈ [200, 900]) and common
(n > 900) tiers move by ≤ 0.013 under any augmentation method. Only the rare
tier carries signal, and the signed effect is negative under all three
methods.

*Table 5.2 — Tier-mean F1 (multi-seed, unweighted species-mean within tier;
± std across the species in the tier).*

| Tier | # species | D1 Baseline | D3 CNP (Δ) | D4 Synthetic (Δ) | D5 LLM-filtered (Δ) |
|---|---:|---:|---:|---:|---:|
| Rare (< 200 train) | 3 | 0.665 ± 0.082 | 0.593 (−0.073) | 0.590 (−0.075) | 0.617 (−0.048) |
| Moderate (200–900) | 7 | 0.854 ± 0.056 | 0.851 (−0.003) | 0.862 (+0.008) | 0.858 (+0.004) |
| Common (> 900) | 6 | 0.907 ± 0.037 | 0.904 (−0.003) | 0.908 (+0.000) | 0.906 (−0.001) |

The 5-fold reading differs in one important respect: D3 CNP yields +0.030
on the rare tier under 5-fold versus −0.073 under multi-seed. This is the
same protocol-dependent reversal as § 5.3.1: k-fold pools 5 × more rare test
images per species and thus samples away from the specific handful of
borderline test cases on the fixed split. Both readings agree that D4 and
D5 harm the rare tier.

Across *both* protocols, however, **no augmentation method improves any rare
species by more than its per-tier standard deviation**, while D4 and D5
reduce rare-tier macro F1 by 0.05 – 0.08. The single unambiguous positive
finding is B. flavidus under D3 CNP on 5-fold (+0.059, p = 0.005); this
result motivates the narrative that *real-texture* augmentation (CNP) is the
only strategy with defensible rare-species utility in our data.

#### 5.3.3 Per-species F1 deltas

Figure 5.1 displays per-species Δ F1 under each augmentation method relative
to D1 (two panels: multi-seed on the fixed split and 5-fold pooled). Both
panels show the same structure: rare-species rows are coloured saturated red
for D4/D5, moderate and common rows are essentially grey.

- [species_f1_delta_multiseed.png](plots/failure/species_f1_delta_multiseed.png)
- [species_f1_delta_kfold.png](plots/failure/species_f1_delta_kfold.png)

Notable species-level findings:

- B. sandersoni drops −0.145 (D3), −0.128 (D4), −0.089 (D5) under
  multi-seed; k-fold gives −0.033 / −0.140 / −0.067. D4 is the worst case
  under both protocols.
- B. flavidus drops −0.091 (D4) multi-seed but gains +0.059 (D3) k-fold.
  D5 hedges between: −0.051 multi-seed / −0.012 k-fold.
- B. ashtoni is the noisiest rare species (n = 6 fixed test, n = 32 k-fold
  pooled). Multi-seed says all three methods are essentially flat
  (−0.037 D3, −0.006 D4, −0.005 D5); k-fold shows D3 positive (+0.064) but
  D4/D5 negative (−0.045 / −0.044).

### 5.4 Failure Mode Analysis

Section 5.3 establishes *that* augmentation harms rare species; § 5.4
explains *why*. The analysis combines per-image prediction tracking,
embedding-space analysis with BioCLIP, and the LLM judge's disagreement
with classifier-relevant features.

#### 5.4.1 Per-image prediction flips

Each test image yields 20 predictions (4 configs × 5 seeds). Collapsing via
majority rule gives one verdict per (image, config). Comparing baseline to
each augmented config produces a 2 × 2 flip category: stable-correct,
stable-wrong, improved (aug rescued a baseline error), or harmed (aug broke
a baseline-correct image).

*Table 5.3 — Rare-species flip counts under D3 / D4 / D5 (multi-seed
majority vote). No rare-species test image is improved by any augmentation
method under this protocol.*

| Species | n test | D3 improved / harmed | D4 improved / harmed | D5 improved / harmed |
|---|---:|---:|---:|---:|
| B. ashtoni | 6 | 0 / 1 | 0 / 0 | 0 / 1 |
| B. sandersoni | 10 | 0 / 1 | 0 / 1 | 0 / 1 |
| B. flavidus | 36 | 0 / 5 | 0 / 8 | 0 / 6 |

Expressed as a flip rate, B. flavidus is harmed on 22.2 % of its D4 test
images — the largest rate in any species × method cell. Common species have
~ 1.4 % flip rates in both directions; their absolute counts are inflated
only by their large test-set sizes (Appendix B).

Figure 5.2 overlays this as a species × method heatmap; the rate panel
normalises by test-set size and makes the rare-tier concentration obvious.

- [flip_category_heatmap.png](plots/failure/flip_category_heatmap.png)

#### 5.4.2 Embedding-space analysis

To investigate the mechanism, we embed every real training image and every
synthetic image with a frozen BioCLIP ViT-B/16 vision encoder
(L2-normalised CLS tokens, 512-d). The choice of BioCLIP over DINOv2
ViT-L/14 is grounded by a 5-NN species diagnostic on the real training set
(overall accuracy 0.657 for BioCLIP vs 0.295 for DINOv2; Appendix D).

Two embedding-space findings are central. First, when real and synthetic
images for the three rare species are projected together, synthetics of each
rare species form a tight, well-separated cluster whose centre is
systematically offset from the centre of the corresponding real cluster
(Fig. 5.3a; cosine distance from each synthetic to its species' real
centroid is shown in Fig. 5.3b):

- Median synthetic → real-centroid cosine distance: 0.31 ashtoni, 0.25
  sandersoni, 0.32 flavidus.
- Real → real-centroid distance (for comparison): 0.1 – 0.2 for all three.

- [embeddings_rare_real_synth.png](plots/embeddings/bioclip_tsne/embeddings_rare_real_synth.png)
- [embeddings_centroid_distance.png](plots/embeddings/bioclip_tsne/embeddings_centroid_distance.png)

Second, the rare-species synthetics cluster **by target species**
(Fig. 5.3a) — they are *not* drifting into each other's embedding region,
and they are *not* landing on top of the confuser species. Instead, they
carve out their own sub-manifold that is a few tenths of cosine-space away
from where real rare bees actually sit. The practical consequence is that
the classifier, while training on this synthetic sub-manifold, is learning
a set of species-discriminative features that are *offset from* the feature
subspace used by real test images of the same species.

The all-species t-SNE (Fig. 5.4, Appendix D) shows that this offset is not
a general "synthetic ≠ real" phenomenon: for common species, synthetic
generation is not reported in this thesis, but the real-image manifold is
well-structured by species. The rare-species gap is specific to the three
species where generation is most needed.

#### 5.4.3 Per-species galleries

Figure 5.5 (three panels, one per rare species) presents 6 sampled real
training images, 6 sampled synthetic images, and all harmed test images
side by side. The visual contrast is pronounced for B. flavidus (only
species with > 3 harmed test images), where synthetic images concentrate on
a narrow stylistic distribution — uniform clean backgrounds, consistent
pose, stylised coloration — while real flavidus vary substantially in pose,
illumination, and degree of occlusion.

- [per_species_gallery_ashtoni.png](plots/failure/per_species_gallery_ashtoni.png)
- [per_species_gallery_sandersoni.png](plots/failure/per_species_gallery_sandersoni.png)
- [per_species_gallery_flavidus.png](plots/failure/per_species_gallery_flavidus.png)

#### 5.4.4 Failure chains

For every rare-species test image harmed under D4 or D5 we retrieve the
five nearest synthetic images that were *actually* present in that
variant's training set (200 per rare species × 3 species = 600 synthetics
per variant). Nearest neighbours are measured by BioCLIP cosine similarity.
Each retrieved synthetic carries its generated-species label and LLM tier.

Figure 5.6a (selected representative chain) shows a harmed B. flavidus test
image under D4: five nearest training synthetics are all B. sandersoni
strict-pass synthetics at cosine similarity 0.60, sitting closer to the
harmed image in BioCLIP space than any D4 flavidus synthetic. Under D4 this
test image is classified as B. sandersoni; under the baseline it is
classified correctly. Figure 5.6b projects the same chain onto the rare-real
+ variant-synthetic t-SNE and shows the retrieval arrows crossing several
cluster boundaries.

- [selected D4 harmed chain](plots/failure/chains_d4_harmed/gallery/flavidus__Bombus_flavidus4921845786.png)
- [same chain on t-SNE](plots/failure/chains_d4_harmed/tsne/flavidus__Bombus_sandersoni4952898591.png)

Aggregate statistics across all 49 D4 harmed chains (Appendix C): the
median test → 5-NN synthetic cosine similarity is 0.56 — well below the
0.7+ range at which the same measurement between two real images of the
same species peaks — which says the "nearest synthetic" is *not actually
close*. It is close only relative to the rest of the synthetic pool.

#### 5.4.5 LLM judge vs classifier-relevant quality

Per-synthetic LLM morphological scores are weakly informative about
classifier-relevant quality. Figure 5.7 is a scatter of synthetic images in
the plane (x = LLM morph mean; y = BioCLIP cosine distance to the correct
species real centroid). Dashed lines mark per-species medians. The upper-
right quadrant ("LLM > median morph but far from real centroid") contains
138 ashtoni / 49 sandersoni / 108 flavidus synthetics — images the LLM
judge passes confidently but that the classifier's feature space places
beyond the real species distribution. This quadrant corresponds to
synthetics that *pass* the current D5 filter yet appear in failure-chain
neighbour sets. Appendix E gives the full 4-quadrant counts.

- [llm_vs_centroid_quadrant.png](plots/failure/llm_vs_centroid_quadrant.png)

#### 5.4.6 Causal attribution: subset ablation

Six additional training runs (seed 42 only) drop one rare species'
synthetic images at a time (D4-no-ashtoni, D4-no-sandersoni, D4-no-flavidus
and the corresponding D5 variants). The question is whether removing a
species' synthetics recovers that species' F1; positive recovery implies
the removed synthetics were collectively *harming* the target species.

*Table 5.4 — Own-species F1 recovery under subset ablation (seed 42).
Positive = that species' synthetics were harmful; negative = helpful.*

| Variant | Dropped species | F1 full → F1 ablated | Recovery | Label (|·|>0.02) |
|---|---|---:|---:|---|
| D4 | ashtoni | 0.545 → 0.727 | **+0.182** | harmful |
| D4 | sandersoni | 0.571 → 0.476 | −0.095 | helpful |
| D4 | flavidus | 0.645 → 0.708 | +0.062 | harmful |
| D5 | ashtoni | 0.727 → 0.727 | +0.000 | neutral |
| D5 | sandersoni | 0.625 → 0.706 | +0.081 | harmful |
| D5 | flavidus | 0.725 → 0.719 | −0.006 | neutral |

Two striking patterns emerge. First, **LLM filtering (D5) reverses the
sandersoni effect**: in D4 sandersoni's synthetics were collectively
helpful (removing them lost −0.095 F1), but once the LLM filter is applied
(D5), sandersoni's *filtered* synthetics become harmful (+0.081 recovery
on removal). The LLM judge is keeping the wrong sandersoni synthetics.
Second, **LLM filtering neutralises the large ashtoni harm** seen in D4
(+0.182 recovery → +0.000 in D5), confirming that the filter does remove
some genuinely bad ashtoni synthetics — but the residual harm for
sandersoni makes D5 no better on aggregate than D4.

Cross-species collateral is substantial (Figure 5.10, full 3 × 3 recovery
matrix). Two representative cells:

- **D4, drop ashtoni → sandersoni F1 changes by −0.150.** The ashtoni
  synthetics were not only harming ashtoni itself; they were also
  providing gradient signal that helped sandersoni, so removing them
  hurts sandersoni. This indicates cross-species feature sharing during
  fine-tuning.
- **D4, drop ashtoni → flavidus F1 changes by +0.124.** The same ashtoni
  synthetics were also harming flavidus. They live in flavidus-like
  regions of BioCLIP space (confirmed by failure chains in § 5.4.4) and
  pull flavidus predictions toward ashtoni at test time.

- [subset_ablation_recovery.png](plots/failure/subset_ablation_recovery.png)

*Interpretive caveat.* Every ablation cell is a single seed-42 run. With
rare-test-set n between 6 and 36, a ±0.05 F1 change corresponds to
flipping 0.3 – 1.8 images and should be read as directional, not
statistically powered. The cross-species collateral numbers (−0.150 on
n = 10 sandersoni) are the most fragile and are interpreted here as
direction-only.

Per-synthetic labels (species-level propagation, see Appendix C) are
written to `RESULTS/failure_analysis/synthetic_labels.csv` and are used
to colour synthetic markers in the § 5.7 (pending) helpful / harmful
embedding overlay.

### 5.5 Expert Calibration Results [PENDING]

Depends on completed expert validation of the 150-image annotation sample.
When labels land: (i) LLM rule AUC-ROC baseline, (ii) BioCLIP linear-probe
filter LOOCV AUC-ROC, (iii) D6-probe downstream macro F1 vs D4 / D5.

### 5.6 Latent Space Analysis

BioCLIP is selected over DINOv2 on the basis of a 5-NN species diagnostic
on the real training images (Appendix D).

Figure 5.8 compares three embedding views (real training images only):

- 16-species overview t-SNE (Figure 5.8a): common species form compact,
  well-separated clusters; rare species flavidus dominates a broader region
  mixed with borealis / perplexus; ashtoni and sandersoni form small
  sub-clusters often surrounded by their baseline confusers.
- Rare-species-only t-SNE (Figure 5.8b): even in isolation, the three rare
  species overlap substantially. This is the baseline indistinguishability
  that pre-exists any synthetic-augmentation question.
- Embedding atlas with thumbnails (Figure 5.8c): thumbnails at true t-SNE
  coordinates confirm the above clusters are pose- and colour-coherent
  rather than artefactual.

- [embeddings_overview.png](plots/embeddings/bioclip_tsne/embeddings_overview.png)
- [embeddings_rare_real_only.png](plots/embeddings/bioclip_tsne/embeddings_rare_real_only.png)
- [embedding_atlas_rare_tsne.png](plots/failure/embedding_atlas_rare_tsne.png)
- [embedding_atlas_all_tsne.png](plots/failure/embedding_atlas_all_tsne.png)

The confusion-pair triplets (Figure 5.9, Appendix E) visually confirm the
baseline-confusion directions identified in § 5.1: sandersoni synthetic
images are distinguishable from real B. vagans, and flavidus synthetic
images are distinguishable from real B. citrinus, yet each set of real
rare-species images remains visually ambiguous against its confuser — the
gap is between *synthetic and real of the same species*, not between
synthetic and the confuser.

- [confusion_triplet_ashtoni__vagans_Smith.png](plots/failure/confusion_triplet_ashtoni__vagans_Smith.png)

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
