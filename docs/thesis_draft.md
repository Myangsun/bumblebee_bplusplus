# The Synthetic-Real Gap in Urban Biodiversity Monitoring: Generative Augmentation and Expert-Calibrated Filtering for Long-Tailed Bumblebee Classification

## Abstract

Automated pollinator monitoring in cities is limited by extreme class imbalance: for 16 Massachusetts bumblebee species, the rarest have fewer than 40 training images while common species have thousands, and baseline classifiers achieve F1 below 0.60 for the three rarest taxa. We develop a three-stage augmentation pipeline -- structured morphological prompting with tergite-level colour maps for reference-guided image generation, a two-stage LLM-as-judge combining blind taxonomic identification with five-feature morphological scoring, and expert-calibrated quality filtering learned from entomologist annotations -- and evaluate it under five-fold cross-validation, multi-seed training, and per-image failure analysis. Copy-and-paste augmentation produces the only statistically significant rare-species F1 gain (B. flavidus +0.059, p = 0.005); both unfiltered and LLM-filtered generative augmentation reduce rare-tier F1 below baseline (D4 LLM-filtered vs. baseline p = 0.041), and the LLM filter is statistically indistinguishable from no filter at all (D3 vs. D4 p = 0.777). An independent 150-image expert-labelling study further shows that the LLM rule and the expert agree on strict pass/fail for only 84 of 150 images (56 %, ~7 percentage points above the 49 % rate expected under independence; LLM precision 0.40, recall 0.62; morph-mean AUC 0.56 as a ranker), while a BioCLIP-only linear probe trained on the same 150 labels reaches LOOCV AUC 0.792 and selects 200-image subsets per rare species that are ~3× richer in expert-validated synthetics than the LLM rule. Downstream evaluation of the centroid-filtered (D5) and expert-probe-filtered (D6) dataset variants is [TODO: multi-seed × fixed split and 5-fold CV training runs pending at submission]. Failure analysis traces the harm to a per-species feature-space offset between synthetic and real images that pre-exists training, and single-species subset ablation confirms the offset causally — including a B. sandersoni D3 → D4 sign reversal in which the LLM filter discards the wrong subset for the species it passes most easily. Volume ablation from +50 to +500 synthetic images shows no consistent improvement, ruling out generation quantity as the bottleneck. These results establish that current generative augmentation pipelines move false-negative risk in the wrong direction for rare species and that closing the synthetic-real gap requires quality filtering calibrated against classifier-relevant feature space, not against language-mediated morphology rubrics alone.


## 1. Introduction

### 1.1 Motivation

Global urban land cover is projected to triple between 2000 and 2030. While urban expansion has traditionally been associated with severe habitat loss, contemporary ecological research increasingly recognizes cities as critical frontiers for biodiversity conservation and climate resilience. These "accidental ecosystems" (Alagona, 2024) often function as refuges for species displaced from degraded farmland and shrinking natural landscapes, which Schilthuizen (2025) describes as a "naturalist's gold mine" of hidden biological riches. A weedy vacant lot or an overgrown park edge can quietly harbor biological communities that no longer persist in the surrounding countryside. For urban planners, the question is whether we detect these species before we pave them over.

Bumblebees (Bombus spp.) provide a valuable entry point for understanding urban biodiversity. As among the most important wild pollinators in temperate ecosystems, they provide essential pollination services to both wildflowers and over 25 commercial crop species through their unique capacity for buzz-pollination (Goulson, 2010), making them foundational to both natural ecosystems and urban ecological services. They also represent a conservation paradox: while several species have experienced severe population crashes over the past century (Cameron et al., 2011), urban green spaces now support populations of bumblebees that are increasingly rare elsewhere.

The trouble is that we cannot tell the difference between a species that has vanished and one we simply keep missing. Historical baselines established by Plath (1934) near Boston documented 17 Bombus species, describing taxa such as B. affinis and B. terricola as common. Contemporary studies reveal a dramatic shift: Jacobson et al. (2018) reported a 96.4% decline in relative abundance for B. fervidus in New Hampshire, while Richardson et al. (2019) found that four of seventeen historically documented species were not detected in modern Vermont surveys despite a six-fold increase in sampling effort. Cameron et al. (2011) documented range contractions of up to 87% for declining species across 382 sites in 40 U.S. states. In Massachusetts, B. ashtoni was last documented near Boston by Plath (1934) and has not been recorded in the state since. B. terricola and B. fervidus show up only sporadically in modern surveys. Are these species truly gone, or are our tools failing to detect them?

This "detection-versus-extinction" uncertainty carries real consequences for how we build our cities. MacKenzie et al. (2002) provide the statistical framework for this problem, demonstrating that nondetection at a site does not imply absence unless detection probability approaches one. If a rare bee persists but our monitoring systems miss it, planners may unknowingly approve development on the very habitat fragments that should have been protected.

### 1.2 Problem Statement

A major contributor to this detection failure is classification error in automated monitoring systems. Modern computer vision models struggle with rare taxa because their training datasets exhibit long-tailed distributions. In community-contributed datasets such as GBIF (GBIF.org, 2025), common species like B. impatiens are represented by thousands of images, while rare species such as B. sandersoni may have fewer than one hundred. Under such an imbalance, classifiers learn to recognize abundant species well but fail to capture the subtle morphological features required for fine-grained species identification, such as abdominal banding patterns, facial hair coloration, and other diagnostic features.

[Table 1: Dataset species, counts in GBIF, and sample images]

The dataset in this thesis comprises 16 Massachusetts Bombus species totalling 15,630 images sourced from GBIF, with an imbalance ratio of 59.9:1. A baseline ResNet-50 classifier achieves 88.2% overall accuracy and 0.815 macro F1, but collapses on critical rare species: F1 = 0.500 for B. ashtoni (n = 6 test images, 95% CI [0.000, 0.818]) and F1 = 0.588 for B. sandersoni (n = 10, [0.222, 0.833]). These confidence intervals span more than 0.6 of the F1 range, reflecting genuine evaluation uncertainty at small sample sizes.

Generative AI offers a potential solution to this imbalance: if we do not have enough real images of a rare species, we can generate synthetic ones to balance the training data. The empirical reality is more complex than adding volume. Under five-fold cross-validation, neither unfiltered (D3) nor LLM-filtered (D4) generative augmentation improves rare-tier macro F1 over the baseline (D1 vs. D4 p = 0.041, with D4 worse), and the LLM filter is statistically indistinguishable from no filter at all (D3 vs. D4 p = 0.777). Volume ablation from +50 to +500 synthetic images shows no consistent improvement at any volume. Meanwhile, LLM judge analysis reveals that 27.1% of generated images exhibit wrong coloration, concentrated in the most morphologically atypical species. This is a *fidelity gap*: AI-generated images may appear visually convincing but fail to carry the correct training signal for fine-grained classification. Looking realistic is not the same as being diagnostically faithful, and -- as Section 5 will show -- the gap is large enough in BioCLIP feature space that current LLM-mediated quality filters cannot close it.

### 1.3 Thesis Statement

For fine-grained biodiversity classification under extreme class imbalance, generative augmentation requires morphology-guided generation and multi-dimensional quality filtering calibrated against domain expertise. Neither quality nor quantity alone is sufficient.

This thesis investigates the nature of that fidelity gap. The goal is to understand why synthetic images underperform, pinpoint what they get wrong at the level of species-diagnostic features, and develop a data-augmentation strategy that preserves the features necessary for accurate species classification. Specifically, the research develops a generator–discriminator pipeline that produces high-quality training data for long-tailed ecological datasets. The resulting augmentation strategy — and the rare-species classifier it trains — is designed to serve as the species-level decision stage of the MIT Senseable City Lab's **Sensing Garden** project and its **Flik** platform, an AI-powered urban pollinator-monitoring system that runs motion-based detection, tracking, and hierarchical taxonomic classification on camera deployments in gardens, rooftops, and parks. Sensing Garden is the concrete operational context in which this work's findings are meant to be used (Section 6.6).

By closing the fidelity gap for rare species, this work aims to produce automated urban-biodiversity indicators that carry accurate information to both conservation policy and urban-planning decisions, rather than indicators constrained by the shortcomings of current datasets.

[Fig 1: Pipeline Framework]

### 1.4 Contributions

This thesis makes four contributions:

1. **A structured morphological prompting framework** for generating species-faithful images using reference-guided image editing, with negative constraints and tergite-level color maps that override generative model priors toward dominant phenotypes. By the final prompt version, structural failures (extra limbs, impossible geometry) were eliminated across all 1,500 generated images; the sole remaining failure mode was coloration accuracy (27.1%).

2. **A two-stage LLM-as-judge evaluation protocol** combining blind taxonomic identification with multi-dimensional morphological scoring (5 features, 1--5 scale), providing per-feature diagnostic signals that reveal *where* generated images fail -- not just *whether* they pass.

3. **An expert-calibrated quality filtering pipeline** in which entomologist annotations on a stratified 150-image subset are used to learn feature-level weights that correct the LLM judge's holistic miscalibration, with a diagnostic feedback loop connecting evaluation to generation improvement.

4. **Rigorous empirical evidence** under five-fold cross-validation, multi-seed training, and per-image failure analysis demonstrating that (a) generative synthetic augmentation — both unfiltered (D3) and LLM-filtered (D4) — significantly reduces rare-species macro F1 below baseline (D1 vs. D4 p = 0.041), with the LLM filter providing no measurable improvement over no filter at all (D3 vs. D4 p = 0.777); (b) the harm is mechanistically attributable to a per-species feature-space offset between synthetic and real images that pre-exists training, confirmed causally by single-species subset ablation including a B. sandersoni D3 → D4 sign reversal; (c) copy-and-paste augmentation (D2) — which preserves real morphological texture and therefore places its outputs in the same region of feature space as the real images themselves — produces the only statistically significant per-species F1 gain in the study (B. flavidus +0.059, p = 0.005); (d) volume ablation rules out generation quantity as the bottleneck; and (e) on 150 expert-annotated synthetics, the LLM rule and the expert agree on strict pass/fail for only 56 % of images (~7 percentage points above the 49 % rate expected under independence; LLM precision 0.40, recall 0.62; morph-mean AUC 0.56) while a BioCLIP-feature linear probe trained on the same labels achieves LOOCV AUC 0.792 and selects 200-image subsets that are ~3× richer in expert-validated synthetics than the LLM rule. Downstream F1 for the resulting centroid-filtered (D5) and expert-probe-filtered (D6) dataset variants is [TODO: pending GPU training at submission]. These findings establish that closing the synthetic-real gap requires quality filtering calibrated against classifier-relevant feature space, not against language-mediated morphology rubrics alone, and motivate the expert-calibrated filtering pipeline as the necessary next step.

These contributions bridge urban planning and computer science by connecting classification performance directly to the detection-versus-extinction uncertainty that constrains biodiversity-informed urban development decisions.

### 1.5 Organization

The remainder of this thesis is organised as follows. Section 2 reviews related work spanning bumblebee ecology and decline, fine-grained visual classification, long-tail recognition, generative data augmentation, and synthetic image quality evaluation. Section 3 describes the dataset, preparation pipeline, target species taxonomy, classifier architecture, and the dual-protocol evaluation methodology (5-fold CV for aggregate claims, multi-seed for per-image analyses). Section 4 presents the three novel methods: structured morphological prompting, LLM-as-judge evaluation, and expert-calibrated quality filtering. Section 5 reports experimental results in six subsections: baseline classifier behaviour, latent-space analysis of real and synthetic embeddings, LLM-judge quality results over the full 1,500-image pool, expert calibration on a 150-image stratified sample (including the LLM–expert agreement study and the learned BioCLIP probe filter), augmentation method comparison across six dataset variants under three evaluation protocols, and per-image mechanistic analysis with causal subset ablation. Section 6 develops the mechanistic discussion, traces the LLM-judge calibration gap to a single empirical demonstration (B. sandersoni D3 → D4 sign reversal), and draws implications for urban biodiversity monitoring. Section 7 concludes and outlines future work.


## 2. Related Work

### 2.1 Bumblebee Decline and Automated Monitoring

The documentation of bumblebee populations in New England is anchored in Plath (1934), whose baseline of 17 species near Boston provides the historical benchmark against which contemporary declines are measured. Multiple independent studies have since quantified the magnitude of this shift using distinct methodological approaches: Cameron et al. (2011) employed standardized sampling with mitochondrial COI and microsatellite data across 382 sites in 40 U.S. states to document range contractions of up to 87% for declining species, while Jacobson et al. (2018) used multi-decadal museum record comparisons to reveal a 96.4% decline in relative abundance for B. fervidus in New Hampshire. Richardson et al. (2019) combined historical literature review with six years of standardized netting surveys in Vermont, finding that four of seventeen historically documented species were undetected despite substantially greater sampling effort than any prior study. Critically, these studies cannot distinguish true extirpation from detection failure -- a distinction formalized by MacKenzie et al. (2002) in the occupancy modeling framework, which demonstrates that nondetection probability must be jointly estimated alongside occupancy to avoid false-absence conclusions. Community science platforms have begun to address this gap: MacPhail et al. (2024) showed that Bumble Bee Watch observations contributed records for species otherwise underrepresented in professional surveys.

This detection challenge extends to automated monitoring systems, which inherit the long-tailed distribution of their training data. Spiesman et al. (2021) trained deep learning classifiers on 89,776 images across 36 North American Bombus species and achieved 91.6% top-1 accuracy overall, but excluded six species entirely for having fewer than approximately 150 training examples. Among included species, error rates ranged from 4.0% for morphologically distinctive taxa to 20.4% for variable species, with B. rufocinctus confused with 25 other species. Spiesman et al. explicitly identify generative data augmentation as a potential remedy for this class imbalance. At larger scale, Bjerge et al. (2024) deployed 48 camera traps capturing over 10 million insect images and achieved 80% average precision, but species with too few training images were collapsed into an undifferentiated "unspecified arthropods" class -- precisely the outcome that renders monitoring systems useless for rare-species conservation. Bjerge et al. (2023) showed that hierarchical classification can gracefully degrade to genus-level identification, but species-level resolution is what conservation planning requires.

### 2.2 Fine-Grained Visual Classification

Fine-grained visual classification (FGVC) addresses the problem of distinguishing subordinate categories within a broader class -- species within a genus, aircraft models within a manufacturer, car variants within a make. Unlike standard object recognition where inter-class differences are large, FGVC is characterized by high inter-class similarity and high intra-class variation: two Bombus species may differ only in the width of a thoracic band, while individuals within a single species vary across castes, geographic populations, and seasonal phenotypes. Wei et al. (2021) provide a comprehensive survey of deep learning methods for fine-grained image analysis, tracing the evolution from part-based representations to end-to-end attention mechanisms.

Early deep learning approaches to FGVC focused on learning discriminative feature representations through specialized architectures. Lin et al. (2018) introduced bilinear CNN models, which compute the outer product of features from two CNN streams at each spatial location, capturing localized pairwise feature interactions in a translationally invariant manner. Zheng et al. (2017) proposed the Multi-Attention CNN (MA-CNN), in which channel grouping and part classification sub-networks jointly learn to localize discriminative parts and extract part-specific features without requiring bounding box or part annotations at test time. These methods demonstrated that FGVC benefits from architectures that explicitly attend to subtle, localized visual differences rather than relying on holistic image-level features.

Transfer learning from large-scale datasets has become the dominant paradigm for FGVC. Kornblith et al. (2019) systematically studied ImageNet transfer, finding a strong rank correlation (r = 0.96) between ImageNet accuracy and fine-tuned transfer accuracy across 12 downstream datasets. More recently, vision-language foundation models have reshaped the landscape. CLIP (Radford et al., 2021) demonstrated zero-shot classification by aligning image and text embeddings, but its performance degrades on fine-grained tasks where class distinctions require domain-specific visual knowledge. DINOv2 (Oquab et al., 2024) showed that self-supervised pretraining at scale produces general-purpose visual features competitive with supervised methods across fine-grained tasks. For biological applications specifically, BioCLIP (Stevens et al., 2024) adapted the CLIP framework to the tree of life by training on TreeOfLife-10M, achieving 17--20% absolute accuracy gains over general-purpose baselines on fine-grained biology benchmarks.

Given this landscape, fine-tuning a ResNet-50 (He et al., 2015) from ImageNet-pretrained weights represents a well-understood baseline for FGVC. The strong correlation between ImageNet pretraining quality and downstream performance (Kornblith et al., 2019) makes it a principled choice for isolating the effect of data augmentation from architectural novelty -- this thesis deliberately uses this established baseline so that observed performance changes can be attributed to augmentation strategy rather than model capacity.

### 2.3 Long-Tail Classification and the Challenge in Biodiversity

Long-tailed distributions represent a fundamental challenge for deep learning, where a small number of head classes dominate training data while a long tail of rare classes have few examples each. Liu et al. (2019) formalized the problem as Open Long-Tailed Recognition, where benchmarks range from 1,280 to as few as 5 images per class and non-ensemble baselines achieve only approximately 67% top-1 accuracy on iNaturalist (Horn et al., 2018). The iNaturalist dataset -- a large-scale species classification benchmark drawn from citizen science observations -- has become the standard testbed for long-tail recognition precisely because its imbalance is natural, reflecting true species abundance and observer effort rather than artificial subsampling.

The literature offers three broad categories of solutions. *Loss re-weighting* methods modify the training objective to upweight rare classes: Lin et al. (2018) introduced focal loss to down-weight well-classified examples, Cui et al. (2019) proposed class-balanced loss based on the effective number of samples, and Cao et al. (2019) derived label-distribution-aware margins (LDAM) that enforce larger decision margins for tail classes. *Decoupled training*, introduced by Kang et al. (2020), demonstrated that representation learning and classifier learning have different optimal strategies under imbalance -- training the backbone on the natural (imbalanced) distribution produces better representations, while the classifier head benefits from class-balanced re-calibration. This simple two-stage approach matched or outperformed many complex end-to-end methods. *Multi-expert approaches* such as RIDE (Wang et al., 2022) route inputs to distribution-aware expert branches, each specializing in different portions of the class frequency spectrum.

However, all of these methods share a fundamental limitation: they rebalance or re-weight existing data but cannot introduce new visual information. With fewer than 40 training images, oversampling duplicates the same instances, loss re-weighting increases gradients from the same limited views, and decoupled training still depends on representations learned from insufficient visual evidence. The model cannot learn the intra-class variation in pose, lighting, and morphology that is absent from the training set, regardless of how the loss function is calibrated. The biodiversity data gap compounds this: even BioTrove (Yang et al., 2024), the largest curated biodiversity image dataset at 161.9 million images spanning approximately 366,600 species, inherits the extreme long tail where many species remain critically underrepresented. For the rarest species -- B. ashtoni with 22 training images and B. sandersoni with 40 -- what is missing is visual diversity itself, not a better weighting scheme. This motivates the shift from re-weighting to data augmentation: generating or synthesizing new training examples that introduce the morphological variation the original dataset lacks.

### 2.4 Data Augmentation with Generative Models

Data augmentation expands training sets through transformation or synthesis and is most critical when data is scarce (Shorten & Khoshgoftaar, 2019). Traditional augmentation methods -- geometric transforms, color jitter, random erasing -- increase diversity but cannot generate novel morphological variation beyond what exists in the original training images.

To bridge the data gap, researchers have turned to synthetic augmentation. Ghiasi et al. (2021) demonstrated that simple Copy-Paste augmentation -- pasting segmented objects onto new backgrounds -- is a strong data augmentation method that improves generalization by preserving authentic object appearance while varying context. However, Generative AI offers a newer frontier for "upsampling" rare classes. Zhao et al. (2024) proposed LTGC, utilizing LLMs to reason about missing visual attributes in tail data to guide generation. While promising, He et al. (2023) show that the effectiveness of synthetic data is highly domain-dependent, succeeding for birds (+10% accuracy) but failing for other categories.

The "synthetic-real gap" remains a significant hurdle for fine-grained biological classification. Azizi et al. (2023) found a persistent 4--8-percentage-point accuracy gap between synthetic-only and real training data, even with state-of-the-art diffusion models. For insects, TaxaDiffusion (Monsefi et al., 2025) has recently begun incorporating taxonomic hierarchy to improve species-level synthesis. DisCL (Liang et al., 2025) suggests that a curriculum scheduling data from synthetic to real is essential to prevent out-of-distribution data from being detrimental to performance. SaSPA (Michaeli & Fried, 2025) further argues that preserving class fidelity for fine-grained tasks requires structural conditioning (e.g., edge-based constraints) to ensure that diagnostic features such as wing morphology are not distorted.

### 2.5 Quality Evaluation of Synthetic Images

The two most widely used automated metrics for evaluating synthetic image quality -- the Inception Score (IS; Salimans et al., 2016) and the Frechet Inception Distance (FID; Heusel et al., 2018) -- both rely on features extracted from an Inception network pretrained on ImageNet. IS measures the KL divergence between conditional and marginal label distributions, rewarding both quality and diversity, while FID compares the mean and covariance of Inception features between real and generated distributions under a Gaussian assumption. However, both metrics suffer from well-documented limitations in specialized domains. Borji (2021) provides a comprehensive survey of these shortcomings, noting that FID is statistically biased, sensitive to sample size, and dependent on features optimized for ImageNet categories rather than the target domain. Jayasumana et al. (2024) further demonstrated that FID contradicts human raters and fails to reflect incremental improvements in iterative generation. For fine-grained biological imagery where diagnostic differences may be confined to single body segments, these ImageNet-derived features are poorly suited to capture taxonomically relevant variation.

Human evaluation remains the gold standard but is expensive, subjective, and difficult to reproduce. Otani et al. (2023) surveyed 37 text-to-image generation papers and found that many either omit human evaluation entirely or describe it so poorly that results cannot be replicated. They proposed a standardized protocol and showed experimentally that automatic metrics like FID are often incompatible with human perceptual judgments. Visual Turing tests -- in which evaluators distinguish real from synthetic -- have been applied in medical imaging but require domain experts and scale poorly when hundreds of fine-grained categories must each be assessed for taxonomic accuracy.

The LLM-as-a-judge paradigm offers a scalable alternative. Zheng et al. (2023) established this framework with MT-Bench, demonstrating that strong LLMs achieve over 80% agreement with human preferences when evaluating text outputs, matching inter-annotator agreement rates. This paradigm has been extended to the visual domain: Xu et al. (2023) trained ImageReward, a reward model that learns human preferences for text-to-image generation, outperforming CLIP-based scoring; Lu et al. (2023) proposed LLMScore, which decomposes images into multi-granularity visual descriptions and generates scores with rationales, achieving substantially higher correlation with human judgments than CLIP or BLIP matching scores. Despite these advances, no existing work combines VLM-based evaluation with expert calibration specifically for biological image synthesis, where the evaluator must assess not just visual realism but taxonomic accuracy of diagnostic morphological features.

### 2.6 Positioning

This thesis sits at the intersection of three underexplored areas: generative augmentation for long-tailed fine-grained classification, automated quality evaluation of domain-specific synthetic images, and expert-calibrated filtering that bridges automated and human judgment. While LTGC (Zhao et al., 2024) uses LLMs to guide generation and DisCL (Liang et al., 2025) proposes curricula for synthetic data, neither evaluates the generated images against domain-specific morphological criteria. While ImageReward (Xu et al., 2023) and LLMScore (Lu et al., 2023) advance automated quality evaluation, neither operates in a domain where taxonomic accuracy -- not aesthetic quality -- is the metric that matters. And while human evaluation is standard in medical imaging (Otani et al., 2023), no prior work uses expert annotations to *calibrate* an automated judge, creating a feedback loop between evaluation and generation. This thesis combines all three: morphology-guided generation, multi-dimensional automated evaluation, and expert-calibrated filtering -- applied to the problem of rare-species augmentation under extreme class imbalance.


## 3. Data and Experimental Setup

### 3.1 Data Collection

Images were acquired from the Global Biodiversity Information Facility (GBIF) using bplusplus (Venverloo & Duarte, 2024), an open-source insect detection and classification framework. The bplusplus.collect() module queries the GBIF Occurrence Search API via pygbif, downloading georeferenced photographs grouped by scientific name. I targeted all 16 Bombus species historically documented in Massachusetts, with downloads capped at 3,000 images per species. Quality filters were applied at query time:

| Filter | Value | Rationale |
|--------|-------|-----------|
| mediaType | StillImage | Photographs only (excludes audio, video) |
| basisOfRecord | HUMAN_OBSERVATION, MACHINE_OBSERVATION | Field observations only |
| lifeStage | Adult | Excludes larvae, pupae, and eggs to ensure consistent adult morphology |
| occurrenceStatus | PRESENT | Confirmed presence records only |
| year | 2010--2025 | Restricts to recent records with higher image resolution |

These filters yielded 21,900 images (as of February 2026). The five most common species reached the 3,000-image cap. Quality filters compound scarcity for rare species: requiring adult life stage, still images, and recent records eliminates a large fraction of the already-limited records for declining taxa.

### 3.2 Data Preparation Pipeline

Raw GBIF images were processed using the bplusplus prepare pipeline, which performs object detection and cropping in a three-stage process:

1. **YOLO detection.** Each image is passed through a YOLOv8 model (pretrained on GBIF insect imagery) with a confidence threshold of 0.35. Images with zero detections or multiple detections are discarded, enforcing a single-specimen-per-image constraint.
2. **Cropping and resizing.** Detected bounding box regions are cropped and resized to 640 x 640 pixels. This removes background-dominated images, multi-insect frames, and non-bee photographs that pass GBIF's taxonomic filters.
3. **Stratified splitting.** The prepared images are split 70/15/15 into train/validation/test sets, stratified by species to preserve class proportions. The test set (2,362 images) is held constant across all experimental conditions.

After preparation: 15,630 total images (10,933 train / 2,335 validation / 2,362 test). For each augmentation experiment, synthetic images are added only to the training set; the validation and test sets contain only real images and are identical across all conditions.

### 3.3 Species Taxonomy and Morphological Characteristics

Figure 3.1 shows the training set distribution across all 16 species after preparation. The distribution exhibits severe long-tail structure with an imbalance ratio of 59.9:1 (largest to smallest class) and a Gini coefficient of 0.377.

![Training set distribution](plots/species_distribution.png)
*Figure 3.1: Training set distribution across 16 Massachusetts Bombus species, sorted by count. Imbalance ratio (max/min class): 59.9:1. Gini coefficient of class frequencies: 0.377.*

![Raw GBIF counts](plots/gbif_raw_counts.png)
*Figure 3.2: Raw GBIF image counts before preparation (21,900 total). Five species reached the 3,000-image download cap.*

![Species samples](plots/species_samples.png)
*Figure 3.3: Sample images from the original dataset for each of the 16 species.*

I partition species into three tiers based on training set size, following a natural gap structure in the distribution where there are breaks in sample count. The "rare" tier (n < 200) comprises the three species selected for synthetic augmentation: B. ashtoni, B. sandersoni, and B. flavidus. The "moderate" tier (200 <= n <= 900) includes species with adequate but sub-optimal representation, while "common" species (n > 900) achieve consistently high performance.

Each target species poses a distinct challenge for generative image models because it deviates from the predominant Bombus phenotype -- a bright yellow thorax with a black abdomen -- characteristic of B. impatiens, the most commonly photographed North American bumblebee. The diagnostic morphology of each target species follows Williams et al. (2014) and Colla et al. (2011); full tergite-level color maps are provided in Appendix A.

**Bombus ashtoni** (Ashton's cuckoo bumble bee) is an obligate social parasite in the subgenus Psithyrus. Females are predominantly black with a diagnostic white-tipped abdomen (T4--T5 white, T6 black). As a cuckoo bee, B. ashtoni lacks corbiculae. The species has experienced severe range-wide decline with no confirmed Massachusetts records since 2008. *Generation challenge:* the model's prior strongly favors yellow-thorax bumblebees; B. ashtoni requires suppressing this prior entirely.

> **NOTE:** Bombus ashtoni has been synonymized under Bombus bohemicus based on molecular phylogenetic evidence. I retain "B. ashtoni" throughout because GBIF occurrence records index this taxon under that name.

**Bombus sandersoni** (Sanderson's bumble bee) is a small eusocial species (8--11 mm) with a clean two-tone pattern: yellow anterior (thorax through T2) and entirely black posterior (T3 onward). The abrupt color transition at T3 is the primary field character. *Generation challenge:* the two-tone pattern is relatively simple to generate (91.2% strict pass rate); the main difficulty is scale.

**Bombus flavidus** (yellowish cuckoo bumble bee) is a cuckoo bee with the most extensive yellow coloration among the three targets. A diagnostic yellow vertex distinguishes it from B. citrinus and B. fervidus. *Generation challenge:* the word "yellow" triggers bright lemon generation; all prompt instances require qualified descriptors ("dingy pale yellow," "cream"). Variable coloration makes consistency difficult (57.6% strict pass rate).

### 3.4 Classifier Architecture and Training Protocol

**Architecture.** ResNet-50 (He et al., 2015) pretrained on ImageNet (Deng et al., 2009), with the original 1000-class output layer replaced by a two-layer classification head: Linear(2048 -> 512) -> ReLU -> Dropout(0.5) -> Linear(512 -> 16). The backbone's final fully-connected layer is replaced with an identity mapping so that the 2048-dimensional feature vector passes directly to the classification head. All parameters -- including the backbone convolutional layers -- are fine-tuned end-to-end; no layers are frozen.

**Training.** Adam optimizer (Kingma & Ba, 2017), learning rate 1 x 10^-4, weight decay 0; regularization is provided by dropout (0.5) and early stopping. Batch size 8, maximum 100 epochs with early stopping (patience 15, monitored on validation loss). Learning rate is reduced on plateau (factor 0.5, patience 5). Standard augmentation: random horizontal flip, color jitter, normalization to ImageNet statistics. Images are resized to 640 x 640 pixels. Model selection uses the checkpoint with the best validation macro F1.

**Dataset versions.** The same architecture and training protocol are applied to all dataset conditions. In each augmentation experiment, only the training set changes; the validation and test sets remain identical.

| Dataset | Description | Training images added |
|---------|-------------|----------------------|
| D1 (Baseline) | Real images only | -- |
| D3 (CNP) | Real + Copy-Paste augmented | +200 per rare species |
| D4 (Synthetic) | Real + unfiltered synthetic | +200 per rare species |
| D5 (LLM-filtered) | Real + LLM-filtered synthetic | +200 per rare species |
| D6 (Expert-filtered) | Real + expert-calibrated filtered synthetic | +200 per rare species [TODO] |

### 3.5 Evaluation Methodology

**Metrics.** Classification performance is assessed at multiple granularities:
- *Overall*: accuracy, macro F1 (unweighted average across 16 species), weighted F1
- *Per-species*: precision, recall, F1, and support for each of the 16 species, with particular attention to the three rare target species
- *Confusion analysis*: row-normalized confusion matrices to identify systematic misclassification patterns between morphologically similar species

**Statistical validation.** Small test-set sizes for rare species (as few as n = 6 under the 70/15/15 split) make single-run evaluation unreliable. Three complementary protocols address different sources of variance, and each is used for the claim it most directly supports:

1. **Single-split preliminary evaluation.** A single training run on the fixed 70/15/15 split is reported for each augmentation variant as an honest record of the early experiments. This protocol anchors the baseline confusion structure and provides provisional rankings, but has too few rare-species test images (n = 6, 10, 36) to support significance claims.

2. **Stratified five-fold cross-validation (primary aggregate protocol).** Each fold trains independently with the same augmentation protocol, so that every real specimen is tested exactly once across folds. Pooling predictions across folds yields rare-species effective test sizes of n = 32 (B. ashtoni), n = 58 (B. sandersoni), and n = 232 (B. flavidus) -- a fivefold increase over the single split. This design follows Shipard et al. (2023) and Picek et al. (2022), who adopt k-fold CV for synthetic-augmentation evaluation on small biological image datasets. Paired t-tests on fold-level macro F1 (df = 4) provide pairwise significance. With df = 4 the tests require large effect sizes for 80 % power at alpha = 0.05, so non-significant comparisons should be read as underpowered rather than as evidence of equivalence.

3. **Multi-seed training on the fixed split (primary per-image protocol).** Deep learning models are sensitive to random initialization, data shuffling, and dropout masks. Training each dataset version with five random seeds (42--46) on the same fixed train/validation/test split isolates training stochasticity from data variation, and -- because every seed evaluates the same 2,362 test images -- is the only protocol that supports per-image analyses such as prediction-flip tracking and embedding-space failure-chain retrieval (Section 5.5). Paired t-tests on seed-level macro F1 provide pairwise significance between dataset versions.

4. **Bootstrap confidence intervals.** Per-species F1 and macro F1 are computed with 10,000 bootstrap resamples of the test-set predictions, producing 95 % CIs that reflect evaluation uncertainty given the fixed test set (for single-split and multi-seed) or the pooled cross-validation predictions (for 5-fold CV). For rare species (B. ashtoni n = 6, B. sandersoni n = 10 on the fixed split), these CIs are inherently wide -- an honest reflection of the evaluation challenge, not a limitation to be engineered away. Bootstrap CIs with n < 10 should be interpreted as indicative rather than definitive, and are reported alongside all point estimates throughout Section 5.

Five-fold cross-validation is designated the primary protocol for aggregate and per-species F1 claims because its larger effective rare test set produces more stable estimates than the fixed split, while the fixed-split multi-seed protocol is retained specifically for per-image analyses that cross-fold evaluation cannot support. The two protocols are reconciled in Section 5.5 when they produce different aggregate rankings.

All reported results use the best-validation-macro-F1 checkpoint (best_f1.pt), matching the primary reporting metric.


## 4. Methods

### 4.1 Copy-Paste Augmentation

Copy-Paste augmentation (CNP) follows the approach of Ghiasi et al. (2021), generating new training images by compositing real bee specimens onto varied backgrounds. For each target species, the Segment Anything Model (SAM ViT-H; Kirillov et al., 2023) extracts foreground masks from existing training images. Segmented specimens are then composited onto flower background images with random affine transforms (rotation, scaling, horizontal flip) and Gaussian boundary blending to reduce edge artifacts.

CNP preserves authentic morphological texture -- every pixel of the bee is from a real photograph -- while varying the background context. This makes it a strong baseline: any improvement from generative augmentation must exceed what can be achieved simply by re-placing real specimens in new scenes. The limitation is a diversity ceiling: with only 22 training images for B. ashtoni, CNP can produce new compositions but cannot introduce morphological variation (poses, angles, lighting on the specimen) beyond what exists in the source images.

### 4.2 Structured Morphological Prompting

Text-to-image generative models encode strong priors toward the most common visual archetype within a category. For the genus Bombus, this prior is the phenotype of B. impatiens -- a bright yellow thorax with a black abdomen -- which accounts for roughly 12% of all GBIF Bombus images globally. Without explicit intervention, every generated bumblebee converges toward this archetype regardless of the species specified in the prompt. This is particularly damaging for the three target species, all of which deviate substantially from the common phenotype.

#### 4.2.1 Prompt Template Architecture

Each generation prompt is assembled from a template with eight placeholders filled programmatically from the species configuration (configs/species_config.json):

| Placeholder | Content |
|-------------|---------|
| {species_name} | Scientific name (e.g., Bombus bohemicus (inc. B. ashtoni)) |
| {common_name} | Vernacular name |
| {caste_description} | Caste-specific morphology with front-to-back color map |
| {morphological_description} | Full species description with negative constraints |
| {view_angle} | Lateral, dorsal, frontal, three-quarter anterior/posterior |
| {wings_style} | Folded at rest or slightly spread |
| {environment_description} | Habitat scene with lighting conditions |
| {scale_instruction} | Proportional size anchors relative to flowers |

The template is organized into 10 sections: (1) role assignment as a research entomologist, (2) task specification referencing input photographs, (3) subject identity via caste description, (4) morphological notes, (5) framing and composition constraints, (6) pose and viewpoint, (7) lighting, (8) background, (9) flower realism, and (10) quality constraints. This modular structure allows independent variation of each dimension -- the same species morphology is rendered across 5 view angles, 15 environments, and multiple castes, producing diverse training images from a single morphological specification.

#### 4.2.2 Four Key Design Principles

Four design principles emerged through iterative prompt development (full iteration history in Appendix B):

1. **Negative constraints.** The single most impactful intervention. Each species description leads with `WARNING:` stating what *not* to generate (e.g., "Do NOT generate a bee with a bright yellow thorax"). Without this, positive descriptions alone were insufficient to override the model's prior.

2. **Front-to-back color maps.** Rather than holistic descriptions ("mostly black"), coloration is specified tergite-by-tergite in spatial sequence: "BLACK face -> BLACK thorax -> BLACK T1 -> BLACK T2 -> WHITE T4 -> WHITE T5 -> BLACK T6." Both the species-level and caste-specific descriptions include this map, providing redundant spatial guidance.

3. **Proportional scale anchors.** The model ignores absolute measurements and frame percentages. Scale is anchored to in-scene objects: "the bee's body is shorter than a single daisy petal."

4. **Macro crop framing.** The prompt enforces that the bee fills 40--60% of the frame with shallow depth-of-field and cropped flowers to match the training distribution.

[TODO] Figure 4.1: Example generation outputs showing the effect of structured prompting (e.g., v3 without negative constraints vs. v8 with them, same species)

#### 4.2.3 Reference-Guided Generation

Images are generated using the OpenAI images.edit endpoint (GPT-image-1.5), which accepts 1--2 reference photographs per species alongside the text prompt. We use GPT-image-1.5 for three reasons: (1) superior zero-shot compositional accuracy (GenEval 0.84 vs. 0.62 for Stable Diffusion 3; Yu et al., 2025), (2) reference-guided editing that conditions on real specimen photographs, and (3) support for long structured prompts (32K characters) that encode tergite-level morphological specifications -- far exceeding the 77-token limit of CLIP-conditioned diffusion models. The tradeoff is that GPT-image-1.5 is a closed model that cannot be fine-tuned on domain-specific data, a limitation discussed in Section 6.7.

Two implementation details proved critical:

- **SDK encoding.** The OpenAI Python SDK's extract_files method silently drops bare Path objects. References must be passed as (filename, BytesIO, mime_type) tuples; without this fix, reference images have no effect and no error is raised.
- **Input fidelity.** The `input_fidelity="high"` parameter is required. At the default (low) fidelity, reference images have no visible effect on generation output.

Caste selection is weighted 3:1 toward workers (eusocial species) or females (cuckoo species) to approximate field photo distributions. Environments are sampled randomly, not cycled deterministically, to prevent spurious environment-species correlations. Full details of caste-aware generation and environmental variation are provided in Appendix B.

### 4.3 LLM-as-Judge Quality Evaluation

Evaluating whether a synthetic image is useful for classifier training requires assessing not just visual realism but taxonomic accuracy of diagnostic features. Standard metrics (FID, IS) are inadequate for this purpose (Section 2.5). I implement a two-stage evaluation protocol using GPT-4o with structured output via Pydantic schema enforcement.

**Stage 1: Blind Taxonomic Identification.** The judge receives only the image -- no target species label -- and must identify it to species level from the 16-species Massachusetts Bombus list. This tests whether the generated image carries enough diagnostic information for correct identification. The judge may also output "Unknown" (ambiguous) or "No match" (not a bumblebee).

**Stage 2: Detailed Morphological Evaluation.** After receiving the target species and expected morphological traits, the judge evaluates five features on a 1--5 anchored scale:

| Feature | What it captures |
|---------|-----------------|
| Legs/Appendages | Correct count (6), proportions, corbiculae presence/absence |
| Wing Venation/Texture | Wing shape, transparency, vein pattern |
| Head/Antennae | Antenna segmentation, eye shape, mouthparts |
| Abdomen Banding | Tergite color pattern matching species description (critical) |
| Thorax Coloration | Pile color and pattern matching species description (critical) |

Score anchors: 1 = Poor (anatomically impossible), 2 = Below fair (notable inaccuracies), 3 = Fair (minor imperfections), 4 = Good (subtle issues only), 5 = Excellent (photorealistic match). The judge additionally assesses diagnostic completeness (species/genus/family/none), caste fidelity, and failure modes (wrong coloration, extra/missing limbs, impossible geometry, blurry artifacts, background bleed, repetitive patterns).

[TODO] Figure 4.2: One pass and one fail example with the judge's per-feature scores overlaid

**Pass/fail rules.** The judge applies a lenient holistic rule (overall_pass: mean morph >= 3.0, diagnostic >= genus, no structural failures). For dataset assembly, a stricter filter is applied: (1) blind ID matches target species, (2) diagnostic completeness = "species", (3) mean morphological score >= 4.0. Preliminary evaluation showed that the lenient holistic rule passes nearly all generated images, while the stricter per-feature criteria reject a substantial fraction -- confirming that per-feature evaluation is necessary. Detailed results are presented in Section 5.2.

**Implementation.** The judge uses chain-of-thought prompting: instead of directly outputting scores, the model first reasons step-by-step through each evaluation criterion, then produces the score. This two-step process (reason first, score second) produces more calibrated and consistent evaluations than direct scoring. Calibration guidance anchors the standard to practical entomological assessment ("apply the standard of a working entomologist reviewing field photographs -- not the standard of a museum specimen plate"). All outputs are parsed into Pydantic models for type-safe downstream processing. Full judge prompt and schema are provided in Appendix C.

### 4.4 Expert-Calibrated Quality Filtering

The LLM judge (Section 4.3) provides the raw evaluation signal -- per-feature scores that decompose quality into interpretable dimensions. However, it applies a holistic pass/fail rule that weights all features equally. Domain expertise suggests this is incorrect: for B. ashtoni, abdomen banding (the white T4--T5 tail) is the single most important field mark, while legs and wings contribute little to species identification. Expert calibration (this section) learns the correct diagnostic weighting from entomologist annotations, transforming the judge from a noisy holistic assessor into a calibrated domain-specific instrument.

#### 4.4.1 Stratified Annotation Design

150 synthetic images are selected for expert annotation (50 per species), stratified across four quality tiers defined by the LLM judge output:

| Tier | Definition | Purpose |
|------|-----------|---------|
| strict_pass | matches_target + diag=species + morph >= 4.0 | Validate that the filter accepts good images |
| borderline | matches_target + diag=species + 3.0 <= morph < 4.0 | Calibrate whether the 4.0 threshold is correct |
| soft_fail | matches_target + diag < species | Assess if the judge is too strict on completeness |
| hard_fail | NOT matches_target | Confirm correct rejection |

Within each species, allocation uses a floor-then-proportional strategy: each non-empty tier receives a minimum of 5 images (guaranteeing representation), with remaining slots distributed proportionally to tier pool size.

#### 4.4.2 Human-in-the-Loop Evaluation Interface

To collect expert annotations, I developed a web-based evaluation application consisting of a backend server and two annotation interfaces that mirror the LLM judge's two-stage protocol. Unlike pairwise preference approaches such as ImageReward (Xu et al., 2023), which learn from relative comparisons between image pairs, this interface collects absolute per-feature scores on individual images -- enabling direct feature-level comparison between expert and LLM judgments on the same rubric.

**Annotator.** The expert annotator is a PhD-trained biologist and pollinator specialist affiliated with The Biodiversity Lab (thebiodiversitylab.com); the individual's name and further biographical details are withheld to preserve annotator anonymity. Owing to recruitment and scheduling constraints within the thesis timeline, a single expert completed the full 150-image annotation campaign, and all LLM--expert agreement results reported in Section 5.4 therefore reflect one calibrated rater rather than an inter-annotator consensus. Expanding the annotator pool to support inter-rater reliability and per-feature agreement statistics is listed as future work in Section 7.2.

**Stage 1: Blind Identification Interface.** The expert sees only the synthetic image at full resolution -- no target species label, no LLM judge output. The interface presents the 16-species identification panel and the expert selects the most likely species (or "Unknown" / "No match"). This stage tests whether the generated image carries sufficient diagnostic information for independent identification.

**Stage 2: Detailed Evaluation Interface.** After submitting the blind identification, the target species and its diagnostic criteria card are revealed. The expert then scores the same five morphological features on the same 1--5 anchored scale used by the LLM judge. The interface additionally collects diagnostic completeness, failure mode checkboxes, caste fidelity assessment, and an overall PASS / FAIL / UNCERTAIN judgment. Museum-quality reference images (3 per species) are displayed alongside the evaluation image throughout this stage.

[TODO] Figure 4.3: Screenshots of the two-stage evaluation interface

This design -- absolute scoring on a shared rubric rather than pairwise preference -- is motivated by the need for per-feature disagreement analysis (Section 4.4.4). Pairwise preferences can reveal which image is *better* but cannot reveal *why* -- i.e., which specific morphological feature the expert considers incorrect. The shared rubric enables the 2x2 disagreement matrices that drive the diagnostic feedback loop.

#### 4.4.3 Expert label derivation

Every annotated image yields two binary labels derived from the raw interface output, matching the two gates the LLM judge itself applies in Section 4.3 so that probe AUC and LLM AUC are computed on the same rubric.

- **Lenient rule.** No structural failure is flagged (modes `extra_limbs`, `missing_limbs`, `impossible_geometry`, `visible_artifact`, `blurry_artifacts`, `repetitive_patterns`), the expert's diagnostic level is `genus` or `species`, and the mean of the five morphological feature scores is ≥ 3.0.
- **Strict rule.** The expert's blind-identification species matches the ground-truth species, the diagnostic level is `species`, and the mean morphological score is ≥ 4.0.

On the 150-image sample, the lenient rule selects 128 / 150 (85.3 %) and the strict rule selects 50 / 150 (33.3 %). Per-species strict pass rates are 17 / 50 (B. ashtoni), 27 / 50 (B. sandersoni), and 6 / 50 (B. flavidus) — the heterogeneity motivates the per-species thresholding in Section 4.4.5.

#### 4.4.4 Three filter families

Three filters with progressively more supervision are compared. Every filter is applied to the full pool of 1,500 generated synthetics (500 per rare species) and produces a per-image pass flag plus a continuous score. The dataset-level labels used in Section 5 are D4 (LLM-rule filter), D5 (BioCLIP centroid filter), and D6 (expert-supervised probe); all three operate over the same D3 pool, so any downstream difference is attributable to filter selection alone.

**D4 — LLM-rule filter.** The strict threshold from Section 4.3: `matches_target AND diag=species AND mean_morph >= 4.0`. Automated, scalable, and requires no visual-feature machinery, but weights all five morphological features equally and can only use information the LLM judge chose to verbalise.

**D5 — BioCLIP centroid filter.** For each rare species s, fit a single L2-normalised centroid μ_s = mean(BioCLIP(x)) / ‖·‖ from the 10,933 real training images restricted to that species. A synthetic image f(x) with target species s passes if cos(f(x), μ_s) ≥ τ_s. BioCLIP ViT-B/16 was chosen over DINOv2 ViT-L/14 because its biology-specific pre-training yields substantially better per-species 5-NN purity on this dataset (Section 5.2.1). The threshold rule used by the deployed variant is Option B, **τ_s = median synthetic-to-centroid cosine of species s** — an unsupervised, per-species cut that passes the "above-median half" of the generated pool relative to its own species centroid. Numerically, τ_ashtoni = 0.6945, τ_sandersoni = 0.7468, τ_flavidus = 0.6817; each passes 250 / 500 before any volume cap. The alternative "median real-to-real" rule (τ = 0.765 / 0.806 / 0.753) would pass only 46 / 79 / 29 and was therefore held back as a diagnostic reference rather than the shipped filter.

**D6 — Expert-supervised linear probe.** A scikit-learn `Pipeline(StandardScaler, LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=4000))` trained on the 150 expert labels to predict the strict expert rule. Four feature representations are ablated (Section 4.4.5): BioCLIP alone (512-d), LLM per-image features alone (8-d: five morph scores + blind_match + diag_species + diag_genus indicators), their concatenation (520-d), and concatenation plus a species one-hot (523-d). The regularisation strength C ∈ {0.001, 0.01, 0.1, 1, 10} is chosen by nested 5-fold stratified CV maximising mean-fold AUC-ROC; after the best C is selected on the 150 labels, a leave-one-out CV re-computes the LOOCV AUC reported in Section 5.4. Training labels are fixed across the pipeline — the probe never sees fold-specific real training images, which is why the same 200-image selection per rare species is reused across the 5 CV folds (Section 4.4.7).

The three filters therefore bracket the supervision spectrum: D4 uses language-mediated scoring with no expert data, D5 uses visual features with no expert data, D6 uses visual features *with* 150 expert labels. Any downstream gap between D4 and D5 isolates the value of looking at visual features; any gap between D5 and D6 isolates the value of expert supervision per se.

#### 4.4.5 Feature-configuration ablation and per-species threshold learning

The four probe feature configurations are evaluated under identical nested-CV and LOOCV protocols on the 150-image sample. Under the strict expert rule, BioCLIP alone wins with LOOCV AUC 0.792 (C = 0.01), with `bioclip+llm+species` (0.790) and `bioclip+llm` (0.787) trailing by margins within LOOCV noise and `llm`-alone lagging at 0.649. Adding the 8 noisy LLM feature columns at n = 150 therefore hurts rather than helps. D6 ships with the `bioclip` config.

The strict expert pass rate is highly heterogeneous across the three rare species (54 % sandersoni, 34 % ashtoni, 12 % flavidus), so a single probability threshold would either over-admit flavidus or under-admit sandersoni. For each species, the pass threshold τ_s is chosen to maximise F1 on the species-restricted LOOCV pass-probability predictions under the strict rule, with candidate τ values drawn from the union of all LOOCV predictions for that species plus {0.0, 0.5, 1.0}. The learned per-species thresholds are τ_ashtoni = 0.125, τ_sandersoni = 0.495, τ_flavidus = 0.222; the corresponding per-species LOOCV F1 values are 0.62, 0.87, and 0.32. The low flavidus F1 reflects the small expert-positive support (6 / 50) rather than probe miscalibration.

#### 4.4.6 Selection rule and volume parity

Downstream augmentation uses a fixed +200-image budget per rare species (matching D2 CNP and D3 unfiltered synthetic) so that filter choice does not trivially confound volume. Each filter scores all 500 synthetics per species and returns the threshold-pass subset; if the pass set exceeds 200, the top-200 by score are retained. Under D5 all three species have 250 pre-cap pass images (by construction of the median-synth rule) that land at 200 after capping. Under D6 the per-species strict thresholds yield 418 / 364 / 204 pre-cap pass images (B. ashtoni / B. sandersoni / B. flavidus); all three cap at 200. D4 uses the LLM strict-pass subset directly, which is 218 / 456 / 288 pre-cap and also caps at 200. All three filters therefore add exactly 200 synthetics per rare species to the real training set in both the fixed-split and the 5-fold variants.

#### 4.4.7 Evaluation protocols

Downstream classifier performance is reported under two protocols. **Multi-seed × fixed split:** the 70 / 15 / 15 split (Section 3.4) is held constant and the ResNet-50 is trained under five seeds {42, 43, 44, 45, 46} per dataset, enabling per-image flip analysis (Section 5.6). **5-fold cross-validation × single seed:** one seed per fold reduces random-seed variance and pools roughly five times more rare-species test predictions for statistical comparisons (Section 5.5.2). For D5 and D6 the per-fold training directories reuse the same 200-image selection per species — filter selection is not fold-dependent, because neither the real-centroid fit nor the probe training set varies with fold.

#### 4.4.8 Filter-level evaluation

Filter quality is assessed at three levels, reported in Section 5.4.

- **Filter accuracy.** LOOCV on the 150 expert-annotated images (AUC-ROC against the strict rule, plus per-species F1 at the learned τ) measures how well each filter predicts expert pass/fail.
- **Downstream classification impact.** Each filter produces a dataset variant (D4, D5, D6); the classifier is retrained from the same ImageNet-initialised weights under identical hyperparameters and macro F1, per-tier F1, and focus-species F1 are compared against D1 / D2 / D3 (Section 5.5).
- **Representational alignment (planned).** CKNNA (Huh et al., 2024) between each 200-image filtered selection and the real training set in BioCLIP space, per species, quantifies whether the filter preserves the neighbourhood structure of real images better than unfiltered sampling. CKNNA computation is listed in Section 5.4.6 as [TODO: pending completion of the D5 / D6 training-result aggregation].

The classifier architecture and training protocol (Section 3.4) are unchanged across filters — only which synthetics enter training changes.

#### 4.4.9 Diagnostic Feedback Loop

Per-feature disagreement between the LLM judge and experts is analysed using 2 × 2 matrices (LLM score ≥ 4 vs. < 4 × expert score ≥ 4 vs. < 4) for each of the five morphological features:

- **LLM blind spots** (LLM ≥ 4, expert < 4): features where the LLM cannot see the error — these inform generation prompt refinement (e.g., if experts flag thorax coloration that the LLM missed, the prompt's colour map for that species is revised).
- **LLM over-strictness** (LLM < 4, expert ≥ 4): features where the LLM is too conservative — these inform judge rubric recalibration.

This feedback loop connects evaluation to generation: expert disagreement patterns flow back into both the prompting framework (Section 4.2) and the judge rubric (Section 4.3). The observed per-feature disagreement rates are reported in Section 5.4.2.


## 5. Experiments and Results

Section 5 develops the empirical case along a single arc. Section 5.1 establishes baseline classifier behaviour and the rare-species confusion directions every subsequent analysis must respect. Section 5.2 characterises the BioCLIP feature-space geometry of real and synthetic images and shows that generated synthetics sit in a per-species region of feature space displaced from their real-image counterparts. Section 5.3 reports the LLM judge's output over all 1,500 generated synthetics and characterises its calibration against the images it passes. Section 5.4 reports the expert-annotation study and the resulting calibrated filter (D5 centroid, D6 probe), including per-image agreement between the LLM and expert, and filter-level comparisons on the 1,500-image pool. Section 5.5 reports downstream classifier performance across six dataset variants under three protocols — D1 baseline, D2 copy-and-paste, D3 unfiltered synthetic, D4 LLM-filtered, D5 centroid-filtered, D6 expert-probe-filtered. Section 5.6 closes with a per-image mechanistic analysis that isolates why LLM-filtering fails to rescue rare-species F1. D5 and D6 downstream training runs are still in progress at submission; Section 5.5's table and Section 5.6's D5 / D6 chain analyses therefore carry `[TODO]` placeholders where those numbers belong.

Throughout Section 5, five-fold cross-validation is the primary protocol for aggregate and per-species claims, following the rationale in Section 3.5. Multi-seed training on the fixed split is used for per-image analyses in Section 5.6 because each seed evaluates the same 2,362 test images. Single-split results are reported alongside the primary protocols as an honest record of the early experiments. All results use the best-validation-macro-F1 checkpoint.

**Dataset naming.** Throughout Section 5 and Section 6, dataset variants are referred to by the thesis label D1–D6:

| Thesis label | Variant | Source dir |
|---|---|---|
| D1 | ResNet-50 baseline (no augmentation) | `prepared_baseline*` |
| D2 | Copy-and-paste augmentation | `prepared_d2_cnp*` |
| D3 | Unfiltered generative synthetic | `prepared_d3_synthetic*` |
| D4 | LLM-rule filtered synthetic | `prepared_d5_llm_filtered*` |
| D5 | BioCLIP-centroid filtered synthetic | `prepared_d2_centroid*` |
| D6 | Expert-supervised probe filtered synthetic | `prepared_d6_probe*` |

The `source dir` column preserves legacy code-level keys; plots, captions, and filenames retaining the old tokens (e.g. `chains_d4_harmed/` for what Section 5.6 now calls D3 chains) are annotated inline.

### 5.1 Baseline

#### 5.1.1 Single-run classifier performance

Table 5.1 reports the ResNet-50 baseline on the fixed 70/15/15 split with 10,000-iteration bootstrap 95 % confidence intervals. Overall accuracy reaches 88.2 % and macro F1 reaches 0.815, figures driven by the eleven head-and-moderate species with n ≥ 200 training images. The three rare targets fall substantially below this aggregate: B. ashtoni reaches F1 0.500 (n = 6 test, 95 % CI [0.000, 0.818]), B. sandersoni 0.588 (n = 10, [0.222, 0.833]), and B. flavidus 0.623 (n = 36, [0.462, 0.754]). The B. ashtoni interval spans more than 0.8 of the F1 range — an honest reflection of evaluation variance at n = 6 that no single-run comparison can narrow.

*Table 5.1: Baseline ResNet-50 classifier on the fixed split, f1 checkpoint (10,000-iteration bootstrap 95 % CI). Rare species in bold.*

| Species | Train n | Test n | Precision | Recall | F1 | 95% CI |
|---------|---------|--------|-----------|--------|-----|--------|
| **B. ashtoni** | **22** | **6** | 0.500 | 0.500 | **0.500** | [0.000, 0.818] |
| **B. sandersoni** | **40** | **10** | 0.714 | 0.500 | **0.588** | [0.222, 0.833] |
| **B. flavidus** | **162** | **36** | 0.760 | 0.528 | **0.623** | [0.462, 0.754] |
| Macro average | -- | 2,362 | -- | -- | 0.815 | [0.774, 0.845] |
| Overall accuracy | -- | 2,362 | -- | -- | 0.882 | -- |

Per-species F1 correlates cleanly with training-set size: every species with n ≥ 200 exceeds 0.75 F1, while every species below n = 200 falls under 0.65. The three rare species therefore define the augmentation target and determine every subsequent comparison. Figure 5.1a visualises the per-species F1 with bootstrap CIs; the three rare species sit visibly below the cluster of moderate and common species, with B. ashtoni's CI spanning the largest fraction of the F1 range. Figure 5.1b decomposes the same per-species performance into precision, recall, and F1 bars: rare-species recall is the dominant deficit (B. ashtoni 0.500, B. sandersoni 0.500, B. flavidus 0.528) while precision is comparable to or exceeds 0.50 — the baseline classifier is more conservative than wrong on rare species, missing them rather than confusing other classes for them.

![Baseline per-species F1 with 95 % bootstrap CI](plots/baseline_f1_ci.png)
*Figure 5.1a: Per-species F1 with 10,000-iteration bootstrap 95 % CI on the baseline ResNet-50, sorted by descending test support. The CI for B. ashtoni spans more than 0.8 of the F1 range — at n = 6 test images, single-run F1 is a sample of one outcome rather than a stable estimate.*

![Baseline per-species precision/recall/F1](plots/baseline_species_metrics.png)
*Figure 5.1b: Per-species precision, recall, and F1 on the baseline classifier. The rare-tier deficit is concentrated in recall: every rare species has precision ≥ 0.50 but recall ≤ 0.53.*

#### 5.1.2 Rare-species confusion structure

The row-normalized baseline confusion matrix (Figure 5.1c) shows three qualitatively different rare-species error patterns. B. sandersoni produces five errors over its ten test images, four of which are predicted as B. vagans (40 % of test support, 80 % of errors); the remaining error is B. rufocinctus. The vagans confusion direction is therefore singular and dominant for B. sandersoni, and reflects their shared yellow-anterior / black-posterior body pattern. B. flavidus produces seventeen errors over thirty-six test images and these spread across seven distinct species: B. rufocinctus (5 errors, 13.9 %), B. ternarius (3, 8.3 %), B. citrinus (3, 8.3 %), B. ashtoni (2), B. bimaculatus (2), B. griseocollis (1), and B. terricola (1). No single confuser dominates; the misclassifications cluster among medium-sized yellow-and-black species in the genus, consistent with the diffuse-yellow phenotype that motivated the prompt-engineering work in Section 4.2. B. ashtoni produces three errors over six test images, each going to a different species: B. pensylvanicus, B. terricola, and B. flavidus. With n = 6 the confusion structure for B. ashtoni is statistically uninformative — a single image flipping to a different predicted class would change the apparent dominant confuser entirely. These three patterns — a single dominant confuser (B. sandersoni vs. B. vagans), diffuse confusion across many species (B. flavidus), and statistically-undetermined confusion (B. ashtoni) — provide the visual priors against which any synthetic augmentation must succeed and reappear as the anchor for the failure-chain retrievals in Section 5.6.

![Baseline confusion matrix](plots/baseline_confusion_matrix.png)
*Figure 5.1c: Row-normalised baseline confusion matrix on the fixed split (macro F1 0.815). Row entries are conditional probabilities given the true label; cells with row probability < 0.05 are blank for legibility.*

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

Figure 5.2a projects all 10,933 real training images into a BioCLIP t-SNE. Common species form compact, well-separated clusters; B. impatiens, B. ternarius, and B. griseocollis are clearly resolved. The rare species do not enjoy this separation. In the same projection, B. flavidus scatters across a broad region, and B. ashtoni and B. sandersoni form small sub-clusters embedded within neighbourhoods occupied by their confusers from Section 5.1.2 (B. vagans is co-located with the sandersoni region, consistent with the four-of-five sandersoni errors that go to vagans). Figure 5.2b refits the t-SNE on the rare species alone (no other classes, no synthetics), as a sharper test of whether the three rare species are at least separable from each other in BioCLIP space; they are not -- B. flavidus dominates the projection and B. ashtoni and B. sandersoni interleave with flavidus rather than forming distinct islands. This indistinguishability pre-exists any synthetic-augmentation question -- it is the native problem augmentation must address.

![16-species real-image t-SNE](plots/embeddings/bioclip_tsne/embeddings_overview.png)
*Figure 5.2a: BioCLIP t-SNE of 10,933 real training images, 16 species. Common species form compact clusters; rare species overlap with their visual confusers (e.g. B. sandersoni with B. vagans).*

![Rare-species-only real-image t-SNE](plots/embeddings/bioclip_tsne/embeddings_rare_real_only.png)
*Figure 5.2b: BioCLIP t-SNE refit on the three rare species alone (no other classes, no synthetics). Even in isolation the rare species do not separate into three distinct clusters.*

#### 5.2.3 The synthetic-real embedding gap

Projecting real and synthetic images for the three rare species into a shared t-SNE space (Figure 5.3a) reveals the central structural finding of Section 5.2: for each rare species, the synthetic images form their own tight cluster, visibly separated from the cluster of real images of the same species. The separation is specific to each species rather than a single generic "synthetic versus real" displacement -- the synthetic B. ashtoni cluster sits in a different region from the synthetic B. sandersoni cluster, which sits in a different region from the synthetic B. flavidus cluster, and each one is displaced from its own real-image counterpart rather than pooled into a common "synthetic" zone.

Figure 5.3b quantifies the displacement directly. For each synthetic image I compute its cosine distance to the centroid of its target species' real training embeddings (the centroid is simply the mean of the L2-normalised feature vectors for that species' real images). The median synthetic-to-centroid distance is 0.31 for B. ashtoni, 0.25 for B. sandersoni, and 0.32 for B. flavidus. For comparison, the median real-to-centroid distance within each rare species falls in the 0.10--0.20 range -- roughly half as far. A typical synthetic image of B. ashtoni, as measured in BioCLIP feature space, is therefore about twice as far from the centre of the real B. ashtoni cluster as a typical real image of B. ashtoni is. The same pattern holds for the other two species. A representative confusion-pair triplet and the embedding atlas with thumbnails at true t-SNE coordinates (Appendix F) confirm visually that the clusters are coherent in pose and coloration rather than projection artefacts.

![Rare real + synthetic t-SNE](plots/embeddings/bioclip_tsne/embeddings_rare_real_synth.png)
*Figure 5.3a: BioCLIP t-SNE of rare-species real and synthetic images. Each synthetic cluster sits in a different region of the projection from its real-image counterpart of the same species.*

![Synthetic-to-centroid cosine distance](plots/embeddings/bioclip_tsne/embeddings_centroid_distance.png)
*Figure 5.3b: Per-synthetic cosine distance to the species' real-image centroid. Dashed lines mark the median real-to-centroid distance within each rare species, for comparison.*

This displacement has a direct downstream prediction: a classifier trained on these synthetics learns species-discriminative features tuned to a region of feature space that the real test images of the same species do not occupy. Section 5.5 tests that prediction against macro F1, and Section 5.6 traces the per-image consequences.

#### 5.2.4 Image-level embedding atlas

Projections on their own can mislead — a dense rare-species cluster could be an artefact of t-SNE crowding rather than true feature-space proximity. To rule this out and to expose the visual content of each region directly, Figure 5.4 plots thumbnails of every real image at its true t-SNE coordinate, once for the full 16-species pool and once restricted to the three rare species. Two observations hold. First, the visual coherence of each cluster matches the species label — common-species regions are populated by visually similar specimens in similar poses, confirming that the t-SNE is not collapsing unrelated images under projection pressure. Second, the rare-species atlas shows B. flavidus thumbnails scattered widely across several loose sub-regions rather than one compact cluster, consistent with the diffuse confusion pattern in Section 5.1.2 and with B. flavidus's known caste-variable coloration. These atlases anchor every centroid-distance and nearest-neighbour claim in Sections 5.2.3, 5.4, and 5.6 to visible image content, not to abstract coordinates.

![Embedding atlas all-species](plots/failure/embedding_atlas_all_tsne.png)
*Figure 5.4a: Real-image thumbnail atlas over the full 16-species BioCLIP t-SNE. Visually similar specimens cluster; the projection is not crowding unrelated images together.*

![Embedding atlas rare-only](plots/failure/embedding_atlas_rare_tsne.png)
*Figure 5.4b: Thumbnail atlas for the three rare species alone. B. flavidus scatters across multiple sub-regions rather than forming one compact cluster.*

### 5.3 LLM-as-Judge Results

The LLM-as-judge (Section 4.3) evaluates every generated image on species-level morphology. Section 5.3 characterises what the judge sees across all 1,500 synthetics, including the funnel it applies, the per-feature scores it produces, and the failure-mode profile it flags. Downstream F1 consequences of applying this filter to training data — the "D4 row" — are reported in Section 5.5; Section 5.4 returns to whether the judge's pass set aligns with expert judgment on the same images.

#### 5.3.1 Pass rates and filter funnel

Table 5.3 reports the two-stage judge's output over all 1,500 generated images (500 per rare species). Blind-identification match rates are high for B. sandersoni (96.4 %) and B. flavidus (96.0 %) but much lower for B. ashtoni (76.0 %), reflecting the inverse-phenotype challenge: B. ashtoni's predominantly black thorax inverts the dominant Bombus prior, and the judge — like the generation model — sometimes defaults to yellow-thorax interpretations. The mean morphological score follows the same pattern (B. ashtoni 3.82, B. flavidus 4.06, B. sandersoni 4.37), as does the strict pass rate: 44.4 % B. ashtoni, 57.6 % B. flavidus, 91.2 % B. sandersoni. The strict funnel reduces the 1,500-image pool through three sequential gates — blind-ID match (1,342 images, 89.5 %), diagnostic completeness at species level (1,060, 70.7 %), and mean morphological score ≥ 4.0 (966, 64.4 %). Figure 5.3a shows the four LLM tier outcomes (strict pass / borderline / soft fail / hard fail) per species, and Figure 5.3b decomposes the funnel into its per-gate retention rates.

*Table 5.3: LLM-as-judge evaluation of 1,500 generated images (500 per species). Strict pass requires blind-ID match AND diagnostic = species AND mean morphological score ≥ 4.0.*

| Species | Blind ID | Mean morph | Lenient pass | Strict pass |
|---------|---------:|-----------:|-------------:|------------:|
| B. ashtoni | 76.0% | 3.82 | 92.0% | 222 / 500 (44.4%) |
| B. flavidus | 96.0% | 4.06 | 99.6% | 288 / 500 (57.6%) |
| B. sandersoni | 96.4% | 4.37 | 100% | 456 / 500 (91.2%) |

The four LLM tiers used for the stratified expert-annotation sample in Section 5.4 are defined from this output: `strict_pass` (all three gates satisfied), `borderline` (all gates except mean morph < 4.0), `soft_fail` (diagnostic level below species), and `hard_fail` (no blind-ID match). Per-tier counts per species are reported in Appendix E.

![LLM tier outcomes on the 1,500-image pool](plots/llm_judge/llm_outcomes_1500.png)
*Figure 5.3a: LLM tier outcomes on the 1,500-image pool. Strict pass requires blind-ID match + diagnostic@species + mean morph ≥ 4.0. B. sandersoni passes at 91.2 % strict; B. flavidus and B. ashtoni concentrate in the borderline and soft-fail tiers respectively.*

![LLM strict-pass funnel per species](plots/llm_judge/llm_funnel_per_species.png)
*Figure 5.3b: Per-species LLM strict-pass funnel. From 500 generated images each, the three sequential gates (blind-ID match → diagnostic@species → morph-mean ≥ 4) progressively thin the pool. The B. ashtoni bottleneck is at the blind-ID gate (24 % loss at the first step); B. flavidus retains blind-ID match almost entirely but loses at the diagnostic-level gate.*

Figure 5.3c decomposes the blind-ID mismatch itself: for the images where the LLM's blind species guess does not match the target, which species does it guess instead? B. ashtoni's misidentifications concentrate in a small number of yellow-thorax Bombus species (B. impatiens and B. griseocollis together account for the majority of wrong guesses), confirming the inverse-phenotype default described above. B. sandersoni and B. flavidus, whose blind-ID rates are already near ceiling, produce diffuse wrong-guess distributions dominated by B. vagans (for B. sandersoni) and a long tail of yellow-banded congeners (for B. flavidus).

![LLM blind-ID breakdown per target species](plots/llm_judge/llm_blind_id_breakdown.png)
*Figure 5.3c: Per-target-species blind-ID outcomes. Correct-match bars are in the canonical per-species colour; wrong-guess bars are ordered by frequency. The dominant wrong guesses for B. ashtoni are yellow-thorax congeners, consistent with the LLM's inverse-phenotype default.*

#### 5.3.2 Per-feature diagnostics

The bottleneck is narrow and localised. The mean per-feature morphological scores hold above 4.0 for fourteen of the fifteen (species × feature) cells; the lone exception is B. ashtoni's thorax coloration mean of 2.98 — far below every other cell (Figure 5.3d). Wrong-coloration is the dominant failure mode overall (27.1 % of all 1,500 images), concentrated in B. ashtoni. Structural failure modes — extra or missing limbs, impossible geometry, visible artefacts, repetitive patterns — register at exactly 0 across all 1,500 images, confirming that the structured-prompting framework of Section 4.2 has eliminated this failure class. The residual gap is a colour-fidelity gap, and it is concentrated in the species that deviates most from the genus-typical phenotype. Figure 5.3e decomposes the LLM-flagged failures into gate-level counts and per-feature-below-3 counts, showing that the "no blind-ID match" gate loses the largest fraction of B. ashtoni images while the "morph mean < 4" gate dominates losses for B. flavidus. Per-angle and per-caste breakdowns — including the frontal-view paradox (high morph mean but low strict-pass rate driven by abdomen-banding occlusion) and the male-caste deficit in B. ashtoni (30.9 % caste-correct vs. 84.3 % female) — are reported in Appendix E. They confirm the judge's scoring behaves coherently across view conditions and do not change the filter-calibration argument that follows.

![LLM per-feature mean morph score](plots/llm_judge/llm_per_feature_heatmap.png)
*Figure 5.3d: LLM mean morphological score per (species × feature) cell on the 1,500-image pool. Fourteen of fifteen cells exceed 4.0; the lone outlier is B. ashtoni thorax coloration at 2.98, the operative bottleneck for B. ashtoni strict-pass.*

![LLM failure-mode decomposition](plots/llm_judge/llm_failure_modes.png)
*Figure 5.3e: LLM-flagged failures on the 1,500-image pool. Left: stacked gate-failure counts per species (no blind-ID match / diagnostic below species / morph-mean below 4). Right: per-feature count of images scoring below 3 on that feature, grouped by species. Structural failure codes (impossible geometry, extra/missing limbs, visible artefacts, repetitive pattern) register at zero across all 1,500 images and are therefore omitted from both panels.*

#### 5.3.3 What the judge measures — and what it does not

The judge measures species-level morphological fidelity as assessed by a vision-language model on a human-visual rubric, and Table 5.3's funnel shows the measurement is informative: the judge correctly identifies the B. ashtoni generation bottleneck, passes B. sandersoni at near-ceiling, and scales morph scores with generation difficulty. What the judge does not measure, however, is whether a synthetic image is useful for the downstream classifier. Section 5.5 reports the direct downstream test (D4 vs. D3 row of Table 5.5); Section 5.4 closes the calibration question by comparing the judge directly against expert judgment on the same 150 images.

### 5.4 Expert Calibration Results

Section 5.4 reports the first direct test of whether the LLM judge's per-feature rubric (Section 5.3) agrees with a domain expert, and whether a learned filter trained on 150 expert labels outperforms the LLM rule and an unsupervised BioCLIP-centroid baseline on the remaining 1,350 unlabelled synthetics. The expert-annotation protocol and probe architecture are specified in Section 4.4; here the results are reported on three levels — 150-label agreement, 1,500-image selection geometry, and image-level pass/fail galleries. Downstream classification F1 for the D5 (centroid) and D6 (probe) variants that this section's filters select is [TODO: pending completion of multi-seed and 5-fold GPU runs] and is reported alongside D1–D4 in Section 5.5.

#### 5.4.1 Expert-annotation outcomes on the 150-image sample

The 150-image stratified sample (50 per rare species, across four LLM tiers) was annotated by a single entomologist under the two-stage protocol of Section 4.4.2. Under the lenient rule (no structural failure + diagnostic at genus-or-better + expert morph-mean ≥ 3.0) 128 / 150 images (85.3%) pass; under the strict rule (blind-ID match + diagnostic at species + expert morph-mean ≥ 4.0) 50 / 150 (33.3%) pass. Per-species strict pass rates are strikingly heterogeneous: 27 / 50 sandersoni (54%), 17 / 50 ashtoni (34%), 6 / 50 flavidus (12%). The distribution between strict-pass, lenient-only, and lenient-fail segments is reported in Figure 5.5: B. flavidus in particular clears the lenient bar in 43 / 50 cases but only 6 / 50 clear the strict bar, so most flavidus synthetics are "passable as a Bombus" without being species-faithful. Even though the LLM judge passes sandersoni at 91.2% strict and flavidus at 57.6% strict on the full 1,500-image pool (Section 5.3), an independent expert on a comparable sample finds only 12% of flavidus synthetics fully acceptable. The calibration gap is therefore largest for the species with the most diffuse and case-variable phenotype.

![Expert-annotation outcomes on the 150-image sample](plots/filters/expert_outcomes_150.png)
*Figure 5.5: Expert-annotation outcomes on the 150-image sample. Strict pass = blind-ID match + diagnostic@species + expert morph-mean ≥ 4.0; lenient pass = no structural failure + diagnostic ≥ genus + morph-mean ≥ 3.0. Lenient-only = lenient pass without strict. Per-species strict rates span 12% (B. flavidus) to 54% (B. sandersoni); lenient-fail rates span 8% (B. sandersoni) to 22% (B. ashtoni).*

Failure-mode attribution on the 150-image sample (Figure 5.6) decomposes the expert-rejection signal. "Species other" (the synthetic does not match the target species to an expert) occurs 52 times, concentrated in B. flavidus (22) and B. ashtoni (16); "wrong coloration" occurs 28 times, with 18 in B. flavidus and 9 in B. ashtoni; structural failures (impossible_geometry, extra_missing_limbs, wrong_scale) together account for 46 images, weighted toward B. ashtoni (22). Because the species-other checkbox is itself a residual ("doesn't look right, but not any of the listed codes"), we classify the expert's free-text note for each species-other flag by keyword into six themes (right panel of Figure 5.6). Face/head anatomy dominates (17 / 52, led by B. flavidus eyes and B. flavidus / B. sandersoni antennae and proboscis), followed by leg anatomy (15 / 52, largely "spidery legs"), body proportion (8 / 52, mostly B. flavidus and B. ashtoni body-to-head ratios), sex/caste mismatch (7 / 52, flagged when an image mixes male and female cues such as pollen baskets on a male-coloured head), posture (3 / 52, images where the bee is off the flower), and color placement (2 / 52, residual after the wrong_coloration code already absorbs most cases). The per-species failure-mode mix defines what the learned filter must learn to reject, and the species-other decomposition shows that what the expert reads as "off" is predominantly fine-grained morphology (eyes, antennae, legs) that the generator has no explicit loss for.

![Expert-flagged failure modes on the 150-image sample](plots/filters/expert_failure_modes_150.png)
*Figure 5.6: Expert-flagged failure modes on the 150-image sample, grouped by target species. Left: checkbox failure codes — wrong-coloration dominates (driven by B. flavidus), structural failures (impossible geometry, wrong scale, extra/missing limbs) concentrate on B. ashtoni. Right: the 52 "species other" free-text notes classified by keyword into six anatomical themes; face/head anatomy is the single largest driver. Image may carry multiple tags.*

#### 5.4.2 LLM–expert agreement

The central quantitative calibration result is in Figures 5.7 and 5.8. Figure 5.7 shows the 2 × 2 confusion of LLM-strict × expert-strict, overall and per species. Over all 150 images, the LLM flags 78 as strict-pass; the expert flags 50 as strict-pass; the intersection is 31. LLM precision against the expert is 0.40, recall 0.62, and the two rules agree on 84/150 images overall (56 %) — only ~7 percentage points above the 49 % agreement expected under independence (given the LLM's 52 % and the expert's 33 % strict-pass marginals). Cohen's κ is not reported because a single annotator supplies the expert labels, so the inter-rater statistic is not defined; the 2 × 2 counts, the LLM precision/recall, and the observed-vs-independence agreement gap together carry the same information. Treating the LLM's continuous mean morph score as a ranker and computing the AUC-ROC against expert strict labels yields 0.56 (strict) / 0.54 (lenient): the LLM morph-mean is only marginally better than random at ranking synthetics by expert quality. This is the quantitative statement of the calibration gap Section 5.3.3 previewed.

![LLM vs expert 2x2](plots/filters/llm_vs_expert_strict_2x2.png)
*Figure 5.7: LLM-strict × expert-strict confusion on the 150-image sample, overall and per species. LLM precision against expert strict = 0.40; recall = 0.62; overall agreement 56 % (84/150), ~7 pp above the 49 % rate expected under independence.*

Figure 5.8 decomposes the disagreement per feature. For each (species × morphological-feature) cell I report the LLM over-strict rate (expert ≥ 4 but LLM < 4 — LLM misses a feature the expert accepts) and the LLM blind-spot rate (LLM ≥ 4 but expert < 4 — LLM accepts a feature the expert rejects). The pattern is structured. Legs-and-appendages and wing-venation-texture show near-zero blind-spot rates across all three species (0.04–0.15), confirming that both rubrics accept these features consistently. Thorax coloration for B. ashtoni produces the largest single disagreement: LLM over-strict rate 0.59 — in more than half the 150 ashtoni thorax calls, the expert accepts coloration the LLM rejects (the LLM is penalising visible black regions the expert reads as diagnostically correct, not as fidelity failures). For B. flavidus, blind-spot rates run 0.26–0.44 across features: the LLM accepts flavidus images that the expert rejects on coloration, head shape, and leg appendages. Sandersoni shows the smallest per-feature disagreement of the three (blind-spot 0.04–0.38, over-strict 0.00–0.11), consistent with its high LLM and expert strict-pass rates.

![LLM vs expert feature heatmap](plots/filters/llm_vs_expert_feature_heatmap.png)
*Figure 5.8: Per-feature LLM-vs-expert disagreement rates on the 150-image sample. Over-strict (expert≥4, LLM<4) and blind-spot (LLM≥4, expert<4) rates per species × feature.*

Figure 5.9 (qualitative galleries) makes the disagreement concrete. Per rare species, the top row shows images the LLM passes strict but the expert rejects — many display visible wrong-coloration in the thorax or abdomen that is obvious to a human expert but does not reach the LLM's 1–5 morph-score cutoff. The bottom row shows images the LLM fails strict but the expert accepts — predominantly images with borderline diagnostic completeness where the expert tolerates partial occlusion that the LLM scores conservatively. Full per-species galleries are in Appendix D; see also `docs/plots/filters/llm_morph_vs_expert_morph.png` for the scatter of per-image LLM-mean vs expert-mean morph scores (MAE ≈ 0.4, Pearson r ≈ 0.35).

![LLM vs expert disagreement galleries](plots/filters/grids/grid_flavidus_disagree_llm_expert.png)
*Figure 5.9: Representative LLM-vs-expert disagreement cases for B. flavidus (analogous galleries for B. ashtoni and B. sandersoni in Appendix D). Top row: LLM pass, expert fail. Bottom row: LLM fail, expert pass.*

#### 5.4.3 The learned probe: feature-config ablation and thresholds

A linear logistic probe trained on the 150 expert labels recovers a meaningful quality signal from BioCLIP features alone. Figure 5.10 reports LOOCV AUC-ROC under four feature-configuration ablations on the same 150 images: BioCLIP-only (512 dims), LLM-features-only (8 dims: five morph scores + blind_match + diag_species + diag_genus indicators), BioCLIP+LLM concatenation (520 dims), and BioCLIP+LLM+species-one-hot (523 dims). BioCLIP alone wins at LOOCV AUC strict 0.792, with the concatenation configurations trailing within LOOCV noise (0.790, 0.787) and the LLM-only configuration substantially below at 0.649. At n = 150, the 8 noisy LLM feature columns hurt rather than help — a decisive signal that the LLM judge's per-feature scores do not carry the discriminative information BioCLIP features already supply. The shipped D6 probe therefore uses the BioCLIP-only configuration with C = 0.01 (chosen by nested 5-fold stratified CV).

![Probe feature-config ablation](plots/filters/probe_feature_config_ablation.png)
*Figure 5.10: Probe feature-configuration ablation. Left-ward bars: LOOCV AUC under the strict rule. Right-ward: under the lenient rule. BioCLIP alone wins at strict AUC 0.792.*

Per-species F1-maximising pass thresholds are learned from the LOOCV pass-probability predictions under the strict rule (Figure 5.11). The learned thresholds are τ_ashtoni = 0.125, τ_sandersoni = 0.495, τ_flavidus = 0.222, with per-species F1 of 0.62, 0.90, and 0.50 at those thresholds (Table 5.4). The large span in τ reflects the per-species strict-pass heterogeneity: B. sandersoni has 27 expert-strict positives on 50 images, so the optimal decision boundary sits near the empirical median of its pass-probability distribution; B. flavidus has 6 positives on 50, so the optimal boundary is pushed downward to a lower probability threshold. Using a single global τ would either over-admit B. sandersoni or under-admit B. flavidus.

![Probe ROC LOOCV](plots/filters/probe_roc_loocv.png)
*Figure 5.11: Per-species LOOCV ROC curves under the strict rule at the BioCLIP-only config. Red markers indicate the learned F1-maximising thresholds τ_ashtoni = 0.125, τ_sandersoni = 0.495, τ_flavidus = 0.222.*

*Table 5.4: Per-species probe accuracy at the learned F1-maximising threshold τ on the 150-image LOOCV sample (BioCLIP-only configuration). At every species the threshold is set such that recall = 1.0 (no expert-strict positive is missed); precision and F1 differ across species, tracking the difficulty ordering already seen in the LLM and embedding-space measures (Section 6.1).*

| Species | τ | TP / FP / FN / TN | Precision | Recall | F1 | Expert-strict positives / 50 |
|---------|---:|:--:|---:|---:|---:|---:|
| **B. ashtoni** | 0.125 | 17 / 21 / 0 / 12 | 0.447 | 1.000 | 0.618 | 17 |
| **B. sandersoni** | 0.495 | 27 / 6 / 0 / 17 | 0.818 | 1.000 | 0.900 | 27 |
| **B. flavidus** | 0.222 | 6 / 12 / 0 / 32 | 0.333 | 1.000 | 0.500 | 6 |

Beyond the τ-thresholded confusion, the score distribution itself is informative: Figure 5.12a histograms the probe's pass-probability separately for expert-strict positives and expert fails on each species. The two distributions visibly separate for B. sandersoni (expert positives concentrate near 0.7, expert fails near 0.3); they overlap heavily for B. ashtoni and B. flavidus, which is why the F1-maximising τ for those two species lands at low values (0.125, 0.222) and the achievable per-species precision tops out at 0.45 and 0.33 respectively. These are not failures of the probe but of the underlying per-species signal-to-noise ratio at n = 50 expert labels per species. Figure 5.12b shows the same probe's reliability-diagram calibration against expert strict labels — within each per-species probability bin the empirical expert-strict pass rate tracks the predicted probability, which is the precondition for using the per-species τ rule defensibly on the 1,500-image pool.

![Probe pass-probability by expert label](plots/filters/probe_score_by_expert_label.png)
*Figure 5.12a: Probe pass-probability distribution split by expert strict label on the 150-image LOOCV sample. Coloured bars are expert-strict positives; grey bars are expert fails; the dashed red line is the per-species F1-maximising τ. The two distributions separate cleanly only for B. sandersoni; B. ashtoni and B. flavidus show heavy overlap, which is why their achievable precision at recall = 1 is bounded.*

![Probe calibration reliability](plots/filters/probe_calibration_reliability.png)
*Figure 5.12b: Probe calibration reliability on the 150-image sample. Size ∝ bin count; dashed line = perfect calibration.*

The most load-bearing single statistic of the probe's quality-as-filter is the expert-label coverage of its 200-image selection on the full 1,500-image pool (Figure 5.13). Of the 150 expert-annotated synthetics, the number that fall into each filter's 200-selection and are expert-strict ✓ is **D4 LLM: 5 / 12 / 1** (B. ashtoni / B. sandersoni / B. flavidus), **D5 centroid: 9 / 13 / 1**, **D6 probe: 16 / 20 / 6**. The probe filter therefore retains roughly **3× as many expert-validated images** as the LLM filter within an identical 200-image budget, with the largest relative gain on B. flavidus (1 → 6) — the species where the LLM judge's calibration was weakest. This is the first direct evidence — independent of downstream classifier training — that expert supervision yields a measurably more expert-aligned 200-image selection than either unsupervised centroid distance or the LLM rule.

![Expert coverage of selected 200](plots/filters/expert_coverage_of_selected200.png)
*Figure 5.13: Expert-label composition of each filter's 200-image selection on the 1,500-image pool. Green: expert-strict pass; orange: lenient-only; red: expert fail; grey: unlabelled. D6 probe selects the largest expert-strict subset across all three species within the same 200-image cap.*

#### 5.4.4 Filter comparison on the 1,500-image pool

The three filters (D4 LLM, D5 centroid, D6 probe) applied to the full 1,500-image pool produce dissimilar pass sets (Figure 5.14 Venn diagrams, Figure 5.15 funnel). Pre-cap pass counts per species are **D4 LLM: 218 / 456 / 288** (B. ashtoni / B. sandersoni / B. flavidus); **D5 centroid: 250 / 250 / 250** by construction of the median-synth threshold rule; **D6 probe: 418 / 364 / 204**. After the cap at 200 per species all three variants are size-matched at +200 per species, ensuring that any downstream F1 difference in Section 5.5 cannot be confounded with training-set volume.

![Per-species filter Venn](plots/filters/venn_llm_centroid_probe.png)
*Figure 5.14: Per-species 3-set Venn of the pass sets on the 1,500-image pool. Overlaps are modest (Jaccard 0.06–0.18 between each pair) — the three filters substantially disagree on which 200 images are best.*

![Filter funnel per species](plots/filters/filter_funnel_per_species.png)
*Figure 5.15: Per-species selection funnel from 500 generated images through LLM-strict / centroid-pass / probe-pass to the capped 200.*

Agreement between the three filter rankings is low. To avoid overloading a single scatter with four orthogonal encodings, the filter-score comparison is reported as two complementary panels. Figure 5.16a plots every synthetic in the 1,500-image pool (D5 centroid cosine on the x-axis, D6 probe pass-probability on the y-axis), coloured only by species — the reader can see each species' joint distribution on the two filter axes directly, without conditioning on any label. Within species the probe and centroid scores correlate only weakly (Spearman ρ ≈ 0.2–0.4), so the two filters select substantially different subsets of the same pool. Figure 5.16b then restricts to the 150 expert-labelled synthetics and adds marker shape (●/✗) for the expert strict label; the question the reader can judge visually is whether expert-strict pass clusters better with high probe-y or with high centroid-x. The expert-strict points (coloured dots) cluster noticeably toward the upper portion of the probe axis but are less cleanly stratified along the centroid axis, which is the image-level statement of the AUC gap reported quantitatively in Section 5.4.3. Score-distribution violins split by LLM tier (in `score_violins_by_expert_tier.png`) corroborate: the probe distinguishes `strict_pass` from `hard_fail` cleanly on B. sandersoni but not on B. flavidus, consistent with the expert labelling asymmetry.

![Filter scores on the 1,500-image pool](plots/filters/centroid_vs_probe_scatter_pool.png)
*Figure 5.16a: Per-synthetic D5 centroid cosine × D6 probe pass-probability on the full 1,500-image pool, coloured by species (Okabe-Ito). Points lie along different within-species distributions on the two axes; within-species Spearman ρ 0.2–0.4.*

![Filter scores on the expert-labelled 150](plots/filters/centroid_vs_probe_scatter_expert.png)
*Figure 5.16b: Same axes restricted to the 150 expert-labelled synthetics, with per-species τ guides. Coloured dots = expert strict pass; grey × = expert fail. Expert-strict points cluster toward the upper (probe) end of the y-axis more cleanly than toward the right (centroid) end of the x-axis.*

Qualitative image grids make the selection differences visible. Figure 5.17 shows a per-species 4-row × 4-column contact sheet for B. flavidus: rows are {LLM-strict only, centroid only, probe only, in all three filters}, columns are 4 random draws. The "all three" row is the tightest coverage of canonical flavidus (pale thorax, characteristic markings); the "LLM-strict" row includes several images with visibly wrong thorax coloration that the probe correctly rejects. Analogous galleries for the two other species, plus boundary-cases within τ ± 0.05 (Figure 5.18), per-filter top-8 / bottom-8 galleries, and LLM-vs-probe / centroid-vs-probe disagreement grids, are in Appendix D.

![Flavidus 4x4 filter grid](plots/filters/grids/grid_flavidus_4x4_by_filter.png)
*Figure 5.17: B. flavidus per-filter 4×4 image grid. Rows: LLM-strict, centroid, probe, all-three. The probe-only and all-three rows are visibly more canonical than the LLM-only row, which contains images with wrong-coloration that pass the LLM rule.*

![Flavidus probe boundary](plots/filters/grids/grid_flavidus_probe_boundary.png)
*Figure 5.18: B. flavidus probe-threshold boundary (τ = 0.222). Top row: images just below τ (probe would reject). Bottom row: just above τ (probe accepts). The cut separates recognisable canonical flavidus from atypical-coloration or off-caste failures.*

#### 5.4.5 Summary and forward pointer

The 150-label protocol exposes a calibration gap (strict-pass agreement 56 % vs. 49 % expected under independence; morph-mean AUC 0.56) between the LLM judge and the expert; a 150-label-supervised BioCLIP probe achieves LOOCV AUC strict 0.792 and selects a 200-image subset per species that is roughly 3× richer in expert-strict images than the LLM-rule filter. Whether this filter-level quality improvement translates into downstream classifier F1 — i.e., whether the probe's 200 are not just more expert-aligned but also more *useful* for training — is the direct question of Section 5.5. That question is currently blocked on completing the multi-seed and 5-fold GPU runs for D5 and D6; the corresponding rows of Table 5.5 are marked `[TODO]`.

CKNNA alignment (Huh et al., 2024) between each filter's 200-image selection and the real-image distribution in BioCLIP space is planned as the complementary filter-level metric and is flagged as [TODO: deferred until D5 / D6 training outputs provide the consolidated embeddings].

### 5.5 Augmentation Method Comparison

#### 5.5.1 Single-run overview

A single-split fixed-test-set training run is the simplest reading of each augmentation variant and makes the per-species picture directly visible before any aggregate protocol is applied. Table 5.4b reports single-split macro F1 with 10,000-iteration bootstrap 95 % CI alongside per-species F1 for the three rare species, and Figure 5.18a visualises the same numbers. D4 has the highest single-split macro F1 (0.834) and D1 the lowest (0.815), but all four completed variants sit inside each other's 95 % CIs — single-split macro F1 does not separate the four variants statistically on its own. The per-species picture is sharper: D2 CNP lifts B. flavidus from 0.623 to 0.719 (+0.096 over baseline), the largest per-species single-split gain of any method; D3 and D4 numerically improve B. flavidus as well (to 0.698 and 0.710) but harm or flatten B. ashtoni and B. sandersoni relative to D2. D5 and D6 are marked `[TODO]` in both the table and the figure.

*Table 5.4b: Single-split fixed-test-set summary (f1 checkpoint). Macro F1 has 10,000-iteration bootstrap 95 % CI; per-species F1 is the point estimate on the fixed test set (n = 6 / 10 / 36 for B. ashtoni / B. sandersoni / B. flavidus).*

| Dataset | Macro F1 | 95 % CI | B. ashtoni F1 | B. sandersoni F1 | B. flavidus F1 |
|---|---:|---|---:|---:|---:|
| **D1 Baseline** | 0.815 | [0.774, 0.845] | 0.500 | 0.588 | 0.623 |
| **D2 CNP** | 0.829 | [0.788, 0.859] | 0.545 | 0.625 | 0.719 |
| **D3 Unfiltered** | 0.823 | [0.781, 0.854] | 0.500 | 0.533 | 0.698 |
| **D4 LLM-filtered** | 0.834 | [0.789, 0.864] | 0.600 | 0.588 | 0.710 |
| **D5 Centroid** | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| **D6 Expert-probe** | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

![Single-run D1-D6 species F1](plots/single_run_species_f1.png)
*Figure 5.18a: Single-run D1–D6 performance on the fixed split. Left: macro F1 with 95 % bootstrap CI; all four completed variants overlap within their intervals. Right: per-species F1 grouped bars for the three rare species; D5 / D6 placeholders will populate when the GPU runs complete. The per-species view is the reading that matters for the rare-tier augmentation question — the macro F1 bar hides species-specific movements.*

Single-split comparisons are underpowered on the rare species: a single test-image flip changes B. ashtoni F1 by up to 0.17 given n = 6. Section 5.5.2 reports the same comparison under the multi-seed and 5-fold cross-validation protocols that were designed to absorb this variance, and Section 5.5.3 reports the pairwise statistical tests on the fold-level macro F1 that the single-split numbers cannot support.

#### 5.5.2 Full wide table (single-split + multi-seed + 5-fold CV)

Table 5.5 reports ResNet-50 classifier performance across the six dataset variants under all three protocols. Rows are the six dataset variants; columns are grouped by protocol — single-split, multi-seed × fixed split (5 seeds × same test set), and 5-fold cross-validation — and within each protocol group four figures are reported: macro F1 with 95 % bootstrap CI, and focus-species F1 for the three rare species (B. ashtoni / B. sandersoni / B. flavidus) alongside overall accuracy. Cells for D5 (centroid filter) and D6 (expert-probe filter) are marked `[TODO]` — training runs were launched as a 4 × 5-task SLURM array on the cluster at submission time and had not finished when this chapter was written.

*Table 5.5: ResNet-50 macro F1, rare-species F1 (B. ashtoni / B. sandersoni / B. flavidus), and overall accuracy across six dataset variants and three evaluation protocols (f1 checkpoint). Single-split F1 is reported with a test-set bootstrap 95 % CI (10 000 resamples); multi-seed F1 is reported as mean ± std across 5 seeds on a fixed split; 5-fold CV F1 is reported as mean ± std across folds, with a pooled-predictions bootstrap 95 % CI. Rare-species F1 under the multi-seed and 5-fold protocols is the mean of per-seed / per-fold F1. D5 and D6 rows are pending the multi-seed and 5-fold CV GPU runs.*

| Dataset | Single-split macro F1 | Single-split 95% CI | Rare F1 (ash / sand / flav) | Acc | Multi-seed macro F1 | Rare F1 (ash / sand / flav) | Acc | 5-fold CV macro F1 | 5-fold pooled 95% CI | Rare F1 (ash / sand / flav) | Acc |
|---|---:|---|---|---:|---:|---|---:|---:|---|---|---:|
| **D1 Baseline** | 0.815 | [0.774, 0.845] | 0.500 / 0.588 / 0.623 | 0.882 | 0.839 ± 0.006 | 0.614 / 0.622 / 0.760 | 0.890 | 0.832 ± 0.013 | [0.818, 0.847] | 0.621 / 0.466 / 0.747 | 0.892 |
| **D2 CNP** | 0.829 | [0.788, 0.859] | 0.545 / 0.625 / 0.719 | 0.886 | 0.822 ± 0.014 | 0.577 / 0.477 / 0.724 | 0.886 | **0.837 ± 0.013** | [0.824, 0.850] | 0.685 / 0.433 / 0.806 | 0.891 |
| **D3 Unfiltered synthetic** | 0.823 | [0.781, 0.854] | 0.500 / 0.533 / 0.698 | 0.887 | 0.828 ± 0.009 | 0.608 / 0.494 / 0.669 | 0.891 | 0.820 ± 0.024 | [0.808, 0.835] | 0.576 / 0.326 / 0.764 | 0.889 |
| **D4 LLM-filtered** | **0.834** | [0.789, 0.864] | 0.600 / 0.588 / 0.710 | 0.886 | 0.831 ± 0.008 | 0.609 / 0.533 / 0.709 | 0.890 | 0.821 ± 0.019 | [0.809, 0.837] | 0.577 / 0.398 / 0.735 | 0.889 |
| **D5 Centroid** | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| **D6 Expert-probe** | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

Two facts dominate the observed D1–D4 rows. First, augmentation effects are concentrated in the rare tier: moderate-tier and common-tier F1 (reported in full in Appendix E) move by ≤ 0.013 under any method, so aggregate macro F1 differences reflect rare-species performance almost entirely. Second, the three protocols produce different aggregate rankings — D4 is best on single-split (0.834), D2 is best on 5-fold CV (0.837), and D1 is best on multi-seed (0.839). The rankings are not contradictory: multi-seed and single-split share the same 6 / 10 / 36 rare test images, so flipping one or two correctly-classified rare images is enough to swap the aggregate ranking; 5-fold CV pools roughly five times more rare-test predictions per species and is the more reliable aggregate reading.

Figure 5.19 visualises per-species delta F1 vs. D1 baseline under both 5-fold and multi-seed protocols. Rare-species rows are coloured saturated under D3 and D4; moderate and common rows are essentially flat.

![Per-species delta F1](plots/failure/species_f1_delta_kfold.png)
*Figure 5.19: Per-species F1 change relative to D1 baseline under 5-fold CV (primary) and multi-seed (fixed split). Rare species highlighted; moderate and common tiers show negligible effects under any method.*

#### 5.5.3 Pairwise paired t-tests

Pairwise paired t-tests on fold-level macro F1 (Table 5.6) identify two comparisons that clear the high-power bar imposed by df = 4. D4 is significantly worse than the baseline (p = 0.041) and significantly worse than D2 (p = 0.030). D3 and D4 are statistically indistinguishable (p = 0.777): the strict LLM filter, applied to the D3 pool, produces no measurable improvement over no filter at all. This non-result is the central empirical motivation for the mechanistic analysis in Section 5.6.

*Table 5.6: Pairwise paired t-tests on fold-level macro F1 (5-fold CV, df = 4). D5 / D6 comparisons pending.*

| Comparison | Mean delta | t | p | Significant |
|------------|-----------:|----:|----:|-------------|
| D1 vs D2 CNP | +0.005 | 1.72 | 0.161 | No |
| D1 vs D3 Unfiltered | -0.012 | -2.17 | 0.096 | No |
| **D1 vs D4 LLM-filtered** | **-0.011** | **-2.98** | **0.041** | **Yes — D4 worse** |
| D2 vs D3 Unfiltered | -0.017 | -2.59 | 0.061 | No |
| **D2 vs D4 LLM-filtered** | **-0.016** | **-3.29** | **0.030** | **Yes — D4 worse** |
| D3 vs D4 LLM-filtered | +0.001 | 0.30 | 0.777 | No (filter no benefit) |
| D1 vs D5 Centroid | [TODO] | [TODO] | [TODO] | [TODO] |
| D1 vs D6 Expert-probe | [TODO] | [TODO] | [TODO] | [TODO] |
| D4 vs D6 (filter comparison) | [TODO] | [TODO] | [TODO] | [TODO] |
| D5 vs D6 (supervision gap) | [TODO] | [TODO] | [TODO] | [TODO] |

Per-species analysis locates the signal. D2 significantly improves B. flavidus F1 over baseline under 5-fold CV (+0.059, p = 0.005), the only statistically significant per-species gain from any method for which training is complete. D3 reduces B. sandersoni F1 by 0.140 under 5-fold CV (marginally significant at p = 0.052), and D4 reduces it by 0.068. B. ashtoni produces large numerical swings (D2 +0.064, D3 −0.045, D4 −0.044 under 5-fold CV) but with 95 % CIs of width 0.27–0.30 due to n = 32 pooled test images, none of these per-species comparisons reach significance.

#### 5.5.4 Ablations: volume, subtractive, and additive

Three ablations probe *why* the D3 and D4 variants harm rare-species F1, each fixing a different alternative explanation. (i) **Volume** — whether the harm resolves if we generate more synthetics of the same quality. (ii) **Subtractive** — whether removing all synthetics of exactly one rare species recovers that species' F1, which would confirm the harmed synthetics are causally responsible rather than merely correlated. (iii) **Additive** — whether adding only one rare species' synthetics to the baseline (while leaving the other two species unaugmented) improves or harms that species, a direct test of whether any *positive* contribution is possible from the synthetic pool at all.

**Volume ablation.** A natural concern is whether the D3 and D4 rare-species harm would resolve at higher synthetic volumes. Figure 5.19 reports macro F1 and rare-species F1 under D3 and D4 at volumes of +50, +100, +200, +300, and +500 images per rare species (single-split evaluation). Neither variant shows a coherent volume–performance trend: D3 macro F1 fluctuates between 0.820 and 0.834 across volumes without a monotone direction, and D4 peaks at +200 but does not maintain the gain at +300 or +500. At +500, both variants regress on B. sandersoni (D3 0.471, D4 0.556) as the synthetic-to-real ratio reaches 12.5:1 for that species. The absence of a volume-dependent improvement establishes that the bottleneck is generation fidelity, not quantity — adding more synthetics of the same quality does not close the embedding-space gap that Section 5.2 identified. A companion background-removal diagnostic (synthetic images evaluated on white backgrounds; full details in Appendix E) confirms the same conclusion from the opposite direction: removing the background does not alter the strict pass rate but increases the wrong-coloration rate by 55 %, pinpointing specimen coloration rather than background confusion as the generation bottleneck.

![Volume ablation](../RESULTS_count_ablation/volume_ablation_trends_with_ci.png)
*Figure 5.19: Volume ablation for D3 and D4 at +50 to +500 synthetic images per rare species (single-split evaluation). No consistent improvement at any volume for either variant.*

**Subtractive ablation — leave one species out.** To confirm the displaced synthetic cluster is causally responsible for the harm — rather than merely correlated with it — I run six additional training jobs (seed 42 only) that each drop all synthetic images of exactly one rare species from D3 or D4. Recovery is defined as F1 under ablation minus F1 under the full variant: positive recovery implies the removed synthetics were collectively harming the target species; negative recovery implies they were collectively helping.

*Table 5.7: Own-species F1 recovery under single-species subtractive ablation (seed 42). Threshold |delta| > 0.02 for a directional label; otherwise neutral. D5 / D6 ablations pending.*

| Variant | Dropped species | F1 full → ablated | Recovery | Label |
|---------|-----------------|---------------------:|---------:|-------|
| D3 | B. ashtoni | 0.545 → 0.727 | **+0.182** | harmful |
| D3 | B. sandersoni | 0.571 → 0.476 | −0.095 | helpful |
| D3 | B. flavidus | 0.645 → 0.708 | +0.062 | harmful |
| D4 | B. ashtoni | 0.727 → 0.727 | +0.000 | neutral |
| D4 | B. sandersoni | 0.625 → 0.706 | **+0.081** | harmful |
| D4 | B. flavidus | 0.725 → 0.719 | −0.006 | neutral |
| D5 | (all three) | [TODO] | [TODO] | [TODO] |
| D6 | (all three) | [TODO] | [TODO] | [TODO] |

Two patterns dominate. First, the LLM filter neutralises the large B. ashtoni harm seen in D3 (+0.182 recovery converts to +0.000 in D4): the filter does remove genuinely bad B. ashtoni synthetics. Second, *the filter reverses the B. sandersoni effect.* Unfiltered B. sandersoni synthetics were collectively helpful in D3 (removing them lost 0.095 F1); filtered B. sandersoni synthetics are collectively harmful in D4 (removing them gains 0.081 F1). Because B. sandersoni has the highest LLM strict-pass rate of the three species (91.2 %), the filter retains nearly all images and the 8.8 % it discards includes the signal that was compensating for the harmful subset. This is the clearest empirical demonstration of LLM-judge miscalibration in the dataset: for the species the filter passes most easily, it discards exactly the wrong subset.

Cross-species collateral effects are substantial and reinforce the embedding-space picture. Dropping B. ashtoni synthetics in D3 simultaneously reduces B. sandersoni F1 by 0.150 and increases B. flavidus F1 by 0.124 — the same image set provides conflicting gradient signal across class decisions, consistent with the failure-chain finding that retrieved nearest synthetics frequently belong to the wrong species. Figure 5.20 shows the full 3 × 3 recovery matrix alongside the own-species recovery bars.

![Subtractive ablation recovery](plots/failure/subset_ablation_recovery.png)
*Figure 5.20: Subtractive ablation recovery. Left: own-species F1 recovery under D3 and D4 by dropped species (old D4/D5 tokens in plot filename correspond to D3/D4 under present thesis numbering). Right: full dropped-vs-measured recovery matrix. Single seed (42); the ablation establishes direction, not magnitude.*

Each ablation cell is a single seed-42 run with rare-species test n between 6 and 36, so a 0.05 F1 change corresponds to flipping 0.3–1.8 images. The analysis is designed to establish direction of effect, not statistically-powered magnitude; the B. sandersoni D3 → D4 sign reversal is a qualitative signal that stochastic noise cannot easily produce, while the cross-species collateral magnitudes should be read as directional only. Per-synthetic labels — propagating the own-species verdicts to every generated image in each variant — are written to `RESULTS/failure_analysis/synthetic_labels.csv`. Extending this causal attribution to D5 and D6 is [TODO: pending those variants' training outputs].

**Additive ablation — add only one species' synthetics.** The subtractive ablation tests whether removing a synthetic subset *recovers* F1; the additive counterpart tests the symmetric question — whether a single rare species' synthetics can *by themselves* help the classifier when added to the baseline without the other two species' pools. Three D1 + single-species runs (seed 42) are registered: (i) D1 + B. sandersoni-only synthetics (the species with the smallest generator–real gap on every measure in Section 6.1, and therefore the most plausible candidate for a positive-signal-in-isolation result); (ii) D1 + B. ashtoni-only; (iii) D1 + B. flavidus-only. All three are `[TODO]` at submission. Under the hypothesis that synthetic augmentation *can* help when the generated distribution sits within the real distribution, the B. sandersoni-only run should lift B. sandersoni F1 above baseline, and B. ashtoni-only and B. flavidus-only should at best leave their target unchanged. A negative or null result across all three species — including B. sandersoni — would argue that any displaced synthetic pool, regardless of per-species fidelity, carries the collateral-class penalty observed under D3.

*Table 5.8: Additive single-species ablation registered predictions (seed 42). `[TODO]` until the three D1 + single-species runs complete.*

| Variant | Added species | Own-species F1 | Delta vs D1 | Label |
|---------|---------------|---------------:|------------:|-------|
| D1 + B. sandersoni only | B. sandersoni | [TODO] | [TODO] | [TODO] |
| D1 + B. ashtoni only | B. ashtoni | [TODO] | [TODO] | [TODO] |
| D1 + B. flavidus only | B. flavidus | [TODO] | [TODO] | [TODO] |

### 5.6 Qualitative analysis

Section 5.5 establishes *that* synthetic augmentation harms rare species and *that* the LLM filter does not fix the harm. Section 5.6 reads the same evidence at the image level — per-image prediction flips, embedding-space nearest-neighbour chains for harmed test images, and filter pass / fail contact grids — so that the statistical patterns of Section 5.5 can be inspected as visible image content rather than aggregate numbers. The analysis uses the multi-seed protocol because per-image flip and chain analyses require every seed to evaluate the same images. Sections 5.6.1–5.6.3 report the analysis on the D3 and D4 variants (unfiltered and LLM-filtered) whose training is complete; D5 and D6 analogues are `[TODO]` pending the multi-seed training runs (Section 5.6.4).

#### 5.6.1 Per-image prediction flips

Each of the 2,362 test images produces 20 predictions across the four complete configs (D1, D2, D3, D4) and five seeds. Collapsing within each config by majority vote yields one verdict per (image, config) pair, and comparing each augmented config against the baseline partitions test images into four cells: stable-correct (both right), stable-wrong (both wrong), improved (augmentation rescued a baseline error), and harmed (augmentation broke a baseline-correct image).

*Table 5.9: Rare-species flip counts under each augmentation method (multi-seed majority vote, fixed split). No rare image is improved by any method. D5 and D6 columns pending.*

| Species | n test | D2 CNP (impr / harm) | D3 Unfiltered (impr / harm) | D4 LLM-filtered (impr / harm) | D5 Centroid | D6 Probe |
|---------|-------:|---------------------|------------------------------|--------------------------------|-------------|----------|
| B. ashtoni | 6 | 0 / 1 | 0 / 0 | 0 / 1 | [TODO] | [TODO] |
| B. sandersoni | 10 | 0 / 1 | 0 / 1 | 0 / 1 | [TODO] | [TODO] |
| B. flavidus | 36 | 0 / 5 | 0 / 8 | 0 / 6 | [TODO] | [TODO] |

The pattern is stark: no rare-species test image is improved by any augmentation method for which training is complete. The effect is not a mixture of improvements and harms that averages out poorly — it is one-directional harm. Expressed as a rate, B. flavidus is harmed on 22.2 % of its D3 test images, the largest cell in any (species × method) pair, while no common species exceeds a 3 % harm rate. Figure 5.21 visualises the species-by-method harm rates; the rare tier carries the signal and the remaining tiers are near zero.

![Flip-category heatmap](plots/failure/flip_category_heatmap.png)
*Figure 5.21: Flip-category rates by species and augmentation method (multi-seed majority vote, fixed split; columns labelled D3/D4/D5 in the original plot artefact correspond to D2/D3/D4 under the present thesis numbering). Rare rows carry the signal; moderate and common rows are near zero.*

#### 5.6.2 Embedding-space failure chains

For every rare-species test image harmed under D3 or D4, I retrieve the five nearest training synthetics of the corresponding variant by BioCLIP cosine similarity, restricted to that variant's actual training pool (600 synthetics per variant). Each retrieved synthetic carries its generated-species label and LLM tier. The aggregate retrieval statistic is informative across all three rare species: the median test-to-5-NN cosine similarity across the 49 D3-harmed chains is 0.56, well below the 0.7+ range typical of two real images of the same species. The "nearest training synthetic" is therefore not close to the harmed test image in absolute terms; it is close only relative to the rest of the synthetic pool. Figure 5.22 shows representative D3 chains for each rare species — a harmed B. ashtoni test image, a harmed B. sandersoni test image, and a harmed B. flavidus test image — followed by its five nearest D3 training synthetics. Three patterns carry across species. (i) The retrieved neighbours frequently belong to a species other than the target: for the eight D3-harmed B. flavidus test images, the classifier predicts B. citrinus (3), B. rufocinctus (2), B. ashtoni (2), and B. griseocollis (1), never the correct B. flavidus and never a single dominant alternative. (ii) For B. ashtoni, the retrieved neighbours include synthetic images whose tergite colour map is correct but whose anatomy drifts toward a non-cuckoo body plan, consistent with the expert's structural-failure flags in Section 5.4.1. (iii) For B. sandersoni the retrieved neighbours are mostly visually correct, which is consistent with the subtractive-ablation reversal (Section 5.5.4): the harmed B. sandersoni chains retrieve synthetics that look right but nonetheless sit in a region the classifier does not reconcile with the real B. sandersoni cluster. Full harmed and improved chain galleries for all three species — including t-SNE projections and per-species improved chains — are in Appendix F. Analogous D5 and D6 chain analyses are `[TODO]` pending D5 / D6 training outputs.

![Representative D3 failure chains for all three rare species](plots/failure/chains_d4_harmed/gallery/flavidus__Bombus_flavidus4512075898.png)
*Figure 5.22: Representative D3 failure chains for the three rare species (currently rendered as the B. flavidus panel; the B. ashtoni and B. sandersoni panels are in Appendix F for the present draft and will be promoted here in the final version; artefact directory `chains_d4_harmed/` under old numbering corresponds to D3 under present thesis numbering). Top row: a harmed rare-species test image followed by its five nearest D3 training synthetics, ranked by BioCLIP cosine similarity.*

#### 5.6.3 Filter pass / fail contact grids

What the expert sees — and what the LLM filter does or does not filter out — is clearest on pages of actual synthetic images. Figure 5.23 arranges per-species contact sheets of representative pass and fail images under each filter (D3 unfiltered, D4 LLM-strict, D5 centroid, D6 probe); D5 and D6 panels are `[TODO]` until the variants' pools are finalised against their final training partitions. Three visual patterns recur. First, the D4 LLM-strict pass cell contains B. flavidus images with wrong coloration (bright lemon rather than the species' characteristic dingy pale yellow) and B. ashtoni images with reduced corbicular anatomy the LLM does not penalise — images the D6 probe correctly rejects. Second, the D4 LLM-strict fail cell contains images the expert read as adequate (borderline diagnostic occlusion or missing a single feature), indicating that the LLM is over-strict in one direction while blind-spotted in another. Third, B. sandersoni — the species with the smallest expert–generator gap (Section 6.1) — shows visually cleaner pass images across every filter but also the clearest example of the LLM's 8.8 %-rejection pathology from the subtractive ablation (Section 5.5.4). Per-species boundary-cases within τ ± 0.05, per-filter top-8 / bottom-8 galleries, and LLM-vs-probe / centroid-vs-probe disagreement grids are in Appendix D.

![D3/D4/D5/D6 per-species filter pass/fail contact grid (B. flavidus shown)](plots/filters/grids/grid_flavidus_4x4_by_filter.png)
*Figure 5.23: Representative filter pass / fail contact sheet for B. flavidus (analogous B. ashtoni and B. sandersoni panels in Appendix D; D5 and D6 panels are `[TODO]` until those variants' final training pools are produced). Rows: LLM-strict only, centroid only, probe only, and all three filters together. The probe-only and all-three rows are visibly more canonical than the LLM-only row, which contains images with wrong coloration that pass the LLM rule.*

#### 5.6.4 D5 / D6 per-image analyses — pending

Per-image flip galleries and embedding-space failure-chain retrievals for the D5 (centroid) and D6 (expert-probe) variants are deferred until the multi-seed training runs complete. The Section 5.4 filter-level evidence predicts that D6 should show substantially fewer rare-species harm cells than D3 / D4 in Table 5.9 and a higher median test-to-5-NN cosine similarity in its failure chains (fewer cross-species nearest neighbours) than D3 or D4; the subtractive ablation in Section 5.5.4 further predicts positive or neutral recovery under D6 for all three rare species (no D3 → D4-style B. sandersoni sign reversal). If these predictions hold, the expert-supervised filter resolves the miscalibration rather than merely relocating it. The hypotheses are registered here against the GPU runs so that the subsequent Section 5.6 update can be read as confirmatory or disconfirmatory rather than post-hoc.


## 6. Discussion

Section 6 interprets the empirical findings of Section 5 along five lines. The synthetic–real gap is not uniform across species: expert, LLM, and embedding-space measures all rank B. sandersoni as the easiest and B. flavidus / B. ashtoni as the harder cases, consistent with how far each species' diagnostic morphology lies from the generator's prior (Section 6.1). Unfiltered generative augmentation (D3) degrades rare-species performance because its synthetic images sit in a displaced region of BioCLIP feature space that is measurable before training and is causally responsible for the F1 harm under subset ablation (Section 6.2). Copy-and-paste augmentation outperforms generative augmentation precisely because its outputs sit in the same region of feature space as the real images themselves (Section 6.3). LLM judges and expert judges make systematically different decisions about the same images — agreement is only 7 percentage points above chance under independence, the LLM morph-mean is a near-random ranker (AUC 0.56), and the LLM-passed / expert-rejected contact-grid quadrant reveals morphology the language rubric does not encode; a BioCLIP-only probe trained on the same 150 labels achieves LOOCV AUC 0.792 and selects a 200-image subset per species that is ~3× richer in expert-validated synthetics than the LLM rule (Section 6.4). The remaining limitations — evaluation power for rare species, single-seed ablation, BioCLIP as a feature-space proxy, pending D5 / D6 training — are honest constraints that do not invalidate the core findings (Sections 6.5 and 6.7).

### 6.1 The synthetic–real gap is non-uniform across species

A consistent pattern in the Section 5 evidence is that the difficulty of generating expert-acceptable synthetic images of a rare species — and the resulting harm to that species under augmentation — is itself per-species, not a single "rare-tier" effect averaged across the three species. The expert who labelled the 150-image stratified sample (Section 5.4.1) ranks the three species in the same order on three independent measures: strict pass rate (B. sandersoni 54 %, B. ashtoni 34 %, B. flavidus 12 %), structural-failure count (B. ashtoni 22 of 50 images flagged for impossible geometry / wrong scale / extra-or-missing limbs, against ≤ 6 each for B. sandersoni and B. flavidus), and "wrong-coloration" / "species other" rate (B. flavidus 22 + 18 of 50, B. ashtoni 16 + 9, B. sandersoni 4 + 1). The same ordering shows up in the LLM judge's strict pass rates (Section 5.3.1: B. sandersoni 91.2 %, B. flavidus 57.6 %, B. ashtoni 44.4 %) and in the BioCLIP feature-space gap (Section 5.2.3: median synthetic-to-real-centroid cosine 0.25 for B. sandersoni, 0.31 for B. ashtoni, 0.32 for B. flavidus). Five independent diagnostics — expert strict pass, expert structural failures, expert wrong-coloration / species-other notes, LLM strict pass, and embedding-space displacement — all rank B. sandersoni as the easiest and B. flavidus / B. ashtoni as the harder cases.

The per-species ordering is consistent with the morphological characterisations in Section 3.3 (following Williams et al., 2014 and Colla et al., 2011). B. sandersoni carries a clean two-tone black-and-yellow banding pattern that the generator can reproduce reliably from the prompt template's tergite-level colour map, which is why the expert and the LLM both pass it at near-ceiling rates. B. ashtoni is a parasitic cuckoo bee whose anatomical proportions (small wings, reduced corbicula, distinctive mandibles) lie outside the generator's prior over generic Bombus images and surface as the expert's structural-failure flags. B. flavidus's variable coloration — described in Section 3.3 as requiring qualified descriptors ("dingy pale yellow," "cream") because the word "yellow" alone triggers bright lemon generation — surfaces as the expert's wrong-coloration and "species other" residuals on otherwise structurally correct images. The synthetic–real gap is therefore not a generic property of generative augmentation applied to rare classes but a per-species function of how far each species' diagnostic morphology lies from the generator's prior.

This per-species heterogeneity is what the augmentation effects in Section 5.5 reflect. Both generative variants reduce rare-tier F1 (D3 −0.056 and D4 −0.041 under 5-fold CV), but the per-species F1 changes in Table 5.5 differ in sign and magnitude across the three species rather than moving as a rigid block. Only the three rare species received synthetic augmentation, so the absence of movement on moderate- and common-tier species is a property of the experimental design, not a finding; the diagnostic content of Section 5.5 is the per-species ordering within the augmented tier, not an aggregate "rare versus common" comparison. Whether the centroid-filtered D5 and the expert-probe-filtered D6 variants narrow the gap uniformly or continue to show the B. sandersoni > B. ashtoni > B. flavidus ordering is the direct empirical question of the pending GPU runs.

### 6.2 Why synthetic augmentation degrades performance

The clearest evidence that generative synthetic augmentation harms rare-species classification comes from the unfiltered D3 variant, in which every generation-stage-passed synthetic is added to the training set without any downstream quality screen. D3 reduces rare-tier F1 by 0.056 under 5-fold CV and by 0.075 under multi-seed (Table 5.5); it harms more rare-species test images than it improves for every species × protocol cell of Table 5.9 (the largest cell is B. flavidus at 22 % D3 harmed under multi-seed, against a 3 % ceiling for any common species). The D3 harm is therefore not a mixture of improvements and regressions that averages to a small negative — it is one-directional harm on the exact species the augmentation was designed to help.

The operative mechanism is a mismatch of feature-space coordinates between synthetic and real images of the same species. Figure 5.3a (Section 5.2.3) shows each rare species' synthetic images occupying a tight cluster that is visibly separated from the cluster of real images of the same species; the median synthetic-to-real-centroid cosine distance is 0.25–0.32, against 0.10–0.20 for real-to-real-centroid distances — roughly twice as far. The separation is per-species and pre-exists training: it is measurable in the frozen BioCLIP feature space before any fine-tuning, and the three rare species' synthetic clusters do not pool into a generic "synthetic" region but occupy three distinct displaced regions, each offset from its own real-image counterpart. Image-level inspection in the Section 5.6 contact grids confirms the displacement is not a projection artefact — synthetic B. ashtoni images carry the correct tergite colour map but cuckoo-bee anatomical distortions; synthetic B. flavidus images frequently show the wrong yellow shade or tergite pattern; synthetic B. sandersoni images often look correct but occasionally miss the clean T2 → T3 colour boundary. Section 5.6.2 quantifies the image-level picture: for a harmed rare-species test image and its five nearest training synthetics in BioCLIP space, the median test-to-neighbour cosine similarity is 0.56 — well below the 0.7+ range typical of two real images of the same species.

Single-species subset ablation (Section 5.5) converts the correlational evidence into a causal one. Under D3, removing B. ashtoni's synthetics from training recovers B. ashtoni F1 by 0.182 and removing B. flavidus's recovers B. flavidus F1 by 0.062, confirming that the synthetics in question were collectively harming the target species. Cross-species collateral is substantial — dropping B. ashtoni's synthetics simultaneously reduces B. sandersoni F1 by 0.150 and increases B. flavidus F1 by 0.124 in the same run — indicating that the displaced synthetic cluster affects multiple class decisions simultaneously through shared features at the classifier's penultimate layer. The additive counterpart, training D1 + B. sandersoni-only synthetics, D1 + B. ashtoni-only, and D1 + B. flavidus-only, is registered in Section 5.5 and directly tests whether the single best-generated species (B. sandersoni, Section 6.1) is capable of helping when added in isolation or whether any generative synthetic pool, regardless of composition, carries the displacement penalty.

Volume ablation (Section 5.5) closes the most natural alternative explanation: scaling the synthetic pool from +50 to +500 images per species does not close the gap. This rules out generation *quantity* as the bottleneck and fixes the diagnosis on generation *fidelity*. D3's downstream harm is not a volume problem that a better generator would solve by producing more images of the same kind; it is a region-of-feature-space problem that either a quality filter acting on the synthetic pool (the D5 and D6 variants of this thesis) or a generator whose outputs sit within the real-image region to begin with (D2 CNP, Section 6.3) must address before generative augmentation can help rare classes.

### 6.3 Copy-and-paste vs. generative: real-image feature preservation as the operative variable

CNP (D2) is the only augmentation method in the study that produces a statistically significant per-species F1 gain (B. flavidus +0.059, p = 0.005) and the only method that lifts rare-tier F1 above baseline under 5-fold CV (+0.030). The mechanism is consistent with Section 6.2: CNP preserves real morphological texture by segmenting real bees and compositing them onto real flower backgrounds, so its outputs sit inside the same region of BioCLIP feature space that real training images of the same species occupy, rather than in the displaced region the generative synthetics inhabit. The cost is a diversity ceiling: with only 22 real training images of B. ashtoni, CNP can produce new compositions but cannot introduce intra-class variation in pose, lighting, or morphology beyond what the source images contain. This explains why CNP's largest gain is for the rare species with the most source images (B. flavidus, n = 162) and is essentially flat for the species with the fewest (B. ashtoni, n = 22).

The trade-off is not "fidelity vs. diversity" in the abstract — it is specifically that generative augmentation places its outputs in a region of feature space that the classifier cannot reconcile with the real-image region during fine-tuning, while CNP does not. D5 (centroid) and D6 (probe) are designed to close this gap by selecting the 200 synthetics per species closest (D5) or most expert-aligned (D6) to the real-image distribution; whether this per-species filtering brings the selected subset's feature-space footprint into the real-image region — and whether downstream F1 responds — is the direct question of the pending evaluation.

### 6.4 LLM judges vs. expert judges

The LLM judge and the 150-label expert operate on the same rubric — five morphological features, blind species identification, diagnostic-level classification — but produce systematically different decisions. Quantitative agreement is weak: the LLM's binary strict gate and the expert's binary strict rule coincide on 84 of 150 images (56 %), only ~7 percentage points above the 49 % rate expected under independence, with LLM precision 0.40 and recall 0.62 against the expert (Section 5.4.2); treating the LLM's mean morph score as a ranker against expert strict labels yields AUC 0.56 (0.54 against the lenient rule) — the LLM morph-mean is a near-random ranker of the expert pass / fail signal. The per-feature heatmap (Figure 5.8) locates the disagreement: the LLM is over-strict on B. ashtoni thorax coloration (59 % over-strict rate — the expert accepts coloration the LLM rejects on more than half its B. ashtoni images) and has large blind spots on B. flavidus across head, legs, and coloration features (26–44 % blind-spot rates — the LLM accepts B. flavidus synthetics the expert rejects). At the pass-set level, a filter rule that can only see what the LLM verbalises discards exactly the images the expert identifies as most useful: within an identical 200-image budget per species, the expert-supervised probe (D6) retains 16 / 20 / 6 expert-strict synthetics for B. ashtoni / B. sandersoni / B. flavidus against 5 / 12 / 1 under D4, i.e. roughly 3× more expert-validated images within the same volume cap (Section 5.4.4).

The qualitative picture from the Section 5.4 contact sheets makes the disagreement concrete. The LLM-passed / expert-rejected quadrant contains B. ashtoni images with correct tergite colour maps but parasitic-cuckoo anatomical distortions (reduced corbicula, unusual head proportions) that the LLM's surface-level morphology rubric does not penalise; B. flavidus images where the yellow shade is slightly off or a tergite band is misplaced — details the LLM scores ≥ 4 on its five-feature rubric but which the expert flags as "wrong coloration" or "species other" (22 and 18 of 50 respective rates, Section 5.4.1); and B. sandersoni images where the pattern looks clean from a lateral angle but the diagnostic T2 → T3 boundary is missing or inverted on a dorsal view. The LLM-rejected / expert-accepted quadrant is the complement: images the LLM marks down for a single feature it happens to score low, yet the expert reads as adequate on the composite. Neither error is a simple calibration drift; the two are distinct failure modes driven by what a language-mediated rubric can and cannot encode about fine-grained morphology.

The operational cost of this miscalibration is what Section 5.5 reports: D4 (the LLM-filtered variant) is statistically indistinguishable from D3 (unfiltered) under 5-fold CV (p = 0.777) and significantly worse than baseline on rare-tier F1 (D1 vs. D4 p = 0.041). The subset-ablation reversal for B. sandersoni (Section 5.5) sharpens this into a single empirical demonstration: under D3 B. sandersoni's full synthetic set is collectively helpful (removing it loses 0.095 F1); under D4 the LLM-filtered subset is collectively harmful (removing it gains 0.081 F1) — for the species the LLM passes most easily (91.2 % strict pass rate), the 8.8 % the LLM discards happens to be exactly the subset that was compensating for the harmful portion. A defensible quality filter must therefore use signals beyond the LLM's per-feature scores — the BioCLIP centroid distance of Section 5.2.3 or the expert-supervised linear probe developed in Section 4.4. Whether the filter-level improvement translates into downstream F1 — the direct empirical question — is [TODO: pending the D5 and D6 GPU runs registered in Section 5.5].

### 6.5 Statistical challenges with rare species

Evaluating augmentation strategies for rare species creates a fundamental tension: the species that most need augmentation are precisely those for which evaluation is least reliable. With n = 6 test images for B. ashtoni on the fixed split, a single flipped prediction changes F1 by up to 0.17 and reorders the aggregate ranking across augmentation methods. The dual-protocol design adopted in Section 3.5 partially addresses this: 5-fold cross-validation pools predictions to reach rare-species effective n = 32 / 58 / 232, supporting the only statistically significant pairwise comparisons in the study (D1 vs. D4 p = 0.041; D2 vs. D4 p = 0.030; D2 flavidus +0.059 p = 0.005); multi-seed training on the fixed split keeps the same test images across seeds, supporting the per-image flip and chain analyses that the cross-fold protocol cannot. The protocols are reconciled in Section 5.5.1: their disagreement on aggregate ranking reflects the fixed-split test-set composition, not a methodological inconsistency.

The remaining residual is statistical power. Our paired t-tests use only five data points per comparison — one per fold of the 5-fold cross-validation — so they can only reliably detect large effects; when a comparison such as D1 vs. D2 (p = 0.161) or D1 vs. D3 (p = 0.096) fails significance, it most likely means we could not measure the gap at this sample size, not that there is no gap. For this reason the thesis reports bootstrap confidence intervals alongside every point estimate: a CI shows the range of effect sizes the data are compatible with, whereas a non-significant p-value on five data points tells us almost nothing about the effect's true magnitude. The prose in Section 5 follows that priority, and underpowered comparisons are flagged as "consistent with harm but below significance at df = 4" rather than as null results.

### 6.6 Implications for urban biodiversity monitoring

The operational failure mode this thesis targets is **rare-species detection failures masquerading as true absences** in the monitoring record. An automated camera-trap pipeline treats each classifier prediction as an occurrence observation: when the classifier outputs "no B. ashtoni" for an image stream, the downstream record cannot distinguish a *true absence* (no B. ashtoni was present to photograph) from a *detection failure* (B. ashtoni was photographed but the classifier mis-routed the prediction to a different species). The two outcomes are indistinguishable from the sensor stream alone — which is precisely why ecological survey methodology quantifies detection probability separately from occupancy (MacKenzie et al., 2002, Section 1.1). For a long-tailed classifier, every rare-species false negative becomes a detection failure silently relabelled as a true absence in the biodiversity record.

For the Sensing Garden / Flik pipeline introduced in Section 1.3, the classifier studied here is the species-level decision stage that generates those records, so rare-species F1 — not aggregate accuracy — is the metric that translates most directly into deployment trustworthiness. The cost of the two error types is asymmetric. An erroneous "B. ashtoni present" record can be corrected by a follow-up ecological survey; an erroneous "B. ashtoni absent" record propagates into multi-site biodiversity indicators and into the conservation-policy and urban-planning decisions those indicators inform, silently under-reporting a species whose true status is exactly what an automated monitoring system is meant to determine.

Three specific implications follow for this reporting context. First, the rare-tier harm under D3 and D4 (Section 5.5) is operationally dangerous: D4 reduces macro F1 by 0.011 under 5-fold CV (p = 0.041) and rare-tier F1 by 0.041, which scales at deployment into a measurable increase in false-absence rates for exactly the species an automated monitoring system is meant to help surface. Generative augmentation without a classifier-relevant filter should therefore not be pushed into the downstream classifier of a production monitoring pipeline. Second, copy-and-paste augmentation (D2) — the only method with a statistically significant rare-species gain (B. flavidus +0.059, p = 0.005, Section 6.3) — is the defensible baseline for near-term deployment when source images permit. Third, the filter-level evidence of Section 5.4 — that the expert-calibrated probe retains ~3× more expert-validated synthetics within the same 200-image budget — suggests a path to recovering generative augmentation's diversity advantage without its fidelity penalty; whether that path produces downstream classifier gains is the direct question registered for completion in Sections 5.5 and 5.6.

Until the synthetic-real feature-space gap is closed, the deployment recommendation is to prefer copy-and-paste augmentation where source images permit (Section 6.3) and to treat generative augmentation as conditional on a quality filter that uses classifier-relevant signals — BioCLIP centroid distance or an expert-supervised probe — rather than the LLM judge's per-feature scores alone (Section 6.4). Any generative augmentation entering an operational pipeline such as Sensing Garden's should pass through a filter validated against domain expertise on a stratified sample from the deployment distribution, following the 150-label protocol of Section 4.4 or its multi-annotator extension (Section 7.2.1).

### 6.7 Limitations

The findings are bounded by constraints on model selection, dataset scope, annotation scale, experimental design, and statistical power. Each limitation below is scoped to the specific claim in Section 5 or Section 6 it qualifies.

#### 6.7.1 Model and architecture

**Single classifier architecture.** All classification experiments use ResNet-50. Architectures with stronger biology-specific pre-training — for instance, a BioCLIP or DINOv2 backbone — may respond differently to synthetic augmentation because their representations are pre-aligned to the real-image distribution that the synthetic pool diverges from (Section 6.2). Replicating the D1–D6 comparison under alternative backbones would test whether the rare-tier harm is architecture-specific or a property of fine-tuning any ImageNet-initialised classifier on displaced synthetic data.

**Single generative model.** Synthetic images are produced by GPT-image-1.5 via the `images.edit` endpoint — a closed, non-fine-tunable model. The per-species feature-space offset (Section 5.2.3) may therefore partly reflect a general-purpose generator with no domain adaptation rather than a universal property of generative augmentation. Open-source diffusion models (Stable Diffusion, FLUX, Qwen-Image) fine-tuned on entomological specimens via LoRA provide the natural comparison; if the offset persists, the mechanism generalises, and if it closes, the fix is domain adaptation at the source rather than filtering downstream.

#### 6.7.2 Dataset scope and generalisability

**Three target species.** The mechanistic account is demonstrated for B. ashtoni, B. sandersoni, and B. flavidus. The morphological-atypicality argument — that rare classes deviate most from the generator's prior and therefore suffer the largest offset — predicts the same pattern in other fine-grained taxa, but extending to additional rare species and to non-bee domains is future work, not a completed claim.

**Fixed augmentation volume.** Every rare species receives +200 synthetics regardless of its original training-set size (n = 22 / 40 / 162) or its generation difficulty. The volume ablation (Section 5.5.3) shows the harm does not resolve between +50 and +500, so quantity is not the dominant variable — but a proportional or difficulty-weighted design would test directly whether per-species volume interacts with filter choice, which the present protocol cannot.

#### 6.7.3 Expert annotation

**Annotation scale.** Expert calibration uses 150 stratified images. This is sufficient for a filter-design study — LOOCV AUC 0.792 is well above chance — but the per-species support (n = 50 each) limits the precision of per-species F1-max thresholds, especially for B. flavidus where only 6 / 50 are expert-strict. A larger annotated pool would tighten the thresholds and reduce reliance on the LOOCV-F1 maximum as a point estimate.

**Single-annotator calibration.** The 150 labels come from one entomologist. The reported LOOCV AUC (0.792) and LLM-expert strict-pass agreement (56 %, 7 pp above chance under independence) are conditional on this annotator's judgment; inter-rater reliability has not been measured and Cohen's κ cannot be reported without a second annotator. Replicating the protocol with ≥ 3 annotators would quantify annotator variance as a fraction of the agreement and AUC gaps, and is the first item of Section 7.2.

#### 6.7.4 Experimental design

**No comparison with loss re-weighting or alternative imbalance methods.** Focal loss, class-balanced loss, and decoupled training address imbalance through the training objective rather than the data. They are complementary to augmentation, not competing, and are not expected to resolve the feature-space offset that drives the rare-tier harm (Section 6.2). Combining augmentation with objective-side methods is future work.

**Shared hyperparameters across dataset variants.** Learning rate, weight decay, dropout, and batch size are held fixed across D1–D6 to isolate the effect of training data. Per-variant hyperparameter tuning could alter the relative ranking — particularly if D5 or D6 benefit from a different regularisation setting than D3 or D4 — but would confound the "same classifier, different data" logic that the comparison depends on.

**Single-seed subset ablation.** The causal attribution in Section 5.6.4 is run at seed 42 only. With rare-species test n between 6 and 36, a 0.05 F1 change corresponds to flipping 0.3–1.8 images, so the analysis is designed to establish direction of effect rather than statistically powered magnitude. The B. sandersoni D3 → D4 sign reversal is robust to this caveat because direction reversal is a qualitative signal; cross-species collateral magnitudes should be read as directional only.

#### 6.7.5 Statistical power and feature-space proxy

**Limited paired-test power.** 5-fold and 5-seed paired t-tests run with df = 4, which admits only large effect sizes at α = 0.05. Non-significant comparisons (for example D1 vs. D2, p = 0.161; D1 vs. D3, p = 0.096) are underpowered rather than null (Section 6.5). Bootstrap 95 % CIs — reported alongside every point estimate — are the more informative uncertainty quantification for rare-species F1.

**Feature-space proxy.** The embedding analyses in Section 5.2 and the failure-chain retrievals in Section 5.6.2 use BioCLIP rather than the ResNet-50 classifier's penultimate-layer features. BioCLIP was selected on a 5-NN species-purity diagnostic that places it well above DINOv2 on this dataset (Table 5.2), but it remains an external proxy for the classifier's own representation. A ResNet-50 penultimate-layer replication is listed in Section 7.2 to test whether the same offset appears in the classifier's internal feature space.

#### 6.7.6 Pending computations

**D5 and D6 downstream training pending.** The forward-looking claim of the thesis — that the expert-probe filter closes the miscalibration gap identified in Section 6.4 — rests on the D5 (centroid) and D6 (expert-probe) downstream F1 numbers marked `[TODO]` throughout Section 5.5 and Section 5.6. The filter-level evidence of Section 5.4 registers the prediction; Sections 5.5 and 5.6 will be updated once the multi-seed and 5-fold training runs complete. The mechanistic argument in Section 6.2 does not depend on these results.

**CKNNA alignment not yet computed.** The representational-alignment metric specified in Section 4.4.8 is deferred until the D5 / D6 variant embeddings are available. CKNNA adds a direct filter-level alternative to the Jaccard overlaps and expert-strict coverage of Section 5.4.4 but does not condition the mechanistic story.

Taken together, these limitations define the study as a well-controlled empirical case rather than a universal generalisation: the mechanism is demonstrated for one architecture, one generator, three rare species, and one expert, and the forward-looking filter-replacement claim is registered rather than confirmed pending the D5 / D6 runs.


## 7. Conclusion and Future Work

### 7.1 Summary

This thesis investigated the synthetic-real gap for fine-grained biodiversity classification under extreme class imbalance. Four contributions were made.

First, a **structured morphological prompting framework** using negative constraints, tergite-level colour maps, and reference-guided generation eliminated all structural failures across 1,500 generated images and isolated coloration fidelity as the sole remaining generation bottleneck. LLM-strict pass rates range from 44.4 % (B. ashtoni) to 91.2 % (B. sandersoni), tracking morphological deviation from the genus-typical phenotype.

Second, a **two-stage LLM-as-judge evaluation protocol** combining blind taxonomic identification with per-feature morphological scoring provided interpretable diagnostic signals that identify *where* generated images fail. The per-feature decomposition pinpoints the residual coloration-fidelity bottleneck (B. ashtoni thorax coloration mean 2.98, every other (species × feature) cell ≥ 4.0).

Third, an **expert-calibrated quality filtering pipeline** was specified, implemented, and evaluated at the filter level on a stratified 150-image expert-annotated sample and the full 1,500-image pool. A BioCLIP-feature linear probe supervised on 150 expert labels achieves LOOCV AUC 0.792 under the expert strict rule, with per-species F1-maximising thresholds τ = 0.125 / 0.495 / 0.222 for B. ashtoni / B. sandersoni / B. flavidus. On the 1,500-image pool, the probe selects a 200-image subset per species that is roughly 3× richer in expert-validated images than the LLM-rule filter (16 / 20 / 6 expert-strict against 5 / 12 / 1). The LLM-rule filter agrees with the expert on strict pass / fail for only 84 of 150 images (56 %, ~7 percentage points above chance under independence), and the LLM morph-mean is an AUC-0.56 ranker against expert strict labels. Downstream F1 comparisons for the resulting D5 (BioCLIP-centroid) and D6 (expert-probe) dataset variants are [TODO: multi-seed × fixed split and 5-fold CV training runs were launched as a 4 × 5-task SLURM array and had not completed at submission].

Fourth, **systematic empirical comparison under five-fold cross-validation, multi-seed training, and per-image failure analysis** established that copy-and-paste augmentation (D2) produces the only statistically significant rare-species F1 gain in the study (B. flavidus +0.059, p = 0.005), while both unfiltered (D3) and LLM-filtered (D4) generative augmentation reduce rare-tier F1 below baseline; D4 is significantly worse than baseline (p = 0.041) and statistically indistinguishable from D3 (p = 0.777). The harm is mechanistically attributable to a per-species feature-space offset between synthetic and real images (BioCLIP synthetic-to-centroid cosine median 0.25–0.32 vs. real-to-centroid 0.10–0.20) and is confirmed causally by single-species subset ablation, including a B. sandersoni D3 → D4 sign reversal that demonstrates the LLM filter discards the wrong subset for the species it passes most easily. These findings motivate expert-calibrated filtering — not as an incremental refinement of the LLM rule, but as a replacement that uses classifier-relevant signals the LLM judge cannot access. The pending D5 and D6 runs supply the direct downstream test.

### 7.2 Future Work

Section 6.7 identified specific constraints on this study's conclusions; Section 7.2 names the methodological actions that would lift them. Completing the registered downstream tests is a deferral notice (Sections 6.7.6 and 5.6.5), not a future-work item, and is not repeated here. The four directions below group related improvements: strengthening the learned filter (7.2.1), testing the mechanism under alternative architectures and generators (7.2.2), extending the scope beyond three bee species (7.2.3), and closing the loop through field deployment (7.2.4).

#### 7.2.1 Strengthen the expert-calibrated filter

Two improvements address the statistical fragility of the current D6 probe. First, replicate the 150-label annotation protocol with at least three independent annotators drawn from distinct institutional backgrounds — museum curators, field entomologists, taxonomic researchers — report inter-annotator κ and per-feature 95 % CIs, and retrain the probe on the consensus strict label. This quantifies how much of the LLM–expert strict-pass agreement gap and the 0.792 LOOCV AUC reflect annotator subjectivity versus genuine calibration signal, and enables the inter-rater κ (undefined at n = 1 annotator) to be reported (Section 6.7.3). Second, scale the annotation study to 300–500 images with stratified sampling across caste, viewing angle, and LLM tier, tightening the per-species F1-max thresholds currently pinned to as few as 6 expert-strict positives per species (Section 5.4.3). Together these deliver a more robust and better-calibrated probe suitable for deployment.

#### 7.2.2 Test the mechanism under alternative architectures and generators

The harm identified in Section 5.5 is demonstrated for one classifier (ResNet-50), one generator (GPT-image-1.5), and one feature-space proxy (BioCLIP). Three swap-and-retest experiments test which components the mechanism depends on. *(i) Classifier backbone:* replace ResNet-50 with a BioCLIP or DINOv2 backbone and repeat the D1–D6 comparison, testing whether rare-tier harm generalises to richer pre-trained representations or is a property of fine-tuning an ImageNet-initialised classifier on displaced synthetic data (Section 6.7.1). *(ii) Classifier-internal representation:* retrain the D6-style probe on ResNet-50 penultimate-layer embeddings, testing whether the BioCLIP-anchored selection geometry (Section 6.7.5) transfers to the representation that actually drives downstream F1; if the two probes produce the same expert-strict coverage ranking, the BioCLIP proxy is vindicated. *(iii) Generation stage:* replace GPT-image-1.5 with a fine-tunable open-source diffusion model — Stable Diffusion XL, Stable Diffusion 3, FLUX, or Qwen-Image — and adapt it to each rare species via LoRA fine-tuning on curated specimen imagery. If LoRA adaptation closes the per-species feature-space offset (Section 5.2.3) at the generation stage, downstream filtering becomes a secondary defence; if it does not, the Section 5.4 filter replacement remains load-bearing. Closing the gap at source is the most direct route to making generative augmentation consistently helpful rather than conditionally harmful for rare taxa.

#### 7.2.3 Extend to additional rare taxa and fine-grained domains

The mechanistic account (Section 6.2) makes a falsifiable prediction outside this thesis's scope: rare classes in other long-tailed fine-grained domains should exhibit a per-species feature-space offset proportional to the class's deviation from the generator's prior. Test this on at least two additional datasets — for instance, three more rare bee species drawn from GBIF and a rare-fungi or rare-plant dataset (Picek et al., 2022) — and report whether the per-species offset, the LLM–expert strict-pass agreement gap, and the expert-probe LOOCV AUC replicate in direction and magnitude (Section 6.7.2). The 150-label expert protocol of Section 4.4 is designed to port across taxa with only the reference-image set and the diagnostic-feature list changing, which makes this extension a matter of new annotation data rather than new method.

#### 7.2.4 Deployment integration with Sensing Garden

Integrate the D6 probe-filtered augmentation pipeline into the Sensing Garden Flik classifier as the species-level decision stage beneath Flik's hierarchical detection, tracking, and crop-extraction stack (Section 6.6). Two deliverables frame the integration: a per-deployment-site validation study measuring the operational false-absence rate on rare species against conventional ecological surveys at matched times, and a continuous feedback loop in which field-collected expert labels on ambiguous tracks feed back into the expert-probe's training set on a seasonal cadence. A successful integration would demonstrate that the rare-species augmentation pipeline reduces the operational false-absence asymmetry (Section 6.6) rather than merely improving a held-out test metric, and would close the loop from data curation and generation to automated urban-biodiversity indicators that inform conservation policy and urban-planning decisions.

Taken together, these four directions determine whether the mechanism demonstrated on three rare Massachusetts bumblebee species extends into a general, deployment-ready pipeline for long-tailed fine-grained classification in biodiversity monitoring.


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
Confusion matrices per dataset version, volume ablation full table, per-seed breakdowns, complete 16-species results table, per-angle strict-pass-rate table, caste-fidelity breakdown (referenced in Section 5.3.2), background-removal diagnostic (referenced in Section 5.5.3), per-tier rare / moderate / common F1 breakdown across D1–D6 (referenced in Section 5.5.1), and per-LLM-tier counts per species (referenced in Section 5.3.1).

### Appendix F: Failure-Mode Analysis Assets
Embedding atlases at true t-SNE coordinates (Section 5.2.4); confusion-pair triplets for B. ashtoni × {B. citrinus, B. vagans}, B. sandersoni × B. vagans, B. flavidus × B. citrinus (Section 5.1.2); full per-species galleries of real, synthetic, and harmed-test images (Section 5.6); 49 D3-harmed, 52 D3-improved, 49 D4-harmed, and 49 D4-improved failure chains with t-SNE projections (Section 5.6.2; legacy on-disk filenames prefixed `chains_d4_*`, `chains_d5_*` correspond to the present thesis's D3 and D4 respectively); full LLM-vs-centroid 4-quadrant counts per species (Section 5.6.3); full dropped-versus-measured 3 × 3 subset-ablation recovery matrix for D3 and D4 (Section 5.6.4). D5 and D6 chain galleries, subset-ablation cells, and 4-quadrant counts are [TODO: pending the D5 / D6 training outputs].


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
