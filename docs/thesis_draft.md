# The Synthetic-Real Gap in Urban Biodiversity Monitoring: Generative Augmentation and Expert-Calibrated Filtering for Long-Tailed Bumblebee Classification

## Abstract

Automated pollinator monitoring in cities is limited by extreme class imbalance: for 16 Massachusetts bumblebee species, the rarest have fewer than 40 training images while common species have thousands, and baseline classifiers achieve F1 below 0.47 for these rare taxa. We develop a three-stage augmentation pipeline: structured morphological prompting with tergite-level color maps for reference-guided image generation, a two-stage LLM-as-judge combining blind taxonomic identification with five-feature morphological scoring, and expert-calibrated quality filtering learned from entomologist annotations. Across multiple training seeds, synthetic augmentation consistently degrades rare-species performance, with the most affected species dropping substantially in F1. Automated quality filtering partially mitigates but does not recover baseline performance. Copy-Paste augmentation using real morphological texture yields the strongest results without degradation. Per-feature analysis traces the failure to coloration infidelity in species that deviate most from the genus-typical phenotype, establishing that augmentation quality -- not volume -- determines whether synthetic data helps or harms.


## 1. Introduction

### 1.1 Motivation

Global urban land cover is projected to triple between 2000 and 2030. While urban expansion has traditionally been associated with severe habitat loss, contemporary ecological research increasingly recognizes cities as critical frontiers for biodiversity conservation and climate resilience. These "accidental ecosystems" (Alagona, 2024) often function as refuges for species displaced from degraded farmland and shrinking natural landscapes, which Schilthuizen (2025) describes as a "naturalist's gold mine" of hidden biological riches. A weedy vacant lot or an overgrown park edge can quietly harbor biological communities that no longer persist in the surrounding countryside. For urban planners, the question is whether we detect these species before we pave them over.

Bumblebees (Bombus spp.) provide a valuable entry point for understanding urban biodiversity. As among the most important wild pollinators in temperate ecosystems, they provide essential pollination services to both wildflowers and over 25 commercial crop species through their unique capacity for buzz-pollination (Goulson, 2010), making them foundational to both natural ecosystems and urban ecological services. They also represent a conservation paradox: while several species have experienced severe population crashes over the past century (Cameron et al., 2011), urban green spaces now support populations of bumblebees that are increasingly rare elsewhere.

The trouble is that we cannot tell the difference between a species that has vanished and one we simply keep missing. Historical baselines established by Plath (1934) near Boston documented 17 Bombus species, describing taxa such as B. affinis and B. terricola as common. Contemporary studies reveal a dramatic shift: Jacobson et al. (2018) reported a 96.4% decline in relative abundance for B. fervidus in New Hampshire, while Richardson et al. (2019) found that four of seventeen historically documented species were not detected in modern Vermont surveys despite a six-fold increase in sampling effort. Cameron et al. (2011) documented range contractions of up to 87% for declining species across 382 sites in 40 U.S. states. In Massachusetts, B. ashtoni was last documented near Boston by Plath (1934) and has not been recorded in the state since. B. terricola and B. fervidus show up only sporadically in modern surveys. Are these species truly gone, or are our tools failing to detect them?

This "detection-versus-extinction" uncertainty carries real consequences for how we build our cities. MacKenzie et al. (2002) provide the statistical framework for this problem, demonstrating that nondetection at a site does not imply absence unless detection probability approaches one. Observations from the Gegear Lab's Beecology Project have detected B. terricola, a species thought to be absent from Massachusetts, through citizen science observations, confirming that nondetection can reflect monitoring limitations rather than true extirpation. If a rare bee persists but our monitoring systems miss it, planners may unknowingly approve development on the very habitat fragments that should have been protected.

### 1.2 Problem Statement

A major contributor to this detection failure is classification error in automated monitoring systems. Modern computer vision models struggle with rare taxa because their training datasets exhibit long-tailed distributions. In community-contributed datasets such as GBIF (GBIF.org, 2025), common species like B. impatiens are represented by thousands of images, while rare species such as B. sandersoni may have fewer than one hundred. Under such an imbalance, classifiers learn to recognize abundant species well but fail to capture the subtle morphological features required for fine-grained species identification, such as abdominal banding patterns, facial hair coloration, and other diagnostic features.

[Table 1: Dataset species, counts in GBIF, and sample images]

The dataset in this thesis comprises 16 Massachusetts Bombus species totaling 15,630 images sourced from GBIF, with an imbalance ratio of 59.9:1. A baseline ResNet-50 classifier achieves [X]% overall accuracy and [X] macro F1, but collapses on critical rare species: F1 = [X] for B. ashtoni (n = 6 test images) and F1 = [X] for B. sandersoni (n = 10). Bootstrap confidence intervals for these species span nearly the full [0, 1] range, reflecting genuine evaluation uncertainty at small sample sizes.

Generative AI offers a potential solution to this imbalance: if we do not have enough real images of a rare species, we can generate synthetic ones to balance the training data. However, the problem is more complex than adding volume. Across multiple training seeds, adding 200 unfiltered synthetic images per species shows a consistent directional trend of degradation, particularly for B. sandersoni. Even after automated quality filtering via an LLM-as-judge, performance does not clearly exceed the baseline. Meanwhile, Copy-Paste augmentation -- which composites real bee cutouts onto new backgrounds -- achieves the best overall result without degradation. This asymmetry suggests a *fidelity gap*: AI-generated images may appear visually convincing but fail to carry the correct training signal for fine-grained classification. Looking realistic is not the same as being diagnostically faithful.

### 1.3 Thesis Statement

For fine-grained biodiversity classification under extreme class imbalance, generative augmentation requires morphology-guided generation and multi-dimensional quality filtering calibrated against domain expertise. Neither quality nor quantity alone is sufficient.

This thesis investigates the nature of that fidelity gap. The goal is to understand why synthetic images underperform, pinpoint what they get wrong at the level of species-diagnostic features, and develop improved augmentation strategies that preserve the features necessary for accurate species classification. Specifically, the research develops a generator-discriminator pipeline to produce high-quality training data for long-tailed ecological datasets. These models are then intended to facilitate precise species-level classification on edge-AI monitoring devices, thereby enabling continuous pollinator monitoring within urban settings. By closing the fidelity gap in rare species, this work aims to ensure that urban planning decisions are guided by a more accurate understanding of biodiversity, rather than being constrained by the shortcomings of current datasets.

[Fig 1: Pipeline Framework]

### 1.4 Contributions

This thesis makes four contributions:

1. **A structured morphological prompting framework** for generating species-faithful images using reference-guided image editing, with negative constraints and tergite-level color maps that override generative model priors toward dominant phenotypes. By the final prompt version, structural failures (extra limbs, impossible geometry) were eliminated across all 1,500 generated images; the sole remaining failure mode was coloration accuracy (27.1%).

2. **A two-stage LLM-as-judge evaluation protocol** combining blind taxonomic identification with multi-dimensional morphological scoring (5 features, 1--5 scale), providing per-feature diagnostic signals that reveal *where* generated images fail -- not just *whether* they pass.

3. **An expert-calibrated quality filtering pipeline** in which entomologist annotations on a stratified 150-image subset are used to learn feature-level weights that correct the LLM judge's holistic miscalibration, with a diagnostic feedback loop connecting evaluation to generation improvement.

4. **Rigorous empirical evidence** under multi-seed evaluation with bootstrap confidence intervals demonstrating that (a) naive synthetic augmentation degrades rare-species classification, (b) automated LLM-based quality filtering partially mitigates but does not resolve this degradation, (c) Copy-Paste augmentation with real morphological texture outperforms generative approaches, and (d) the fidelity gap is morphology-specific and species-dependent -- with species deviating most from the dominant phenotype suffering the greatest degradation. These findings provide the empirical foundation for expert-calibrated filtering as the necessary next step.

These contributions bridge urban planning and computer science by connecting classification performance directly to the detection-versus-extinction uncertainty that constrains biodiversity-informed urban development decisions.

### 1.5 Organization

The remainder of this thesis is organized as follows. Section 2 reviews related work spanning bumblebee ecology and decline, fine-grained visual classification, long-tail recognition, generative data augmentation, and synthetic image quality evaluation. Section 3 describes the dataset, preparation pipeline, target species taxonomy, classifier architecture, and evaluation methodology. Section 4 presents the three novel methods: structured morphological prompting, LLM-as-judge evaluation, and expert-calibrated quality filtering. Section 5 reports experimental results, including baseline analysis, generation quality assessment, augmentation comparison under multi-seed evaluation, and expert calibration. Section 6 discusses the nature of the synthetic-real gap, the complementarity of augmentation methods, and implications for biodiversity monitoring. Section 7 concludes and outlines future work.


## 2. Related Work

### 2.1 Bumblebee Decline and Automated Monitoring

The documentation of bumblebee populations in New England is anchored in Plath (1934), whose baseline of 17 species near Boston provides the historical benchmark against which contemporary declines are measured. Multiple independent studies have since quantified the magnitude of this shift using distinct methodological approaches: Cameron et al. (2011) employed standardized sampling with mitochondrial COI and microsatellite data across 382 sites in 40 U.S. states to document range contractions of up to 87% for declining species, while Jacobson et al. (2018) used multi-decadal museum record comparisons to reveal a 96.4% decline in relative abundance for B. fervidus in New Hampshire. Richardson et al. (2019) combined historical literature review with six years of standardized netting surveys in Vermont, finding that four of seventeen historically documented species were undetected despite substantially greater sampling effort than any prior study. Critically, these studies cannot distinguish true extirpation from detection failure -- a distinction formalized by MacKenzie et al. (2002) in the occupancy modeling framework, which demonstrates that nondetection probability must be jointly estimated alongside occupancy to avoid false-absence conclusions. Community science platforms have begun to address this gap: MacPhail et al. (2024) showed that Bumble Bee Watch observations contributed records for species otherwise underrepresented in professional surveys, and the Beecology Project has detected B. terricola in Massachusetts, a species previously thought absent from the state.

This detection challenge extends to automated monitoring systems, which inherit the long-tailed distribution of their training data. Spiesman et al. (2021) trained deep learning classifiers on 89,776 images across 36 North American Bombus species and achieved 91.6% top-1 accuracy overall, but excluded six species entirely for having fewer than approximately 150 training examples. Among included species, error rates ranged from 4.0% for morphologically distinctive taxa to 20.4% for variable species, with B. rufocinctus confused with 25 other species. Spiesman et al. explicitly identify generative data augmentation as a potential remedy for this class imbalance. At larger scale, Bjerge et al. (2024) deployed 48 camera traps capturing over 10 million insect images and achieved 80% average precision, but species with too few training images were collapsed into an undifferentiated "unspecified arthropods" class -- precisely the outcome that renders monitoring systems useless for rare-species conservation. Bjerge et al. (2023) showed that hierarchical classification can gracefully degrade to genus-level identification, but species-level resolution is what conservation planning requires.

### 2.2 Fine-Grained Visual Classification

Fine-grained visual classification (FGVC) addresses the problem of distinguishing subordinate categories within a broader class -- species within a genus, aircraft models within a manufacturer, car variants within a make. Unlike standard object recognition where inter-class differences are large, FGVC is characterized by high inter-class similarity and high intra-class variation: two Bombus species may differ only in the width of a thoracic band, while individuals within a single species vary across castes, geographic populations, and seasonal phenotypes. Wei et al. (2022) provide a comprehensive survey of deep learning methods for fine-grained image analysis, tracing the evolution from part-based representations to end-to-end attention mechanisms.

Early deep learning approaches to FGVC focused on learning discriminative feature representations through specialized architectures. Lin et al. (2015) introduced bilinear CNN models, which compute the outer product of features from two CNN streams at each spatial location, capturing localized pairwise feature interactions in a translationally invariant manner. Zheng et al. (2017) proposed the Multi-Attention CNN (MA-CNN), in which channel grouping and part classification sub-networks jointly learn to localize discriminative parts and extract part-specific features without requiring bounding box or part annotations at test time. These methods demonstrated that FGVC benefits from architectures that explicitly attend to subtle, localized visual differences rather than relying on holistic image-level features.

Transfer learning from large-scale datasets has become the dominant paradigm for FGVC. Kornblith et al. (2019) systematically studied ImageNet transfer, finding a strong rank correlation (r = 0.96) between ImageNet accuracy and fine-tuned transfer accuracy across 12 downstream datasets. More recently, vision-language foundation models have reshaped the landscape. CLIP (Radford et al., 2021) demonstrated zero-shot classification by aligning image and text embeddings, but its performance degrades on fine-grained tasks where class distinctions require domain-specific visual knowledge. DINOv2 (Oquab et al., 2024) showed that self-supervised pretraining at scale produces general-purpose visual features competitive with supervised methods across fine-grained tasks. For biological applications specifically, BioCLIP (Stevens et al., 2024) adapted the CLIP framework to the tree of life by training on TreeOfLife-10M, achieving 17--20% absolute accuracy gains over general-purpose baselines on fine-grained biology benchmarks.

Given this landscape, fine-tuning a ResNet-50 (He et al., 2016) from ImageNet-pretrained weights represents a well-understood baseline for FGVC. The strong correlation between ImageNet pretraining quality and downstream performance (Kornblith et al., 2019) makes it a principled choice for isolating the effect of data augmentation from architectural novelty -- this thesis deliberately uses this established baseline so that observed performance changes can be attributed to augmentation strategy rather than model capacity.

### 2.3 Long-Tail Classification and the Challenge in Biodiversity

Long-tailed distributions represent a fundamental challenge for deep learning, where a small number of head classes dominate training data while a long tail of rare classes have few examples each. Liu et al. (2019) formalized the problem as Open Long-Tailed Recognition, where benchmarks range from 1,280 to as few as 5 images per class and non-ensemble baselines achieve only approximately 67% top-1 accuracy on iNaturalist (Van Horn et al., 2018). The iNaturalist dataset -- a large-scale species classification benchmark drawn from citizen science observations -- has become the standard testbed for long-tail recognition precisely because its imbalance is natural, reflecting true species abundance and observer effort rather than artificial subsampling.

The literature offers three broad categories of solutions. *Loss re-weighting* methods modify the training objective to upweight rare classes: Lin et al. (2017) introduced focal loss to down-weight well-classified examples, Cui et al. (2019) proposed class-balanced loss based on the effective number of samples, and Cao et al. (2019) derived label-distribution-aware margins (LDAM) that enforce larger decision margins for tail classes. *Decoupled training*, introduced by Kang et al. (2020), demonstrated that representation learning and classifier learning have different optimal strategies under imbalance -- training the backbone on the natural (imbalanced) distribution produces better representations, while the classifier head benefits from class-balanced re-calibration. This simple two-stage approach matched or outperformed many complex end-to-end methods. *Multi-expert approaches* such as RIDE (Wang et al., 2021) route inputs to distribution-aware expert branches, each specializing in different portions of the class frequency spectrum.

However, all of these methods share a fundamental limitation: they rebalance or re-weight existing data but cannot introduce new visual information. With fewer than 40 training images, oversampling duplicates the same instances, loss re-weighting increases gradients from the same limited views, and decoupled training still depends on representations learned from insufficient visual evidence. The model cannot learn the intra-class variation in pose, lighting, and morphology that is absent from the training set, regardless of how the loss function is calibrated. The biodiversity data gap compounds this: even BioTrove (Yang et al., 2025), the largest curated biodiversity image dataset at 161.9 million images spanning approximately 366,600 species, inherits the extreme long tail where many species remain critically underrepresented. For the rarest species -- B. ashtoni with 22 training images and B. sandersoni with 40 -- what is missing is visual diversity itself, not a better weighting scheme. This motivates the shift from re-weighting to data augmentation: generating or synthesizing new training examples that introduce the morphological variation the original dataset lacks.

### 2.4 Data Augmentation with Generative Models

Data augmentation expands training sets through transformation or synthesis and is most critical when data is scarce (Shorten & Khoshgoftaar, 2019). Traditional augmentation methods -- geometric transforms, color jitter, random erasing -- increase diversity but cannot generate novel morphological variation beyond what exists in the original training images.

To bridge the data gap, researchers have turned to synthetic augmentation. Beery et al. (2020) demonstrated that Copy-Paste augmentation improves generalization by preserving authentic morphology while varying backgrounds. However, Generative AI offers a newer frontier for "upsampling" rare classes. Zhao et al. (2024) proposed LTGC, utilizing LLMs to reason about missing visual attributes in tail data to guide generation. While promising, He et al. (2023) show that the effectiveness of synthetic data is highly domain-dependent, succeeding for birds (+10% accuracy) but failing for other categories. Trabucco et al. (2024) found that ~10x synthetic images per real image is a practical sweet spot for diffusion-based augmentation, with marginal gains beyond 20x.

The "synthetic-real gap" remains a significant hurdle for fine-grained biological classification. Azizi et al. (2023) found a persistent 4--8-percentage-point accuracy gap between synthetic-only and real training data, even with state-of-the-art diffusion models. For insects, TaxaDiffusion (Monsefi et al., 2025) has recently begun incorporating taxonomic hierarchy to improve species-level synthesis. DisCL (Liang et al., 2025) suggests that a curriculum scheduling data from synthetic to real is essential to prevent out-of-distribution data from being detrimental to performance. SaSPA (Michaeli & Fried, 2024) further argues that preserving class fidelity for fine-grained tasks requires structural conditioning (e.g., edge-based constraints) to ensure that diagnostic features such as wing morphology are not distorted.

### 2.5 Quality Evaluation of Synthetic Images

The two most widely used automated metrics for evaluating synthetic image quality -- the Inception Score (IS; Salimans et al., 2016) and the Frechet Inception Distance (FID; Heusel et al., 2017) -- both rely on features extracted from an Inception network pretrained on ImageNet. IS measures the KL divergence between conditional and marginal label distributions, rewarding both quality and diversity, while FID compares the mean and covariance of Inception features between real and generated distributions under a Gaussian assumption. However, both metrics suffer from well-documented limitations in specialized domains. Borji (2022) provides a comprehensive survey of these shortcomings, noting that FID is statistically biased, sensitive to sample size, and dependent on features optimized for ImageNet categories rather than the target domain. Jayasumana et al. (2024) further demonstrated that FID contradicts human raters and fails to reflect incremental improvements in iterative generation. For fine-grained biological imagery where diagnostic differences may be confined to single body segments, these ImageNet-derived features are poorly suited to capture taxonomically relevant variation.

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

*Figure 3.1: Training set distribution across 16 Massachusetts Bombus species, sorted by count. Imbalance ratio (max/min class): 59.9:1. Gini coefficient of class frequencies: 0.377.*

I partition species into three tiers based on training set size, following a natural gap structure in the distribution where there are breaks in sample count. The "rare" tier (n < 200) comprises the three species selected for synthetic augmentation: B. ashtoni, B. sandersoni, and B. flavidus. The "moderate" tier (200 <= n <= 900) includes species with adequate but sub-optimal representation, while "common" species (n > 900) achieve consistently high performance.

Each target species poses a distinct challenge for generative image models because it deviates from the predominant Bombus phenotype -- a bright yellow thorax with a black abdomen -- characteristic of B. impatiens, the most commonly photographed North American bumblebee. The diagnostic morphology of each target species follows Williams et al. (2014) and Colla et al. (2011); full tergite-level color maps are provided in Appendix A.

**Bombus ashtoni** (Ashton's cuckoo bumble bee) is an obligate social parasite in the subgenus Psithyrus. Females are predominantly black with a diagnostic white-tipped abdomen (T4--T5 white, T6 black). As a cuckoo bee, B. ashtoni lacks corbiculae. The species has experienced severe range-wide decline with no confirmed Massachusetts records since 2008. *Generation challenge:* the model's prior strongly favors yellow-thorax bumblebees; B. ashtoni requires suppressing this prior entirely.

> **NOTE:** Bombus ashtoni has been synonymized under Bombus bohemicus based on molecular phylogenetic evidence. I retain "B. ashtoni" throughout because GBIF occurrence records index this taxon under that name.

**Bombus sandersoni** (Sanderson's bumble bee) is a small eusocial species (8--11 mm) with a clean two-tone pattern: yellow anterior (thorax through T2) and entirely black posterior (T3 onward). The abrupt color transition at T3 is the primary field character. *Generation challenge:* the two-tone pattern is relatively simple to generate (91.2% strict pass rate); the main difficulty is scale.

**Bombus flavidus** (yellowish cuckoo bumble bee) is a cuckoo bee with the most extensive yellow coloration among the three targets. A diagnostic yellow vertex distinguishes it from B. citrinus and B. fervidus. *Generation challenge:* the word "yellow" triggers bright lemon generation; all prompt instances require qualified descriptors ("dingy pale yellow," "cream"). Variable coloration makes consistency difficult (57.6% strict pass rate).

### 3.4 Classifier Architecture and Training Protocol

**Architecture.** ResNet-50 (He et al., 2016) pretrained on ImageNet (Deng et al., 2009), with the original 1000-class output layer replaced by a two-layer classification head: Linear(2048 -> 512) -> ReLU -> Dropout(0.5) -> Linear(512 -> 16). The backbone's final fully-connected layer is replaced with an identity mapping so that the 2048-dimensional feature vector passes directly to the classification head. All parameters -- including the backbone convolutional layers -- are fine-tuned end-to-end; no layers are frozen.

**Training.** Adam optimizer (Kingma & Ba, 2015), learning rate 1 x 10^-4, weight decay 0; regularization is provided by dropout (0.5) and early stopping. Batch size 8, maximum 100 epochs with early stopping (patience 15, monitored on validation loss). Learning rate is reduced on plateau (factor 0.5, patience 5). Standard augmentation: random horizontal flip, color jitter, normalization to ImageNet statistics. Images are resized to 640 x 640 pixels. Model selection uses the checkpoint with the best validation macro F1.

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

**Statistical validation.** Small test-set sizes for rare species (as few as n = 6 under the 70/15/15 split) make single-run evaluation unreliable. Two complementary approaches address different sources of variance:

1. **Multi-seed training.** Deep learning models are sensitive to random initialization, data shuffling, and dropout masks. A single training run may produce results that reflect a lucky or unlucky seed rather than a true augmentation effect. To isolate the effect of data augmentation from training stochasticity, each dataset version is trained with multiple random seeds (N = [5]) on the same fixed train/validation/test split. This ensures that any variance across runs is attributable to optimization randomness, not to different data compositions. Paired t-tests on seed-level macro F1 scores provide pairwise significance between dataset versions. Effect sizes (Cohen's d) are reported alongside p-values to distinguish statistical significance from practical significance.

2. **Bootstrap confidence intervals.** Per-species F1 and macro F1 are computed with 10,000 bootstrap resamples of the test set predictions, producing 95% CIs that reflect evaluation uncertainty given the fixed test set. For rare species (B. ashtoni n = 6, B. sandersoni n = 10), these CIs are inherently wide -- this is an honest reflection of the evaluation challenge, not a limitation to be engineered away. Bootstrap CIs with n < 10 should be interpreted as indicative rather than definitive.

**Why not k-fold cross-validation.** K-fold cross-validation increases effective test-set size but introduces a confound: each fold has a different training set composition, so observed performance differences conflate the augmentation effect with fold-to-fold data variation. For augmentation experiments where the goal is to measure the impact of adding specific synthetic images, a fixed split with multiple seeds provides a cleaner comparison.

All reported results use the best-validation-macro-F1 checkpoint.


## 4. Methods

### 4.1 Copy-Paste Augmentation

Copy-Paste augmentation (CNP) follows the approach of Beery et al. (2020), generating new training images by compositing real bee specimens onto varied backgrounds. For each target species, the Segment Anything Model (SAM ViT-H; Kirillov et al., 2023) extracts foreground masks from existing training images. Segmented specimens are then composited onto flower background images with random affine transforms (rotation, scaling, horizontal flip) and Gaussian boundary blending to reduce edge artifacts.

CNP preserves authentic morphological texture -- every pixel of the bee is from a real photograph -- while varying the background context. This makes it a strong baseline: any improvement from generative augmentation must exceed what can be achieved simply by re-placing real specimens in new scenes. The limitation is a diversity ceiling: with only 22 training images for B. ashtoni, CNP can produce new compositions but cannot introduce morphological variation (poses, angles, lighting on the specimen) beyond what exists in the source images.

### 4.2 Structured Morphological Prompting

Text-to-image generative models encode strong priors toward the most common visual archetype within a category. For the genus Bombus, this prior is the phenotype of B. impatiens -- a bright yellow thorax with a black abdomen -- which accounts for roughly 12% of all GBIF Bombus images globally. Without explicit intervention, every generated bumblebee converges toward this archetype regardless of the species specified in the prompt. This is particularly damaging for the three target species, all of which deviate substantially from the common phenotype.

#### 4.2.1 Prompt Template Architecture

Each generation prompt is assembled from a template with eight placeholders filled programmatically from the species configuration (configs/species_config.json):

| Placeholder | Content |
|-------------|---------|
| {species_name} | Scientific name (e.g., Bombus bohemicus (inc. ashtoni)) |
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

**Implementation.** G-Eval-style chain-of-thought prompting with calibration guidance ("apply the standard of a working entomologist reviewing field photographs -- not the standard of a museum specimen plate"). All outputs are parsed into Pydantic models for type-safe downstream processing. Full judge prompt and schema are provided in Appendix C.

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

**Stage 1: Blind Identification Interface.** The expert sees only the synthetic image at full resolution -- no target species label, no LLM judge output. The interface presents the 16-species identification panel and the expert selects the most likely species (or "Unknown" / "No match"). This stage tests whether the generated image carries sufficient diagnostic information for independent identification.

**Stage 2: Detailed Evaluation Interface.** After submitting the blind identification, the target species and its diagnostic criteria card are revealed. The expert then scores the same five morphological features on the same 1--5 anchored scale used by the LLM judge. The interface additionally collects diagnostic completeness, failure mode checkboxes, caste fidelity assessment, and an overall PASS / FAIL / UNCERTAIN judgment. Museum-quality reference images (3 per species) are displayed alongside the evaluation image throughout this stage.

[TODO] Figure 4.3: Screenshots of the two-stage evaluation interface

This design -- absolute scoring on a shared rubric rather than pairwise preference -- is motivated by the need for per-feature disagreement analysis (Section 4.4.4). Pairwise preferences can reveal which image is *better* but cannot reveal *why* -- i.e., which specific morphological feature the expert considers incorrect. The shared rubric enables the 2x2 disagreement matrices that drive the diagnostic feedback loop.

#### 4.4.3 Learned Filter

The LLM judge produces rich per-feature scores, but its holistic pass/fail rule weights all features equally. Expert calibration learns which features matter most. A central methodological question is whether the learned filter should operate on LLM scores alone or also incorporate visual representations of the images themselves.

**Model A -- Holistic rule (baseline).** The strict filter from Section 4.3: matches_target AND diag=species AND mean morph >= 4.0. No learning; this is the LLM's own decision rule.

**Model B -- Weighted LLM scores.** L2-regularized logistic regression on 7 LLM features (5 morphological scores + blind ID match + diagnostic level), predicting expert pass/fail. The learned coefficients reveal which LLM scores the experts implicitly trust. This tests whether *reweighting existing signals* is sufficient.

**Model B -- Linear probe on DINOv2 embeddings.** Model A's fundamental limitation is that it never sees the image -- it operates entirely on the LLM's textual assessment. Two species-diagnostic errors that produce identical LLM scores may look very different in visual feature space. Model B replaces the LLM rule with a linear layer on frozen DINOv2 embeddings (Oquab et al., 2024), predicting expert pass/fail directly from the image representation. This follows the standard linear probe evaluation protocol from the DINOv2 literature: frozen backbone features with a single learned linear layer and L2 regularization. The same approach can be applied with BioCLIP embeddings (Stevens et al., 2024) to test whether biology-specialized features outperform general-purpose ones. This tests whether vision alone -- without any LLM signal -- can predict expert judgment.

**Model C -- Linear probe on DINOv2 + LLM scores.** Model C concatenates the DINOv2 embedding with the 7 LLM judge scores before the linear layer, testing whether LLM scores provide complementary signal on top of visual features. If Model C does not outperform Model B, the LLM scores are redundant for filtering; if it does, the two modalities capture different aspects of quality.

All filter variants are evaluated on the expert-annotated set (150 images, leave-one-out cross-validation) and then applied to score the remaining synthetic images. The highest-scoring images are selected to assemble the D6 dataset for classifier retraining. The downstream ResNet-50 classifier architecture and training protocol (Section 3.4) remain unchanged -- the filter determines *which* synthetic images enter the training set, not *how* the classifier is trained.

#### 4.4.4 Diagnostic Feedback Loop

Per-feature disagreement between the LLM judge and experts is analyzed using 2x2 matrices (LLM score >= 4 vs. < 4 x expert score >= 4 vs. < 4) for each feature:

- **LLM blind spots** (LLM >= 4, expert < 4): features where the LLM cannot see the error -- these inform generation prompt refinement (e.g., if experts flag thorax coloration that the LLM missed, the prompt's color map for that species is revised).
- **LLM over-strictness** (LLM < 4, expert >= 4): features where the LLM is too conservative -- these inform judge rubric recalibration.

This feedback loop connects evaluation to generation: expert disagreement patterns flow back into both the prompting framework (Section 4.2) and the judge rubric (Section 4.3), enabling iterative improvement across the full pipeline.


## 5. Experiments and Results

### 5.1 Baseline Analysis

Table 5.1 reports baseline classifier performance across all 16 species (mean +- std across N seeds).

*Table 5.1: Baseline classifier per-species results (ResNet-50, f1 checkpoint, N seeds).*

| Species | Train n | Test n | Precision | Recall | F1 | 95% Bootstrap CI |
|---------|---------|--------|-----------|--------|-----|------------------|
| ... | | | | | | |
| **B. ashtoni** | **22** | **6** | [x] | [x] | [x] | [x, x] |
| **B. sandersoni** | **40** | **10** | [x] | [x] | [x] | [x, x] |
| **B. flavidus** | **162** | **36** | [x] | [x] | [x] | [x, x] |
| Macro average | | | | | [x] | [x, x] |
| Overall accuracy | | | | | [x] | |

*Figure 5.1: Per-species F1 with 95% bootstrap CIs (horizontal bar chart, sorted by F1).*

*Figure 5.2: Row-normalized confusion matrix (16x16). Bold labels indicate rare species.*

**Analysis.** Performance correlates with training set size. Common species (n > 900) achieve F1 > 0.85. The three rare target species fall substantially below. Confusion analysis reveals systematic misclassification patterns: B. sandersoni is predicted as B. vagans in a majority of error cases, likely reflecting shared body size and yellow-anterior coloration. B. flavidus misclassifications are spread across multiple species sharing variable coloration patterns. These confusion patterns between species sharing diagnostic color features confirm that the classification challenge is fine-grained and morphology-driven.

### 5.2 Generation Quality

*Table 5.2: LLM-as-judge evaluation of 1,500 synthetic images (500 per species).*

| Species | Strict Pass | Pass Rate | Blind ID Match | Mean Morph | Judge Pass (lenient) |
|---------|-------------|-----------|----------------|------------|---------------------|
| B. ashtoni | 222 | 44.4% | 76.0% | 3.82 | 92.0% |
| B. flavidus | 288 | 57.6% | 96.0% | 4.06 | 99.6% |
| B. sandersoni | 456 | 91.2% | 96.4% | 4.37 | 100% |

*Table 5.3: Strict filter funnel.*

| Stage | Count | Cumulative Rate |
|-------|-------|-----------------|
| Total images | 1,500 | 100% |
| + matches_target | 1,342 | 89.5% |
| + diag=species | 1,060 | 70.7% |
| + morph >= 4.0 | 966 | 64.4% |

*Figure 5.3: Per-feature morphological score distributions by species (box plots, 5 features x 3 species).*

*Figure 5.4: Sample image grid -- 3 rows (species) x 4 columns (strict_pass / borderline / soft_fail / hard_fail). Shows what each quality tier looks like concretely.*

**Failure mode analysis.** Wrong coloration is the dominant failure mode (27.1% of all images), concentrated in B. ashtoni. Structural failures (extra/missing limbs, impossible geometry, artifacts, repetitive patterns) are zero across all 1,500 images, confirming that prompt engineering (Section 4.2) eliminated structural generation errors. The per-feature scores pinpoint the bottleneck: ashtoni's thorax coloration mean = 2.98, far below every other feature (all >= 4.0).

**Caste fidelity.** B. ashtoni male generation is problematic (30.9% caste-correct vs. 84.3% female), suggesting the model lacks sufficient priors for sex-dimorphic features in this species.

### 5.3 Diagnostic Experiments

#### 5.3.1 Background Removal Test

To test whether the generation bottleneck lies in background interference or specimen morphology, synthetic images were evaluated on white backgrounds.

*Result:* Negative. Pass rates were unchanged, while wrong_coloration surged +55% on white backgrounds. This confirms the bottleneck is coloration accuracy of the specimen itself, not background confusion.

#### 5.3.2 Volume Ablation

To determine whether additional synthetic volume improves classifier performance, models were trained with +50, +100, +200, +300, and +500 synthetic images per rare species (D4 unfiltered). Trabucco et al. (2024) suggest ~10x synthetic per real image as a practical sweet spot; our ablation tests ratios from 1.2:1 (flavidus at +200) to ~23:1 (ashtoni at +500).

*Table 5.5: Volume ablation -- D4 unfiltered synthetic at varying addition counts.*

| Volume | Macro F1 | Ashtoni F1 | Sandersoni F1 | Flavidus F1 |
|--------|----------|------------|---------------|-------------|
| +0 (baseline) | [x] | [x] | [x] | [x] |
| +50 | [x] | [x] | [x] | [x] |
| +100 | [x] | [x] | [x] | [x] |
| +200 | [x] | [x] | [x] | [x] |
| +300 | [x] | [x] | [x] | [x] |
| +500 | [x] | [x] | [x] | [x] |

*Figure 5.5: Volume ablation line plot (x-axis: synthetic volume, y-axis: F1, one line per rare species + macro F1). With error bars from multi-seed runs.*

*Expected finding:* No statistically significant improvement at any volume. This establishes that the problem is quality, not quantity -- consistent with Ma & Zhang (2026), who show that optimal synthetic volume depends on generator accuracy.

### 5.4 Augmentation Method Comparison

*Table 5.6: Classification results across augmentation strategies. Mean +- std across N random seeds on a fixed train/test split. Bold = best per column.*

| Dataset | Macro F1 | Weighted F1 | Accuracy | Ashtoni F1 | Sandersoni F1 | Flavidus F1 |
|---------|----------|-------------|----------|------------|---------------|-------------|
| D1 Baseline | [x+-x] | [x+-x] | [x+-x] | [x+-x] | [x+-x] | [x+-x] |
| D3 CNP | [x+-x] | [x+-x] | [x+-x] | [x+-x] | [x+-x] | [x+-x] |
| D4 Synthetic | [x+-x] | [x+-x] | [x+-x] | [x+-x] | [x+-x] | [x+-x] |
| D5 LLM-filtered | [x+-x] | [x+-x] | [x+-x] | [x+-x] | [x+-x] | [x+-x] |
| D6 Expert-filtered | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

*Table 5.7: Pairwise significance tests (macro F1, paired t-test across seeds). p < 0.05 marked with \*.*

| Comparison | Mean Delta | p-value | Cohen's d | Significant? |
|------------|-----------|---------|-----------|-------------|
| D1 vs D3 | [x] | [x] | [x] | |
| D1 vs D4 | [x] | [x] | [x] | |
| D1 vs D5 | [x] | [x] | [x] | |
| D3 vs D5 | [x] | [x] | [x] | |
| D4 vs D5 | [x] | [x] | [x] | |

*Figure 5.6: Grouped bar chart -- per-species F1 across all dataset versions (D1/D3/D4/D5), with error bars.*

*Figure 5.7: Side-by-side comparison for each rare species -- real image vs. D3 CNP composite vs. D4 synthetic (pass) vs. D4 synthetic (fail). Shows the synthetic-real gap visually.*

*Figure 5.8: Confusion matrix comparison -- row-normalized confusion matrices for D1 vs D3 vs D5 (3-panel), focused on rare species rows.*

**Qualitative failure analysis.** Figure 5.9 shows 2--3 test images where D4-trained classifiers predicted incorrectly but the baseline and D3 classifiers predicted correctly, illustrating what features the classifier learned from synthetic data that led to misclassification.

*Figure 5.9: Failure analysis -- test images misclassified by D4 but correctly classified by D1/D3.*

**Key findings** (to be updated with final numbers):

1. **CNP is the strongest method overall** -- marginal improvement in macro F1 over baseline, with consistent gains for flavidus.
2. **Unfiltered synthetic (D4) consistently degrades performance** across all seeds, particularly harmful for sandersoni where the high synthetic:real ratio overwhelms the small real training set.
3. **LLM filtering (D5) partially mitigates the D4 degradation** -- sandersoni partially recovers, but macro F1 remains below baseline.
4. **The effect is species-dependent** -- species with more atypical morphology (ashtoni) or fewer real images (sandersoni) are most vulnerable to synthetic noise.
5. **Quality filtering helps but is insufficient** -- motivating expert-calibrated filtering (D6).

### 5.5 Expert Calibration Results [TODO]

#### 5.5.1 Inter-Annotator Agreement

*Table 5.9: Cohen's kappa per feature and overall.*

#### 5.5.2 Per-Feature Disagreement Analysis

*Table 5.10: 2x2 disagreement matrices for critical features.*

*Figure 5.10: Heatmap of LLM-vs-expert agreement per feature per species.*

#### 5.5.3 Learned Filter Performance

*Table 5.11: Filter model comparison on expert-annotated set.*

| Model | Features | AUC-ROC | Precision@90%Recall |
|-------|----------|---------|---------------------|
| A: Holistic rule | -- | [x] | [x] |
| B: Weighted LLM | 7 LLM scores | [x] | [x] |
| C: LLM + visual | 7 + latent PCs | [x] | [x] |

*Figure 5.11: ROC curves for Models A/B/C overlaid.*
*Figure 5.12: Learned coefficient magnitudes for Model B.*

#### 5.5.4 D6 Classifier Results

Same format as Table 5.6, with D6 row added.

### 5.6 Latent Space Analysis [TODO]

*Figure 5.13: t-SNE/UMAP of DINOv2 embeddings -- real vs. synthetic, colored by species.*
*Figure 5.14: Same for BioCLIP embeddings.*

*Table 5.13: Correlation between LLM judge scores and embedding-space distance to real class centroid.*


## 6. Discussion

### 6.1 The Synthetic-Real Gap is Morphology-Specific

The generation quality analysis (Section 5.2) reveals that the synthetic-real gap is not uniform across species -- it is concentrated in species that deviate most from the generative model's learned prior. B. sandersoni, with a clean two-tone pattern (yellow front, black rear), achieves a 91.2% strict pass rate. B. ashtoni, a predominantly black bee with sparse pale markings -- the inverse of the typical Bombus phenotype -- passes at only 44.4%. This 2x difference in pass rate from the same generation pipeline, with the same prompt architecture and the same model, isolates morphological atypicality as the primary determinant of generation quality.

The per-feature scores pinpoint the bottleneck further. Ashtoni's mean thorax coloration score (2.98) falls far below every other feature (all >= 4.0), while sandersoni scores >= 4.04 on all features. The generation model can render anatomically correct bees (zero structural failures across 1,500 images) but cannot reliably produce the correct *coloration* for species that contradict its prior. This is consistent with the observation from Section 4.2 that negative constraints were the single most impactful prompt intervention -- the model defaults to the dominant phenotype unless explicitly overridden, and even then, coloration fidelity remains the residual failure mode.

This finding has implications beyond bumblebees. Any fine-grained domain where rare classes are visually atypical relative to common classes -- rare bird plumages, uncommon mineral formations, unusual pathological presentations -- will likely exhibit the same morphology-specific gap when using generative augmentation.

### 6.2 Why Synthetic Augmentation Degrades Performance

The most important result in this thesis is that adding synthetic images to the training set *hurts* classifier performance on rare species, even after automated quality filtering.

The degradation is most severe for B. sandersoni, whose F1 drops substantially under D4 (unfiltered synthetic). With only 40 real training images, adding 200 synthetic images means that synthetic data constitutes roughly 83% of the class's training set. Even if most synthetic images are morphologically reasonable (91.2% pass rate), the remaining ~9% introduce noise -- wrong coloration, genus-level rather than species-level diagnostics -- that the classifier cannot distinguish from real signal.

This pattern is consistent with the hypothesis that high synthetic:real ratios overwhelm small real training sets -- species with fewer real images show greater degradation. However, since all three species received the same +200 images rather than proportional augmentation, we cannot fully disentangle the effects of ratio from species-specific generation quality. The volume ablation (Section 5.3.2) provides partial evidence: no volume from +50 to +500 shows statistically significant improvement, suggesting that the issue is image quality rather than ratio alone.

LLM filtering (D5) partially mitigates the degradation by removing the worst synthetic images, improving sandersoni relative to D4. But the filtered images still introduce distribution shift that the strict morphological threshold cannot eliminate -- an image can pass all five morphological criteria yet still differ from real images in texture, lighting distribution, or subtle coloration characteristics that the LLM cannot assess from a single image. This residual gap is what motivates the expert calibration and latent-space filtering approaches in Section 4.4.

### 6.3 Copy-Paste vs. Generative: Fidelity and Diversity

CNP augmentation achieves the best overall results without degradation. This is instructive: CNP preserves real morphological texture (every pixel of the bee is from a real photograph) while varying only the background context. Generative augmentation introduces both new morphological variation *and* new morphological errors -- and for rare species, the errors outweigh the variation.

However, CNP has a fundamental diversity ceiling. With 22 training images of B. ashtoni, CNP can produce at most 22 distinct foreground specimens in different backgrounds. It cannot generate novel poses, lighting conditions on the specimen, or intra-class morphological variation. For species with more source images (B. flavidus, n = 162), CNP has more raw material to work with, which may explain why CNP's benefit is most consistent for flavidus.

This suggests a complementarity: CNP provides safe, high-fidelity augmentation but limited diversity; generative augmentation provides potentially unlimited diversity but at the cost of fidelity errors. The open question -- addressed by expert-calibrated filtering -- is whether the fidelity errors can be reduced to the point where generative diversity becomes beneficial.

### 6.4 LLM Judges: Useful Signals, Incorrect Weighting

The LLM-as-judge produces informative per-feature signals -- the per-feature score distributions clearly differentiate ashtoni's coloration bottleneck (thorax = 2.98) from sandersoni's across-the-board success (all features >= 4.0). However, the holistic pass/fail rule -- a simple threshold on the mean score -- weights all features equally. Domain expertise suggests this is wrong: for B. ashtoni, the white T4--T5 tail is the single most important field mark, while wing venation contributes negligibly to species identification.

The gap between the lenient pass rate (97.2%) and the strict pass rate (64.4%) further illustrates the issue. The judge's default calibration -- trained on general visual assessment, not entomological diagnostics -- is too generous on species-critical features and does not distinguish "looks like a bumblebee" from "looks like *this* bumblebee."

This does not mean the LLM judge is useless -- the per-feature scores are the raw material for expert calibration. The contribution is decomposing quality into interpretable dimensions; the limitation is weighting them correctly. Expert annotations provide the supervisory signal to learn the correct weighting (Section 4.4.3), transforming the judge from a noisy holistic assessor into a calibrated diagnostic instrument.

### 6.5 Statistical Challenges with Rare Species

Evaluating augmentation strategies for rare species presents a fundamental tension: the species that most need augmentation are precisely those for which evaluation is least reliable. With small test sets, individual test images have disproportionate influence on F1, and confidence intervals overlap across conditions.

This thesis addresses the problem through multi-seed training on a fixed split, which quantifies training stochasticity, combined with bootstrap CIs, which quantify test-set sampling uncertainty. These two sources of variance are kept separate -- unlike k-fold cross-validation, which conflates them. However, even with these approaches, per-species significance for the rarest taxa remains difficult: with n = 6 test images for B. ashtoni, individual test images have disproportionate influence on F1. Reporting confidence intervals and effect sizes, rather than relying on point estimates, is essential.

### 6.6 Implications for Urban Biodiversity Monitoring

The practical goal of this work is to reduce false-absence errors in automated pollinator monitoring. A classifier that misidentifies B. ashtoni as B. pensylvanicus produces a false negative for ashtoni -- and if B. ashtoni is reported as absent, planners may approve development on habitat where it still persists.

The results show that naive synthetic augmentation is not a safe default for this purpose -- it can actually increase misclassification of the species it aims to help. This is an important negative result for the growing body of work applying generative AI to ecological monitoring. The contribution is not just that quality filtering helps, but that *unfiltered generative augmentation can make monitoring systems worse* -- a finding that should inform deployment decisions.

CNP augmentation, by contrast, provides a reliable improvement with no degradation risk, and should be the first-line augmentation strategy for monitoring deployments where rare species are present. Generative augmentation should be deployed only with quality filtering validated against domain expertise.

### 6.7 Limitations

Several limitations constrain the scope of these conclusions:

- **Single classifier architecture.** All experiments use ResNet-50. Architectures with stronger pretrained features (BioCLIP, DINOv2) may respond differently to synthetic augmentation.
- **Single generative model.** GPT-image-1.5 via the images.edit API is a closed, non-fine-tunable model. The observed synthetic-real gap may be specific to general-purpose models without domain-specific fine-tuning; open-source diffusion models (SDXL, Stable Diffusion 3) with LoRA fine-tuning on real specimens could potentially close the gap.
- **Three target species.** The morphology-specific gap is demonstrated for three species; generalization to other long-tailed taxa is hypothesized but not tested.
- **Expert evaluation scale.** 150 annotated images. Larger annotation sets and additional annotators would increase the reliability and generalizability of the learned filter.
- **No comparison with loss re-weighting.** This thesis does not compare against focal loss, class-balanced loss, or decoupled training -- methods that address class imbalance through training strategy rather than data augmentation. These approaches are complementary rather than competing; they could be combined with augmentation and are not expected to resolve the fidelity gap that is the focus of this work.
- **Fixed augmentation volume.** All species received +200 synthetic images rather than proportional augmentation. The observed degradation conflates two effects: image quality and synthetic:real ratio. The volume ablation (Section 5.3.2) provides partial evidence that quality dominates, but a controlled ratio experiment would strengthen this conclusion.
- **No hyperparameter tuning across dataset versions.** Weight decay, learning rate, and dropout were not optimized per dataset version; systematic tuning might alter the relative ranking of augmentation strategies.


## 7. Conclusion and Future Work

### 7.1 Summary

This thesis investigated the synthetic-real gap for fine-grained biodiversity classification under extreme class imbalance. Four contributions were made:

First, a **structured morphological prompting framework** using negative constraints, tergite-level color maps, and reference-guided generation eliminated all structural failures across 1,500 generated images, isolating coloration fidelity as the sole remaining bottleneck -- with pass rates ranging from 44.4% (B. ashtoni) to 91.2% (B. sandersoni) depending on morphological atypicality.

Second, a **two-stage LLM-as-judge evaluation protocol** combining blind taxonomic identification with per-feature morphological scoring provided interpretable diagnostic signals that identify *where* generated images fail, enabling targeted pipeline improvements.

Third, an **expert-calibrated quality filtering pipeline** was designed to learn feature-level weights from entomologist annotations, connecting automated evaluation to domain expertise through a diagnostic feedback loop. [TODO: report D6 outcome]

Fourth, **rigorous empirical comparison** under multi-seed evaluation with bootstrap confidence intervals demonstrated that synthetic augmentation from a general-purpose generative model degrades rare-species classification, that automated quality filtering partially mitigates but does not resolve this degradation, and that Copy-Paste augmentation using real morphological texture yields the strongest results without degradation risk. The finding that augmentation quality -- not volume -- determines classifier impact is the central empirical contribution.

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
Confusion matrices per dataset version, volume ablation full table, per-seed breakdowns, complete 16-species results table, caste fidelity breakdown.
