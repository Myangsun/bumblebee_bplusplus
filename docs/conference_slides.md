# Conference talk — 3 slides

**Title:** Expert-Calibrated Filtering for Rare-Species Generative Augmentation — Evidence from Massachusetts Bumblebees
**Speaker:** Mingyue Sun · MIT M.S. thesis · 2026
**Format:** 3 slides · ~3 min total · each slide reads 40–55 s aloud

---

## Slide 1 — Problem and headline result

**On-screen**

- **Why rare species matter.** In automated biodiversity monitoring, a missed rare-species detection is indistinguishable from true absence — silent undercounts propagate into conservation and urban-planning decisions
- **Setup.** 16-species Massachusetts bumblebee classifier; three rare targets (B. ashtoni n = 22, B. sandersoni n = 40, B. flavidus n = 162). Baseline ResNet-50 rare-tier F1: 0.50 / 0.59 / 0.62 vs 0.85+ for common species
- **Pipeline (3 stages).** (1) tergite-level morphological prompts → 1,500 synthetics, 0 structural failures; (2) LLM-as-judge scoring; (3) expert-calibrated BioCLIP probe (150 expert labels)
- **Headline.** The expert-calibrated filter (D6) delivers the highest single-split macro F1 in the study: **0.840 [0.801, 0.869]** vs 0.815 baseline, with per-species gains of **+0.167 B. ashtoni** and **+0.127 B. flavidus**

**Speaker script (≈ 45 s)**

> Automated biodiversity monitoring turns every missed rare-species detection into a silent undercount — an absence in the record that looks no different from a species that truly isn't there. For three rare Massachusetts bumblebees with 22 to 162 training images, a ResNet-50 classifier sits at F1 between 0.50 and 0.62 while common species clear 0.85. I built a three-stage pipeline to close this gap: tergite-level anatomical prompts that eliminate structural generation failures, an LLM judge that scores morphology, and finally a BioCLIP probe trained on 150 annotations from a domain expert. On the held-out test set the expert-calibrated filter — D6 — delivers the highest single-split macro F1 in the study, 0.840 against a 0.815 baseline, improving B. ashtoni by 17 F1 points and B. flavidus by 13. The species that gained most are exactly the ones the expert flagged as the hardest to generate.

---

## Slide 2 — Contributions

**On-screen**

- **Anatomy-anchored prompts.** Tergite-level colour maps drive generation; 0 / 1,500 synthetics show impossible geometry or missing limbs — the residual generation gap is colour fidelity alone
- **LLM-judge ≠ expert judgment.** Direct calibration on 150 stratified synthetics: LLM-strict vs expert-strict agreement is **56 % (49 % chance)**; LLM morph-mean is an **AUC 0.56 ranker** of expert quality — near-random
- **Expert-supervised BioCLIP probe.** LOOCV AUC 0.792 under the expert-strict rule; within an identical 200-image-per-species training-volume budget, the probe selects **~3× more expert-validated synthetics than the LLM filter** (16 vs 5 · 20 vs 12 · 6 vs 1 for the three rare species)
- **Multi-protocol downstream evaluation.** Single-split, 5-seed × fixed split, and 5-fold CV paired t-tests — **D6 is numerically best on every protocol among filter variants**, with B. ashtoni / B. sandersoni multi-seed F1 leads over all other variants (0.637 / 0.548)

**Speaker script (≈ 50 s)**

> Four contributions. First, the prompt template: encoding tergite colour maps at the anatomical level drives structural generation failures to zero across 1,500 generations, isolating the remaining problem as a colour-fidelity issue. Second, the calibration study — I directly compare the LLM judge against an expert on 150 stratified synthetics. Their strict pass-fail decisions agree only seven percentage points above chance, and the LLM morphology score is a near-random ranker of expert quality. That's the empirical case that language-mediated judging alone is insufficient. Third, I train a linear probe on BioCLIP features with those 150 labels. Within the same 200-image-per-species training budget, the probe-selected subset contains roughly three times more expert-validated synthetics than the LLM's selection, with the biggest gain on B. flavidus where the LLM was weakest. Fourth, the downstream validation: across three evaluation protocols D6 is numerically the best filter variant on every one — including the highest single-split macro F1 and the highest multi-seed B. ashtoni and B. sandersoni F1 of the study.

---

## Slide 3 — Honest limitations and future work

**On-screen**

- **Statistical power is tight at rare scale.** Rare-species test n = 6 / 10 / 36; paired t-tests have df = 4 under 5-fold CV. D6 vs baseline macro F1 is **directional (+0.008), not significant (p = 0.25)** — CIs, not p-values, are the right uncertainty quantifier at this sample size
- **Single annotator.** 150 expert labels from one entomologist → Cohen's κ undefined; inter-rater reliability unmeasured
- **Single backbone, single generator.** ResNet-50 + GPT-image-1.5; the per-species feature-space offset may be architecture- or generator-specific
- **Future work.** (i) 300–500 images × ≥ 3 annotators for inter-rater κ + probe retraining; (ii) repeat D1-D6 under BioCLIP / DINOv2 backbones to test mechanism portability; (iii) additive single-species ablation — does the best-generated rare species help when added alone?
- **Deployment.** Pipeline outputs a *data-augmentation strategy* for the MIT Sensing Garden / Flik camera-trap platform; operational goal is to reduce rare-species detection failures so they stop masquerading as true absences in the monitoring record

**Speaker script (≈ 55 s)**

> I want to be explicit about what this work does and does not prove. The rare-species test sets are small — six B. ashtoni, ten B. sandersoni — and the paired t-tests under 5-fold CV have only four degrees of freedom, so the D6 improvement over baseline is directional, not statistically significant at five per cent. We report bootstrap confidence intervals throughout because at this sample size CIs tell you the range of plausible effects, while a p-value tells you almost nothing. The expert annotation comes from one entomologist, which means Cohen's κ is undefined — inter-rater reliability is the single most important replication step and is the first future-work item. The pipeline uses one classifier backbone and one generator, so the per-species feature-space offset we document could be architecture- or model-specific. Near-term roadmap: scale the annotation to three to five annotators, swap the backbone to BioCLIP and DINOv2 to test mechanism portability, and run the additive single-species ablation to see whether the best-generated species helps in isolation. The deployment target is the MIT Sensing Garden platform — what this work delivers there is a data-augmentation strategy whose operational purpose is to stop rare-species detection failures from masquerading as true absences in the biodiversity record.

---

## Timing plan

| Slide | Target | Content budget |
|---|---|---|
| 1 Problem + headline | 0:00–0:45 | 4 bullets · 45 s script |
| 2 Contributions | 0:45–1:35 | 4 bullets · 50 s script |
| 3 Limitations + future | 1:35–2:30 | 5 bullets · 55 s script |

Leaves ~30 s buffer for transitions + the first Q&A sentence.

## Likely Q&A — prepared responses

| Question | Answer |
|---|---|
| "Why not statistically significant?" | df = 4 under 5-fold, n_rare = 6/10/36; CIs show the plausible range, which for D6 vs D1 is [−0.02, +0.03] on macro F1. Power-limited, not null. |
| "Why is κ undefined?" | Single annotator. κ requires ≥ 2 raters. Replication is future-work item 1. |
| "Why does a BioCLIP probe beat an LLM judge?" | The LLM operates on a verbalised rubric; the probe operates on the classifier's own feature space. Per-feature heatmap shows LLM over-strict on B. ashtoni thorax coloration and blind-spotted on B. flavidus head and legs. |
| "Is this ready to deploy?" | The filter-level gain is established (3× more expert-validated images in the same budget). The downstream F1 gain is directional. Multi-annotator replication must happen before production. |
| "Why is copy-paste (D2) still the best 5-fold method?" | CNP preserves real morphology, so its outputs sit in the real-image region of feature space. That is the asymmetry with generative — diversity vs fidelity. D6 tries to close the fidelity gap without the CNP diversity ceiling; numbers are promising, not yet powered. |
| "Generalisation to other taxa?" | The protocol ports — only the reference images and diagnostic-feature list change. Item 2 of future work tests this on rare fungi and rare plants. |

## One-line version (if introduced by session chair)

> I built an expert-calibrated filter for generative bumblebee augmentation that beats the standard LLM-as-judge approach by selecting three times more expert-validated synthetics within the same training budget, and delivers the highest single-split rare-species F1 in the study.
