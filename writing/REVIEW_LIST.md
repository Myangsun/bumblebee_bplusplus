# Original Review List — ECCV 2026 Submission

**Paper:** *True Absence or Detection Failure: Generative Augmentation and Expert-Calibrated
Filtering for Long-Tailed Bumblebee Classification*
**Reviewer:** Julia Chae (advisor) · captured in `writing/Review.md` · 2026-07-05
**Implementation tasks:** see `writing/TODO.md` (each review maps to a `T#` task there).

Julia left 21 comments, grouped below by type. Each has its timestamp, the paraphrased ask, and
where it lands in the current paper (`main.tex` / `supplementary.tex`).

## Writing / clarity (quick)

- **R1 · 2:21pm** — "This is confusingly worded and I'm not sure comes across cleanly to someone
  reading the paper for the first time — possible reword?" *(unanchored; likely an intro/abstract
  topic sentence.)*
- **R2 · 2:22pm** — Edit accepted: "an obvious remedy" → "a possible remedy." — §1 Introduction,
  `main.tex:132`.
- **R3 · 2:57pm** — "This is a bit unclear, what does this mean?" *(unanchored.)*
- **R4 · 3:06pm** — Edit accepted: "deployed 200-image selection" → "selected 200-image subset." —
  §5.4, near `main.tex:647`.

## Consistency / correctness bugs

- **R5 · 2:25pm** — Pipeline figure implies ResNet-50 is **linear-probed**, text says
  **fine-tuned**. Fix the figure (should be fine-tuned). — Fig 1 `Diagram_A.jpg`; `main.tex:299`,
  `:324`.
- **R6 · 2:44pm** — "Trained from scratch, fine-tuned, or linear-probe? Figure implies linear
  probe, one section says fine-tuned, another says trained from scratch — make consistent." —
  `main.tex:299` ("fine-tuned end-to-end") **directly contradicts** `main.tex:324` ("trained from
  scratch").
- **R7 · 2:24pm** — "What does *structured morphological prompting* mean — that's for generation,
  right? Isn't it applied to all GPT-Image generations? The pool is identical other than filtering.
  Phrasing makes it seem specific to the expert-calibrated probe." — §1.4 contribution bullet 2,
  `main.tex:173–177`.

## Missing citations

- **R8 · 2:33pm** — Add and reframe:
  - *Fill-Up: Balancing Long-Tailed Data with Generative Models* (long-tail classification via
    textual inversion) — related work, potentially a baseline.
  - *Personalized Representation from Personalized Generation* (representation improvement in
    data-scarce regimes with synthetic data).
  - Note: both use **fine-tuned generators** (textual inversion / DreamBooth) because off-the-shelf
    generators are weak in the long-tail regime — **unlike our off-the-shelf `gpt-image-1.5`.** A
    framing point, not just a citation. — §2, `main.tex:211–224`.

## Design-choice justification

- **R9 · 2:45pm** — "Why 200, when that isn't the count needed to balance classes? Any scaling
  analysis (+50/+100/+200 vs +500 or complete fill-up)? Otherwise +200 seems arbitrary." — volume
  ablation `supplementary.tex:132` currently spans **only +50/+100/+200/+300**.
- **R10 · 2:50pm** — "Justify the generation design. Did you test **text-guided** vs
  **reference-guided**? Why 500/species not fill-up to ~900 total? How was the structured template
  selected — what else was tried?" — §4.2, `main.tex:351–381`.
- **R11 · 2:53pm** — "Group 4.3/4.4/4.5 as *Improving Fidelity of Reference-Based Generative Images
  with Filtering*, with a sub-subsection each for LLM-as-judge, BioCLIP centroid, Expert-Calibrated
  Probe." — §4.3–4.5, `main.tex:383–466`.
- **R12 · 2:55pm** — "Could we try **BioCLIP2** instead of BioCLIP?" — D5 backbone, §4.5
  `main.tex:415–423`; probe embedding §4.6.

## Figures / visualization

- **R13 · 2:40pm** — "Include a figure of the three target species and why they're difficult (e.g.,
  show confusion species)?" — §3, `main.tex:286–295`.
- **R14 · 2:59pm** — "Regenerate this figure — titles/legends tiny and unreadable; right panel too
  small to see shape differences." — Fig 4 t-SNE, `main.tex:539–548`.
- **R15 · 3:03pm** — "Any figures/visualizations of successful images that match the correct human
  classification?" — §5.4.
- **R16 · 3:02pm** — "What is 0.1–0.2 referring to? Real-image distance to its own centroid? Is
  synthetic-to-centroid the real species centroid?" — §5.2, `main.tex:554–556`.

## Expert-filter framing

- **R17 · 3:09pm** — "Expert filter retained more, but what is the **upper limit / oracle** of what
  it should have retained? 27/17/6? Make it clear." — §5.4, `main.tex:647–649` (D6 retains 16/20/6).
- **R18 · 7:26pm** — "Fig 6b is confusing; most of the 200 selected aren't expert-labelled so the
  point is muddied. Instead show **filtering classification performance against expert labels** for
  LLM vs Centroid vs Probe." — Fig 5(b) `expert_coverage_of_selected200.png`, `main.tex:660–665`.

## Major — new experiments / restructure

- **R19 · 8:27pm (MAJOR)** — "Missing baselines. Add classical long-tail baselines (oversampling,
  class-balanced loss, LDAM-DRW, balanced softmax); simple augmentation baselines (RandAugment,
  MixUp); and mimic **Fill-Up's training recipe** (synthetic+real, then real-only finetune with
  balanced softmax). Without these we'll get many questions on how the numbers compare."
- **R20 · 8:13pm (MAJOR)** — "Bootstrap single-split + multi-seed + five-fold CV is too much for
  main results — they contradict each other. Also ensure **no data bleeding between folds**: only
  synthetic images generated from a training-split sample may be used in that fold's training."
- **R21 · 8:20pm (MAJOR)** — "Remove single-split and five-fold CV; keep one clean train/val/test
  split with no bleeding (CP-source and reference images all in train) and **run over 10 seeds**."

## Advisor summary email — the four consolidated majors

1. **Missing baselines/experiments** — classical long-tail + non-generative aug + Fill-Up two-stage
   recipe (= R19); make the contribution clear vs Fill-Up and recent long-tail generative work.
2. **Statistical presentation muddies the signal** — one primary protocol, others to supplementary;
   preference is **single locked split × ≥10 seeds** (= R20 + R21).
3. **Leakage** — all real-data-dependent steps (CP sources, BioCLIP centroids, filtering/
   calibration) must be fold-local / train-only; no val/test image in the generation pipeline
   (= R20).
4. **Arbitrary design choices** — justify +200 (ablation or fill-to-target); state **ResNet-50 =
   efficient/field-deployable** framing so the right comparison is ResNet-50-compatible long-tail
   baselines, not foundation-model classifiers (= R9 + backbone framing).
