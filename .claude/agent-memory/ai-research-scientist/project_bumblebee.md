---
name: bumblebee_bplusplus project context
description: Core facts about the MA bumblebee species classification project, experiments run, and current research direction
type: project
---

Mia Sun is building a ResNet-50 classifier for 16 Massachusetts bumblebee species (GBIF data). Primary challenge: extreme long-tail — B. ashtoni (23 train, 6 test) and B. sandersoni (39 train, 9 test) are near-extirpated.

**Current pipeline:** GPT-image-1 synthetic generation → GPT-4o LLM-as-judge filtering → ResNet-50 classifier

**Key results as of 2026-03:**
- Baseline macro F1: 0.810 [0.767, 0.841]
- D5 (LLM-filtered synthetic, best volume V_200): macro F1 0.852 [0.814, 0.879] — not stat. sig. at 95%
- Copy-paste (CNP) also improves F1 but is limited in diversity
- Volume ablation (50–300 synth/species): no statistically significant improvement; 95% CIs overlap
- Background removal test: did NOT help; primary failure driver is borderline coloration accuracy, not backgrounds
- LLM judge: wrong_coloration is the dominant failure mode for B. ashtoni (15% fail rate); B. sandersoni passes at 99.3%

**New direction (2026-03-18):** Collect ~150 expert (entomologist) labels (50/species x 3 species) and use them as training signal to build a learned filter.

**Key constraints:**
- 150 expert labels total, cannot do multiple rounds
- Test sets are tiny (n=6, n=10) — single-run F1 unreliable; always use bootstrap CI + multi-seed
- Need to filter 1000+ synthetic images

**Why:** Expert labels represent ground truth on morphological accuracy; LLM judge has a ceiling because it cannot reliably detect borderline coloration inaccuracies.

**How to apply:** Recommend reward model / learned filter approach using expert labels. CLIP/DINOv2 linear probe is the highest-priority experiment given the constraint set.

**Research question investigated (2026-04-08): Pairwise preference learning + DINOv2 fine-tuning**
- Verdict: not recommended at N=150. Bradley-Terry on PCA-compressed frozen features is mathematically equivalent to the composite logistic regression plan (LLM scores + BioCLIP PCs) but adds annotation complexity without information gain. DINOv2 backbone fine-tuning (LoRA or full) is infeasible — EPP=0.003 with LoRA at N=150. Only feasible pairwise scenario: targeted follow-up if kappa for borderline tier falls below 0.50 (collect ~50–100 intra-species pairs within borderline images only to resolve rank ordering at the threshold boundary).
- Use BioCLIP embeddings (not raw DINOv2) as backbone for composite model — domain-aligned to biological specimens via TreeOfLife-10M training.
- Structured rubric annotation (absolute binary labels + 5 per-feature scores) strictly dominates pairwise for this task because: (1) directly supervises the binary decision boundary, (2) enables LLM judge per-feature calibration, (3) provides prompt refinement signal.
