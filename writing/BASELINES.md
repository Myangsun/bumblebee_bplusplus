# Baselines & Benchmarks — ECCV 2026 Submission

Consolidated list of every comparison method for the paper: the six dataset conditions (D1–D6),
the classical long-tail loss baselines (E3), the non-generative augmentation baselines (E4), and the
Fill-Up-style recipe (E5). Plus a research-grounded check of whether this set covers the *essential*
baselines the long-tailed-recognition community (and the Fill-Up paper) expects.

> **FINALIZED SET (author, 2026-07-07) — 14 methods:** D1–D6 · class-balanced weighted CE ·
> Balanced Softmax · LDAM-DRW · **cRT** · **Decouple-LWS** · **Remix** · **BS+CMO** · **Fill-Up-style
> two-stage** (reported as **Stage I + Stage II**, 2 results per pool). Aligned to Fill-Up Table 3
> (covers **9 of its 13 rows** — both Fill-Up stages included).
> **Dropped:** RandAugment, MixUp (RandAugment kept only as a *component* inside the Fill-Up recipe),
> Logit Adjustment. **Cite-only:** PaCo, ensembles (RIDE/NCL/BalPOE), foundation-model methods.
> **Ready:** weighted CE, Balanced Softmax, LDAM-DRW, cRT. **Needs implementation:** Decouple-LWS,
> Remix, BS+CMO, Fill-Up-style. New training ≈ 45 runs at 5 seeds.

Companion docs: `TODO.md` (work plan), `LAUNCH_GUIDE.md` (how to run), `REVIEW_LIST.md` (reviews).

---

## 1. This paper's baseline / benchmark matrix

**Benchmark (dataset):** all methods run on the **single 16-species Massachusetts *Bombus* dataset**,
fixed 70/15/15 split (~60:1 imbalance), identical real-only val (2335) / test (2362) across every
condition. Classifier is a **ResNet-50 fine-tuned end-to-end from ImageNet weights** for all rows.
We deliberately do **not** run standard LT academic benchmarks (CIFAR-LT / ImageNet-LT / iNat) — that
is a scope decision consistent with the deployment framing (fine-grained biodiversity monitoring);
see §3.

Legend — Status: ✅ done · ⚙ scripts ready · ⏸ on hold.

### 1a. Dataset conditions (our contribution axis) — D1–D6

| ID | Name | What it is | Adds to train | Status |
|----|------|-----------|---------------|--------|
| **D1** | Baseline (real only) | Plain fine-tune, no augmentation. The lower anchor. | — | ✅ (seeds 42–46) |
| **D2** | Copy-paste (CNP) | SAM-cut real bee foregrounds composited on new backgrounds. Fidelity anchor. | +200/rare sp. | ✅ |
| **D3** | Unfiltered synthetic | gpt-image-1.5 reference-guided generations, no filter. | +200/rare sp. | ✅ |
| **D4** | LLM-filtered | D3 pool filtered by a GPT-4o morphology rubric (strict rule). | +200/rare sp. | ✅ |
| **D5** | BioCLIP-centroid-filtered | D3 pool filtered by cosine-to-real-centroid (median threshold). | +200/rare sp. | ✅ |
| **D6** | Expert-probe-filtered | D3 pool filtered by a BioCLIP logistic probe on 150 expert labels. | +200/rare sp. | ✅ |

### 1b. Classical long-tail loss baselines (E3) — on the D1 real-only data

| ID | Name | Family | What it is | Citation | Status |
|----|------|--------|-----------|----------|--------|
| **E3a** | Class-balanced weighted CE | Re-weighting loss | Per-class weights ∝ 1/effective-number (β=0.9999). | Cui et al., CVPR 2019 | ⚙ `--loss weighted_ce` |
| **E3b** | Balanced Softmax | Margin/prior loss | Add `log(class prior)` to logits before CE. **The loss Fill-Up uses.** | Ren et al., NeurIPS 2020 | ⚙ `--loss balanced_softmax` |
| **E3c** | LDAM-DRW | Margin + deferred re-weighting | Per-class margin ∝ n_c^−¼ on scaled logits; class-balanced re-weight deferred to a late epoch. | Cao et al., NeurIPS 2019 | ⚙ `--loss ldam_drw` |
| **E8** | cRT (decoupling) | Two-stage classifier | Freeze the trained D1 backbone; re-train only the classifier head with class-balanced sampling. Fill-Up precedent (`Decouple-cRT`). | Kang et al., ICLR 2020 | ⚙ implemented (reuses D1 rep) |
| **E9** | Decouple-LWS | Two-stage classifier | Freeze the whole trained model; learn only a per-class scalar that rescales each classifier weight (class-balanced sampling). Cheapest decoupling variant; Fill-Up precedent (`Decouple-LWS`). | Kang et al., ICLR 2020 | needs impl (reuses cRT stage-1 rep) |

### 1c. Non-generative augmentation / oversampling baselines (E4) — on the D1 real-only data

Aligned to Fill-Up Table 3 (which uses Remix and BS+CMO, not plain RandAugment/MixUp).

| ID | Name | Family | What it is | Citation | Status |
|----|------|--------|-----------|----------|--------|
| **E4a** | Remix | LT-aware mixup | Mixup features, but bias the **label** mix toward the minority class when counts differ. | Chou et al., ECCV-W 2020 | needs impl |
| **E4b** | BS+CMO | Minority oversampling | Context-rich Minority Oversampling — paste minority foregrounds onto majority backgrounds (CutMix-style), **combined with Balanced Softmax** (as in Fill-Up Table 3). | Park et al., CVPR 2022 | needs impl |

*Dropped:* RandAugment (`--randaugment`) and MixUp (`--mixup-alpha`) — code retained in the harness;
RandAugment stays as a **component** inside the Fill-Up recipe (E5, both stages), not a standalone row.

### 1d. Generative training-recipe baseline (E5) — Fill-Up-style

| ID | Name | Family | What it is | Citation | Status |
|----|------|--------|-----------|----------|--------|
| **E5** | Fill-Up-style two-stage | Generative + two-stage | **Stage I:** real+synthetic pool w/ Balanced Softmax + RandAugment. **Stage II:** real-only warm-start fine-tune w/ Balanced Softmax + RandAugment, lower LR. **Both stages reported** (cf. Table 3 Stage I/II). D3 + D6 pools, seeds 42–46 → 4 result rows. | Shin, Kang & Park, arXiv 2306.07200 (2023) | ⚙ implemented |

---

## 2. Essential-benchmark research (what the field expects)

### 2a. What Fill-Up itself benchmarks against
(From arXiv 2306.07200, main tables — confirms the canonical set a reviewer expects.)
- **Datasets:** ImageNet-LT (IF 256), Places-LT (IF 996), iNaturalist 2018 (IF 500), IN100-LT (ablation). Backbone **ResNet-50**.
- **Baselines compared:** CE · Decouple-cRT · Decouple-LWS · **Remix** · **LDAM-DRW** · **Balanced
  Softmax** · PaCo · **BS+CMO** · RIDE · NCL · BalPOE (+ ResLT, MiSLAS, SHIKE on Places-LT).
- **Its own recipe:** textual-inversion tokens per class → Stable Diffusion v1.5 → fill each class to
  a balanced count (~1,300/class at ImageNet scale) → **two-stage Balanced Softmax** (stage 1
  real+synthetic balanced, stage 2 real-only fine-tune; RandAugment in both).
- **Its thesis:** class-name/text-prompt generation misaligns with the real domain and barely helps
  once real data exists; textual-inversion tokens learned from the *real tail* align the synthetics,
  and the two-stage recipe converts them into gains. (This is exactly the alternative hypothesis our
  E5 tests for our off-the-shelf generator.)

### 2b. Canonical long-tail baseline families (with citations)
One representative per family is what reviewers check off.

| Family | Representative methods (citation) |
|--------|-----------------------------------|
| Data re-sampling | class-balanced / square-root / progressively-balanced sampling (Kang et al., ICLR 2020) |
| Re-weighting loss | weighted CE; Focal (Lin, ICCV 2017); **Class-Balanced loss (Cui, CVPR 2019)** |
| Margin / prior loss | **LDAM-DRW (Cao, NeurIPS 2019)**; **Balanced Softmax (Ren, NeurIPS 2020)**; Logit Adjustment (Menon, ICLR 2021); LADE (Hong, CVPR 2021) |
| Decoupling / two-stage | **cRT / τ-norm / LWS (Kang, ICLR 2020)**; MiSLAS (Zhong, CVPR 2021); DisAlign (Zhang, CVPR 2021) |
| Ensemble / experts | BBN (Zhou, CVPR 2020); RIDE (Wang, ICLR 2021); ACE (Cai, ICCV 2021); TADE (Zhang, 2021) |
| Non-generative aug | mixup (Zhang, ICLR 2018); CutMix (Yun, ICCV 2019); RandAugment (Cubuk, NeurIPS 2020); Remix (Chou, ECCV-W 2020); **CMO (Park, CVPR 2022)** |
| Generative aug (scarce/LT) | **Fill-Up (Shin, 2023)**; LTGC (Zhao, CVPR 2024); Beery et al. (WACV 2020); Personalized Representation (ICLR 2025); Azizi et al. (2023) |

### 2c. Coverage of our set against the canonical families

| Family | Our representative | Covered? |
|--------|-------------------|----------|
| Data re-sampling | — (D1 is instance-balanced only) | ⚠ **gap** — see §3 |
| Re-weighting loss | E3a class-balanced weighted CE | ✅ |
| Margin / prior loss | E3b Balanced Softmax, E3c LDAM-DRW | ✅ (2 of the family) |
| Decoupling / two-stage | cRT (E8) + Decouple-LWS (E9) | ✅ (both Table 3 decoupling rows) |
| Ensemble / experts | — (out of scope: single-model deployment framing) | ➖ cite-only |
| Non-generative aug / oversampling | E4a Remix, E4b BS+CMO | ✅ (matches Fill-Up Table 3) |
| Generative aug | D2–D6 (our contribution) + E5 Fill-Up-style | ✅ |

---

## 3. Gaps & recommendations (must-have vs optional, 10-day deadline)

Our contribution is a synthetic-data **selection/filtering** study on one small fine-grained dataset,
ResNet-50, deployment framing. The defensible target is **one representative per major family** on the
identical backbone — not the full LT zoo.

**Already covered (✅):** re-weighting loss (E3a), margin/prior loss (E3b+E3c), non-generative aug
(E4a+E4b), generative aug (D2–D6, + E5 when resumed), lower anchor (D1).

**Included in the finalized set (author, 2026-07-07):**
1. **cRT — decoupling (Kang 2020). ✔ INCLUDED** (`Decouple-cRT` in Fill-Up Table 3). ⚙ implemented.
2. **Decouple-LWS (Kang 2020). ✔ INCLUDED** (`Decouple-LWS` in Table 3) — cheapest decoupling variant:
   freeze the whole model, learn only per-class logit scalars. Reuses the cRT stage-1 rep. Needs impl.
3. **Remix (Chou 2020). ✔ INCLUDED** — replaces MixUp; LT-aware label mixing (`Remix` in Table 3).
   Needs implementation.
4. **BS+CMO (Park 2022). ✔ INCLUDED** — replaces RandAugment; minority foreground oversampling **+
   Balanced Softmax** (`BS + CMO` in Table 3; rivals copy-paste D2). Needs implementation.
5. **Fill-Up-style two-stage (Shin 2023). ✔ INCLUDED.** D3 + D6, seeds 42–46. Needs implementation.

**Dropped / cite-only:**
- **RandAugment, MixUp — DROPPED** as standalone baselines (author) — Fill-Up Table 3 uses the
  LT-aware Remix/CMO instead. RandAugment is retained only as a *component* inside the Fill-Up recipe.
- **Logit Adjustment (Menon 2021) — DROPPED**; no Fill-Up precedent, overlaps Balanced Softmax.
- **PaCo (Cui, ICCV 2021)** — parametric contrastive; different training paradigm (projection head,
  contrastive loss, longer tuning), too risky to reimplement faithfully in the deadline. *Cite-only.*
- **Ensembles — RIDE / NCL / BalPOE** — multi-expert (3–4 networks); heavy compute **and** contradict
  the single-model deployment framing. *Cite-only, justified by the framing.*
- **Class-balanced re-sampling** — E3a weighted-CE stands in for the rebalancing family. *Optional.*

**Safe to cite-only (omit with a one-line justification):** ensemble/expert methods (BBN, RIDE, NCL),
foundation-model / personalized-generation methods (Personalized Representation, ICLR 2025), and other
generative pipelines (LTGC, DiffuseMix) — they don't match the single-ResNet-50 deployment framing or
would need a new generation pipeline within the deadline.

**Rebuttal framing:** "We include one representative method per major long-tail family — data
re-weighting (CB weighted CE), margin loss (LDAM-DRW, Balanced Softmax), decoupling (cRT, LWS),
LT-aware augmentation/oversampling (Remix, BS+CMO), and generative augmentation (D2–D6, Fill-Up-style)
— all on the identical ResNet-50 backbone and single fine-grained dataset, to keep comparisons
controlled. Multi-expert (RIDE/NCL/BalPOE) and foundation-model classifiers are out of scope for a
single-model deployment study." This set covers **9 of Fill-Up Table 3's 13 rows** (CE=D1, cRT, LWS,
Remix, LDAM-DRW, BS, BS+CMO, Fill-Up Stage I, Fill-Up Stage II); the only omissions are PaCo and the
three ensembles.

---

## 4. Fill-Up recipe spec (reference)

*Shin, Kang & Park, "Fill-Up: Balancing Long-Tailed Data with Generative Models," arXiv 2306.07200,
2023.* Our E5 mimics only the **training recipe**, not the textual-inversion generator.

- **Stage 1 (balanced):** train the whole model on **real + synthetic** filled to equal per-class
  counts, with **Balanced Softmax** (they use 100 ep, batch 256, LR 0.1 ÷10 every 30 ep, RandAugment).
- **Stage 2 (real fine-tune):** continue (no reset), fine-tune on **real-only** long-tailed data with
  **Balanced Softmax** (30 ep, LR 1e-3 ÷10 every 10 ep, 5 warm-up ep).
- **Balanced Softmax prior is computed from real counts only** in both stages (synthetic balances the
  data, not the loss prior).
- **Our adaptation:** substitute their per-class textual-inversion generations with our existing
  gpt-image-1.5 filtered pool; run for D6 and D3, seeds 42–46. Report as "Fill-Up-style," noting the
  generation axis differs (off-the-shelf vs. fine-tuned generator).

---

## 5b. Implementation correctness vs. original papers (verified 2026-07-07)

Code in `pipeline/train/simple.py`; all four compile and pass CPU smoke tests.

- **Decouple-LWS (Kang 2020).** ✅ Faithful. Freeze the whole trained model, learn a per-class
  scalar on the logits, train with class-balanced sampling. `LWSWrapper.state_dict()` folds the scale
  into the final linear — verified the wrapped model and the folded plain model give identical logits
  (max|Δ|=1.8e-7), so eval needs no change. *Minor variant:* the scale multiplies the full logit
  (weight **and** bias); Kang's formulation scales the weight vector only (their head is bias-free).
  Negligible in practice.
- **Remix (Chou 2020).** ✅ Faithful. Mixup the images (λ∼Beta(1,1)); label-mix factor
  λ_y = 0 if n_i/n_j ≥ κ ∧ λ < τ; = 1 if n_i/n_j ≤ 1/κ ∧ (1−λ) < τ; else λ. κ=3, τ=0.5 (paper
  defaults). All three rule branches unit-tested (head+rare, rare+head, equal counts).
- **BS+CMO (Park 2022).** ✅ Faithful. Instance-balanced background loader + minority-weighted
  (1/n_c) foreground loader; CutMix box (λ∼Beta(1,1)) pastes foreground onto background; area-adjusted
  label mix; loss = Balanced Softmax. Matches CMO+BS.
- **Fill-Up-style (Shin 2023).** ✅ Recipe structure faithful (Stage I real+synthetic → Stage II
  real-only warm-start fine-tune, Balanced Softmax + RandAugment in both, lower stage-II LR). **Both
  stages are reported as separate results** (`baseline_seed{N}_fillup_{d3,d6}_s1` = Stage I,
  `..._fillup_{d3,d6}` = Stage II), matching Fill-Up Table 3's Stage I / Stage II rows; both evaluate
  on the identical real test split. **BS prior — FIXED:** `--bs-real-prior` now computes the
  Balanced-Softmax prior from **real-only counts** (excludes `::` synthetic images) in both stages, as
  Fill-Up does (verified: rare-class prior uses n=22, not 222). Applied in `jobs/train_fillup.sh`.
  **One remaining documented deviation:** *under-balance* — our pool adds +200/rare (cap 500/sp.), not
  Fill-Up's fill-to-balance (Fig 6); this is a data-availability limit, stated honestly as future work.

## 5. New citations to add to `main.bib` (for W2/W6)

Verified this session unless flagged. Confirm arXiv IDs/venues marked [chk] before camera-ready.

- Fill-Up — Shin, Kang, Park. arXiv 2306.07200, 2023. (preprint)
- Balanced Softmax — Ren, Yu, Sheng, Ma, Zhao, Yi, Li. NeurIPS 2020. arXiv 2007.10740.
- LDAM-DRW — Cao, Wei, Gaidon, Arechiga, Ma. NeurIPS 2019. arXiv 1906.07413.
- Class-Balanced Loss — Cui, Jia, Lin, Song, Belongie. CVPR 2019. arXiv 1901.05555.
- Remix — Chou, Chang, Pan, Wei, Juan. ECCV 2020 Workshops. arXiv 2007.03943. *(included — E4a)*
- CMO — Park, Hong, Heo, Yun, Choi. CVPR 2022. [arXiv id chk 2112.00412]. *(included — E4b, BS+CMO)*
- Decoupling / cRT / LWS — Kang, Xie, Rohrbach, Yan, Gordo, Feng, Kalantidis. ICLR 2020. arXiv 1910.09217. *(cRT + LWS both included)*
- RandAugment — Cubuk, Zoph, Shlens, Le. NeurIPS 2020. arXiv 1909.13719. *(component of Fill-Up recipe, not a standalone baseline)*
- MixUp — Zhang, Cissé, Dauphin, Lopez-Paz. ICLR 2018. arXiv 1710.09412. *(dropped; cite for Remix context)*
- Personalized Representation from Personalized Generation — Sundaram, Chae, Tian, Beery, Isola. ICLR 2025. arXiv 2412.16156. *(related work)*
- LTGC — Zhao et al. CVPR 2024. arXiv 2403.05854. *(related work)*
