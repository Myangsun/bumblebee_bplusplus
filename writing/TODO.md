# TODO — ECCV 2026 Revision

**Paper:** `writing/paper-template-Latest/main.tex`, `supplementary.tex` · **Bib:** `main.bib`
**Original reviews:** `writing/REVIEW_LIST.md` (each task cites the `R#` it resolves)

Two phases, run in order. **Phase 1 = experiments** — run every model/ablation and write results to
disk; **touch no `.tex` file.** **Phase 2 = writing & figures** — starts only after all Phase-1
results are in, and does every `main.tex` / `supplementary.tex` / `main.bib` edit.

Rationale: (a) the author wants all results before any writing; (b) keeping compute out of the
`.tex` files means Phase-1 agents never clobber each other's paper edits, and Phase-2 numbers are
never stale.

`[COMPUTE]` = requires training runs. `→ deps` = must wait for the named task.

---

# PHASE 1 — Experiments (run first; output = result files only, no `.tex` edits)

- [x] **E1 — Leakage audit** `[COMPUTE: none — RESOLVED by audit]` · R20, advisor #3 · AS1
  - **Audit verdict (2026-07-06):** The **fixed locked split is leak-free.** CP foregrounds are
    train-only (`copy_paste.py:178`), the D5 centroid is fit on a train-only embedding cache
    (`run_filter.py:43-48`), the D6 probe is fit on the 150 expert-labeled *synthetic* images
    (never real val/test), and the gpt-image-1.5 references are **external photos** (author-confirmed),
    not GBIF images. **No re-fit needed.**
  - **Only the 5-fold CV leaked** (held-out reals informed CP sources + centroid via the reused
    global pool) — resolved by **dropping CV entirely** (author decision; see W1).
  - **Carried into writing (W1):** (a) add the no-leakage guarantee paragraph to §3; (b) FIX the
    factual error at `main.tex:481` — the probe is fit on 150 expert-labeled **synthetic** images,
    **not** the "full real training set"; only the **centroid** uses real training images.

- [x] **E2 — 5-seed locked-split sweep** `[COMPUTE: none — already run]` · R20, R21, advisor #2
  - **Decision (author, 2026-07-06):** Standardise on **5 seeds (42–46)**, not 10. These runs
    **already exist** in `RESULTS/` for all six variants — **no new training needed.** Data verified
    leak-free and share an identical real-only val/test (2335 / 2362) across D1–D6; each augmented
    variant adds exactly **+200 per rare species** (ashtoni 22→222, sandersoni 40→240,
    flavidus 162→362). Use the `@f1` checkpoint. Paper↔`RESULTS/` stem map:
    D1=`baseline`, D2=`d3_cnp`, D3=`d4_synthetic`, D4=`d5_llm_filtered`, D5=`d2_centroid`, D6=`d6_probe`.
  - **Tradeoff (AS5):** 5 seeds keeps paired-t **df=4** (underpowered), against the advisor's 10-seed
    suggestion — flagged; author chose 5. If a 10-seed (47–51) sweep was launched earlier, `scancel`
    it; the 47–51 scripts have been removed.
  - **Output (exists):** per-seed macro-F1 + per-species F1 for D1–D6 at seeds 42–46 in `RESULTS/`.

- [ ] **E3 — Classical long-tail baselines** `[COMPUTE: moderate — scripts ready]` · R19, advisor #1 · AS2
  - **Goal:** Show the null augmentation result holds even against standard long-tail machinery.
  - **Run:** On the baseline (real-only) data × **5 seeds (42–46)**: class-balanced weighted CE
    (Cui 2019), Balanced Softmax (Ren 2020), LDAM-DRW (Cao 2019) = **15 runs**. Scripts ready:
    `jobs/train_loss_baselines.sh` (+ `jobs/evaluate_baselines.sh`). Code in `simple.py`/`run.py`
    (`--loss`), smoke-tested.
  - **Output:** per-seed metrics `RESULTS/baseline_seed{42..46}_{wce,bsm,ldam}_gbif`.

- [ ] **E4 — LT-aware augmentation / oversampling baselines** `[COMPUTE: moderate — needs impl]` · R19
  - **Goal:** Match Fill-Up Table 3's non-generative baselines (Remix, BS+CMO), not generic aug.
  - **Decision (author, 2026-07-07):** **dropped RandAugment + MixUp**; use **Remix** (Chou 2020,
    LT-aware label mixing) and **BS+CMO** (Park 2022, minority-foreground oversampling + Balanced
    Softmax). On baseline data × **5 seeds (42–46)** = **10 runs**.
  - **Impl:** Remix = `train_epoch` label-mix bias by class count; CMO = minority-sampled foreground
    CutMix-paste + Balanced Softmax loss. New flags + SLURM launcher; update `evaluate_baselines.sh`
    tags (`remix`, `cmo`). RandAugment/MixUp code stays in the harness (RandAugment is used inside E5).
  - **Output:** per-seed metrics `RESULTS/baseline_seed{42..46}_{remix,cmo}_gbif`.

- [x] **E5 — Fill-Up-style two-stage recipe** `[COMPUTE: moderate]` · R19, advisor #1 · AS2 · ⚙ implemented
  - **Scope:** two-stage on the **existing synthetic pool** (no textual inversion). **Stage I** =
    real+synthetic pool (Balanced Softmax + RandAugment); **Stage II** = real-only warm-start
    fine-tune (Balanced Softmax + RandAugment, lower LR). **Both stages reported** (Table 3 Stage I/II).
    D3 + D6 pools, seeds 42–46 = 10 runs → 4 result rows. Script `jobs/train_fillup.sh`; tags
    `fillup_{d3,d6}_s1` (I) and `fillup_{d3,d6}` (II). "Fill-Up-style," not a reproduction.
  - **BS prior FIXED:** `--bs-real-prior` computes the Balanced-Softmax prior from real-only counts in
    both stages (Fill-Up-faithful), applied in the launcher. **One remaining deviation:** under-balance
    pool (Fig 6) — data-availability limit, stated as future work (BASELINES §5b).

- [ ] **E8 — cRT (decoupling) baseline** `[COMPUTE: light — needs impl]` · R19 · Fill-Up precedent
  - **Goal:** Cover the decoupling family (Kang 2020) — the one Fill-Up itself compares against
    (`Decouple-cRT`).
  - **Run:** Reuse the **existing D1 checkpoints** (seeds 42–46): freeze the backbone, re-train **only
    the classifier head** with class-balanced sampling. ~5 runs, cheap (head-only).
  - **Impl:** a `--decouple-crt` path in `simple.py` that loads a D1 checkpoint, freezes `backbone`,
    and trains `classifier` with a class-balanced sampler. New SLURM launcher + eval. ✅ implemented.
  - **Output:** `RESULTS/baseline_seed{42..46}_crt_gbif`.

- [ ] **E9 — Decouple-LWS baseline** `[COMPUTE: light — needs impl]` · R19 · Fill-Up precedent (`Decouple-LWS`)
  - **Goal:** Second decoupling row from Table 3 — cheapest variant, freeze the whole model and learn
    only a per-class logit scalar.
  - **Run:** Reuse the cRT stage-1 representation (seeds 42–46); freeze backbone **and** classifier,
    learn a 16-dim per-class scale on the final logits with class-balanced sampling. ~5 runs.
  - **Impl:** a `--decouple-lws` path in `simple.py` (learnable per-class scale on `classifier`
    output; only that vector trains). Add `lws` tag to `metrics.py` discovery + eval; SLURM launcher.
  - **Output:** `RESULTS/baseline_seed{42..46}_lws_gbif`.

- [ ] **E6 — Extended volume ablation** `[COMPUTE: light–moderate]` · R9 *(stretch — constrained)*
  - **Goal:** Remove the "+200 is arbitrary" objection.
  - **Pool ceiling verified (2026-07-06): 500 generated images/species** only
    (`RESULTS_kfold/synthetic_generation`, 500/500/500). Consequences:
    - **+500** reachable **only for D3 unfiltered** (uses the whole pool); filtered variants
      (D4/D5/D6) pass <500, so +500 filtered is not achievable.
    - **Fill-to-target (~900/species) is infeasible** for all variants — needs ~740/860/880
      synthetic for flav/sand/ash, above the 500 available, without generating more.
  - **Recommendation:** do NOT run costly gpt-image-1.5 generation for the deadline. Optionally add a
    single **+500 D3-unfiltered** ceiling point; otherwise rely on the existing flat +50→+300 sweep
    and justify +200 as volume-parity in W3. Fill-to-target = future work, stated honestly.

- [ ] **E7 — BioCLIP2 embeddings** `[COMPUTE: light]` · R12 *(lowest priority)*
  - **Goal:** Robustness note for "could we try BioCLIP2?".
  - **Current backbone verified: BioCLIP v1** (`embeddings.py:59`, `hf-hub:imageomics/bioclip`,
    ViT-B/16). E7 swaps this to `hf-hub:imageomics/bioclip-2`, re-embeds the pool, recomputes D5 5-NN
    purity and optionally D6 probe AUC. No ResNet-50 retraining.
  - **Caveat:** `embeddings.py:60` hardcodes another user's path (`/home/su/bioclip/src`, guarded by
    `exists()`); load bioclip-2 via `open_clip` directly.

**Phase-1 gate:** every metric file exists on disk; no `.tex` file has been edited yet.

**Phase-1 status:** E1 resolved (audit). **E2 done** (5-seed runs already exist). Remaining runs are
E3 (15) + E4 (10) — scripts ready, independent, parallel. E6/E7 are optional/constrained.

---

# PHASE 2 — Writing & figures (after ALL Phase-1 results in)

> The three results-integration tasks (W1, W2, W3) all edit the **same downstream table** in
> `main.tex`/`supplementary.tex` — assign them to **one agent or serialize** them. W4–W11 are
> mostly disjoint prose/figures and can parallelize.

- [ ] **W1 — Integrate primary results & restructure protocol** · R20, R21, AS5 · uses E2
  - **Goal:** One primary protocol in the main text; retire the "three protocols disagree" story.
  - **Changes:** Rebuild the main downstream table (`main.tex:704–750`) from E2 as mean±sd macro-F1 +
    per-species F1 on the fixed test set. **DROP five-fold CV entirely** (it leaked — see E1; author
    decision) and remove the single-split-bootstrap-as-a-separate-protocol column; fold bootstrap CIs
    into the **5-seed** reporting. Re-run paired t-tests on the **5-seed** data (**df=4**), replacing
    `supplementary.tex:242–272`. Rewrite the §5.6 "different rankings" framing (there is now one
    protocol). Update the abstract, §1.4, §5, §6 numbers. (Author chose 5 seeds over the advisor's
    10; df=4 is underpowered — keep downstream claims conservative, per the existing §5 hedge.)
  - **Leakage writing (from E1):** (a) add a no-leakage guarantee paragraph to §3 (`:297–313`) —
    CP sources, references (external), centroid, probe, thresholds all train-only, val/test never in
    the generation pipeline; (b) FIX `main.tex:481` — probe is fit on 150 expert-labeled **synthetic**
    images, not the "full real training set"; only the centroid uses real training images.

- [ ] **W2 — Baseline rows & narrative** · R19, AS2 · uses E3, E4, E5
  - **Goal:** Put the new baselines in the main comparison so reviewers can see how the numbers stack.
  - **Changes:** Add long-tail (E3), augmentation (E4), and Fill-Up (E5) rows to the main table; write
    a "baselines" paragraph in §5; add bib entries (Balanced Softmax, LDAM-DRW, RandAugment, MixUp,
    Fill-Up). Discuss whether the Fill-Up incorporation method changes the verdict (§5/§6).

- [ ] **W3 — Volume ablation write-up + +200 justification** · R9 · uses E6
  - **Changes:** Justify the +200 budget in §3/§4 as volume-parity supported by the existing flat
    +50→+300 sweep (`supplementary.tex:132–151`); optionally add the +500 D3-unfiltered ceiling point
    if E6 runs it. State plainly that fill-to-target (~900/species) is **beyond the 500-image
    generated pool** and left to future work (do not imply it was tested).

- [ ] **W4 — BioCLIP2 robustness line** · R12 · uses E7
  - **Changes:** Add a BioCLIP2 comparison line to §4.5/§5.4 or the supplementary, or a justified
    deferral if E7 was skipped.

- [ ] **W5 — Training-story consistency + Diagram_A** · R5, R6
  - **Goal:** One consistent training story across text and Figure 1.
  - **Verified from code (2026-07-06):** `simple.py:114-116` loads `ResNet50_Weights.DEFAULT`
    (ImageNet-pretrained), replaces `fc` with a new head, and **never freezes the backbone**; the
    optimizer gets all `model.parameters()` → **fine-tuned end-to-end from ImageNet.** So §3 is
    correct; §4 `:324` "trained from scratch" is **wrong**; the figure's linear-probe implication is
    **wrong**.
  - **Changes:** Standardise on **"fine-tuned end-to-end from ImageNet-pretrained weights."** Fix
    `main.tex:324` ("trained from scratch" → this). Regenerate `Diagram_A.jpg` so it does not depict a
    frozen backbone / linear probe. `main.tex:299`, `:324`.

- [ ] **W6 — Missing citations + reframe generator novelty** · R8, advisor #1
  - **Changes:** Add BibTeX for **Fill-Up: Balancing Long-Tailed Data with Generative Models** and
    **Personalized Representation from Personalized Generation**. In §2 (`main.tex:211–224`) cite both
    and state they use **fine-tuned generators** (textual inversion / DreamBooth) because off-the-shelf
    models are weak in data-scarce regimes, **whereas we use off-the-shelf `gpt-image-1.5` and study
    *selection* from its pool.** (Coordinate bib with W2.)

- [ ] **W7 — ResNet-50 = deployment-framing statement** · advisor #4, AS3
  - **Changes:** One sentence in §1 (~`:143`) and one in §3 (`:297`): the goal is efficient,
    field-deployable monitoring, so we fix ResNet-50 and benchmark against ResNet-50-compatible
    long-tail methods rather than foundation-model classifiers.

- [ ] **W8 — Quick clarity fixes** · R1, R2, R3, R4, R7, R16, R17
  - **Changes:**
    - R2: `main.tex:132` "an obvious remedy" → "a possible remedy."
    - R4: §5.4 "deployed 200-image selection" → "selected 200-image subset" (`~:647`).
    - R7: §1.4 bullet 2 (`:173–177`) — clarify structured morphological prompting generates the
      **entire** pool shared by D3–D6; it is not a D6-only step.
    - R16: §5.2 (`:554–556`) — define distances: 0.1–0.2 = real-image cosine distance to that
      species' **own real centroid**; synthetic-to-centroid = distance to the **same real centroid**.
    - R17: §5.4 (`:647–649`) — state the oracle ceiling of expert-strict positives (27/17/6 from the
      150 labels, or correct achievable max) so D6's 16/20/6 is read against a maximum.
    - R1: abstract (`main.tex:76–80`) — reword "selects subsets roughly 3× richer in expert-validated
      images at the same budget"; make the comparison explicit (vs LLM/centroid, counting
      expert-strict positives in the +200 cap). Coordinate with W11 so it does not over-sell.
    - R3: §5 intro (`main.tex:490–493`) — clarify "better-powered intermediate signals… primary
      evidence"; state plainly *why* they are better-powered (large synthetic pool / 150 expert
      labels) than the tiny rare-tier test set.

- [ ] **W9 — Regenerate figures + add qualitative panels** · R13, R14, R15
  - **Changes:** (R14) Regenerate Fig 4 t-SNE with larger fonts/legends and an enlarged right panel
    (Okabe-Ito, integer axes, legend outside, white bg). (R13) Add a §3 figure of the three rare
    targets beside their confusers (*sandersoni*/*vagans*; *ashtoni* dark-white vs yellow-worker
    prior; *flavidus* vs yellow-banded congeners) — reuse thesis `species_samples`. (R15) Add a §5.4
    panel of **successful** synthetics the expert accepted, to balance the failure figure. Assets in
    `docs/plots/`, `thesis/images/`; scripts in `scripts/`.

- [ ] **W10 — Restructure §4.3–4.5 + redesign Fig 6b** · R11, R18
  - **Changes:** (R11) Group §4.3/4.4/4.5 under **"Improving Fidelity of Reference-Based Generative
    Images with Filtering"** with three sub-subsections (§4.2 generation stays separate). (R18)
    Replace Fig 5(b) with **filter-vs-expert-label classification performance** (precision/recall/F1
    or agreement) for LLM vs Centroid vs Probe on the 150 expert labels; keep coverage in
    text/supplementary if useful. `main.tex:383–466`, `:660–665`.

- [ ] **W11 — Reconcile abstract / contribution framing with the null result** · AS4, ties to R1
  - **Goal:** Remove the tension where the abstract and §5.4 sell D6 ("3× richer") while §5.6/§6 walk
    it back to no reliable downstream gain — own the dissociation as the finding.
  - **Changes:** One consistent claim across abstract (`:64–92`), §1.4 bullet 3 (`:166–187`), §5.4
    (`:635–649`), §5.6/§6: better *selection* yields a better *intermediate* signal that does **not**
    convert to a separable downstream gain — that is the result, not a shortfall.

---

## Traceability — every review → task

| Review | Task | Review | Task | Review | Task |
|--------|------|--------|------|--------|------|
| R1 | W8, W11 | R8 | W6 | R15 | W9 |
| R2 | W8 | R9 | E6, W3 | R16 | W8 |
| R3 | W8 | R10 | W6 (framing) + R10 note | R17 | W8 |
| R4 | W8 | R11 | W10 | R18 | W10 |
| R5 | W5 | R12 | E7, W4 | R19 | E3, E4, E5, W2 |
| R6 | W5 | R13 | W9 | R20 | E1, E2, W1 |
| R7 | W8 | R14 | W9 | R21 | E2, W1 |

All 21 reviews covered. **Additional suggestions** (my reviewer read): AS1 leakage → E1; AS2 null
result needs airtight baselines → E3/E4/E5/W2; AS3 deployment framing → W7; AS4 own the
selection-vs-downstream dissociation → W11; AS5 df=4 underpowered → E2/W1.

---

## Open decisions (flag before executing)

1. **Commit the 60-run E2 sweep?** Everything downstream depends on it; if yes, queue it first.
2. **Stretch depth:** E5 (Fill-Up) vs E6 (volume) compete for GPU budget — E5 has higher reviewer
   value if only one fits.
3. **Keep CV at all?** Advisor prefers dropping it. If kept in supplementary it must be fold-local
   (E1); otherwise remove to avoid a leakage objection.

## Note on R10

The "text-guided vs reference-guided" comparison has no existing experiment. If it was never run,
address it in prose (justify the reference-guided choice) in W6 rather than fabricating a result —
confirm with the author whether a small text-guided comparison is feasible (would become a Phase-1
run if so).
