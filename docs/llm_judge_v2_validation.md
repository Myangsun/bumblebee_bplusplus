# LLM judge v1 → v2 refinement: validation summary

Generated from `scripts/plot_judge_stability.py` + the raw JSON results in
`RESULTS_kfold/llm_judge_eval/*` and `RESULTS_kfold/llm_judge_decomposed/*`.

## Motivation

The contact-sheet panels at
`RESULTS_kfold/llm_judge_eval/contact_Bombus_<species>.png` showed a
viewpoint-skewed failure pattern: frontal / head-on poses of *B. ashtoni*
appear disproportionately in the FAIL grid relative to lateral and dorsal
poses. A first pass at the data attributed this to the LLM-judge rubric's
critical-feature veto (rule 6 of `scripts/llm_judge.py`), but a more careful
trace identified a different mechanism: the judge's binary
`not_visible: bool` flag is defined as "not visible OR obscured," which
conflates *out of frame* with *in frame but not diagnostically exposed*. On
frontal *B. ashtoni* images, 21 % of `abdomen_banding` judgements correctly
mark the tergites `not_visible`, and the remaining 79 % split between
guessing a low score (dragging `mean_morph` down) and downgrading
`diagnostic_completeness` to `"genus"` (which lands the image in the
soft_fail tier and therefore in the contact-sheet FAIL panel).

A second question surfaced while testing v2: **is v1 stable enough for
per-image verdicts to be meaningful at all?** This document reports a
test-retest validation across all three rare species and answers both
questions.

## Experimental setup

For each of *B. ashtoni*, *B. sandersoni*, *B. flavidus*, a stratified
80-image cohort was drawn from the single-run v1 judge output:

- 40 images (or all available) from the v1 *soft_fail* tier
  (`matches_target` true, `diag` ≠ `"species"` — the contact-sheet FAIL cell)
- 40 images from the v1 *strict_pass* tier (`matches_target` true,
  `diag = "species"`, `mean_morph ≥ 4.0`)

Sample sizes: n = 80 (ashtoni), 66 (sandersoni — only 26 soft_fails exist
in the 500-image pool), 80 (flavidus). Total N = 226.

Each cohort was scored by four judge configurations:

| config | description |
|---|---|
| v1 run 1 | original `scripts/llm_judge.py` output from `results.json` |
| v1 run 2 | same prompt, same code, fresh API call (`scripts/llm_judge_rerun.py`) |
| v2 run 1 | `scripts/llm_judge_decomposed.py` (tri-state visibility) |
| v2 run 2 | same, second fresh run |

All runs use GPT-4o at `temperature = 0` with structured Pydantic output.

## Key finding 1: v1 is not a stable instrument

The single-run v1 verdicts that populate `results.json` (and the contact
sheets, strict-pass funnels, per-species heterogeneity statistics in
Section 5.3 of the thesis) flip on identical re-runs at a high rate:

| species | n | v1 `overall_pass` flip rate | v1 `diag` level flip rate |
|---|---:|---:|---:|
| B. ashtoni | 80 | **21.2 %** | 20.0 % |
| B. sandersoni | 66 | 3.0 % | 28.8 % |
| B. flavidus | 80 | 1.2 % | 28.7 % |
| **overall** | **226** | **8.8 %** | **26.1 %** |

*B. ashtoni* in particular loses 17 of 80 pass/fail verdicts (21.2 %)
between two identical calls. The `diag` field flip rate is 20–29 % across
all three species — i.e., on a given image the judge revises its
species-vs-genus commitment roughly a quarter of the time. This is the
field that drives the contact-sheet FAIL panel.

Implication: per-image v1 claims in the thesis (specific failure
modes on specific images, per-image filter agreement with expert, the
150-label LLM-vs-expert 56 % strict-pass agreement reported in §5.4.2) are
built on a measurement whose noise floor is ~5 pp above v2's.

## Key finding 2: v2 tri-state is materially more stable

Replacing the binary `not_visible` flag with a tri-state
`visibility ∈ {visible, not_assessable, not_visible}`, and excluding
non-`visible` features from `mean_morph`, produces:

| species | n | v2 `overall_pass` flip | v2 `diag` flip | v2 visibility-pattern exact match |
|---|---:|---:|---:|---:|
| B. ashtoni | 80 | **3.8 %** | **7.5 %** | 83.8 % |
| B. sandersoni | 66 | 1.5 % | 1.5 % | 75.8 % |
| B. flavidus | 80 | 5.0 % | 7.5 % | 76.2 % |
| **overall** | **226** | **3.5 %** | **5.8 %** | **78.8 %** |

Overall flip rate drops from 8.8 % to 3.5 % (2.5× improvement). On the
species where v1 was most unstable (*B. ashtoni*), the improvement is
5.6×. Diag-level flip rate drops from 26.1 % to 5.8 % across all three
species — the field that matters most for the contact-sheet FAIL panel is
now stable.

On *B. flavidus* the `overall_pass` flip rate is marginally higher under
v2 (5.0 %) than v1 (1.2 %), because the v1 lenient rule
(`mean_morph ≥ 3.0` + `diag ∈ {species, genus}`) was extremely permissive
on flavidus already and had little to disagree with on re-run. This is a
property of v1's rule, not evidence that v2 is less reliable — v2's diag
flip rate (7.5 %) is 4× lower than v1's (28.7 %) on the same cohort.

## Key finding 3: the tri-state flag is used meaningfully, and species-specifically

v2 invokes `not_assessable` and `not_visible` states at rates that track
each species' pose-diagnostic-surface geometry:

| species | feature with highest skip rate | % skipped | interpretation |
|---|---|---:|---|
| B. ashtoni | head_antennae (n_v) + abdomen_banding (n_a) | 15 % + 16 % | frontal poses hide the dorsal tergite pattern and sometimes the face |
| B. sandersoni | **abdomen_banding** (n_a) | **46 %** | the diagnostic T4 white tip requires dorsal view; most lateral-anterior poses legitimately skip it |
| B. flavidus | head_antennae (n_a) | 28 % | broad face-angle limits |

The sandersoni finding is particularly notable: nearly half the judge's
abdomen_banding judgements use `not_assessable` correctly, rather than
guessing a score and propagating noise into `mean_morph` and `diag`. This
is the kind of per-feature information the v1 binary flag could not
represent, and it explains why sandersoni's v1 `diag` flip rate was 28.8 %
while v2's is 1.5 % — v2 simply opts out of the ambiguous calls.

## What the refinement does NOT claim

1. **v2 does not "recover" soft_fail images as a block.** The soft_fail
   cohort pass rates are already high under v1 (because `diag = "genus"`
   satisfies v1's lenient overall_pass rule) and remain comparably high
   under v2. The value of v2 is in the *stability* and *mechanism
   visibility* of the verdicts, not in flipping aggregate pass rates.
2. **v2 does not address identity errors.** Images where the LLM blind-ID
   is wrong (hard_fail tier) require a different fix — likely pose-
   invariant BioCLIP features, which is what the D6 expert-probe filter
   already uses. v2 is a rubric refinement for the coverage-confound
   failure mode only.
3. **v2 has not been run on the full 1,500-image pool**, and no D4′
   downstream variant has been built. This validation is a targeted
   experiment to show v1's per-image noise floor and v2's targeted fix;
   downstream F1 effects are deferred.

## Files

- Judge scripts:
  [scripts/llm_judge.py](../scripts/llm_judge.py) (v1, unchanged),
  [scripts/llm_judge_decomposed.py](../scripts/llm_judge_decomposed.py) (v2),
  [scripts/llm_judge_rerun.py](../scripts/llm_judge_rerun.py) (helper for v1 rerun).
- Raw results:
  `RESULTS_kfold/llm_judge_eval/rerun_{ashtoni,sandersoni,flavidus}_tier.json`
  (v1 run 2),
  `RESULTS_kfold/llm_judge_decomposed/{species}_tier_v2trivis{,_rerun}.json`
  (v2 runs).
- Test-file manifests:
  `RESULTS_kfold/llm_judge_decomposed/{species}_tier_test_files_meta.json`.
- Plot:
  [docs/plots/judge_decomposed/stability_3species.png](plots/judge_decomposed/stability_3species.png)
  / .pdf — four-panel summary.
- Plotting script:
  [scripts/plot_judge_stability.py](../scripts/plot_judge_stability.py).

## Next steps (deferred)

- Run v2 on the full 1,500-image pool; build a D4′ filtered partition
  (top-200 per species by v2 rule); retrain ResNet-50 and compare D4 /
  D4′ / D6 under the same three evaluation protocols.
- Validate v2 against the 150 expert labels: strict-pass agreement and
  morph-mean AUC, side-by-side with v1's 56 % / 0.56 baseline from
  thesis §5.4.2.
- Report the v1 test-retest noise floor as a limitation of thesis §5.3
  per-image claims (not yet incorporated into the thesis text).
