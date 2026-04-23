# LLM Judge Analysis — 2026-04-03

## Strict Filter Rules (D5 Assembly)

An image passes the strict filter if **ALL** of the following hold:

1. **Blind ID matches target species** (`matches_target == True`)
2. **Diagnostic completeness == "species"** (not genus/family/none)
3. **Mean morphological score >= 4.0**

The judge's own `overall_pass` is lenient (morph >= 3.0, diag >= genus, allows wrong_coloration). The strict filter above is what actually selects images for the D5 dataset.

---

## Summary

| Metric | Value |
|--------|-------|
| Total images evaluated | 1500 |
| Judge overall_pass (lenient) | 1458/1500 (97.2%) |
| Strict filter pass | 966/1500 (64.4%) |

## Filter Funnel

| Stage | Count | Rate |
|-------|-------|------|
| Total images | 1500 | 100% |
| matches_target | 1342 | 89.5% |
| + diag=species | 1060 | 70.7% |
| + morph>=4.0 | 966 | 64.4% |

## Per-Species Metrics

| Species | Total | Strict Pass | Pass Rate | Blind ID Match | Diag=species | Mean Morph | Judge Pass |
|---------|-------|-------------|-----------|----------------|--------------|------------|------------|
| Bombus_ashtoni | 500 | 222 | 44.4% | 380 (76.0%) | 298 (59.6%) | 3.82 | 460 (92.0%) |
| Bombus_flavidus | 500 | 288 | 57.6% | 480 (96.0%) | 306 (61.2%) | 4.06 | 498 (99.6%) |
| Bombus_sandersoni | 500 | 456 | 91.2% | 482 (96.4%) | 456 (91.2%) | 4.37 | 500 (100%) |

## Blind ID Breakdown

| Target Species | Identified As | Count |
|----------------|---------------|-------|
| Bombus_ashtoni | Bombus ashtoni | 341 |
| Bombus_ashtoni | Bombus ternarius | 63 |
| Bombus_ashtoni | Bombus bohemicus | 39 |
| Bombus_ashtoni | Bombus impatiens | 19 |
| Bombus_ashtoni | Unknown | 17 |
| Bombus_ashtoni | No match | 14 |
| Bombus_ashtoni | Other | 7 |
| Bombus_flavidus | Bombus flavidus | 480 |
| Bombus_flavidus | Bombus ternarius | 15 |
| Bombus_flavidus | Bombus impatiens | 5 |
| Bombus_sandersoni | Bombus sandersoni | 482 |
| Bombus_sandersoni | Bombus impatiens | 9 |
| Bombus_sandersoni | Unknown | 7 |
| Bombus_sandersoni | Other | 2 |

## Caste Fidelity

| Species / Caste | Correct | Total | Accuracy |
|-----------------|---------|-------|----------|
| Bombus_ashtoni / female | 296 | 351 | 84.3% |
| Bombus_ashtoni / male | 46 | 149 | 30.9% |
| Bombus_flavidus / female | 362 | 363 | 99.7% |
| Bombus_flavidus / male | 130 | 137 | 94.9% |
| Bombus_sandersoni / worker | 286 | 294 | 97.3% |
| Bombus_sandersoni / queen | 111 | 115 | 96.5% |
| Bombus_sandersoni / male | 90 | 91 | 98.9% |

## Failure Modes

| Failure Mode | Count | Rate |
|--------------|-------|------|
| wrong_coloration | 406 | 27.1% |
| extra_missing_limbs | 0 | 0.0% |
| impossible_geometry | 0 | 0.0% |
| blurry_artifacts | 0 | 0.0% |
| background_bleed | 0 | 0.0% |
| flower_unrealistic | 0 | 0.0% |
| repetitive_pattern | 0 | 0.0% |

## Diagnostic Completeness

| Level | Count | Rate |
|-------|-------|------|
| species | 1060 | 70.7% |
| genus | 438 | 29.2% |
| family | 1 | 0.1% |
| none | 1 | 0.1% |

## D4 Training Results (all 500 synthetic per species, unfiltered)

| Species | F1 (D4 +500) | F1 (prev D5 +200) | Delta |
|---------|-------------|-------------------|-------|
| Bombus_ashtoni | 0.67 | 0.67 | 0.00 |
| Bombus_sandersoni | 0.59 | 0.71 | -0.12 |
| Bombus_flavidus | 0.78 | (new species) | — |
| **Macro F1** | **0.84** | **0.85** | **-0.01** |

Sandersoni collapsed from 0.71 to 0.59 at +500 volume. Confirms +200 remains the sweet spot.

## Key Findings

1. **V8 prompts eliminated structural failures**: zero limb/geometry/artifact/repetitive issues across all 1500 images
2. **Wrong coloration is the only failure mode** (27.1%), concentrated in ashtoni (predominantly black bee is hard to generate)
3. **Ashtoni male generation is problematic**: only 30.9% caste-correct vs 84%+ for females
4. **Ashtoni is the filtering bottleneck**: 44% strict pass rate vs 91% for sandersoni
5. **+500 synthetic volume hurts sandersoni** (F1 0.59 vs 0.71 at +200) — high synthetic:real ratio causes degradation
6. **Strict filter yields enough images for +200**: ashtoni 222, flavidus 288, sandersoni 456

---

## D5 Dataset Rules

**Filter**: strict (matches_target + diag=species + morph>=4.0)
**Volume**: `--add 200` per species

```bash
python scripts/assemble_dataset.py \
  --mode llm_filtered --add 200 \
  --judge-results RESULTS/llm_judge_eval/results.json \
  --name d5_llm_filtered --force
```

Expected train counts: ashtoni 222, sandersoni 240, flavidus 362.

---

## Expert Validation Dataset Design (150 images)

### Quality Tiers

Each image is classified into one of four tiers based on the LLM judge output:

| Tier | Definition | Purpose |
|------|-----------|---------|
| **strict_pass** | matches_target + diag=species + morph>=4.0 | Images that would enter D5 — experts validate the filter accepts good images |
| **borderline** | matches_target + diag=species + 3.0<=morph<4.0 | Near-miss images — experts calibrate whether the 4.0 threshold is correct |
| **soft_fail** | matches_target + diag<species | Correct species but insufficient diagnostic detail — experts assess if the judge is too strict on completeness |
| **hard_fail** | NOT matches_target | Wrong species or unrecognizable — experts confirm the judge correctly rejects these |

### Available Pool

| Tier | Ashtoni | Sandersoni | Flavidus |
|------|---------|------------|----------|
| strict_pass | 222 | 456 | 288 |
| borderline | 76 | 0 | 18 |
| soft_fail | 82 | 26 | 174 |
| hard_fail | 120 | 18 | 20 |

### Stratification Strategy: Proportional with Floor

50 images per species. Allocation algorithm:

1. **Floor**: each non-empty tier gets a minimum of 5 images (guarantees small tiers are represented)
2. **Proportional fill**: remaining slots (50 minus floor total) are distributed proportionally to tier pool size
3. **Cap**: no tier can exceed its pool size

#### Allocation Breakdown

**Bombus_ashtoni** (4 non-empty tiers, floor = 4 x 5 = 20, remaining = 30)

| Tier | Pool | Proportion | Floor | + Proportional | Final |
|------|------|-----------|-------|----------------|-------|
| strict_pass | 222 | 44.4% | 5 | + 13 | 18 |
| borderline | 76 | 15.2% | 5 | + 5 | 10 |
| soft_fail | 82 | 16.4% | 5 | + 5 | 10 |
| hard_fail | 120 | 24.0% | 5 | + 7 | 12 |
| **Total** | **500** | | | | **50** |

**Bombus_sandersoni** (3 non-empty tiers, floor = 3 x 5 = 15, remaining = 35)

| Tier | Pool | Proportion | Floor | + Proportional | Final |
|------|------|-----------|-------|----------------|-------|
| strict_pass | 456 | 91.2% | 5 | + 32 | 37 |
| borderline | 0 | — | — | — | 0 |
| soft_fail | 26 | 5.2% | 5 | + 2 | 7 |
| hard_fail | 18 | 3.6% | 5 | + 1 | 6 |
| **Total** | **500** | | | | **50** |

**Bombus_flavidus** (4 non-empty tiers, floor = 4 x 5 = 20, remaining = 30)

| Tier | Pool | Proportion | Floor | + Proportional | Final |
|------|------|-----------|-------|----------------|-------|
| strict_pass | 288 | 57.6% | 5 | + 18 | 23 |
| borderline | 18 | 3.6% | 5 | + 1 | 6 |
| soft_fail | 174 | 34.8% | 5 | + 10 | 15 |
| hard_fail | 20 | 4.0% | 5 | + 1 | 6 |
| **Total** | **500** | | | | **50** |

### Why Floor-then-Proportional?

The goal of expert validation is **qualitative signal and threshold calibration**, not statistical hypothesis testing. Experts need to:
- See enough of each tier to form patterns (what does a borderline image look like?)
- Calibrate the judge's threshold (should morph≥4.0 be lowered to 3.0?)
- Identify where the judge disagrees with human judgment

Floor-then-proportional was chosen because:
1. **Representative**: the allocation mirrors the true quality distribution of each species' generated pool. Sandersoni mostly passes (91%) and the expert set reflects that — experts see what the generator actually produces.
2. **Minimum floor (5 per non-empty tier)**: guarantees even small tiers (e.g. sandersoni hard_fail, 18 images) get enough samples for experts to identify patterns, rather than being washed out by the dominant tier.
3. **Simple and reproducible**: no tuning parameters beyond the floor value.

Alternative considered: equal allocation per tier (maximizes per-tier signal but over-represents rare failures relative to the true distribution, which could bias expert impressions of overall quality).
