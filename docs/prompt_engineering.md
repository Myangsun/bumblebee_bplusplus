# Prompt Engineering for Synthetic Bumblebee Image Generation

## Overview

We use OpenAI's `gpt-image-1.5` model to generate photorealistic synthetic bumblebee images for training data augmentation. The generation pipeline lives in `notebooks/prompt_testing.ipynb` and is designed for rapid iteration on prompt text, reference images, and configuration parameters.

The goal: produce images that are morphologically faithful enough to improve a ResNet-50 classifier on rare species (B. sandersoni, B. ashtoni, B. flavidus) where real GBIF training data is scarce (36–314 images).

---

## Iteration Workflow

Each experiment follows a tight loop:

1. **Edit config cell** — set species, N images, experiment name
2. **Load reference images** — real photos from `references/{species}/`
3. **Generate batch** — calls `images.edit` (reference-guided) or `images.generate` (text-only)
4. **Inspect grid** — visual check against real GBIF samples
5. **Diagnose failures** — identify which morphological features are wrong
6. **Refine** — adjust the relevant prompt component (morphology text, caste description, scale, environment, or template structure)
7. **Repeat**

### What to iterate on

| Component | When to change | Location |
|---|---|---|
| `SPECIES_MORPHOLOGY` | Species colours/banding are wrong | Cell 3 |
| `CASTE_OPTIONS` | Caste-specific features are missing | Cell 5 |
| `build_scale_instruction()` | Bee is too large/small relative to flowers | Cell 5 |
| `ENVIRONMENTS` | Background is repetitive or biasing the classifier | Cell 5 |
| `PROMPT_TEMPLATE` | Structural issues (section ordering, emphasis) | Cell 5 |
| Reference images | Model ignores morphology details | `references/{species}/` |
| `input_fidelity` | Reference images have no effect on output | Cell 7 (`generate_single`) |
| `MODEL_CONFIG` | Resolution/quality tradeoffs | Cell 5 |

### A/B comparison

`compare_experiments()` generates side-by-side grids from two configs. Use it to isolate the effect of a single change (e.g., updated morphology text vs. original).

---

## Final Prompt Structure

The prompt is assembled by `build_prompt()` from a template with 8 placeholders:

```
{species_name}              ← scientific name (e.g., "Bombus bohemicus (inc. ashtoni)")
{common_name}               ← vernacular name (e.g., "Ashton's cuckoo bumble bee")
{caste_description}         ← worker/queen/male description with colour map
{morphological_description} ← detailed species morphology with negative constraints
{view_angle}                ← lateral / dorsal / three-quarter anterior / etc.
{wings_style}               ← folded or slightly spread
{environment_description}   ← randomly sampled habitat scene
{scale_instruction}         ← species-specific size + proportional flower anchors
```

### Template sections (in order)

1. **Role** — "You are an expert in insect morphology and photorealistic rendering"
2. **Task** — References input photos, defines the generation objective
3. **Subject identity** — Caste-specific description (inserted via `{caste_description}`)
4. **Morphological notes** — Full species description (inserted via `{morphological_description}`)
5. **Scale and proportion** — Species-specific body size with flower-based anchors
6. **Pose and viewpoint** — View angle, wing position, natural variation
7. **Lighting variation** — Outdoor only, varying sun angle/softness
8. **Background requirements** — Environment description, depth of field, flower density
9. **Flower realism** — Imperfections, asymmetry, natural variation
10. **Quality constraints** — No artifacts, no repeating compositions, species-agnostic backgrounds

### Design principles

- **Negative constraints are critical.** For species that deviate from the "generic yellow bumble bee" archetype, explicit "DO NOT" instructions prevent the model from reverting to common patterns. Example: B. ashtoni descriptions start with "WARNING: This is NOT a typical yellow-and-black bumble bee."
- **Front-to-back colour maps** in caste descriptions (e.g., "pale yellow collar → BLACK thorax → BLACK T1 → BLACK T2 → ...") give the model a spatial sequence to follow.
- **Reference images + `input_fidelity="high"`** are essential. Without `input_fidelity="high"`, the `images.edit` endpoint treats reference photos at low resolution and they have no visible effect on output. With it, generated bees match reference morphology noticeably better.
- **Environments are species-agnostic.** 15 diverse environments are randomly sampled (not cycled deterministically) to prevent the classifier from learning background shortcuts.
- **Scale uses proportional anchors, not absolute sizes.** The model doesn't understand millimetres or frame percentages. Instead: "a white clover head is ~20 mm (same size as the bee or LARGER)." The instruction explicitly warns that "generated bees tend to be TOO LARGE" because that is the observed error direction.

---

## Key Configuration Details

### Reference images

Stored in `references/{species_name}/`. The `images.edit` endpoint accepts them as PNG BytesIO tuples — bare `Path` objects fail with the OpenAI SDK. Conversion handles EXIF rotation and RGBA→RGB compositing.

```
references/
├── Bombus_ashtoni/
│   ├── bombus_ashtoni.jpg
│   └── bombus_ashtoni_s.jpg
├── Bombus_sandersoni/
│   ├── sandersoni_photo.jpg
│   └── Bombus-sandersoni.jpg
└── Bombus_flavidus/
    ├── bombus-flavidus.png
    └── bombus-flavidus_s.png
```

### Caste selection

Defined only for the 3 target species in `CASTE_OPTIONS`. Each entry includes a detailed colour map specific to that caste. Selection is weighted 3:1 toward workers (eusocial species) or females (cuckoo species) to match natural field photo distributions. Caste is recorded in output filenames and manifests.

| Species | Available castes | Notes |
|---|---|---|
| B. sandersoni | worker, queen, male | Eusocial — workers most common |
| B. ashtoni | female, male | Cuckoo (Psithyrus) — no workers |
| B. flavidus | female, male | Cuckoo (Psithyrus) — no workers |

### Model config

```python
MODEL_CONFIG = {
    "model": "gpt-image-1.5",
    "size": "1024x1024",
    "quality": "high",
    "output_format": "png",
}
```

### Output structure

```
RESULTS/prompt_testing/{experiment_name}/
├── {species}_{0000}_{caste}_{view_angle}.png
├── {species}_{0001}_{caste}_{view_angle}.png
├── ...
├── grid.png
└── manifest.json
```

The manifest records model config, prompts, caste, view angle, and environment for each image — enabling full reproducibility of any experiment.

---

## Iteration History (Key Decisions)

| Version | Change | Reason |
|---|---|---|
| v1 | Generic 3-option scale (small/medium/large) | Initial baseline |
| v2 | Species-specific scale with mm body lengths | Generic scale didn't differentiate species |
| v3 | Proportional flower anchors, removed frame_pct | Model doesn't understand mm or frame percentages |
| v4 | Added negative constraints to morphology | B. ashtoni was generating as generic yellow bee |
| v5 | Added caste descriptions with front-to-back colour maps | Missing sex dimorphism, incomplete colour guidance |
| v6 | Expanded environments from 5→15, random sampling | Deterministic cycling caused repetitive backgrounds |
| v7 | Added `input_fidelity="high"` to `images.edit` | Reference images had no visible effect at default (low) fidelity |
| v8 | Fixed SDK image passing (Path→BytesIO tuples) | Bare Path/BytesIO objects fail with OpenAI SDK `extract_files` |

### Lessons learned

1. **Text-to-image models default to common patterns.** Without strong negative constraints, every bumble bee looks like B. impatiens (yellow thorax, black abdomen). Rare species need explicit "this is NOT..." instructions.
2. **Reference images matter, but only at high fidelity.** The `input_fidelity` parameter is not documented prominently — defaulting to low silently makes references decorative.
3. **Proportional reasoning works better than absolute.** "The bee should be the same size as a clover head" is actionable; "the bee is 12 mm" is not.
4. **Randomize everything the classifier shouldn't learn.** Environments, lighting, flower arrangement, and composition must vary randomly. Deterministic cycling creates spurious correlations.
5. **Iterate visually.** The grid display + side-by-side comparison is the fastest feedback loop. Quantitative evaluation (LLM judge, expert review) comes after prompt engineering stabilizes.
