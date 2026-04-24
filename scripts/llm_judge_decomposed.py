#!/usr/bin/env python3
"""LLM-as-judge v2 — tri-state visibility refinement.

Motivation
----------
v1 (scripts/llm_judge.py) has a binary `not_visible` flag per feature. The
prompt defines this as "not visible or obscured," but fine-grained bumblebee
morphology has a THIRD state the prompt cannot express: *visible in frame
but not diagnostically assessable from this viewpoint*. A frontal shot of
B. ashtoni shows the bee clearly but does not expose the dorsal T1-T5
tergite pattern that actually defines the species. In v1 the judge either
(a) marks `not_visible=True` (happens for 21% of frontal shots on
abdomen_banding) and the feature is correctly skipped, OR (b) decides the
tergites are "sort of visible from this angle" and assigns a low guess
score, which drags `mean_morph` down and pushes the image into soft_fail.

v2 expands `not_visible: bool` into `visibility: str` with three levels:

  visible         — feature is in frame and its diagnostic surface is
                    exposed well enough to grade 1-5.
  not_assessable  — feature is in frame but the viewpoint/pose does not
                    expose the diagnostic surface for this species.
  not_visible     — feature is out of frame, occluded, or destroyed.

Features scored `not_assessable` or `not_visible` are excluded from
mean_morph. The judge reports diagnostic_completeness ("species" / "genus"
/ "family" / "none") independently — a typical frontal shot with the
tergites not_assessable will naturally be reported as "genus" by the
judge because the species-diagnostic surface is not in view.

Pass rule (pre-committed, anchor-derived; not expert-tuned):
  matches_target=True
  AND diag in {"species", "genus"}          -- lenient diag; genus OK because
                                               not_assessable tergites
                                               legitimately prevent species ID.
  AND mean_morph (visible features only) >= 4.0   -- strict anchor, "Good or better"
  AND no repetitive_pattern failure

Downstream code in scripts/assemble_dataset.py consumes `overall_pass` and
`morphological_fidelity.<feature>.score / not_visible` — the schema below
preserves both fields (with `not_visible` set True whenever visibility !=
"visible") so the LLM-filter funnel does not need to change.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path
from typing import Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

from pipeline.augment.synthetic import SPECIES_MORPHOLOGY, SPECIES_DATA, IMAGE_EXTENSIONS
from pipeline.config import RESULTS_DIR

load_dotenv()

MODEL = "gpt-4o"
DEFAULT_IMAGE_DIR = RESULTS_DIR / "synthetic_generation"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "llm_judge_decomposed"
SAVE_INTERVAL = 25

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

SPECIES_LIST = [
    "Bombus affinis", "Bombus ashtoni", "Bombus bimaculatus",
    "Bombus borealis", "Bombus citrinus", "Bombus fervidus",
    "Bombus flavidus", "Bombus griseocollis", "Bombus impatiens",
    "Bombus pensylvanicus", "Bombus perplexus", "Bombus rufocinctus",
    "Bombus sandersoni", "Bombus ternarius", "Bombus terricola",
    "Bombus vagans",
]

MORPHOLOGICAL_FEATURES = [
    ("legs_appendages", "Legs/Appendages"),
    ("wing_venation_texture", "Wing Venation/Texture"),
    ("head_antennae", "Head/Antennae"),
    ("abdomen_banding", "Abdomen Banding"),
    ("thorax_coloration", "Thorax Coloration"),
]

# Pre-committed pass thresholds; not tuned on expert labels.
# mean_morph >= 4.0 mirrors v1's "Good or better" strict-rule anchor.
MEAN_MORPH_THRESHOLD = 4.0


SYSTEM_PROMPT = """\
You are an entomologist reviewing synthetic bumblebee images for a species \
classification training dataset. Evaluate each image using the two-stage \
protocol below.

Think step-by-step through each criterion before assigning scores.

═══ STAGE 1: BLIND IDENTIFICATION ═══

Identify the specimen from visual evidence alone. The dataset contains \
16 Bombus species — choose from:

  Family: Apidae
  Genus: Bombus
  Species (pick one, "Unknown", or "No match"):
    Bombus affinis, Bombus ashtoni, Bombus bimaculatus, Bombus borealis,
    Bombus citrinus, Bombus fervidus, Bombus flavidus, Bombus griseocollis,
    Bombus impatiens, Bombus pensylvanicus, Bombus perplexus,
    Bombus rufocinctus, Bombus sandersoni, Bombus ternarius,
    Bombus terricola, Bombus vagans

If too ambiguous for species-level ID, set species="Unknown".
If not a Bombus bumble bee at all, set species="No match".

═══ STAGE 2: DETAILED EVALUATION ═══

── 2A. Per-feature visibility and morphological fidelity ──

For each of the five features below, first pick a VISIBILITY state, then \
score only if the feature is "visible".

Visibility states (critical — read carefully):
  • visible         — the feature is in frame AND the viewpoint exposes \
its diagnostic surface well enough for a trained entomologist to grade its \
fidelity on a 1-5 scale.
  • not_assessable  — the feature is technically in frame (some pixels \
present) BUT the pose/viewpoint does NOT expose the diagnostic surface \
for this species. Example: a frontal (head-on) shot where the abdomen \
appears only as a small silhouette behind the face — the dorsal tergite \
pattern that defines the species is not in view, so you cannot fairly \
grade abdomen_banding from this angle. Use this state rather than guessing \
a low score. This is the correct state for body parts that are present \
but rendered at an angle that hides their diagnostic content.
  • not_visible     — the feature is out of frame, fully occluded by \
flowers/leaves, cropped off, or destroyed by image artefacts. Nothing to \
grade.

Features:
  • Legs/Appendages — correct count (6), proportions, hair/spurs
  • Wing Venation/Texture — wing shape, transparency, vein pattern
  • Head/Antennae — antenna segmentation, eye shape, mouthparts
  • Abdomen Banding — tergite colour pattern matching species description \
(critical for Bombus species ID)
  • Thorax Coloration — pile colour/pattern matching species description \
(critical for Bombus species ID)

Score anchors (apply only to features marked "visible"):
  1 = Poor: feature has clear errors (wrong count, wrong colour, \
distorted shape, anatomically impossible)
  2 = Below fair: feature is present but with notable inaccuracies
  3 = Fair: feature is roughly correct with only minor imperfections that \
would not mislead a classifier
  4 = Good: feature is accurate with only subtle issues visible on close \
inspection
  5 = Excellent: feature is photorealistic and matches the species description

CRITICAL RULE: Do not penalise a feature for being off-angle. If you \
cannot see the diagnostic surface, use "not_assessable" rather than \
scoring low. Scores 1-2 should only be given to features you CAN see and \
which show clear errors.

── 2B. Diagnostic Completeness ──

Report the highest taxonomic level at which this image could support \
identification:
  "species" — diagnostic surfaces (typically dorsal thorax + tergites) \
are assessable and consistent with the target species.
  "genus"   — Bombus is confidently identifiable but species-diagnostic \
surfaces are not_assessable or not_visible from this viewpoint.
  "family"  — Apidae identifiable, but genus not confident.
  "none"    — not identifiable at all.

If critical features (abdomen_banding, thorax_coloration) are both either \
not_assessable or not_visible, the correct level is usually "genus", \
because the species-diagnostic surface simply isn't in view.

── 2C. Image Quality Failures ══

Select all that apply:
  • quality_no_failure:   No Failure
  • blurry_artifacts:     Blurry/Visual Artifacts
  • background_bleed:     Background Bleed/Contamination
  • flower_unrealistic:   Unrealistic Flower Geometry
  • repetitive_pattern:   Repetitive/Cloned Patterns
  • quality_other:        Other (describe)

═══ PASS/FAIL RULE ═══

overall_pass = true if ALL of:
  1. blind_identification.species is NOT "No match".
  2. diag in {"species", "genus"}.
  3. mean of visible-feature scores is >= 4.0.
  4. image_quality.repetitive_pattern is false.

Otherwise overall_pass = false.

Features marked not_assessable or not_visible are excluded from the \
mean-score calculation (do not score them as low — mark them correctly and \
let the rule skip them).

═══ CALIBRATION GUIDANCE ═══

These are AI-generated training images. Apply the standard of a working \
entomologist reviewing field photographs. A head-on shot of B. ashtoni \
with nothing visibly wrong but tergites out of view should end up with \
abdomen_banding = not_assessable, diag = "genus", and pass if the \
remaining visible features score well. A dorsal shot with a wrong-coloured \
thorax should end up with thorax_coloration = visible, score = 1 or 2, \
and fail on mean_morph."""


class BlindIdentification(BaseModel):
    family: str = Field(description="Taxonomic family")
    genus: str = Field(description="Taxonomic genus")
    species: str = Field(
        description="Species name from the 16-species list, 'Unknown', or 'No match'"
    )
    matches_target: bool = Field(
        description="Whether blind ID matches target species at genus+species level"
    )


class MorphFeatureScore(BaseModel):
    visibility: Literal["visible", "not_assessable", "not_visible"] = Field(
        description="'visible' = diagnostic surface in frame and assessable; "
                    "'not_assessable' = in frame but viewpoint does not expose "
                    "diagnostic surface; 'not_visible' = out of frame / occluded."
    )
    score: Optional[int] = Field(
        default=None, ge=1, le=5,
        description="1-5 score; required only when visibility == 'visible', "
                    "otherwise null.",
    )
    notes: str = Field(description="Brief assessment.")


class MorphologicalFidelity(BaseModel):
    legs_appendages: MorphFeatureScore
    wing_venation_texture: MorphFeatureScore
    head_antennae: MorphFeatureScore
    abdomen_banding: MorphFeatureScore
    thorax_coloration: MorphFeatureScore


class DiagnosticCompleteness(BaseModel):
    level: Literal["species", "genus", "family", "none"] = Field(
        description="Highest taxonomic level the image supports."
    )


class ImageQualityFailures(BaseModel):
    quality_no_failure: bool = Field(description="True if no image quality issues")
    blurry_artifacts: bool = Field(default=False)
    background_bleed: bool = Field(default=False)
    flower_unrealistic: bool = Field(default=False)
    repetitive_pattern: bool = Field(default=False)
    quality_other: Optional[str] = Field(default=None)


class JudgeVerdictV2(BaseModel):
    chain_of_thought: str = Field(
        description="Step-by-step reasoning before assigning scores."
    )
    blind_identification: BlindIdentification
    morphological_fidelity: MorphologicalFidelity
    diagnostic_completeness: DiagnosticCompleteness
    image_quality: ImageQualityFailures
    overall_pass: bool = Field(
        description="True only if all pass/fail rules are satisfied."
    )
    summary: str = Field(description="One-sentence recommendation.")


def encode_image_b64(path: Path) -> str:
    mime = MIME_TYPES.get(path.suffix.lower(), "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


_CASTE_FILENAME_RE = re.compile(r"[^:]+::(\d+)::([^:]+)::")


def _extract_caste_from_filename(filename: str, species: str) -> tuple[str | None, str | None]:
    m = _CASTE_FILENAME_RE.match(filename)
    if not m:
        return None, None
    caste_name = m.group(2)
    sp_data = SPECIES_DATA.get(species, {})
    caste_opts = sp_data.get("caste_options", {})
    return caste_name, caste_opts.get(caste_name)


def _validate_blind_id(verdict: dict, species: str) -> dict:
    morph = SPECIES_MORPHOLOGY.get(species, {})
    config_name = morph.get("species_name", species.replace("_", " "))
    blind_species = verdict["blind_identification"]["species"].lower().strip()
    acceptable = {config_name.lower().strip(), species.replace("_", " ").lower()}
    if "(" in config_name:
        base = config_name.split("(")[0].strip().lower()
        acceptable.add(base)
        inner = config_name.split("(")[1].rstrip(")").strip().lower()
        for part in inner.replace("inc.", "").replace("=", "").split(","):
            part = part.strip()
            if part:
                genus = config_name.split()[0].lower()
                acceptable.add(f"{genus} {part}")
    verdict["blind_identification"]["matches_target"] = (blind_species in acceptable)
    return verdict


def _compute_mean_morph(verdict: dict) -> float:
    """Mean of scores across features where visibility == 'visible'.

    Raises ValueError if the model returns visibility == 'visible' but no
    score — that is a schema contract violation the caller should surface as
    an `error` row rather than silently pull the mean down.
    """
    morph = verdict.get("morphological_fidelity", {})
    scores = []
    for feat_id, _ in MORPHOLOGICAL_FEATURES:
        feat = morph.get(feat_id, {}) or {}
        vis = feat.get("visibility", "visible")
        if vis == "visible":
            if feat.get("score") is None:
                raise ValueError(
                    f"{feat_id}: visibility='visible' but score is None "
                    "(schema violation from LLM)."
                )
            scores.append(feat["score"])
    return sum(scores) / len(scores) if scores else 0.0


def _enforce_pass_rule(verdict: dict) -> dict:
    """Override overall_pass deterministically from the anchor rule."""
    bi = verdict["blind_identification"]
    diag = verdict["diagnostic_completeness"]["level"]
    mm = _compute_mean_morph(verdict)
    no_rep = not verdict["image_quality"].get("repetitive_pattern", False)
    verdict["mean_morph_visible"] = mm
    verdict["overall_pass"] = (
        bi["species"] != "No match"
        and diag in ("species", "genus")
        and mm >= MEAN_MORPH_THRESHOLD
        and no_rep
    )
    return verdict


def _backfill_legacy_fields(verdict: dict) -> dict:
    """Add legacy `not_visible` boolean per feature so downstream code
    (scripts/assemble_dataset.py, plotting scripts) keeps working.
    """
    morph = verdict.get("morphological_fidelity", {})
    for feat_id, _ in MORPHOLOGICAL_FEATURES:
        feat = morph.get(feat_id, {}) or {}
        vis = feat.get("visibility", "visible")
        feat["not_visible"] = (vis != "visible")
        morph[feat_id] = feat
    return verdict


def judge_single_image(client: OpenAI, image_path: Path, species: str) -> dict:
    morph = SPECIES_MORPHOLOGY.get(species, {})
    species_name = morph.get("species_name", species.replace("_", " "))
    description = morph.get("morphological_description", "No description available.")

    caste_name, caste_desc = _extract_caste_from_filename(image_path.name, species)
    caste_section = (
        f"Expected caste: {caste_name}. {caste_desc}\n"
        if caste_name and caste_desc else ""
    )

    user_text = (
        f"Target Species: {species_name}\n"
        f"Expected Traits: {description}\n"
        f"{caste_section}"
        "Evaluate the attached synthetic image. For each morphological "
        "feature, pick visibility FIRST, then score only if 'visible'. "
        "Use 'not_assessable' rather than guessing a low score when the "
        "viewpoint does not expose the diagnostic surface."
    )

    data_uri = encode_image_b64(image_path)
    completion = client.chat.completions.parse(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                ],
            },
        ],
        response_format=JudgeVerdictV2,
    )
    verdict = completion.choices[0].message.parsed.model_dump()
    verdict = _validate_blind_id(verdict, species)
    verdict = _backfill_legacy_fields(verdict)
    verdict = _enforce_pass_rule(verdict)
    return verdict


def main():
    ap = argparse.ArgumentParser(description="LLM judge v2 (tri-state visibility).")
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--files-list", type=Path, required=True,
                    help="Text file with one image filename per line.")
    ap.add_argument("--species", type=str, required=True)
    ap.add_argument("--output-name", type=str, default="results.json")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / args.output_name

    existing = []
    if results_path.exists():
        existing = json.loads(results_path.read_text()).get("results", [])
    done = {r["file"] for r in existing}

    files = [
        (args.image_dir / args.species / ln.strip()).resolve()
        for ln in args.files_list.read_text().splitlines()
        if ln.strip()
    ]
    client = OpenAI()

    results = list(existing)
    for idx, fp in enumerate(files, 1):
        if fp.name in done:
            continue
        print(f"[{idx}/{len(files)}] {fp.name} ...", end=" ", flush=True)
        try:
            verdict = judge_single_image(client, fp, args.species)
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"file": fp.name, "species": args.species,
                            "error": str(e), "overall_pass": False})
            continue
        # Count visibility states for console readability
        vis_counts = {"v": 0, "na": 0, "nv": 0}
        for feat_id, _ in MORPHOLOGICAL_FEATURES:
            v = verdict["morphological_fidelity"][feat_id].get("visibility", "visible")
            if v == "visible": vis_counts["v"] += 1
            elif v == "not_assessable": vis_counts["na"] += 1
            else: vis_counts["nv"] += 1
        mm = verdict.get("mean_morph_visible", 0.0)
        diag = verdict["diagnostic_completeness"]["level"]
        status = "PASS" if verdict["overall_pass"] else "FAIL"
        print(f"{status}  (vis={vis_counts['v']}/na={vis_counts['na']}/nv={vis_counts['nv']} "
              f"mm={mm:.1f} diag={diag})  {verdict['summary']}")
        results.append({"file": fp.name, "species": args.species, **verdict})
        if idx % SAVE_INTERVAL == 0:
            results_path.write_text(json.dumps({
                "model": MODEL,
                "mean_morph_threshold": MEAN_MORPH_THRESHOLD,
                "results": results,
            }, indent=2))

    results_path.write_text(json.dumps({
        "model": MODEL,
        "mean_morph_threshold": MEAN_MORPH_THRESHOLD,
        "results": results,
    }, indent=2))
    passed = sum(1 for r in results if r.get("overall_pass"))
    print(f"\nDone. {passed}/{len(results)} passed. Wrote {results_path}")


if __name__ == "__main__":
    main()
