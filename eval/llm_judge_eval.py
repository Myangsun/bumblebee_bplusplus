#!/usr/bin/env python3
"""
LLM-as-a-judge evaluation for synthetic bumblebee images.

Applies multi-dimensional rubric scoring to generated images before they
enter the training dataset, and can also evaluate classification outputs.

Rubric dimensions (pass/fail + reasoning):
  1. Plausibility      — Does the image look like a real camera-trap photograph?
  2. Morphology         — Are species-specific traits correctly rendered?
  3. Environment        — Is the background a natural outdoor habitat (not studio)?
  4. Artefact-free      — No cloned patterns, repeated geometry, or rendering glitches?
  5. Pose consistency   — Does the view angle match the requested perspective?

Workflow:
  python llm_judge_eval.py run
  python llm_judge_eval.py run --image-dir edit_results --out judge_results.json
  python llm_judge_eval.py run --threshold 0.8

Reference:
  https://ai.pydantic.dev/evals/evaluators/llm-judge/
"""

import base64
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
DEFAULT_IMAGE_DIRS = [
    SCRIPT_DIR / "batch_results",
    SCRIPT_DIR / "edit_results",
]
DEFAULT_OUTPUT = SCRIPT_DIR / "judge_results.json"

MODEL = "gpt-4o"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Species metadata (for morphology checks) ─────────────────────────────────

SPECIES_META = {
    "Bombus_sandersoni": {
        "species_name": "Bombus sandersoni",
        "morphological_description": (
            "Bumblebee with yellow hair on thorax, yellow band on first abdominal "
            "segment, and black abdomen with mixed yellow hairs on the sides"
        ),
    },
    "Bombus_ashtoni": {
        "species_name": "Bombus ashtoni",
        "morphological_description": (
            "Cuckoo bumblebee with sparse pale-yellow hair on thorax, largely black "
            "abdomen with pale-yellow hair patches on anterior segments, and darker "
            "more robust exoskeleton than typical bumblebees"
        ),
    },
}

VARIATIONS = {
    "lateral": "lateral (side view, wings folded over abdomen)",
    "dorsal": "dorsal (top-down view, wings folded over abdomen)",
    "three-quarter_anterior": "three-quarter anterior (front-angled, wings slightly spread)",
    "three-quarter_posterior": "three-quarter posterior (rear-angled, wings folded)",
    "frontal": "frontal (head-on view, wings slightly spread)",
}


# ── Structured output schema ─────────────────────────────────────────────────

class RubricDimension(BaseModel):
    """A single rubric dimension evaluation."""
    passed: bool = Field(description="Whether this dimension passes the quality check.")
    score: float = Field(
        ge=0.0, le=1.0,
        description="Quality score from 0.0 (terrible) to 1.0 (excellent).",
    )
    reason: str = Field(description="Specific explanation for the score.")


class JudgeVerdict(BaseModel):
    """Full multi-dimensional judge verdict for a synthetic image."""

    plausibility: RubricDimension = Field(
        description=(
            "Does the image look like a real camera-trap photograph? "
            "Check for realistic lighting, depth of field, natural composition. "
            "Fail if it looks like a studio render, 3D model, or digital painting."
        ),
    )
    morphology: RubricDimension = Field(
        description=(
            "Are the species-specific morphological traits correctly rendered? "
            "Check hair/pile colour, banding patterns, body proportions, "
            "number of legs, wing structure. Fail if anatomy is wrong."
        ),
    )
    environment: RubricDimension = Field(
        description=(
            "Is the background a natural outdoor habitat with flowers? "
            "Check for realistic vegetation, varied flower arrangement, "
            "natural depth. Fail if background is blank, studio, or artificial."
        ),
    )
    artefact_free: RubricDimension = Field(
        description=(
            "Is the image free of generation artefacts? "
            "Check for cloned/repeated patterns, distorted geometry, "
            "unnatural textures, blending seams, extra limbs. "
            "Fail if visible artefacts would be noticed by a domain expert."
        ),
    )
    pose_consistency: RubricDimension = Field(
        description=(
            "Does the insect's pose and camera angle match the requested view? "
            "Fail if the perspective is clearly wrong (e.g. dorsal when lateral was requested)."
        ),
    )
    overall_pass: bool = Field(
        description="True only if ALL dimensions pass. A single failure means overall fail.",
    )
    overall_score: float = Field(
        ge=0.0, le=1.0,
        description="Mean of all dimension scores.",
    )
    summary: str = Field(
        description="One-sentence overall assessment and recommendation (keep / regenerate).",
    )


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert entomologist and image-quality reviewer for a synthetic training dataset.

Your task is to evaluate a computer-generated bumblebee image against a strict rubric.
The image was generated to augment a real dataset for species classification training,
so it must be indistinguishable from real camera-trap photographs and morphologically accurate.

Evaluate each rubric dimension independently. Be strict — a single obvious flaw in any
dimension should cause that dimension to fail. Training data quality directly impacts
classifier performance.

You will be told:
- The target species and its key morphological traits
- The requested view angle / pose

Score each dimension 0.0–1.0 and give a concrete reason.
Set overall_pass = True only if every dimension passes."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def encode_image_b64(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def collect_images(*dirs: Path) -> list[Path]:
    images = []
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(p)
    return images


def parse_filename(filename: str) -> tuple[str | None, str | None]:
    """
    Parse filename convention: Bombus_sandersoni-lateral_0.png
    Returns (species_slug, view_label) or (None, None).
    """
    stem = Path(filename).stem
    for slug in SPECIES_META:
        if stem.startswith(slug):
            rest = stem[len(slug):]
            if rest.startswith("-"):
                # e.g. "-lateral_0" → "lateral"
                view_part = rest[1:]
                # Strip trailing _N index
                for v_key in VARIATIONS:
                    if view_part.startswith(v_key):
                        return slug, v_key
            return slug, None
    return None, None


# ── Core judge ───────────────────────────────────────────────────────────────

def judge_image(client: OpenAI, image_path: Path) -> dict:
    """Run the full rubric evaluation on a single image."""
    species_slug, view_key = parse_filename(image_path.name)

    # Build context about what was requested
    context_parts = []
    if species_slug and species_slug in SPECIES_META:
        meta = SPECIES_META[species_slug]
        context_parts.append(f"Target species: {meta['species_name']}")
        context_parts.append(f"Key morphological traits: {meta['morphological_description']}")
    else:
        context_parts.append("Target species: unknown (evaluate general bumblebee plausibility)")

    if view_key and view_key in VARIATIONS:
        context_parts.append(f"Requested view angle: {VARIATIONS[view_key]}")
    else:
        context_parts.append("Requested view angle: not specified")

    context = "\n".join(context_parts)
    data_uri = encode_image_b64(image_path)

    completion = client.chat.completions.parse(
        model=MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Evaluate this synthetic bumblebee image.\n\n{context}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri, "detail": "high"},
                    },
                ],
            },
        ],
        response_format=JudgeVerdict,
    )

    verdict: JudgeVerdict = completion.choices[0].message.parsed
    return verdict.model_dump()


# ── Evaluation run ───────────────────────────────────────────────────────────

def run_evaluation(image_dirs: list[Path], output_path: Path, threshold: float = 0.7):
    """Judge all images and produce a quality report."""
    client = OpenAI()
    images = collect_images(*image_dirs)

    if not images:
        print(f"No images found in {[str(d) for d in image_dirs]}")
        return

    print(f"Found {len(images)} images to evaluate\n")

    results = []
    pass_count = 0
    dim_names = ["plausibility", "morphology", "environment", "artefact_free", "pose_consistency"]
    dim_scores = {d: [] for d in dim_names}

    for img_path in images:
        print(f"  Judging {img_path.name} …", end=" ", flush=True)
        verdict = judge_image(client, img_path)

        passed = verdict["overall_pass"]
        score = verdict["overall_score"]
        if passed:
            pass_count += 1

        status = "PASS" if passed else "FAIL"
        print(f"{status}  (score={score:.2f})  {verdict['summary']}")

        for dim in dim_names:
            dim_scores[dim].append(verdict[dim]["score"])

        results.append({
            "file": img_path.name,
            "source_dir": img_path.parent.name,
            "overall_pass": passed,
            "overall_score": score,
            "summary": verdict["summary"],
            "dimensions": {
                dim: {
                    "passed": verdict[dim]["passed"],
                    "score": verdict[dim]["score"],
                    "reason": verdict[dim]["reason"],
                }
                for dim in dim_names
            },
        })

    # ── Summary report ──
    n = len(results)
    print(f"\n{'─' * 60}")
    print(f"Total images evaluated: {n}")
    print(f"Overall pass rate: {pass_count}/{n} = {pass_count / n:.1%}")
    print(f"Pass threshold: {threshold}")

    above_threshold = sum(1 for r in results if r["overall_score"] >= threshold)
    print(f"Images above score threshold ({threshold}): {above_threshold}/{n}")

    print(f"\nPer-dimension mean scores:")
    for dim in dim_names:
        scores = dim_scores[dim]
        mean = sum(scores) / len(scores) if scores else 0
        pass_rate = sum(1 for s in scores if s >= 0.5) / len(scores) if scores else 0
        print(f"  {dim:22s}: mean={mean:.2f}  pass_rate={pass_rate:.1%}")

    # Per-source directory breakdown
    source_dirs = sorted(set(r["source_dir"] for r in results))
    if len(source_dirs) > 1:
        print(f"\nPer-source breakdown:")
        for sd in source_dirs:
            subset = [r for r in results if r["source_dir"] == sd]
            sd_pass = sum(1 for r in subset if r["overall_pass"])
            sd_mean = sum(r["overall_score"] for r in subset) / len(subset)
            print(f"  {sd}: {sd_pass}/{len(subset)} pass, mean_score={sd_mean:.2f}")

    # Flag worst images for regeneration
    failed = [r for r in results if not r["overall_pass"]]
    if failed:
        print(f"\nImages flagged for regeneration ({len(failed)}):")
        for r in sorted(failed, key=lambda x: x["overall_score"]):
            failing_dims = [
                dim for dim in dim_names if not r["dimensions"][dim]["passed"]
            ]
            print(f"  {r['file']} (score={r['overall_score']:.2f}) — failing: {', '.join(failing_dims)}")

    # ── Save ──
    report = {
        "model": MODEL,
        "threshold": threshold,
        "total_images": n,
        "pass_rate": pass_count / n if n else 0,
        "dimension_means": {
            dim: sum(dim_scores[dim]) / len(dim_scores[dim]) if dim_scores[dim] else 0
            for dim in dim_names
        },
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved to {output_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge quality evaluation for synthetic bumblebee images"
    )
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Evaluate all images against the quality rubric")
    p_run.add_argument(
        "--image-dir",
        type=Path,
        action="append",
        default=None,
        help="Directory of images to evaluate (repeatable; default: batch_results + edit_results)",
    )
    p_run.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON path (default: judge_results.json)",
    )
    p_run.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Minimum overall_score to consider an image acceptable (default: 0.7)",
    )

    args = parser.parse_args()

    if args.cmd == "run":
        dirs = args.image_dir if args.image_dir else DEFAULT_IMAGE_DIRS
        run_evaluation(dirs, args.out, threshold=args.threshold)
    else:
        parser.print_help()
