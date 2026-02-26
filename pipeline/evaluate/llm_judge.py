#!/usr/bin/env python3
"""
LLM-as-a-judge evaluation for synthetic bumblebee images.

Applies a multi-dimensional rubric (plausibility, morphology, environment,
artefact-free, pose consistency) to generated images using GPT-4o structured output.

Importable API
--------------
    from pipeline.evaluate.llm_judge import run
    run(image_dirs=["eval/batch_results"], output="judge_results.json")

CLI
---
    python pipeline/evaluate/llm_judge.py run
    python pipeline/evaluate/llm_judge.py run --image-dir eval/batch_results --threshold 0.8
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path
from typing import List, Optional

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
DEFAULT_IMAGE_DIRS = [SCRIPT_DIR / "batch_results", SCRIPT_DIR / "edit_results"]
DEFAULT_OUTPUT = SCRIPT_DIR / "judge_results.json"

MODEL = "gpt-4o"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Species metadata ──────────────────────────────────────────────────────────

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


# ── Structured output schema ──────────────────────────────────────────────────


class RubricDimension(BaseModel):
    passed: bool = Field(description="Whether this dimension passes the quality check.")
    score: float = Field(ge=0.0, le=1.0, description="Quality score 0.0–1.0.")
    reason: str = Field(description="Specific explanation for the score.")


class JudgeVerdict(BaseModel):
    plausibility: RubricDimension = Field(
        description="Does the image look like a real camera-trap photograph? "
                    "Fail if it looks like a studio render or 3D model."
    )
    morphology: RubricDimension = Field(
        description="Are species-specific traits correctly rendered? "
                    "Check hair/pile colour, banding, body proportions, legs, wings."
    )
    environment: RubricDimension = Field(
        description="Is the background a natural outdoor habitat with flowers? "
                    "Fail if background is blank, studio, or artificial."
    )
    artefact_free: RubricDimension = Field(
        description="Is the image free of generation artefacts? "
                    "Fail if cloned patterns, distorted geometry, or extra limbs are visible."
    )
    pose_consistency: RubricDimension = Field(
        description="Does the insect pose match the requested view angle?"
    )
    overall_pass: bool = Field(description="True only if ALL dimensions pass.")
    overall_score: float = Field(ge=0.0, le=1.0, description="Mean of all dimension scores.")
    summary: str = Field(description="One-sentence assessment and recommendation (keep / regenerate).")


SYSTEM_PROMPT = """\
You are an expert entomologist and image-quality reviewer for a synthetic training dataset.

Your task is to evaluate a computer-generated bumblebee image against a strict rubric.
The image was generated to augment a real dataset for species classification training,
so it must be indistinguishable from real camera-trap photographs and morphologically accurate.

Evaluate each rubric dimension independently. Be strict — a single obvious flaw should fail
that dimension. Training data quality directly impacts classifier performance.

Score each dimension 0.0–1.0 and give a concrete reason.
Set overall_pass = True only if every dimension passes."""


# ── Helpers ───────────────────────────────────────────────────────────────────


def encode_image_b64(path: Path) -> str:
    suffix = path.suffix.lower()
    mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp"}.get(suffix, "image/png")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"


def collect_images(*dirs: Path) -> List[Path]:
    images = []
    for d in dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(p)
    return images


def parse_filename(filename: str):
    """Parse species slug and view from filename like 'Bombus_sandersoni-lateral_0.png'."""
    stem = Path(filename).stem
    for slug in SPECIES_META:
        if stem.startswith(slug):
            rest = stem[len(slug):]
            if rest.startswith("-"):
                view_part = rest[1:]
                for v_key in VARIATIONS:
                    if view_part.startswith(v_key):
                        return slug, v_key
            return slug, None
    return None, None


# ── Core judge ────────────────────────────────────────────────────────────────


def judge_image(client: OpenAI, image_path: Path) -> dict:
    """Run the full rubric evaluation on a single image."""
    species_slug, view_key = parse_filename(image_path.name)

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
                    {"type": "text", "text": f"Evaluate this synthetic bumblebee image.\n\n{context}"},
                    {"type": "image_url", "image_url": {"url": data_uri, "detail": "high"}},
                ],
            },
        ],
        response_format=JudgeVerdict,
    )

    verdict: JudgeVerdict = completion.choices[0].message.parsed
    return verdict.model_dump()


# ── Evaluation run ────────────────────────────────────────────────────────────


def run(
    image_dirs: Optional[List[Path | str]] = None,
    output: Path | str = DEFAULT_OUTPUT,
    threshold: float = 0.7,
) -> dict:
    """
    Judge all images in the given directories against the quality rubric.

    Args:
        image_dirs: Directories containing images to evaluate.
                    Defaults to eval/batch_results + eval/edit_results.
        output: Path for the JSON report.
        threshold: Minimum overall_score to consider an image acceptable.

    Returns:
        Full evaluation report dict.
    """
    client = OpenAI()
    dirs = [Path(d) for d in image_dirs] if image_dirs else DEFAULT_IMAGE_DIRS
    output = Path(output)
    images = collect_images(*dirs)

    if not images:
        print(f"No images found in {[str(d) for d in dirs]}")
        return {}

    print(f"Found {len(images)} images to evaluate\n")

    DIM_NAMES = ["plausibility", "morphology", "environment", "artefact_free", "pose_consistency"]
    results = []
    pass_count = 0
    dim_scores = {d: [] for d in DIM_NAMES}

    for img_path in images:
        print(f"  Judging {img_path.name} ...", end=" ", flush=True)
        verdict = judge_image(client, img_path)

        passed = verdict["overall_pass"]
        score = verdict["overall_score"]
        if passed:
            pass_count += 1

        status = "PASS" if passed else "FAIL"
        print(f"{status}  (score={score:.2f})  {verdict['summary']}")

        for dim in DIM_NAMES:
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
                for dim in DIM_NAMES
            },
        })

    n = len(results)
    print(f"\n{'─' * 60}")
    print(f"Total evaluated: {n}")
    print(f"Pass rate: {pass_count}/{n} = {pass_count / n:.1%}")
    above = sum(1 for r in results if r["overall_score"] >= threshold)
    print(f"Above threshold ({threshold}): {above}/{n}")

    print("\nPer-dimension mean scores:")
    for dim in DIM_NAMES:
        scores = dim_scores[dim]
        mean = sum(scores) / len(scores) if scores else 0
        print(f"  {dim:22s}: {mean:.2f}")

    failed = [r for r in results if not r["overall_pass"]]
    if failed:
        print(f"\nFlagged for regeneration ({len(failed)}):")
        for r in sorted(failed, key=lambda x: x["overall_score"]):
            failing_dims = [d for d in DIM_NAMES if not r["dimensions"][d]["passed"]]
            print(f"  {r['file']} (score={r['overall_score']:.2f}) — failing: {', '.join(failing_dims)}")

    report = {
        "model": MODEL,
        "threshold": threshold,
        "total_images": n,
        "pass_rate": pass_count / n if n else 0,
        "dimension_means": {
            dim: sum(dim_scores[dim]) / len(dim_scores[dim]) if dim_scores[dim] else 0
            for dim in DIM_NAMES
        },
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"\nFull report saved to {output}")
    return report


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="LLM-as-a-judge quality evaluation for synthetic bumblebee images"
    )
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Evaluate images against the quality rubric")
    p_run.add_argument(
        "--image-dir", type=Path, action="append", default=None,
        help="Image directory to evaluate (repeatable; default: batch_results + edit_results)",
    )
    p_run.add_argument("--out", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path")
    p_run.add_argument("--threshold", type=float, default=0.7,
                       help="Min score to consider acceptable (default: 0.7)")

    args = parser.parse_args()

    if args.cmd == "run":
        dirs = args.image_dir if args.image_dir else None
        run(image_dirs=dirs, output=args.out, threshold=args.threshold)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
