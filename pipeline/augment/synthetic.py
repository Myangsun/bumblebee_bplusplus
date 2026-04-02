#!/usr/bin/env python3
"""
Generate synthetic bumblebee images using OpenAI gpt-image-1.5 edit API.

Uses reference images + morphological prompts with angle/environment variations.
Supports both direct (synchronous) and batch (async) workflows.

Importable API
--------------
    from pipeline.augment.synthetic import run
    run(species=["Bombus_ashtoni"], count=500)

CLI (step-by-step)
------------------
    python pipeline/augment/synthetic.py upload
    python pipeline/augment/synthetic.py build --species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus --count 500
    python pipeline/augment/synthetic.py submit
    python pipeline/augment/synthetic.py status --poll
    python pipeline/augment/synthetic.py download

CLI (all-in-one)
----------------
    python pipeline/augment/synthetic.py run --species Bombus_ashtoni Bombus_sandersoni Bombus_flavidus --count 5
"""

from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import time
from pathlib import Path
from typing import Optional

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dotenv import load_dotenv
from openai import OpenAI

from pipeline.config import PROJECT_ROOT, RESULTS_DIR, load_species_config

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL = "gpt-image-1.5"
SIZE = "1024x1024"
QUALITY = "medium"
OUTPUT_FORMAT = "jpeg"

REFERENCES_DIR = PROJECT_ROOT / "references"
PROMPT_TEMPLATE_FILE = PROJECT_ROOT / "configs" / "prompt_template.txt"
OUTPUT_DIR = RESULTS_DIR / "synthetic_generation"

# Batch workflow artifact filenames (relative to output_dir)
_FILE_IDS_NAME = "file_ids.json"
_BATCH_INPUT_NAME = "batchinput_edit.jsonl"
_BATCH_ID_NAME = "batch_edit_id.txt"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# ── Load species config (single source of truth) ─────────────────────────────

_CONFIG = load_species_config()
ENVIRONMENTS = _CONFIG["environments"]
VARIATIONS = _CONFIG["variations"]
SPECIES_DATA = _CONFIG["species"]
DEFAULT_CASTE = _CONFIG.get("default_caste", "worker")

# Backward-compatible alias used by scripts/llm_judge.py
SPECIES_MORPHOLOGY = SPECIES_DATA


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_prompt_template() -> str:
    """Load the prompt template from configs/prompt_template.txt."""
    return PROMPT_TEMPLATE_FILE.read_text().strip()


def get_caste_description(species: str, caste: str | None = None) -> tuple[str, str]:
    """Return (caste_name, caste_description) for a species.

    If caste is None, randomly selects from available castes weighted
    towards workers/females (most common in field photos).
    """
    if species not in SPECIES_DATA or "caste_options" not in SPECIES_DATA[species]:
        c = caste or DEFAULT_CASTE
        return c, f"This is a {c}."

    options = SPECIES_DATA[species]["caste_options"]
    if caste is not None:
        if caste in options:
            return caste, options[caste]
        else:
            c = list(options.keys())[0]
            return c, options[c]

    # Random selection weighted towards workers/females (most common in field)
    weights_cfg = SPECIES_DATA[species].get("caste_weights", {})
    castes = list(options.keys())
    if weights_cfg:
        weights = [weights_cfg.get(c, 1) for c in castes]
    elif "worker" in castes:
        weights = [3 if c == "worker" else 1 for c in castes]
    elif "female" in castes:
        weights = [3 if c == "female" else 1 for c in castes]
    else:
        weights = [1] * len(castes)
    c = random.choices(castes, weights=weights, k=1)[0]
    return c, options[c]


def build_scale_instruction(species: str) -> str:
    """Build a species-specific scale instruction using proportional anchors."""
    info = SPECIES_DATA.get(species, {}).get("body_size")
    if info is None:
        body, label = "10-16", "medium"
    else:
        body = info["size_mm"]
        label = info["label"]
    return (
        f"This is a {label} bumblebee, body length {body} mm. "
        f"CRITICAL SIZE CONSTRAINT — the bee must be SMALL relative to the flowers. "
        f"Real proportions: a bumblebee body is shorter than a single daisy petal. "
        f"On white clover (~20 mm head), the bee's body is roughly the same width "
        f"as the flower head — never wider. "
        f"On aster or coneflower (~40 mm head), the flower head is 2–3× wider "
        f"than the bee. "
        f"On goldenrod, the bee clings to a single tiny sprig and is larger than "
        f"any individual floret but smaller than the full panicle. "
        f"On thistle, the bee perches on top and the spiny flower head is at least "
        f"as wide as the bee's body. "
        f"When in doubt, draw the bee SMALLER. An undersized bee in a large flower "
        f"looks like a real field photo. An oversized bee on a tiny flower looks "
        f"obviously synthetic."
    )


def _fill_template(
    template: str, species: str, variation: dict, environment: str
) -> tuple[str, str]:
    """Fill prompt template placeholders with species/variation/environment data.

    Returns (prompt, caste_name) so caste can be included in filenames.
    """
    species_info = SPECIES_DATA[species]
    caste_name, caste_desc = get_caste_description(species)
    scale = build_scale_instruction(species)

    prompt = template.format(
        species_name=species_info["species_name"],
        common_name=species_info.get("common_name", ""),
        caste_description=caste_desc,
        morphological_description=species_info["morphological_description"],
        view_angle=variation["view_angle"],
        wings_style=variation["wings_style"],
        environment_description=environment,
        scale_instruction=scale,
    )
    return prompt, caste_name


def _species_slug(species: str) -> str:
    """Normalize species name for file paths (already underscore-separated)."""
    return species.replace(" ", "_")


def _reference_images(species: str, references_dir: Path = REFERENCES_DIR) -> list[Path]:
    """Return all reference image paths for a species."""
    ref_dir = references_dir / _species_slug(species)
    if not ref_dir.is_dir():
        raise FileNotFoundError(
            f"Reference folder not found: {ref_dir}\n"
            f"Expected: references/{_species_slug(species)}/"
        )
    images = sorted(
        p for p in ref_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {ref_dir}")
    return images


# ── Step 1: Upload reference images ──────────────────────────────────────────


def upload_references(
    species_list: list[str] | None = None,
    references_dir: Path = REFERENCES_DIR,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, list[str]]:
    """
    Upload reference images to OpenAI Files API.

    Returns and saves a manifest: { species_slug: [file_id, ...] }
    """
    client = OpenAI()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, list[str]] = {}

    if species_list is None:
        species_list = list(SPECIES_DATA.keys())

    for species in species_list:
        slug = _species_slug(species)
        paths = _reference_images(species, references_dir)
        ids = []
        print(f"\n{species}  ({len(paths)} reference image(s))")
        for p in paths:
            with p.open("rb") as fh:
                uploaded = client.files.create(file=fh, purpose="user_data")
            print(f"  Uploaded {p.name}  ->  {uploaded.id}")
            ids.append(uploaded.id)
        manifest[slug] = ids

    file_ids_path = output_dir / _FILE_IDS_NAME
    file_ids_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nFile IDs saved to {file_ids_path}")
    return manifest


# ── Step 2: Build batch JSONL ─────────────────────────────────────────────────


def build_batch(
    species_list: list[str] | None = None,
    count: int = 500,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    """
    Build batchinput_edit.jsonl with ``count`` requests per species.

    Requires file_ids.json from a prior ``upload`` step.
    Randomly samples ENVIRONMENTS and cycles through VARIATIONS for diversity.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    file_ids_path = output_dir / _FILE_IDS_NAME
    if not file_ids_path.exists():
        raise FileNotFoundError(
            f"{file_ids_path} not found — run `upload` first."
        )
    manifest: dict[str, list[str]] = json.loads(file_ids_path.read_text())
    template = _load_prompt_template()

    if species_list is None:
        species_list = list(SPECIES_DATA.keys())

    lines = []
    for species in species_list:
        slug = _species_slug(species)
        file_ids = manifest.get(slug)
        if not file_ids:
            raise KeyError(f"No file IDs for {slug} in {file_ids_path}")

        print(f"\n{species}: building {count} requests")

        for i in range(count):
            variation = VARIATIONS[i % len(VARIATIONS)]
            environment = random.choice(ENVIRONMENTS)
            prompt, caste_name = _fill_template(template, species, variation, environment)
            label = variation["view_angle"].replace(" ", "_")
            custom_id = f"{slug}::{i:04d}::{caste_name}::{label}"

            record = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/images/edits",
                "body": {
                    "model": MODEL,
                    "prompt": prompt,
                    "n": 1,
                    "size": SIZE,
                    "quality": QUALITY,
                    "output_format": OUTPUT_FORMAT,
                    "input_fidelity": "high",
                    "images": [{"file_id": fid} for fid in file_ids],
                },
            }
            lines.append(json.dumps(record))

    batch_path = output_dir / _BATCH_INPUT_NAME
    batch_path.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {len(lines)} requests to {batch_path}")
    return batch_path


# ── Step 3: Submit batch ──────────────────────────────────────────────────────


def submit_batch(output_dir: Path = OUTPUT_DIR) -> str:
    """Upload the JSONL and create a batch. Returns batch ID."""
    client = OpenAI()
    batch_path = output_dir / _BATCH_INPUT_NAME
    if not batch_path.exists():
        raise FileNotFoundError(f"{batch_path} not found — run `build` first.")

    print("Uploading batch input file...")
    with batch_path.open("rb") as fh:
        file_obj = client.files.create(file=fh, purpose="batch")
    print(f"  File ID: {file_obj.id}")

    print("Creating batch...")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/images/edits",
        completion_window="24h",
        metadata={"description": "bumblebee synthetic generation"},
    )
    print(f"  Batch ID: {batch.id}  status: {batch.status}")

    batch_id_path = output_dir / _BATCH_ID_NAME
    batch_id_path.write_text(batch.id)
    print(f"Batch ID saved to {batch_id_path}")
    return batch.id


# ── Step 4: Poll status ──────────────────────────────────────────────────────


def _validate_batch_id(batch_id: str) -> str:
    clean = batch_id.strip()
    if not clean.startswith("batch_") or len(clean) > 64 or " " in clean:
        raise ValueError(f"Malformed batch ID: {clean!r}")
    return clean


def poll_batch(
    batch_id: str | None = None,
    poll: bool = True,
    interval: int = 60,
    output_dir: Path = OUTPUT_DIR,
):
    """Check batch status, optionally polling until done."""
    client = OpenAI()
    if batch_id is None:
        batch_id = (output_dir / _BATCH_ID_NAME).read_text().strip()
    batch_id = _validate_batch_id(batch_id)

    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"[{time.strftime('%H:%M:%S')}] status={batch.status}  "
            f"total={counts.total}  completed={counts.completed}  failed={counts.failed}"
        )
        if batch.status in ("completed", "failed", "expired", "cancelled"):
            if batch.output_file_id:
                print(f"Output file ID: {batch.output_file_id}")
            if batch.error_file_id:
                print(f"Error  file ID: {batch.error_file_id}")
            return batch
        if not poll:
            return batch
        time.sleep(interval)


# ── Step 5: Download results ─────────────────────────────────────────────────


def download_results(
    batch_id: str | None = None,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, int]:
    """
    Download batch results, decode images, and save per-species.

    Returns dict mapping species -> number of images saved.
    """
    client = OpenAI()
    if batch_id is None:
        batch_id = (output_dir / _BATCH_ID_NAME).read_text().strip()
    batch_id = _validate_batch_id(batch_id)

    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"Batch not yet complete (status={batch.status}). Aborting.")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    species_counts: dict[str, int] = {}
    ext = "jpg" if OUTPUT_FORMAT == "jpeg" else OUTPUT_FORMAT

    if batch.output_file_id:
        raw = client.files.content(batch.output_file_id).text
        raw_path = output_dir / "raw_edit_output.jsonl"
        raw_path.write_text(raw)
        print(f"Saved raw output -> {raw_path}")

        for line in raw.strip().splitlines():
            entry = json.loads(line)
            cid = entry["custom_id"]
            # Extract species from custom_id: "Bombus_ashtoni::0001::female::lateral"
            species = cid.split("::")[0]

            if entry.get("error"):
                print(f"  {cid}: ERROR - {entry['error']}")
                summary.append({"custom_id": cid, "error": entry["error"]})
                continue

            body = entry["response"]["body"]
            images = body.get("data", [])
            species_dir = output_dir / species
            species_dir.mkdir(parents=True, exist_ok=True)

            saved = []
            for j, img in enumerate(images):
                b64 = img.get("b64_json")
                if b64:
                    img_path = species_dir / f"{cid}_{j}.{ext}"
                    img_path.write_bytes(base64.b64decode(b64))
                    saved.append(str(img_path))
                    species_counts[species] = species_counts.get(species, 0) + 1
                elif img.get("url"):
                    saved.append(img["url"])
            summary.append({"custom_id": cid, "images": saved})

        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSummary -> {output_dir / 'summary.json'}")
    else:
        print("No output file (all requests may have failed).")

    if batch.error_file_id:
        err_raw = client.files.content(batch.error_file_id).text
        (output_dir / "errors.jsonl").write_text(err_raw)
        print(f"Errors  -> {output_dir / 'errors.jsonl'}")

    print("\nDownload complete:")
    for sp, cnt in sorted(species_counts.items()):
        print(f"  {sp}: {cnt} images -> {output_dir / sp}")

    return species_counts


# ── All-in-one orchestrator ──────────────────────────────────────────────────


def run(
    species: list[str] | None = None,
    count: int = 500,
    references_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    poll_interval: int = 60,
) -> dict[str, int]:
    """
    Run the full generation pipeline: upload -> build -> submit -> poll -> download.

    Processes one species per batch to stay under the OpenAI enqueued token
    limit (1M tokens for gpt-image-1.5).

    Args:
        species: Species to generate for (default: all in SPECIES_DATA).
        count: Images to generate per species.
        references_dir: Override reference images directory.
        output_dir: Override output directory.
        poll_interval: Seconds between batch status checks.

    Returns:
        Dict mapping species -> number of images saved.
    """
    if species is None:
        species = list(SPECIES_DATA.keys())
    if references_dir is None:
        references_dir = REFERENCES_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR

    print("=" * 70)
    print("SYNTHETIC IMAGE GENERATION")
    print(f"  Model: {MODEL}  Size: {SIZE}  Quality: {QUALITY}  Format: {OUTPUT_FORMAT}")
    print(f"  Species: {species}")
    print(f"  Count per species: {count}")
    print(f"  Total requests: {count * len(species)}")
    print("=" * 70)

    # Step 1: Upload all references once
    print("\n--- Step 1: Upload reference images ---")
    upload_references(species, references_dir, output_dir)

    # Steps 2-5: Process one species at a time to stay under token limit
    all_counts: dict[str, int] = {}
    for i, sp in enumerate(species, 1):
        print(f"\n{'=' * 70}")
        print(f"BATCH {i}/{len(species)}: {sp}  ({count} images)")
        print(f"{'=' * 70}")

        print(f"\n--- Build batch for {sp} ---")
        build_batch([sp], count, output_dir)

        print(f"\n--- Submit batch for {sp} ---")
        submit_batch(output_dir)

        print(f"\n--- Poll batch for {sp} ---")
        poll_batch(poll=True, interval=poll_interval, output_dir=output_dir)

        print(f"\n--- Download results for {sp} ---")
        sp_counts = download_results(output_dir=output_dir)
        all_counts.update(sp_counts)

    print(f"\n{'=' * 70}")
    print("ALL BATCHES COMPLETE")
    for sp, cnt in sorted(all_counts.items()):
        print(f"  {sp}: {cnt} images")
    print(f"  Total: {sum(all_counts.values())} images")
    return all_counts


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic bumblebee images with gpt-image-1.5 edit API"
    )
    sub = parser.add_subparsers(dest="cmd")

    # upload
    sub.add_parser("upload", help="Upload reference images -> file_ids.json")

    # build
    p_build = sub.add_parser("build", help="Build batchinput_edit.jsonl")
    p_build.add_argument("--species", nargs="+", help="Species to generate")
    p_build.add_argument("--count", type=int, default=500,
                         help="Images per species (default: 500)")

    # submit
    sub.add_parser("submit", help="Submit batch from batchinput_edit.jsonl")

    # status
    p_st = sub.add_parser("status", help="Check batch status")
    p_st.add_argument("--id", help="Batch ID (default: read batch_edit_id.txt)")
    p_st.add_argument("--poll", action="store_true")
    p_st.add_argument("--interval", type=int, default=60)

    # download
    p_dl = sub.add_parser("download", help="Download batch results")
    p_dl.add_argument("--id", help="Batch ID (default: read batch_edit_id.txt)")

    # run (all-in-one)
    p_run = sub.add_parser("run", help="Run full pipeline (upload -> download)")
    p_run.add_argument("--species", nargs="+", help="Species to generate")
    p_run.add_argument("--count", type=int, default=500,
                        help="Images per species (default: 500)")
    p_run.add_argument("--output-dir", type=Path, help="Override output directory")
    p_run.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between status checks (default: 60)")

    args = parser.parse_args()

    if args.cmd == "upload":
        upload_references()
    elif args.cmd == "build":
        build_batch(species_list=args.species, count=args.count)
    elif args.cmd == "submit":
        submit_batch()
    elif args.cmd == "status":
        poll_batch(
            batch_id=getattr(args, "id", None),
            poll=args.poll,
            interval=args.interval,
        )
    elif args.cmd == "download":
        download_results(batch_id=getattr(args, "id", None))
    elif args.cmd == "run":
        run(
            species=args.species,
            count=args.count,
            output_dir=args.output_dir,
            poll_interval=args.poll_interval,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
