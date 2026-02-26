#!/usr/bin/env python3
"""
Image generation via the OpenAI Images Edit API (/v1/images/edits).

Reference images are auto-discovered from:
  references/<Species_Name>/*.{jpg,jpeg,png,webp}

Workflow options:
  direct   — calls the edit API immediately for every species × variation
  build    — writes batchinput_edit.jsonl (requires pre-uploaded file IDs)
  upload   — uploads all reference images and saves a file_ids.json manifest
  submit   — submits a batch from batchinput_edit.jsonl (needs upload first)
  status   — polls a batch
  download — downloads batch results

Quick start (direct, no batch):
  python image_edit_generate.py direct

Batch workflow:
  python image_edit_generate.py upload
  python image_edit_generate.py build
  python image_edit_generate.py submit
  python image_edit_generate.py status --poll
  python image_edit_generate.py download
"""

import base64
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

TEMPLATE_FILE = Path(__file__).with_name("prompt.json")
REFERENCES_DIR = Path(__file__).with_name("references")
OUTPUT_DIR = Path(__file__).with_name("edit_results")
FILE_IDS_FILE = Path(__file__).with_name("file_ids.json")
BATCH_INPUT = Path(__file__).with_name("batchinput_edit.jsonl")
BATCH_ID_FILE = Path(__file__).with_name("batch_edit_id.txt")

MODEL = "gpt-image-1.5"  # or "gpt-image-1.5" / "gpt-image-1-mini"
SIZE = "1024x1024"
QUALITY = "medium"  # "low" | "medium" | "high"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

MIME_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

# ── Species dataset ────────────────────────────────────────────────────────────

VARIATIONS = [
    {"view_angle": "lateral", "wings_style": "folded over the abdomen at rest"},
    {"view_angle": "dorsal", "wings_style": "folded over the abdomen at rest"},
    {"view_angle": "three-quarter anterior", "wings_style": "slightly spread"},
    {
        "view_angle": "three-quarter posterior",
        "wings_style": "folded over the abdomen at rest",
    },
    {"view_angle": "frontal", "wings_style": "slightly spread"},
]

SPECIES_LIST = [
    {
        "species_name": "Bombus sandersoni",
        "morphological_description": (
            "Bumblebee with yellow hair on thorax, yellow band on first abdominal segment, "
            "and black abdomen with mixed yellow hairs on the sides"
        ),
    },
    {
        "species_name": "Bombus ashtoni",
        "morphological_description": (
            "Cuckoo bumblebee with sparse pale-yellow hair on thorax, largely black abdomen "
            "with pale-yellow hair patches on anterior segments, and darker more robust "
            "exoskeleton than typical bumblebees"
        ),
    },
]

# ── Helpers ────────────────────────────────────────────────────────────────────


def load_template() -> str:
    text = TEMPLATE_FILE.read_text().strip()
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    return text


def fill_template(template: str, species: dict) -> str:
    return (
        template.replace("species_name", species["species_name"])
        .replace("view_angle", species["view_angle"])
        .replace(
            "morphological_description (e.g. Bumblebee with velvety black hair and extensive bright red tail)",
            species["morphological_description"],
        )
        .replace("wings_style", species["wings_style"])
    )


def species_slug(species: dict) -> str:
    return species["species_name"].replace(" ", "_")


def reference_images(species: dict) -> list[Path]:
    """Return all reference image paths for a species (matched by folder name)."""
    folder_name = species_slug(species)
    ref_dir = REFERENCES_DIR / folder_name
    if not ref_dir.is_dir():
        raise FileNotFoundError(
            f"Reference folder not found: {ref_dir}\n"
            f"Expected: references/{folder_name}/"
        )
    images = sorted(
        p for p in ref_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"No images found in {ref_dir}")
    return images


# ── Step A (optional): Upload reference images → file IDs ─────────────────────


def upload_references() -> dict[str, list[str]]:
    """
    Upload every reference image to the Files API (purpose='vision').
    Returns and saves a manifest: { species_slug: [file_id, ...] }
    """
    client = OpenAI()
    manifest: dict[str, list[str]] = {}

    for species in SPECIES_LIST:
        slug = species_slug(species)
        paths = reference_images(species)
        ids = []
        print(f"\n{species['species_name']}  ({len(paths)} image(s))")
        for p in paths:
            with p.open("rb") as fh:
                uploaded = client.files.create(file=fh, purpose="user_data")
            print(f"  Uploaded {p.name}  →  {uploaded.id}")
            ids.append(uploaded.id)
        manifest[slug] = ids

    FILE_IDS_FILE.write_text(json.dumps(manifest, indent=2))
    print(f"\nFile IDs saved to {FILE_IDS_FILE}")
    return manifest


# ── Step B: Direct edit API calls (no batch) ──────────────────────────────────


def run_direct():
    """
    Call /v1/images/edits synchronously for every species × variation.
    Reference images are passed as (filename, bytes, mime_type) tuples.
    """
    client = OpenAI()
    template = load_template()
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = []

    for species in SPECIES_LIST:
        slug = species_slug(species)
        refs = reference_images(species)
        print(f"\n{'─' * 60}")
        print(f"Species : {species['species_name']}")
        print(f"Refs    : {[r.name for r in refs]}")

        for var in VARIATIONS:
            merged = {**species, **var}
            prompt = fill_template(template, merged)
            label = var["view_angle"].replace(" ", "_")
            custom_id = f"{slug}-{label}"
            print(f"  Generating {label} …", end=" ", flush=True)

            # Build (filename, bytes, mime_type) tuples for multipart upload
            image_data = [
                (r.name, r.read_bytes(), MIME_TYPES.get(r.suffix.lower(), "image/png"))
                for r in refs
            ]
            response = client.images.edit(
                model=MODEL,
                image=image_data if len(image_data) > 1 else image_data[0],
                prompt=prompt,
                n=1,
                size=SIZE,
                quality=QUALITY,
            )

            saved = []
            for j, img in enumerate(response.data):
                if img.b64_json:
                    out_path = OUTPUT_DIR / f"{custom_id}_{j}.png"
                    out_path.write_bytes(base64.b64decode(img.b64_json))
                    saved.append(str(out_path))
                    print(f"saved → {out_path.name}", end="  ")
                elif img.url:
                    saved.append(img.url)
                    print(f"url → {img.url}", end="  ")
            print()
            summary.append({"custom_id": custom_id, "images": saved})

    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone. Summary → {OUTPUT_DIR / 'summary.json'}")


# ── Step C: Build batch .jsonl (requires uploaded file IDs) ───────────────────


def build_batch_file():
    """
    Build batchinput_edit.jsonl using pre-uploaded file IDs.
    Run `upload` first to generate file_ids.json.
    """
    if not FILE_IDS_FILE.exists():
        raise FileNotFoundError(f"{FILE_IDS_FILE} not found — run `upload` first.")
    manifest: dict[str, list[str]] = json.loads(FILE_IDS_FILE.read_text())
    template = load_template()
    lines = []

    for species in SPECIES_LIST:
        slug = species_slug(species)
        file_ids = manifest.get(slug)
        if not file_ids:
            raise KeyError(f"No file IDs found for {slug} in {FILE_IDS_FILE}")

        for var in VARIATIONS:
            merged = {**species, **var}
            prompt = fill_template(template, merged)
            label = var["view_angle"].replace(" ", "_")
            custom_id = f"{slug}-{label}"

            # JSON body: "images" (plural) = array of file-ID objects.
            # "image" (singular) is only for multipart/form-data (DALL-E 2).
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
                    "images": [{"file_id": fid} for fid in file_ids],
                },
            }
            lines.append(json.dumps(record))

    BATCH_INPUT.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} requests to {BATCH_INPUT}")


# ── Step D: Submit batch ───────────────────────────────────────────────────────


def submit_batch():
    client = OpenAI()

    print("Uploading batch input file…")
    with BATCH_INPUT.open("rb") as fh:
        file_obj = client.files.create(file=fh, purpose="batch")
    print(f"  File ID: {file_obj.id}")

    print("Creating batch…")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/images/edits",
        completion_window="24h",
        metadata={"description": "simone image-edit batch"},
    )
    print(f"  Batch ID: {batch.id}  status: {batch.status}")
    BATCH_ID_FILE.write_text(batch.id)
    print(f"Batch ID saved to {BATCH_ID_FILE}")
    return batch.id


# ── Step E: Poll status ───────────────────────────────────────────────────────


def _validate_batch_id(batch_id: str) -> str:
    """Raise if batch_id looks malformed (not a plain batch_* token)."""
    clean = batch_id.strip()
    if not clean.startswith("batch_") or len(clean) > 64 or " " in clean:
        raise ValueError(
            f"Malformed batch ID ({len(clean)} chars): {clean!r}\n"
            f"Check {BATCH_ID_FILE} — it may contain extra text."
        )
    return clean


def check_status(batch_id: str | None = None, poll: bool = False, interval: int = 60):
    client = OpenAI()
    if batch_id is None:
        batch_id = BATCH_ID_FILE.read_text().strip()
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
            break
        time.sleep(interval)


# ── Step F: Download batch results ────────────────────────────────────────────


def download_results(batch_id: str | None = None):
    client = OpenAI()
    if batch_id is None:
        batch_id = BATCH_ID_FILE.read_text().strip()
    batch_id = _validate_batch_id(batch_id)

    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"Batch not yet complete (status={batch.status}). Aborting.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = []

    if batch.output_file_id:
        raw = client.files.content(batch.output_file_id).text
        output_file = OUTPUT_DIR / "raw_edit_output.jsonl"
        output_file.write_text(raw)
        print(f"Saved raw output → {output_file}")

        for line in raw.strip().splitlines():
            entry = json.loads(line)
            cid = entry["custom_id"]
            if entry.get("error"):
                print(f"  {cid}: ERROR — {entry['error']}")
                summary.append({"custom_id": cid, "error": entry["error"]})
                continue

            body = entry["response"]["body"]
            images = body.get("data", [])
            saved = []
            for j, img in enumerate(images):
                if img.get("b64_json"):
                    img_path = OUTPUT_DIR / f"{cid}_{j}.png"
                    img_path.write_bytes(base64.b64decode(img["b64_json"]))
                    saved.append(str(img_path))
                    print(f"  {cid}: saved → {img_path.name}")
                elif img.get("url"):
                    saved.append(img["url"])
                    print(f"  {cid}: url → {img['url']}")
            summary.append({"custom_id": cid, "images": saved})

        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSummary → {OUTPUT_DIR / 'summary.json'}")
    else:
        print("No output file (all requests may have failed).")

    if batch.error_file_id:
        err_raw = client.files.content(batch.error_file_id).text
        (OUTPUT_DIR / "errors.jsonl").write_text(err_raw)
        print(f"Errors  → {OUTPUT_DIR / 'errors.jsonl'}")
        for line in err_raw.strip().splitlines():
            print(" ", line)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenAI image-edit generation with reference images"
    )
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("direct", help="Run edits synchronously (no batch)")
    sub.add_parser("upload", help="Upload reference images → file_ids.json")
    sub.add_parser("build", help="Build batchinput_edit.jsonl (needs upload first)")
    sub.add_parser("submit", help="Submit batch from batchinput_edit.jsonl")

    p_st = sub.add_parser("status", help="Check batch status")
    p_st.add_argument("--id", help="Batch ID (default: read batch_edit_id.txt)")
    p_st.add_argument("--poll", action="store_true")
    p_st.add_argument("--interval", type=int, default=60)

    p_dl = sub.add_parser("download", help="Download batch results")
    p_dl.add_argument("--id", help="Batch ID (default: read batch_edit_id.txt)")

    args = parser.parse_args()

    if args.cmd == "direct":
        run_direct()
    elif args.cmd == "upload":
        upload_references()
    elif args.cmd == "build":
        build_batch_file()
    elif args.cmd == "submit":
        submit_batch()
    elif args.cmd == "status":
        check_status(
            batch_id=getattr(args, "id", None),
            poll=args.poll,
            interval=args.interval,
        )
    elif args.cmd == "download":
        download_results(batch_id=getattr(args, "id", None))
    else:
        parser.print_help()
