#!/usr/bin/env python3
"""
Remove backgrounds from synthetic bumblebee images using Grounded-SAM.

Uses Grounding DINO for text-prompted detection ("bumblebee") → bounding box,
then SAM for precise segmentation within the box → composite on white background.

Pipeline:
    RESULTS/synthetic_generation/{species}/*.jpg
        → Grounded-SAM segmentation
        → RESULTS/synthetic_generation_nobg/{species}/*.jpg

CLI
---
    # Remove backgrounds from all species
    python scripts/remove_background.py

    # Specific species only
    python scripts/remove_background.py --species Bombus_ashtoni Bombus_sandersoni

    # Custom detection threshold
    python scripts/remove_background.py --box-threshold 0.25 --text-threshold 0.2
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.config import RESULTS_DIR
from pipeline.augment.copy_paste import load_sam, central_click_mask

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

SOURCE_DIR = RESULTS_DIR / "synthetic_generation"
OUTPUT_DIR = RESULTS_DIR / "synthetic_generation_nobg"


# ── Grounding DINO helpers ────────────────────────────────────────────────────


def _default_gdino_config() -> str:
    """Resolve the GroundingDINO config shipped inside the installed package."""
    import groundingdino
    pkg_dir = Path(groundingdino.__file__).parent
    return str(pkg_dir / "config" / "GroundingDINO_SwinT_OGC.py")


def load_grounding_dino(
    config_path: str | None,
    weights_path: str,
    device: str = "cuda",
):
    """Load Grounding DINO model."""
    from groundingdino.util.inference import load_model
    if config_path is None:
        config_path = _default_gdino_config()
    model = load_model(config_path, weights_path, device=device)
    return model


def detect_with_grounding_dino(
    model,
    image: np.ndarray,
    text_prompt: str = "bumblebee",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> np.ndarray | None:
    """
    Run Grounding DINO detection, return the best bounding box as [x1, y1, x2, y2]
    in pixel coordinates, or None if no detection.
    """
    from groundingdino.util.inference import predict
    import torchvision.transforms.functional as F

    h, w = image.shape[:2]
    pil_img = Image.fromarray(image)
    transform = F.to_tensor(pil_img)

    boxes, logits, _ = predict(
        model=model,
        image=transform,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if len(boxes) == 0:
        return None

    # Take highest confidence detection
    best_idx = logits.argmax().item()
    box = boxes[best_idx].cpu().numpy()  # cx, cy, w, h normalized

    # Convert from normalized cxcywh to pixel xyxy
    cx, cy, bw, bh = box
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)

    return np.array([x1, y1, x2, y2])


def sam_mask_from_box(
    predictor,
    image: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """Run SAM with a box prompt, return binary mask (uint8, 0/1)."""
    predictor.set_image(image)
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True,
    )
    best = masks[int(np.argmax(scores))]
    return best.astype(np.uint8)


# ── Core removal ─────────────────────────────────────────────────────────────


def remove_bg_single(
    grounding_model,
    sam_predictor,
    img_path: Path,
    output_path: Path,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
) -> dict:
    """
    Remove background from a single image.

    Returns dict with method used and success status.
    """
    pil_img = Image.open(img_path).convert("RGB")
    rgb = np.array(pil_img)

    method = "grounded_sam"
    box = detect_with_grounding_dino(
        grounding_model, rgb,
        text_prompt="bumblebee",
        box_threshold=box_threshold,
        text_threshold=text_threshold,
    )

    if box is not None:
        mask = sam_mask_from_box(sam_predictor, rgb, box)
    else:
        # Fallback: center-click SAM (from copy_paste.py)
        method = "central_click_fallback"
        _, mask = central_click_mask(sam_predictor, pil_img)

    # Composite on solid background
    h, w = rgb.shape[:2]
    bg = np.full((h, w, 3), bg_color, dtype=np.uint8)

    # Feather edges slightly for cleaner compositing
    mask_float = cv2.GaussianBlur(mask.astype(np.float32), (5, 5), 1.0)
    mask_3c = np.stack([mask_float] * 3, axis=-1)
    composite = (mask_3c * rgb.astype(np.float32) + (1 - mask_3c) * bg.astype(np.float32))
    composite = composite.astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(composite).save(output_path, quality=95)

    return {"method": method, "success": True}


# ── Main pipeline ────────────────────────────────────────────────────────────


def run(
    source_dir: Path = SOURCE_DIR,
    output_dir: Path = OUTPUT_DIR,
    species: list[str] | None = None,
    bg_color: tuple[int, int, int] = (255, 255, 255),
    sam_checkpoint: Path = Path("checkpoints/sam_vit_h.pth"),
    gdino_config: str | None = None,
    gdino_weights: str = "checkpoints/groundingdino_swint_ogc.pth",
    box_threshold: float = 0.3,
    text_threshold: float = 0.25,
    force: bool = False,
):
    """Remove backgrounds from all synthetic images."""
    if not source_dir.exists():
        raise FileNotFoundError(f"Source not found: {source_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Discover species
    all_species = sorted(
        d.name for d in source_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )
    if species:
        all_species = [s for s in all_species if s in species]

    if not all_species:
        print("No species directories found.")
        return

    print(f"Species to process: {all_species}")

    # Load models
    print(f"Loading Grounding DINO from {gdino_weights} ...")
    grounding_model = load_grounding_dino(gdino_config, gdino_weights, device=device)

    print(f"Loading SAM from {sam_checkpoint} ...")
    sam_predictor = load_sam(sam_checkpoint)

    # Process
    manifest = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "bg_color": list(bg_color),
        "box_threshold": box_threshold,
        "text_threshold": text_threshold,
        "timestamp": datetime.now().isoformat(),
        "species": {},
    }

    total_processed = total_failed = total_fallback = 0

    for sp in all_species:
        sp_src = source_dir / sp
        sp_out = output_dir / sp
        images = sorted(
            p for p in sp_src.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            print(f"\n{sp}: no images, skipping")
            continue

        # Skip if already done (unless --force)
        if not force and sp_out.exists():
            existing = len(list(sp_out.iterdir()))
            if existing >= len(images):
                print(f"\n{sp}: {existing} images already exist, skipping (use --force)")
                manifest["species"][sp] = {
                    "total": len(images), "processed": existing,
                    "failed": 0, "fallback": 0, "skipped": True,
                }
                continue

        sp_out.mkdir(parents=True, exist_ok=True)
        print(f"\n{sp}: processing {len(images)} images")

        processed = failed = fallback = 0

        for img_path in tqdm(images, desc=f"  {sp}", unit="img"):
            out_path = sp_out / img_path.name
            try:
                result = remove_bg_single(
                    grounding_model, sam_predictor,
                    img_path, out_path,
                    bg_color=bg_color,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
                processed += 1
                if result["method"] == "central_click_fallback":
                    fallback += 1
            except Exception as e:
                print(f"  WARN: {img_path.name}: {e}")
                failed += 1

        total_processed += processed
        total_failed += failed
        total_fallback += fallback

        manifest["species"][sp] = {
            "total": len(images),
            "processed": processed,
            "failed": failed,
            "fallback": fallback,
        }
        print(f"  Done: {processed} processed, {fallback} fallback, {failed} failed")

    # Save manifest
    manifest_path = output_dir / "bg_removal_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{'=' * 60}")
    print("BACKGROUND REMOVAL COMPLETE")
    print(f"  Processed:  {total_processed}")
    print(f"  Fallback:   {total_fallback}")
    print(f"  Failed:     {total_failed}")
    print(f"  Output:     {output_dir}")
    print(f"  Manifest:   {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Remove backgrounds from synthetic images using Grounded-SAM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR,
                        help="Source image directory (default: RESULTS/synthetic_generation)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: RESULTS/synthetic_generation_nobg)")
    parser.add_argument("--species", nargs="+", default=None,
                        help="Species to process (default: all)")
    parser.add_argument("--bg-color", type=int, nargs=3, default=[255, 255, 255],
                        help="Background RGB color (default: 255 255 255)")
    parser.add_argument("--sam-checkpoint", type=Path,
                        default=Path("checkpoints/sam_vit_h.pth"),
                        help="SAM checkpoint path")
    parser.add_argument("--gdino-config", type=str, default=None,
                        help="Grounding DINO config path (default: auto-detect from package)")
    parser.add_argument("--gdino-weights", type=str,
                        default="checkpoints/groundingdino_swint_ogc.pth",
                        help="Grounding DINO weights path")
    parser.add_argument("--box-threshold", type=float, default=0.3,
                        help="Grounding DINO box confidence threshold")
    parser.add_argument("--text-threshold", type=float, default=0.25,
                        help="Grounding DINO text confidence threshold")
    parser.add_argument("--force", action="store_true",
                        help="Re-process even if output exists")

    args = parser.parse_args()
    run(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        species=args.species,
        bg_color=tuple(args.bg_color),
        sam_checkpoint=args.sam_checkpoint,
        gdino_config=args.gdino_config,
        gdino_weights=args.gdino_weights,
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
        force=args.force,
    )


if __name__ == "__main__":
    main()
