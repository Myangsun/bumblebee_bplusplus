#!/usr/bin/env python3
"""
Extract bumblebee cutouts using SAM (smart multi-point) and save as RGBA PNGs.

This is step 1 of the 2-step copy-paste augmentation process.
Extracts high-quality cutouts that you can manually review before pasting.

Segmentation Strategy:
- Tries center point first (best for centered subjects)
- If confidence < 0.8 OR segmented area < 10% of image, tries 4 adjacent points
- Returns result with highest confidence score
- Uses both confidence score and mask area to detect segmentation failures

Usage example:
  python scripts/extract_cutouts.py \\
    --targets Bombus_sandersoni\\
    --dataset-root GBIF_MA_BUMBLEBEES \\
    --sam-checkpoint checkpoints/sam_vit_h.pth

Notes:
- Reads images from GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/
- Saves RGBA cutouts to CACHE_CNP/cutouts/<species>/cutout_XXXXX_<source_name>.png
- Cutout filenames include original source image names for traceability
- You can manually delete low-quality cutouts before running step 2
- Works best when bee is roughly centered in image
- Automatically falls back to adjacent points if center is too small/low confidence
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
import torch

try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "segment-anything is required. Install via:\n"
        "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
        f"Import error: {e}"
    )


RNG = random.Random(42)


def _list_images(p: Path) -> List[Path]:
    """List all image files in a directory."""
    exts = (".jpg", ".jpeg", ".png")
    return [q for q in p.rglob("*") if q.suffix.lower() in exts]


def _list_images_excluding_augmented(p: Path) -> List[Path]:
    """List image files excluding previously augmented ones."""
    imgs = _list_images(p)
    return [q for q in imgs if not q.name.startswith("aug_")]


def _ensure_dir(p: Path) -> None:
    """Create directory if it doesn't exist."""
    p.mkdir(parents=True, exist_ok=True)


def load_sam(checkpoint_path: Path, model_type: str = "vit_h") -> SamPredictor:
    """Load SAM model from checkpoint."""
    if not Path(checkpoint_path).exists():
        raise SystemExit(f"SAM checkpoint not found: {checkpoint_path}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.eval()
    return SamPredictor(sam)


def central_click_mask(
    predictor: SamPredictor,
    pil_img: Image.Image,
    confidence_threshold: float = 0.8,
    min_area_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate mask from SAM with smart multi-point fallback.

    Strategy:
    1. Try center point first (usually best for centered subjects)
    2. If confidence is low OR area is too small, try adjacent points
    3. Return the result with highest confidence

    Args:
        predictor: SAM predictor
        pil_img: PIL image
        confidence_threshold: If center confidence >= threshold AND area is good, use immediately (default 0.8)
        min_area_ratio: Minimum mask area as fraction of image (default 0.1 = 10%)

    Returns:
        img: RGB image array
        mask: binary mask (0/1)
    """
    img = np.array(pil_img.convert("RGB"))
    h, w = img.shape[:2]
    image_area = h * w
    min_pixels = image_area * min_area_ratio

    predictor.set_image(img)

    # Try center first
    point = np.array([[w // 2, h // 2]], dtype=np.float32)
    label = np.array([1], dtype=np.int32)
    masks, scores, _ = predictor.predict(
        point_coords=point, point_labels=label, multimask_output=True
    )
    best_mask = masks[int(np.argmax(scores))].astype(np.uint8)
    best_score = float(np.max(scores))
    best_area = np.sum(best_mask)

    # If center is good enough (high confidence AND large enough area), return early
    if best_score >= confidence_threshold and best_area >= min_pixels:
        return img, best_mask

    # Otherwise, try adjacent points
    offset = min(h, w) // 6  # 1/6 of shorter dimension
    fallback_points = [
        (w // 2, h // 2 - offset),  # Above center
        (w // 2, h // 2 + offset),  # Below center
        (w // 2 - offset, h // 2),  # Left of center
        (w // 2 + offset, h // 2),  # Right of center
    ]

    with torch.no_grad():
        for px, py in fallback_points:
            px = max(0, min(px, w - 1))
            py = max(0, min(py, h - 1))

            point = np.array([[px, py]], dtype=np.float32)
            label = np.array([1], dtype=np.int32)
            masks, scores, _ = predictor.predict(
                point_coords=point, point_labels=label, multimask_output=True
            )
            mask = masks[int(np.argmax(scores))].astype(np.uint8)
            score = float(np.max(scores))

            # Keep if better than current best
            if score > best_score:
                best_score = score
                best_mask = mask

    return img, best_mask


def make_cutout_rgba(img_rgb: np.ndarray, mask01: np.ndarray, feather: int = 3) -> np.ndarray:
    """
    Create RGBA cutout with soft alpha blending.

    Args:
        img_rgb: RGB image array
        mask01: binary mask (0/1)
        feather: Gaussian blur sigma for soft edges

    Returns:
        RGBA image array
    """
    # Feather the mask for soft edges
    alpha = cv2.GaussianBlur(mask01 * 255, (0, 0), feather).astype(np.uint8)
    return np.dstack([img_rgb, alpha])  # RGBA


def _cutout_cache_dir(species: str) -> Path:
    """Get cutout cache directory for species."""
    d = Path("CACHE_CNP") / "cutouts" / species
    _ensure_dir(d)
    return d


def _save_cutout(species: str, idx: int, rgba: np.ndarray, source_name: str = "") -> Path:
    """
    Save RGBA cutout to cache directory.

    Args:
        species: Species name
        idx: Index number
        rgba: RGBA image array
        source_name: Original source image filename (without extension)
    """
    d = _cutout_cache_dir(species)
    if source_name:
        # Include original filename in cutout name
        p = d / f"cutout_{idx:05d}_{source_name}.png"
    else:
        p = d / f"cutout_{idx:05d}.png"
    Image.fromarray(rgba).save(p)
    return p


def _load_cached_cutouts(species: str) -> int:
    """Count existing cached cutouts for a species."""
    d = _cutout_cache_dir(species)
    return len(list(d.glob("cutout_*.png")))


def normalize_species_name(name: str) -> str:
    """Normalize species name for folder naming."""
    return name.replace(" ", "_")


def extract_cutouts_for_species(
    predictor: SamPredictor,
    dataset_root: Path,
    species: str,
    extract_all: bool = False,
) -> dict:
    """
    Extract cutouts from all images of a species.

    Args:
        predictor: SAM predictor
        dataset_root: Root dataset directory (e.g., GBIF_MA_BUMBLEBEES)
        species: Species folder name
        extract_all: If True, re-extract all. If False, only extract new ones.

    Returns:
        Dictionary with extraction statistics
    """
    sp = normalize_species_name(species)
    train_dir = dataset_root / "prepared_cnp" / "train"
    sp_dir = train_dir / sp

    if not sp_dir.exists():
        return {
            "species": sp,
            "status": "error",
            "message": f"Species directory not found: {sp_dir}",
            "extracted": 0,
            "failed": 0,
        }

    # Get source images (excluding previously augmented)
    src_imgs = _list_images_excluding_augmented(sp_dir)
    if not src_imgs:
        return {
            "species": sp,
            "status": "error",
            "message": f"No training images found in {sp_dir}",
            "extracted": 0,
            "failed": 0,
        }

    # Get starting index for new cutouts
    cached_count = _load_cached_cutouts(sp)
    start_idx = cached_count if not extract_all else 0

    if extract_all and cached_count > 0:
        print(f"[{sp}] Clearing {cached_count} previously cached cutouts...")
        shutil.rmtree(_cutout_cache_dir(sp))
        start_idx = 0

    print(f"[{sp}] Extracting cutouts from {len(src_imgs)} images...")
    if cached_count > 0 and not extract_all:
        print(f"[{sp}] {cached_count} cutouts already cached, extracting new ones...")

    extracted = 0
    failed = 0
    failed_list = []

    for i, p in enumerate(src_imgs):
        try:
            pil = Image.open(p).convert("RGB")
            rgb, m = central_click_mask(predictor, pil)
            rgba = make_cutout_rgba(rgb, m, feather=3)
            # Use source filename (without extension) for traceability
            source_stem = p.stem  # filename without extension
            _save_cutout(sp, start_idx + i, rgba, source_name=source_stem)
            extracted += 1

            if (extracted + failed) % 20 == 0:
                print(f"  [{sp}] Progress: {extracted + failed}/{len(src_imgs)}")
        except Exception as e:
            print(f"  [WARN] Cutout failed for {p.name}: {e}")
            failed += 1
            failed_list.append(str(p.name))

    print(f"[{sp}] ✓ Extracted {extracted} cutouts, {failed} failed")

    return {
        "species": sp,
        "status": "success",
        "extracted": extracted,
        "failed": failed,
        "failed_images": failed_list[:10],  # Show first 10 failed
        "total_images": len(src_imgs),
        "cached_count": cached_count,
        "new_cached_count": _load_cached_cutouts(sp),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Extract bumblebee cutouts using SAM (step 1 of 2)"
    )
    ap.add_argument(
        "--targets",
        nargs="+",
        required=True,
        help="List of species folder names (e.g., Bombus_sandersoni Bombus_bohemicus)",
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("GBIF_MA_BUMBLEBEES"),
        help="Root containing prepared_cnp/train",
    )
    ap.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("checkpoints/sam_vit_h.pth"),
        help="Path to SAM checkpoint file",
    )
    ap.add_argument(
        "--extract-all",
        action="store_true",
        help="Re-extract all cutouts (clear cache first)",
    )
    args = ap.parse_args()

    # Load SAM
    print(f"Loading SAM from {args.sam_checkpoint}...")
    predictor = load_sam(args.sam_checkpoint)

    # Extract cutouts for each species
    results_log: List[dict] = []
    for species in args.targets:
        result = extract_cutouts_for_species(
            predictor,
            args.dataset_root,
            species,
            extract_all=args.extract_all,
        )
        results_log.append(result)

    # Save extraction log
    _ensure_dir(Path("RESULTS") / "cutout_extraction")
    log_path = Path("RESULTS") / "cutout_extraction" / "extraction_log.json"
    log_path.write_text(json.dumps(results_log, indent=2))

    # Summary
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    total_extracted = sum(r.get("extracted", 0) for r in results_log)
    total_failed = sum(r.get("failed", 0) for r in results_log)
    print(f"Total extracted: {total_extracted}")
    print(f"Total failed: {total_failed}")
    print(f"\nCutout locations:")
    for species in args.targets:
        sp = normalize_species_name(species)
        d = _cutout_cache_dir(sp)
        count = len(list(d.glob("cutout_*.png")))
        print(f"  {sp}: {count} cutouts in {d}")
    print(f"\n✓ Log written to: {log_path}")
    print("\nNext step: Review cutouts in CACHE_CNP/cutouts/<species>/")
    print("Delete low-quality cutouts, then run: python scripts/paste_cutouts.py")


if __name__ == "__main__":
    main()
