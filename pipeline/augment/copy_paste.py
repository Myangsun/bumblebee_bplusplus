#!/usr/bin/env python3
"""
Copy-paste augmentation: extract SAM cutouts then paste them onto flower backgrounds.

This module unifies the two-step process:
  Step 1 — Extract bumblebee cutouts with SAM (segment-anything)
  Step 2 — Paste cutouts onto flower backgrounds

Importable API
--------------
    from pipeline.augment.copy_paste import extract_cutouts, generate_composites

CLI (step 1 + step 2 in sequence)
---
    python pipeline/augment/copy_paste.py \\
        --targets Bombus_sandersoni Bombus_ashtoni \\
        --dataset-root GBIF_MA_BUMBLEBEES \\
        --sam-checkpoint checkpoints/sam_vit_h.pth \\
        --per-class-count 100

Notes
-----
- Reads images from GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/
- Saves RGBA cutouts to CACHE_CNP/cutouts/<species>/
- Pastes composites back to GBIF_MA_BUMBLEBEES/prepared_cnp/train/<species>/
- Run step 1 only: --extract-only
- Run step 2 only: --paste-only (requires cached cutouts)
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pipeline.config import GBIF_DATA_DIR, RESULTS_DIR

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    from segment_anything import sam_model_registry, SamPredictor
    _SAM_AVAILABLE = True
except ImportError:
    _SAM_AVAILABLE = False

RNG = random.Random(42)

# ── SAM helpers ───────────────────────────────────────────────────────────────


def _list_images(p: Path) -> List[Path]:
    exts = (".jpg", ".jpeg", ".png")
    if not p.exists():
        return []
    return [q for q in p.rglob("*") if q.suffix.lower() in exts]


def _list_images_excluding_augmented(p: Path) -> List[Path]:
    return [q for q in _list_images(p) if not q.name.startswith("aug_")]


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _cutout_cache_dir(species: str) -> Path:
    d = Path("CACHE_CNP") / "cutouts" / species
    _ensure_dir(d)
    return d


def load_sam(checkpoint_path: Path, model_type: str = "vit_h") -> "SamPredictor":
    if not _SAM_AVAILABLE:
        raise SystemExit(
            "segment-anything is required:\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git"
        )
    if not Path(checkpoint_path).exists():
        raise SystemExit(f"SAM checkpoint not found: {checkpoint_path}")
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint_path))
    sam.eval()
    return SamPredictor(sam)


def central_click_mask(
    predictor: "SamPredictor",
    pil_img: Image.Image,
    confidence_threshold: float = 0.8,
    min_area_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate SAM mask with smart multi-point fallback strategy."""
    import torch
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

    if best_score >= confidence_threshold and best_area >= min_pixels:
        return img, best_mask

    # Fallback: try adjacent points
    offset = min(h, w) // 6
    fallback_points = [
        (w // 2, h // 2 - offset),
        (w // 2, h // 2 + offset),
        (w // 2 - offset, h // 2),
        (w // 2 + offset, h // 2),
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
            if score > best_score:
                best_score = score
                best_mask = mask

    return img, best_mask


def make_cutout_rgba(img_rgb: np.ndarray, mask01: np.ndarray, feather: int = 3) -> np.ndarray:
    alpha = cv2.GaussianBlur(mask01 * 255, (0, 0), feather).astype(np.uint8)
    return np.dstack([img_rgb, alpha])


# ── Step 1: Extract ───────────────────────────────────────────────────────────


def extract_cutouts(
    predictor: "SamPredictor",
    dataset_root: Path,
    species: str,
    extract_all: bool = False,
) -> dict:
    """
    Extract RGBA cutouts from all images of a species using SAM.

    Args:
        predictor: Loaded SAM predictor.
        dataset_root: Root dataset directory (e.g., GBIF_MA_BUMBLEBEES).
        species: Species folder name.
        extract_all: If True, clear cache and re-extract.

    Returns:
        Statistics dict.
    """
    sp = species.replace(" ", "_")
    sp_dir = dataset_root / "prepared_cnp" / "train" / sp

    if not sp_dir.exists():
        return {"species": sp, "status": "error", "message": f"Not found: {sp_dir}", "extracted": 0, "failed": 0}

    src_imgs = _list_images_excluding_augmented(sp_dir)
    if not src_imgs:
        return {"species": sp, "status": "error", "message": f"No training images in {sp_dir}", "extracted": 0, "failed": 0}

    cached_count = len(list(_cutout_cache_dir(sp).glob("cutout_*.png")))
    start_idx = cached_count if not extract_all else 0

    if extract_all and cached_count > 0:
        print(f"[{sp}] Clearing {cached_count} previously cached cutouts...")
        shutil.rmtree(_cutout_cache_dir(sp))
        start_idx = 0

    print(f"[{sp}] Extracting cutouts from {len(src_imgs)} images...")

    extracted = 0
    failed = 0
    failed_list: List[str] = []

    for i, p in enumerate(src_imgs):
        try:
            pil = Image.open(p).convert("RGB")
            rgb, m = central_click_mask(predictor, pil)
            rgba = make_cutout_rgba(rgb, m, feather=3)
            d = _cutout_cache_dir(sp)
            cutout_path = d / f"cutout_{start_idx + i:05d}_{p.stem}.png"
            Image.fromarray(rgba).save(cutout_path)
            extracted += 1
            if (extracted + failed) % 20 == 0:
                print(f"  [{sp}] Progress: {extracted + failed}/{len(src_imgs)}")
        except Exception as e:
            print(f"  [WARN] Cutout failed for {p.name}: {e}")
            failed += 1
            failed_list.append(str(p.name))

    print(f"[{sp}] Extracted {extracted} cutouts, {failed} failed")
    return {
        "species": sp, "status": "success",
        "extracted": extracted, "failed": failed,
        "failed_images": failed_list[:10],
        "total_images": len(src_imgs),
    }


# ── Step 2: Paste ─────────────────────────────────────────────────────────────


def load_cutouts(species: str) -> List[Tuple[np.ndarray, str]]:
    d = _cutout_cache_dir(species)
    cutouts = []
    for p in sorted(d.glob("cutout_*.png")):
        try:
            arr = np.array(Image.open(p).convert("RGBA"))
            if arr.shape[2] == 4:
                source_name = p.stem.split("_", 2)[2] if "_" in p.stem else p.stem
                cutouts.append((arr, source_name))
        except Exception as e:
            print(f"  [WARN] Failed to load {p.name}: {e}")
    return cutouts


def rotate_rgba(rgba: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0:
        return rgba
    h, w = rgba.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3:4]
    rotated_rgb = cv2.warpAffine(rgb, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    rotated_alpha = cv2.warpAffine(alpha, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.dstack([rotated_rgb, rotated_alpha])


def resize_with_ratio(rgba: np.ndarray, target_short_side: int) -> np.ndarray:
    h, w = rgba.shape[:2]
    scale = target_short_side / max(1, min(h, w))
    new_wh = (max(1, int(w * scale)), max(1, int(h * scale)))
    return cv2.resize(rgba, new_wh, interpolation=cv2.INTER_AREA)


def preprocess_flower(rgb: np.ndarray, scale_factor: int = 3, crop_size: int = 640) -> np.ndarray:
    h, w = rgb.shape[:2]
    enlarged = cv2.resize(rgb, (int(w * scale_factor), int(h * scale_factor)), interpolation=cv2.INTER_CUBIC)
    eh, ew = enlarged.shape[:2]
    y_start = max(0, (eh - crop_size) // 2)
    x_start = max(0, (ew - crop_size) // 2)
    cropped = enlarged[y_start:y_start + crop_size, x_start:x_start + crop_size, :]
    if cropped.shape[0] < crop_size or cropped.shape[1] < crop_size:
        cropped = cv2.copyMakeBorder(
            cropped, 0, max(0, crop_size - cropped.shape[0]),
            0, max(0, crop_size - cropped.shape[1]),
            cv2.BORDER_REFLECT_101
        )
    return cropped


def alpha_paste(bg_rgb: np.ndarray, fg_rgba: np.ndarray, center_xy: Tuple[int, int]) -> np.ndarray:
    fh, fw = fg_rgba.shape[:2]
    x0 = int(center_xy[0] - fw // 2)
    y0 = int(center_xy[1] - fh // 2)
    x1, y1 = max(0, x0), max(0, y0)
    x2, y2 = min(bg_rgb.shape[1], x0 + fw), min(bg_rgb.shape[0], y0 + fh)
    if x2 <= x1 or y2 <= y1:
        return bg_rgb
    fg_crop = fg_rgba[y1 - y0: y1 - y0 + (y2 - y1), x1 - x0: x1 - x0 + (x2 - x1)]
    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    a = fg_crop[:, :, 3:4].astype(np.float32) / 255.0
    bg_crop = bg_rgb[y1:y2, x1:x2, :].astype(np.float32)
    comp = (a * fg_rgb + (1 - a) * bg_crop).astype(np.uint8)
    out = bg_rgb.copy()
    out[y1:y2, x1:x2, :] = comp
    return out


def generate_composites(
    cutout_species_list: List[str],
    flower_images: List[Path],
    output_dir: Path,
    per_class_count: int = 100,
    size_ratio_range: Tuple[float, float] = (0.5, 0.7),
    rotation_range: Tuple[float, float] = (-180.0, 180.0),
    paste_position: str = "center",
) -> dict:
    """
    Paste pre-extracted cutouts onto flower backgrounds.

    Args:
        cutout_species_list: Species names with cached cutouts.
        flower_images: Flower background image paths.
        output_dir: Train directory where augmented images will be saved.
        per_class_count: Number of composites per species.
        size_ratio_range: (min, max) cutout size as fraction of background short side.
        rotation_range: (min, max) rotation degrees.
        paste_position: "center" or "random".

    Returns:
        Statistics dict.
    """
    _ensure_dir(output_dir)
    if not flower_images:
        raise SystemExit("No flower images found")

    results_log: List[dict] = []
    stats: dict = {"total_generated": 0, "total_failed": 0, "per_species": {}}

    for species in cutout_species_list:
        sp = species.replace(" ", "_")
        cutouts = load_cutouts(sp)
        if not cutouts:
            print(f"[WARN] No cutouts found for {sp}")
            continue

        print(f"\n[{sp}] Generating {per_class_count} composites using {len(cutouts)} cutouts...")
        sp_output_dir = output_dir / sp
        _ensure_dir(sp_output_dir)

        generated = 0
        failed = 0

        while generated < per_class_count:
            try:
                cutout_array, source_name = RNG.choice(cutouts)
                cutout = cutout_array.copy()
                bg_path = RNG.choice(flower_images)
                bg_img = np.array(Image.open(bg_path).convert("RGB"))
                bg_img = preprocess_flower(bg_img, scale_factor=3, crop_size=640)
                bh, bw = bg_img.shape[:2]

                angle = RNG.uniform(rotation_range[0], rotation_range[1])
                cutout = rotate_rgba(cutout, angle)
                ratio = RNG.uniform(size_ratio_range[0], size_ratio_range[1])
                target_size = int(min(bh, bw) * ratio)
                cutout = resize_with_ratio(cutout, target_size)

                if paste_position == "center":
                    cx, cy = bw // 2, bh // 2
                else:
                    cx = RNG.randint(int(0.15 * bw), int(0.85 * bw))
                    cy = RNG.randint(int(0.15 * bh), int(0.85 * bh))

                composite = alpha_paste(bg_img, cutout, (cx, cy))
                output_path = sp_output_dir / f"aug_{generated:05d}_{source_name}.png"
                Image.fromarray(composite).save(output_path)

                results_log.append({
                    "species": sp, "output": str(output_path),
                    "background": str(bg_path), "source_cutout": source_name,
                    "paste_position": [int(cx), int(cy)],
                    "rotation_angle": float(angle), "size_ratio": float(ratio),
                })
                generated += 1
                if generated % 50 == 0:
                    print(f"  [{sp}] {generated}/{per_class_count} done")

            except Exception as e:
                print(f"  [WARN] Composite failed: {e}")
                failed += 1
                if failed > per_class_count * 2:
                    break

        print(f"[{sp}] Generated {generated}, {failed} failed")
        stats["per_species"][sp] = {"generated": generated, "failed": failed, "target": per_class_count}
        stats["total_generated"] += generated
        stats["total_failed"] += failed

    return {"stats": stats, "results_log": results_log}


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(
        description="Copy-paste augmentation: extract SAM cutouts and paste onto flowers"
    )
    ap.add_argument("--targets", nargs="+", required=True,
                    help="Species folder names to augment")
    ap.add_argument("--dataset-root", type=Path, default=GBIF_DATA_DIR,
                    help="Dataset root containing prepared_cnp/train")
    ap.add_argument("--sam-checkpoint", type=Path, default=Path("checkpoints/sam_vit_h.pth"),
                    help="SAM checkpoint path")
    ap.add_argument("--flower-dir", type=Path, default=Path("Flowers"),
                    help="Directory with flower background images")
    ap.add_argument("--output-subdir", type=str, default="prepared_cnp",
                    help="Output subdirectory name (e.g., prepared_cnp, prepared_cnp_100)")
    ap.add_argument("--per-class-count", type=int, default=100,
                    help="Augmented images per species (default: 100)")
    ap.add_argument("--size-ratio-range", type=float, nargs=2, default=[0.5, 0.7],
                    help="Cutout size range as fraction of background (default: 0.5 0.7)")
    ap.add_argument("--rotation-range", type=float, nargs=2, default=[-180, 180],
                    help="Rotation range in degrees (default: -180 180)")
    ap.add_argument("--paste-position", choices=["center", "random"], default="center",
                    help="Paste position on background")
    ap.add_argument("--extract-all", action="store_true",
                    help="Re-extract all cutouts (clears cache)")
    ap.add_argument("--extract-only", action="store_true",
                    help="Only extract cutouts, skip pasting")
    ap.add_argument("--paste-only", action="store_true",
                    help="Only paste (skip extraction, requires cached cutouts)")

    args = ap.parse_args()

    # ── Step 1: Extract ──
    if not args.paste_only:
        print(f"Loading SAM from {args.sam_checkpoint}...")
        predictor = load_sam(args.sam_checkpoint)

        results_log: List[dict] = []
        for species in args.targets:
            result = extract_cutouts(predictor, args.dataset_root, species, args.extract_all)
            results_log.append(result)

        log_dir = RESULTS_DIR / "cutout_extraction"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "extraction_log.json"
        log_path.write_text(json.dumps(results_log, indent=2))

        print("\n" + "=" * 60)
        print(f"Total extracted: {sum(r.get('extracted', 0) for r in results_log)}")
        print(f"Total failed:    {sum(r.get('failed', 0) for r in results_log)}")
        print(f"Log: {log_path}")

        if args.extract_only:
            print("\nReview cutouts in CACHE_CNP/cutouts/<species>/ then re-run with --paste-only")
            return

    # ── Step 2: Paste ──
    train_dir = args.dataset_root / args.output_subdir / "train"
    if not train_dir.exists():
        raise SystemExit(f"Train directory not found: {train_dir}")

    print(f"\nLoading flower images from {args.flower_dir}...")
    flower_images = _list_images(args.flower_dir)
    if not flower_images:
        raise SystemExit(f"No images found in {args.flower_dir}")
    print(f"Found {len(flower_images)} flower images")

    print("\n" + "=" * 60)
    print("GENERATING COMPOSITES")
    print("=" * 60)
    result = generate_composites(
        args.targets,
        flower_images,
        train_dir,
        args.per_class_count,
        tuple(args.size_ratio_range),
        tuple(args.rotation_range),
        args.paste_position,
    )

    log_dir = RESULTS_DIR / "paste_composites"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "generation_log.json"
    log_path.write_text(json.dumps(result["results_log"], indent=2))

    stats = result["stats"]
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total generated: {stats['total_generated']}")
    print(f"Total failed:    {stats['total_failed']}")
    for sp, sp_stats in stats["per_species"].items():
        print(f"  {sp}: {sp_stats['generated']}/{sp_stats['target']}")
    print(f"\nOutput: {train_dir.absolute()}")
    print(f"Log:    {log_path.absolute()}")
    print("Done!")


if __name__ == "__main__":
    main()
