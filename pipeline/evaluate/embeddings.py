#!/usr/bin/env python3
"""
Extract frozen vision-foundation-model embeddings (DINOv2, BioCLIP) for the
expert-calibrated filter and latent-space failure analysis.

Importable API
--------------
    from pipeline.evaluate.embeddings import extract, load_cache

    features, paths, labels = extract(
        model="dinov2",
        image_paths=[...],
        output_path="RESULTS/embeddings/dinov2_real.npz",
    )

CLI
---
    python pipeline/evaluate/embeddings.py --model dinov2 \
        --source prepared_split:train --output RESULTS/embeddings/dinov2_real.npz

Output format (NPZ)
-------------------
    features     : (N, D) float32, L2-normalized
    image_paths  : (N,) unicode strings (absolute paths)
    species      : (N,) unicode strings (parent directory names)
    model_id     : scalar string, the backbone identifier
    resolution   : scalar int, input image size
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from pipeline.config import GBIF_DATA_DIR, PROJECT_ROOT, RESULTS_DIR

DEFAULT_OUTPUT_DIR = RESULTS_DIR / "embeddings"
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# DINOv2 defaults (Oquab et al., 2024).
DINOV2_MODEL_ID = "dinov2_vitl14"
DINOV2_RESOLUTION = 518
DINOV2_MEAN = (0.485, 0.456, 0.406)
DINOV2_STD = (0.229, 0.224, 0.225)

# BioCLIP defaults (Stevens et al., 2024).
BIOCLIP_MODEL_ID = "hf-hub:imageomics/bioclip"
BIOCLIP_SRC = Path("/home/su/bioclip/src")


class _ImageDataset(Dataset):
    def __init__(self, paths: Sequence[Path], transform):
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        with Image.open(path) as img:
            tensor = self.transform(img.convert("RGB"))
        species = path.parent.name
        if not species.startswith("Bombus_"):
            raise ValueError(
                f"Unexpected parent dir '{species}' for {path}; "
                "expected a species slug like 'Bombus_ashtoni'"
            )
        return tensor, str(path), species


# ── Image collection ──────────────────────────────────────────────────────────


def collect_images_from_dir(root: Path) -> List[Path]:
    """Recursively gather supported images under ``root`` in deterministic order."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory not found: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)


def collect_from_source(source: str) -> List[Path]:
    """
    Resolve a semantic source string to a concrete list of image paths.

    Accepted forms:
        "prepared_split:train"      → GBIF prepared_split/train
        "prepared_split:all"        → union of train+valid+test
        "synthetic:RESULTS_kfold"   → RESULTS_kfold/synthetic_generation
        "<path>"                    → literal directory path
    """
    if source.startswith("prepared_split:"):
        split = source.split(":", 1)[1]
        base = GBIF_DATA_DIR / "prepared_split"
        if split == "all":
            paths: List[Path] = []
            for s in ("train", "valid", "test"):
                d = base / s
                if d.exists():
                    paths.extend(collect_images_from_dir(d))
            return paths
        return collect_images_from_dir(base / split)

    if source.startswith("synthetic:"):
        results_subdir = source.split(":", 1)[1]
        return collect_images_from_dir(PROJECT_ROOT / results_subdir / "synthetic_generation")

    return collect_images_from_dir(Path(source))


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_dinov2(device: torch.device, model_id: str = DINOV2_MODEL_ID):
    """Load DINOv2 via torch.hub and a matching preprocessing transform."""
    model = torch.hub.load("facebookresearch/dinov2", model_id)
    model.eval().to(device)
    transform = transforms.Compose([
        transforms.Resize(DINOV2_RESOLUTION, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(DINOV2_RESOLUTION),
        transforms.ToTensor(),
        transforms.Normalize(mean=DINOV2_MEAN, std=DINOV2_STD),
    ])
    return model, transform, DINOV2_RESOLUTION


def _load_bioclip(device: torch.device, model_id: str = BIOCLIP_MODEL_ID,
                  bioclip_src: Path = BIOCLIP_SRC):
    """Load BioCLIP via open_clip (matches pipeline/evaluate/bioclip.py)."""
    try:
        import open_clip  # type: ignore
    except ImportError:
        if bioclip_src.exists():
            sys.path.append(str(bioclip_src))
            import open_clip  # type: ignore
        else:
            raise
    model, _, preprocess = open_clip.create_model_and_transforms(model_id, device=device)
    model.eval()
    # open_clip preprocess resolution is model-dependent; expose the actual value.
    resolution = getattr(model.visual, "image_size", None)
    if isinstance(resolution, (tuple, list)):
        resolution = int(resolution[0])
    elif resolution is None:
        resolution = 224
    return model, preprocess, int(resolution)


# ── Embedding forward pass ────────────────────────────────────────────────────


@torch.no_grad()
def _encode_batch(model, images: torch.Tensor, backbone: str) -> torch.Tensor:
    """Return L2-normalized CLS features for ``images`` on ``model``."""
    if backbone == "dinov2":
        feats = model.forward_features(images)["x_norm_clstoken"]
    elif backbone == "bioclip":
        feats = model.encode_image(images)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return F.normalize(feats.float(), dim=-1)


# ── Public API ────────────────────────────────────────────────────────────────


def extract(
    model: str,
    image_paths: Sequence[Path | str],
    output_path: Path | str,
    batch_size: int = 32,
    num_workers: int = 4,
    device: str = "auto",
    dinov2_model_id: str = DINOV2_MODEL_ID,
    bioclip_model_id: str = BIOCLIP_MODEL_ID,
    bioclip_src: Path = BIOCLIP_SRC,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Extract L2-normalized CLS embeddings for a list of image paths and save to NPZ.

    Args:
        model: "dinov2" or "bioclip".
        image_paths: Images to encode.
        output_path: Destination NPZ file.
        batch_size, num_workers: Loader parameters.
        device: "cuda", "cpu", or "auto".

    Returns:
        (features, image_paths_as_strings, species_names) — features shape (N, D).
    """
    image_paths = [Path(p) for p in image_paths]
    if not image_paths:
        raise ValueError("extract() received an empty image_paths list")

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if model == "dinov2":
        backbone = "dinov2"
        net, transform, resolution = _load_dinov2(device, dinov2_model_id)
        model_id = dinov2_model_id
    elif model == "bioclip":
        backbone = "bioclip"
        net, transform, resolution = _load_bioclip(device, bioclip_model_id, Path(bioclip_src))
        model_id = bioclip_model_id
    else:
        raise ValueError(f"Unknown model: {model!r} (expected 'dinov2' or 'bioclip')")

    loader = DataLoader(
        _ImageDataset(image_paths, transform),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    features_chunks: List[np.ndarray] = []
    paths_out: List[str] = []
    species_out: List[str] = []

    first_batch = True
    for images, batch_paths, batch_species in tqdm(loader, desc=f"Encoding ({model})"):
        feats = _encode_batch(net, images.to(device, non_blocking=True), backbone)
        if first_batch:
            assert feats.ndim == 2 and feats.shape[-1] > 0, \
                f"Unexpected feature shape from {backbone}: {tuple(feats.shape)}"
            first_batch = False
        features_chunks.append(feats.cpu().numpy().astype(np.float32))
        paths_out.extend(batch_paths)
        species_out.extend(batch_species)

    features = np.concatenate(features_chunks, axis=0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        features=features,
        image_paths=np.array(paths_out, dtype="U512"),
        species=np.array(species_out, dtype="U128"),
        model_id=np.array(model_id, dtype="U256"),
        resolution=np.array(resolution, dtype=np.int32),
    )
    print(f"Saved {features.shape} embeddings to {output_path}")
    return features, paths_out, species_out


def load_cache(path: Path | str) -> dict:
    """Load an NPZ produced by :func:`extract` as a dict of numpy arrays."""
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_sources(values: Iterable[str]) -> List[Path]:
    all_paths: List[Path] = []
    seen = set()
    for source in values:
        for p in collect_from_source(source):
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                all_paths.append(p)
    return all_paths


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract DINOv2 or BioCLIP CLS embeddings and cache to NPZ.",
    )
    parser.add_argument("--model", choices=("dinov2", "bioclip"), default="dinov2")
    parser.add_argument(
        "--source", action="append", required=True,
        help=("Repeatable. 'prepared_split:train|valid|test|all', "
              "'synthetic:<results_dir>', or a literal directory path."),
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="Output NPZ path (e.g. RESULTS/embeddings/dinov2_real.npz)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dinov2-model-id", default=DINOV2_MODEL_ID)
    parser.add_argument("--bioclip-model-id", default=BIOCLIP_MODEL_ID)
    parser.add_argument("--bioclip-src", type=Path, default=BIOCLIP_SRC)
    args = parser.parse_args(argv)

    paths = _parse_sources(args.source)
    print(f"Found {len(paths)} images from {len(args.source)} source(s)")

    extract(
        model=args.model,
        image_paths=paths,
        output_path=args.output,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        dinov2_model_id=args.dinov2_model_id,
        bioclip_model_id=args.bioclip_model_id,
        bioclip_src=args.bioclip_src,
    )


if __name__ == "__main__":
    main()
