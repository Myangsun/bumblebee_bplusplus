#!/usr/bin/env python3
"""
Generate PCA and t-SNE visualizations of the training dataset using BioCLIP embeddings.

Importable API
--------------
    from pipeline.evaluate.bioclip import run
    run(data_root="GBIF_MA_BUMBLEBEES/prepared_split", split="train")

CLI
---
    python pipeline/evaluate/bioclip.py
    python pipeline/evaluate/bioclip.py --data-root GBIF_MA_BUMBLEBEES/prepared_split --split train
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Make project root importable when running as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from pipeline.config import GBIF_DATA_DIR, PROJECT_ROOT

DEFAULT_DATA_ROOT = GBIF_DATA_DIR / "prepared_split"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "plots" / "bioclip_embeddings"
DEFAULT_BIOCLIP_SRC = Path("/home/su/bioclip/src")


# ── Dataset ───────────────────────────────────────────────────────────────────


class ImagePathDataset(Dataset):
    def __init__(self, items: Sequence[Tuple[Path, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        with Image.open(path) as img:
            tensor = self.transform(img.convert("RGB"))
        return tensor, label, str(path)


# ── Data collection ───────────────────────────────────────────────────────────


def _collect_images(
    dataset_dir: Path,
    per_class_limit: Optional[int],
    max_images: Optional[int],
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[str]]:
    rng = random.Random(seed)
    entries: List[Tuple[Path, int]] = []
    class_names: List[str] = []
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for label_idx, class_dir in enumerate(sorted(p for p in dataset_dir.iterdir() if p.is_dir())):
        all_images = [p for p in class_dir.rglob("*") if p.suffix.lower() in supported_exts]
        if not all_images:
            continue
        rng.shuffle(all_images)
        if per_class_limit is not None:
            all_images = all_images[:per_class_limit]
        entries.extend((p, label_idx) for p in all_images)
        class_names.append(class_dir.name)
        if max_images is not None and len(entries) >= max_images:
            entries = entries[:max_images]
            break

    if not entries:
        raise RuntimeError(f"No images found under {dataset_dir}")

    return entries, class_names


# ── BioCLIP loading ───────────────────────────────────────────────────────────


def _import_open_clip(bioclip_src: Path):
    try:
        import open_clip  # type: ignore
        return open_clip
    except ImportError:
        if bioclip_src.exists():
            sys.path.append(str(bioclip_src))
            import open_clip  # type: ignore
            return open_clip
        raise


# ── Embedding extraction ──────────────────────────────────────────────────────


def _extract_embeddings(model, device, loader):
    features, labels, paths = [], [], []
    model.eval()
    with torch.no_grad():
        for images, lbls, batch_paths in tqdm(loader, desc="Encoding images"):
            feats = F.normalize(model.encode_image(images.to(device)), dim=-1)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
            paths.extend(batch_paths)
    return np.concatenate(features), np.concatenate(labels), paths


# ── Projections ───────────────────────────────────────────────────────────────


def _run_pca(features: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=0).fit_transform(features)


def _run_tsne(features: np.ndarray, perplexity: float, n_iter: int, seed: int) -> np.ndarray:
    effective_perplexity = min(perplexity, max(5.0, (features.shape[0] - 1) / 3))
    return TSNE(
        n_components=2, learning_rate="auto", init="pca",
        perplexity=effective_perplexity, n_iter=n_iter, random_state=seed,
    ).fit_transform(features)


# ── Plotting ──────────────────────────────────────────────────────────────────


def _plot_embedding(coords, labels, class_names, title, output_path):
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    for idx, label_id in enumerate(unique_labels):
        mask = labels == label_id
        plt.scatter(coords[mask, 0], coords[mask, 1], label=class_names[label_id],
                    s=12, alpha=0.75, color=cmap(idx))
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(markerscale=1.5, fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _write_csv(path, coords, labels, class_names, image_paths):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "label_id", "label_name", "image_path"])
        for (x, y), lid, img_path in zip(coords, labels, image_paths):
            writer.writerow([x, y, lid, class_names[lid], img_path])


# ── Public API ────────────────────────────────────────────────────────────────


def run(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    split: str = "train",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    batch_size: int = 32,
    num_workers: int = 4,
    model_id: str = "hf-hub:imageomics/bioclip",
    per_class_limit: Optional[int] = 200,
    max_images: Optional[int] = 2000,
    skip_pca: bool = False,
    skip_tsne: bool = False,
    tsne_perplexity: float = 30.0,
    tsne_iterations: int = 1000,
    seed: int = 42,
    device: str = "auto",
    save_embeddings: bool = False,
    bioclip_src: Path = DEFAULT_BIOCLIP_SRC,
) -> None:
    """
    Extract BioCLIP embeddings and generate PCA/t-SNE visualizations.

    Args:
        data_root: Dataset root (should contain split subdirectory).
        split: Dataset split to visualize (train/valid/test).
        output_dir: Where to save plots and CSV files.
        batch_size: BioCLIP inference batch size.
        num_workers: DataLoader workers.
        model_id: BioCLIP model identifier for open_clip.
        per_class_limit: Max images per class (None = no limit).
        max_images: Global image cap (None = no limit).
        skip_pca: Skip PCA visualization.
        skip_tsne: Skip t-SNE visualization.
        tsne_perplexity: t-SNE perplexity.
        tsne_iterations: t-SNE optimization iterations.
        seed: Random seed.
        device: "cuda", "cpu", or "auto".
        save_embeddings: If True, save raw .npz embeddings.
        bioclip_src: Path to BioCLIP repo src/ for importing open_clip.
    """
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    dataset_dir = (data_root / split).resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {dataset_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    open_clip = _import_open_clip(Path(bioclip_src))

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"Loading dataset from {dataset_dir}")
    items, class_names = _collect_images(dataset_dir, per_class_limit, max_images, seed)
    print(f"Sampling {len(items)} images across {len(class_names)} classes")

    print(f"Loading BioCLIP model ({model_id}) on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(model_id, device=device)

    loader = DataLoader(
        ImagePathDataset(items, preprocess),
        batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False,
    )

    features, label_ids, image_paths = _extract_embeddings(model, device, loader)
    print(f"Extracted embeddings: {features.shape}")

    if save_embeddings:
        emb_path = output_dir / f"{split}_bioclip_embeddings.npz"
        np.savez_compressed(emb_path, features=features, labels=label_ids,
                            image_paths=np.array(image_paths), class_names=np.array(class_names))
        print(f"Saved embeddings to {emb_path}")

    if not skip_pca:
        print("Running PCA...")
        pca_coords = _run_pca(features)
        pca_plot = output_dir / f"{split}_pca.png"
        _plot_embedding(pca_coords, label_ids, class_names, f"BioCLIP PCA ({split})", pca_plot)
        _write_csv(output_dir / f"{split}_pca.csv", pca_coords, label_ids, class_names, image_paths)
        print(f"PCA saved to {pca_plot}")

    if not skip_tsne:
        print("Running t-SNE (may take a few minutes)...")
        tsne_coords = _run_tsne(features, tsne_perplexity, tsne_iterations, seed)
        tsne_plot = output_dir / f"{split}_tsne.png"
        _plot_embedding(tsne_coords, label_ids, class_names, f"BioCLIP t-SNE ({split})", tsne_plot)
        _write_csv(output_dir / f"{split}_tsne.csv", tsne_coords, label_ids, class_names, image_paths)
        print(f"t-SNE saved to {tsne_plot}")

    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Visualize training data with BioCLIP embeddings (PCA / t-SNE)"
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model-id", type=str, default="hf-hub:imageomics/bioclip")
    parser.add_argument("--per-class-limit", type=int, default=200)
    parser.add_argument("--max-images", type=int, default=2000)
    parser.add_argument("--skip-pca", action="store_true")
    parser.add_argument("--skip-tsne", action="store_true")
    parser.add_argument("--tsne-perplexity", type=float, default=30.0)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-embeddings", action="store_true")
    parser.add_argument("--bioclip-src", type=Path, default=DEFAULT_BIOCLIP_SRC)

    args = parser.parse_args()

    run(
        data_root=args.data_root,
        split=args.split,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        model_id=args.model_id,
        per_class_limit=None if args.per_class_limit == -1 else args.per_class_limit,
        max_images=None if args.max_images == -1 else args.max_images,
        skip_pca=args.skip_pca,
        skip_tsne=args.skip_tsne,
        tsne_perplexity=args.tsne_perplexity,
        tsne_iterations=args.tsne_iterations,
        seed=args.seed,
        device=args.device,
        save_embeddings=args.save_embeddings,
        bioclip_src=args.bioclip_src,
    )


if __name__ == "__main__":
    main()
