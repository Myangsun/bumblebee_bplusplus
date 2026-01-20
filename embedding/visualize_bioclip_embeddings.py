#!/usr/bin/env python3
"""
Generate PCA and t-SNE visualizations of the training dataset using BioCLIP embeddings.

This script loads the BioCLIP model (via the local BioCLIP repository or the open_clip
package), extracts embeddings for images in the specified dataset, and produces
2D projections (PCA / t-SNE) along with scatter plots and optional cached features.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = REPO_ROOT / "GBIF_MA_BUMBLEBEES" / "prepared_split"
DEFAULT_BIOCLIP_SRC = Path("/home/su/bioclip/src")


def _import_open_clip(bioclip_src: Path):
    """
    Import open_clip either from the default Python environment or the local BioCLIP repo.
    """
    try:
        import open_clip  # type: ignore
        return open_clip
    except ImportError:
        if bioclip_src.exists():
            sys.path.append(str(bioclip_src))
            import open_clip  # type: ignore
            return open_clip
        raise


class ImagePathDataset(Dataset):
    """
    Simple dataset that loads images from precomputed (path, label) tuples.
    """

    def __init__(self, items: Sequence[Tuple[Path, int]], transform):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)
        return tensor, label, str(path)


def _collect_images(
    dataset_dir: Path,
    per_class_limit: int | None,
    max_images: int | None,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[str]]:
    """
    Build a flat list of (image_path, label_id) tuples from the dataset directory.
    """
    rng = random.Random(seed)
    entries: List[Tuple[Path, int]] = []
    class_names = []
    supported_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for label_idx, class_dir in enumerate(sorted(p for p in dataset_dir.iterdir() if p.is_dir())):
        all_images = [
            img_path
            for img_path in class_dir.rglob("*")
            if img_path.suffix.lower() in supported_exts
        ]
        if not all_images:
            continue

        rng.shuffle(all_images)
        if per_class_limit is not None:
            all_images = all_images[:per_class_limit]

        labeled = [(img_path, label_idx) for img_path in all_images]
        entries.extend(labeled)
        class_names.append(class_dir.name)

        if max_images is not None and len(entries) >= max_images:
            entries = entries[:max_images]
            break

    if not entries:
        raise RuntimeError(f"No images found under {dataset_dir}")

    return entries, class_names


def _build_dataloader(items, transform, batch_size, num_workers):
    dataset = ImagePathDataset(items, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        drop_last=False,
    )


def _extract_embeddings(model, device, loader):
    features = []
    labels = []
    paths = []
    model.eval()

    with torch.no_grad():
        for images, lbls, batch_paths in tqdm(loader, desc="Encoding images"):
            images = images.to(device)
            feats = model.encode_image(images)
            feats = F.normalize(feats, dim=-1)
            features.append(feats.cpu().numpy())
            labels.append(lbls.numpy())
            paths.extend(batch_paths)

    feature_array = np.concatenate(features, axis=0)
    label_array = np.concatenate(labels, axis=0)
    return feature_array, label_array, paths


def _run_pca(features: np.ndarray):
    reducer = PCA(n_components=2, random_state=0)
    return reducer.fit_transform(features)


def _run_tsne(features: np.ndarray, perplexity: float, iterations: int, seed: int):
    max_perplexity = max(5.0, (features.shape[0] - 1) / 3)
    effective_perplexity = min(perplexity, max_perplexity)
    reducer = TSNE(
        n_components=2,
        learning_rate="auto",
        init="pca",
        perplexity=effective_perplexity,
        n_iter=iterations,
        random_state=seed,
    )
    return reducer.fit_transform(features)


def _plot_embedding(coords, labels, class_names, title, output_path):
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", len(unique_labels))

    for idx, label_id in enumerate(unique_labels):
        mask = labels == label_id
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            label=class_names[label_id],
            s=12,
            alpha=0.75,
            color=cmap(idx),
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(markerscale=1.5, fontsize=8, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def _write_projection_csv(path, coords, labels, class_names, image_paths):
    with open(path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["x", "y", "label_id", "label_name", "image_path"])
        for (x_val, y_val), label_id, img_path in zip(coords, labels, image_paths):
            writer.writerow([x_val, y_val, label_id, class_names[label_id], img_path])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute BioCLIP embeddings for the training data and run PCA / t-SNE.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Path to the prepared dataset root (should contain train/valid/test).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (train/valid/test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "plots" / "bioclip_embeddings",
        help="Directory where plots and embeddings will be saved.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for BioCLIP inference.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="hf-hub:imageomics/bioclip",
        help="Identifier passed to open_clip.create_model_and_transforms.",
    )
    parser.add_argument(
        "--per-class-limit",
        type=int,
        default=200,
        help="Maximum number of images sampled per class (set -1 for no limit).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2000,
        help="Global cap on the number of images processed (set -1 for no limit).",
    )
    parser.add_argument(
        "--skip-pca",
        action="store_true",
        help="Skip PCA projection.",
    )
    parser.add_argument(
        "--skip-tsne",
        action="store_true",
        help="Skip t-SNE projection.",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=30.0,
        help="Base perplexity for t-SNE.",
    )
    parser.add_argument(
        "--tsne-iterations",
        type=int,
        default=1000,
        help="Number of optimization steps for t-SNE.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when sampling images.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda, cpu, or auto).",
    )
    parser.add_argument(
        "--save-embeddings",
        action="store_true",
        help="Persist raw embeddings/labels/paths to an .npz archive.",
    )
    parser.add_argument(
        "--bioclip-src",
        type=Path,
        default=DEFAULT_BIOCLIP_SRC,
        help="Path to the BioCLIP repository's src directory for importing open_clip.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_dir = (args.data_root / args.split).resolve()
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset split not found: {dataset_dir}")

    per_class_limit = None if args.per_class_limit == -1 else args.per_class_limit
    max_images = None if args.max_images == -1 else args.max_images
    args.output_dir.mkdir(parents=True, exist_ok=True)

    open_clip = _import_open_clip(args.bioclip_src)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    print(f"Loading dataset from {dataset_dir}")
    items, class_names = _collect_images(
        dataset_dir,
        per_class_limit=per_class_limit,
        max_images=max_images,
        seed=args.seed,
    )
    print(f"Sampling {len(items)} images across {len(class_names)} classes.")

    print(f"Loading BioCLIP model ({args.model_id}) on device {device}.")
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_id,
        device=device,
    )

    loader = _build_dataloader(
        items,
        transform=preprocess,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    features, label_ids, image_paths = _extract_embeddings(model, device, loader)
    print(f"Extracted embeddings with shape {features.shape}.")

    if args.save_embeddings:
        embedding_path = args.output_dir / f"{args.split}_bioclip_embeddings.npz"
        np.savez_compressed(
            embedding_path,
            features=features,
            labels=label_ids,
            image_paths=np.array(image_paths),
            class_names=np.array(class_names),
        )
        print(f"Saved raw embeddings to {embedding_path}.")

    if not args.skip_pca:
        print("Running PCA...")
        pca_coords = _run_pca(features)
        pca_plot_path = args.output_dir / f"{args.split}_pca.png"
        _plot_embedding(
            pca_coords,
            label_ids,
            class_names,
            title=f"BioCLIP PCA ({args.split})",
            output_path=pca_plot_path,
        )
        _write_projection_csv(
            args.output_dir / f"{args.split}_pca.csv",
            pca_coords,
            label_ids,
            class_names,
            image_paths,
        )
        print(f"PCA artifacts written to {pca_plot_path}")
    else:
        pca_coords = None

    if not args.skip_tsne:
        print("Running t-SNE (this may take a few minutes)...")
        tsne_coords = _run_tsne(
            features,
            perplexity=args.tsne_perplexity,
            iterations=args.tsne_iterations,
            seed=args.seed,
        )
        tsne_plot_path = args.output_dir / f"{args.split}_tsne.png"
        _plot_embedding(
            tsne_coords,
            label_ids,
            class_names,
            title=f"BioCLIP t-SNE ({args.split})",
            output_path=tsne_plot_path,
        )
        _write_projection_csv(
            args.output_dir / f"{args.split}_tsne.csv",
            tsne_coords,
            label_ids,
            class_names,
            image_paths,
        )
        print(f"t-SNE artifacts written to {tsne_plot_path}")

    print("Done.")


if __name__ == "__main__":
    main()
