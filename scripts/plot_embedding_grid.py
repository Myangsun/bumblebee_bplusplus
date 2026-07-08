#!/usr/bin/env python3
"""6-panel grid of embedding overviews — {BioCLIP, DINOv2} × {PCA, t-SNE, UMAP}.

Composes the six pre-rendered `embeddings_overview.png` files (one per
backbone × projection pair) into a single 2 × 3 grid figure for Appendix F.
Each cell shows all 16 species in the chosen embedding; row labels are the
backbone, column labels are the projection.

Output: docs/plots/embeddings/backbone_projection_grid.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT  # noqa: E402

EMB_DIR = PROJECT_ROOT / "docs/plots/embeddings"
OUT = EMB_DIR / "backbone_projection_grid"

BACKBONES = ["bioclip", "dinov2"]
BACKBONE_LABELS = {"bioclip": "BioCLIP ViT-B/16", "dinov2": "DINOv2 ViT-L/14"}
PROJECTIONS = ["pca", "tsne", "umap"]
PROJECTION_LABELS = {"pca": "PCA", "tsne": "t-SNE", "umap": "UMAP"}


def main() -> None:
    fig, axes = plt.subplots(
        len(BACKBONES), len(PROJECTIONS),
        figsize=(15, 9.5), facecolor="white",
    )
    for i, backbone in enumerate(BACKBONES):
        for j, projection in enumerate(PROJECTIONS):
            ax = axes[i, j]
            path = EMB_DIR / f"{backbone}_{projection}" / "embeddings_overview.png"
            if not path.exists():
                ax.text(0.5, 0.5, f"missing\n{path.name}", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#888")
                ax.set_axis_off()
                continue
            img = Image.open(path).convert("RGB")
            ax.imshow(img)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(PROJECTION_LABELS[projection], fontsize=14,
                             fontweight="bold", pad=10)
            if j == 0:
                ax.text(-0.04, 0.5, BACKBONE_LABELS[backbone], rotation=90,
                        ha="right", va="center", fontsize=13, fontweight="bold",
                        transform=ax.transAxes)

    fig.suptitle(
        "Embedding overviews — backbone × projection grid (16 Massachusetts Bombus species)",
        fontsize=14, y=0.995,
    )
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    fig.savefig(f"{OUT}.png", dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{OUT}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}.png")
    print(f"wrote {OUT}.pdf")


if __name__ == "__main__":
    main()
