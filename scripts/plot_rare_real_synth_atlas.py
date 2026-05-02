#!/usr/bin/env python3
"""All-image variant of the rare-species real+synthetic tSNE atlas.

Same tSNE projection (rare real + rare synthetic, BioCLIP features,
perplexity 20, seed 42) used by ``plot_rare_real_synth_anchored.py``,
but instead of overlaying 9 anchor thumbnails, this version uses 2-D
grid binning to spread thumbnails of as many real and synthetic images
as can fit without overlap. Real frames are solid species colour,
synthetic frames are dashed species colour.

Output:
  docs/plots/embeddings/bioclip_tsne/embeddings_rare_real_synth_atlas.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT  # noqa: E402
from pipeline.evaluate.embeddings import load_cache  # noqa: E402

CACHE_DIR = PROJECT_ROOT / "RESULTS/embeddings"
OUT_DIR = PROJECT_ROOT / "docs/plots/embeddings/bioclip_tsne"

RARE = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")
SPECIES_PALETTE = {
    "Bombus_ashtoni":    "#0072B2",
    "Bombus_sandersoni": "#E69F00",
    "Bombus_flavidus":   "#009E73",
}

# Tuned to the ~1700-point rare pool. Larger grid = more thumbnails (smaller).
GRID = 22
THUMB_PX = 96
ZOOM = 0.35


def _square_thumb(path: Path, side: int = THUMB_PX) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((side, side))
    return np.asarray(img)


def _resolve_path(p: str) -> Path:
    cand = Path(p)
    if cand.exists():
        return cand
    s = str(p)
    for marker in ("GBIF_MA_BUMBLEBEES", "RESULTS_kfold", "RESULTS"):
        if marker in s:
            tail = s[s.index(marker):]
            local = PROJECT_ROOT / tail
            if local.exists():
                return local
    return cand


def grid_sample(coords: np.ndarray, grid: int) -> np.ndarray:
    x_min, x_max = float(coords[:, 0].min()), float(coords[:, 0].max())
    y_min, y_max = float(coords[:, 1].min()), float(coords[:, 1].max())
    cell_w = (x_max - x_min) / grid
    cell_h = (y_max - y_min) / grid
    if cell_w == 0 or cell_h == 0:
        return np.array([], dtype=int)
    col_idx = np.clip(((coords[:, 0] - x_min) / cell_w).astype(int), 0, grid - 1)
    row_idx = np.clip(((coords[:, 1] - y_min) / cell_h).astype(int), 0, grid - 1)
    chosen = []
    for gy in range(grid):
        for gx in range(grid):
            mask = (col_idx == gx) & (row_idx == gy)
            members = np.where(mask)[0]
            if members.size == 0:
                continue
            cx = x_min + (gx + 0.5) * cell_w
            cy = y_min + (gy + 0.5) * cell_h
            d2 = (coords[members, 0] - cx) ** 2 + (coords[members, 1] - cy) ** 2
            chosen.append(int(members[int(np.argmin(d2))]))
    return np.array(chosen, dtype=int)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    real = load_cache(CACHE_DIR / "bioclip_real_train.npz")
    synth = load_cache(CACHE_DIR / "bioclip_synthetic.npz")

    real_mask = np.isin(real["species"], list(RARE))
    real_feats = real["features"][real_mask]
    real_paths = real["image_paths"][real_mask]
    real_sp = real["species"][real_mask]

    synth_feats = synth["features"]
    synth_paths = synth["image_paths"]
    synth_sp = synth["species"]

    combined = np.concatenate([real_feats, synth_feats], axis=0)
    n_real = real_feats.shape[0]
    perp = min(20.0, max(5.0, (combined.shape[0] - 1) / 3))
    print(f"[atlas] tSNE on {combined.shape[0]} pts (perplexity={perp:.1f})...")
    coords = TSNE(n_components=2, init="pca", learning_rate="auto",
                  perplexity=perp, random_state=42).fit_transform(combined)
    coords_real, coords_synth = coords[:n_real], coords[n_real:]

    fig, ax = plt.subplots(figsize=(13, 10), facecolor="white")

    # Faint background scatter for context.
    for sp in RARE:
        m = real_sp == sp
        if m.any():
            ax.scatter(coords_real[m, 0], coords_real[m, 1],
                       c=[SPECIES_PALETTE[sp]], s=14, alpha=0.30,
                       marker="o", edgecolors="white", linewidths=0.2,
                       label=f"real {sp.replace('Bombus_', 'B. ')} (n={int(m.sum())})")
    for sp in RARE:
        m = synth_sp == sp
        if m.any():
            ax.scatter(coords_synth[m, 0], coords_synth[m, 1],
                       c=[SPECIES_PALETTE[sp]], s=14, alpha=0.25,
                       marker="X", edgecolors="black", linewidths=0.2,
                       label=f"synth {sp.replace('Bombus_', 'B. ')} (n={int(m.sum())})")

    # Grid-sampled thumbnails over the joint coordinate grid.
    sample_idx = grid_sample(coords, GRID)
    n_thumbs = 0
    for idx in sample_idx:
        if idx < n_real:
            xy = coords_real[idx]
            sp = str(real_sp[idx])
            path = _resolve_path(str(real_paths[idx]))
            ls = "solid"
        else:
            j = idx - n_real
            xy = coords_synth[j]
            sp = str(synth_sp[j])
            path = _resolve_path(str(synth_paths[j]))
            ls = "dashed"
        col = SPECIES_PALETTE.get(sp, "#777")
        try:
            arr = _square_thumb(path)
        except Exception as exc:
            print(f"  thumbnail fail {path}: {exc}")
            continue
        ab = AnnotationBbox(
            OffsetImage(arr, zoom=ZOOM),
            tuple(xy), frameon=True, pad=0.10, zorder=6,
            bboxprops=dict(edgecolor=col, lw=1.6, linestyle=ls),
        )
        ax.add_artist(ab)
        n_thumbs += 1
    print(f"  rendered {n_thumbs} thumbnails (grid={GRID}×{GRID})")

    ax.set_xlabel("TSNE-1")
    ax.set_ylabel("TSNE-2")
    ax.set_title("BIOCLIP tSNE — rare species real (●) + synthetic (✕)\n"
                 "All-image atlas: every grid cell shows the closest real or "
                 "synthetic image. Real frames solid; synthetic frames dashed; "
                 "frame colour = species.",
                 fontsize=11)
    ax.legend(markerscale=1.4, fontsize=8.5, frameon=False,
              loc="center left", bbox_to_anchor=(1.01, 0.5))

    fig.tight_layout()
    out_png = OUT_DIR / "embeddings_rare_real_synth_atlas.png"
    out_pdf = OUT_DIR / "embeddings_rare_real_synth_atlas.pdf"
    fig.savefig(out_png, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
