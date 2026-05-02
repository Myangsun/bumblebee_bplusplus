#!/usr/bin/env python3
"""Combined real+synthetic tSNE atlas with one anchored real-vs-synthetic
triplet per rare species.

Per rare species (B. ashtoni, B. sandersoni, B. flavidus):
  - 1 real anchor: the real-training image closest to that species' real
    centroid (most canonical example).
  - 2 synthetic samples: the closest synthetic to the real anchor (cosine),
    and the farthest synthetic of the same species.

Background scatter is the same rare-real + rare-synth tSNE used in
docs/plots/embeddings/bioclip_tsne/embeddings_rare_real_synth.png. On top
we overlay the 3 thumbnails per species at their actual tSNE coordinates
and connect the real anchor to each of its two paired synthetics with a
line annotated with cosine distance.

Outputs:
  docs/plots/embeddings/bioclip_tsne/embeddings_rare_real_synth_anchored.{png,pdf}
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
THUMB_PX = 96  # crop side in pixels


def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def _square_thumb(path: Path, side: int = THUMB_PX) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((side, side))
    return np.asarray(img)


def _resolve_path(p: str) -> Path:
    """Caches use absolute paths recorded at extraction time on a different
    cluster mount. Re-resolve into the local PROJECT_ROOT if the literal path
    no longer exists."""
    cand = Path(p)
    if cand.exists():
        return cand
    # Strip any leading mount prefix and graft onto PROJECT_ROOT.
    s = str(p)
    for marker in ("GBIF_MA_BUMBLEBEES", "RESULTS_kfold", "RESULTS"):
        if marker in s:
            tail = s[s.index(marker):]
            local = PROJECT_ROOT / tail
            if local.exists():
                return local
    return cand  # may not exist; caller handles


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

    # tSNE on the same combined pool as the source figure.
    combined = np.concatenate([real_feats, synth_feats], axis=0)
    n_real = real_feats.shape[0]
    perp = min(20.0, max(5.0, (combined.shape[0] - 1) / 3))
    print(f"[anchored-tsne] fitting on {combined.shape[0]} pts (perplexity={perp:.1f})...")
    coords = TSNE(n_components=2, init="pca", learning_rate="auto",
                  perplexity=perp, random_state=42).fit_transform(combined)
    coords_real, coords_synth = coords[:n_real], coords[n_real:]

    # ── Pick anchors and pair samples per species ────────────────────────────
    triplets = []  # list of dicts per species
    for sp in RARE:
        rm = real_sp == sp
        sm = synth_sp == sp
        rfeat = _l2(real_feats[rm])
        sfeat = _l2(synth_feats[sm])
        rpath = real_paths[rm]
        spath = synth_paths[sm]

        centroid = _l2(rfeat.mean(axis=0, keepdims=True))[0]
        # Real anchor = real image closest to its own species centroid.
        anchor_idx_local = int(np.argmax(rfeat @ centroid))
        anchor_feat = rfeat[anchor_idx_local]

        # Synth nearest / farthest by cosine distance to the real anchor.
        sims = sfeat @ anchor_feat
        nearest_local = int(np.argmax(sims))
        farthest_local = int(np.argmin(sims))
        d_near = float(1.0 - sims[nearest_local])
        d_far = float(1.0 - sims[farthest_local])

        # Re-locate these picks in the global tSNE coordinate arrays.
        real_global_idx = int(np.flatnonzero(rm)[anchor_idx_local])
        synth_global_idx_near = int(np.flatnonzero(sm)[nearest_local])
        synth_global_idx_far = int(np.flatnonzero(sm)[farthest_local])
        triplets.append({
            "species": sp,
            "real_xy": coords_real[real_global_idx],
            "near_xy": coords_synth[synth_global_idx_near],
            "far_xy":  coords_synth[synth_global_idx_far],
            "real_path": _resolve_path(str(rpath[anchor_idx_local])),
            "near_path": _resolve_path(str(spath[nearest_local])),
            "far_path":  _resolve_path(str(spath[farthest_local])),
            "d_near": d_near,
            "d_far":  d_far,
        })
        print(f"  {sp}: d_near={d_near:.3f}  d_far={d_far:.3f}  "
              f"anchor={Path(rpath[anchor_idx_local]).name}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8.5))

    # Background scatter — keep faint so the thumbnails carry the eye.
    for sp in RARE:
        m = real_sp == sp
        if m.any():
            ax.scatter(coords_real[m, 0], coords_real[m, 1],
                       c=[SPECIES_PALETTE[sp]], s=22, alpha=0.45,
                       marker="o", edgecolors="white", linewidths=0.3,
                       label=f"real {sp.replace('Bombus_', 'B. ')} (n={int(m.sum())})")
    for sp in RARE:
        m = synth_sp == sp
        if m.any():
            ax.scatter(coords_synth[m, 0], coords_synth[m, 1],
                       c=[SPECIES_PALETTE[sp]], s=22, alpha=0.35,
                       marker="X", edgecolors="black", linewidths=0.3,
                       label=f"synth {sp.replace('Bombus_', 'B. ')} (n={int(m.sum())})")

    # Triplets: thin connector lines + thumbnails (solid frame for real,
    # dashed frame for synthetic). All frames use the species colour.
    for t in triplets:
        col = SPECIES_PALETTE[t["species"]]
        rx, ry = t["real_xy"]
        nx, ny = t["near_xy"]
        fx, fy = t["far_xy"]
        # Thin connectors, real → each synthetic.
        for (sx, sy) in [(nx, ny), (fx, fy)]:
            ax.plot([rx, sx], [ry, sy], color=col, linewidth=0.8,
                    alpha=0.85, zorder=4)

        # Three thumbnails: real (solid species-colour frame), nearest synth
        # and farthest synth (dashed species-colour frames).
        for (xy, path, ls) in [
            ((rx, ry), t["real_path"], "solid"),
            ((nx, ny), t["near_path"], "dashed"),
            ((fx, fy), t["far_path"],  "dashed"),
        ]:
            try:
                arr = _square_thumb(path)
            except Exception as exc:
                print(f"  thumbnail fail {path}: {exc}")
                continue
            ab = AnnotationBbox(
                OffsetImage(arr, zoom=0.55),
                xy, frameon=True, pad=0.18, zorder=6,
                bboxprops=dict(edgecolor=col, lw=2.2, linestyle=ls),
            )
            ax.add_artist(ab)

    ax.set_xlabel("TSNE-1")
    ax.set_ylabel("TSNE-2")
    ax.set_title("BIOCLIP tSNE — rare species real (●) + synthetic (✕)\n"
                 "Per species: 1 canonical real anchor (solid frame) connected to "
                 "its closest and farthest synthetic of the same species (dashed frames).",
                 fontsize=11)
    # Two compact legends: scatter + line styles.
    ax.legend(markerscale=1.1, fontsize=8.5, frameon=False,
              loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    out_png = OUT_DIR / "embeddings_rare_real_synth_anchored.png"
    out_pdf = OUT_DIR / "embeddings_rare_real_synth_anchored.pdf"
    fig.savefig(out_png, dpi=240, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
