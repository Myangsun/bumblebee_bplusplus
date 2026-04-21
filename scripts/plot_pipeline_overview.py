#!/usr/bin/env python3
"""Render a 3×8 figure showing, per rare species:
original train image | cutout | CNP | synthetic @ 5 angles.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import GBIF_DATA_DIR, PROJECT_ROOT

SPECIES = ("Bombus_ashtoni", "Bombus_flavidus", "Bombus_sandersoni")
ANGLES = ("lateral", "dorsal", "three-quarter_anterior", "three-quarter_posterior", "frontal")

TRAIN_ROOT = GBIF_DATA_DIR / "prepared" / "train"
CUTOUT_ROOT = PROJECT_ROOT / "CACHE_CNP" / "cutouts"
CNP_ROOT = PROJECT_ROOT / "RESULTS_kfold" / "cnp_generation" / "train"
SYNTH_ROOT = PROJECT_ROOT / "RESULTS_kfold" / "synthetic_generation"

OUT_DIR = PROJECT_ROOT / "docs" / "plots"
OUT_PNG = OUT_DIR / "pipeline_overview.png"
OUT_PDF = OUT_DIR / "pipeline_overview.pdf"

THUMB = 300
ID_RE = re.compile(r"Bombus_\w+?(\d+)")


def image_id(path: Path) -> str | None:
    m = ID_RE.search(path.stem)
    return m.group(1) if m else None


def index_by_id(paths):
    idx = {}
    for p in paths:
        iid = image_id(p)
        if iid and iid not in idx:
            idx[iid] = p
    return idx


def pick_triplet(species: str):
    train_imgs = sorted(TRAIN_ROOT.joinpath(species).glob("*.jpg"))
    cutouts = index_by_id(sorted(CUTOUT_ROOT.joinpath(species).glob("*.png")))
    cnps = index_by_id(sorted(CNP_ROOT.joinpath(species).glob("*.png")))
    for img in train_imgs:
        iid = image_id(img)
        if iid in cutouts and iid in cnps:
            return img, cutouts[iid], cnps[iid]
    raise RuntimeError(f"No matching original/cutout/CNP triplet for {species}")


def pick_synthetic(species: str, angle: str) -> Path:
    matches = sorted(SYNTH_ROOT.joinpath(species).glob(f"*::{angle}_*.jpg"))
    if not matches:
        raise RuntimeError(f"No synthetic {angle} for {species}")
    return matches[0]


def load_thumb(path: Path) -> Image.Image:
    img = Image.open(path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    img.thumbnail((THUMB, THUMB), Image.LANCZOS)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    col_titles = [
        "Original (train)",
        "Cutout",
        "CNP",
        *[a.replace("_", " ").replace("three-quarter", "3/4") for a in ANGLES],
    ]
    n_rows, n_cols = len(SPECIES), len(col_titles)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(1.8 * n_cols, 1.9 * n_rows + 0.4),
        gridspec_kw={"wspace": 0.05, "hspace": 0.05},
    )

    for r, species in enumerate(SPECIES):
        orig, cutout, cnp = pick_triplet(species)
        tiles = [orig, cutout, cnp] + [pick_synthetic(species, a) for a in ANGLES]
        for c, tile_path in enumerate(tiles):
            ax = axes[r, c]
            ax.imshow(load_thumb(tile_path))
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(col_titles[c], fontsize=10)
            if c == 0:
                pretty = species.replace("Bombus_", "B. ")
                ax.set_ylabel(pretty, fontsize=11, style="italic",
                              rotation=90, labelpad=8)

    fig.suptitle(
        "Pipeline inputs per rare species: real photo → cutout → CNP → synthetic (5 angles)",
        fontsize=12, y=0.995,
    )
    fig.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    print(f"Wrote {OUT_PNG}")
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
