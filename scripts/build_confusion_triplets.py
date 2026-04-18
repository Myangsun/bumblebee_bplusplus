#!/usr/bin/env python3
"""
Task 1 T1.7 — confusion-pair triplets.

For each (rare target, primary confuser) pair defined by the baseline
confusion matrix, render a three-column figure:

    column A — real images of the rare target species
    column B — synthetic images of the rare target species
    column C — real images of the confuser species

This asks: do the synthetic target images visually drift toward the confuser,
explaining why augmentation frequently mis-routes rare-species test images
to the confuser class?

Primary confuser pairs (from baseline confusion matrix, see §5 of thesis):
    B. ashtoni     ↔ B. citrinus, B. vagans_Smith
    B. sandersoni  ↔ B. vagans_Smith
    B. flavidus    ↔ B. citrinus

Outputs:
    docs/plots/failure/confusion_triplet_{target}__{confuser}.png
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from pipeline.config import GBIF_DATA_DIR, PROJECT_ROOT

RARE_CONFUSERS: Dict[str, Tuple[str, ...]] = {
    "Bombus_ashtoni":    ("Bombus_citrinus", "Bombus_vagans_Smith"),
    "Bombus_sandersoni": ("Bombus_vagans_Smith",),
    "Bombus_flavidus":   ("Bombus_citrinus",),
}
REAL_TRAIN_ROOT = GBIF_DATA_DIR / "prepared_split" / "train"
SYNTHETIC_ROOT = PROJECT_ROOT / "RESULTS_kfold" / "synthetic_generation"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "plots" / "failure"
SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
THUMB_SIDE = 280


def _list_images(directory: Path) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUPPORTED_EXTS)


def _load_thumb(path: Path, side: int = THUMB_SIDE) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((side, side), Image.LANCZOS)
        return np.asarray(img)


def _short(sp: str) -> str:
    return sp.replace("Bombus_", "B. ")


def render_triplet(target: str, confuser: str,
                   target_real: List[Path], target_synth: List[Path],
                   confuser_real: List[Path],
                   output_path: Path, n_per_col: int,
                   rng: random.Random) -> None:
    target_real_sample = rng.sample(target_real, min(n_per_col, len(target_real)))
    target_synth_sample = rng.sample(target_synth, min(n_per_col, len(target_synth)))
    confuser_sample = rng.sample(confuser_real, min(n_per_col, len(confuser_real)))

    n_rows = max(len(target_real_sample), len(target_synth_sample),
                 len(confuser_sample), 1)

    fig, axes = plt.subplots(n_rows, 3, figsize=(9.5, 3.0 * n_rows), squeeze=False)
    col_headers = [
        f"target REAL — {_short(target)}\n(n={len(target_real)})",
        f"target SYNTHETIC — {_short(target)}\n(n={len(target_synth)})",
        f"confuser REAL — {_short(confuser)}\n(n={len(confuser_real)})",
    ]
    col_edge = ["#2ca02c", "#1f77b4", "#d62728"]

    for col_i, (sources, header, edge) in enumerate(zip(
            (target_real_sample, target_synth_sample, confuser_sample),
            col_headers, col_edge)):
        for row_i in range(n_rows):
            ax = axes[row_i, col_i]
            ax.set_xticks([]); ax.set_yticks([])
            if row_i >= len(sources):
                ax.axis("off")
                continue
            try:
                ax.imshow(_load_thumb(sources[row_i]))
            except Exception:
                ax.text(0.5, 0.5, "missing", ha="center", va="center",
                        transform=ax.transAxes, fontsize=7)
                continue
            for spine in ax.spines.values():
                spine.set_color(edge); spine.set_linewidth(1.6)
            if row_i == 0:
                ax.set_title(header, fontsize=10)

    fig.suptitle(f"Confusion triplet — {_short(target)} (target) vs "
                 f"{_short(confuser)} (primary confuser)\n"
                 "Do synthetic target images visually lie closer to real target "
                 "or real confuser?",
                 fontsize=10.5, y=0.995)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {output_path.relative_to(PROJECT_ROOT)}  "
          f"({len(target_real_sample)} / {len(target_synth_sample)} / {len(confuser_sample)})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-per-col", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    for target, confusers in RARE_CONFUSERS.items():
        target_real = _list_images(REAL_TRAIN_ROOT / target)
        target_synth = _list_images(SYNTHETIC_ROOT / target)
        if not target_real or not target_synth:
            print(f"[skip] {target}: missing real or synthetic images")
            continue
        for confuser in confusers:
            confuser_real = _list_images(REAL_TRAIN_ROOT / confuser)
            if not confuser_real:
                print(f"[skip] {target} × {confuser}: missing confuser images")
                continue
            short_t = target.replace("Bombus_", "")
            short_c = confuser.replace("Bombus_", "")
            out = args.output_dir / f"confusion_triplet_{short_t}__{short_c}.png"
            render_triplet(target, confuser, target_real, target_synth,
                           confuser_real, out, args.n_per_col, rng)

    print("Done.")


if __name__ == "__main__":
    main()
