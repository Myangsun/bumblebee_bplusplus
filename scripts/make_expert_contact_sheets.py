#!/usr/bin/env python3
"""Per-species expert-label contact sheets.

Mirrors the LLM judge contact sheets at
RESULTS_kfold/llm_judge_eval/contact_Bombus_<species>.png but uses the
expert labels from RESULTS/expert_validation_results/jessie_all_150.csv.

For each rare species, produces a PNG with:
  - Top grid (green header):  STRICT PASS images, sorted by expert morph
                              mean descending.
  - Bottom grid (red header): FAIL images (not strict pass), sorted by
                              expert morph mean ascending.

Output: docs/plots/filters/contact_expert_Bombus_<species>.png
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("/home/msun14/bumblebee_bplusplus")
CSV = ROOT / "RESULTS/expert_validation_results/jessie_all_150.csv"
SYNTH_DIR = ROOT / "RESULTS_kfold/synthetic_generation"
OUT_DIR = ROOT / "docs/plots/filters"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RARE = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]

MORPH_COLS = [
    "morph_legs_appendages", "morph_wing_venation_texture",
    "morph_head_antennae", "morph_abdomen_banding", "morph_thorax_coloration",
]
STRUCTURAL_CODES = {
    "extra_limbs", "missing_limbs", "impossible_geometry",
    "visible_artifact", "visible_artifacts", "blurry_artifacts",
    "repetitive_patterns",
}


def load_expert() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    df["ground_truth_species"] = df["ground_truth_species"].str.replace(" ", "_", regex=False)
    df["blind_id_species"] = df["blind_id_species"].fillna("").str.replace(" ", "_", regex=False)
    df["failure_dict"] = df["failure_modes"].apply(
        lambda s: json.loads(s) if isinstance(s, str) else {}
    )
    df["expert_morph_mean"] = df[MORPH_COLS].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    df["has_structural"] = df["failure_dict"].apply(
        lambda d: bool(set(d.get("all") or []) & STRUCTURAL_CODES)
    )
    df["strict"] = (
        (df["blind_id_species"] == df["ground_truth_species"])
        & (df["diagnostic_level"] == "species")
        & (df["expert_morph_mean"] >= 4.0)
    )
    df["basename"] = df["image_path"].str.split("/").str[-1]
    return df


def _load_thumb(basename: str, species: str, width: int) -> Image.Image | None:
    path = SYNTH_DIR / species / basename
    if not path.exists():
        return None
    img = Image.open(path).convert("RGB")
    w, h = img.size
    new_h = int(h * (width / w))
    return img.resize((width, new_h), Image.LANCZOS)


def _font(size: int, bold: bool = False):
    paths = [
        "/usr/share/fonts/liberation-sans/LiberationSans-"
        + ("Bold" if bold else "Regular") + ".ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans"
        + ("-Bold" if bold else "") + ".ttf",
    ]
    for p in paths:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def make_sheet(species: str, rows_df: pd.DataFrame, cols: int = 5, rows: int = 4,
               thumb_w: int = 200):
    sp_short = species.replace("Bombus_", "B. ")

    pass_df = rows_df[rows_df.strict].sort_values("expert_morph_mean", ascending=False)
    fail_df = rows_df[~rows_df.strict].sort_values("expert_morph_mean", ascending=True)

    per_grid = cols * rows
    pass_items = list(pass_df.itertuples(index=False))[:per_grid]
    fail_items = list(fail_df.itertuples(index=False))[:per_grid]

    # Load thumbnails.
    pass_thumbs = [(_load_thumb(r.basename, species, thumb_w), r) for r in pass_items]
    fail_thumbs = [(_load_thumb(r.basename, species, thumb_w), r) for r in fail_items]
    pass_thumbs = [(t, r) for t, r in pass_thumbs if t is not None]
    fail_thumbs = [(t, r) for t, r in fail_thumbs if t is not None]
    if not pass_thumbs and not fail_thumbs:
        print(f"  [{species}] no thumbnails found — skipping")
        return

    all_thumbs = pass_thumbs + fail_thumbs
    max_h = max(t.height for t, _ in all_thumbs)

    pad = 6
    title_h = 44
    label_h = 32
    caption_h = 18
    cell_h = max_h + caption_h

    grid_h = rows * (cell_h + pad) + pad
    total_w = cols * (thumb_w + pad) + pad
    total_h = title_h + label_h + grid_h + 12 + label_h + grid_h + 16

    sheet = Image.new("RGB", (total_w, total_h), (249, 249, 247))
    draw = ImageDraw.Draw(sheet)
    f_title = _font(20, bold=True)
    f_label = _font(15, bold=True)
    f_cap = _font(11)

    # Title.
    n_pass = int(pass_df.shape[0])
    n_fail = int(fail_df.shape[0])
    title = (f"{sp_short} — Expert label contact sheet  "
             f"(strict pass = {n_pass} / {len(rows_df)}, fail = {n_fail} / {len(rows_df)})")
    draw.text((pad, 10), title, fill=(30, 30, 30), font=f_title)

    def draw_grid(thumbs, y0, label, color):
        draw.rectangle([(0, y0), (total_w, y0 + label_h)], fill=color)
        draw.text((pad, y0 + 8), label, fill=(255, 255, 255), font=f_label)
        y_grid = y0 + label_h
        for idx, (thumb, r) in enumerate(thumbs):
            ri, ci = idx // cols, idx % cols
            if ri >= rows:
                break
            x = pad + ci * (thumb_w + pad)
            y = y_grid + pad + ri * (cell_h + pad)
            sheet.paste(thumb, (x, y))
            # Per-image caption line: blind-id match flag + expert morph mean.
            blind_ok = "✓" if r.blind_id_species == r.ground_truth_species else "✗"
            cap = f"m={r.expert_morph_mean:.1f}  blind-ID {blind_ok}  diag={r.diagnostic_level}"
            draw.text((x + 2, y + thumb.height + 2), cap, fill=(60, 60, 60), font=f_cap)

    y_pass = title_h
    n_show_pass = len(pass_thumbs)
    draw_grid(pass_thumbs, y_pass,
              f"EXPERT STRICT PASS  ({n_show_pass} of {n_pass} shown)",
              (134, 170, 152))  # #86aa98

    y_fail = title_h + label_h + grid_h + 12
    n_show_fail = len(fail_thumbs)
    draw_grid(fail_thumbs, y_fail,
              f"EXPERT FAIL  ({n_show_fail} of {n_fail} shown, worst morph first)",
              (216, 106, 106))  # #d86a6a

    out = OUT_DIR / f"contact_expert_{species}.png"
    sheet.save(out, quality=92)
    print(f"saved: {out}")


def main():
    df = load_expert()
    for sp in RARE:
        sub = df[df.ground_truth_species == sp]
        make_sheet(sp, sub)


if __name__ == "__main__":
    main()
