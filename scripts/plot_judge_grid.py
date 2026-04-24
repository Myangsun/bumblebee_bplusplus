#!/usr/bin/env python3
"""
Plot grids of PASS vs FAIL synthetic images per species, based on LLM judge results.

Creates:
  1. Pass grid (random sample)
  2. Fail grid (sorted by mean score, worst first)
  3. Failure analysis chart (pass rate by angle, feature scores, blind ID, background)

Usage:
    python scripts/plot_judge_grid.py
    python scripts/plot_judge_grid.py --rows 4 --cols 5
    python scripts/plot_judge_grid.py --species Bombus_ashtoni
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from pipeline.config import RESULTS_DIR

DEFAULT_SYNTH_DIR = RESULTS_DIR / "synthetic_generation"
DEFAULT_JUDGE_RESULTS = RESULTS_DIR / "llm_judge_eval" / "results.json"

# Module-level references set by main() for use in helper functions
SYNTH_DIR = DEFAULT_SYNTH_DIR
JUDGE_RESULTS = DEFAULT_JUDGE_RESULTS

ANGLE_ORDER = ["lateral", "dorsal", "three-quarter_anterior", "three-quarter_posterior", "frontal"]
ANGLE_LABELS = ["Lateral", "Dorsal", "3/4 Anterior", "3/4 Posterior", "Frontal"]
FEATURE_ORDER = ["abdomen_banding", "thorax_coloration", "head_antennae", "legs_appendages", "wing_venation_texture"]
FEATURE_LABELS = ["Abdomen\nBanding", "Thorax\nColor", "Head\nAntennae", "Legs\nAppend.", "Wing\nTexture"]
BG_ORDER = ["yellow_flowers", "pink_purple_flowers", "green_foliage", "mixed_other"]
BG_LABELS = ["Yellow\nFlowers", "Pink/Purple\nFlowers", "Green\nFoliage", "Mixed/\nOther"]
BG_COLORS_MAP = {"yellow_flowers": "#f1c40f", "pink_purple_flowers": "#e91e9e",
                  "green_foliage": "#27ae60", "mixed_other": "#95a5a6"}


def strict_pass(r: dict, min_score: float = 4.0) -> bool:
    if not r.get("blind_identification", {}).get("matches_target", False):
        return False
    if r.get("diagnostic_completeness", {}).get("level") != "species":
        return False
    morph = r.get("morphological_fidelity", {})
    scores = [v["score"] for v in morph.values()
              if isinstance(v, dict) and "score" in v and not v.get("not_visible", False)]
    if not scores or (sum(scores) / len(scores)) < min_score:
        return False
    return True


def mean_morph_score(r: dict) -> float:
    morph = r.get("morphological_fidelity", {})
    scores = [v["score"] for v in morph.values()
              if isinstance(v, dict) and "score" in v and not v.get("not_visible", False)]
    return sum(scores) / len(scores) if scores else 0


def parse_angle(filename: str) -> str:
    m = re.match(r"^.+?::\d+::(.+?)_\d+\.jpg$", filename)
    return m.group(1) if m else "unknown"


def classify_background(img_path: Path) -> str:
    """Classify dominant background color from border pixels."""
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        h, w = arr.shape[:2]
        border = np.concatenate([
            arr[:h//7, :, :].reshape(-1, 3),
            arr[-h//7:, :, :].reshape(-1, 3),
            arr[:, :w//7, :].reshape(-1, 3),
            arr[:, -w//7:, :].reshape(-1, 3),
        ])
        r, g, b = border.mean(axis=0)
        if r > 140 and g > 130 and b < 100 and r > b * 1.5:
            return "yellow_flowers"
        elif r > 130 and b > 90 and g < 130:
            return "pink_purple_flowers"
        elif g > r and g > b and g > 90:
            return "green_foliage"
        else:
            return "mixed_other"
    except Exception:
        return "mixed_other"


def get_annotation(r: dict) -> str:
    blind = r.get("blind_identification", {})
    blind_sp = blind.get("species", "?")
    if blind_sp and "_" in blind_sp:
        blind_sp = blind_sp.split("_")[-1]

    morph = r.get("morphological_fidelity", {})
    scores = {}
    for k, v in morph.items():
        if isinstance(v, dict) and "score" in v and not v.get("not_visible", False):
            scores[k] = v.get("score", "?")

    abd = scores.get("abdomen_banding", "?")
    thx = scores.get("thorax_coloration", "?")
    mean_s = mean_morph_score(r)
    angle = parse_angle(r["file"])
    diag = r.get("diagnostic_completeness", {}).get("level", "?")

    return (f"ID:{blind_sp} | {diag} | {angle}\n"
            f"abd:{abd} thx:{thx} mean:{mean_s:.1f}")


def plot_image_grid(images: list[dict], title: str, rows: int, cols: int,
                    color: str) -> plt.Figure | None:
    """Plot a grid of images. Images should already be ordered as desired."""
    if not images:
        return None

    n = min(rows * cols, len(images))
    selected = images[:n]

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx in range(rows * cols):
        r_idx, c_idx = divmod(idx, cols)
        ax = axes[r_idx, c_idx]
        ax.axis("off")

        if idx >= n:
            continue

        r = selected[idx]
        img_path = SYNTH_DIR / r.get("species", "") / r["file"]
        if not img_path.exists():
            ax.text(0.5, 0.5, "Not found", ha="center", va="center", transform=ax.transAxes)
            continue

        img = mpimg.imread(str(img_path))
        ax.imshow(img)
        annotation = get_annotation(r)
        ax.set_title(annotation, fontsize=6.5, color=color, fontweight="bold",
                     pad=2, loc="left", family="monospace")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_combined_grid(species: str, sp_pass: list, sp_fail: list,
                       rows: int, cols: int, seed: int = 42) -> plt.Figure | None:
    """Plot PASS and FAIL grids side by side in a single figure."""
    if not sp_pass and not sp_fail:
        return None

    sp_short = species.replace("Bombus_", "B. ").replace("_", " ")

    random.seed(seed)
    n_grid = rows * cols
    pass_sample = random.sample(sp_pass, min(n_grid, len(sp_pass))) if sp_pass else []
    fail_sorted = sorted(sp_fail, key=mean_morph_score)[:n_grid]

    total_cols = cols * 2 + 1  # gap column in middle
    fig, axes = plt.subplots(rows, total_cols,
                             figsize=(total_cols * 2.8, rows * 3.2),
                             gridspec_kw={"width_ratios": [1]*cols + [0.15] + [1]*cols})

    # Turn off all axes first
    for ax_row in axes:
        for ax in ax_row:
            ax.axis("off")

    def _fill_half(images, col_offset, color):
        for idx in range(rows * cols):
            r_idx, c_idx = divmod(idx, cols)
            ax = axes[r_idx, col_offset + c_idx]
            if idx >= len(images):
                continue
            r = images[idx]
            img_path = SYNTH_DIR / r.get("species", "") / r["file"]
            if not img_path.exists():
                ax.text(0.5, 0.5, "Not found", ha="center", va="center",
                        transform=ax.transAxes)
                continue
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            annotation = get_annotation(r)
            ax.set_title(annotation, fontsize=5.5, color=color, fontweight="bold",
                         pad=2, loc="left", family="monospace")

    _fill_half(pass_sample, 0, "#2ecc71")
    _fill_half(fail_sorted, cols + 1, "#e74c3c")

    # Titles
    fig.suptitle(f"{species.replace('_', ' ')} — LLM Judge Pass vs Fail",
                 fontsize=14, fontweight="bold", y=0.99)

    # Sub-headers
    left_center = cols / 2 / total_cols
    right_center = (cols + 1 + cols / 2) / total_cols
    fig.text(left_center, 0.95, "PASS (strict filter)",
             ha="center", fontsize=12, fontweight="bold", color="#2ecc71")
    fig.text(left_center, 0.93,
             f"{species.replace('_', ' ')} — PASS ({len(sp_pass)} total)",
             ha="center", fontsize=9, color="black")
    fig.text(right_center, 0.95, "FAIL (strict filter)",
             ha="center", fontsize=12, fontweight="bold", color="#e74c3c")
    fig.text(right_center, 0.93,
             f"{species.replace('_', ' ')} — FAIL ({len(sp_fail)} total)",
             ha="center", fontsize=9, color="black")

    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


def _nobg_annotation(orig_r: dict, nobg_r: dict) -> tuple[str, str]:
    """Build annotation strings for original and nobg versions of same image."""
    def _ann(r):
        blind = r.get("blind_identification", {})
        blind_sp = blind.get("species", "?")
        if blind_sp and "_" in blind_sp:
            blind_sp = blind_sp.split("_")[-1]
        morph = r.get("morphological_fidelity", {})
        scores = {}
        for k, v in morph.items():
            if isinstance(v, dict) and "score" in v and not v.get("not_visible", False):
                scores[k] = v.get("score", "?")
        abd = scores.get("abdomen_banding", "?")
        thx = scores.get("thorax_coloration", "?")
        ms = mean_morph_score(r)
        diag = r.get("diagnostic_completeness", {}).get("level", "?")
        status = "PASS" if r.get("overall_pass") else "FAIL"
        return f"{status} | ID:{blind_sp} | {diag}\nabd:{abd} thx:{thx} mean:{ms:.1f}"
    return _ann(orig_r), _ann(nobg_r)


def plot_nobg_overlay(orig_results_path: Path, nobg_results_path: Path,
                      orig_synth_dir: Path, nobg_synth_dir: Path,
                      output_dir: Path, species_list: list[str] | None = None,
                      rows: int = 4, cols: int = 5):
    """Plot side-by-side original vs nobg images for flipped cases.

    Creates per-species grids showing:
      - Pass-to-Fail flips (originally passed, failed after bg removal)
      - Fail-to-Pass flips (originally failed, passed after bg removal)
    Each pair shows original (left) and nobg (right) of the same image.
    """
    orig_data = json.loads(orig_results_path.read_text())
    nobg_data = json.loads(nobg_results_path.read_text())

    orig_idx = {r["file"]: r for r in orig_data["results"] if "error" not in r}
    nobg_idx = {r["file"]: r for r in nobg_data["results"] if "error" not in r}
    common = sorted(set(orig_idx.keys()) & set(nobg_idx.keys()))

    if species_list is None:
        species_list = sorted(set(orig_idx[f]["species"] for f in common))

    for sp in species_list:
        sp_files = [f for f in common if orig_idx[f]["species"] == sp]
        sp_short = sp.replace("Bombus_", "B. ")

        # Categorize flips
        p2f = sorted([f for f in sp_files
                       if orig_idx[f]["overall_pass"] and not nobg_idx[f]["overall_pass"]],
                      key=lambda f: mean_morph_score(nobg_idx[f]))
        f2p = sorted([f for f in sp_files
                       if not orig_idx[f]["overall_pass"] and nobg_idx[f]["overall_pass"]],
                      key=lambda f: mean_morph_score(orig_idx[f]))

        for flip_type, flip_files, title_label in [
            ("pass_to_fail", p2f, "PASS → FAIL (bg removal hurt)"),
            ("fail_to_pass", f2p, "FAIL → PASS (bg removal helped)"),
        ]:
            if not flip_files:
                continue

            # Each pair takes 2 columns; show `cols` pairs per row
            n_pairs = min(rows * cols, len(flip_files))
            pair_cols = cols
            pair_rows = (n_pairs + pair_cols - 1) // pair_cols

            total_cols = pair_cols * 2  # orig + nobg alternating
            fig, axes = plt.subplots(pair_rows, total_cols,
                                     figsize=(total_cols * 2.5, pair_rows * 3.2))
            if pair_rows == 1:
                axes = axes[np.newaxis, :]

            for ax_row in axes:
                for ax in ax_row:
                    ax.axis("off")

            for idx in range(n_pairs):
                r_idx, c_idx = divmod(idx, pair_cols)
                fname = flip_files[idx]
                orig_r = orig_idx[fname]
                nobg_r = nobg_idx[fname]
                orig_ann, nobg_ann = _nobg_annotation(orig_r, nobg_r)

                # Original image
                ax_orig = axes[r_idx, c_idx * 2]
                orig_img_path = orig_synth_dir / sp / fname
                if orig_img_path.exists():
                    img = mpimg.imread(str(orig_img_path))
                    ax_orig.imshow(img)
                orig_color = "#2ecc71" if orig_r["overall_pass"] else "#e74c3c"
                ax_orig.set_title(f"ORIG: {orig_ann}", fontsize=5.5, color=orig_color,
                                  fontweight="bold", pad=2, loc="left", family="monospace")

                # NoBG image
                ax_nobg = axes[r_idx, c_idx * 2 + 1]
                nobg_img_path = nobg_synth_dir / sp / fname
                if nobg_img_path.exists():
                    img = mpimg.imread(str(nobg_img_path))
                    ax_nobg.imshow(img)
                nobg_color = "#2ecc71" if nobg_r["overall_pass"] else "#e74c3c"
                ax_nobg.set_title(f"NOBG: {nobg_ann}", fontsize=5.5, color=nobg_color,
                                  fontweight="bold", pad=2, loc="left", family="monospace")

            fig.suptitle(f"{sp_short} — {title_label} (n={len(flip_files)})",
                         fontsize=14, fontweight="bold", y=1.01)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            out = output_dir / f"nobg_overlay_{sp}_{flip_type}.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            fig.savefig(Path(out).with_suffix(".pdf"), dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out}")


def plot_failure_analysis(species: str, sp_pass: list, sp_fail: list,
                          bg_data: dict) -> plt.Figure:
    """Create a multi-panel failure analysis chart."""
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.4)

    total = len(sp_pass) + len(sp_fail)
    sp_short = species.replace("Bombus_", "B. ")

    # --- Panel 1: Pass rate by angle ---
    ax1 = fig.add_subplot(gs[0, 0])
    angle_pass = Counter()
    angle_total = Counter()
    for r in sp_pass + sp_fail:
        angle = parse_angle(r["file"])
        angle_total[angle] += 1
        if r in sp_pass:
            angle_pass[angle] += 1

    rates = [100 * angle_pass.get(a, 0) / max(angle_total.get(a, 1), 1) for a in ANGLE_ORDER]
    bars = ax1.bar(ANGLE_LABELS, rates, color=["#2ecc71" if r >= 50 else "#e74c3c" for r in rates],
                   edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Pass Rate (%)")
    ax1.set_title("Pass Rate by Viewpoint", fontweight="bold", fontsize=11)
    ax1.set_ylim(0, 110)
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{rate:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.tick_params(axis="x", labelsize=8)

    # --- Panel 2: Feature scores pass vs fail ---
    ax2 = fig.add_subplot(gs[0, 1])
    pass_means = []
    fail_means = []
    for feat in FEATURE_ORDER:
        p_scores = [r.get("morphological_fidelity", {}).get(feat, {}).get("score", 0)
                     for r in sp_pass
                     if isinstance(r.get("morphological_fidelity", {}).get(feat), dict)
                     and not r["morphological_fidelity"][feat].get("not_visible", False)]
        f_scores = [r.get("morphological_fidelity", {}).get(feat, {}).get("score", 0)
                     for r in sp_fail
                     if isinstance(r.get("morphological_fidelity", {}).get(feat), dict)
                     and not r["morphological_fidelity"][feat].get("not_visible", False)]
        pass_means.append(sum(p_scores)/len(p_scores) if p_scores else 0)
        fail_means.append(sum(f_scores)/len(f_scores) if f_scores else 0)

    x = np.arange(len(FEATURE_ORDER))
    w = 0.35
    ax2.bar(x - w/2, pass_means, w, label="Pass", color="#2ecc71", edgecolor="white")
    ax2.bar(x + w/2, fail_means, w, label="Fail", color="#e74c3c", edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels(FEATURE_LABELS, fontsize=8)
    ax2.set_ylabel("Mean Score (1-5)")
    ax2.set_title("Feature Scores: Pass vs Fail", fontweight="bold", fontsize=11)
    ax2.set_ylim(0, 5.5)
    ax2.legend(fontsize=9)
    ax2.axhline(y=4.0, color="gray", linestyle="--", alpha=0.5)

    # --- Panel 3: Blind ID breakdown for failures ---
    ax3 = fig.add_subplot(gs[0, 2])
    blind_ids = Counter()
    for r in sp_fail:
        blind_sp = r.get("blind_identification", {}).get("species", "Unknown")
        if blind_sp == species.replace("_", " "):
            blind_ids["Target\n(correct)"] += 1
        elif blind_sp == "Unknown":
            blind_ids["Unknown"] += 1
        else:
            short = blind_sp.replace("Bombus ", "B. ")
            blind_ids[short] += 1

    if blind_ids:
        labels = list(blind_ids.keys())
        sizes = list(blind_ids.values())
        colors_pie = []
        for lbl in labels:
            if lbl == "Unknown":
                colors_pie.append("#95a5a6")
            elif "correct" in lbl:
                colors_pie.append("#f39c12")
            else:
                colors_pie.append("#e74c3c")
        wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct="%1.0f%%",
                                            colors=colors_pie, textprops={"fontsize": 8})
        for t in autotexts:
            t.set_fontsize(8)
            t.set_fontweight("bold")
    ax3.set_title("Blind ID of Failed Images", fontweight="bold", fontsize=11)

    # --- Panel 4: Pass rate by background ---
    ax4 = fig.add_subplot(gs[0, 3])
    bg_rates = []
    bg_colors = []
    bg_labels_used = []
    bg_counts = []
    for bg_type, bg_label in zip(BG_ORDER, BG_LABELS):
        t = bg_data["total"].get(bg_type, 0)
        if t == 0:
            continue
        p = bg_data["pass"].get(bg_type, 0)
        rate = 100 * p / t
        bg_rates.append(rate)
        bg_colors.append(BG_COLORS_MAP[bg_type])
        bg_labels_used.append(bg_label)
        bg_counts.append(f"(n={t})")

    bars4 = ax4.bar(range(len(bg_rates)), bg_rates, color=bg_colors, edgecolor="white", linewidth=0.5)
    ax4.set_xticks(range(len(bg_rates)))
    ax4.set_xticklabels([f"{lbl}\n{cnt}" for lbl, cnt in zip(bg_labels_used, bg_counts)], fontsize=8)
    ax4.set_ylabel("Pass Rate (%)")
    ax4.set_title("Pass Rate by Background", fontweight="bold", fontsize=11)
    ax4.set_ylim(0, 110)
    for bar, rate in zip(bars4, bg_rates):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                 f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- Panel 5: Score distribution histogram ---
    ax5 = fig.add_subplot(gs[1, 0])
    pass_scores = [mean_morph_score(r) for r in sp_pass]
    fail_scores = [mean_morph_score(r) for r in sp_fail]
    bins = np.arange(1.5, 5.5, 0.25)
    ax5.hist(pass_scores, bins=bins, alpha=0.7, color="#2ecc71", label=f"Pass (n={len(sp_pass)})", edgecolor="white")
    ax5.hist(fail_scores, bins=bins, alpha=0.7, color="#e74c3c", label=f"Fail (n={len(sp_fail)})", edgecolor="white")
    ax5.axvline(x=4.0, color="gray", linestyle="--", alpha=0.7)
    ax5.set_xlabel("Mean Morphological Score")
    ax5.set_ylabel("Count")
    ax5.set_title("Score Distribution", fontweight="bold", fontsize=11)
    ax5.legend(fontsize=9)

    # --- Panel 6: Pass/Fail counts by angle ---
    ax6 = fig.add_subplot(gs[1, 1])
    angle_fail_count = Counter(parse_angle(r["file"]) for r in sp_fail)
    fail_by_angle = [angle_fail_count.get(a, 0) for a in ANGLE_ORDER]
    pass_by_angle = [angle_pass.get(a, 0) for a in ANGLE_ORDER]
    total_by_angle = [angle_total.get(a, 0) for a in ANGLE_ORDER]

    ax6.barh(ANGLE_LABELS, pass_by_angle, color="#2ecc71", label="Pass", edgecolor="white")
    ax6.barh(ANGLE_LABELS, fail_by_angle, left=pass_by_angle, color="#e74c3c", label="Fail", edgecolor="white")
    for i, (p, f, t) in enumerate(zip(pass_by_angle, fail_by_angle, total_by_angle)):
        ax6.text(t + 1, i, f"{p}/{t}", va="center", fontsize=9)
    ax6.set_xlabel("Image Count")
    ax6.set_title("Pass/Fail Counts by Angle", fontweight="bold", fontsize=11)
    ax6.legend(fontsize=9, loc="lower right")

    # --- Panel 7: Angle x Background heatmap ---
    ax7 = fig.add_subplot(gs[1, 2:])
    # Build pass rate matrix: angles x backgrounds
    matrix = []
    annot = []
    bg_types_present = [bg for bg in BG_ORDER if bg_data["total"].get(bg, 0) > 0]
    bg_labels_present = [BG_LABELS[BG_ORDER.index(bg)] for bg in bg_types_present]

    for angle in ANGLE_ORDER:
        row = []
        row_annot = []
        for bg_type in bg_types_present:
            key = f"{angle}_{bg_type}"
            t = bg_data["angle_bg_total"].get(key, 0)
            p = bg_data["angle_bg_pass"].get(key, 0)
            if t > 0:
                rate = 100 * p / t
                row.append(rate)
                row_annot.append(f"{rate:.0f}%\n({p}/{t})")
            else:
                row.append(float("nan"))
                row_annot.append("")
        matrix.append(row)
        annot.append(row_annot)

    matrix_arr = np.array(matrix, dtype=float)
    im = ax7.imshow(matrix_arr, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax7.set_xticks(range(len(bg_labels_present)))
    ax7.set_xticklabels(bg_labels_present, fontsize=9)
    ax7.set_yticks(range(len(ANGLE_LABELS)))
    ax7.set_yticklabels(ANGLE_LABELS, fontsize=9)
    for i in range(len(ANGLE_ORDER)):
        for j in range(len(bg_types_present)):
            if annot[i][j]:
                ax7.text(j, i, annot[i][j], ha="center", va="center", fontsize=8, fontweight="bold")
    ax7.set_title("Pass Rate: Angle x Background", fontweight="bold", fontsize=11)
    plt.colorbar(im, ax=ax7, label="Pass Rate (%)", shrink=0.8)

    fig.suptitle(f"{sp_short} — LLM Judge Failure Analysis",
                 fontsize=15, fontweight="bold", y=1.0)
    return fig


def compute_bg_data(species: str, sp_pass: list, sp_fail: list) -> dict:
    """Compute background classification data for a species."""
    bg_pass = Counter()
    bg_fail = Counter()
    bg_total = Counter()
    angle_bg_pass = Counter()
    angle_bg_total = Counter()

    pass_set = set(id(r) for r in sp_pass)

    print(f"  Classifying backgrounds for {len(sp_pass) + len(sp_fail)} images...")
    for r in sp_pass + sp_fail:
        img_path = SYNTH_DIR / species / r["file"]
        bg = classify_background(img_path)
        angle = parse_angle(r["file"])
        key = f"{angle}_{bg}"
        bg_total[bg] += 1
        angle_bg_total[key] += 1
        if id(r) in pass_set:
            bg_pass[bg] += 1
            angle_bg_pass[key] += 1
        else:
            bg_fail[bg] += 1

    return {"pass": bg_pass, "fail": bg_fail, "total": bg_total,
            "angle_bg_pass": angle_bg_pass, "angle_bg_total": angle_bg_total}


def plot_nobg_comparison(orig_results_path: Path, nobg_results_path: Path,
                         output_dir: Path, species_list: list[str] | None = None):
    """Plot combined original vs no-background comparison grid."""
    orig_data = json.loads(orig_results_path.read_text())
    nobg_data = json.loads(nobg_results_path.read_text())

    orig_idx = {r["file"]: r for r in orig_data["results"] if "error" not in r}
    nobg_idx = {r["file"]: r for r in nobg_data["results"] if "error" not in r}
    common = sorted(set(orig_idx.keys()) & set(nobg_idx.keys()))

    if species_list is None:
        species_list = sorted(set(orig_idx[f]["species"] for f in common))

    VIEWPOINTS = ["lateral", "dorsal", "three-quarter_anterior", "three-quarter_posterior", "frontal"]
    VP_LABELS = ["Lateral", "Dorsal", "3/4 Anterior", "3/4 Posterior", "Frontal"]

    C_ORIG = "#F44336"
    C_NOBG = "#4CAF50"

    for sp in species_list:
        sp_files = [f for f in common if orig_idx[f]["species"] == sp]
        sp_short = sp.replace("Bombus_", "B. ")

        # --- Gather per-species stats ---
        sp_stats = {
            "orig_pass": 0, "nobg_pass": 0, "total": len(sp_files),
            "fail_to_pass": 0, "pass_to_fail": 0,
        }
        vp_stats = {}
        yellow_stats = {"orig_pass": 0, "nobg_pass": 0, "total": 0}
        other_stats = {"orig_pass": 0, "nobg_pass": 0, "total": 0}

        for f in sp_files:
            o_pass = orig_idx[f].get("overall_pass", False)
            n_pass = nobg_idx[f].get("overall_pass", False)
            if o_pass: sp_stats["orig_pass"] += 1
            if n_pass: sp_stats["nobg_pass"] += 1
            if not o_pass and n_pass: sp_stats["fail_to_pass"] += 1
            if o_pass and not n_pass: sp_stats["pass_to_fail"] += 1

            vp = parse_angle(f)
            if vp not in vp_stats:
                vp_stats[vp] = {"orig_pass": 0, "nobg_pass": 0, "total": 0}
            vp_stats[vp]["total"] += 1
            if o_pass: vp_stats[vp]["orig_pass"] += 1
            if n_pass: vp_stats[vp]["nobg_pass"] += 1

            # Yellow bg heuristic from environment index
            parts = f.split("::")
            if len(parts) >= 2:
                try:
                    seq = int(parts[1])
                    env_idx = (seq // 5) % 5
                    is_yellow = env_idx in {1, 3}
                except ValueError:
                    is_yellow = False
            else:
                is_yellow = False

            bucket = yellow_stats if is_yellow else other_stats
            bucket["total"] += 1
            if o_pass: bucket["orig_pass"] += 1
            if n_pass: bucket["nobg_pass"] += 1

        # --- Feature scores ---
        feat_keys = ["abdomen_banding", "thorax_coloration", "head_antennae",
                     "legs_appendages", "wing_venation_texture"]
        feat_labels = ["Abdomen\nBanding", "Thorax\nColor", "Head\nAntennae",
                       "Legs", "Wings"]
        orig_feat = []
        nobg_feat = []
        for fk in feat_keys:
            o_scores = []
            n_scores = []
            for f in sp_files:
                o_m = orig_idx[f].get("morphological_fidelity", {}).get(fk, {})
                n_m = nobg_idx[f].get("morphological_fidelity", {}).get(fk, {})
                if isinstance(o_m, dict) and "score" in o_m and not o_m.get("not_visible", False):
                    o_scores.append(o_m["score"])
                if isinstance(n_m, dict) and "score" in n_m and not n_m.get("not_visible", False):
                    n_scores.append(n_m["score"])
            orig_feat.append(sum(o_scores) / len(o_scores) if o_scores else 0)
            nobg_feat.append(sum(n_scores) / len(n_scores) if n_scores else 0)

        # --- Failure mode counts ---
        failure_keys = ["wrong_coloration", "background_bleed", "blurry_artifacts",
                        "extra_missing_limbs", "flower_unrealistic"]
        failure_labels = ["Wrong\nColor", "BG\nBleed", "Blurry", "Limbs", "Flower"]
        orig_failures = []
        nobg_failures = []
        for fk in failure_keys:
            oc = sum(1 for f in sp_files
                     if orig_idx[f].get("species_fidelity", {}).get(fk, False)
                     or orig_idx[f].get("image_quality", {}).get(fk, False))
            nc = sum(1 for f in sp_files
                     if nobg_idx[f].get("species_fidelity", {}).get(fk, False)
                     or nobg_idx[f].get("image_quality", {}).get(fk, False))
            orig_failures.append(oc)
            nobg_failures.append(nc)

        # === PLOT 2x2 GRID ===
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        w = 0.35

        # Panel 1: Pass rate by viewpoint
        ax = axes[0, 0]
        vps = [v for v in VIEWPOINTS if v in vp_stats]
        vp_lbls = [VP_LABELS[VIEWPOINTS.index(v)] for v in vps]
        x = np.arange(len(vps))
        orig_vp = [vp_stats[v]["orig_pass"] / vp_stats[v]["total"] for v in vps]
        nobg_vp = [vp_stats[v]["nobg_pass"] / vp_stats[v]["total"] for v in vps]
        b1 = ax.bar(x - w/2, orig_vp, w, label="Original", color=C_ORIG, alpha=0.8)
        b2 = ax.bar(x + w/2, nobg_vp, w, label="No Background", color=C_NOBG, alpha=0.8)
        for bar, val in zip(b1, orig_vp):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=8)
        for bar, val in zip(b2, nobg_vp):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(vp_lbls, fontsize=9)
        ax.set_ylabel("Pass Rate (overall_pass)")
        ax.set_title("Pass Rate by Viewpoint")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 2: Yellow vs Other background
        ax = axes[0, 1]
        bg_labels = ["Yellow BG\n(goldenrod/dandelion)", "Other BG"]
        x2 = np.arange(2)
        o_bg = [yellow_stats["orig_pass"] / max(yellow_stats["total"], 1),
                other_stats["orig_pass"] / max(other_stats["total"], 1)]
        n_bg = [yellow_stats["nobg_pass"] / max(yellow_stats["total"], 1),
                other_stats["nobg_pass"] / max(other_stats["total"], 1)]
        b1 = ax.bar(x2 - w/2, o_bg, w, label="Original", color=C_ORIG, alpha=0.8)
        b2 = ax.bar(x2 + w/2, n_bg, w, label="No Background", color=C_NOBG, alpha=0.8)
        for bar, val in zip(b1, o_bg):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=9)
        for bar, val in zip(b2, n_bg):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x2)
        ax.set_xticklabels(bg_labels)
        ax.set_ylabel("Pass Rate")
        ax.set_title("Yellow vs Other Backgrounds")
        ax.set_ylim(0, 1.15)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.text(0.5, -0.12, f"n={yellow_stats['total']} yellow, n={other_stats['total']} other",
                transform=ax.transAxes, ha="center", fontsize=8, color="gray")

        # Panel 3: Feature scores
        ax = axes[1, 0]
        x3 = np.arange(len(feat_keys))
        ax.bar(x3 - w/2, orig_feat, w, label="Original", color="#FF9800", alpha=0.8)
        ax.bar(x3 + w/2, nobg_feat, w, label="No Background", color="#2196F3", alpha=0.8)
        for i, (ov, nv) in enumerate(zip(orig_feat, nobg_feat)):
            ax.text(i - w/2, ov + 0.05, f"{ov:.2f}", ha="center", va="bottom", fontsize=8)
            ax.text(i + w/2, nv + 0.05, f"{nv:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x3)
        ax.set_xticklabels(feat_labels, fontsize=9)
        ax.set_ylabel("Mean Score (1-5)")
        ax.set_title("Morphological Feature Scores")
        ax.set_ylim(0, 5.5)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Panel 4: Failure mode counts
        ax = axes[1, 1]
        x4 = np.arange(len(failure_keys))
        ax.bar(x4 - w/2, orig_failures, w, label="Original", color=C_ORIG, alpha=0.8)
        ax.bar(x4 + w/2, nobg_failures, w, label="No Background", color=C_NOBG, alpha=0.8)
        for i, (ov, nv) in enumerate(zip(orig_failures, nobg_failures)):
            ax.text(i - w/2, ov + 1, str(ov), ha="center", va="bottom", fontsize=8)
            ax.text(i + w/2, nv + 1, str(nv), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x4)
        ax.set_xticklabels(failure_labels, fontsize=9)
        ax.set_ylabel("Count")
        ax.set_title("Failure Mode Counts")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        # Summary annotation
        orig_rate = sp_stats["orig_pass"] / sp_stats["total"]
        nobg_rate = sp_stats["nobg_pass"] / sp_stats["total"]
        delta = nobg_rate - orig_rate
        sign = "+" if delta >= 0 else ""
        summary = (f"Overall: {orig_rate:.1%} → {nobg_rate:.1%} ({sign}{delta:.1%})  |  "
                   f"Flips: {sp_stats['fail_to_pass']} F→P, {sp_stats['pass_to_fail']} P→F")

        plt.suptitle(f"{sp_short} — Original vs No-Background Comparison (n={sp_stats['total']})",
                     fontsize=14, fontweight="bold", y=1.02)
        fig.text(0.5, 0.99, summary, ha="center", fontsize=10, color="gray")
        plt.tight_layout()
        out = output_dir / f"nobg_comparison_{sp}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        fig.savefig(Path(out).with_suffix(".pdf"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out}")


def main():
    parser = argparse.ArgumentParser(description="Plot pass/fail image grids from LLM judge")
    parser.add_argument("--rows", type=int, default=4, help="Rows per grid (default: 4)")
    parser.add_argument("--cols", type=int, default=5, help="Columns per grid (default: 5)")
    parser.add_argument("--species", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "llm_judge_eval")
    parser.add_argument("--judge-results", type=Path, default=None,
                        help="Path to results.json (default: RESULTS/llm_judge_eval/results.json)")
    parser.add_argument("--synth-dir", type=Path, default=None,
                        help="Path to synthetic images dir (default: RESULTS/synthetic_generation)")
    parser.add_argument("--compare-nobg", action="store_true",
                        help="Generate original vs no-background comparison plots")
    parser.add_argument("--nobg-results", type=Path, default=None,
                        help="Path to nobg results.json (for --compare-nobg)")
    args = parser.parse_args()

    global SYNTH_DIR, JUDGE_RESULTS
    if args.synth_dir:
        SYNTH_DIR = args.synth_dir
    if args.judge_results:
        JUDGE_RESULTS = args.judge_results

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Compare mode ---
    if args.compare_nobg:
        orig_path = args.judge_results or DEFAULT_JUDGE_RESULTS
        nobg_path = args.nobg_results or (RESULTS_DIR / "llm_judge_eval_nobg" / "results.json")
        orig_synth = args.synth_dir or DEFAULT_SYNTH_DIR
        nobg_synth = RESULTS_DIR / "synthetic_generation_nobg"
        print(f"Comparing: {orig_path} vs {nobg_path}")
        plot_nobg_comparison(orig_path, nobg_path, output_dir, args.species)
        plot_nobg_overlay(orig_path, nobg_path, orig_synth, nobg_synth,
                          output_dir, args.species, args.rows, args.cols)
        print("\nDone.")
        return

    data = json.loads(JUDGE_RESULTS.read_text())
    all_results = data.get("results", [])

    pass_results = [r for r in all_results if strict_pass(r)]
    fail_results = [r for r in all_results if not strict_pass(r)]
    print(f"Loaded {len(all_results)} results: {len(pass_results)} pass, {len(fail_results)} fail")

    if args.species:
        species_list = args.species
    else:
        species_list = sorted(set(r.get("species", "") for r in all_results if r.get("species")))

    n_grid = args.rows * args.cols

    for sp in species_list:
        print(f"\n{sp}:")
        sp_pass = [r for r in pass_results if r.get("species") == sp]
        sp_fail = [r for r in fail_results if r.get("species") == sp]
        print(f"  Pass: {len(sp_pass)}, Fail: {len(sp_fail)}")

        # PASS grid: random sample
        random.seed(args.seed)
        sp_pass_shuffled = random.sample(sp_pass, min(n_grid, len(sp_pass)))
        fig_pass = plot_image_grid(
            sp_pass_shuffled,
            f"{sp.replace('_', ' ')} — PASS ({len(sp_pass)} total, {len(sp_pass_shuffled)} random shown)",
            args.rows, args.cols, "#2ecc71",
        )
        if fig_pass:
            out = output_dir / f"grid_{sp}_PASS.png"
            fig_pass.savefig(out, dpi=150, bbox_inches="tight")
            fig_pass.savefig(Path(out).with_suffix(".pdf"), dpi=150, bbox_inches="tight")
            plt.close(fig_pass)
            print(f"  Saved: {out}")

        # FAIL grid: sorted by mean score (worst first)
        sp_fail_sorted = sorted(sp_fail, key=mean_morph_score)
        n_shown = min(n_grid, len(sp_fail_sorted))
        fig_fail = plot_image_grid(
            sp_fail_sorted,
            f"{sp.replace('_', ' ')} — FAIL ({len(sp_fail)} total, {n_shown} worst shown)",
            args.rows, args.cols, "#e74c3c",
        )
        if fig_fail:
            out = output_dir / f"grid_{sp}_FAIL.png"
            fig_fail.savefig(out, dpi=150, bbox_inches="tight")
            fig_fail.savefig(Path(out).with_suffix(".pdf"), dpi=150, bbox_inches="tight")
            plt.close(fig_fail)
            print(f"  Saved: {out}")

        # Combined PASS + FAIL side by side
        fig_combined = plot_combined_grid(sp, sp_pass, sp_fail,
                                          args.rows, args.cols, args.seed)
        if fig_combined:
            out = output_dir / f"grid_{sp}_combined.png"
            fig_combined.savefig(out, dpi=150, bbox_inches="tight")
            fig_combined.savefig(Path(out).with_suffix(".pdf"), dpi=150, bbox_inches="tight")
            plt.close(fig_combined)
            print(f"  Saved: {out}")

        # Failure analysis chart (with background classification)
        bg_data = compute_bg_data(sp, sp_pass, sp_fail)
        fig_analysis = plot_failure_analysis(sp, sp_pass, sp_fail, bg_data)
        out = output_dir / f"analysis_{sp}.png"
        fig_analysis.savefig(out, dpi=150, bbox_inches="tight")
        fig_analysis.savefig(Path(out).with_suffix(".pdf"), dpi=150, bbox_inches="tight")
        plt.close(fig_analysis)
        print(f"  Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
