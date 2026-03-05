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

SYNTH_DIR = RESULTS_DIR / "synthetic_generation"
JUDGE_RESULTS = RESULTS_DIR / "llm_judge_eval" / "results.json"

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


def main():
    parser = argparse.ArgumentParser(description="Plot pass/fail image grids from LLM judge")
    parser.add_argument("--rows", type=int, default=4, help="Rows per grid (default: 4)")
    parser.add_argument("--cols", type=int, default=5, help="Columns per grid (default: 5)")
    parser.add_argument("--species", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "llm_judge_eval")
    args = parser.parse_args()

    data = json.loads(JUDGE_RESULTS.read_text())
    all_results = data.get("results", [])

    pass_results = [r for r in all_results if strict_pass(r)]
    fail_results = [r for r in all_results if not strict_pass(r)]
    print(f"Loaded {len(all_results)} results: {len(pass_results)} pass, {len(fail_results)} fail")

    if args.species:
        species_list = args.species
    else:
        species_list = sorted(set(r.get("species", "") for r in all_results if r.get("species")))

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
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
            plt.close(fig_fail)
            print(f"  Saved: {out}")

        # Failure analysis chart (with background classification)
        bg_data = compute_bg_data(sp, sp_pass, sp_fail)
        fig_analysis = plot_failure_analysis(sp, sp_pass, sp_fail, bg_data)
        out = output_dir / f"analysis_{sp}.png"
        fig_analysis.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig_analysis)
        print(f"  Saved: {out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
