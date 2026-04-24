#!/usr/bin/env python3
"""Generate §4 figures.

- Fig 4.1 (prompt iteration): side-by-side B. ashtoni samples from v3 (no negative
  constraints) and v8 (final prompt with negatives + tergite colour maps), same
  pose indices.
- Fig 4.2 (judge overlay): one PASS and one FAIL example with the LLM judge's
  per-feature scores drawn on top of the image.
- Fig 4.3 (interface schematic): a two-panel schematic of the two-stage expert
  evaluation interface (blind ID stage, detailed evaluation stage) built from
  text boxes, since live-browser screenshots are not available in this repo.

Outputs: docs/plots/pipeline_v3_vs_v8.{png,pdf},
         docs/plots/judge_overlay_pass_fail.{png,pdf},
         docs/plots/expert_interface_schematic.{png,pdf}
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/plots"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, stem: str):
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")


# ── Fig 4.1 ────────────────────────────────────────────────────────────────

def fig_4_1_prompt_iteration():
    v3_dir = ROOT / "RESULTS_prompt_iteration/prompt_testing/ashtoni_v3"
    v8_dir = ROOT / "RESULTS_prompt_iteration/prompt_testing/ashtoni_v8_new_prompt"

    # Pick the first 3 file indices that exist in BOTH versions.
    common = []
    for v3_img in sorted(v3_dir.glob("Bombus_ashtoni_*.png")):
        stem = v3_img.stem
        # match v8 file with the same "Bombus_ashtoni_0000" prefix ignoring caste
        prefix = "_".join(stem.split("_")[:3])  # Bombus_ashtoni_0000
        v8_candidates = sorted(v8_dir.glob(f"{prefix}_*.png"))
        if v8_candidates:
            common.append((v3_img, v8_candidates[0]))
        if len(common) == 3:
            break

    if not common:
        raise SystemExit("no shared v3/v8 files found")

    fig, axes = plt.subplots(2, len(common), figsize=(4 * len(common), 8))
    for col, (v3_img, v8_img) in enumerate(common):
        axes[0, col].imshow(Image.open(v3_img))
        axes[0, col].set_axis_off()
        axes[1, col].imshow(Image.open(v8_img))
        axes[1, col].set_axis_off()
    axes[0, 0].set_title("v3 prompt (no negative constraints, no tergite map)",
                         loc="left", fontsize=12, y=1.02)
    axes[1, 0].set_title("v8 prompt (negatives + tergite colour map + caste targeting)",
                         loc="left", fontsize=12, y=1.02)
    fig.suptitle("Figure 4.1 — Structured prompting effect on B. ashtoni generation",
                 fontsize=13, y=0.99)
    save(fig, "pipeline_v3_vs_v8")
    plt.close(fig)
    print("wrote pipeline_v3_vs_v8.{png,pdf}")


# ── Fig 4.2 ────────────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "legs_appendages": "Legs/Appendages",
    "wing_venation_texture": "Wing Venation",
    "head_antennae": "Head/Antennae",
    "abdomen_banding": "Abdomen Banding",
    "thorax_coloration": "Thorax Coloration",
}


def _extract_scores(record):
    """Parse chain_of_thought text for the 5 morph scores. Falls back to 0 if missing."""
    cot = record.get("chain_of_thought", "")
    scores = {}
    for key, label in FEATURE_LABELS.items():
        # search for e.g. "Legs/Appendages:" then look for "Score: N" within 120 chars
        needle = label
        idx = cot.find(needle)
        if idx < 0:
            scores[key] = None
            continue
        window = cot[idx: idx + 250]
        score_idx = window.lower().find("score")
        if score_idx < 0:
            scores[key] = None
            continue
        # next digit after "score"
        after = window[score_idx: score_idx + 40]
        import re
        m = re.search(r"[:*]\s*(\d)", after)
        scores[key] = int(m.group(1)) if m else None
    return scores


def fig_4_2_judge_overlay():
    judge_results_path = ROOT / "RESULTS_kfold/llm_judge_eval/results.json"
    judge = json.loads(judge_results_path.read_text())
    records = judge["results"]

    # Find one pass (matches_target + diag=species + morph>=4) and one fail.
    # Use the presence of "Matches Target: True" and mean morph >= 4 as heuristic.
    pass_record, fail_record = None, None
    for r in records:
        if pass_record is not None and fail_record is not None:
            break
        cot = r.get("chain_of_thought", "")
        matches = "Matches Target**: True" in cot or "Matches Target: True" in cot
        scores = _extract_scores(r)
        valid = [s for s in scores.values() if s is not None]
        if not valid:
            continue
        mean = sum(valid) / len(valid)
        if pass_record is None and matches and mean >= 4.0:
            pass_record = (r, scores, mean)
        if fail_record is None and (not matches or mean < 3.0):
            fail_record = (r, scores, mean)

    if pass_record is None or fail_record is None:
        raise SystemExit("could not find suitable pass + fail records")

    # Locate the source image for each record.
    synth_root = ROOT / "RESULTS_prompt_iteration/synthetic_generation"

    def find_image(fname):
        # filename like "Bombus_ashtoni::0000::female::lateral_0.jpg"
        parts = fname.split("::")
        species = parts[0]
        candidates = list((synth_root / species).rglob("*"))
        # match on last token (e.g. "lateral_0") inside any filename
        tail = parts[-1].replace(".jpg", "").replace(".png", "")
        for c in candidates:
            if tail in c.name:
                return c
        # fallback: first image in that species subdir
        imgs = list((synth_root / species).glob("*.png")) + list((synth_root / species).glob("*.jpg"))
        return imgs[0] if imgs else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, (rec, scores, mean), label in (
        (axes[0], pass_record, "PASS"),
        (axes[1], fail_record, "FAIL"),
    ):
        img_path = find_image(rec["file"])
        if img_path and img_path.exists():
            ax.imshow(Image.open(img_path))
        ax.set_axis_off()
        species_short = rec["species"].replace("Bombus_", "B. ")
        ax.set_title(f"{label} — {species_short}    mean morph = {mean:.1f}",
                     fontsize=13, loc="left")
        # Annotate per-feature scores in a side panel
        lines = [f"{FEATURE_LABELS[k]}: {v if v is not None else '—'}"
                 for k, v in scores.items()]
        textstr = "\n".join(lines)
        ax.text(1.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f7f7f7",
                          edgecolor="#888"))
    fig.suptitle("Figure 4.2 — LLM-judge scoring: PASS vs FAIL example",
                 fontsize=13, y=1.02)
    save(fig, "judge_overlay_pass_fail")
    plt.close(fig)
    print("wrote judge_overlay_pass_fail.{png,pdf}")


# ── Fig 4.3 ────────────────────────────────────────────────────────────────

def fig_4_3_interface_schematic():
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))

    def box(ax, text, y, height=0.12, color="#ffffff", edge="#333333", fontsize=10, weight="normal"):
        rect = patches.Rectangle((0.05, y), 0.9, height, facecolor=color, edgecolor=edge, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(0.5, y + height / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)

    # Stage 1 panel
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title("Stage 1 — Blind Identification", fontsize=13, weight="bold", loc="left")
    box(ax, "Synthetic image (no label shown)", 0.78, height=0.14, color="#e6f0ff", weight="bold")
    box(ax, "Select species from 16-way panel", 0.56, height=0.12)
    box(ax, "Diagnostic level: species / genus / family / none", 0.40, height=0.10, color="#f5f5f5")
    box(ax, "Unknown   /   No match   options always available", 0.26, height=0.08, color="#f5f5f5")
    box(ax, "→ Submit advances to Stage 2", 0.08, height=0.10, color="#dcead5", weight="bold")

    # Stage 2 panel
    ax = axes[1]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_axis_off()
    ax.set_title("Stage 2 — Detailed Morphological Evaluation", fontsize=13, weight="bold", loc="left")
    box(ax, "Target species + diagnostic criteria card revealed", 0.82, height=0.1, color="#e6f0ff", weight="bold")
    box(ax, "Museum reference images (×3) shown alongside", 0.70, height=0.08, color="#f5f5f5")
    # 5 feature rows
    features = ["Legs/Appendages (1–5)", "Wing Venation (1–5)", "Head/Antennae (1–5)",
                "Abdomen Banding (1–5)", "Thorax Coloration (1–5)"]
    for i, feat in enumerate(features):
        box(ax, feat, 0.58 - i * 0.08, height=0.07, color="#ffffff")
    # checkboxes and overall verdict
    box(ax, "Failure-mode checkboxes (wrong colour / missing limb / ...)", 0.16, height=0.08, color="#fef2e0")
    box(ax, "Overall verdict: PASS  /  FAIL  /  UNCERTAIN", 0.04, height=0.08, color="#dcead5", weight="bold")

    fig.suptitle("Figure 4.3 — Schematic of the two-stage expert evaluation interface", fontsize=13, y=1.01)
    save(fig, "expert_interface_schematic")
    plt.close(fig)
    print("wrote expert_interface_schematic.{png,pdf}")


if __name__ == "__main__":
    fig_4_1_prompt_iteration()
    fig_4_2_judge_overlay()
    fig_4_3_interface_schematic()
