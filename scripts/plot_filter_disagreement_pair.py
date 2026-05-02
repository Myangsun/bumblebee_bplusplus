#!/usr/bin/env python3
"""Two-image disagreement panel: where automated filters and the expert split.

Image A: passes both LLM-strict and centroid filters but FAILS expert strict.
Image B: passes expert strict but FAILS the automated filters.

Each panel shows the synthetic image alongside per-filter pass/fail status,
per-feature scores from the LLM and the expert, and the expert's verbatim
free-text failure note.

Output: docs/plots/filters/expert_vs_automated_disagreement_pair.{png,pdf}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT  # noqa: E402

ROOT = PROJECT_ROOT
SYN_ROOT = ROOT / "RESULTS_kfold/synthetic_generation"
OUT = ROOT / "docs/plots/filters/expert_vs_automated_disagreement_pair.png"

TAU_CENTROID = {"Bombus_ashtoni": 0.6945,
                "Bombus_sandersoni": 0.7468,
                "Bombus_flavidus": 0.6817}

PAIR = [
    {
        "label": "Image A — automated filters pass, expert FAILS",
        "file": "Bombus_ashtoni::0395::female::lateral_0.jpg",
        "species": "Bombus_ashtoni",
        "llm": {
            "verdict": "PASS (strict)", "verdict_color": "#2ca02c",
            "blind_id": "B. bohemicus / ashtoni  (matches target ✓)",
            "diag": "species",
            "morph_mean": 4.00,
            "scores": [("legs", 4), ("wings", 4), ("head/ant", 4),
                       ("abdomen", 5), ("thorax", 3)],
            "summary": '"Legs appear correct with dense hair … '
                       'matching the key field mark." (LLM did not '
                       'flag the leg-placement / posture defect.)',
        },
        "centroid": {"score": 0.704, "tau": TAU_CENTROID["Bombus_ashtoni"],
                     "verdict": "PASS  0.704 ≥ τ 0.694",
                     "verdict_color": "#2ca02c"},
        "expert": {
            "verdict": "FAIL (strict)", "verdict_color": "#d62728",
            "blind_id": "B. ashtoni (correctly identified)",
            "diag": "species",
            "morph_mean": 3.80,
            "scores": [("legs", 3), ("wings", 4), ("head/ant", 4),
                       ("abdomen", 4), ("thorax", 4)],
            "modes": ["species_other"],
            "note": '"back legs aren\'t placed quite right, '
                    'should be on flower"',
            "why_fail": "morph mean 3.80 < 4  (legs scored 3)",
        },
    },
    {
        "label": "Image B — expert passes, automated filters FAIL",
        "file": "Bombus_ashtoni::0119::male::lateral_0.jpg",
        "species": "Bombus_ashtoni",
        "llm": {
            "verdict": "FAIL (strict)", "verdict_color": "#d62728",
            "blind_id": "B. ternarius  (matches target ✗)",
            "diag": "genus",
            "morph_mean": 3.40,
            "scores": [("legs", 4), ("wings", 4), ("head/ant", 4),
                       ("abdomen", 3), ("thorax", 2)],
            "summary": '"Significant yellow coloration … incorrect for '
                       'B. ashtoni." (LLM applied female-typed colour '
                       'rubric to a male; yellow on B. ashtoni males '
                       'is in fact correct.)',
        },
        "centroid": {"score": 0.707, "tau": TAU_CENTROID["Bombus_ashtoni"],
                     "verdict": "PASS  0.707 ≥ τ 0.694",
                     "verdict_color": "#2ca02c"},
        "expert": {
            "verdict": "PASS (strict)", "verdict_color": "#2ca02c",
            "blind_id": "B. ashtoni (correctly identified)",
            "diag": "species",
            "morph_mean": 4.80,
            "scores": [("legs", 5), ("wings", 5), ("head/ant", 5),
                       ("abdomen", 5), ("thorax", 4)],
            "modes": ["species_no_failure", "quality_no_failure"],
            "note": "no failure modes flagged",
            "why_fail": "—",
        },
    },
]


def _annot_block(ax, x, y, header, header_color, lines, header_fs=10.5,
                 line_fs=9, dy=0.045):
    ax.text(x, y, header, ha="left", va="top",
            fontsize=header_fs, fontweight="bold", color=header_color,
            transform=ax.transAxes)
    for i, line in enumerate(lines):
        ax.text(x, y - dy * (i + 1), line, ha="left", va="top",
                fontsize=line_fs, transform=ax.transAxes,
                family="monospace")


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.5),
                             gridspec_kw={"width_ratios": [1, 1.55]},
                             facecolor="white")

    for row, item in enumerate(PAIR):
        ax_img, ax_txt = axes[row, 0], axes[row, 1]

        img_path = SYN_ROOT / item["species"] / item["file"]
        img = Image.open(img_path).convert("RGB")
        ax_img.imshow(img)
        ax_img.set_axis_off()
        ax_img.set_title(item["label"], fontsize=11.5, loc="left",
                         fontweight="bold", pad=8)

        ax_txt.set_axis_off()
        ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)

        # File header
        ax_txt.text(0.0, 1.0, item["file"], ha="left", va="top",
                    fontsize=9.5, family="monospace", color="#444",
                    transform=ax_txt.transAxes)
        ax_txt.text(0.0, 0.955, f"target species: {item['species']}",
                    ha="left", va="top", fontsize=9.5,
                    transform=ax_txt.transAxes)

        # LLM block
        llm = item["llm"]
        llm_lines = [
            f"  verdict          {llm['verdict']}",
            f"  blind ID         {llm['blind_id']}",
            f"  diag level       {llm['diag']}",
            f"  morph mean       {llm['morph_mean']:.2f}",
            "  per-feature      " + "  ".join(f"{n}={s}" for n, s in llm["scores"]),
            f"  rationale        {llm['summary']}",
        ]
        _annot_block(ax_txt, 0.0, 0.88, "LLM-strict filter",
                     llm["verdict_color"], llm_lines)

        # Centroid block
        c = item["centroid"]
        cent_lines = [f"  verdict          {c['verdict']}"]
        _annot_block(ax_txt, 0.0, 0.535, "BioCLIP centroid filter",
                     c["verdict_color"], cent_lines)

        # Expert block
        e = item["expert"]
        exp_lines = [
            f"  verdict          {e['verdict']}",
            f"  blind ID         {e['blind_id']}",
            f"  diag level       {e['diag']}",
            f"  morph mean       {e['morph_mean']:.2f}",
            "  per-feature      " + "  ".join(f"{n}={s}" for n, s in e["scores"]),
            f"  failure modes    {', '.join(e['modes'])}",
            f"  free-text note   {e['note']}",
            f"  why strict-fail  {e['why_fail']}",
        ]
        _annot_block(ax_txt, 0.0, 0.45, "Expert annotation (Jessie)",
                     e["verdict_color"], exp_lines)

    fig.suptitle("Where automated filters and the expert disagree — "
                 "two B. ashtoni examples (raw 150-image expert sample)",
                 fontsize=12.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT}")
    print(f"wrote {OUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
