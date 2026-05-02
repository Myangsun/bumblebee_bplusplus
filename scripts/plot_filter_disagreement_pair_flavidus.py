#!/usr/bin/env python3
"""Same-species disagreement panel for B. flavidus.

Image A: passes BOTH automated filters (LLM-strict + centroid) but FAILS expert.
Image B: passes expert but FAILS BOTH the LLM-strict and the centroid filter.

Output: docs/plots/filters/expert_vs_automated_disagreement_pair_flavidus.{png,pdf}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT  # noqa: E402

ROOT = PROJECT_ROOT
SYN_ROOT = ROOT / "RESULTS_kfold/synthetic_generation"
OUT = ROOT / "docs/plots/filters/expert_vs_automated_disagreement_pair_flavidus.png"
TAU_FLA = 0.6817

PAIR = [
    {
        "label": "Image A — automated filters PASS, expert FAILS",
        "file": "Bombus_flavidus::0192::female::three-quarter_anterior_0.jpg",
        "species": "Bombus_flavidus",
        "llm": {
            "verdict": "PASS (strict)", "verdict_color": "#2ca02c",
            "blind_id": "B. flavidus  (matches target ✓)",
            "diag": "species",
            "morph_mean": 4.20,
            "scores": [("legs", 4), ("wings", 4), ("head/ant", 4),
                       ("abdomen", 4), ("thorax", 5)],
            "summary": '"Head is mostly black with some yellow hairs above '
                       'the antenna base, consistent with the species." '
                       '(LLM did not flag the missing antenna or '
                       'malformed mouthparts.)',
        },
        "centroid": {"verdict": "PASS  0.785 ≥ τ 0.682",
                     "verdict_color": "#2ca02c"},
        "expert": {
            "verdict": "FAIL (strict)", "verdict_color": "#d62728",
            "blind_id": "B. flavidus (called male; GT = female)",
            "diag": "species",
            "morph_mean": 3.20,
            "scores": [("legs", 4), ("wings", 4), ("head/ant", 1),
                       ("abdomen", 3), ("thorax", 4)],
            "modes": ["species_other"],
            "note": '"missing antenna, mouth doesn\'t seem right"',
            "why_fail": "morph mean 3.20 < 4  (head/antennae = 1)",
        },
    },
    {
        "label": "Image B — expert PASS, both LLM and centroid FAIL",
        "file": "Bombus_flavidus::0028::female::three-quarter_posterior_0.jpg",
        "species": "Bombus_flavidus",
        "llm": {
            "verdict": "FAIL (strict)", "verdict_color": "#d62728",
            "blind_id": "B. ternarius  (matches target ✗)",
            "diag": "genus",
            "morph_mean": 3.60,
            "scores": [("legs", 4), ("wings", 4), ("head/ant", 3),
                       ("abdomen", 3), ("thorax", 4)],
            "summary": '"Banding pattern … more consistent with '
                       'Bombus ternarius." (LLM mis-identified as '
                       'a non-target species and demoted diag to '
                       'genus.)',
        },
        "centroid": {"verdict": "FAIL  0.604 < τ 0.682",
                     "verdict_color": "#d62728"},
        "expert": {
            "verdict": "PASS (strict)", "verdict_color": "#2ca02c",
            "blind_id": "B. flavidus (correctly identified)",
            "diag": "species",
            "morph_mean": 4.25,
            "scores": [("legs", 5), ("wings", 4), ("head/ant", "N/A"),
                       ("abdomen", 4), ("thorax", 4)],
            "modes": ["species_no_failure", "quality_no_failure"],
            "note": "no failure modes flagged "
                    "(head/antennae not visible from this angle)",
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
        img = Image.open(SYN_ROOT / item["species"] / item["file"]).convert("RGB")
        ax_img.imshow(img); ax_img.set_axis_off()
        ax_img.set_title(item["label"], fontsize=11.5, loc="left",
                         fontweight="bold", pad=8)

        ax_txt.set_axis_off(); ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
        ax_txt.text(0.0, 1.0, item["file"], ha="left", va="top",
                    fontsize=9.5, family="monospace", color="#444",
                    transform=ax_txt.transAxes)
        ax_txt.text(0.0, 0.955, f"target species: {item['species']}",
                    ha="left", va="top", fontsize=9.5,
                    transform=ax_txt.transAxes)

        L = item["llm"]
        llm_lines = [
            f"  verdict          {L['verdict']}",
            f"  blind ID         {L['blind_id']}",
            f"  diag level       {L['diag']}",
            f"  morph mean       {L['morph_mean']:.2f}",
            "  per-feature      " + "  ".join(f"{n}={s}" for n, s in L["scores"]),
            f"  rationale        {L['summary']}",
        ]
        _annot_block(ax_txt, 0.0, 0.88, "LLM-strict filter",
                     L["verdict_color"], llm_lines)

        c = item["centroid"]
        _annot_block(ax_txt, 0.0, 0.535, "BioCLIP centroid filter",
                     c["verdict_color"], [f"  verdict          {c['verdict']}"])

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
                 "two B. flavidus examples (raw 150-image expert sample)",
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
