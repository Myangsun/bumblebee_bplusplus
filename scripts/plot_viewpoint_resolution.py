#!/usr/bin/env python3
"""Viewpoint-resolution plot: does v2 tri-state actually fix the frontal-shot
pattern that dominated the v1 contact-sheet FAIL panels?

The v1 contact sheets at RESULTS_kfold/llm_judge_eval/contact_Bombus_<species>.png
show soft_fail + hard_fail images. Frontal poses are massively over-represented
in those FAIL panels because v1's binary `not_visible` flag forced the judge to
either mark the abdomen invisible (21% of frontal ashtoni) or guess a low score
on a feature it couldn't fairly grade — in both paths, the judge then reported
diag="genus", landing the image in soft_fail.

This plot shows two things, per species:

  Left panel: v1 soft_fail rate by viewpoint, computed over the FULL 500-image
              pool (not just our 80-image cohort). This is the raw
              contact-sheet problem.

  Right panel: on images we re-scored with v2 (the 80-image tier cohort), the
               v2 verdict by viewpoint — broken into PASS (correct tri-state
               handling) vs FAIL. Also shows % of each viewpoint where v2
               marks abdomen_banding `not_assessable` or `not_visible`, which
               is the mechanism by which v2 removes the noise.

Output: docs/plots/judge_decomposed/viewpoint_resolution_3species.{png,pdf}
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT  # noqa: E402

ROOT = PROJECT_ROOT
OUT = ROOT / "docs/plots/judge_decomposed"

SPECIES = [
    ("ashtoni", "Bombus_ashtoni", "B. ashtoni", "#0072B2"),
    ("sandersoni", "Bombus_sandersoni", "B. sandersoni", "#E69F00"),
    ("flavidus", "Bombus_flavidus", "B. flavidus", "#009E73"),
]

ANGLES = ["frontal", "three-quarter_anterior", "lateral",
          "three-quarter_posterior", "dorsal"]
ANGLE_LABEL = {
    "frontal": "frontal",
    "three-quarter_anterior": "3/4 ant.",
    "lateral": "lateral",
    "three-quarter_posterior": "3/4 post.",
    "dorsal": "dorsal",
}


def angle_of(fn: str) -> str:
    parts = fn.split("::")
    return parts[3].rsplit("_", 1)[0] if len(parts) >= 4 else "?"


def morph_mean_v1(r):
    m = r.get("morphological_fidelity", {})
    s = [f["score"] for k, f in m.items()
         if isinstance(f, dict)
         and not f.get("not_visible", False)
         and f.get("score") is not None]
    return sum(s) / len(s) if s else 0.0


def v1_tier(r):
    if not r.get("blind_identification", {}).get("matches_target"):
        return "hard_fail"
    if r.get("diagnostic_completeness", {}).get("level") != "species":
        return "soft_fail"
    if morph_mean_v1(r) >= 4.0:
        return "strict_pass"
    return "borderline"


def load_v1_full(species_dir: str) -> list[dict]:
    rows = json.load(open(ROOT / "RESULTS_kfold/llm_judge_eval/results.json"))["results"]
    return [r for r in rows if r.get("species") == species_dir]


def load_v2_cohort(stem: str):
    meta = json.load(open(ROOT / f"RESULTS_kfold/llm_judge_decomposed/{stem}_tier_test_files_meta.json"))
    v2 = {r["file"]: r for r in json.load(open(ROOT / f"RESULTS_kfold/llm_judge_decomposed/{stem}_tier_v2trivis.json"))["results"]}
    files = meta["soft_fails"] + meta["strict_passes"]
    return meta, files, v2


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 11.5), sharex=False)

    for row_idx, (stem, species_dir, pretty, color) in enumerate(SPECIES):
        # ── Left: v1 soft_fail rate by angle, on full 500-image pool ──
        ax_l = axes[row_idx, 0]
        v1_all = load_v1_full(species_dir)
        counts_by_angle = Counter(angle_of(r["file"]) for r in v1_all)
        soft_by_angle = Counter(angle_of(r["file"]) for r in v1_all
                                if v1_tier(r) == "soft_fail")
        rates = [soft_by_angle.get(a, 0) / max(1, counts_by_angle.get(a, 1)) * 100
                 for a in ANGLES]
        bars = ax_l.bar(range(len(ANGLES)), rates, color=color, edgecolor="black")
        for bar, v, a in zip(bars, rates, ANGLES):
            n = counts_by_angle.get(a, 0)
            sf = soft_by_angle.get(a, 0)
            ax_l.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                      f"{sf}/{n}", ha="center", fontsize=9)
        ax_l.set_xticks(range(len(ANGLES)))
        ax_l.set_xticklabels([ANGLE_LABEL[a] for a in ANGLES], fontsize=10)
        ax_l.set_ylabel("v1 soft_fail rate (%)", fontsize=10)
        ax_l.set_ylim(0, max(rates + [1]) * 1.35 if rates else 1)
        ax_l.grid(axis="y", alpha=0.3)
        ax_l.set_title(f"{pretty}  —  v1 soft_fail rate by viewpoint\n(full pool, n={len(v1_all)}; soft_fail = contact-sheet FAIL cell)",
                       fontsize=11)
    
        # ── Right: v2 verdict × viewpoint on tier cohort, with tri-state usage ──
        ax_r = axes[row_idx, 1]
        meta, files, v2 = load_v2_cohort(stem)
        # For each angle: v2 pass / v2 fail / % abdomen skipped
        angle_data = {a: {"pass": 0, "fail": 0, "abd_skip": 0, "n": 0} for a in ANGLES}
        for f in files:
            a = angle_of(f)
            if a not in angle_data: continue
            r = v2[f]
            angle_data[a]["n"] += 1
            if r.get("overall_pass"):
                angle_data[a]["pass"] += 1
            else:
                angle_data[a]["fail"] += 1
            vis = (r.get("morphological_fidelity", {}) or {}).get("abdomen_banding", {}).get("visibility", "visible")
            if vis in ("not_assessable", "not_visible"):
                angle_data[a]["abd_skip"] += 1
        x = np.arange(len(ANGLES))
        w = 0.38
        pass_h = [angle_data[a]["pass"] for a in ANGLES]
        fail_h = [angle_data[a]["fail"] for a in ANGLES]
        skip_pct = [angle_data[a]["abd_skip"] / max(1, angle_data[a]["n"]) * 100
                    for a in ANGLES]
        ns = [angle_data[a]["n"] for a in ANGLES]
        b_pass = ax_r.bar(x - w/2, pass_h, w, color="#2e7d4b", edgecolor="black", label="v2 PASS")
        b_fail = ax_r.bar(x - w/2, fail_h, w, bottom=pass_h, color="#c5443a",
                          edgecolor="black", label="v2 FAIL")
        # Overlay skip % on right y-axis
        ax_r2 = ax_r.twinx()
        b_skip = ax_r2.bar(x + w/2, skip_pct, w, color="#888", alpha=0.6,
                           edgecolor="black", label="v2 abdomen_banding\nnot_assessable / not_visible (%)")
        for bar, v in zip(b_skip, skip_pct):
            if v > 0:
                ax_r2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                           f"{v:.0f}%", ha="center", fontsize=9, color="#333")
        # Pass-rate labels on top of pass+fail stacks
        for i, a in enumerate(ANGLES):
            total = angle_data[a]["n"]
            if total > 0:
                p = angle_data[a]["pass"]
                ax_r.text(i - w/2, total + 0.3, f"{p}/{total}",
                          ha="center", fontsize=9)
        ax_r.set_xticks(x); ax_r.set_xticklabels([ANGLE_LABEL[a] for a in ANGLES], fontsize=10)
        ax_r.set_ylabel("v2 verdict count (stacked, PASS+FAIL)", fontsize=10)
        ax_r2.set_ylabel("% of v2 judgements where abdomen_banding is skipped", fontsize=10)
        ax_r.set_title(f"{pretty}  —  v2 verdict + tri-state usage by viewpoint\n(80-image cohort: 40 v1 soft_fail + 40 v1 strict_pass)",
                       fontsize=11)
        # Combined legend
        lines1, labels1 = ax_r.get_legend_handles_labels()
        lines2, labels2 = ax_r2.get_legend_handles_labels()
        ax_r.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9, framealpha=0.95)
        ax_r2.set_ylim(0, max(skip_pct + [5]) * 1.35 if skip_pct else 1)
        ax_r.grid(axis="y", alpha=0.3)
    
    fig.suptitle("Does v2 tri-state resolve the viewpoint-coverage confound?  "
                 "Left: v1 soft_fail rate by pose (the problem).  "
                 "Right: v2 verdict + abdomen-skip rate by pose (the fix).",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        out = OUT / f"viewpoint_resolution_3species.{ext}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
