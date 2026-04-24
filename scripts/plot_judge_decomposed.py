#!/usr/bin/env python3
"""Per-species diagnostic plots for v2 (tri-state visibility) judge.

For each rare species (ashtoni, sandersoni, flavidus):

  (A) mean_morph_visible scatter, coloured by v2 diag level, marker by v1 tier.
      The pass region (mm >= 4 AND diag in {species, genus}) is shaded.
      Lets the reader see why images pass or fail under v2.

  (B) Contact-sheet grid of v1 tier × v2 verdict transitions (strict_pass and
      soft_fail rows only, since those are the two tiers we sampled from):
          row 1: v1 strict_pass → v2 PASS (stable good)
          row 2: v1 strict_pass → v2 FAIL (v2 stricter)
          row 3: v1 soft_fail   → v2 PASS (v2 reclassifies correctly)
          row 4: v1 soft_fail   → v2 FAIL (genuine fidelity or coverage issues)
      Thumbs annotated with (mm, diag, vis_skip_count).

Outputs under docs/plots/judge_decomposed/:
  {species}_mm_scatter.{png,pdf}
  {species}_contact_transitions.{png,pdf}
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline.config import PROJECT_ROOT  # noqa: E402

ROOT = PROJECT_ROOT
OUT = ROOT / "docs/plots/judge_decomposed"

SPECIES = [
    ("ashtoni", "Bombus_ashtoni", "B. ashtoni", "#0072B2"),
    ("sandersoni", "Bombus_sandersoni", "B. sandersoni", "#E69F00"),
    ("flavidus", "Bombus_flavidus", "B. flavidus", "#009E73"),
]


def morph_mean_v1(r):
    m = r.get("morphological_fidelity", {})
    s = [f["score"] for k, f in m.items()
         if isinstance(f, dict)
         and not f.get("not_visible", False)
         and f.get("score") is not None]
    return sum(s) / len(s) if s else 0.0


def classify_v1(r):
    if not r.get("blind_identification", {}).get("matches_target"):
        return "hard_fail"
    if r.get("diagnostic_completeness", {}).get("level") != "species":
        return "soft_fail"
    if morph_mean_v1(r) >= 4.0:
        return "strict_pass"
    return "borderline"


def load_triplet(stem: str):
    meta = json.load(open(ROOT / f'RESULTS_kfold/llm_judge_decomposed/{stem}_tier_test_files_meta.json'))
    v1 = {r['file']: r for r in json.load(open(ROOT / 'RESULTS_kfold/llm_judge_eval/results.json'))['results']}
    v2 = {r['file']: r for r in json.load(open(ROOT / f'RESULTS_kfold/llm_judge_decomposed/{stem}_tier_v2trivis.json'))['results']}
    files = meta['soft_fails'] + meta['strict_passes']
    rows = []
    for f in files:
        r1, r2 = v1[f], v2[f]
        vis_list = [r2['morphological_fidelity'][k].get('visibility', 'visible')
                    for k in ('legs_appendages','wing_venation_texture','head_antennae',
                              'abdomen_banding','thorax_coloration')]
        rows.append({
            'file': f,
            'v1_tier': classify_v1(r1),
            'v2_pass': bool(r2.get('overall_pass')),
            'v2_mm': r2.get('mean_morph_visible', 0.0),
            'v2_diag': (r2.get('diagnostic_completeness') or {}).get('level'),
            'v2_vis': vis_list,
            'v2_skipped': sum(1 for v in vis_list if v != 'visible'),
        })
    return meta, rows


# ── Panel A: mean_morph × diag scatter ──────────────────────────────────────


DIAG_COLOR = {"species": "#3a8a56", "genus": "#d08a3a", "family": "#c5443a", "none": "#777"}
TIER_MARKER = {"strict_pass": "o", "soft_fail": "X", "borderline": "s", "hard_fail": "^"}


def plot_mm_scatter(stem, pretty, rows):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    # Shade pass region: mm >= 4 AND diag in {species, genus}
    ax.add_patch(Rectangle((3.5, -0.5), 5.5 - 3.5, 2.0,
                           facecolor="#d0e8c8", alpha=0.35, zorder=0))
    ax.axvline(4.0 - 0.5, color="#555", lw=0.8, ls="--", zorder=1)

    import random
    random.seed(0)

    diag_order = ["species", "genus", "family", "none"]
    diag_y = {d: i for i, d in enumerate(diag_order)}

    plotted = set()
    for r in rows:
        y = diag_y.get(r['v2_diag'], 3) + random.uniform(-0.15, 0.15)
        x = r['v2_mm'] + random.uniform(-0.06, 0.06)
        marker = TIER_MARKER.get(r['v1_tier'], 'o')
        color = DIAG_COLOR.get(r['v2_diag'], '#777')
        edge = '#111' if r['v2_pass'] else '#c5443a'
        ax.scatter([x], [y], c=color, marker=marker, s=55, alpha=0.8,
                   edgecolors=edge, linewidths=1.0, zorder=3)

    # Legend dummies
    for tier, marker in TIER_MARKER.items():
        if tier in ('borderline', 'hard_fail'): continue  # not sampled
        ax.scatter([], [], c='#888', marker=marker, s=55, edgecolors='#111',
                   label=f"v1 {tier.replace('_', ' ')}")
    for diag in ("species", "genus"):
        ax.scatter([], [], c=DIAG_COLOR[diag], marker='s', s=55, edgecolors='none',
                   label=f"v2 diag = {diag}")
    ax.scatter([], [], facecolors='none', edgecolors='#c5443a', marker='o', s=70,
               linewidths=1.2, label="v2 FAIL (red edge)")

    ax.set_xlim(2.8, 5.2)
    ax.set_ylim(-0.5, 3.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(diag_order)
    ax.set_xlabel("v2 mean_morph over VISIBLE features", fontsize=11)
    ax.set_ylabel("v2 diagnostic_completeness level", fontsize=11)
    ax.set_title(
        f"{pretty}  (n={len(rows)}: v1 soft_fail + v1 strict_pass)\n"
        "Green shading = v2 pass region (mm ≥ 4 AND diag ∈ {species, genus})",
        fontsize=11)
    ax.legend(loc='lower left', fontsize=9, framealpha=0.95)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ('png', 'pdf'):
        out = OUT / f"{stem}_mm_scatter.{ext}"
        fig.savefig(out, dpi=200, bbox_inches='tight')
        print(f"wrote {out}")
    plt.close(fig)


# ── Panel B: contact-transition sheet ───────────────────────────────────────


def plot_contact_transitions(stem, species_dir, pretty, rows):
    img_dir = ROOT / f'RESULTS_kfold/synthetic_generation/{species_dir}'
    cells = {
        ("strict_pass", True): [r for r in rows if r['v1_tier'] == 'strict_pass' and r['v2_pass']],
        ("strict_pass", False): [r for r in rows if r['v1_tier'] == 'strict_pass' and not r['v2_pass']],
        ("soft_fail", True): [r for r in rows if r['v1_tier'] == 'soft_fail' and r['v2_pass']],
        ("soft_fail", False): [r for r in rows if r['v1_tier'] == 'soft_fail' and not r['v2_pass']],
    }
    titles = {
        ("strict_pass", True):  f"v1 strict_pass → v2 PASS  (stable good)  n={len(cells[('strict_pass', True)])}",
        ("strict_pass", False): f"v1 strict_pass → v2 FAIL  (v2 stricter)  n={len(cells[('strict_pass', False)])}",
        ("soft_fail", True):    f"v1 soft_fail  → v2 PASS  (v2 correctly reclassifies)  n={len(cells[('soft_fail', True)])}",
        ("soft_fail", False):   f"v1 soft_fail  → v2 FAIL  (genuine fidelity / coverage issues)  n={len(cells[('soft_fail', False)])}",
    }

    N_PER = 8; THUMB = 220; GAP = 10; HEADER = 32; CAP = 26
    LEFT_PAD = 30; TOP_PAD = 76
    W = LEFT_PAD + N_PER * (THUMB + GAP) + GAP
    H = TOP_PAD + 4 * (HEADER + CAP + THUMB + GAP) + GAP
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font_title = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
        font_row = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        font_cap = ImageFont.truetype("DejaVuSans.ttf", 11)
    except Exception:
        font_title = font_row = font_cap = ImageFont.load_default()

    draw.text((LEFT_PAD, 12),
              f"{pretty}: v1 tier → v2 verdict transitions (tri-state rubric)",
              fill="black", font=font_title)
    draw.text((LEFT_PAD, 38),
              "mm = mean_morph_visible,  diag = diagnostic_completeness,  "
              "skip = # features marked not_assessable/not_visible (out of 5)",
              fill="#333", font=font_cap)

    y = TOP_PAD
    for key in [("strict_pass", True), ("strict_pass", False),
                ("soft_fail", True), ("soft_fail", False)]:
        draw.text((LEFT_PAD, y), titles[key], fill="black", font=font_row)
        y += HEADER
        items = cells[key][:N_PER]
        for col in range(N_PER):
            x = LEFT_PAD + col * (THUMB + GAP)
            if col >= len(items):
                continue
            r = items[col]
            img = Image.open(img_dir / r['file']).convert("RGB")
            w, h = img.size; side = min(w, h)
            img = img.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2)).resize((THUMB, THUMB))
            canvas.paste(img, (x, y))
            cap = f"mm={r['v2_mm']:.1f}  diag={r['v2_diag']}  skip={r['v2_skipped']}"
            tw = draw.textlength(cap, font=font_cap)
            draw.text((x + (THUMB - tw) // 2, y + THUMB + 4), cap,
                      fill="black", font=font_cap)
        y += THUMB + CAP + GAP

    for ext in ('png', 'pdf'):
        out = OUT / f"{stem}_contact_transitions.{ext}"
        canvas.save(out)
        print(f"wrote {out}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, species_dir, pretty, _color in SPECIES:
        meta, rows = load_triplet(stem)
        plot_mm_scatter(stem, pretty, rows)
        plot_contact_transitions(stem, species_dir, pretty, rows)


if __name__ == "__main__":
    main()
