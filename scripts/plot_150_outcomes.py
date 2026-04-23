#!/usr/bin/env python3
"""Figures for Section 5.4.1: expert-annotation outcomes on the 150-image sample.

Produces two figures:
  1. expert_outcomes_150.png  — per-species strict / lenient-only / fail
     stacked bars, overall rollup, and a per-species decomposition of the
     atomic pass criteria (blind-ID match, diagnostic=species, morph-mean
     thresholds, no structural failure).
  2. expert_failure_modes_150.png — failure-mode frequency per species
     (checkbox codes), species_other free-text themes, and a "positive"
     summary row (no-failure flags + caste / sex accuracy).

Styling:
  - Species palette chosen in a muted orange / purple / green family for
    print-friendly academic use. ashtoni = #D89060, sandersoni = #8E7AB5,
    flavidus = #6FA987.
  - Outcome palette: strict = #86aa98 (sage), lenient-only = #c9c9c4 (grey),
    fail = #d86a6a (coral). Background = #f9f9f7.

Output: docs/plots/filters/
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

ROOT = Path("/home/msun14/bumblebee_bplusplus")
CSV = ROOT / "RESULTS/expert_validation_results/jessie_all_150.csv"
OUT = ROOT / "docs/plots/filters"
OUT.mkdir(parents=True, exist_ok=True)

RARE = ["Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus"]
SHORT = {"Bombus_ashtoni": "B. ashtoni",
         "Bombus_sandersoni": "B. sandersoni",
         "Bombus_flavidus": "B. flavidus"}

SPECIES_COLOR = {
    "Bombus_ashtoni":    "#0072B2",
    "Bombus_sandersoni": "#E69F00",
    "Bombus_flavidus":   "#009E73",
}

OUTCOME_COLORS = {
    "Strict pass":       "#86aa98",
    "Lenient pass only": "#c9c9c4",
    "Fail (lenient)":    "#d86a6a",
}
BG = "#f9f9f7"

STRUCTURAL_CODES = {
    "extra_limbs", "missing_limbs", "impossible_geometry",
    "visible_artifact", "visible_artifacts", "blurry_artifacts",
    "repetitive_patterns",
}

MORPH_COLS = [
    "morph_legs_appendages", "morph_wing_venation_texture",
    "morph_head_antennae", "morph_abdomen_banding", "morph_thorax_coloration",
]


def _sex(x: str | float):
    if x in ("worker", "queen", "female"):
        return "female"
    if x == "male":
        return "male"
    return None


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "figure.dpi": 120,
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


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
    df["blind_id_match"] = df["blind_id_species"] == df["ground_truth_species"]
    df["diag_species"] = df["diagnostic_level"] == "species"
    df["diag_genus_or_better"] = df["diagnostic_level"].isin(["genus", "species"])
    df["morph_ge_4"] = df["expert_morph_mean"] >= 4.0
    df["morph_ge_3"] = df["expert_morph_mean"] >= 3.0
    df["no_structural"] = ~df["has_structural"]
    df["lenient"] = df["no_structural"] & df["diag_genus_or_better"] & df["morph_ge_3"]
    df["strict"] = df["blind_id_match"] & df["diag_species"] & df["morph_ge_4"]
    df["gt_sex"] = df["caste_ground_truth"].map(_sex)
    df["bl_sex"] = df["blind_id_caste"].map(_sex)
    df["sex_match"] = df["gt_sex"] == df["bl_sex"]
    return df


def _integer_y(ax, top):
    ax.set_ylim(0, top)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=6))


def _bar_labels(ax, bars, vals, ymax, color="#1a1a1a", fs=8):
    pad = ymax * 0.015
    for rect, v in zip(bars, vals):
        if v <= 0:
            continue
        ax.text(rect.get_x() + rect.get_width() / 2, v + pad, str(int(v)),
                ha="center", va="bottom", fontsize=fs, color=color)


# ---------- Figure A: outcomes + atomic pass criteria ----------

def plot_outcomes(df: pd.DataFrame):
    cats = ["Strict pass", "Lenient pass only", "Fail (lenient)"]
    counts = {c: [] for c in cats}
    for sp in RARE:
        sub = df[df.ground_truth_species == sp]
        s = int(sub.strict.sum())
        l = int(sub.lenient.sum())
        n = len(sub)
        counts["Strict pass"].append(s)
        counts["Lenient pass only"].append(l - s)
        counts["Fail (lenient)"].append(n - l)

    fig = plt.figure(figsize=(11.5, 8.0))
    gs = fig.add_gridspec(
        2, 3, height_ratios=[1, 1], width_ratios=[3, 1, 0.9],
        hspace=0.6, wspace=0.35,
    )
    ax_sp = fig.add_subplot(gs[0, 0])
    ax_tot = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")
    ax_cri = fig.add_subplot(gs[1, :])

    x = np.arange(len(RARE))
    bottoms = np.zeros(len(RARE))
    for c in cats:
        vals = np.array(counts[c])
        ax_sp.bar(x, vals, bottom=bottoms, color=OUTCOME_COLORS[c], label=c,
                  edgecolor=BG, linewidth=1.0, width=0.62)
        for i, v in enumerate(vals):
            if v > 0:
                ax_sp.text(i, bottoms[i] + v / 2, str(int(v)),
                           ha="center", va="center",
                           fontsize=10, color="#1a1a1a")
        bottoms += vals
    ax_sp.set_xticks(x)
    ax_sp.set_xticklabels([SHORT[sp] for sp in RARE], fontstyle="italic")
    ax_sp.set_ylabel("Images (n = 50 per species)")
    _integer_y(ax_sp, 50)
    ax_sp.set_title("(a) Per-species outcome", loc="left")

    # Outcome legend in its own axis so it never overlaps a bar.
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=OUTCOME_COLORS[c], edgecolor=BG, label=c) for c in cats]
    ax_leg.legend(handles=handles, loc="center left", title="Outcome",
                  title_fontsize=10, borderaxespad=0.0)

    totals = [sum(counts[c]) for c in cats]
    bottom = 0
    for c, v in zip(cats, totals):
        ax_tot.bar([0], [v], bottom=[bottom], color=OUTCOME_COLORS[c],
                   width=0.55, edgecolor=BG, linewidth=1.0)
        ax_tot.text(0, bottom + v / 2, f"{int(v)}",
                    ha="center", va="center", fontsize=10, color="#1a1a1a")
        bottom += v
    ax_tot.set_xticks([0])
    ax_tot.set_xticklabels(["All 150"])
    _integer_y(ax_tot, 150)
    ax_tot.set_ylabel("Images")
    ax_tot.set_title("(b) Overall", loc="left")

    # --- Panel (c): atomic pass criteria per species ---
    CRITERIA = [
        ("Blind-ID match\n(species)",   "blind_id_match"),
        ("Diagnostic =\nspecies",       "diag_species"),
        ("Morph-mean\n≥ 4.0",           "morph_ge_4"),
        ("Morph-mean\n≥ 3.0",           "morph_ge_3"),
        ("No structural\nfailure",      "no_structural"),
    ]
    xc = np.arange(len(CRITERIA))
    w = 0.26
    offsets = {RARE[0]: -w, RARE[1]: 0.0, RARE[2]: +w}
    for sp in RARE:
        sub = df[df.ground_truth_species == sp]
        vals = [int(sub[col].sum()) for _, col in CRITERIA]
        bars = ax_cri.bar(xc + offsets[sp], vals, width=w,
                          color=SPECIES_COLOR[sp], label=SHORT[sp],
                          edgecolor=BG, linewidth=0.8)
        _bar_labels(ax_cri, bars, vals, 50, fs=8)
    ax_cri.set_xticks(xc)
    ax_cri.set_xticklabels([c for c, _ in CRITERIA], fontsize=9)
    ax_cri.set_ylabel("Images passing (of 50 per species)")
    _integer_y(ax_cri, 55)
    ax_cri.grid(axis="y", linestyle=":", alpha=0.35, color="#888888")
    ax_cri.set_axisbelow(True)
    ax_cri.set_title(
        "(c) Atomic pass criteria per species  "
        "(strict = Blind-ID ∧ Diagnostic=species ∧ Morph ≥ 4.0; "
        "lenient = No-structural ∧ Diagnostic ≥ genus ∧ Morph ≥ 3.0)",
        loc="left",
    )
    ax_cri.legend(title="Target species", title_fontsize=9,
                  loc="upper center", bbox_to_anchor=(0.5, -0.28),
                  ncols=3, prop={"style": "italic"})

    fig.suptitle(
        "Expert-annotation outcomes on the 150-image sample",
        fontsize=12.5, y=0.995,
    )
    fig.savefig(OUT / "expert_outcomes_150.png")
    fig.savefig(OUT / "expert_outcomes_150.pdf")
    plt.close(fig)
    print(f"saved: {OUT/'expert_outcomes_150.png'}")


# ---------- Figure B: failure modes + species_other + positives/caste ----------

THEME_ORDER = [
    ("Sex / caste mismatch", [r"\bif (this is )?male\b", r"suggests male",
                              r"for male", r"male thorax", r"pollen basket"]),
    ("Posture / context",    [r"flower", r"\bresting", r"flat surface"]),
    ("Leg anatomy",          [r"\blegs?\b", r"spidery", r"spider-?like"]),
    ("Face / head anatomy",  [r"\bantenna", r"proboscis", r"\beyes?\b",
                              r"\bmouth", r"\bface", r"\bhead"]),
    ("Body proportion",      [r"\bbody\b", r"abdomen", r"thorax",
                              r"too large", r"too big", r"too wide",
                              r"very wide", r"very large", r"proportion"]),
    ("Color placement/hue",  [r"color", r"yellow", r"orange", r"\bband",
                              r"coloring", r"vibrant"]),
]


def classify_text(txt: str) -> str:
    t = (txt or "").lower()
    if not t.strip():
        return "Unlabelled other"
    for name, patterns in THEME_ORDER:
        for p in patterns:
            if re.search(p, t):
                return name
    return "Unlabelled other"


CODE_LABELS = {
    "wrong_coloration":    "Wrong\ncoloration",
    "impossible_geometry": "Impossible\ngeometry",
    "wrong_scale":         "Wrong\nscale",
    "extra_missing_limbs": "Extra /\nmissing limbs",
    "repetitive_pattern":  "Repetitive\npattern",
    "flower_unrealistic":  "Flower\nunrealistic",
    "blurry_artifacts":    "Blurry\nartifacts",
    "quality_other":       "Other\nquality",
    "species_other":       "Species other\n(see panel b)",
}


def _grouped_bars(ax, labels, counts_dict, ymax, ylabel, title,
                  show_legend=False, bar_label_fs=7.5):
    x = np.arange(len(labels))
    w = 0.26
    offsets = {RARE[0]: -w, RARE[1]: 0.0, RARE[2]: +w}
    for sp in RARE:
        vals = [counts_dict[lab][sp] for lab in labels]
        bars = ax.bar(x + offsets[sp], vals, width=w,
                      color=SPECIES_COLOR[sp], label=SHORT[sp],
                      edgecolor=BG, linewidth=0.8)
        _bar_labels(ax, bars, vals, ymax, fs=bar_label_fs)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel(ylabel)
    _integer_y(ax, ymax)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", linestyle=":", alpha=0.35, color="#888888")
    ax.set_axisbelow(True)
    if show_legend:
        ax.legend(title="Target species", title_fontsize=9,
                  loc="upper right", prop={"style": "italic"})


def plot_failure_modes(df: pd.DataFrame):
    skip = {"species_no_failure", "quality_no_failure"}

    codes_per_sp = defaultdict(lambda: defaultdict(int))
    for sp in RARE:
        for d in df[df.ground_truth_species == sp].failure_dict:
            for c in (d.get("all") or []):
                if c in skip:
                    continue
                codes_per_sp[c][sp] += 1

    themes_per_sp = defaultdict(lambda: defaultdict(int))
    for sp in RARE:
        sub = df[df.ground_truth_species == sp]
        mask = sub.failure_dict.apply(lambda d: "species_other" in (d.get("species") or []))
        for d in sub[mask].failure_dict:
            themes_per_sp[classify_text(d.get("species_other_text", ""))][sp] += 1

    def _tot(d):
        return sum(d.values())

    code_order = sorted(codes_per_sp.keys(), key=lambda c: -_tot(codes_per_sp[c]))
    code_labels = [CODE_LABELS.get(c, c.replace("_", " ")) for c in code_order]
    codes_lookup = {CODE_LABELS.get(c, c.replace("_", " ")): codes_per_sp[c] for c in code_order}
    code_ymax = max(max(codes_per_sp[c].values()) for c in code_order)
    code_ymax = int(np.ceil(code_ymax * 1.25))

    theme_order = [t for t, _ in THEME_ORDER] + ["Unlabelled other"]
    theme_order = [t for t in theme_order if _tot(themes_per_sp[t]) > 0]
    theme_order = sorted(theme_order, key=lambda t: -_tot(themes_per_sp[t]))
    theme_ymax = max(max(themes_per_sp[t].values()) for t in theme_order)
    theme_ymax = int(np.ceil(theme_ymax * 1.25))

    # Positive / caste summary
    POS_LABELS = [
        "No species\nfailure flag",
        "No quality\nfailure flag",
        "Blind-ID\nspecies match",
        "Sex (♀/♂)\nmatch",
    ]
    pos_per_sp = defaultdict(lambda: defaultdict(int))
    for sp in RARE:
        sub = df[df.ground_truth_species == sp]
        pos_per_sp[POS_LABELS[0]][sp] = int(
            sub.failure_dict.apply(lambda d: "species_no_failure" in (d.get("species") or [])).sum()
        )
        pos_per_sp[POS_LABELS[1]][sp] = int(
            sub.failure_dict.apply(lambda d: "quality_no_failure" in (d.get("quality") or [])).sum()
        )
        pos_per_sp[POS_LABELS[2]][sp] = int(sub.blind_id_match.sum())
        pos_per_sp[POS_LABELS[3]][sp] = int(sub.sex_match.sum())

    fig = plt.figure(figsize=(11.0, 9.0))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 1], hspace=0.65)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[2, 0])

    _grouped_bars(
        ax_a, code_labels, codes_lookup, code_ymax,
        "Image count", "(a) Checkbox failure modes",
        show_legend=True,
    )

    so_total = sum(sum(1 for d in df[df.ground_truth_species == sp].failure_dict
                       if "species_other" in (d.get("species") or []))
                   for sp in RARE)
    _grouped_bars(
        ax_b, theme_order, themes_per_sp, theme_ymax,
        "Image count",
        f"(b) \"Species other\" free-text themes  (n = {so_total} notes)",
    )

    _grouped_bars(
        ax_c, POS_LABELS, pos_per_sp, 55,
        "Images (of 50 per species)",
        "(c) Positive flags and caste accuracy  (higher = better; 50 = ceiling)",
    )

    fig.suptitle(
        "Expert-flagged failure modes and positive metrics on the 150-image sample",
        fontsize=12.5, y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "Panels (a, b): an image may carry multiple tags. Panel (c): ceiling is 50 images per species.",
        ha="center", fontsize=8.5, style="italic", color="#444444",
    )
    fig.savefig(OUT / "expert_failure_modes_150.png")
    fig.savefig(OUT / "expert_failure_modes_150.pdf")
    plt.close(fig)
    print(f"saved: {OUT/'expert_failure_modes_150.png'}")

    audit_rows = []
    for sp in RARE:
        sub = df[df.ground_truth_species == sp]
        mask = sub.failure_dict.apply(lambda d: "species_other" in (d.get("species") or []))
        for _, row in sub[mask].iterrows():
            txt = row.failure_dict.get("species_other_text", "")
            audit_rows.append({
                "species": sp,
                "theme": classify_text(txt),
                "text": txt,
                "basename": Path(row.image_path).name,
            })
    pd.DataFrame(audit_rows).to_csv(OUT / "expert_species_other_themes.csv", index=False)
    print(f"saved: {OUT/'expert_species_other_themes.csv'}")


def main():
    _style()
    df = load_expert()
    plot_outcomes(df)
    plot_failure_modes(df)


if __name__ == "__main__":
    main()
