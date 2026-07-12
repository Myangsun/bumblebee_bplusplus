#!/usr/bin/env python3
"""Aggregate every experiment (D1-D6 dataset variants, long-tail learning,
augmentation, Fill-Up) into one comparison table.

Rows computed from raw @f1 test_results (mean +/- sd over seeds 42..46):
  D1-D4  -> RESULTS_seeds/{config}_seed{seed}@f1_seed_test_results_*.json
  methods, Fill-Up -> RESULTS/baseline_seed{seed}_{tag}_gbif/test_results_f1.json
D5-centroid and D6-probe multiseed raw files are not on disk; their values are
taken verbatim from the published Table 5.5 (validated: recomputed D1/D2 match
that table exactly). Emits a console summary and the LaTeX table body.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44, 45, 46)
RARE = ("Bombus_ashtoni", "Bombus_sandersoni", "Bombus_flavidus")  # ash/sand/flav

# key kinds: ("seeds", cfg) RESULTS_seeds ; ("dir", tag) RESULTS ;
#            ("const", dict) verbatim from Table 5.5.
# Grouped by method family (not data provenance): D1-D6 are split into the
# family each one belongs to -- D2 is augmentation, D3-D6 are generative.
METHODS = [
    ("D1 Baseline",        "base",     ("seeds", "baseline")),
    ("Class-Balanced",     "ltlearn",  ("dir", "wce")),  # Cui 2019 effective-number CB reweighting
    ("Balanced-Softmax",   "ltlearn",  ("dir", "bsm")),
    ("LDAM-DRW",           "ltlearn",  ("dir", "ldam")),
    ("cRT",                "ltlearn",  ("dir", "crt")),
    ("LWS",                "ltlearn",  ("dir", "lws")),
    ("D2 Copy-Paste",      "aug",      ("seeds", "d3_cnp")),      # CNP = copy-and-paste
    ("Remix",              "aug",      ("dir", "remix")),
    ("CMO",                "aug",      ("dir", "cmo")),
    ("D3 Unfiltered",      "gen",      ("seeds", "d4_synthetic")),
    ("D4 LLM-filter",      "gen",      ("seeds", "d5_llm_filtered")),
    ("D5 Centroid",        "gen",      ("const", {"macro_m": 0.827, "macro_s": 0.002,
                                                  "rare_m": [0.62, 0.49, 0.70], "acc_m": 0.889})),
    ("D6 Expert-probe",    "gen",      ("const", {"macro_m": 0.834, "macro_s": 0.008,
                                                  "rare_m": [0.64, 0.55, 0.70], "acc_m": 0.890})),
    ("Fill-Up D3 Stage I",  "gen",     ("dir", "fillup_d3_s1")),
    ("Fill-Up D3 Stage II", "gen",     ("dir", "fillup_d3")),
    ("Fill-Up D6 Stage I",  "gen",     ("dir", "fillup_d6_s1")),
    ("Fill-Up D6 Stage II", "gen",     ("dir", "fillup_d6")),
]

GROUP_HEADERS = {
    "base":     "\\emph{Baseline}",
    "ltlearn":  "\\emph{Long-tail learning}",
    "aug":      "\\emph{Data augmentation}",
    "gen":      "\\emph{Generative augmentation (Fill-Up)}",
}


def _seeds_file(config: str, seed: int) -> Path:
    m = sorted((ROOT / "RESULTS_seeds").glob(
        f"{config}_seed{seed}@f1_seed_test_results_*.json"))
    if not m:
        raise FileNotFoundError(f"{config} seed{seed}")
    return m[-1]


def _dir_file(tag: str, seed: int) -> Path:
    p = ROOT / "RESULTS" / f"baseline_seed{seed}_{tag}_gbif" / "test_results_f1.json"
    if not p.exists():
        raise FileNotFoundError(str(p))
    return p


def per_seed_stats(d: dict) -> dict:
    sm = d["species_metrics"]
    return {"macro": float(np.mean([m["f1"] for m in sm.values()])),
            "rare": [sm[r]["f1"] for r in RARE],
            "acc": d["overall_accuracy"]}


def agg(kind: str, ref) -> dict:
    if kind == "const":
        return ref
    macro, acc, rare = [], [], [[], [], []]
    for s in SEEDS:
        f = _seeds_file(ref, s) if kind == "seeds" else _dir_file(ref, s)
        st = per_seed_stats(json.loads(f.read_text()))
        macro.append(st["macro"]); acc.append(st["acc"])
        for i in range(3):
            rare[i].append(st["rare"][i])
    return {"macro_m": float(np.mean(macro)), "macro_s": float(np.std(macro, ddof=1)),
            "acc_m": float(np.mean(acc)), "acc_s": float(np.std(acc, ddof=1)),
            "rare_m": [float(np.mean(r)) for r in rare]}


rows = [(label, group, agg(spec[0], spec[1])) for label, group, spec in METHODS]

# Per-column maxima on the DISPLAYED (rounded) values, so ties bold together.
macro_max = max(round(r[2]["macro_m"], 3) for r in rows)
acc_max = max(round(r[2]["acc_m"] * 100, 1) for r in rows)
rare_max = [max(round(r[2]["rare_m"][i], 2) for r in rows) for i in range(3)]


def rare_str(a):
    return "/".join(f"{v:.2f}"[1:] for v in a["rare_m"])


def fmt_macro(a, bold=True):
    sd = f"$\\pm${a['macro_s']:.3f}" if "macro_s" in a else ""
    s = f"{a['macro_m']:.3f}{sd}"
    return f"\\textbf{{{s}}}" if bold and round(a["macro_m"], 3) == macro_max else s


def fmt_rare(a):
    # No per-species bolding in the rare column (kept plain for readability).
    return "/".join(f"{v:.2f}"[1:] for v in a["rare_m"])


def fmt_acc(a, bold=True):
    s = f"{a['acc_m']*100:.1f}"
    return f"\\textbf{{{s}}}" if bold and round(a["acc_m"] * 100, 1) == acc_max else s


print("\n=== Console summary (mean +/- sd over seeds 42..46) ===")
print(f"{'Method':22s} {'MacroF1':>14s} {'Rare ash/sand/flav':>20s} {'Acc':>7s}")
for label, group, a in rows:
    sd = f"±{a['macro_s']:.3f}" if "macro_s" in a else "  (T5.5)"
    print(f"{label:22s} {a['macro_m']:.3f}{sd}  {rare_str(a):>18s}  {a['acc_m']*100:.1f}%")

# Citation keys appended to the method name in the table (references live in
# the table, not the prose). D1-D6 are defined in the Method section (no cite).
CITES = {
    "D1 Baseline": "he_deep_2015",
    "D2 Copy-Paste": "ghiasi_simple_2021",
    "Class-Balanced": "cui_class-balanced_2019",
    "Balanced-Softmax": "ren_balanced_2020",
    "LDAM-DRW": "cao_learning_2019",
    "cRT": "kang_decoupling_2020",
    "LWS": "kang_decoupling_2020",
    "Remix": "chou_remix_2020",
    "CMO": "park_majority_2022",
    "Fill-Up D3 Stage I": "shin_fill-up_2023",
    "Fill-Up D3 Stage II": "shin_fill-up_2023",
    "Fill-Up D6 Stage I": "shin_fill-up_2023",
    "Fill-Up D6 Stage II": "shin_fill-up_2023",
}

print("\n=== LaTeX body ===")
prev = None
for label, group, a in rows:
    if prev is not None and group != prev:
        print("    \\midrule")
    prev = group
    name = f"{label}~\\cite{{{CITES[label]}}}" if label in CITES else label
    print(f"    {name} & {fmt_macro(a)} & {fmt_rare(a)} & {fmt_acc(a)} \\\\")
