#!/usr/bin/env python3
"""
Task 2 — verification report comparing D5 (LLM strict_pass), D2 (BioCLIP
centroid), and D6 (expert-supervised probe) 200-image selections per
rare species. Writes ``RESULTS/filters/d2_d6_assembly_check.md``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from pipeline.config import RESULTS_DIR
from pipeline.evaluate.filters import (RARE_SPECIES, load_expert_labels,
                                        load_llm_judge)
from pipeline.config import PROJECT_ROOT


def _top_n(scores_path: Path, species: str, n: int = 200) -> list[str]:
    d = json.loads(scores_path.read_text())
    rows = [r for r in d["scores"] if r["species"] == species]
    rows.sort(key=lambda r: -float(r["score"]))
    return [r["basename"] for r in rows[:n]]


def _d5_selection(llm_judge_path: Path, species: str, n: int = 200) -> list[str]:
    """D5 picks the first N strict-pass synthetics, by id order
    (matches scripts/assemble_dataset.py used to build prepared_d5)."""
    lj = load_llm_judge(llm_judge_path)
    basenames = sorted([b for b in lj.basename_to_strict_pass if lj.basename_to_strict_pass[b]
                         and b.startswith(species)])
    return basenames[:n]


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    scores_dir = RESULTS_DIR / "filters"
    out_md = scores_dir / "d2_d6_assembly_check.md"

    centroid_path = scores_dir / "centroid_scores.json"
    probe_path = scores_dir / "probe_scores.json"
    llm_judge_path = PROJECT_ROOT / "RESULTS_kfold" / "llm_judge_eval" / "results.json"
    expert_csv = RESULTS_DIR / "expert_validation_results" / "jessie_all_150.csv"

    expert = load_expert_labels(expert_csv)
    lj = load_llm_judge(llm_judge_path)

    lines: list[str] = []
    lines.append("# D2 / D6 Assembly Verification\n")
    lines.append("Three +200-per-species synthetic selections compared: "
                 "**D5** (existing LLM strict_pass), **D2** (BioCLIP centroid "
                 "distance), **D6** (expert-supervised LogisticRegression probe).\n")

    # ── 1. Per-variant per-species counts ────────────────────────────────────
    lines.append("## 1. Per-variant selection counts\n")
    lines.append("| Species | D5 selection | D2 selection | D6 selection |")
    lines.append("|---|---:|---:|---:|")
    selections: dict[str, dict[str, set[str]]] = {}
    for sp in RARE_SPECIES:
        d5 = _d5_selection(llm_judge_path, sp)
        d2 = _top_n(centroid_path, sp)
        d6 = _top_n(probe_path, sp)
        selections[sp] = {"D5": set(d5), "D2": set(d2), "D6": set(d6)}
        lines.append(f"| {sp.replace('Bombus_', 'B. ')} | {len(d5)} | {len(d2)} | {len(d6)} |")

    # ── 2. Jaccard overlap (per species, pairwise) ──────────────────────────
    lines.append("\n## 2. Pairwise Jaccard overlap per species\n")
    lines.append("| Species | D5 ∩ D2 | D5 ∩ D6 | D2 ∩ D6 |")
    lines.append("|---|---:|---:|---:|")
    for sp in RARE_SPECIES:
        s = selections[sp]
        j_d5_d2 = _jaccard(s["D5"], s["D2"])
        j_d5_d6 = _jaccard(s["D5"], s["D6"])
        j_d2_d6 = _jaccard(s["D2"], s["D6"])
        lines.append(f"| {sp.replace('Bombus_', 'B. ')} | {j_d5_d2:.3f} | {j_d5_d6:.3f} | {j_d2_d6:.3f} |")

    # ── 3. Score-distribution per variant per species ──────────────────────
    lines.append("\n## 3. Score distributions per variant\n")
    lines.append("| Filter | Species | min | 25% | median | 75% | max |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for name, path in (("D2 centroid", centroid_path), ("D6 probe", probe_path)):
        d = json.loads(path.read_text())
        for sp in RARE_SPECIES:
            vals = np.array([r["score"] for r in d["scores"] if r["species"] == sp])
            p = np.percentile(vals, [0, 25, 50, 75, 100])
            lines.append(f"| {name} | {sp.replace('Bombus_', 'B. ')} | "
                         f"{p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {p[3]:.3f} | {p[4]:.3f} |")

    # ── 4. LLM tier breakdown of each variant's selection ───────────────────
    lines.append("\n## 4. LLM tier of each variant's 200-image selection\n")
    lines.append("Per variant and species, number of selected synthetics whose LLM `strict_pass` flag is True.\n")
    lines.append("| Species | D5 strict | D2 strict | D6 strict |")
    lines.append("|---|---:|---:|---:|")
    for sp in RARE_SPECIES:
        def count_strict(sel: set[str]) -> int:
            return sum(1 for b in sel if lj.basename_to_strict_pass.get(b, False))
        lines.append(f"| {sp.replace('Bombus_', 'B. ')} | "
                     f"{count_strict(selections[sp]['D5'])} | "
                     f"{count_strict(selections[sp]['D2'])} | "
                     f"{count_strict(selections[sp]['D6'])} |")

    # ── 5. Expert-label coverage of each variant's selection ────────────────
    lines.append("\n## 5. Expert-label coverage of each variant's selection\n")
    lines.append(
        "Of the 150 expert-annotated synthetics, how many fall in each "
        "variant's 200-image selection, and of those, how many pass the "
        "expert **strict** rule?\n"
    )
    lines.append("| Species | D5 covered (strict✓) | D2 covered (strict✓) | D6 covered (strict✓) |")
    lines.append("|---|---|---|---|")
    for sp in RARE_SPECIES:
        def cov(sel: set[str]) -> tuple[int, int]:
            exp_in = [b for b in sel if b in expert.basename_to_strict]
            exp_pass = sum(1 for b in exp_in if expert.basename_to_strict[b])
            return len(exp_in), exp_pass
        d5c, d5p = cov(selections[sp]["D5"])
        d2c, d2p = cov(selections[sp]["D2"])
        d6c, d6p = cov(selections[sp]["D6"])
        lines.append(f"| {sp.replace('Bombus_', 'B. ')} | "
                     f"{d5c} ({d5p}) | {d2c} ({d2p}) | {d6c} ({d6p}) |")

    # ── 6. Sanity: selection files exist and have right counts ──────────────
    from pipeline.config import GBIF_DATA_DIR
    lines.append("\n## 6. Assembled prepared directories\n")
    lines.append("| Variant | Path | train images per rare species |")
    lines.append("|---|---|---|")
    for variant_name, prep_dir in (
        ("D2 centroid", GBIF_DATA_DIR / "prepared_d2_centroid"),
        ("D6 probe",    GBIF_DATA_DIR / "prepared_d6_probe"),
    ):
        per_sp_counts = []
        for sp in RARE_SPECIES:
            tr = prep_dir / "train" / sp
            n_total = sum(1 for p in tr.glob("*") if p.is_file() or p.is_symlink()) if tr.exists() else 0
            n_synth = sum(1 for p in tr.glob("*") if "::" in p.name) if tr.exists() else 0
            per_sp_counts.append(f"{sp.replace('Bombus_', 'B. ')}: {n_total} total / {n_synth} synth")
        lines.append(f"| {variant_name} | `{prep_dir.name}` | {', '.join(per_sp_counts)} |")

    out_md.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out_md}")
    print()
    # Also print the key overlap table to stdout
    print("\n".join(lines))


if __name__ == "__main__":
    main()
