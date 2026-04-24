# GPU rerun checklist (2026-04-23)

Scope: after the 2026-04-23 audit, these training/evaluation jobs were
modified for reproducibility (added `--seed`, fixed focus-species mismatches).
Rerunning them will OVERWRITE existing artefacts with numerically close but
not byte-identical results. Expect every macro F1 in the final tables to move
by ≤ 0.01; re-run `scripts/dump_final_metrics.py` after each block to refresh
`docs/final_metrics.md`, then diff thesis Tables 5.4b / 5.5 / 5.6 / 5.7 / 5.8
/ 5.9 before committing.

## Priority 1 — thesis-table reruns (optional; only if you want full reproducibility)

| Block | Script | Array size | Wall-time | Output overwritten |
|---|---|---:|---:|---|
| 5-fold CV (D1–D4) | `jobs/kfold_train.sh` | 20 | 6 h/task | `RESULTS_kfold/{baseline,d3_cnp,d4_synthetic,d5_llm_filtered}_fold{0..4}_gbif/` |
| 5-fold CV (D5) | `jobs/train_d2_centroid_kfold.sh` | 5 | 4 h/task | `RESULTS/d2_centroid_fold{0..4}_gbif/` |
| 5-fold CV (D6) | `jobs/train_d6_probe_kfold.sh` | 5 | 34 h/task | `RESULTS/d6_probe_fold{0..4}_gbif/` |

After reruns:
```bash
sbatch jobs/kfold_evaluate.sh        # re-eval all folds at f1 checkpoint
sbatch jobs/kfold_evaluate2.sh       # D5/D6 extended
python scripts/dump_final_metrics.py
python scripts/build_species_f1_csv.py
python scripts/analyze_flips.py
```

Multi-seed does not need rerun — `jobs/seed_train.sh`, `train_d2_centroid_multiseed.sh`, and `train_d6_probe_multiseed.sh` already pass `--seed "$SEED"` explicitly (seeds 42–46). Their current artefacts in `RESULTS_seeds/` and `RESULTS/` are already reproducible.

## Priority 2 — single-split reruns (needed ONLY if you want an independently trained single-split comparator; otherwise the seed-42 row from the multi-seed protocol already provides the correct value)

| Script | Seed | Focus-species |
|---|---|---|
| `jobs/train_baseline.sh` | 42 | ashtoni / sandersoni / flavidus |
| `jobs/augment_and_train_d3_cnp.sh` | 42 | ashtoni / sandersoni / flavidus |
| `jobs/assemble_and_train_d4_synthetic.sh` | 42 | ashtoni / sandersoni / flavidus |
| `jobs/train_d4_synthetic.sh` | 42 | ashtoni / sandersoni / flavidus |
| `jobs/assemble_and_train_d5_llm_filtered.sh` | 42 | ashtoni / sandersoni / flavidus |
| `jobs/train_d5_llm_filtered.sh` | 42 | ashtoni / sandersoni / flavidus |

Output: `RESULTS/{variant}_single_seed42_gbif/`. The suffix `single_seed42` separates these from the legacy unseeded single-split artefacts that were used historically. The thesis does not need these to complete Tables 5.4b / 5.5 — both tables read from `RESULTS_seeds/{variant}_seed42@f1_*.json`, which is the multi-seed seed-42 run.

## Priority 3 — volume ablation full grid (run this next)

Expanded from the legacy D3/D4-only volume ablation to the full
4-variant × 4-volume grid (D3 unfiltered, D4 LLM-filtered, D5 centroid,
D6 expert-probe × 50 / 100 / 200 / 300). Seeded at 42, 3 focus-species,
f1 checkpoint. ~96 GPU-hours end-to-end.

```bash
# Step 1 — assemble 16 prepared_* directories (CPU only, ~1h)
sbatch jobs/volume_ablation_assemble.sh

# Step 2 — train 16 models (array 0-15, ~6h each, GPU)
sbatch --dependency=afterok:<assemble_jobid> jobs/volume_ablation_train.sh

# Step 3 — evaluate at f1 checkpoint (suffix volume_ablation)
sbatch --dependency=afterok:<train_jobid> jobs/volume_ablation_evaluate.sh

# Step 4 — regenerate Figure 5.21 and update final_metrics
python scripts/plot_volume_ablation.py
python scripts/dump_final_metrics.py
```

Expected artefacts after step 3:
`RESULTS/{d4_synthetic,d5_llm_filtered,d2_centroid,d6_probe}_{50,100,200,300}@f1_volume_ablation_test_results_*.json`.
The volume=200 runs for each variant should match (within floating-point
noise) the corresponding D3/D4/D5/D6 seed-42 row in Table 5.5 — this
serves as an internal consistency check on the assembly pipeline.

The legacy D3/D4-only runs in `RESULTS_count_ablation/` can be archived
after the full grid lands; Figure 5.21 now points at
`docs/plots/volume_ablation_trends.png`.

## Priority 4 — ablations (already seeded, no rerun needed)

`train_subset_ablation.sh`, `train_subset_ablation_d5_d6.sh`, and
`train_additive_ablation.sh` already passed `--seed 42` explicitly. Tables
5.7 and 5.8 cells match their raw JSONs to 4 decimals — no rerun required.

## Post-rerun checklist

After any rerun, in order:

1. `sbatch jobs/<rerun>.sh` — launch the training array.
2. Wait for completion; check `jobs/logs/*.out` and `.err`.
3. `sbatch jobs/<corresponding evaluate_f1_*.sh>` — re-eval at f1 checkpoint if needed.
4. `python scripts/dump_final_metrics.py` — regenerate `docs/final_metrics.md`.
5. Compare `final_metrics.md` against the thesis tables; update any cell that moved.
6. `python scripts/build_species_f1_csv.py` — regenerate per-species CSVs.
7. `python scripts/analyze_flips.py` — regenerate flip analysis.
8. `python scripts/plot_section4_figures.py` (for §4) and the various `scripts/plot_*.py` and `make_*_plots.py` as relevant — regenerate figures (now with .png + .pdf pairs).

## What is NOT reproducible even after reruns

- The LLM judge (GPT-4o) scores in `RESULTS_kfold/llm_judge_eval/results.json` — API calls are non-deterministic. Re-running would drift within noise but not reproduce exactly.
- The synthetic image generations in `RESULTS_prompt_iteration/synthetic_generation/` — GPT-image-1.5 is non-deterministic. The existing 1,500 images are the canonical thesis inputs and should NOT be regenerated.
- The 150 expert labels in `RESULTS/expert_validation_results/jessie_all_150.csv` — one-time human annotation.
