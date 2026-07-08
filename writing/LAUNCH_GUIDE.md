# Experiment Launch Guide — baseline sweep

Prepared for the author to submit. **I have started no jobs.** All scripts are ready to `sbatch` and
carry no `--account`/`--qos` directives.

**Seed policy: 5 seeds (42–46) everywhere.** E1 (leakage) needs no compute; **E2 (D1–D6) already
exists** in `RESULTS/`. The runs to launch are the long-tail/decoupling/aug/Fill-Up baselines below.
All train on the **real-only baseline** (`prepared_split`) except Fill-Up stage 1, so every row is an
apples-to-apples comparison against synthetic augmentation (reviewer R19), on the identical
**ResNet-50 fine-tuned from ImageNet**.

---

## What's left to run

**Cluster limit: `--time` max is 6h.** A single training run is ~3.3h (worst case ~4h at 100
epochs), so the two-stage methods (cRT, Fill-Up) are **split into stage-1 and stage-2 jobs**, each
≤6h, chained with `--dependency=afterok`.

| Baseline | Method(s) | Script(s) | Tasks |
|----------|-----------|-----------|-------|
| E3 losses | weighted CE, Balanced Softmax, LDAM-DRW | `jobs/train_loss_baselines.sh` | 15 |
| E4 aug | Remix, BS+CMO | `jobs/train_aug_baselines.sh` | 10 |
| cRT | stage 1 rep → stage 2 head | `jobs/train_crt_stage1.sh` → `jobs/train_crt_stage2.sh` | 5 + 5 |
| LWS | after cRT stage 1 | `jobs/train_lws_baseline.sh` | 5 |
| Fill-Up | Stage I → Stage II (both reported) | `jobs/train_fillup_stage1.sh` → `jobs/train_fillup_stage2.sh` | 10 + 10 |

All 4 new methods (LWS, Remix, BS+CMO, Fill-Up) are implemented, compiled, and CPU smoke-tested.

---

## Launch (capture each jobid `sbatch` prints)

```bash
cd /home/msun14/bumblebee_bplusplus

# Independent — submit now (loss baselines already running):
sbatch jobs/train_aug_baselines.sh          # E4 Remix + CMO (10)

# cRT: stage 1 → stage 2; LWS also gates on cRT stage 1
CRT1=$(sbatch --parsable jobs/train_crt_stage1.sh)                          # rep (5)
sbatch --dependency=afterok:$CRT1 jobs/train_crt_stage2.sh                  # cRT head (5)
sbatch --dependency=afterok:$CRT1 jobs/train_lws_baseline.sh               # LWS (5)

# Fill-Up: Stage I → Stage II
FU1=$(sbatch --parsable jobs/train_fillup_stage1.sh)                        # Stage I (10)
sbatch --dependency=afterok:$FU1 jobs/train_fillup_stage2.sh                # Stage II (10)

# Evaluate after ALL training finishes (list every training jobid):
sbatch --dependency=afterok:<LOSS>:<AUG>:<CRT1>:<CRT2>:<LWS>:<FU1>:<FU2> \
       jobs/evaluate_baselines.sh                                          # 11 tags × 42–46 = 55
```

`--parsable` makes `sbatch` print just the jobid so you can capture it into a variable as above. The
**E2 primary sweep (D1–D6) needs no launch** — its metrics already exist.

> **If a job still hits the time limit:** it means a single stage exceeded 6h. Lower `--epochs` (e.g.
> `--epochs 60`) in that script's `python run.py` line, or reduce `--patience`, then resubmit — the
> `@f1` checkpoint is saved continuously so a shorter cap still yields a usable model. Timed-out jobs
> keep their `latest_checkpoint.pt`, so you can also add `--resume` and resubmit to continue.

---

## Outputs & how to read them

- Training writes to `RESULTS/baseline_seed{N}_{tag}_gbif/`; tags: `wce, bsm, ldam, crt, lws, remix,
  cmo, fillup_d3_s1, fillup_d3, fillup_d6_s1, fillup_d6`. Use the **`best_f1.pt` (@f1)** checkpoint
  (best val macro-F1 — the primary metric).
- **Fill-Up reports both stages:** `fillup_{d3,d6}_s1` = Stage I (after the real+synthetic stage),
  `fillup_{d3,d6}` = Stage II (after the real-only fine-tune) — both evaluated on the real test split.
- cRT/LWS stage-1 (`baseline_seed{N}_crtbase_gbif`) is **not evaluated** (intermediate rep only).
- Evaluation writes `RESULTS/<model>@f1_seed_test_results_*.json` — use the `@f1` files.
- D1–D6 stems: D1=`baseline`, D2=`d3_cnp`, D3=`d4_synthetic`, D4=`d5_llm_filtered`, D5=`d2_centroid`,
  D6=`d6_probe`, each `_seed{42..46}_gbif`.
- Every model shares the identical real-only test split (`prepared_split/test`, 2362 images), so all
  numbers are directly comparable.

---

## Method notes (defaults; correctness verified vs. papers — see BASELINES.md §5b)

- **weighted_ce** — class-balanced effective-number weighting (Cui 2019, β=0.9999).
- **balanced_softmax** — add `log(train prior)` to logits before CE (Ren 2020).
- **ldam_drw** — LDAM margin (∝ n_c^−¼, s=30) + DRW at epoch `min(10, epochs/2)`=**10**
  (`--drw-epoch N` to override). Fixed from 20→10 after review: at 20 the best-F1 checkpoint was
  reached *before* DRW fired (patience 15), so DRW never influenced the reported model. At 10 (< 15)
  post-DRW epochs are eligible for @f1 selection. Verify via the `activated at epoch` log line vs
  `best_f1_epoch`.
- **remix** — mixup images (λ∼Beta(1,1)); label mix biased to the minority (κ=3, τ=0.5).
- **cmo** — instance-balanced background + minority-weighted foreground; CutMix paste; area-weighted
  label; **Balanced Softmax** loss (auto-selected by `--cmo`).
- **crt** — per seed: stage 1 trains the D1 rep (instance-balanced CE), stage 2 (`--decouple-crt
  --init-from …`) freezes the backbone, re-inits + retrains the head with a class-balanced sampler.
  Two-stage because the D1 seed checkpoints were pruned. Result `baseline_seed{N}_crt_gbif`.
- **lws** — reuses the cRT stage-1 rep; freezes the whole model and learns only a per-class logit
  scale (class-balanced sampler). The scale is folded into the final linear at save (verified
  logit-identical), so eval loads a plain model. Result `baseline_seed{N}_lws_gbif`.
- **fillup_d3(_s1) / fillup_d6(_s1)** — **Stage I** (`_s1`): train on the D3 (`d4_synthetic`) or D6
  (`d6_probe`) pool with Balanced Softmax + RandAugment. **Stage II** (no `_s1`): warm-start real-only
  fine-tune (`--init-from`, `--lr 1e-5`, Balanced Softmax + RandAugment). Both stages use
  **`--bs-real-prior`** so the Balanced-Softmax prior stays at the real long-tail (Fill-Up-faithful).
  Both stages are reported and evaluated on the real test split (cf. Fill-Up Table 3 Stage I/II).
  **One remaining deviation:** under-balance pool (+200/rare, cap 500/sp.), not fill-to-balance — a
  data-availability limit, stated as future work (BASELINES.md §5b).
- **RandAugment / MixUp** — dropped as standalone baselines; RandAugment retained as a component of
  the Fill-Up recipe. `--randaugment` / `--mixup-alpha` flags remain in the harness.

---

## Code changes (for review)

All in `pipeline/train/simple.py`, `run.py`, `pipeline/evaluate/metrics.py`. Compile + CPU smoke
tests pass (LWS fold-equivalence, Remix rule, CMO/Remix train steps, losses, DRW, eval regex).

- `simple.py` — long-tail losses (`build_criterion`, `BalancedSoftmaxLoss`, `LDAMLoss`); MixUp/**Remix/CMO**
  in `train_epoch`; DRW in `train_model`; **cRT** + **LWS** (`LWSWrapper`, `_resolve_init_checkpoint`,
  freeze/reinit, class-balanced sampler, CMO foreground loader via `_infinite_iter`); generalized
  `--init-from` warm-start (Fill-Up stage 2). Threaded through `run()` + CLI.
- `run.py` — `--loss/--drw-epoch/--randaugment/--mixup-alpha/--decouple-crt/--decouple-lws/--remix/--cmo/--bs-real-prior/--init-from`.
- `metrics.py` — `_discover_method_baseline_models()` tags now include
  `crt, lws, remix, cmo, fillup_d3, fillup_d6`.

**Backward compatible:** with no new flags, training behaves exactly as before.

---

## After the runs — what consumes the results

- **W1** rebuilds the main downstream table from the 5-seed D1–D6 results, re-runs paired t-tests
  (**df=4**), and drops CV.
- **W2** adds these baseline rows (Table 3-aligned: 9/13 rows, incl. Fill-Up Stage I + II) and the
  §5 baselines paragraph.

See `TODO.md` for the full writing plan and `BASELINES.md` for the finalized 14-method set +
correctness notes.
