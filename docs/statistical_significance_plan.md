# Statistical Significance Plan

## The Problem

Per-species F1 with support of 6 (ashtoni) and 10 (sandersoni) has very high variance. A single misclassification changes F1 by 10-20%. We can't reliably claim one model is better than another.

## Approach: Three Complementary Methods

### 1. Bootstrap Confidence Intervals on Per-Species Metrics

**What:** Resample the test set predictions with replacement (10,000 iterations), compute F1 each time, report 95% CI.

**Why:** Shows the uncertainty band around reported F1. If CIs overlap between two models, the difference is not significant.

**Input:** `detailed_predictions` from `test_results.json` (already stored: image path, ground truth, prediction, correct flag).

**Implementation:** New script `scripts/statistical_tests.py`

```python
def bootstrap_per_species_f1(predictions, n_bootstrap=10000):
    """Resample predictions, compute per-species F1 each time."""
    # Group by species, resample within species, compute F1
    # Return: {species: {mean, std, ci_lower, ci_upper}}
```

### 2. McNemar's Test on Paired Predictions

**What:** For two models evaluated on the same test set, count images where model A is right and B is wrong (and vice versa). McNemar's chi-squared test determines if the disagreement is significant.

**Why:** Uses the full 2,362 test images (not just rare species), so it has much more statistical power. Tests overall accuracy difference.

**Input:** Two `test_results.json` files with `detailed_predictions` — match by `image_path`.

```python
def mcnemars_test(predictions_a, predictions_b):
    """Paired test on 2,362 test images."""
    # b = A_correct & B_wrong, c = A_wrong & B_correct
    # chi2 = (|b-c| - 1)^2 / (b+c), p-value from chi2 distribution
```

### 3. Multi-Seed Training (Mean +/- Std)

**What:** Train each dataset configuration 3-5 times with different random seeds. Report mean +/- std of per-species F1.

**Why:** Controls for training randomness (weight init, data shuffling, dropout). If the variance across seeds is larger than the difference between models, the improvement is not robust.

**Changes needed in `pipeline/train/simple.py`:**
1. Add `--seed` CLI argument (default: 42)
2. Add `set_seed()` call at start of `run()`
3. Store seed in `training_metadata.json`

```python
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Output structure:**
```
RESULTS/
├── d5_llm_filtered_seed42_gbif/
├── d5_llm_filtered_seed43_gbif/
├── d5_llm_filtered_seed44_gbif/
```

**SLURM job for multi-seed:**
```bash
for SEED in 42 43 44; do
    sbatch jobs/train_multiseed.sh <dataset> $SEED
done
```

## Statistical Tests Script

**New file:** `scripts/statistical_tests.py`

Functions:
- `bootstrap_ci(results_json, n_bootstrap=10000)` — per-species F1 with 95% CI
- `mcnemars_test(results_a, results_b)` — chi2 statistic, p-value, effect size
- `multi_seed_summary(result_jsons)` — mean +/- std per-species F1
- `significance_report(model_pairs)` — full comparison table with all three methods

CLI:
```bash
# Bootstrap CIs for a single model
python scripts/statistical_tests.py bootstrap --results RESULTS/baseline_gbif/test_results.json

# McNemar's between two models
python scripts/statistical_tests.py mcnemar --model-a RESULTS/baseline_gbif --model-b RESULTS/d5_llm_filtered_gbif

# Multi-seed summary
python scripts/statistical_tests.py multi-seed --pattern "RESULTS/d5_llm_filtered_seed*_gbif"

# Full report
python scripts/statistical_tests.py report --models baseline d3_cnp d4_synthetic d5_llm_filtered
```

## Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `pipeline/train/simple.py` | Modify | Add `--seed`, `set_seed()`, store seed in metadata |
| `scripts/statistical_tests.py` | Create | Bootstrap CIs, McNemar's test, multi-seed summary |
| `jobs/train_multiseed.sh` | Create | SLURM job for multi-seed training |

## Verification

1. **Seed reproducibility:** Train twice with same seed → identical results
2. **Bootstrap CIs:** Run on baseline → CIs for ashtoni (n=6) should be wide (~0.3-1.0), confirming unreliability
3. **McNemar's:** Run on baseline vs D5 → expect p < 0.05 on overall accuracy (2,362 images)
4. **Multi-seed:** 3+ seeds → report mean +/- std; check if improvement exceeds std
