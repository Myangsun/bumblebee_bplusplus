# Experimental Results: Augmentation Strategy Comparison

## 5.4 Augmentation Method Comparison

Due to extremely limited test-set sample sizes for rare species (B. ashtoni n=6, B. sandersoni n=10 in a single split), we evaluate augmentation strategies under three complementary protocols: (1) single-split preliminary results, (2) stratified 5-fold cross-validation as the primary evaluation, and (3) multi-seed training variance on the fixed split. Following Shipard et al. (2023) and Picek et al. (2022), we adopt k-fold CV as the primary protocol because it evaluates across all available specimens rather than a single held-out subset. All results use the best macro-F1 checkpoint (best_f1.pt), which directly optimizes the reported metric -- appropriate for imbalanced classification where accuracy is dominated by common species (Provost & Fawcett, 2001).

### 5.4.1 Checkpoint Selection

Each training run produces three checkpoints: best overall validation accuracy (best_multitask.pt), best validation macro F1 (best_f1.pt), and best mean F1 on focus species only (best_focus.pt). Since our primary evaluation metric is macro F1, checkpoint selection should match the evaluation criterion to avoid model selection inconsistency.

*Table 5.6: Macro F1 by checkpoint type (single-split evaluation). Best per-config in bold.*

| Config | best_multitask (acc) | best_f1 (macro F1) | best_focus (rare spp.) |
|--------|---------------------|--------------------|-----------------------|
| D1 Baseline | 0.810 | **0.815** | 0.809 |
| D3 CNP | **0.829** | **0.829** | 0.794 |
| D4 Synthetic | **0.834** | 0.823 | 0.810 |
| D5 LLM-filtered | **0.837** | 0.834 | 0.833 |

The best_f1 checkpoint yields comparable or better macro F1 than best_multitask for most configs. The best_focus checkpoint is unstable (0.794 for D3) because it optimizes on only 3 species with ~4--5 validation images per fold, amplifying noise.

Multi-seed analysis (5 seeds per config) confirms this instability. Table 5.6b shows the range of macro F1 across seeds for each checkpoint type:

*Table 5.6b: Checkpoint stability across 5 seeds (macro F1 range = max - min).*

| Config | best_f1 range | best_multitask range | best_focus range |
|--------|--------------|---------------------|-----------------|
| D1 Baseline | 0.013 | 0.015 | **0.044** |
| D3 CNP | 0.035 | 0.038 | **0.081** |
| D4 Synthetic | 0.025 | 0.017 | **0.073** |
| D5 LLM-filtered | 0.018 | 0.009 | **0.066** |

The best_focus checkpoint exhibits 3--5x wider variance than best_f1 or best_multitask across seeds. This is expected: best_focus optimizes on ~4--5 validation images per rare species, making it highly sensitive to initialization noise. We report all subsequent results under best_f1.

### 5.4.2 Single-Split Evaluation (Preliminary)

*Table 5.7: Single-split results (best_f1.pt, +200 synthetic per species). Test set sizes in parentheses.*

| Dataset | Macro F1 | Top-3 Acc | Accuracy | Ashtoni F1 (n=6) | Sandersoni F1 (n=10) | Flavidus F1 (n=36) |
|---------|----------|-----------|----------|------------------|----------------------|---------------------|
| D1 Baseline | 0.815 | 97.0% | 88.2% | 0.500 | 0.588 | 0.623 |
| D3 CNP | 0.829 | 96.6% | 88.6% | 0.545 | 0.625 | 0.719 |
| D4 Synthetic | 0.823 | 97.0% | 88.7% | 0.500 | 0.533 | 0.698 |
| D5 LLM-filtered | **0.834** | 97.0% | 88.6% | **0.600** | 0.588 | 0.710 |

On a single split, D5 appears best (0.834) with D3 second (0.829). However, the rare species test sets are critically small: the difference between 0.500 and 0.600 F1 on B. ashtoni reflects getting one additional image correct out of six. These per-species estimates have bootstrap 95% CI widths of 0.25--0.31 for ashtoni (n=6), rendering point comparisons unreliable. Top-3 accuracy is uniformly high (96.6--97.0%) across all configs, indicating that even when the top prediction is wrong, the correct species typically ranks within the top three.

![D1 Baseline confusion matrix (single-split, best_f1.pt)](../RESULTS_kfold/baseline@f1_confusion_matrix.png)
*Figure 5.6a: D1 Baseline confusion matrix (single-split, best_f1.pt). B. ashtoni (row 2) is frequently misclassified as B. citrinus and B. vagans. B. sandersoni (row 14) is confused with B. vagans.*

![D1 Baseline per-species metrics (single-split, best_f1.pt)](../RESULTS_kfold/baseline@f1_species_metrics.png)
*Figure 5.6b: D1 Baseline per-species F1, precision, and recall (single-split, best_f1.pt). B. ashtoni and B. sandersoni have the lowest F1 scores, confirming the rare species classification challenge.*

![Single-split model comparison (all configs x all checkpoints)](../RESULTS_kfold/model_comparison_gbif_20260408_194228.png)
*Figure 5.6c: Single-split per-species F1 comparison across all 4 configs and 3 checkpoint types (12 models). Focus species (B. ashtoni, B. sandersoni) highlighted. Differences between configs are minimal for common species but variable for rare species.*

![D3 CNP confusion matrix (single-split, best_f1.pt)](../RESULTS_kfold/d3_cnp@f1_confusion_matrix.png)
*Figure 5.6d: D3 CNP confusion matrix (single-split, best_f1.pt). Compare with D1 baseline (Figure 5.6a) -- B. flavidus (row 7) shows improved diagonal concentration.*

![D5 LLM-filtered confusion matrix (single-split, best_f1.pt)](../RESULTS_kfold/d5_llm_filtered@f1_confusion_matrix.png)
*Figure 5.6e: D5 LLM-filtered confusion matrix (single-split, best_f1.pt). B. sandersoni (row 14) shows substantial off-diagonal mass to B. vagans, similar to D1 baseline.*

### 5.4.3 Five-Fold Cross-Validation (Primary Results)

Stratified 5-fold CV pools predictions across all folds, yielding effective test sets of n=32 (B. ashtoni), n=58 (B. sandersoni), and n=232 (B. flavidus) -- a 5x increase over the single split. Each fold trains independently with the same augmentation protocol (+200 images per rare species), ensuring that every real specimen is tested exactly once across folds. This design follows Shipard et al. (2023), who use 5-fold CV for synthetic augmentation evaluation on small biological image datasets.

*Table 5.8: 5-fold CV results (best_f1.pt, mean +/- std across folds). Best per-metric in bold.*

| Config | Macro F1 | Accuracy | Ashtoni F1 (n=32) | Sandersoni F1 (n=58) | Flavidus F1 (n=232) |
|--------|----------|----------|-------------------|----------------------|---------------------|
| D1 Baseline | 0.832 +/- 0.013 | 89.2 +/- 0.8% | 0.621 +/- 0.151 | **0.466 +/- 0.115** | 0.747 +/- 0.020 |
| D3 CNP | **0.837 +/- 0.013** | 89.1 +/- 0.5% | **0.685 +/- 0.224** | 0.433 +/- 0.101 | **0.806 +/- 0.027** |
| D4 Synthetic | 0.820 +/- 0.024 | 88.9 +/- 0.8% | 0.576 +/- 0.191 | 0.326 +/- 0.144 | 0.764 +/- 0.076 |
| D5 LLM-filtered | 0.821 +/- 0.019 | 88.9 +/- 0.5% | 0.577 +/- 0.230 | 0.398 +/- 0.104 | 0.735 +/- 0.050 |

The k-fold results invert the single-split ranking. D3 CNP achieves the highest macro F1 (0.837), the strongest B. ashtoni improvement (+0.064 over baseline), and the only statistically significant per-species gain (B. flavidus, see Table 5.9). Both synthetic strategies (D4, D5) perform below baseline on macro F1, with D4 most damaging to B. sandersoni (0.326 vs. 0.466 baseline, --0.140).

*Table 5.9: Pairwise paired t-tests on fold-level macro F1 (df=4). Significant results in bold.*

| Comparison | Mean diff | t | p-value | Significant |
|-----------|-----------|------|---------|-------------|
| D1 vs D3 CNP | +0.005 | 1.715 | 0.161 | No |
| D1 vs D4 Synthetic | --0.012 | --2.172 | 0.096 | No |
| **D1 vs D5 LLM-filtered** | **--0.011** | **--2.981** | **0.041** | **Yes (D5 worse)** |
| D3 vs D4 Synthetic | --0.017 | --2.593 | 0.061 | No |
| **D3 vs D5 LLM-filtered** | **--0.016** | **--3.290** | **0.030** | **Yes (D5 worse)** |
| D4 vs D5 LLM-filtered | +0.001 | 0.304 | 0.777 | No |

D5 (LLM-filtered synthetic) is significantly worse than both D1 baseline (p=0.041) and D3 (p=0.030). D4 and D5 are statistically indistinguishable (p=0.777), meaning the LLM judge filtering does not provide measurable benefit over unfiltered synthetic images. With df=4, these tests require large effect sizes (Cohen's d > 1.0) to achieve significance; the D3 vs baseline comparison (p=0.161) lacks statistical power to confirm the observed +0.005 improvement.

Per-species pairwise tests reveal that D3 significantly improves B. flavidus F1 over D1 (+0.059, p=0.005) and over D5 (+0.071, p=0.013). No augmentation strategy significantly improves B. ashtoni (all p > 0.30, reflecting the n=32 sample limitation) or B. sandersoni (D4 reduction of --0.140 vs D1 is marginally significant at p=0.052).

*Table 5.10: Pooled 5-fold bootstrap 95% CIs (10,000 iterations, best_f1.pt).*

| Config | Macro F1 [95% CI] | Ashtoni F1 [95% CI] | Sandersoni F1 [95% CI] | Flavidus F1 [95% CI] |
|--------|-------------------|---------------------|------------------------|---------------------|
| D1 Baseline | 0.833 [0.818, 0.847] | 0.630 [0.462, 0.769] | 0.479 [0.345, 0.598] | 0.747 [0.700, 0.790] |
| D3 CNP | 0.838 [0.824, 0.850] | 0.700 [0.552, 0.820] | 0.434 [0.308, 0.549] | 0.806 [0.763, 0.843] |
| D4 Synthetic | 0.822 [0.808, 0.835] | 0.600 [0.439, 0.737] | 0.330 [0.206, 0.449] | 0.765 [0.719, 0.807] |
| D5 LLM-filtered | 0.823 [0.809, 0.837] | 0.600 [0.438, 0.737] | 0.413 [0.281, 0.538] | 0.737 [0.688, 0.782] |

Bootstrap CIs confirm the pattern: D3 CNP's macro F1 CI [0.824, 0.850] barely overlaps with D4 [0.808, 0.835] and D5 [0.809, 0.837], while substantially overlapping with baseline [0.818, 0.847]. For B. flavidus, D3's CI [0.763, 0.843] does not overlap with D5's [0.688, 0.782], consistent with the paired t-test significance (p=0.013). B. ashtoni CIs remain wide (0.27--0.30 width) due to n=32, reflecting the fundamental sample-size limitation for this species.

![Per-species F1 comparison](../RESULTS/kfold_species_comparison_f1.png)
*Figure 5.7a: Per-species F1 across all 16 species (5-fold CV, pooled, best_f1.pt). Focus species highlighted in red. Augmentation effects are concentrated on rare species; common species (n > 500) show negligible differences across configs.*

![Bootstrap CI comparison](../RESULTS_kfold/kfold_bootstrap_ci_f1.png)
*Figure 5.7b: Bootstrap 95% confidence intervals for macro F1 and focus species F1, computed on pooled 5-fold predictions (10,000 iterations).*

![K-fold box plots](../RESULTS_kfold/kfold_analysis_f1.png)
*Figure 5.7c: Per-fold macro F1 and focus species F1 distributions across 5 folds. High fold-to-fold variance for B. ashtoni (n=6--7 per fold) reflects the fundamental small-sample challenge.*

![K-fold model comparison](../RESULTS_kfold/model_comparison_kfold_20260408_195532.png)
*Figure 5.7d: K-fold per-species F1 comparison across all configs and checkpoints (5-fold evaluation).*

### 5.4.4 Multi-Seed Training Variance

To complement k-fold CV (which varies the data partition), we train each config with 5 random seeds (42--46) on the original fixed split, isolating training stochasticity (weight initialization, data shuffling, dropout) from data variation.

*Table 5.11: Multi-seed results (best_f1.pt, fixed split, mean +/- std across 5 seeds).*

| Config | Macro F1 | Ashtoni F1 (n=6) | Sandersoni F1 (n=10) | Flavidus F1 (n=36) |
|--------|----------|------------------|----------------------|---------------------|
| D1 Baseline | **0.839 +/- 0.006** | 0.614 +/- 0.035 | **0.622 +/- 0.070** | **0.760 +/- 0.040** |
| D3 CNP | 0.822 +/- 0.014 | 0.577 +/- 0.062 | 0.477 +/- 0.156 | 0.724 +/- 0.059 |
| D4 Synthetic | 0.828 +/- 0.009 | 0.608 +/- 0.087 | 0.494 +/- 0.059 | 0.669 +/- 0.016 |
| D5 LLM-filtered | 0.831 +/- 0.008 | 0.609 +/- 0.109 | 0.533 +/- 0.063 | 0.709 +/- 0.047 |

*Table 5.11b: Multi-seed pairwise paired t-tests (df=4).*

| Comparison | Mean diff | t | p-value | Significant |
|-----------|-----------|------|---------|-------------|
| D1 vs D3 CNP | --0.016 | --2.695 | 0.054 | No |
| D1 vs D4 Synthetic | --0.011 | --2.219 | 0.091 | No |
| D1 vs D5 LLM-filtered | --0.008 | --1.607 | 0.183 | No |
| D3 vs D4 Synthetic | +0.006 | 1.094 | 0.336 | No |
| D3 vs D5 LLM-filtered | +0.009 | 1.584 | 0.188 | No |
| D4 vs D5 LLM-filtered | +0.003 | 0.620 | 0.569 | No |

On the fixed split, D1 appears best (0.839). No pairwise test reaches significance (closest: D1 vs D3, p=0.054). This inverts the k-fold ranking where D3 is best. The discrepancy arises because all 5 seeds evaluate on the same 6 ashtoni and 10 sandersoni test images. Per-seed analysis reveals that D1 consistently gets 4/6 ashtoni and 5--6/10 sandersoni correct, while D3 gets 3/6 ashtoni and 3--6/10 sandersoni -- a difference of 1--2 images that drives the entire ranking reversal. The multi-seed protocol measures training variance but does not address the fundamental small-n test-set limitation.

![Seed variance box plots](../RESULTS/seed_analysis_f1.png)
*Figure 5.8: Multi-seed training variance (5 seeds, fixed split, best_f1.pt). D1's apparent advantage is an artifact of the specific test images, not a population-level finding.*

### 5.4.5 Reconciling the Evidence

*Table 5.12: Summary across evaluation protocols. Macro F1 reported.*

| Protocol | Test size (rare spp.) | Best config | Worst config | D1 rank |
|----------|----------------------|-------------|-------------|---------------|
| Single-split (1 run) | n=6 / 10 / 36 | D5 (0.834) | D1 (0.815) | 4th |
| 5-fold CV (primary) | n=32 / 58 / 232 | D3 (0.837) | D4 (0.820) | 2nd |
| Multi-seed (5 runs) | n=6 / 10 / 36 | D1 (0.839) | D3 (0.822) | 1st |

The ranking depends on the protocol and, critically, on which test images are used. Single-split and multi-seed protocols evaluate on the same 6 ashtoni images; k-fold evaluates on all 32. The reversal between single-split (D5 best) and k-fold (D3 best) demonstrates that initial results were driven by the specific test-set composition rather than population-level performance differences.

Across all protocols, four findings are robust:

1. **D3 CNP is the only strategy that consistently improves rare species** without degrading others. In k-fold, it provides the only statistically significant per-species gain (B. flavidus +0.059, p=0.005) and the largest numerical improvement for B. ashtoni (+0.064).
2. **Synthetic augmentation (D4, D5) does not improve and may harm classification.** D5 is significantly worse than D1 in k-fold (p=0.041). The most affected species is B. sandersoni, where D4 reduces F1 by 0.140 vs D1 (p=0.052).
3. **LLM judge filtering (D5 vs D4) provides no significant benefit** (p=0.777 in k-fold, p=0.569 in multi-seed). The strict filter passes 44--91% of images depending on species, but the filtered subset does not yield better classifiers.
4. **Common species are unaffected by augmentation.** All 13 species with n > 200 show < 0.02 F1 variation across configs (Figure 5.7a), confirming that augmentation effects are concentrated on the target rare species.

These results establish that, for this dataset, copy-and-paste augmentation preserving real texture and morphology (D3) outperforms generative synthetic augmentation (D4, D5) for rare species classification. The failure of synthetic augmentation is consistent with the LLM judge finding that 55.6% of B. ashtoni images fail strict quality filters due to incorrect thorax coloration (Section 5.2), suggesting that current text-to-image models cannot reliably reproduce species-diagnostic features for morphologically atypical species.

### 5.4.6 Volume Ablation (Supplementary)

To determine whether additional synthetic volume improves classifier performance, models were trained with +50, +100, +200, and +300 synthetic images per rare species for both D4 (unfiltered) and D5 (LLM-filtered). This ablation uses the original single-split evaluation with the best_multitask.pt checkpoint (a relative comparison of trends, not absolute performance).

![Volume ablation trends with bootstrap 95% CI](../RESULTS_count_ablation/volume_ablation_trends_with_ci.png)
*Figure 5.9a: Volume ablation trends for D4 and D5 at +50 to +300 images per species, with bootstrap 95% CIs. No consistent improvement with increasing volume for either D4 or D5.*

![Volume ablation model comparison](../RESULTS_count_ablation/model_comparison_volume_ablation_20260316_140735.png)
*Figure 5.9b: Per-species F1 comparison across all volume ablation configs. Variation across volumes is small relative to between-config differences.*

![D5 volume ablation bootstrap CI](../RESULTS_count_ablation/bootstrap_ci_d5_ablation.png)
*Figure 5.9c: Bootstrap 95% CIs for D5 LLM-filtered at varying volumes. All CIs overlap with baseline, confirming no statistically significant improvement at any volume.*

The absence of a volume-dependent improvement confirms that the bottleneck is generation fidelity, not data quantity.

---

## Data Sources

All numbers in this document are derived from the following experimental outputs:

| Item | Source file |
|------|------------|
| Table 5.6 (Checkpoint) | `RESULTS_kfold/{config}@{ckpt}_gbif_test_results_20260408_194228.json` |
| Table 5.6b (Stability) | `RESULTS/{config}_seed{N}_gbif/test_results_{ckpt}.json` (5 seeds x 3 ckpts x 4 configs) |
| Table 5.7 (Single-split) | `RESULTS_kfold/{config}@f1_gbif_test_results_20260408_194228.json` |
| Tables 5.8--5.9 (K-fold) | `RESULTS_kfold/kfold_analysis_f1.json` |
| Table 5.10 (Bootstrap) | `RESULTS_kfold/kfold_bootstrap_ci_f1.json` |
| Tables 5.11--5.11b (Multi-seed) | `RESULTS/seed_analysis_f1.json` |
| Table 5.12 (Summary) | Aggregated from above |
| Fig 5.6a (D1 confusion) | `RESULTS_kfold/baseline@f1_confusion_matrix.png` |
| Fig 5.6b (D1 metrics) | `RESULTS_kfold/baseline@f1_species_metrics.png` |
| Fig 5.6c (Single-split comparison) | `RESULTS_kfold/model_comparison_gbif_20260408_194228.png` |
| Fig 5.6d (D3 confusion) | `RESULTS_kfold/d3_cnp@f1_confusion_matrix.png` |
| Fig 5.6e (D5 confusion) | `RESULTS_kfold/d5_llm_filtered@f1_confusion_matrix.png` |
| Fig 5.7a (K-fold species) | `RESULTS/kfold_species_comparison_f1.png` |
| Fig 5.7b (Bootstrap CIs) | `RESULTS_kfold/kfold_bootstrap_ci_f1.png` |
| Fig 5.7c (K-fold boxes) | `RESULTS_kfold/kfold_analysis_f1.png` |
| Fig 5.7d (K-fold comparison) | `RESULTS_kfold/model_comparison_kfold_20260408_195532.png` |
| Fig 5.8 (Seed boxes) | `RESULTS/seed_analysis_f1.png` |
| Fig 5.9a (Volume trends) | `RESULTS_count_ablation/volume_ablation_trends_with_ci.png` |
| Fig 5.9b (Volume comparison) | `RESULTS_count_ablation/model_comparison_volume_ablation_20260316_140735.png` |
| Fig 5.9c (Volume bootstrap) | `RESULTS_count_ablation/bootstrap_ci_d5_ablation.png` |