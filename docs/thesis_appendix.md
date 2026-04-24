## Appendices

### Appendix A: Bumblebee Morphology and Field Guide
Full tergite-level color maps, caste variation tables, and reference photographs for B. ashtoni, B. sandersoni, and B. flavidus. Morphological comparison table.

### Appendix B: Full Prompt Template and Iteration History
Final prompt template (v10), prompt evolution table (v1--v10), environmental and viewpoint configuration.

### Appendix C: LLM Judge Prompt and Pydantic Schema
Complete system prompt, structured output schema, strict filter rules.

### Appendix D: Expert Calibration Results -- Detailed Analysis
Inter-annotator agreement tables, per-feature disagreement matrices, learned filter coefficients, ROC analysis, latent space visualizations.

### Appendix E: Per-Species Detailed Results
Confusion matrices per dataset version, volume ablation full table, per-seed breakdowns, complete 16-species results table, per-angle strict-pass-rate table, caste-fidelity breakdown (referenced in Section 5.3.2), background-removal diagnostic (referenced in Section 5.5.3), per-tier rare / moderate / common F1 breakdown across D1–D6 (referenced in Section 5.5.1), and per-LLM-tier counts per species (referenced in Section 5.3.1).

### Appendix F: Failure-Mode Analysis Assets
Embedding atlases at true t-SNE coordinates (Section 5.2.4); confusion-pair triplets for B. ashtoni × {B. citrinus, B. vagans}, B. sandersoni × B. vagans, B. flavidus × B. citrinus (Section 5.1.2); full per-species galleries of real, synthetic, and harmed-test images (Section 5.6); harmed-and-improved failure-chain galleries with t-SNE projections for all four generative-filter variants (Section 5.6.2).

On-disk chain galleries and thesis-to-filename mapping (seed 42, f1 checkpoint):

| Thesis variant | On-disk directory | Harmed chains | Improved chains |
|---|---|---:|---:|
| D3 unfiltered | `docs/plots/failure/chains_d4_synthetic_{harmed,improved}/` | 49 | 52 |
| D4 LLM-filtered | `docs/plots/failure/chains_d5_llm_filtered_{harmed,improved}/` | 49 | 49 |
| D5 centroid | `docs/plots/failure/chains_d2_centroid_{harmed,improved}/` | 10 | 1 |
| D6 expert-probe | `docs/plots/failure/chains_d6_probe_{harmed,improved}/` | 7 | 1 |

Each gallery contains `gallery/` (horizontal-strip test + 5-NN synthetic panels) and `tsne/` (per-chain BioCLIP t-SNE scatters with test→neighbour arrows). The sharp drop in chain count from D3/D4 → D5/D6 is itself a per-image reading of the Table 5.7 subtractive-ablation result: quality filtering reduces the number of harmed rare-species test images at seed 42 by roughly 5–7×. D6 produces zero harmed chains for B. ashtoni and B. sandersoni — the per-image form of Table 5.7's "no harmful cells" result — while its 7 harmed chains are predominantly B. flavidus. Legacy naming note: on-disk directories prefixed `chains_d4_*` correspond to thesis D3 (unfiltered) and `chains_d5_*` correspond to thesis D4 (LLM-filtered) — the disk names track the experimental configuration slug rather than the thesis variant label.

Additional assets: full dropped-versus-measured 3 × 3 subtractive-ablation recovery matrix for D3, D4, D5 and D6 (Section 5.5.4, `RESULTS/failure_analysis/subset_ablation_recovery.csv`); additive single-species cross-species matrix (Section 5.5.4).


---
