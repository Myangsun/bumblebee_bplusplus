# Baselines Walkthrough — algorithms + code, to learn & reimplement

A teaching guide to the **8 additional baselines** (beyond the D1–D6 data conditions). For each:
the **idea**, the **algorithm/math**, the **exact code position**, the **flag to run it**, and a
**"build it yourself"** note. All code is in `pipeline/train/simple.py` unless stated.

## The shared harness (what's held constant)
Everything below plugs into ONE training loop; only the marked axis changes.
- Model: `SimpleClassifier` (`simple.py:99`) — ResNet-50, ImageNet weights, head `2048→512→16`.
- Loop: `train_model` (`:480`) calls `train_epoch` (`:365`) + `validate_epoch`; early stops on val
  acc (patience 15); saves `best_f1.pt` (the @f1 checkpoint = best val macro-F1).
- Loss is built once by `build_criterion` (`:278`); default = `nn.CrossEntropyLoss`.
- Per-class training counts: `compute_class_counts` (`:217`) — the raw material for every long-tail method.

**Mental model — the 8 fall into 4 families:**
| Family | What it changes | Baselines |
|--------|-----------------|-----------|
| Loss (objective) | how each sample is scored → gradients | weighted CE, Balanced Softmax, LDAM-DRW |
| Augmentation (data on-the-fly) | the batches the model sees | Remix, BS+CMO |
| Decoupling (2-stage classifier) | freeze rep, re-fit classifier | cRT, LWS |
| Generative recipe (2-stage) | how synthetic data is incorporated | Fill-Up-style |

---

# Family 1 — Loss re-weighting / margin
These change **only** the loss. Same data, same predictions; the loss re-scales or shifts before
backprop, so the gradients (and thus the learned model) differ. Selection is still @f1.

## 1. Class-balanced weighted CE — Cui et al., CVPR 2019
- **Idea:** rare classes contribute little to plain CE (few images). Re-weight each class's loss so
  rare classes count more. Weight ∝ inverse "effective number of samples" (not raw inverse frequency
  — overlapping samples give diminishing returns).
- **Math:** effective number `E_n = (1−β^n)/(1−β)`; weight `w_c ∝ (1−β)/(1−β^{n_c})`, β=0.9999,
  normalized to sum to #classes. Then plain CE with those class weights.
- **Code:** `_class_balanced_weights` (`:231`); wired in `build_criterion` (`:294`):
  ```python
  return nn.CrossEntropyLoss(weight=_class_balanced_weights(class_counts).to(device))
  ```
- **Run:** `--loss weighted_ce`.
- **Build it yourself:** compute counts → weight vector → pass `weight=` to `nn.CrossEntropyLoss`.
  That's the whole method (2 lines). Everything else is the shared loop.

## 2. Balanced Softmax — Ren et al., NeurIPS 2020
- **Idea:** the classifier's softmax is biased toward frequent classes. Correct it by **adding the
  log class-prior to the logits** before the softmax, so the loss accounts for the train imbalance.
- **Math:** loss = `CE(z + log(prior), y)`, where `prior_c = n_c / N`. (Adding `log n_c` is
  equivalent up to a constant that cancels in the softmax.)
- **Code:** `BalancedSoftmaxLoss` (`:239`):
  ```python
  self.register_buffer("log_prior", torch.log(prior + 1e-12))
  def forward(self, logits, target):
      return F.cross_entropy(logits + self.log_prior, target)
  ```
- **Run:** `--loss balanced_softmax`. (Also the loss inside Fill-Up and BS+CMO.)
- **Build it yourself:** precompute `log_prior` (shape `[C]`); add to logits inside a tiny
  `nn.Module` that calls `F.cross_entropy`. One line of real logic.

## 3. LDAM-DRW — Cao et al., NeurIPS 2019
- **Idea:** two ingredients. (a) **LDAM**: give rare classes a *larger decision margin* (so the model
  must be more confident to predict a common class over a rare one). (b) **DRW** (deferred
  re-weighting): train normally first, then *switch on* class-balanced weights partway through — this
  learns good features first, then re-balances the classifier.
- **Math:** per-class margin `m_c = C / n_c^{1/4}` (scaled so the max = 0.5). Subtract `m_y` from the
  *true* class logit, scale all logits by `s=30`, then CE. DRW: from a chosen epoch, apply
  class-balanced weights to that CE.
- **Code:** `LDAMLoss` (`:252`) — margins in `__init__`, the forward subtracts the margin on the true
  class only:
  ```python
  logits_m = torch.where(index, logits - margin, logits)   # margin on true class
  return F.cross_entropy(self.s * logits_m, target, weight=self.weight)
  ```
  DRW trigger lives in `train_model` (`:521-524`): at `drw_epoch` it calls `set_drw_weight` (`:267`).
  Default `drw_epoch = min(10, epochs//2)` (`:1000`) — **must be < patience** or the best-F1
  checkpoint is chosen before DRW fires (this was the review fix).
- **Run:** `--loss ldam_drw` (`--drw-epoch N` to override).
- **Build it yourself:** the margin forward is ~5 lines; the tricky part is the **schedule** — you
  need the training loop to swap the loss weights at a set epoch, so the loss can't be a fixed object.
  Study how `train_model` reaches into `criterion` mid-loop.

---

# Family 2 — Augmentation / oversampling
These change the **batches**, inside `train_epoch` (`:365`). The branch order is
`cmo → remix → mixup → plain`.

## 4. Remix — Chou et al., ECCV-W 2020
- **Idea:** MixUp (blend two images + their labels) but **bias the blended label toward the rare
  class** — so a head+rare mix is labeled mostly as the rare class, giving the tail more signal.
- **Math:** mix images `x = λx_i + (1−λ)x_j`, λ∼Beta(1,1). Label weight `λ_y`: `0` if
  `n_i/n_j ≥ κ ∧ λ<τ`; `1` if `n_i/n_j ≤ 1/κ ∧ (1−λ)<τ`; else `λ`. (κ=3, τ=0.5.)
- **Code:** `train_epoch` (`:397-411`):
  ```python
  ratio = cc[y_i] / cc[y_j]
  lam_y = torch.full((N,), lam, device=device)
  lam_y[(ratio >= kappa) & (lam < tau)] = 0.0
  lam_y[(ratio <= 1/kappa) & ((1-lam) < tau)] = 1.0
  loss = (lam_y*ce_i + (1-lam_y)*ce_j).mean()
  ```
- **Run:** `--remix`.
- **Build it yourself:** start from MixUp (`:413-420`), then replace the scalar label weight with the
  per-sample `λ_y` rule. Needs per-class counts in the loop and `F.cross_entropy(..., reduction="none")`.

## 5. BS+CMO — Park et al., CVPR 2022 (Context-rich Minority Oversampling + Balanced Softmax)
- **Idea:** paste a **rare-class foreground** (CutMix patch) onto a **common-class background**, so
  rare classes get seen in many contexts. The label is area-weighted. Combined with Balanced Softmax.
- **Math:** two streams — background = normal loader, foreground = minority-weighted sampler. CutMix
  box (λ∼Beta(1,1)); paste; `λ_adj = 1 − box_area/(H·W)`;
  `loss = λ_adj·BS(bg_label) + (1−λ_adj)·BS(fg_label)`.
- **Code:** the foreground stream is a 2nd loader `fg_loader` (`:930`) with a class-balanced sampler
  (`_class_balanced_sampler`, `:904`), fed batch-by-batch via `_infinite_iter` (`:320`); the paste +
  loss are in `train_epoch` (`:377-395`):
  ```python
  images[:, :, y1:y2, x1:x2] = fg_images[:, :, y1:y2, x1:x2]
  lam_adj = 1.0 - ((y2-y1)*(x2-x1)/(H*W))
  loss = lam_adj*criterion(outputs, labels) + (1-lam_adj)*criterion(outputs, fg_labels)
  ```
  `--cmo` also auto-switches the loss to Balanced Softmax (`:1010`).
- **Run:** `--cmo`.
- **Build it yourself:** the new concept vs Remix is the **second DataLoader** with a minority sampler
  and an infinite iterator, plus CutMix box math. The label mix is by pixel area, not a Beta draw.

---

# Family 3 — Decoupling (2-stage classifier)
Kang et al., ICLR 2020: **good features come from normal (instance-balanced) training; the imbalance
problem lives in the classifier.** So train the backbone once, then re-fit *only* the classifier with
class-balanced sampling. Both reuse a shared **stage-1 representation** (`baseline_seed{N}_crtbase`,
made by `jobs/train_crt_stage1.sh` = a plain D1 run).

## 6. cRT (classifier re-training)
- **Idea:** freeze the backbone, **re-initialize the classifier head**, retrain just the head with a
  class-balanced sampler.
- **Code:** model surgery in `run()` (`:961-977`):
  ```python
  model.load_state_dict(state["model_state_dict"])   # stage-1 rep
  for p in model.backbone.parameters(): p.requires_grad = False
  for m in model.classifier:
      if isinstance(m, nn.Linear): m.reset_parameters()   # fresh head
  ```
  Class-balanced sampler: `_class_balanced_sampler` (`:904`, weight `1/n_c`); optimizer only sees
  trainable params (`:1006`, the `requires_grad` filter).
- **Run:** stage 2 = `--decouple-crt --init-from baseline_seed{N}_crtbase`.
- **Build it yourself:** load checkpoint → set `backbone.requires_grad_(False)` → `reset_parameters()`
  on the head → build a `WeightedRandomSampler(weights=1/count[label])` → optimizer over trainable
  params only.

## 7. Decouple-LWS (learnable weight scaling)
- **Idea:** even cheaper than cRT — **don't retrain the head, just learn one scalar per class** that
  rescales its classifier weights (fixes the norm imbalance between head/tail).
- **Math:** logit_c → `γ_c · logit_c`; learn the 16 `γ_c` with class-balanced sampling, everything
  else frozen.
- **Code:** `LWSWrapper` (`:327`) — a frozen base model + a learnable `logit_scale`:
  ```python
  def forward(self, x): return self.base(x) * self.logit_scale
  def state_dict(...):   # fold scale into final linear so eval loads a plain model
      sd[wkey] = sd[wkey] * scale.view(-1,1); sd[bkey] = sd[bkey] * scale
  ```
  Wired at `run()` `:968-971` (freeze all, wrap).
- **Run:** stage 2 = `--decouple-lws --init-from baseline_seed{N}_crtbase`.
- **Build it yourself:** wrap the frozen model, add `nn.Parameter(torch.ones(C))`, multiply logits by
  it, train only that vector. The one subtlety is **folding** the scale into the final `Linear` at
  save time so evaluation needs no special code — study `LWSWrapper.state_dict`.

---

# Family 4 — Generative two-stage recipe

## 8. Fill-Up-style — Shin, Kang & Park, 2023
- **Idea:** don't just concatenate synthetic images. Two stages: **(I)** train on real+synthetic
  (rich, balanced-ish) then **(II)** fine-tune on real-only to re-anchor to the true distribution —
  Balanced Softmax in both, with the prior kept at the **real** long-tail.
- **Code:** no single new function — it's **orchestration** + one reused capability:
  - The reused capability is the generic **warm-start**: `--init-from` *without* decouple loads a full
    checkpoint and fine-tunes all params (`run()` `:961` block, the `else` path).
  - The **real-only prior**: `compute_class_counts(real_only=True)` (`:217`, excludes `::` synthetic
    files) → passed as `bs_prior_counts` to `build_criterion` (`:990-994`).
  - The two stages are two `run.py` calls in `jobs/train_fillup_stage1.sh` (train on `d4_synthetic`/
    `d6_probe`, `--loss balanced_softmax --bs-real-prior --randaugment`) and
    `jobs/train_fillup_stage2.sh` (`--dataset raw --init-from <stage1> --lr 1e-5 ...`).
- **Run:** `sbatch train_fillup_stage1.sh` then `--dependency=afterok:<s1> train_fillup_stage2.sh`.
  Both stages are reported (Stage I = `_s1` tag, Stage II = the plain tag), like Table 3.
- **Build it yourself:** the pieces are (a) a warm-start loader, (b) a prior that ignores synthetic
  images, (c) two training calls chained by a checkpoint. No custom loss/layer — it's a *recipe*.

---

## How to read a run's config
Every run writes `training_metadata.json` → `hyperparameters` records `loss_type`, `drw_epoch`,
`randaugment`, `mixup_alpha`, `remix`, `cmo`, `bs_real_prior`, `decouple_crt/lws`, `init_from` — so
you can confirm exactly which axis was active. The reported metric is the `@f1` checkpoint.

## Suggested learning path
1. **Balanced Softmax** (1 line) → understand loss vs. selection-metric.
2. **Weighted CE** (weight vector) → the re-weighting family.
3. **LDAM-DRW** → margins + the mid-training schedule (hardest loss).
4. **MixUp → Remix** → batch-level augmentation + label rules.
5. **BS+CMO** → a second minority-sampled data stream.
6. **cRT → LWS** → freezing + the state_dict fold trick.
7. **Fill-Up** → composing everything into a two-stage recipe.
