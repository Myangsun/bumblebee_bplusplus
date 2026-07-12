# Parallel Writing/Figures Session — Handoff Brief

**Read this first.** You are the **parallel writing/figures session**, working *alongside* a main
session that owns the experiments and all results-dependent writing. This brief tells you exactly what
is yours, what to never touch, the concrete tasks (with `file:line`), and the git workflow that keeps
us from clobbering each other.

---

## 0. Context (what the paper is)

ECCV 2026 submission on **synthetic-data augmentation + expert-calibrated filtering** for long-tailed
fine-grained bumblebee classification (16 MA *Bombus* species, ~60:1 imbalance). Files:
`paper-template-Latest/main.tex`, `paper-template-Latest/supplementary.tex`, bibliography
`paper-template-Latest/main.bib`. Method conditions **D1–D6**: D1 real-only baseline; D2 copy-paste;
D3 unfiltered synthetic; D4 LLM-filtered; D5 BioCLIP-centroid-filtered; D6 expert-probe-filtered. Each
augmented condition adds **+200 synthetic images per rare species** (volume parity).

Right now: the main session has SLURM baseline jobs running (losses, aug, cRT/LWS, Fill-Up). Once they
land, the main session rebuilds the results table and stats. **That work is blocked on compute — which
is why we split off the results-independent writing/figures to you, to run in parallel.**

---

## 1. Git workflow — MANDATORY (prevents clobbering)

Work on a **separate branch in a separate worktree** so we never edit the same file on the same branch
at the same time:

```bash
# the MAIN session (or you, once) creates the worktree:
cd /home/msun14/bumblebee_bplusplus
git worktree add ../bb-writing writing-figures

# YOU work exclusively inside:
cd ../bb-writing
```

- Commit small, self-contained chunks with clear messages.
- **Do not merge to `main` yourself.** When a chunk is clean, tell the author; merges happen
  deliberately after a quick conflict check against the main session's in-flight edits.
- Before starting a chunk, `git log --oneline -5 main` to see if the main session has touched your
  target region.

---

## 2. File / region ownership

| Area | Owner | Notes |
|------|-------|-------|
| **Figures / diagram assets** (Diagram_A, t-SNE, ROC, qualitative panels) | **YOU** | asset files + their captions |
| **`main.bib` + citation insertions** | **YOU** | separate file, ~zero collision |
| **Intro / Related Work / Dataset prose polish** (non-results, non-CV, non-protocol) | **YOU** | bounded sections only — see §4 |
| **Results table** (`:~708–753`) + §5 numbers | **MAIN** | rebuilt from 5-seed runs |
| **Baseline rows + §5 baselines paragraph** | **MAIN** | Fill-Up I/II, cRT, LWS, losses, aug |
| **Everything CV-related** (drop five-fold) | **MAIN** | cross-cutting, atomic — see §3 |
| **Probe/centroid fitting sentence** (`:480–481`) | **MAIN** | entangled with CV drop |

---

## 3. DO NOT TOUCH (owned by main session — will cause conflicts)

These are cross-cutting or results-dependent. Leave them entirely alone, **including in the intro and
abstract**, even though it's tempting to polish them:

1. **Any mention of five-fold / CV / folds.** The paper currently frames five-fold CV as "the primary"
   protocol; the main session is *removing CV entirely* (it leaks) and re-centering on
   locked-split × 5 seeds. This edit spans abstract→intro→methods→results→stats and must be done as one
   atomic change. Affected lines to AVOID: `:82`, `:179`, `:308–309`, `:480–481`, `:700`, `:708`,
   `:717`, `:753`. (LOOCV for the probe/centroid calibration at `:436`, `:460–463`, `:639`, `:656` is a
   *different thing* — that's the filter's internal cross-val, not the classifier protocol, and it
   stays. Still, don't edit those either; they're methods-adjacent.)
2. **The main results table** and any sentence citing a macro-F1 / accuracy / p-value number.
3. **The probe/centroid "fitted once on the full real training set" sentence** (`:480–481`) — it's
   factually wrong *and* CV-entangled; the main session will fix it (the probe is fit on 150 synthetic
   labels, not the real training set; only the centroid uses real training images).

If you spot an error inside a DO-NOT-TOUCH region, **write it in `WRITING_NOTES.md`** (create it) for
the main session — don't edit in place.

---

## 4. Your task list (concrete, results-independent)

### T1 — Fix Diagram_A (the pipeline figure) — HIGH PRIORITY
The figure appears to depict **ResNet-50 with a frozen backbone**. That is **wrong**: the shared
classifier is **fine-tuned end-to-end** from ImageNet weights (verified in
`pipeline/train/simple.py:99–129` — no `requires_grad=False` anywhere; and the prose already says so at
[main.tex:299](paper-template-Latest/main.tex#L299)).

- The **only frozen encoder** in the system is **BioCLIP** — used by the D5 centroid filter and the D6
  probe ([main.tex:174](paper-template-Latest/main.tex#L174), `:434`, `:635`: "logistic probe on frozen
  BioCLIP"). A diagram can easily mislabel *which* encoder is frozen.
- **Correct depiction:** BioCLIP (filter encoder) = ❄️ frozen; ResNet-50 (classifier) = 🔥 fine-tuned
  end-to-end. Remove any frozen/lock/snowflake marker from the ResNet-50 box.
- The source `Diagram_A.jpg` is **not committed** in the repo — locate the editable source (Figma /
  draw.io / PPT / Illustrator) from the author, fix it, re-export, and place the `.jpg` next to
  `main.tex`. The prose/caption ([main.tex:158–163](paper-template-Latest/main.tex#L158-L163)) is
  already correct ("train a shared ResNet-50") — **do not change the caption**, only the image.

### T2 — Citations / `main.bib` (reviewers R6, R12)
- Fill citation gaps flagged in `REVIEW_LIST.md` (R6, R12). Add missing entries to `main.bib` and
  `\cite{}` them at the relevant prose anchors — **but only in prose regions you own** (§2). If the
  natural anchor is inside a DO-NOT-TOUCH region, log it in `WRITING_NOTES.md` instead.
- Bib hygiene: dedupe, fix malformed entries, consistent venue/year formatting. `main.bib` is a
  separate file — safe to work freely.

### T3 — Prose polish (bounded)
Polish only these, and only for clarity/grammar/flow — **no claims about protocol, CV, or metrics**:
- Related Work (`:~200–240`, `:~860–880`)
- Dataset / generation description (`:~320–370`, `:~419–441`) — but NOT `:480–481`.
- Qualitative figure captions (t-SNE `:542`, ROC `:656`, generation panels `:332`).
Keep edits surgical; note anything larger in `WRITING_NOTES.md`.

---

## 5. Cross-references to read

- `writing/TODO.md` — full plan. Your work = the results-independent slice of Phase 2 (W-items on
  figures/citations/prose). W1/W2 (table + baseline rows) are the main session's.
- `writing/REVIEW_LIST.md` — all 21 reviews + advisor's 4 majors (source for T2 citation gaps).
- `writing/BASELINES.md` / `writing/BASELINES_WALKTHROUGH.md` — baseline definitions (context only;
  the §5 baselines paragraph is the main session's to write).

## 6. Handback checklist (per chunk)
- [ ] Edited only files/regions you own (§2), touched nothing in §3.
- [ ] Committed on `writing-figures` with a clear message; did **not** merge to `main`.
- [ ] Logged any out-of-scope issues in `WRITING_NOTES.md`.
- [ ] Told the author the chunk is ready for a conflict-checked merge.
