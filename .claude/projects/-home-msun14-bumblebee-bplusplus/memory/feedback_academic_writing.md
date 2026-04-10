---
name: Academic writing standards
description: Guidelines for writing thesis/paper sections — abstract structure, conciseness, presenting negative results, avoiding hype
type: feedback
---

User wants academic writing at NeurIPS/MIT thesis standard. Previous abstract attempt was too dense with numbers and read like a results dump.

**Why:** The user is writing a graduate thesis (20-30 pages) that should meet top-tier CS publication standards.

**How to apply:**

## Abstract: 6-Sentence Template
1. Context/Problem (1 sentence, broad area + why it matters)
2. Specific gap (what's unsolved)
3. Approach — name the method, be concrete, active voice
4. Key insight — what makes it work / what distinguishes this from generic
5. Results — quantitative, with baselines, dataset names
6. Implication — grounded, no hype

## Abstract Anti-patterns to Avoid
- Literature review abstract (>1 sentence of context)
- Vague claims without numbers ("promising", "significant")
- Hype words ("novel", "groundbreaking", "for the first time")
- Method-only (no results) or results-only (no method)
- Burying the lede (contribution appears late)
- Too many numbers crammed together — pick the 2-3 most important

## General Writing Rules
- Every paragraph needs a topic sentence that can stand alone
- No weasel words: "fairly", "quite", "somewhat", "arguably"
- Active voice > passive voice
- Precise comparisons: "X achieves 5.2% higher F1 than Y on Z" not "X is better"
- "So what?" test on every sentence
- No "In this paper, we..." — it's obvious
- No undefined acronyms in abstract
- No citations in abstract (NeurIPS style)

## Presenting Negative/Mixed Results
- Frame as insight, not failure: "reveals that Y" not "fails at X"
- Characterize WHEN it works and WHEN it does not
- Use confidence intervals and statistical tests
- Be upfront in abstract if results are mixed

## Section Structure (ML paper)
- Intro: problem → why hard → what we do → contribution list → roadmap
- Related Work: by theme, not chronology. End with positioning paragraph.
- Method: system overview figure first, then top-down components
- Experiments: datasets, baselines, metrics justification, main table, ablation, qualitative examples, statistical tests
- Discussion: what was surprising, where it fails, what you'd change
- Conclusion: contribution (1 sent), key result (1 sent), limitations, specific future work
