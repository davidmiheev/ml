# NVIDIA Nemotron Model Reasoning Challenge

[Competition](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge) ·
Mar–Jun 2026 · metric: accuracy (`\boxed{}` answer, exact string or 1e-2 relative) ·
4,354 teams · $106k + DGX Sparks

## TL;DR

- **This was a trace-design competition, not a training competition.** Every top
  team did plain LoRA SFT on traces written by deterministic Python solvers. The
  ranking was decided by how much of each puzzle family the solver could cover,
  and how *learnable* its trace was within the 7,680-token budget.
- **The hard part was two categories: bit manipulation and cryptarithm.** The
  other five (cipher, numeral, unit conversion, gravity, numeric equations) were
  ~solved by the April Open Progress Prize. Gold went to whoever reverse-engineered
  the hidden generator of those two most completely.
- **The winning idea was a memorize-vs-compute split.** The adapter memorizes a
  precomputed catalog (cryptarithm "signature → candidate digit assignments",
  valid 8-bit rule sequences). The trace then only *verifies* candidates step by
  step. 1st and 3rd place arrived at this independently. 3rd made it explicit
  with a separate memorization training stage.
- **No RL, no distillation, no fancy losses.** Several teams report that focal
  loss, token reweighting, curricula, DPO, model soups and LoRA merges did not
  beat 1-epoch cross-entropy SFT.
- **The public LB (~250 problems) was nearly useless.** It had a wall at 0.86.
  Teams that trusted a decent local validation set over the public LB were rewarded
  in the private shake-up (3rd, 10th).

## The problem as the winners saw it

- **Setup.** Each prompt is an "Alice's Wonderland" puzzle: a few input→output
  examples of a hidden transformation plus one query. The 9,500 train problems
  fall into 7 families: bit manipulation (1,602), text cipher, numeral (Roman),
  unit conversion, gravity (`d = ½gt²` with an altered `g`), equation-numeric
  (two 2-digit numbers and a symbolic operator) and cryptarithm (the same, but
  digits are also hidden behind symbols).
- **Constraints that shaped everything:**
  - A LoRA rank ≤ 32 on Nemotron-3-Nano-30B-A3B (hybrid Mamba/attention MoE).
  - vLLM at `temperature=0`, `max_tokens=7680`.
  - No tool execution at inference. The model *is* the program, so all "compute"
    must be written out token by token.
- **Two facts made the deterministic trace the only sane design.** Greedy
  decoding means you only need the argmax token right at every step. Problems
  are generator-made, so every family has a finite rule grammar that can be
  recovered from `train.csv`.
- **Leaderboard noise.**
  - Teams estimated the test set at ~500 problems split 250/250, so one problem
    ≈ 0.004 on the LB.
  - vLLM at temperature 0 was still not bit-deterministic. The same adapter
    could move ±0.01 between submissions (reported by the 1st- and 5th-place
    teams).
- **Tokenizer quirks.** Nemotron tokenizes bits and digits roughly one character
  per token, so long binary columns are expensive. HEX compression of bit
  columns freed ~25–28% of the token budget (1st, 3rd and 6th place). Nemotron
  is also bad at spelling, splitting and concatenating symbols (OPP). Tokens
  like `}` inside `\boxed{}` broke the metric's extraction, and 2nd place
  switched to a `The final answer is:` format.

## Solutions

### Open Progress Prize: Tong Hui Kang ("SFT to maximize minimum logprob") · [post](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915) · [code @ bb199bc](https://github.com/tonghuikang/nemotron/tree/bb199bcdf7f38765dc7d7f37cf0e0019aec37284)

The baseline almost every gold team built on. Public LB ~0.85 in April.

- **Thesis.** "Nemotron can act as a simple computer after LoRA training." He
  bet *against* RL ("I already have the optimal policy") and *against*
  distilling larger models ("the optimal policy should be generated with code").
- **Trace-design principles:**
  - Deterministic: only the argmax matters.
  - Each token derived from as few sources as possible (division broken into
    subtraction and addition).
  - Coverage: keep iterating over all 77 cipher words even after a match, so
    rare tokens get trained.
  - Tokenization-aware.
  - Diversity and conciseness are explicit *non-goals*.
- **Diagnostic: minimum per-trace logprob.** Any trace with a token below
  ln 2 (0.69) is at risk of flipping under greedy decoding. He inspected exactly
  those tokens and rewrote the trace. *Code check:* the default loss in
  `train_sft.py` is plain `CrossEntropyLossConfig`. "Max-min-logprob" is the
  selection and debugging objective, not a custom loss. A weighted-CE variant
  that down-weights already-confident tokens exists in `loss_config.py` as an
  option.
- **Bit manipulation solver** ([post](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/690307)):
  - Find a per-output-bit expression (identity / NOT / constant /
    AND/OR/XOR of two input bits).
  - Then pick the longest consistent left and right "runs".
  - 85.1% solver ceiling.
- **Cryptarithm.** Only `concat` / `reverse concat` handled (7.9%).
- **Tinker gotcha.** Tinker LoRAs needed lossy conversion to the PEFT layout:
  - unfusing experts;
  - an SVD merge of `gate_proj`+`x_proj` → `in_proj`, which retains only ~75%
    of singular mass.

  This caused training-serving skew: missing `\boxed`, degenerate repeats.
  2nd place later measured the cost: switching Tinker → Megatron-Bridge alone
  moved bit-manipulation accuracy from 0.81 to 0.89.

### 1st: NullSira (GoodMeatDay, re, reopon), private 0.920 / public 0.91 · [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/1st-place-solution) · [code @ 1082d40](https://github.com/xrwr/kaggle-nvidia-nemotron-model-reasoning-challenge-1st-place/tree/1082d409b124040540007a9591f43b6693eb7ff9)

- **Pipeline.**
  - Generate synthetic problems programmatically from `train.csv` statistics.
  - Attach solver traces.
  - Train an SFT LoRA in Unsloth: r=32, α=32, lr 2e-4 cosine, `adamw_bnb_8bit`,
    eff. batch 16, 1 epoch over 220k samples / 890M tokens. Target modules
    include `lm_head` and Mamba `in_proj`/`out_proj`.
  - Continue training on a bit-heavy 140k pool at lr 1e-5.
  - About 170 GPU-hours on a single RTX PRO 6000.
- **Equation-numeric.** A 24-rule inventory (join, ±1 variants of
  add/sub/mul, `abs`, `max mod min`, and "flip" versions on digit-reversed
  operands) plus two corpus invariants:
  - *mode*: all non-join operators in a prompt share normal-or-flip;
  - *group*: no two operators share a family.

  Solver ceiling: 95% on deduce, 53% on guess.
- **Cryptarithm.**
  - The insight: enumerate all `00..99 × 00..99 × 22 rules` once and normalize
    each `left+right+output` digit string into a *signature*. That gives 4,205
    signatures, and each maps to a short list of candidate digit assignments.
  - The model memorizes this catalog through repetition in the SFT data.
  - The trace then:
    1. looks up candidate counts per equation;
    2. picks the anchor with the fewest candidates (a length-8 multiplication
       signature is preferred if within `max(30, 2×min)`);
    3. orders the remaining equations by unknown count;
    4. runs a printed DFS with mode/group pruning.

  This alone took them to 0.89 public / 0.908 private.
- **Bit manipulation.** Hui Kang's solver, plus three changes:
  1. Bit strings of length ≥4 written in HEX. Median generated tokens went
     6771 → 4888.
  2. Extra rule families: majority, 3-input parity, choice, `sel_nand_xnor`,
     and AND/OR/XOR compositions.
  3. *Repair.* Enumerate all valid 8-bit transformations into 5,238 per-bit
     rule sequences. Project the heuristic sequence onto the nearest catalog
     entries by Hamming distance and verify those against the examples.

  The repair idea came from an *accidental* training sample whose trace was the
  solver's but whose `\boxed{}` was the ground truth. The model learned to
  "fix" the final answer.
- **Checkpoint selection.**
  - Periodic Kaggle submissions served as a noisy external validation.
  - The final pick was the extra-training adapter, submitted twice.
  - Their best *unselected* submission scored 0.932 private.
- **Code notes (not in the writeup):**
  - Run lineage `100 → 102 → 103 → 104 → 106` (`configs/sft/experiment/`).
  - Run 104 is a re-run after fixing **gradient-accumulation loss
    normalization** in their linear-CE loss.
  - Run 105 (continuation at lr 5e-5) *regressed*, so 106 restarted from 104 at
    lr 1e-5.
  - 106 stopped at 6,500 steps (0.74 epoch). Run 107 tried to finish the
    remaining 19k samples at lr 2e-6.
  - The team states every line of code was written by Codex; the ideas were
    theirs.

### 2nd: vli, private 0.908 / public 0.884 · [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-place-solution) · [code @ 597dcaa](https://github.com/livctr/nvidia-nemotron/tree/597dcaaa9e7aebe81b61092260f9f1343c5a2fcc)

- **Synthetic only, train.csv as validation.** All 57.6k training examples
  (159M tokens) are synthetic, and the full `train.csv` is the validation set.
- **Bit manipulation: 99.06% solver, 97.3% emitted.**
  - Hypothesized generator: atoms `SHL_a`, `ROT_b`, `SHR_c` with shifts in
    1..7, combined into no-op / one-op / two-op / majority / choice shapes.
  - Because every shift is ≥1, the bit-0 and bit-7 "corner regions" are at most
    2-input functions.
  - The trace enumerates corner pairs, derives the middle-region operator from
    them (AND/AND → majority, ANDNOT/identity → choice), and verifies.
  - Full derivation in `nemotron/data/docs/bit_manipulation_bit_formulas.md`.
- **Text cipher.** Error-correction loop over the 77-word vocabulary and the 4
  sentence templates. It scores words by matched letters within ±1 length:
  98.5% → 99.5%.
- **Infrastructure finding.**
  - Megatron-Bridge maps LoRA to PEFT *exactly*, while Tinker needs the lossy
    SVD. That difference alone was worth +8 points on bits.
  - A late-found bug in the QKV split of the extractor (Megatron interleaves Q/K/V
    per group) barely moved the score: 0.884 → 0.880 public, 0.908 private
    unchanged.
- **Negative results.** Focal loss, token-loss reweighting and multi-stage
  schedules did not beat plain CE. Returns diminish after ~80M tokens.
- **Code note.** `configs/final.yaml` shows *failure-driven data generation*:
  - per-word query weights (`key: 10, book: 10, door: 8`) to upsample confusions
    seen in eval;
  - difficulty buckets for under-determined cipher letters;
  - a 1% forced "over-run" branch so a rare solver fallback gets training
    signal.

### 3rd: YS-L, private 0.900 · [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/3rd-place-solution) · [code @ e37af39](https://github.com/YS-L/nvidia-nemotron-reasoning-3rd-place-solution/tree/e37af3928a8f511e248dd94bb80fb9e109ab1531)

- **Numeric equation solver.** Rewritten from scratch with exactly 13
  operations and "simple" vs "reverse" structures applied globally. Ceiling:
  93.8% deduce, 39% guess.
  - A co-occurrence table showed operator *families*: `a*b` never appears with
    `a*b±1` in the same problem.
  - So the best guess for an unseen operator is a not-yet-used family.
- **Cryptarithm as a CSP.**
  - Multiplication-family signatures give 3,678 signatures and 11,334
    `(signature, structure, op)` triples (`data/artifacts/cryptarithm/pattern_lookup.json`).
  - Intersect candidate mappings from all multiplication examples to get
    "anchors".
  - Backtrack the rest with ≤2 unknowns per step.
  - Solves 24.2% within budget.
- **Two-stage training is the key idea here.**
  - *Stage 1* trains only on packed `<RECALL_DOMAINS>` / `<RECALL_ROWS>` drills,
    16 per sample with request tokens masked. That is 27.6k steps and 25 h,
    reaching 97% exact recall.
  - *Stage 2* trains on real traces starting from the stage-1 adapter, with 3k
    drills mixed in to prevent forgetting.
  - Ablation: direct 12/17 → two-stage 16/17 solved cryptarithms, private
    0.888 → 0.900.
- **Validation.** A 10% stratified holdout, with augmented children of held-out
  rows also excluded. It tracked *private* well and *public* badly: one of
  their 0.85-public submissions scored 0.90 private.
- **Training.** Effective batch 4 worked better than 16 (smaller batches
  under-trained less). `MoE tie weights = 0`.

### 4th: Dipam Chakraborty (/make-no-mistakes), private 0.900 · [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/4th-place-solution)

- **~$120 total compute.** Colab Pro G4 plus Kaggle weekly quota. 27.7k samples
  in the final mix.
- **Bit manipulation: 99.4% with only 5 templates** (Majority, Choice, and
  three 2-op shapes over SHL/SHR/ROT).
  - Two *anchor bit positions* collapse the search. At bit 7 every left shift
    is dead; at bit 0 every right shift is dead. So the shift amounts can be
    read off two quadrant tables, then every joint candidate is verified.
- **"Trainable trace" rules**, enforced by audit scripts:
  - *No hidden compute.*
  - *Locality*: every line computable from the ~10–20 lines above it.
  - *Causal consistency*: the trace shape can't depend on the future winner.
  - *Reference integrity.*
  - *Few uniform rules.*
- **Metric to watch: first-token divergence / greedy next-token match** on
  unseen traces. In their experience 80% token match ⇒ >95% generation match.
- **Workflow.** All code by Claude. A throwaway web UI diffed generations
  against gold traces, and the LLM was pointed at it to propose format fixes.

### 5th: Domdolus Tolus ("Jack of Trades, Loser of Cryptarithm"), private 0.896 · [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/domdolus-tolus-solution)

- **The team said it had near-zero ML experience.** It spent most of its time
  solving puzzles by hand.
- **Equation: "2 worlds × 4 families × 13 rules"** solve 100% when the answer
  is peeked.
  - Tie-break priorities plus a lookup string of operator orderings handled
    ambiguous deduce/guess cases: 96% solver.
- **Training.**
  - 15 sequential SFT iterations on Kaggle notebooks (12-hour limit), ~2B tokens
    total.
  - Then *LoRA averaging* (soup of the before and after iterations) and soups of
    3 models trained from the same checkpoint on different data.
  - They report the soups stabilized LB variance. *10th place reported the
    opposite*: seed soups and SVD merges degraded to the worst member.
- **Negative results.**
  - Pseudo-GRPO via iterated DPO did not learn.
  - Short CoTs failed even with 250k examples.
  - Unsloth 4-bit plus packing "broke" Nemotron.
  - DoRA isn't supported by vLLM.

### 6th (2nd public): Alehandreus & Yurnero, private 0.896 · [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/2nd-public-6th-private-place-solution)

- **Validation by LB probing per domain.**
  - Train the adapter to answer `\boxed{DUMMY}` for every category except the
    one under test.
  - The public LB then reports a *pure single-domain* score (each 0.01 ≈ 3
    problems).
- **Bit-template table.** Counts of `train.csv` problems per formula, e.g.
  `shl XOR shr` 368, `(shl XOR NOT rotr) OR shr` 303. All arithmetic done in
  HEX with a printed cheat-sheet.
- **Data-mix sensitivity.**
  - More data for any one domain didn't hurt the others, *except* bit
    manipulation, which degraded with any mix change. Final datasets were
    66–69% bit tokens.
  - They chose the safer 0.25B-token dataset over the 0.5B one with more
    cryptarithm. That cost them 2nd–3rd private.

### 7th: Prateek & team, private 0.888 · [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/7th-place-solution-solving-bitmanipulation-via-st) · [arXiv](https://arxiv.org/abs/2606.23672)

- **Bit manipulation as string matching**, without arithmetic:
  - find "bases" by minimal bit-flip similarity between outputs;
  - deduce truth tables;
  - backtrack on collisions;
  - force **single-bit tokenization**.
- Uses **dynamic loss masking** so the model learns to hypothesize,
  self-check and backtrack.
- They joined 10 days before the end and skipped cryptarithm entirely.

### 9th: fate (solo), private 0.884 · [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/9th-place-solution-human-auditable-cot)

**No synthetic expansion at all.** They hand-curated Hui Kang's traces into
consistent, human-auditable targets:

- bits 1364 → 1447 verified;
- equation 561 → 633;
- bit traces oversampled 5×;
- 1 epoch.

### 10th: Rick, neilus, eikichi, private 0.880 / public 0.860 · [writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/writeups/10th-place-solution)

- **Oracle ceiling 90.6% on train.** They tracked "oracle vs imitation" errors
  separately.
- **Training changes.** Re-added low-logprob problems as 2× priority samples
  (3× overfit). Raised `lora_alpha/r` to 4, which improved CV.
- **Negative results.**
  - Curriculum learning, two-stage targeted training, SVD merge and seed soups
    all failed.
  - Dropping low-confidence traces *hurt*.
- **Selection.** They chose the higher-validation run over the higher-public
  one (0.852 → 0.880 private).

## What actually mattered

Ranked by the evidence in the writeups:

| Lever | Evidence | Who |
| --- | --- | --- |
| Recovering the generator grammar for **bit manipulation** (3-atom templates, nonzero shifts, corner/anchor bits) | Solver ceilings went from 85% (OPP) to 97–99.4%. That is the single biggest gain across teams | 1st, 2nd, 3rd, 4th, 6th, 10th |
| **Memorize a precomputed catalog, verify in the trace** (cryptarithm signatures, valid bit-rule sequences) | 1st: catalog alone → 0.908 private. 3rd: two-stage memorize-then-solve lifted 0.888 → 0.900 | 1st, 3rd |
| **Token budget engineering** (HEX bit columns, unique delimiters, early-exit verification) | −28% median tokens (1st) enabled whole new rule families within 7,680 tokens | 1st, 2nd, 3rd, 6th |
| **Learnable trace format** (locality, no hidden compute, causal consistency) | 4th: 82% → >95% bit accuracy at equal training data. 9th: gold with no synthetic expansion | 4th, 9th, OPP |
| **Lossless adapter export** | Tinker→Megatron-Bridge alone: bits 0.81 → 0.89 | 2nd, OPP |
| **Validation independent of the public LB** | 3rd, 10th and 5th all report the public/private gap. 3rd's 0.85-public submission scored 0.90 private | 3rd, 10th, 6th (per-domain probes) |
| Training tricks (focal loss, reweighting, curricula, DPO, soups, merges) | Mostly neutral or negative. Soups were mixed (helped 5th, hurt 10th) | 2nd, 5th, 10th |

Cryptarithm decided the final top 3. Even the winners solved only ~40% of
*deduce* problems and ~12% of *guess* problems.

## Transferable lessons

1. **When a small model must "run an algorithm" under greedy decoding, design
   the trace like a program for a weak CPU.**
   - Every token should be locally derivable.
   - Long-range state should be copied, not recomputed.
   - Branches should be determined only by what is already written.
   - Measure min-logprob or first-divergence, not mean loss.
2. **Split memory from compute.** Precompute lookup tables offline, train the
   model to recall them (a separate drill stage helps), and let the trace do
   only verification. This turns an intractable search into a short one.
3. **Recover the data generator before modeling.** Co-occurrence tables, "peek
   the answer" solvability checks and template counts are cheap and gave the
   largest gains here.
4. **Check the export path.** Any lossy conversion (SVD, fused→unfused,
   quantization) between training and serving can cost more than all
   hyperparameter tuning.
5. **Use small public test sets as a sanity check only.** Build a validation set
   that excludes augmented children of held-out rows, and select by it.

## Open questions

- 1st place's GA loss-normalization fix (run 104) isn't discussed in the
  writeup. How much of the final gap did it account for?
- LoRA soups helped 5th place but not 10th. Was the difference same-checkpoint
  forks (5th) vs independent seeds (10th)?
- Could RL on the memorized-catalog traces close the cryptarithm "guess" gap
  (~12%), or is it information-theoretically under-determined from the prompt?
