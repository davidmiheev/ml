# Смысловой переключатель — episodic few-shot rule induction

Solution for the Yandex contest task "Смысловой переключатель / Поймай правило за хвост"
(corpus `semantic-switch-v2`).  Each episode hides a binary rule: eight labelled
support messages (4 positive / 4 negative), four queries to label.  The metric is
**Episode Exact Match** — an episode scores only if all four queries are right.

Full task statement: [`task.txt`](task.txt).

## What is here

| path | what |
| --- | --- |
| `starter/solution/` | the submission — the only files the contest accepts |
| `dev/` | local tooling: synthetic corpus generator, scorer, validation harness, degenerate-input tests |
| `examples/` | the organiser's tiny schema dummies |

The organiser's `starter/runtime/`, `starter/assets/` (frozen `rubert-tiny2`),
`starter/tests/` and `package_submission.py` are **not** redistributed here.  To run
anything, unpack the official starter archive and drop `starter/solution/` from this
repository over theirs.

## Constraints the solution works under

CPU only, 2 cores, 4 GB, no network, train + predict together ≤ 25 min, backbone frozen
at its pinned SHA-256, ≤ 150 000 tensor scalars and ≤ 2 MiB in the adapter, ≤ 58 input
tokens, and prediction may not train, use gradients, or read the training data.

## Approach

The frozen `rubert-tiny2` is used purely as a sentence encoder; every distinct message is
encoded **once** (dedup + length-sorted dynamic padding — ~10× cheaper than padding all
48 000 training texts to 58 tokens).  Everything learned lives in a small adapter:

1. **Episode-local geometry.** Features are standardised with global train statistics,
   then centred on the mean of the episode's own twelve messages and L2-normalised.
   Removing the episode's common topic direction is worth ~10 EM points on its own.
2. **Nearest centroid.** The rule direction is `unit(c⁺) − unit(c⁻)` over the support
   set; a query is scored by its cosine along that direction.  Cosine beat squared
   Euclidean by ~3 EM points.
3. **Rule codebook + self-calibrating gate.** Rule directions mined from the training
   episodes (all twelve labels are known there, so they are far cleaner than an
   eight-shot estimate) are k-means clustered into a codebook.  At prediction time the
   episode's own direction attends over the codebook against a learnable *null* slot: a
   recognised rule borrows the mined direction, an unrecognised one keeps its own.
   This is the same idea as the official soft-prompt pool, moved into the metric space.
4. **Joint decoding for exact match.** Four independent decisions do not maximise exact
   match.  The distribution of the number of positive queries per episode is estimated
   on train (Laplace-smoothed) and the four labels are decoded by joint MAP, with the
   corpus base rate divided out.  Reported probabilities are the joint marginals,
   clamped so `label == 1` iff `probability >= 0.5`.
5. **Data-driven model selection.** Candidates (no codebook / coarse / fine, and the
   blend ceiling) are scored over three independent 3-way splits on episodes no fitting
   stage has seen.  Each candidate is scored in **both** regimes — rules that recur in
   the corpus, and rules never mined (simulated by hiding the best codebook match *and
   its whole cosine neighbourhood*, since one rule spreads over many entries).  The rank
   is the mean, so retrieval must earn its place under either assumption, and a richer
   candidate must clear a margin before displacing a simpler one.  The winner is refit
   with the selection split folded back in.

### What was tried and rejected

- **Learned projections** (full 312→192, low-rank, diagonal).  They lift the score on
  episodes whose rule recurs in training and *lose* more than they gain the moment a
  rule is new — on the held-out-rule proxy a full projection scored 0.41 EM against
  0.60 for the untouched frozen space.  Unlike retrieval, that failure cannot be
  simulated at training time, so it cannot be guarded against; the frozen space is left
  undistorted.
- **ZCA whitening** of the embedding space: neutral at α=0.25, sharply harmful beyond.
- **Extra encoder views** (mean-pool, max-pool, layer-1 mean): all worse than CLS alone.
- **Cluster-grouped validation** as a proxy for unseen rules: rule-direction clusters
  leak too much (0.77 purity) to rank candidates honestly.

## Results on the local proxy

The organiser ships only one-episode dummies, so `dev/gen_synth.py` generates a proxy
corpus — templated Russian support messages with latent attributes, one hidden rule per
episode, hard negatives sharing the confounding attributes.  Numbers are Episode Exact
Match / per-query accuracy on 1000 episodes.

| regime | starter baseline | this solution |
| --- | --- | --- |
| rules held out from training (adversarial) | 2.1 / 35.9 | **51.4 / 74.1** |
| rules recurring in training | 9.0 / 55.4 | **71.3 / 84.6** |
| held-out rules, unbalanced query counts | — | **26.7 / 69.3** |

The proxy is a stand-in, not a prediction of the hidden corpus — it is deliberately
harsher about rule generalisation than the real corpus is likely to be.

Budget on the worst case (all 48 000 training texts distinct, 2 threads):
**train 77 s, predict 14 s** against a 1500 s limit; adapter 20 294 scalars / 82 008
bytes against limits of 150 000 / 2 097 152.

## Running it

```bash
# from the unpacked starter, with this repo's starter/solution/ copied in
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -m starter.solution.train \
  --train train.jsonl --adapter runs/adapter.safetensors --seed 1729
TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 python -m starter.solution.predict \
  --data test.jsonl --adapter runs/adapter.safetensors --output runs/predictions.jsonl
python -m pytest -q starter/tests/test_contract.py
python package_submission.py --source starter/solution --output submission.zip
```

The CLI is unchanged from the starter.  `--epochs`, `--batch-size` and `--learning-rate`
act as lower bounds on the internal schedule, so the run is sane whatever the organiser
passes.  Two additions: `--time-budget` (seconds for the training phase, default 840 —
candidates are dropped rather than overrunning) and `--report-eval` (a labelled JSONL
scored for every candidate, diagnostics only).

Local validation, from a directory holding `contest/` (the unpacked starter) and `dev/`:

```bash
python dev/gen_synth.py --out dev/data                      # proxy corpus
python dev/gen_synth.py --out dev/data_free --query-balance free
python dev/split_eval.py dev/data/train.jsonl dev/data/heldin_train.jsonl dev/data/heldin_eval.jsonl 1000
dev/validate.sh                                             # all regimes + contract + packaging + edge cases
```
