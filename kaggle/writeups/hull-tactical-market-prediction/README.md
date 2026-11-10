# Hull Tactical: Market Prediction

[Competition](https://www.kaggle.com/competitions/hull-tactical-market-prediction) ·
training phase Sep–Dec 2025, live forecasting phase to Jun 2026 ·
metric: volatility- and underperformance-penalized Sharpe ·
3,677 teams · $100k

## TL;DR

- **This competition is mostly a lesson in what *not* to trust.**
  - During the training phase the public LB scored the last 180 rows of
    `train.csv`, so labels were public by design. LB scores above 3 (Sharpe!)
    were pure overfit.
  - The real ranking came from ~6 months of live S&P 500 data after submissions
    closed.
- **The final top Sharpe-variant scores are ~3.8 (1st lunalu 3.845, 2nd
  FastScramblers 3.842).** They come from a single, roughly half-year live
  window. The forum raised early on how likely a random submission was to win
  on a sample this short.
- **Top solutions are largely undocumented.** As of this analysis, the 4th place
  is the only top-10 writeup. A forum thread ("Where are the top solutions?")
  asks the same. **None of the top writeups link a code repository.** The two
  writeups worth studying are 4th (no ML) and 61st (one LightGBM + binary
  allocation).
- **The documented winners won on portfolio construction:** sparse signal,
  inverse-volatility blending, volatility targeting, and conservative
  allocation. They did not win on forecasting accuracy.

## The problem as the winners saw it

- **Data.** Daily rows with anonymized feature groups: `M*` market, `E*` macro,
  `I*` rates, `P*` valuation, `V*` volatility, `S*` sentiment, `MOM*`
  momentum, `D*` dummies. History goes back decades, with heavy early
  missingness.
  - Targets are `forward_returns` and `market_forward_excess_returns` (minus the
    rolling 5-year mean, MAD-winsorized).
  - The test API serves *lagged* returns, so a model can update online.
- **Output.** A daily S&P 500 allocation in [0, 2]. Leverage up to 2× is
  allowed.
- **Metric.** A Sharpe ratio variant that penalizes realized volatility above
  1.2× the market's and penalizes underperforming the market's return. The
  exact code is in the
  [metric notebook](https://www.kaggle.com/code/metric/hull-competition-sharpe);
  this analysis relies on the evaluation page's description. Consequences:
  - leverage is free until the volatility cap, then quickly expensive;
  - sitting in cash is safe but loses to the market term;
  - there is **no transaction-cost term**, a point raised early in the forum.
    Daily-flipping short-horizon reversal strategies are therefore much more
    attractive here than in real trading.
- **Validation is the whole game.** The host's sticky notice said the training
  phase test rows were part of train "by design" and recommended walk-forward,
  non-overlapping out-of-sample evaluation. AmbrosM's top-voted post ("Stop
  wasting your time here!", 139 votes) showed:
  - high-LB public notebooks had *negative* cross-validated R²;
  - only the host's starter notebook had positive R² and a realistic Sharpe.

## Solutions

### 4th: YannFb (solo), "Technical Model, No Learning: Short Term Reversal" · [writeup](https://www.kaggle.com/competitions/hull-tactical-market-prediction/writeups/4th-place-technical-model-no-learning-short-te)

- **No ML model.** The pipeline is a rule-based **short-horizon mean-reversion**
  alpha plus portfolio engineering. The author researched the alpha
  *before* the competition and deliberately did not tune it on competition
  data. They ran "only a handful of complete experiments".
- **Inverse-volatility blending.**
  - The main alpha is sparse (often inactive), so its realized volatility is
    unstable.
  - A few weak auxiliary signals are added as *stabilizers*, not alpha.
    Weights are `w_i ∝ 1/σ_i` (rolling volatility).
  - No covariance estimation and no optimizer. They argue covariance-estimation
    error would swamp mean-variance gains with so few signals.
- **Volatility-targeting overlay.**
  - `L_t = σ_target / σ̂_portfolio,t`, then `clip(L_t · w_t, 0, 2)`.
  - The volatility estimate uses a long window and is updated only
    periodically, to avoid chasing noise.
  - In the author's words, this overlay "contributed more to the final
    leaderboard score than almost any modification to the alpha itself."
- **Backtest they report:**

  | Configuration | Ann. return | Ann. vol | Sharpe | Max DD |
  | --- | ---: | ---: | ---: | ---: |
  | Signal only | 5.2% | 7.7% | 0.68 | −13.6% |
  | Signal + vol targeting | 9.4% | 14.3% | 0.66 | −25.8% |
  | Buy & hold | 6.3% | 19.2% | 0.33 | −63.8% |

  Vol targeting barely changes Sharpe, but it scales the return up toward the
  metric's market-return hurdle while staying under the 1.2× volatility cap.
  That is exactly what this metric pays for.
- **Known weakness.** Mean reversion underperforms in persistent low-volatility
  bull trends. The author's regret is not ensembling slight variants of the
  signal.

### 61st: Nikitas Rafael Nikitas (silver), "Single Model, Simple Decision" · [writeup](https://www.kaggle.com/competitions/hull-tactical-market-prediction/writeups/single-model-simple-decision-61st-place-silver) · [notebook](https://kaggle.com/code/rafanikitas/hull-eda-training-pipeline)

- **Features.** 14 hand-picked features × (lags 1–20, rolling mean/std over
  2–60) = 224 temporal features.
- **Model.** One LightGBM. Hyperparameters were chosen by Optuna on mean
  **Spearman** across 4 time-series folds, i.e. ranking rather than magnitude.
  CatBoost/XGBoost ensembles scored *lower*.
- **Binary allocation {0, 1}** instead of the allowed [0, 2]. This acts as
  regularization: it ignores small prediction differences and needs no leverage
  calibration. Dynamic 0–2 allocation scored lower.

### Forum material that mattered more than most notebooks

- **[Important Notice on Training Data Usage](https://www.kaggle.com/competitions/hull-tactical-market-prediction/discussion/608087)**
  (host). Train/test overlap was intentional. Use walk-forward validation.
- **[Stop wasting your time here!](https://www.kaggle.com/competitions/hull-tactical-market-prediction/discussion/608088)**
  (AmbrosM). Add CV R² to every notebook and distrust LB > 3. In the comments,
  Robert Hatch notes the metric has no transaction costs, so daily reversal
  edges that die in real trading can score here.
- **[The possibility of a random submission winning](https://www.kaggle.com/competitions/hull-tactical-market-prediction/discussion/608135)**
  and **[The REAL "do-nothing" baseline is 0.469](https://www.kaggle.com/competitions/hull-tactical-market-prediction/discussion/611071)**:
  useful context on how noisy a single live window is, and where the
  do-nothing baseline sits. I read only the titles of these two threads, not
  their contents.
- **[Micro Alphas (paper by Hull Tactical)](https://www.kaggle.com/competitions/hull-tactical-market-prediction/discussion/614618)**:
  the host's own view of how weak, many-signal alphas are combined.

## What actually mattered

These conclusions come from the two writeups and the forum consensus. Without
top-3 writeups, they can't be verified against the winners.

| Lever | Evidence |
| --- | --- |
| **Not fitting on the leaked public window** | Host notice. AmbrosM's CV-R² analysis. High-LB public notebooks were overfit by construction |
| **Volatility targeting / exposure control** | 4th place: the overlay contributed more than any alpha change, scaling returns toward the market hurdle under the 1.2× vol cap |
| **Conservative, low-degree-of-freedom allocation** | 4th (inverse-vol, no optimizer), 61st (binary 0/1, single model). Both report more complex variants did worse |
| **Prior economic rationale for the signal** | 4th's alpha predates the competition and was intentionally untuned |
| **Short-horizon reversal + no transaction costs** | 4th's signal family. The forum noted the metric does not charge for daily turnover |
| Bigger ensembles, richer models | 61st: GBDT ensembles worse than a single LGBM. Several writeups say complexity amplified live errors |

## Transferable lessons

1. **In finance competitions, read the data page for leaks *by design*.** If the
   public LB scores rows you trained on, ignore it entirely and build a
   walk-forward, purged validation from day one.
2. **Optimize the metric's geometry, not R².** With a volatility-capped,
   market-relative Sharpe, the allocation map (vol targeting, clipping,
   inverse-vol weights) is a bigger lever than forecast accuracy.
3. **Prefer few degrees of freedom when the evaluation window is short.** Binary
   or inverse-vol allocations and untuned, economically motivated signals
   generalized best among the documented solutions.
4. **Treat a single live window as a lottery with skill-weighted tickets.** Pick
   final submissions for robustness across regimes (bull vs volatile), not for
   backtest peaks.

## Open questions

- What did 1st–3rd (lunalu, FastScramblers, wynzaro) actually do? This needs
  revisiting if their writeups appear. Was it more leverage under the volatility
  cap, or a genuinely different signal?
- How much of the top-10 spread is explained by the realized market regime
  (Dec 2025–Jun 2026) rather than method? A post-hoc regime analysis on Hull's
  [published daily data](https://www.kaggle.com/competitions/hull-tactical-market-prediction/discussion/737498)
  could answer this.
