# Fork in the Road
*At the heart of every machine learning algorithm is a binary choice. One fork in the road can make the difference between a successful model and noise...*

![GitHub Actions](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)
![Python](https://img.shields.io/badge/python-%233670A0.svg?style=for-the-badge&logo=python&logoColor=ffdd54)

Welcome to Fork in the Road!
This software suite is a config-driven machine learning pipeline that scans US equities for technical breakout setups, predicts whether each one will hit a profit target before a stop loss, and simulates the resulting P&L.

Built as a personal project to work through a full applied-ML lifecycle end to end: data acquisition and caching, feature engineering, labeling, training, calibration, validation, position sizing, backtesting, and hyperparameter search — with the statistical failure modes that make financial ML particularly easy to get wrong treated as first-class design constraints rather than afterthoughts.

Development and documentation assisted by Claude.

**Python 3.12+ · ~6,100 lines of implementation · ~3,900 lines of tests · 328 tests**

> **Not investment advice, and not a profitable trading system.** The results below are a modest measured edge on a limited backtest with known, documented biases. This is a software and methodology exercise. See [Results](#results) and [Known limitations](#known-limitations) for an honest accounting.

---


## Table of contents

- [Why this problem](#why-this-problem)
- [Architecture](#architecture)
- [The pipeline, stage by stage](#the-pipeline-stage-by-stage)
- [Results](#results)
- [Known limitations](#known-limitations)
- [Getting started](#getting-started)
- [Command reference](#command-reference)
- [Testing](#testing)
- [Where this goes next](#where-this-goes-next)

---


## Why this problem

What greater problem to solve than the most widely-studied one in modern society? Many say that the stock market is as predictable as Ohio's spring weather. Others say that they have the winning algorithm that rakes in returns when used properly. In the end of the day, the stock market is a collection of inumerable inputs that result in price action. Perfection is a scaling issue of data, one that Fork in the Road does not solve. Instead, this framework aims to perfect the statistics such that the little data that is available is best modeled.

Predicting stock movement is a deliberately unforgiving domain to build in. It has properties that punish sloppy engineering in ways that stay invisible until they matter:

- **Lookahead bias is silent.** A single feature computed with data the model wouldn't have had at prediction time produces spectacular backtest results and zero real-world value. Nothing crashes.
- **Labels overlap in time.** A 15-day forward-looking label means adjacent training rows share outcome windows, so a naive random train/test split leaks the future into the past.
- **Classes are imbalanced and the base rate is low.** Roughly 19% of scanned candidates succeeded in this dataset. A model that predicts "failure" every time scores 81% accuracy and is worthless.
- **Accuracy is the wrong metric anyway.** What matters is whether expected value per trade clears the payout structure's breakeven point — which moves every time the target or stop changes.

Every one of those is a correctness problem before it is a modeling problem, which made this a good vehicle for practicing defensive design.

---


## Architecture

The system is a linear pipeline of independently testable stages. Each stage reads a validated config, does one thing, and hands a plain `DataFrame` or dataclass to the next.

```
                          config/*.yaml
                               │
                    ┌──────────┴──────────┐
                    │  Pydantic schemas   │  fail fast, at load time
                    └──────────┬──────────┘
                               │
  ┌────────────┐      ┌────────▼────────┐      ┌──────────────┐
  │  yfinance  │─────▶│    YFProxy      │◀────▶│  PriceCache  │
  └────────────┘      │  (sole I/O      │      │  (parquet +  │
                      │   boundary)     │      │   coverage)  │
                      └────────┬────────┘      └──────────────┘
                               │
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
 ┌─────────────┐       ┌──────────────┐       ┌──────────────┐
 │  Scanner    │       │ FeatureEngine│       │   Labeler    │
 │ which rows? │       │  what are    │       │ what happened│
 │             │       │  they like?  │       │  next?       │
 └──────┬──────┘       └──────┬───────┘       └──────┬───────┘
        │                     │                      │
        └─────────────────────┼──────────────────────┘
                              ▼
                      ┌───────────────┐      ┌────────────────┐
                      │ DatasetBuilder│◀────▶│ CandidateCache │
                      └───────┬───────┘      └────────────────┘
                              ▼
                    ┌─────────────────┐
                    │  ModelTrainer   │  XGBoost + probability calibration
                    └────────┬────────┘
                             │
     ┌───────────────┬───────┴────────┬─────────────────┐
     ▼               ▼                ▼                 ▼
┌──────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐
│Evaluator │  │WalkForward  │  │PositionSizer│  │LivePredictor │
│          │  │ Validator   │  │  → Backtest │  │ (today's     │
│          │  │             │  │   Simulator │  │  candidates) │
└──────────┘  └──────┬──────┘  └──────┬──────┘  └──────────────┘
                     │                │
                     └────────┬───────┘
                              ▼
                   ┌────────────────────┐
                   │ Optuna tuning      │  searches labeling + model +
                   │ (scoring/objective)│  feature selection jointly
                   └────────────────────┘
```


### Module map

| Package | Responsibility |
|---|---|
| `fitr.config_schemas` | Pydantic models for all seven config files. Cross-field validation (a scan condition can't reference an undefined metric; a feature's params must match its function signature). |
| `fitr.data` | `YFProxy` — the only module that touches yfinance. Retry/backoff, incremental cache extension, trading calendar derived from real price-index data rather than assumed. |
| `fitr.features` | 28 pure indicator functions plus a config-driven `FeatureEngine` that dispatches to them. All indicators return price-normalized ratios, so a $5 stock and a $500 stock are comparable. |
| `fitr.scanning` | `BreakoutScanner` — evaluates configurable setup definitions (AND/OR condition trees) to find candidates. Includes an `explain()` path showing per-condition pass/fail. |
| `fitr.labeling` | `OutcomeLabeler` — walks forward from each candidate to determine SUCCESS / FAILURE / INCONCLUSIVE, with a conservative tie-break when daily OHLC can't disambiguate intraday ordering. |
| `fitr.dataset` | `DatasetBuilder` (assembly + cooldown) and `CandidateCache` (fingerprint-keyed scan reuse). |
| `fitr.modeling` | Training, calibration, evaluation, walk-forward validation, position sizing, backtesting, live prediction. |
| `fitr.tuning` | Optuna objective plus the shared scoring function that defines "better". |

---

## The pipeline, stage by stage

### 1. Data access — `YFProxy`

Every network call in the project goes through one class. Nothing else imports `yfinance`, which means the entire pipeline can be tested offline against fake price paths, and retry/rate-limit logic lives in exactly one place.

The cache does **incremental** extension rather than re-download. This mattered enormously in practice: a scanner walking forward one trading day at a time requests a trailing window whose end date advances daily, so a naive "not covered → refetch everything" cache re-downloads a ~500-day window per ticker per day and provides nearly zero benefit for the exact workload it exists to serve.

Cache coverage is tracked by **requested date range** in a JSON sidecar, not inferred from the returned rows. The distinction is subtle and was a real bug: request `2023-01-01` and the earliest row returned is `2023-01-03`, because Jan 1 was a Sunday and Jan 2 a holiday. Inferring coverage from the data alone concludes "the head is still missing" and re-issues the same empty fetch on every subsequent call, forever — once per ticker per call, across a 1,760-ticker universe.

The trading calendar is derived from a reference ticker's own price index rather than a hardcoded holiday list, so it is correct by construction for whatever period the data actually covers.

### 2. Feature engineering — `indicators.py` + `FeatureEngine`

28 pure functions (`rsi`, `macd_histogram_slope`, `relative_volume`, `atr_normalized`, `close_location_value`, `relative_return_vs_benchmark`, …), each taking a DataFrame and returning a single float. No hidden state, no I/O, individually unit-tested against hand-computed values.

`FeatureEngine` reads `features.yaml` and dispatches. Adding a feature is a config edit:

```yaml
- name: rsi_14
  function: rsi
  params: {window: 14}

- name: return_20d_vs_market
  function: relative_return_vs_benchmark
  params: {window: 20}
  benchmark: market          # resolves to SPY, defined once at the top
```

The schema validates at load time that the named function exists, that its params match the real signature, and that any referenced benchmark is defined — so a typo fails immediately with a clear message instead of surfacing as a `NaN` column 40 minutes into a dataset build.

**Point-in-time discipline** is enforced structurally: every feature computation receives an `as_of_date`, and `YFProxy.get_history` truncates to it. A feature *cannot* see future data, because the data isn't in the frame it's handed.

### 3. Scanning — `BreakoutScanner`

Setups are declarative condition trees:

```yaml
- name: volatility_squeeze_breakout
  logic: AND
  conditions:
    - {metric: bb_width, operator: "<", value: 0.08}
    - {metric: rel_volume, operator: ">", value: 1.5}
    - {metric: dist_from_20d_high, operator: ">", value: -0.02}
    - {metric: close_loc, operator: ">", value: 0.7}
```

Five setups ship in the example config. `scanner.explain(ticker, date)` returns a per-condition breakdown, which `tools/debug_scanner.py` surfaces for manual verification against a charting platform.

### 4. Labeling — `OutcomeLabeler`

Walks forward over the configured horizon and returns the first threshold touched. Three decisions worth noting:

- **Ambiguous days resolve pessimistically.** If a day's high crosses the target *and* its low crosses the stop, daily OHLC can't say which came first — so the stop is assumed. Assuming the favorable outcome would systematically inflate the measured success rate.
- **Too-recent candidates return `INCONCLUSIVE`**, not `FAILURE`, and are excluded from training rather than silently mislabeled.
- **Rich outcome metadata is retained** — `trigger_reason`, `forward_return`, `trading_days_held` — which is what makes real P&L simulation possible later, instead of assuming every failure loses exactly the full stop.

That last point produced a genuinely useful diagnostic. A candidate that drifts up 3% without ever reaching a 12% target is labeled `FAILURE` (`horizon_expired_flat`) but carries a *positive* `forward_return`. That is precisely why backtest win rate (any positive return) and success precision (full target hit) legitimately diverge — two different questions, not a bug in either.

### 5. Dataset assembly — `DatasetBuilder` + `CandidateCache`

Applies a per-ticker cooldown (default 5 trading days) so a single underlying move doesn't produce five near-duplicate, correlated training rows.

The expensive part of a rebuild — scanning every trading day across the universe — is cached under a SHA-256 fingerprint of `(scanning config content, universe, date range)`. Change only `features.yaml` or `labeling.yaml` and the scan is reused entirely:

**Measured: 53.35s → 0.05s on a rebuild after a labeling-only change.**

Fingerprinting the config *content* rather than the file path means two equivalent configs loaded from different paths correctly share a cache entry, and any semantic change correctly invalidates it.

### 6. Training and calibration — `ModelTrainer`

XGBoost gradient-boosted trees. Class imbalance is handled via `scale_pos_weight` rather than resampling — fabricating synthetic market rows has no defensible meaning, whereas reweighting the loss uses the real data as-is.

Label encoding is explicit (`SUCCESS → 1`, `FAILURE → 0`) rather than delegated to a library's internal encoder, because "class 1 is the positive class" is relied on downstream by the evaluator, predictor, and backtester. That should be a fact about the code, not an accident of alphabetical sort order.

**Probability calibration** (Platt/isotonic via `CalibratedClassifierCV`) is what makes probability *values* meaningful rather than just their ranking. The raw model was severely overconfident:

| Bucket | Mean predicted | Actual rate | Gap |
|---|---|---|---|
| 50–60% | 55.0% | 25.1% | **+29.8** |
| 60–70% | 64.6% | 28.8% | **+35.8** |
| 70–80% | 73.6% | 32.3% | **+41.4** |
| 80–90% | 82.3% | 26.7% | **+55.7** |

After sigmoid calibration, gaps compressed to single digits across most buckets. Calibration is a strictly monotonic transform, so it never changes ranking — but it does change how many candidates clear a *fixed* threshold, which is exactly what makes Kelly-based sizing trustworthy.

Calibration is fit on its own held-out chronological tail slice, carved *before* the base model trains, so it never sees data the base model was fit on:

```
[  fit  ][ early-stopping validation ][ calibration ]  │  test.csv (untouched)
```

### 7. Validation — `WalkForwardValidator`

A single train/test split answers "did this work once?" Walk-forward answers "does this work repeatedly?" The date range is divided into sequential folds; each trains on everything prior and tests on the next window.

This immediately surfaced something a single split would have hidden: one fold produced **zero** predictions above threshold. Its test window opened at the trough of a sharp drawdown for which the training data contained no analogue — and a properly calibrated model *should* lose confidence when extrapolating into unfamiliar conditions. That's correct behavior, and it is only visible because the per-fold report exposes the calibration curve and per-setup breakdown rather than just an aggregate mean.

### 8. Position sizing — `PositionSizer`

Two methods: fixed-fractional and fractional Kelly.

The Kelly formula here is **not** the textbook `f* = (bp − q)/b`. That derivation assumes a loss costs the entire wager, which is not this system's payoff structure — a stopped-out trade loses `stop_loss_return` of the position, not all of it. Re-deriving by maximizing expected log wealth for proportional gain/loss:

```
maximize  p·ln(1 + f·T) + q·ln(1 − f·S)

  d/df = pT/(1+fT) − qS/(1−fS) = 0
     ⟹ f* = (pT − qS)/(S·T) = p/S − q/T
```

Both formulas agree on where the fraction crosses zero — which is why a breakeven test passed either way and didn't catch the error — but disagree on magnitude everywhere else. The corrected version reproduces the project's breakeven-precision formula exactly at `f* = 0`: `p* = S/(T+S)`.

A practical consequence worth documenting: with a tight stop, raw Kelly is enormous (625% of equity at `p=0.5` with an 8%/4% payout), so `max_position_fraction` is not a nicety — it is load-bearing.

### 9. Backtesting — `BacktestSimulator`

Walks the test set chronologically, sizes each trade against current equity, applies the **real** `forward_return`, and compounds. Reports total return, win rate, average win/loss, max drawdown, average holding period, and a Sharpe-*like* ratio — named that way deliberately, because it is neither annualized nor computed over fixed calendar intervals, so it is valid for comparing this system's own configs against each other but not against externally quoted Sharpe ratios.

### 10. Hyperparameter search — `fitr.tuning`

Optuna TPE search over labeling parameters, model hyperparameters, and feature-group selection **jointly**, so interactions between them (a longer holding horizon plausibly favors longer-lookback features) are searchable rather than invisible. Scanning criteria are excluded because varying them forces a genuine rescan — a fundamentally different cost tier.

Two guardrails matter here:

**Trials never touch `test.csv`.** Each trial scores via walk-forward over the tuning-visible portion only. Searching hundreds of configs against a fixed holdout is p-hacking with extra steps; the winner would be whichever config best fits that holdout's noise.

**Scoring is expected value per trade, not precision.** This was a real bug found by inspecting tuning output. Because target and stop are *themselves* being searched, raw precision isn't comparable across trials — shrinking the target and widening the stop makes SUCCESS trivially easier to hit and inflates precision without improving economics. The tuner exploited exactly that, converging on `target=0.04 / stop=0.10`, which needs **71.4% precision just to break even** and is net-negative even at 70% precision. Expected value weighs each outcome by its own payout size and cannot be gamed the same way:

```
EV = p·T − (1−p)·S
```

Computed directly rather than by running a backtest, deliberately: a backtest-based score requires a fixed sizing scheme, and Kelly's fraction scales with `1/S`, which would just relocate the same confound into the sizing math.

---

## Results

Measured on ~21,600 candidates from 2023-01 through 2025-12, US equities, with an **18.8% base success rate**.

Best trial from the tuning run:

```json
{
  "score": 0.0098,
  "params": {
    "horizon_trading_days": 16,
    "target_return": 0.122,
    "stop_loss_magnitude": 0.0233,
    "cooldown_trading_days": 1,
    "n_estimators": 478,
    "max_depth": 7,
    "learning_rate": 0.1085,
    "include_macd_family": true,
    "include_market_context_family": true,
    "include_rsi_family": false,
    "include_volume_family": false,
    "include_volatility_family": false
  }
}
```

Applying the tuned labeling parameters (12.2% target / −2.3% stop → **15.8% breakeven precision**) and backtesting the held-out period:

| Threshold | Trades | Win rate | Total return | Walk-forward mean precision |
|---|---|---|---|---|
| 0.158 | 231 | 21.6% | **+17.0%** | 16% |
| 0.175 | ~100 | 24.3% | **+15.6%** | 17% |

Both configurations clear the 15.8% breakeven bar on both the backtest and the multi-period walk-forward mean. The walk-forward figure is the more trustworthy of the two, since it averages across several distinct market periods rather than one favorable six-month window.

**How to read this honestly:** the edge is real but thin, measured over a limited period, and close enough to breakeven that ordinary execution friction could plausibly erase it. The asymmetric target/stop ratio does most of the work — the model does not need to be right often, only slightly more often than 15.8%. Note also that the tuner selected a **2.3% stop**, tight enough to be vulnerable to routine daily noise; the search has no concept of execution practicality and will happily optimize into regions that only work frictionlessly.

---

## Known limitations

Stated plainly, because a portfolio project that hides its caveats is less useful than one that names them.

1. **Close-price entry assumption.** Labels use the candidate day's close as the entry price, but a live trader acting on that signal can only enter at the next day's open at the earliest. Breakout candidates frequently gap up overnight, so backtested returns capture movement that isn't actually tradeable. This is a one-directional overstatement, not symmetric noise, and it interacts badly with the tuned 2.3% stop.
2. **No concurrent-capital modeling.** Each trade is sized against full current equity as though nothing else were open. When signals cluster, a real portfolio would be splitting capital across simultaneous positions.
3. **No transaction costs or slippage.** Neither commissions nor bid-ask spread are modeled.
4. **Survivorship bias.** The universe is current listings; delisted companies are absent, which biases historical results upward.
5. **Feature-group selection is coarse.** Features are toggled in family bundles to keep the search space tractable, so a strong individual indicator can be dropped alongside weaker group-mates. ATR(14) — the evaluator's top-ranked single feature — was cut this way when its volatility family was excluded.
6. **Feature values are recomputed every tuning trial.** The scan cache is reused, but features and labels are computed in the same pass, so feature computation repeats even when only labeling varies. Correct, but wasteful.
7. **Limited history.** Three years is enough to detect a signal and not enough to be confident it persists across regimes.

---

## Getting started

```bash
git clone https://github.com/<you>/fork-in-the-road.git
cd fork-in-the-road

python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

python -m scripts.init_configs      # required: creates config/*.yaml from the tracked examples
python -m pytest                    # 328 tests, no network required
```

Only `config/*_example.yaml` files are tracked in git — working configs stay local, so experiment-specific tuning never lands in a commit. `init_configs` bootstraps them and never overwrites without `--force`.

A first end-to-end run:

```bash
# Build a dataset (network-bound; start small to sanity-check)
python -m scripts.build_dataset --start 2023-01-01 --end 2023-06-30 --split-date 2023-05-01 -v

# Train, then evaluate against the held-out split
python -m scripts.train_model --train data/processed/train.csv --test data/processed/test.csv

# Simulate P&L with position sizing
python -m scripts.backtest --model data/models/<model>.joblib --test data/processed/test.csv \
    --position-sizing-config config/position_sizing.yaml --threshold 0.33

# Validate stability across multiple time periods
python -m scripts.walk_forward --data data/processed/full.csv \
    --position-sizing-config config/position_sizing.yaml

# Today's ranked candidates with suggested allocations
python -m scripts.scan_live --model data/models/<model>.joblib \
    --position-sizing-config config/position_sizing.yaml --equity 100000
```

Trimming `config/universe.txt` (1,760 tickers by default) is the fastest way to shorten a first run.

---

## Command reference

### Pipeline scripts

| Script | Purpose |
|---|---|
| `scripts/init_configs.py` | One-time setup: create working configs from the tracked examples. |
| `scripts/build_dataset.py` | Scan → featurize → label → write `full.csv` / `train.csv` / `test.csv`. Reuses the scan cache automatically; `--force-rescan` overrides. |
| `scripts/train_model.py` | Train (with optional calibration and early stopping), save model plus metadata sidecar, optionally evaluate. |
| `scripts/evaluate_model.py` | Load a saved model and report accuracy, precision/recall, confusion matrix, per-setup breakdown, calibration curve, feature importances. |
| `scripts/backtest.py` | Return-aware P&L simulation using real forward returns and configured position sizing. |
| `scripts/walk_forward.py` | Multi-fold sequential validation; optional per-fold backtests and trade logs. |
| `scripts/tune.py` | Optuna search over labeling + model + feature-selection space. Writes `trial_log.csv` and `best_params.json` as *suggestions*, never auto-applied. |
| `scripts/scan_live.py` | Rank today's candidates with calibrated probabilities and suggested equity allocations. |

### Debug tools

Manual-verification utilities, each printing full intermediate values so results can be checked by hand against a charting platform:

| Tool | Purpose |
|---|---|
| `tools/inspect_ticker.py` | Raw indicator values for one ticker/date. |
| `tools/compute_features.py` | Run `features.yaml` against one ticker/date. |
| `tools/debug_scanner.py` | Per-condition pass/fail breakdown for a setup match. |
| `tools/debug_labeler.py` | Forward price walk showing exactly which day and threshold triggered a label. |
| `tools/debug_position_sizer.py` | Full sizing computation, including a `--sweep` table across probabilities. |

---

## Testing

```bash
python -m pytest              # 328 tests, ~9 seconds, no network
```

Testing approach:

- **No network in the test suite.** `yfinance.download` is monkeypatched with hand-constructed price paths, so tests are deterministic and offline.
- **Hand-computed expected values.** Indicator, Kelly, EV, and equity-curve tests assert exact arithmetic derived by hand, not "whatever the code returned when it was written."
- **Regression tests name the bug they prevent.** Several tests exist because a specific defect shipped; their docstrings explain the failure mode so the test isn't later "simplified" back into uselessness.

Two examples of bugs caught this way, both the same class — duplicated logic that drifted:

- `format_predictions()` accessed `model.feature_importances_` directly, which works on a raw `XGBClassifier` and raises on a calibrated one. Fixed by consolidating into a single `TrainedModel.feature_importances()`. The underlying test gap was that *no* test called `format_predictions()` at all.
- The tuning objective kept its own hardcoded copy of the non-feature column list. When `trading_days_held` was added to the trainer's exclusion set, that copy wasn't updated — silently passing a post-outcome column to the model as a feature and bypassing the leakage guard entirely. Fixed by making `NON_FEATURE_COLUMNS` a single shared constant.

---

## Where this goes next

Roughly in priority order:

1. **Next-day-open entry pricing.** The highest-value correctness fix. Redefining entry as the next session's open removes the overnight-gap overstatement, at the cost of a full dataset rebuild and retrain.
2. **Execution-cost modeling.** A configurable slippage/commission haircut, to test whether the measured edge survives realistic friction. Given how thin the margin is, this is close to a go/no-go question.
3. **Per-feature caching.** Separating feature computation from labeling so tuning trials that vary only labeling stop recomputing identical feature values — the direct unlock for per-feature (rather than per-group) tuning.
4. **Feature-parameter search.** Tuning indicator windows themselves (`rsi_14` vs `rsi_10`) rather than only which features are included. Depends on (3) to be affordable.
5. **Concurrent-position portfolio simulation.** Model real capital constraints when signals cluster.
6. **Per-setup models.** The five setups may not share optimal parameters. Worth testing empirically on the high-volume setups; the low-volume ones almost certainly benefit more from pooling than from isolation.
7. **More history and broader regime coverage**, which every item above ultimately depends on for its conclusions to mean anything.

---

## License

MIT — see [LICENSE](LICENSE).
