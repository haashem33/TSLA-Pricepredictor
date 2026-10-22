# TSLA Intraday Tokenized Forecasting Model — Build Instructions for Claude Code

> Put this file in the root of an empty repository as `CLAUDE.md`, then start Claude Code there and say:
> **"Read CLAUDE.md and start Phase 1. Stop at the end of each phase and report the gate results."**

---

## 0. Mission and ground rules

Build a research pipeline that forecasts the **distribution** of TSLA's next 5-minute to 60-minute returns from tokenized market data, using a small causal Transformer. The goal is an honest, leakage-free estimate of whether any edge exists. It is **not** to produce an impressive-looking backtest.

Rules for Claude Code (follow these at all times):

1. **Work phase by phase** (section 12). At the end of every phase, stop, run the gate checks, and report the numbers to the user. Do not start the next phase until the user says so.
2. **Never use future data.** Every scaler, bin edge, cluster, HMM, feature selection and hyperparameter must be fit on training data only. Every rolling statistic uses data up to and including bar t, never t+1.
3. **Never invent results.** Report only numbers produced by code that ran. If something fails, say so.
4. **Write tests** for every data transform, especially the leakage tests in section 11.
5. **Cache raw data locally** (`data/raw/`). Never re-download what is already cached. Respect the data provider's rate limits.
6. **Log every experiment** to `experiments/registry.csv` (section 10.6). The number of trials is needed for the deflated Sharpe ratio.
7. **Targets are returns or volatility, never price levels.**
8. Keep the model **small** (section 8.3). Overfitting is the main risk, not lack of capacity.
9. Ask the user before: adding paid data sources, changing the target stock, or touching the final holdout or the forward test (section 10.5).
10. Use one git commit per completed task with a clear message.
11. **Never use the test month for any choice** (early stopping, calibration, temperature, hyperparameters, feature selection, thresholds). Inside each fold, the inner validation set is the last 2 months of the training window, purged and embargoed from the rest of training (section 10.1).
12. Recorded decisions (with dates and reasons) live in `reports/decisions.md`. If this file and the decisions log disagree, the decisions log wins and this file must be updated.

---

## 1. Fixed decisions

| Decision | Value |
| --- | --- |
| Target stock | TSLA only |
| Main bars | 5-minute bars, regular session only (09:30–16:00 US/Eastern), 78 bars per day |
| Context streams | SPY and QQQ 5-minute bars (market), TSLA/SPY/QQQ daily bars (slow context) |
| Price adjustment | Split- and dividend-adjusted (TSLA split 5:1 in Aug 2020 and 3:1 in Aug 2022 — verify adjustment is correct around both). **Data v2 (D9):** TSLA 2016-01..2022-08 is built from unadjusted (Alpaca `Adjustment.RAW`) 1-min bars divided by the cumulative split factor in floating point (no rounding), volumes × factor; the vendor's split-adjusted prices are rounded to $0.01 and are not used for TSLA before 2022-09. SPY/QQQ: dividend back-adjusted |
| Forecast horizons | h = 1, 3, 12 bars (5, 15, 60 minutes) as training targets; **primary trading horizon h = 12** (backtests and gates are judged on h = 12) |
| Volatility scale | EWMA × seasonal σ̂ / S_{t,h} (section 5.2, 9.1) is the scale for tokens and labels. GARCH-t is a feature and a benchmark, not the scale |
| Horizon rule | Labels never cross the session close; drop samples within h bars of the close for horizon h |
| Primary target | Volatility-scaled forward log return, as a categorical distribution over bins |
| Secondary targets | Forward realized volatility; triple-barrier outcome |
| Context length | 256 bars (~3.3 trading days), option to test 512 |
| Model | 2–4 layer causal Transformer, d_model = 128, ~0.74–1.5M parameters (tiny fallback: d_model = 64, ~0.2M) |
| Validation | Expanding-window walk-forward with purging and embargo; inner validation = last 2 months of each training window; final untouched holdout (2026-01-01..2026-10-09) plus a sealed forward test (after 2026-10-09) |
| Success metrics | Out-of-sample IC (section 10.6 definition: z-space, pooled per monthly fold), ICIR, t-stat, binomial p of IC>0 folds, hit rate, CRPS, net Sharpe after costs, break-even cost, deflated Sharpe ratio — each for the full period, 2023–2025, and the most recent 24 folds (information only; gates use the D10 recency check) |
| Hardware | GTX 1070 (8 GB): train in **FP32** (no bf16/fp16) |
| Later experiments (not now) | Dollar bars; training on many stocks. (Self-supervised pre-training and a fine-tuned Kronos-mini comparison are part of Phase 4, sections 9.4 and 10.3.) |

---

## 2. Tech stack

- Python 3.11+, managed with `uv` or `pip` + `requirements.txt`.
- Data: `pandas`, `numpy`, `pyarrow` (store as Parquet), `pandas_market_calendars` (sessions, half-days, holidays).
- Statistics: `statsmodels` (ADF, KPSS, Ljung–Box), `arch` (GARCH), `hmmlearn` (HMM), `scikit-learn` (k-means, scaling, metrics), `scipy`.
- Baselines: `lightgbm`, `catboost`.
- Deep learning: `torch` (2.x), FP32 on the GTX 1070 (Pascal has no bf16 and slow fp16; do not use mixed precision); fall back to CPU.
- Config: `pydantic` or `omegaconf` YAML configs in `configs/`.
- Testing: `pytest`. Lint: `ruff`.
- Plots: `matplotlib` (saved to `reports/figures/`).

---

## 3. Repository layout

```
.
├── CLAUDE.md
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   ├── tokenizer.yaml
│   ├── model.yaml
│   └── backtest.yaml
├── data/
│   ├── raw/            # cached API responses (gitignored)
│   ├── interim/        # cleaned bars (Parquet)
│   └── processed/      # features, tokens, labels per fold
├── src/tslaq/
│   ├── data/           # download.py, clean.py, calendar.py
│   ├── features/       # returns.py, volatility.py, indicators.py, stats_models.py, regime.py, context.py
│   ├── selection/      # ic.py, redundancy.py, importance.py, condprob.py
│   ├── tokenizer/      # fields.py, bins.py, shapes.py, tokenizer.py
│   ├── labels/         # forward_returns.py, realized_vol.py, triple_barrier.py, weights.py
│   ├── cv/             # walk_forward.py, purge.py, cpcv.py
│   ├── models/         # baselines.py, embeddings.py, transformer.py, heads.py, losses.py
│   ├── train/          # train.py, calibrate.py, ensemble.py
│   ├── backtest/       # positions.py, costs.py, metrics.py, dsr.py
│   └── utils/          # registry.py, seed.py, io.py
├── tests/
├── experiments/registry.csv
├── reports/            # one markdown report per phase + figures
└── requirements.txt
```

---

## 4. Data (Phase 1)

### 4.1 Sources

Use the Alpha Vantage API (key in environment variable `ALPHAVANTAGE_API_KEY`, never committed). Check the current API docs before coding; endpoints and parameters change.

- `TIME_SERIES_INTRADAY` for TSLA, SPY and QQQ with `interval=5min`, `adjusted=true`, `extended_hours=false` (the default includes pre- and post-market bars, which we do not want), `outputsize=full`, and `month=YYYY-MM`. One call returns one symbol for one month. History is available month by month back to 2000-01 (TSLA listed in June 2010). Download at least 5 years (~180 calls for three symbols).
- **Daily bars: build them from the 5-minute bars** (open of the first bar, max high, min low, close of the last bar, summed volume). Do not depend on `TIME_SERIES_DAILY_ADJUSTED`, which may require a premium key, or on the free daily endpoint, whose free history may be limited.
- `EARNINGS` for TSLA: use `reportedDate` and, if present, `reportTime` (before open / after close) to decide which session the earnings reaction falls in. TSLA has typically reported after the close, so the reaction usually lands in the next session; if `reportTime` is missing, assume after the close and log the assumption.
- Optional later: `NEWS_SENTIMENT`, `REALTIME_PUT_CALL_RATIO` / historical options. Not in v1.

**Rate limits and resumable downloads.** The free key allows about 25 requests per day and 5 per minute; paid plans remove the daily cap. The downloader must:

- wait at least 12 seconds between calls on a free key (configurable in `configs/data.yaml`);
- save each month's raw response to `data/raw/<symbol>/<YYYY-MM>.json` immediately, and skip any month already saved;
- detect the API's rate-limit / "Information" / "Note" messages, stop cleanly, and resume from the next missing month on the next run;
- keep a `data/raw/download_log.csv` (symbol, month, status, timestamp) and report progress (months done / months needed).

On a free key, a 5-year download takes about 8 days of daily runs; tell the user before starting and ask whether to proceed or use a paid key.

If the user prefers another data vendor, keep the same interface: `download.py` returns raw frames, `clean.py` produces the canonical schema.

### 4.2 Canonical bar schema (Parquet, one file per symbol)

`timestamp` (tz-aware, US/Eastern, bar **end** time), `open`, `high`, `low`, `close`, `volume`, `session_date`, `bar_index` (0–77), `is_half_day`.

### 4.3 Cleaning rules

- Keep regular-session bars only. Use `pandas_market_calendars` (NYSE) for sessions and half-days.
- Missing bars: forward-fill price with zero volume and set a `missing` flag; drop days with more than 5% missing bars.
- Detect bad ticks: a bar whose |log return| exceeds 15 × the rolling median absolute return is flagged and inspected, not silently deleted. Report how many.
- Verify adjustment: log-returns across the two TSLA split dates must not show a −80% / −67% jump.
- Write a data-quality report to `reports/phase1_data.md`: date range, bar counts per year, missing-bar stats, split check, return histogram.

---

## 5. Core transforms (Phase 2)

All formulas below are causal: at bar t they use only bars ≤ t.

### 5.1 Returns

Log return: `r_t = ln(C_t / C_{t-1})`. The first bar of each day uses the previous session's close, so it contains the overnight gap. Store an additional `r_intraday_t = ln(C_t / O_t)` and `gap_t = ln(O_t / C_{t-1})` for bar_index 0.

### 5.2 Intraday volatility scale σ̂

Intraday volatility is U-shaped over the day, so the scale combines a slow EWMA level with a time-of-day profile:

```
seasonal_k  = median over the last 60 sessions of |r| at bar_index k   (computed from past sessions only)
u_t         = r_t / seasonal_{k(t)}                                     (de-seasonalized return)
level²_t    = λ · level²_{t-1} + (1 − λ) · u²_{t-1},   λ = 0.97 (tune on train)
σ̂_{t|t-1}   = sqrt(level²_t) · seasonal_{k(t)}
```

`σ̂_{t|t-1}` must only use information up to t−1. Treat bar_index 0 (overnight gap) with its own profile value.

### 5.3 Normalization helpers

- Rolling z-score with past-only window: `z = (x_t − mean_{t-w..t-1}) / std_{t-w..t-1}`.
- Rolling percentile rank over a past window.
- Clip continuous features to [−5, 5] after z-scoring.

### 5.4 Stationarity checks

For every continuous feature, run ADF (reject unit root, p < 0.05) and KPSS (fail to reject stationarity) on the training period. Log results; features that fail either test get further normalization (or a smaller fractional-differencing d) or are dropped.

### 5.5 Fractional differentiation (optional feature)

Fixed-width-window fractional differencing of log price:
`w_0 = 1, w_k = −w_{k−1} · (d − k + 1) / k`, truncated when |w_k| < 1e-4. Find the smallest d in [0, 1] (step 0.05) that passes ADF on training data. Use it as one continuous feature.

### 5.6 Library pitfalls that leak the future (read before using these libraries)

| Library / tool | Leaky usage | Correct usage |
| --- | --- | --- |
| `hmmlearn` | `predict_proba` / `predict` (forward–backward and Viterbi use future observations) | Fit with `hmmlearn` on training data, then compute **filtered** state probabilities yourself with the forward recursion: `p_pred = p_prev @ transmat`; `p_t ∝ p_pred * emission_likelihood(x_t)`; normalize |
| `arch` (GARCH) | Using `res.conditional_volatility` from a fit that included validation/test data; refitting on test data | Fit on training data; then build a fixed-parameter model over the full series (`model.fix(res.params)`) or recurse forward manually, so each σ_t uses only past data and trained parameters |
| `statsmodels` state space / Kalman | `smoothed_state` | `filtered_state` (or the one-step-ahead `predicted_state`) |
| `scipy.signal` | `filtfilt`, centered windows | `lfilter` or explicit trailing windows |
| `pandas` | `rolling(..., center=True)`, `shift(-k)` in features, `fillna(method="bfill")` | Trailing windows only; negative shifts only when building labels; forward-fill only |
| wavelets / FFT | Transform over the whole series | Transform the trailing window ending at t only |
| `sklearn` scalers, k-means, quantile bins | `fit` on all data | `fit` on the training fold only, `transform` everywhere |

Add a test for each of these in `tests/` (section 11).

---

## 6. Features (Phase 2)

Compute everything multi-scale. Windows are in 5-minute bars (78 = one day).

### 6.1 Candle shape (per bar)

- body = (C − O) / O; body_ratio = (C − O) / (H − L + ε)
- range = (H − L) / O
- upper_shadow = (H − max(O, C)) / O; lower_shadow = (min(O, C) − L) / O
- close_location = (2C − H − L) / (H − L + ε) ∈ [−1, 1]

### 6.2 Relation to the previous candle

- dH = (H_t − H_{t−1}) / ATR_t; dL = (L_t − L_{t−1}) / ATR_t
- dC = (C_t − C_{t−1}) / ATR_t
- pos_in_prev = (C_t − L_{t−1}) / (H_{t−1} − L_{t−1} + ε)
- body_overlap = overlap of the two candle bodies / union of the bodies

### 6.3 Trend and momentum

- Momentum: log return over k ∈ {3, 6, 12, 24, 78} bars, divided by σ̂·√k
- C / EMA_n − 1 for n ∈ {12, 26, 78}; EMA_12 / EMA_26 − 1
- MACD (12, 26, 9): (MACD − signal) / (C · σ̂)
- RSI(14): (RSI − 50) / 50
- Stochastic %K(14) / 100
- Linear-regression slope of ln C over 24 and 78 bars × window, plus R²
- ADX(14) / 100
- **Intraday momentum** (Gao, Han, Li & Zhou, JFE 2018: for SPY, the first half-hour return measured from the previous close predicted the last half-hour return, more strongly on volatile and high-volume days; a later replication found weaker results in recent years, so test it rather than assume it):
    - `r_first30` = ln(C at 10:00 / previous session close), available from bar_index 5 onward (0 / MISSING before)
    - `r_12th30` = ln(C at 15:30 / C at 15:00), available from 15:30 onward
    - their signs, and `r_first30` × first-half-hour relative volume and × first-half-hour realized volatility
    - In the Phase 2 conditional tables (6.9), test specifically: P(up over the last half-hour | sign of r_first30), overall and by year.

### 6.4 Volatility and range

- ATR(14) / C
- Parkinson volatility over 12 and 78 bars: sqrt(mean((ln H/L)²) / (4 ln 2))
- Garman–Klass volatility over 78 bars
- Bollinger %B(20, 2) and bandwidth
- Volatility ratio σ(12) / σ(78)
- GARCH(1,1) with Student-t innovations on de-seasonalized returns u_t, fit on each training fold, then applied to later data with parameters held fixed (section 5.6); features = conditional σ forecast (GARCH-t σ, and its h-bar forecast × seasonal) and standardized residual
- HAR realized-variance features: log of realized variance over the last 1, 5 and 22 completed sessions (sum of r² over those sessions' bars, known at the open), plus the current session's realized variance so far, de-seasonalized by the cumulative seasonal profile

### 6.5 Volume and intraday-specific

- Relative volume by time of day: ln(V_t / median of V at the same bar_index over the last 20 sessions)
- Session VWAP deviation: C_t / VWAP_session,t − 1 (VWAP resets each day)
- OBV slope over 24 bars, z-scored
- Amihud illiquidity: |r_t| / (C_t · V_t), log, rolling mean over 78
- Time-of-day: bar_index (categorical) and minutes to close

### 6.6 Statistical-model features

- Kalman local-linear-trend filter on ln C: filtered slope and innovation (z-scored). Fit noise variances on training data.
- Hurst exponent (R/S) over 256 bars; variance ratio VR(12) over 256 bars
- Autocorrelation of r at lag 1 over 78 bars. Caution: on trade prices, bid–ask bounce creates spurious negative lag-1 autocorrelation in short-interval returns. Any reversal signal must be re-tested with a one-bar execution delay (section 10.4) before it counts.
- Shannon entropy of 10-bin quantized returns over 78 bars
- HMM: 3-state Gaussian HMM on [u_t, ln σ̂_t] fit on training data; use the **filtered** (forward-algorithm) state probabilities only — never the smoothed (Viterbi/backward) ones. Do not use `hmmlearn`'s `predict_proba` for this; see section 5.6.

### 6.7 Market and daily context

**D12:** on TSLA sessions where QQQ or SPY bars are missing (e.g. QQQ 2018-05-02/03, SPY 2019-08-12), keep the TSLA rows and set the market features/tokens to MISSING (also for forward-filled market bars and the market bar-0 return after a missing market session).

- SPY and QQQ: same-bar return z-scores, 12-bar momentum, relative volume
- Rolling beta of TSLA to QQQ over 390 bars (5 sessions); residual return r_TSLA − β·r_QQQ (z-scored)
- Daily (known at the open, i.e. from the previous completed day): previous day return / daily ATR, daily ATR / C, C_prev / SMA_20(daily) − 1, 5-day momentum, today's gap
- Event flags: earnings-reaction session (the session after an after-close report, or the report day for a before-open report), the session after that, option expiry (third Friday), half-day

### 6.8 Feature selection (Phase 2 deliverable)

On training folds only:

1. IC of each feature vs. realized z at h ∈ {1, 3, 12}, using the section 10.6 definition (Spearman pooled within each calendar month of the training period; ICIR = mean/std across months; t-stat; binomial p of IC>0 months), plus per-time-slot IC; report by year, for 2023–2025 and for the last 24 months of the training window.
2. IC decay curve per feature (h = 1 … 24).
3. Cluster features by |Spearman correlation| distance (hierarchical clustering, threshold 0.7).
4. Clustered MDA with LightGBM under purged CV: permute a whole cluster at once.
5. Run the conditional up-probability analysis (section 6.9).
6. Keep 20–40 features: the most stable 1–2 per useful cluster by IC, **plus** any feature with weak IC that passes the extreme-bucket rule in 6.9. Save the list to `configs/features.yaml` with the reason each was kept (IC, extreme lift, or both).

Write `reports/phase2_features.md` with tables and figures.

### 6.9 Conditional up-probability analysis (Phase 2 deliverable)

Goal: for every feature, measure how the probability that TSLA goes up next changes with the feature's value. The information coefficient misses effects that are non-monotonic, extremes-only, or regime-dependent; this analysis catches them. Run it on training folds only.

**Outcomes** (for each horizon h ∈ {1, 3, 12}, never crossing the session close):

- `up_h = 1 if r_{t→t+h} > 0, else 0`. Exclude exact zeros from the up/down share and report how many there were.
- `dir_cost_h ∈ {up, down, flat}`: up if r > c_rt, down if r < −c_rt, else flat, where c_rt = round-trip cost from `configs/backtest.yaml`.
- Also record the forward return r_{t→t+h} and z_{t→t+h} for payoff statistics.

**Single-feature tables.** For each feature f and horizon h:

1. Decile edges fit on the training fold. Also compute extra tail buckets: bottom/top 2% and 5%.
2. For each bucket k: n, n_eff = n / h, p̂_k = P(up | bucket k), base rate p̂_0, lift_k = p̂_k − p̂_0, Wilson 95% interval using n_eff, P(up beyond cost), P(down beyond cost), mean and median forward return, mean z.
3. Day-block bootstrap (resample whole sessions, 1,000 draws) for the extreme-bucket lift as a cross-check of the Wilson interval.
4. Shape summary: Spearman correlation of bucket index vs p̂_k (monotonicity), extreme spread p̂_D10 − p̂_D1, and a label: monotonic / U-shaped / extremes-only / none.
5. Single-feature ROC AUC for `up_h` and mutual information between the bucket and `up_h`.
6. Stability: the same table per calendar year; count the years where the extreme-bucket lift has the same sign as the full-period lift.

**Multiple testing.** For every (feature, bucket, horizon) cell, a binomial test of p̂_k vs p̂_0 using n_eff. Apply Benjamini–Hochberg at FDR 10% across all cells. Report how many cells survive.

**Extreme-bucket keep rule.** Keep a feature with weak IC if at least one tail or extreme decile bucket (a) survives Benjamini–Hochberg, (b) has the same lift sign in ≥ 70% of years, and (c) has a mean forward return with the same sign as the lift.

**Pairwise tables.** 5 × 5 quintile grids (min 500 bars per cell, otherwise mark the cell "insufficient") of P(up) and lift for:

- RSI(14) × VREG (volatility regime)
- VWAP deviation × TOD (time-of-day slot)
- 12-bar momentum × HMM state
- relative volume × RET token group (down / flat / up)
- gap (first bars of the day) × TOD within the first hour
- QQQ same-bar return × TSLA residual return

Also report the interaction effect: lift in each cell minus the sum of the two single-feature lifts.

**Token tables (run again in Phase 3 once tokens exist).** P(up) and lift by each value of RET, SHAPE, REL, GAP, VOLU, VREG, HMM and TOD, and by two-token sequences (previous RET → current RET, previous RET → current SHAPE, REL × VOLU). Use only sequences with n ≥ 500. Add these to `reports/phase3_tokenizer.md`.

**Outputs.**

- `reports/tables/condprob/<feature>_h<h>.csv` (one row per bucket) and `<feature>_h<h>_by_year.csv`
- `reports/figures/condprob/<feature>_h<h>.png`: bar chart of P(up) per bucket with Wilson error bars and a horizontal base-rate line
- `reports/figures/condprob/pair_<a>__<b>_h<h>.png`: heatmap of lift with cell counts
- A summary table in `reports/phase2_features.md`: features ranked by the size of their significant, stable extreme lift, with shape label, IC and AUC side by side

**Interpretation rules to state in the report.** With ~10,000 bars per decile, a 95% interval is about ±1 percentage point at h = 1 and wider at longer horizons. Lifts inside that band are noise. A lift that disappears after costs is not tradable on its own but can still help the model in combination.

---

## 7. Tokenizer (Phase 3)

One composite token per bar. Each field has its own small vocabulary and its own embedding table; embeddings are summed.

### 7.1 Return token (RET)

```
z_t        = r_t / σ̂_{t|t-1}
tok_ret_t  = number of edges e_k with clip(z_t, −6, 6) ≥ e_k,   e_k = quantile_train(k / B)
```

- B_in = 64 bins for inputs.
- Output head bins: B_out = 32 (single stock; can test 64 with label smoothing).
- Bin edges are fit on the training fold only, saved with the model, and never refit on test data.
- De-tokenization uses the mean of training z values inside each bin (not the midpoint), times σ̂.

### 7.2 Field specification

| Field | Definition | Vocab |
| --- | --- | --- |
| RET | Volatility-scaled return z_t (7.1) | 64 |
| SHAPE | k-means (fit on train) of [body/ATR, upper_shadow/ATR, lower_shadow/ATR, close_location, range/ATR] | 32 |
| REL | Relation to previous candle: {inside, outside, higher-high-higher-low, lower-high-lower-low, mixed} × {bullish engulfing, bearish engulfing, none} | 15 |
| GAP | Overnight gap / σ̂ for bar_index 0, quantile bins; a dedicated "no gap" token for all other bars | 9 + 1 |
| VOLU | Relative volume by time of day (6.5), quantile bins | 12 |
| VREG | Percentile of σ̂_t within its trailing 20-session history | 8 |
| HMM | Filtered most-likely HMM state | 3 + unknown |
| MKT | QQQ same-bar return z-score, quantile bins | 16 |
| TOD | bar_index bucketed into 13 half-hour slots | 13 |
| DOW | Day of week | 5 |
| EVENT | earnings-reaction session, session after it, opex, half-day, none (see 6.7) | 5 |
| Special | PAD, MASK, BOS, MISSING (per field), DAY_SEP | 5 |

Every field must have a MISSING token for warm-up periods and data gaps (including missing QQQ/SPY sessions for MKT, D12).

### 7.3 Input vector per bar

```
e_t = Σ_fields E_f[tok_f,t] + W_c · clip(x_cont_t, −5, 5) + positional encoding
```

`x_cont_t` = the 20–40 selected continuous features (section 6.8), z-scored with training statistics.

### 7.4 Tokenizer checks (must pass before Phase 4)

- Bin occupancy: every RET bin has ≥ 50 training samples; merge sparse tail bins otherwise.
- Round-trip: correlation between de-tokenized and true returns, and RMSE in σ units — report both.
- Leakage test (section 11).
- **Gate 3 — token-only baseline:** LightGBM on the last 24 bars of tokens (one-hot or ordinal) predicting the h = 3 and h = 12 return sign and z. If the out-of-sample IC (section 10.6) is not above zero with a stable sign across years **and** does not pass the recency check (section 10.6, D10: full period t ≥ 2; 2023–2025 mean IC > 0 with t ≥ 1), stop and report to the user before building the network.

Write `reports/phase3_tokenizer.md`.

---

## 8. Model (Phase 4)

### 8.1 Architecture

| # | Layer | Specification |
| --- | --- | --- |
| 1 | Field embeddings | One table per field (7.2), dim 128, summed |
| 2 | Continuous projection | Variable-selection network with **hidden size 32**: each feature → linear 1→32 → its own gated residual network (32→32, GLU); a selection GRN over the concatenated 30×32 vector → softmax feature weights; weighted sum → linear 32→128. Save the weights for interpretation. Ablation option: replace with a single linear layer (n_features → 128) |
| 3 | Positional encoding | Rotary (RoPE) in attention; TOD and DOW tokens already give calendar info |
| 4 | Backbone | 2–4 pre-norm blocks: causal multi-head self-attention (4 heads), SwiGLU feed-forward (4 × d), RMSNorm, dropout 0.1–0.2 |
| 5 | Market cross-attention (optional, test in ablation) | Queries from TSLA stream, keys/values from a QQQ/SPY token stream |
| 6 | Return heads | One linear + softmax over B_out bins per horizon h ∈ {1, 3, 12} |
| 7 | Volatility head | Linear → log realized variance over next h bars |
| 8 | Barrier head | Linear → softmax over {upper hit, lower hit, timeout} |

Causal masking is mandatory: position t attends only to positions ≤ t. Predictions are made from the hidden state at the last bar of the window (and, for training efficiency, at every position with a valid label).

### 8.2 Regularization

- AdamW, weight decay 0.05, dropout 0.1–0.2
- Random token masking: replace 10% of input tokens with MASK during training
- Gaussian jitter (σ = 0.05) on continuous inputs
- HL-Gauss label smoothing for the return heads (Farebrother et al., 2024, "Stop Regressing"): the target distribution is a Gaussian N(z_true, σ_s²) integrated over each bin's edges (CDF differences, renormalized), then cross-entropy against the predicted bin probabilities. Default σ_s = the median bin width in z units; tune in {0.5, 1, 2} × that on validation.
- Early stopping on **inner-validation IC** (last 2 months of the training window, section 10.1), not validation loss and never the test month
- Seed ensemble of 5 models **only for the final configuration**; single seed during development and ablations

### 8.3 Size

Three configurations, defined in `configs/model.yaml`:

| Config | Layers | d_model | Heads | FFN hidden | VSN | Approx. parameters |
| --- | --- | --- | --- | --- | --- | --- |
| `tiny` (fallback) | 2 | 64 | 4 | 256 | hidden 16 | ~0.2M |
| `base` (start here) | 2 | 128 | 4 | 512 | hidden 32 | ~0.74M (~0.57M with linear projection instead of VSN) |
| `large` (max) | 4 | 128 | 4 | 512 | hidden 32 | ~1.26M (~1.52M with market cross-attention) |

Approximate breakdown for `base`: token embeddings ~29k (≈228 token values × 128), VSN ~170k (30 features), each Transformer block ~262k (attention 4 × 128² + SwiGLU 3 × 128 × 512 + norms), output heads ~13k, optional market cross-attention ~66k per block.

Rules:

- Start with `base`. Move to `large` only if it wins the paired fold-level comparison (section 10.6: share of folds won, Wilcoxon on fold ICs).
- If `base` overfits (training loss keeps falling while validation IC falls), switch to `tiny` before adding more regularization.
- Print the exact parameter count (`sum(p.numel() for p in model.parameters())`) at the start of every run and log it to the registry. If it differs from the table by more than 20%, report why.
- Never exceed ~2M parameters on TSLA-only data.

---

## 9. Labels and losses (Phase 4)

### 9.1 Labels (computed per horizon h; never crossing the session close)

- **Horizon scale.** Because intraday volatility depends on time of day, do not use σ̂_t·√h. Use the forecast scale over the actual bars ahead: `S_{t,h} = level_t · sqrt( Σ_{i=1..h} seasonal²_{k(t+i)} )` (section 5.2), computed with information up to t only.
- **Return bin:** z_{t→t+h} = ln(C_{t+h} / C_t) / S_{t,h}, binned with frozen output edges.
- **Realized volatility:** ln(Σ_{i=1..h} r²_{t+i} + ε).
- **Triple barrier:** upper = +k·S_{t,h}, lower = −k·S_{t,h}, time limit = h bars, k ∈ {1.0, 1.5} (config). Label = first barrier touched using bar highs and lows. If both are touched within the same bar, the order is unknown: label it as the **lower** barrier (conservative, as if the stop-loss hit first) and report how many such cases occurred.
- **Sample weights:** average uniqueness = mean over the label window of 1 / (number of concurrent labels). Use as loss weights.

### 9.2 Loss

```
L = λ1 · CE_smooth(return bins, all horizons) + λ2 · QLIKE(σ̂², σ²) + λ3 · CE(barrier)
QLIKE(σ̂², σ²) = σ²/σ̂² − ln(σ²/σ̂²) − 1
default λ = (1.0, 0.5, 0.5)
```

### 9.3 Training settings

AdamW, lr 3e-4, warm-up 2% of steps then cosine decay, batch 256 windows, gradient clipping 1.0, **FP32** on the GTX 1070. Temperature-scale the return heads on the **inner validation** set (last 2 months of the training window) after training.

- **Retraining cadence:** neural models retrain **quarterly** (each model predicts the next 3 test months), warm-started from the previous fold's weights; tree models stay monthly. Report how many epochs each warm start ran.
- **Timing:** time the first training run and report projected total Phase 4 hours to the user before starting the full Phase 4 grid.
- **Volatility head target (D11):** it must beat GARCH-t×seasonal on out-of-sample QLIKE **on identical rows** (RV > 0, no forward-filled bar in the label window), fold by fold, with a Diebold–Mariano test on the QLIKE differential (day-clustered errors). The Phase 1 v2 values 1.435 / 0.522 / 0.248 (h = 1 / 3 / 12) are reference values only. Also report QLIKE with RV floored at the 1st percentile of the training-fold RV (robustness: h = 1 QLIKE is dominated by a few tiny-RV rows).

### 9.4 Optional self-supervised pre-training (Phase 4 ablation)

A 2025 study of time-series foundation models on daily excess returns (Rahimikia, Ni & Wang, arXiv 2511.18578) found that off-the-shelf models underperformed CatBoost/LightGBM and fine-tuning did not close the gap, while models **pre-trained from scratch on financial data** gained substantially. Test the in-project version of this:

1. Pre-train the same backbone with next-token prediction (RET token of the next bar) plus 15% masked-token reconstruction, on TSLA training bars only.
2. Variant: add QQQ and SPY token sequences (same tokenizer, fit on TSLA training data) as extra unlabeled sequences.
3. Fine-tune each on the supervised TSLA labels (9.1) and compare with training from scratch, fold by fold.
4. Keep pre-training (and any extra symbols) only if it wins the paired fold-level comparison on TSLA (section 10.6). Never pre-train on data after a fold's training end.

---

## 10. Validation and backtest (Phases 4–5)

### 10.1 Splits

- Expanding-window walk-forward: train on all data up to month M, validate on month M+1, step one month.
- **Purge:** remove training samples whose label window overlaps the validation period.
- **Embargo:** additionally drop the 78 bars (one session) after each validation block from training.
- Every fold refits: scalers, bin edges, SHAPE k-means, HMM, GARCH, feature selection, model.
- **Inner validation:** inside each fold, the last 2 calendar months of the training window are the inner validation set, used for early stopping, calibration, temperature scaling and hyperparameter choice. Purge and embargo it from the inner training part exactly as above. After choices are made, models may be refit on the full training window with the chosen settings (fixed number of iterations). The test month is used only for scoring.
- **Training-window ablation (Phase 4):** expanding window vs rolling 3-year window vs recency-weighted samples (exponential decay of sample weight with age; half-life chosen on inner validation).

### 10.2 Combinatorial purged CV (Phase 5)

Split the development period into N = 10 blocks, test on every combination of k = 2 blocks with purge and embargo, and reconstruct the backtest paths to get a distribution of Sharpe ratios.

### 10.3 Baselines (must be reported next to every model result)

1. Zero-return forecast (and historical-mean forecast)
2. EWMA/seasonal σ̂ and GARCH-t for volatility (compared on identical rows, fold by fold, D11; Phase 1 v2 reference QLIKE: GARCH-t 1.435 / 0.522 / 0.248 at h = 1 / 3 / 12)
3. Logistic regression, LightGBM and CatBoost on the selected continuous features (tree ensembles were the strongest benchmarks in the 2025 study above)
4. LightGBM on tokens (Gate 3 model)
5. Phase 4 reference models (optional but recommended): Kronos-mini (4.1M parameters, 2,048-bar context; github.com/shiyu-coder/Kronos) zero-shot and fine-tuned on TSLA training folds with the repo's fine-tuning scripts; optionally Chronos-2. Run them on exactly the same folds and metrics. Expect weak zero-shot results; the fine-tuned version is the fair comparison.

### 10.4 From forecast to position

```
E[r]  = Σ_bins p_bin · (bin mean z) · S_{t,h}     (for the chosen trading horizon; S_{t,h} from 9.1)
w_t   = c · E[r] / S²_{t,h},   c ∈ [0.25, 0.5],   |w_t| ≤ w_max (config, default 1.0)
```

- **Execution timing:** a signal computed at the close of bar t is filled at the **open of bar t+1** (config option: VWAP of bar t+1). P&L is measured from the fill price, not from C_t. Report results with and without an extra one-bar delay; an edge that disappears with one more bar of delay is likely bid–ask bounce or noise.
- No-trade band: only change position when |Δw| exceeds a threshold (config).
- No new positions in the last 10 minutes; flatten at the session's final bar (no overnight positions in v1), charging normal costs.
- Costs per side (config): half-spread + commission + slippage, default 3 bps total per side; always also report results at 2× costs. Shorts: add a borrow cost parameter.
- **Break-even analysis (every backtest):** the cost per side at which net Sharpe = 0, and net results at 0.5, 1, 3 and 6 bp per side. Primary trading horizon h = 12.

### 10.5 Final holdout

Reserve the most recent 9 months as a holdout: **2026-01-01..2026-10-09**, sealed. Do not load it in any phase until the user explicitly approves evaluating the frozen design, and evaluate it once.

**Forward test (second sealed set):** data after 2026-10-09. It is downloaded only at the end, after the holdout evaluation, with the user's approval, and evaluated once with the same frozen design. (The holdout may have been seen in the earlier `tsla-diffusion` project; the forward test is the only set guaranteed unseen.)

### 10.6 Metrics and registry

For every run append one row to `experiments/registry.csv`: timestamp, git hash, config hash, phase, model, n_params, fold scheme, IC (h=1,3,12), ICIR, hit rate, CRPS, Brier, net Sharpe (per-period and annualized), Sortino, max drawdown, turnover, Sharpe at 2× costs, break-even cost, n_trials_so_far.

Metrics definitions:

- **IC (primary)** = Spearman correlation between predicted z and realized z (z = forward return / S_{t,h}), **pooled within each monthly test fold**. Report the mean over folds; **ICIR = mean / std across folds**; **t = ICIR · sqrt(n_folds)**; and a **one-sided binomial p-value** for the share of folds with IC > 0 (H0: p = 0.5).
- **Cross-check:** per-time-slot IC = Spearman across days at each bar_index, averaged over slots (and shown per half-hour slot).
- **Do not use** Spearman per day averaged over days: for one stock it ranks bars over time within a day, and overlapping forward windows bias it (on a random walk a 78-bar momentum signal scores ≈ −0.31 at h = 12; see `tests/test_baselines.py::test_ic_metric_bias_placebo`). Do not rank E[r] = ẑ·S either: that rewards ranking by volatility (a constant-mean forecast gets a non-zero IC).
- **Recency:** every metric is reported for the full period, for 2023–2025, and for the most recent 24 folds (the last 24 are for information only).
- **Recency check (decision D10)** — a gate passes the recency check only if **(a)** the full-period result passes with **t ≥ 2** and **(b)** over **2023–2025 (36 folds)** the mean IC is **> 0 with t ≥ 1**. For Sharpe-based gates, use the same rule on the per-fold net returns (full period t ≥ 2; 2023–2025 mean > 0 with t ≥ 1).
- **Execution delay:** every IC is also reported with one bar of delay (prediction at t vs realized z from t+1 to t+1+h).
- **Model comparisons (Phase 4 and later)** are paired by fold: share of folds won, Wilcoxon signed-rank test on fold ICs, and a Diebold–Mariano test on CRPS / QLIKE loss differentials with day-clustered (session-level) standard errors. Raw averages alone are not evidence.
- Hit rate and P(up) calibration. For reference, the 2025 study found best-model directional accuracy of only about 51–52% on daily returns; treat substantially higher out-of-sample accuracy on 5-minute TSLA data as a red flag to investigate for leakage.
- CRPS from the predicted bin distribution.
- Calibration: reliability diagram for P(up) and the barrier head.

### 10.7 Deflated Sharpe ratio

Use per-period (5-minute or daily-aggregated) Sharpe, not annualized:

```
PSR(SR*) = Φ( (SR̂ − SR*) · sqrt(T − 1) / sqrt(1 − γ3·SR̂ + (γ4 − 1)/4 · SR̂²) )
SR*      = sqrt(V) · ( (1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e)) ),   γ ≈ 0.5772
DSR      = PSR(SR*)
```

T = number of return observations, γ3 = skewness, γ4 = raw kurtosis, N = number of trials in the registry, V = variance of the trials' Sharpe ratios. Report DSR with every final result.

---

## 11. Required tests

- `test_no_lookahead_features`: compute all features on data truncated at a random time t, and on the full data; values at t must be identical. Repeat for 50 random t.
- `test_no_lookahead_tokens`: same for tokens, using bin edges and clusters fit on a fixed training period.
- `test_labels_alignment`: label at t uses only bars t+1 … t+h, and never crosses the session close.
- `test_purge_embargo`: no training sample's label window overlaps validation; embargo rows are removed.
- `test_split_adjustment`: no ±60%+ log-return jump on TSLA split dates.
- `test_tokenizer_roundtrip`: de-tokenized return correlation above a threshold (report the value).
- `test_causal_mask`: changing input at position t+1 does not change model output at position t.
- `test_dsr`: DSR equals PSR when N = 1; matches a hand-computed example.
- `test_hmm_filtered`: the regime probability at t is unchanged when data after t is deleted (fails if `predict_proba` is used).
- `test_garch_fixed_params`: GARCH σ at t is unchanged when data after t is deleted, given fixed trained parameters.
- `test_execution_delay`: in the backtest, the fill for a signal at bar t uses bar t+1's price; shifting all prices after t does not change the signal at t.
- `test_horizon_scale`: S_{t,h} equals level_t · sqrt(Σ seasonal²) over the right slots and never uses bars after t.
- `test_condprob`: on synthetic data where P(up) is known by construction per bucket (e.g. 0.45 / 0.50 / 0.55), the estimated p̂_k falls inside its Wilson interval; on pure noise, the Benjamini–Hochberg step flags roughly no cells.

---

## 12. Phases and gates

| Phase | Work | Gate (must pass, report numbers) |
| --- | --- | --- |
| 1. Data and baselines | Sections 4, 5.1–5.2, 10.1, 10.3 (zero/GARCH/LightGBM on simple returns), tests | Leakage and alignment tests pass; data report complete; baselines reproducible |
| 2. Features | Sections 5.3–5.5, 6 (including 6.9) | Feature report with IC tables and conditional up-probability tables, charts and pair heatmaps; number of cells surviving Benjamini–Hochberg reported; 20–40 features selected, each with a stated reason (stable IC and/or stable extreme lift) |
| 3. Tokenizer | Section 7 | Tokenizer checks pass; **token-only LightGBM shows out-of-sample IC > 0 (section 10.6) with a stable sign and passes the recency check (D10)** — otherwise stop and discuss with the user |
| 4. Model | Sections 8, 9 (including 9.4), 10.3 | Transformer beats LightGBM and CatBoost on IC and CRPS in **paired fold-level tests** (share of folds won, Wilcoxon on fold ICs, Diebold–Mariano on CRPS with day-clustered errors; report per fold); the Transformer itself passes the recency check (D10); volatility head beats GARCH-t QLIKE on identical rows, fold by fold, with Diebold–Mariano (D11); compared with fine-tuned Kronos-mini; ablations: without continuous features, without market stream, with/without self-supervised pre-training, 2 vs 4 layers, expanding vs rolling 3-year vs recency-weighted training window |
| 5. Backtest and selection | Section 10.2, 10.4, 10.6, 10.7 | At h = 12: net Sharpe positive at 2× costs and passing the recency check (D10: full period t ≥ 2; 2023–2025 mean > 0 with t ≥ 1); DSR > 0.95; break-even cost reported; then ask the user before evaluating the holdout once (and, later, the forward test once) |
| 6. Paper trading (only if the user asks) | Live inference script on new bars, logging predictions vs outcomes | Live IC over ≥ 3 months within the backtest range before any real capital |

Each phase ends with `reports/phaseN_*.md`: what was built, the gate numbers, failures, and the proposed next step.

---

## 13. Pitfalls checklist (re-check at every gate)

- [ ] Split-adjusted prices verified around Aug 2020 and Aug 2022
- [ ] Every fitted object (scaler, bins, k-means, HMM, GARCH, feature list) fit on training fold only
- [ ] σ̂ uses data up to t−1 only; time-of-day profile uses past sessions only
- [ ] HMM uses filtered probabilities, not smoothed; Kalman uses filtered states; GARCH parameters fixed after training (section 5.6)
- [ ] Label and position scales use S_{t,h} (time-of-day aware), not σ̂_t·√h
- [ ] Backtest fills at bar t+1, never at the close used to compute the signal; results also shown with one extra bar of delay
- [ ] Earnings flags placed on the correct session using the report time
- [ ] Labels never cross the session close; purge and embargo applied
- [ ] Overlapping labels down-weighted by uniqueness
- [ ] Conditional up-probability tables use n_eff = n / h and Benjamini–Hochberg; no "edge" claimed from a single uncorrected cell
- [ ] Targets are returns/volatility, never price levels
- [ ] Baselines reported next to every model result
- [ ] Costs charged; results also shown at 2× costs
- [ ] Every run logged to the registry
- [ ] Holdout untouched until the user approves; forward test not downloaded until the end
- [ ] No choice (early stopping, calibration, hyperparameters, features) used the test month; inner validation = last 2 training months
- [ ] IC computed per section 10.6 (z-space, pooled per fold), with t-stat, binomial p, 2023–2025 and last-24-fold results, and the one-bar-delay version
- [ ] Model comparisons are paired fold-level tests, not raw averages
- [ ] Neural training in FP32
- [ ] A suspiciously high Sharpe (e.g. > 3 on 5-minute single-stock data after costs) is treated as a bug until proven otherwise

---

## 14. Background reading (for context, not required to code)

- Kronos (financial K-line tokenizer, AAAI 2026): https://arxiv.org/abs/2508.02739
- Chronos (scaling + quantization tokenizer): https://arxiv.org/abs/2403.07815
- PatchTST: https://arxiv.org/abs/2211.14730
- Gu, Kelly & Xiu, Empirical Asset Pricing via ML: https://papers.ssrn.com/abstract=3159577
- Bailey & López de Prado, The Deflated Sharpe Ratio: https://papers.ssrn.com/abstract=2460551
- López de Prado, *Advances in Financial Machine Learning* (2018): ch. 3 labeling, 4 sample weights, 5 fractional differentiation, 7 purged CV, 8 feature importance, 12 CPCV
- Rahimikia, Ni & Wang, Re(Visiting) Time Series Foundation Models in Finance (2025): https://arxiv.org/abs/2511.18578
- Gao, Han, Li & Zhou, Market Intraday Momentum (JFE 2018), DOI 10.1016/j.jfineco.2018.05.009
- Heston, Korajczyk & Sadka, Intraday Patterns in the Cross-section of Stock Returns (bid-ask bounce): https://arxiv.org/abs/1005.3535
- Farebrother et al., Stop Regressing (HL-Gauss loss, 2024): https://arxiv.org/abs/2403.03950
- Kronos code, Kronos-mini and fine-tuning scripts: https://github.com/shiyu-coder/Kronos
- Alpha Vantage API documentation: https://www.alphavantage.co/documentation/
