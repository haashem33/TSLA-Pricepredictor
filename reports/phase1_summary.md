# Phase 1 — Data and baselines: summary and gate

Detailed data-quality tables: [`phase1_data_quality.md`](phase1_data_quality.md). Raw metric tables: `reports/tables/phase1_*.csv`.

## What was built

| Area | Module | Notes |
|---|---|---|
| Raw cache | `src/tslaq/data/download.py` | Source `local_alpaca` copies the user's Alpaca SIP 1-min files to `data/raw/<SYM>/<YYYY-MM>.parquet` (360 files, 185 MB), logged in `data/raw/download_log.csv`. The Alpha Vantage month-by-month downloader (12 s spacing, stops on `Note`/`Information`, resumable) is kept behind the same interface but **not used**: no API intraday calls were needed. |
| Holdout guard | `download.py`, `configs/data.yaml` | Holdout = **2026-01-01 → 2026-10-09** (the last ~9 months before the source's sealed end). Months ≥ 2026-01 are never copied or loaded (`PermissionError`; tested). |
| Calendar | `data/calendar.py` | NYSE schedule via `pandas_market_calendars`; half-days = 42 bars. |
| Cleaning | `data/clean.py` | 1-min → 5-min, regular session, bar-end NY timestamps; missing bars ffilled + flagged; sessions with > 5% missing dropped; bad ticks flagged; SPY/QQQ dividend back-adjustment (Alpha Vantage DIVIDENDS); daily bars built from 5-min bars; earnings → reaction session. |
| Returns | `features/returns.py` | `r`, `r_intraday`, `gap`. |
| σ̂ and S_{t,h} | `features/volatility.py` | 60-session seasonal median \|r\| per slot (past sessions only, half-day safe), EWMA level, σ̂_{t\|t−1}, S_{t,h} from the calendar's session length. λ tuned on 2016-01…2017-12 only. |
| Labels | `labels/forward_returns.py` | fwd log return, z = fwd/S, log RV, never crossing the close. |
| CV | `cv/walk_forward.py`, `cv/purge.py` | Expanding monthly walk-forward, 96 folds (val 2018-01 … 2025-12), purge + 78-bar embargo. |
| Baselines | `models/baselines.py`, `features/stats_models.py`, `scripts/run_phase1_baselines.py` | Zero, historical mean, LightGBM on past vol-scaled returns (12 lags, 24/78-bar sums, bar_index); seasonal-only, EWMA×seasonal, GARCH(1,1)-t×seasonal (fit per fold, fixed-parameter forward recursion). |
| Registry | `utils/registry.py` → `experiments/registry.csv` | 6 rows (n_trials = 6). |

## Gate 1 results

### 1. Leakage and alignment tests — **PASS (19/19)**

`test_no_lookahead_features` (50 random truncation points, synthetic data, plus 10 on real TSLA 2020), perturbation test, `test_horizon_scale`, seasonal-uses-past-only, `test_garch_fixed_params`, `test_labels_alignment`, labels-use-only-t..t+h, `test_purge_embargo` (generic + walk-forward), `test_split_adjustment`, aggregation/half-day/holiday, missing-bar fill, dividend adjustment, holdout guard, interim-has-no-holdout, baseline-feature no-lookahead, LightGBM reproducibility, IC-metric placebo.

The tests caught two real bugs, both fixed: (a) any half-day in the trailing 60 sessions blanked the afternoon seasonal slots, so σ̂ coverage was 154k instead of 190k bars; (b) the burn-in cutoff and session length depended on how much data existed after t.

### 2. Data report — **complete**

- TSLA 195,024 bars / 2,510 sessions, 2016-01-04 → 2025-12-31 (SPY 2,513, QQQ 2,511 sessions).
- Dropped sessions: TSLA 4. Three are vendor gaps (2016-02-22, 2018-05-02, 2018-05-03, also missing in QQQ). The fourth, **2018-08-07, was a real trading halt**. 9 missing bars ffilled per symbol.
- Bad ticks flagged: TSLA 84 (8 spike-and-revert). The largest are real events (earnings gaps, 2024-11-06, 2025-04-07), so none were deleted.
- Splits: bar-0 log return +0.96% on 2020-08-31 and +0.43% on 2022-08-25 (an unadjusted split would show −161% / −110%).
- Dividends: the source is not dividend-adjusted (SPY ex-date gap −0.45% before vs −0.03% after adjustment).
- λ: QLIKE-best = 0.97 on the training window (grid 0.90–0.995). z = fwd/S has std 1.03 / 0.99 / 1.00 at h = 1/3/12, so it is well calibrated.

### 3. Baselines — **reproducible**

Two full 96-fold runs gave the identical prediction digest `77ff73e7a68b`.

**Return forecasts** (out-of-sample, 96 monthly folds; IC = Spearman of predicted vs realized z, pooled within each fold):

| h | model | fold IC mean | ICIR (folds) | folds IC>0 | slot IC | hit rate | CRPS (z) | Brier |
|---|---|---|---|---|---|---|---|---|
| 1 | zero | – | – | – | – | – | 0.5606 | 0.2500 |
| 1 | hist_mean | – | – | – | – | 0.503 | 0.5606 | 0.2500 |
| 1 | LightGBM | **0.0248** | 0.81 | 81% | 0.0233 | 0.507 | 0.5605 | 0.2499 |
| 3 | zero | – | – | – | – | – | 0.5406 | 0.2500 |
| 3 | LightGBM | 0.0204 | 0.61 | 69% | 0.0188 | 0.504 | 0.5407 | 0.2503 |
| 12 | zero | – | – | – | – | – | 0.5431 | 0.2500 |
| 12 | LightGBM | 0.0107 | 0.17 | 59% | 0.0191 | 0.504 | 0.5437 | 0.2506 |

LightGBM fold-IC by year (h=1): 2018 .022, 2019 .025, 2020 .055, 2021 .033, 2022 .032, 2023 .019, **2024 .009, 2025 .003**. The signal is decaying. At h = 3 and h = 12 it is about 0 or negative in 2023–2025. LightGBM does **not** beat the zero forecast on CRPS or Brier at any horizon. The h = 1 IC may be partly bid–ask bounce, so it must survive the one-bar execution-delay test before it counts.

**Volatility forecasts** (QLIKE, lower is better; target Σ r² over the next h bars):

| h | seasonal only | EWMA × seasonal | GARCH-t × seasonal |
|---|---|---|---|
| 1 | 1.621 | 1.324 | **1.307** |
| 3 | 0.822 | 0.530 | **0.521** |
| 12 | 0.512 | 0.252 | **0.248** |

GARCH-t edges out EWMA in every year. Fitted parameters are stable across folds: α 0.057–0.068, β 0.920–0.932, ν 5.8–6.6.

## Failures and issues found

1. **The IC metric in CLAUDE.md §10.6 ("Spearman per day, then averaged") is biased for a single stock.** Within one day it ranks bars over time, and the overlapping forward windows create a mechanical negative correlation with any path-dependent predictor. On a pure random walk, a 78-bar momentum feature scores daily IC = −0.31 at h = 12 (placebo test in `tests/test_baselines.py`). The LightGBM "daily IC" was −0.12 at h = 12, an artifact. I also found that IC on E[r] = ẑ·S rewards plain volatility ranking (a constant-mean model gets a nonzero IC). **Proposed replacement:** IC in z-space, pooled per monthly fold (ICIR = mean/std across folds), with per-time-slot IC (across days) as a cross-check. The registry rows use this; the biased per-day number is kept in the CSVs as `daily_ic(biased)`.
2. The registry's 6 Phase 1 rows were re-scored with the corrected metric on the same predictions; no extra trials were added.
3. The Phase 1 baseline uses only lagged returns. CatBoost and logistic regression, and LightGBM on the selected features, come in Phase 2/4 (§10.3).

## Notes for you

- No paid data was added. Alpha Vantage was used only for DIVIDENDS/SPLITS (3 calls).
- **The holdout may not be truly unseen.** The data README says the earlier `tsla-diffusion` project kept only days *after 2026-10-09* unseen, so 2026-01 → 2026-10-09 may already have been looked at in that project. If so, the only clean final test is data after 2026-10-09 (which would need downloading later).
- Git identity was set repo-locally (`haash` / your account email) to make commits.

## Proposed next step

Phase 2 (features, IC tables, conditional up-probability analysis), after you decide on the IC metric definition.
