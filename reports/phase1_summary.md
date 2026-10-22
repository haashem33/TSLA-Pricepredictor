# Phase 1 — Data and baselines: summary and gate

Detailed data-quality tables: [`phase1_data_quality.md`](phase1_data_quality.md) (data v2) and [`phase1_data_quality_v1.md`](phase1_data_quality_v1.md). Raw metric tables: `reports/tables/phase1_*_{v1,v2}.csv`. Decisions: [`decisions.md`](decisions.md).

> **Current data version: v2 (decision D9, 2026-10-22).** TSLA 2016-01..2022-08 is rebuilt from unadjusted prices because the vendor's split-adjusted prices were rounded to $0.01. The original Phase 1 results below are **v1**. The v2 re-run and before/after numbers are in the section [Data v2 re-run](#data-v2-re-run-2026-10-22) at the end. All later phases use v2.

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

---

## Data v2 re-run (2026-10-22)

**What changed (D9).**
- Downloaded 80 months of TSLA 1-minute SIP bars with `Adjustment.RAW` (2016-01..2022-08) plus 2 check months, 42 MB, to `data/raw/TSLA_raw/`. Keys are in the gitignored `.env`.
- Built 5-minute bars from raw prices, then divided by the cumulative split factor (15 before 2020-08-31, 3 until 2022-08-24) in floating point; volume × factor.
- From 2022-09 the vendor files are used as before. SPY/QQQ are unchanged.

**Checks**

| check | result |
|---|---|
| Post-split months rebuilt from RAW vs current bars (2022-10: 1,638 bars; 2023-03: 1,794 bars) | **exact match**: 0 mismatches in timestamp, OHLC, volume, trade_count, vwap |
| 1-min RAW vs vendor file, 2022-10 | identical (18,468 rows, all columns) |
| Split day 2020-08-31: v2 gap vs explicit ln(O_raw / (C_prev_raw / 5)) = ln(444.55 / (2213.52/5)) | 0.004161 = 0.004161; bar-0 log return 0.009635 (v1: 0.009644) |
| Split day 2022-08-25: v2 gap vs explicit ln(302.38 / (891.45/3)) | 0.017447 = 0.017447; bar-0 log return 0.004312 (v1: 0.004312) |
| v2 vs v1 close before 2022-08-25 | median \|diff\| 0.57 bp, p99 3.4 bp, max 4.9 bp (= the removed rounding) |
| Sessions | TSLA keeps 2,512 (v1: 2,510): the RAW download has 2018-05-02/03, which were missing from the split-adjusted download. 2016-02-22 and the 2018-08-07 halt are still dropped. |
| Tests | **26/26 pass** on v2. New: split factor, post-split exact match (×2), split-day bar-0 explicit (×2), v2 not rounded and close to v1. |

**B1 before/after (TSLA, intraday 5-min bars).** Cent-multiple share = 1-min regular-session closes in the series each version uses.

| year | exact-zero r v1 → v2 | AC(1) v1 → v2 | cent-multiple v1 → v2 |
|---|---|---|---|
| 2016 | 18.1% → 0.79% | +0.002 → +0.007 | 99.1% → 6.4% |
| 2017 | 14.5% → 0.44% | −0.033 → −0.029 | 100% → 5.8% |
| 2018 | 9.3% → 0.27% | −0.018 → −0.018 | 100% → 5.8% |
| 2019 | 13.8% → 0.45% | −0.030 → −0.026 | 100% → 6.0% |
| 2020 | 2.2% → 0.09% | −0.036 → −0.037 | 100% → 9.4% |
| 2021 | 0.94% → 0.11% | −0.005 → −0.005 | 100% → 19.7% |
| 2022 | 0.44% → 0.13% | −0.020 → −0.020 | 84.9% → 33.0% |
| 2023–2025 | unchanged (0.30%, 0.34%, 0.17%) | unchanged | unchanged (48–54%) |

SPY/QQQ are identical in v1 and v2. As predicted in `phase1_checks.md`, rounding drove the exact zeros but only a small part of the negative AC(1).

**λ grid (2016-01..2017-12, QLIKE of σ̂² vs r²):** v2 −11.679 / −11.698 / −11.707 / **−11.709** / −11.703 / −11.687 / −11.669 for λ = 0.90 / 0.93 / 0.95 / **0.97** / 0.98 / 0.99 / 0.995. λ = 0.97 again (v1 best was also 0.97, at −11.663).

**Baselines v1 → v2 (96 folds; IC = z-space fold IC; v2 prediction digest `8de32dc7a7df`).**

| h | LightGBM IC | ICIR | folds IC>0 | slot IC | hit rate | CRPS (z) LGB / zero |
|---|---|---|---|---|---|---|
| 1 | 0.0248 → 0.0247 | 0.81 → 0.86 | 81% → 86% | 0.023 → 0.024 | 0.507 → 0.509 | v2: 0.5606 / 0.5608 |
| 3 | 0.0204 → 0.0215 | 0.61 → 0.66 | 69% → 74% | 0.019 → 0.021 | 0.504 → 0.504 | v2: 0.5427 / 0.5427 |
| 12 | 0.0107 → 0.0106 | 0.17 → 0.17 | 59% → 61% | 0.019 → 0.020 | 0.504 → 0.505 | v2: 0.5469 / 0.5463 |

| h | QLIKE v1 → v2: seasonal | EWMA × seasonal | GARCH-t × seasonal |
|---|---|---|---|
| 1 | 1.621 → 1.752 | 1.324 → 1.449 | 1.307 → **1.435** |
| 3 | 0.822 → 0.826 | 0.530 → 0.532 | 0.521 → **0.522** |
| 12 | 0.512 → 0.515 | 0.252 → 0.253 | 0.248 → **0.248** |

The h = 1 QLIKE increase is mostly a sample effect, not a worse forecast. QLIKE drops rows with RV = 0: v1 had 3.5% such rows at h = 1 (artificial zeros), v2 has 0.3%. The newly included rows have tiny RV, which QLIKE penalizes heavily (GARCH-t QLIKE 4.68 on them). On the 148,590 rows with RV > 0 in both versions, GARCH-t h = 1 is 1.307 (v1) vs 1.326 (v2); h = 3 and h = 12 are equal. GARCH-t still beats EWMA at every horizon. **The volatility-head target is now the v2 numbers 1.435 / 0.522 / 0.248** (D9).

**B2 delay table v1 → v2 (LightGBM, fold IC in z space; D10 status for h = 12).**

| h | variant | full IC (t) v1 → v2 | 2023–25 IC (t) v1 → v2 | last 24 IC (t) v1 → v2 |
|---|---|---|---|---|
| 1 | no delay | 0.0240 (7.8) → 0.0239 (8.1) | 0.0096 (2.3) → 0.0133 (3.2) | 0.0055 (1.1) → 0.0081 (1.5) |
| 1 | 1-bar delay | 0.0126 (6.1) → 0.0085 (3.8) | 0.0071 (2.1) → 0.0049 (1.4) | 0.0069 (1.6) → 0.0031 (0.8) |
| 3 | no delay | 0.0189 (5.6) → 0.0198 (6.0) | 0.0084 (1.6) → 0.0118 (2.3) | 0.0039 (0.6) → 0.0075 (1.1) |
| 3 | 1-bar delay | 0.0134 (4.2) → 0.0141 (4.5) | 0.0093 (1.9) → 0.0095 (1.9) | 0.0070 (1.2) → 0.0062 (1.0) |
| 12 | no delay | 0.0098 (1.5) → 0.0103 (1.6) | −0.0047 (−0.4) → −0.0023 (−0.2) | −0.0030 (−0.2) → 0.0006 (0.0) |
| 12 | 1-bar delay | 0.0104 (1.6) → 0.0109 (1.7) | −0.0038 (−0.3) → −0.0015 (−0.1) | −0.0019 (−0.1) → 0.0013 (0.1) |

(Same rows for both variants: t+1+h within the session. v2 by year is in `reports/tables/check_b2_v2.csv`.)

**Reading.**
- On v2, one bar of delay cuts the h = 1 IC by about two thirds (0.024 → 0.0085), more than on v1. The h = 1 baseline is mostly next-bar bounce.
- At h = 12, the primary horizon, delay changes nothing. The lagged-returns baseline **fails the D10 recency check**: full-period t = 1.6 < 2, and 2023–2025 mean IC is −0.002. This is expected of a returns-only baseline and is not a gate for Phase 1.

**Registry.** 6 v2 rows were added with `data_version = v2` and `rerun_of = "phase1 v1 baselines (same model/config)"`. `n_trials_so_far` stays **6**: re-runs do not count as trials. New columns `data_version` and `rerun_of` were added; the v1 rows are marked `v1`.
