# Phase 3 follow-up — pre-registered baseline confirmation at h = 12 (D24, D25)

Date 2026-10-26. Data v2, development period 2016-01..2025-12, 96 monthly out-of-sample folds 2018-01..2025-12.
Holdout (2026-01-01..2026-10-09) and forward test not loaded. D24 and D25 were recorded in `reports/decisions.md`
(and CLAUDE.md §10.3 item 7, §12) and committed (eae3cb5) **before any D25 model was trained**. Gate 3 stays failed
and was not re-run or tuned.

> **Approximation (applies to every model-null number below):** the mirror null uses the **last fold's (2025-12)
> fitted parameters for GARCH-t, HMM, Kalman and frac-diff in every fold**, and the real run's feature lists and
> versions (D23 construction). Null frames were rebuilt deterministically from seeds 3000..3019 (Gate 3 had not
> cached them).

**Result: one model passes. M3 (equal-weight composite of the D19 feature scores) passes the fixed D10 and the D22
model-null condition at h = 12 with the delayed outcome (t 4.33 vs null threshold 2.11; 2023–25 IC 0.0188, t 1.90).
M1, M2 and M4 fail.** Under the pre-registered rule, Phase 4 *may* proceed with M3 as the D4 baseline. **You decide**;
read the caveats in §8 first. In short, M3's pass largely re-states the Phase 2 single-feature results, it is not
significantly better than M1 or Gate-3 B, and its simple sign rule does not cover the estimated half-spread.

## 1. Decisions (verbatim summary of D24 / D25)

- **D24 (Phase 3 outcome):** Gate 3 failed (D23 result) and stays failed. The tokenizer passed all §7.4 checks and is
  kept as built. The tokens showed no h = 12 information (0 BH token cells at h = 12 in any fold; model A t 0.15) and
  made LightGBM worse when added to B.
- **D25 (baseline confirmation, pre-registered):** M1 = Phase 1 `lgb_returns` retrained in the D23 pipeline; M2 =
  ridge on the by_fold v2 features (inputs standardized on the fitting rows, clipped ±5, NaN → 0; α per fold from
  {0.1, 1, 10, 100, 1000} by D19 IC on the D6 inner block, ties → larger α, then refit on the full window); M3 =
  equal-weight mean of sign_train·(c(x) − mean_train c(x)) over the fold's by_fold v2 features with basis IC / both,
  scored with the D19 feature formula; M4 = CatBoost (300 iterations, depth 4, RMSE, seed 0, thread_count 1, no
  tuning). D23 rows, folds, purge + 78-bar embargo, D19 model scoring, first-fold V_s from the D6 block. K = 20 null
  draws per model (seeds 3000..3019). Pass = fixed D10 **and** D22 model null at h = 12 delayed. Full text including
  every implementation choice: `reports/decisions.md`, D25.

## 2. Results (D19 anchored IC v2)

Table: `reports/tables/phase3_baselines/baseline_results.csv` (fold ICs: `baseline_fold_ic.csv`; null:
`baseline_model_null_summary.csv`, per draw `baseline_model_null_per_draw.csv`). "thr" = null mean t + 2 null sd.

### 2.1 h = 12 (gate horizon)

| model | delay | IC full | t full | IC 23–25 | t 23–25 | fixed D10 | null mean IC | null mean t (sd) | thr | model null | **D25 pass** |
|---|---|---|---|---|---|---|---|---|---|---|---|
| M1 lgb_returns | **yes** | 0.0203 | 3.54 | 0.0078 | 0.95 | **fail** (23–25 t < 1) | −0.0009 | −0.19 (1.00) | 1.81 | pass | **no** |
| M1 | no | 0.0219 | 3.92 | 0.0098 | 1.22 | pass | −0.0007 | −0.17 (0.92) | 1.67 | pass | – |
| M2 ridge | **yes** | 0.0173 | 2.01 | 0.0031 | 0.30 | **fail** | −0.0014 | −0.20 (1.23) | 2.26 | **fail** | **no** |
| M2 | no | 0.0199 | 2.30 | 0.0050 | 0.48 | fail | −0.0014 | −0.19 (1.19) | 2.20 | pass | – |
| **M3 composite** | **yes** | **0.0294** | **4.33** | **0.0188** | **1.90** | **pass** | +0.0018 | 0.22 (0.94) | **2.11** | **pass** | **YES** |
| M3 | no | 0.0294 | 4.34 | 0.0220 | 2.13 | pass | +0.0018 | 0.23 (0.91) | 2.05 | pass | – |
| M4 CatBoost | **yes** | 0.0073 | 0.85 | −0.0016 | −0.14 | **fail** | +0.0002 | 0.04 (0.95) | 1.94 | **fail** | **no** |
| M4 | no | 0.0106 | 1.23 | 0.0009 | 0.08 | fail | +0.0004 | 0.08 (0.94) | 1.96 | fail | – |
| *Gate-3 B (ref.)* | *yes* | *0.0167* | *1.92* | *0.0071* | *0.58* | *fail* | *−0.0001* | *−0.01 (1.17)* | *2.34* | *fail* | *–* |

All null mean ICs are within ±0.0018 (limit ±0.003). Pooled null mean z_12 = 0.0024 (target 0.0019; D22 drift not
recalibrated, as in D23).

Other metrics at h = 12 delayed: share of folds with IC > 0 / binomial p: M1 69% / 0.0002, M2 50% / 0.54,
M3 61% / 0.016, M4 61% / 0.016. Last 24 folds (information): M1 IC 0.0031 t 0.30, M2 −0.0090 t −0.75,
M3 0.0092 t 0.81, M4 −0.0095 t −0.79. Old monthly Spearman (biased, information only) h = 12: M1 0.010, M2 0.009,
M3 0.002, M4 0.017. Hit rate h = 12: M1 0.505, M2 0.501, M3 0.500, M4 0.506.

Fold IC by year, h = 12 delayed:

| year | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| 2018 | 0.011 | 0.049 | 0.034 | 0.060 |
| 2019 | 0.044 | 0.031 | 0.029 | −0.042 |
| 2020 | 0.045 | 0.047 | 0.063 | 0.034 |
| 2021 | 0.014 | −0.009 | 0.012 | −0.008 |
| 2022 | 0.026 | 0.011 | 0.041 | 0.020 |
| 2023 | 0.017 | 0.027 | 0.038 | 0.014 |
| 2024 | 0.011 | −0.007 | −0.003 | 0.001 |
| 2025 | −0.005 | −0.011 | 0.021 | −0.020 |

### 2.2 h = 1 and h = 3 (information; no null)

| model | h | delay | IC full | t full | IC 23–25 | t 23–25 | fixed D10 |
|---|---|---|---|---|---|---|---|
| M1 | 1 | no / yes | 0.0239 / 0.0078 | 8.78 / 3.83 | 0.0104 / 0.0068 | 3.48 / 2.41 | pass / pass |
| M1 | 3 | no / yes | 0.0292 / 0.0192 | 9.94 / 6.52 | 0.0253 / 0.0190 | 5.90 / 4.39 | pass / pass |
| M2 | 1 | no / yes | 0.0220 / 0.0139 | 7.41 / 5.05 | 0.0046 / 0.0040 | 1.55 / 1.11 | pass / pass |
| M2 | 3 | no / yes | 0.0247 / 0.0156 | 5.49 / 3.70 | 0.0072 / 0.0044 | 1.26 / 0.72 | pass / fail |
| M3 | 1 | no / yes | 0.0155 / 0.0132 | 6.86 / 5.41 | 0.0066 / 0.0084 | 2.01 / 2.69 | pass / pass |
| M3 | 3 | no / yes | 0.0210 / 0.0220 | 5.22 / 5.44 | 0.0145 / 0.0139 | 2.54 / 2.54 | pass / pass |
| M4 | 1 | no / yes | 0.0311 / 0.0115 | 9.82 / 4.35 | 0.0133 / 0.0066 | 3.34 / 1.78 | pass / pass |
| M4 | 3 | no / yes | 0.0268 / 0.0172 | 6.24 / 4.28 | 0.0166 / 0.0102 | 2.88 / 2.01 | pass / pass |

## 3. CRPS vs the zero forecast (Gaussian, training residual sd; DM with day-clustered errors)

Same rows; zero forecast = mean 0, sd = training sd of clipped z. M3 is a score, not a z forecast: no CRPS.

| model | h = 1: CRPS / zero / DM t | h = 3 | h = 12 |
|---|---|---|---|
| M1 | 0.5606 / 0.5608 / −2.24 (p 0.03, better) | 0.5428 / 0.5427 / +0.74 | 0.5470 / 0.5463 / +1.98 (p 0.048, worse) |
| M2 | 0.5607 / 0.5608 / −1.17 | 0.5430 / 0.5427 / +2.38 (worse) | 0.5473 / 0.5463 / +2.57 (worse) |
| M4 | 0.5622 / 0.5608 / +5.94 (worse) | 0.5475 / 0.5427 / +5.79 (worse) | 0.5643 / 0.5463 / +10.30 (worse) |

No model beats the zero forecast on CRPS at h = 12; only M1 at h = 1 is (barely) better.

## 4. Paired fold tests (`baseline_paired.csv`; Wilcoxon signed-rank on 96 fold ICs)

| comparison | delay | mean IC (first / second) | diff | share won | Wilcoxon p two-sided / greater |
|---|---|---|---|---|---|
| **M3 vs M1** | **yes** | 0.0294 / 0.0203 | +0.0091 | 58% | 0.16 / **0.080** |
| M3 vs M1 | no | 0.0294 / 0.0219 | +0.0075 | 58% | 0.24 / 0.12 |
| **M3 vs Gate-3 B** | **yes** | 0.0294 / 0.0167 | +0.0127 | 56% | 0.15 / **0.075** |
| M3 vs Gate-3 B | no | 0.0294 / 0.0194 | +0.0100 | 54% | 0.27 / 0.13 |
| M2 vs M1 | yes | 0.0173 / 0.0203 | −0.0030 | 49% | 0.65 / 0.67 |
| M4 vs M1 | yes | 0.0073 / 0.0203 | −0.0129 | 45% | 0.18 / 0.91 |
| M1 vs Gate-3 B | yes | 0.0203 / 0.0167 | +0.0036 | 50% | 0.66 / 0.33 |
| M2 vs Gate-3 B | yes | 0.0173 / 0.0167 | +0.0006 | 50% | 0.93 / 0.54 |
| M4 vs Gate-3 B | yes | 0.0073 / 0.0167 | −0.0094 | 40% | 0.080 / 0.96 |

At h = 12 no model is significantly better than M1 or Gate-3 B at 5%. h = 1 / 3 rows are in the CSV (B and M4 lead
there; M1 beats M2/M3 at h = 1 without delay).

**M1 vs the stored Phase 1 predictions** (`m1_vs_phase1.csv`): same rows (132,012 at h = 12), correlation 0.931 /
0.945 / 0.959 (h = 1 / 3 / 12), mean |diff| 0.017 z, max 0.13 z; none identical (num_threads 1 vs 4 and the D23 row
set change the trees). Scores agree: stored h = 12 delayed t 3.56 vs retrained 3.54.

## 5. Family-wise check (information)

On each of the 20 null seeds, does any of M1–M4 pass fixed D10 and its own null threshold (h = 12 delayed)?
**1 of 20 seeds (5%)**: seed 3009, M3 (null t 2.58 ≥ 2.11, 2023–25 t 1.10). Per-model null false-pass: M1 0/20,
M2 0/20, M3 1/20, M4 0/20. With 20 draws, the 5% has a wide interval (binomial 95% CI ≈ 0.1–25%).

## 6. Break-even of a simple sign rule (information only, D25 (i)) next to the estimated spread

Rule: position = sign(ẑ) (M3: sign of s), enter at the next bar's open, exit at the close of bar t+h, trades
non-overlapping within the session, no entry in the last 10 minutes. Sharpe annualized from daily P&L.
`breakeven_rule_backtest.csv` (all costs, t-stats, extra-delay rows).

| model | h | period | trades/day | break-even bp/side | Sharpe gross | net 0.5 bp | net 1 bp | net 3 bp | net 6 bp | +1-bar delay break-even |
|---|---|---|---|---|---|---|---|---|---|---|
| Gate-3 B | 1 | 2018–25 | 74.7 | 0.45 | 3.67 | −0.39 | −4.45 | −20.6 | −44.6 | 0.14 |
| Gate-3 B | 1 | 2023–25 | 74.6 | 0.25 | 2.29 | −2.36 | −7.00 | −25.5 | −52.7 | 0.05 |
| Gate-3 B | 3 | 2018–25 | 24.9 | 0.85 | 2.33 | 0.96 | −0.42 | −5.90 | −14.1 | 0.42 |
| Gate-3 B | 3 | 2023–25 | 24.9 | 0.47 | 1.35 | −0.10 | −1.55 | −7.34 | −16.0 | 0.28 |
| Gate-3 B | 12 | 2018–25 | 6.0 | **1.41** | 0.96 | 0.62 | 0.28 | −1.08 | −3.11 | 1.33 |
| Gate-3 B | 12 | 2023–25 | 6.0 | **1.33** | 0.98 | 0.61 | 0.25 | −1.23 | −3.43 | 1.30 |
| M3 | 1 | 2018–25 | 74.7 | 0.11 | 0.98 | −3.44 | −7.86 | −25.5 | −51.5 | 0.06 |
| M3 | 3 | 2018–25 | 24.9 | 0.01 | 0.04 | −1.31 | −2.66 | −8.05 | −16.1 | 0.10 |
| M3 | 12 | 2018–25 | 6.0 | **0.39** | 0.25 | −0.07 | −0.39 | −1.67 | −3.58 | 0.39 |
| M3 | 12 | 2023–25 | 6.0 | **0.31** | 0.22 | −0.13 | −0.48 | −1.88 | −3.97 | 0.27 |

TSLA effective spread per year from 1-minute regular-session bars (`spread_by_year.csv`; daily estimates, yearly
mean; EDGE = Ardia–Guidotti–Kroencke via `bidask` 2.1.0, CS = Corwin–Schultz 2-bar, gap-adjusted, negatives → 0):

| year | median price | EDGE full bp | **EDGE half bp** | CS full bp | CS half bp | 1 tick in bp |
|---|---|---|---|---|---|---|
| 2016 | 209 | 4.28 | 2.14 | 3.74 | 1.87 | 0.48 |
| 2017 | 318 | 2.75 | 1.37 | 2.98 | 1.49 | 0.31 |
| 2018 | 318 | 3.28 | 1.64 | 4.47 | 2.24 | 0.31 |
| 2019 | 261 | 2.37 | 1.19 | 3.47 | 1.73 | 0.38 |
| 2020 | 660 | 3.28 | 1.64 | 7.21 | 3.61 | 0.15 |
| 2021 | 728 | 1.96 | 0.98 | 4.77 | 2.38 | 0.14 |
| 2022 | 736 | 2.03 | 1.01 | 6.13 | 3.07 | 0.14 |
| 2023 | 222 | 1.01 | **0.51** | 4.83 | 2.42 | 0.45 |
| 2024 | 211 | 1.26 | **0.63** | 4.84 | 2.42 | 0.47 |
| 2025 | 345 | 1.30 | **0.65** | 5.56 | 2.78 | 0.29 |

(2016–2022-08 prices are unadjusted, so "median price" mixes pre- and post-split levels in 2020 and 2022; spreads in
bp are not affected.) CS is 2–5× EDGE from 2020 on and rises with volatility, a known upward bias of CS on
high-volatility intraday data; EDGE is the primary estimate.

**Reading.** Against the 2023–25 EDGE half-spread of about 0.5–0.65 bp (before commission, slippage and impact):
M3's sign rule at h = 12 breaks even at 0.31 bp per side, **below the spread alone**. Gate-3 B at h = 12 breaks even at
1.33 bp per side, about 2× the half-spread, but below the default 3 bp per side; it is also stable under the extra
bar of delay. h = 1 and h = 3 rules do not survive realistic costs. Note that M3 has the highest IC but the weakest
sign-rule P&L. D19 weights each row by |s|·|z| in σ units, while the sign rule takes equal-size positions in return
space. The composite's information sits in its large scores, not in the sign of the typical bar.

## 7. Timing, budget, registry

- Real walk-forward: projected ~17–20 min with 5 workers (last fold 72 s, first 23 s); actual ~20 min wall
  (65 min of fold compute: M1 21, M2 7, M3 2, M4 34 min).
- Model null: the timed draw (seed 3000) took M1 76 s, M2 39 s, M3 14 s, M4 128 s, plus 18 s for the frame, so
  275 s per seed. 20 × 275 s = 1.53 h ≤ 10 h, so **K = 20 for all four models, none "unconfirmed"**. Draws 3001..3019 ran
  22:55–00:20 (≈ 1.5 h).
- M2 α per fold (h = 12): 1000 in 64 folds, 0.1 in 28, 100 in 4 (inner ICs differ only in the 3rd–4th digit; the
  grid is weak shrinkage at 0.2–1.6M rows, as noted in D25). M3 used 12–21 IC/both features per fold (median 17).
- Registry: 12 model trials `d25_M{1..4}_h{1,3,12}` and 6 rule-backtest trials `d25_rule_{Gate3B,M3}_h{1,3,12}`.
  **n_trials_so_far = 36.** Null draws are diagnostics, not trials.
- Tests: `tests/test_baselines_d25.py` (rule backtest fills at t+1 open / exits at t+h close, non-overlap, session
  limits, trade list unchanged when future prices change) passes.

## 8. Decision-rule outcome and caveats

**D25 rule:** at least one model passes (M3 only), so **Phase 4 may proceed with M3 as the D4 baseline**, i.e. the
passing model with the highest full-period t at h = 12 delayed. The decision is yours. Before deciding, weigh:

1. **M3 is not independent confirmation.** Its inputs are the per-fold by_fold v2 lists (training-only selection),
   but the research process has already seen Phase 2's full-period out-of-sample single-feature results on these same
   96 folds (20 features pass D10\* at h = 12 with or without the delay, and 17 of them appear in M3's
   per-fold lists, e.g. vwap_dev and ema_dev_78 in all 96 folds, bb_pctb, linreg_slope_78, mom_78, mom_3, r_first30). An equal-weight average of them passing is close to a restatement of that finding. The model null
   controls the metric, not this reuse of the evaluation period. Only the sealed holdout and forward test can confirm it.
2. **The margin is thin on the recency part.** 2023–25 t 1.90 (needs ≥ 1), but 2024 IC −0.003 and the last 24 folds t
   0.81. One of 20 null seeds also passes for M3 (5%).
3. **No model is significantly better than M1 or Gate-3 B at h = 12** (M3 vs M1 one-sided p 0.080, vs B 0.075).
4. **No economic edge yet.** M3's sign rule breaks even at 0.31 bp per side, below the estimated half-spread. Gate-3 B
   breaks even at about 1.3 bp, which is above the half-spread but below the 3 bp default cost. No model beats the
   zero forecast on CRPS at h = 12, and M3 has no distribution forecast, so Phase 4's CRPS comparison would need the
   zero forecast / M1 / M4 as references.
5. The model null uses the last-fold fitted-parameter approximation (stated above).

**My recommendation.** The pre-registered rule allows Phase 4, but the evidence for a *tradable* h = 12 edge is weak.
If you proceed, a Transformer would have to beat M3 on paired fold ICs (D4) and also show a sign / position rule that
clears realistic costs (§10.4, Phase 5). A cheaper option is to freeze M3 (and Gate-3 B) as they are and spend the
single holdout evaluation on them before building Phase 4. That needs your approval (D7) and uses up the holdout. The
other option is to conclude with a final report. Nothing has been started.

## 9. Files

Code: `src/tslaq/models/baselines_d25.py`, `scripts/run_phase3_baselines.py` (`--real`, `--null M.. --draws a b`,
`--summarize --log`, `--backtest --log`), `scripts/phase3_spread.py`, `tests/test_baselines_d25.py`. Tables:
`reports/tables/phase3_baselines/`. Checkpoints (gitignored): `data/processed/phase3_baselines/`.
