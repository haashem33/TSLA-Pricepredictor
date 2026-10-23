# Phase 2 follow-up — diagnostics of the daily signal

Requested 2026-10-23 after the Gate 2 review. Scripts: `scripts/phase2_daily_diagnostics.py` (B1–B4) and
`scripts/phase2_daily_placebo.py` (placebo, 2018 check). Tables: `reports/tables/phase2_daily/`. Figure:
`reports/figures/phase2_daily_slot_ic_h12.png`. Same out-of-sample convention as Phase 2: 96 monthly folds
2018-01..2025-12, the sign in month M is the sign of the mean monthly IC over all months before M, raw feature
versions. Holdout not loaded.

## Main finding: the within-month IC of the daily features is a mechanical artefact

A session-permutation placebo removes all real day-to-day predictability. It shuffles whole sessions, keeping
each session's intraday path and its gap, high and low relative to the previous close. Prices are then rebuilt
and the daily features recomputed. On the shuffled data the 10.6 IC (Spearman pooled within each calendar
month) is **as large as on the real data**:

| feature | target | window | real IC (unsigned) | placebo mean (200 perms) | placebo 5–95% | z vs placebo |
|---|---|---|---|---|---|---|
| d_sma20_dev | z_12, bar level | month | −0.062 | −0.063 | −0.080 … −0.048 | 0.09 |
| d_mom5 | z_12, bar level | month | −0.045 | −0.036 | −0.052 … −0.021 | −0.97 |
| d_atr_c | z_12, bar level | month | +0.051 | +0.044 | +0.030 … +0.059 | 0.72 |
| d_sma20_dev | close-to-close (C_{D−1}→C_D) | month | −0.209 | −0.224 | −0.259 … −0.195 | 0.71 |
| d_mom5 | close-to-close | month | −0.137 | −0.131 | −0.167 … −0.098 | −0.32 |
| d_atr_c | close-to-close | month | +0.180 | +0.150 | +0.117 … +0.188 | 1.32 |
| d_sma20_dev | close-to-close | year | +0.015 | −0.024 | −0.059 … +0.010 | 1.89 |
| d_sma20_dev | close-to-close | all 2018–2025 | +0.043 | −0.003 | −0.036 … +0.027 | 2.31 |
| d_mom5 | close-to-close | all | +0.017 | −0.003 | −0.040 … +0.029 | 0.93 |
| d_atr_c | close-to-close | all | −0.022 | +0.004 | −0.029 … +0.040 | −1.25 |

(`placebo_session_permutation.csv`.) Mechanism: these features behave like a lagged price level (d_sma20_dev and
d_mom5 rise with price; d_atr_c = ATR / C falls with price). They are persistent over about a month. Spearman
within a 21-session month demeans both sides within the month, so a feature that tracks the price level gets a
negative correlation with the month's own returns (positive for 1/C-type features), even with no predictability.
This is the same bias D1 found for the per-day IC (78-bar momentum ≈ −0.31 on a random walk), one level up. It
is a property of the metric, not a data leak (B1).

Consequences:

- The Phase 2 single-feature D10 passes of **d_sma20_dev, d_mom5 and d_atr_c at h = 12 are explained by this
  bias** (real IC inside the placebo 5–95% band). **amihud** (log mean of |r| / (C·V)) also contains 1/C and
  shows the same pattern, but it was not put through the placebo, because permuting volume is not set up.
  Treat it as suspect, not cleared.
- Once the within-month demeaning is removed, nothing significant remains. Over the whole period in one window,
  real d_sma20_dev vs the next close-to-close return is +0.043 (z 2.3 against the placebo). That is
  **momentum**, the opposite sign of the "reversal" the monthly IC suggested. It is the strongest leftover, it
  is a single number among several tested, and it is not confirmed by the rule P&L below.
- The same bias can affect **any model prediction with a slow (multi-day) component** scored by the 10.6 IC,
  which includes Gate 3 model B/C (by_fold lists contain d_* features) and Phase 4. This needs a decision; see
  the end.

## B1. Leak check of the daily-context features

- **Truncation rebuild on real data:** raw TSLA 5-minute bars were truncated at 20 random intraday times
  (2018–2025, bar_index 1–75). Returns, σ̂, daily context, event flags and amihud were rebuilt from the
  truncated bars, and the last 30 sessions up to t were compared. **All 20 are identical** (max relative
  difference 0.0) for d_prev_ret_atr, d_atr_c, d_sma20_dev, d_mom5, har_rv1/5/22, gap_z, rv_today_deseas,
  amihud, earn_react, earn_after, opex, half_day, dow (`b1_truncation_rebuild.csv`).
- **Constant within the session:** every d_*, har_* and event flag has exactly one value per session.
- **Equal to a computation from sessions ≤ D−1 only:** in 300 random sessions, an independent loop
  implementation (daily bars from sessions before D, true range and Wilder ATR written out explicitly) matches
  to ≤ 7.5e-16 relative.
- **amihud is not a daily feature:** it is a rolling 78-bar mean of |r| / (C·V), updated every bar. It passed
  the same truncation check. The earnings flags come from the Phase 1 reaction-session table
  (reportedDate + reportTime). They are constant within the session and depend only on the session date.
- **The existing `test_no_lookahead_features` did rebuild the daily bars from truncated data.** `build_features`
  calls `daily_context`, which aggregates the (truncated) 5-minute frame itself, so no precomputed daily table
  is used. The returns and σ̂ columns of the synthetic test frame were computed once on the full series and
  then truncated; their causality is covered by the Phase 1 tests. The new real-data check rebuilds those too.

## B2. Where the effect happens (10.6-style monthly IC; read with the placebo above)

Signed out-of-sample IC, 96 folds (`b2_daily_target_ic.csv`):

| feature | open-to-close | t | close-to-close (D) | t | next session C_D→C_{D+1} | t | z_12 bar | t |
|---|---|---|---|---|---|---|---|---|
| d_sma20_dev | 0.177 | 8.5 | 0.209 | 10.1 | 0.166 | 7.5 | 0.062 | 6.8 |
| d_mom5 | 0.106 | 5.2 | 0.137 | 6.4 | 0.097 | 4.6 | 0.045 | 5.2 |
| d_atr_c | 0.150 | 6.9 | 0.180 | 7.5 | 0.130 | 6.0 | 0.051 | 5.6 |
| amihud | 0.106 | 5.1 | 0.107 | 5.1 | 0.093 | 4.2 | 0.045 | 5.2 |

2023–2025 values are similar (0.08–0.20, t 2.3–6.3). Every year is positive for every feature and target except one cell
(amihud next-session 2019, −0.01; by-year table in the CSV). The next-session IC is as large as the same-session IC, which a real one-day
effect would not produce but the window bias does. Scales: open-to-close is divided by
level(before bar 0) × √(Σ seasonal² over bars 1..n−1), so the intraday part of bar 0 is not in the scale.
Close-to-close uses bars 0..n−1. The amihud value is the previous session's last bar, known at the open.

**Per-time-slot profile at h = 12** (`b2_slot_ic_h12.csv`, figure above). The signed IC of d_sma20_dev is
highest for forecasts made at 09:30–10:00 (0.103, t 5.0). It falls through the morning (0.04–0.06), peaks again
at 13:00 (0.092), and is ≈ 0 or negative after 14:00 in 2023–2025 (−0.07 at 14:30). d_mom5 and d_atr_c follow
the same profile. These are 10.6-style ICs, so they carry the same bias.

## B3. TSLA-specific or market-wide (`b3_residual_qqq_ic.csv`)

| feature | TSLA z_12 (same rows) | TSLA residual z_12 (minus β·QQQ) | QQQ z_12 with TSLA's feature | QQQ's own feature vs QQQ z_12 |
|---|---|---|---|---|
| d_sma20_dev | 0.062 (t 6.8) | 0.055 (t 5.7) | 0.030 (t 3.1) | 0.060 (t 8.3) |
| d_mom5 | 0.045 (5.2) | 0.040 (4.4) | 0.020 (2.2) | 0.036 (4.4) |
| d_atr_c | 0.051 (5.5) | 0.024 (2.6) | 0.048 (5.0) | 0.054 (6.0) |
| amihud | 0.046 (5.2) | 0.035 (3.8) | 0.016 (1.9) | 0.032 (4.2) |

The "effect" survives removing the market and appears just as strongly in QQQ with QQQ's own features. That is
what the window bias predicts for any price-level-like feature of any asset. It does not point to a
TSLA-specific signal.

## B4. Single-feature rule baselines (D16), logged as trials 8 and 9 in the registry

Unit position, sign from each fold's training h = 12 IC (d_sma20_dev −1 in all 96 folds; sign_first30 +1 in 90,
−1 in 6), zero threshold. Costs are per side; a round trip is charged on every traded day. Daily returns,
annualized √252. Fold t is computed on the monthly net returns (D10). Tables: `b4_rule_*.csv`, `_delay1.csv`,
`_daily.csv`.

**(a) d_sma20_dev:** short if above the 20-day mean, else long; fill at the open of the 09:35 bar; exit at the
close of the last bar.

| period | days | gross bp/day | gross SR | SR 0.5bp | SR 1bp | SR 3bp | SR 6bp | break-even bp/side |
|---|---|---|---|---|---|---|---|---|
| 2018–2025 | 2010 | 7.9 | 0.42 | 0.37 | 0.32 | 0.10 | −0.22 | 3.97 |
| 2023–2025 | 752 | 6.3 | 0.36 | 0.30 | 0.24 | 0.02 | −0.33 | 3.13 |
| 2018 | 250 | 58.3 | 3.48 | 3.42 | 3.36 | **3.12** | 2.76 | 29.1 |
| 2019 | 252 | 7.7 | 0.57 | 0.50 | 0.42 | 0.12 | −0.32 | 3.84 |
| 2020 | 253 | −26.5 | −1.08 | −1.12 | −1.16 | −1.32 | −1.57 | — |
| 2021 | 252 | 12.1 | 0.67 | 0.62 | 0.56 | 0.34 | 0.01 | 6.06 |
| 2022 | 251 | −6.5 | −0.29 | −0.34 | −0.39 | −0.57 | −0.84 | — |
| 2023 | 250 | 6.2 | 0.37 | 0.31 | 0.25 | 0.01 | −0.34 | 3.10 |
| 2024 | 252 | 12.2 | 0.78 | 0.72 | 0.65 | 0.40 | 0.01 | 6.08 |
| 2025 | 250 | 0.4 | 0.02 | −0.03 | −0.08 | −0.28 | −0.58 | 0.18 |

D10 on fold net returns at 3 bp: full t 0.33; 2023–25 mean +10 bp/month, t 0.04 → **fails**. At 6 bp it also
fails. With one extra bar of delay (fill at 09:40): gross SR 0.51, net 3 bp 0.17.

*2018 net Sharpe 3.1 > 2 (investigated as a suspected leak).* B1 rules out a data leak. With the signal lagged
one more session, 2018 gross SR is still 2.66, so it is not a same-day timing artefact. It is concentrated:
the top 5 days give 35% of the year's net P&L, and without the top 10 days SR is 1.6. 2018 TSLA daily sd was
267 bp. Over 8 years × 2 rules, one year at about 3σ is consistent with chance, and the full period is
SR 0.42 gross. I read it as a high-volatility year, not an edge.

**(b) sign_first30 (Gao et al.):** position = sign of the first-half-hour return (from the previous close);
fill at the open of the 15:30 bar (bar n−6 on half-days); exit at the close.

| period | gross bp/day | gross SR | SR 0.5bp | SR 1bp | SR 3bp | SR 6bp | break-even bp/side |
|---|---|---|---|---|---|---|---|
| 2018–2025 | 2.9 | 0.66 | 0.43 | 0.20 | −0.72 | −2.11 | 1.43 |
| 2023–2025 | 5.3 | 1.28 | 1.04 | 0.80 | −0.17 | −1.62 | 2.65 |
| 2024 | 13.5 | 3.44 | 3.19 | 2.93 | 1.91 | 0.37 | 6.73 |
| other years | −0.9 … 3.1 | −0.19 … 1.11 | | | −1.88 … −0.64 | | |

D10 at 3 bp: full t −1.97 → **fails**. With one extra bar of delay: gross SR 0.76, net 3 bp −0.76. The gross
effect is in the paper's direction, but it is smaller than costs except in 2024.

No net Sharpe above 2 except the 2018 d_sma20_dev year discussed above.

## Decision needed before Gate 3

The 10.6 IC (pooled within monthly folds) is biased for predictors with multi-day persistence. Gate 3 (D15) B/C
and every Phase 4 model would be scored with it. Options:

1. **Training-centred IC (my recommendation).** Map predictions and outcomes with their training-period
   location rather than the test month's own. For example, use Pearson on prediction − training mean vs
   realized z (z has mean ≈ 0, so it is not re-centred), or Spearman after ranking against the training
   distribution. With no in-window demeaning, the expected value is 0 under no predictability. Keep the fold
   structure, t-stat and D10 unchanged, and validate with this session-permutation placebo (expected ≈ 0).
2. **Keep the 10.6 IC but subtract a placebo**: re-score each model on session-permuted data. This is exact,
   but it requires re-running every model on permuted inputs, which is expensive.
3. **Judge daily-horizon content only by P&L** (rule/backtest Sharpe and fold-level D10 on net returns), which
   has no window demeaning.

Under any option, the Phase 2 single-feature D10 list should be re-run with the corrected metric, and so should
the Phase 2 `stable IC` selection reasons, which also used monthly IC. Intraday features with short memory
are much less exposed, but the same placebo can check them.
