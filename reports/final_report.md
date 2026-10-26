# TSLA intraday forecasting: final report

Date 2026-10-26. This report is for a reader who has not followed the project. Every number links to the report or
table it comes from. Decisions D1–D28 are in [`decisions.md`](decisions.md).

## Summary

- **Question.** Can tokenized 5-minute market data and a small Transformer forecast TSLA's next 5- to 60-minute
  returns well enough to trade after costs? The goal was an honest, leakage-free answer, not an impressive backtest.
- **Answer.** No tradable edge was confirmed during development, and the Transformer (Phase 4) was never built,
  because the earlier, cheaper gates did not justify it.
  - A small, real-looking signal exists at the 60-minute horizon: intraday trend continuation, about 2.7 effective
    signals.
  - Its information is roughly the size of TSLA's trading costs. The best development strategy (Gate-3 B, rule R1)
    clears realistic costs on paper, but not the deflated-Sharpe hurdle (DSR 0.000), and it failed its own model gate.
  - The single holdout evaluation of that frozen candidate (2026-01-01..2026-10-09, 194 sessions) **failed its
    pre-registered criterion**. Net P&L at realistic cost was −725 bp (net Sharpe −0.49), and gross P&L was ≈ 0
    (+75 bp), even though the forecasts kept a weakly positive IC (0.016, t 1.15).
- **Recommendation.** Stop. Do not trade this design, do not build Phase 4 on it, and keep the forward test sealed
  unless a new, separately pre-registered idea justifies using it (§10).

## 1. The question

The project asked whether TSLA's next-bar to next-hour **return distribution** can be forecast from tokenized market
data with a small causal Transformer (CLAUDE.md §0). It was set up to give an honest estimate of whether any edge
exists, with these safeguards:

- no future data in any fitted object;
- walk-forward validation with purge and embargo;
- a registry of every trial, used for the deflated Sharpe ratio;
- a sealed holdout and a sealed forward test.

The **primary trading horizon is h = 12 bars (60 minutes)** (D2). Shorter horizons (5 and 15 minutes) are dominated
by costs and bid–ask bounce. Every signal is also scored with one extra bar of execution delay, so that bounce
cannot pass as an edge.

## 2. Data

| set | period | use | status |
|---|---|---|---|
| development | 2016-01-04..2025-12-31 (TSLA 2,512 sessions) | features, selection, all models and gates; 96 monthly out-of-sample folds 2018-01..2025-12 | used |
| holdout | 2026-01-01..2026-10-09 | one evaluation of one frozen design (D7, D28) | **used once** (§7) |
| forward test | after 2026-10-09 | the only data guaranteed unseen (the holdout may have been seen in an earlier project) | **sealed** |

- **Source.** TSLA, SPY and QQQ Alpaca SIP 1-minute bars, aggregated to 5-minute regular-session bars (78 per day)
  ([`phase1_summary.md`](phase1_summary.md), [`phase1_data_quality.md`](phase1_data_quality.md)).
- **Data v2 (D9).** The vendor's split-adjusted TSLA prices before the 2022 split were rounded to $0.01. In 2016
  this made 18% of 5-minute returns exactly zero. TSLA 2016-01..2022-08 was therefore rebuilt from unadjusted prices
  divided by the split factor in floating point. Exact zeros fell from 18.1% to 0.79% in 2016
  ([`phase1_summary.md`, data v2 re-run](phase1_summary.md#data-v2-re-run-2026-10-22)).
- **Other cleaning.** SPY/QQQ are dividend back-adjusted. Earnings dates come from Alpha Vantage and are mapped to
  the reaction session. Sessions with more than 5% missing bars are dropped (4 in TSLA). Bad ticks are flagged, not
  deleted.

## 3. Methods

- **Volatility scale.** Returns are scaled by σ̂ = EWMA level × time-of-day profile (λ = 0.97, tuned on 2016–2017
  only). The label is z = forward log return / S_{t,h}, where S_{t,h} is the scale over the actual bars ahead.
  Labels never cross the session close.
- **Features (Phase 2).** 83 causal features: candle shape, bar-to-bar relations, trend and momentum (EMA, RSI,
  MACD, stochastics, regression slopes), intraday momentum (Gao et al.), volatility and range, volume and VWAP,
  statistical models (GARCH-t, filtered HMM, Kalman, Hurst, entropy, frac-diff), market context (SPY/QQQ, residual
  vs QQQ), daily context and event flags.
  - Every fitted object (GARCH, HMM, Kalman, frac-diff d, stationarity versions, clusters, bins) is refit per fold
    on training data only.
  - Selection is per fold (`configs/features.yaml`, 17–32 features per fold).
  - Report: [`phase2_features.md`](phase2_features.md), [`phase2_rescore.md`](phase2_rescore.md).
- **Tokenizer (Phase 3).** One composite token per bar with 11 fields (vol-scaled return in 64 bins, k-means candle
  shape, candle relation, gap, relative volume, volatility regime, HMM state, QQQ return, time of day, day of week,
  event). Fitted per fold. It passed all checks: round-trip correlation 0.99 in z, no lookahead
  ([`phase3_tokenizer.md`](phase3_tokenizer.md) §1–2).
- **Models.** All are gradient-boosted trees or linear models with fixed parameters (no tuning):
  - LightGBM on lagged returns (Phase 1, "M1");
  - LightGBM on tokens (A), on the selected features (B), and on both (C) (Gate 3);
  - ridge (M2), an equal-weight feature composite (M3) and CatBoost (M4) (D25).
- **Validation.** Expanding-window monthly walk-forward, 96 folds (2018-01..2025-12). Training samples whose label
  window overlaps the test month are purged, and a 78-bar (one-session) embargo follows. The inner validation block
  (the last 2 training months) is used for any choice. The test month is never used for one.
- **Tests.** 65 tests pass. They cover no-lookahead for features, tokens, HMM (filtered) and GARCH (fixed
  parameters), label alignment, purge and embargo, the split adjustment, execution delay, DSR, and the metric null.

## 4. The metric problem and its fix

The project's biggest methodological finding was that the planned **information coefficient (IC) was biased**,
which mattered more than any model choice.

1. **Per-day Spearman (Phase 1).** For one stock this ranks bars over time within a day. Overlapping 60-minute
   windows then correlate mechanically with any path-dependent feature: a 78-bar momentum feature scores −0.31 on a
   pure random walk (D1; `tests/test_baselines.py::test_ic_metric_bias_placebo`). It was replaced by a Spearman
   pooled within each monthly fold.
2. **Monthly-pooled Spearman (Phase 2).** This also failed. Daily-context features such as distance from the 20-day
   average behave like a lagged price level. Demeaning within the test month uses the month's own average, which is
   only known at month end. That injects the month's own returns into the score.
   - A session-permutation placebo (shuffling whole days, so no real predictability is left) produced ICs as large
     as on the real data: d_sma20_dev −0.063 placebo vs −0.062 real
     ([`phase2_daily_diagnostics.md`](phase2_daily_diagnostics.md)).
   - So the "daily reversal" signals that passed in Phase 2 were artefacts.
3. **Anchored IC v2 (D19).** Nothing from the test month is used except each row's own score and outcome. The
   score is centred and scaled with training statistics, the outcome is not demeaned, and the normalisation uses
   training moments ([`phase2_metric_fix_v2.md`](phase2_metric_fix_v2.md)).
4. **Null tests (D18, D20–D22).** A metric may be used in a gate only after it scores ≈ 0 on data built to have no
   predictability: a random-sign "mirror" of TSLA, with a drift calibrated to the real mean.
   - D19 passed the synthetic and placebo tests. On the mirror null with drift it narrowly failed on two
     seed-sample effects (dow, earn_react), first with a mis-specified drift and then with the corrected one
     ([`phase2_metric_fix_v3.md`](phase2_metric_fix_v3.md)).
   - Under the rule fixed before that run, the formula was not revised again. Instead (D22), each feature is judged
     against **its own 300-draw null distribution**, and every model gate also requires the model to score ≈ 0
     when trained and tested on K null draws.
   - With D22, the null's mean ICs are within ±0.0016 and the fixed-rule false-pass rate is 1.80% (nominal ≈ 1.67%)
     ([`phase2_rescore.md`](phase2_rescore.md) §2).
   - Limitation: the null uses the last fold's fitted GARCH/HMM/Kalman/frac-diff parameters in every fold.

## 5. Results by phase and gate

The gate rule (D10) is: full-period t ≥ 2, **and** over 2023–2025 a mean IC > 0 with t ≥ 1. Models must also pass
the D22 model null (null mean IC within ±0.003, and real t ≥ null mean t + 2 null sd). "h = 12 delayed" = 60-minute
outcome measured from one bar later.

| step | what | key numbers (h = 12 delayed unless stated) | outcome | source |
|---|---|---|---|---|
| Phase 1 | data v2, σ̂ scale, baselines | lagged-returns LightGBM: h = 1 IC 0.024 → 0.0085 with one bar of delay (bounce); old metric h = 12 t 1.7, 2023–25 IC −0.002. GARCH-t QLIKE 1.435 / 0.522 / 0.248 (h = 1/3/12) | gate passed (tests, data, reproducibility) | [`phase1_summary.md`](phase1_summary.md) |
| Gate 2 | features, per-fold selection | 19 features pass D10\* at h = 12 (19 delayed); ≈ 2.7 effective signals; the 6 daily-context passers of the old metric all fail | passed (re-confirmed under D19 + D22) | [`phase2_rescore.md`](phase2_rescore.md) §3, §8 |
| Gate 3 | LightGBM on tokens (A), features (B), both (C) | C: IC 0.0078, t 1.39 (null threshold 2.09); A: t 0.15; B: IC 0.0167, t 1.92 (threshold 2.34); C vs B: 49% of folds, Wilcoxon p 0.77 | **failed**; tokens add nothing | [`phase3_tokenizer.md`](phase3_tokenizer.md) §4–5 |
| D25 | pre-registered baselines M1–M4 | M3 (equal-weight composite): IC 0.0294, t 4.33, 2023–25 t 1.90, null threshold 2.11 → **pass**; M1 t 3.54 but 2023–25 t 0.95; M2 t 2.01; M4 t 0.85; 1 of 20 null seeds has a passing model | M3 passes (but largely restates Gate 2) | [`phase3_baselines.md`](phase3_baselines.md) §2, §5, §8 |
| D27 | early economic test (M3, B, M1 × rules R1/R2) | B R1: net Sharpe 1.11 at D26 cost, 0.10 at 2× D26, 2023–25 1.06, +1 bar delay 0.62; DSR 0.000 for every trial | **no trial passes** (DSR) | [`phase5_early.md`](phase5_early.md) §2–4 |
| D28 | single holdout evaluation of frozen B R1 (2026-01..2026-10-09) | reproduction on fold 2025-12: corr 1.000; net P&L at D26 (0.749 bp/side) **−725 bp**, Sharpe −0.49; gross +75 bp; break-even 0.07 bp/side; IC delayed 0.0157, t 1.15 (10 months) | **criterion failed** | [`holdout_eval.md`](holdout_eval.md) |
| Phase 4 | Transformer | – | **not started** (the D27 rule recommended concluding) | [`decisions.md`](decisions.md) D27 |

Other results:

- **No model beats the zero forecast on CRPS at h = 12** (Gate 3 and D25:
  [`phase3_baselines.md`](phase3_baselines.md) §3).
- **Out-of-sample hit rates are 0.500–0.507** (D25 §2.1).
- Phase 4's volatility-head target (beat GARCH-t on identical rows) was never tested.

## 6. What survived and what did not

**Survived** (D22 single-feature tests at h = 12, with and without delay;
[`phase2_rescore.md`](phase2_rescore.md) §3, §7):

1. **Intraday trend / oscillator continuation.** rsi14, bb_pctb, stoch_k14, EMA deviations, momentum over 3–24 bars,
   VWAP deviation and the bar-relation features. D19 IC 0.007–0.036, t 2.3–4.8. They survive the one-bar delay, and
   their sign is continuation, the opposite of what bid–ask bounce would produce.
2. **Day-level momentum.** mom_78, ema_dev_78, r_first30, sign_first30 (Gao et al. intraday momentum). IC
   0.022–0.038. mom_78 is weak in 2023–25.
3. **Residual vs QQQ.** Marginal; passes only without the delay.

These 19 passers are about **2.7 effective signals** (eigenvalues 11.0, 3.4, 1.3 of their fold-IC correlation
matrix). No null draw reached 19 passers, and the null averages 1.3 passers per draw.

**Did not survive:**

- **Daily reversal and illiquidity.** d_sma20_dev (old t 6.26 → 0.93), d_mom5, d_atr_c, amihud and d_prev_ret_atr.
  These were the old metric's artefact.
- **The tokens.** No token cell survives Benjamini–Hochberg at h = 12 in any fold. Tokens alone have t 0.15, and
  adding them to B makes LightGBM worse (significantly at h = 1 and 3)
  ([`phase3_tokenizer.md`](phase3_tokenizer.md) §3, §5). What the tokens do carry is 1-bar reversal at h = 1, which
  is bid–ask bounce. With a fixed-parameter LightGBM encoding, this is a negative result about the tokens. It does
  not test how a Transformer would use them.
- **The short-horizon edge.** h = 1 and h = 3 ICs are larger (B: t 12.8 and 9.1), but one bar of delay cuts most of
  it, and the rules lose money at realistic costs ([`phase3_baselines.md`](phase3_baselines.md) §6).
- **Model combination at h = 12.** Only the equal-weight composite M3 passes. Fitted models (B, C, ridge, CatBoost)
  do not beat it significantly and mostly fail their nulls.

## 7. Economics and the holdout

- **Spreads.** TSLA's effective half-spread (EDGE estimator from 1-minute bars) fell from 2.14 bp (2016) to
  0.51–0.65 bp (2023–2025) ([`tables/phase3_baselines/spread_by_year.csv`](tables/phase3_baselines/spread_by_year.csv)).
  The realistic cost (D26) is the prior year's half-spread + 0.10 bp SEC fee: 0.61–1.74 bp per side, 0.82 bp on
  average in 2023–25. **This cost was chosen after the D25 break-even numbers were seen.**
- **Break-even costs per side** (sign rule, h = 12): Gate-3 B 1.33–1.41 bp, M3 0.31–0.39 bp
  ([`phase3_baselines.md`](phase3_baselines.md) §6). With rule R1: B 2.64 bp (2.01 bp with one extra bar of delay),
  M3 0.78 bp, M1 1.00 bp ([`phase5_early.md`](phase5_early.md) §2).
- **The best development candidate.** Gate-3 B with R1 (position = clip(ẑ/q, ±1), band 0.25, fill at the next open,
  flat at the close):
  - net Sharpe 1.11 at D26 cost over 2018–2025, positive in 6 of 8 years;
  - only 0.10 at 2× D26, and −0.29 at the old 3 bp default;
  - CPCV: 9 paths with median net Sharpe 0.68, all positive, but with optimistic approximations
    ([`phase5_early.md`](phase5_early.md) §5).
  - It is a model that **failed Gate 3**. The IC gate and the economic test disagree on which model is useful
    (M3 has the best IC but the worst P&L).
- **Holdout (D28).** With development concluded, the user approved spending the holdout on the frozen B R1 as a
  "no-failure" check. This was a deviation from the D27 rule, and it does not make B R1 a pass. Design, criterion
  and interpretation were committed before any holdout data was downloaded.
  - **Reproduction.** A from-scratch reproduction of development fold 2025-12 matched the stored predictions exactly
    (correlation 1.000, max difference 0), with the same 27 features and the same R1 P&L (−215.16 bp)
    ([`tables/holdout_d28/reproduction_check.csv`](tables/holdout_d28/reproduction_check.csv)).
  - **Criterion.** On the holdout, total net P&L at D26 cost was **−725 bp**, so the criterion (> 0) **fails**
    ([`tables/holdout_d28/holdout_results.csv`](tables/holdout_d28/holdout_results.csv)).
  - **Other numbers.**
    - Net Sharpe −0.49 at D26, −1.02 at 2× D26, −2.09 at 3 bp.
    - Gross +75 bp (Sharpe 0.05), so break-even was 0.07 bp per side, against 2.64 bp in development.
    - With one extra bar of delay: −675 bp net, +126 bp gross.
    - Hit rate 0.485; 12.5 fills and 5.5 units of turnover per day.
  - **By month.** Net P&L is positive in 3 of 10 months. September alone lost 1,364 bp and January gained 701 bp
    ([`tables/holdout_d28/holdout_by_month.csv`](tables/holdout_d28/holdout_by_month.csv)).
  - **IC.** The D19 IC stayed about the size seen in development but is not significant over 10 months: h = 12
    0.0138 (t 0.97), delayed 0.0157 (t 1.15), positive in 7 of 10 months
    ([`tables/holdout_d28/holdout_ic_by_month.csv`](tables/holdout_d28/holdout_ic_by_month.csv)). The information,
    if real, is far too small in return terms to pay even a 0.75 bp half-spread.
  - **Power.** The holdout has low power: a failure had about a 17% chance even if the development Sharpe of about
    1.1 were true. The result is consistent with the deflated-Sharpe verdict and with no tradable edge.

## 8. Multiple testing

- **Trials.** The registry ([`../experiments/registry.csv`](../experiments/registry.csv)) holds **43
  trials**: 6 Phase 1, 3 Phase 2 screens and rules, 9 Gate 3, 18 D25, 6 D27 and 1 D28. Re-runs are not counted.
- **Repeated looks.** The same 96 development folds were looked at repeatedly across phases.
- **Deflated Sharpe ratio (D27).** It is 0.000 for every rule trial with the pre-registered V (variance of all rule
  trials' Sharpe ratios, 0.2773). That V implies SR\* = 1.16 per day, about 18 annualized, which no strategy could
  meet.
  - With a like-for-like V (the 6 D27 trials only), the best DSR is 0.196 (B R1). That is still far below 0.95.
  - Without any deflation, PSR(0) of B R1 is 0.999 ([`phase5_early.md`](phase5_early.md) §4).
- **Family-wise check.** In D25, 1 of 20 null seeds had a passing model (5%).

## 9. Limitations

1. **Single stock, single venue model.** TSLA only. No cross-sectional confirmation, no market-impact or
   queue-position model. Fills assume the next bar's open.
2. **Null approximation.** All model nulls use the last fold's fitted GARCH-t / HMM / Kalman / frac-diff parameters
   in every fold, and the real run's feature lists.
3. **Costs after the fact.** The D26 cost was chosen after the D25 break-even numbers were seen. The EDGE estimate
   is itself an estimate; Corwin–Schultz gives 2–5× higher spreads from 2020 on.
4. **Reuse of the evaluation period.** M3's constituents are the Phase 2 passers evaluated on the same folds. The
   CPCV used the last fold's feature list (selected on data that includes most test blocks).
5. **Holdout power and status.** About 194 sessions cannot distinguish a net Sharpe of 1 from 0 with confidence.
   The holdout may also have been seen in the earlier `tsla-diffusion` project. Only the forward test is guaranteed
   unseen.
6. **Phase 4 untested.** The Transformer, self-supervised pre-training and the Kronos comparison were not built, so
   the project says nothing about them directly. It does show that the inputs they would use carry little h = 12
   information beyond simple trend features.

## 10. Recommendations

1. **Stop the project as designed.** Development produced no trial that passes its pre-registered gates, and the
   one holdout evaluation failed. Per D28, do not start paper trading (Phase 6) with Gate-3 B R1 and do not commit
   capital.
2. **Do not build Phase 4 on this evidence.** The tokens carry no h = 12 information in the tests run here. The best
   feature model's information is about the size of the half-spread. D27 set the bar a Transformer would have to
   clear: h = 12 delayed IC ≥ 0.02 with a passing economic test. Nothing so far suggests it can.
3. **Keep the forward test sealed.** It is the only data guaranteed unseen. Use it only for a genuinely new idea that
   is fully pre-registered (design, criterion and interpretation) before it is opened, and then only once.
4. **If the question is reopened, change the economics, not the model.** The small intraday trend-continuation
   signal is consistent across the 96 development folds and kept a positive (non-significant) IC on the holdout. It
   could only matter where costs are much lower than a TSLA half-spread, or in a cross-section of many stocks, where
   a common signal can be pooled. Either would be a new project with its own null tests and holdout.
5. **Reuse the method.** The parts worth keeping are independent of the result:
   - the anchored IC v2 metric with per-feature and per-model mirror-null calibration (D19, D22);
   - the session-permutation placebo that exposed the monthly-IC artefact;
   - the pre-registration discipline (decisions recorded and committed before each run).

## Files

- **Reports.**
  - [`decisions.md`](decisions.md) (D1–D28);
  - [`phase1_summary.md`](phase1_summary.md), [`phase2_features.md`](phase2_features.md),
    [`phase2_daily_diagnostics.md`](phase2_daily_diagnostics.md), the `phase2_metric_fix*.md` reports and
    [`phase2_rescore.md`](phase2_rescore.md);
  - [`phase3_tokenizer.md`](phase3_tokenizer.md), [`phase3_baselines.md`](phase3_baselines.md),
    [`phase5_early.md`](phase5_early.md) and [`holdout_eval.md`](holdout_eval.md).
- **Registry.** [`../experiments/registry.csv`](../experiments/registry.csv).
- **Code.** `src/tslaq/`, `scripts/`, `tests/` (65 tests).
