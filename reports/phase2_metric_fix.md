# Phase 2 — metric fix (D17 anchored IC) and metric null test (D18): FAILED

Date: 2026-10-24. Data v2. Holdout and forward test not loaded. Diagnostics only, so none of these runs is
logged as a trial in the registry. Phase 3 has not started.

**Result: D17 as recorded fails D18.** The anchored IC removes most of the bias of the old monthly-pooled
Spearman, but not all of it. On data with no predictability by construction:

- persistent features still score a non-zero mean IC: d_sma20_dev −0.012 to −0.021 (old metric −0.05 to −0.08);
- the D10 false-pass rate is above nominal: 2.3 passes per draw out of 85 features, against 1.4 expected, and
  17% of draws for d_sma20_dev, against 1.7% nominal.

The user's instruction was "If it fails, stop and report. Do not adjust the metric based on these results",
so steps C (re-scoring Phase 2) and D's re-selection were **not run**. `configs/features.yaml` is unchanged
(v1, old metric), and no `features_v1_biased.yaml` or D16 re-run was written. The candidate fixes in §5 are
options for the user to choose from. None of them has been adopted.

## 1. The metric under test (D17, as recorded in `decisions.md`)

Per monthly test fold M, with anchors from the fold's training rows (all months before M):

- feature score s = clip((x − median_train) / IQR_train, ±5) (training sd if the IQR is 0; for binary flags,
  x − training mean); signed by sign_train = sign of the mean anchored IC over the training months;
- target y = clip(z, ±5) − m_train;
- IC = Σ s·y / √(Σ s² · Σ y²) over the test month.

Code: `tslaq.backtest.metrics.anchored_ic`, `tslaq.selection.ic.score_anchors / feature_scores /
anchored_group_ic`, `tslaq.selection.rescore.anchored_fold`.

## 2. Synthetic null (`tests/test_metric_null.py`)

The data: a random walk with U-shaped seasonal volatility; 72 months × 21 sessions × 78 bars; 6 seeds; months
12–71 scored (360 folds). It has two daily features known at the open: level_like (previous close / 20-day mean − 1,
like d_sma20_dev) and inv_price (1 / previous close). h = 12.

| Test | Result |
|---|---|
| `test_old_metric_shows_window_bias` | **passes**: old metric level_like −0.0756 (t −18.5), inv_price +0.1027 (t +26.6) |
| `test_anchored_ic_basic` | passes |
| `test_anchored_ic_no_bias_on_null` | **FAILS**: anchored level_like −0.0195 (t −4.4) |

I left the failing test in place on purpose (not xfailed). It is the D18 condition.

**Decomposition**, from 20-seed prototypes (scratchpad scripts, not in the repo). Each cell is the mean
anchored IC for level_like / inv_price, with se ≈ 0.002:

| Variant | level_like | inv_price |
|---|---|---|
| D17 as defined | −0.0207 | +0.0301 |
| D17 with m = 0 | −0.0217 | +0.0218 |
| numerator Σ s·y with the denominator from training second moments | −0.0003 | +0.0177 |
| same with m = 0 | −0.0015 | +0.0058 |
| score frozen at its month-start value | −0.0028 | +0.012 |
| frozen with m = 0 | −0.003 | +0.0035 |

The decomposition points to two mechanisms:

1. **Ratio bias (level-like features).** Σ s² over the test month depends on that month's own returns for a
   persistent feature: a month that moves the price far from its 20-day mean inflates |s|. The normalisation
   by √Σs² then correlates with the numerator. With a fixed (training) denominator the bias is gone
   (−0.0003).
2. **The −m·mean(s) term (non-stationary features).** A feature that drifts away from its training median
   (1/price on a trending path) has mean(s) ≠ 0 in the test month, so subtracting m_train adds −m·Σs, a
   term that has nothing to do with prediction. inv_price falls from +0.018 to +0.006 when m = 0.

## 3. Session-permutation placebo (B1, 200 permutations)

Script `scripts/phase2_metric_placebo.py`, table `reports/tables/phase2_metric_fix/placebo_both_metrics.csv`.
Whole sessions are shuffled (each keeps its intraday path, gap and range relative to the previous close). Daily
closes are rebuilt and the daily features recomputed. 96 folds 2018-01..2025-12, h = 12.

| Feature | Metric | Real unsigned | Placebo mean | Placebo sd | Placebo p05..p95 | Within ±0.005 or 1 sd? |
|---|---|---|---|---|---|---|
| d_sma20_dev | old (biased) | −0.0617 | −0.0626 | 0.0096 | −0.080..−0.048 | no |
| d_mom5 | old (biased) | −0.0452 | −0.0364 | 0.0092 | −0.052..−0.021 | no |
| d_atr_c | old (biased) | +0.0512 | +0.0444 | 0.0095 | 0.030..0.059 | no |
| amihud | old (biased) | +0.0451 | +0.0023 | 0.0080 | −0.011..0.015 | yes* |
| d_sma20_dev | anchored | −0.0257 | **−0.0155** | 0.0098 | −0.032..−0.001 | **no** |
| d_mom5 | anchored | −0.0227 | −0.0085 | 0.0097 | −0.025..0.007 | yes (1 sd) |
| d_atr_c | anchored | +0.0058 | **+0.0116** | 0.0095 | −0.005..0.026 | **no** |
| amihud | anchored | +0.0159 | +0.0036 | 0.0079 | −0.011..0.016 | yes* |

Signed out-of-sample, anchored: d_sma20_dev real 0.0257 (t 2.74), placebo mean 0.0127 (sd 0.0115; mean
placebo t 1.37, p95 t 3.29). d_mom5 real 0.0227 (t 2.55), placebo 0.0046. The delayed-outcome results are almost
the same (CSV).

\* The amihud placebo is not comparable to the other rows. Each session keeps its dollar volume, so shuffling
sessions destroys amihud's own persistence. The mirror null (§4) keeps volume in place.

Under the anchored metric, the real IC of every one of the four features lies inside or near its placebo band
(|z| ≤ 1.6; amihud z 1.55 unsigned, 1.83 signed).

## 4. Random-sign mirror null (B2, 99 draws)

Script `scripts/phase2_metric_null.py`. Per-draw output is in `data/processed/phase2_metric_null/` (gitignored),
and the tables are in `reports/tables/phase2_metric_fix/null_*.csv` and `null_summary.json`.

- **Construction.** Each TSLA bar gets an independent random sign. A minus sign mirrors (open, high, low, close,
  vwap) around the previous close in log space, with high and low swapped. Volume is kept, and prices are
  rebuilt from the first price. |r|, ranges, volatility clustering and the persistence of the features survive,
  but future returns cannot be predicted from the past. SPY and QQQ are the real series.
- **Features.** All 85 Phase 2 features are recomputed on the mirrored series. **Approximation:** GARCH-t, HMM,
  Kalman and frac-diff use the already-fitted parameters of the last fold (2025-12) for all folds; the real run
  refit them per fold. Each fold's feature version (raw or rolling-z) comes from its checkpoint.
- **Scoring.** 96 folds, h = 12 with and without the one-bar delay, signs from the training months with the same
  metric.
- **Budget.** A single draw took 130 s. With 4 workers it was about 4 min per draw per worker, so 99 draws
  (0–98) fit into the ~1.5 h budget. The run was stopped at the budget, and the finished draws are kept.
- r_12th30 and sign_12th30 exist only from 15:30, where no h = 12 label fits before the close, so they cannot
  be scored (not counted as failures).

**D10 false passes per draw** (85 features; nominal = 85 × 1.67% = 1.42, where 1.67% is the D10 pass rate for
iid N(0, 1) fold ICs):

| Target | Anchored mean (min..max) | Old mean (min..max) | Nominal |
|---|---|---|---|
| z_12 | **2.31** (0..20), median 1; 0 passes in 32% of draws, ≥ 5 in 14% | 9.90 (1..28); ≥ 5 in 89% | 1.42 |
| z_12 delayed | **2.14** (0..19) | 8.94 (1..28) | 1.42 |

The draw-to-draw spread is wide because the features are strongly correlated: one draw can push a whole
cluster through D10 at once.

**Per-feature D10 false-pass share, anchored (old in brackets), z_12:** d_sma20_dev 17% (100%), d_mom5 11%
(86%), rsi14 8% (26%), vwap_dev 8% (60%), mom_3 7% (12%), bar_index 7% (4%), fracdiff_lnc 6% (78%),
mom_78 6% (43%), ema12_26 5% (23%), minutes_to_close 5% (4%). Nominal is 1.7%.

**Mean IC over draws, anchored unsigned (sd over draws), z_12:** d_sma20_dev **−0.0120** (0.0089), d_mom5
−0.0085 (0.0093), vwap_dev −0.0075 (0.0092), ema_dev_78 −0.0061 (0.0084), d_atr_c +0.0049 (0.0091). The
other features are within ±0.005. Under the stated condition (within ±0.005 or within 1 null sd), only
**d_sma20_dev fails** (both targets); all signed means pass. Under the old metric, 21 features fail the
unsigned condition and 12 the signed one. d_sma20_dev's old null mean of −0.052 matches its real old IC of
−0.062.

## 5. Verdict and options (for the user; nothing adopted)

**Pass condition (step B):** placebo and null means within ±0.005 of 0 (or within 1 null sd), and a D10
false-pass count close to nominal. **It is not met:**

- the placebo fails for d_sma20_dev and d_atr_c;
- the null mean fails for d_sma20_dev;
- the D10 false passes are 1.6× nominal overall, and 5–10× nominal for the persistent trend and level features;
- the synthetic D18 test fails.

D17 is much better than the old metric (about 4× fewer false passes; the d_sma20_dev bias is cut from −0.05
to −0.012), but it is not yet a valid gate metric under D18.

Candidate changes for the user to decide on. Each would need its own decision entry and a fresh run of D18
(synthetic test + placebo + mirror null) before use:

1. **Training-moment denominator.** IC = Σ s·y / (n · √(E_train[s²] · E_train[y²])), so the test month
   enters only through the numerator. In the synthetic null it removes the level-feature bias
   (−0.0207 → −0.0003). On its own it leaves the 1/price bias (+0.018).
2. **Drop the target anchor (m = 0), or make the score's mean zero over the training rows.** This takes out
   the −m·Σs term for drifting features. Combined with option 1: −0.0015 / +0.0058 in the synthetic null,
   both within about 2 se of 0.
3. **Frozen anchors / month-start scores.** This is closer to how a daily feature would be traded, but it
   changes what is being measured. Synthetic: −0.003 / +0.0035 with m = 0.
4. **Use the null as the reference distribution.** Judge D10 against each feature's own null (e.g. real t
   vs the mirror-null t quantile) instead of the iid nominal. This costs ~4 min of compute per draw, and the
   null uses approximate fitted parameters.

Recommendation: options 1 + 2 together. They are the smallest change that addresses both identified
mechanisms, and they keep the "model's predicted z as is" score for models. Whatever is chosen, rerun the full
D18 battery, unchanged, before any Phase 2 re-scoring.

## 6. Status of Phase 2 numbers

Until a metric passes D18, the Phase 2 IC tables, the per-fold feature lists (`configs/features.yaml`, old
metric) and the D10 passes in `phase2_features.md` §4 stand as computed, but they are **biased, information
only**. The D10 passes of the persistent daily features in particular are explained by the metric bias (§3,
§4).
