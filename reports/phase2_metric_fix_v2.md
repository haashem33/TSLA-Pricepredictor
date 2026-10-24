# Phase 2 — anchored IC v2 (D19) against the pre-registered null criteria (D20): FAILED

Date: 2026-10-24. Data v2 (bars 2016-01-04..2025-12-31). Holdout and forward test not loaded. Diagnostics only,
so no registry trials. Phase 3 has not started.

**Result: D19 fails D20.**

- Criteria 1 (synthetic) and 2 (session-permutation placebo) pass.
- Criterion 3 passes for the mirror null without drift.
- Criterion 3 **fails for the mirror null with real drift**: 10–12 volatility-level features have a non-zero mean
  IC, and the pooled D10 false-pass rate is 2.35% / 2.44% against a limit of 2.2%.

As pre-registered, I did not revise the formula. Step C (re-scoring Phase 2, re-selection, features.yaml v2,
the D16 re-run, Phase 1 re-scoring) was **not run**. `configs/features.yaml` is unchanged (v1, old metric).
The D20 fallback applies; its projected cost is in §5.

## 1. D19 as implemented

Per monthly fold M, with anchors from the training rows (all months before M):

- c(x) = clip((x − median_train) / IQR_train, ±5); the training sd is used if the IQR is 0. Binary flags
  (≤ 2 distinct training values) are coded 0/1, with 1 = the larger value.
- s = sign_train · (c(x) − mean_train c(x)). y = clip(z, ±5).
- IC = Σ s·y / (n·√(V_s·V_y)), where n = rows with finite s and y, V_s = training mean of (c − mean c)², and
  V_y = training mean of y².
- sign_train = sign of the mean D19 IC over the training months, with the same anchors.
- Models: s = ẑ − training mean of clipped z, with V_s from all earlier out-of-sample months (the first fold
  uses the inner-validation block).

Code:
- `tslaq.selection.ic.score_anchors_v2 / feature_scores_v2 / v2_group_ic`
- `tslaq.backtest.metrics.ic_v2 / fold_ic_v2_model`
- `tslaq.selection.rescore.anchored_fold_v2`

The model path is unit-tested (`test_v2_basic`) but has not been used on real predictions.

## 2. Decisions on reading D20, fixed before any result was seen

- **Criterion 3 on both targets.** It is applied to h = 12 and to h = 12 with the one-bar delay. Both must
  pass in each variant.
- **Draw counts.** Exactly 60 no-drift draws (seeds 1000..1059, the same mirrors as the D17 run) and 40 drift
  draws (seeds 1000..1039: the same mirrors, plus drift).
- **Mirror-null drift.** "Mean log return per bar" was read literally: the mean of ln(C_t/C_{t−1}) over *all*
  bars 2016–2025, overnight bar included, is 1.73e-5. This is added to every bar (every price of row t × e^{μt}).
  - The mean z_12 in the drift null is therefore **0.028**, about 15× the real regular-session mean of z_12
    (0.0019). Most of TSLA's drift is overnight; the intraday mean (bar_index > 0) is only 3.5e-6.
  - So the drift variant is a much stronger test than the real data. Whether this is the reading you intended is
    your call. I did not re-run it any other way.
- **Synthetic drift.** A constant per-bar μ = 4.27e-7, chosen so that E[z_12] = 0.00192 (the real mean of
  z_12). The synthetic test uses 100 seeds, fixed before running (se of the mean IC ≈ 0.0012).

## 3. D20 criteria

| # | Criterion | Value | Result |
|---|---|---|---|
| 1a | Synthetic, zero drift: level_like / inv_price mean IC within ±0.003 | +0.0012 / −0.0013 (se 0.0012 / 0.0017; 6,000 folds) | **pass** |
| 1b | Synthetic, real drift: level_like / inv_price_rz within ±0.003 | +0.0012 / −0.0016 (raw inv_price −0.0010, reported only) | **pass** |
| 2 | Placebo (200 permutations), unsigned placebo mean within ±0.003: d_sma20_dev / d_mom5 / d_atr_c | +0.0005 / +0.0003 / +0.0010 (sd ≈ 0.0098); amihud +0.0033, reported only | **pass** |
| 3a | Mirror null, 60 draws, h = 12: every feature's mean within ±0.003 or 3 se | all 83 scoreable features pass (max \|mean\| 0.0034, d_mom5, se 0.0014) | pass |
| 3a | pooled D10 false-pass rate ≤ 2.2% | 1.69% (1.4 per draw out of 83; nominal 1.67%) | pass |
| 3a | no feature above 8% | max 6.7% (obv_slope_z) | pass |
| 3a′ | same, h = 12 delayed | all features pass; pooled 1.77%; max 6.7% (earn_react) | pass |
| 3b | Mirror null + drift, 40 draws, h = 12: every feature's mean within ±0.003 or 3 se | **10 fail**: hmm_p2 −0.0108, hmm_p0 +0.0078, garch_logvar12 −0.0069, parkinson_78 −0.0063, gk_78 −0.0059, vreg_level_pct −0.0058, vreg_sigma_pct −0.0056, rv_today_deseas −0.0053, garch_sigma −0.0050, range −0.0041 | **FAIL** |
| 3b | pooled D10 false-pass rate ≤ 2.2% | **2.35%** (1.95 per draw) | **FAIL** |
| 3b | no feature above 8% | **vreg_level_pct 12.5%, fracdiff_lnc 10%** | **FAIL** |
| 3b′ | same, h = 12 delayed | **12 fail the mean condition** (the above, plus bar_index and minutes_to_close); pooled **2.44%**; vreg_level_pct **12.5%** | **FAIL** |

r_12th30 and sign_12th30 cannot be scored at h = 12: they exist only from 15:30, and no h = 12 label fits
before the close. With 40 draws, the per-feature D10 rates are coarse: 8% is 3.2 draws, 12.5% is 5 draws.

Tables are in `reports/tables/phase2_metric_fix_v2/`:
- `null_v2_summary.json`, `null_v2_per_feature.csv`, `null_v2_d10_passes_per_draw.csv`, `null_v2_per_draw_feature.csv`;
- `placebo_all_metrics.csv` (old, D17 and D19 side by side), `placebo_meta.json`.

The real-data values in the placebo table are reference points only, not a gate result. D19 signed
out-of-sample IC at h = 12: d_sma20_dev +0.0049 (t 0.50), d_mom5 +0.0116 (t 1.22), d_atr_c −0.0064,
amihud −0.0072. All four are inside their placebo bands. The old metric gave t 6.8, 5.2, 5.6 and 5.2.

## 4. Why the drift variant fails

Every no-drift draw has a drift twin with the same seed. Over the 40 pairs, the IC difference drift − no-drift
is almost the same in every pair (sd ≈ 0.0001). For example, garch_logvar12 moves by −0.0083, hmm_p2 by
−0.0074 and garch_sigma by −0.0061.

These features measure |r| and the range, and the drift barely touches them. What changes is y: with a drift,
y has a non-zero mean ȳ. Write y = ȳ + (y − ȳ). The numerator then gains a term ȳ·Σ s, which adds about
ȳ · mean_test(s) / √(V_s V_y) to the IC.

For volatility-level features, mean_test(s) ≠ 0: the test month's volatility level differs systematically from
the expanding training mean. D19 centres the score at its training mean, not at the test month's mean, so this
term is not removed. That is by design (D19 uses nothing from the test month). It is the D17 mechanism 2 moved
from the target to the score: under drift, y is not centred.

The term is linear in ȳ. Scaled to the real mean z_12 (0.0019 instead of 0.028), the worst feature would shift
by about 0.0083 × 0.0019 / 0.028 ≈ 0.0006. This is an analytic extrapolation, not a test result. The real
data has a positive mean z, so the effect exists there too, probably about 15× smaller than in this null.

## 5. The D20 fallback and its projected cost

The formula is not revised again. Under the fallback:

**(a) Single features: D10 with null quantiles.**
- Each feature's real signed t (full period, and 2023–2025) is compared with the same feature's mirror-null t
  distribution, in place of the fixed thresholds t ≥ 2 / t ≥ 1.
- The 100 D19 draws already saved (60 without drift, 40 with) give a first version. The pooled null t quantiles
  at h = 12 are 95%: 1.73 / 1.86 and 97.5%: 2.04 / 2.23 (no drift / drift).
- Per-feature quantiles at the D10 tail (about 98%) need about 200–500 draws per feature. That means
  100–400 more draws, at about 255 s per draw per worker with 5 workers (about 51 s per draw overall), so
  **about 1.5–6 h**.
- You need to choose the null variant. The drift null contains the ȳ·mean(s) term that the real data also has,
  but with ≈ 15× the real drift. A drift scaled to the real mean of z would need a new run of ≈ 40+ draws
  (≈ 35 min). That is a choice about the null, not about the formula.

**(b) Exact per-fold fitted parameters in the null.**
- Instead of the last-fold approximation, GARCH, HMM, Kalman and frac-d would be refit per fold per draw.
- The Phase 2 walk-forward took 6,424 s for 96 folds. The fits are a fraction of that, but at least ~0.5 h per
  draw: **≥ 100 h for 200 draws**. Not practical. I suggest keeping the approximation and stating it.

**(c) Model gates: each gated model must score ≈ 0 when trained and tested on mirror-null data.**
- Gate 3 (LightGBM A/B/C, fixed parameters): one 96-fold walk-forward per variant per draw, plus the null
  features and tokens (~2 min per draw). The Phase 1 LightGBM walk-forward was never timed, so I can't project
  this yet. The first step would be to time one null walk-forward of model C.
- Phase 4 (Transformer): a full quarterly-retrained Phase 4 run per draw. This will be the dominant cost, and
  realistically allows only ~10–20 draws.

## 6. Status

- **D19 is recorded but fails D20.** Under the fallback, it can be used in a gate only together with the
  null-quantile comparison of §5 (a), and for models with the null run of §5 (c).
- **Phase 2 numbers are unchanged:** the IC tables, D10 passes and per-fold feature lists stand as computed
  with the old metric (biased, information only). Under every metric tried, the daily-feature D10 passes look
  like metric artefacts.
- **Tests:** `tests/test_metric_null.py` has been updated (D19, 100 seeds, zero and real drift) and passes
  (4 tests).
