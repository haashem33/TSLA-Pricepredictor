# Phase 2 re-score with D19 + D22 (D20 fallback, final)

Date 2026-10-25. Data v2, development period 2016-01..2025-12 (96 monthly out-of-sample folds 2018-01..2025-12).
Holdout and forward test not loaded. No feature, fit or model was refitted: every fold was rebuilt from its cached
Phase 2 checkpoint and only re-scored.

> **Approximation (applies to every number from the null below):** the mirror null uses the **last fold's
> (2025-12) fitted parameters for GARCH-t, HMM, Kalman and frac-diff in all folds**; each fold's feature versions
> (raw / rolling-z) come from that fold's checkpoint. The 19 features that pass D10\* (§3) are price and volume
> transforms without fitted parameters, so for them the approximation does not change the feature values.

## 1. D22 definition (decisions.md, 2026-10-25, final)

- **Metric:** D19 anchored IC v2, unchanged: c(x) = clip((x − median_train)/IQR_train, ±5), s = sign_train·(c − mean_train c),
  y = clip(z, ±5), IC = Σ s·y / (n·√(V_s·V_y)) with training moments, per monthly fold.
- **Null:** random-sign mirror null + the D21 calibrated constant drift on intraday bars (bar_index > 0), **300 new
  draws, seeds 2000..2299**. Calibration (D21 procedure on the new seeds, before any feature was scored):
  no-drift pooled mean z_12 b = −0.000340 (per-draw sd 0.0092), μ = (0.001924 − b)/1663.5 = **1.3609e-6 per bar**.
  Realised null mean z_12 over the 300 draws: **0.001924** (target 0.001924). The old 60 / 40 draws are not used.
- **Single-feature D10\*** (h = 12, with and without the one-bar delay): (a) real full-period signed t ≥ the feature's
  own null 97.7% quantile of full-period signed t; (b) real 2023–2025 mean IC > 0 and 2023–2025 t ≥ its own null
  84.1% quantile. For information: empirical p = (1 + #null draws ≥ real on both t's)/301 and BH at FDR 10% across
  the 83 features (features whose 2023–2025 t is undefined get no p and are left out of BH: hmm_p2, which has no
  valid month after mid-2022).
- **Training-side selection:** the stable-IC |t| ≥ 2 and the delayed |t| ≥ 2 are replaced by the feature's own null
  95.4% quantile of |full-period t| (h = 12 and h = 12 delayed respectively), the same for every fold and also used
  for the h = 1 and h = 3 conditions (no null was run for those); features without a null quantile keep 2.0
  (r_12th30, sign_12th30: not scorable at h = 12). Extreme-bucket rule unchanged.
- **Model gates (Gate 3, Phase 4):** K mirror-null draws per model; pass only if null mean IC within ±0.003 and
  real t ≥ null mean t + 2 null sd, in addition to D10. **Not applied retroactively to Phase 1** (§6).
- D16 rule signs and clustered MDA use D19.

## 2. Null quantiles (300 draws)

Tables: `reports/tables/phase2_rescore/null_quantiles.csv` (all 83 features × {h = 12, h = 12 delayed}),
`null_per_draw_t.csv` (per draw), `null_quantiles_summary.json`.

| check | h = 12 | h = 12 delayed |
|---|---|---|
| max \|null mean unsigned IC\| over features | 0.0016 | 0.0016 |
| pooled D10 false-pass rate, fixed thresholds (t ≥ 2; 23–25 mean > 0, t ≥ 1) | **1.80%** | **1.85%** |
| highest per-feature fixed-D10 rate | 4.0% (half_day) | 4.3% (garch_resid) |
| range of null q97.7 of full-period t | 1.68 – 2.36 | 1.66 – 2.50 |
| range of null q84.1 of 2023–25 t | 0.79 – 1.63 | 0.76 – 1.50 |
| range of null q95.4 of \|full-period t\| (selection threshold) | 1.72 – 2.33 | 1.75 – 2.25 |
| range of null sd of full-period t | 0.92 – 1.10 | 0.93 – 1.13 |

(For information: the nominal iid rate of the fixed D10 rule is about 1.67%.) Top 20 features by |null mean IC|, h = 12
(plus the delayed quantiles used by D10\* and selection):

| feature | null mean IC | se | t full q84.1 | t full q95.4 | t full q97.7 | t 23–25 q84.1 | t 23–25 q95.4 | t 23–25 q97.7 | \|t\| q95.4 | delayed t full q97.7 | delayed t 23–25 q84.1 | delayed \|t\| q95.4 | D10 fixed rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r_first30 | 0.0016 | 0.0006 | 1.02 | 1.54 | 1.77 | 0.95 | 1.68 | 1.91 | 1.77 | 1.66 | 0.95 | 1.74 | 0.010 |
| dow | -0.0015 | 0.0005 | 1.08 | 1.77 | 2.01 | 1.18 | 2.04 | 2.37 | 2.02 | 2.01 | 1.13 | 2.04 | 0.023 |
| hmm_p2 | -0.0014 | 0.0011 | 1.00 | 1.73 | 2.05 | – | – | – | 1.99 | 2.14 | – | 2.01 | 0.000 |
| sign_first30 | 0.0014 | 0.0006 | 1.00 | 1.59 | 1.88 | 0.92 | 1.45 | 1.52 | 1.91 | 1.85 | 0.91 | 1.94 | 0.013 |
| ema_dev_78 | 0.0012 | 0.0006 | 0.99 | 1.59 | 2.16 | 1.00 | 1.61 | 2.02 | 1.93 | 1.91 | 0.97 | 1.94 | 0.013 |
| mom_78 | 0.0012 | 0.0005 | 1.00 | 1.77 | 1.96 | 0.86 | 1.64 | 2.04 | 1.97 | 1.92 | 0.84 | 1.94 | 0.013 |
| linreg_slope_78 | 0.0011 | 0.0006 | 0.94 | 1.69 | 2.08 | 0.99 | 1.45 | 1.92 | 1.97 | 2.06 | 1.01 | 1.98 | 0.013 |
| linreg_slope_24 | 0.0011 | 0.0005 | 0.97 | 1.47 | 1.79 | 1.10 | 1.70 | 1.90 | 1.80 | 1.71 | 1.06 | 1.89 | 0.010 |
| ema12_26 | 0.0010 | 0.0006 | 0.99 | 1.67 | 1.94 | 1.03 | 1.65 | 1.98 | 1.92 | 1.93 | 1.05 | 1.87 | 0.013 |
| ema_dev_26 | 0.0008 | 0.0005 | 1.00 | 1.67 | 1.96 | 1.08 | 1.79 | 2.00 | 1.92 | 1.91 | 1.09 | 1.91 | 0.017 |
| earn_react | -0.0008 | 0.0006 | 0.97 | 1.59 | 1.91 | 1.14 | 1.78 | 2.07 | 2.02 | 1.91 | 1.15 | 1.98 | 0.013 |
| har_rv22 | -0.0008 | 0.0005 | 0.95 | 1.70 | 2.08 | 1.06 | 1.65 | 1.97 | 2.06 | 2.11 | 1.08 | 2.06 | 0.023 |
| mom_24 | 0.0007 | 0.0005 | 0.95 | 1.65 | 1.92 | 1.01 | 1.71 | 2.23 | 1.92 | 1.95 | 1.00 | 1.95 | 0.013 |
| opex | -0.0007 | 0.0005 | 0.89 | 1.74 | 1.93 | 0.97 | 2.03 | 2.31 | 1.93 | 1.93 | 0.95 | 1.92 | 0.017 |
| first30_x_rv | 0.0007 | 0.0006 | 1.01 | 1.81 | 2.06 | 1.20 | 1.75 | 1.95 | 2.00 | 2.05 | 1.23 | 2.00 | 0.030 |
| linreg_r2_78 | 0.0006 | 0.0005 | 1.00 | 1.64 | 2.07 | 0.84 | 1.50 | 1.66 | 2.13 | 2.08 | 0.82 | 2.15 | 0.017 |
| fracdiff_lnc | 0.0006 | 0.0005 | 1.13 | 1.73 | 1.97 | 1.20 | 1.89 | 2.10 | 2.18 | 2.14 | 1.18 | 2.21 | 0.023 |
| first30_x_relvol | 0.0006 | 0.0005 | 0.96 | 1.84 | 1.95 | 1.01 | 1.67 | 1.99 | 2.13 | 1.94 | 1.02 | 2.13 | 0.013 |
| spy_relvol | -0.0006 | 0.0004 | 0.96 | 1.78 | 2.02 | 0.87 | 1.75 | 2.00 | 2.04 | 2.06 | 0.92 | 2.05 | 0.020 |
| half_day | -0.0006 | 0.0005 | 1.15 | 1.90 | 2.12 | 1.15 | 1.72 | 1.95 | 2.08 | 2.13 | 1.11 | 2.05 | 0.040 |

## 3. D10\* at h = 12, old metric vs D19 / D22

Table: `reports/tables/phase2_rescore/d10_star_h12.csv` (D10\* detail) and `d10_old_vs_d19_d22.csv` (side by side,
all horizons). "old" = monthly-pooled Spearman (biased, information only).

| | h = 12 | h = 12 delayed |
|---|---|---|
| old metric, fixed D10 | 8 | 7 |
| D19, fixed D10 (information) | 18 | 21 |
| D19, part (a) only | 22 | 23 |
| D19, part (b) only | 33 | 29 |
| **D19 + D22, D10\*** | **19** | **19** |
| BH 10% on the empirical p (information) | 31 | 29 |

18 features pass D10\* on both; resid_qqq_z passes only without the delay (delayed 2023–25 t 0.90 < 0.97),
close_location only with the delay (no-delay full t 1.94 < 2.08). BH counts are larger than the D10\* counts
because the empirical p has a floor of 1/301 and includes features that clear one part but not the other; it is
information only.

Rows below: every feature that passes either the old D10 or D10\*.

**h = 12**

| feature | old t full | old t 23–25 | old D10 | D19 IC full | D19 t full | null q97.7 | D19 IC 23–25 | D19 t 23–25 | null q84.1 | D10* | emp p | BH 10% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rsi14 | -1.82 | -0.55 | no | 0.0364 | 4.83 | 2.08 | 0.0342 | 2.79 | 1.11 | yes | 0.0033 | yes |
| mom_78 | 1.27 | -1.77 | no | 0.0378 | 4.40 | 1.96 | 0.0120 | 0.94 | 0.86 | yes | 0.0033 | yes |
| ema_dev_78 | 0.66 | 1.11 | no | 0.0344 | 4.25 | 2.16 | 0.0203 | 1.81 | 1.00 | yes | 0.0033 | yes |
| ema12_26 | -0.60 | -0.26 | no | 0.0321 | 3.86 | 1.94 | 0.0245 | 2.17 | 1.03 | yes | 0.0033 | yes |
| bb_pctb | 0.80 | 0.21 | no | 0.0259 | 3.74 | 2.12 | 0.0273 | 2.27 | 1.09 | yes | 0.0033 | yes |
| stoch_k14 | -0.39 | 0.24 | no | 0.0250 | 3.68 | 2.36 | 0.0269 | 2.19 | 1.05 | yes | 0.0033 | yes |
| vwap_dev | -1.19 | -1.42 | no | 0.0308 | 3.56 | 2.25 | 0.0240 | 1.87 | 1.09 | yes | 0.0033 | yes |
| ema_dev_26 | -0.07 | 0.59 | no | 0.0282 | 3.47 | 1.96 | 0.0277 | 2.08 | 1.08 | yes | 0.0033 | yes |
| mom_24 | -0.64 | 0.56 | no | 0.0271 | 3.27 | 1.92 | 0.0226 | 1.79 | 1.01 | yes | 0.0033 | yes |
| mom_12 | -0.06 | 0.14 | no | 0.0224 | 2.86 | 2.16 | 0.0271 | 2.02 | 0.98 | yes | 0.0033 | yes |
| r_first30 | 1.90 | 1.32 | no | 0.0300 | 2.76 | 1.77 | 0.0264 | 2.10 | 0.95 | yes | 0.0033 | yes |
| mom_6 | 0.95 | 0.51 | no | 0.0168 | 2.53 | 2.30 | 0.0210 | 1.71 | 1.06 | yes | 0.0100 | yes |
| ema_dev_12 | 1.27 | 0.30 | no | 0.0188 | 2.42 | 2.05 | 0.0247 | 1.71 | 1.07 | yes | 0.0100 | yes |
| dL | 3.51 | 1.17 | yes | 0.0071 | 2.39 | 2.15 | 0.0112 | 1.75 | 1.02 | yes | 0.0100 | yes |
| dC | 3.88 | 0.29 | no | 0.0066 | 2.38 | 2.16 | 0.0138 | 2.70 | 1.01 | yes | 0.0033 | yes |
| mom_3 | 2.44 | 0.76 | no | 0.0120 | 2.33 | 2.29 | 0.0170 | 1.76 | 1.09 | yes | 0.0166 | yes |
| dH | 2.68 | -1.79 | no | 0.0080 | 2.32 | 2.05 | 0.0396 | 2.77 | 1.63 | yes | 0.0033 | yes |
| sign_first30 | 2.26 | 1.48 | yes | 0.0219 | 2.28 | 1.88 | 0.0218 | 1.56 | 0.92 | yes | 0.0033 | yes |
| resid_qqq_z | 1.29 | -0.35 | no | 0.0071 | 2.20 | 1.96 | 0.0131 | 2.28 | 0.87 | yes | 0.0133 | yes |
| d_mom5 | 4.94 | 2.43 | yes | 0.0117 | 1.19 | 1.98 | 0.0028 | 0.22 | 0.91 | no | 0.0930 | no |
| spy_z | 3.32 | 1.25 | yes | 0.0043 | 1.02 | 1.89 | -0.0045 | -0.80 | 0.84 | no | 0.1229 | no |
| d_atr_c | 3.10 | 1.48 | yes | 0.0087 | 0.96 | 2.17 | -0.0012 | -0.09 | 0.91 | no | 0.1495 | no |
| d_sma20_dev | 6.26 | 3.10 | yes | 0.0098 | 0.93 | 1.97 | 0.0041 | 0.31 | 1.09 | no | 0.1196 | no |
| amihud | 3.13 | 1.85 | yes | 0.0008 | 0.09 | 1.98 | 0.0103 | 0.62 | 1.10 | no | 0.2292 | no |
| d_prev_ret_atr | 2.41 | 2.77 | yes | -0.0034 | -0.41 | 1.90 | 0.0180 | 1.38 | 1.13 | no | 0.1163 | no |

**h = 12, one-bar-delayed outcome**

| feature | old t full | old t 23–25 | old D10 | D19 IC full | D19 t full | null q97.7 | D19 IC 23–25 | D19 t 23–25 | null q84.1 | D10* | emp p | BH 10% |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rsi14 | -0.37 | -0.39 | no | 0.0360 | 4.80 | 2.10 | 0.0312 | 2.68 | 1.05 | yes | 0.0033 | yes |
| mom_78 | 1.57 | -1.52 | no | 0.0378 | 4.39 | 1.92 | 0.0114 | 0.90 | 0.84 | yes | 0.0033 | yes |
| ema_dev_78 | -1.10 | -0.62 | no | 0.0349 | 4.28 | 1.91 | 0.0189 | 1.71 | 0.97 | yes | 0.0033 | yes |
| ema12_26 | -0.66 | -0.70 | no | 0.0317 | 3.77 | 1.93 | 0.0236 | 2.11 | 1.05 | yes | 0.0033 | yes |
| vwap_dev | -0.84 | -0.47 | no | 0.0315 | 3.66 | 2.21 | 0.0222 | 1.76 | 1.07 | yes | 0.0033 | yes |
| ema_dev_26 | -1.05 | 0.54 | no | 0.0286 | 3.53 | 1.91 | 0.0252 | 2.01 | 1.09 | yes | 0.0033 | yes |
| bb_pctb | 0.26 | 0.29 | no | 0.0243 | 3.49 | 2.10 | 0.0236 | 2.04 | 1.10 | yes | 0.0033 | yes |
| stoch_k14 | -1.17 | 0.10 | no | 0.0236 | 3.48 | 2.09 | 0.0230 | 1.93 | 1.09 | yes | 0.0066 | yes |
| close_location | 0.19 | 0.25 | no | 0.0088 | 3.32 | 2.33 | 0.0081 | 1.99 | 1.10 | yes | 0.0033 | yes |
| mom_24 | -1.17 | -0.92 | no | 0.0272 | 3.28 | 1.95 | 0.0227 | 1.79 | 1.00 | yes | 0.0033 | yes |
| dL | 0.44 | 1.69 | no | 0.0090 | 2.96 | 2.26 | 0.0078 | 1.27 | 1.09 | yes | 0.0066 | yes |
| mom_12 | -0.99 | 0.20 | no | 0.0215 | 2.81 | 2.09 | 0.0236 | 1.91 | 1.02 | yes | 0.0033 | yes |
| dC | 1.59 | 0.81 | no | 0.0077 | 2.79 | 2.33 | 0.0087 | 1.60 | 1.08 | yes | 0.0166 | yes |
| r_first30 | 2.20 | 1.31 | yes | 0.0308 | 2.79 | 1.66 | 0.0264 | 2.06 | 0.95 | yes | 0.0033 | yes |
| dH | -0.95 | -2.29 | no | 0.0096 | 2.57 | 1.84 | 0.0518 | 4.14 | 1.50 | yes | 0.0033 | yes |
| mom_3 | 0.37 | 0.88 | no | 0.0128 | 2.50 | 2.19 | 0.0141 | 1.53 | 1.11 | yes | 0.0133 | yes |
| ema_dev_12 | 0.42 | 0.32 | no | 0.0190 | 2.45 | 2.06 | 0.0216 | 1.57 | 1.11 | yes | 0.0066 | yes |
| mom_6 | 0.78 | 0.51 | no | 0.0159 | 2.44 | 2.19 | 0.0189 | 1.62 | 1.10 | yes | 0.0133 | yes |
| sign_first30 | 2.30 | 1.52 | yes | 0.0224 | 2.30 | 1.85 | 0.0223 | 1.56 | 0.91 | yes | 0.0033 | yes |
| d_mom5 | 4.85 | 2.27 | yes | 0.0120 | 1.21 | 1.99 | 0.0021 | 0.16 | 0.95 | no | 0.0864 | no |
| d_atr_c | 3.09 | 1.41 | yes | 0.0094 | 1.02 | 2.13 | -0.0014 | -0.11 | 0.90 | no | 0.1395 | no |
| d_sma20_dev | 6.11 | 2.91 | yes | 0.0103 | 0.96 | 2.00 | 0.0040 | 0.30 | 1.10 | no | 0.1163 | no |
| amihud | 3.11 | 1.75 | yes | 0.0013 | 0.13 | 1.95 | 0.0096 | 0.58 | 1.10 | no | 0.2193 | no |
| d_prev_ret_atr | 2.40 | 2.70 | yes | -0.0044 | -0.52 | 1.90 | 0.0182 | 1.38 | 1.12 | no | 0.1229 | no |

What changed:

- **All 6 daily-context passers of the old metric fail D10\***: d_sma20_dev (t 6.26 → 0.93), d_mom5 (4.94 → 1.19),
  d_atr_c (3.10 → 0.96), amihud (3.13 → 0.09), d_prev_ret_atr (2.41 → −0.41), and spy_z (3.32 → 1.02). This matches the
  session-permutation placebo (`reports/phase2_daily_diagnostics.md`): the old results were the within-month
  demeaning artefact.
- The same artefact pushed **intraday trend features down** under the old metric (rsi14 old t −1.82, ema12_26 −0.60,
  mom_24 −0.64, vwap_dev ≈ −0.8). Under D19 they have t 3.3–4.8 against null 97.7% quantiles of 1.9–2.2.
- All 19 passers have a training sign of +1 in (nearly) every fold, i.e. **continuation**, not reversal
  (r_first30 −1 in 4 folds, sign_first30 in 6, dH +1 in 54 of 54 defined folds).

h = 1 and h = 3 (D19 t, information only; `rescore_d19_walkforward_summary.csv`): 17 features pass the fixed D10
at h = 1 and 17 at h = 3. Top by t: h = 1 mom_78 5.53, linreg_slope_78 5.09, r_first30 4.44, rsi14 4.36,
bb_pctb 4.20; h = 3 rsi14 5.57, bb_pctb 5.20, mom_78 5.12, stoch_k14 4.96. body_ratio and close_location have
full-period t ≈ 3.9 at h = 1 but negative 2023–25 means.

**IC decay h = 1..24** (`ic_decay_d19.csv`, out-of-sample signed D19 IC). For the trend group the IC rises with the
horizon: rsi14 0.011 / 0.023 / 0.036 / 0.051 at h = 1 / 3 / 12 / 24, mom_78 0.014 / 0.023 / 0.038 / 0.052,
r_first30 0.013 / 0.019 / 0.030 / 0.040, with t roughly flat (rsi14 4.4 / 5.6 / 4.8 / 5.1). dH is non-monotonic
(0.008 / 0.001 / 0.008 / 0.013).

**Per-slot IC** (`slot_ic_d19.csv`, cross-check): the slot-averaged D19 IC gives the same ranking and size as the
pooled monthly IC (h = 12: rsi14 0.036 t 4.84, mom_78 0.038 t 4.41, ema_dev_78 0.035 t 4.27, ema12_26 0.032 t 3.88,
bb_pctb 0.026 t 3.74). The signal is not coming from one time-of-day slot.

## 4. Per-fold selection (configs/features.yaml version 2)

`configs/features.yaml` is now version 2 (D19 + D22 thresholds); the old list is kept as
`configs/features_v1_biased.yaml`. Tables: `selection_changes_by_fold.csv` (per fold: n_v1, n_v2, overlap, gained,
lost), `feature_selection_v1_vs_v2.csv` (share of folds per feature), `rescore_d19_selected_by_fold.csv`,
`mda_d19_lastfold.csv`.

| | v1 (biased) | v2 (D19 + D22) |
|---|---|---|
| features per fold, mean (min–max) | 24.9 (12–35) | 24.8 (17–32) |
| folds with < 20 features | 21 (2018-01..2020-04) | **10** (2018-01, -02, -04, -08, 2019-06..10, 2020-02; 17–19 each) |
| overlap with v1 per fold, mean (min–max) | – | 17.3 (8–24) |

Biggest gains (share of folds v1 → v2): ema_dev_78 0 → 100%, vwap_dev 6 → 100%, bb_pctb 20 → 99%,
pos_in_prev 0 → 80%, obv_slope_z 0 → 62%, entropy_78 1 → 46%, mom_24 51 → 82%, close_location 46 → 77%,
dC 0 → 31%, rsi14 0 → 22%.
Biggest losses: d_sma20_dev 100 → 20%, amihud 81 → 1%, d_atr_c 70 → 9%, kalman_innov 68 → 14%,
kalman_slope 71 → 20%, d_mom5 73 → 29%, earn_after 67 → 25%, d_prev_ret_atr 42 → 0%, mom_6 61 → 26%,
body 26 → 0%. Unchanged near 100%: macd_hist (extreme-bucket rule), mom_3, linreg_slope_78, mom_78, dL.
Every kept feature has a stated basis (IC, extreme, or both) in features.yaml.
Example, last fold 2025-12: 33 → 27 features, 20 in both. Gained bb_pctb, body_ratio, dC, ema_dev_78, obv_slope_z,
rsi14, vwap_dev. Lost amihud, d_atr_c, d_mom5, d_prev_ret_atr, d_sma20_dev, ema_dev_12, fracdiff_lnc, gap_z,
garch_resid, hmm_p0, kalman_innov, linreg_r2_24, mom_12.

## 5. Rule baselines (D16)

Table: `reports/tables/phase2_rescore/rule_d_sma20_dev_d19.csv`. Registry rows rule_d_sma20_dev (rerun_of
"rule_d_sma20_dev 2026-10-23T20:58:32 (b1dff46)") and phase2_feature_screen (rerun_of
"phase2_feature_screen 2026-10-22T22:51:40 (b2fda20)").

| rule | sign | gross Sharpe (ann.) | net 3 bp | net 6 bp | break-even bp/side | fold t full (3 bp) | 2023–25 mean / t (3 bp) | D10 |
|---|---|---|---|---|---|---|---|---|
| (a) d_sma20_dev, D19 sign | −1 in all 96 folds | 0.42 | 0.10 | −0.22 | 3.97 | 0.33 | +0.001 / 0.04 | fail |
| (a) same, +1 bar delay | | 0.51 | 0.17 | −0.16 | 4.58 | 0.56 | −0.001 / −0.06 | fail |
| (b) sign_first30, unchanged | | 0.66 | −0.72 | −2.11 | 1.43 | −1.97 | −0.31 | fail |

The D19 training sign of d_sma20_dev equals the old sign in every fold, so rule (a) trades exactly as before and its
numbers are unchanged (gross 0.42, break-even 3.97 bp/side). Neither rule passes D10 net of costs. These remain
the baselines that every later model must beat net of costs on the same folds.

## 6. Phase 1 predictions re-scored with D19

Table: `reports/tables/phase2_rescore/phase1_preds_d19.csv`. 95 folds (2018-02..2025-12): no inner-validation
predictions were stored, so the first fold 2018-01 only seeds V_s. **The model-null part of D22 is not applied
retroactively to Phase 1**; the fixed-D10 result below is information only.

| model | h | IC full | t full | IC>0 folds | IC 23–25 | t 23–25 | fixed D10 |
|---|---|---|---|---|---|---|---|
| lgb_returns | 1 | 0.0234 | 9.16 | 88% | 0.0121 | 3.99 | pass |
| lgb_returns | 3 | 0.0306 | 10.10 | 86% | 0.0255 | 6.21 | pass |
| lgb_returns | 12 | 0.0221 | 3.93 | 66% | 0.0120 | 1.49 | pass |
| hist_mean | 12 | −0.0010 | −0.16 | 52% | 0.0025 | 0.32 | fail |
| zero | 12 | 0.0059 | 0.56 | 58% | 0.0023 | 0.29 | fail |

(Old Phase 1 h = 12 fold IC of lgb_returns: 0.0106, ICIR 0.17.) The historical-mean and zero forecasts score ≈ 0, as
expected of the metric.

## 7. Which h = 12 signals remain (honest summary)

- **19 features pass D10\* at h = 12 and 19 with the delay, 18 on both.** They are not 19 independent findings. The
  correlation matrix of their monthly out-of-sample ICs has eigenvalues 11.0, 3.4, 1.3, 0.8, …, i.e. **about 2.7
  effective signals**, in three groups:
  1. **Intraday trend / oscillator continuation** (rsi14, bb_pctb, stoch_k14, ema_dev_12/26, ema12_26, mom_3..24,
     vwap_dev, and the bar-relation features dH, dL, dC, close_location; monthly IC correlations 0.6–0.95).
     IC 0.007–0.036, t 2.3–4.8, and it survives the one-bar delay. So this is not bid–ask bounce: the sign is
     continuation, which bounce would push the other way.
  2. **Day-level momentum**: mom_78, ema_dev_78, r_first30, sign_first30 (Gao et al. intraday momentum; IC 0.022–0.038).
     mom_78 is weak in 2023–25 (t 0.94, just above its null q84.1 of 0.86).
  3. **Residual vs QQQ** (resid_qqq_z): passes only without the delay; marginal.
- The multi-day reversal and illiquidity/vol-level signals reported in Phase 2 v1 (d_sma20_dev, d_mom5, amihud,
  d_atr_c, d_prev_ret_atr) **do not survive**. Their old results were the metric artefact.
- Chance passes: in the 300 null draws, **1.3 features per draw pass D10\* on average** (per-feature rate 1.6%). The
  most in one draw is 12 at h = 12 and 13 delayed, because correlated features pass together. **No null draw reaches 19.**
  All 19 real passers have full-period t ≥ null mean t + 2 null sd. The empirical p is at its floor (1/301) for 14
  of the 19 at h = 12 and 13 of the 19 delayed.
- Sizes are small: D19 IC ≈ 0.02–0.04 on clipped scaled 60-minute returns. These are single-feature screens. Whether
  they add up to a tradable edge net of costs is for Gate 3 / Phase 4, which use the D22 model-null requirement.

## 8. Gate 2 (re-confirmed)

| requirement | result |
|---|---|
| IC tables, condprob tables, charts, pair heatmaps | done (`reports/phase2_features.md`); IC tables now D19 (this report) |
| BH survivors reported | unchanged (extreme-bucket cells not re-scored): per fold h = 1 / 3 / 12: 138 / 29 / 0.8; delayed h = 1 / 3: 60 / 4.3 |
| 20–40 features selected per fold, each with a reason | 86 of 96 folds 20–32; **10 folds 17–19**, accepted per D13 (not padded); every feature has a basis (IC / extreme / both) |
| single features passing D10\* at h = 12 | 19 (19 with the delay; 18 on both), ≈ 2.7 effective signals |
| metric null | 300-draw mirror null with calibrated drift: max \|null mean IC\| 0.0016, pooled fixed-D10 false-pass rate 1.80% / 1.85% |

**Gate 2 is re-confirmed under D19 + D22, and Phase 2 is complete.** Phase 3 has not been started.

Total fold re-score time 15,151 s (96 folds; fold timings in the pickles under `data/processed/phase2_rescore_v2/`).
The 300-draw null took about 4.5 h on 5 workers.

## 9. Files

Scripts: `scripts/phase2_metric_null_v2.py --calibrate-fallback` / `--variant fallback --draws 0 299`,
`scripts/phase2_null_quantiles.py`, `scripts/phase2_rescore_v2.py --folds-only` / `--summarize`.
Tables in `reports/tables/phase2_rescore/`: null_fallback_calibration.json, null_per_draw_t.csv, null_quantiles.csv,
null_quantiles_summary.json, d10_star_h12.csv, d10_old_vs_d19_d22.csv, rescore_d19_walkforward_summary.csv,
rescore_d19_oos_signed_ic_by_fold.csv, ic_decay_d19.csv, slot_ic_d19.csv, selection_changes_by_fold.csv,
feature_selection_v1_vs_v2.csv, rescore_d19_selected_by_fold.csv, mda_d19_lastfold.csv, rule_d_sma20_dev_d19.csv,
phase1_preds_d19.csv, rescore_d19_summary.json.
