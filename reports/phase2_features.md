# Phase 2 — Features and selection (Gate 2 report)

Data v2. Walk-forward with 96 monthly folds (validation months 2018-01 .. 2025-12). Every fold refits on its own
training window (sessions before the month): GARCH-t, HMM, Kalman LLT MLE, frac-diff d, ADF/KPSS versioning,
clustering, clustered MDA, conditional up-probability cells + BH, and the feature list. Holdout (2026-01-01..)
and forward test were not loaded. Run time 6,424 s of fold compute (projection was 6,738 s). Script:
`scripts/run_phase2_walkforward.py`; per-fold checkpoints in `data/processed/phase2_folds/`. Registry row:
`phase2_feature_screen`. Tests: 48 pass (leakage over all 85 features with 50 random truncations, HMM, Kalman,
condprob).

**"Out-of-sample IC" below** = the 10.6 definition applied to a single feature used as a forecast: Spearman of
the feature against realized z in the validation month M, pooled over the month, multiplied by the sign of that
feature's IC on fold M's training months. Feature values in month M come from fold M's fitted objects. t =
mean/std × √n_folds. "Delay" = same with the one-bar-delayed outcome (z from t+1 to t+1+h).

## 1. Kalman local-linear-trend fits per fold

Table: `reports/tables/phase2_fitted_params_by_fold.csv` (one row per fold: σ² of irregular/level/trend, their
ratio to var(Δln C), bound flags, log-lik, convergence, plus GARCH/HMM/frac-d).

| | irregular | level | trend |
|---|---|---|---|
| share of var(Δy), fold 2018-01 | 0.024 | 0.952 | ≈ 0 |
| share of var(Δy), fold 2025-12 | 5e-5 | 0.9999 | ≈ 0 |
| folds at the lower optimizer bound (log var(Δy) − 30) | 0 | 0 | 0 |
| folds at the upper bound (log var(Δy) + 5) | 0 | 0 | 0 |
| folds "near zero" (σ² < 1e-8 × var(Δy)) | 0 | 0 | **89 / 96** |
| not converged | 0 / 96 | | |

The MLE puts almost all variance into the level (random walk) and drives the **trend variance to ~0 in 89 of 96
folds** without hitting the hard bound. Hence `kalman_slope` is close to a very slow fixed-gain drift estimate
and `kalman_innov` is close to the scaled 1-bar return. Neither is a bug, but the slope carries little that a
long momentum window does not already carry (it clusters with momentum features in most folds).

Other fits (same table): GARCH-t α 0.062–0.069, β 0.916–0.930, ν 6.0–6.6; HMM u-variances ordered low→high in
every fold (e.g. 2.01 / 2.98 / 6.87 in 2025-12), log-lik spread across the 5 EM starts ≤ 0.15; frac-diff d varies
between 0.15 and 0.70 across folds.

## 2. Selection is per fold

`configs/features.yaml` holds **one list per validation month** (`by_fold`), with the feature version (raw or
rolling-z) and basis. Phase 4 must use `by_fold[<month>]` for each fold; no list chosen across folds is provided.
Long form: `reports/tables/phase2_selected_by_fold.csv`.

- Features per fold: mean 24.9, median 25, min 12, max 35. By year: 2018 14.5, 2019 18.8, 2020 20.9, 2021 25.3,
  2022 25.2, 2023 29.0, 2024 32.1, 2025 33.8.
- **21 folds (2018-01 .. 2020-04) have fewer than 20 features**, below the 20–40 target. The list was not padded:
  with 2–4 years of training data, fewer features pass the stability rules. No fold exceeds 40.
- 55 distinct features were selected in at least one fold (of 85 built).

### Per-feature table (h = 12, out-of-sample)

Share = share of the 96 folds that selected the feature. IC/t = out-of-sample signed IC at h = 12 (full = all
folds with the feature; 23–25 = 2023–2025 folds). Shape = modal h = 12 condprob shape across folds (every
feature is "none" at h = 12 because almost no h = 12 cell survives BH, see §3). Reason = the reason in the
feature's last selected fold (IC = stable training IC; ext = extreme-bucket rule; h1/h3 lifts in pp).

| feature | share | IC full | t full | IC 23–25 | t 23–25 | D10 | delay t | reason kept (last selected fold) |
|---|---|---|---|---|---|---|---|---|
| d_sma20_dev | 1.00 | 0.0594 | 6.26 | 0.0441 | 3.10 | **yes** | 6.11 | IC (h1/h3/h12 t −7.4/−8.0/−7.8) |
| macd_hist | 1.00 | 0.0174 | 2.14 | 0.0118 | 0.85 | no | 1.73 | IC h1/h3 + ext (h1 D1 +2.5pp) |
| mom_3 | 0.99 | 0.0094 | 2.44 | 0.0054 | 0.76 | no | 0.37 | IC h1 + ext (h1 B2 +5.2pp) |
| linreg_slope_78 | 0.96 | 0.0071 | 0.89 | −0.0264 | −2.28 | no | 1.02 | ext (h1 B2 −3.0pp) |
| dL | 0.93 | 0.0086 | 3.51 | 0.0050 | 1.17 | **yes** | 0.44 | IC h1 + ext (h1 B2 +3.6pp) |
| mom_78 | 0.92 | 0.0096 | 1.27 | −0.0180 | −1.77 | no | 1.57 | ext (h1 T5 +1.5pp) |
| r_first30 | 0.85 | 0.0168 | 1.90 | 0.0145 | 1.32 | no (delay: yes) | 2.20 | IC h1/h3/h12 + ext |
| spy_mom12 | 0.81 | 0.0094 | 1.24 | −0.0052 | −0.48 | no | 0.91 | IC h1/h3 + ext |
| amihud | 0.81 | 0.0306 | 3.13 | 0.0295 | 1.85 | **yes** | 3.11 | IC h3/h12 |
| stoch_k14 | 0.74 | −0.0025 | −0.39 | 0.0026 | 0.24 | no | −1.17 | ext (h1 B2 +4.4pp, h3 B2 +5.1pp) |
| d_mom5 | 0.73 | 0.0438 | 4.94 | 0.0316 | 2.43 | **yes** | 4.85 | IC h1/h3/h12 |
| kalman_slope | 0.71 | 0.0089 | 1.10 | 0.0152 | 1.20 | no | 0.90 | ext (h1 D1 −1.3pp) |
| qqq_mom12 | 0.71 | 0.0113 | 1.53 | 0.0032 | 0.28 | no | 1.24 | IC h1/h3 + ext |
| d_atr_c | 0.70 | 0.0299 | 3.10 | 0.0220 | 1.48 | **yes** | 3.09 | IC h3/h12 |
| kalman_innov | 0.68 | 0.0097 | 4.14 | 0.0018 | 0.43 | no | 2.28 | IC h1/h3/h12 + ext |
| earn_after | 0.67 | 0.0114 | 0.66 | 0.0072 | 0.29 | no | 0.60 | IC h3 |
| bar_index | 0.66 | −0.0127 | −1.65 | −0.0067 | −0.58 | no | −1.36 | ext (h1 T5 +2.5pp) |
| spy_z | 0.65 | 0.0128 | 3.32 | 0.0067 | 1.25 | **yes** | 2.15 | IC h1/h3/h12 + ext |
| ema_dev_12 | 0.64 | 0.0075 | 1.27 | 0.0030 | 0.30 | no | 0.42 | IC h1 + ext |
| beta_qqq | 0.63 | 0.0062 | 0.64 | 0.0187 | 1.35 | no | 0.49 | ext (h1 B2 −2.6pp) |
| minutes_to_close | 0.61 | −0.0118 | −1.61 | −0.0227 | −2.14 | no | −1.45 | ext (h1 B5 +2.5pp) |
| dow | 0.61 | 0.0181 | 1.92 | 0.0082 | 0.49 | no | 1.88 | IC h1/h3 |
| mom_6 | 0.61 | 0.0051 | 0.95 | 0.0048 | 0.51 | no | 0.78 | ext (h1 B5 +3.2pp, h3 B2 +4.0pp) |
| resid_qqq_z | 0.59 | 0.0035 | 1.29 | −0.0016 | −0.35 | no | −0.04 | ext (h1 D1 +1.4pp) |
| dH | 0.56 | 0.0088 | 2.68 | −0.0293 | −1.79 | no | −0.95 | IC h1/h3 + ext |
| sign_first30 | 0.52 | 0.0209 | 2.26 | 0.0191 | 1.48 | **yes** | 2.30 | IC h1/h3/h12 |
| fracdiff_lnc | 0.52 | 0.0189 | 2.55 | 0.0082 | 0.92 | no | 2.26 | IC h1/h3 + ext |
| body_ratio | 0.51 | 0.0118 | 4.68 | 0.0024 | 0.53 | no | 1.58 | IC h1/h3/h12 + ext |
| mom_24 | 0.51 | −0.0044 | −0.64 | 0.0061 | 0.56 | no | −1.17 | ext (h3 B5 +2.7pp) |
| close_location | 0.46 | 0.0054 | 2.13 | −0.0017 | −0.47 | no | 0.19 | ext (h1 B2 +5.8pp, T2 −5.2pp) |
| first30_x_rv | 0.44 | −0.0137 | −1.41 | −0.0154 | −1.07 | no | −1.69 | ext (h1 B2 −3.0pp) |
| d_prev_ret_atr | 0.42 | 0.0205 | 2.41 | 0.0396 | 2.77 | **yes** | 2.40 | IC h3/h12 |
| linreg_r2_24 | 0.40 | 0.0074 | 0.96 | −0.0032 | −0.32 | no | 0.89 | IC h3 |
| first30_x_relvol | 0.32 | −0.0097 | −0.98 | −0.0192 | −1.39 | no | −1.20 | ext (h1 B2 −3.2pp) |
| gap_z | 0.32 | −0.0008 | −0.09 | 0.0136 | 0.96 | no | 0.07 | IC h1 |
| qqq_z | 0.28 | 0.0114 | 2.45 | 0.0041 | 0.23 | no | 1.67 | IC h1/h3/h12 + ext |
| body | 0.26 | 0.0102 | 4.22 | 0.0024 | 0.53 | no | 2.05 | IC h1/h3/h12 + ext |
| bb_pctb | 0.20 | 0.0050 | 0.80 | 0.0021 | 0.21 | no | 0.26 | ext (h1 B2 +3.7pp) |
| lower_shadow | 0.17 | −0.0083 | −1.82 | −0.0112 | −1.70 | no | 0.57 | ext (h1 B5 +1.4pp) |
| earn_react | 0.16 | 0.0230 | 1.11 | 0.0060 | 0.16 | no | 1.10 | IC h1/h3/h12 |
| hmm_p0 | 0.13 | 0.0091 | 0.78 | −0.0058 | −0.27 | no | 0.87 | IC h1 |
| opex | 0.11 | 0.0069 | 0.91 | −0.0026 | −0.28 | no | 0.82 | IC h12 |
| garch_resid | 0.09 | 0.0099 | 4.12 | 0.0025 | 0.59 | no | 1.93 | IC h1/h3/h12 + ext |
| (12 more, each selected in ≤ 6 folds) | | | | | | | | see CSV |

Full table: `reports/tables/phase2_feature_selection_summary.csv` (also basis counts across folds and the
h = 12 shape in the last fold). `spy_z`, `dH`, `qqq_z` have fewer than 96 out-of-sample folds (62 / 54 / 27):
in the other folds both their raw and rolling-z versions failed ADF/KPSS and they were dropped there.

Most "stable IC" reasons are at h = 1/3 and are **reversal** signals (negative IC) of bar-level returns
(body, dC, kalman_innov, garch_resid, spy_z …). Their h = 12 out-of-sample IC is small (≈ 0.01) and close to zero
in 2023–2025.

## 3. Benjamini–Hochberg survivors by horizon (per fold, FDR 10%)

Cells per fold ≈ 1,015–1,025 per horizon (85 features × 14 buckets, categorical by value). Table:
`reports/tables/phase2_bh_counts_by_fold.csv`.

| outcome | h | mean per fold | min | median | max |
|---|---|---|---|---|---|
| no delay | 1 | 138.4 | 72 | 142.5 | 185 |
| no delay | 3 | 29.1 | 3 | 33.5 | 56 |
| no delay | 12 | **0.8** | 0 | 0 | 3 |
| one-bar delay | 1 | 59.9 | 0 | 73 | 116 |
| one-bar delay | 3 | 4.3 | 0 | 5 | 12 |

Mean per fold by training-window end year:

| year | h1 | h3 | h12 | h1 delayed | h3 delayed |
|---|---|---|---|---|---|
| 2018 | 84.3 | 5.9 | 0.0 | 1.2 | 0.0 |
| 2019 | 97.9 | 9.1 | 0.0 | 3.4 | 0.0 |
| 2020 | 125.5 | 23.3 | 0.0 | 27.3 | 0.2 |
| 2021 | 142.9 | 46.2 | 0.1 | 79.1 | 6.2 |
| 2022 | 156.3 | 47.5 | 1.0 | 78.6 | 8.0 |
| 2023 | 156.9 | 38.9 | 2.0 | 82.2 | 6.0 |
| 2024 | 175.2 | 33.9 | 2.2 | 105.8 | 6.8 |
| 2025 | 168.1 | 28.0 | 0.9 | 101.2 | 7.0 |

Overlap (summed over folds, `phase2_bh_bounce_overlap_by_fold.csv`): of 13,286 no-delay h = 1 survivors, 4,734
(36%) also survive with the delayed outcome; at h = 3, 390 of 2,795 (14%). Most short-horizon cells are 1-bar
reversal effects that vanish with one bar of delay (bid–ask bounce / microstructure, 6.6 caution). With training
windows ending in 2018–2019 almost nothing survives the delay at all.

**h = 12: 74 surviving (fold, cell) pairs in total, all 74 hold under the day-block bootstrap** (1,000
draws of training sessions; the 95% bootstrap interval of the lift excludes 0 with the same sign).
`reports/tables/phase2_h12_bh_cells_bootstrap_by_fold.csv`. They are three cells:

| feature | bucket | folds surviving | first – last | mean lift | example (fold 2025-04) |
|---|---|---|---|---|---|
| mom_78 | D10 | 37 | 2021-08 – 2025-05 | +4.7pp | p̂ 0.546 vs 0.504, n_eff 1,155, boot [+2.2, +6.3]pp |
| mom_78 | T5 | 21 | 2022-10 – 2025-04 | +6.3pp | p̂ 0.563, n_eff 564, boot [+3.0, +8.9]pp |
| linreg_slope_78 | D10 | 16 | 2022-04 – 2025-04 | +4.0pp | p̂ 0.541, n_eff 1,367, boot [+1.7, +5.6]pp |

These are **in-sample (training-window)** cells: high 78-bar (one-day) momentum → higher P(up) over the next
hour, with year-sign share 0.89–1.0. They do not carry out of sample: mom_78 and linreg_slope_78 have
out-of-sample h = 12 IC of −0.018 (t −1.8) and −0.026 (t −2.3) in 2023–2025, and no h = 12 cell survives in any
fold after 2025-05.

## 4. Single-feature D10 check at h = 12

> **Metric fix 2026-10-24:** the D17 anchored IC was tested against the D18 null and **failed** (see
> `reports/phase2_metric_fix.md`). It cuts the bias about 4× but does not remove it for persistent features. The numbers below
> are the old monthly-pooled Spearman (biased, information only) and have not been re-scored.
>
> **2026-10-24 (v2):** the corrected metric D19 also failed the pre-registered null criteria D20 (mirror null with drift),
> see `reports/phase2_metric_fix_v2.md`. Under D19 the h = 12 signed IC of d_sma20_dev is +0.005 (t 0.5). The D20 fallback applies.
>
> **Amended 2026-10-23:** a session-permutation placebo shows that the within-month IC of d_sma20_dev, d_mom5
> and d_atr_c is a mechanical bias of the monthly-pooled metric for multi-day-persistent features (real IC inside
> the placebo band); amihud is suspect for the same reason. See `reports/phase2_daily_diagnostics.md`. The D10
> passes below for these features are not evidence of predictability.

D10: full-period t ≥ 2 **and** 2023–2025 mean IC > 0 with t ≥ 1. Tested on 83 features with h = 12
out-of-sample IC (`reports/tables/phase2_walkforward_summary.csv`, column `d10_pass`).

**8 features pass at h = 12**; 7 pass with the one-bar-delayed outcome.

| feature | IC full | t full | binom p (IC>0 folds) | IC 23–25 | t 23–25 | IC last 24 | t last 24 | delay: t full / t 23–25 |
|---|---|---|---|---|---|---|---|---|
| d_sma20_dev | 0.0594 | 6.26 | 0.0002 | 0.0441 | 3.10 | 0.0495 | 3.37 | 6.11 / 2.91 |
| d_mom5 | 0.0438 | 4.94 | 0.0003 | 0.0316 | 2.43 | 0.0346 | 2.28 | 4.85 / 2.27 |
| dL | 0.0086 | 3.51 | 0.045 | 0.0050 | 1.17 | 0.0062 | 1.30 | 0.44 / – (fails) |
| spy_z (62 folds) | 0.0128 | 3.32 | 0.008 | 0.0067 | 1.25 | 0.0067 | 1.25 | 2.15 / – (fails) |
| amihud | 0.0306 | 3.13 | 0.041 | 0.0295 | 1.85 | 0.0215 | 1.13 | 3.11 / 1.75 |
| d_atr_c | 0.0299 | 3.10 | 0.012 | 0.0220 | 1.48 | 0.0116 | 0.71 | 3.09 / 1.41 |
| d_prev_ret_atr | 0.0205 | 2.41 | 0.305 | 0.0396 | 2.77 | 0.0317 | 1.92 | 2.40 / 2.70 |
| sign_first30 | 0.0209 | 2.26 | 0.003 | 0.0191 | 1.48 | 0.0391 | 2.73 | 2.30 / 1.52 |

Delayed-outcome passers: d_sma20_dev, d_mom5, amihud, d_atr_c, d_prev_ret_atr, sign_first30, r_first30
(r_first30 does not pass without delay: t 1.90). dL and spy_z fail with the delay: they are 1-bar reversal.

Interpretation and caveats:

- The strongest are **daily-context features** (known at the open, constant within the session in raw form):
  d_sma20_dev and d_mom5 with a negative training sign, i.e. multi-day **reversal** (TSLA below its 20-day mean →
  higher scaled return over the next hour). They are in the same cluster and their monthly out-of-sample ICs
  correlate 0.77. amihud and d_atr_c (positive sign; illiquidity / daily-range level) have ICs that correlate
  0.4–0.6 with d_sma20_dev across months, so the 8 passes are **not 8 independent pieces of evidence**:
  roughly three groups (daily reversal + vol/illiquidity level, first-half-hour momentum, bar-level reversal).
- Multiple testing: 83 features were tested and the sign is chosen per fold from training data. Under the null,
  about 2 features would reach t ≥ 2 by chance. A Bonferroni-style bar (one-sided 5% / 83, t ≈ 3.0) is cleared
  by d_sma20_dev, d_mom5, dL, spy_z, amihud and d_atr_c; dL and spy_z fail the delay test.
- A daily feature has only ~21 distinct values per month, so each monthly IC is noisy; the t-stat across 96
  independent months accounts for this, but the effective sample is ~2,000 sessions, not ~190,000 bars.
- These are single-feature screens, not a model result. Gate 3/4 decide whether any of this survives in a model.

## 5. Other Phase 2 deliverables (last fold = training 2016-01 .. 2025-11, unless noted)

- **Stationarity** (`phase2_stationarity_lastfold.csv`): of 77 continuous features, 45 pass raw, 26 pass after a
  rolling z-score (1,560 bars), 6 dropped (dH, dL, pos_in_prev, mom_3, r_12th30, hmm_p2 in that fold). The
  version is decided per fold, so a feature dropped in one fold can be used in another.
- **Clusters** (|Spearman| distance, average linkage, threshold 0.7): 53 clusters in the last fold
  (`phase2_clusters_lastfold.csv`).
- **Clustered MDA** (LightGBM, purged 5-fold CV, permute whole clusters; `phase2_mda_lastfold.csv`): CV IC of the
  full model 0.027 / 0.032 / 0.023 at h = 1/3/12. Top h = 12 clusters by MDA t: first30_x_rv (t 5.3), the short
  momentum/oscillator cluster (2.8), the 24-bar trend cluster (2.5), dow (2.3), hurst_256 (2.0).
- **IC decay** h = 1..24: `phase2_ic_decay.csv` / `reports/figures/phase2_ic_decay.png`.
- **Per-time-slot IC** (cross-check): `phase2_slot_ic_h{1,3,12}.csv`, `phase2_slot_ic_summary.csv`.
- **Condprob tables and charts**: 472 tables in `reports/tables/condprob/` (`<feature>_h<h>.csv` and
  `_by_year.csv`) and 257 figures in `reports/figures/condprob/` (bars with Wilson intervals, pair heatmaps).
  Bootstrap of extreme buckets: `phase2_bootstrap_extremes.csv`; AUC and MI: `phase2_auc_mi_lastfold.csv`.
- **Intraday momentum (Gao et al.)**, `phase2_intraday_momentum.csv`, last-half-hour P(up) by sign of r_first30,
  2016–2025: 0.544 after a positive first half hour (n 1,275 days) vs 0.515 after a negative one (n 1,189); lifts
  +1.4 / −1.5pp, Wilson intervals overlap (0.517–0.571 vs 0.486–0.543). Same sign as the paper in 8 of 10 years
  (reversed in 2016 and 2023). Mean last-30-min return 4.4 bp vs 0.3 bp. Weak, not significant on its own; the
  related 12th-half-hour sign gives a similar result. `sign_first30` passes D10 at h = 12 (above).
- **Pair grids** (`phase2_pair_grids.csv`): 465 sufficient cells (≥ 500 bars), 26 survive BH across all pair
  cells. All survivors are at h = 1 (QQQ z × residual 8, relvol × RET group 6, VWAP dev × TOD 5, mom_12 × HMM 3,
  RSI × VREG-σ̂ 2) except two at h = 3; none at h = 12. Largest |interaction| (cell lift minus the two single
  lifts) is ≤ 4.3pp, in the gap × first-hour grid, which has no BH survivor.
- **VREG**: both definitions were computed (σ̂ percentile and realized-level percentile within 20 sessions). The
  RSI × VREG grid is reported for both; σ̂ percentile gives 2 h = 1 survivors, level percentile none. The
  tokenizer (Phase 3) will use the σ̂ percentile per 7.2 unless you prefer otherwise.

## 6. Gate 2 status

| requirement | result |
|---|---|
| IC tables, condprob tables, charts, pair heatmaps | done (§2–5) |
| BH survivors reported | yes; h = 1 / 3 / 12: 138 / 29 / 0.8 per fold; delayed h = 1 / 3: 60 / 4.3 |
| 20–40 features selected per fold with reason | 75 folds yes; **21 folds (2018-01..2020-04) have 12–19** |
| single feature passing D10 at h = 12 | 8 (6 also pass the delay and the ~3.0 multiplicity bar is cleared by 4 of those: d_sma20_dev, d_mom5, amihud, d_atr_c) |

## 7. Deviations and caveats

- `hmmlearn` has no Python 3.14 wheel and could not be built here; the HMM is our own numba implementation
  (log-space recursions, ≥ 5 EM starts, variance floors, states ordered by variance after each fit), tested for
  monotone EM log-lik, recovery of a known 3-state model, and filtered-probability causality.
- The Kalman LLT is our own numba filter with exact diffuse start, fit by L-BFGS-B MLE; it matches a 50-digit
  reference to 1e-8 (statsmodels drifts ~5e-5 because it freezes P at steady state). Trend variance → ~0 (§1).
- Hurst and VR use de-seasonalized returns u.
- BH at h = 12 uses n_eff = n/12 and is conservative; h = 12 cells almost never survive.
- Fewer than 20 features in the 21 earliest folds (§2).
- A few ruff style warnings remain in the new modules (no functional effect).

**Proposed next step (Phase 3, on your go-ahead):** build the tokenizer (7.1–7.4) with per-fold edges/k-means,
re-run the token condprob tables, and run Gate 3 (token-only LightGBM, D10 at h = 3 and h = 12).
