# Decisions log

Each entry: date, decision, reason. This log wins over CLAUDE.md if they ever disagree (CLAUDE.md rule 12).

## 2026-10-22 — after Phase 1 review (approved by the user)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D1 | ~~**Primary IC** = Spearman of predicted z vs realized z, pooled within each monthly test fold.~~ **Amended by D17, then by D19 (anchored IC v2).** Fold structure, ICIR, t and binomial p are kept. ICIR = mean/std across folds; also t = ICIR·√n_folds and a one-sided binomial p-value for the share of folds with IC > 0. Cross-check: per-time-slot IC across days. The per-day IC is retired. | The per-day IC of one stock ranks bars over time within a day and is biased by overlapping windows: on a random walk, 78-bar momentum scores −0.31 at h = 12. Ranking E[r] = ẑ·S rewards volatility ranking. | §1, §6.8, §7.4, §10.6, §12, §13 |
| D2 | **Primary trading horizon h = 12** (60 min); h = 1 and h = 3 remain training targets. Every backtest reports the break-even cost per side (net Sharpe = 0) and results at 0.5, 1, 3, 6 bp per side. | Short horizons are dominated by costs and microstructure. | §1, §10.4, §12 |
| D3 | **Recency:** all metrics for the full period and for 2023–2025 separately; ~~every gate must also hold on the most recent 24 folds~~ (gate part replaced by D10). Phase 4 ablation: expanding vs rolling 3-year vs recency-weighted training window. | The Phase 1 baseline IC decays to about 0 in 2024–2025. | §1, §10.1, §10.6, §12 |
| D4 | **Model comparisons** (Phase 4+): paired fold-level tests (share of folds won, Wilcoxon on fold ICs) and Diebold–Mariano on CRPS/QLIKE with day-clustered errors. | Raw averages over 96 noisy folds are not evidence. | §8.3, §9.4, §10.6, §12 |
| D5 | **EWMA × seasonal** stays the scale for tokens and labels. Phase 2 adds GARCH-t σ and HAR features (RV over the last 1, 5, 22 sessions). The volatility head must beat GARCH-t QLIKE 1.307 / 0.521 / 0.248 (h = 1 / 3 / 12) — numbers updated to data v2 in D9: 1.435 / 0.522 / 0.248. | GARCH-t is only marginally better than EWMA (1–3%); keep the simple scale and use GARCH as a feature and benchmark. | §1, §6.4, §9.3, §10.3 |
| D6 | **Inner validation:** in each fold, the last 2 months of the training window (purged and embargoed) are used for early stopping, calibration, temperature and hyperparameters. The test month is never used for a choice. | Prevents selection on the test month. The Phase 1 LightGBM used no early stopping or tuning (fixed parameters), so **no re-run is needed** (see `phase1_checks.md`, B4). | §0 rule 11, §8.2, §9.3, §10.1 |
| D7 | **Holdout** 2026-01-01..2026-10-09 stays sealed. **Second sealed forward test** on data after 2026-10-09, downloaded only at the end. | The holdout may have been seen in the earlier `tsla-diffusion` project. | §0 rule 9, §1, §10.5, §13 |
| D8 | **GTX 1070:** FP32 only (no bf16/fp16). Neural models retrain quarterly with warm starts from the previous fold; trees stay monthly. 5-seed ensemble only for the final configuration. Time the first training run and report projected Phase 4 hours before starting it. | Pascal GPU: no bf16, slow fp16; compute budget. | §1, §2, §8.2, §9.3 |

## 2026-10-22 — data v2 and recency rule (approved by the user)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D9 | **P1 approved: data v2.** TSLA 2016-01..2022-08 is rebuilt from unadjusted Alpaca SIP 1-min bars (`Adjustment.RAW`, `data/raw/TSLA_raw/`), divided by the cumulative split factor in floating point (no rounding), volumes × factor. The bar-0 returns on 2020-08-31 and 2022-08-25 are checked explicitly against ln(O_raw / (C_prev_raw / ratio)). From 2022-09 the vendor files are used unchanged (raw == adjusted; verified bar-for-bar on 2022-10 and 2023-03). SPY/QQQ unchanged. v2 is the default (`configs/data.yaml: data_version`); v1 is kept in `data/interim/v1/` for comparison. Phase 1 baselines were re-run on v2 and logged as re-runs (`rerun_of`), not new trials. **Consequence for D5:** the volatility-head target is now the v2 GARCH-t QLIKE **1.435 / 0.522 / 0.248** (h = 1 / 3 / 12). h = 1 rose from 1.307 mainly because QLIKE drops RV = 0 rows and v1 had 3.5% artificial zeros (v2: 0.3%); on common rows v1 and v2 give 1.307 vs 1.326. | v1 prices were rounded to $0.01 (18% exact-zero 5-min returns in 2016). | §1 (price adjustment), §9.3, §10.3 |
| D10 | **Recency check (replaces the gate part of D3).** *(IC per D19 since 2026-10-24; D17 superseded.)* A gate passes the recency check if (a) the full-period result passes with **t ≥ 2** and (b) over **2023–2025 (36 folds)** the mean IC is **> 0 with t ≥ 1**. Last-24-fold numbers are still reported, for information only. | At h = 12 the per-fold IC sd is about 0.06–0.075 (0.062 over all 96 folds, 0.073 over the last 24), so 24 folds reach t = 2 only for IC ≳ 0.026–0.03. That is above the ≈ 0.021 IC needed to break even at 3 bp per side, so the old rule would reject real edges. | §1, §7.4, §10.6, §12 |

## 2026-10-22 — before Phase 2 (approved by the user)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D11 | **Volatility-head comparison (refines D5/D9).** The Phase 4 volatility head is compared with GARCH-t×seasonal **on identical rows** (RV > 0 and no forward-filled bar in the label window), **fold by fold**, with the Diebold–Mariano test on the QLIKE loss differential (day-clustered errors, D4). The v2 QLIKE values 1.435 / 0.522 / 0.248 (h = 1 / 3 / 12) are **reference values only, not a fixed target**. Robustness check: also report QLIKE with RV floored at the 1st percentile of the training-fold RV. | h = 1 QLIKE is dominated by a few tiny-RV rows (D9: the RV = 0 exclusion alone moved it from 1.307 to 1.435), so a fixed number on a different row set is not a fair target. | §1, §9.3, §10.3, §12 |
| D12 | **Missing market sessions.** On TSLA sessions where QQQ or SPY bars are missing (QQQ 2018-05-02/03, SPY 2019-08-12, and any others found), **keep the TSLA rows** and set the market features and tokens to **MISSING**. The same applies to forward-filled market bars, and to the market bar-0 return after a missing market session, because it spans more than one night. | Dropping TSLA sessions for a market-data gap loses target data. MISSING is already a token for every field. | §6.7, §7.2 |

## 2026-10-23 — after Phase 2 review (approved by the user)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D13 | **Short feature lists are accepted.** Folds with fewer than 20 selected features (Phase 2: 21 folds, 2018-01..2020-04, 12–19 features) are used as they are. Lists are **never padded** to reach 20. | Padding would add features that failed the stability rules on that fold's training data. | §6.8, §12 |
| D14 | **VREG = percentile of σ̂_t** within its trailing 20-session history, as specified in §7.2. This is fixed by the spec, **not chosen from Phase 2 results** (Phase 2 computed both the σ̂ and the realized-level percentile). | Avoids a data-driven choice among tokenizer definitions. | §7.2 |
| D15 | **Gate 3 redefined.** *(IC per D19 since 2026-10-24; D17 superseded.)* Per fold, three LightGBM variants with the **fixed Phase 1 parameters (no tuning)**: **A** = tokens only (last 24 bars); **B** = that fold's `by_fold` continuous features only; **C** = tokens + that fold's `by_fold` features. Score h = 12 (primary), h = 1 and h = 3, each with and without the one-bar-delayed outcome. **Gate 3 passes if C passes D10 at h = 12 with the delayed outcome.** The value of the tokens is reported as C vs B (share of folds won, Wilcoxon on fold ICs). **If A fails and C is not better than B, stop and discuss before Phase 4.** | Phase 2 found single features (daily context) that pass D10 at h = 12 on their own; a tokens-only gate would ignore them, and a gate on C alone would not show whether tokens add anything. | §7.4, §12 |
| D16 | **Single-feature rule baselines** *(sign_train per D19 since 2026-10-24; D17 superseded)* (Phase 2 follow-up B4: d_sma20_dev one position per day 09:35 → close; sign_first30 last half-hour as in Gao et al.; sign from each fold's training IC, unit position) are added to the §10.3 baselines. **Every later model must beat them** (net, same costs, same folds). | A model that does not beat a one-feature rule adds nothing tradable. | §10.3 |

## 2026-10-23 — metric fix after the daily-signal diagnostics (approved by the user, option 1)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D17 | **Superseded by D19 on 2026-10-24 (fails D18, `phase2_metric_fix.md`).** ~~**Anchored IC replaces the monthly-pooled Spearman as the primary metric** (amends D1; D10, D15 and D16 now use it). Nothing is centred, ranked or demeaned with test-month statistics. Per fold: **score** s_t = for a model, its predicted z as is (zero = no position); for a single feature, sign_train × (x_t − median_train(x)) / IQR_train(x), clipped to ±5 (training sd if IQR = 0; for binary flags x − training mean); sign_train, median, IQR and sd come from the fold's training rows only. **Target** y_t = clip(z_t, −5, 5) − m, m = mean of clipped z over the fold's training rows. **Anchored IC** of the test month = Σ s·y / √(Σ s² · Σ y²). Fold structure, ICIR, t = ICIR·√n_folds, binomial p, the D10 thresholds and the delayed-outcome variant are unchanged. The per-time-slot cross-check uses the same formula. Training-side choices (sign_train, the stable-IC selection rule, clustered MDA scoring) use the same formula on months inside the training window, with that fold's anchors. The old monthly Spearman is still reported, labelled "biased, information only".~~ | Demeaning within the test month uses the month's average forecast, which is known only at month end. For persistent features this injects the month's own returns with a minus sign (session-permutation placebo, `phase2_daily_diagnostics.md`). | §1, §6.8, §10.6, §11, §12, §13 |
| D18 | **Metric null test.** Before any metric is used in a gate, it must score ≈ 0 on data with no predictability by construction, and the share of features passing D10 there must be near the nominal rate. `tests/test_metric_null.py` is the fast synthetic version (random walk with seasonal volatility, a persistent level-like feature and a 1/price feature; the old metric must show the bias, the new one must not). The full-data version is the random-sign mirror null in `reports/phase2_metric_fix.md`. | A gate metric that rewards no-signal data cannot separate signal from artefact. | §11, §12 |

> **Status 2026-10-24: D17 fails D18 (superseded by D19)** (synthetic test: anchored level_like IC −0.0195, t −4.4; placebo mean for d_sma20_dev −0.0155, d_atr_c +0.0116; mirror null 2.3 D10 passes per draw vs 1.4 nominal, d_sma20_dev 17% vs 1.7%). Per D18, D17 is **not used in any gate** yet; the "IC per D17" notes on D10, D15 and D16 take effect only once a metric passes D18. Candidate fixes await the user's decision: `reports/phase2_metric_fix.md` §5.

## 2026-10-24 — corrected metric and pre-registered null criteria (approved by the user)

The choice follows from the two mechanisms in `phase2_metric_fix.md` §2, not from real-data results. This is the only revision: if D19 fails D20, the D20 fallback applies and the formula is not revised again.

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D19 | **Anchored IC v2 = primary metric** (supersedes D17). D10, D15, D16 and all training-side choices (sign_train, stable-IC rule, clustered MDA scoring) use it. Nothing from the test month is used except each test row's own s_t and y_t. **Features:** c(x) = clip((x − median_train) / IQR_train, ±5) (training sd if IQR = 0; binary flags as 0/1); score s_t = sign_train × (c(x_t) − mean_train(c(x))), i.e. centred at its training **mean** after clipping. **Models:** s_t = ẑ_t − training mean of clipped z (forecast relative to the historical-mean forecast). **Target:** y_t = clip(z_t, ±5), nothing subtracted. **Fold IC = Σ s·y / (n · √(V_s · V_y))**, n = rows with finite s and y; V_y = training mean of y²; V_s = training mean of s² for features; for models, the mean of s² over all earlier out-of-sample test months (the first fold uses the D6 inner-validation block), never in-sample predictions. Fold structure, t = ICIR·√n_folds, binomial p, D10 thresholds, the delayed-outcome variant and the slot cross-check are unchanged. The old monthly Spearman is still reported as "biased, information only". | (1) The test month's Σs² correlates with the numerator for persistent features. (2) With expanding windows the training mean of z is nearly constant across folds, so the m·Σs term repeats with the same sign every fold. A per-fold bias b adds about b/0.06·√96 ≈ 163·b to t, so the bias must be ≤ ~0.003. | §1, §6.8, §10.6, §11, §12, §13 |
| D20 | **Pre-registered pass criteria for D18** (replace "±0.005 or 1 sd"). D19 passes only if ALL hold: **(1) synthetic test:** level_like and inv_price mean IC within ±0.003, both with zero drift and with a constant drift equal to the real 2016–2025 regular-session mean of z; in the drift variant inv_price is tested in its rolling-z form (the raw form is reported only). **(2) session-permutation placebo (200 permutations):** d_sma20_dev, d_mom5, d_atr_c unsigned placebo mean within ±0.003 (amihud reported only). **(3) mirror null (≥ 60 draws) and mirror null + real drift (≥ 40 draws; after mirroring, add the constant real 2016–2025 mean log return per bar):** pooled over all draws, every feature's mean unsigned IC within ±0.003 or within 3 se of 0; pooled D10 false-pass rate across all features and draws ≤ 2.2% (nominal 1.67%); no single feature above 8%. **Fallback if any criterion fails:** stop and do not revise the formula; single features are then judged against their own mirror-null t distribution (D10 with null quantiles), and every model gate also requires the model to score ≈ 0 when trained and tested on mirror-null data. Report what failed and the projected compute cost of the fallback. | Criteria fixed before the run, so the metric cannot be tuned to the null results. | §10.6, §11, §12 |

> **Status 2026-10-24: D19 fails D20 as then specified** (`reports/phase2_metric_fix_v2.md`; the criterion-3 drift was corrected by D21 on 2026-10-25, this run is kept as a 15× stress test). Criteria 1 and 2 and the no-drift mirror null pass; the mirror null + drift fails (10–12 volatility-level features with mean IC up to −0.011; pooled D10 false passes 2.35% / 2.44% > 2.2%; vreg_level_pct 12.5%). The drift was read as the mean log return over all bars incl. the overnight bar (1.73e-5/bar), giving mean z_12 = 0.028 in the null vs 0.0019 in the real data. Per D20 the formula is not revised; the **D20 fallback applies** (single features vs their own mirror-null t distribution; models must score ≈ 0 on mirror-null data). Phase 2 has not been re-scored; `configs/features.yaml` is still v1 (old metric).

## 2026-10-25 — correction of the D20 criterion-3 drift (approved by the user)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D21 | **Correction of the D20 criterion-3 drift.** The D19 formula is not changed. The drift variant of the mirror null uses a constant per-bar drift on intraday bars (bar_index > 0), calibrated so that the null's mean z_12 equals the real regular-session mean of z_12 (0.0019, within ±10%), the same calibration as criterion 1. All other D20 criteria, thresholds, seeds (1000..1039) and the last-fold fitted-parameter approximation are unchanged. The 2026-10-24 run (drift 1.73e-5 per bar on all bars, mean z_12 0.028, ≈ 15× real) stays in the record as a stress test and documented limitation. **Decision rule fixed before the run:** if the corrected drift variant passes all D20 criterion-3 conditions (h = 12 and h = 12 delayed), D19 is adopted as the gate metric; if it fails, the D20 fallback applies with no further revision of the formula or the test. | The D20 criterion-3 wording ("mean log return per bar") was a specification error: criterion 1 already defined the drift as the real regular-session mean of z, and criterion 3 should have matched it. The all-bar mean is dominated by the overnight gap, which never enters an intraday label. | §10.6, §11 |

> **Status 2026-10-25: the corrected drift variant fails D20 criterion 3** (`reports/phase2_metric_fix_v3.md`): calibrated μ = 1.4885e-6 per intraday bar, null mean z_12 0.001924; pooled D10 false passes 1.93% / 2.02% (pass), but dow mean IC +0.0032 / +0.0033 (limit 0.0031 / 0.0033 = 3 se) and earn_react 10% D10 at h = 12 delayed (4 of 40 draws). Both are present in the paired no-drift draws 0..39 with the same seeds (drift effect ≤ 0.0001 for these two; ≤ 0.0007 for any feature). Per the D21 rule, **D19 is not adopted as the gate metric and the D20 fallback applies**; no further revision of the formula or the test. Phase 2 not re-scored; `configs/features.yaml` still v1.

## 2026-10-25 — D20 fallback specification (approved by the user; FINAL)

The D21 result stands: D19 is not adopted on its own and the pre-registered D20 fallback applies. **This is final: no further changes to the formula, the null tests or their criteria.**

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D22 | **D20 fallback specification.** **Metric:** the D19 formula, unchanged. **Null:** mirror null with the D21 calibrated intraday drift (constant drift on bar_index > 0 such that the pooled null mean z_12 equals the real regular-session mean of z_12, 0.0019 ± 10%; implemented with the D21 procedure on the new seeds: μ = (0.001924 − b) / (12·mean(1/S_12)), b = pooled mean z_12 of the mirrored series without drift for seeds 2000..2299, computed before any feature is scored), **300 new independent draws, seeds 2000..2299**; the earlier 60 no-drift and 40 drift1x draws are not used for quantiles; last-fold fitted-parameter approximation kept and stated in every report; each draw stores the per-month signed and unsigned D19 IC per feature for h = 12 and h = 12 delayed. **Single-feature D10\*:** a feature passes if (a) its real full-period signed t ≥ its own null 97.7% quantile of full-period signed t, and (b) its real 2023–2025 mean IC > 0 and its 2023–2025 t ≥ its own null 84.1% quantile of 2023–2025 t. For information only: empirical p = (1 + number of null draws at least as extreme on both parts) / 301, and Benjamini–Hochberg at FDR 10% across features. **Training-side selection** (stable-IC rule and the delayed rule): the \|t\| ≥ 2.0 thresholds are replaced by the feature's own null 95.4% quantile of \|full-period t\| (approximation: the same quantile for every fold's training window, and the h = 12 / h = 12-delayed quantiles are also used for h = 1 and h = 3, which have no null; both stated). **Model gates (Gate 3, Phase 4):** the model is also trained and tested on K mirror-null draws (K fixed in the Phase 3 instructions after timing one run); a model passes only if its null mean IC is within ±0.003 **and** its real t ≥ null mean t + 2 × null sd of t, in addition to D10. **D16 rule signs and clustered MDA scoring use D19.** | Pre-registered fallback after D19 failed the D21 run on seed-sample effects (dow, earn_react, both present in the paired no-drift draws). Calibrating each feature against its own null removes any remaining bias without another formula change. | §6.8, §10.6, §11, §12, §13 |

**D22 result (2026-10-25):** 300-draw null done (μ = 1.3609e-6 per intraday bar, null mean z_12 0.001924; max |null mean IC| 0.0016; pooled fixed-D10 false-pass 1.80% / 1.85%). D10\* at h = 12: 19 features (19 delayed), ≈ 2.7 effective signals (intraday trend continuation, day-level momentum, residual vs QQQ); old daily-context passers fail. `configs/features.yaml` version 2 (old list in `configs/features_v1_biased.yaml`); 10 folds have 17–19 features (D13). D16 rule (a) unchanged (D19 sign = old sign in all folds). Gate 2 re-confirmed; Phase 2 complete. Report: `reports/phase2_rescore.md`.

## Closed proposals

| # | Proposal | Status |
|---|---|---|
| P1 | Compute TSLA returns from **unadjusted** 1-minute prices (Alpaca `Adjustment.RAW`, same free keys as the earlier project), dividing by the cumulative split factor in floating point (no rounding), with explicit handling of the two split sessions. | Split-adjusted prices before 2022-08-25 are rounded to $0.01 (100% exact cent multiples in 2017–2021 vs about 50% after the split). In 2016 the effective tick was 7.2 bp instead of about 0.5 bp, and 18% of 5-min returns were exactly zero. See `phase1_checks.md`, B1. **Approved 2026-10-22 → D9.** |

## 2026-10-25 — Gate 3 specification under D19 + D22 (instructed by the user; recorded before any Gate 3 model was trained)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D23 | **Gate 3 under D19 + D22** (refines D15). **Models:** LightGBM with the fixed Phase 1 parameters (`tslaq.models.baselines.LGB_PARAMS`) and the Phase 1 objective (L2 regression on z_h clipped to ±6), no tuning; only `num_threads` (a compute setting) may differ. **A** = tokens of the last 24 bars (bars t−23..t; the 11 fields RET, SHAPE, REL, GAP, VOLU, VREG, HMM, MKT, TOD, DOW, EVENT; 264 integer-coded LightGBM categorical columns, MISSING is its own code); **B** = that fold's `by_fold` features from `configs/features.yaml` v2, in the fold's version (raw / rolling-z), values from the fold's own fitted objects (Phase 2 checkpoint), unscaled (trees); **C** = A + B inputs. One model per variant and horizon h = 1, 3, 12. The delayed outcome is a second scoring of the same predictions, not a new model. 96 monthly folds (2018-01..2025-12), rows with finite z_h and S_h, purge + 78-bar embargo as in Phase 1 (`monthly_folds`). The tokenizer (all fitted parts) is refit per fold on that fold's training rows. Each variant × horizon is one trial in `experiments/registry.csv` (9 trials). **Scoring:** D19 model formula: s = ẑ − training mean of the clipped (±5) outcome (sessions before the fold month); y = clip(outcome, ±5); V_y = training mean of y²; V_s = mean s² over all earlier out-of-sample months; the first fold (2018-01) uses the D6 inner-validation block (2017-11..2017-12, predicted by the same variant trained on rows before 2017-11 with purge + embargo). For the delayed outcome the training mean and V_y are those of the delayed outcome. **Gate 3 passes only if** model C at h = 12 with the delayed outcome passes the fixed D10 (full-period t ≥ 2; 2023–2025 mean IC > 0 with t ≥ 1) **and** the D22 model-null condition (null mean IC within ±0.003 and real t ≥ null mean t + 2 × null sd of t, both on the delayed h = 12 outcome). **Model null:** random-sign mirror null with the D22 calibrated intraday drift (μ = 1.3609e-6 per bar on bar_index > 0, not recalibrated on the new seeds; the realised pooled mean z_12 is reported), new seeds 3000, 3001, …; SPY/QQQ real; parameter-free features recomputed exactly, GARCH-t / HMM / Kalman / frac-diff with the **last fold's (2025-12) fitted parameters in every fold** (approximation, stated in every report); the tokenizer is refit per fold on the null data (RET edges, SHAPE k-means, GAP / VOLU / VREG / MKT quantile bins), except the HMM token, which uses the same last-fold HMM as the features; each fold's feature list and versions are those of the real run. Each null draw scores h = 12 and h = 12 delayed. **Budget:** first one complete null walk-forward of C (all 96 folds, h = 12, seed 3000) is timed in the same configuration as later draws and counts as draw 1. Then K = 20 if 20 draws of C fit in 12 hours (K × time per draw ≤ 12 h), else K = 15 if they fit, else K = 10; if 10 do not fit, stop and report the timing. C gets the null first; B and then A get the same K only if their K draws also fit in the remaining part of the 12 hours; a variant without a null is reported as "unconfirmed". Null runs are diagnostics, not trials. **Token value:** C vs B paired by fold (share of folds won, Wilcoxon signed-rank on fold ICs), for every horizon with and without delay; "C better than B" means mean fold-IC difference > 0 and one-sided Wilcoxon p < 0.05 at h = 12 delayed. C vs the Phase 1 `lgb_returns` predictions re-scored with D19 (no retraining) on the same folds (2018-02..2025-12, because Phase 1 has no inner-validation block). **If A fails (D10 or model null) and C is not better than B, stop and discuss before Phase 4 (D15).** | D15 predates D19/D22; the gate must use the adopted metric and the D22 model-null requirement, with every implementation choice fixed before the run. | §7.4, §10.6, §12 |

> **D23 result (2026-10-25): Gate 3 FAILS** (`reports/phase3_tokenizer.md`). C at h = 12 with the delayed outcome: IC 0.0078, t 1.39 (< 2; 2023–25 IC 0.0097, t 1.28); model null K = 20 (seeds 3000..3019): null mean IC +0.0005, threshold t 0.07 + 2 × 1.01 = 2.09 → fail. A fails (t 0.15), B fails at h = 12 (delayed t 1.92, null threshold 2.34). C is not better than B (h = 12 delayed: 49% of folds, Wilcoxon one-sided p 0.77; significantly worse at h = 1 and 3). **D15 stop condition met: stop and discuss before Phase 4.** K = 20 for A, B and C within the 12-hour budget (8.47 h used); 9 trials logged (n_trials_so_far = 18).

## 2026-10-25 — Phase 3 outcome and pre-registered baseline confirmation (approved by the user after the D15 discussion; recorded before any D25 model was trained)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D24 | **Phase 3 outcome.** Gate 3 **failed** (D23 result) and stays failed: it is not re-run or tuned. The tokenizer passed all §7.4 checks and is **kept as built**. The tokens showed no h = 12 information (0 BH token cells at h = 12 in any fold; model A h = 12 delayed t 0.15) and made LightGBM worse when added to B (C vs B: 49% of folds at h = 12 delayed, significantly worse at h = 1 and 3). | Record the result before deciding on Phase 4. | §7.4, §12 |
| D25 | **Baseline confirmation at h = 12 (pre-registered).** Uses baselines that §10.3 requires anyway. **Common to all models:** the D23 rows (finite z_h and S_h), 96 monthly folds 2018-01..2025-12, purge + 78-bar embargo (`gate3.split`), D19 model scoring and the D23 V_s rule (first fold: the D6 inner-validation block 2017-11..12, predicted by the same model trained on rows before 2017-11 with purge + embargo), one model per h = 1, 3, 12; the delayed outcome is a second scoring of the same predictions; each model × horizon is one trial. Tree / linear targets = the Phase 1 objective (z_h clipped ±6). **M1** = Phase 1 `lgb_returns`: same inputs (`simple_return_features`: 12 lagged σ̂-scaled returns, 24/78-bar scaled sums, bar_index), `LGB_PARAMS` with num_threads = 1, retrained in the D23 pipeline (D23 rows; LightGBM handles NaN inputs, Phase 1 dropped rows with a NaN input); report correlation / max difference with the stored Phase 1 predictions on common rows. **M2** = ridge (`sklearn.linear_model.Ridge`, intercept, penalty α·‖w‖² on the sum of squared errors) on the fold's `by_fold` v2 features (fold versions); inputs standardized with the fitting rows' mean / sd, clipped ±5, NaN → 0 after standardization; α per fold from {0.1, 1, 10, 100, 1000}: an inner model per α is fit on the rows before the fold's inner block (the last 2 calendar months before M, purge + embargo) and scored on that block with the D19 form Σs·y / (n·√(V_s·V_y)), s = ẑ − inner-training mean of clipped z, V_s = mean s² over the block itself (no earlier out-of-sample months exist for it), V_y = inner-training mean of y²; highest IC wins, ties → larger α; then refit on the full training window with the chosen α. Fold 2018-01's chosen-α inner predictions are its D6 V_s block. (Note: with 0.2–1.6M rows this grid means weak shrinkage; it is fixed as given.) **M3** = equal-weight composite: features of the fold's `by_fold` v2 list with basis "IC" or "both" ("extreme" excluded); per feature the D19 anchors (median / IQR, clip ±5, training mean of c) from sessions < M and sign_train = sign of the mean D19 monthly IC at the same h (undelayed z_h) over the training months (months < M with ≥ 200 rows; the Phase 2 rescore rule); s = mean over features with a finite score of sign_train·(c(x) − mean_train c(x)) (NaN if none); nothing else fitted. Scored with the **D19 feature formula**: V_s = training mean of s² (sessions < M), V_y = training mean of y²; same in the null, where anchors and signs are recomputed on the null data. M3 has no z forecast, so no CRPS. **M4** = `CatBoostRegressor(iterations = 300, depth = 4, loss_function = RMSE, random_seed = 0, thread_count = 1)`, other parameters default, on the fold's `by_fold` v2 features, no tuning. **Model null:** K = 20 per model, seeds 3000..3019, the D23 construction (mirror + μ = 1.3609e-6 per intraday bar, SPY/QQQ real, last-fold fitted parameters for GARCH-t / HMM / Kalman / frac-diff, the real run's feature lists and versions); Gate 3 null frames were not cached, so they are rebuilt deterministically from the same seeds; h = 12 with and without delay. **Pass:** at h = 12 with the delayed outcome, fixed D10 (full t ≥ 2; 2023–2025 mean IC > 0 with t ≥ 1) **and** the D22 model-null condition (null mean IC within ±0.003 and real t ≥ null mean t + 2 × null sd of t). **Family-wise check (information):** share of the 20 seeds in which at least one of M1–M4 passes, on that null draw, the fixed D10 and its model's null threshold. **Decision rule:** if ≥ 1 model passes, Phase 4 may proceed with the passing model with the highest full-period t at h = 12 delayed as the D4 baseline the Transformer must beat; if none passes, the recommendation is to conclude the project with a final report instead of Phase 4; the user decides. **Information only (not a gate; D2 unchanged):** (i) for Gate-3 B and the best of M1–M4 (highest full-period t at h = 12 delayed), a rule backtest per h = 1, 3, 12 on the out-of-sample predictions of that h: position = sign(ẑ) (M3: sign of s), signal at bar t enters at the open of bar t+1 (t+2 with the extra delay) and exits at the close of bar t+h (t+h+1), trades in a session non-overlapping (next signal bar ≥ the previous one + h), no entry in the last 10 minutes, exit no later than the session's last bar; return = sign·ln(exit/entry) − 2·cost; daily P&L over all test sessions, Sharpe = mean/sd·√252, t = mean/sd·√n_days; break-even cost per side = mean daily gross / (2 × mean trades per day); gross and net at 0.5, 1, 3, 6 bp per side; full period and 2023–2025; with and without the extra delay; one trial per model × h. (ii) TSLA effective spread per year 2016–2025 from 1-minute regular-session bars (bar start 09:30–15:59 ET; unadjusted Alpaca 1-min 2016-01..2022-08, vendor 1-min from 2022-09): EDGE (Ardia, Guidotti & Kroencke 2024, package `bidask`) per session, yearly mean and median; Corwin–Schultz cross-check (2-bar estimator on consecutive 1-min bars with the CS gap adjustment, negative estimates set to 0, daily mean); full spread and half-spread in bp, compared with the break-even costs of (i). **Budget:** one timed null draw per model first; if 20 draws of all four do not fit in 10 hours, run M1, M2, M3, M4 in that order and mark a model without a complete null as "unconfirmed". | Gate 3 failed; before deciding on Phase 4, check with pre-registered, untuned baselines whether any model clears the h = 12 gate at all. | §10.3, §12 |

> **D25 result (2026-10-26):** `reports/phase3_baselines.md`. At h = 12 delayed: **M3 passes** (IC 0.0294, t 4.33; 2023–25 IC 0.0188, t 1.90; null K = 20 mean IC +0.0018, threshold 0.22 + 2 × 0.94 = 2.11). M1 fails D10 (2023–25 t 0.95; null passes), M2 fails D10 and the null (t 2.01 < 2.26), M4 fails both (t 0.85). Family-wise: 1 of 20 null seeds has a passing model (M3, seed 3009). **Decision rule: Phase 4 may proceed with M3 as the D4 baseline; the user decides.** Caveats: M3's constituents overlap the Phase 2 D10\* passers evaluated on the same folds (17 of 20); M3 is not significantly better than M1 or Gate-3 B (one-sided Wilcoxon p 0.080 / 0.075); its sign rule breaks even at 0.31 bp per side vs an EDGE half-spread of 0.5–0.65 bp (2023–25); Gate-3 B's h = 12 rule breaks even at 1.3–1.4 bp. n_trials_so_far = 36.

## 2026-10-26 — realistic costs and an early, pre-registered economic test (approved by the user; recorded before any D27 backtest was run)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D26 | **Realistic cost model.** Cost per side in calendar year Y = the **EDGE mean half-spread of year Y−1** (`reports/tables/phase3_baselines/spread_by_year.csv`, column `edge_mean_half_bp`; causal: prior year only) **+ 0.10 bp** (SEC Section 31 fee, $20.60 per $1M on sales only, averaged over buys and sells ≈ 0.10 bp per side; FINRA TAF negligible). Values: 2018 1.474, 2019 1.738, 2020 1.287, 2021 1.742, 2022 1.078, 2023 1.113, 2024 0.607, 2025 0.729 bp per side. Every result is also reported at **2× D26** and at the fixed **0.5 / 1 / 3 / 6 bp** per side (D2). The 3 bp default in `configs/backtest.yaml` is kept for older reports. (CPCV only: 2016 has no prior year in the table, so 2016 uses its own-year value; stated.) **This cost choice was made after the D25 break-even numbers were seen**, although the spread measurement itself does not depend on any model. | The fixed 3 bp default is about 2–5× the measured TSLA half-spread in 2023–2025; a model-free, prior-year estimate is a more realistic and still causal cost. | §10.4, §12 |
| D27 | **Early Phase 5 test (pre-registered).** No retraining except the optional CPCV. **Candidates:** M3, Gate-3 B and M1 at h = 12, using their stored D25 / D23 out-of-sample predictions as they are (`data/processed/phase3_baselines/real/`, `data/processed/phase3/real/`, variant B); ẑ = the stored prediction (M3: its score s). **Threshold q** per fold = 90th percentile of \|ẑ\| on the fold's D6 inner-validation block; only fold 2018-01 has stored inner predictions (2017-11..12), so every later fold M uses the previous fold's out-of-sample month M−1 (causal). **Execution (both rules, CLAUDE.md §10.4):** signal at the close of bar t, fill at the open of bar t+1 (t+2 with the extra delay); no fills in the session's last two bars (last 10 minutes) other than the forced exit; flat at the close of the session's last bar; cost = D26 cost × \|Δw\| on every change of position, including the final flattening. **R1 proportional:** target w = clip(ẑ/q, −1, 1); the position changes to the target only when \|w_target − w_current\| > 0.25 (no-trade band); bars without a finite ẑ keep the current position. P&L = Σ w × log return between consecutive fill points (open to open; the last bar open to close). **R2 strong signal:** when \|ẑ_t\| > q, take sign(ẑ_t) at unit size, enter at the open of t+1(+d) and exit at the close of bar t+12(+d); non-overlapping (next signal bar ≥ t+12); trades whose exit would fall after the session's last bar are skipped (D25 convention); otherwise flat. **Trials:** 3 candidates × 2 rules = 6 trials in `experiments/registry.csv` (net_sharpe_period = daily Sharpe at D26 cost; sharpe_2x_costs at 2× D26). **Reported per trial:** trades per day (R1: fills per day), turnover (mean daily Σ\|Δw\|), hit rate and mean bp per trade (R1: a trade = a holding segment between consecutive fills with w ≠ 0, return per unit position), break-even cost per side (mean daily gross / mean daily turnover), gross and net Sharpe (daily P&L over all test sessions incl. flat days, annualized √252, and daily per-period), Sortino (downside deviation √mean(min(r,0)²)), max drawdown of the cumulative daily log P&L; for 2018–2025 and 2023–2025; at D26, 2× D26 and 0.5 / 1 / 3 / 6 bp; with and without one extra bar of delay. **DSR (§10.7)** at D26 cost on full-period daily net returns (no extra delay): N = n_trials_so_far after logging the 6 trials; V = variance of the per-period (daily) Sharpe ratios logged in the registry for all rule-backtest trials (D16, D25, D27; reruns excluded; D16 and D25 are logged at their own costs); report γ3, γ4 (raw), T, SR*. **PASS** (Phase 5 gate as adapted): full-period net Sharpe > 0 at 2× D26 **and** 2023–2025 net Sharpe > 0 at D26 **and** full-period net Sharpe > 0 at D26 with one extra bar of delay **and** DSR > 0.95. **Optional CPCV (§10.2), time-boxed to 6 h,** for the single trial with the highest full-period net Sharpe at D26 cost: development sessions 2016-01..2025-12 in N = 10 contiguous blocks of equal session count, every k = 2 test-block combination (45 splits), training = the other blocks purged (label windows overlapping a test block) and embargoed (78 bars after each test block), the candidate's recipe retrained per split (M3: anchors and sign_train on the training rows; LightGBM: fixed parameters); fitted features (GARCH-t / HMM / Kalman / frac-diff) and the feature list are the last fold's (2025-12) — approximation, stated; q = 90th percentile of \|ẑ\| on the training rows (M3, unfitted) or, for LightGBM, on a 2-month inner block of training sessions immediately before the first test block (or after the last test block if none precede it), predicted by a model trained on the remaining training rows; the same rule, D26 costs and execution; report the Sharpe of the 9 reconstructed paths (median, share > 0) and of the 45 splits. Skip and say so if it does not fit. **Information:** if null-draw predictions of M1, M3 or B are stored on disk, run R1 and R2 on them and report the null net Sharpe distribution at D26 cost; and for each candidate the approximate IC needed for a net Sharpe of 0.5 at 2× D26 cost, IC × (required gross edge / observed gross edge), required gross = 0.5·sd(daily net)/√252 + mean daily cost at 2× D26, IC = the D25 h = 12 delayed IC (an approximation that assumes gross edge ∝ IC at fixed turnover). **Decision rule (fixed now):** if at least one trial passes, freeze it (code, parameters, config hash) and stop; the next step is the user's decision on the single holdout evaluation (D7) as a "no-failure" check, then paper trading. If none passes, the recommendation is to conclude with a final report; Phase 4 only as a user-chosen long shot with the IC target above as its explicit bar. | An economic check on predictions that already exist costs little and answers whether the h = 12 information can pay realistic costs before Phase 4 or the holdout is spent. | §10.2, §10.4, §10.7, §12 |

> **D27 result (2026-10-26): no trial passes** (`reports/phase5_early.md`). Gate-3 B passes the three Sharpe conditions with both rules (R1: full 2× D26 0.10, 2023–25 D26 1.06, +1 delay D26 0.62; R2: 0.55 / 0.47 / 0.49) but every trial has DSR 0.000 (pre-registered V = 0.2773 over 14 rule trials, SR\* = 1.163 per day; information: with V of the 6 D27 trials only, best DSR 0.196). M3 and M1 fail the cost / delay conditions. CPCV (B R1, 45 splits, last-fold features and list): 9 paths median net Sharpe 0.68 at D26, all > 0. Null Sharpes not run (null predictions not stored). IC target for net Sharpe 0.5 at 2× D26: B 0.016–0.020, M1 0.053–0.058, M3 0.062–0.112. M3's fold-2018-01 q used its score on the D6 block (2017-11..12, anchors from sessions < 2017-11), as M3 had no stored inner scores. **Decision rule: recommendation = conclude with a final report; Phase 4 only as a user-chosen long shot with IC ≥ 0.02 (h = 12 delayed) and a passing D27 backtest as its bar.** n_trials_so_far = 42.

## 2026-10-26 — final holdout evaluation (approved by the user; recorded before any holdout data was downloaded or loaded)

Development is concluded (D27: no trial passes). As the final out-of-sample step the single best development candidate, Gate-3 B with rule R1, is frozen and evaluated **once** on the holdout (D7). **This deviates from the D27 decision rule**, which linked the holdout to a passing trial, **and does not make B R1 a pass.** The forward test (data after 2026-10-09) stays sealed. Phase 4 is not started.

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D28 | **Final holdout evaluation of the frozen Gate-3 B R1.** **Frozen design** = exactly the development recipe, applied per holdout month M = 2026-01..2026-10 (October only up to 2026-10-09), each month using only sessions before M: data v2 cleaning (TSLA, SPY, QQQ Alpaca SIP 1-minute bars, same adjustment handling as v2: split-adjusted source bars from 2022-09 on, SPY/QQQ dividend back-adjusted; regular session, 5-minute bars, missing-bar and bad-session rules unchanged); the Phase 2 per-fold fits (GARCH-t, HMM, Kalman, frac-diff, ADF/KPSS raw / rolling-z versions, clusters, condprob cells for the extreme-bucket rule; `scripts/run_phase2_walkforward.py::run_fold`) and the per-fold feature selection with the D22 thresholds and the existing 300-draw null quantiles (`reports/tables/phase2_rescore/null_quantiles.csv`; no new null; `scripts/phase2_rescore_v2.py::run_fold` + `d22_select`); LightGBM with the fixed Phase 1 parameters (`LGB_PARAMS`, num_threads = 1) on that fold's list (Gate-3 B, h = 12, D23 rows, purge + 78-bar embargo; `tslaq.models.gate3.fold_predictions`, variant B); rule R1 exactly as in D27 (q = 90th percentile of \|ẑ\| over the previous month's out-of-sample predictions, 2026-01 uses the stored 2025-12 predictions; w = clip(ẑ/q, ±1); band 0.25; signal at the close of bar t, fill at the open of bar t+1; no fills in the last 10 minutes; flat at the close; `tslaq.backtest.positions.r1_fills` / `r1_pnl`); D26 cost for 2026 = EDGE 2025 mean half-spread + 0.10 bp (`spread_by_year.csv`). The git commit hash and the config hash of the frozen code are recorded in the freeze note below, before the reproduction check. **Reproduction check, before any holdout data is loaded:** run the frozen pipeline from scratch for development fold 2025-12 (on the development data only) and report the correlation and max difference vs the stored Gate-3 B h = 12 predictions, the feature list vs `configs/features.yaml` v2, and the R1 P&L for that month vs the D27 result. **If the correlation is below 0.99, stop and report without touching the holdout.** **Primary criterion ("no-failure" check), fixed now:** total net P&L of R1 at D26 cost over the whole holdout > 0. **Reported, not criteria:** net Sharpe and total P&L at D26, 2× D26, 3 bp and gross; the same with one extra bar of delay; per-month P&L; turnover and trades (fills) per day; hit rate; D19 IC at h = 12 with and without delay (per month and mean, with t over the holdout months; s = ẑ − training mean of clipped z, V_y = training mean of y², V_s = mean s² over all earlier out-of-sample months incl. 2018-01..2025-12). **Interpretation, fixed now:** the holdout has low power. If B R1's true annualized net Sharpe is about 1.1, the chance of a positive holdout P&L is about 83%; with no edge it is 50%. A pass is consistent with the development result but is not confirmation. If the criterion passes, the recommended next step is paper trading (Phase 6) on the forward period with its own pre-registered criteria, for at least 3 months, before any real money. If it fails, the recommendation is to stop. **One evaluation only:** the holdout is scored once; no parameter, rule, cost, feature or code change after any holdout number is seen. One trial in `experiments/registry.csv`. **Data:** TSLA, SPY, QQQ 1-minute bars 2026-01-02..2026-10-09 from Alpaca (SIP feed); TSLA earnings dates for the period from Alpha Vantage `EARNINGS` (as in Phase 1); SPY/QQQ dividends for the period from Alpha Vantage `DIVIDENDS` (as in data v2). Nothing after 2026-10-09 is requested. | User-approved final out-of-sample step after development concluded without a passing trial. Fixing the design, the criterion and its interpretation before the data is seen keeps the single evaluation honest. | §10.5, §12 |

> **D28 freeze (2026-10-26, before the reproduction check and before any holdout data was downloaded):** frozen code = git commit **65a13dcc0c7a1dc00b8b74e51f7a5cb8fd563752** (`scripts/run_holdout_d28.py` and the modules it calls), clean worktree; **config hash da5d24529c** (`run_holdout_d28.frozen_config()`: LightGBM parameters, Phase 2 settings, data and backtest configs, R1 settings, SHA-1 of the null-quantile and spread tables). The only changes to existing code were in the data layer (an optional end date and extra months for the holdout build, and a `TSLAQ_DATA_VERSION` override); the v2 bars and earnings sessions rebuilt with the changed code are bit-identical to the stored v2 files, and the test suite passes (65 tests). The 2026 earnings dates were re-checked against Alpha Vantage `EARNINGS` (2026-01-28, 04-22, 07-22, post-market; unchanged in the file); TSLA `SPLITS` lists no split after 2022-08-25; the 2026 SPY/QQQ dividends were already in the Alpha Vantage files of data v2.

> **D28 result (2026-10-26): the primary criterion FAILS** (`reports/holdout_eval.md`).
> - **Reproduction check (fold 2025-12, before any holdout data):** correlation 1.000, max difference 0.0; feature
>   list identical; R1 P&L −215.16 bp, the same as D27. The holdout was then downloaded and evaluated once with the
>   frozen code (no code or config change after the freeze; config hash da5d24529c at evaluation).
> - **Holdout, 194 sessions, 2026-01-02..2026-10-09:** total net R1 P&L at D26 (0.749 bp per side) **−725 bp**,
>   net Sharpe −0.49 (t −0.43). Gross +75 bp (Sharpe 0.05; break-even 0.07 bp per side vs 2.64 in development);
>   2× D26 −1,526 bp; 3 bp −3,129 bp. With one extra bar of delay: −675 bp at D26, gross +126 bp. Hit rate 0.485;
>   12.5 fills and 5.5 turnover per day. Net P&L positive in 3 of 10 months (September −1,364 bp).
> - **D19 IC at h = 12:** mean 0.0138 (t 0.97); delayed 0.0157 (t 1.15); 7 of 10 months positive.
> - **Recommendation as fixed in D28: stop.** No paper trading, no Phase 4 on this design; the forward test stays
>   sealed. One registry trial `d28_holdout_R1_Gate3B_h12`; **n_trials_so_far = 43**. Final report:
>   `reports/final_report.md`.

## 2026-10-31 — new pre-registered study: SPY intraday momentum (instructed by the user; recorded before any backtest)

The TSLA project is concluded (D28) and none of its results change. This is a separate study on SPY. It replicates and
tests the strategy in Zarattini, Aziz & Barbon, "Beat the Market: An Effective Intraday Momentum Strategy for S&P500
ETF (SPY)" (SSRN; SFI Research Paper 24-97).

- **Source.** The version of 2025-02-03 (first version 2024-05-10), from alexandria.unisg.ch, bitstream
  a99aba00-f967-49b3-aceb-f544dc386e0b, SHA-256 41d635d162551c4add329d2a11cd9571aea391befd00e472ed81c0070bb207ad.
  The PDF is not committed.
- **Secondary reference.** The authors' public Python code (concretumgroup.com). Where the paper is silent, D29
  follows the code. Where the two disagree, D29 follows the paper. The ambiguities and their resolutions are A1–A20
  in [`spy_momentum_plan.md`](spy_momentum_plan.md).
- **What stays off limits.** No order is placed, and no order-routing code is written. The forward test (data after
  2026-10-09) stays sealed.

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D29 | **SPY intraday momentum study, pre-registered.** The rules, data, execution, costs, periods, criteria, order of evaluation and trial count are as specified below the table. Primary configuration: noise area (14 days, VM 1) + VWAP trailing stop, semi-hourly decisions, volatility-targeted sizing (2% daily, cap 4×). Pass criteria: P2 (2024-03..2025-12) net Sharpe > 0.5 at realistic cost, > 0 with one extra bar of delay and > 0 at 2× realistic cost; P3 (2026-01-02..2026-10-09) net P&L > 0 at realistic cost. P1 (2016-01..2024-02) is a replication, reported against a fixed tolerance. | A published, fully specified rule that has a post-publication period in our data is a cheap, honest test of whether a documented intraday edge survives realistic costs. The rules, costs, tolerance and criteria are fixed before any SPY backtest, so the result cannot be tuned. | separate study; reuses the §10.4 / §10.7 conventions |

### D29 specification

**Rules (as extracted).** Notation and the first three steps:

- **Notation.** All bars are 1-minute bars. Minute m is the bar ending at 09:30 + m minutes. O_t is the open of day
  t's first bar. C_{t−1} is the previous session's last regular-session close.

1. **Move from the open.** move_{d,m} = |C_{d,m} / O_d − 1|.
2. **Average move.** σ_{t,m} = the mean of move_{t−i,m} over i = 1..14, using the 14 previous sessions in the data.
   All 14 are required.
3. **Noise area.**
   - UB_{t,m} = max(O_t, C_{t−1}) · (1 + VM · σ_{t,m}).
   - LB_{t,m} = min(O_t, C_{t−1}) · (1 − VM · σ_{t,m}).
   - VM = 1.

The remaining steps:

4. **VWAP.** VWAP_{t,m} = Σ (H+L+C)/3 · V / Σ V, accumulated from 09:30 over the regular session only.
5. **Decision times.** Positions change only at 10:00, 10:30, …, 15:30 (on half-days 10:00..12:30), using the close
   of the bar that ends at that time.
6. **Primary configuration** (paper Table 3, row "Curr.Band + VWAP, Dyn.").
   - The signal is +1 if C > UB and C > VWAP, −1 if C < LB and C < VWAP, and 0 otherwise.
   - At each decision the position becomes signal × N_t.
   - This is the paper's entry plus its trailing stop (max(UB, VWAP) for longs, min(LB, VWAP) for shorts), checked
     only at decision times.
   - Re-entry is allowed whenever the condition holds again (as in the authors' code).
7. **Exit.** Flat at the close of the session's last bar (16:00; 13:00 on half-days).
8. **Sizing.**
   - Shares: N_t = round(AUM_{t−1} / O_t · min(4, 0.02 / σ_SPY,t)).
   - σ_SPY,t is the sample sd (ddof 1) of the dividend-adjusted daily close-to-close returns ret_{t−i}, i = 1..14.
   - Start: $100,000 on the first session with complete look-backs (≈ 2016-01-25), then compounding.
   - Daily return: dollar P&L_t / AUM_{t−1}.
   - Prices: N_t and the per-share costs use **unadjusted** prices. Signals use the dividend back-adjusted series.
     Intraday returns are the same in both; on an ex-date the adjusted C_{t−1} equals the unadjusted C_{t−1} − D, as
     in the code.

**Secondary configurations (replication only).**
- **(a) "Opp.Band, 100%".** When flat, at a decision, go long if C > UB or short if C < LB. Hold until a decision at
  which C is beyond the opposite band, then reverse. Flat at the close. N_t = round(AUM_{t−1} / O_t).
- **(b) "Curr.Band + VWAP, 100%".** Rule 6 with N_t = round(AUM_{t−1} / O_t).

**No tuning.** Lookback 14, VM 1, volatility target 2% and leverage cap 4× are fixed as in the paper.

**Data.**
- **Source.** SPY Alpaca SIP 1-minute regular-session bars (bar start 09:30..15:59 ET).
  - 2016-01..2025-12 come from `data/raw/SPY/`, the data v2 source.
  - 2026-01-02..2026-10-09 come from `data/raw/holdout/SPY/`, the data v2h source. The code asserts that no bar after
    2026-10-09 16:00 ET is read.
  - The `SPY_1min_sip_2026-*` files in the local source directory are never read.
- **Cleaning.** The v2 rules at 1-minute resolution:
  - NYSE calendar with half-days;
  - missing minutes forward-filled at the previous close with zero volume;
  - sessions with more than 5% missing minutes dropped, plus every session that is missing or dropped in SPY v2 / v2h
    (for example 2019-08-12);
  - bad ticks flagged, not deleted;
  - dividend back-adjustment with `data/raw/dividends/SPY.csv` (`clean.dividend_adjust`).
- **Data check before any backtest.**
  - Aggregated to 5 minutes, the 1-minute frame must reproduce the SPY 5-minute log returns in `data/interim/v2`
    (2016–2025) and `v2h` (2026): max |difference| ≤ 1e-10, same sessions.
  - The cache must be dividend-unadjusted (checked against the dividend file on ex-dates).

**Execution.**
- **E0 (main).** The signal is taken at the decision bar's close and filled at the open of the next 1-minute bar.
  That price is Open_HH:MM, the paper's fill reference.
- **E1.** One extra bar of delay: the fill is at the open of HH:MM+1.
- **Close exit.** At the last bar's close in both E0 and E1.
- **P&L.** Shares × the price change between fills.

**Costs.** Charged per order on |Δshares|, for every position change including the close exit.
- **C_paper** (the paper's own assumption, §3 and §4.6). $0.0035 per share commission + $0.001 per share slippage.
  No minimum; the $0.0020 volume tier is not modelled.
- **C_real** (realistic), the sum of:
  - **half-spread:** the SPY EDGE mean half-spread of year Y−1 × notional. EDGE is Ardia, Guidotti & Kroencke 2024,
    computed per session on 1-minute OHLC with the yearly mean taken, i.e. the D25 (ii) / D26 procedure applied to SPY.
    2016 uses its own year, because there is no 2015 data; 2026 uses 2025.
  - **commission:** $0.0035 per share, minimum $0.35 per order.
  - **SEC fee:** 0.10 bp × notional (Section 31 fee averaged over buys and sells, as in D26).
- **2× C_real.** Every component of C_real doubled.
- **Gross.** Results without costs are reported as well.

**Periods.** All metrics come from the daily returns within each period of one continuous simulation. Sharpe = mean /
sd of daily returns × √252, with no risk-free rate.

| period | dates | role |
|---|---|---|
| P1 | 2016-01..2024-02 | replication |
| P2 | 2024-03..2025-12 | post-publication test |
| P3 | 2026-01-02..2026-10-09 | second out-of-sample check |

SPY 2026 data is in v2h. D28 used it only as TSLA market context, never for an SPY strategy.

**Pre-registered criteria (primary configuration).**
1. **P1: report only.**
   - **Comparison.** The paper's monthly table (FAQ Q24), months 2016-02..2024-02: annualized Sharpe from monthly
     returns 1.37, CAGR 20.1%, vol 14.2%. Ours is taken at C_paper with E0.
   - **"Within a reasonable range"** means all of these hold:
     - (i) the correlation of monthly returns is ≥ 0.80;
     - (ii) the Sharpe from monthly returns is within ±0.30 of 1.37;
     - (iii) the CAGR is within ±4 pp of 20.1%;
     - (iv) the yearly return has the same sign in at least 7 of the 8 years 2016 (from February)..2023.
   - **A miss is reported, not fixed.** Code changes are allowed only to correct deviations from the D29 rules, and
     each one is documented with before and after numbers.
2. **P2 PASS** if all three hold:
   - net Sharpe > 0.5 at C_real (E0);
   - net Sharpe > 0 at C_real with E1;
   - net Sharpe > 0 at 2× C_real (E0).
3. **P3 PASS** if the net P&L (compounded total return) is > 0 at C_real (E0).

**Order of evaluation.**
- **Stage A** runs P1 only and loads no bar after 2024-02-29.
- **Freeze.** The code is then frozen, and its commit and config hash are recorded.
- **Stage B** runs P2 and P3 once with the frozen code. Nothing changes after any P2 or P3 number has been seen.
- Each stage needs the user's approval.

**Benchmark.** SPY buy-and-hold, 1×, dividend-adjusted close-to-close (overnight included), over the same periods and
with the same metrics.

**Reported, not criteria.** For every configuration × {gross, C_paper, C_real, 2× C_real} × {E0, E1} × period, and
per year and month:
- **Returns and risk:** total return, CAGR, vol, Sharpe, t of the mean daily return, Sortino, MDD, skew, worst and
  best day.
- **Hit ratio:** positive days among days with a trade, and among all days.
- **Market exposure:** alpha and beta vs SPY (daily OLS, Newey–West t).
- **Trading:** trades and trades per day, average P&L per share, the long and short legs separately, and the
  break-even cost per side (gross P&L / traded notional, in bp).
- **Comparisons:** the 2024-05..2025-12 slice of P2 (information only), and the authors' own post-publication
  figures (2024 +32.2%, 2025-01 −1.2%).
- **DSR (§10.7), information only:** N = n_trials_so_far; V = the variance of the three D29 configurations' P2 daily
  Sharpe ratios, and also with the D27 V.

**Trials.**
- n_trials_so_far continues from **43**.
- D29 adds **3 trials** (the primary and the two secondary configurations), giving 46.
- Stage B rows are logged with `rerun_of` pointing to the Stage A rows.
- Cost levels, E1 and the periods are additional scorings of the same trial, not new trials.

**Out of scope.**
- The paper's §4 analyses: VIX regimes, daily patterns, day of week, the VM and lookback sweeps, gamma / RSI, I-Star.
- Other tickers and futures.

## 2026-10-31 — D29 amendments (instructed by the user with the Stage A approval; recorded before any backtest code was run)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D29a | **Amendments to D29.** **D29a-1 (replaces resolution A2):** σ_{t,m} = the mean of move_{d,m} over the 14 most recent previous sessions d that contain minute m. Half-days have no minutes after their close, and dropped or missing sessions are absent. 14 such sessions are required. **D29a-2 (cost floor):** in C_real, the half-spread per side = max(SPY EDGE mean half-spread of year Y−1, half a tick = $0.005 per share). The spread table reports the yearly EDGE half-spread in bp next to the half-tick in bp (at the year's median price), and which one binds. In the backtest the floor is applied per order, at the fill price: cost = max(EDGE_{Y−1} bp × notional, $0.005 × shares). **D29a-3 (interpretation, fixed now):** P2 has limited power. If the paper's Sharpe of about 1.3 is real, the chance of meeting "Sharpe > 0.5" over 22 months is about 86%; with no edge it is about 25% (normal approximation, se of the annualized Sharpe ≈ √(12/22) = 0.74). If P2 and P3 both pass, the next step is paper trading on live data for at least 3 months, on an instrument the user can trade from the EU (likely Micro E-mini S&P 500 futures, MES), with its own pre-registered criteria and cost model. If P2 fails, the recommendation is to stop. A pass is not proof. | (1) Requiring the 14 previous calendar sessions would block afternoon trading for about 14 sessions after every half-day. (2) SPY's quoted spread is almost always one tick, so an EDGE estimate below half a tick is not a credible cost. (3) The interpretation is fixed before any P2 number exists. | separate study |

> **D29 Stage A result and freeze (2026-10-31; P1 only, no SPY bar after 2024-02-29 used, P2 / P3 not run)**
> ([`spy_stageA.md`](spy_stageA.md)).
> - **Data check passed.** 2,052 sessions; 159,516 five-minute bars matched v2 with max \|Δ log return\| 4.4e-16; the
>   cache is dividend-unadjusted (32 ex-dates, factor difference ≤ 2.2e-16).
> - **P1 is within the D29 tolerance** (primary, C_paper, E0, 2016-02..2024-02 vs the paper's monthly table):
>   correlation 0.993, Sharpe 1.26 vs 1.37, CAGR 18.3% vs 20.1%, yearly sign 8 of 8. No code fix was needed.
> - **Frozen code:** git commit **8163aa6** (`scripts/spy_stageA.py`, `src/tslaq/spy/`), **config hash 3d8abf0044**
>   (`reports/tables/spy_stageA/frozen_config.json`). Commits after 8820812 changed no strategy logic: 6ce1b73 fixed
>   the units of the paper column in `p1_yearly.csv`, and 8163aa6 filters the v2 reads at load time. Both were
>   verified to reproduce `p1_summary.csv` byte for byte.
> - **Trials:** 3 trials logged (`d29_spy_momentum_*`, git 8820812); **n_trials_so_far = 46**.

## 2026-10-31 — D29 Stage B specification (instructed by the user with the Stage B approval; recorded before any P2 or P3 strategy number was computed)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D29b | **Stage B (P2, P3) under the frozen D29 code.** The specification is below the table. | The extension of data and costs and the one-time evaluation are fixed before any P2 or P3 number exists. | separate study |

### D29b specification

**(1) Frozen strategy code.**
- `git diff 8163aa6 HEAD -- src/tslaq/spy/momentum.py src/tslaq/spy/metrics.py` is empty (checked at HEAD 1c28016:
  0 lines).
- Only the loaders (`src/tslaq/spy/data.py`) and the runner (a new `scripts/spy_stageB.py`) may be extended.
- Constants, rules and cost formulas are those of config hash 3d8abf0044. The only additions are the half-spreads for
  cost years 2025 and 2026, from (3).

**(2) Data extension.**
- **P2 bars:** `data/raw/SPY/2024-03..2025-12`.
- **P3 bars:** `data/raw/holdout/SPY/2026-01..2026-10`, read with a timestamp filter at read time, so nothing at or
  after 2026-10-09 20:00 UTC (16:00 ET) is loaded. The loader asserts this.
- **Cleaning:** the same D29 rules and the same dividend back-adjustment (SPY dividends up to 2026-10-09). Sessions
  are dropped if they are absent from v2 (through 2025-12) or v2h (2026).
- **Data check before the backtest:** aggregated to 5 minutes, the frame reproduces the SPY log returns of
  `data/interim/v2` (2016-01..2025-12) and `data/interim/v2h` (2026-01-02..2026-10-09):
  - max |difference| ≤ 1e-10;
  - the same sessions;
  - the unadjusted-cache check on both.
- **Reproduction check before any P2 / P3 output:** the continuous simulation's daily returns over P1 must reproduce
  the Stage A daily files for every configuration × cost × execution (max |difference| ≤ 1e-12). If either check
  fails, the run stops and writes no P2 / P3 number.

**(3) Costs.**
- The SPY EDGE spread table is extended to 2024 and 2025 with the same code (`scripts/spy_stageA.py::spread_table`
  procedure). It is model-free and is computed and committed in a separate step before the Stage B run.
- Cost year Y uses EDGE of Y−1 with the D29a-2 half-tick floor: 2025 uses 2024 and 2026 uses 2025. Earlier years are
  unchanged.

**(4) Information note (fixed now).**
- The authors have already published 2024 (+32.2%) and 2025-01 (−1.2%) for this strategy, so P2 is partly known.
- The least-known parts are 2025 and P3. 2025 (full year) is reported separately next to P2 and P3.
- This is information only. The D29 pass criteria are unchanged.

**(5) One evaluation only.**
- The Stage B runner refuses to run if its marker file `reports/tables/spy_stageB/stageB_done.json` exists, as in
  D28.
- Nothing changes after any P2 or P3 number has been seen.

**(6) Evaluation.**
- **Simulation:** one continuous simulation from 2016-01-26 to 2026-10-09.
- **Criteria (D29):** applied to the primary configuration.
- **Reported:** every D29 "reported, not criteria" metric for all 3 configurations × {gross, C_paper, C_real,
  2× C_real} × {E0, E1}, for P2, P3, 2024-05..2025-12 and 2025, per year and per month, plus SPY buy-and-hold.
- **DSR (information):** N = n_trials_so_far (46; Stage B rows are re-runs); V = the variance of the three
  configurations' P2 daily Sharpe ratios at C_real E0, and also the D27 V (0.2773).
- **Interpretation:** as fixed in D29a-3.
- **Registry:** Stage B rows are logged with `rerun_of` = the Stage A model names `d29_spy_momentum_*`.

**(7) Disclosure.** While checking the holdout file format before this entry, the timestamp column of
`data/raw/holdout/SPY/2026-10.parquet` was read for its min / max only. Its max is 2026-10-09 23:59 UTC, i.e. it
contains post-market bars of 2026-10-09 after 16:00 ET. No price or volume after 16:00 ET was read. From here on the
file is read with the timestamp filter in (2).

> **D29b result (2026-10-31; single Stage B run, frozen strategy code 8163aa6 / 3d8abf0044)**
> ([`spy_stageB.md`](spy_stageB.md)).
> - **Checks passed.** Data check v2 2016–2025 and v2h 2026 (max \|Δ log return\| 4.5e-16 / 0.0, same sessions);
>   P1 reproduction exact for all 24 runs.
> - **P2 (2024-03..2025-12): PASS.** Net Sharpe 0.59 at C_real (E0), 0.20 at C_real with E1, 0.37 at 2× C_real.
> - **P3 (2026-01-02..2026-10-09): FAIL.** Net P&L −13.6% at C_real (Sharpe −1.38).
> - **Information.** 2025 at C_real −4.8% (SPY +17.7%). 2024-05..2025-12 Sharpe 0.46. Our 2024 (+32.5% at paper
>   cost) matches the authors' published +32.2%, so the P2 pass rests on already-published months.
> - **No pre-registered next step.** D29a-3 does not cover "P2 passes, P3 fails". Recommendation (not
>   pre-registered): stop; no paper trading on this strategy. The user decides.
> - **Registry and forward test.** Three re-run rows (n_trials_so_far 46). The forward test stays sealed.
