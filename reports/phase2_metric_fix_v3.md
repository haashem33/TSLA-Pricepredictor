# Phase 2 — D21 corrected drift test for D19: FAILED (narrowly, on seed-sample effects, not on drift)

Date: 2026-10-25. Data v2 (bars 2016-01-04..2025-12-31). Holdout and forward test not loaded. Diagnostics only,
so no registry trials. Phase 3 has not started.

**Result: the corrected drift variant fails D20 criterion 3. Under the D21 decision rule (fixed before the run),
D19 is not adopted as the gate metric, and the D20 fallback applies.**

The variant misses two of its conditions:
- one feature's mean IC: dow +0.0032 / +0.0033, against ±0.003 or 3 se = 0.0031 / 0.0033;
- one feature's D10 rate at h = 12 delayed: earn_react 10%, i.e. 4 of 40 draws, against the 8% limit.

Both misses are already present in the paired no-drift draws with the same seeds (§3). The drift itself moves
no feature's IC by more than 0.0007.

As the rule requires, I did not revise the formula or the test. Step C was **not run**:
- `configs/features.yaml` is unchanged (v1, old metric);
- no `features_v1_biased.yaml` was created;
- no D16 re-run or Phase 1 re-scoring was done.

The step C script (`scripts/phase2_rescore_v2.py`) and the D19 MDA scorer are committed but have not been run.

## 1. The D21 correction and why

D20 criterion 1 defined the drift as "the real 2016–2025 regular-session mean of z". The criterion-3 wording
("mean log return per bar") did not match it. Read literally, it gave 1.73e-5 per bar on all bars, which is
dominated by the overnight gap (that gap never enters an intraday label). The mean z_12 in the null was then
0.028, about 15× real.

D21 replaces it with a constant drift on intraday bars (bar_index > 0), calibrated so that the null's mean
z_12 equals the real one. Nothing else changes: thresholds, seeds 1000..1039 and the last-fold fitted-parameter
approximation are the same. The 15× run stays on record as a stress test (`phase2_metric_fix_v2.md`).

**Calibration**, fixed before the run:
- μ = (0.001924 − b) / (12 · mean(1/S_12)) = **1.4885e-6 per intraday bar**.
- b = −0.000552 is the pooled mean z_12 of the paired no-drift draws 0..39.
- 12 · mean(1/S_12) = 1663.5 on the real rows; the mirror keeps |r|, so S_12 is unchanged.
- Measured pooled mean z_12 of the 40 drift draws: **0.001924**, within ±10% of the real 0.001924.

## 2. D20 criterion 3 with the corrected drift (40 draws, seeds 1000..1039; 83 scoreable features)

| Condition | h = 12 | h = 12 delayed |
|---|---|---|
| Null mean z_12 = real ± 10% | 0.001924 — pass | same rows — pass |
| Every feature's mean unsigned IC within ±0.003 or 3 se | **fail: dow +0.0032 (se 0.0010)**; all other features pass (hmm_p2 −0.0040 and linreg_slope_78 +0.0034 pass on 3 se) | **fail: dow +0.0033 (se 0.0010)** |
| Pooled D10 false-pass rate ≤ 2.2% (nominal 1.67%) | 1.93% (1.60 per draw) — pass | 2.02% (1.68 per draw) — pass |
| No feature above 8% | max 7.5% (earn_react) — pass | **fail: earn_react 10.0% (4 of 40)** |
| **Variant result** | **FAIL** | **FAIL** |

For reference, the no-drift null (60 draws, unchanged from 2026-10-24) passes both targets:
- every feature passes the mean condition (dow +0.0012, earn_react +0.0018);
- pooled false-pass rate 1.69% / 1.77%;
- worst single feature 6.7%.

Tables (`reports/tables/phase2_metric_fix_v3/`): `null_v3_summary.json`, `null_v3_per_feature.csv`,
`null_v3_d10_passes_per_draw.csv`, `null_v3_per_draw_feature.csv`.

## 3. Where the two failures come from

D21 pairs the drift draws with no-drift draws 0..39: the same mirror seeds, and so the same random signs. The
two failures are already in those 40 no-drift draws:

| Feature, target | No-drift draws 0..39 | Drift draws 0..39 | Paired drift effect |
|---|---|---|---|
| dow, h = 12 | mean +0.0032, 0 D10 passes | mean +0.0032, 0 D10 passes | −0.00002 |
| dow, h = 12 delayed | +0.0033 | +0.0033 | −0.00002 |
| earn_react, h = 12 | 3 D10 passes | 3 (the same 3 draws) | −0.00007 |
| earn_react, h = 12 delayed | 4 D10 passes | 4 (the same 4 draws) | −0.00007 |

So these two features fail on the 40-seed subsample, not because of the drift. dow and earn_react are
deterministic calendar and event features, so their null IC depends only on the random signs. Over all 60
no-drift draws they pass (dow +0.0012; earn_react 5.0% / 6.7%).

With 40 draws, the 8% limit means "more than 3 draws". If each feature passed D10 independently at the nominal
1.67%, the chance that at least one of 83 features reaches 4 passes is roughly 30%. That is a property of the
pre-registered thresholds at 40 draws.

This explanation does not change the result. D21 fixed the rule before the run, and the rule says fail.

## 4. Drift sensitivity (B2: from the 2026-10-24 15× pairs, no new compute)

Per feature: sensitivity = mean IC change (drift − no drift) / mean z_12 change (+0.0288), over 40 pairs. The
implied drift-induced bias is sensitivity × k × 0.00192, for k = 1× and 3× real drift. Top 15 by |sensitivity|,
h = 12 (full table, including the delayed target: `drift_sensitivity.csv`):

| Feature | Sensitivity (se) | Implied bias 1× | Implied bias 3× | Predicted 1× shift | Measured 1× shift |
|---|---|---|---|---|---|
| garch_logvar12 | −0.290 (0.0006) | −0.00056 | −0.00167 | −0.00072 | −0.00072 |
| hmm_p2 | −0.259 (0.0012) | −0.00050 | −0.00149 | −0.00064 | −0.00064 |
| vreg_sigma_pct | −0.213 (0.0005) | −0.00041 | −0.00123 | −0.00053 | −0.00053 |
| garch_sigma | −0.210 (0.0005) | −0.00040 | −0.00121 | −0.00052 | −0.00052 |
| vreg_level_pct | −0.196 (0.0006) | −0.00038 | −0.00113 | −0.00048 | −0.00048 |
| parkinson_78 | −0.193 (0.0002) | −0.00037 | −0.00111 | −0.00048 | −0.00048 |
| gk_78 | −0.188 (0.0002) | −0.00036 | −0.00108 | −0.00047 | −0.00047 |
| hmm_p0 | +0.179 (0.0013) | +0.00034 | +0.00103 | +0.00044 | +0.00044 |
| rv_today_deseas | −0.178 (0.0007) | −0.00034 | −0.00103 | −0.00044 | −0.00044 |
| parkinson_12 | −0.152 (0.0002) | −0.00029 | −0.00088 | −0.00038 | −0.00038 |
| range | −0.149 (0.0002) | −0.00029 | −0.00086 | −0.00037 | −0.00037 |
| atr_c | −0.133 (0.0002) | −0.00026 | −0.00077 | −0.00033 | −0.00033 |
| har_rv1 | −0.127 (0.0002) | −0.00024 | −0.00073 | −0.00031 | −0.00031 |
| har_rv22 | −0.118 (0.0002) | −0.00023 | −0.00068 | −0.00029 | −0.00029 |
| kalman_slope | +0.110 (0.052) | +0.00021 | +0.00063 | +0.00027 | +0.00004 |

- **No feature is flagged.** No implied bias at 1× exceeds ±0.003: the maximum is 0.00056, and 0.0017 at 3×.
- The predicted 1× shift uses the actual paired mean z_12 shift of the corrected run (+0.00248, slightly above
  1× because the 40 seeds have a negative baseline).
- **Predicted and measured shifts agree** (correlation 0.991 at h = 12, 0.994 delayed; the largest measured
  shift is 0.00072). The ȳ·mean(s) mechanism of `phase2_metric_fix_v2.md` §4 is linear in the drift, as assumed.
- The 15× run used an all-bar drift, so for intraday-only features the sensitivity is approximate. The one poor
  fit is kalman_slope, whose paired shift is noisy (se 0.05).

## 5. Consequence: the D20 fallback

D19 stays recorded as the formula but **not** as an adopted gate metric. Under the fallback:

1. **Single features:** D10 compares each feature's real signed t (full period, and 2023–2025) with that
   feature's own mirror-null t distribution.
2. **Every model gate** (Gate 3, Phase 4) also requires the model to score ≈ 0 when trained and tested on
   mirror-null data.

Projected cost, as in `phase2_metric_fix_v2.md` §5:

| Item | Cost |
|---|---|
| Per-feature null quantiles at the D10 tail (≈ 98%) | ≈ 200–500 independent draws. 60 independent no-drift draws exist. The 40 drift1x draws reuse the same seeds, so they add almost no independent information. ≈ 51 s per draw on 5 workers → ≈ 2–6 h for 140–440 more draws |
| Exact per-fold refits in the null | ≥ 100 h; not practical. I suggest keeping the last-fold approximation |
| Gate 3 null (LightGBM A/B/C, 96 folds per draw) | not yet timed; first step: time one null walk-forward of model C |
| Phase 4 null (Transformer) | a full Phase 4 run per draw; dominant cost, realistically 10–20 draws |

You need to choose which null to use for the quantiles: the no-drift null, or the corrected-drift null with
new, independent seeds. The B2 sensitivities say the two differ by ≤ 0.0007 in mean IC per feature.

## 6. Status

- D19 has failed D20 / D21; **the D20 fallback applies**.
- The Phase 2 IC tables, D10 passes and `configs/features.yaml` (v1) still stand as computed with the old metric
  (biased, information only).
- `tests/test_metric_null.py` passes (D20 criterion 1).
- Code added this round, committed:
  - `scripts/phase2_metric_null_v2.py`: `drift1x` variant, `--summarize-v3`;
  - `scripts/phase2_metric_drift_sensitivity.py`;
  - `scripts/phase2_rescore_v2.py` (step C, **not run**);
  - `clustered_mda(score="v2")` (D19 scorer: V_s from the other blocks' out-of-fold predictions; not run on
    real data).
