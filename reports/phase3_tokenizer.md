# Phase 3 — Tokenizer and Gate 3 (D15 + D23)

Date 2026-10-25. Data v2, development period 2016-01..2025-12, 96 monthly out-of-sample folds 2018-01..2025-12.
Holdout (2026-01-01..2026-10-09) and forward test not loaded. Gate 3 specification **D23** was recorded in
`reports/decisions.md` (and CLAUDE.md §7.4, §10.6, §12) and committed (5675860) **before any Gate 3 model was trained**.

> **Approximation (applies to every model-null number below):** the mirror null uses the **last fold's (2025-12)
> fitted parameters for GARCH-t, HMM, Kalman and frac-diff in every fold**; the HMM token in the null comes from the
> same last-fold HMM. All other tokenizer parts are refit per fold on the null data; feature lists and versions are
> those of the real run.

**Result: Gate 3 FAILS.** Model C at h = 12 with the delayed outcome has full-period t = 1.39 (< 2) and fails the
model-null condition (needs t ≥ 2.09). A (tokens only) fails as well, and C is **not** better than B. **The D15 stop
condition applies: stop and discuss before Phase 4.** Phase 4 has not been started.

## 1. Tokenizer design (§7.1–7.3, D12, D14)

One composite token per bar = 11 field codes. Code: `src/tslaq/tokenizer/` (`fields.py`, `bins.py`, `shapes.py`,
`tokenizer.py`). Every fitted part is fit **per fold on that fold's training rows only** (sessions before the
validation month) and saved per fold (`data/processed/phase3/tokenizers/<M>.pkl`, 96 files).

| field | input at bar t (causal) | codes | fitted per fold |
|---|---|---|---|
| RET | z_t = r_t / σ̂_{t\|t−1}, clipped ±6 | 64 quantile bins (63 edges = training quantiles k/64); bins with < 50 training rows merged into the inner neighbour, tails first | edges; training bin means of z for de-tokenization |
| RET out (Phase 4 heads) | z_{t→t+h}, clipped ±6 | 32 quantile bins for each h = 1, 3, 12, same merge rule | edges; bin means |
| SHAPE | [(C−O)/ATR, upper shadow/ATR, lower shadow/ATR, close_location, (H−L)/ATR], Wilder ATR(14) incl. bar t | k-means, 32 clusters, on training-standardized inputs clipped ±5 (≤ 100k training rows, 4 starts, seed 0); clusters ordered by mean body | mean/sd, centres |
| REL | previous candle: inside / outside / higher-high-higher-low / lower-high-lower-low (strict inequalities) / mixed (equal high or low) × bullish engulfing / bearish engulfing / none | 15 | rule |
| GAP | gap / σ̂ on bar_index 0 | 9 quantile bins of the training bar-0 gaps; code 9 = "no gap" on every other bar | edges |
| VOLU | ln(V_t / median V at the same bar_index, last 20 sessions) | 12 quantile bins | edges |
| VREG | percentile of σ̂_t within its trailing 20 sessions (D14) | 8 quantile bins | edges |
| HMM | filtered most-likely state of the fold's 3-state Gaussian HMM on [u, ln σ̂] (numba forward recursion; Phase 2 fold fit, states ordered by variance) | 3 | HMM (fold checkpoint) |
| MKT | QQQ same-bar return / QQQ σ̂; MISSING on D12 market gaps | 16 quantile bins | edges |
| TOD | bar_index // 6 | 13 | rule |
| DOW | day of week | 5 | rule |
| EVENT | earnings-reaction session > session after it > opex > half-day > none (priority on overlap) | 5 | rule |

Each field's vocabulary = its value codes followed by **MISSING, PAD, MASK, BOS, DAY_SEP**: 183 value codes, 238
embedding rows. MISSING = warm-up, missing/forward-filled TSLA bars (and the next bar for REL), non-finite inputs,
D12 market gaps. Vocabulary sizes are fixed, so a merged bin leaves a code unused instead of renumbering.

## 2. Tokenizer checks (§7.4) — pass

Table: `reports/tables/phase3/tokenizer_checks_summary.csv` (per fold: `tokenizer_checks_by_fold.csv`).

| check | result (min / median / max over 96 folds) |
|---|---|
| 1. RET occupancy | 64 bins in every fold; min training rows per bin 527 / 1,457 / 2,944; **0 merges in any fold** (RET output bins h = 1/3/12: min 862–1,048 rows, 0 merges) |
| 2. Round-trip, RET (input bins) | corr(de-tokenized z, true z clipped ±6): train 0.991 / 0.991 / 0.992, test month 0.981 / 0.992 / 0.997. corr in return space (× σ̂): train 0.957 / 0.978 / 0.981, test 0.765 / 0.994 / 0.999 (the 0.765 month, 2019-10, has one return beyond 6σ that the token clips). RMSE in σ units: train 0.133 / 0.137 / 0.139, test 0.082 / 0.133 / 0.209 |
| 2. Round-trip, output bins | corr in z: h = 1 train 0.985, h = 3 0.983, h = 12 0.980 (medians); test 0.938–0.995; RMSE 0.15–0.20 σ (median) |
| 3. HMM token | valid in all 96 folds; training state shares 0.34–0.37 / 0.33–0.38 / 0.20–0.26 (calm / middle / high-vol); MISSING ≤ 0.5% of any test month; 2.6–13% of training rows (see note) |
| 4. Tests | `tests/test_tokenizer.py`: 7 tests incl. **`test_no_lookahead_tokens`** (edges / k-means / HMM fit on a fixed period, 50 random truncations, tokens and the 24-bar lag matrix identical) and **`test_tokenizer_roundtrip`** (synthetic: RET corr 0.983, RMSE 0.20 σ; output bins 0.981 / 0.986 / 0.989). **Full suite 59 / 59 pass.** |
| field MISSING in test months | 0 in every field and fold except fold 2018-01 (≤ 0.7%) and MKT in 2018-05 (9.2%: the QQQ gap 2018-05-02/03, D12) |
| SHAPE | all 32 clusters used in every fold; smallest cluster 89 (2018-01) to 577 training rows |
| GAP | 9 bins in every fold |

Note on HMM MISSING in training: the filtered probabilities are NaN where σ̂ is undefined, mainly the 60-session
seasonal-profile warm-up at the start of 2016. That warm-up is 13% of the shortest training window (fold 2018-01) and
2.6% of the longest (2025-12); test months have ≤ 0.5%.

**Why `hmm_p2` has no valid month after mid-2022 (Phase 2), and why the HMM token is still valid.** `hmm_p2` (the
filtered probability of the high-variance state) is a continuous feature and went through the Phase 2 ADF/KPSS
screen. It fails KPSS in raw form in every fold (p ≤ 0.024) because the high-vol regime is persistent (p₂₂ ≈
0.94–0.95), so its level shifts with the market regime (e.g. 2020–2022). The rolling-z version passed until fold
2022-06 (KPSS p 0.05–0.10) and fails from 2022-07 onward (p 0.010–0.021): in the folds 2022-07..2025-12 the feature
was **dropped by the stationarity screen**, so it has no out-of-sample month in 2023–2025. The HMM fit itself is
stable in those folds (state u-variances e.g. 2.09/2.94/7.87 in 2022-07, 2.01/2.98/6.87 in 2025-12). The HMM token
is the categorical argmax state; it is not subject to the ADF/KPSS screen (as `hmm_state` in Phase 2), has no
MISSING in test months beyond 0.5%, and all 3 states occur in every fold's training data (test-month shares vary
with the regime, e.g. high-vol state 0.08–0.97 of a month).

## 3. Token conditional up-probability tables (§6.9) — training rows only

Per fold, on the fold's training rows; h = 1, 3, 12 with and without the one-bar delay; n ≥ 500 per cell; Wilson
and binomial test with n_eff = n/h; BH at FDR 10% across all token cells of one fold and outcome. Tables:
`reports/tables/phase3/token_condprob_summary.csv` (every cell that survives BH in any fold: number of folds,
mean lift, last-fold n / lift / Wilson / year-sign share), `token_condprob_bh_by_fold.csv`,
`token_condprob_lastfold.csv` (all cells, last fold, with by-year sign share), `token_condprob_insufficient.csv`.

- **Sequences:** no 64 × 64 RET→RET and no 64 × 32 RET→SHAPE cell reaches 500 bars in any fold (4,096 / 2,048
  cells, ~40–80 bars each). They are reported as insufficient. A **supplementary** family with RET in 8 groups of 8
  bins (RET8→RET8, RET8→SHAPE) has its own BH. REL × VOLU: ~40 of 180 cells sufficient. GAP: each bar-0 bin has
  ~270 training bars, so only the "no gap" cell is tested.
- BH survivors per fold (mean, spec family): **h = 12: 0, h = 12 delayed: 0 — in every fold**; h = 3: 0.1
  (delayed 0); h = 1: 30 (delayed 18.5). Grouped family: h = 1 18.5 (delayed 2.9), h = 3 and h = 12 0.
- What survives is short-horizon **reversal**: h = 1, RET bin 0 (most negative z) → P(up next bar) +6.8 pp (96/96
  folds), RET8 0→0 +9.6 pp, RET8 6→7 −6.7 pp; REL "lower-high-lower-low" +1.9 pp, "higher-high-higher-low" −1.7 pp
  (96/96). At h = 3 only REL (lh_ll +1.6 pp, 58 folds; hh_hl −1.5 pp, 40 folds) and 3 SHAPE cells, and **none
  survives the one-bar delay at h = 3**. With the delay the h = 1 count drops by ~40%. This is the bid–ask bounce /
  1-bar reversal already seen in Phase 2, and nothing of it reaches h = 12.

## 4. Gate 3 (D23)

LightGBM with the fixed Phase 1 parameters (`num_threads = 1`), objective L2 on z_h clipped ±6, no tuning. A = 264
token columns (11 fields × bars t−23..t, categorical); B = the fold's `by_fold` features (`configs/features.yaml` v2,
17–32 per fold); C = A + B. One model per variant and h; 96 folds, purge + 78-bar embargo; the first fold's V_s from
its D6 inner-validation block (2017-11..12). Scoring: D19 model formula. Fold ICs: `reports/tables/phase3/gate3_fold_ic.csv`.

### 4.1 Gate 3 table

`reports/tables/phase3/gate3_results.csv`, `gate3_model_null_summary.csv` (per draw: `gate3_model_null_per_draw.csv`).
Model null: K = 20 draws for each of A, B, C (seeds 3000..3019), h = 12 only. "thr" = null mean t + 2 null sd.

| variant | h | delay | IC full | t full | IC 23–25 | t 23–25 | fixed D10 | null mean IC | null mean t (sd) | thr | model null | 
|---|---|---|---|---|---|---|---|---|---|---|---|
| **C** | **12** | **yes** | **0.0078** | **1.39** | 0.0097 | 1.28 | **fail** | +0.0005 | 0.07 (1.01) | 2.09 | **fail** |
| C | 12 | no | 0.0105 | 1.82 | 0.0114 | 1.52 | fail | +0.0005 | 0.08 (0.95) | 1.98 | fail |
| C | 3 | no / yes | 0.0091 / 0.0040 | 3.32 / 1.44 | 0.0036 / 0.0022 | 1.08 / 0.76 | pass / fail | – | – | – | no null (h ≠ 12) |
| C | 1 | no / yes | 0.0068 / 0.0024 | 3.21 / 1.15 | 0.0035 / 0.0003 | 1.19 / 0.14 | pass / fail | – | – | – | no null |
| B | 12 | yes | 0.0167 | 1.92 | 0.0071 | 0.58 | fail | −0.0001 | −0.01 (1.17) | 2.34 | fail |
| B | 12 | no | 0.0194 | 2.25 | 0.0094 | 0.76 | fail | +0.0001 | 0.03 (1.18) | 2.38 | fail |
| B | 3 | no / yes | 0.0378 / 0.0235 | 9.09 / 5.85 | 0.0262 / 0.0154 | 4.28 / 2.66 | pass / pass | – | – | – | no null |
| B | 1 | no / yes | 0.0395 / 0.0119 | 12.76 / 5.34 | 0.0219 / 0.0082 | 6.00 / 2.35 | pass / pass | – | – | – | no null |
| A | 12 | yes | 0.0007 | 0.15 | 0.0094 | 1.27 | fail | +0.0004 | 0.03 (1.21) | 2.45 | fail |
| A | 12 | no | 0.0026 | 0.51 | 0.0087 | 1.19 | fail | +0.0005 | 0.06 (1.11) | 2.27 | fail |
| A | 3 | no / yes | 0.0039 / 0.0014 | 1.45 / 0.50 | 0.0004 / 0.0001 | 0.13 / 0.04 | fail / fail | – | – | – | no null |
| A | 1 | no / yes | 0.0041 / −0.0006 | 1.92 / −0.27 | −0.0014 / −0.0013 | −0.43 / −0.57 | fail / fail | – | – | – | no null |

Other metrics (same file): share of folds with IC > 0 at h = 12 delayed: C 57% (binomial p 0.09), B 59% (0.04), A 48%;
last 24 folds h = 12 delayed: C IC 0.0057 t 0.62, B −0.0004 t −0.03, A 0.0087 t 1.09. Hit rate h = 12: C 0.503,
B 0.507, A 0.499. CRPS (z, Gaussian with training residual sd) h = 12: A/C 0.5482, B 0.5492 vs the zero forecast
0.5463 (Phase 1): **no variant beats the zero forecast on CRPS**. Old monthly Spearman (biased, information only)
h = 12: C 0.010, B 0.019, A 0.001. Fold IC by year (C, h = 12 delayed): 2018 .010, 2019 .005, 2020 −.002,
2021 .017, 2022 .004, 2023 .018, 2024 .012, 2025 −.001.

The null behaves as intended: null mean IC within ±0.0005 for every variant (limit ±0.003), null fixed-D10
false-pass rate 0% (C, B) and 5% (A, 1 of 20). Pooled null mean z_12 = 0.0024 (target 0.0019; D22 drift not
recalibrated on the 20 new seeds, as fixed in D23; the per-draw sd of the no-drift mean z_12 is 0.009, so 20
draws are within noise).

### 4.2 Verdict (D23)

| condition | result |
|---|---|
| C, h = 12 delayed, fixed D10 | **fail**: full t 1.39 < 2 (2023–25 part passes: IC 0.0097 > 0, t 1.28 ≥ 1) |
| C, h = 12 delayed, model null | null mean IC +0.0005 (within ±0.003: yes); real t 1.39 < 0.07 + 2 × 1.01 = 2.09: **fail** |
| **Gate 3** | **FAIL** |
| A fails? | **yes** (h = 12 delayed t 0.15; null threshold 2.45) |
| C better than B? (mean diff > 0 and one-sided Wilcoxon p < 0.05 at h = 12 delayed) | **no** (§5) |
| D15 stop condition (A fails and C not better than B) | **met → stop and discuss before Phase 4** |

## 5. Value of the tokens: paired comparisons

`reports/tables/phase3/gate3_paired.csv`. Share of folds won by the first model; Wilcoxon signed-rank on fold ICs.
Phase 1 `lgb_returns` = the stored Phase 1 predictions (12 lagged scaled returns, 24/78-bar sums, bar_index; no
retraining) re-scored with D19, same 95 folds (2018-02..2025-12).

| comparison | h | delay | mean IC (first / second) | diff | share won | Wilcoxon p two-sided / greater |
|---|---|---|---|---|---|---|
| **C vs B** | **12** | **yes** | 0.0078 / 0.0167 | −0.0088 | 49% | 0.46 / 0.77 |
| C vs B | 12 | no | 0.0105 / 0.0194 | −0.0089 | 49% | 0.43 / 0.79 |
| C vs B | 3 | no / yes | 0.0091 / 0.0378, 0.0040 / 0.0235 | −0.029 / −0.019 | 27% / 27% | < 1e-4 / < 1e-4 |
| C vs B | 1 | no / yes | 0.0068 / 0.0395, 0.0024 / 0.0119 | −0.033 / −0.010 | 14% / 44% | < 1e-4 / 0.007 |
| C vs A | 12 | yes | 0.0078 / 0.0007 | +0.0071 | 56% | 0.10 / 0.052 |
| **C vs Phase 1 lgb_returns** | **12** | **yes** | 0.0071 / 0.0203 | −0.0132 | 46% | 0.15 / 0.93 |
| C vs Phase 1 lgb_returns | 12 | no | 0.0096 / 0.0221 | −0.0125 | 45% | 0.15 / 0.93 |
| C vs Phase 1 lgb_returns | 3 | no / yes | 0.0090 / 0.0306, 0.0039 / 0.0208 | −0.022 / −0.017 | 21% / 32% | < 1e-4 / < 1e-4 |
| C vs Phase 1 lgb_returns | 1 | no / yes | 0.0072 / 0.0234, 0.0022 / 0.0080 | −0.016 / −0.006 | 27% / 39% | < 1e-4 / 0.018 |

Phase 1 lgb_returns under D19 (information; no model null was run for it): h = 12 IC 0.0221 t 3.93 (delayed 0.0203,
t 3.56), 2023–25 t 1.49 (delayed 1.19).

**Reading.** Adding the 264 token columns to B makes the model **worse** at every horizon, significantly so at
h = 1 and h = 3, and indistinguishable at h = 12. The tokens alone carry little (A ≈ 0 at h = 12, ≈ 0.004 at
h = 1/3 and negative after the delay). With fixed parameters (300 trees, 15 leaves, column subsample 0.8), the 264
categorical columns take most of the splits away from the 17–32 continuous features. That is a finding about this
LightGBM encoding; D23 forbids tuning it, and it does not by itself say how a Transformer with embeddings would
use the tokens. C is also below the Phase 1 lagged-returns model. B (the selected features) is strong at h = 1 and
h = 3 (delayed h = 3 t 5.85, 2023–25 t 2.66), but at h = 12 it misses D10 (delayed t 1.92) and its model-null
threshold (2.34).

## 6. Timing, K and registry

- Real walk-forward: projected 0.73 h with 5 workers after timing the last fold (199 s); actual 47 min
  (10:41–11:28), 13,696 s of fold compute for models + 266 s for checks and token tables.
- Model null (budget 12 h from 11:28:30, the start of the timed draw; script `scripts/phase3_null_chain.py`,
  log `data/processed/phase3/null_chain.log`):
  - C: timed first draw (seed 3000) **654 s** → 20 × 654 s = 3.6 h ≤ 12 h → **K = 20**; done 15:03.
  - B: timed draw 356 s; 20 × 356 s = 1.98 h ≤ 8.42 h remaining → K = 20; done 16:59.
  - A: timed draw 561 s; 20 × 561 s = 3.12 h ≤ 6.48 h remaining → K = 20; done 19:57.
  - Total 8.47 h of the 12 h. **No variant is "unconfirmed".**
- Registry: 9 trials `gate3_{A,B,C}_h{1,3,12}` (phase 3; `rank_ic_h*` = D19 mean fold IC; notes carry delayed IC,
  2023–25 and the model-null result). `n_trials_so_far` = 18. Null draws are diagnostics, not trials.

## 7. Failures, deviations and caveats

- **Gate 3 fails** (C h = 12 delayed: D10 and model null). D15 stop condition met.
- The 64-bin RET sequences cannot be tested at n ≥ 500; a grouped RET8 family is reported separately (own BH).
- Model null approximation: last-fold fitted parameters (GARCH-t, HMM, Kalman, frac-diff) and the last-fold HMM for
  the HMM token in every fold.
- The D6 inner-validation model of the first fold uses fold 2018-01's feature list and fitted features (fit on
  data < 2018-01, which includes the inner block); it only sets V_s for fold 2018-01.
- `num_threads = 1` instead of Phase 1's 4 (compute setting, D23).
- Rows: finite z_h and S_h (Phase 1 additionally required its own lagged-return features; slightly different row sets).

## 8. Proposed next step (for discussion; nothing started)

Per D15 we stop here. Options for you to decide between:

1. **Proceed to Phase 4 anyway**, on the grounds that the LightGBM categorical encoding is a poor test of tokens for
   a Transformer (embeddings, attention over 256 bars). Gate 3 would be recorded as failed; Phase 4's own gates
   (paired tests vs LightGBM/CatBoost, D10, model null) would then decide. Risk: Phase 3 gave no evidence that the
   tokens carry h = 12 information (A ≈ 0; no token cell survives BH at h = 12 in any fold).
2. **Pre-register one reduced token variant** for LightGBM (e.g. fewer lags of RET/SHAPE/REL plus the day-level
   fields), as a new trial with its own null, before deciding. This is close to tuning on Gate 3 and costs trials.
3. **Re-examine the h = 12 target.** Even B fails h = 12 with the null, while B is strong at h = 1/3 (and survives
   the delay at h = 3: t 5.85, 2023–25 t 2.66). The h = 12 single features passed D10\* individually (Phase 2), but a
   model combining them does not. A change of primary horizon would be a new decision (D2) and must not be chosen
   from these results without care.

My recommendation is to discuss 1 vs 3 before any further run: the evidence says the tokens add nothing in this
form, and the remaining signal is at h = 1–3, where costs dominate (D2).

## 9. Files

Code: `src/tslaq/tokenizer/*`, `src/tslaq/models/gate3.py`, `src/tslaq/selection/token_condprob.py`,
`scripts/run_phase3_gate3.py` (`--time-one`, `--real`, `--null V --draws a b`, `--summarize --log`),
`scripts/phase3_null_chain.py`, `tests/test_tokenizer.py`. Tables: `reports/tables/phase3/`. Checkpoints (gitignored):
`data/processed/phase3/` (real folds, tokenizers, null draws, logs).
