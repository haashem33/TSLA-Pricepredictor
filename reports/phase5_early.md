# Early Phase 5 test: economic check of the h = 12 predictions (D26, D27)

Date 2026-10-26. Data v2, out-of-sample period 2018-01..2025-12 (96 monthly folds, 2,010 test sessions). The
holdout (2026-01-01..2026-10-09) and the forward test were not loaded: the bar files end 2025-12-31, and the script
asserts this. D26 and D27 were recorded in `reports/decisions.md` (and CLAUDE.md §10.4, §10.7, §12) and committed
(e2b82a1) **before any D27 backtest was run**. No model was retrained except in the optional CPCV.

**Result: no trial passes the pre-registered D27 gate.** Gate-3 B is the only candidate with positive net Sharpe
under every non-DSR condition (R1: full period 1.11 at D26 cost, 0.10 at 2× D26, 2023–25 1.06, +1 bar delay 0.62;
R2: 0.93 / 0.55 / 0.47 / 0.49). Both B trials fail on the DSR, which is 0.000 with the pre-registered V (SR\* = 1.16
per day). M3 and M1 also fail the cost and delay conditions. **Decision rule: none passes, so the recommendation is to
conclude with a final report; Phase 4 only as a long shot you choose, with the IC target in §6 as its explicit bar.**
Read §7 first: as pre-registered, the DSR condition could not have been met by any plausible strategy.

## 1. Decisions (as recorded)

- **D26 (realistic cost).** Cost per side in calendar year Y = the EDGE mean half-spread of year Y−1
  (`reports/tables/phase3_baselines/spread_by_year.csv`, prior year only) + 0.10 bp (SEC Section 31 fee, $20.60 per
  $1M on sales only, averaged over buys and sells; FINRA TAF negligible). Per side: 2018 1.474, 2019 1.738,
  2020 1.287, 2021 1.742, 2022 1.078, 2023 1.113, 2024 0.607, 2025 0.729 bp (time-weighted mean 1.22 bp in
  2018–25, 0.82 bp in 2023–25). Every result is also given at 2× D26 and at 0.5 / 1 / 3 / 6 bp per side.
  **This cost choice was made after the D25 break-even numbers were seen.** The spread measurement itself does not
  depend on any model.
- **D27 (early Phase 5 test, pre-registered).** Full text: `reports/decisions.md`, D27. In short:
  - **Candidates.** M3, Gate-3 B and M1, using their stored h = 12 out-of-sample predictions (M3: its score s).
  - **Threshold q.** q = 90th percentile of |ẑ| on the D6 block for fold 2018-01, and on the previous
    out-of-sample month for every later fold.
  - **R1 (proportional).** w = clip(ẑ/q, ±1), with a no-trade band of 0.25.
  - **R2 (strong signal).** If |ẑ| > q, hold a unit position with sign(ẑ) for 12 bars, non-overlapping.
  - **Execution.** Fill at the next bar's open, no fills in the last 10 minutes, flat at the close.
  - **Trials.** 6 trials in total.
  - **Pass.** All four must hold: full-period net Sharpe > 0 at 2× D26; 2023–25 net Sharpe > 0 at D26; full-period
    net Sharpe > 0 at D26 with one extra bar of delay; DSR > 0.95.
- **One clarification made while running.** M3 has no stored D6-block scores. Its fold-2018-01 q uses M3's score on
  2017-11..12, computed with anchors and signs from sessions before 2017-11 and fold 2018-01's feature list. This is
  the D6 analogue of the models' inner predictions; only anchors are computed, no model is fitted. Every other q
  follows D27 as written.

Definitions used in the tables:

- **Trade.** R1: a fill. R2: a round trip.
- **Turnover.** Mean daily Σ|Δw|, including the final flattening.
- **Hit rate and bp/trade.** R1: holding segments between fills, per unit position. R2: trades.
- **Break-even.** Mean daily gross / mean daily turnover.
- **Sharpe.** Daily P&L over all 2,010 test sessions (flat days included), × √252.
- **Sortino.** Uses the downside deviation √mean(min(r, 0)²).
- **Max drawdown.** Of the cumulative daily log P&L, in log-return percentage points. Values above 100 are possible
  because the series is summed in log units.

## 2. Results per trial

`reports/tables/phase5_early/backtest_results.csv` has every cost level and both periods, with and without the extra
delay (one row per trial × delay × period × cost). Sharpe values are annualized; "t" = daily mean / sd × √n_days.

| candidate | rule | +delay | period | trades/day | turnover/day | hit | bp/trade | break-even bp/side | D26 bp/side | Sharpe gross | **D26** | t D26 | Sortino D26 | MDD % D26 | **2×D26** | **3 bp** | 0.5 bp | 1 bp | 6 bp |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| M3 | R1 | 0 | 2018-2025 | 23.29 | 10.78 | 0.443 | 0.52 | 0.78 | 1.22 | 0.72 | **-0.39** | -1.11 | -0.63 | 128.4 | **-1.48** | **-1.98** | 0.26 | -0.20 | -4.53 |
| M3 | R1 | 0 | 2023-2025 | 24.51 | 11.45 | 0.440 | 0.35 | 0.91 | 0.82 | 0.92 | **0.08** | 0.14 | 0.14 | 28.0 | **-0.74** | **-2.06** | 0.41 | -0.09 | -4.86 |
| M3 | R1 | 1 | 2018-2025 | 23.29 | 10.78 | 0.479 | 0.41 | 0.91 | 1.22 | 0.87 | **-0.29** | -0.81 | -0.46 | 95.2 | **-1.42** | **-1.94** | 0.39 | -0.08 | -4.61 |
| M3 | R1 | 1 | 2023-2025 | 24.51 | 11.45 | 0.475 | 0.29 | 0.59 | 0.82 | 0.65 | **-0.26** | -0.45 | -0.41 | 49.3 | **-1.16** | **-2.58** | 0.10 | -0.45 | -5.63 |
| M3 | R2 | 0 | 2018-2025 | 1.77 | 3.55 | 0.500 | 3.87 | 1.93 | 1.22 | 0.57 | **0.21** | 0.60 | 0.34 | 53.0 | **-0.15** | **-0.32** | 0.43 | 0.28 | -1.25 |
| M3 | R2 | 0 | 2023-2025 | 1.79 | 3.59 | 0.501 | 6.28 | 3.14 | 0.82 | 0.98 | **0.73** | 1.26 | 1.21 | 24.5 | **0.47** | **0.04** | 0.83 | 0.67 | -0.93 |
| M3 | R2 | 1 | 2018-2025 | 1.76 | 3.52 | 0.508 | 2.40 | 1.20 | 1.22 | 0.37 | **-0.01** | -0.02 | -0.01 | 54.5 | **-0.39** | **-0.57** | 0.22 | 0.06 | -1.54 |
| M3 | R2 | 1 | 2023-2025 | 1.79 | 3.57 | 0.506 | 2.98 | 1.49 | 0.82 | 0.49 | **0.22** | 0.39 | 0.36 | 32.1 | **-0.05** | **-0.51** | 0.33 | 0.16 | -1.56 |
| Gate-3 B | R1 | 0 | 2018-2025 | 18.01 | 8.28 | 0.512 | 1.96 | 2.64 | 1.22 | 2.11 | **1.11** | 3.12 | 1.71 | 45.5 | **0.10** | **-0.29** | 1.71 | 1.31 | -2.67 |
| Gate-3 B | R1 | 0 | 2023-2025 | 15.99 | 7.18 | 0.503 | 1.69 | 2.25 | 0.82 | 1.68 | **1.06** | 1.83 | 1.62 | 27.0 | **0.44** | **-0.56** | 1.31 | 0.93 | -2.76 |
| Gate-3 B | R1 | 1 | 2018-2025 | 18.01 | 8.28 | 0.504 | 1.43 | 2.01 | 1.22 | 1.67 | **0.62** | 1.75 | 0.96 | 44.6 | **-0.43** | **-0.83** | 1.25 | 0.84 | -3.30 |
| Gate-3 B | R1 | 1 | 2023-2025 | 15.99 | 7.18 | 0.503 | 1.15 | 1.31 | 0.82 | 1.05 | **0.39** | 0.67 | 0.59 | 28.7 | **-0.27** | **-1.34** | 0.65 | 0.25 | -3.67 |
| Gate-3 B | R2 | 0 | 2018-2025 | 1.43 | 2.85 | 0.528 | 8.37 | 4.19 | 1.22 | 1.32 | **0.93** | 2.63 | 1.45 | 29.1 | **0.55** | **0.37** | 1.16 | 1.00 | -0.57 |
| Gate-3 B | R2 | 0 | 2023-2025 | 1.32 | 2.65 | 0.511 | 4.56 | 2.28 | 0.82 | 0.73 | **0.47** | 0.81 | 0.68 | 25.3 | **0.21** | **-0.23** | 0.57 | 0.41 | -1.18 |
| Gate-3 B | R2 | 1 | 2018-2025 | 1.41 | 2.81 | 0.524 | 5.44 | 2.72 | 1.22 | 0.90 | **0.49** | 1.39 | 0.74 | 41.0 | **0.09** | **-0.09** | 0.73 | 0.57 | -1.08 |
| Gate-3 B | R2 | 1 | 2023-2025 | 1.31 | 2.61 | 0.510 | 1.40 | 0.70 | 0.82 | 0.23 | **-0.03** | -0.06 | -0.05 | 31.2 | **-0.30** | **-0.75** | 0.07 | -0.10 | -1.73 |
| M1 | R1 | 0 | 2018-2025 | 28.22 | 14.31 | 0.502 | 0.69 | 1.00 | 1.22 | 1.46 | **-0.37** | -1.05 | -0.57 | 105.2 | **-2.18** | **-2.90** | 0.73 | -0.01 | -7.11 |
| M1 | R1 | 0 | 2023-2025 | 26.04 | 12.86 | 0.497 | 0.32 | 0.64 | 0.82 | 0.95 | **-0.25** | -0.43 | -0.39 | 47.7 | **-1.43** | **-3.41** | 0.21 | -0.52 | -7.58 |
| M1 | R1 | 1 | 2018-2025 | 28.22 | 14.31 | 0.501 | 0.65 | 0.99 | 1.22 | 1.54 | **-0.41** | -1.16 | -0.63 | 88.2 | **-2.34** | **-3.11** | 0.76 | -0.02 | -7.60 |
| M1 | R1 | 1 | 2023-2025 | 26.04 | 12.86 | 0.492 | 0.04 | 0.44 | 0.82 | 0.68 | **-0.58** | -1.00 | -0.89 | 55.9 | **-1.83** | **-3.91** | -0.10 | -0.87 | -8.29 |
| M1 | R2 | 0 | 2018-2025 | 1.84 | 3.69 | 0.502 | 2.80 | 1.40 | 1.22 | 0.58 | **0.07** | 0.19 | 0.10 | 51.7 | **-0.45** | **-0.67** | 0.37 | 0.17 | -1.93 |
| M1 | R2 | 0 | 2023-2025 | 1.73 | 3.47 | 0.498 | 0.76 | 0.38 | 0.82 | 0.17 | **-0.19** | -0.33 | -0.28 | 21.5 | **-0.55** | **-1.16** | -0.05 | -0.27 | -2.51 |
| M1 | R2 | 1 | 2018-2025 | 1.81 | 3.62 | 0.500 | 1.64 | 0.82 | 1.22 | 0.34 | **-0.17** | -0.49 | -0.25 | 44.6 | **-0.69** | **-0.91** | 0.13 | -0.08 | -2.17 |
| M1 | R2 | 1 | 2023-2025 | 1.70 | 3.40 | 0.504 | 0.55 | 0.28 | 0.82 | 0.12 | **-0.24** | -0.42 | -0.35 | 25.8 | **-0.61** | **-1.23** | -0.10 | -0.33 | -2.61 |

Per-period (daily, not annualized) net Sharpe at D26, full period: M3 R1 −0.0247, M3 R2 +0.0134, B R1 +0.0696,
B R2 +0.0587, M1 R1 −0.0235, M1 R2 +0.0043. Net Sharpe at D26 by year (information only;
`sharpe_by_year_d26.csv`):

| year | M3 R1 | M3 R2 | B R1 | B R2 | M1 R1 | M1 R2 |
|---|---|---|---|---|---|---|
| 2018 | −0.52 | −0.08 | −0.03 | 1.05 | −2.82 | −1.63 |
| 2019 | −2.34 | −0.98 | −0.65 | −0.94 | −2.31 | 0.25 |
| 2020 | −0.67 | 0.23 | 2.67 | 2.23 | 1.26 | 1.11 |
| 2021 | −0.39 | −0.30 | 0.81 | 1.36 | −1.33 | 0.42 |
| 2022 | 0.06 | 0.35 | 1.70 | 1.22 | 0.72 | 0.47 |
| 2023 | 0.13 | 1.06 | 0.75 | 0.82 | −0.24 | −0.17 |
| 2024 | −0.22 | 0.05 | 1.44 | 0.86 | −1.18 | −0.57 |
| 2025 | 0.27 | 1.00 | 1.01 | −0.08 | 0.55 | 0.12 |

Reading:

- **Gate-3 B R1 is the strongest trial.**
  - Its break-even is 2.64 bp per side, about 2× the D26 cost, and it is net positive at D26 in 6 of 8 years.
  - Its edge is thin against costs. At 2× D26 the full-period Sharpe is only 0.10. With one extra bar of delay the
    break-even falls to 2.01 bp, and the result turns negative at 2× D26.
  - At the old 3 bp default, every R1 trial is negative.
- **M3 has the highest IC but the weakest P&L.** R1 breaks even at 0.78 bp, below D26. This matches the D25
  observation that its D19 information sits in a few large scores, not in the sign of typical bars.
- **M1 does not cover D26 under either rule.**

## 3. Pass table (D27 gate)

`reports/tables/phase5_early/pass_table.csv`

| candidate | rule | full 2× D26 > 0 | 2023–25 D26 > 0 | full D26 +1 delay > 0 | DSR > 0.95 | **pass** |
|---|---|---|---|---|---|---|
| M3 | R1 | −1.48 ✗ | 0.08 ✓ | −0.29 ✗ | 0.000 ✗ | **no** |
| M3 | R2 | −0.15 ✗ | 0.73 ✓ | −0.01 ✗ | 0.000 ✗ | **no** |
| Gate-3 B | R1 | 0.10 ✓ | 1.06 ✓ | 0.62 ✓ | 0.000 ✗ | **no** |
| Gate-3 B | R2 | 0.55 ✓ | 0.47 ✓ | 0.49 ✓ | 0.000 ✗ | **no** |
| M1 | R1 | −2.18 ✗ | −0.25 ✗ | −0.41 ✗ | 0.000 ✗ | **no** |
| M1 | R2 | −0.45 ✗ | −0.19 ✗ | −0.17 ✗ | 0.000 ✗ | **no** |

## 4. Deflated Sharpe ratio (§10.7)

`reports/tables/phase5_early/dsr.csv`. Inputs: full-period daily net returns at D26 cost, no extra delay; T = 2,010
days; N = 42 (n_trials_so_far after logging the 6 D27 trials). V is the variance of the per-period Sharpe ratios of
the 14 rule-backtest trials in the registry (`dsr_rule_trials_in_V.csv`): D16 ×2, D25 ×6 (logged at 3 bp) and
D27 ×6 (at D26). That gives V = 0.2773 and SR\* = 1.163 per day.

| candidate | rule | SR̂ per day | γ3 (skew) | γ4 (raw kurt.) | T | SR\* | **DSR** | PSR(0) | *info: SR\* with V of D27 only (0.00161)* | *info: DSR* |
|---|---|---|---|---|---|---|---|---|---|---|
| M3 | R1 | −0.0247 | 1.21 | 9.96 | 2010 | 1.163 | **0.000** | 0.138 | 0.0887 | 0.000 |
| M3 | R2 | 0.0134 | 1.06 | 9.69 | 2010 | 1.163 | **0.000** | 0.728 | 0.0887 | 0.000 |
| Gate-3 B | R1 | 0.0696 | 0.20 | 8.28 | 2010 | 1.163 | **0.000** | 0.999 | 0.0887 | 0.196 |
| Gate-3 B | R2 | 0.0587 | 0.41 | 12.80 | 2010 | 1.163 | **0.000** | 0.996 | 0.0887 | 0.088 |
| M1 | R1 | −0.0235 | 0.69 | 9.19 | 2010 | 1.163 | **0.000** | 0.149 | 0.0887 | 0.000 |
| M1 | R2 | 0.0043 | 0.52 | 11.86 | 2010 | 1.163 | **0.000** | 0.576 | 0.0887 | 0.000 |

V is dominated by the D25 h = 1 trials at 3 bp (per-day Sharpe −1.30 and −1.61). A per-day SR\* of 1.163 means an
annualized Sharpe of about 18, so with this V no strategy could pass. With V from the 6 like-for-like D27 trials only
(information, not the pre-registered rule), SR\* = 0.089 per day (1.41 annualized), and the best DSR is 0.196 (B R1).
That is still far below 0.95. Without any deflation, PSR(0) for B R1 is 0.999.

## 5. CPCV (§10.2): Gate-3 B, R1 (highest full-period net Sharpe at D26)

`reports/tables/phase5_early/cpcv_sharpes.csv`, `cpcv_summary.json`.

**Setup.**

- Sessions 2016-01-04..2025-12-31 in 10 blocks of equal session count (block boundaries in the summary JSON).
- 45 splits with k = 2 test blocks each.
- Purge plus 78-bar embargo.
- LightGBM retrained per split with the fixed Phase 1 parameters.
- q from a 42-session inner block of training sessions, predicted by a model trained on the rest.
- R1 at D26 cost. 2016 uses its own-year spread, since there is no prior year in the table.

**Run time.** One split took 11 s, so the projected total was about 2 min with 5 workers. The actual run took about
3 min wall time (532 s of fit compute), well inside the 6-hour box.

| | n | median net Sharpe (D26) | mean | min | max | share > 0 | median gross Sharpe |
|---|---|---|---|---|---|---|---|
| **paths** | 9 | **0.68** | 0.69 | 0.51 | 0.89 | **100%** | 1.69 |
| splits | 45 | 0.64 | 0.63 | −0.92 | 1.96 | 82% | 1.78 |

**Approximations that make this optimistic (stated per D27):**

1. The fitted features (GARCH-t, HMM, Kalman, frac-diff) and the 27-feature list are the last fold's (2025-12).
   That list was selected on data through 2025-11, which includes most test blocks.
2. In CPCV, training uses data from after the test blocks.
3. This trial was picked for CPCV because it had the best Sharpe of the six.

The CPCV shows that the B R1 result is not driven by one period. It does not correct for selection.

## 6. Null Sharpes and the IC target (information)

**Null-draw Sharpes: not run.** The D23/D25 null draws on disk (`data/processed/phase3/null_B/`,
`data/processed/phase3_baselines/null_M1/`, `null_M3/`) store only per-fold ICs, not predictions, and the null
frames were deleted after scoring. D27 made this step conditional on stored predictions, so it was skipped.
Regenerating them would mean retraining on the null draws: about 275 s per seed for all four D25 models, and about
1.5 min per seed for B.

**IC target.** `reports/tables/phase5_early/ic_target.csv`. This is the approximate h = 12 delayed IC needed for a
net Sharpe of 0.5 at 2× D26: IC × (required gross / observed gross). It assumes the gross edge scales linearly with
IC at fixed turnover, which is a rough approximation.

| candidate | rule | IC (h12 delayed) | observed gross bp/day | required gross bp/day | cost 2× D26 bp/day | ratio | **IC target** |
|---|---|---|---|---|---|---|---|
| M3 | R1 | 0.0294 | 8.46 | 32.24 | 26.24 | 3.81 | **0.112** |
| M3 | R2 | 0.0294 | 6.86 | 14.55 | 8.66 | 2.12 | **0.062** |
| Gate-3 B | R1 | 0.0167 | 21.87 | 26.04 | 20.87 | 1.19 | **0.020** |
| Gate-3 B | R2 | 0.0167 | 11.94 | 11.54 | 7.01 | 0.97 | **0.016** |
| M1 | R1 | 0.0203 | 14.26 | 40.77 | 35.82 | 2.86 | **0.058** |
| M1 | R2 | 0.0203 | 5.16 | 13.54 | 9.12 | 2.62 | **0.053** |

The bar depends on how a model's IC turns into P&L. B's information converts well (IC 0.016–0.020 would suffice),
but M3's and M1's do not (0.05–0.11). A Phase 4 model would therefore need both of these:

- the D10 + D22 IC gate at h = 12 delayed, with an IC of **≥ 0.02** if its P&L conversion is like B's;
- its own R1/R2 backtest passing D27.

## 7. Caveats

1. **The DSR condition was not passable as specified.** V mixes trials at different costs and horizons, so
   SR\* = 1.16 per day (about 18 annualized). Even with a like-for-like V, the best DSR is 0.20. Changing V now would
   be a post-hoc revision, so the pre-registered result stands. You decide whether this changes how you read the
   outcome.
2. **B is a model that failed Gate 3** (h = 12 delayed t 1.92 vs null threshold 2.34), yet it has the best P&L. M3,
   which passed D25, has the weakest P&L. So the IC gate and the economic test disagree on which model is useful.
3. **Costs were chosen after the D25 break-even numbers were seen.** They are an estimate (EDGE from 1-minute bars)
   without market impact, queue position or open-auction effects. At the old 3 bp default, only B R2 stays positive
   (0.37).
4. **There are 42 trials in the registry, and the same 96 folds have been looked at repeatedly.** Only the sealed
   holdout and forward test are untouched.

## 8. Decision-rule outcome

**No trial passes D27, so nothing is frozen.** Under the rule fixed in D27, the recommendation is to **conclude the
project with a final report**. Phase 4 is an option only as a long shot you choose explicitly, with the §6 IC target
(≥ 0.02 at h = 12 delayed and a passing D27 backtest) as its bar. The holdout is not proposed: D27 links it to a
passing trial. Nothing further has been started.

## 9. Files, registry, tests

- **Code.** `src/tslaq/backtest/positions.py` (R1 / R2), `src/tslaq/backtest/dsr.py`, `scripts/run_phase5_early.py`
  (`--log`), `scripts/run_phase5_cpcv.py` (`--time`, `--workers`).
- **Tests.** `tests/test_rules_d27.py` passes. It checks that fills happen at t+1(+d), the band, the session limits,
  hand-computed P&L and flattening, R2 timing / threshold / non-overlap, that fills and earlier P&L do not change
  when future prices change, and that DSR = PSR at N = 1 against a hand example.
- **Registry.** 6 trials `d27_{R1,R2}_{M3,Gate3B,M1}_h12` (phase 5). **n_trials_so_far = 42.**
- **Tables.** `reports/tables/phase5_early/`. CPCV checkpoints are in `data/processed/phase5_cpcv/` (gitignored).
