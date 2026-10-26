# Final holdout evaluation of the frozen Gate-3 B R1 (D28)

Date 2026-10-26. Holdout 2026-01-01..2026-10-09 (194 sessions), **evaluated once**. The forward test (after
2026-10-09) stays sealed, and Phase 4 was not started.

**Result: the primary criterion FAILS.**

- Total net P&L of R1 at D26 cost (0.749 bp per side) over the 194 holdout sessions is **−725 bp** (net Sharpe −0.49,
  t −0.43).
- Gross P&L is +75 bp (gross Sharpe 0.05), so the strategy broke even only at **0.07 bp per side**, against 2.64 bp in
  development.
- The forecasts kept a weakly positive D19 IC (h = 12 delayed: mean 0.0157, t 1.15, positive in 7 of 10 months), but
  it did not turn into P&L.
- Under the interpretation fixed in D28, **the recommendation is to stop.**

## 1. Decision D28 (recorded and committed before any holdout data was downloaded)

D28 was recorded in [`decisions.md`](decisions.md) and in CLAUDE.md §10.5 and §12, and committed as 3feeb97. The
frozen code was committed as 65a13dc, and the freeze note as f3a40be. All three commits came before the reproduction
check and before the download. In short:

- **Why.** Development is concluded: D27 found no passing trial. As the final out-of-sample step, the single best
  development candidate, Gate-3 B with rule R1, is frozen and evaluated once on the holdout (D7). **This deviates
  from the D27 decision rule**, which linked the holdout to a passing trial, **and does not make B R1 a pass.**
- **Frozen design.** This is exactly the development recipe, applied per holdout month M = 2026-01..2026-10
  (October only up to 2026-10-09). Each month uses only sessions before M.
  - data v2 cleaning;
  - the Phase 2 per-fold fits (GARCH-t, HMM, Kalman, frac-diff, ADF/KPSS raw / rolling-z versions, clusters,
    condprob cells);
  - the per-fold feature selection with the D22 thresholds and the existing 300-draw null quantiles (no new null);
  - LightGBM with the fixed Phase 1 parameters on that fold's list (Gate-3 B, h = 12, D23 rows, purge + 78-bar
    embargo);
  - rule R1 exactly as in D27: q = 90th percentile of |ẑ| over the previous month's out-of-sample predictions
    (2026-01 uses the stored 2025-12 predictions); w = clip(ẑ/q, ±1); band 0.25; fill at the next bar's open; no
    fills in the last 10 minutes; flat at the close;
  - D26 cost for 2026 = EDGE 2025 mean half-spread + 0.10 bp = **0.749 bp per side** (0.649 + 0.10).
- **Frozen code.** Git commit **65a13dcc0c7a1dc00b8b74e51f7a5cb8fd563752**, config hash **da5d24529c**
  (`scripts/run_holdout_d28.py::frozen_config`).
- **Reproduction check.** Before any holdout data was loaded, the frozen pipeline was run for development fold
  2025-12. The rule fixed in advance: if the correlation with the stored predictions is below 0.99, stop.
- **Primary criterion ("no-failure" check), fixed in advance.** Total net P&L of R1 at D26 cost over the whole
  holdout > 0.
- **Reported, but not criteria.**
  - net Sharpe and total P&L at D26, 2× D26, 3 bp and gross; the same with one extra bar of delay;
  - per-month P&L; turnover and trades per day; hit rate;
  - D19 IC at h = 12 with and without delay (per month and mean, with t over the holdout months).
- **Interpretation, fixed in advance.**
  - The holdout has low power. If B R1's true annualized net Sharpe is about 1.1, the chance of a positive holdout
    P&L is about 83%. With no edge it is 50%.
  - A pass is consistent with the development result but is not confirmation.
  - If the criterion passes, the recommended next step is paper trading (Phase 6) on the forward period, with its
    own pre-registered criteria, for at least 3 months before any real money. If it fails, the recommendation is
    to stop.
- **One evaluation only.** No parameter, rule, cost, feature or code change after any holdout number is seen. One
  trial in `experiments/registry.csv`.

## 2. Data

Table: [`tables/holdout_d28/holdout_data_checks.json`](tables/holdout_d28/holdout_data_checks.json).

**Source.**
- TSLA, SPY and QQQ Alpaca SIP 1-minute bars, split-adjusted (as in the v2 source files from 2022-09), plus TSLA
  unadjusted bars for the split check.
- Requests were capped at 2026-10-10 00:00 New York; the script asserts that no later bar was received.
- Source check on the development month 2025-12: Alpaca's split-adjusted bars equal the local v2 source files
  exactly (all rows, OHLCV difference 0 for all three symbols;
  [`tables/holdout_d28/source_check_2025-12.csv`](tables/holdout_d28/source_check_2025-12.csv)).

**Events.**
- **Earnings.** Alpha Vantage `EARNINGS` (as in Phase 1) lists the 2026 reports on 2026-01-28, 04-22 and 07-22, all
  post-market, which matches the file used since Phase 1. The reaction sessions are 2026-01-29, 04-23 and 07-23.
  The 2026-10-21 report falls after the holdout.
- **Splits and dividends.** `SPLITS` shows no TSLA split after 2022-08-25, and TSLA raw = split-adjusted in every
  holdout month (difference 0). SPY/QQQ 2026 dividends come from the Alpha Vantage `DIVIDENDS` files of data v2.

**Cleaning** with the v2 code (`data/interim/v2h/`):
- 194 sessions, 15,132 bars per symbol, 2026-01-02..2026-10-09; no missing bars, no dropped sessions, no half-days.
- Bad ticks flagged (not deleted): TSLA 3, SPY 2 (+1 range), QQQ 3.

**Development rows unchanged.**
- TSLA development bars in v2h are **bit-identical** to v2 (195,180 rows).
- SPY/QQQ have identical timestamps and log returns (max difference 1.8e-15). Only the back-adjustment factor of
  the 2026 dividends scales their earlier price levels.
- The evaluation also re-scored all 96 development months of Gate-3 B on the v2h frame: max |IC difference| vs the
  stored Gate 3 fold ICs 1e-16, with and without delay.

## 3. Reproduction check (fold 2025-12, development data only)

Table: [`tables/holdout_d28/reproduction_check.csv`](tables/holdout_d28/reproduction_check.csv). Run from scratch:
Phase 2 fold fit (115 s), D19 re-score (177 s), D22 selection, and LightGBM B (9 s).

| check | result |
|---|---|
| predictions (h = 12) | 1,416 new vs 1,416 stored, all common; **correlation 1.000000, max \|difference\| 0.0** |
| feature list | identical to `configs/features.yaml` v2, fold 2025-12 (27 features, same versions and order) |
| R1 2025-12, q = 0.1497 (stored 2025-11), D26 0.729 bp | net P&L **−215.16 bp** (new) = −215.16 bp (stored predictions) = −215.16 bp (D27 daily file); 356 fills each; max daily difference vs D27 1e-12 bp |
| rule (corr ≥ 0.99) | **pass** → holdout evaluated |

## 4. Primary criterion (fixed in D28)

| criterion | value | result |
|---|---|---|
| total net P&L of R1 at D26 cost over the holdout > 0 | **−725.0 bp** (194 sessions, 2026-01-02..2026-10-09) | **FAIL** |

## 5. Every reported number

Table: [`tables/holdout_d28/holdout_results.csv`](tables/holdout_d28/holdout_results.csv).

- P&L is the sum of daily log returns in bp. Sharpe is annualized from daily P&L over all 194 sessions (√252); t =
  daily mean / sd × √194.
- "Trades" = fills. The hit rate and bp/segment are per holding segment between fills, per unit position (as in D27).
- Break-even = mean daily gross / mean daily turnover.

| extra delay | cost | bp/side | **total P&L bp** | mean bp/day | net Sharpe | t | Sortino | max DD bp | share of days > 0 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | **D26** | 0.749 | **−725.0** | −3.74 | **−0.49** | −0.43 | −0.75 | 2,158 | 42.3% |
| 0 | 2× D26 | 1.499 | −1,525.5 | −7.86 | −1.02 | −0.90 | −1.53 | 2,603 | 40.7% |
| 0 | 3 bp | 3.000 | −3,128.8 | −16.13 | −2.09 | −1.84 | −2.94 | 3,941 | 38.1% |
| 0 | gross | 0 | +75.5 | +0.39 | +0.05 | +0.05 | +0.08 | 1,713 | 42.8% |
| 1 | D26 | 0.749 | −674.7 | −3.48 | −0.50 | −0.44 | −0.77 | 1,614 | 44.3% |
| 1 | 2× D26 | 1.499 | −1,475.2 | −7.60 | −1.09 | −0.96 | −1.61 | 2,067 | 41.8% |
| 1 | 3 bp | 3.000 | −3,078.5 | −15.87 | −2.26 | −1.99 | −3.14 | 3,462 | 38.7% |
| 1 | gross | 0 | +125.8 | +0.65 | +0.09 | +0.08 | +0.15 | 1,169 | 47.9% |

| extra delay | fills (trades) per day | turnover per day (Σ\|Δw\|) | holding segments | hit rate | mean bp per segment | break-even bp/side |
|---|---|---|---|---|---|---|
| 0 | 12.46 | 5.51 | 2,418 | 0.485 | +0.20 | **0.07** |
| 1 | 12.46 | 5.51 | 2,418 | 0.486 | −0.14 | 0.12 |

The development values of the same trial (D27, 2018–2025, no extra delay) for comparison: 18.0 fills and 8.28
turnover per day, hit rate 0.512, +1.96 bp per segment, break-even 2.64 bp per side, net Sharpe 1.11 at D26
([`phase5_early.md`](phase5_early.md) §2).

## 6. Per month

Tables: [`tables/holdout_d28/holdout_by_month.csv`](tables/holdout_d28/holdout_by_month.csv) (also fills,
turnover and q) and [`tables/holdout_d28/holdout_q_by_month.csv`](tables/holdout_d28/holdout_q_by_month.csv). All
values are P&L in bp.

| month | sessions | q | fills | net D26 | net 2× D26 | net 3 bp | gross | +1 delay net D26 | +1 delay gross |
|---|---|---|---|---|---|---|---|---|---|
| 2026-01 | 20 | 0.170 | 208 | +700.6 | +636.7 | +508.7 | +764.6 | +520.3 | +584.2 |
| 2026-02 | 19 | 0.148 | 290 | +189.6 | +94.1 | −97.2 | +285.0 | −153.4 | −57.9 |
| 2026-03 | 22 | 0.131 | 365 | −89.3 | −218.5 | −477.4 | +40.0 | −98.6 | +30.7 |
| 2026-04 | 21 | 0.154 | 273 | −241.1 | −332.8 | −516.6 | −149.3 | −121.4 | −29.7 |
| 2026-05 | 20 | 0.165 | 215 | −24.6 | −92.6 | −228.8 | +43.4 | −132.1 | −64.1 |
| 2026-06 | 21 | 0.151 | 225 | −75.4 | −148.0 | −293.4 | −2.8 | +78.6 | +151.2 |
| 2026-07 | 22 | 0.145 | 284 | −63.2 | −158.4 | −349.1 | +32.0 | +66.2 | +161.4 |
| 2026-08 | 21 | 0.156 | 227 | +245.0 | +175.5 | +36.3 | +314.5 | +99.8 | +169.3 |
| 2026-09 | 21 | 0.118 | 276 | **−1,363.8** | −1,462.6 | −1,660.6 | −1,264.9 | −911.7 | −812.8 |
| 2026-10 (to 10-09) | 7 | 0.165 | 55 | −2.9 | −18.8 | −50.5 | +12.9 | −22.3 | −6.4 |
| **total** | **194** | | **2,418** | **−725.0** | −1,525.5 | −3,128.8 | +75.5 | −674.7 | +125.8 |

Net P&L at D26 is positive in 3 of 10 months. September 2026 alone loses 1,364 bp, and January gains 701 bp. For
information only, not a criterion: without September the total would be +639 bp. The criterion was fixed over the
whole holdout, so this does not change the result, and dropping the worst month after seeing it would be exactly
the kind of post-hoc choice D28 rules out.

## 7. D19 IC at h = 12

Table: [`tables/holdout_d28/holdout_ic_by_month.csv`](tables/holdout_d28/holdout_ic_by_month.csv).

- s = ẑ − training mean of clipped z (sessions before the month); y = clip(z, ±5).
- V_y = training mean of y².
- V_s = mean s² over all earlier out-of-sample months, i.e. 2018-01..2025-12 and the earlier holdout months (the
  first development fold uses the D6 block).

| month | IC h = 12 | IC h = 12 delayed | predictions |
|---|---|---|---|
| 2026-01 | +0.0567 | +0.0487 | 1,320 |
| 2026-02 | +0.0417 | +0.0407 | 1,254 |
| 2026-03 | −0.0045 | −0.0037 | 1,452 |
| 2026-04 | +0.0195 | +0.0265 | 1,386 |
| 2026-05 | −0.0292 | −0.0271 | 1,320 |
| 2026-06 | +0.0638 | +0.0707 | 1,386 |
| 2026-07 | +0.0554 | +0.0591 | 1,452 |
| 2026-08 | +0.0130 | +0.0113 | 1,386 |
| 2026-09 | −0.0815 | −0.0721 | 1,386 |
| 2026-10 | +0.0030 | +0.0028 | 462 |
| **mean (sd), t over 10 months** | **+0.0138 (0.045), t 0.97** | **+0.0157 (0.043), t 1.15** | 12,804 |
| share of months with IC > 0 | 70% | 70% | |

Development reference for Gate-3 B at h = 12 delayed: IC 0.0167, t 1.92 over 96 folds; 2023–25 IC 0.0071
([`phase3_tokenizer.md`](phase3_tokenizer.md) §4.1). The holdout IC is about the same size as in development. Over 10
months it is not significant, and it does not show up as P&L.

**Reading (descriptive; no change follows from it).**
- The months line up only loosely. The monthly IC and the monthly gross P&L have the same sign in 6 of 10 months
  (correlation 0.76, driven by January and September), and September is the worst month on both.
- What changed from development is the size of the gross edge: +0.2 bp per holding segment vs +1.96 bp, and gross
  +0.39 bp per day vs +21.9 bp per day. As in D25/D27, the D19 IC weights rows by |s|·|z| in σ units, while R1's P&L
  is in return space with capped positions. A positive IC therefore does not guarantee a gross edge large enough to
  cover even a 0.75 bp half-spread.

## 8. Interpretation (as fixed in D28)

- **The primary criterion fails** (−725 bp at D26).
- **Power.** Under the D28 power statement, a failure has about a 17% chance if B R1's true net Sharpe were about
  1.1, and 50% if it has no edge. So the failure is evidence against the development result, but weak evidence on
  its own. The observed net Sharpe is −0.49 (t −0.43).
- **Delay check.** The extra-delay variant gives the same picture (−675 bp at D26), and gross P&L is ≈ 0 in both
  variants.
- **Recommendation, as fixed in D28: stop.** Do not start paper trading or Phase 4 on this design, and do not
  revise the design on these numbers. The forward test (after 2026-10-09) stays sealed. Unsealing it would only make
  sense for a new, separately pre-registered design, which is not proposed here.

## 9. Registry, timing and files

- **Registry.** One trial, `d28_holdout_R1_Gate3B_h12` (phase 5, config hash da5d24529c, data version v2h):
  net_sharpe_period −0.031, net_sharpe_ann −0.49, sharpe_2x_costs −1.02, rank_ic_h12 = mean D19 IC 0.0138.
  **n_trials_so_far = 43.**
- **Run times.**
  - Reproduction check: 6.3 min wall.
  - Holdout pipeline with 5 workers: Phase 2 fits 12.7 min, re-score 7.8 min, LightGBM 1.1 min.
  - Evaluation: about 2 min.
  - Download: 40 month files, about 1 min.
- **Code.** `scripts/run_holdout_d28.py` (`--freeze`, `--source-check`, `--repro`, `--download`, `--build`,
  `--holdout`, `--evaluate --log`). The evaluation refuses to run twice (`holdout_summary.json` is the marker).
- **Tables** in [`tables/holdout_d28/`](tables/holdout_d28/): `reproduction_check.csv`, `source_check_2025-12.csv`,
  `holdout_data_checks.json`, `holdout_results.csv`, `holdout_by_month.csv`, `holdout_ic_by_month.csv`,
  `holdout_q_by_month.csv`, `holdout_daily.csv`, `holdout_summary.json`.
- **Holdout per-month feature lists** (27–31 features; gitignored with the checkpoints):
  `data/processed/holdout_d28/holdout/feature_lists.yaml`.
