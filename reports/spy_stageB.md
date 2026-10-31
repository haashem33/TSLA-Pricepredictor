# SPY intraday momentum: Stage B (P2 and P3)

Date 2026-10-31. Specification: D29, D29a and D29b in [`decisions.md`](decisions.md). Stage A:
[`spy_stageA.md`](spy_stageA.md).

This is the single Stage B evaluation (marker `tables/spy_stageB/stageB_done.json`). It used the frozen strategy code
(commit 8163aa6, config hash 3d8abf0044). No order was placed, no order-routing code exists, and no bar at or after
2026-10-09 16:00 ET was loaded.

## Summary

| period | criterion (primary, D29) | value | result |
|---|---|---|---|
| **P2** 2024-03..2025-12 (461 sessions) | net Sharpe > 0.5 at C_real, E0 | **0.59** | pass |
| | net Sharpe > 0 at C_real, E1 (one extra bar of delay) | **0.20** | pass |
| | net Sharpe > 0 at 2× C_real, E0 | **0.37** | pass |
| | **P2 overall** | | **PASS** |
| **P3** 2026-01-02..2026-10-09 (194 sessions) | net P&L > 0 at C_real, E0 | **−13.6%** (Sharpe −1.38) | **FAIL** |

**Information note (D29b (4)).** The P2 pass is carried by 2024, which the authors had already published (+32.2%; we
get +29.0% at real cost and +32.5% at paper cost). The two least-known parts are both negative:
- **2025, full year:** −4.8% at real cost (Sharpe −0.25), against SPY buy-and-hold +17.7%.
- **P3:** −13.6%, against SPY +15.1%.
- **P2 without the paper's own data months** (2024-05..2025-12): Sharpe 0.46 at real cost, below the 0.5 bar.

**Interpretation.**
- D29a-3 fixed what to do if P2 and P3 both pass (paper trading) and if P2 fails (stop). **It does not cover "P2
  passes, P3 fails"**, so no pre-registered next step applies.
- **My recommendation: stop. Do not paper trade.** Every number since the paper's data ended has been weak or
  negative. A pass would not have been proof (D29a-3), and a split result is weaker still.
- The decision is yours.

## 1. Data, costs and checks

**Frozen code.** `git diff 8163aa6 HEAD -- src/tslaq/spy/momentum.py src/tslaq/spy/metrics.py` is empty (0 lines) at
every Stage B commit. Only the loaders (`src/tslaq/spy/data.py`) and the runners changed:
- `61d7847`: full-range build, holdout reader, v2 / v2h check, Stage B runner;
- `58cfb3d`: spread table, and round-trip float parsing in the runner.

The Stage A 1-minute frame rebuilt with the extended loaders is identical to the stored one.

**Data check** ([`tables/spy_stageB/data_check.csv`](tables/spy_stageB/data_check.csv)): all pass.
- **Frame:** 2,707 sessions, 2016-01-04..2026-10-09. The last bar ends at 2026-10-09 16:00 ET.
- **v2, 2016–2025:** 2,513 sessions and 195,258 five-minute bars, all matched; max |Δ log return| 4.5e-16; 40
  ex-dates, factor difference ≤ 2.2e-16.
- **v2h, 2026:** 194 sessions and 15,132 bars, all matched; max |Δ log return| 0.0; 3 ex-dates, factor difference
  ≤ 1.1e-16.
- **Dropped:** 2019-08-12 only (missing minutes).
- **Flags:** 21 half-days; 74 forward-filled minutes; 404 bad-tick and 76 bad-range flags (flagged, not deleted).

**P1 reproduction (D29b (2)).** Every one of the 24 configuration × cost × execution daily series over P1 reproduces
the Stage A files exactly (max |Δ| 0.0) ([`tables/spy_stageB/p1_reproduction.csv`](tables/spy_stageB/p1_reproduction.csv)).

**Spread table (D29b (3))** ([`tables/spy_stageB/spread_by_year.csv`](tables/spy_stageB/spread_by_year.csv)).
- It was computed and committed (58cfb3d) before any P2 / P3 strategy number.
- The 2016–2023 rows are byte-identical to Stage A.
- New rows:

  | year | EDGE half-spread (bp) | half tick at median price (bp) | binds at median price |
  |---|---|---|---|
  | 2024 | 0.115 | 0.092 | EDGE |
  | 2025 | 0.156 | 0.081 | EDGE |

- Cost years: 2025 uses 0.115 bp and 2026 uses 0.156 bp, with the half-tick floor applied per order.
- Realized total cost per side for the primary configuration (spread, commission and SEC fee): 0.28 bp in P2 and
  0.30 bp in P3.

## 2. Primary configuration (Curr.Band + VWAP, dynamic sizing)

[`tables/spy_stageB/summary_by_window.csv`](tables/spy_stageB/summary_by_window.csv) holds every metric for all 3
configurations × 4 costs × 2 executions × windows (P1, P2, P3, 2024-05..2025-12, 2025, full).

| window | cost | exec | total return | CAGR | vol | Sharpe | t | Sortino | MDD | hit (traded days) | alpha (NW t) | beta | break-even bp/side |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **P2** | gross | E0 | 22.3% | 11.6% | 15.0% | 0.81 | 1.09 | 1.54 | 13.6% | 44.9% | 14.0% (1.26) | −0.10 | 0.87 |
| P2 | paper | E0 | 20.3% | 10.6% | 15.0% | 0.75 | 1.01 | 1.42 | 14.1% | 44.9% | 13.1% (1.17) | −0.10 | |
| P2 | **real** | **E0** | **15.1%** | 8.0% | 15.0% | **0.59** | 0.79 | 1.10 | 15.4% | 44.1% | 10.7% (0.96) | −0.10 | |
| P2 | 2× real | E0 | 8.4% | 4.5% | 15.0% | 0.37 | 0.50 | 0.67 | 17.2% | 42.6% | 7.4% (0.66) | −0.10 | |
| P2 | gross | E1 | 9.8% | 5.3% | 14.8% | 0.42 | 0.57 | 0.73 | 17.4% | 47.8% | 8.2% (0.77) | −0.11 | 0.40 |
| P2 | real | E1 | 3.4% | 1.8% | 14.8% | 0.20 | 0.27 | 0.33 | 19.2% | 46.7% | 4.9% (0.46) | −0.11 | |
| P2 | 2× real | E1 | −2.7% | −1.5% | 14.9% | −0.03 | −0.04 | −0.04 | 20.8% | 46.0% | 1.6% (0.15) | −0.10 | |
| **P3** | gross | E0 | −10.8% | −13.8% | 13.1% | −1.07 | −0.94 | −1.65 | 11.7% | 39.1% | −14.3% (−0.91) | 0.02 | −1.08 |
| P3 | paper | E0 | −11.4% | −14.6% | 13.2% | −1.13 | −0.99 | −1.74 | 12.3% | 39.1% | | | |
| P3 | **real** | **E0** | **−13.6%** | −17.3% | 13.2% | **−1.38** | −1.21 | −2.08 | 14.4% | 38.3% | −18.5% (−1.17) | 0.02 | |
| P3 | 2× real | E0 | −16.4% | −20.7% | 13.3% | −1.69 | −1.48 | −2.49 | 17.0% | 35.2% | | | |
| P3 | real | E1 | −9.0% | −11.5% | 13.3% | −0.85 | −0.75 | −1.34 | 9.8% | 37.5% | | | |
| 2024-05..2025-12 | real | E0 | 10.3% | 6.0% | 15.1% | 0.46 | 0.60 | 0.87 | 15.4% | 44.0% | 9.2% (0.77) | −0.11 | 0.76 (gross) |
| 2024-05..2025-12 | real | E1 | −0.2% | −0.1% | 14.9% | 0.06 | 0.08 | 0.11 | 19.2% | 46.1% | | | |
| **2025** | gross | E0 | −1.9% | −1.9% | 15.4% | −0.05 | −0.05 | −0.09 | 13.6% | 44.4% | 0.0% (0.00) | −0.04 | −0.17 |
| 2025 | real | E0 | −4.8% | −4.8% | 15.5% | −0.25 | −0.24 | −0.44 | 15.4% | 43.7% | −3.0% (−0.20) | −0.04 | |
| 2025 | real | E1 | −10.2% | −10.3% | 15.0% | −0.65 | −0.65 | −1.06 | 19.2% | 45.1% | | | |

**SPY buy-and-hold:**

| window | total return | Sharpe | MDD |
|---|---|---|---|
| P2 | 37.5% | 1.16 | 18.7% |
| P3 | 15.1% | 1.47 | 8.9% |
| 2024-05..2025-12 | 38.7% | 1.27 | 18.7% |
| 2025 | 17.7% | 0.98 | 18.7% |

**Other primary-configuration numbers at real cost with E0** (P1 | P2 | P3):
- **Orders:** 3,598 | 807 | 374. Round trips per day: 0.88 | 0.88 | 0.96.
- **Trade hit ratio:** 37.7% | 40.1% | 33.7%.
- **Average P&L per share:** gross $0.132 | $0.072 | −$0.138; net $0.107 | $0.040 | −$0.182.
- **Turnover:** 4.8 | 4.7 | 5.4 × equity per day.
- **Skew:** 2.0 | 3.2 | 1.5. Worst / best day in P2: −3.1% / +9.5%.
- **Legs (sum of daily returns):**

  | leg | P1 | P2 | P3 |
  |---|---|---|---|
  | long | +77.7% | +4.1% | −2.9% |
  | short | +42.6% | +12.0% | −11.1% |

  In P2 the short leg carries the result. Long trades have earned nothing since 2024-05 (+0.1%).

**Delay sensitivity.** One extra bar of delay costs much more after 2024 than before. Gross break-even falls from 2.1
bp per side (P1) to 0.87 bp (P2, E0) and 0.40 bp (P2, E1). The margin between the edge and the cost is now small.

### Yearly (primary, %)

[`tables/spy_stageB/yearly.csv`](tables/spy_stageB/yearly.csv) holds every run.

| year | gross E0 | paper E0 | real E0 | 2× real E0 | real E1 | SPY B&H | authors (published) |
|---|---|---|---|---|---|---|---|
| 2016 | −10.8 | −13.4 | −16.8 | −22.4 | −20.2 | 21.7 | −12.8 |
| 2017 | −6.3 | −9.3 | −13.5 | −20.1 | −15.0 | 21.7 | −6.9 |
| 2018 | 58.7 | 55.4 | 50.5 | 42.7 | 51.0 | −4.5 | 61.1 |
| 2019 | 7.8 | 5.8 | 2.8 | −2.0 | −0.6 | 31.1 | 6.9 |
| 2020 | 26.3 | 25.1 | 23.3 | 20.3 | 25.8 | 18.3 | 26.8 |
| 2021 | 31.4 | 29.7 | 25.6 | 20.1 | 24.6 | 28.7 | 34.8 |
| 2022 | 25.9 | 24.9 | 23.0 | 20.2 | 23.7 | −18.2 | 24.4 |
| 2023 | 41.3 | 39.6 | 35.2 | 29.4 | 27.2 | 26.2 | 37.2 |
| **2024** | 33.9 | **32.5** | 29.0 | 24.3 | 22.2 | 24.9 | **32.2** |
| **2025** | −1.9 | −2.7 | **−4.8** | −7.6 | −10.2 | 17.7 | (2025-01: −1.2; ours −1.4) |
| **2026 (to 10-09)** | −10.8 | −11.4 | **−13.6** | −16.4 | −9.0 | 15.1 | |

(2016 starts 2016-01-26 for us; the authors' 2016 is the full year.)

### Monthly, 2024-01..2026-10 (primary, %)

[`tables/spy_stageB/monthly.csv`](tables/spy_stageB/monthly.csv) holds every run, all months from 2016.

| month | paper E0 | real E0 | real E1 | 2× real E0 | SPY | authors |
|---|---|---|---|---|---|---|
| 2024-01 | 8.6 | 8.4 | 8.8 | 8.1 | 1.6 | 8.8 |
| 2024-02 | −1.3 | −1.6 | −2.4 | −2.0 | 5.2 | −1.5 |
| 2024-03 | −0.7 | −0.9 | −1.2 | −1.3 | 3.3 | −0.4 |
| 2024-04 | 5.8 | 5.4 | 4.9 | 4.9 | −4.0 | 5.8 |
| 2024-05 | −4.4 | −4.6 | −4.9 | −4.8 | 5.0 | −4.3 |
| 2024-06 | 1.5 | 1.3 | 1.1 | 1.1 | 3.5 | 1.6 |
| 2024-07 | 8.6 | 8.3 | 8.1 | 7.9 | 1.2 | 8.2 |
| 2024-08 | −2.9 | −3.0 | −3.3 | −3.2 | 2.4 | −2.8 |
| 2024-09 | 4.3 | 4.2 | 2.0 | 4.0 | 2.2 | 4.1 |
| 2024-10 | 6.5 | 6.3 | 5.0 | 5.9 | −1.0 | 6.7 |
| 2024-11 | −2.9 | −3.2 | −2.8 | −3.5 | 6.0 | −2.6 |
| 2024-12 | 6.7 | 6.4 | 6.1 | 6.0 | −2.4 | 5.7 |
| 2025-01 | −1.4 | −1.5 | −1.5 | −1.7 | 2.7 | −1.2 |
| 2025-02 | 7.8 | 7.6 | 5.7 | 7.3 | −1.3 | |
| 2025-03 | 0.0 | −0.1 | 0.6 | −0.4 | −5.6 | |
| 2025-04 | 4.3 | 4.2 | 3.8 | 4.1 | −0.9 | |
| 2025-05 | −2.2 | −2.3 | −2.0 | −2.4 | 6.3 | |
| 2025-06 | −1.9 | −2.1 | −3.6 | −2.4 | 5.1 | |
| 2025-07 | −1.5 | −1.7 | −2.9 | −2.0 | 2.3 | |
| 2025-08 | −2.9 | −3.1 | −4.3 | −3.4 | 2.1 | |
| 2025-09 | −5.4 | −5.7 | −5.5 | −6.0 | 3.6 | |
| 2025-10 | 6.2 | 5.9 | 3.6 | 5.6 | 2.4 | |
| 2025-11 | −3.0 | −3.2 | −2.7 | −3.4 | 0.2 | |
| 2025-12 | −1.9 | −2.1 | −1.2 | −2.3 | 0.1 | |
| 2026-01 | −3.5 | −3.8 | −4.9 | −4.2 | 1.5 | |
| 2026-02 | −0.4 | −0.6 | 0.3 | −0.9 | −0.8 | |
| 2026-03 | 0.7 | 0.5 | 3.3 | 0.1 | −5.0 | |
| 2026-04 | −0.9 | −1.0 | −1.0 | −1.2 | 10.5 | |
| 2026-05 | −3.3 | −3.7 | −3.3 | −4.2 | 5.3 | |
| 2026-06 | 4.0 | 3.7 | 3.8 | 3.4 | −1.1 | |
| 2026-07 | −3.0 | −3.2 | −3.0 | −3.6 | 0.1 | |
| 2026-08 | −1.3 | −1.5 | −1.3 | −1.7 | 2.7 | |
| 2026-09 | −3.3 | −3.7 | −2.2 | −4.2 | −0.3 | |
| 2026-10 (to 10-09) | −0.7 | −0.8 | −0.7 | −0.9 | 2.1 | |

Our monthly returns track the authors' published months through 2025-01 closely, including 11 post-publication months
(2024-03..2025-01). The implementation is faithful. The decline after January 2025 is in the strategy, not in the
replication: 13 of the 21 months from 2025-02 to 2026-10 are negative at real cost.

## 3. Secondary configurations (information)

Real cost, E0; all 32 cost × execution rows per window are in `summary_by_window.csv`.

| configuration | P2 Sharpe | P2 total | P3 Sharpe | P3 total | 2025 total | P2, E1 Sharpe |
|---|---|---|---|---|---|---|
| Opp.Band, 100% | 0.70 | 12.1% | −1.13 | −5.6% | +15.8% | 0.54 |
| Curr.Band + VWAP, 100% | 0.35 | 3.7% | −1.21 | −4.5% | +0.3% | 0.04 |

All three configurations lose money in P3 at every cost level and with both executions. The opposite-band variant had
a good 2025 (Sharpe 1.36) that did not carry into 2026.

**Full period 2016-01-26..2026-10-09, at real cost with E0:**

| | total return | Sharpe |
|---|---|---|
| primary | 204.5% | 0.79 |
| Opp.Band, 100% | 51.2% | 0.48 |
| Curr.Band + VWAP, 100% | 54.7% | 0.65 |
| SPY buy-and-hold | 393.3% | 0.94 |

## 4. DSR (information only)

Table: [`tables/spy_stageB/dsr_p2_info.csv`](tables/spy_stageB/dsr_p2_info.csv). N = 46.

**With V = the variance of the three configurations' P2 daily Sharpe ratios at real cost, E0** (1.4e-4; SR\* = 0.025
per day):

| configuration | DSR | PSR(0) |
|---|---|---|
| primary | 0.61 | 0.80 |
| Opp.Band, 100% | 0.68 | |
| Curr.Band + VWAP, 100% | 0.48 | |

**With the D27 V (0.2773):** DSR 0.000 for all three.

None reaches 0.95. PSR(0) of the primary configuration over P2 alone is only 0.80, a one-sided probability of about
80% that its true P2 Sharpe is above zero.

## 5. Interpretation (D29a-3) and recommendation

D29a-3, fixed before Stage A:
- **Power.** P2 has limited power: with a true Sharpe of about 1.3 the chance of "Sharpe > 0.5" is about 86%; with no
  edge it is about 25%.
- **Next steps.** If P2 and P3 both pass, paper trade on live data for at least 3 months (on an EU-tradable instrument
  such as MES), with its own pre-registered criteria and cost model. If P2 fails, stop.
- **A pass is not proof.**

**What happened.**
- **P2 passed narrowly:** 0.59 against 0.5. A no-edge strategy clears that bar about a quarter of the time.
- **P3 failed clearly:** −13.6%, Sharpe −1.38.
- **This case is not pre-registered.** D29a-3 does not say what to do when P2 passes and P3 fails.

**Recommendation (mine, not pre-registered): stop. Do not start paper trading on this strategy.**

Reasons:
1. **The P2 pass rests on published data.** It is carried by 2024, which the authors had already published (our
   replication matches it to 0.3 pp). On the parts nobody had seen (2025 and P3), the strategy lost money at every
   cost level: −4.8% and −13.6% at real cost, while SPY rose 17.7% and 15.1%.
2. **P2 without the paper's own months fails the bar.** Over 2024-05..2025-12 the Sharpe is 0.46 at real cost, and
   0.06 with one extra bar of delay.
3. **The edge has shrunk to about the size of the cost.** Gross break-even fell from 2.1 bp per side (P1) to 0.87 bp
   (P2), and with one bar of delay to 0.40 bp. P3 is negative gross.
4. **All three configurations lose in P3, under every cost and both delays.**
5. **The DSR is far below 0.95.**

**Limits.** P3 is only 194 sessions; a Sharpe of −1.38 over 0.77 years has a standard error of about 1.1. The result
is consistent with a weakened edge, not proof that none exists. The trend-day regime the strategy needs (high
volatility, persistent intraday moves) was rare in 2025–2026. Continuing anyway would be a new, separately
pre-registered decision, for example paper trading MES with criteria fixed in advance.

## 6. Process notes and disclosures

- **Run once.** The Stage B run executed once (`stageB_done.json`, status "done"). Nothing changed after any P2 / P3
  number was seen.
- **Dry run before it.** A dry run of the same runner on Stage A data only (P1 sub-windows standing in for P2 / P3,
  output to the scratchpad, no registry, no marker) checked the code paths first. It computed no P2 / P3 number.
- **Timestamp read (disclosed in D29b (7)).** Before D29b, the min / max of the `timestamp` column of the 2026-10
  holdout file were read. Its max, 2026-10-09 23:59 UTC, shows post-market bars after 16:00 ET. No price was read.
  Every later read filters at read time.
- **Holdout bars read by a test.** The loader-cap test (`test_stage_b_holdout_loader_is_capped`) reads the 2026-10
  P3 bars before 16:00 ET to check the cap. It computes no strategy number.
- **Spread check flag.** The first `--spread` run printed "identical to Stage A: False". The cause was the CSV float
  parser, which differs by up to 8e-17. The files are byte-identical, and the runner now parses the table with
  round-trip precision.
- **Registry.** Three Stage B rows, `d29_spy_momentum_*`, phase `spy_B`, `rerun_of` = the Stage A rows. n_trials_so_far
  stays at **46**. Stage B config hash (cost years extended): f532e2ee90; strategy config hash 3d8abf0044.
- **Forward test.** It stays sealed. Nothing after 2026-10-09 16:00 ET was loaded.
