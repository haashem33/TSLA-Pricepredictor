# SPY intraday momentum: Stage A (replication period P1)

Date 2026-10-31. Specification: D29 and D29a in [`decisions.md`](decisions.md); plan in
[`spy_momentum_plan.md`](spy_momentum_plan.md).

Stage A covers **P1 = 2016-01..2024-02 only**. No P2 or P3 number has been computed. No order-routing code exists.

## Summary

- **Data check: pass.** The 1-minute SPY frame, aggregated to 5 minutes, reproduces the v2 log returns exactly (max
  difference 4.4e-16) on the same 2,052 sessions. The source cache is dividend-unadjusted.
- **Replication: within the D29 tolerance on all four checks.** Primary configuration, paper costs, fill at the next
  open (E0), months 2016-02..2024-02, compared with the paper's monthly table:

  | check | ours | paper | tolerance | result |
  |---|---|---|---|---|
  | correlation of monthly returns | 0.993 | | ≥ 0.80 | pass |
  | Sharpe from monthly returns | 1.26 | 1.37 | ±0.30 | pass (−0.11) |
  | CAGR | 18.3% | 20.1% | ±4 pp | pass (−1.8 pp) |
  | same yearly sign, 2016–2023 | 8 of 8 | | ≥ 7 | pass |

- **Costs.**
  - Realistic cost averages **0.39 bp per side**; the paper's own assumption is 0.13 bp.
  - The strategy's gross break-even is **2.1 bp per side**, so costs take a modest share of the edge in P1.
  - Primary net Sharpe over P1:

    | | Sharpe |
    |---|---|
    | gross | 1.37 |
    | paper costs | 1.24 |
    | realistic costs | 1.03 |
    | 2× realistic costs | 0.69 |
    | one extra bar of delay, realistic costs | 0.92 |

  - SPY buy-and-hold: Sharpe 0.86 over the same days.
- **No code fix** was needed for any rule deviation. The code is frozen at commit **8163aa6**, config hash
  **3d8abf0044**. Three registry trials were logged (n_trials_so_far 43 → **46**).
- **Read with care.** P1 lies inside the paper's own sample. This shows that our implementation matches theirs. It is
  not evidence of an edge. The test is P2.

## 1. Data (step 1)

Table: [`tables/spy_stageA/data_check.csv`](tables/spy_stageA/data_check.csv). Dividend detail:
[`tables/spy_stageA/data_check_dividends.csv`](tables/spy_stageA/data_check_dividends.csv).

**Source and cleaning.**
- SPY Alpaca SIP 1-minute bars from `data/raw/SPY/2016-01..2024-02`, cleaned with the D29 rules
  (`src/tslaq/spy/data.py`). Coverage: 2016-01-04..2024-02-29, 2,052 sessions (15 half-days), 797,580 minutes.
- **Missing minutes:** 104 before session drops; 74 are kept, forward-filled at the previous close with zero volume.
- **Dropped session:** 2019-08-12 (7.7% of minutes missing; it is also absent from v2).
- **Bad ticks:** 278 flagged, plus 52 range flags, on the 1-minute bars. They are flagged, not deleted, and the
  strategy does not use the flags.

**Check against v2.** Aggregated to 5-minute closes, the frame matches `data/interim/v2/SPY_5min.parquet`:
- 2,052 vs 2,052 sessions;
- 159,516 vs 159,516 bars, none unmatched;
- max |difference in log return| **4.4e-16** (limit 1e-10).

**Unadjusted cache.** The ratio of the v2 adjusted close to our raw close is constant between ex-dates (relative sd
4.9e-16). At each of the 32 ex-dates it jumps by exactly 1 − D / C_prev (max difference 2.2e-16). The smallest jump is
0.31%. So the source is dividend-unadjusted, and the back-adjustment is applied correctly.

**Stage A data boundary.**
- The 1-minute loader refuses any month after 2024-02. The spread table uses 2016–2023 only. Both are covered by a
  test.
- **Disclosure.** The first three runs of the data check (one dry run, the registered run, and the table-fix re-run)
  read the v2 SPY 5-minute file in full (to 2025-12-31) and only then filtered it to ≤ 2024-02-29. The session-set
  helper likewise read the file's whole `session_date` column. No later row entered any computation or output.
  - Commit 8163aa6 now filters at read time.
  - The re-run with that commit reproduces `p1_summary.csv` byte for byte.
  - These v2 SPY 5-minute bars had already been used through 2025-12 as TSLA market context. They have never been
    examined for this strategy.

## 2. SPY spread and the D29a-2 cost floor (step 2)

Tables: [`tables/spy_stageA/spread_by_year.csv`](tables/spy_stageA/spread_by_year.csv) (per session:
`spread_daily.csv`). EDGE uses the D25 (ii) procedure on the unadjusted 1-minute bars; no session had a NaN or
negative estimate.

| year | sessions | median price | EDGE half-spread (bp) | half tick (bp, at median price) | binds (at median price) | used as cost in year |
|---|---|---|---|---|---|---|
| 2016 | 252 | 210.27 | 0.162 | 0.238 | half tick | 2016 (own year), 2017 |
| 2017 | 251 | 243.71 | 0.144 | 0.205 | half tick | 2018 |
| 2018 | 251 | 273.96 | 0.155 | 0.183 | half tick | 2019 |
| 2019 | 252 | 291.75 | 0.125 | 0.171 | half tick | 2020 |
| 2020 | 253 | 326.66 | 0.187 | 0.153 | EDGE | 2021 |
| 2021 | 252 | 428.77 | 0.115 | 0.117 | half tick | 2022 |
| 2022 | 251 | 403.89 | 0.198 | 0.124 | EDGE | 2023 |
| 2023 | 250 | 428.45 | 0.123 | 0.117 | EDGE (barely) | 2024 |

- **Per order in the backtest:** half-spread = max(EDGE_{Y−1} bp × notional, $0.005 × shares), at the fill price.
  The "binds" column above compares year Y's own EDGE with the half tick at year Y's median price. For orders, what
  matters is year Y−1's EDGE against the half tick at the fill price.
- **Total realistic cost** (half-spread, $0.0035 per share commission with a $0.35 minimum, and the 0.10 bp SEC fee)
  averages **0.387 bp per side** of traded notional for the primary configuration over P1.
- **Paper cost** ($0.0045 per share) averages **0.133 bp**.

## 3. Implementation and tests (step 3)

Code: `src/tslaq/spy/momentum.py` (rules), `metrics.py`, `data.py`, and the runner `scripts/spy_stageA.py`.

**Tests:** `tests/test_spy_momentum.py`, 13 tests, all passing. The full suite passed 77 of 77 before the guard test
was added; that test passes too. The tests cover:
- **No lookahead (truncation).** σ, UB/LB, VWAP, σ_SPY, the previous close and the day open are identical when the
  data is cut at 30 random points, and unchanged when every later session is altered.
- **D29a-1.** After a half-day, the minute-300 σ skips the half-day and averages the 14 previous sessions that contain
  that minute. Minute 100 uses the 14 calendar sessions. σ is NaN when only 13 sessions contain the minute, and
  afternoon trading resumes the day after a half-day.
- **σ_SPY.** It is computed from the 14 returns ending at yesterday's close.
- **Fills.** E0 fills at the open of minute m+1, E1 at minute m+2. The position is flat at the last close, and no
  position carries overnight. A signal does not change when later prices change. On a half-day the decisions run
  10:00..12:30 and the exit is at the 13:00 close.
- **Opposite-band holding.** The secondary configuration (a) keeps its position until the opposite band is crossed.
- **Costs.** The paper cost is checked; the $0.35 minimum and the half-tick floor bind on a small order; EDGE and the
  per-share commission apply on a large order; 2× doubles everything; a reversal is one order whose cost is split
  between the two trades.
- **Sizing.** Share rounding and the 4× leverage cap.
- **Stage A guard.** Loaders refuse, or filter out, data after 2024-02.

## 4. P1 results (step 4)

**Period and conventions.**
- P1 runs from **2016-01-26** (the first session with full look-backs) to 2024-02-29, 2,037 sessions.
- The account starts at $100,000 and compounds.
- Sharpe is computed on daily returns, annualized with √252, with no risk-free rate.

**Tables.**
- Every configuration × cost × execution:
  [`tables/spy_stageA/p1_summary.csv`](tables/spy_stageA/p1_summary.csv).
- Yearly: [`tables/spy_stageA/p1_yearly.csv`](tables/spy_stageA/p1_yearly.csv).
- Monthly, all runs: `p1_monthly_all.csv`.
- Comparison with the paper: [`p1_monthly_comparison.csv`](tables/spy_stageA/p1_monthly_comparison.csv),
  [`p1_yearly_comparison.csv`](tables/spy_stageA/p1_yearly_comparison.csv) and `p1_comparison_by_cost.csv`.
- Verdict: `p1_verdict.json`.

### 4.1 Primary configuration (Curr.Band + VWAP, dynamic sizing)

| cost | exec | total return | CAGR | vol | Sharpe | t | Sortino | MDD | alpha (NW t) | beta |
|---|---|---|---|---|---|---|---|---|---|---|
| gross | E0 | 353.8% | 20.6% | 14.4% | 1.37 | 3.89 | 2.61 | 22.3% | 20.7% (4.1) | −0.06 |
| paper | E0 | 291.7% | 18.4% | 14.4% | 1.24 | 3.53 | 2.34 | 25.4% | 18.8% (3.8) | −0.06 |
| **real** | **E0** | **206.3%** | **14.9%** | 14.5% | **1.03** | 2.93 | 1.90 | 29.8% | 15.8% (3.1) | −0.06 |
| 2× real | E0 | 106.6% | 9.4% | 14.5% | 0.69 | 1.96 | 1.23 | 39.0% | 10.9% (2.2) | −0.06 |
| gross | E1 | 294.8% | 18.5% | 14.3% | 1.26 | 3.58 | 2.38 | 25.5% | 19.0% (3.8) | −0.06 |
| paper | E1 | 240.7% | 16.4% | 14.3% | 1.13 | 3.22 | 2.11 | 28.5% | 17.2% (3.4) | −0.06 |
| real | E1 | 166.3% | 12.9% | 14.3% | 0.92 | 2.61 | 1.68 | 33.1% | 14.1% (2.8) | −0.06 |
| 2× real | E1 | 79.6% | 7.5% | 14.4% | 0.58 | 1.64 | 1.02 | 42.3% | 9.2% (1.8) | −0.06 |
| SPY buy & hold | | 211.7% | 15.1% | 18.3% | 0.86 | 2.44 | 1.20 | 33.8% | | |

Other numbers at real cost with E0 (paper values for 2007–2024 in brackets):
- **Hit ratio:** 42.5% of traded days [43%]; 25.5% of all days. The strategy trades on 60% of days.
- **Distribution:** skew 2.0 [1.22]; worst day −5.2% [−4.5%]; best day +9.1% [+9.1%].
- **Orders:** 1.77 per day. The paper's "trades per day" is 1.8; its counts match our orders (entries + exits), not
  our round trips.
- **Round trips:** 0.88 per day. Trade hit ratio 37.7% (38.9% at paper cost) [37%].
- **Average P&L per share:** gross $0.132, net $0.107 (paper cost $0.123) [$0.09 over 2007–2024; the FAQ says $0.18
  over the last seven years].
- **Turnover:** 4.8× equity per day.
- **Break-even:** 2.10 bp per side (gross run; 1.98 bp in the real-cost run, whose sizes are smaller).
- **Legs:** sum of daily returns 77.7% long (Sharpe 1.12) and 42.6% short (Sharpe 0.45). Both legs are positive, as
  in the paper's FAQ Q5.

**Yearly returns** (primary, E0; paper = the dynamic version over the same months; 2016 from February for the
comparison; 2024 = January–February):

| year | gross | paper cost | real | 2× real | real, E1 | paper (same months) | SPY B&H |
|---|---|---|---|---|---|---|---|
| 2016 | −10.8% | −13.4% (Feb–Dec −13.8%) | −16.8% | −22.4% | −20.2% | −11.9% | 21.7% |
| 2017 | −6.3% | −9.3% | −13.5% | −20.1% | −15.0% | −7.0% | 21.7% |
| 2018 | 58.7% | 55.4% | 50.5% | 42.7% | 51.0% | 60.9% | −4.5% |
| 2019 | 7.8% | 5.8% | 2.8% | −2.0% | −0.6% | 6.8% | 31.1% |
| 2020 | 26.3% | 25.1% | 23.3% | 20.3% | 25.8% | 26.6% | 18.3% |
| 2021 | 31.4% | 29.7% | 25.6% | 20.1% | 24.6% | 34.9% | 28.7% |
| 2022 | 25.9% | 24.9% | 23.0% | 20.2% | 23.7% | 24.3% | −18.2% |
| 2023 | 41.3% | 39.6% | 35.2% | 29.4% | 27.2% | 37.3% | 26.2% |
| 2024 (Jan–Feb) | 7.4% | 7.2% | 6.7% | 5.9% | 6.1% | 7.2% | 6.9% |

(SPY B&H 2016 is from 2016-01-26.)

**Observation (not a criterion).** Our gross run is closer to the paper's table (monthly Sharpe 1.40, CAGR 20.5%)
than our paper-cost run (1.26, 18.3%) (`p1_comparison_by_cost.csv`). The likely reason is that the paper's costs, as
actually charged, were lower than $0.0045 per share on every order:
- the $0.0020 commission tier applied on about 20% of days;
- the authors' code charges no slippage.

The P1 verdict uses the paper-cost run, as pre-registered.

### 4.2 Secondary configurations (replication only)

| configuration | cost | total | CAGR | vol | Sharpe | MDD | hit (traded days) | paper 2007–2024: IRR / vol / Sharpe / MDD / hit |
|---|---|---|---|---|---|---|---|---|
| Opp.Band, 100% | paper, E0 | 52.4% | 5.4% | 8.9% | 0.63 | 13.4% | 54.6% | 6.2% / 10.9% / 0.61 / 21% / 54% |
| Opp.Band, 100% | real, E0 | 43.0% | 4.5% | 8.9% | 0.54 | 15.0% | 53.8% | |
| Opp.Band, 100% | real, E1 | 40.1% | 4.3% | 9.0% | 0.51 | 15.8% | 53.8% | |
| Curr.Band + VWAP, 100% | paper, E0 | 71.0% | 6.9% | 6.8% | 1.01 | 11.6% | 43.8% | 9.7% / 7.7% / 1.24 / 12% / 43% |
| Curr.Band + VWAP, 100% | real, E0 | 56.4% | 5.7% | 6.8% | 0.85 | 11.8% | 42.5% | |
| Curr.Band + VWAP, 100% | real, E1 | 53.8% | 5.5% | 6.7% | 0.83 | 11.0% | 42.8% | |

- **Opp.Band, 100%.** Our hit ratios and orders per day match the paper's Table 4 (1.23 orders per day vs 1.3).
- **Curr.Band + VWAP, 100%.** It has the same signals as the primary configuration, at 1× instead of
  volatility-targeted size.
- **Skew** of the Opp.Band variant is −0.17 (paper −1.24 over 2007–2024, which includes 2008).

### 4.3 DSR (information only)

D29 defines the DSR on P2. For information over P1, with N = 46 and V = the variance of the three configurations' P1
daily Sharpe ratios at real cost (2.4e-4; SR\* = 0.035 per day):

| configuration | DSR | PSR(0) |
|---|---|---|
| primary | 0.92 | 0.999 |
| Curr.Band + VWAP, 100% | 0.80 | |
| Opp.Band, 100% | 0.48 | |

Table: [`tables/spy_stageA/p1_dsr_info.csv`](tables/spy_stageA/p1_dsr_info.csv).

This V reflects only three closely related variants, so it understates the multiple testing. P1 is also in-sample for
the paper.

## 5. Rule-deviation check (step 5)

P1 is within the tolerance, so no fix was needed. Two things were still checked:
- **Trade count.** It matches once "trades" is read as orders.
- **Hit and skew.** Both are consistent with the paper.

**Code changes after the first registered run** (8820812). Neither touches strategy logic, and both were verified to
reproduce `p1_summary.csv` byte for byte:
- 6ce1b73: `p1_yearly.csv` stored the paper column in percent; it is now in fractions.
- 8163aa6: the v2 reads are filtered at load time (§1).

**Process note.** One dry run without registry logging, on the same code, preceded the registered run. It was used
to catch crashes and produced the same numbers.

## 6. Freeze (step 6)

- **Frozen code:** commit **8163aa6** (`src/tslaq/spy/`, `scripts/spy_stageA.py`).
- **Config hash:** **3d8abf0044** (`tables/spy_stageA/frozen_config.json`). It covers the rule constants, the cost
  constants, `configs/data.yaml`, the SHA-1 of `spread_by_year.csv` and the half-spread per cost year (2024: 0.123 bp).
- **Record:** the freeze note is in `decisions.md` under D29.
- **Registry:** three trials, `d29_spy_momentum_{primary, opp_band_1x, vwap_1x}`, phase `spy_A`, git 8820812, config
  3d8abf0044; n_trials_so_far = **46**.

Stage B (P2 2024-03..2025-12 and P3 2026-01-02..2026-10-09, once, with this code) still needs code for the later
data. The loaders are capped at 2024-02 and would need an approved, documented extension. They would read P2 from
`data/raw/SPY/` and P3 from `data/raw/holdout/SPY/`, with nothing after 2026-10-09. The data check would run on v2 and
v2h. The spread years would extend to 2024–2025 for the cost years 2025–2026. **This needs the user's approval.**
