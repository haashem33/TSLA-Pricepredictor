# SPY intraday momentum study: plan (Step 1)

Date 2026-10-31. The binding specification is **D29** in [`decisions.md`](decisions.md); this file summarises it, lists
what the paper reports, and records every ambiguity with its resolution. **No backtest has been run.**

**Paper.** Zarattini, Aziz & Barbon, "Beat the Market: An Effective Intraday Momentum Strategy for S&P500 ETF (SPY)",
SSRN / SFI Research Paper 24-97. Version of 2025-02-03 (first version 2024-05-10), from
[alexandria.unisg.ch](https://alexandria.unisg.ch/server/api/core/bitstreams/a99aba00-f967-49b3-aceb-f544dc386e0b/content)
(SHA-256 41d635d1…b207ad). The paper used IQFeed 1-minute SPY data from May 2007 to April 2024, backtested in Matlab.
The authors' Python code (concretumgroup.com) is a secondary reference.

## 1. Rules (paper §3)

**Noise area.**
- **Per day.** For each of the previous 14 days and each time of day HH:MM, take the absolute move from the open,
  |Close_HH:MM / Open_9:30 − 1|.
- **σ.** The average of that move over the 14 days, separately for each time of day.
- **Bands.**
  - UB = max(Open_t, Close_{t−1}) · (1 + VM·σ).
  - LB = min(Open_t, Close_{t−1}) · (1 − VM·σ).
  - VM = 1. Using the previous close adjusts the bands for the overnight gap.

**Trading.**
- **Entry.** Long above the noise area, short below it.
- **Timing.** Trading happens only at HH:00 and HH:30. The first position is taken at 10:00. Stops are also checked
  only at those times.
- **Exits.** Flat at the 16:00 close.
- **Base model.** The opposite band is the stop. A cross of the opposite band reverses the position.
- **Final model.** The trailing stop is max(UB, VWAP) for longs and min(LB, VWAP) for shorts. VWAP is computed over
  regular hours only.

**Sizing.**
- **Base.** 100% of the previous day's equity, shares = AUM_{t−1} / Open_t.
- **Final.** Shares = AUM_{t−1} / Open_t · min(4, 2% / σ_SPY,t), where σ_SPY,t is the sd of the last 14 daily
  returns. Leverage is capped at 4×.

**Costs.** $0.0035 per share commission (IBKR entry level) plus $0.001 per share slippage.
- The slippage figure comes from more than 1,000 live $100k market orders sent just before HH:MM, measured against
  Open_HH:MM.
- The paper says about 20% of days would qualify for the $0.0020 commission tier.

## 2. What the paper reports (2007-05..2024-04, net of the costs above)

| strategy | total return | IRR | vol | Sharpe | hit | MDD | alpha | beta | trades |
|---|---|---|---|---|---|---|---|---|---|
| Opp.Band, 100% | 178% | 6.2% | 10.9% | 0.61 | 54% | 21% | 7.1% | −0.05 | 5,494 |
| Curr.Band + VWAP, 100% | 380% | 9.7% | 7.7% | 1.24 | 43% | 12% | 9.9% | −0.03 | 7,668 |
| **Curr.Band + VWAP, dynamic** | **1,985%** | **19.6%** | **14.3%** | **1.33** | 43% | 25% | 19.6% | −0.07 | 7,668 |
| SPY buy & hold | 227% | 7.2% | 20.2% | 0.45 | 54% | 56% | | | |

Source: Tables 1–4 and A1. Further reported figures for the dynamic version:
- **Distribution.** Skew 1.22, worst day −4.5%, best day +9.1%.
- **Trading.** 1.8 trades per day, average P&L $0.09 per share.
- **Lookback.** Sharpe at other lookbacks (FAQ Q6): 1.23–1.50; 14 days gives 1.35.
- **Later version.** FAQ Q22, through 2024-12: Sharpe 1.35, total return 2,382%.

**Yearly returns of the dynamic version** (FAQ Q4 / Q24, %; 2024 and 2025-01 are the authors' post-publication
update):

| 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025-01 |
|---|---|---|---|---|---|---|---|---|---|
| −12.8 | −6.9 | 61.1 | 6.9 | 26.8 | 34.8 | 24.4 | 37.2 | 32.2 | −1.2 |

The compounded monthly values reproduce these yearly figures within 0.2 pp. Over the months 2016-02..2024-02, the
monthly table gives an annualized Sharpe of 1.37 (from monthly returns), a CAGR of 20.1% and vol of 14.2%. These are
the P1 reference numbers.

## 3. Ambiguities and resolutions

| # | issue | resolution |
|---|---|---|
| A1 | The printed move formula lost its absolute-value bars in the PDF. | Absolute value: the text says "absolute move", and the code uses `abs`. |
| A2 | 14-day window: the code allows 13 observations (`min_periods = 13`). | All 14 previous sessions are required (paper). |
| A3 | σ_SPY window: the paper uses ret_{t−1..t−14}; the code uses t−15..t−2 (it skips yesterday) and its comment says 15. | Paper: the 14 returns ending at yesterday's close. |
| A4 | σ_SPY return type is not defined. | Daily close-to-close, dividend-adjusted (as in the code). |
| A5 | The VWAP price is not defined in this paper. | Typical price (H+L+C)/3 × volume, regular session only (code; the paper's footnote 2). |
| A6 | Does the final model's entry also require VWAP? | Yes, as in the code. Under the paper's stop, a long below VWAP would be closed at the very next check. |
| A7 | Are stops checked continuously or semi-hourly? | Semi-hourly only (paper: "Stop losses can also be triggered only on semi-hourly intervals"). |
| A8 | Re-entry after a stop. | Allowed at any later decision where the condition holds (code). |
| A9 | First and last decision times. | 10:00 (paper §4.3) to 15:30. The code's 15:59 signal is never acted on. |
| A10 | Fill price. | E0: the open of the bar starting at HH:MM (the paper's slippage reference, Open_HH:MM). The code fills at the decision bar's close, about the same price. |
| A11 | Close exit. | The last bar's close, i.e. market-on-close. It is not delayed in E1. |
| A12 | Share rounding. | `round` (code). It can exceed the 4× cap by less than one share. |
| A13 | Commission details. | C_paper: $0.0035 + $0.001 per share, no minimum, no volume tier. C_real: $0.35 minimum per order (IBKR fixed; also in the code). |
| A14 | Sharpe and hit-ratio definitions. | Sharpe: daily, √252, no risk-free rate (as in the code). Hit ratio: reported both over days with a trade and over all days. |
| A15 | Dividends in the gap adjustment. | A back-adjusted series, which equals the code's C_{t−1} − D on an ex-date. Shares and costs use unadjusted prices. |
| A16 | Half-days are not mentioned. | Decisions from 10:00 to 12:30, flat at the 13:00 close. |
| A17 | Our data starts 2016-01-04, so look-backs end ≈ 2016-01-22. | Trading starts ≈ 2016-01-25. January 2016 is excluded from the P1 monthly comparison. |
| A18 | The P2 start (2024-03) is inside the paper's data, which ends 2024-04; the first version is dated 2024-05-10. | Keep the user's window for the criterion, and also report 2024-05..2025-12 for information. |
| A19 | Reported figures differ between versions (trades 7,668 vs 7,964; 1× total 380% vs 393%) and different sample ends. | The monthly table (Q24) is the reference for P1. Table 1–4 values are context only. |
| A20 | Data vendor: IQFeed vs Alpaca SIP, which may differ in minute open/close prints and volumes. | This is why P1 is judged against a tolerance, not exact equality. |

## 4. Our test (summary of D29)

- **Data.**
  - SPY 1-minute, data v2 (2016-01..2025-12, `data/raw/SPY/`) and v2h (2026-01-02..2026-10-09,
    `data/raw/holdout/SPY/`).
  - v2 cleaning at 1-minute resolution and dividend back-adjustment.
  - Check before any backtest: the 5-minute aggregation must reproduce the v2 / v2h SPY returns.
  - Nothing after 2026-10-09 is read.
- **Configurations.** The primary configuration (Curr.Band + VWAP, dynamic) is judged against the criteria. Opp.Band
  100% and Curr.Band + VWAP 100% are run for replication. Nothing is tuned.
- **Execution.** E0 fills at the next bar's open; E1 adds one extra bar of delay. Both are flat at the close.
- **Costs.**
  - C_paper: $0.0035 + $0.001 per share.
  - C_real: the SPY EDGE half-spread of the prior year + $0.0035 per share (min $0.35) + 0.10 bp SEC fee.
  - Also 2× C_real and gross.
- **Criteria (primary configuration).**

  | period | dates | criterion |
  |---|---|---|
  | P1 | 2016-01..2024-02 | report only; "within range" = monthly correlation ≥ 0.80, Sharpe within ±0.30 of 1.37, CAGR within ±4 pp of 20.1%, yearly sign agreement in ≥ 7 of 8 years (C_paper, E0) |
  | P2 | 2024-03..2025-12 | PASS if net Sharpe > 0.5 at C_real, > 0 at C_real with E1, and > 0 at 2× C_real |
  | P3 | 2026-01-02..2026-10-09 | PASS if net P&L > 0 at C_real |

- **Benchmark.** SPY buy-and-hold over the same periods.
- **Order.**
  - Stage A: P1 only, with no data after 2024-02 loaded.
  - Freeze: code commit and config hash recorded.
  - Stage B: P2 and P3, run once.
  - Each stage needs the user's approval.
- **Trials.** n_trials_so_far continues from 43; this study adds 3 trials (→ 46).

## 5. Next step (not started)

Step 2 = Stage A, once the user approves:
1. the 1-minute SPY build and the data check;
2. the SPY EDGE spread table;
3. the backtest code with tests (no lookahead in the bands, VWAP and σ_SPY; fill at t+1; costs);
4. the P1 report against the tolerance above.
