# Phase 1 follow-up checks (2026-10-22)

Nothing in the pipeline was changed by these checks. Tables: `reports/tables/check_b1_rounding.csv`, `check_b2_b3_delay_ic.csv`, `check_b3_slot_ic.csv`.

## B1. Price rounding, exact zeros, lag-1 autocorrelation

Intraday bars only (bar_index > 0, non-missing). Cent-multiple share uses the source 1-minute closes in the regular session (split-adjusted, before my dividend adjustment).

| year | TSLA zero r | TSLA AC(1) | TSLA cent-multiple | TSLA median adj. price | SPY zero r | SPY AC(1) | QQQ zero r | QQQ AC(1) |
|---|---|---|---|---|---|---|---|---|
| 2016 | 18.1% | +0.002 | 99.1% | 13.93 | 2.4% | −0.005 | 3.8% | −0.002 |
| 2017 | 14.5% | −0.033 | 100% | 21.18 | 3.4% | −0.015 | 4.1% | −0.013 |
| 2018 | 9.3% | −0.018 | 100% | 21.21 | 1.8% | −0.021 | 1.9% | −0.011 |
| 2019 | 13.8% | −0.030 | 100% | 17.36 | 2.0% | −0.016 | 2.5% | −0.006 |
| 2020 | 2.2% | −0.036 | 100% | 80.37 | 1.0% | −0.016 | 0.9% | −0.004 |
| 2021 | 0.9% | −0.005 | 100% | 242.50 | 1.0% | −0.021 | 0.8% | −0.010 |
| 2022 | 0.4% | −0.020 | 84.9% | 271.41 | 0.6% | −0.017 | 0.5% | −0.010 |
| 2023 | 0.3% | +0.003 | 48.3% | 222.05 | 0.8% | +0.015 | 0.6% | +0.019 |
| 2024 | 0.3% | −0.007 | 51.5% | 211.79 | 0.8% | −0.028 | 0.7% | −0.030 |
| 2025 | 0.2% | −0.001 | 53.8% | 345.36 | 0.6% | +0.014 | 0.7% | +0.010 |

**Finding: the split-adjusted TSLA prices are rounded to $0.01.** Unadjusted prices (after 2022-08-25, and SPY/QQQ in every year) are exact cents only about 50–75% of the time, because of sub-penny prints. Adjusted TSLA prices are exact cents 100% of the time in 2017–2021. 2022 shows 85% because only January–August is rounded.

Size of the effect, assuming independent rounding of both prices (noise sd = $0.01/√6/P):

| year | effective tick (bp) | true tick (bp, ≈ tick/15 or /3) | rounding-noise sd (bp) | 5-min return sd (bp) | noise variance share | implied AC(1) from rounding |
|---|---|---|---|---|---|---|
| 2016 | 7.2 | 0.48 | 2.9 | 25.5 | 1.3% | −0.007 |
| 2017 | 4.7 | 0.31 | 1.9 | 18.9 | 1.0% | −0.005 |
| 2018 | 4.7 | 0.31 | 1.9 | 29.5 | 0.4% | −0.002 |
| 2019 | 5.8 | 0.38 | 2.4 | 23.3 | 1.0% | −0.005 |
| 2020 | 1.3 | ≈0.1–0.4 | 0.5 | 48.1 | 0.01% | ≈0 |
| 2021–22 | 0.4 | 0.13 | 0.2 | 31–37 | ≈0 | ≈0 |

So rounding explains the high exact-zero share before 2020 (18% → 0.2%), but only about −0.005 of the observed −0.02 to −0.04 lag-1 autocorrelation. The rest is real microstructure (bid–ask bounce), which SPY/QQQ show too (−0.01 to −0.03). The discreteness still distorts the return distribution, the RET token edges near zero, and any feature built from small returns in 2016–2019.

**Unadjusted 1-minute data is not in `Desktop\Data`.** The earlier project downloaded with `Adjustment.SPLIT` (`tsla-diffusion/download_data.py`), and its Alpaca keys are in `tsla-diffusion/.env` (I checked that the file exists, not its contents). **Proposal (P1 in `decisions.md`):** download TSLA 1-minute bars with `Adjustment.RAW` for 2016-01..2022-08 (≈80 monthly requests, free plan, SIP feed). Build 5-minute bars from raw prices, then divide by the cumulative split factor in floating point (no rounding). On the two split sessions (2020-08-31, 2022-08-25), compute the bar-0 return as ln(O_raw / (C_prev_raw / factor)). Intraday returns then come straight from unrounded raw prices. Volumes are multiplied by the factor. SPY/QQQ never split and need no change.

## B2. Baseline IC with one bar of execution delay

LightGBM (Phase 1), z-space fold IC. "Delay" = prediction at t vs realized z from t+1 to t+1+h. Both variants are scored on the same rows (those where t+1+h is in the same session).

| h | variant | period | IC | ICIR | t | folds IC>0 | binomial p | folds |
|---|---|---|---|---|---|---|---|---|
| 1 | no delay | 2018–2025 | 0.0240 | 0.80 | 7.82 | 80% | <1e-4 | 96 |
| 1 | delay 1 bar | 2018–2025 | 0.0126 | 0.63 | 6.13 | 74% | <1e-4 | 96 |
| 1 | no delay | 2023–2025 | 0.0096 | 0.39 | 2.32 | 72% | 0.006 | 36 |
| 1 | delay 1 bar | 2023–2025 | 0.0071 | 0.34 | 2.06 | 64% | 0.066 | 36 |
| 1 | no delay | last 24 | 0.0055 | 0.22 | 1.06 | 63% | 0.15 | 24 |
| 1 | delay 1 bar | last 24 | 0.0069 | 0.33 | 1.64 | 63% | 0.15 | 24 |
| 3 | no delay | 2018–2025 | 0.0189 | 0.58 | 5.64 | 70% | 1e-4 | 96 |
| 3 | delay 1 bar | 2018–2025 | 0.0134 | 0.43 | 4.23 | 60% | 0.026 | 96 |
| 3 | no delay | 2023–2025 | 0.0084 | 0.27 | 1.61 | 56% | 0.31 | 36 |
| 3 | delay 1 bar | 2023–2025 | 0.0093 | 0.31 | 1.86 | 56% | 0.31 | 36 |
| 3 | no delay | last 24 | 0.0039 | 0.13 | 0.64 | 46% | 0.73 | 24 |
| 3 | delay 1 bar | last 24 | 0.0070 | 0.24 | 1.16 | 50% | 0.58 | 24 |
| 12 | no delay | 2018–2025 | 0.0098 | 0.16 | 1.52 | 60% | 0.026 | 96 |
| 12 | delay 1 bar | 2018–2025 | 0.0104 | 0.16 | 1.60 | 59% | 0.041 | 96 |
| 12 | no delay | 2023–2025 | −0.0047 | −0.07 | −0.43 | 50% | 0.57 | 36 |
| 12 | delay 1 bar | 2023–2025 | −0.0038 | −0.06 | −0.34 | 50% | 0.57 | 36 |
| 12 | no delay | last 24 | −0.0030 | −0.04 | −0.21 | 54% | 0.42 | 24 |
| 12 | delay 1 bar | last 24 | −0.0019 | −0.03 | −0.13 | 58% | 0.27 | 24 |

By year (fold IC; no delay → delay):

| year | h=1 | h=3 | h=12 |
|---|---|---|---|
| 2018 | 0.021 → 0.017 | 0.020 → 0.017 | 0.012 → 0.012 |
| 2019 | 0.025 → 0.008 | 0.008 → 0.002 | 0.016 → 0.019 |
| 2020 | 0.053 → 0.027 | 0.043 → 0.030 | 0.038 → 0.036 |
| 2021 | 0.034 → 0.013 | 0.029 → 0.017 | 0.021 → 0.021 |
| 2022 | 0.031 → 0.015 | 0.026 → 0.014 | 0.006 → 0.007 |
| 2023 | 0.018 → 0.007 | 0.017 → 0.014 | −0.008 → −0.008 |
| 2024 | 0.007 → 0.000 | −0.002 → −0.000 | 0.002 → 0.004 |
| 2025 | 0.004 → 0.014 | 0.010 → 0.014 | −0.008 → −0.008 |

**Reading:** one bar of delay halves the h = 1 IC over the full period (0.024 → 0.013), which is consistent with a large bounce/next-bar component. At h = 12, the primary horizon, the delay changes nothing, but there is no baseline edge in 2023–2025 or the last 24 folds (t ≈ −0.2 to −0.4). Nothing passes the recency rule. This is only a lagged-returns baseline, so it says nothing yet about richer features.

## B3. Fold IC by year (h = 3, h = 12) and per-time-slot IC

Fold IC by year without delay is the left value in each cell of the B2 by-year table. Per-time-slot IC (Spearman across days at each bar_index, LightGBM, no delay), averaged within half-hour slots (full 78-slot table in `check_b3_slot_ic.csv`):

| slot start | h=1 2018–25 | h=1 2023–25 | h=12 2018–25 | h=12 2023–25 |
|---|---|---|---|---|
| 09:30 | 0.018 | 0.012 | 0.023 | −0.003 |
| 10:00 | 0.020 | 0.010 | 0.007 | −0.023 |
| 10:30 | 0.009 | −0.008 | 0.014 | 0.004 |
| 11:00 | 0.029 | 0.014 | 0.049 | 0.045 |
| 11:30 | 0.025 | 0.029 | 0.010 | −0.001 |
| 12:00 | 0.030 | 0.025 | −0.005 | −0.014 |
| 12:30 | 0.013 | −0.006 | 0.015 | 0.016 |
| 13:00 | 0.046 | 0.019 | 0.035 | 0.069 |
| 13:30 | 0.013 | −0.010 | 0.027 | 0.011 |
| 14:00 | 0.016 | 0.000 | 0.005 | −0.073 |
| 14:30 | 0.020 | −0.009 | 0.031 | 0.004 |
| 15:00 | 0.026 | 0.017 | – | – |
| 15:30 | 0.042 | 0.033 | – | – |

Share of the 78 slots with IC > 0: h=1 84% (2018–25) / 60% (2023–25); h=12 69% / 49%. Slot ICs over 2023–2025 use about 750 days per slot, so ±0.07 is roughly 2 standard errors. The 13:00 and 14:00 values for h = 12 are within noise, and no time-of-day pattern should be read from this table. (The h = 12 column has no labels after 15:00.)

## B4. Did the Phase 1 LightGBM use the test month for any choice?

**No.**
- Hyperparameters were fixed before the first run (`LGB_PARAMS`: 300 trees, lr 0.03, 15 leaves, min_child_samples 1000) and never tuned. There was no early stopping (`fit` has no `eval_set`).
- The Gaussian predictive sd came from training residuals.
- The historical mean and the seasonal-only constant came from training rows.
- GARCH-t was fit on u before the test month.
- λ = 0.97 was tuned once on 2016-01..2017-12, before the first test month (2018-01).

The test month was used only for scoring, so under D6 no re-run is required. Caveat: the residual sd was fit in-sample on training data, not on an inner validation set. Under D6, future calibration uses the inner validation months.
