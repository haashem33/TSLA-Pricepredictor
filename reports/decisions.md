# Decisions log

Each entry: date, decision, reason. This log wins over CLAUDE.md if they ever disagree (CLAUDE.md rule 12).

## 2026-10-22 — after Phase 1 review (approved by the user)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D1 | **Primary IC** = Spearman of predicted z vs realized z, pooled within each monthly test fold. ICIR = mean/std across folds; also t = ICIR·√n_folds and a one-sided binomial p-value for the share of folds with IC > 0. Cross-check: per-time-slot IC across days. The per-day IC is retired. | The per-day IC of one stock ranks bars over time within a day and is biased by overlapping windows: on a random walk, 78-bar momentum scores −0.31 at h = 12. Ranking E[r] = ẑ·S rewards volatility ranking. | §1, §6.8, §7.4, §10.6, §12, §13 |
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
| D10 | **Recency check (replaces the gate part of D3).** A gate passes the recency check if (a) the full-period result passes with **t ≥ 2** and (b) over **2023–2025 (36 folds)** the mean IC is **> 0 with t ≥ 1**. Last-24-fold numbers are still reported, for information only. | At h = 12 the per-fold IC sd is about 0.06–0.075 (0.062 over all 96 folds, 0.073 over the last 24), so 24 folds reach t = 2 only for IC ≳ 0.026–0.03. That is above the ≈ 0.021 IC needed to break even at 3 bp per side, so the old rule would reject real edges. | §1, §7.4, §10.6, §12 |

## 2026-10-22 — before Phase 2 (approved by the user)

| # | Decision | Reason | Where in CLAUDE.md |
|---|---|---|---|
| D11 | **Volatility-head comparison (refines D5/D9).** The Phase 4 volatility head is compared with GARCH-t×seasonal **on identical rows** (RV > 0 and no forward-filled bar in the label window), **fold by fold**, with the Diebold–Mariano test on the QLIKE loss differential (day-clustered errors, D4). The v2 QLIKE values 1.435 / 0.522 / 0.248 (h = 1 / 3 / 12) are **reference values only, not a fixed target**. Robustness check: also report QLIKE with RV floored at the 1st percentile of the training-fold RV. | h = 1 QLIKE is dominated by a few tiny-RV rows (D9: the RV = 0 exclusion alone moved it from 1.307 to 1.435), so a fixed number on a different row set is not a fair target. | §1, §9.3, §10.3, §12 |
| D12 | **Missing market sessions.** On TSLA sessions where QQQ or SPY bars are missing (QQQ 2018-05-02/03, SPY 2019-08-12, and any others found), **keep the TSLA rows** and set the market features and tokens to **MISSING**. The same applies to forward-filled market bars, and to the market bar-0 return after a missing market session, because it spans more than one night. | Dropping TSLA sessions for a market-data gap loses target data. MISSING is already a token for every field. | §6.7, §7.2 |

## Closed proposals

| # | Proposal | Status |
|---|---|---|
| P1 | Compute TSLA returns from **unadjusted** 1-minute prices (Alpaca `Adjustment.RAW`, same free keys as the earlier project), dividing by the cumulative split factor in floating point (no rounding), with explicit handling of the two split sessions. | Split-adjusted prices before 2022-08-25 are rounded to $0.01 (100% exact cent multiples in 2017–2021 vs about 50% after the split). In 2016 the effective tick was 7.2 bp instead of about 0.5 bp, and 18% of 5-min returns were exactly zero. See `phase1_checks.md`, B1. **Approved 2026-10-22 → D9.** |
