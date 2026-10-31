# TSLA intraday forecasting and SPY intraday momentum: two pre-registered studies

A research repository with two concluded studies. Both were run with pre-registered decisions, leakage tests,
walk-forward or staged evaluation, a trial registry and sealed out-of-sample data. **Neither found a tradable edge.**

## Results

**Read first:** [`reports/final_report.md`](reports/final_report.md). Every decision, with its date and reason, is in
[`reports/decisions.md`](reports/decisions.md) (D1–D29c).

| study | question | outcome | report |
|---|---|---|---|
| **TSLA** (D1–D28) | Can tokenized 5-minute data and a small model forecast TSLA's next 5–60 minute returns well enough to trade after costs? | No edge confirmed. Gate 3 failed, no development trial passed (DSR 0.000), and the single holdout evaluation (2026-01-01..2026-10-09) lost −725 bp net. Stopped. The Transformer (Phase 4) was not built. | [`final_report.md`](reports/final_report.md) §1–10 |
| **SPY** (D29–D29c) | Does the published SPY intraday momentum strategy of Zarattini, Aziz & Barbon (2024) survive after publication? | The replication matched the paper (monthly correlation 0.993). Post-publication, P2 passed narrowly (Sharpe 0.59) but P3 failed (−13.6%), and 2025 lost −4.8%. Over the full period: 204.5% vs SPY buy-and-hold 393.3%. Stopped, no paper trading (the user's decision after seeing the results). | [`final_report.md`](reports/final_report.md) §11, [`spy_stageA.md`](reports/spy_stageA.md), [`spy_stageB.md`](reports/spy_stageB.md) |

## Setup

- **Requirements.** Python ≥ 3.11 (developed on Windows with a local `.venv`).

  ```bash
  python -m venv .venv
  .venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
  .venv/Scripts/pip install -e .
  ```

- **Credentials.** Put them in a `.env` file in the repository root. It is gitignored and never committed, and the
  code never prints it. Only the variable names are listed here:

  | variable | used for |
  |---|---|
  | `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | Alpaca SIP 1-minute bars (`src/tslaq/data/download_alpaca.py`, holdout download in `scripts/run_holdout_d28.py`) |
  | `ALPHAVANTAGE_API_KEY` | Alpha Vantage (`configs/data.yaml`). In this project, earnings, dividends and splits were fetched once and stored under `data/raw/earnings/` and `data/raw/dividends/` |

- **Optional environment variables.**
  - `TSLAQ_DATA_VERSION` overrides `configs/data.yaml: data_version`. It is used only by the D28 holdout run
    (`v2h`).
  - `PYTHONHASHSEED` is set by some scripts for reproducibility.
- **Development source.** The 1-minute source files were read from a local folder (`configs/data.yaml:
  local_source_dir`) and copied into `data/raw/<SYMBOL>/<YYYY-MM>.parquet`.

## Reproducing the studies

All commands run from the repository root with the project's interpreter (`.venv/Scripts/python` on Windows).

**Tests and lint.**

```bash
python -m pytest -q
ruff check .
```

### TSLA study (D1–D28)

Run in this order; each script's docstring lists its options:

| step | command |
|---|---|
| data v1 / v2 (5-minute bars) | `python -m tslaq.data.clean v2` (after the raw 1-minute files are cached) |
| Phase 1 checks and baselines | `python scripts/phase1_checks.py v2`, `python scripts/run_phase1_baselines.py` |
| Phase 2 features and selection | `python scripts/run_phase2_walkforward.py` |
| Phase 2 metric tests and null | `phase2_metric_null*.py`, `phase2_metric_placebo.py`, `phase2_daily_*.py`, `phase2_null_quantiles.py`, `phase2_rescore_v2.py` |
| Gate 3 and its null | `python scripts/run_phase3_gate3.py --real`, `python scripts/phase3_null_chain.py` |
| D25 baselines and spreads | `python scripts/run_phase3_baselines.py --real`, `python scripts/phase3_spread.py` |
| D27 early test and CPCV | `python scripts/run_phase5_early.py`, `python scripts/run_phase5_cpcv.py` |
| D28 holdout (run once, done) | `python scripts/run_holdout_d28.py --freeze / --source-check / --repro / --download / --build / --holdout / --evaluate --log` |

The full walk-forward and the null runs take many hours.

### SPY study (D29–D29c)

| step | command | notes |
|---|---|---|
| Stage A (P1 2016-01..2024-02) | `python scripts/spy_stageA.py --no-registry` | Rebuilds the data check, spread table, P1 backtests and comparison tables. Without `--no-registry` it would log 3 new registry rows. |
| Stage B spread table | `python scripts/spy_stageB.py --spread` | Model-free; 2016–2025. |
| Stage B dry run | `python scripts/spy_stageB.py --dry <output dir>` | Stage A data only. |
| Stage B evaluation | `python scripts/spy_stageB.py --run` | **Run once.** It refuses to run again because `reports/tables/spy_stageB/stageB_done.json` exists (D29b). |

**Frozen code.**
- Strategy: commit `8163aa6`, config hash `3d8abf0044`.
- `src/tslaq/spy/momentum.py` and `metrics.py` are unchanged since then.

## Data layout

```
data/raw/            cached 1-minute bars: TSLA/, SPY/, QQQ/ (vendor, split-adjusted), TSLA_raw/ (unadjusted),
                     holdout/ (2026-01..2026-10-09, D28), dividends/, earnings/, download_log.csv
data/interim/v1, v2  5-minute regular-session bars 2016-2025 (v2 = TSLA rebuilt from unadjusted prices, D9)
data/interim/v2h     v2 + holdout 2026-01-02..2026-10-09 (D28)
data/interim/spy1m/  SPY 1-minute frame and daily strategy returns (D29)
data/processed/      per-fold features, tokens, predictions, null draws
reports/             one report per phase / stage, decisions.md, tables/, figures/
experiments/         registry.csv
```

**Gitignored, not in the repository:**
- `.env`;
- all of `data/interim/` and `data/processed/`;
- the Parquet files under `data/raw/` (`data/raw/**/*.parquet`);
- `.venv/`, caches, `*.egg-info/`.

The CSV files under `data/raw/` (dividends, earnings, splits, download log) are tracked. Vendor bar data is not
redistributed.

## Sealed data

- **Holdout (2026-01-01..2026-10-09).** Used once for TSLA (D28) and once for the SPY P3 check (D29b). It is spent.
- **Forward test (any bar after 2026-10-09 16:00 ET).** **Sealed.** It has never been downloaded or read for either
  study. The data layer refuses holdout months by default. The SPY holdout reader filters out everything from
  2026-10-09 16:00 ET on, at read time.
  - The 2026-10 holdout file also contains post-market bars of 2026-10-09 (to 19:59 ET). They have never been read for
    prices.
  - Per the final report, the forward test should be used only for a new, fully pre-registered idea, and only once.

## Trial registry

`experiments/registry.csv` logs one row per run: git hash, config hash, phase, metrics and `n_trials_so_far`. Re-runs
carry `rerun_of` and do not add a trial.

**`n_trials_so_far = 46`:** 43 TSLA trials (D1–D28) + 3 SPY configurations (D29 Stage A; the Stage B rows are
re-runs). This count is the N used in the deflated Sharpe ratio.
