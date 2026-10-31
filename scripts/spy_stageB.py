"""D29 Stage B (D29b): P2 = 2024-03..2025-12 and P3 = 2026-01-02..2026-10-09 with the frozen D29 strategy code.

  --spread  SPY EDGE spread table 2016-2025 (model-free; D29b (3)); run and commit before --run
  --dry     exercise every reporting path on Stage A data only (P1 sub-windows stand in for P2 / P3; output to the
            directory given after --dry; no registry, no marker)
  --run     the single Stage B evaluation (refuses to run if the marker exists; D29b (5))

Order inside --run: data build (nothing after 2026-10-09 16:00 ET) -> data checks (v2, v2h) -> P1 reproduction of the
Stage A daily files -> marker written -> simulation metrics for P2 / P3 / 2024-05..2025-12 / 2025 -> registry.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tslaq.backtest.dsr import dsr
from tslaq.spy import metrics as mt
from tslaq.spy import momentum as mo
from tslaq.spy.data import STAGE_B_RAW_END, build_spy_1min, build_spy_1min_full, data_check
from tslaq.utils.io import ROOT, config_hash
from tslaq.utils.registry import log_run, n_trials

sys.path.insert(0, str(ROOT / "scripts"))
from spy_stageA import frozen_config, run_all, spread_table

TAB_A = ROOT / "reports" / "tables" / "spy_stageA"
TAB = ROOT / "reports" / "tables" / "spy_stageB"
INTERIM_A = ROOT / "data" / "interim" / "spy1m"
MARKER = TAB / "stageB_done.json"
START = pd.Timestamp("2016-01-26")
WINDOWS = {"P1": ("2016-01-26", "2024-02-29"), "P2": ("2024-03-01", "2025-12-31"),
           "P3": ("2026-01-02", "2026-10-09"), "P2_after_paper_data": ("2024-05-01", "2025-12-31"),
           "Y2025": ("2025-01-01", "2025-12-31"), "full": ("2016-01-26", "2026-10-09")}
DRY_WINDOWS = {"P1": ("2016-01-26", "2021-12-31"), "P2": ("2022-01-01", "2023-10-31"),
               "P3": ("2023-11-01", "2024-02-29"), "P2_after_paper_data": ("2022-03-01", "2023-10-31"),
               "Y2025": ("2023-01-01", "2023-12-31"), "full": ("2016-01-26", "2024-02-29")}
D27_V = 0.2773


def spread_step():
    TAB.mkdir(parents=True, exist_ok=True)
    Y, hs, D = spread_table("2026-01", STAGE_B_RAW_END)
    Y.to_csv(TAB / "spread_by_year.csv", index=False)
    D.to_csv(TAB / "spread_daily.csv", index=False)
    A = pd.read_csv(TAB_A / "spread_by_year.csv", float_precision="round_trip")
    same = np.allclose(Y[Y.year <= 2023].drop(columns="binding_at_median_price").to_numpy(float),
                       A.drop(columns="binding_at_median_price").to_numpy(float), rtol=0, atol=0)
    print(Y.round(3).to_string(index=False))
    print("2016-2023 identical to Stage A table:", same)
    print("half-spread by cost year (bp):", {k: round(v, 4) for k, v in hs.items()})


def load_hs() -> dict:
    Y = pd.read_csv(TAB / "spread_by_year.csv", float_precision="round_trip")
    e = dict(zip(Y["year"], Y["edge_mean_half_bp"]))
    hs = {y: e[y - 1] if (y - 1) in e else e[y] for y in range(2016, 2027)}
    frozen = json.loads((TAB_A / "frozen_config.json").read_text())["half_spread_bp_by_cost_year"]
    for k, v in frozen.items():
        assert hs[int(k)] == v, f"cost year {k} changed"
    return hs


def window_rows(res, bench, windows):
    rows = []
    for w, (a, b) in windows.items():
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        for (cfg, cost, delay), (daily, trades) in res.items():
            d = daily[(daily.session_date >= a) & (daily.session_date <= b)]
            t = trades[(trades.session_date >= a) & (trades.session_date <= b)] if len(trades) else trades
            rows.append({"window": w, "from": a.date(), "to": b.date(), "config": cfg, "cost": cost,
                         "exec": f"E{delay}", **mt.strategy_stats(d, t, bench)})
        bb = bench[(bench.index >= a) & (bench.index <= b)]
        rows.append({"window": w, "from": a.date(), "to": b.date(), "config": "spy_buy_hold", "cost": "none",
                     "exec": "close-to-close", **mt.perf(bb.to_numpy())})
    return pd.DataFrame(rows)


def criteria(S: pd.DataFrame) -> dict:
    def g(w, cost, ex, col):
        r = S[(S.window == w) & (S.config == "primary") & (S.cost == cost) & (S.exec == ex)]
        return float(r[col].iloc[0])
    v = {"P2_sharpe_real_E0": g("P2", "real", "E0", "sharpe"), "P2_sharpe_real_E1": g("P2", "real", "E1", "sharpe"),
         "P2_sharpe_real2x_E0": g("P2", "real2x", "E0", "sharpe"),
         "P3_total_return_real_E0": g("P3", "real", "E0", "total_return")}
    v["P2_c1"] = v["P2_sharpe_real_E0"] > 0.5
    v["P2_c2"] = v["P2_sharpe_real_E1"] > 0
    v["P2_c3"] = v["P2_sharpe_real2x_E0"] > 0
    v["P2_PASS"] = bool(v["P2_c1"] and v["P2_c2"] and v["P2_c3"])
    v["P3_PASS"] = bool(v["P3_total_return_real_E0"] > 0)
    return v


def evaluate(df, hs, out: Path, windows: dict, log: bool, info_extra: dict):
    ind = mo.add_indicators(df)
    res = run_all(ind, hs, START)
    # P1 reproduction (D29b (2)) before any later number is written
    rep = []
    for (cfg, cost, delay), (daily, _) in res.items():
        a = pd.read_parquet(INTERIM_A / f"daily_{cfg}_{cost}_E{delay}.parquet")
        b = daily[daily.session_date <= a.session_date.max()].reset_index(drop=True)
        same_dates = len(a) == len(b) and bool((a.session_date.to_numpy() == b.session_date.to_numpy()).all())
        diff = float(np.abs(a["ret"].to_numpy() - b["ret"].to_numpy()).max()) if same_dates else np.inf
        rep.append({"config": cfg, "cost": cost, "exec": f"E{delay}", "n_days": len(a), "same_dates": same_dates,
                    "max_abs_diff_ret": diff, "pass": same_dates and diff <= 1e-12})
    R = pd.DataFrame(rep)
    out.mkdir(parents=True, exist_ok=True)
    R.to_csv(out / "p1_reproduction.csv", index=False)
    print("P1 reproduction: all pass =", bool(R["pass"].all()), "max diff", R["max_abs_diff_ret"].max())
    if not R["pass"].all():
        raise SystemExit("P1 reproduction failed; stopping before any P2 / P3 number (D29b (2))")
    if log:
        MARKER.write_text(json.dumps({"status": "started", **info_extra}, indent=1, default=str))
    daily_close = df.groupby("session_date")["close"].last()
    bench = daily_close / daily_close.shift(1) - 1
    S = window_rows(res, bench, windows)
    S.to_csv(out / "summary_by_window.csv", index=False)
    months = {f"{c}|{k}|E{d}": mt.monthly(v[0]) for (c, k, d), v in res.items()}
    years = {f"{c}|{k}|E{d}": mt.yearly(v[0]) for (c, k, d), v in res.items()}
    b = bench[bench.index >= START]
    bdf = pd.DataFrame({"session_date": b.index, "ret": b.to_numpy()})
    M = pd.DataFrame(months)
    M["spy_buy_hold"] = mt.monthly(bdf)
    M["paper_dynamic"] = mt.paper_monthly_series().reindex(M.index)
    M.index.name = "month"
    M.to_csv(out / "monthly.csv")
    Yt = pd.DataFrame(years)
    Yt["spy_buy_hold"] = mt.yearly(bdf)
    Yt["paper_dynamic_full_year"] = pd.Series(mt.PAPER_YEARLY) / 100
    Yt.index.name = "year"
    Yt.to_csv(out / "yearly.csv")
    ddir = (INTERIM_A / "stageB") if log else out
    ddir.mkdir(parents=True, exist_ok=True)
    for (c, k, d), (daily, _) in res.items():
        daily.to_parquet(ddir / f"daily_{c}_{k}_E{d}.parquet", index=False)
    v = criteria(S)
    # DSR (information): V over the three configurations' P2 daily Sharpe at real E0, and the D27 V
    a, z = (pd.Timestamp(x) for x in windows["P2"])
    p2 = {c: res[(c, "real", 0)][0] for c in mo.CONFIGS}
    p2 = {c: d[(d.session_date >= a) & (d.session_date <= z)]["ret"].to_numpy() for c, d in p2.items()}
    sr = {c: float(x.mean() / x.std(ddof=1)) for c, x in p2.items()}
    Vp2 = float(np.var(list(sr.values()), ddof=1))
    N = n_trials()
    D = [{"config": c, "V_source": vs, **dsr(p2[c], N, V)} for c in mo.CONFIGS
         for vs, V in (("P2_three_configs", Vp2), ("D27", D27_V))]
    pd.DataFrame(D).to_csv(out / "dsr_p2_info.csv", index=False)
    print(json.dumps(v, indent=1))
    if log:
        ch = info_extra["config_hash"]
        for c in mo.CONFIGS:
            r = S[(S.window == "P2") & (S.config == c) & (S.cost == "real") & (S.exec == "E0")].iloc[0]
            r2 = S[(S.window == "P2") & (S.config == c) & (S.cost == "real2x") & (S.exec == "E0")].iloc[0]
            r3 = S[(S.window == "P3") & (S.config == c) & (S.cost == "real") & (S.exec == "E0")].iloc[0]
            log_run({"config_hash": ch, "phase": "spy_B", "model": f"d29_spy_momentum_{c}", "n_params": "",
                     "fold_scheme": "D29b Stage B, P2 2024-03..2025-12 (P3 2026-01-02..2026-10-09 in notes); "
                                    "fill next open (E0); C_real (D29a-2)",
                     "hit_rate_h12": r["hit_traded_days"], "net_sharpe_period": sr[c], "net_sharpe_ann": r["sharpe"],
                     "sortino": r["sortino"], "max_drawdown": r["mdd"], "turnover": r["turnover_per_day"],
                     "sharpe_2x_costs": r2["sharpe"], "data_version": "v2 + v2h (SPY 1-min)",
                     "rerun_of": f"d29_spy_momentum_{c}",
                     "notes": f"Stage B; P3 total return C_real E0 {r3['total_return']:.4f}, Sharpe {r3['sharpe']:.3f}"})
        MARKER.write_text(json.dumps({"status": "done", **info_extra, **v}, indent=1, default=str))
        print("registry n_trials_so_far =", n_trials())
    return S, v


def main():
    args = sys.argv[1:]
    if "--spread" in args:
        return spread_step()
    if "--dry" in args:
        out = Path(args[args.index("--dry") + 1])
        df, _ = build_spy_1min()
        hs = load_hs()
        return evaluate(df, hs, out, DRY_WINDOWS, log=False, info_extra={})
    if "--run" in args:
        if MARKER.exists():
            raise SystemExit(f"{MARKER} exists: Stage B is evaluated once (D29b (5)); not re-running.")
        TAB.mkdir(parents=True, exist_ok=True)
        hs = load_hs()
        df, info = build_spy_1min_full()
        chk = {}
        _, c1 = data_check(df[df.session_date <= pd.Timestamp("2025-12-31")], "v2")
        _, c2 = data_check(df, "v2h", start=pd.Timestamp("2026-01-01"))
        for k, c in (("v2_2016_2025", c1), ("v2h_2026", c2)):
            chk.update({f"{k}:{x}": y for x, y in c.items()})
        chk.update({"n_sessions": int(df.session_date.nunique()), "n_minutes": len(df),
                    "first_session": str(df.session_date.min().date()), "last_session": str(df.session_date.max().date()),
                    "last_bar_end": str(df.timestamp.max()), "n_missing_minutes_kept": int(df.missing.sum()),
                    "n_bad_tick": int(df.bad_tick.sum()), "n_bad_range": int(df.bad_range.sum()),
                    "n_half_days": int(df.groupby("session_date")["is_half_day"].first().sum()),
                    "dropped_missing_frac": ";".join(str(d.date()) for d in info["dropped_missing_frac"]["session_date"]),
                    "dropped_not_in_v2_v2h": ";".join(str(d.date()) for d in info["dropped_not_in_v2"]),
                    "n_dividends_applied": len(info["dividends"])})
        pd.DataFrame([chk]).T.rename(columns={0: "value"}).to_csv(TAB / "data_check.csv")
        ok = all(chk[f"{k}:pass_returns"] and chk[f"{k}:pass_unadjusted"] for k in ("v2_2016_2025", "v2h_2026"))
        print("data check pass:", ok)
        if not ok:
            raise SystemExit("data check failed; stopping before any backtest (D29b (2))")
        cfg = frozen_config(hs) | {"stage": "B", "spread_sha1_stageB":
                                   hashlib.sha1((TAB / "spread_by_year.csv").read_bytes()).hexdigest()}
        cfg["half_spread_bp_by_cost_year"] = {str(k): v for k, v in hs.items()}
        (TAB / "stageB_config.json").write_text(json.dumps(cfg, indent=2, default=str))
        info_extra = {"config_hash_stageB": config_hash(cfg), "config_hash": "3d8abf0044",
                      "frozen_strategy_commit": "8163aa6"}
        return evaluate(df, hs, TAB, WINDOWS, log=True, info_extra=info_extra)
    print(__doc__)


if __name__ == "__main__":
    main()
