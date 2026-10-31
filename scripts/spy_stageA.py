"""D29 Stage A: SPY intraday momentum replication period P1 = 2016-01..2024-02 only (D29, D29a).

Steps: (1) SPY 1-minute frame + data check; (2) SPY EDGE spread table 2016-2023 and the D29a-2 floor; (3) backtests
of the primary and both secondary configurations at gross / C_paper / C_real / 2x C_real, E0 / E1; (4) P1 metrics,
buy-and-hold, the D29 tolerance check against the paper's monthly table; (5) 3 registry trials.
No bar after 2024-02-29 is loaded (tslaq.spy.data.STAGE_A_END).

Usage: python scripts/spy_stageA.py [--no-registry]
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np
import pandas as pd
from bidask import edge

from tslaq.backtest.dsr import dsr
from tslaq.data.download import month_range
from tslaq.spy import metrics as mt
from tslaq.spy import momentum as mo
from tslaq.spy.data import STAGE_A_END, build_spy_1min, data_check, load_raw_1min
from tslaq.utils.io import ROOT, config_hash, load_config
from tslaq.utils.registry import log_run, n_trials

TAB = ROOT / "reports" / "tables" / "spy_stageA"
INTERIM = ROOT / "data" / "interim" / "spy1m"
P1 = ("2016-01-01", "2024-02-29")
TOL = {"corr_min": 0.80, "sharpe_ref": 1.37, "sharpe_tol": 0.30, "cagr_ref": 0.201, "cagr_tol": 0.04,
       "sign_years_min": 7}


def spread_table() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """D25 (ii) EDGE procedure on SPY source 1-minute bars (unadjusted), 2016-2023; D29a-2 half-tick comparison."""
    rows = []
    for m in month_range("2016-01", "2024-01"):
        d = load_raw_1min([m])
        ts = d["timestamp"].dt.tz_convert("America/New_York")
        mins = ts.dt.hour * 60 + ts.dt.minute
        keep = (mins >= 570) & (mins < 960)
        d = d[keep].assign(session=ts.dt.date[keep])
        for s, g in d.groupby("session"):
            g = g.sort_values("timestamp")
            if len(g) < 30:
                continue
            o, h, lo, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
            rows.append({"session": s, "n_bars": len(g), "edge": float(edge(o, h, lo, c)),
                         "price": float(np.median(c))})
    D = pd.DataFrame(rows)
    D["year"] = pd.to_datetime(D["session"]).dt.year
    Y = D.groupby("year").agg(n_sessions=("session", "size"), median_price=("price", "median"),
                              edge_mean=("edge", "mean"), edge_median=("edge", "median"),
                              edge_nan=("edge", lambda x: int(x.isna().sum())),
                              edge_negative=("edge", lambda x: int((x < 0).sum()))).reset_index()
    Y["edge_mean_full_bp"] = Y["edge_mean"] * 1e4
    Y["edge_mean_half_bp"] = Y["edge_mean"] * 1e4 / 2
    Y["edge_median_half_bp"] = Y["edge_median"] * 1e4 / 2
    Y["half_tick_bp_at_median_price"] = mo.HALF_TICK / Y["median_price"] * 1e4
    Y["binding_at_median_price"] = np.where(Y["half_tick_bp_at_median_price"] > Y["edge_mean_half_bp"],
                                            "half_tick", "edge")
    Y = Y.drop(columns=["edge_mean", "edge_median"])
    # cost year Y uses EDGE of Y-1 (2016 its own year)
    e = dict(zip(Y["year"], Y["edge_mean_half_bp"]))
    hs = {y: e[y - 1] if (y - 1) in e else e[y] for y in range(2016, 2025)}
    return Y, hs, D


def run_all(ind: pd.DataFrame, hs: dict, start: pd.Timestamp):
    res = {}
    for cfg in mo.CONFIGS:
        for delay in (0, 1):
            plans = mo.day_plans(ind, cfg, delay=delay)
            for cost in mo.COSTS:
                res[(cfg, cost, delay)] = mo.simulate(plans, cfg, cost, hs, start=start)
    return res


def frozen_config(hs: dict) -> dict:
    spread_sha = hashlib.sha1((TAB / "spread_by_year.csv").read_bytes()).hexdigest()
    return {"study": "D29/D29a", "lookback": mo.LOOKBACK, "vm": mo.VM, "target_vol": mo.TARGET_VOL,
            "max_lev": mo.MAX_LEV, "decisions": list(mo.DECISION_MINUTES), "commission": mo.COMMISSION,
            "min_commission": mo.MIN_COMMISSION, "paper_slippage": mo.PAPER_SLIPPAGE, "half_tick": mo.HALF_TICK,
            "sec_bp": mo.SEC_BP, "aum0": 100_000, "data": load_config("data"), "spread_sha1": spread_sha,
            "half_spread_bp_by_cost_year": {str(k): v for k, v in hs.items()}}


def main(log: bool = True):
    TAB.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)
    # (1) data
    df, info = build_spy_1min(STAGE_A_END)
    assert df["session_date"].max() <= pd.Timestamp(P1[1])
    tab_div, chk = data_check(df)
    chk.update({"n_missing_minutes_before_drop": info["n_missing_minutes"],
                "dropped_sessions_missing_frac": ";".join(str(d.date()) for d in info["dropped_missing_frac"]["session_date"]),
                "dropped_sessions_not_in_v2": ";".join(str(d.date()) for d in info["dropped_not_in_v2"]),
                "n_sessions": int(df["session_date"].nunique()), "n_minutes": len(df),
                "n_missing_minutes_kept": int(df["missing"].sum()), "n_bad_tick": int(df["bad_tick"].sum()),
                "n_bad_range": int(df["bad_range"].sum()), "n_half_days": int(df.groupby("session_date")["is_half_day"].first().sum()),
                "first_session": str(df["session_date"].min().date()), "last_session": str(df["session_date"].max().date())})
    pd.DataFrame([chk]).T.rename(columns={0: "value"}).to_csv(TAB / "data_check.csv")
    tab_div.to_csv(TAB / "data_check_dividends.csv", index=False)
    print("data check:", {k: chk[k] for k in ("pass_returns", "pass_unadjusted", "logret_max_abs_diff")})
    if not (chk["pass_returns"] and chk["pass_unadjusted"]):
        raise SystemExit("data check failed; stopping before any backtest")
    df.to_parquet(INTERIM / "SPY_1min_stageA.parquet", index=False)
    # (2) spreads
    Y, hs, D = spread_table()
    Y.to_csv(TAB / "spread_by_year.csv", index=False)
    D.to_csv(TAB / "spread_daily.csv", index=False)
    print(Y.round(3).to_string(index=False))
    # (3) backtests
    ind = mo.add_indicators(df)
    plans0 = mo.day_plans(ind, "primary")
    start = next(p.session_date for p in plans0 if p.tradable)
    res = run_all(ind, hs, start)
    daily_close = df.groupby("session_date")["close"].last()
    bench = (daily_close / daily_close.shift(1) - 1)
    # (4) metrics
    rows, months, years = [], {}, {}
    for (cfg, cost, delay), (daily, trades) in res.items():
        daily.to_parquet(INTERIM / f"daily_{cfg}_{cost}_E{delay}.parquet", index=False)
        s = mt.strategy_stats(daily, trades, bench)
        rows.append({"config": cfg, "cost": cost, "exec": f"E{delay}", **s})
        months[(cfg, cost, delay)] = mt.monthly(daily)
        years[(cfg, cost, delay)] = mt.yearly(daily)
    S = pd.DataFrame(rows)
    b = bench[bench.index >= start]
    bh = {"config": "spy_buy_hold", "cost": "none", "exec": "close-to-close", **mt.perf(b.to_numpy())}
    S = pd.concat([S, pd.DataFrame([bh])], ignore_index=True)
    S.to_csv(TAB / "p1_summary.csv", index=False)
    # yearly / monthly tables
    bench_df = pd.DataFrame({"session_date": b.index, "ret": b.to_numpy()})
    Yt = pd.DataFrame({f"{c}|{k}|E{d}": v for (c, k, d), v in years.items()})
    Yt["spy_buy_hold"] = mt.yearly(bench_df)
    Yt["paper_dynamic"] = pd.Series(mt.PAPER_YEARLY)
    Yt.index.name = "year"
    Yt.to_csv(TAB / "p1_yearly.csv")
    Mt = pd.DataFrame({f"{c}|{k}|E{d}": v for (c, k, d), v in months.items()})
    Mt["spy_buy_hold"] = mt.monthly(bench_df)
    Mt.index.name = "month"
    Mt.to_csv(TAB / "p1_monthly_all.csv")
    # tolerance (primary, C_paper, E0) against the paper's monthly table, months 2016-02..2024-02
    ours = months[("primary", "paper", 0)]
    paper = mt.paper_monthly_series()
    rng = pd.period_range("2016-02", "2024-02", freq="M")
    C = pd.DataFrame({"ours": ours.reindex(rng), "paper": paper.reindex(rng)})
    C["diff"] = C["ours"] - C["paper"]
    C.index.name = "month"
    C.to_csv(TAB / "p1_monthly_comparison.csv")

    def mstats(x):
        return float(x.mean() / x.std(ddof=1) * np.sqrt(12)), float(np.prod(1 + x) ** (12 / len(x)) - 1), \
            float(x.std(ddof=1) * np.sqrt(12))
    sh_o, cg_o, vol_o = mstats(C["ours"])
    sh_p, cg_p, vol_p = mstats(C["paper"])
    corr = float(C["ours"].corr(C["paper"]))
    ysign = []
    for y in range(2016, 2024):
        mm = C[C.index.year == y]
        yo, yp = float(np.prod(1 + mm["ours"]) - 1), float(np.prod(1 + mm["paper"]) - 1)
        ysign.append({"year": y, "ours": yo, "paper": yp, "same_sign": bool(np.sign(yo) == np.sign(yp))})
    YS = pd.DataFrame(ysign)
    YS.to_csv(TAB / "p1_yearly_comparison.csv", index=False)
    n_sign = int(YS["same_sign"].sum())
    verdict = {"n_months": len(C), "corr_monthly": corr, "sharpe_monthly_ours": sh_o, "sharpe_monthly_paper": sh_p,
               "cagr_ours": cg_o, "cagr_paper": cg_p, "vol_ours": vol_o, "vol_paper": vol_p,
               "years_same_sign": n_sign,
               "ok_corr": corr >= TOL["corr_min"], "ok_sharpe": abs(sh_o - TOL["sharpe_ref"]) <= TOL["sharpe_tol"],
               "ok_cagr": abs(cg_o - TOL["cagr_ref"]) <= TOL["cagr_tol"], "ok_sign": n_sign >= TOL["sign_years_min"]}
    verdict["within_range"] = bool(verdict["ok_corr"] and verdict["ok_sharpe"] and verdict["ok_cagr"]
                                   and verdict["ok_sign"])
    # same comparison for every cost / execution of the primary config (information)
    extra = []
    for cost in mo.COSTS:
        for d in (0, 1):
            x = months[("primary", cost, d)].reindex(rng)
            s_, c_, _ = mstats(x)
            extra.append({"cost": cost, "exec": f"E{d}", "corr_monthly": float(x.corr(C["paper"])),
                          "sharpe_monthly": s_, "cagr": c_})
    pd.DataFrame(extra).to_csv(TAB / "p1_comparison_by_cost.csv", index=False)
    # DSR, information only (D29 defines it on P2; here V = variance of the 3 configs' P1 daily Sharpe at C_real E0)
    sr = {c: float(res[(c, "real", 0)][0]["ret"].mean() / res[(c, "real", 0)][0]["ret"].std(ddof=1))
          for c in mo.CONFIGS}
    V = float(np.var(list(sr.values()), ddof=1))
    N_after = n_trials() + (3 if log else 0)
    dsr_rows = [{"config": c, **dsr(res[(c, "real", 0)][0]["ret"].to_numpy(), N_after, V)} for c in mo.CONFIGS]
    pd.DataFrame(dsr_rows).to_csv(TAB / "p1_dsr_info.csv", index=False)
    cfg = frozen_config(hs)
    ch = config_hash(cfg)
    verdict["config_hash"] = ch
    verdict["start_session"] = str(start.date())
    (TAB / "p1_verdict.json").write_text(json.dumps(verdict, indent=2, default=str))
    (TAB / "frozen_config.json").write_text(json.dumps(cfg, indent=2, default=str))
    print(json.dumps(verdict, indent=2, default=str))
    # (5) registry: 3 trials
    if log:
        for c in mo.CONFIGS:
            r_real = S[(S.config == c) & (S.cost == "real") & (S.exec == "E0")].iloc[0]
            r_2x = S[(S.config == c) & (S.cost == "real2x") & (S.exec == "E0")].iloc[0]
            log_run({"config_hash": ch, "phase": "spy_A", "model": f"d29_spy_momentum_{c}", "n_params": "",
                     "fold_scheme": f"D29 Stage A, P1 {start.date()}..{P1[1]}; fill next open (E0); C_real (D29a-2)",
                     "hit_rate_h12": r_real["hit_traded_days"], "net_sharpe_period": sr[c],
                     "net_sharpe_ann": r_real["sharpe"], "sortino": r_real["sortino"],
                     "max_drawdown": r_real["mdd"], "turnover": r_real["turnover_per_day"],
                     "sharpe_2x_costs": r_2x["sharpe"], "data_version": "v2 (SPY 1-min)",
                     "notes": f"SPY intraday momentum (Zarattini/Aziz/Barbon 2024), config {c}; P1 only; "
                              f"break-even {r_real['break_even_bp_per_side']:.3f} bp/side"})
        print("registry: n_trials_so_far =", n_trials())


if __name__ == "__main__":
    main(log="--no-registry" not in sys.argv)
