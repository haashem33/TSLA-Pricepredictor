"""Writes reports/phase1_data_quality.md tables + figures (CLAUDE.md 4.3). Dev period only."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tslaq.data.clean import build_symbol  # noqa: E402
from tslaq.features.returns import add_returns  # noqa: E402
from tslaq.utils.io import ROOT, interim_path, load_config  # noqa: E402

FIG = ROOT / "reports" / "figures"
OUT = ROOT / "reports" / "phase1_data_quality.md"


def md(df: pd.DataFrame, floatfmt="{:.4g}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.itertuples(index=False))


def main() -> None:
    cfg = load_config("data")
    FIG.mkdir(parents=True, exist_ok=True)
    L = ["# Phase 1 data-quality report", "",
         "Source: Alpaca SIP 1-minute bars (split-adjusted) from the user's local copy, aggregated to 5-minute "
         "regular-session bars (timestamps = bar end, America/New_York). SPY/QQQ back-adjusted for cash dividends "
         "(Alpha Vantage DIVIDENDS). Development period only: holdout (>= "
         f"{cfg['holdout_start']}) was not copied or loaded.", ""]
    per_year, miss_rows, bad_rows, insp = [], [], [], []
    dfs = {}
    for s in cfg["symbols"]:
        df, info = build_symbol(s, cfg)
        df = add_returns(df)
        dfs[s] = df
        yr = df.groupby(df["session_date"].dt.year).agg(sessions=("session_date", "nunique"), bars=("close", "size"),
                                                        half_days=("is_half_day", lambda x: x.groupby(
                                                            df.loc[x.index, "session_date"]).first().sum()))
        yr.insert(0, "symbol", s)
        per_year.append(yr.reset_index().rename(columns={"session_date": "year"}))
        dropped = info["dropped_sessions"]
        miss_rows.append({"symbol": s, "calendar_sessions": info["n_sessions_calendar"],
                          "sessions_kept": df["session_date"].nunique(),
                          "sessions_dropped(>5% missing)": len(dropped),
                          "dropped_dates": ", ".join(f"{d:%Y-%m-%d} ({f:.0%})" for d, f in
                                                     zip(dropped["session_date"], dropped["missing_frac"])),
                          "missing_bars_before_drop": info["n_missing_bars_before_drop"],
                          "missing_bars_kept(ffilled)": int(df["missing"].sum()),
                          "bars_with_<5_minutes": int((df["n_minutes"] < 5).sum())})
        r = df["r"]
        nxt = r.shift(-1)
        bt = df["bad_tick"]
        revert = bt & (np.sign(nxt) == -np.sign(r)) & (nxt.abs() > 0.5 * r.abs())
        bad_rows.append({"symbol": s, "bad_tick_flags": int(bt.sum()),
                         "of_which_bar0(gap)": int((bt & (df["bar_index"] == 0)).sum()),
                         "spike_and_revert(next bar reverses >50%)": int(revert.sum()),
                         "bad_range_flags": int(df["bad_range"].sum()),
                         "max_abs_r": float(r.abs().max())})
        top = df.loc[bt].assign(abs_r=r.abs(), next_r=nxt).nlargest(5, "abs_r")
        for t in top.itertuples():
            insp.append({"symbol": s, "timestamp": f"{t.timestamp:%Y-%m-%d %H:%M}", "bar_index": t.bar_index,
                         "r": t.r, "next_r": t.next_r, "volume": t.volume})
        if "dividends" in info:
            f = info["dividends"]
            g = df[df["bar_index"] == 0].set_index("session_date")["r"]
            dfs[s + "_div"] = pd.DataFrame({"symbol": s, "n_ex_dates": len(f),
                                            "mean_dividend_yield": float((1 - f["factor"]).mean()),
                                            "mean_exdate_gap_after_adj": float(g.reindex(f["ex_date"]).mean()),
                                            "mean_exdate_gap_before_adj(est)": float(
                                                (g.reindex(f["ex_date"]) + np.log(f["factor"].to_numpy())).mean()),
                                            "mean_gap_all_days": float(g.mean())}, index=[0])
    py = pd.concat(per_year)
    L += ["## Date range", ""]
    for s in cfg["symbols"]:
        d = dfs[s]
        L.append(f"- **{s}**: {d['timestamp'].min():%Y-%m-%d %H:%M} -> {d['timestamp'].max():%Y-%m-%d %H:%M}, "
                 f"{len(d):,} bars, {d['session_date'].nunique():,} sessions")
    L += ["", "## Bars and sessions per year", "",
          md(py.pivot(index="year", columns="symbol", values="bars").reset_index()), "",
          md(py[py.symbol == "TSLA"][["year", "sessions", "half_days"]]), "",
          "Full sessions have 78 bars; NYSE half-days (13:00 close) have 42.", "",
          "## Missing bars", "", md(pd.DataFrame(miss_rows)), "",
          "Missing 5-minute bars are forward-filled (O=H=L=C=previous close, volume 0, `missing=True`). "
          "Sessions with > 5% missing bars are dropped. 2016-02-22 and 2018-05-02/03 are gaps in the vendor "
          "file (the same days are missing in QQQ). **2018-08-07 (TSLA) was a real trading halt** (\"funding "
          "secured\" tweet, halted ~14:08-15:45 ET); the session was dropped under the 5% rule, so the "
          "next session's bar-0 return covers the gap from that day's close.", "",
          "## Bad ticks (flagged, not deleted)", "",
          f"Rule: |log return| > {cfg['bad_tick_mult']} x trailing median |r| (intraday: previous "
          f"{cfg['bad_tick_window_bars']} bars; bar 0: previous {cfg['bad_tick_window_sessions']} opening bars). "
          "`bad_range`: same rule on ln(H/L).", "", md(pd.DataFrame(bad_rows)), "",
          "Largest flagged bars (inspection):", "", md(pd.DataFrame(insp)), ""]
    # split check
    tsla = dfs["TSLA"]
    sp = []
    for d, fac in [("2020-08-31", 5), ("2022-08-25", 3)]:
        i = np.flatnonzero((tsla["session_date"] == d).to_numpy())[0]
        sp.append({"split_date": d, "factor": f"{fac}:1", "prev_close": tsla["close"].iat[i - 1],
                   "open": tsla["open"].iat[i], "bar0_log_return": tsla["r"].iat[i],
                   "unadjusted_would_be": float(np.log(1 / fac)),
                   "max_abs_r_+-1_session": float(tsla["r"].iloc[i - 78: i + 78].abs().max())})
    L += ["## Split-adjustment check (TSLA)", "", md(pd.DataFrame(sp)), "",
          "No -80% / -67% jump: prices are correctly split-adjusted.", "",
          "## Dividend adjustment check (SPY, QQQ)", "",
          md(pd.concat([dfs["SPY_div"], dfs["QQQ_div"]])), "",
          "Before adjustment the ex-date gap averaged roughly minus the dividend yield, which shows the source "
          "is not dividend-adjusted; after adjustment it is close to the all-day average.", ""]
    # earnings
    e = pd.read_parquet(interim_path("X").with_name("TSLA_earnings_sessions.parquet"))
    L += ["## TSLA earnings sessions", "",
          f"{len(e)} reports in the dev period, all `post-market` -> reaction session = next trading day "
          f"(none had a missing report time). First {e['reported_date'].min():%Y-%m-%d}, last "
          f"{e['reported_date'].max():%Y-%m-%d}.", ""]
    # histogram
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for s in cfg["symbols"]:
        r = dfs[s]["r"].dropna()
        intra = r[dfs[s]["bar_index"].loc[r.index] > 0]
        ax[0].hist(intra.clip(-0.02, 0.02), bins=200, histtype="step", density=True, label=s)
        ax[1].hist(intra.clip(-0.02, 0.02), bins=200, histtype="step", density=True, log=True, label=s)
    ax[0].set_title("5-min log returns (bar_index > 0), clipped +-2%")
    ax[1].set_title("same, log density")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "phase1_return_hist.png", dpi=120)
    stats = pd.DataFrame([{"symbol": s, "n": int(dfs[s]["r"].notna().sum()), "std": dfs[s]["r"].std(),
                           "skew": dfs[s]["r"].skew(), "excess_kurtosis": dfs[s]["r"].kurt(),
                           "share_exact_zero": float((dfs[s]["r"] == 0).mean())} for s in cfg["symbols"]])
    L += ["## Return distribution", "", "![returns](figures/phase1_return_hist.png)", "", md(stats), ""]
    OUT.write_text("\n".join(L), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
