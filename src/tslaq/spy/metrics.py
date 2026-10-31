"""Performance metrics for the D29 study (daily returns of a compounding account)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skew

# Paper monthly returns (%), FAQ Q24 table, dynamic Curr.Band + VWAP version; 2024 and 2025-01 post-publication.
PAPER_MONTHLY = {
    2016: [-1.0, -1.9, 1.3, -1.4, 0.4, -7.3, -3.1, -5.2, 8.4, -5.0, 1.4, 0.7],
    2017: [-3.3, 0.9, 1.7, -3.3, -2.4, 0.4, -1.8, 3.6, -1.1, 2.9, -2.2, -2.3],
    2018: [3.5, 8.9, 7.2, 3.5, -4.7, 2.0, 6.2, -1.7, -3.5, 13.4, 3.0, 12.5],
    2019: [0.9, -0.7, 2.9, 0.9, -5.1, -0.5, 1.4, 6.8, -1.4, 2.6, -1.7, 1.0],
    2020: [5.3, 1.9, -0.5, -1.8, -0.2, 8.0, 0.1, -1.8, 14.3, 2.8, -1.1, -1.9],
    2021: [7.8, 3.1, 0.7, 6.1, 2.6, -1.0, 2.8, 1.7, 2.8, 3.3, -3.2, 4.1],
    2022: [-5.5, 2.8, 0.2, 9.0, 2.1, 0.5, 0.2, 6.3, -1.0, 5.8, 1.6, 0.7],
    2023: [2.9, -1.3, 7.8, 1.8, 2.9, 4.1, 2.2, 6.0, 2.9, -1.1, 0.5, 3.8],
    2024: [8.8, -1.5, -0.4, 5.8, -4.3, 1.6, 8.2, -2.8, 4.1, 6.7, -2.6, 5.7],
}
PAPER_YEARLY = {2016: -12.8, 2017: -6.9, 2018: 61.1, 2019: 6.9, 2020: 26.8, 2021: 34.8, 2022: 24.4, 2023: 37.2,
                2024: 32.2}


def paper_monthly_series() -> pd.Series:
    idx, val = [], []
    for y, v in PAPER_MONTHLY.items():
        for k, x in enumerate(v):
            idx.append(pd.Period(f"{y}-{k + 1:02d}", "M"))
            val.append(x / 100)
    return pd.Series(val, index=pd.PeriodIndex(idx), name="paper")


def monthly(daily: pd.DataFrame, col: str = "ret") -> pd.Series:
    m = pd.to_datetime(daily["session_date"]).dt.to_period("M")
    return (1 + daily[col]).groupby(m).prod() - 1


def yearly(daily: pd.DataFrame, col: str = "ret") -> pd.Series:
    y = pd.to_datetime(daily["session_date"]).dt.year
    return (1 + daily[col]).groupby(y).prod() - 1


def max_drawdown(r: np.ndarray) -> float:
    eq = np.cumprod(1 + r)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    return float((1 - eq / peak).max())


def perf(r: pd.Series | np.ndarray) -> dict:
    r = np.asarray(r, float)
    n = len(r)
    sd = r.std(ddof=1)
    down = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    tot = float(np.prod(1 + r) - 1)
    return {"n_days": n, "total_return": tot, "cagr": float((1 + tot) ** (252 / n) - 1),
            "vol": float(sd * np.sqrt(252)), "sharpe": float(r.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
            "t_mean": float(r.mean() / sd * np.sqrt(n)) if sd > 0 else np.nan,
            "sortino": float(r.mean() / down * np.sqrt(252)) if down > 0 else np.nan,
            "mdd": max_drawdown(r), "skew": float(skew(r)), "worst_day": float(r.min()), "best_day": float(r.max())}


def alpha_beta(r: np.ndarray, bench: np.ndarray, lags: int = 5) -> dict:
    X = sm.add_constant(np.asarray(bench, float))
    f = sm.OLS(np.asarray(r, float), X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"alpha_ann": float(f.params[0] * 252), "alpha_t_nw": float(f.tvalues[0]),
            "beta": float(f.params[1]), "beta_t_nw": float(f.tvalues[1])}


def strategy_stats(daily: pd.DataFrame, trades: pd.DataFrame, bench: pd.Series) -> dict:
    """All D29 'reported, not criteria' numbers for one run over the rows of `daily`."""
    r = daily["ret"].to_numpy()
    out = perf(r)
    traded = daily["n_orders"] > 0
    out["hit_traded_days"] = float((daily.loc[traded, "ret"] > 0).mean()) if traded.any() else np.nan
    out["hit_all_days"] = float((daily["ret"] > 0).mean())
    out["share_days_traded"] = float(traded.mean())
    out.update(alpha_beta(r, bench.reindex(daily["session_date"]).to_numpy()))
    out["n_trades"] = len(trades)
    out["trades_per_day"] = len(trades) / len(daily)
    out["n_orders"] = int(daily["n_orders"].sum())
    if len(trades):
        out["avg_gross_per_share"] = float((trades["gross"] / trades["shares"]).mean())
        out["avg_net_per_share"] = float(((trades["gross"] - trades["cost"]) / trades["shares"]).mean())
        out["trade_hit"] = float(((trades["gross"] - trades["cost"]) > 0).mean())
        out["max_trade_loss_pct"] = float(((trades["gross"] - trades["cost"]) / (trades["shares"] * trades["entry"])).min())
        out["max_trade_gain_pct"] = float(((trades["gross"] - trades["cost"]) / (trades["shares"] * trades["entry"])).max())
    out["break_even_bp_per_side"] = float(daily["pnl_gross"].sum() / daily["traded_notional"].sum() * 1e4)
    out["cost_bp_per_side"] = float(daily["cost"].sum() / daily["traded_notional"].sum() * 1e4)
    out["turnover_per_day"] = float((daily["traded_notional"] / daily["aum_prev"]).mean())
    for leg in ("long", "short"):
        lr = (daily[f"{leg}_net"] / daily["aum_prev"]).to_numpy()
        out[f"{leg}_sum_ret"] = float(lr.sum())
        out[f"{leg}_sharpe"] = float(lr.mean() / lr.std(ddof=1) * np.sqrt(252))
        out[f"{leg}_gross_sum_ret"] = float((daily[f"{leg}_gross"] / daily["aum_prev"]).sum())
    return out
