"""Placebo for the daily-signal IC (Phase 2 follow-up).

The 10.6 IC pools Spearman within each calendar month. A daily feature that behaves like a lagged price level
(d_sma20_dev, d_mom5, d_atr_c = ATR / C) is persistent within the month, and the month's own demeaning creates a
mechanical correlation with the month's returns (the same effect D1 found for the per-day IC one level down).
Here whole sessions are permuted (destroying any day-to-day predictability while keeping each session's
intraday path, gap, high and low relative to the previous close), prices are rebuilt, the daily features are
recomputed from the permuted closes, and the same unsigned IC is measured. Windows: calendar month (as in
10.6), calendar year, and the whole 2018-2025 period in one Spearman.
Also: the 2018 d_sma20_dev rule year (net Sharpe > 2) is checked for concentration and for a timing leak.
Usage: python scripts/phase2_daily_placebo.py
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from tslaq.data.pipeline import choose_lambda, load_bars
from tslaq.features.context import daily_context
from tslaq.features.returns import add_returns
from tslaq.features.volatility import add_sigma
from tslaq.labels.forward_returns import add_labels
from tslaq.utils.io import ROOT, load_config

warnings.filterwarnings("ignore")
TAB = ROOT / "reports" / "tables" / "phase2_daily"
N_PERM = 200
FEATS = ["d_sma20_dev", "d_mom5", "d_atr_c"]


def spearman(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 15:
        return np.nan
    a, b = rankdata(x[ok]), rankdata(y[ok])
    return float(np.corrcoef(a, b)[0, 1])


def daily_feats(c, h, low):
    """Same formulas as context.daily_context, on daily close/high/low arrays, shifted to 'known at open'."""
    d = pd.DataFrame({"close": c, "high": h, "low": low})
    pc = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - pc).abs(), (d["low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f = pd.DataFrame({"d_sma20_dev": d["close"] / d["close"].rolling(20).mean() - 1,
                      "d_mom5": np.log(d["close"] / d["close"].shift(5)), "d_atr_c": atr / d["close"]}).shift(1)
    return f


def window_ics(x, y, keys):
    """Mean of per-window Spearman over windows in `keys` (array of window labels)."""
    return np.nanmean([spearman(x[keys == k], y[keys == k]) for k in pd.unique(keys)])


def main():
    cfg = load_config("data")
    v = cfg["volatility"]
    raw = load_bars("TSLA")
    lam, _ = choose_lambda(raw, load_config("backtest")["walk_forward"]["first_val_month"], cfg)
    L = add_labels(add_sigma(add_returns(raw), lam, v["seasonal_window_sessions"], v["burn_in_sessions"], (12,)), (12,))
    sess = L["session_date"].to_numpy()
    first = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1]
    last = np.r_[first[1:] - 1, len(sess) - 1]
    O, H, Lo, C = (L[c].to_numpy() for c in ["open", "high", "low", "close"])
    dH = np.maximum.reduceat(H, first)
    dL = np.minimum.reduceat(Lo, first)
    dC = C[last]
    pc = np.r_[np.nan, dC[:-1]]
    # per-session relatives to the previous close (permutable units)
    rel = pd.DataFrame({"c": dC / pc, "h": dH / pc, "l": dL / pc})
    seas2 = np.r_[0, np.cumsum(np.nan_to_num((L["seasonal"] ** 2).to_numpy()))]
    lvl0 = np.sqrt(L["level2"].to_numpy()[first])
    S_cc = lvl0 * np.sqrt(seas2[last + 1] - seas2[first])
    cc = np.log(rel["c"].to_numpy()) / S_cc
    # real-data check of the identity with context.daily_context
    real = daily_feats(dC, dH, dL)
    ref = daily_context(L)[FEATS].to_numpy()[first]
    assert np.allclose(real.to_numpy(), ref, equal_nan=True, rtol=1e-12, atol=1e-15)
    # bar-level h = 12 z grouped by session
    z12 = L["z_12"].to_numpy()
    n_s = len(first)
    sd = pd.Series(sess[first])
    month = sd.dt.to_period("M").astype(str).to_numpy()
    year = sd.dt.year.to_numpy()
    oos = (sd >= pd.Timestamp("2018-01-01")).to_numpy()
    bar_sess = np.repeat(np.arange(n_s), last - first + 1)

    def measure(feat_df, cc_v, z_by_bar, bar_month, bar_oos, bar_sid):
        out = {}
        for f in FEATS:
            x = feat_df[f].to_numpy()
            for name, keys in [("month", month), ("year", year.astype(str)), ("all", np.full(n_s, "all"))]:
                out[(f, "cc", name)] = window_ics(x[oos], cc_v[oos], keys[oos])
            xb = x[bar_sid]
            out[(f, "z12_bar", "month")] = window_ics(xb[bar_oos], z_by_bar[bar_oos], bar_month[bar_oos])
        return out

    bar_month = month[bar_sess]
    bar_oos = oos[bar_sess]
    R = measure(real, cc, z12, bar_month, bar_oos, bar_sess)
    rng = np.random.default_rng(0)
    seg = [np.arange(a, b + 1) for a, b in zip(first, last)]
    P = []
    valid = np.flatnonzero(np.isfinite(rel["c"].to_numpy()))
    for i in range(N_PERM):
        perm = np.r_[0, rng.permutation(valid)]  # position 0 keeps the first session (no previous close)
        r = rel.to_numpy()[perm]
        c = np.r_[dC[0], dC[0] * np.cumprod(r[1:, 0])]
        cprev = np.r_[np.nan, c[:-1]]
        f = daily_feats(c, r[:, 1] * cprev, r[:, 2] * cprev)
        f.iloc[0] = np.nan
        ccp = cc[perm]
        # bar-level z of the permuted sessions, in the permuted order; months follow the position
        order = np.concatenate([seg[j] for j in perm])
        sid = np.repeat(np.arange(n_s), [len(seg[j]) for j in perm])
        P.append(measure(f, ccp, z12[order], month[sid], oos[sid], sid))
        if (i + 1) % 50 == 0:
            print(f"perm {i + 1}/{N_PERM}", flush=True)
    rows = []
    for key in R:
        pv = np.array([p[key] for p in P])
        rows.append({"feature": key[0], "target": key[1], "window": key[2], "real_ic_unsigned": R[key],
                     "placebo_mean": pv.mean(), "placebo_sd": pv.std(), "placebo_p05": np.quantile(pv, 0.05),
                     "placebo_p95": np.quantile(pv, 0.95), "real_minus_placebo": R[key] - pv.mean(),
                     "z_vs_placebo": (R[key] - pv.mean()) / pv.std(),
                     "share_placebo_as_extreme": float(np.mean(np.abs(pv - pv.mean()) >= abs(R[key] - pv.mean())))})
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "placebo_session_permutation.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string(index=False))

    # 2018 rule year (net Sharpe > 2): concentration and timing-leak check
    D = pd.read_csv(TAB / "b4_rule_d_sma20_dev_daily.csv", parse_dates=["session"])
    y = D[D["session"].dt.year == 2018].copy()
    net = y["ret"] - 6e-4 * y["pos"].abs()
    srt = np.sort(net.to_numpy())[::-1]
    chk = {"n_days": len(y), "net_sharpe_ann": float(net.mean() / net.std() * np.sqrt(252)),
           "net_sharpe_ann_without_top5": float(srt[5:].mean() / srt[5:].std() * np.sqrt(252)),
           "net_sharpe_ann_without_top10": float(srt[10:].mean() / srt[10:].std() * np.sqrt(252)),
           "share_of_net_from_top5_days": float(srt[:5].sum() / srt.sum()),
           "daily_sd_bp": float(y["ret"].std() * 1e4)}
    # timing leak check: same rule with the signal lagged one more session (d_sma20_dev known at D-1's open)
    x = real["d_sma20_dev"].to_numpy()
    xs = pd.Series(np.r_[np.nan, x[:-1]], index=sd)
    entry = first + 1
    gross = pd.Series(np.log(C[last] / O[entry]), index=sd)
    for lag_name, xx in [("signal_D", pd.Series(x, index=sd)), ("signal_lag1", xs)]:
        g = (-np.sign(xx) * gross)[sd.dt.year.to_numpy() == 2018]
        chk[f"gross_sharpe_2018_{lag_name}"] = float(g.mean() / g.std() * np.sqrt(252))
        gg = (-np.sign(xx) * gross)[oos]
        chk[f"gross_sharpe_2018_2025_{lag_name}"] = float(gg.mean() / gg.std() * np.sqrt(252))
    print(json.dumps(chk, indent=1))
    (TAB / "rule_2018_check.json").write_text(json.dumps(chk, indent=1))


if __name__ == "__main__":
    main()
