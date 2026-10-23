"""Phase 2 follow-up diagnostics of the daily signal (user request 2026-10-23, items B1-B4).

B1  leak check of the daily-context features on real data: rebuild from raw 5-minute bars truncated at 20
    random intraday times; constancy within the session; independent re-computation from sessions <= D-1.
B2  where the effect happens: out-of-sample IC of d_sma20_dev, d_mom5, d_atr_c, amihud vs the session's
    open-to-close return, the same session's close-to-close return (C_{D-1} -> C_D) and the next session's
    (C_D -> C_{D+1}); by year and 2023-2025; per-time-slot IC profile at h = 12.
B3  TSLA-specific or market-wide: h = 12 IC against TSLA's residual return (minus beta x QQQ), and QQQ's own
    daily features against QQQ's h = 12 z.
B4  single-feature rule baselines (D16): d_sma20_dev one position per day (09:35 bar open -> session close),
    sign_first30 last half hour (Gao et al.). Sign from each fold's training IC; unit position. Each logged to
    the registry as a trial.

Out-of-sample convention (as in Phase 2): fold M = calendar month 2018-01 .. 2025-12; the sign of a feature in
month M is the sign of its mean monthly IC over the training months (all months before M, from 2016-02).
Raw feature versions are used throughout (the daily features are parameter-free).
Usage: python scripts/phase2_daily_diagnostics.py
"""
from __future__ import annotations

import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from tslaq.backtest.metrics import fold_ic_stats
from tslaq.data.pipeline import choose_lambda, load_bars
from tslaq.features.context import add_market, daily_context, event_flags
from tslaq.features.indicators import bar_features
from tslaq.features.returns import add_returns
from tslaq.features.volatility import add_sigma
from tslaq.labels.forward_returns import add_labels
from tslaq.utils.io import ROOT, config_hash, interim_path, load_config
from tslaq.utils.registry import log_run

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
TAB = ROOT / "reports" / "tables" / "phase2_daily"
FIG = ROOT / "reports" / "figures"
TAB.mkdir(parents=True, exist_ok=True)
FEATS = ["d_sma20_dev", "d_mom5", "d_atr_c", "amihud"]
DAILY = ["d_prev_ret_atr", "d_atr_c", "d_sma20_dev", "d_mom5", "har_rv1", "har_rv5", "har_rv22"]
EVENTS = ["earn_react", "earn_after", "opex", "half_day", "dow"]
OOS = pd.period_range("2018-01", "2025-12", freq="M").astype(str)
COSTS_BP = (0.0, 0.5, 1.0, 3.0, 6.0)


def base(sym, raw, lam, cfg, hz=(1, 3, 12)):
    v = cfg["volatility"]
    return add_sigma(add_returns(raw), lam, v["seasonal_window_sessions"], v["burn_in_sessions"], hz)


def spearman(x, y, min_obs):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < min_obs:
        return np.nan
    a, b = rankdata(x[ok]), rankdata(y[ok])
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def monthly(x, y, month, min_obs):
    return pd.Series({m: spearman(x[month == m], y[month == m], min_obs) for m in pd.unique(month)}).sort_index()


def train_sign(ic: pd.Series) -> pd.Series:
    """Per OOS month M: sign of the mean IC over months < M."""
    return pd.Series({M: np.sign(ic[ic.index < M].dropna().mean()) for M in OOS})


def oos_summary(ic: pd.Series, label: dict) -> list[dict]:
    sg = train_sign(ic)
    s = (sg * ic.reindex(OOS)).dropna()
    yr = pd.Index(s.index).str[:4].astype(int)
    rows = []
    for name, sel in [("2018-2025", np.ones(len(s), bool)), ("2023-2025", yr >= 2023)] + \
                     [(str(y), yr == y) for y in range(2018, 2026)]:
        st = fold_ic_stats(s[sel])
        rows.append({**label, "period": name, "ic": st["ic"], "t": st["t"], "pos_share": st["pos_share"],
                     "n_folds": st["n_folds"], "sign_mode": float(sg.mode().iloc[0])})
    full, rec = rows[0], rows[1]
    for r in rows:
        r["d10_pass"] = bool(full["t"] >= 2 and rec["ic"] > 0 and rec["t"] >= 1)
    return rows


# ---------------------------------------------------------------- B1
def leak_check(raw, lam, cfg, react, L, F):
    """Rebuild daily/event/amihud features from raw bars truncated at random intraday times."""
    cols = DAILY + ["gap_z", "rv_today_deseas", "amihud"] + EVENTS
    rng = np.random.default_rng(23)
    k = raw["bar_index"].to_numpy()
    cand = np.flatnonzero((raw["session_date"] >= pd.Timestamp("2018-01-01")).to_numpy() & (k > 0) & (k < 76))
    rows = []
    for i in np.sort(rng.choice(cand, 20, replace=False)):
        tr = base("TSLA", raw.iloc[: i + 1].reset_index(drop=True), lam, cfg)
        g = pd.concat([daily_context(tr), event_flags(tr, react), bar_features(tr)[["amihud"]]], axis=1)
        lo = max(0, i + 1 - 78 * 30)  # compare the last 30 sessions up to and including bar t
        worst = 0.0
        for c in cols:
            a, b = F[c].to_numpy()[lo: i + 1], g[c].to_numpy()[lo: i + 1]
            same_nan = np.array_equal(np.isnan(a), np.isnan(b))
            m = ~np.isnan(a)
            d = float(np.max(np.abs(a[m] - b[m]) / np.maximum(1.0, np.abs(a[m])))) if m.any() else 0.0
            worst = max(worst, d if same_nan else np.inf)
        rows.append({"t_row": int(i), "timestamp": str(raw["timestamp"].iloc[i]), "bar_index": int(k[i]),
                     "max_rel_diff_all_cols": worst, "identical": worst == 0.0})
    T = pd.DataFrame(rows)
    # constancy within the session
    const = F[DAILY + EVENTS].groupby(L["session_date"].to_numpy()).nunique(dropna=False).max()
    # independent re-computation from sessions <= D-1 (explicit loops, no pandas rolling/ewm)
    sess = L["session_date"].to_numpy()
    u = pd.unique(sess)
    first = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1]
    last = np.r_[first[1:] - 1, len(sess) - 1]
    O, H, Lo, C = (L[c].to_numpy() for c in ["open", "high", "low", "close"])
    dH = np.array([H[a: b + 1].max() for a, b in zip(first, last)])
    dL = np.array([Lo[a: b + 1].min() for a, b in zip(first, last)])
    dC = C[last]
    sample = rng.choice(np.arange(40, len(u)), 300, replace=False)
    diffs = {c: 0.0 for c in ["d_prev_ret_atr", "d_atr_c", "d_sma20_dev", "d_mom5"]}
    for j in sample:  # value for session u[j] from daily bars 0..j-1 only
        hi, lo_, cl = dH[:j], dL[:j], dC[:j]
        tr_ = np.empty(j)
        tr_[0] = hi[0] - lo_[0]
        for q in range(1, j):
            tr_[q] = max(hi[q] - lo_[q], abs(hi[q] - cl[q - 1]), abs(lo_[q] - cl[q - 1]))
        # Wilder ATR as pandas ewm(alpha=1/14, adjust=False, min_periods=14): seeded at the first TR
        a = tr_[0]
        for q in range(1, j):
            a = a + (tr_[q] - a) / 14
        ref = {"d_prev_ret_atr": (cl[-1] - cl[-2]) / a, "d_atr_c": a / cl[-1],
               "d_sma20_dev": cl[-1] / cl[-20:].mean() - 1, "d_mom5": np.log(cl[-1] / cl[-6])}
        row = first[j]
        for c, v in ref.items():
            diffs[c] = max(diffs[c], abs(F[c].iat[row] - v) / max(1.0, abs(v)))
    return T, const, diffs


# ---------------------------------------------------------------- B2/B3 daily table
def daily_table(L, F):
    sess = L["session_date"].to_numpy()
    first = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1]
    last = np.r_[first[1:] - 1, len(sess) - 1]
    seas2 = (L["seasonal"] ** 2).to_numpy()
    cs = np.r_[0, np.cumsum(np.nan_to_num(seas2))]
    lvl0 = np.sqrt(L["level2"].to_numpy()[first])  # EWMA level known before bar 0 (info <= D-1 close)
    S_cc = lvl0 * np.sqrt(cs[last + 1] - cs[first])       # bars 0..n-1 (bar 0 includes the gap)
    S_oc = lvl0 * np.sqrt(cs[last + 1] - cs[first + 1])   # bars 1..n-1 (bar-0 intraday part not in the scale)
    O, C = L["open"].to_numpy(), L["close"].to_numpy()
    prevC = np.r_[np.nan, C[last][:-1]]
    D = pd.DataFrame({"session": sess[first], "month": pd.Series(sess[first]).dt.to_period("M").astype(str)})
    D["oc"] = np.log(C[last] / O[first]) / S_oc
    D["cc"] = np.log(C[last] / prevC) / S_cc
    D["cc_next"] = np.r_[D["cc"].to_numpy()[1:], np.nan]
    for f in ["d_sma20_dev", "d_mom5", "d_atr_c"]:
        D[f] = F[f].to_numpy()[first]
    D["amihud"] = np.r_[np.nan, F["amihud"].to_numpy()[last][:-1]]  # previous session's last value
    return D


def slot_profile(L, F, month):
    """Per half-hour slot: monthly Spearman (pooled within month x slot) of the feature vs z_12, signed by
    the feature's training sign at h = 12 (pooled bar-level monthly IC)."""
    z = L["z_12"].to_numpy()
    k = L["bar_index"].to_numpy()
    slot = k // 6
    rows = []
    for f in FEATS:
        x = F[f].to_numpy()
        sg = train_sign(monthly(x, z, month, 200))
        for s in range(13):
            m = slot == s
            if not np.isfinite(z[m]).any():
                continue
            ic = monthly(x[m], z[m], month[m], 50)
            v = (sg * ic.reindex(OOS)).dropna()
            yr = pd.Index(v.index).str[:4].astype(int)
            a, b = fold_ic_stats(v), fold_ic_stats(v[yr >= 2023])
            rows.append({"feature": f, "slot": s, "start": f"{9 + (30 + 30 * s) // 60:02d}:{(30 + 30 * s) % 60:02d}",
                         "ic": a["ic"], "t": a["t"], "ic_2325": b["ic"], "t_2325": b["t"], "n_folds": a["n_folds"]})
    P = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for f in FEATS:
        p = P[P["feature"] == f]
        ax.plot(p["slot"], p["ic"], marker="o", label=f)
    ax.axhline(0, color="k", lw=0.8)
    p0 = P[P["feature"] == FEATS[0]]
    ax.set_xticks(p0["slot"].tolist())
    ax.set_xticklabels(p0["start"].tolist(), rotation=45)
    ax.set_ylabel("signed OOS IC vs z_12 (mean over 96 folds)")
    ax.set_title("Daily-signal IC by half-hour slot of the forecast bar (h = 12)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "phase2_daily_slot_ic_h12.png", dpi=110)
    plt.close(fig)
    return P


# ---------------------------------------------------------------- B4 rules
def rule_returns(L, F, month, rule: str, extra_delay: int = 0):
    """Per-session gross log return of a unit position and the position sign. Signal at the close of bar
    t is filled at the open of bar t+1 (+ extra_delay bars); exit at the close of the session's last bar."""
    sess = L["session_date"].to_numpy()
    first = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1]
    last = np.r_[first[1:] - 1, len(sess) - 1]
    O, C = L["open"].to_numpy(), L["close"].to_numpy()
    z = L["z_12"].to_numpy()
    if rule == "d_sma20_dev":
        x = F["d_sma20_dev"].to_numpy()
        sig_row = first            # known at the open; signal at the close of bar 0 -> fill at the 09:35 bar
        val = np.sign(x[sig_row])
    else:  # sign_first30, Gao et al.: enter at the open of the bar starting 30 min before the close
        x = F["sign_first30"].to_numpy()
        sig_row = last - 6         # close of the bar ending 30 min before the close (15:30 on full days)
        val = x[sig_row]
    entry = sig_row + 1 + extra_delay
    gross = np.log(C[last] / O[entry])
    sg = train_sign(monthly(x, z, month, 200))  # fold sign from the training-period h = 12 IC
    m = pd.Series(sess[first]).dt.to_period("M").astype(str).to_numpy()
    pos = sg.reindex(m).to_numpy() * val
    out = pd.DataFrame({"session": sess[first], "month": m, "pos": pos, "ret": pos * gross})
    return out[np.isin(m, OOS)].dropna().reset_index(drop=True), sg


def rule_metrics(R: pd.DataFrame) -> pd.DataFrame:
    yr = R["session"].dt.year
    rows = []
    for name, sel in [("2018-2025", yr >= 0), ("2023-2025", yr >= 2023)] + [(str(y), yr == y) for y in range(2018, 2026)]:
        r, p = R.loc[sel, "ret"].to_numpy(), np.abs(R.loc[sel, "pos"].to_numpy())
        row = {"period": name, "n_days": int(sel.sum()), "share_long": float((R.loc[sel, "pos"] > 0).mean()),
               "hit_rate": float((r[p > 0] > 0).mean()), "mean_gross_bp": r.mean() * 1e4,
               "breakeven_bp_per_side": r.mean() / (2 * p.mean()) * 1e4}
        for c in COSTS_BP:
            net = r - 2 * c / 1e4 * p
            sr = net.mean() / net.std()
            row[f"sharpe_ann_{c:g}bp"] = sr * np.sqrt(252)
        # D10 on per-fold (monthly) net returns at 3 bp per side
        rows.append(row)
    M = pd.DataFrame(rows)
    for c in (3.0, 6.0):
        net = R["ret"] - 2 * c / 1e4 * np.abs(R["pos"])
        mo = net.groupby(R["month"]).sum()
        a, b = fold_ic_stats(mo), fold_ic_stats(mo[mo.index.str[:4].astype(int) >= 2023])
        M[f"fold_t_full_{c:g}bp"] = a["t"]
        M[f"fold_mean_2325_{c:g}bp"] = b["ic"]
        M[f"fold_t_2325_{c:g}bp"] = b["t"]
        M[f"d10_pass_{c:g}bp"] = bool(a["t"] >= 2 and b["ic"] > 0 and b["t"] >= 1)
    return M


def main():
    cfg = load_config("data")
    wf = load_config("backtest")["walk_forward"]
    raw = load_bars("TSLA")
    lam, _ = choose_lambda(raw, wf["first_val_month"], cfg)
    L = add_labels(base("TSLA", raw, lam, cfg), (12,))
    qraw = load_bars("QQQ")
    lq, _ = choose_lambda(qraw, wf["first_val_month"], cfg)
    Q = add_labels(base("QQQ", qraw, lq, cfg), (12,))
    react = pd.read_parquet(interim_path("TSLA").with_name("TSLA_earnings_sessions.parquet"))["reaction_session"]
    bf = bar_features(L)
    F = pd.concat([daily_context(L), event_flags(L, react), bf[["amihud", "sign_first30"]],
                   add_market(L, {"qqq": Q})[["beta_qqq"]]], axis=1)
    month = L["session_date"].dt.to_period("M").astype(str).to_numpy()
    out = {}

    # B1
    T, const, diffs = leak_check(raw, lam, cfg, react, L, F)
    T.to_csv(TAB / "b1_truncation_rebuild.csv", index=False)
    out["b1"] = {"truncations": len(T), "all_identical": bool(T["identical"].all()),
                 "max_rel_diff": float(T["max_rel_diff_all_cols"].max()),
                 "max_distinct_values_per_session": const.to_dict(),
                 "independent_recompute_max_rel_diff": diffs}
    print("B1", json.dumps(out["b1"], indent=1, default=float), flush=True)

    # B2 daily targets
    D = daily_table(L, F)
    rows = []
    for f in FEATS:
        for tgt in ["oc", "cc", "cc_next"]:
            rows += oos_summary(monthly(D[f].to_numpy(), D[tgt].to_numpy(), D["month"].to_numpy(), 15),
                                {"feature": f, "target": tgt})
        rows += oos_summary(monthly(F[f].to_numpy(), L["z_12"].to_numpy(), month, 200),
                            {"feature": f, "target": "z12_bar"})
    B2 = pd.DataFrame(rows)
    B2.to_csv(TAB / "b2_daily_target_ic.csv", index=False)
    SP = slot_profile(L, F, month)
    SP.to_csv(TAB / "b2_slot_ic_h12.csv", index=False)

    # B3 residual and QQQ
    qk = Q[["session_date", "bar_index", "fwd_12", "z_12"]].rename(columns={"fwd_12": "q_fwd", "z_12": "q_z"})
    J = L[["session_date", "bar_index"]].merge(qk, on=["session_date", "bar_index"], how="left")
    resid = (L["fwd_12"].to_numpy() - F["beta_qqq"].to_numpy() * J["q_fwd"].to_numpy()) / L["S_12"].to_numpy()
    rows = []
    zt = L["z_12"].to_numpy()
    both = np.isfinite(resid)
    for f in FEATS:
        x = F[f].to_numpy()
        rows += oos_summary(monthly(np.where(both, x, np.nan), zt, month, 200), {"feature": f, "target": "tsla_z12 (same rows)"})
        rows += oos_summary(monthly(x, resid, month, 200), {"feature": f, "target": "tsla_resid_z12"})
        rows += oos_summary(monthly(x, J["q_z"].to_numpy(), month, 200), {"feature": f, "target": "qqq_z12 (TSLA feature)"})
    QF = pd.concat([daily_context(Q), bar_features(Q)[["amihud"]]], axis=1)
    qmonth = Q["session_date"].dt.to_period("M").astype(str).to_numpy()
    for f in FEATS:
        rows += oos_summary(monthly(QF[f].to_numpy(), Q["z_12"].to_numpy(), qmonth, 200),
                            {"feature": f"QQQ {f}", "target": "qqq_z12"})
    B3 = pd.DataFrame(rows)
    B3.to_csv(TAB / "b3_residual_qqq_ic.csv", index=False)

    # B4 rules
    bt = load_config("backtest")
    res = {}
    for rule in ["d_sma20_dev", "sign_first30"]:
        R, sg = rule_returns(L, F, month, rule)
        Rd, _ = rule_returns(L, F, month, rule, extra_delay=1)
        M, Md = rule_metrics(R), rule_metrics(Rd)
        M.to_csv(TAB / f"b4_rule_{rule}.csv", index=False)
        Md.to_csv(TAB / f"b4_rule_{rule}_delay1.csv", index=False)
        R.to_csv(TAB / f"b4_rule_{rule}_daily.csv", index=False)
        res[rule] = {"sign_counts": sg.value_counts().to_dict(), "metrics": M, "delay": Md}
        full = M.iloc[0]
        net3 = R["ret"] - 2 * 3e-4 * np.abs(R["pos"])
        net6 = R["ret"] - 2 * 6e-4 * np.abs(R["pos"])
        eq = net3.cumsum()
        down = net3[net3 < 0]
        log_run({"config_hash": config_hash({"rule": rule, "costs": bt["costs"]}), "phase": 2,
                 "model": f"rule_{rule}", "n_params": 0,
                 "fold_scheme": "expanding monthly WF 2018-01..2025-12; sign from training-month IC (h=12)",
                 "net_sharpe_period": net3.mean() / net3.std(), "net_sharpe_ann": full["sharpe_ann_3bp"],
                 "sortino": net3.mean() / down.std() * np.sqrt(252), "max_drawdown": float((eq - eq.cummax()).min()),
                 "turnover": float(2 * np.abs(R["pos"]).mean()), "sharpe_2x_costs": net6.mean() / net6.std() * np.sqrt(252),
                 "data_version": cfg["data_version"],
                 "notes": (f"D16 single-feature rule baseline (B4); daily returns; gross Sharpe {full['sharpe_ann_0bp']:.2f}; "
                           f"break-even {full['breakeven_bp_per_side']:.2f} bp/side; +1 bar delay net 3bp Sharpe "
                           f"{Md.iloc[0]['sharpe_ann_3bp']:.2f}")})
    pd.set_option("display.width", 250)
    print(B2[B2["period"].isin(["2018-2025", "2023-2025"])].round(4).to_string(index=False))
    print(SP.round(4).to_string(index=False))
    print(B3[B3["period"].isin(["2018-2025", "2023-2025"])].round(4).to_string(index=False))
    for rule, r in res.items():
        print(rule, r["sign_counts"])
        print(r["metrics"].round(3).to_string(index=False))
        print("delay+1:", r["delay"].iloc[:2].round(3).to_string(index=False))
    (TAB / "summary.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
