"""Early, pre-registered Phase 5 test (decisions D26, D27) on stored out-of-sample h = 12 predictions.

Candidates: M3 and M1 (D25, data/processed/phase3_baselines/real/), Gate-3 B (D23, data/processed/phase3/real/).
Rules R1 (proportional, band 0.25) and R2 (strong signal, 12-bar hold), q = 90th percentile of |zhat| on the D6
block (fold 2018-01) or the previous out-of-sample month. Costs: D26 (prior-year EDGE half-spread + 0.10 bp),
2x D26 and fixed 0 / 0.5 / 1 / 3 / 6 bp per side. No model is retrained (CPCV: scripts/run_phase5_cpcv.py).
Holdout never loaded (the bar files end 2025-12-31).

Usage: python scripts/run_phase5_early.py [--log]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_phase3_gate3 as rp3

from tslaq.backtest import positions as bp
from tslaq.backtest.dsr import dsr
from tslaq.models import baselines_d25 as b
from tslaq.models import gate3 as g3
from tslaq.utils.io import ROOT, config_hash, load_config
from tslaq.utils.registry import REGISTRY, log_run, n_trials

warnings.filterwarnings("ignore")
H = 12
TAB = ROOT / "reports" / "tables" / "phase5_early"
D25 = ROOT / "data" / "processed" / "phase3_baselines" / "real"
SPREAD = ROOT / "reports" / "tables" / "phase3_baselines" / "spread_by_year.csv"
SEC_FEE_BP = 0.10
Q_PCT = 90.0
BAND = 0.25
FIXED_BP = (0.0, 0.5, 1.0, 3.0, 6.0)
CANDIDATES = ("M3", "Gate-3 B", "M1")
RULES = ("R1", "R2")
PERIODS = (("2018-2025", 2018), ("2023-2025", 2023))


def d26_cost_by_year(own_year_first: bool = False) -> dict:
    """D26: cost per side (bp) in year Y = EDGE mean half-spread of Y-1 + 0.10 bp. With own_year_first the first
    table year uses its own value (CPCV only; no prior year exists)."""
    S = pd.read_csv(SPREAD).set_index("year")["edge_mean_half_bp"]
    out = {int(y) + 1: float(v) + SEC_FEE_BP for y, v in S.items()}
    if own_year_first:
        out[int(S.index.min())] = float(S.iloc[0]) + SEC_FEE_BP
    return out


# ------------------------------------------------------------------ predictions and thresholds

def m3_inner(fr: dict) -> pd.DataFrame:
    """M3 has no stored D6 block: its score on 2017-11..12 with anchors / signs from sessions < 2017-11 and fold
    2018-01's feature list (the D6 analogue of the models' inner predictions; nothing is fitted but anchors)."""
    M = g3.MONTHS[0]
    feats = [(f, v) for f, v, k in b.by_fold_basis()[M] if k in ("IC", "both")]
    s, _, _ = b.composite(fr, g3.fold_ff(fr, M), feats, g3.INNER[0], H)
    _, te = g3.split(fr, H, g3.INNER)
    return pd.DataFrame({"fold": "inner", "pos": te, "pred": s[te]})


def load_preds(fr: dict) -> dict:
    P = []
    for M in g3.MONTHS:
        with open(D25 / f"{M}.pkl", "rb") as fh:
            P.append(pickle.load(fh)["preds"])
    P = pd.concat(P, ignore_index=True)
    P = P[P["h"] == H]
    GB = []
    for M in g3.MONTHS:
        with open(rp3.REAL / f"{M}.pkl", "rb") as fh:
            GB.append(pickle.load(fh)["preds"])
    GB = pd.concat(GB, ignore_index=True)
    out = {"M1": P[P["model"] == "M1"], "M3": P[P["model"] == "M3"],
           "Gate-3 B": GB[(GB["variant"] == "B") & (GB["h"] == H)]}
    out = {k: v[["fold", "pos", "pred"]].reset_index(drop=True) for k, v in out.items()}
    if not (out["M3"]["fold"] == "inner").any():
        out["M3"] = pd.concat([m3_inner(fr), out["M3"]], ignore_index=True)
    return out


def thresholds(P: pd.DataFrame) -> dict:
    """q per fold: D6 block for the first fold, else the previous fold's out-of-sample month."""
    a = lambda g: float(np.nanpercentile(np.abs(g["pred"].to_numpy()), Q_PCT))
    q = {g3.MONTHS[0]: a(P[P["fold"] == "inner"])}
    for prev, M in zip(g3.MONTHS[:-1], g3.MONTHS[1:]):
        q[M] = a(P[P["fold"] == prev])
    return q


# ------------------------------------------------------------------ one candidate x rule x delay

def simulate(L: pd.DataFrame, sid, last, P: pd.DataFrame, q: dict, rule: str, d: int) -> dict:
    """Daily gross P&L, daily turnover, number of fills / trades per day and per-trade unit returns."""
    O, C = L["open"].to_numpy(), L["close"].to_numpy()
    Q = P[P["fold"] != "inner"]
    pos, z = Q["pos"].to_numpy(), Q["pred"].to_numpy()
    qq = Q["fold"].map(q).to_numpy(float)
    days = np.unique(sid[pos])
    if rule == "R1":
        F = bp.r1_fills(sid, last, pos, z, qq, d, BAND)
        R = bp.r1_pnl(O, C, sid, last, F)
        on = np.isin(sid, days)
        gross = pd.Series(R["gross"][on]).groupby(sid[on]).sum()
        turn = pd.Series(R["dw"][on]).groupby(sid[on]).sum()
        nfill = pd.Series(1.0, index=sid[F["bar"].to_numpy()]).groupby(level=0).sum()
        unit, usid = R["seg_unit"], R["seg_sid"]
    else:
        T = bp.r2_trades(O, C, sid, last, pos, z, qq, d, H)
        gross = T.groupby("sid")["gross"].sum()
        turn = 2.0 * T.groupby("sid").size().astype(float)
        nfill = T.groupby("sid").size().astype(float)
        unit, usid = T["gross"].to_numpy(), T["sid"].to_numpy()
    idx = pd.Index(days)
    return {"gross": gross.reindex(idx, fill_value=0.0), "turn": turn.reindex(idx, fill_value=0.0),
            "ntr": nfill.reindex(idx, fill_value=0.0), "unit": unit, "usid": usid}


def perf(net: pd.Series) -> dict:
    sd = net.std()
    dd = np.sqrt(np.mean(np.minimum(net.to_numpy(), 0.0) ** 2))
    cum = net.cumsum()
    return {"sharpe_ann": float(net.mean() / sd * np.sqrt(252)), "sharpe_day": float(net.mean() / sd),
            "t": float(net.mean() / sd * np.sqrt(len(net))), "sortino_ann": float(net.mean() / dd * np.sqrt(252)),
            "max_dd_pct": float((cum.cummax() - cum).max() * 100), "ann_ret_pct": float(net.mean() * 252 * 100)}


def table_rows(name: str, rule: str, d: int, S: dict, year: np.ndarray, cost_y: dict) -> tuple[list, dict]:
    rows, nets = [], {}
    for per, lo in PERIODS:
        m = year >= lo
        g, tv, n = S["gross"][m], S["turn"][m], S["ntr"][m]
        um = np.isin(S["usid"], S["gross"].index[m])
        u = S["unit"][um]
        base = {"candidate": name, "rule": rule, "extra_delay": d, "period": per, "n_days": int(m.sum()),
                "trades_per_day": float(n.mean()), "turnover_per_day": float(tv.mean()),
                "n_trades": len(u), "hit_rate": float(np.mean(u > 0)) if len(u) else np.nan,
                "mean_bp_per_trade": float(np.mean(u) * 1e4) if len(u) else np.nan,
                "break_even_bp_per_side": float(g.mean() / tv.mean() * 1e4) if tv.mean() > 0 else np.nan,
                "exposure_days_share": float(np.mean(tv > 0))}
        c26 = np.array([cost_y[y] for y in year[m]])
        costs = {"D26": c26, "2xD26": 2 * c26, **{("gross" if c == 0 else f"{c:g}bp"): np.full(m.sum(), c)
                                                   for c in FIXED_BP}}
        for tag, c in costs.items():
            net = g - c / 1e4 * tv.to_numpy()
            rows.append(base | {"cost": tag, "mean_cost_bp_per_side": float(np.mean(c))} | perf(net))
            nets[(per, tag)] = net
    return rows, nets


# ------------------------------------------------------------------ main

def registry_rule_sharpes() -> pd.DataFrame:
    R = pd.read_csv(REGISTRY)
    R = R[R["rerun_of"].isna() & (R["model"].str.contains("rule") | R["model"].str.startswith("d27_"))
          & R["net_sharpe_period"].notna()]
    return R[["model", "net_sharpe_period"]]


def main(log: bool):
    TAB.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fr = g3.real_frame()
    L = fr["L"]
    assert L["session_date"].max() < pd.Timestamp("2026-01-01"), "holdout must stay sealed"
    sid, last = bp.session_arrays(fr["sess"])
    sess_by_sid = pd.Series(fr["sess"]).groupby(sid).first()
    preds = load_preds(fr)
    print(f"frame + predictions: {time.time() - t0:.0f}s", flush=True)
    cost_y = d26_cost_by_year()
    rows, NETS, Q = [], {}, []
    for name in CANDIDATES:
        P = preds[name]
        q = thresholds(P)
        Q += [{"candidate": name, "fold": M, "q": v} for M, v in q.items()]
        for rule in RULES:
            for d in (0, 1):
                t1 = time.time()
                S = simulate(L, sid, last, P, q, rule, d)
                year = pd.DatetimeIndex(sess_by_sid.loc[S["gross"].index]).year.to_numpy()
                r, nets = table_rows(name, rule, d, S, year, cost_y)
                rows += r
                NETS[(name, rule, d)] = (nets, pd.DatetimeIndex(sess_by_sid.loc[S["gross"].index]))
                print(f"{name} {rule} d={d}: {time.time() - t1:.0f}s", flush=True)
    BT = pd.DataFrame(rows)
    BT.to_csv(TAB / "backtest_results.csv", index=False)
    pd.DataFrame(Q).to_csv(TAB / "thresholds_q_by_fold.csv", index=False)
    # daily net P&L at D26 (no extra delay), full period, for DSR and CPCV reference
    daily = pd.DataFrame({f"{n}|{r}": NETS[(n, r, 0)][0][("2018-2025", "D26")].to_numpy()
                          for n in CANDIDATES for r in RULES}, index=NETS[(CANDIDATES[0], "R1", 0)][1])
    daily.index.name = "session"
    daily.to_csv(TAB / "daily_net_d26.csv")

    def get(n, r, d, per, cost, col="sharpe_ann"):
        x = BT[(BT["candidate"] == n) & (BT["rule"] == r) & (BT["extra_delay"] == d) & (BT["period"] == per)
               & (BT["cost"] == cost)]
        return float(x[col].iloc[0])

    # ---- registry (6 trials), then DSR with N = n_trials_so_far after logging
    dv = load_config("data")["data_version"]
    if log:
        for n in CANDIDATES:
            for r in RULES:
                cfg = {"d27": r, "candidate": n, "h": H, "q_pct": Q_PCT, "band": BAND if r == "R1" else None,
                       "costs": "D26", "sec_fee_bp": SEC_FEE_BP}
                log_run({"config_hash": config_hash(cfg), "phase": 5,
                         "model": f"d27_{r}_{n.replace(' ', '').replace('-', '')}_h12",
                         "fold_scheme": "stored out-of-sample h=12 predictions 2018-01..2025-12; fill next open; D26 costs",
                         "net_sharpe_period": get(n, r, 0, "2018-2025", "D26", "sharpe_day"),
                         "net_sharpe_ann": get(n, r, 0, "2018-2025", "D26"),
                         "sortino": get(n, r, 0, "2018-2025", "D26", "sortino_ann"),
                         "max_drawdown": get(n, r, 0, "2018-2025", "D26", "max_dd_pct"),
                         "turnover": get(n, r, 0, "2018-2025", "D26", "turnover_per_day"),
                         "sharpe_2x_costs": get(n, r, 0, "2018-2025", "2xD26"), "data_version": dv,
                         "notes": (f"D27 early Phase 5; break-even {get(n, r, 0, '2018-2025', 'D26', 'break_even_bp_per_side'):.2f} bp/side; "
                                   f"gross Sharpe {get(n, r, 0, '2018-2025', 'gross'):.2f}; 2023-25 net D26 "
                                   f"{get(n, r, 0, '2023-2025', 'D26'):.2f}; +1 bar delay net D26 "
                                   f"{get(n, r, 1, '2018-2025', 'D26'):.2f}")})
    N = n_trials()
    RS = registry_rule_sharpes()
    V = float(RS["net_sharpe_period"].var(ddof=1))
    V27 = float(RS[RS["model"].str.startswith("d27_")]["net_sharpe_period"].var(ddof=1))
    D, verdict = [], []
    for n in CANDIDATES:
        for r in RULES:
            x = daily[f"{n}|{r}"].to_numpy()
            a = dsr(x, N, V)
            a27 = dsr(x, N, V27)
            D.append({"candidate": n, "rule": r, **a, "V_d27_only": V27, "sr_star_d27_only": a27["sr_star"],
                      "dsr_d27_only": a27["dsr"], "n_rule_trials_in_V": len(RS)})
            c1 = get(n, r, 0, "2018-2025", "2xD26")
            c2 = get(n, r, 0, "2023-2025", "D26")
            c3 = get(n, r, 1, "2018-2025", "D26")
            verdict.append({"candidate": n, "rule": r, "sharpe_full_2xD26": c1, "sharpe_2325_D26": c2,
                            "sharpe_full_D26_delay1": c3, "dsr": a["dsr"], "c1": c1 > 0, "c2": c2 > 0,
                            "c3": c3 > 0, "c4": a["dsr"] > 0.95,
                            "pass": bool(c1 > 0 and c2 > 0 and c3 > 0 and a["dsr"] > 0.95),
                            "sharpe_full_D26": get(n, r, 0, "2018-2025", "D26")})
    pd.DataFrame(D).to_csv(TAB / "dsr.csv", index=False)
    VD = pd.DataFrame(verdict)
    VD.to_csv(TAB / "pass_table.csv", index=False)
    RS.to_csv(TAB / "dsr_rule_trials_in_V.csv", index=False)

    # ---- IC target (information): IC needed for net Sharpe 0.5 at 2x D26
    ic = pd.read_csv(ROOT / "reports" / "tables" / "phase3_baselines" / "baseline_results.csv")
    ic = ic[(ic["h"] == 12) & ic["delayed"]].set_index("model")["ic_full"].to_dict()
    gb = pd.read_csv(ROOT / "reports" / "tables" / "phase3" / "gate3_results.csv")
    ic["Gate-3 B"] = float(gb[(gb["variant"] == "B") & (gb["h"] == 12) & gb["delayed"]]["ic_full"].iloc[0])
    T = []
    for n in CANDIDATES:
        for r in RULES:
            S = simulate(L, sid, last, preds[n], thresholds(preds[n]), r, 0)
            year = pd.DatetimeIndex(sess_by_sid.loc[S["gross"].index]).year.to_numpy()
            cost2 = 2 * np.array([cost_y[y] for y in year]) / 1e4 * S["turn"].to_numpy()
            net2 = S["gross"].to_numpy() - cost2
            obs = float(S["gross"].mean())
            req = 0.5 * float(np.std(net2, ddof=1)) / np.sqrt(252) + float(cost2.mean())
            T.append({"candidate": n, "rule": r, "ic_h12_delayed": ic[n], "observed_gross_bp_per_day": obs * 1e4,
                      "required_gross_bp_per_day": req * 1e4, "mean_cost_2xD26_bp_per_day": cost2.mean() * 1e4,
                      "ratio": req / obs if obs > 0 else np.nan,
                      "ic_target": ic[n] * req / obs if obs > 0 else np.nan})
    pd.DataFrame(T).to_csv(TAB / "ic_target.csv", index=False)

    summ = {"n_trials_so_far": N, "V_all_rule_trials": V, "V_d27_only": V27, "n_rule_trials_in_V": len(RS),
            "passing": VD[VD["pass"]][["candidate", "rule"]].to_dict("records"),
            "best_full_D26": VD.sort_values("sharpe_full_D26").iloc[-1][["candidate", "rule", "sharpe_full_D26"]].to_dict(),
            "cost_d26_by_year": cost_y, "null_predictions_stored": False,
            "seconds": time.time() - t0}
    (TAB / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    pd.set_option("display.width", 250)
    show = BT[BT["cost"].isin(["gross", "D26", "2xD26", "3bp"])]
    print(show[["candidate", "rule", "extra_delay", "period", "cost", "trades_per_day", "turnover_per_day", "hit_rate",
                "mean_bp_per_trade", "break_even_bp_per_side", "sharpe_ann"]].round(3).to_string(index=False))
    print(pd.DataFrame(D).round(4).to_string(index=False))
    print(VD.round(3).to_string(index=False))
    print(pd.DataFrame(T).round(4).to_string(index=False))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true")
    main(ap.parse_args().log)
