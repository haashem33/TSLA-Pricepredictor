"""D28: final holdout evaluation of the frozen Gate-3 B with rule R1 (h = 12), evaluated once.

Frozen design (reports/decisions.md, D28): per month M, using only sessions before M,
  1. the Phase 2 per-fold fits (scripts/run_phase2_walkforward.py::run_fold: GARCH-t, HMM, Kalman, frac-diff,
     ADF/KPSS versions, clusters, condprob cells),
  2. the D19 re-score of the fold and the D22 selection with the existing 300-draw null quantiles
     (scripts/phase2_rescore_v2.py::run_fold + d22_select),
  3. LightGBM B with the fixed Phase 1 parameters on that list, h = 12 (tslaq.models.gate3.fold_predictions),
  4. rule R1 exactly as in D27 (scripts/run_phase5_early.py::simulate; q from the previous month's out-of-sample
     predictions), D26 cost for 2026 = EDGE 2025 half-spread + 0.10 bp.

Stages (in this order):
  --freeze            print the git commit hash and the config hash of the frozen code
  --source-check      Alpaca split-adjusted 1-min bars of the development month 2025-12 vs the local v2 source files
  --repro             frozen pipeline for development fold 2025-12 on the development data (no holdout data);
                      correlation with the stored Gate-3 B predictions, feature list, R1 P&L vs D27
  --download          holdout 1-min bars 2026-01-02..2026-10-09 (Alpaca SIP; split-adjusted, plus TSLA raw)
  --build             data/interim/v2h (v2 cleaning of 2016-01..2026-10-09) and identity checks vs v2
  --holdout           frozen pipeline for the holdout months (data version v2h)
  --evaluate --log    the single evaluation: tables, primary criterion, one registry trial
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import pickle
import subprocess
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import phase2_rescore_v2 as prs
import run_phase2_walkforward as rp2
import run_phase3_gate3 as rp3
import run_phase5_early as rp5
import yaml

from tslaq.backtest import positions as bp
from tslaq.backtest.metrics import fold_ic_v2_model
from tslaq.features.volatility import seasonal_profile
from tslaq.models import gate3 as g3
from tslaq.selection import phase2 as p2
from tslaq.selection import rescore as rs
from tslaq.utils.io import ROOT, config_hash, interim_path, load_config
from tslaq.utils.registry import log_run

warnings.filterwarnings("ignore")
H = 12
HCFG = load_config("data")["holdout_eval"]
HV, HMONTHS, END = HCFG["version"], HCFG["months"], HCFG["end_date"]
HRAW = ROOT / HCFG["raw_dir"]
OUT = ROOT / "data" / "processed" / "holdout_d28"
TAB = ROOT / "reports" / "tables" / "holdout_d28"
REPRO_M = "2025-12"
SYMS = ("TSLA", "SPY", "QQQ")
G: dict = {}


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:10]


def frozen_config() -> dict:
    return {"lgb": g3.lgb_params(), "phase2_settings": p2.SETTINGS, "data": load_config("data"),
            "backtest": load_config("backtest"),
            "d28": {"variant": "B", "h": H, "rule": "R1", "q_pct": rp5.Q_PCT, "band": rp5.BAND,
                    "no_fill_last_bars": bp.NO_FILL_LAST, "embargo": g3.EMBARGO, "rz_window": rs.RZ_WINDOW,
                    "sec_fee_bp": rp5.SEC_FEE_BP},
            "null_quantiles_sha1": sha1(prs.TAB / "null_quantiles.csv"),
            "spread_by_year_sha1": sha1(rp5.SPREAD)}


def freeze_info() -> dict:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True)
    return {"git_commit": head, "worktree_clean": dirty.strip() == "", "config_hash": config_hash(frozen_config())}


# ------------------------------------------------------------------ data

def source_check() -> pd.DataFrame:
    """Alpaca SIP split-adjusted bars of 2025-12 (development month) vs the local v2 source files."""
    from tslaq.data.download_alpaca import _client, fetch_month
    cli, rows = _client(), []
    for s in SYMS:
        a = fetch_month(cli, s, "2025-12", "split")
        b = pd.read_parquet(ROOT / "data" / "raw" / s / "2025-12.parquet")
        m = a.merge(b, on="timestamp", suffixes=("_a", "_b"))
        rows.append({"symbol": s, "rows_alpaca": len(a), "rows_local": len(b), "rows_common": len(m),
                     **{f"max_abs_diff_{c}": float((m[f"{c}_a"] - m[f"{c}_b"]).abs().max())
                        for c in ("open", "high", "low", "close", "volume")}})
    D = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    D.to_csv(TAB / "source_check_2025-12.csv", index=False)
    return D


def download():
    """Holdout months, nothing after END (request capped at END + 1 day 00:00 New York)."""
    from tslaq.data.download import _log
    from tslaq.data.download_alpaca import _client, fetch_month
    cap = pd.Timestamp(END, tz="America/New_York") + pd.Timedelta(days=1)
    cap = cap.tz_convert("UTC")
    cli = None
    for s, adj in [(s, "split") for s in SYMS] + [("TSLA", "raw")]:
        d = HRAW / (s if adj == "split" else f"{s}_raw")
        d.mkdir(parents=True, exist_ok=True)
        for M in HMONTHS:
            p = d / f"{M}.parquet"
            if p.exists():
                continue
            cli = cli or _client()
            df = fetch_month(cli, s, M, adj, cap)
            assert pd.to_datetime(df["timestamp"], utc=True).max() < cap
            df.to_parquet(p, index=False)
            _log(f"{s}_holdout_{adj}", M, "ok")
            print(f"{s} {adj} {M}: {len(df)} rows", flush=True)


def build() -> dict:
    from tslaq.data.clean import build_symbol, daily_from_bars, earnings_sessions
    from tslaq.data.download import month_range
    cfg = load_config("data")
    months = month_range(cfg["start_month"], cfg["holdout_start"][:7])
    out = {}
    interim_path("X", HV).parent.mkdir(parents=True, exist_ok=True)
    # TSLA split check: raw == split-adjusted in the holdout (no split after 2022-08)
    chk = []
    for M in HMONTHS:
        a = pd.read_parquet(HRAW / "TSLA" / f"{M}.parquet")
        b = pd.read_parquet(HRAW / "TSLA_raw" / f"{M}.parquet")
        m = a.merge(b, on="timestamp", suffixes=("_s", "_r"))
        chk.append({"month": M, "rows_split": len(a), "rows_raw": len(b), "common": len(m),
                    "max_abs_close_diff": float((m["close_s"] - m["close_r"]).abs().max()),
                    "max_abs_volume_diff": float((m["volume_s"] - m["volume_r"]).abs().max())})
    out["tsla_raw_vs_split"] = chk
    for s in SYMS:
        extra = pd.concat([pd.read_parquet(HRAW / s / f"{M}.parquet") for M in HMONTHS], ignore_index=True)
        df, info = build_symbol(s, cfg, "v2", months, end_date=END, extra_m1=extra)
        assert df["session_date"].max() <= pd.Timestamp(END)
        df.to_parquet(interim_path(s, HV), index=False)
        daily_from_bars(df).to_parquet(interim_path(s, HV).with_name(f"{s}_daily.parquet"), index=False)
        old = pd.read_parquet(interim_path(s, "v2"))
        new_dev = df[df["session_date"] < pd.Timestamp("2026-01-01")].reset_index(drop=True)
        same_ts = len(new_dev) == len(old) and bool((new_dev["timestamp"].to_numpy() == old["timestamp"].to_numpy()).all())
        r_new = np.diff(np.log(new_dev["close"].to_numpy()))
        r_old = np.diff(np.log(old["close"].to_numpy()))
        h = df[df["session_date"] >= pd.Timestamp("2026-01-01")]
        out[s] = {"rows_dev_v2": len(old), "rows_dev_v2h": len(new_dev), "dev_timestamps_identical": same_ts,
                  "dev_frames_identical": bool(same_ts and new_dev.equals(old)),
                  "dev_max_abs_logret_diff": float(np.max(np.abs(r_new - r_old))) if same_ts else None,
                  "holdout_sessions": int(h["session_date"].nunique()), "holdout_bars": len(h),
                  "holdout_first": str(h["session_date"].min().date()), "holdout_last": str(h["session_date"].max().date()),
                  "holdout_missing_bars": int(h["missing"].sum()), "holdout_bad_tick": int(h["bad_tick"].sum()),
                  "holdout_bad_range": int(h["bad_range"].sum()), "holdout_half_days": int(h.groupby("session_date")["is_half_day"].first().sum()),
                  "dropped_sessions_holdout": [str(d.date()) for d in info["dropped_sessions"]["session_date"]
                                               if d >= pd.Timestamp("2026-01-01")],
                  "sessions_without_data": info["sessions_without_data"]}
        print(s, json.dumps(out[s]), flush=True)
    E = earnings_sessions(cfg, end_date=END)
    E.to_parquet(interim_path("X", HV).with_name("TSLA_earnings_sessions.parquet"), index=False)
    out["earnings_reaction_sessions_holdout"] = [str(d.date()) for d in E["reaction_session"]
                                                 if d >= pd.Timestamp("2026-01-01")]
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "holdout_data_checks.json").write_text(json.dumps(out, indent=1, default=str))
    return out


# ------------------------------------------------------------------ frozen pipeline, one stage per pool

def _init_p2():
    G["D"] = rp2.prepare()
    prof = seasonal_profile(G["D"]["L"], load_config("data")["volatility"]["seasonal_window_sessions"])
    G["prof"], G["pos_s"] = prof.to_numpy(), prof.index.get_indexer(G["D"]["L"]["session_date"])
    G["cost"] = 2 * load_config("backtest")["costs"]["bps_per_side"] / 1e4


def job_phase2(args) -> float:
    M, ckd = args
    path = ckd / f"{M}.pkl"
    if path.exists():
        return 0.0
    if "D" not in G:
        _init_p2()
    r = rp2.run_fold(M, G["D"], G["cost"], G["prof"], G["pos_s"], report=False)
    with open(path, "wb") as fh:
        pickle.dump(r, fh)
    print(f"phase2 fit {M}: {r['seconds']:.0f}s", flush=True)
    return r["seconds"]


def job_rescore(args) -> float:
    M, ckd, rsd = args
    prs.CK, prs.OUT = ckd, rsd
    sec = prs.run_fold(M)
    print(f"rescore {M}: {sec:.0f}s", flush=True)
    return sec


def job_b(args) -> pd.DataFrame:
    M, ckd, versions = args
    g3.CK = ckd
    if "fr" not in G:
        G["fr"] = g3.real_frame()
    R = g3.fold_predictions(G["fr"], M, versions, variants=("B",), horizons=(H,))
    print(f"B {M}: {R['seconds']:.0f}s, {len(R['preds'])} predictions", flush=True)
    return R["preds"].assign(seconds=R["seconds"])


def select(months, rsd: Path) -> dict:
    q12, qd12 = prs.d22_thresholds()
    out = {}
    for M in months:
        with open(rsd / f"{M}.pkl", "rb") as fh:
            r = pickle.load(fh)
        S = prs.d22_select(r, q12, qd12)
        out[M] = [{"feature": x.feature, "version": x.version, "basis": x.basis} for x in S[S["kept"]].itertuples()]
    return out


def run_pipeline(months, tag: str, workers: int) -> tuple[pd.DataFrame, dict, dict]:
    base = OUT / tag
    ckd, rsd = base / "phase2_folds", base / "rescore"
    for d in (ckd, rsd):
        d.mkdir(parents=True, exist_ok=True)
    w = max(1, min(workers, len(months)))
    t = {}
    t0 = time.time()
    with Pool(w) as pool:
        pool.map(job_phase2, [(M, ckd) for M in months], chunksize=1)
    t["phase2_fit_wall_s"] = time.time() - t0
    t0 = time.time()
    with Pool(w) as pool:
        pool.map(job_rescore, [(M, ckd, rsd) for M in months], chunksize=1)
    t["rescore_wall_s"] = time.time() - t0
    lists = select(months, rsd)
    (base / "feature_lists.yaml").write_text(yaml.safe_dump(lists, sort_keys=False, width=200), encoding="utf-8")
    pp = base / "preds_B_h12.parquet"
    t0 = time.time()
    if not pp.exists():
        with Pool(w) as pool:
            P = pool.map(job_b, [(M, ckd, {x["feature"]: x["version"] for x in lists[M]}) for M in months],
                         chunksize=1)
        pd.concat(P, ignore_index=True).to_parquet(pp, index=False)
    t["lightgbm_wall_s"] = time.time() - t0
    (base / "timing.json").write_text(json.dumps(t, indent=1))
    print(f"{tag} timing: {json.dumps(t)}", flush=True)
    return pd.read_parquet(pp), lists, t


# ------------------------------------------------------------------ helpers shared by --repro and --evaluate

def stored_b() -> pd.DataFrame:
    P = []
    for M in g3.MONTHS:
        with open(rp3.REAL / f"{M}.pkl", "rb") as fh:
            P.append(pickle.load(fh)["preds"])
    P = pd.concat(P, ignore_index=True)
    return P[(P["variant"] == "B") & (P["h"] == H)][["fold", "pos", "pred"]].reset_index(drop=True)


def q90(pred) -> float:
    return float(np.nanpercentile(np.abs(np.asarray(pred, float)), rp5.Q_PCT))


def day_frame(fr: dict, S: dict, sid_dates: pd.Series) -> pd.DataFrame:
    d = pd.DatetimeIndex(sid_dates.loc[S["gross"].index])
    return pd.DataFrame({"session": d, "month": d.to_period("M").astype(str), "gross": S["gross"].to_numpy(),
                         "turnover": S["turn"].to_numpy(), "fills": S["ntr"].to_numpy()})


def perf(net: np.ndarray) -> dict:
    net = pd.Series(net)
    sd = net.std()
    dd = np.sqrt(np.mean(np.minimum(net.to_numpy(), 0.0) ** 2))
    cum = net.cumsum()
    return {"total_pnl_bp": float(net.sum() * 1e4), "mean_daily_bp": float(net.mean() * 1e4),
            "sharpe_ann": float(net.mean() / sd * np.sqrt(252)) if sd > 0 else np.nan,
            "sharpe_day": float(net.mean() / sd) if sd > 0 else np.nan,
            "t": float(net.mean() / sd * np.sqrt(len(net))) if sd > 0 else np.nan,
            "sortino_ann": float(net.mean() / dd * np.sqrt(252)) if dd > 0 else np.nan,
            "max_dd_bp": float((cum.cummax() - cum).max() * 1e4), "share_days_positive": float((net > 0).mean())}


def ic_stats(s: pd.Series) -> dict:
    s = s.dropna()
    sd = s.std(ddof=1)
    return {"n_months": len(s), "ic_mean": float(s.mean()), "ic_sd": float(sd),
            "t": float(s.mean() / sd * np.sqrt(len(s))) if sd > 0 else np.nan, "share_ic_pos": float((s > 0).mean())}


# ------------------------------------------------------------------ reproduction check

def repro(workers: int) -> dict:
    assert os.environ.get("TSLAQ_DATA_VERSION") in (None, "", "v2")
    P, lists, timing = run_pipeline([REPRO_M], "repro", workers)
    fr = g3.real_frame()
    L = fr["L"]
    assert L["session_date"].max() < pd.Timestamp("2026-01-01"), "holdout must not be loaded in the repro check"
    SB = stored_b()
    old = SB[SB["fold"] == REPRO_M]
    m = old.merge(P[["pos", "pred"]], on="pos", suffixes=("_stored", "_new"))
    with open(ROOT / "configs" / "features.yaml", encoding="utf-8") as fh:
        yl = yaml.safe_load(fh)["by_fold"][REPRO_M]
    same_list = [(x["feature"], x["version"]) for x in yl] == [(x["feature"], x["version"]) for x in lists[REPRO_M]]
    # R1 for the month, q from the stored 2025-11 predictions (as in D27)
    sid, last = bp.session_arrays(fr["sess"])
    sid_dates = pd.Series(fr["sess"]).groupby(sid).first()
    q = {REPRO_M: q90(SB[SB["fold"] == "2025-11"]["pred"])}
    cost = rp5.d26_cost_by_year()[2025]
    res = {}
    for tag, Q in (("new", P.assign(fold=REPRO_M)[["fold", "pos", "pred"]]), ("stored", old)):
        D = day_frame(fr, rp5.simulate(L, sid, last, Q, q, "R1", 0), sid_dates)
        res[tag] = D.assign(net=D["gross"] - cost / 1e4 * D["turnover"])
    d27 = pd.read_csv(rp5.TAB / "daily_net_d26.csv", parse_dates=["session"]).set_index("session")["Gate-3 B|R1"]
    d27 = d27[d27.index.to_period("M").astype(str) == REPRO_M]
    out = {"fold": REPRO_M, "n_stored": len(old), "n_new": len(P), "n_common": len(m),
           "corr": float(np.corrcoef(m["pred_stored"], m["pred_new"])[0, 1]),
           "max_abs_diff": float((m["pred_stored"] - m["pred_new"]).abs().max()),
           "mean_abs_diff": float((m["pred_stored"] - m["pred_new"]).abs().mean()),
           "sd_stored_pred": float(m["pred_stored"].std()),
           "feature_list_identical": same_list, "n_features_new": len(lists[REPRO_M]), "n_features_stored": len(yl),
           "q": q[REPRO_M], "d26_cost_2025_bp": cost,
           "r1_net_pnl_bp_new": float(res["new"]["net"].sum() * 1e4),
           "r1_net_pnl_bp_stored_preds": float(res["stored"]["net"].sum() * 1e4),
           "r1_net_pnl_bp_d27": float(d27.sum() * 1e4),
           "r1_gross_pnl_bp_new": float(res["new"]["gross"].sum() * 1e4),
           "r1_fills_new": float(res["new"]["fills"].sum()), "r1_fills_stored": float(res["stored"]["fills"].sum()),
           "r1_daily_net_max_abs_diff_new_vs_d27_bp": float(np.max(np.abs(
               res["new"].set_index("session")["net"].reindex(d27.index).to_numpy() - d27.to_numpy())) * 1e4),
           **{k: v for k, v in timing.items()}}
    out["pass_corr_0.99"] = bool(out["corr"] >= 0.99)
    TAB.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(TAB / "reproduction_check.csv", index=False)
    print(json.dumps(out, indent=1), flush=True)
    return out


# ------------------------------------------------------------------ the single evaluation

def evaluate(log: bool) -> dict:
    assert os.environ.get("TSLAQ_DATA_VERSION") == HV
    marker = TAB / "holdout_summary.json"
    if marker.exists():
        raise SystemExit(f"{marker} exists: the holdout is evaluated once (D28); not re-running.")
    HP = pd.read_parquet(OUT / "holdout" / "preds_B_h12.parquet")
    assert sorted(HP["fold"].unique()) == HMONTHS
    fr = g3.real_frame()
    L = fr["L"]
    assert L["session_date"].max() <= pd.Timestamp(END)
    SB = stored_b()
    # the development rows of the v2h frame are the v2 rows in the same positions
    Lv2 = pd.read_parquet(interim_path("TSLA", "v2"))
    ndev = len(Lv2)
    assert (L["timestamp"].to_numpy()[:ndev] == Lv2["timestamp"].to_numpy()).all()
    assert int((L["session_date"] < pd.Timestamp("2026-01-01")).sum()) == ndev
    sid, last = bp.session_arrays(fr["sess"])
    sid_dates = pd.Series(fr["sess"]).groupby(sid).first()
    Q = HP[["fold", "pos", "pred"]]
    q = {HMONTHS[0]: q90(SB[SB["fold"] == g3.MONTHS[-1]]["pred"])}
    for prev, M in itertools.pairwise(HMONTHS):
        q[M] = q90(Q[Q["fold"] == prev]["pred"])
    cost_y = rp5.d26_cost_by_year()
    c26 = cost_y[2026]
    costs = {"D26": c26, "2xD26": 2 * c26, "3bp": 3.0, "gross": 0.0}
    rows, months, daily = [], [], []
    for d in (0, 1):
        S = rp5.simulate(L, sid, last, Q, q, "R1", d)
        D = day_frame(fr, S, sid_dates)
        u = S["unit"]
        base = {"extra_delay": d, "n_days": len(D), "fills_per_day": float(D["fills"].mean()),
                "turnover_per_day": float(D["turnover"].mean()), "n_holding_segments": len(u),
                "hit_rate": float(np.mean(u > 0)) if len(u) else np.nan,
                "mean_bp_per_segment": float(np.mean(u) * 1e4) if len(u) else np.nan,
                "break_even_bp_per_side": float(D["gross"].mean() / D["turnover"].mean() * 1e4)}
        for tag, c in costs.items():
            net = D["gross"] - c / 1e4 * D["turnover"]
            rows.append(base | {"cost": tag, "cost_bp_per_side": c} | perf(net.to_numpy()))
            D[f"net_{tag}"] = net
            g = D.groupby("month")
            for M, x in g:
                months.append({"month": M, "extra_delay": d, "cost": tag, "n_days": len(x),
                               "pnl_bp": float(x[f"net_{tag}"].sum() * 1e4), "fills": float(x["fills"].sum()),
                               "turnover": float(x["turnover"].sum()), "q": q[M]})
        daily.append(D.assign(extra_delay=d))
    R = pd.DataFrame(rows)
    MB = pd.DataFrame(months)
    # D19 IC at h = 12, with and without delay; V_s from all earlier out-of-sample months (incl. development)
    allm = [*g3.MONTHS, *HMONTHS]
    PA = pd.concat([SB, Q], ignore_index=True)
    ics, icsum, devchk = {}, [], {}
    stored_ic = pd.read_csv(ROOT / "reports" / "tables" / "phase3" / "gate3_fold_ic.csv")
    for dl in (False, True):
        y = g3.outcome(fr, H, dl)
        zbar, vy = g3.train_moments(fr, y, [*allm, g3.INNER[0]])
        inn = PA[PA["fold"] == "inner"]
        s0 = inn["pred"].to_numpy() - zbar[g3.INNER[0]]
        vs0 = float(np.mean(s0[np.isfinite(y[inn["pos"].to_numpy()])] ** 2))
        X = PA[PA["fold"] != "inner"]
        ic = fold_ic_v2_model(X["pred"], y[X["pos"].to_numpy()], X["fold"], zbar, vy, vs0)
        key = "delayed" if dl else "no_delay"
        ics[key] = ic[HMONTHS]
        icsum.append({"outcome": key, **ic_stats(ic[HMONTHS])})
        si = stored_ic[(stored_ic["variant"] == "B") & (stored_ic["h"] == H) & (stored_ic["delayed"] == dl)]
        devchk[key] = float(np.max(np.abs(ic[g3.MONTHS].to_numpy() - si.set_index("fold")["ic"].reindex(g3.MONTHS).to_numpy())))
    ICM = pd.DataFrame({"month": HMONTHS, "ic_h12": ics["no_delay"].to_numpy(), "ic_h12_delayed": ics["delayed"].to_numpy(),
                        "n_predictions": Q.groupby("fold").size().reindex(HMONTHS).to_numpy()})
    main = R[(R["extra_delay"] == 0) & (R["cost"] == "D26")].iloc[0]
    crit = bool(main["total_pnl_bp"] > 0)
    TAB.mkdir(parents=True, exist_ok=True)
    R.to_csv(TAB / "holdout_results.csv", index=False)
    MB.to_csv(TAB / "holdout_by_month.csv", index=False)
    ICM.to_csv(TAB / "holdout_ic_by_month.csv", index=False)
    pd.concat(daily, ignore_index=True).to_csv(TAB / "holdout_daily.csv", index=False)
    pd.DataFrame([{"month": M, "q": v} for M, v in q.items()]).to_csv(TAB / "holdout_q_by_month.csv", index=False)
    summ = {"criterion": "total net P&L of R1 at D26 cost over the holdout > 0",
            "criterion_pass": crit, "total_net_pnl_bp_d26": float(main["total_pnl_bp"]),
            "net_sharpe_ann_d26": float(main["sharpe_ann"]), "d26_cost_2026_bp_per_side": c26,
            "n_sessions": int(main["n_days"]), "first_session": str(daily[0]["session"].min().date()),
            "last_session": str(daily[0]["session"].max().date()), "ic_summary": icsum,
            "dev_fold_ic_max_abs_diff_vs_stored": devchk, "freeze": freeze_info()}
    if log:
        r2x = R[(R["extra_delay"] == 0) & (R["cost"] == "2xD26")].iloc[0]
        r1d = R[(R["extra_delay"] == 1) & (R["cost"] == "D26")].iloc[0]
        log_run({"config_hash": summ["freeze"]["config_hash"], "phase": 5, "model": "d28_holdout_R1_Gate3B_h12",
                 "fold_scheme": "D28 frozen design, monthly folds 2026-01..2026-10 (to 2026-10-09); fill next open; D26 cost",
                 "rank_ic_h12": icsum[0]["ic_mean"], "hit_rate_h12": float(main["hit_rate"]),
                 "net_sharpe_period": float(main["sharpe_day"]), "net_sharpe_ann": float(main["sharpe_ann"]),
                 "sortino": float(main["sortino_ann"]), "max_drawdown": float(main["max_dd_bp"]),
                 "turnover": float(main["turnover_per_day"]), "sharpe_2x_costs": float(r2x["sharpe_ann"]),
                 "data_version": HV,
                 "notes": (f"D28 single holdout evaluation; criterion total net P&L at D26 > 0: {crit} "
                           f"({main['total_pnl_bp']:.1f} bp); +1 bar delay net D26 {r1d['total_pnl_bp']:.1f} bp, Sharpe "
                           f"{r1d['sharpe_ann']:.2f}; IC h12 mean {icsum[0]['ic_mean']:+.4f} t {icsum[0]['t']:.2f}, "
                           f"delayed {icsum[1]['ic_mean']:+.4f} t {icsum[1]['t']:.2f}")})
    marker.write_text(json.dumps(summ, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(R.round(3).to_string(index=False))
    print(MB[MB["cost"].isin(["D26", "gross"])].round(2).to_string(index=False))
    print(ICM.round(4).to_string(index=False))
    print(json.dumps(summ, indent=1, default=str), flush=True)
    return summ


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    for f in ("freeze", "source-check", "repro", "download", "build", "holdout", "evaluate", "log"):
        ap.add_argument(f"--{f}", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    if a.freeze:
        print(json.dumps(freeze_info(), indent=1))
    if a.source_check:
        print(source_check().to_string(index=False))
    if a.repro:
        repro(a.workers)
    if a.download:
        download()
    if a.build:
        build()
    if a.holdout or a.evaluate:
        os.environ["TSLAQ_DATA_VERSION"] = HV
    if a.holdout:
        run_pipeline(HMONTHS, "holdout", a.workers)
    if a.evaluate:
        evaluate(a.log)
