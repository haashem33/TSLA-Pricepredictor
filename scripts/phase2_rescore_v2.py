"""D21 step C: re-score Phase 2 with the D19 anchored IC v2 (no refits).

Per fold M (96 folds 2018-01..2025-12) the fold's feature frame is rebuilt from its checkpoint
(data/processed/phase2_folds/<M>.pkl: fitted GARCH-t / HMM / Kalman / frac-d parameters and feature versions;
tslaq.selection.rescore.fold_features, verified to reproduce the stored frame exactly). Then, with fold M's
D19 anchors (training rows = months < M):
- monthly D19 IC on the training months and month M for h = 1..24 and the delayed h = 1, 3, 12;
- the stable-IC table and rule (p2.feature_ic_table, settings unchanged) on D19 training-month IC;
- out-of-sample signed IC in month M (sign from the training months), h = 1, 3, 12 with and without delay,
  and h = 1..24 for the decay curve;
- slot IC: D19 IC within each half-hour slot of month M (same anchors and training moments), signed;
- clustered MDA (same clusters, k, rows and LightGBM parameters) scored with D19 (score="v2");
- the extreme-bucket rule unchanged (stored condprob cells), and p2.select -> the fold's v2 feature list.
Then: configs/features.yaml v2 (old file kept as configs/features_v1_biased.yaml), the D10 table old vs D19,
the selection changes, D16 rule (a) with the D19 training sign (registry, rerun_of), and the stored Phase 1
predictions re-scored with the D19 model IC (no retraining).

Usage: python scripts/phase2_rescore_v2.py --time-one        (one fold, prints the projected run time)
       python scripts/phase2_rescore_v2.py --workers 4        (all folds, resumable, then summaries)
       python scripts/phase2_rescore_v2.py --summarize
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from phase2_daily_diagnostics import rule_metrics

from tslaq.backtest.metrics import Z_CLIP, fold_ic_stats, fold_ic_v2_model
from tslaq.data.pipeline import load_bars
from tslaq.selection import phase2 as p2
from tslaq.selection import rescore as rs
from tslaq.selection.ic import feature_scores_v2, score_anchors_v2, v2_group_ic
from tslaq.selection.importance import MDA_PARAMS, clustered_mda
from tslaq.utils.io import ROOT, config_hash, load_config
from tslaq.utils.registry import log_run

warnings.filterwarnings("ignore")
CK = ROOT / "data" / "processed" / "phase2_folds"
OUT = ROOT / "data" / "processed" / "phase2_rescore_v2"
TAB = ROOT / "reports" / "tables" / "phase2_metric_fix_v3"
MONTHS = [str(p) for p in pd.period_range("2018-01", "2025-12", freq="M")]
H = (1, 3, 12)
DECAY_H = tuple(range(1, 25))
SEL_KEYS = [*[str(h) for h in H], *[f"d{h}" for h in H]]
SLOT_KEYS = ("1", "3", "12", "d12")
G: dict = {}


def init():
    lam = rs.lambdas()
    L = rs.base_frame(load_bars("TSLA"), lam["TSLA"], DECAY_H)
    F_free, Frz_free = rs.free_features(L, rs.market_frames(lam), rs.reaction_sessions())
    prof, pos = rs.profile(L)
    Z = {str(h): L[f"z_{h}"].to_numpy() for h in DECAY_H} | {f"d{h}": L[f"zd_{h}"].to_numpy() for h in H}
    G.update({"lam": lam, "L": L, "F_free": F_free, "Frz_free": Frz_free, "prof": prof, "pos": pos, "Z": Z,
              "month": L["session_date"].dt.to_period("M").astype(str).to_numpy(),
              "slot": L["bar_index"].to_numpy() // 6})


def slot_ic(X: pd.DataFrame, A: dict, vys: dict, M: str) -> pd.DataFrame:
    """Unsigned D19 IC of each feature within each half-hour slot of month M (fold anchors), long format."""
    te = np.flatnonzero(G["month"] == M)
    sl = G["slot"][te]
    order = np.argsort(sl, kind="stable")
    rows_ = te[order]
    S = feature_scores_v2(X.to_numpy(float)[rows_], A)
    out = []
    for k in SLOT_KEYS:
        y = np.clip(G["Z"][k][rows_], -Z_CLIP, Z_CLIP)
        g = v2_group_ic(S, y, sl[order], A["vs"], vys[k], 50)
        g.columns = X.columns
        g = g.stack(future_stack=True).rename("ic").reset_index()
        g.columns = ["slot", "feature", "ic"]
        g["h"] = k
        out.append(g)
    return pd.concat(out, ignore_index=True)


def run_fold(M: str) -> float:
    path = OUT / f"{M}.pkl"
    if path.exists():
        return 0.0
    if not G:
        init()
    t0 = time.time()
    with open(CK / f"{M}.pkl", "rb") as fh:
        r = pickle.load(fh)
    versions = r["versions"]
    L, month = G["L"], G["month"]
    X = rs.fold_features(L, G["F_free"], G["Frz_free"], rs.fold_params(r), G["prof"], G["pos"], versions)
    IC, A, vys = rs.anchored_fold_v2(X, G["Z"], month, M)
    IC["version"] = IC["feature"].map(versions)
    cutoff = f"{M}-01"
    T = p2.feature_ic_table(IC, versions, cutoff)
    W = IC.pivot_table(index="month", columns=["h", "feature"], values="ic", dropna=False)
    tr_m = W.index < M
    sign = {k: np.sign(W[k][tr_m].mean()) for k in W.columns.get_level_values(0).unique()}
    icm = {k: W[k].loc[M] if M in W.index else pd.Series(dtype=float) for k in sign}
    oos = []
    for k, sk in sign.items():
        for f in X.columns:
            v, sg = icm[k].get(f, np.nan), sk.get(f, np.nan)
            if np.isfinite(v) and np.isfinite(sg):
                oos.append({"month": M, "feature": f, "version": versions[f], "h": k, "ic": v, "sign": sg,
                            "signed_ic": sg * v})
    SL = slot_ic(X, A, vys, M)
    SL["sign"] = [sign[k].get(f, np.nan) for k, f in zip(SL["h"], SL["feature"])]
    SL["month"] = M
    tr = L["session_date"].to_numpy() < np.datetime64(cutoff)
    mda = {}
    params = MDA_PARAMS | {"num_threads": 1}
    for h in H:
        le = L[f"label_end_{h}"].to_numpy()
        z = L[f"z_{h}"].to_numpy()
        pos = np.flatnonzero(tr & np.isfinite(z) & (le >= 0))[:: p2.SETTINGS["mda"]["row_step"]]
        mda[h] = clustered_mda(X, z, le, pos, r["clusters"], p2.SETTINGS["mda"]["k"], params=params, score="v2")
    K = p2.extreme_keep(r["C"])
    S = p2.select(T, K, r["clusters"], mda)
    S["version"] = S["feature"].map(versions)
    out = {"month": M, "seconds": time.time() - t0, "T": T, "oos": pd.DataFrame(oos), "slot": SL,
           "mda": pd.concat(mda, names=["h"]), "S": S, "versions": versions,
           "train_ic": IC[IC["month"] < M] if M == MONTHS[-1] else None}
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    print(f"{M}: {out['seconds']:.0f}s kept={int(S['kept'].sum())}", flush=True)
    return out["seconds"]


def wf_table(OOS: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (f, key), g in OOS.groupby(["feature", "h"]):
        s = g.set_index("month")["signed_ic"].sort_index()
        yr = pd.to_datetime(s.index).year
        a, b, c = fold_ic_stats(s), fold_ic_stats(s[yr >= 2023]), fold_ic_stats(s.iloc[-24:])
        rows.append({"feature": f, "h": key, "ic_full": a["ic"], "t_full": a["t"], "pos_full": a["pos_share"],
                     "binom_p_full": a["binom_p"], "n_full": a["n_folds"], "ic_2325": b["ic"], "t_2325": b["t"],
                     "n_2325": b["n_folds"], "ic_last24": c["ic"], "t_last24": c["t"]})
    WF = pd.DataFrame(rows)
    WF["d10_pass"] = (WF["t_full"] >= 2) & (WF["ic_2325"] > 0) & (WF["t_2325"] >= 1)
    return WF


def rule_d19(L: pd.DataFrame, F: pd.DataFrame, extra_delay: int = 0):
    """D16 rule (a), d_sma20_dev: identical to scripts/phase2_daily_diagnostics.py rule_returns except that the
    fold sign is the sign of the mean D19 IC (h = 12) over the training months, with that fold's anchors."""
    sess = L["session_date"].to_numpy()
    first = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1]
    last = np.r_[first[1:] - 1, len(sess) - 1]
    O, C = L["open"].to_numpy(), L["close"].to_numpy()
    x = F["d_sma20_dev"].to_numpy()
    y = np.clip(L["z_12"].to_numpy(), -Z_CLIP, Z_CLIP)
    month = L["session_date"].dt.to_period("M").astype(str).to_numpy()
    sg = {}
    for M in MONTHS:
        tr = month < M
        A = score_anchors_v2(x[tr][:, None])
        ic = v2_group_ic(feature_scores_v2(x[tr][:, None], A), y[tr], month[tr], A["vs"],
                         float(np.nanmean(y[tr] ** 2)), 200)[0]
        sg[M] = np.sign(ic.mean())
    sg = pd.Series(sg)
    val = np.sign(x[first])
    entry = first + 1 + extra_delay
    gross = np.log(C[last] / O[entry])
    m = pd.Series(sess[first]).dt.to_period("M").astype(str).to_numpy()
    pos = sg.reindex(m).to_numpy() * val
    out = pd.DataFrame({"session": sess[first], "month": m, "pos": pos, "ret": pos * gross})
    return out[np.isin(m, MONTHS)].dropna().reset_index(drop=True), sg


def phase1_rescore(L: pd.DataFrame) -> pd.DataFrame:
    """D19 model IC of the stored Phase 1 return predictions (no retraining). zbar and V_y per fold from the
    training rows (sessions before the fold month); V_s from earlier out-of-sample folds. No inner-validation
    predictions were stored, so the first fold (2018-01) only seeds V_s and is not scored."""
    R = pd.read_parquet(ROOT / "data" / "processed" / "phase1_return_preds_v2.parquet")
    sess = L["session_date"].to_numpy()
    rows = []
    for (model, h), g in R.groupby(["model", "h"]):
        z = L[f"z_{h}"].to_numpy()
        y = np.clip(z, -Z_CLIP, Z_CLIP)
        g = g.sort_values("pos")
        folds = sorted(g["fold"].unique())
        zbar, vy = {}, {}
        for f in folds:
            tr = sess < np.datetime64(f"{f}-01")
            zbar[f], vy[f] = float(np.nanmean(y[tr])), float(np.nanmean(y[tr] ** 2))
        f0 = g["fold"] == folds[0]
        s0 = g.loc[f0, "pred_z"].to_numpy() - zbar[folds[0]]
        vs0 = float(np.nanmean(s0**2))
        sub = g[~f0]
        ic = fold_ic_v2_model(sub["pred_z"], z[sub["pos"].to_numpy()], sub["fold"], zbar, vy, vs0)
        yr = pd.Index(ic.index).str[:4].astype(int)
        a, b = fold_ic_stats(ic), fold_ic_stats(ic[yr >= 2023])
        rows.append({"model": model, "h": h, "n_folds": a["n_folds"], "ic_full": a["ic"], "t_full": a["t"],
                     "pos_share": a["pos_share"], "binom_p": a["binom_p"], "ic_2325": b["ic"], "t_2325": b["t"],
                     "d10_pass": bool(a["t"] >= 2 and b["ic"] > 0 and b["t"] >= 1)})
    return pd.DataFrame(rows)


def summarize():
    TAB.mkdir(parents=True, exist_ok=True)
    folds = {}
    for M in MONTHS:
        with open(OUT / f"{M}.pkl", "rb") as fh:
            folds[M] = pickle.load(fh)
    old = {}
    for M in MONTHS:
        with open(CK / f"{M}.pkl", "rb") as fh:
            r = pickle.load(fh)
        old[M] = r["S"]
    OOS = pd.concat([r["oos"] for r in folds.values()], ignore_index=True)
    OOS.to_csv(TAB / "rescore_d19_oos_signed_ic_by_fold.csv", index=False)
    WF = wf_table(OOS[OOS["h"].isin(SEL_KEYS)])
    WF.to_csv(TAB / "rescore_d19_walkforward_summary.csv", index=False)
    WO = pd.read_csv(ROOT / "reports" / "tables" / "phase2_walkforward_summary.csv", dtype={"h": str})
    cols = ["ic_full", "t_full", "ic_2325", "t_2325", "d10_pass"]
    D10 = WO[["feature", "h", *cols]].add_prefix("old_").rename(columns={"old_feature": "feature", "old_h": "h"}).merge(
        WF[["feature", "h", *cols]].add_prefix("d19_").rename(columns={"d19_feature": "feature", "d19_h": "h"}),
        on=["feature", "h"], how="outer")
    D10 = D10.sort_values(["h", "d19_t_full"], ascending=[True, False])
    D10.to_csv(TAB / "d10_old_vs_d19.csv", index=False)
    # decay: out-of-sample signed IC by h, and the last fold's training-window mean IC (raw versions)
    dec = OOS[~OOS["h"].str.startswith("d")].copy()
    dec["h"] = dec["h"].astype(int)
    DEC = dec.groupby(["feature", "h"])["signed_ic"].agg(
        ic_oos="mean", t_oos=lambda s: fold_ic_stats(s.reset_index(drop=True))["t"]).reset_index()
    TI = folds[MONTHS[-1]]["train_ic"]
    TI = TI[~TI["h"].str.startswith("d")].assign(h=lambda d: d["h"].astype(int))
    DEC = DEC.merge(TI.groupby(["feature", "h"])["ic"].mean().rename("ic_train_lastfold").reset_index(),
                    on=["feature", "h"], how="left")
    DEC.to_csv(TAB / "ic_decay_d19.csv", index=False)
    # slot IC: per fold, mean over slots of the signed slot IC; then over folds
    SL = pd.concat([r["slot"] for r in folds.values()], ignore_index=True)
    SL["signed"] = SL["ic"] * SL["sign"]
    pf = SL.groupby(["h", "feature", "month"])["signed"].mean().rename("slot_ic").reset_index()
    SLT = pf.groupby(["h", "feature"])["slot_ic"].agg(
        slot_ic_mean="mean", slot_t=lambda s: fold_ic_stats(s.reset_index(drop=True))["t"],
        n_folds="count").reset_index()
    SLT.to_csv(TAB / "slot_ic_d19.csv", index=False)
    # selection: v2 lists, changes per fold, per-feature shares
    v1_path = ROOT / "configs" / "features_v1_biased.yaml"
    if not v1_path.exists():
        shutil.copyfile(ROOT / "configs" / "features.yaml", v1_path)
    with open(v1_path, encoding="utf-8") as fh:
        v1 = yaml.safe_load(fh)
    chg, sel = [], []
    for M, r in folds.items():
        k2 = r["S"][r["S"]["kept"]]
        a = {x["feature"] for x in v1["by_fold"][M]}
        b = set(k2["feature"])
        chg.append({"fold": M, "n_v1": len(a), "n_v2": len(b), "n_both": len(a & b),
                    "gained": ", ".join(sorted(b - a)), "lost": ", ".join(sorted(a - b))})
        sel.append(k2.assign(fold=M))
    CH = pd.DataFrame(chg)
    CH.to_csv(TAB / "selection_changes_by_fold.csv", index=False)
    SEL = pd.concat(sel, ignore_index=True)
    SEL.to_csv(TAB / "rescore_d19_selected_by_fold.csv", index=False)
    s1 = pd.Series([x["feature"] for M in MONTHS for x in v1["by_fold"][M]]).value_counts() / len(MONTHS)
    s2 = SEL["feature"].value_counts() / len(MONTHS)
    FS = pd.DataFrame({"share_folds_v1": s1, "share_folds_v2": s2}).fillna(0.0)
    FS["basis_v2"] = SEL.groupby("feature")["basis"].agg(lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().items()))
    FS["delta"] = FS["share_folds_v2"] - FS["share_folds_v1"]
    FS.sort_values("share_folds_v2", ascending=False).to_csv(TAB / "feature_selection_v1_vs_v2.csv")
    cfg = {"version": 2, "metric": "D19 anchored IC v2 (stable-IC rule and clustered MDA); extreme-bucket rule unchanged",
           "generated": time.strftime("%Y-%m-%d"), "data_version": load_config("data")["data_version"],
           "previous_version": "configs/features_v1_biased.yaml (old monthly-pooled Spearman, biased)",
           "note": v1["note"], "settings": v1["settings"],
           "by_fold": {M: [{"feature": x.feature, "version": x.version, "basis": x.basis}
                           for x in r["S"][r["S"]["kept"]].itertuples()] for M, r in folds.items()}}
    with open(ROOT / "configs" / "features.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True, width=200)
    folds[MONTHS[-1]]["mda"].to_csv(TAB / "mda_d19_lastfold.csv")
    # D16 rule (a) with the D19 training sign
    L = G["L"]
    F = G["F_free"]
    R, sg = rule_d19(L, F)
    Rd, _ = rule_d19(L, F, extra_delay=1)
    RM, RMd = rule_metrics(R), rule_metrics(Rd)
    RM.assign(delay="none").pipe(lambda d: pd.concat([d, RMd.assign(delay="+1 bar")])).to_csv(
        TAB / "rule_d_sma20_dev_d19.csv", index=False)
    bt = load_config("backtest")
    full = RM.iloc[0]
    net3, net6 = R["ret"] - 2 * 3e-4 * np.abs(R["pos"]), R["ret"] - 2 * 6e-4 * np.abs(R["pos"])
    eq = net3.cumsum()
    log_run({"config_hash": config_hash({"rule": "d_sma20_dev", "costs": bt["costs"], "sign": "D19"}), "phase": 2,
             "model": "rule_d_sma20_dev", "n_params": 0,
             "fold_scheme": "expanding monthly WF 2018-01..2025-12; sign from training-month D19 IC (h=12)",
             "net_sharpe_period": net3.mean() / net3.std(), "net_sharpe_ann": full["sharpe_ann_3bp"],
             "sortino": net3.mean() / net3[net3 < 0].std() * np.sqrt(252),
             "max_drawdown": float((eq - eq.cummax()).min()), "turnover": float(2 * np.abs(R["pos"]).mean()),
             "sharpe_2x_costs": net6.mean() / net6.std() * np.sqrt(252), "data_version": load_config("data")["data_version"],
             "rerun_of": "rule_d_sma20_dev 2026-10-23T20:58:32 (b1dff46)",
             "notes": (f"D16 rule (a) re-run with the D19 training sign (D21); sign counts {sg.value_counts().to_dict()}; "
                       f"gross Sharpe {full['sharpe_ann_0bp']:.2f}; break-even {full['breakeven_bp_per_side']:.2f} bp/side; "
                       f"+1 bar delay net 3bp Sharpe {RMd.iloc[0]['sharpe_ann_3bp']:.2f}")})
    log_run({"config_hash": config_hash(p2.SETTINGS, {"metric": "D19"}), "phase": 2, "model": "phase2_feature_screen",
             "n_params": 0, "fold_scheme": "expanding monthly WF 2018-01..2025-12, cached per-fold fits; selection per fold",
             "data_version": load_config("data")["data_version"],
             "rerun_of": "phase2_feature_screen 2026-10-22T22:51:40 (b2fda20)",
             "notes": (f"re-scored with D19 (D21), no refits; D10 h12 single-feature passes: "
                       f"{int(WF[(WF['h'] == '12') & WF['d10_pass']].shape[0])}")})
    P1 = phase1_rescore(L)
    P1.to_csv(TAB / "phase1_preds_d19.csv", index=False)
    s = {"n_folds": len(folds), "fold_seconds_total": float(sum(r["seconds"] for r in folds.values())),
         "d10_pass_d19": {k: WF[(WF["h"] == k) & WF["d10_pass"]]["feature"].tolist() for k in SEL_KEYS},
         "d10_pass_old": {k: WO[(WO["h"] == k) & WO["d10_pass"]]["feature"].tolist() for k in SEL_KEYS},
         "selected_per_fold_v1": CH["n_v1"].describe().to_dict(), "selected_per_fold_v2": CH["n_v2"].describe().to_dict(),
         "folds_below_20_v2": int((CH["n_v2"] < 20).sum()),
         "rule_d_sma20_dev_d19": {"sign_counts": sg.value_counts().to_dict(),
                                  **RM.iloc[0][["sharpe_ann_0bp", "sharpe_ann_3bp", "breakeven_bp_per_side"]].to_dict(),
                                  **{k: RM.iloc[0][k] for k in RM.columns if k.startswith("d10_pass")}},
         "phase1_d19": P1.to_dict("records")}
    (TAB / "rescore_d19_summary.json").write_text(json.dumps(s, indent=1, default=str))
    print(json.dumps(s, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--time-one", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.time_one:
        t0 = time.time()
        init()
        t1 = time.time()
        sec = run_fold(MONTHS[-1])
        print(f"init {t1 - t0:.0f}s, one fold {sec:.0f}s -> projected {96 * sec / 60:.0f} min single, "
              f"{96 * sec / 60 / 4 * 1.3:.0f} min with 4 workers", flush=True)
    if a.workers:
        with Pool(a.workers, initializer=init) as pool:
            pool.map(run_fold, MONTHS, chunksize=1)
    if a.summarize or a.workers:
        if not G:
            init()
        summarize()
