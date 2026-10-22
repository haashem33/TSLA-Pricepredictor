"""Phase 2 walk-forward: per-fold refits over the 96 monthly folds (validation 2018-01 .. 2025-12).

Fold M (training = sessions before M-01, CLAUDE.md 10.1): fit GARCH-t / HMM / Kalman / frac-d on training
rows -> apply with fixed parameters -> ADF/KPSS (rolling-z retest) -> monthly IC on training months ->
clustering, clustered MDA, conditional up-probability cells + BH (also one-bar-delayed outcomes at h = 1, 3)
-> day-block bootstrap of every BH-surviving h = 12 cell -> the fold's own feature list. The out-of-sample
part is the signed IC of each feature in month M (sign from that fold's training IC; month-M feature values
from that fold's fitted objects). Each fold is checkpointed to data/processed/phase2_folds/<M>.pkl.

Usage: python scripts/run_phase2_walkforward.py            (resumes from checkpoints)
       python scripts/run_phase2_walkforward.py --summarize (summary tables only)
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings

import numpy as np
import pandas as pd
import yaml

from tslaq.backtest.metrics import fold_ic_stats
from tslaq.data.pipeline import choose_lambda, load_bars
from tslaq.features.build import CATEGORICAL, FITTED, build_features, fit_params, fitted_features
from tslaq.features.normalize import rolling_z
from tslaq.features.returns import add_returns
from tslaq.features.volatility import add_sigma, seasonal_profile
from tslaq.labels.forward_returns import add_labels
from tslaq.selection import phase2 as p2
from tslaq.selection.condprob import session_weights
from tslaq.selection.ic import delayed, monthly_ic
from tslaq.selection.importance import clustered_mda
from tslaq.selection.redundancy import cluster_features, spearman_matrix
from tslaq.selection.stationarity import stationarity
from tslaq.utils.io import ROOT, config_hash, interim_path, load_config
from tslaq.utils.registry import log_run

warnings.filterwarnings("ignore")
H = (1, 3, 12)
DECAY_H = tuple(range(1, 25))
MONTHS = [str(p) for p in pd.period_range("2018-01", "2025-12", freq="M")]
CK = ROOT / "data" / "processed" / "phase2_folds"
TAB = ROOT / "reports" / "tables"
KEYS = [*H, *[f"d{h}" for h in H]]


def prepare():
    cfg, wf = load_config("data"), load_config("backtest")["walk_forward"]
    v = cfg["volatility"]

    def base(sym, hz):
        raw = load_bars(sym)
        lam, _ = choose_lambda(raw, wf["first_val_month"], cfg)
        return add_sigma(add_returns(raw), lam, v["seasonal_window_sessions"], v["burn_in_sessions"], hz), lam

    L, lam = base("TSLA", DECAY_H)
    L = add_labels(L, DECAY_H)
    sess = L["session_date"].to_numpy()
    for h in H:  # one-bar-delayed outcomes: from t+1 to t+1+h, aligned to t
        L[f"fwdd_{h}"] = delayed(L[f"fwd_{h}"].to_numpy(), sess)
        L[f"zd_{h}"] = delayed(L[f"z_{h}"].to_numpy(), sess)
    (spy, ls), (qqq, lq) = base("SPY", ()), base("QQQ", ())
    react = pd.read_parquet(interim_path("TSLA").with_name("TSLA_earnings_sessions.parquet"))["reaction_session"]
    P0 = fit_params(L, "2018-01-01")  # only used to get the parameter-free columns; fitted ones are per fold
    F_all = build_features(L, {"spy": spy, "qqq": qqq}, [P0], react)
    free = [c for c in F_all.columns if c not in FITTED]
    month = L["session_date"].dt.to_period("M").astype(str).to_numpy()
    Z = {h: L[f"z_{h}"].to_numpy() for h in DECAY_H}
    for h in H:
        Z[f"d{h}"] = L[f"zd_{h}"].to_numpy()
    Frz_free = pd.DataFrame({f: rolling_z(F_all[f], p2.RZ_WINDOW) for f in free if f not in CATEGORICAL})
    IC_free = pd.concat([monthly_ic(F_all[free], Z, month).assign(version="raw"),
                         monthly_ic(Frz_free, {k: Z[k] for k in KEYS}, month).assign(version="rz")], ignore_index=True)
    IC_free["h"] = IC_free["h"].astype(str)
    return {"L": L, "F_free": F_all[free], "Frz_free": Frz_free, "IC_free": IC_free, "month": month, "Z": Z,
            "lambdas": {"TSLA": lam, "SPY": ls, "QQQ": lq}, "react": react}


def kalman_diag(P: dict, L: pd.DataFrame, cutoff: str) -> dict:
    """Fitted Kalman variances and bound checks. L-BFGS-B bounds are log(var(dy)) - 30 .. + 5 (see
    fit_kalman_llt); 'near zero' = variance < 1e-8 x var(dy)."""
    y = np.log(L.loc[L["session_date"] < pd.Timestamp(cutoff), "close"].to_numpy())[-60000:]
    dv = float(np.var(np.diff(y)))
    k = P["kalman"]
    out = {"var_dy": dv, "converged": k.get("converged"), "loglik": k.get("loglik")}
    for name, val in zip(["irregular", "level", "trend"], k["params"]):
        lx = np.log(val)
        out[f"s2_{name}"] = val
        out[f"ratio_{name}"] = val / dv
        out[f"at_lower_bound_{name}"] = bool(lx <= np.log(dv) - 30 + 1e-3)
        out[f"at_upper_bound_{name}"] = bool(lx >= np.log(dv) + 5 - 1e-3)
        out[f"near_zero_{name}"] = bool(val < 1e-8 * dv)
    return out


def run_fold(M: str, D: dict, cost: float, prof, pos_s, report: bool) -> dict:
    t0 = time.time()
    L, month = D["L"], D["month"]
    cutoff = f"{M}-01"
    tr = L["session_date"].to_numpy() < np.datetime64(cutoff)
    P = fit_params(L, cutoff)
    FF = fitted_features(L, P, prof, pos_s)
    F = pd.concat([D["F_free"], FF], axis=1)
    fit_cont = [f for f in FF if f not in CATEGORICAL]
    Frz = pd.concat([D["Frz_free"], pd.DataFrame({f: rolling_z(FF[f], p2.RZ_WINDOW) for f in fit_cont})], axis=1)
    stat, versions = [], {}
    for f in F.columns:
        if f in CATEGORICAL:
            versions[f] = "raw"
            continue
        s = stationarity(F[f].to_numpy()[tr])
        row = {"feature": f, "adf_p": s["adf_p"], "kpss_p": s["kpss_p"], "raw_pass": s["pass"]}
        if s["pass"]:
            versions[f] = "raw"
        else:
            s2 = stationarity(Frz[f].to_numpy()[tr])
            row.update({"rz_adf_p": s2["adf_p"], "rz_kpss_p": s2["kpss_p"], "rz_pass": s2["pass"]})
            if s2["pass"]:
                versions[f] = "rz"
        row["version"] = versions.get(f, "drop")
        stat.append(row)
    # monthly IC of the fitted features on training months and month M (fold-M parameters)
    upto = month <= M
    keys = DECAY_H + tuple(KEYS[3:]) if report else KEYS
    Zk = {k: D["Z"][k][upto] for k in keys}
    IC_fit = pd.concat([monthly_ic(FF[upto], Zk, month[upto]).assign(version="raw"),
                        monthly_ic(Frz.loc[upto, fit_cont], {k: D["Z"][k][upto] for k in KEYS}, month[upto])
                        .assign(version="rz")], ignore_index=True)
    IC_fit["h"] = IC_fit["h"].astype(str)
    IC = pd.concat([D["IC_free"][D["IC_free"]["month"] <= M], IC_fit], ignore_index=True)
    T = p2.feature_ic_table(IC, versions, cutoff)
    # out-of-sample: signed IC in month M; sign from this fold's training IC
    oos = []
    ICM = IC[IC["month"] == M].set_index(["feature", "version", "h"])["ic"]
    for r in T.itertuples():
        for key, sign_src in [(str(r.h), r.ic), (f"d{r.h}", r.d_ic)]:
            v = ICM.get((r.feature, r.version, key), np.nan)
            if np.isfinite(v) and np.isfinite(sign_src):
                oos.append({"month": M, "feature": r.feature, "version": r.version, "h": key,
                            "ic": v, "sign": np.sign(sign_src), "signed_ic": np.sign(sign_src) * v})
    X = pd.DataFrame({f: (Frz[f] if v == "rz" else F[f]) for f, v in versions.items()})
    clusters = cluster_features(spearman_matrix(X.iloc[np.flatnonzero(tr)[::6]]), p2.SETTINGS["cluster_threshold"])
    mda = {}
    for h in H:
        le = L[f"label_end_{h}"].to_numpy()
        z = L[f"z_{h}"].to_numpy()
        pos = np.flatnonzero(tr & np.isfinite(z) & (le >= 0))[:: p2.SETTINGS["mda"]["row_step"]]
        mda[h] = clustered_mda(X, z, le, pos, clusters, p2.SETTINGS["mda"]["k"])
    C, BY, edges, AUC = p2.condprob_cells(X, L, tr, cost)
    Cd, _, _, _ = p2.condprob_cells(X, L, tr, cost, horizons=(1, 3), prefix="d", full=False)
    K = p2.extreme_keep(C)
    S = p2.select(T, K, clusters, mda)
    S["version"] = S["feature"].map(versions)
    SH = p2.shapes(C)
    # day-block bootstrap of every BH-surviving h = 12 cell (training sessions only)
    codes, _ = pd.factorize(L["session_date"])
    W = session_weights(int(L.loc[tr, "session_date"].nunique()), 1000, seed=int(M.replace("-", "")))
    B12 = p2.bootstrap_cells(C[(C["h"] == 12) & C["bh_sig"]], X, L, tr, edges, W, codes)
    out = {
        "month": M, "seconds": time.time() - t0, "n_train": int(tr.sum()),
        "kalman": kalman_diag(P, L, cutoff), "garch": P["garch"], "hmm": P["hmm"].as_dict(), "hmm_runs": P["hmm_runs"],
        "frac_d": P["frac_d"], "stationarity": pd.DataFrame(stat), "versions": versions, "T": T, "oos": pd.DataFrame(oos),
        "clusters": clusters, "mda": pd.concat(mda, names=["h"]), "S": S, "shapes": SH,
        "bh_counts": C.groupby("h")["bh_sig"].agg(["sum", "size"]).reset_index(),
        "bh_counts_delayed": Cd.groupby("h")["bh_sig"].agg(["sum", "size"]).reset_index(),
        "C": C, "Cd": Cd[Cd["bh_sig"]], "B12": B12, "AUC": AUC,
    }
    if report:
        out["report"] = {"X": X, "C": C, "BY": BY, "edges": edges, "versions": versions, "IC": IC, "F": F}
    return out


def summarize(D: dict | None):
    folds = {}
    for M in MONTHS:
        f = CK / f"{M}.pkl"
        if f.exists():
            with open(f, "rb") as fh:
                folds[M] = pickle.load(fh)
    n = len(folds)
    print(f"{n} folds loaded")
    # 1. Kalman variances per fold
    KD = pd.DataFrame([{"fold": M, **r["kalman"], "frac_d": r["frac_d"], "garch_alpha": r["garch"]["alpha"],
                        "garch_beta": r["garch"]["beta"], "garch_nu": r["garch"]["nu"],
                        "hmm_var_u": "/".join(f"{c[0][0]:.2f}" for c in r["hmm"]["covs"]),
                        "hmm_starts_ll_range": max(x["loglik"] for x in r["hmm_runs"]) - min(x["loglik"] for x in r["hmm_runs"]),
                        "seconds": r["seconds"]} for M, r in folds.items()])
    KD.to_csv(TAB / "phase2_fitted_params_by_fold.csv", index=False)
    # 2. selection per fold
    SEL = pd.concat([r["S"].assign(fold=M) for M, r in folds.items()], ignore_index=True)
    SEL.to_csv(TAB / "phase2_selected_by_fold.csv", index=False)
    kept = SEL[SEL["kept"]]
    share = kept.groupby("feature")["fold"].nunique() / n
    OOS = pd.concat([r["oos"] for r in folds.values()], ignore_index=True)
    OOS.to_csv(TAB / "phase2_oos_signed_ic_by_fold.csv", index=False)
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
    WF.to_csv(TAB / "phase2_walkforward_summary.csv", index=False)
    SHP = pd.concat([r["shapes"].assign(fold=M) for M, r in folds.items()], ignore_index=True)
    sh12 = SHP[SHP["h"] == 12]
    modal = sh12.groupby("feature")["shape"].agg(lambda s: s.value_counts().index[0])
    modal_share = sh12.groupby("feature")["shape"].agg(lambda s: s.value_counts().iloc[0] / len(s))
    last = folds[MONTHS[-1]] if MONTHS[-1] in folds else folds[list(folds)[-1]]
    last_shape = last["shapes"][last["shapes"]["h"] == 12].set_index("feature")["shape"]
    basis = kept.groupby("feature")["basis"].agg(lambda s: ", ".join(f"{k} {v}" for k, v in s.value_counts().items()))
    last_reason = kept.sort_values("fold").groupby("feature")["reason"].last()
    last_fold_sel = kept.sort_values("fold").groupby("feature")["fold"].last()
    w12 = WF[WF["h"] == "12"].set_index("feature")
    w12d = WF[WF["h"] == "d12"].set_index("feature")
    FS = pd.DataFrame({"share_folds_selected": share}).join(
        w12[["ic_full", "t_full", "ic_2325", "t_2325", "d10_pass"]]).join(
        w12d[["ic_full", "t_full", "d10_pass"]].add_prefix("delay_")).assign(
        shape_h12_modal=modal, shape_h12_modal_share=modal_share, shape_h12_last_fold=last_shape,
        basis_counts=basis, last_selected_fold=last_fold_sel, reason_last_selected=last_reason)
    FS = FS[FS["share_folds_selected"].notna()].sort_values("share_folds_selected", ascending=False)
    FS.to_csv(TAB / "phase2_feature_selection_summary.csv")
    # 3. BH counts by horizon (+ delayed), and bootstrap of h = 12 survivors
    BH = pd.concat([r["bh_counts"].assign(fold=M, outcome="no delay") for M, r in folds.items()]
                   + [r["bh_counts_delayed"].assign(fold=M, outcome="1-bar delay") for M, r in folds.items()],
                   ignore_index=True)
    BH.to_csv(TAB / "phase2_bh_counts_by_fold.csv", index=False)
    B12 = pd.concat([r["B12"].assign(fold=M) for M, r in folds.items() if len(r["B12"])], ignore_index=True) \
        if any(len(r["B12"]) for r in folds.values()) else pd.DataFrame()
    B12.to_csv(TAB / "phase2_h12_bh_cells_bootstrap_by_fold.csv", index=False)
    # bounce check: no-delay h = 1 / 3 survivors that are not significant with the delayed outcome
    bounce = []
    for M, r in folds.items():
        C, Cd = r["C"], r["Cd"]
        for h in (1, 3):
            nd = C[(C["h"] == h) & C["bh_sig"]][["feature", "bucket"]]
            dd = Cd[Cd["h"] == h][["feature", "bucket"]]
            both = nd.merge(dd, on=["feature", "bucket"])
            bounce.append({"fold": M, "h": h, "sig_no_delay": len(nd), "sig_delayed": len(dd), "sig_both": len(both)})
    BO = pd.DataFrame(bounce)
    BO.to_csv(TAB / "phase2_bh_bounce_overlap_by_fold.csv", index=False)
    # Phase 4 feature lists: one per fold (never a single list chosen across folds)
    cfg = {"generated": time.strftime("%Y-%m-%d"), "data_version": load_config("data")["data_version"],
           "note": ("Feature selection is per fold. Phase 4 must use by_fold[<validation month>] for that fold; no "
                    "single list chosen across folds may be used. 'rz' = rolling z-score over the previous "
                    f"{p2.RZ_WINDOW} bars (used when the raw feature failed ADF/KPSS on that fold's training data)."),
           "settings": {k: v for k, v in p2.SETTINGS.items() if k != "mda"} | {"mda_k": p2.SETTINGS["mda"]["k"]},
           "by_fold": {M: [{"feature": x.feature, "version": x.version, "basis": x.basis}
                           for x in r["S"][r["S"]["kept"]].itertuples()] for M, r in folds.items()}}
    with open(ROOT / "configs" / "features.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True, width=200)
    agg = BH.groupby(["outcome", "h"])["sum"].describe()[["mean", "min", "50%", "max"]]
    summary = {
        "n_folds": n,
        "n_features_built": int(len(last["versions"]) + (last["stationarity"]["version"] == "drop").sum()),
        "selected_per_fold": kept.groupby("fold").size().describe().to_dict(),
        "n_distinct_selected": int(kept["feature"].nunique()),
        "cells_per_fold_last": last["bh_counts"].set_index("h")["size"].to_dict(),
        "bh_last_fold": last["bh_counts"].set_index("h")["sum"].to_dict(),
        "bh_last_fold_delayed": last["bh_counts_delayed"].set_index("h")["sum"].to_dict(),
        "bh_by_outcome_h": {f"{a} h{h}": v for (a, h), v in agg.round(1).to_dict("index").items()},
        "h12_bh_cells_total": int(len(B12)), "h12_bh_cells_boot_hold": int(B12["boot_holds"].sum()) if len(B12) else 0,
        "kalman_any_bound": int(KD.filter(like="at_").to_numpy().any(axis=1).sum()),
        "kalman_any_near_zero": int(KD.filter(like="near_zero").to_numpy().any(axis=1).sum()),
        "kalman_not_converged": int((~KD["converged"].astype(bool)).sum()),
        "d10_pass_h12": WF[(WF["h"] == "12") & WF["d10_pass"]]["feature"].tolist(),
        "d10_pass_d12": WF[(WF["h"] == "d12") & WF["d10_pass"]]["feature"].tolist(),
        "n_features_tested_h12": int((WF["h"] == "12").sum()),
        "fold_seconds_total": float(KD["seconds"].sum()),
    }
    (TAB / "phase2_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))
    return folds, summary


def report_extras(folds, D):
    """Training-window tables/figures for the last fold (training 2016-01 .. 2025-11)."""
    r = folds[MONTHS[-1]]["report"]
    cost = 2 * load_config("backtest")["costs"]["bps_per_side"] / 1e4
    B, PG, IMR, SLT, DEC = p2.stage_report(r, r["F"], D["L"], r["IC"], cost)
    SLT.to_csv(TAB / "phase2_slot_ic_summary.csv", index=False)
    folds[MONTHS[-1]]["T"].to_csv(TAB / "phase2_feature_ic_train_lastfold.csv", index=False)
    folds[MONTHS[-1]]["AUC"].to_csv(TAB / "phase2_auc_mi_lastfold.csv", index=False)
    folds[MONTHS[-1]]["shapes"].to_csv(TAB / "phase2_shapes_lastfold.csv", index=False)
    folds[MONTHS[-1]]["stationarity"].to_csv(TAB / "phase2_stationarity_lastfold.csv", index=False)
    folds[MONTHS[-1]]["mda"].to_csv(TAB / "phase2_mda_lastfold.csv")
    folds[MONTHS[-1]]["clusters"].to_csv(TAB / "phase2_clusters_lastfold.csv")
    sel = folds[MONTHS[-1]]["S"]
    p2.decay_figure(DEC, list(sel[sel["kept"]]["feature"][:12]), p2.FIG / "phase2_ic_decay.png")
    print(f"report extras: bootstrap {len(B)}, pair cells {int(PG['sufficient'].sum())} "
          f"(BH {int(PG['bh_sig_pairs'].sum())})")


def main(summ_only: bool):
    CK.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    D = prepare()
    print(f"prepared {time.time() - t0:.0f}s", flush=True)
    if not summ_only:
        cost = 2 * load_config("backtest")["costs"]["bps_per_side"] / 1e4
        prof_df = seasonal_profile(D["L"], load_config("data")["volatility"]["seasonal_window_sessions"])
        prof, pos_s = prof_df.to_numpy(), prof_df.index.get_indexer(D["L"]["session_date"])
        for M in MONTHS:
            path = CK / f"{M}.pkl"
            if path.exists():
                continue
            r = run_fold(M, D, cost, prof, pos_s, report=(M == MONTHS[-1]))
            with open(path, "wb") as fh:
                pickle.dump(r, fh)
            print(f"{M}: {r['seconds']:.0f}s n_train={r['n_train']} kept={int(r['S']['kept'].sum())} "
                  f"BH={r['bh_counts']['sum'].tolist()} BHd={r['bh_counts_delayed']['sum'].tolist()} "
                  f"h12boot={int(r['B12']['boot_holds'].sum()) if len(r['B12']) else 0}/{len(r['B12'])} "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)
    folds, summary = summarize(D)
    report_extras(folds, D)
    (TAB / "phase2_lambdas.json").write_text(json.dumps(D["lambdas"]))
    if not summ_only:
        log_run({"config_hash": config_hash(p2.SETTINGS, load_config("data")), "phase": 2,
                 "model": "phase2_feature_screen", "n_params": 0,
                 "fold_scheme": "expanding monthly WF 2018-01..2025-12, per-fold refits; selection per fold",
                 "data_version": load_config("data")["data_version"],
                 "notes": (f"{summary['n_features_built']} features x 3 h screened per fold; per-feature signed "
                           f"monthly OOS IC; D10 h12 single-feature passes: {len(summary['d10_pass_h12'])}")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summarize", action="store_true")
    main(ap.parse_args().summarize)
