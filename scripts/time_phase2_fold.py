"""Time one Phase 2 walk-forward fold end-to-end with per-fold refits (CLAUDE.md 10.1), and project the
96-fold total. Nothing is written except reports/tables/phase2_fold_timing.csv.

Per fold (validation month M, training = sessions before M):
  1. fit GARCH-t, HMM (5 EM starts), Kalman LLT MLE, frac-diff d on training rows
  2. apply the fitted objects (causal filters) to all rows
  3. ADF/KPSS on training rows (rolling-z retest for failures)
  4. monthly IC tables (fitted features recomputed; parameter-free features cached once)
  5. |Spearman| clustering, clustered MDA (LightGBM, purged 5-fold CV, h = 1/3/12)
  6. conditional up-probability cells + Benjamini-Hochberg, extreme-bucket rule, selection
Usage: python scripts/time_phase2_fold.py
"""
from __future__ import annotations

import importlib.util
import time
import warnings

import numpy as np
import pandas as pd

from tslaq.data.pipeline import choose_lambda, load_bars
from tslaq.features.build import FITTED, build_features, fit_params, fitted_features
from tslaq.features.normalize import rolling_z
from tslaq.features.returns import add_returns
from tslaq.features.volatility import add_sigma, seasonal_profile
from tslaq.labels.forward_returns import add_labels
from tslaq.selection.ic import delayed, monthly_ic
from tslaq.selection.importance import clustered_mda
from tslaq.selection.redundancy import cluster_features, spearman_matrix
from tslaq.selection.stationarity import stationarity
from tslaq.utils.io import ROOT, interim_path, load_config

warnings.filterwarnings("ignore")
spec = importlib.util.spec_from_file_location("p2", ROOT / "scripts" / "run_phase2_analysis.py")
p2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p2)
H = (1, 3, 12)


def main():
    timings = []
    t0 = time.time()
    cfg, wf = load_config("data"), load_config("backtest")["walk_forward"]
    v = cfg["volatility"]

    def base(sym, hz):
        raw = load_bars(sym)
        lam, _ = choose_lambda(raw, wf["first_val_month"], cfg)
        return add_sigma(add_returns(raw), lam, v["seasonal_window_sessions"], v["burn_in_sessions"], hz)

    L = add_labels(base("TSLA", tuple(range(1, 25))), tuple(range(1, 25)))
    mkt = {"spy": base("SPY", ()), "qqq": base("QQQ", ())}
    react = pd.read_parquet(interim_path("TSLA").with_name("TSLA_earnings_sessions.parquet"))["reaction_session"]
    timings.append({"step": "once: base frames + labels", "seconds": time.time() - t0})

    # parameter-free features: computed once (they do not depend on the fold); fitted ones per fold
    t = time.time()
    P0 = fit_params(L, "2018-01-01")
    F_all = build_features(L, mkt, [P0], react)
    free = [c for c in F_all.columns if c not in FITTED]
    timings.append({"step": "once: parameter-free features (+ one throwaway fit)", "seconds": time.time() - t})
    t = time.time()
    month = L["session_date"].dt.to_period("M").astype(str).to_numpy()
    sess = L["session_date"].to_numpy()
    Z = {h: L[f"z_{h}"].to_numpy() for h in range(1, 25)}
    for h in H:
        Z[f"d{h}"] = delayed(L[f"z_{h}"].to_numpy(), sess)
    Zs = {k: Z[k] for k in [*H, *[f"d{h}" for h in H]]}
    Frz_free = pd.DataFrame({f: rolling_z(F_all[f], p2.RZ_WINDOW) for f in free if f not in p2.CATEGORICAL})
    IC_free = pd.concat([monthly_ic(F_all[free], Z, month).assign(version="raw"),
                         monthly_ic(Frz_free, Zs, month).assign(version="rz")])
    timings.append({"step": "once: monthly IC of parameter-free features (raw h=1..24 + delayed, rz)", "seconds": time.time() - t})

    prof_df = seasonal_profile(L, v["seasonal_window_sessions"])
    prof, pos_s = prof_df.to_numpy(), prof_df.index.get_indexer(L["session_date"])
    cost = 2 * load_config("backtest")["costs"]["bps_per_side"] / 1e4
    for M in ["2018-01", "2025-12"]:
        cutoff = f"{M}-01"
        tr = sess < np.datetime64(cutoff)
        n_tr = int(tr.sum())
        step = {}

        def tick(name, t1):
            step[name] = time.time() - t1
            return time.time()

        t1 = time.time()
        P = fit_params(L, cutoff)
        t1 = tick("1 fit GARCH/HMM/Kalman/frac-d", t1)
        FF = fitted_features(L, P, prof, pos_s)
        F = pd.concat([F_all[free], FF], axis=1)
        t1 = tick("2 apply fitted filters", t1)
        Frz = pd.concat([Frz_free, pd.DataFrame({f: rolling_z(FF[f], p2.RZ_WINDOW) for f in FF if f not in p2.CATEGORICAL})], axis=1)
        versions = {}
        for f in F.columns:
            if f in p2.CATEGORICAL:
                versions[f] = "raw"
                continue
            if stationarity(F[f].to_numpy()[tr])["pass"]:
                versions[f] = "raw"
            elif stationarity(Frz[f].to_numpy()[tr])["pass"]:
                versions[f] = "rz"
        t1 = tick("3 ADF/KPSS", t1)
        trm = np.isin(month, pd.unique(month[tr]))
        fit_cont = [f for f in FF if f not in p2.CATEGORICAL]
        IC_fit = pd.concat([monthly_ic(FF[trm], {k: z[trm] for k, z in Zs.items()}, month[trm]).assign(version="raw"),
                            monthly_ic(Frz.loc[trm, fit_cont], {k: z[trm] for k, z in Zs.items()}, month[trm]).assign(version="rz")])
        IC = pd.concat([IC_free, IC_fit], ignore_index=True)
        IC["h"] = IC["h"].astype(str)
        T = p2.feature_ic_table(IC, versions, cutoff)
        t1 = tick("4 monthly IC (fitted feats) + IC stats", t1)
        X = pd.DataFrame({f: (Frz[f] if ver == "rz" else F[f]) for f, ver in versions.items()})
        clusters = cluster_features(spearman_matrix(X.iloc[np.flatnonzero(tr)[::6]]), 0.7)
        t1 = tick("5a clustering", t1)
        mda = {}
        for h in H:
            le = L[f"label_end_{h}"].to_numpy()
            z = L[f"z_{h}"].to_numpy()
            pos = np.flatnonzero(tr & np.isfinite(z) & (le >= 0))[:: p2.SETTINGS["mda"]["row_step"]]
            mda[h] = clustered_mda(X, z, le, pos, clusters, p2.SETTINGS["mda"]["k"])
        t1 = tick("5b clustered MDA (3 h x 5 CV folds)", t1)
        C, BY, edges, AUC = p2.condprob_cells(X, L, tr, cost)
        S = p2.select(T, p2.extreme_keep(C), clusters, mda)
        t1 = tick("6 condprob cells + BH + selection", t1)
        tot = sum(step.values())
        for k, s in step.items():
            timings.append({"step": f"fold {M}: {k}", "seconds": s, "n_train": n_tr})
        timings.append({"step": f"fold {M}: TOTAL", "seconds": tot, "n_train": n_tr})
        print(f"fold {M}: n_train={n_tr} total {tot:.0f}s  " + "  ".join(f"{k}={s:.0f}s" for k, s in step.items()),
              f"| selected {int(S['kept'].sum())}, clusters {clusters.nunique()}, BH {int(C['bh_sig'].sum())}/{len(C)}",
              flush=True)
    Tm = pd.DataFrame(timings)
    # projection: per-fold time linear in training rows between the two measured folds
    a = Tm[Tm["step"] == "fold 2018-01: TOTAL"].iloc[0]
    b = Tm[Tm["step"] == "fold 2025-12: TOTAL"].iloc[0]
    sizes = []
    for M in pd.period_range("2018-01", "2025-12", freq="M").astype(str):
        sizes.append(int((sess < np.datetime64(f"{M}-01")).sum()))
    sizes = np.array(sizes)
    slope = (b["seconds"] - a["seconds"]) / (b["n_train"] - a["n_train"])
    proj = float(np.sum(a["seconds"] + slope * (sizes - a["n_train"])))
    once = float(Tm[Tm["step"].str.startswith("once")]["seconds"].sum())
    Tm = pd.concat([Tm, pd.DataFrame([{"step": "PROJECTED 96 folds (linear in n_train)", "seconds": proj},
                                      {"step": "PROJECTED total incl. one-time steps", "seconds": proj + once}])])
    Tm.to_csv(ROOT / "reports" / "tables" / "phase2_fold_timing.csv", index=False)
    print(Tm.to_string(index=False))


if __name__ == "__main__":
    main()
