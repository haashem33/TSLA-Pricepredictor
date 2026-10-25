"""Phase 3: per-fold tokenizer, tokenizer checks, token condprob tables, and Gate 3 (D15 + D23).

Real walk-forward (96 folds 2018-01..2025-12; holdout never loaded): per fold M the tokenizer is fit on sessions
before M and saved (data/processed/phase3/tokenizers/<M>.pkl); tokenizer checks; token conditional up-probability
cells on the training rows (h = 1, 3, 12, with and without the one-bar delay); LightGBM A / B / C for h = 1, 3, 12
(fixed Phase 1 parameters, num_threads = 1); the first fold also predicts its D6 inner-validation block
(2017-11..2017-12) to seed V_s. Checkpoint per fold: data/processed/phase3/real/<M>.pkl.

Model null (D23): random-sign mirror of TSLA (seed 3000 + i) plus the D22 calibrated drift on intraday bars;
SPY/QQQ real; parameter-free features recomputed, GARCH-t / HMM / Kalman / frac-diff with the LAST fold's
(2025-12) fitted parameters in every fold (approximation); tokenizer refit per fold on the null data (HMM token from
the same last-fold HMM); each fold's real feature list and versions; one variant at h = 12, scored on z_12 and the
delayed z_12. Result per draw: data/processed/phase3/null_<variant>/draw_<seed>.parquet.

Usage: python scripts/run_phase3_gate3.py --time-one                 (last fold, prints projected real-run time)
       python scripts/run_phase3_gate3.py --real --workers 5
       python scripts/run_phase3_gate3.py --null C --draws 0 0 --workers 5   (timed first null draw)
       python scripts/run_phase3_gate3.py --summarize [--log]
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
from phase2_metric_null import mirror
from phase2_metric_null_v2 import add_drift

from tslaq.backtest.metrics import fold_ic_v2_model
from tslaq.data.pipeline import load_bars
from tslaq.features.build import fitted_features
from tslaq.models import gate3 as g3
from tslaq.models.baselines import LGB_PARAMS, lgb_n_params
from tslaq.selection import rescore as rs
from tslaq.selection.token_condprob import token_cells
from tslaq.utils.io import ROOT, config_hash, load_config
from tslaq.utils.registry import log_run

warnings.filterwarnings("ignore")
BASE = ROOT / "data" / "processed" / "phase3"
REAL = BASE / "real"
TOKDIR = BASE / "tokenizers"
CACHE = BASE / "null_cache"
TAB = ROOT / "reports" / "tables" / "phase3"
CAL = ROOT / "reports" / "tables" / "phase2_rescore" / "null_fallback_calibration.json"
SEED0 = 3000
G: dict = {}


def null_dir(v: str) -> Path:
    return BASE / f"null_{v}"


# ------------------------------------------------------------------ real walk-forward

def init_real():
    G["fr"] = g3.real_frame()
    G["lists"] = g3.by_fold_lists()


def real_job(M: str) -> float:
    path = REAL / f"{M}.pkl"
    if path.exists():
        return 0.0
    if "fr" not in G:
        init_real()
    fr = G["fr"]
    R = g3.fold_predictions(fr, M, G["lists"][M], inner=(M == g3.MONTHS[0]), tok_dir=TOKDIR)
    t1 = time.time()
    checks = g3.tokenizer_checks(fr, M, R["tok"], R["Xin"], R["T"])
    L = fr["L"]
    tr = fr["sess"] < np.datetime64(f"{M}-01")
    year = L["session_date"].dt.year.to_numpy() if M == g3.MONTHS[-1] else None
    cost = g3.load_config_costs()
    cells = []
    for h in g3.H:
        for d in (False, True):
            fwd = L[f"fwdd_{h}" if d else f"fwd_{h}"].to_numpy()
            z = L[f"zd_{h}" if d else f"z_{h}"].to_numpy()
            D = token_cells(R["T"], fwd, z, tr, h, cost, year)
            D["outcome"] = f"d{h}" if d else str(h)
            cells.append(D)
    out = {"month": M, "preds": R["preds"], "info": R["info"], "checks": checks,
           "cells": pd.concat(cells, ignore_index=True), "seconds_models": R["seconds"],
           "seconds_tables": time.time() - t1}
    REAL.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(out, fh)
    print(f"{M}: models {R['seconds']:.0f}s, checks+tables {out['seconds_tables']:.0f}s", flush=True)
    return R["seconds"] + out["seconds_tables"]


# ------------------------------------------------------------------ model null

def build_null(i: int) -> Path:
    """Null frame of draw i (seed 3000 + i), pickled for the fold workers."""
    if "lam" not in G:
        G["lam"] = rs.lambdas()
        G["mkt"] = rs.market_frames(G["lam"])
        G["react"] = rs.reaction_sessions()
        G["raw"] = load_bars("TSLA")
        G["P_last"] = rs.fold_params(g3.load_ckpt(g3.MONTHS[-1]))
        G["mu"] = json.loads(CAL.read_text())["mu"]
    raw = mirror(G["raw"], np.random.default_rng(SEED0 + i))
    raw = add_drift(raw, G["mu"], raw["bar_index"].to_numpy() > 0)
    L = rs.base_frame(raw, G["lam"]["TSLA"], (12,))
    F_free, Frz_free = rs.free_features(L, G["mkt"], G["react"])
    fr = g3.finish_frame(L, F_free, Frz_free, (12,))
    fr["FF"] = fitted_features(L, G["P_last"], fr["prof"], fr["pos"])
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"draw_{SEED0 + i}.pkl"
    with open(path, "wb") as fh:
        pickle.dump(fr, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def null_job(arg) -> pd.DataFrame:
    path, M, variant = arg
    if G.get("null_path") != path:
        with open(path, "rb") as fh:
            G["fr"] = pickle.load(fh)
        G["null_path"] = path
        G.setdefault("lists", g3.by_fold_lists())
    R = g3.fold_predictions(G["fr"], M, G["lists"][M], variants=(variant,), horizons=(12,),
                            inner=(M == g3.MONTHS[0]))
    return R["preds"]


def run_null(variant: str, draws: list[int], workers: int):
    od = null_dir(variant)
    od.mkdir(parents=True, exist_ok=True)
    with Pool(workers) as pool:
        for i in draws:
            out = od / f"draw_{SEED0 + i}.parquet"
            if out.exists():
                continue
            t0 = time.time()
            path = build_null(i)
            t1 = time.time()
            P = pd.concat(pool.map(null_job, [(path, M, variant) for M in g3.MONTHS], chunksize=1), ignore_index=True)
            with open(path, "rb") as fh:
                fr = pickle.load(fh)
            rows = {}
            for d in (False, True):
                rows["ic_d12" if d else "ic_12"] = g3.score_model(fr, P[P["h"] == 12], 12, d)
            R = pd.DataFrame(rows).rename_axis("fold").reset_index()
            R["seed"], R["variant"] = SEED0 + i, variant
            R["mean_z12"] = float(np.nanmean(fr["L"]["z_12"].to_numpy()))
            R["seconds"] = time.time() - t0
            R["seconds_frame"] = t1 - t0
            R.to_parquet(out)
            path.unlink()
            s = {k: g3.ic_summary(R.set_index("fold")[k]) for k in ("ic_12", "ic_d12")}
            print(f"null {variant} seed {SEED0 + i}: {time.time() - t0:.0f}s (frame {t1 - t0:.0f}s); "
                  f"h12 IC {s['ic_12']['ic_full']:+.4f} t {s['ic_12']['t_full']:+.2f}; "
                  f"d12 IC {s['ic_d12']['ic_full']:+.4f} t {s['ic_d12']['t_full']:+.2f}", flush=True)


# ------------------------------------------------------------------ summary

def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    w2 = wilcoxon(d) if len(d) > 1 else None
    w1 = wilcoxon(d, alternative="greater") if len(d) > 1 else None
    return {"n_folds": len(d), "mean_a": float(a[d.index].mean()), "mean_b": float(b[d.index].mean()),
            "mean_diff": float(d.mean()), "share_a_wins": float((d > 0).mean()),
            "wilcoxon_p_two_sided": float(w2.pvalue), "wilcoxon_p_greater": float(w1.pvalue)}


def phase1_ic(fr: dict, h: int, delayed: bool) -> pd.Series:
    """Phase 1 lgb_returns predictions (no retraining) re-scored with D19; fold 2018-01 only seeds V_s."""
    R = pd.read_parquet(ROOT / "data" / "processed" / "phase1_return_preds_v2.parquet")
    g = R[(R["model"] == "lgb_returns") & (R["h"] == h)].sort_values("pos")
    y = g3.outcome(fr, h, delayed)
    zbar, vy = g3.train_moments(fr, y, g3.MONTHS)
    f0 = g["fold"] == g3.MONTHS[0]
    s0 = g.loc[f0, "pred_z"].to_numpy() - zbar[g3.MONTHS[0]]
    ok0 = np.isfinite(y[g.loc[f0, "pos"].to_numpy()])
    sub = g[~f0]
    return fold_ic_v2_model(sub["pred_z"], y[sub["pos"].to_numpy()], sub["fold"], zbar, vy, float(np.mean(s0[ok0] ** 2)))


def summarize(log: bool):
    TAB.mkdir(parents=True, exist_ok=True)
    folds = {}
    for M in g3.MONTHS:
        with open(REAL / f"{M}.pkl", "rb") as fh:
            folds[M] = pickle.load(fh)
    fr = g3.real_frame()
    P = pd.concat([r["preds"] for r in folds.values()], ignore_index=True)
    # ---- Gate 3 real results
    fic, res = [], []
    mom = {(h, d): g3.train_moments(fr, g3.outcome(fr, h, d), [*g3.MONTHS, g3.INNER[0]]) for h in g3.H for d in (0, 1)}
    for v in g3.VARIANTS:
        for h in g3.H:
            Pv = P[(P["variant"] == v) & (P["h"] == h)]
            side = g3.side_metrics(fr, Pv, h)
            for d in (False, True):
                ic = g3.score_model(fr, Pv, h, d, mom[(h, int(d))])
                fic.append(pd.DataFrame({"variant": v, "h": h, "delayed": d, "fold": ic.index, "ic": ic.to_numpy()}))
                res.append({"variant": v, "h": h, "delayed": d, **g3.ic_summary(ic), **side})
    FIC = pd.concat(fic, ignore_index=True)
    FIC.to_csv(TAB / "gate3_fold_ic.csv", index=False)
    RES = pd.DataFrame(res)
    # ---- model null
    NUL, nd = [], {}
    for v in g3.VARIANTS:
        files = sorted(null_dir(v).glob("draw_*.parquet"))
        if not files:
            continue
        D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        nd[v] = D
        for seed, g in D.groupby("seed"):
            for k, d in (("ic_12", False), ("ic_d12", True)):
                NUL.append({"variant": v, "seed": seed, "delayed": d, "mean_z12": g["mean_z12"].iloc[0],
                            "seconds": g["seconds"].iloc[0], **g3.ic_summary(g.set_index("fold")[k])})
    NU = pd.DataFrame(NUL)
    NU.to_csv(TAB / "gate3_model_null_per_draw.csv", index=False)
    nsum = []
    for (v, d), g in NU.groupby(["variant", "delayed"]) if len(NU) else []:
        real = RES[(RES["variant"] == v) & (RES["h"] == 12) & (RES["delayed"] == d)].iloc[0]
        thr = g["t_full"].mean() + 2 * g["t_full"].std()
        nsum.append({"variant": v, "h": 12, "delayed": d, "K": len(g), "seeds": f"{g['seed'].min()}..{g['seed'].max()}",
                     "null_mean_ic": g["ic_full"].mean(), "null_mean_ic_se": g["ic_full"].std() / np.sqrt(len(g)),
                     "null_mean_t": g["t_full"].mean(), "null_sd_t": g["t_full"].std(), "null_t_threshold": thr,
                     "null_fixed_d10_rate": g["d10"].mean(), "null_mean_z12": g["mean_z12"].mean(),
                     "real_ic": real["ic_full"], "real_t": real["t_full"],
                     "null_ic_within_0.003": bool(abs(g["ic_full"].mean()) <= 0.003),
                     "real_t_above_threshold": bool(real["t_full"] >= thr),
                     "model_null_pass": bool(abs(g["ic_full"].mean()) <= 0.003 and real["t_full"] >= thr),
                     "mean_seconds_per_draw": g["seconds"].mean()})
    NS = pd.DataFrame(nsum)
    NS.to_csv(TAB / "gate3_model_null_summary.csv", index=False)
    ncols = ["K", "null_mean_ic", "null_mean_t", "null_sd_t", "null_t_threshold", "model_null_pass"]
    if len(NS):
        RES = RES.merge(NS[["variant", "delayed", *ncols]], on=["variant", "delayed"], how="left")
    for c in ncols:
        if c not in RES:
            RES[c] = np.nan
    RES["model_null_pass"] = RES["model_null_pass"].astype(object)
    RES.loc[RES["h"] != 12, ncols] = np.nan
    RES["null_status"] = np.where(RES["h"] != 12, "no null (h != 12)",
                                  np.where(RES["K"].notna(), "null run", "unconfirmed"))
    RES.to_csv(TAB / "gate3_results.csv", index=False)
    # ---- paired comparisons
    W = FIC.pivot_table(index="fold", columns=["variant", "h", "delayed"], values="ic")
    PR = []
    for h in g3.H:
        for d in (False, True):
            PR.append({"comparison": "C vs B", "h": h, "delayed": d, **paired(W[("C", h, d)], W[("B", h, d)])})
            PR.append({"comparison": "C vs A", "h": h, "delayed": d, **paired(W[("C", h, d)], W[("A", h, d)])})
            p1 = phase1_ic(fr, h, d)
            c = W[("C", h, d)].loc[p1.index]
            PR.append({"comparison": "C vs Phase 1 lgb_returns (D19)", "h": h, "delayed": d, **paired(c, p1),
                       "phase1_ic_full": g3.ic_summary(p1)["ic_full"], "phase1_t_full": g3.ic_summary(p1)["t_full"],
                       "phase1_t_2325": g3.ic_summary(p1)["t_2325"], "phase1_d10": g3.ic_summary(p1)["d10"]})
    PRD = pd.DataFrame(PR)
    PRD.to_csv(TAB / "gate3_paired.csv", index=False)
    # ---- tokenizer checks
    CH = pd.DataFrame([r["checks"] for r in folds.values()])
    CH.drop(columns=["ret_merges"]).to_csv(TAB / "tokenizer_checks_by_fold.csv", index=False)
    num = CH.drop(columns=["fold", "ret_merges"]).astype(float)
    CS = pd.DataFrame({"min": num.min(), "median": num.median(), "max": num.max(),
                       "fold_of_min": [CH.loc[num[c].idxmin(), "fold"] for c in num.columns]})
    CS.to_csv(TAB / "tokenizer_checks_summary.csv")
    # ---- token condprob
    C = pd.concat([r["cells"].assign(fold=M) for M, r in folds.items()], ignore_index=True)
    cells = C[C["cell"] != "_insufficient"]
    BHC = cells.groupby(["fold", "outcome", "family", "table"]).agg(n_cells=("cell", "size"), bh_sig=("bh_sig", "sum")
                                                                    ).reset_index()
    BHC.to_csv(TAB / "token_condprob_bh_by_fold.csv", index=False)
    last = C[C["fold"] == g3.MONTHS[-1]]
    last.to_csv(TAB / "token_condprob_lastfold.csv", index=False)
    sv = cells[cells["bh_sig"]].groupby(["outcome", "family", "table", "cell"]).agg(
        folds_bh=("fold", "nunique"), mean_lift=("lift", "mean"), first=("fold", "min"), last=("fold", "max")).reset_index()
    lt = last[last["cell"] != "_insufficient"].set_index(["outcome", "family", "table", "cell"])
    sv = sv.join(lt[["n", "n_eff", "p_up", "lift", "wilson_lo", "wilson_hi", "mean_fwd", "p_up_cost", "p_dn_cost",
                     "year_sign_share", "bh_sig"]].add_prefix("lastfold_"), on=["outcome", "family", "table", "cell"])
    sv = sv.sort_values(["outcome", "folds_bh"], ascending=[True, False])
    sv.to_csv(TAB / "token_condprob_summary.csv", index=False)
    ins = C[C["cell"] == "_insufficient"].groupby(["outcome", "table"])["n"].mean().rename("mean_insufficient_cells")
    ins.to_csv(TAB / "token_condprob_insufficient.csv")
    # ---- verdict (D23)
    def row(v, h, d):
        return RES[(RES["variant"] == v) & (RES["h"] == h) & (RES["delayed"] == d)].iloc[0]
    c12d = row("C", 12, True)
    cvb = PRD[(PRD["comparison"] == "C vs B") & (PRD["h"] == 12) & PRD["delayed"]].iloc[0]
    a12d = row("A", 12, True)
    c_better = bool(cvb["mean_diff"] > 0 and cvb["wilcoxon_p_greater"] < 0.05)
    a_null = bool(a12d["model_null_pass"]) if pd.notna(a12d["model_null_pass"]) else None
    c_null = bool(c12d["model_null_pass"]) if pd.notna(c12d["model_null_pass"]) else None
    a_fails = (not a12d["d10"]) or (a_null is False)
    verdict = {"gate3_pass": bool(c12d["d10"] and c_null is True),
               "C_h12_delayed_d10": bool(c12d["d10"]), "C_h12_delayed_model_null": c_null,
               "A_fails": bool(a_fails), "A_null_status": a12d["null_status"], "C_better_than_B": c_better,
               "stop_and_discuss_D15": bool(a_fails and not c_better)}
    s = {"verdict": verdict, "fold_seconds_models": float(sum(r["seconds_models"] for r in folds.values())),
         "fold_seconds_tables": float(sum(r["seconds_tables"] for r in folds.values())),
         "null_K": NS.groupby("variant")["K"].max().to_dict() if len(NS) else {},
         "lgb_params": g3.lgb_params()}
    (TAB / "gate3_summary.json").write_text(json.dumps(s, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(RES[["variant", "h", "delayed", "ic_full", "t_full", "ic_2325", "t_2325", "d10", "null_status",
               "null_mean_ic", "null_mean_t", "null_sd_t", "model_null_pass"]].round(4).to_string(index=False))
    print(PRD.round(4).to_string(index=False))
    print(json.dumps(s, indent=1, default=str))
    if log:
        log_trials(RES, NS)


def log_trials(RES: pd.DataFrame, NS: pd.DataFrame):
    dv = load_config("data")["data_version"]
    for v in g3.VARIANTS:
        for h in g3.H:
            a = RES[(RES["variant"] == v) & (RES["h"] == h) & ~RES["delayed"]].iloc[0]
            b = RES[(RES["variant"] == v) & (RES["h"] == h) & RES["delayed"]].iloc[0]
            null = ""
            if h == 12:
                n = NS[(NS["variant"] == v) & NS["delayed"]] if len(NS) else NS
                null = (f"; model null (d12) K={int(n['K'].iloc[0])} mean IC {n['null_mean_ic'].iloc[0]:+.4f} "
                        f"mean t {n['null_mean_t'].iloc[0]:+.2f} sd t {n['null_sd_t'].iloc[0]:.2f} pass "
                        f"{bool(n['model_null_pass'].iloc[0])}") if len(n) else "; model null: unconfirmed"
            log_run({"config_hash": config_hash(LGB_PARAMS, {"gate3": v, "h": h, "lags": 24, "threads": g3.THREADS}),
                     "phase": 3, "model": f"gate3_{v}_h{h}", "n_params": lgb_n_params(),
                     "fold_scheme": "expanding monthly WF 2018-01..2025-12, purge+embargo 78; tokenizer refit per fold",
                     f"rank_ic_h{h}": a["ic_full"], f"icir_h{h}": a["icir_full"], f"hit_rate_h{h}": a["hit_rate"],
                     f"crps_h{h}": a["crps_z"], f"brier_h{h}": a["brier"], "data_version": dv,
                     "notes": (f"D23 Gate 3; rank_ic column = D19 mean fold IC; t {a['t_full']:.2f}, 2023-25 IC "
                               f"{a['ic_2325']:+.4f} t {a['t_2325']:.2f}, D10 {a['d10']}; delayed IC {b['ic_full']:+.4f} "
                               f"t {b['t_full']:.2f}, 2023-25 {b['ic_2325']:+.4f} t {b['t_2325']:.2f}, D10 {b['d10']}{null}")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-one", action="store_true")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--null", choices=list(g3.VARIANTS))
    ap.add_argument("--draws", nargs=2, type=int)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    TOKDIR.mkdir(parents=True, exist_ok=True)
    if a.time_one:
        t0 = time.time()
        init_real()
        t1 = time.time()
        sec = real_job(g3.MONTHS[-1])
        n_tr = {M: int((G["fr"]["sess"] < np.datetime64(f"{M}-01")).sum()) for M in g3.MONTHS}
        proj = sum(sec * n_tr[M] / n_tr[g3.MONTHS[-1]] for M in g3.MONTHS)
        print(f"init {t1 - t0:.0f}s; last fold {sec:.0f}s -> projected {proj / 3600:.2f} h single worker, "
              f"{proj / 3600 / a.workers * 1.15:.2f} h with {a.workers} workers", flush=True)
    if a.real:
        with Pool(a.workers, initializer=init_real) as pool:
            pool.map(real_job, list(reversed(g3.MONTHS)), chunksize=1)
    if a.null:
        run_null(a.null, list(range(a.draws[0], a.draws[1] + 1)), a.workers)
    if a.summarize:
        summarize(a.log)
