"""D25 baseline confirmation at h = 12 (pre-registered in reports/decisions.md before any model was trained).

Real walk-forward (96 folds 2018-01..2025-12; holdout never loaded): M1 (Phase 1 lgb_returns, retrained), M2
(ridge, alpha from the inner block), M3 (equal-weight D19 composite), M4 (CatBoost), h = 1, 3, 12; the first
fold also predicts its D6 inner-validation block. Checkpoint per fold: data/processed/phase3_baselines/real/<M>.pkl.

Model null: the D23 construction (seed 3000 + i, mirror + D22 drift, last-fold fitted parameters), h = 12, each
model run separately on the same null frame so each has its own timing. Result per model and draw:
data/processed/phase3_baselines/null_<model>/draw_<seed>.parquet.

Usage: python scripts/run_phase3_baselines.py --real --workers 5
       python scripts/run_phase3_baselines.py --null M1 M2 M3 M4 --draws 0 0 --workers 5   (timed first draw)
       python scripts/run_phase3_baselines.py --summarize [--log]
       python scripts/run_phase3_baselines.py --backtest [--log]
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
from scipy.stats import norm, wilcoxon

sys.path.insert(0, str(Path(__file__).parent))
import run_phase3_gate3 as rp3

from tslaq.backtest.metrics import crps_gaussian, fold_ic
from tslaq.models import baselines_d25 as b
from tslaq.models import gate3 as g3
from tslaq.models.baselines import LGB_PARAMS, lgb_n_params
from tslaq.utils.io import ROOT, config_hash, load_config
from tslaq.utils.registry import log_run

warnings.filterwarnings("ignore")
BASE = ROOT / "data" / "processed" / "phase3_baselines"
REAL = BASE / "real"
TAB = ROOT / "reports" / "tables" / "phase3_baselines"
G: dict = {}
COSTS_BP = (0.0, 0.5, 1.0, 3.0, 6.0)


def null_dir(m: str) -> Path:
    return BASE / f"null_{m}"


# ------------------------------------------------------------------ real walk-forward

def init_real():
    G["fr"] = g3.real_frame()
    G["lists"] = b.by_fold_basis()


def real_job(M: str) -> float:
    path = REAL / f"{M}.pkl"
    if path.exists():
        return 0.0
    if "fr" not in G:
        init_real()
    R = b.fold_predictions(G["fr"], M, G["lists"], inner=(M == g3.MONTHS[0]))
    REAL.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(R, fh)
    print(f"{M}: {R['seconds']:.0f}s " + " ".join(f"{k} {v:.0f}s" for k, v in R["seconds_by_model"].items()),
          flush=True)
    return R["seconds"]


# ------------------------------------------------------------------ model null

def null_frame(i: int) -> Path:
    path = rp3.CACHE / f"draw_{rp3.SEED0 + i}.pkl"
    return path if path.exists() else rp3.build_null(i)  # reuse a cached null frame if one exists


def null_job(arg) -> pd.DataFrame:
    path, M, model = arg
    if G.get("null_path") != path:
        with open(path, "rb") as fh:
            G["fr"] = pickle.load(fh)
        G["null_path"] = path
        G.setdefault("lists", b.by_fold_basis())
    R = b.fold_predictions(G["fr"], M, G["lists"], models=(model,), horizons=(12,), inner=(M == g3.MONTHS[0]))
    return R["preds"]


def run_null(models: list[str], draws: list[int], workers: int):
    with Pool(workers) as pool:
        for i in draws:
            seed = rp3.SEED0 + i
            todo = [m for m in models if not (null_dir(m) / f"draw_{seed}.parquet").exists()]
            if not todo:
                continue
            t0 = time.time()
            path = null_frame(i)
            with open(path, "rb") as fh:
                fr = pickle.load(fh)
            tf = time.time() - t0
            for m in todo:
                t1 = time.time()
                P = pd.concat(pool.map(null_job, [(path, M, m) for M in g3.MONTHS], chunksize=1), ignore_index=True)
                rows = {("ic_d12" if d else "ic_12"): b.score(fr, P, m, 12, d) for d in (False, True)}
                R = pd.DataFrame(rows).rename_axis("fold").reset_index()
                R["seed"], R["model"] = seed, m
                R["mean_z12"] = float(np.nanmean(fr["L"]["z_12"].to_numpy()))
                R["seconds"] = time.time() - t1
                R["seconds_frame"] = tf
                null_dir(m).mkdir(parents=True, exist_ok=True)
                R.to_parquet(null_dir(m) / f"draw_{seed}.parquet")
                s = {k: g3.ic_summary(R.set_index("fold")[k]) for k in ("ic_12", "ic_d12")}
                print(f"null {m} seed {seed}: {R['seconds'].iloc[0]:.0f}s (frame {tf:.0f}s); "
                      f"h12 IC {s['ic_12']['ic_full']:+.4f} t {s['ic_12']['t_full']:+.2f}; "
                      f"d12 IC {s['ic_d12']['ic_full']:+.4f} t {s['ic_d12']['t_full']:+.2f}", flush=True)
            path.unlink()


# ------------------------------------------------------------------ summary

def paired(a: pd.Series, c: pd.Series) -> dict:
    d = (a - c).dropna()
    return {"n_folds": len(d), "mean_first": float(a[d.index].mean()), "mean_second": float(c[d.index].mean()),
            "mean_diff": float(d.mean()), "share_first_wins": float((d > 0).mean()),
            "wilcoxon_p_two_sided": float(wilcoxon(d).pvalue),
            "wilcoxon_p_greater": float(wilcoxon(d, alternative="greater").pvalue)}


def dm_day(loss_diff: np.ndarray, sess: np.ndarray) -> tuple[float, float]:
    """Diebold-Mariano t on a loss differential with day-clustered errors (daily sums, iid across days)."""
    D = pd.Series(loss_diff).groupby(sess).agg(["sum", "size"])
    n = D["size"].sum()
    mean = D["sum"].sum() / n
    se = np.sqrt(((D["sum"] - mean * D["size"]) ** 2).sum()) / n
    t = mean / se
    return float(t), float(2 * norm.sf(abs(t)))


def side(fr: dict, P: pd.DataFrame, model: str, h: int) -> dict:
    Q = P[P["fold"] != "inner"]
    pos = Q["pos"].to_numpy()
    z = fr["L"][f"z_{h}"].to_numpy()[pos]
    fwd = fr["L"][f"fwd_{h}"].to_numpy()[pos]
    p = Q["pred"].to_numpy()
    m = np.isfinite(p) & (p != 0) & np.isfinite(fwd) & (fwd != 0)
    out = {"hit_rate": float(np.mean(np.sign(p[m]) == np.sign(fwd[m]))), "long_share": float(np.mean(p[m] > 0)),
           "spearman_monthly_biased": float(fold_ic(p, z, Q["fold"].to_numpy()).mean())}
    if model != "M3":
        c = crps_gaussian(p, Q["sd"].to_numpy(), z)
        c0 = crps_gaussian(np.zeros_like(p), Q["sd0"].to_numpy(), z)
        ok = np.isfinite(c) & np.isfinite(c0)
        t, pv = dm_day(c[ok] - c0[ok], fr["sess"][pos][ok])
        out |= {"crps_z": float(c[ok].mean()), "crps_zero": float(c0[ok].mean()),
                "crps_diff": float((c[ok] - c0[ok]).mean()), "crps_dm_t": t, "crps_dm_p": pv}
    return out


def load_real() -> pd.DataFrame:
    P, info = [], []
    for M in g3.MONTHS:
        with open(REAL / f"{M}.pkl", "rb") as fh:
            R = pickle.load(fh)
        P.append(R["preds"])
        info += R["info"]
        G.setdefault("secs", []).append({"fold": M, "seconds": R["seconds"], **R["seconds_by_model"]})
    G["info"] = info
    return pd.concat(P, ignore_index=True)


def summarize(log: bool):
    TAB.mkdir(parents=True, exist_ok=True)
    fr = g3.real_frame()
    P = load_real()
    pd.DataFrame(G["secs"]).to_csv(TAB / "real_fold_seconds.csv", index=False)
    info = pd.DataFrame([i for i in G["info"] if i["model"] == "M2"])
    info.to_csv(TAB / "m2_alpha_by_fold.csv", index=False)
    m3 = pd.DataFrame([{k: v for k, v in i.items() if k != "signs"} for i in G["info"] if i["model"] == "M3"])
    m3.to_csv(TAB / "m3_features_by_fold.csv", index=False)
    # ---- real results
    fic, res = [], []
    for m in b.MODELS:
        for h in g3.H:
            Pm = P[(P["model"] == m) & (P["h"] == h)]
            sd = side(fr, Pm, m, h)
            for d in (False, True):
                ic = b.score(fr, Pm, m, h, d)
                fic.append(pd.DataFrame({"model": m, "h": h, "delayed": d, "fold": ic.index, "ic": ic.to_numpy()}))
                res.append({"model": m, "h": h, "delayed": d, **g3.ic_summary(ic), **sd})
    FIC = pd.concat(fic, ignore_index=True)
    FIC.to_csv(TAB / "baseline_fold_ic.csv", index=False)
    RES = pd.DataFrame(res)
    # ---- model null
    NUL = []
    for m in b.MODELS:
        for f in sorted(null_dir(m).glob("draw_*.parquet")):
            g = pd.read_parquet(f)
            for k, d in (("ic_12", False), ("ic_d12", True)):
                NUL.append({"model": m, "seed": int(g["seed"].iloc[0]), "delayed": d,
                            "mean_z12": g["mean_z12"].iloc[0], "seconds": g["seconds"].iloc[0],
                            **g3.ic_summary(g.set_index("fold")[k])})
    NU = pd.DataFrame(NUL)
    nsum = []
    for (m, d), g in NU.groupby(["model", "delayed"]) if len(NU) else []:
        real = RES[(RES["model"] == m) & (RES["h"] == 12) & (RES["delayed"] == d)].iloc[0]
        thr = g["t_full"].mean() + 2 * g["t_full"].std()
        nsum.append({"model": m, "h": 12, "delayed": d, "K": len(g), "complete": len(g) == 20,
                     "seeds": f"{g['seed'].min()}..{g['seed'].max()}", "null_mean_ic": g["ic_full"].mean(),
                     "null_mean_ic_se": g["ic_full"].std() / np.sqrt(len(g)), "null_mean_t": g["t_full"].mean(),
                     "null_sd_t": g["t_full"].std(), "null_t_threshold": thr,
                     "null_fixed_d10_rate": g["d10"].mean(), "null_mean_z12": g["mean_z12"].mean(),
                     "real_ic": real["ic_full"], "real_t": real["t_full"],
                     "null_ic_within_0.003": bool(abs(g["ic_full"].mean()) <= 0.003),
                     "real_t_above_threshold": bool(real["t_full"] >= thr),
                     "model_null_pass": bool(len(g) == 20 and abs(g["ic_full"].mean()) <= 0.003
                                             and real["t_full"] >= thr),
                     "mean_seconds_per_draw": g["seconds"].mean()})
    NS = pd.DataFrame(nsum)
    if len(NU):
        NU = NU.merge(NS[["model", "delayed", "null_t_threshold"]], on=["model", "delayed"])
        NU["gate_pass_in_null"] = NU["d10"] & (NU["t_full"] >= NU["null_t_threshold"])
    NU.to_csv(TAB / "baseline_model_null_per_draw.csv", index=False)
    NS.to_csv(TAB / "baseline_model_null_summary.csv", index=False)
    ncols = ["K", "null_mean_ic", "null_mean_t", "null_sd_t", "null_t_threshold", "model_null_pass"]
    RES = RES.merge(NS[["model", "delayed", *ncols]], on=["model", "delayed"], how="left") if len(NS) else RES
    for c in ncols:
        if c not in RES:
            RES[c] = np.nan
    RES["model_null_pass"] = RES["model_null_pass"].astype(object)
    RES.loc[RES["h"] != 12, ncols] = np.nan
    RES["null_status"] = np.where(RES["h"] != 12, "no null (h != 12)",
                                  np.where(RES["K"] == 20, "null run (K = 20)", "unconfirmed"))
    RES["gate_pass"] = (RES["h"] == 12) & RES["delayed"] & RES["d10"] & (RES["model_null_pass"] == True)
    RES.to_csv(TAB / "baseline_results.csv", index=False)
    # ---- family-wise check (information)
    fw = {}
    if len(NU):
        D = NU[NU["delayed"]].pivot_table(index="seed", columns="model", values="gate_pass_in_null", aggfunc="first")
        fw = {"n_seeds": len(D), "models": list(D.columns), "seeds_any_pass": int(D.any(axis=1).sum()),
              "share_any_pass": float(D.any(axis=1).mean()), "per_model_pass": D.sum().astype(int).to_dict()}
    # ---- paired tests: each model vs M1 and vs Gate-3 B
    W = FIC.pivot_table(index="fold", columns=["model", "h", "delayed"], values="ic")
    GB = pd.read_csv(ROOT / "reports" / "tables" / "phase3" / "gate3_fold_ic.csv")
    GB = GB[GB["variant"] == "B"].pivot_table(index="fold", columns=["h", "delayed"], values="ic")
    PR = []
    for h in g3.H:
        for d in (False, True):
            for m in b.MODELS:
                if m != "M1":
                    PR.append({"comparison": f"{m} vs M1", "h": h, "delayed": d, **paired(W[(m, h, d)], W[("M1", h, d)])})
                PR.append({"comparison": f"{m} vs Gate-3 B", "h": h, "delayed": d,
                           **paired(W[(m, h, d)], GB[(h, d)])})
    PRD = pd.DataFrame(PR)
    PRD.to_csv(TAB / "baseline_paired.csv", index=False)
    # ---- M1 vs stored Phase 1 predictions
    ph1 = pd.read_parquet(ROOT / "data" / "processed" / "phase1_return_preds_v2.parquet")
    ph1 = ph1[ph1["model"] == "lgb_returns"]
    mt = []
    for h in g3.H:
        a = P[(P["model"] == "M1") & (P["h"] == h) & (P["fold"] != "inner")][["pos", "pred"]]
        c = ph1[ph1["h"] == h][["pos", "pred_z"]]
        j = a.merge(c, on="pos")
        mt.append({"h": h, "n_d25": len(a), "n_phase1": len(c), "n_common": len(j),
                   "corr": float(np.corrcoef(j["pred"], j["pred_z"])[0, 1]),
                   "max_abs_diff": float((j["pred"] - j["pred_z"]).abs().max()),
                   "mean_abs_diff": float((j["pred"] - j["pred_z"]).abs().mean()),
                   "share_identical_1e-9": float(((j["pred"] - j["pred_z"]).abs() < 1e-9).mean())})
    MT = pd.DataFrame(mt)
    MT.to_csv(TAB / "m1_vs_phase1.csv", index=False)
    # ---- decision rule (D25)
    gate = RES[(RES["h"] == 12) & RES["delayed"]].set_index("model")
    passing = gate[gate["gate_pass"]]
    best_all = gate["t_full"].idxmax()
    verdict = {"passing_models": list(passing.index),
               "phase4_baseline": (passing["t_full"].idxmax() if len(passing) else None),
               "decision": ("Phase 4 may proceed" if len(passing) else
                            "no model passes: recommendation is to conclude the project with a final report (user decides)"),
               "best_model_by_t_h12_delayed": best_all, "family_wise": fw,
               "null_status": {m: ("K = 20" if (len(NS) and ((NS["model"] == m) & NS["complete"]).any()) else "unconfirmed")
                               for m in b.MODELS}}
    (TAB / "baseline_summary.json").write_text(json.dumps(verdict, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(RES[["model", "h", "delayed", "ic_full", "t_full", "ic_2325", "t_2325", "d10", "null_mean_ic",
               "null_mean_t", "null_sd_t", "null_t_threshold", "model_null_pass", "gate_pass"]].round(4).to_string(index=False))
    print(PRD.round(4).to_string(index=False))
    print(MT.round(4).to_string(index=False))
    print(json.dumps(verdict, indent=1, default=str))
    if log:
        log_trials(RES, NS)


def n_params(m: str) -> int:
    if m == "M1":
        return lgb_n_params()
    if m == "M2":
        return 1 + max(len(v) for v in b.by_fold_basis().values())
    if m == "M3":
        return 0
    return b.CAT_PARAMS["iterations"] * 2 ** b.CAT_PARAMS["depth"]


def log_trials(RES: pd.DataFrame, NS: pd.DataFrame):
    dv = load_config("data")["data_version"]
    cfg = {"M1": LGB_PARAMS | {"num_threads": 1}, "M2": {"alphas": list(b.ALPHAS), "clip": b.STD_CLIP},
           "M3": {"basis": ["IC", "both"]}, "M4": b.CAT_PARAMS}
    for m in b.MODELS:
        for h in g3.H:
            a = RES[(RES["model"] == m) & (RES["h"] == h) & ~RES["delayed"]].iloc[0]
            d = RES[(RES["model"] == m) & (RES["h"] == h) & RES["delayed"]].iloc[0]
            null = ""
            if h == 12:
                n = NS[(NS["model"] == m) & NS["delayed"]] if len(NS) else NS
                null = (f"; model null (d12) K={int(n['K'].iloc[0])} mean IC {n['null_mean_ic'].iloc[0]:+.4f} mean t "
                        f"{n['null_mean_t'].iloc[0]:+.2f} sd t {n['null_sd_t'].iloc[0]:.2f} thr "
                        f"{n['null_t_threshold'].iloc[0]:.2f} pass {bool(n['model_null_pass'].iloc[0])}; "
                        f"D25 gate {bool(d['gate_pass'])}") if len(n) else "; model null: unconfirmed"
            row = {"config_hash": config_hash(cfg[m], {"d25": m, "h": h}), "phase": 3, "model": f"d25_{m}_h{h}",
                   "n_params": n_params(m), "fold_scheme": "expanding monthly WF 2018-01..2025-12, purge+embargo 78 (D23 rows)",
                   f"rank_ic_h{h}": a["ic_full"], f"icir_h{h}": a["icir_full"], f"hit_rate_h{h}": a["hit_rate"],
                   "data_version": dv,
                   "notes": (f"D25 baseline; rank_ic column = D19 mean fold IC; t {a['t_full']:.2f}, 2023-25 IC "
                             f"{a['ic_2325']:+.4f} t {a['t_2325']:.2f}, D10 {a['d10']}; delayed IC {d['ic_full']:+.4f} "
                             f"t {d['t_full']:.2f}, 2023-25 {d['ic_2325']:+.4f} t {d['t_2325']:.2f}, D10 {d['d10']}{null}")}
            if m != "M3":
                row[f"crps_h{h}"] = a["crps_z"]
            log_run(row)


# ------------------------------------------------------------------ break-even rule backtest (information)

def session_arrays(L: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    sess = L["session_date"].to_numpy()
    pos = np.arange(len(L))
    last = pd.Series(pos).groupby(sess).transform("max").to_numpy()
    return sess, last


def rule_trades(L: pd.DataFrame, last: np.ndarray, pos: np.ndarray, pred: np.ndarray, h: int, d: int) -> pd.DataFrame:
    """Sign rule: signal at bar t -> enter at the open of t+1+d, exit at the close of t+h+d; non-overlapping
    trades within a session (next signal bar >= previous + h); no entry in the last 10 minutes (two 5-min bars);
    exit no later than the session's last bar."""
    O, C = L["open"].to_numpy(), L["close"].to_numpy()
    sess = L["session_date"].to_numpy()
    o = np.argsort(pos)
    pos, pred = pos[o], pred[o]
    rows, nxt, cur = [], -1, None
    for t, p in zip(pos, pred):
        if sess[t] != cur:
            cur, nxt = sess[t], -1
        e, x = t + 1 + d, t + h + d
        if t < nxt or not np.isfinite(p) or p == 0 or x > last[t] or e > last[t] - 2:
            continue
        sg = 1.0 if p > 0 else -1.0
        rows.append((t, sess[t], sg, sg * np.log(C[x] / O[e])))
        nxt = t + h
    return pd.DataFrame(rows, columns=["pos", "session", "side", "gross"])


def backtest_table(L: pd.DataFrame, P: pd.DataFrame, name: str, h: int) -> list[dict]:
    sess, last = session_arrays(L)
    Q = P[P["fold"] != "inner"]
    days_all = pd.Index(np.unique(sess[Q["pos"].to_numpy()]))
    out = []
    for d in (0, 1):
        T = rule_trades(L, last, Q["pos"].to_numpy(), Q["pred"].to_numpy(), h, d)
        for per, lo in (("2018-2025", 2018), ("2023-2025", 2023)):
            days = days_all[pd.DatetimeIndex(days_all).year >= lo]
            Tp = T[pd.DatetimeIndex(T["session"]).year >= lo]
            gross = Tp.groupby("session")["gross"].sum().reindex(days, fill_value=0.0)
            ntr = Tp.groupby("session").size().reindex(days, fill_value=0).astype(float)
            r = {"model": name, "h": h, "extra_delay": d, "period": per, "n_days": len(days), "n_trades": len(Tp),
                 "trades_per_day": float(ntr.mean()), "long_share": float((Tp["side"] > 0).mean()),
                 "trade_hit_rate": float((Tp["gross"] > 0).mean()), "mean_trade_gross_bp": float(Tp["gross"].mean() * 1e4),
                 "break_even_bp_per_side": float(gross.mean() / (2 * ntr.mean()) * 1e4)}
            for c in COSTS_BP:
                net = gross - 2 * c / 1e4 * ntr
                tag = "gross" if c == 0 else f"net_{c:g}bp"
                r[f"sharpe_{tag}"] = float(net.mean() / net.std() * np.sqrt(252))
                r[f"t_{tag}"] = float(net.mean() / net.std() * np.sqrt(len(net)))
                r[f"ann_ret_{tag}"] = float(net.mean() * 252)
            out.append(r)
    return out


def backtest(log: bool):
    TAB.mkdir(parents=True, exist_ok=True)
    L = g3.real_frame()["L"]
    summ = json.loads((TAB / "baseline_summary.json").read_text())
    best = summ["best_model_by_t_h12_delayed"]
    P = load_real()
    GB = []
    for M in g3.MONTHS:
        with open(rp3.REAL / f"{M}.pkl", "rb") as fh:
            GB.append(pickle.load(fh)["preds"])
    GB = pd.concat(GB, ignore_index=True)
    GB = GB[GB["variant"] == "B"]
    rows = []
    for name, src in (("Gate-3 B", GB), (best, P[P["model"] == best])):
        for h in g3.H:
            t0 = time.time()
            rows += backtest_table(L, src[src["h"] == h], name, h)
            print(f"backtest {name} h={h}: {time.time() - t0:.0f}s", flush=True)
    BT = pd.DataFrame(rows)
    BT.to_csv(TAB / "breakeven_rule_backtest.csv", index=False)
    pd.set_option("display.width", 250)
    print(BT[["model", "h", "extra_delay", "period", "trades_per_day", "long_share", "break_even_bp_per_side",
              "sharpe_gross", "sharpe_net_0.5bp", "sharpe_net_1bp", "sharpe_net_3bp", "sharpe_net_6bp"]].round(3).to_string(index=False))
    if log:
        bc = load_config("backtest")["costs"]["bps_per_side"]
        for name in ("Gate-3 B", best):
            for h in g3.H:
                g = BT[(BT["model"] == name) & (BT["h"] == h)].set_index(["extra_delay", "period"])
                a = g.loc[(0, "2018-2025")]
                k = f"net_{bc:g}bp"
                log_run({"config_hash": config_hash({"rule": "sign", "model": name, "h": h, "costs": list(COSTS_BP)}),
                         "phase": 3, "model": f"d25_rule_{name.replace(' ', '').replace('-', '')}_h{h}",
                         "fold_scheme": "out-of-sample predictions 2018-01..2025-12; non-overlapping sign rule, fill next open",
                         "net_sharpe_period": a[f"sharpe_{k}"] / np.sqrt(252), "net_sharpe_ann": a[f"sharpe_{k}"],
                         "turnover": a["trades_per_day"], "sharpe_2x_costs": a[f"sharpe_net_{2 * bc:g}bp"],
                         "data_version": load_config("data")["data_version"],
                         "notes": (f"D25 (i) information only; break-even {a['break_even_bp_per_side']:.2f} bp/side; gross "
                                   f"Sharpe {a['sharpe_gross']:.2f}; 2023-25 break-even "
                                   f"{g.loc[(0, '2023-2025'), 'break_even_bp_per_side']:.2f} bp; +1 bar delay break-even "
                                   f"{g.loc[(1, '2018-2025'), 'break_even_bp_per_side']:.2f} bp")})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--null", nargs="+", choices=list(b.MODELS))
    ap.add_argument("--draws", nargs=2, type=int)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--log", action="store_true")
    a = ap.parse_args()
    if a.real:
        with Pool(a.workers, initializer=init_real) as pool:
            pool.map(real_job, list(reversed(g3.MONTHS)), chunksize=1)
    if a.null:
        run_null(a.null, list(range(a.draws[0], a.draws[1] + 1)), a.workers)
    if a.summarize:
        summarize(a.log)
    if a.backtest:
        backtest(a.log)
