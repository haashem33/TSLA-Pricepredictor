"""D17 validation B2: random-sign mirror null for all features (diagnostic, not a trial).

Each draw gives every TSLA bar an independent random sign. A minus sign mirrors the bar around the previous
close in log space: (o, h, l, c, vwap) relative to C_{t-1} -> (-o, -l, -h, -c, -vwap), so high and low swap.
Volume is kept. Prices are rebuilt from the first real price, so |returns|, ranges, the volatility clustering
and the price-level-like persistence of the features survive, but the sign of every future return is
independent of the past: there is no predictability by construction. SPY / QQQ are the real series.
All features are recomputed on the mirrored series: parameter-free ones exactly; GARCH-t, HMM, Kalman and
frac-diff with the already-fitted parameters of the LAST fold (2025-12), applied to all folds (no refits; an
approximation: the real run used each fold's own parameters). Each fold's feature versions (raw / rolling-z)
come from its checkpoint.

Scored on the 96 folds, h = 12 with and without the one-bar delay, under the old metric (Spearman pooled within
the month, biased, information only) and the D17 anchored IC; signs from each fold's training months with the
same metric. Per draw the number of features passing D10, and per feature the mean IC under each metric.

Usage: python scripts/phase2_metric_null.py --draws 0 1         (draw indices to run; resumes)
       python scripts/phase2_metric_null.py --workers 3 --draws 1 30
       python scripts/phase2_metric_null.py --summarize
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from tslaq.backtest.metrics import fold_ic_stats
from tslaq.data.pipeline import load_bars
from tslaq.features.build import CATEGORICAL, fitted_features
from tslaq.features.normalize import rolling_z
from tslaq.selection import rescore as rs
from tslaq.selection.ic import spearman_cols
from tslaq.utils.io import ROOT

warnings.filterwarnings("ignore")
CK = ROOT / "data" / "processed" / "phase2_folds"
OUT = ROOT / "data" / "processed" / "phase2_metric_null"
TAB = ROOT / "reports" / "tables" / "phase2_metric_fix"
MONTHS = [str(p) for p in pd.period_range("2018-01", "2025-12", freq="M")]
KEYS = ("12", "d12")
G: dict = {}


def mirror(raw: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Random-sign mirror of every bar around the previous close (log space); row 0 is kept."""
    C = raw["close"].to_numpy()
    cp = np.r_[np.nan, C[:-1]]
    rel = {c: np.log(raw[c].to_numpy() / cp) for c in ["open", "high", "low", "close", "vwap"]}
    s = rng.choice([-1.0, 1.0], len(raw))
    s[0] = 1.0
    neg = s < 0
    new = {"open": s * rel["open"], "close": s * rel["close"], "vwap": s * rel["vwap"],
           "high": np.where(neg, -rel["low"], rel["high"]), "low": np.where(neg, -rel["high"], rel["low"])}
    lc = np.log(C[0]) + np.r_[0.0, np.cumsum(new["close"][1:])]
    cpn = np.exp(np.r_[np.nan, lc[:-1]])
    out = raw.copy()
    for c, x in new.items():
        v = cpn * np.exp(x)
        v[0] = raw[c].iat[0]
        out[c] = v
    out["close"] = np.exp(lc)
    return out


def init():
    lam = rs.lambdas()
    folds = {}
    for M in MONTHS:
        with open(CK / f"{M}.pkl", "rb") as fh:
            r = pickle.load(fh)
        folds[M] = r["versions"] if M != MONTHS[-1] else r
    last = folds[MONTHS[-1]]
    G.update({"lam": lam, "mkt": rs.market_frames(lam), "react": rs.reaction_sessions(), "raw": load_bars("TSLA"),
              "versions": {M: (v["versions"] if M == MONTHS[-1] else v) for M, v in folds.items()},
              "P": rs.fold_params(last)})


def run_draw(i: int) -> float:
    path = OUT / f"draw_{i:03d}.parquet"
    if path.exists():
        return 0.0
    if not G:
        init()
    t0 = time.time()
    raw = mirror(G["raw"], np.random.default_rng(1000 + i))
    L = rs.base_frame(raw, G["lam"]["TSLA"], (12,))
    F_free, Frz_free = rs.free_features(L, G["mkt"], G["react"])
    prof, pos = rs.profile(L)
    FF = fitted_features(L, G["P"], prof, pos)
    Rw = pd.concat([F_free, FF], axis=1).replace([np.inf, -np.inf], np.nan)
    Rz = pd.concat([Frz_free, pd.DataFrame({f: rolling_z(FF[f], rs.RZ_WINDOW) for f in FF if f not in CATEGORICAL})],
                   axis=1).replace([np.inf, -np.inf], np.nan)
    month = L["session_date"].dt.to_period("M").astype(str).to_numpy()
    Z = {"12": L["z_12"].to_numpy(), "d12": L["zd_12"].to_numpy()}
    # old metric: monthly Spearman of every column (raw and rz) once per draw
    starts = np.r_[0, np.flatnonzero(month[1:] != month[:-1]) + 1]
    um, ends = month[starts], np.r_[starts[1:], len(month)]
    old = {}
    for ver, Fr in (("raw", Rw), ("rz", Rz)):
        X = Fr.to_numpy(float)
        for k, z in Z.items():
            old[(ver, k)] = pd.DataFrame([spearman_cols(X[a:b], z[a:b], 200) for a, b in zip(starts, ends)],
                                         index=um, columns=Fr.columns)
    rows = []
    for M in MONTHS:
        ver = G["versions"][M]
        X = pd.DataFrame({f: (Rz[f] if v == "rz" else Rw[f]) for f, v in ver.items()})
        IC, _, _, _ = rs.anchored_fold(X, Z, month, M)
        W = IC.pivot_table(index="month", columns=["h", "feature"], values="ic")
        for k in KEYS:
            A = W[k]
            a_m, a_sign = A.loc[M], np.sign(A[A.index < M].mean())
            for f, v in ver.items():
                O = old[(v, k)][f]
                rows.append((M, f, k, v, a_m.get(f, np.nan), a_sign.get(f, np.nan), O.loc[M],
                             np.sign(O[O.index < M].mean())))
    D = pd.DataFrame(rows, columns=["fold", "feature", "h", "version", "anch_ic", "anch_sign", "old_ic", "old_sign"])
    D["draw"] = i
    OUT.mkdir(parents=True, exist_ok=True)
    D.to_parquet(path)
    sec = time.time() - t0
    print(f"draw {i}: {sec:.0f}s", flush=True)
    return sec


def d10(s: pd.Series) -> bool:
    s = s.dropna()
    yr = pd.Index(s.index).str[:4].astype(int)
    a, b = fold_ic_stats(s), fold_ic_stats(s[yr >= 2023])
    return bool(a["t"] >= 2 and b["ic"] > 0 and b["t"] >= 1)


def nominal_rate(n_sim: int = 200_000, n: int = 96, seed: int = 0) -> float:
    """D10 pass rate for iid N(0, 1) fold ICs (96 folds, last 36 = 2023-2025)."""
    x = np.random.default_rng(seed).standard_normal((n_sim, n))
    t = x.mean(1) / x.std(1, ddof=1) * np.sqrt(n)
    r = x[:, -36:]
    tr = r.mean(1) / r.std(1, ddof=1) * 6
    return float(((t >= 2) & (r.mean(1) > 0) & (tr >= 1)).mean())


def summarize():
    files = sorted(OUT.glob("draw_*.parquet"))
    D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    n_draws = D["draw"].nunique()
    D["anch_signed"] = D["anch_ic"] * D["anch_sign"]
    D["old_signed"] = D["old_ic"] * D["old_sign"]
    per = []
    for (dr, f, k), g in D.groupby(["draw", "feature", "h"]):
        g = g.set_index("fold")
        per.append({"draw": dr, "feature": f, "h": k, "n_folds": int(g["anch_ic"].notna().sum()),
                    "anch_unsigned": g["anch_ic"].mean(), "anch_signed": g["anch_signed"].mean(),
                    "anch_t": fold_ic_stats(g["anch_signed"])["t"], "anch_d10": d10(g["anch_signed"]),
                    "old_unsigned": g["old_ic"].mean(), "old_signed": g["old_signed"].mean(),
                    "old_t": fold_ic_stats(g["old_signed"])["t"], "old_d10": d10(g["old_signed"])})
    P = pd.DataFrame(per)
    P.to_csv(TAB / "null_per_draw_feature.csv", index=False)
    nom = nominal_rate()
    cnt = P.groupby(["draw", "h"]).agg(n_features=("feature", "size"), anch_d10=("anch_d10", "sum"),
                                       old_d10=("old_d10", "sum")).reset_index()
    cnt["nominal_expected"] = cnt["n_features"] * nom
    cnt.to_csv(TAB / "null_d10_passes_per_draw.csv", index=False)
    feat = P.groupby(["feature", "h"]).agg(
        n_draws=("draw", "size"),
        anch_unsigned_mean=("anch_unsigned", "mean"), anch_unsigned_sd=("anch_unsigned", "std"),
        anch_signed_mean=("anch_signed", "mean"), anch_signed_sd=("anch_signed", "std"),
        anch_d10_share=("anch_d10", "mean"),
        old_unsigned_mean=("old_unsigned", "mean"), old_unsigned_sd=("old_unsigned", "std"),
        old_signed_mean=("old_signed", "mean"), old_signed_sd=("old_signed", "std"),
        old_d10_share=("old_d10", "mean")).reset_index()
    for m in ("anch", "old"):
        # pass condition (B): null mean within +-0.005 of 0, or within 1 null sd (sd across draws)
        for kind in ("unsigned", "signed"):
            mu, sd = feat[f"{m}_{kind}_mean"], feat[f"{m}_{kind}_sd"]
            feat[f"{m}_{kind}_ok"] = ((mu.abs() <= 0.005) | (mu.abs() <= sd)).where(mu.notna())
    feat.to_csv(TAB / "null_per_feature.csv", index=False)
    s = {"n_draws": n_draws, "nominal_d10_rate_iid": nom,
         "d10_passes_per_draw": {k: {m: {"mean": float(g[f"{m}_d10"].mean()), "min": int(g[f"{m}_d10"].min()),
                                         "max": int(g[f"{m}_d10"].max())} for m in ("anch", "old")}
                                 | {"n_features": int(g["n_features"].iloc[0]),
                                    "nominal_expected": float(g["nominal_expected"].iloc[0])}
                                 for k, g in cnt.groupby("h")},
         "features_failing_mean_condition": {
             k: {f"{m}_{kind}": g.loc[g[f"{m}_{kind}_ok"] == False, "feature"].tolist()
                 for m in ("anch", "old") for kind in ("unsigned", "signed")} for k, g in feat.groupby("h")},
         "not_scoreable": {k: g.loc[g["anch_unsigned_mean"].isna(), "feature"].tolist() for k, g in feat.groupby("h")}}
    (TAB / "null_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    pd.set_option("display.width", 250)
    print(feat.sort_values(["h", "anch_unsigned_mean"]).round(4).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", nargs=2, type=int, default=None, help="first and last draw index (inclusive)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.draws:
        idx = list(range(a.draws[0], a.draws[1] + 1))
        if a.workers > 1:
            with Pool(a.workers, initializer=init) as pool:
                pool.map(run_draw, idx, chunksize=1)
        else:
            init()
            for i in idx:
                run_draw(i)
    if a.summarize:
        summarize()
