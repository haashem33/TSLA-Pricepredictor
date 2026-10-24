"""D20 criterion 3: random-sign mirror null, and mirror null + real drift, scored with the D19 anchored IC v2
(diagnostic, not a trial).

Same construction as scripts/phase2_metric_null.py (same mirror, seeds 1000 + i, all features recomputed,
GARCH-t / HMM / Kalman / frac-diff with the last fold's (2025-12) fitted parameters for all folds, each fold's
feature versions from its checkpoint). Variant "drift": after mirroring, the constant real 2016-2025 mean log
return per bar (all bars, overnight bar included) is added to every bar, i.e. every price of row t is multiplied
by exp(mu * t). Scored on the 96 folds at h = 12 with and without the one-bar delay; sign_train = sign of the
mean D19 IC over the training months (same anchors).

D20 criterion 3, applied to each variant and to each target (h = 12 and h = 12 delayed; all four must pass):
every feature's mean unsigned IC over draws within +-0.003 or within 3 se of 0; pooled D10 false-pass rate over
all scoreable features and draws <= 2.2%; no single feature above 8%.

Usage: python scripts/phase2_metric_null_v2.py --variant nodrift --workers 5 --draws 0 59
       python scripts/phase2_metric_null_v2.py --variant drift --workers 5 --draws 0 39
       python scripts/phase2_metric_null_v2.py --summarize
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import phase2_metric_null as p1

from tslaq.backtest.metrics import fold_ic_stats
from tslaq.features.build import CATEGORICAL, fitted_features
from tslaq.features.normalize import rolling_z
from tslaq.selection import rescore as rs
from tslaq.utils.io import ROOT

warnings.filterwarnings("ignore")
OUT = {v: ROOT / "data" / "processed" / f"phase2_metric_null_v2_{v}" for v in ("nodrift", "drift")}
TAB = ROOT / "reports" / "tables" / "phase2_metric_fix_v2"
MIN_DRAWS = {"nodrift": 60, "drift": 40}
TOL, MAX_POOLED, MAX_FEATURE = 0.003, 0.022, 0.08
G = p1.G


def real_mu(raw: pd.DataFrame) -> float:
    """Mean log return per bar over all bars (2016-2025, overnight bar included)."""
    c = np.log(raw["close"].to_numpy())
    return float((c[-1] - c[0]) / (len(c) - 1))


def add_drift(m: pd.DataFrame, mu: float) -> pd.DataFrame:
    out = m.copy()
    f = np.exp(mu * np.arange(len(m)))
    for c in ["open", "high", "low", "close", "vwap"]:
        out[c] = m[c].to_numpy() * f
    return out


def run_draw(arg) -> float:
    i, variant = arg
    path = OUT[variant] / f"draw_{i:03d}.parquet"
    if path.exists():
        return 0.0
    if not G:
        p1.init()
    t0 = time.time()
    raw = p1.mirror(G["raw"], np.random.default_rng(1000 + i))
    if variant == "drift":
        raw = add_drift(raw, real_mu(G["raw"]))
    L = rs.base_frame(raw, G["lam"]["TSLA"], (12,))
    F_free, Frz_free = rs.free_features(L, G["mkt"], G["react"])
    prof, pos = rs.profile(L)
    FF = fitted_features(L, G["P"], prof, pos)
    Rw = pd.concat([F_free, FF], axis=1).replace([np.inf, -np.inf], np.nan)
    Rz = pd.concat([Frz_free, pd.DataFrame({f: rolling_z(FF[f], rs.RZ_WINDOW) for f in FF if f not in CATEGORICAL})],
                   axis=1).replace([np.inf, -np.inf], np.nan)
    month = L["session_date"].dt.to_period("M").astype(str).to_numpy()
    Z = {"12": L["z_12"].to_numpy(), "d12": L["zd_12"].to_numpy()}
    zmean = float(np.nanmean(Z["12"]))
    rows = []
    for M in p1.MONTHS:
        ver = G["versions"][M]
        X = pd.DataFrame({f: (Rz[f] if v == "rz" else Rw[f]) for f, v in ver.items()})
        IC, _, _ = rs.anchored_fold_v2(X, Z, month, M)
        W = IC.pivot_table(index="month", columns=["h", "feature"], values="ic", dropna=False)
        for k in p1.KEYS:
            A = W[k]
            a_m, a_sign = A.loc[M], np.sign(A[A.index < M].mean())
            for f, v in ver.items():
                rows.append((M, f, k, v, a_m.get(f, np.nan), a_sign.get(f, np.nan)))
    D = pd.DataFrame(rows, columns=["fold", "feature", "h", "version", "ic", "sign"])
    D["draw"], D["variant"], D["mean_z12"] = i, variant, zmean
    OUT[variant].mkdir(parents=True, exist_ok=True)
    D.to_parquet(path)
    sec = time.time() - t0
    print(f"{variant} draw {i}: {sec:.0f}s", flush=True)
    return sec


def summarize():
    TAB.mkdir(parents=True, exist_ok=True)
    nom = p1.nominal_rate()
    per, crit = [], {}
    for variant, out in OUT.items():
        files = sorted(out.glob("draw_*.parquet"))
        if not files:
            continue
        D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        D["signed"] = D["ic"] * D["sign"]
        for (dr, f, k), g in D.groupby(["draw", "feature", "h"]):
            g = g.set_index("fold")
            per.append({"variant": variant, "draw": dr, "feature": f, "h": k,
                        "n_folds": int(g["ic"].notna().sum()), "unsigned": g["ic"].mean(),
                        "signed": g["signed"].mean(), "t": fold_ic_stats(g["signed"])["t"],
                        "d10": p1.d10(g["signed"]), "mean_z12": g["mean_z12"].iloc[0]})
    P = pd.DataFrame(per)
    P.to_csv(TAB / "null_v2_per_draw_feature.csv", index=False)
    P = P[P["n_folds"] > 0]  # not scoreable at h = 12 (r_12th30, sign_12th30)
    feat = P.groupby(["variant", "h", "feature"]).agg(
        n_draws=("draw", "size"), unsigned_mean=("unsigned", "mean"), unsigned_sd=("unsigned", "std"),
        signed_mean=("signed", "mean"), signed_sd=("signed", "std"), d10_rate=("d10", "mean")).reset_index()
    feat["unsigned_se"] = feat["unsigned_sd"] / np.sqrt(feat["n_draws"])
    feat["mean_ok"] = (feat["unsigned_mean"].abs() <= TOL) | (feat["unsigned_mean"].abs() <= 3 * feat["unsigned_se"])
    feat["d10_ok"] = feat["d10_rate"] <= MAX_FEATURE
    feat.to_csv(TAB / "null_v2_per_feature.csv", index=False)
    cnt = P.groupby(["variant", "h", "draw"]).agg(n_features=("feature", "size"), d10=("d10", "sum")).reset_index()
    cnt.to_csv(TAB / "null_v2_d10_passes_per_draw.csv", index=False)
    for (variant, k), g in feat.groupby(["variant", "h"]):
        n_draws = int(g["n_draws"].max())
        pooled = float(P[(P["variant"] == variant) & (P["h"] == k)]["d10"].mean())
        c = {"n_draws": n_draws, "enough_draws": n_draws >= MIN_DRAWS[variant],
             "mean_z12": float(P[P["variant"] == variant]["mean_z12"].mean()),
             "n_features": len(g), "features_failing_mean": g.loc[~g["mean_ok"], "feature"].tolist(),
             "max_abs_unsigned_mean": float(g["unsigned_mean"].abs().max()),
             "pooled_d10_rate": pooled, "pooled_d10_ok": pooled <= MAX_POOLED,
             "max_feature_d10_rate": float(g["d10_rate"].max()),
             "max_feature_d10": g.loc[g["d10_rate"].idxmax(), "feature"],
             "features_above_8pct": g.loc[~g["d10_ok"], "feature"].tolist(),
             "mean_d10_per_draw": float(cnt[(cnt["variant"] == variant) & (cnt["h"] == k)]["d10"].mean())}
        c["pass"] = bool(c["enough_draws"] and not c["features_failing_mean"] and c["pooled_d10_ok"]
                         and not c["features_above_8pct"])
        crit[f"{variant}_{k}"] = c
    s = {"nominal_d10_rate_iid": nom, "tolerance": TOL, "max_pooled_d10": MAX_POOLED,
         "max_feature_d10": MAX_FEATURE, "criteria": crit, "criterion_3_pass": all(c["pass"] for c in crit.values())}
    (TAB / "null_v2_summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    pd.set_option("display.width", 250)
    for (variant, k), g in feat.groupby(["variant", "h"]):
        print(variant, k)
        print(g.reindex(g["unsigned_mean"].abs().sort_values(ascending=False).index).head(10).round(4)
              .to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=list(OUT), default="nodrift")
    ap.add_argument("--draws", nargs=2, type=int, default=None, help="first and last draw index (inclusive)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    if a.draws:
        idx = [(i, a.variant) for i in range(a.draws[0], a.draws[1] + 1)]
        if a.workers > 1:
            with Pool(a.workers, initializer=p1.init) as pool:
                pool.map(run_draw, idx, chunksize=1)
        else:
            p1.init()
            for i in idx:
                run_draw(i)
    if a.summarize:
        summarize()
