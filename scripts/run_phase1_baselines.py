"""Phase 1 baselines over the expanding monthly walk-forward (dev period only; holdout never loaded).

Usage: python scripts/run_phase1_baselines.py [--folds N]   (N limits the number of folds, for checks)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from tslaq.backtest.metrics import brier_up, crps_gaussian, fold_ic, hit_rate, ic_summary, per_slot_ic, qlike
from tslaq.cv.walk_forward import monthly_folds
from tslaq.data.pipeline import HORIZONS, build_base, choose_lambda, load_bars
from tslaq.labels.forward_returns import EPS
from tslaq.models.baselines import (
    LGB_PARAMS,
    Z_CLIP,
    fit_predict_lgb,
    garch_var_forecast,
    lgb_n_params,
    seasonal_sum_sq,
    simple_return_features,
)
from tslaq.utils.io import ROOT, config_hash, load_config
from tslaq.utils.registry import log_run
from tslaq.utils.seed import set_seed

OUT = ROOT / "reports" / "tables"


def main(max_folds: int | None, log: bool) -> None:
    set_seed(0)
    dcfg, bcfg = load_config("data"), load_config("backtest")
    wf = bcfg["walk_forward"]
    lam, lam_tab = choose_lambda(load_bars("TSLA"), wf["first_val_month"], dcfg)
    df = build_base("TSLA", lam, dcfg)
    X = simple_return_features(df)
    feat_ok = X.notna().all(axis=1).to_numpy() & np.isfinite(df["sigma_hat"].to_numpy())
    ret_preds, vol_preds, fold_rows = [], [], []
    garch_cache: dict = {}
    month = df["session_date"].dt.to_period("M").astype(str).to_numpy()
    t0 = time.time()
    for h in HORIZONS:
        S = df[f"S_{h}"].to_numpy()
        z = df[f"z_{h}"].to_numpy()
        ok = feat_ok & np.isfinite(z) & np.isfinite(S)
        le = np.where(ok, df[f"label_end_{h}"].to_numpy(), -1)
        folds = monthly_folds(df["session_date"], le, wf["first_val_month"], wf["last_val_month"], wf["embargo_bars"])
        folds = folds[:max_folds] if max_folds else folds
        sss = seasonal_sum_sq(df, h)
        for f in folds:
            tr, va = f.train_idx, f.val_idx
            # ---- return baselines (targets in z units; E[r] = z_pred * S) ----
            mu = float(np.mean(np.clip(z[tr], -Z_CLIP, Z_CLIP)))
            sd = float(np.std(np.clip(z[tr], -Z_CLIP, Z_CLIP)))
            p_lgb, _, sd_lgb = fit_predict_lgb(X.iloc[tr], z[tr], X.iloc[va])
            for name, pz, s in [("zero", np.zeros(len(va)), sd), ("hist_mean", np.full(len(va), mu), sd),
                                ("lgb_returns", p_lgb, sd_lgb)]:
                ret_preds.append(pd.DataFrame({"h": h, "model": name, "fold": f.val_month, "pos": va,
                                               "pred_z": pz, "sd_z": s}))
            # ---- volatility baselines (forecast of sum r^2 over t+1..t+h) ----
            c_seas = float(np.nanmean(df["u"].to_numpy()[tr] ** 2))
            if f.val_month not in garch_cache:  # fit on all u before the validation month, all horizons
                garch_cache[f.val_month] = garch_var_forecast(df, np.flatnonzero(month < f.val_month), HORIZONS)
            garch_f, gp = garch_cache[f.val_month]
            for name, fc in [("vol_seasonal", c_seas * sss[va]), ("vol_ewma", S[va] ** 2),
                             ("vol_garch_t", garch_f[h][va])]:
                vol_preds.append(pd.DataFrame({"h": h, "model": name, "fold": f.val_month, "pos": va, "f": fc}))
            fold_rows.append({"h": h, "fold": f.val_month, "n_train": len(tr), "n_val": len(va),
                              "garch_alpha": gp["alpha"], "garch_beta": gp["beta"], "garch_nu": gp["nu"]})
        print(f"h={h}: {len(folds)} folds done, {time.time() - t0:.0f}s", flush=True)
    R = pd.concat(ret_preds, ignore_index=True)
    V = pd.concat(vol_preds, ignore_index=True)
    info = pd.DataFrame(fold_rows)
    day = df["session_date"].to_numpy()
    slot = df["bar_index"].to_numpy()
    fwd = {h: df[f"fwd_{h}"].to_numpy() for h in HORIZONS}
    zz = {h: df[f"z_{h}"].to_numpy() for h in HORIZONS}
    SS = {h: df[f"S_{h}"].to_numpy() for h in HORIZONS}
    # realized variance sum r^2 over t+1..t+h; values below 1e-11 are exact zeros plus float residue
    rvv = {}
    for h in HORIZONS:
        x = np.exp(df[f"rv_{h}"].to_numpy()) - EPS
        rvv[h] = np.where(x < 1e-11, 0.0, x)

    def ret_metrics(g, h):
        p = g["pos"].to_numpy()
        er = g["pred_z"].to_numpy() * SS[h][p]
        y = fwd[h][p]
        ic = ic_summary(er, y, day[p])
        ic = {"daily_ic(biased)": ic["rank_ic"], "daily_icir(biased)": ic["icir"], "pooled_ic": ic["pooled_ic"]}
        fic = fold_ic(er, y, g["fold"].to_numpy())
        ytrue = zz[h][p]
        pz = g["pred_z"].to_numpy()
        fz = fold_ic(pz, ytrue, g["fold"].to_numpy())
        return {**ic, "z_fold_ic": fz.mean(), "z_fold_icir": fz.mean() / fz.std() if len(fz) > 1 else np.nan,
                "z_folds_pos": float((fz > 0).mean()) if len(fz) else np.nan,
                "z_slot_ic": per_slot_ic(pz, ytrue, slot[p]), "slot_ic": per_slot_ic(er, y, slot[p]),
                "fold_ic_mean": fic.mean(), "fold_icir": fic.mean() / fic.std() if len(fic) > 1 else np.nan,
                "folds_ic_pos": float((fic > 0).mean()), "hit_rate": hit_rate(er, y),
                "crps_z": float(np.mean(crps_gaussian(g["pred_z"].to_numpy(), g["sd_z"].to_numpy(), ytrue))),
                "brier": brier_up(norm.cdf(g["pred_z"].to_numpy() / g["sd_z"].to_numpy()), y),
                "n": len(p), "up_rate": float(np.mean(y[y != 0] > 0))}

    def vol_metrics(g, h):
        p = g["pos"].to_numpy()
        f, y = g["f"].to_numpy(), rvv[h][p]
        m = np.isfinite(f) & (f > 0) & (y > 0)
        return {"qlike": qlike(y, f), "zero_rv_share": float(np.mean(y[np.isfinite(f)] == 0)), "mse_log": float(np.mean((np.log(y[m]) - np.log(f[m])) ** 2)),
                "spearman": float(spearmanr(f[m], y[m]).statistic), "n": int(m.sum())}

    rows = [{"h": h, "model": m, **ret_metrics(g, h)} for (h, m), g in R.groupby(["h", "model"])]
    ret_sum = pd.DataFrame(rows)
    ret_fold = pd.DataFrame([{"h": h, "model": m, "fold": fo, **ret_metrics(g, h)}
                             for (h, m, fo), g in R[R["model"] == "lgb_returns"].groupby(["h", "model", "fold"])])
    R["year"] = pd.to_datetime(R["fold"]).dt.year
    ret_year = pd.DataFrame([{"h": h, "model": m, "year": y, **ret_metrics(g, h)}
                             for (h, m, y), g in R[R["model"] == "lgb_returns"].groupby(["h", "model", "year"])])
    vol_sum = pd.DataFrame([{"h": h, "model": m, **vol_metrics(g, h)} for (h, m), g in V.groupby(["h", "model"])])
    V["year"] = pd.to_datetime(V["fold"]).dt.year
    vol_year = pd.DataFrame([{"h": h, "model": m, "year": y, **vol_metrics(g, h)}
                             for (h, m, y), g in V.groupby(["h", "model", "year"])])
    digest = hashlib.sha1(np.ascontiguousarray(R["pred_z"].to_numpy()).tobytes()
                          + np.ascontiguousarray(np.nan_to_num(V["f"].to_numpy())).tobytes()).hexdigest()[:12]
    version = dcfg.get("data_version", "v1")
    suffix = f"_{version}" + (f"_check{max_folds}" if max_folds else "")
    OUT.mkdir(parents=True, exist_ok=True)
    pred_dir = ROOT / "data" / "processed"
    pred_dir.mkdir(parents=True, exist_ok=True)
    R.to_parquet(pred_dir / f"phase1_return_preds{suffix}.parquet", index=False)
    V.to_parquet(pred_dir / f"phase1_vol_preds{suffix}.parquet", index=False)
    ret_sum.to_csv(OUT / f"phase1_returns_summary{suffix}.csv", index=False)
    ret_fold.to_csv(OUT / f"phase1_lgb_by_fold{suffix}.csv", index=False)
    ret_year.to_csv(OUT / f"phase1_lgb_by_year{suffix}.csv", index=False)
    vol_sum.to_csv(OUT / f"phase1_vol_summary{suffix}.csv", index=False)
    vol_year.to_csv(OUT / f"phase1_vol_by_year{suffix}.csv", index=False)
    info.to_csv(OUT / f"phase1_fold_info{suffix}.csv", index=False)
    lam_tab.to_csv(OUT / f"phase1_lambda_tuning{suffix}.csv", index=False)
    meta = {"data_version": version, "lambda": lam, "prediction_digest": digest, "n_folds": int(info["fold"].nunique()),
            "lgb_params": LGB_PARAMS, "seconds": round(time.time() - t0)}
    (OUT / f"phase1_meta{suffix}.json").write_text(json.dumps(meta, indent=2))
    print(ret_sum.to_string())
    print(vol_sum.to_string())
    print(meta)
    if log:
        chash = config_hash(dcfg, bcfg, LGB_PARAMS)
        scheme = (f"expanding monthly WF {wf['first_val_month']}..{wf['last_val_month']}, "
                  f"purge+embargo {wf['embargo_bars']}")
        # on a new data version the same models are re-runs of the v1 trials, not new trials
        rerun = {"rerun_of": "phase1 v1 baselines (same model/config)"} if version != "v1" else {}
        for m in ["zero", "hist_mean", "lgb_returns"]:
            row = {"config_hash": chash, "phase": 1, "model": m, "fold_scheme": scheme,
                   "n_params": lgb_n_params() if m == "lgb_returns" else (1 if m == "hist_mean" else 0),
                   "data_version": version, **rerun, "notes": f"digest={digest}; lambda={lam}"}
            for h in HORIZONS:
                s = ret_sum[(ret_sum["h"] == h) & (ret_sum["model"] == m)].iloc[0]
                row.update({f"rank_ic_h{h}": s["z_fold_ic"], f"icir_h{h}": s["z_fold_icir"],
                            f"hit_rate_h{h}": s["hit_rate"], f"crps_h{h}": s["crps_z"], f"brier_h{h}": s["brier"]})
            log_run(row)
        for m in ["vol_seasonal", "vol_ewma", "vol_garch_t"]:
            q = "; ".join(f"h{h} qlike={vol_sum[(vol_sum.h == h) & (vol_sum.model == m)]['qlike'].iloc[0]:.4f}"
                          for h in HORIZONS)
            log_run({"config_hash": chash, "phase": 1, "model": m, "fold_scheme": scheme,
                     "n_params": {"vol_seasonal": 1, "vol_ewma": 1, "vol_garch_t": 4}[m],
                     "data_version": version, **rerun, "notes": f"{q}; digest={digest}"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()
    main(a.folds, log=not a.no_log and a.folds is None)
