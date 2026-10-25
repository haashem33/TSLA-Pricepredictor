"""D25 baseline confirmation (pre-registered, reports/decisions.md): M1 = Phase 1 lgb_returns retrained in the
D23 pipeline, M2 = ridge on the fold's by_fold v2 features (alpha from the D6 inner-validation block by D19 IC),
M3 = equal-weight composite of the D19 feature scores (basis IC / both), M4 = CatBoost (fixed parameters).

Rows, folds, purge / embargo, the first-fold V_s block and the D19 model scoring are those of Gate 3
(`tslaq.models.gate3`). M3 is not a model: its score is centred at training anchors and scored with the D19
feature formula (V_s = training mean of s^2).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from tslaq.backtest.metrics import Z_CLIP as SCORE_CLIP
from tslaq.backtest.metrics import ic_v2
from tslaq.models import gate3 as g3
from tslaq.models.baselines import Z_CLIP, simple_return_features
from tslaq.selection.ic import feature_scores_v2, score_anchors_v2, v2_group_ic
from tslaq.utils.io import ROOT

MODELS = ("M1", "M2", "M3", "M4")
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
CAT_PARAMS = {"iterations": 300, "depth": 4, "loss_function": "RMSE", "random_seed": 0, "thread_count": 1,
              "verbose": 0, "allow_writing_files": False}
STD_CLIP = 5.0
SIGN_MIN_OBS = 200  # months with >= 200 rows enter sign_train (Phase 2 rescore rule)


def by_fold_basis() -> dict:
    """{M: [(feature, version, basis), ...]} from configs/features.yaml v2."""
    import yaml
    with open(ROOT / "configs" / "features.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["version"] == 2
    return {M: [(x["feature"], x["version"], x["basis"]) for x in v] for M, v in cfg["by_fold"].items()}


def m1_matrix(fr: dict) -> np.ndarray:
    if "X1" not in fr:
        fr["X1"] = simple_return_features(fr["L"]).replace([np.inf, -np.inf], np.nan).to_numpy(float)
    return fr["X1"]


def prev_months(M: str, k: int = 2) -> tuple:
    p = pd.Period(M, "M")
    return tuple(str(p - i) for i in range(k, 0, -1))


# ------------------------------------------------------------------ fitters (target: z clipped +-6, Phase 1)

def fit_cat(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> tuple[np.ndarray, float]:
    from catboost import CatBoostRegressor
    m = CatBoostRegressor(**CAT_PARAMS)
    yt = np.clip(y[tr], -Z_CLIP, Z_CLIP)
    m.fit(X[tr], yt)
    return m.predict(X[te]), float(np.std(yt - m.predict(X[tr])))


def _standardize(X: np.ndarray, tr: np.ndarray) -> np.ndarray:
    mu = np.nanmean(X[tr], axis=0)
    sd = np.nanstd(X[tr], axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = np.clip((X - mu) / sd, -STD_CLIP, STD_CLIP)
    return np.where(np.isfinite(Z), Z, 0.0)


def fit_ridge(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    Xs = _standardize(X, tr)
    yt = np.clip(y[tr], -Z_CLIP, Z_CLIP)
    m = Ridge(alpha=alpha).fit(Xs[tr], yt)
    return m.predict(Xs[te]), float(np.std(yt - m.predict(Xs[tr])))


def inner_ic(pred: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray) -> float:
    """D19 form on one inner block: s = pred - inner-training mean of clipped y, V_s = mean s^2 over the block,
    V_y = inner-training mean of clipped y^2."""
    yc = np.clip(y, -SCORE_CLIP, SCORE_CLIP)
    s = pred - float(np.nanmean(yc[tr]))
    ok = np.isfinite(yc[te])
    return ic_v2(s, yc[te], float(np.mean(s[ok] ** 2)), float(np.nanmean(yc[tr] ** 2)))


def choose_alpha(fr: dict, XB: np.ndarray, h: int, M: str) -> tuple[float, dict, pd.DataFrame]:
    """Alpha by D19 IC on the fold's inner block (2 months before M); ties -> larger alpha. Returns the alpha,
    the IC per alpha and the chosen-alpha inner predictions (fold 2018-01: its D6 V_s block)."""
    y = fr["L"][f"z_{h}"].to_numpy()
    tr, te = g3.split(fr, h, prev_months(M))
    ics, preds = {}, {}
    for a in ALPHAS:
        p, sd = fit_ridge(XB, y, tr, te, a)
        ics[a], preds[a] = inner_ic(p, y, tr, te), (p, sd)
    best = max(ALPHAS, key=lambda a: (ics[a] if np.isfinite(ics[a]) else -np.inf, a))
    p, sd = preds[best]
    return best, ics, pd.DataFrame({"pos": te, "pred": p, "sd": sd})


# ------------------------------------------------------------------ M3 composite

def composite(fr: dict, FF: pd.DataFrame, feats: list, M: str, h: int) -> tuple[np.ndarray, float, dict]:
    """Composite score s (all rows), its training V_s (sessions < M), and sign_train per feature."""
    X = g3.feature_matrix(fr, FF, dict(feats)).to_numpy(float)
    tr = fr["sess"] < np.datetime64(f"{M}-01")
    A = score_anchors_v2(X[tr])
    S = feature_scores_v2(X, A)
    y = np.clip(fr["L"][f"z_{h}"].to_numpy(), -SCORE_CLIP, SCORE_CLIP)
    vy = float(np.nanmean(y[tr] ** 2))
    G = v2_group_ic(S[tr], y[tr], fr["month"][tr], A["vs"], vy, SIGN_MIN_OBS)
    sign = np.sign(G.mean(axis=0).to_numpy())
    use = np.isfinite(sign) & (sign != 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows -> NaN score
        s = np.nanmean(S[:, use] * sign[use], axis=1) if use.any() else np.full(len(X), np.nan)
    vs = float(np.nanmean(s[tr] ** 2))
    return s, vs, {f: float(sg) for (f, _), sg in zip(feats, sign)}


# ------------------------------------------------------------------ one fold

def fold_predictions(fr: dict, M: str, lists: dict, models=MODELS, horizons=g3.H, inner: bool = False) -> dict:
    """Predictions of every model x horizon for month M (and the first fold's inner block). Columns: model, h,
    fold, pos, pred, sd, vs (M3 only), sd0 (training sd of clipped z, for the zero-forecast CRPS)."""
    t0 = time.time()
    FF = g3.fold_ff(fr, M)
    feats = lists[M]
    versions = {f: v for f, v, _ in feats}
    ic_feats = [(f, v) for f, v, b in feats if b in ("IC", "both")]
    XB = g3.feature_matrix(fr, FF, versions).to_numpy(float)
    X1 = m1_matrix(fr) if "M1" in models else None
    preds, info, secs = [], [], {m: 0.0 for m in models}
    for h in horizons:
        y = fr["L"][f"z_{h}"].to_numpy()
        tr, te = g3.split(fr, h, (M,))
        sd0 = float(np.std(np.clip(y[tr], -Z_CLIP, Z_CLIP)))
        blocks = [(M, tr, te)]
        if inner:
            blocks.append(("inner", *g3.split(fr, h, g3.INNER)))

        def add(m, tag, pos, p, sd, vs=np.nan, h=h, sd0=sd0):
            preds.append(pd.DataFrame({"model": m, "h": h, "fold": tag, "pos": pos, "pred": p, "sd": sd,
                                       "vs": vs, "sd0": sd0}))

        for m in models:
            t1 = time.time()
            if m == "M1":
                for tag, a, b in blocks:
                    add(m, tag, b, *g3.fit_predict(X1, y, a, b, []))
            elif m == "M4":
                for tag, a, b in blocks:
                    add(m, tag, b, *fit_cat(XB, y, a, b))
            elif m == "M2":
                alpha, ics, P_in = choose_alpha(fr, XB, h, M)
                add(m, M, te, *fit_ridge(XB, y, tr, te, alpha))
                if inner:  # fold 2018-01: its alpha-selection block is the D6 block
                    assert prev_months(M) == g3.INNER
                    add(m, "inner", P_in["pos"].to_numpy(), P_in["pred"].to_numpy(), P_in["sd"].iloc[0])
                info.append({"model": m, "h": h, "fold": M, "alpha": alpha,
                             **{f"inner_ic_{a:g}": v for a, v in ics.items()}})
            elif m == "M3":
                s, vs, sign = composite(fr, FF, ic_feats, M, h)
                add(m, M, te, s[te], np.nan, vs)
                info.append({"model": m, "h": h, "fold": M, "n_features": len(ic_feats),
                             "n_signed": int(sum(np.isfinite(v) and v != 0 for v in sign.values())),
                             "signs": sign})
            secs[m] += time.time() - t1
        info.append({"model": "_rows", "h": h, "fold": M, "n_train": len(tr), "n_test": len(te)})
    return {"month": M, "preds": pd.concat(preds, ignore_index=True), "info": info, "seconds": time.time() - t0,
            "seconds_by_model": secs}


# ------------------------------------------------------------------ scoring

def score(fr: dict, P: pd.DataFrame, model: str, h: int, delayed: bool, moments=None) -> pd.Series:
    """D19 fold IC per month. Models (M1, M2, M4): D23 model scoring. M3: D19 feature formula with the fold's
    training V_s of the composite."""
    if model != "M3":
        return g3.score_model(fr, P, h, delayed, moments)
    y = np.clip(g3.outcome(fr, h, delayed), -SCORE_CLIP, SCORE_CLIP)
    if moments is None:
        moments = g3.train_moments(fr, g3.outcome(fr, h, delayed), g3.MONTHS)
    _, vy = moments
    out = {}
    for M, g in P[P["fold"] != "inner"].groupby("fold"):
        out[M] = ic_v2(g["pred"].to_numpy(), y[g["pos"].to_numpy()], float(g["vs"].iloc[0]), vy[M])
    return pd.Series(out, dtype=float).sort_index()
