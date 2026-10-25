"""Gate 3 (CLAUDE.md 7.4; decisions D15 + D23): LightGBM A (tokens of the last 24 bars), B (the fold's by_fold
features), C (both), fixed Phase 1 parameters and objective, one model per horizon, 96 monthly folds with purge
and embargo; tokenizer refit per fold on the fold's training rows. Scored with the D19 model formula.

A `frame` dict holds the base frame L (bars, returns, sigma, labels incl. one-bar-delayed outcomes), the
parameter-free features F_free and their rolling-z versions Frz_free, and either `fold_ff` (real data: a function
M -> the fold's fitted features from its Phase 2 checkpoint) or a fixed `FF` (mirror null: last-fold parameters).
"""
from __future__ import annotations

import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import norm

from tslaq.backtest.metrics import Z_CLIP as SCORE_CLIP
from tslaq.backtest.metrics import crps_gaussian, fold_ic, fold_ic_stats, fold_ic_v2_model
from tslaq.cv.purge import purge_embargo_mask
from tslaq.data.pipeline import load_bars
from tslaq.features.build import fitted_features
from tslaq.features.normalize import rolling_z
from tslaq.models.baselines import LGB_PARAMS, Z_CLIP
from tslaq.selection import rescore as rs
from tslaq.tokenizer.fields import FIELDS, special, token_inputs
from tslaq.tokenizer.tokenizer import Tokenizer, lag_matrix
from tslaq.utils.io import ROOT, load_config

H = (1, 3, 12)
MONTHS = [str(p) for p in pd.period_range("2018-01", "2025-12", freq="M")]
INNER = ("2017-11", "2017-12")  # D6 inner-validation block of the first fold (seeds V_s, D19)
VARIANTS = ("A", "B", "C")
EMBARGO = 78
THREADS = 1  # compute setting only (D23); identical in the real run and every null draw
CK = ROOT / "data" / "processed" / "phase2_folds"


def lgb_params() -> dict:
    return LGB_PARAMS | {"num_threads": THREADS}


def by_fold_lists() -> dict:
    import yaml
    with open(ROOT / "configs" / "features.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert cfg["version"] == 2
    return {M: {x["feature"]: x["version"] for x in v} for M, v in cfg["by_fold"].items()}


def load_ckpt(M: str) -> dict:
    with open(CK / f"{M}.pkl", "rb") as fh:
        return pickle.load(fh)


def finish_frame(L: pd.DataFrame, F_free: pd.DataFrame, Frz_free: pd.DataFrame, horizons) -> dict:
    prof, pos = rs.profile(L)
    fr = {"L": L, "F_free": F_free, "Frz_free": Frz_free, "prof": prof, "pos": pos, "horizons": tuple(horizons),
          "month": L["session_date"].dt.to_period("M").astype(str).to_numpy(),
          "sess": L["session_date"].to_numpy()}
    return fr


def real_frame(horizons=H) -> dict:
    lam = rs.lambdas()
    L = rs.base_frame(load_bars("TSLA"), lam["TSLA"], horizons)
    F_free, Frz_free = rs.free_features(L, rs.market_frames(lam), rs.reaction_sessions())
    fr = finish_frame(L, F_free, Frz_free, horizons)
    fr["lam"] = lam
    return fr


def fold_ff(fr: dict, M: str) -> pd.DataFrame:
    """Fitted features used for fold M: the fold's own Phase 2 fit (real) or the fixed last-fold fit (null)."""
    if "FF" in fr:
        return fr["FF"]
    return fitted_features(fr["L"], rs.fold_params(load_ckpt(M)), fr["prof"], fr["pos"])


def feature_matrix(fr: dict, FF: pd.DataFrame, versions: dict) -> pd.DataFrame:
    cols = {}
    for f, v in versions.items():
        if f in FF:
            cols[f] = rolling_z(FF[f], rs.RZ_WINDOW) if v == "rz" else FF[f]
        else:
            cols[f] = fr["Frz_free"][f] if v == "rz" else fr["F_free"][f]
    return pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan)


def label_end(fr: dict, h: int) -> np.ndarray:
    L = fr["L"]
    ok = np.isfinite(L[f"z_{h}"].to_numpy()) & np.isfinite(L[f"S_{h}"].to_numpy())
    return np.where(ok, L[f"label_end_{h}"].to_numpy(), -1)


def split(fr: dict, h: int, months: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Train = rows before the first of `months` (purged against the block, 78-bar embargo); val = rows in the
    block with a valid label. Identical to cv.walk_forward.monthly_folds for a one-month block."""
    month = fr["month"]
    le = label_end(fr, h)
    pos = np.arange(len(month))
    inb = np.isin(month, months)
    val = pos[inb & (le >= 0)]
    a, b = int(pos[inb].min()), int(pos[inb].max())
    cand = pos[month < months[0]]
    keep = purge_embargo_mask(cand, le, [(a, b)], EMBARGO)
    return cand[keep], val


def fit_predict(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray, cat: list) -> tuple[np.ndarray, float]:
    m = lgb.LGBMRegressor(**lgb_params())
    yt = np.clip(y[tr], -Z_CLIP, Z_CLIP)
    m.fit(X[tr], yt, categorical_feature=cat if cat else "auto")
    return m.predict(X[te]), float(np.std(yt - m.predict(X[tr])))


def fold_tokens(fr: dict, FF: pd.DataFrame, cutoff: str) -> tuple[Tokenizer, pd.DataFrame, pd.DataFrame]:
    """Tokenizer fit on rows with session < cutoff; returns (tokenizer, token inputs, tokens)."""
    X = token_inputs(fr["L"], fr["F_free"], FF["hmm_state"].to_numpy())
    tr = fr["sess"] < np.datetime64(cutoff)
    Z = {h: fr["L"][f"z_{h}"].to_numpy() for h in fr["horizons"]}
    tok = Tokenizer(horizons=fr["horizons"]).fit(X, tr, Z, cutoff)
    return tok, X, tok.transform(X)


def design(LM: np.ndarray, XB: pd.DataFrame, variant: str) -> tuple[np.ndarray, list]:
    if variant == "A":
        return LM, list(range(LM.shape[1]))
    if variant == "B":
        return XB.to_numpy(float), []
    return np.hstack([LM.astype(float), XB.to_numpy(float)]), list(range(LM.shape[1]))


def fold_predictions(fr: dict, M: str, versions: dict, variants=VARIANTS, horizons=H, inner: bool = False,
                     tok_dir=None) -> dict:
    """Fit every variant x horizon for fold M and predict month M (and, for the first fold, the inner block)."""
    t0 = time.time()
    FF = fold_ff(fr, M)
    tok, Xin, T = fold_tokens(fr, FF, f"{M}-01")
    if tok_dir is not None:
        tok.save(tok_dir / f"{M}.pkl")
    LM, _ = lag_matrix(T)
    XB = feature_matrix(fr, FF, versions)
    preds, info = [], []
    blocks = [((M,), LM, M)]
    if inner:
        _, _, Ti = fold_tokens(fr, FF, f"{INNER[0]}-01")
        blocks.append((INNER, lag_matrix(Ti)[0], "inner"))
    for months, LMb, tag in blocks:
        for h in horizons:
            tr, te = split(fr, h, months)
            y = fr["L"][f"z_{h}"].to_numpy()
            for v in variants:
                Xv, cat = design(LMb, XB, v)
                p, sd = fit_predict(Xv, y, tr, te, cat)
                preds.append(pd.DataFrame({"variant": v, "h": h, "fold": tag, "pos": te, "pred": p, "sd": sd}))
                info.append({"variant": v, "h": h, "fold": tag, "n_train": len(tr), "n_test": len(te)})
    return {"month": M, "preds": pd.concat(preds, ignore_index=True), "info": pd.DataFrame(info), "tok": tok,
            "T": T, "Xin": Xin, "FF_hmm": FF["hmm_state"].to_numpy(), "seconds": time.time() - t0}


# ------------------------------------------------------------------ scoring (D19 model formula)

def outcome(fr: dict, h: int, delayed: bool) -> np.ndarray:
    return fr["L"][(f"zd_{h}" if delayed else f"z_{h}")].to_numpy()


def train_moments(fr: dict, y: np.ndarray, months) -> tuple[dict, dict]:
    """Per fold: training mean of the clipped outcome and training mean of its square (sessions < fold month)."""
    yc = np.clip(y, -SCORE_CLIP, SCORE_CLIP)
    zbar, vy = {}, {}
    for M in months:
        tr = fr["sess"] < np.datetime64(f"{M}-01")
        zbar[M], vy[M] = float(np.nanmean(yc[tr])), float(np.nanmean(yc[tr] ** 2))
    return zbar, vy


def score_model(fr: dict, P: pd.DataFrame, h: int, delayed: bool, moments: dict | None = None) -> pd.Series:
    """D19 fold IC per month for one variant x horizon. P has columns fold, pos, pred (incl. fold 'inner')."""
    y = outcome(fr, h, delayed)
    if moments is None:
        moments = train_moments(fr, y, [*MONTHS, INNER[0]])
    zbar, vy = moments
    inn = P[P["fold"] == "inner"]
    s0 = inn["pred"].to_numpy() - zbar[INNER[0]]
    ok0 = np.isfinite(y[inn["pos"].to_numpy()])
    vs0 = float(np.mean(s0[ok0] ** 2))
    Q = P[P["fold"] != "inner"]
    return fold_ic_v2_model(Q["pred"], y[Q["pos"].to_numpy()], Q["fold"], zbar, vy, vs0)


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.sort_index()
    yr = pd.Index(ic.index).str[:4].astype(int)
    a, b, c = fold_ic_stats(ic), fold_ic_stats(ic[yr >= 2023]), fold_ic_stats(ic.iloc[-24:])
    return {"n_folds": a["n_folds"], "ic_full": a["ic"], "icir_full": a["icir"], "t_full": a["t"],
            "pos_share": a["pos_share"], "binom_p": a["binom_p"], "ic_2325": b["ic"], "t_2325": b["t"],
            "n_2325": b["n_folds"], "ic_last24": c["ic"], "t_last24": c["t"],
            "d10": bool(a["t"] >= 2 and b["ic"] > 0 and b["t"] >= 1)}


def side_metrics(fr: dict, P: pd.DataFrame, h: int) -> dict:
    """Hit rate, Gaussian CRPS (training residual sd), Brier of P(up), and the biased monthly Spearman."""
    Q = P[P["fold"] != "inner"]
    pos = Q["pos"].to_numpy()
    z = fr["L"][f"z_{h}"].to_numpy()[pos]
    fwd = fr["L"][f"fwd_{h}"].to_numpy()[pos]
    p, sd = Q["pred"].to_numpy(), Q["sd"].to_numpy()
    m = (p != 0) & (fwd != 0) & np.isfinite(fwd)
    sp = fold_ic(p, z, Q["fold"].to_numpy())
    pu = norm.cdf(p / sd)
    mb = np.isfinite(fwd) & (fwd != 0)
    return {"hit_rate": float(np.mean(np.sign(p[m]) == np.sign(fwd[m]))),
            "crps_z": float(np.mean(crps_gaussian(p, sd, z))), "brier": float(np.mean((pu[mb] - (fwd[mb] > 0)) ** 2)),
            "spearman_monthly_biased": float(sp.mean())}


# ------------------------------------------------------------------ tokenizer checks (7.4)

def tokenizer_checks(fr: dict, M: str, tok: Tokenizer, Xin: pd.DataFrame, T: pd.DataFrame) -> dict:
    L = fr["L"]
    tr = fr["sess"] < np.datetime64(f"{M}-01")
    te = fr["month"] == M
    out = {"fold": M, "ret_n_bins": len(tok.ret.edges) + 1, "ret_min_train_count": int(tok.ret.counts.min()),
           "ret_merged": len(tok.ret.merged), "ret_merges": tok.ret.merged,
           "shape_min_train": int(np.bincount(T["SHAPE"].to_numpy()[tr], minlength=33)[:32].min()),
           "shape_max_train": int(np.bincount(T["SHAPE"].to_numpy()[tr], minlength=33)[:32].max()),
           "shape_inertia": tok.shape.inertia, "gap_n_bins": len(tok.gap.edges) + 1}
    z = Xin["ret_z"].to_numpy()
    zh = tok.detok_ret(T["RET"].to_numpy())
    r, sig = L["r"].to_numpy(), L["sigma_hat"].to_numpy()
    for tag, sel in (("train", tr), ("test", te)):
        ok = sel & np.isfinite(zh) & np.isfinite(z)
        out[f"ret_corr_{tag}"] = float(np.corrcoef(r[ok], zh[ok] * sig[ok])[0, 1])
        out[f"ret_corr_z_{tag}"] = float(np.corrcoef(np.clip(z[ok], -6, 6), zh[ok])[0, 1])
        out[f"ret_rmse_sigma_{tag}"] = float(np.sqrt(np.mean((zh[ok] - np.clip(z[ok], -6, 6)) ** 2)))
        for h, b in tok.out.items():
            zt = L[f"z_{h}"].to_numpy()
            m = sel & np.isfinite(zt)
            zo = b.detokenize(b.transform(zt[m]))
            out[f"out{h}_corr_{tag}"] = float(np.corrcoef(zo, np.clip(zt[m], -6, 6))[0, 1])
            out[f"out{h}_rmse_{tag}"] = float(np.sqrt(np.mean((zo - np.clip(zt[m], -6, 6)) ** 2)))
        hm = T["HMM"].to_numpy()[sel]
        for k in range(3):
            out[f"hmm{k}_share_{tag}"] = float(np.mean(hm == k))
        out[f"hmm_missing_share_{tag}"] = float(np.mean(hm == special("HMM", "MISSING")))
    for h, b in tok.out.items():
        out[f"out{h}_min_train_count"] = int(b.counts.min())
        out[f"out{h}_merged"] = len(b.merged)
    for f in FIELDS:
        x = T[f].to_numpy()
        out[f"missing_share_test_{f}"] = float(np.mean(x[te] == special(f, "MISSING")))
        out[f"codes_used_train_{f}"] = len(np.unique(x[tr][x[tr] < FIELDS[f]]))
    return out


def load_config_costs() -> float:
    return 2 * load_config("backtest")["costs"]["bps_per_side"] / 1e4
