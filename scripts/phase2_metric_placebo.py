"""D17 / D19 validation (D20 criterion 2): session-permutation placebo under the old metric, the D17 anchored
IC (superseded) and the D19 anchored IC v2 (diagnostic, not a trial).

Same permutation as scripts/phase2_daily_placebo.py: whole sessions are shuffled (each session keeps its
intraday path, its gap, high and low relative to the previous close), daily closes are rebuilt and the daily
features recomputed. amihud: each session keeps its dollar volume (C x V per bar); share volume = dollar volume /
rebuilt price, so |r| / (C x V) of every bar moves with its session and the 78-bar rolling mean is recomputed on
the permuted series. Bar-level target z_12 (and the one-bar-delayed z) moves with its session; months follow the
position in the rebuilt series.

Metrics over the 96 folds 2018-01..2025-12 (fold = calendar month M, training = all rows before M):
- old (biased, information only): Spearman pooled within month M; sign_train = sign of the mean over months < M;
- anchored (D17): s = (x - median_train) / IQR_train clipped to +-5, y = clip(z, +-5) - m_train,
  IC = sum(s y) / sqrt(sum s^2 sum y^2) in month M; sign_train = sign of the mean anchored IC over months < M
  (same anchors);
- anchored v2 (D19): s = c(x) - mean_train c(x), c = clip((x - median_train) / IQR_train, +-5), y = clip(z, +-5),
  IC = sum(s y) / (n sqrt(V_s V_y)) with training moments; sign_train from the training months.
Reported unsigned (mean IC over folds, the bias) and signed (out-of-sample, as used by D10).
Usage: python scripts/phase2_metric_placebo.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from phase2_daily_placebo import daily_feats

from tslaq.backtest.metrics import Z_CLIP, fold_ic_stats
from tslaq.data.pipeline import load_bars
from tslaq.features.indicators import bar_features
from tslaq.selection import rescore as rs
from tslaq.selection.ic import (
    anchored_group_ic,
    feature_scores,
    feature_scores_v2,
    score_anchors,
    score_anchors_v2,
    spearman_cols,
    v2_group_ic,
)
from tslaq.utils.io import ROOT

warnings.filterwarnings("ignore")
TAB = ROOT / "reports" / "tables" / "phase2_metric_fix_v2"  # D17-only results: reports/tables/phase2_metric_fix
TAB.mkdir(parents=True, exist_ok=True)
N_PERM = 200
FEATS = ["d_sma20_dev", "d_mom5", "d_atr_c", "amihud"]
OOS = [str(p) for p in pd.period_range("2018-01", "2025-12", freq="M")]


def amihud_from(illiq: np.ndarray) -> np.ndarray:
    return np.log(pd.Series(illiq).rolling(78, min_periods=60).mean().to_numpy())


def score_all(X: np.ndarray, Y: dict, month: np.ndarray) -> dict:
    """Per target key: arrays (n_folds x n_feat) of old and anchored fold IC (unsigned) and training signs."""
    um = pd.unique(month)
    starts = np.r_[0, np.flatnonzero(month[1:] != month[:-1]) + 1]
    bounds = dict(zip(um, zip(starts, np.r_[starts[1:], len(month)])))
    out = {}
    for key, z in Y.items():
        old = np.array([spearman_cols(X[a:b], z[a:b], 200) for a, b in (bounds[m] for m in um)])
        old = pd.DataFrame(old, index=um)
        zc = np.clip(z, -Z_CLIP, Z_CLIP)
        o_u, o_s, a_u, a_s, v_u, v_s = [], [], [], [], [], []
        for M in OOS:
            a, b = bounds[M]
            past = old.loc[old.index < M]
            o_u.append(old.loc[M].to_numpy())
            o_s.append(np.sign(past.mean().to_numpy()))
            centre, scale = score_anchors(X[:a])
            m = np.nanmean(zc[:a])
            G = anchored_group_ic(feature_scores(X[:b], centre, scale), zc[:b] - m, month[:b], 200)
            a_u.append(G.iloc[-1].to_numpy())
            a_s.append(np.sign(G.iloc[:-1].mean().to_numpy()))
            A = score_anchors_v2(X[:a])
            G2 = v2_group_ic(feature_scores_v2(X[:b], A), zc[:b], month[:b], A["vs"], np.nanmean(zc[:a] ** 2), 200)
            v_u.append(G2.iloc[-1].to_numpy())
            v_s.append(np.sign(G2.iloc[:-1].mean().to_numpy()))
        out[key] = {"old": np.array(o_u), "old_sign": np.array(o_s), "anch": np.array(a_u), "anch_sign": np.array(a_s),
                    "v2": np.array(v_u), "v2_sign": np.array(v_s)}
    return out


def summary(res: dict) -> dict:
    """(feature, key, metric, kind) -> value: mean over folds of unsigned IC, signed IC and the signed t."""
    v = {}
    for key, d in res.items():
        for met in ("old", "anch", "v2"):
            u, s = d[met], d[met] * d[f"{met}_sign"]
            for j, f in enumerate(FEATS):
                st = fold_ic_stats(pd.Series(s[:, j]))
                v[(f, key, met, "unsigned")] = float(np.nanmean(u[:, j]))
                v[(f, key, met, "signed")] = st["ic"]
                v[(f, key, met, "signed_t")] = st["t"]
    return v


def main():
    t0 = time.time()
    lam = rs.lambdas()
    L = rs.base_frame(load_bars("TSLA"), lam["TSLA"], (12,))
    sess = L["session_date"].to_numpy()
    first = np.r_[0, np.flatnonzero(sess[1:] != sess[:-1]) + 1]
    last = np.r_[first[1:] - 1, len(sess) - 1]
    n_s = len(first)
    H_, Lo, C = (L[c].to_numpy() for c in ["high", "low", "close"])
    dH, dL, dC = np.maximum.reduceat(H_, first), np.minimum.reduceat(Lo, first), C[last]
    pc = np.r_[np.nan, dC[:-1]]
    rel = np.column_stack([dC / pc, dH / pc, dL / pc])
    v = L["volume"].where(~L["missing"] & (L["volume"] > 0)).to_numpy()
    illiq = np.abs(L["r"].to_numpy()) / (C * v)  # = |r| / dollar volume; invariant to the rebuilt price level
    ref = bar_features(L)["amihud"].to_numpy()
    assert np.allclose(amihud_from(illiq), ref, equal_nan=True, rtol=1e-12)
    z = {"12": L["z_12"].to_numpy(), "d12": L["zd_12"].to_numpy()}
    month_s = pd.Series(sess[first]).dt.to_period("M").astype(str).to_numpy()
    seg_len = last - first + 1
    seg = [np.arange(a, b + 1) for a, b in zip(first, last)]

    def bar_matrix(daily: pd.DataFrame, ill: np.ndarray, sid: np.ndarray) -> np.ndarray:
        return np.column_stack([daily[f].to_numpy()[sid] for f in FEATS[:3]] + [amihud_from(ill)])

    sid0 = np.repeat(np.arange(n_s), seg_len)
    month_b = month_s[sid0]
    real = summary(score_all(bar_matrix(daily_feats(dC, dH, dL), illiq, sid0), z, month_b))
    print(f"real scored {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    valid = np.flatnonzero(np.isfinite(rel[:, 0]))
    P = []
    t1 = time.time()
    for i in range(N_PERM):
        perm = np.r_[0, rng.permutation(valid)]  # position 0 keeps the first session (no previous close)
        r = rel[perm]
        c = np.r_[dC[0], dC[0] * np.cumprod(r[1:, 0])]
        cprev = np.r_[np.nan, c[:-1]]
        f = daily_feats(c, r[:, 1] * cprev, r[:, 2] * cprev)
        f.iloc[0] = np.nan
        order = np.concatenate([seg[j] for j in perm])
        sid = np.repeat(np.arange(n_s), seg_len[perm])
        X = bar_matrix(f, illiq[order], sid)
        P.append(summary(score_all(X, {k: x[order] for k, x in z.items()}, month_s[sid])))
        if i == 0:
            print(f"one permutation {time.time() - t1:.1f}s -> projected {N_PERM * (time.time() - t1) / 60:.0f} min",
                  flush=True)
        if (i + 1) % 25 == 0:
            print(f"perm {i + 1}/{N_PERM} {time.time() - t0:.0f}s", flush=True)
    rows = []
    for key in real:
        pv = np.array([p[key] for p in P])
        rows.append({"feature": key[0], "target": f"z_{key[1]}" if key[1] == "12" else "z_12 delayed",
                     "metric": {"old": "old (biased, information only)", "anch": "anchored (D17, superseded)",
                                "v2": "anchored v2 (D19)"}[key[2]],
                     "statistic": {"unsigned": "mean fold IC (unsigned)", "signed": "mean signed OOS IC",
                                   "signed_t": "t of signed OOS IC"}[key[3]],
                     "real": real[key], "placebo_mean": np.nanmean(pv), "placebo_sd": np.nanstd(pv),
                     "placebo_p05": np.nanquantile(pv, 0.05), "placebo_p95": np.nanquantile(pv, 0.95),
                     "z_vs_placebo": (real[key] - np.nanmean(pv)) / np.nanstd(pv), "n_perm": int(np.isfinite(pv).sum())})
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "placebo_all_metrics.csv", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string(index=False))
    (TAB / "placebo_meta.json").write_text(json.dumps({"n_perm": N_PERM, "seconds": time.time() - t0,
                                                       "lambda_tsla": lam["TSLA"]}, indent=1))


if __name__ == "__main__":
    main()
