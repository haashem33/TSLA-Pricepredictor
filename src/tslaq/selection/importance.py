"""Clustered MDA with LightGBM under purged CV (CLAUDE.md 6.8 step 4; Lopez de Prado ch. 8).

Inside a training window: K contiguous blocks; for each block, train on the other rows after purging
(label overlap) and a 78-bar embargo, score the IC (z space) on the block, then permute all features
of one cluster jointly (same row permutation) and re-score. Importance = IC_base - IC_permuted.
score="anchored" (D17): the model's prediction as is vs clip(z, -5, 5) - m, m = mean clipped z over the
CV-training rows; not the default, since D17 fails D18 (reports/phase2_metric_fix.md);
score="spearman" (default): the old pooled Spearman (biased, information only).
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from tslaq.backtest.metrics import Z_CLIP, anchored_ic
from tslaq.cv.purge import purge_embargo_mask
from tslaq.selection.ic import spearman_cols

MDA_PARAMS = {
    "objective": "regression", "learning_rate": 0.05, "n_estimators": 150, "num_leaves": 15,
    "min_child_samples": 1000, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "random_state": 0, "deterministic": True, "force_row_wise": True,
    "num_threads": 6, "verbose": -1,
}


def purged_kfold(pos: np.ndarray, label_end: np.ndarray, k: int = 5, embargo: int = 78):
    """Yield (train_pos, test_pos) over contiguous blocks of `pos` (row positions in the full frame)."""
    blocks = np.array_split(np.sort(pos), k)
    for b in blocks:
        a, e = int(b[0]), int(b[-1])
        cand = pos[(pos < a) | (pos > e)]
        keep = purge_embargo_mask(cand, label_end, [(a, e)], embargo)
        yield cand[keep], b


def clustered_mda(X: pd.DataFrame, y: np.ndarray, label_end: np.ndarray, pos: np.ndarray, clusters: pd.Series,
                  k: int = 5, seed: int = 0, params: dict | None = None, score: str = "spearman") -> pd.DataFrame:
    """X, y, label_end are full-frame arrays; `pos` = training row positions to use."""
    rng = np.random.default_rng(seed)
    yc = np.clip(y, -6, 6)
    groups = {c: list(clusters.index[clusters == c]) for c in sorted(clusters.unique())}
    col = {c: i for i, c in enumerate(X.columns)}
    Xn = X.to_numpy(np.float32)
    rows = []
    for fold, (tr, te) in enumerate(purged_kfold(pos, label_end, k)):
        m = lgb.LGBMRegressor(**(params or MDA_PARAMS)).fit(Xn[tr], yc[tr])
        Xt = Xn[te]
        if score == "anchored":
            yt = np.clip(y[te], -Z_CLIP, Z_CLIP) - np.nanmean(np.clip(y[tr], -Z_CLIP, Z_CLIP))

            def ic(p, yt=yt):
                return anchored_ic(p, yt, 50)
        else:
            def ic(p, yt=y[te]):
                return spearman_cols(p[:, None], yt, 50)[0]
        base = ic(m.predict(Xt))
        perm = rng.permutation(len(te))
        for c, feats in groups.items():
            Xp = Xt.copy()
            idx = [col[f] for f in feats]
            Xp[:, idx] = Xt[perm][:, idx]
            s = ic(m.predict(Xp))
            rows.append({"fold": fold, "cluster": c, "base_ic": base, "perm_ic": s, "mda": base - s})
    R = pd.DataFrame(rows)
    g = R.groupby("cluster")["mda"]
    out = pd.DataFrame({"mda_mean": g.mean(), "mda_se": g.std() / np.sqrt(g.count())})
    out["mda_t"] = out["mda_mean"] / out["mda_se"]
    out["base_ic"] = R.groupby("cluster")["base_ic"].mean()
    out["features"] = [", ".join(groups[c]) for c in out.index]
    return out
