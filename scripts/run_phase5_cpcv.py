"""Optional CPCV (CLAUDE.md 10.2; decision D27) for the D27 trial with the highest full-period net Sharpe at D26
cost: Gate-3 B (LightGBM, fixed Phase 1 parameters, by_fold features) with rule R1, h = 12.

Development sessions 2016-01..2025-12 in N = 10 contiguous blocks of equal session count; all 45 pairs of test
blocks; training = the other blocks, purged (label windows overlapping a test block) and embargoed (78 bars after
each test block). Fitted features (GARCH-t / HMM / Kalman / frac-diff) and the feature list are the last fold's
(2025-12): approximation, stated in the report. q = 90th percentile of |zhat| on a 42-session (about 2-month)
inner block of training sessions immediately before the first test block (after the last test block if none
precede it; before the last test block if neither exists), predicted by a model trained on the remaining
training rows (purged and embargoed around the inner block). R1 and D26 costs as in the main test (2016 uses its
own-year spread). Sharpe over the 9 reconstructed paths and the 45 splits.

Usage: python scripts/run_phase5_cpcv.py --time        (one split, timing)
       python scripts/run_phase5_cpcv.py --workers 5     (all splits, then summary)
"""
from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import run_phase5_early as pe

from tslaq.backtest import positions as bp
from tslaq.cv.purge import purge_embargo_mask
from tslaq.models import gate3 as g3

H = 12
N_BLOCKS, K_TEST = 10, 2
INNER_SESSIONS = 42
LAST_FOLD = "2025-12"
OUT = pe.ROOT / "data" / "processed" / "phase5_cpcv"
TAB = pe.TAB
G: dict = {}


def init():
    fr = g3.real_frame()
    FF = g3.fold_ff(fr, LAST_FOLD)
    G["fr"] = fr
    G["X"] = g3.feature_matrix(fr, FF, g3.by_fold_lists()[LAST_FOLD]).to_numpy(float)
    G["y"] = fr["L"][f"z_{H}"].to_numpy()
    G["le"] = g3.label_end(fr, H)
    sess = fr["sess"]
    days = np.unique(sess)
    G["days"] = days
    G["block_of_day"] = np.minimum((np.arange(len(days)) * N_BLOCKS) // len(days), N_BLOCKS - 1)
    G["block"] = G["block_of_day"][np.searchsorted(days, sess)]


def spans(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [a, b] position ranges where mask is True."""
    p = np.flatnonzero(mask)
    if not len(p):
        return []
    cut = np.flatnonzero(np.diff(p) > 1)
    return list(zip(p[np.r_[0, cut + 1]], p[np.r_[cut, len(p) - 1]]))


def inner_days(test: tuple) -> np.ndarray:
    days, bod = G["days"], G["block_of_day"]
    lo = int(np.flatnonzero(bod == test[0])[0])
    hi = int(np.flatnonzero(bod == test[-1])[-1])
    if lo >= INNER_SESSIONS:
        return days[lo - INNER_SESSIONS:lo]
    if hi + 1 + 1 + INNER_SESSIONS <= len(days):  # skip one session (the 78-bar embargo) after the last test block
        return days[hi + 2:hi + 2 + INNER_SESSIONS]
    lo2 = int(np.flatnonzero(bod == test[-1])[0])
    return days[lo2 - INNER_SESSIONS:lo2]


def split_job(test: tuple) -> dict:
    if "fr" not in G:
        init()
    t0 = time.time()
    fr, X, y, le, blk = G["fr"], G["X"], G["y"], G["le"], G["block"]
    pos = np.arange(len(y))
    is_test = np.isin(blk, test)
    te = pos[is_test & (le >= 0)]
    cand = pos[~is_test]
    tr = cand[purge_embargo_mask(cand, le, spans(is_test), g3.EMBARGO)]
    # inner block for q (training sessions only)
    ib = np.isin(fr["sess"], inner_days(test)) & ~is_test
    ite = pos[ib & (le >= 0)]
    icand = tr[~ib[tr]]
    itr = icand[purge_embargo_mask(icand, le, spans(ib), g3.EMBARGO)]
    p_in, _ = g3.fit_predict(X, y, itr, ite, [])
    q = float(np.nanpercentile(np.abs(p_in), pe.Q_PCT))
    p, _ = g3.fit_predict(X, y, tr, te, [])
    return {"test": test, "pos": te, "pred": p, "q": q, "n_train": len(tr), "n_inner_train": len(itr),
            "n_inner": len(ite), "seconds": time.time() - t0}


def run_split(test: tuple) -> float:
    path = OUT / f"split_{test[0]}_{test[1]}.pkl"
    if path.exists():
        return 0.0
    R = split_job(test)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(R, fh)
    print(f"split {test}: {R['seconds']:.0f}s q {R['q']:.3f} train {R['n_train']}", flush=True)
    return R["seconds"]


def paths(splits: list[tuple]) -> list[dict]:
    """Path j of block g = the j-th split (lexicographic) whose test set contains g; 9 paths for N = 10, k = 2."""
    n_paths = len([s for s in splits if 0 in s])
    out = [{} for _ in range(n_paths)]
    for g in range(N_BLOCKS):
        for j, s in enumerate([s for s in splits if g in s]):
            out[j][g] = s
    return out


def summarize():
    if "fr" not in G:
        init()
    fr = G["fr"]
    L = fr["L"]
    sid, last = bp.session_arrays(fr["sess"])
    sess_by_sid = pd.Series(fr["sess"]).groupby(sid).first()
    cost_y = pe.d26_cost_by_year(own_year_first=True)
    splits = list(itertools.combinations(range(N_BLOCKS), K_TEST))
    R = {}
    for s in splits:
        with open(OUT / f"split_{s[0]}_{s[1]}.pkl", "rb") as fh:
            R[s] = pickle.load(fh)
    blk = G["block"]

    def run(P: pd.DataFrame, q: dict) -> pd.Series:
        S = pe.simulate(L, sid, last, P, q, "R1", 0)
        year = pd.DatetimeIndex(sess_by_sid.loc[S["gross"].index]).year.to_numpy()
        c = np.array([cost_y[y] for y in year])
        return S["gross"] - c / 1e4 * S["turn"].to_numpy(), S["gross"]

    rows = []
    for s in splits:
        r = R[s]
        P = pd.DataFrame({"fold": str(s), "pos": r["pos"], "pred": r["pred"]})
        net, gross = run(P, {str(s): r["q"]})
        rows.append({"kind": "split", "id": str(s), "n_days": len(net), "q": r["q"],
                     "sharpe_net_d26": float(net.mean() / net.std() * np.sqrt(252)),
                     "sharpe_gross": float(gross.mean() / gross.std() * np.sqrt(252))})
    for j, pth in enumerate(paths(splits)):
        parts = []
        for g, s in pth.items():
            r = R[s]
            m = blk[r["pos"]] == g
            parts.append(pd.DataFrame({"fold": str(s), "pos": r["pos"][m], "pred": r["pred"][m]}))
        P = pd.concat(parts, ignore_index=True)
        net, gross = run(P, {str(s): R[s]["q"] for s in pth.values()})
        rows.append({"kind": "path", "id": f"path_{j}", "n_days": len(net), "q": np.nan,
                     "sharpe_net_d26": float(net.mean() / net.std() * np.sqrt(252)),
                     "sharpe_gross": float(gross.mean() / gross.std() * np.sqrt(252))})
    T = pd.DataFrame(rows)
    T.to_csv(TAB / "cpcv_sharpes.csv", index=False)
    summ = {}
    for k in ("path", "split"):
        x = T[T["kind"] == k]["sharpe_net_d26"]
        g = T[T["kind"] == k]["sharpe_gross"]
        summ[k] = {"n": len(x), "median_net": float(x.median()), "mean_net": float(x.mean()),
                   "min_net": float(x.min()), "max_net": float(x.max()), "share_net_gt0": float((x > 0).mean()),
                   "median_gross": float(g.median())}
    days = G["days"]
    summ["blocks"] = [{"block": b, "first": str(days[G["block_of_day"] == b][0])[:10],
                       "last": str(days[G["block_of_day"] == b][-1])[:10]} for b in range(N_BLOCKS)]
    summ["seconds_total_fit"] = float(sum(r["seconds"] for r in R.values()))
    (TAB / "cpcv_summary.json").write_text(json.dumps(summ, indent=1))
    print(T.round(3).to_string(index=False))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    splits = list(itertools.combinations(range(N_BLOCKS), K_TEST))
    if a.time:
        t0 = time.time()
        init()
        print(f"frame: {time.time() - t0:.0f}s", flush=True)
        s = run_split(splits[-1])
        print(f"one split: {s:.0f}s; projected {len(splits)} splits / {a.workers} workers: "
              f"{s * len(splits) / a.workers / 60:.0f} min (+ frame per worker)", flush=True)
    else:
        with Pool(a.workers) as pool:
            pool.map(run_split, splits, chunksize=1)
        summarize()
