"""Forecast -> position rules of the early Phase 5 test (CLAUDE.md 10.4; decision D27).

Bars are rows of the base frame L in time order; `sess` is the session id per bar and `last` the position of the
session's last bar. A signal at bar t (its close) fills at the open of bar t+1+d (d = extra delay). No fills in the
session's last two bars (last 10 minutes) except the forced exit at the close of the last bar.

R1 proportional: target w = clip(zhat / q, -1, 1); the position moves to the target only if |target - w| > band.
R2 strong signal: |zhat| > q -> sign(zhat) at unit size from the open of t+1+d to the close of t+h+d,
non-overlapping within a session; trades whose exit would fall after the session's last bar are skipped (D25).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NO_FILL_LAST = 2  # bars at the end of the session without new fills (10 minutes of 5-minute bars)


def session_arrays(sess: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(integer session id per bar, position of the session's last bar per bar); bars must be in time order."""
    sid = pd.factorize(sess)[0]
    pos = np.arange(len(sess))
    last = pd.Series(pos).groupby(sid).transform("max").to_numpy()
    return sid, last


def r1_fills(sid: np.ndarray, last: np.ndarray, pos: np.ndarray, zhat: np.ndarray, q: np.ndarray, d: int,
             band: float = 0.25) -> pd.DataFrame:
    """Fills of R1: bar of the fill (its open), new position, signal bar."""
    o = np.argsort(pos, kind="stable")
    pos, zhat, q = pos[o], zhat[o], q[o]
    rows, cur, cs = [], 0.0, None
    for t, z, qq in zip(pos, zhat, q):
        if sid[t] != cs:
            cs, cur = sid[t], 0.0
        e = t + 1 + d
        if not (np.isfinite(z) and np.isfinite(qq) and qq > 0) or e > last[t] - NO_FILL_LAST:
            continue
        w = float(np.clip(z / qq, -1.0, 1.0))
        if abs(w - cur) > band:
            rows.append((e, w, t))
            cur = w
    return pd.DataFrame(rows, columns=["bar", "w", "signal"])


def r1_pnl(O: np.ndarray, C: np.ndarray, sid: np.ndarray, last: np.ndarray, F: pd.DataFrame) -> dict:
    """Per-bar position path of R1 and its P&L. The position set at the open of bar e is held open-to-open; the
    last bar is held open-to-close and flattened at its close. Returns per-bar gross P&L, per-bar |dw| (fills and
    the final flattening) and holding segments (unit return, sign) for the hit rate."""
    n = len(O)
    w = pd.Series(np.nan, index=np.arange(n))
    w.iloc[F["bar"].to_numpy()] = F["w"].to_numpy()
    # session starts are flat until the first fill
    first = np.r_[True, sid[1:] != sid[:-1]]
    w[first & w.isna().to_numpy()] = 0.0
    w = w.groupby(sid).ffill().fillna(0.0).to_numpy()
    is_last = np.arange(n) == last
    nxt_open = np.r_[O[1:], np.nan]
    r = np.where(is_last, np.log(C / O), np.log(nxt_open / O))
    gross = w * r
    prev = np.r_[0.0, w[:-1]]
    prev[first] = 0.0
    dw = np.abs(w - prev) + np.where(is_last, np.abs(w), 0.0)
    # holding segments: runs of bars with the same position inside a session, w != 0
    seg_id = np.cumsum(first | (w != prev))
    S = pd.DataFrame({"seg": seg_id, "w": w, "r": r, "sid": sid}).groupby("seg").agg(
        w=("w", "first"), r=("r", "sum"), sid=("sid", "first"))
    S = S[S["w"] != 0]
    return {"gross": gross, "dw": dw, "seg_unit": np.sign(S["w"].to_numpy()) * S["r"].to_numpy(),
            "seg_sid": S["sid"].to_numpy()}


def r2_trades(O: np.ndarray, C: np.ndarray, sid: np.ndarray, last: np.ndarray, pos: np.ndarray, zhat: np.ndarray,
              q: np.ndarray, d: int, h: int = 12) -> pd.DataFrame:
    """Trades of R2: signal bar, entry bar, exit bar, side, gross log return."""
    o = np.argsort(pos, kind="stable")
    pos, zhat, q = pos[o], zhat[o], q[o]
    rows, nxt, cs = [], -1, None
    for t, z, qq in zip(pos, zhat, q):
        if sid[t] != cs:
            cs, nxt = sid[t], -1
        e, x = t + 1 + d, t + h + d
        if t < nxt or not (np.isfinite(z) and np.isfinite(qq)) or abs(z) <= qq:
            continue
        if x > last[t] or e > last[t] - NO_FILL_LAST:
            continue
        sg = 1.0 if z > 0 else -1.0
        rows.append((t, e, x, sid[t], sg, sg * np.log(C[x] / O[e])))
        nxt = t + h
    return pd.DataFrame(rows, columns=["signal", "entry", "exit", "sid", "side", "gross"])
