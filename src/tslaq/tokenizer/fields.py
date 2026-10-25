"""Token fields (CLAUDE.md 7.2) and their raw, causal inputs.

Every input at bar t uses bars <= t only (sigma_hat_{t|t-1}, Wilder ATR including bar t's true range, the
previous candle, trailing-window percentiles). Each field's vocabulary is its V value codes 0..V-1 followed by
the specials MISSING = V, PAD = V+1, MASK = V+2, BOS = V+3, DAY_SEP = V+4 (one embedding table per field).
The nominal V is fixed, so a fold whose bins were merged simply leaves some value codes unused.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tslaq.features.indicators import true_range, wilder

# field -> number of value codes (MISSING and the other specials come after these)
FIELDS = {"RET": 64, "SHAPE": 32, "REL": 15, "GAP": 10, "VOLU": 12, "VREG": 8, "HMM": 3, "MKT": 16, "TOD": 13,
          "DOW": 5, "EVENT": 5}
SPECIALS = ("MISSING", "PAD", "MASK", "BOS", "DAY_SEP")
N_GAP_BINS = 9          # GAP value codes 0..8 are quantile bins of the bar-0 gap; 9 = "no gap" (bars 1..)
NO_GAP = N_GAP_BINS
REL_RANGE = ("inside", "outside", "hh_hl", "lh_ll", "mixed")
REL_ENGULF = ("none", "bull_engulf", "bear_engulf")
EVENTS = ("earn_react", "earn_after", "opex", "half_day", "none")  # priority order when flags overlap
SHAPE_COLS = ("body_atr", "upper_atr", "lower_atr", "close_location", "range_atr")


def special(field: str, name: str) -> int:
    return FIELDS[field] + SPECIALS.index(name)


def vocab_size(field: str) -> int:
    return FIELDS[field] + len(SPECIALS)


def rel_codes(df: pd.DataFrame) -> np.ndarray:
    """Relation to the previous candle, 5 range classes x 3 engulfing classes = code range*3 + engulf.

    Strict inequalities: inside H < H1 and L > L1; outside H > H1 and L < L1; hh_hl H > H1 and L > L1;
    lh_ll H < H1 and L < L1; mixed = an equal high or low (H == H1 or L == L1). Bullish engulfing: C > O,
    C1 < O1, O <= C1, C >= O1 and the bodies differ; bearish the mirror image. NaN where there is no previous bar.
    """
    O, H, L, C = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    O1, H1, L1, C1 = (np.r_[np.nan, x[:-1]] for x in (O, H, L, C))
    rng = np.full(len(df), 4)
    rng[(H < H1) & (L > L1)] = 0
    rng[(H > H1) & (L < L1)] = 1
    rng[(H > H1) & (L > L1)] = 2
    rng[(H < H1) & (L < L1)] = 3
    diff = (O != C1) | (C != O1)
    bull = (C > O) & (C1 < O1) & (O <= C1) & (C >= O1) & diff
    bear = (C < O) & (C1 > O1) & (O >= C1) & (C <= O1) & diff
    eng = np.where(bull, 1, np.where(bear, 2, 0))
    out = (rng * 3 + eng).astype(float)
    out[~np.isfinite(H1)] = np.nan
    return out


def event_codes(F: pd.DataFrame) -> np.ndarray:
    out = np.full(len(F), EVENTS.index("none"))
    for name in reversed(EVENTS[:-1]):  # highest priority written last
        out[F[name].to_numpy() == 1] = EVENTS.index(name)
    return out


def token_inputs(L: pd.DataFrame, F_free: pd.DataFrame, hmm_state: np.ndarray) -> pd.DataFrame:
    """Raw per-bar inputs of every field. L = base frame (bars + returns + sigma); F_free = parameter-free
    Phase 2 features of the same rows (relvol, vreg_sigma_pct, qqq_z, close_location, event flags, dow);
    hmm_state = filtered most-likely HMM state (NaN = unknown). NaN means MISSING."""
    miss = L["missing"].to_numpy(bool)
    miss1 = np.r_[True, miss[:-1]]
    atr = wilder(true_range(L), 14).to_numpy()
    O, H, Lo, C = (L[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_ok = np.where(atr > 0, atr, np.nan)
        z = L["r"].to_numpy() / L["sigma_hat"].to_numpy()
        gap_z = L["gap"].to_numpy() / L["sigma_hat"].to_numpy()
    k = L["bar_index"].to_numpy()
    out = pd.DataFrame({
        "ret_z": np.where(miss, np.nan, z),
        "body_atr": (C - O) / atr_ok,
        "upper_atr": (H - np.maximum(O, C)) / atr_ok,
        "lower_atr": (np.minimum(O, C) - Lo) / atr_ok,
        "close_location": F_free["close_location"].to_numpy(),
        "range_atr": (H - Lo) / atr_ok,
        "rel": np.where(miss | miss1, np.nan, rel_codes(L)),
        "gap_z": np.where(k == 0, np.where(miss, np.nan, gap_z), 0.0),
        "relvol": F_free["relvol"].to_numpy(),
        "vreg": F_free["vreg_sigma_pct"].to_numpy(),
        "hmm": np.asarray(hmm_state, float),
        "mkt": F_free["qqq_z"].to_numpy(),
        "tod": (k // 6).astype(float),
        "dow": L["session_date"].dt.dayofweek.to_numpy().astype(float),
        "event": event_codes(F_free).astype(float),
        "is_gap_bar": (k == 0),
    }, index=L.index)
    for c in SHAPE_COLS:
        out.loc[miss, c] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)
