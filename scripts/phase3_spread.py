"""D25 (ii): TSLA effective bid-ask spread per year 2016-2025 from 1-minute regular-session bars (information only).

Sources: unadjusted Alpaca 1-min bars 2016-01..2022-08 (data/raw/TSLA_raw, D9) and the vendor 1-min bars from
2022-09 (data/raw/TSLA; raw == adjusted from then on). Regular session = bar start 09:30..15:59 US/Eastern.
Per session: EDGE (Ardia, Guidotti & Kroencke 2024, package `bidask`) on the session's 1-min OHLC, and the
Corwin-Schultz 2-bar estimator on consecutive 1-min bars (gap adjustment of CS 2012, negative estimates set to 0,
daily mean). Per year: mean and median of the daily estimates, full spread and half-spread in bp.

Usage: python scripts/phase3_spread.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from bidask import edge

from tslaq.utils.io import ROOT

RAW = ROOT / "data" / "raw"
TAB = ROOT / "reports" / "tables" / "phase3_baselines"
K = 3 - 2 * np.sqrt(2)


def month_files() -> list:
    out = []
    for p in pd.period_range("2016-01", "2025-12", freq="M"):
        src = RAW / ("TSLA_raw" if p <= pd.Period("2022-08", "M") else "TSLA") / f"{p}.parquet"
        out.append(src)
    return out


def corwin_schultz(o, h, lo, c) -> float:
    """Mean over consecutive bar pairs of the CS (2012) 2-bar spread, gap-adjusted, negatives set to 0."""
    h1, l1 = h[1:].copy(), lo[1:].copy()
    cp = c[:-1]
    up = l1 > cp  # next bar entirely above the previous close: shift down
    dn = h1 < cp  # entirely below: shift up
    adj = np.where(up, l1 - cp, np.where(dn, h1 - cp, 0.0))
    h1, l1 = h1 - adj, l1 - adj
    h0, l0 = h[:-1], lo[:-1]
    beta = np.log(h0 / l0) ** 2 + np.log(h1 / l1) ** 2
    gamma = np.log(np.maximum(h0, h1) / np.minimum(l0, l1)) ** 2
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / K - np.sqrt(gamma / K)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(np.mean(np.maximum(s, 0.0)))


def main():
    rows = []
    for f in month_files():
        d = pd.read_parquet(f)
        ts = d["timestamp"].dt.tz_convert("America/New_York")
        mins = ts.dt.hour * 60 + ts.dt.minute
        d = d[(mins >= 570) & (mins < 960)].assign(session=ts.dt.date[(mins >= 570) & (mins < 960)])
        for s, g in d.groupby("session"):
            g = g.sort_values("timestamp")
            o, h, lo, c = (g[k].to_numpy(float) for k in ("open", "high", "low", "close"))
            if len(g) < 30:
                continue
            rows.append({"session": s, "n_bars": len(g), "edge": float(edge(o, h, lo, c)),
                         "cs": corwin_schultz(o, h, lo, c), "price": float(np.median(c))})
        print(f.name, flush=True)
    D = pd.DataFrame(rows)
    D["year"] = pd.to_datetime(D["session"]).dt.year
    TAB.mkdir(parents=True, exist_ok=True)
    D.to_csv(TAB / "spread_daily.csv", index=False)
    Y = D.groupby("year").agg(n_sessions=("session", "size"), median_price=("price", "median"),
                              edge_mean=("edge", "mean"), edge_median=("edge", "median"),
                              cs_mean=("cs", "mean"), cs_median=("cs", "median"), edge_nan=("edge", lambda x: x.isna().sum()))
    for k in ("edge_mean", "edge_median", "cs_mean", "cs_median"):
        Y[f"{k}_full_bp"] = Y[k] * 1e4
        Y[f"{k}_half_bp"] = Y[k] * 1e4 / 2
    Y["tick_bp_at_median_price"] = 0.01 / Y["median_price"] * 1e4
    Y = Y.drop(columns=["edge_mean", "edge_median", "cs_mean", "cs_median"]).reset_index()
    Y.to_csv(TAB / "spread_by_year.csv", index=False)
    pd.set_option("display.width", 250)
    print(Y.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
