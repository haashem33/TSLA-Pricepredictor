"""Phase 1 follow-up checks by data version (reports/phase1_checks.md).

B1: by year, share of exact-zero 5-min returns, lag-1 autocorrelation (intraday bars), and share of
    regular-session 1-min closes that are exact multiples of $0.01 (in the price series that version uses).
B2: LightGBM baseline fold IC (z space) with and without one bar of execution delay.

Usage: python scripts/phase1_checks.py v2 [--b2]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from tslaq.backtest.metrics import fold_ic, fold_ic_stats
from tslaq.data.clean import split_factor
from tslaq.data.download import load_raw_month, month_range
from tslaq.data.pipeline import build_base
from tslaq.utils.io import ROOT, interim_path, load_config

OUT = ROOT / "reports" / "tables"


def one_min_closes(symbol: str, year: int, version: str, cfg: dict) -> np.ndarray:
    v2 = cfg["v2"]
    months = month_range(f"{year}-01", f"{year + 1}-01")
    if version == "v2" and symbol == cfg["target"]:
        parts = [pd.read_parquet(ROOT / v2["raw_dir"] / f"{m}.parquet") if m < v2["raw_end_month_exclusive"]
                 else load_raw_month(symbol, m, cfg) for m in months]
    else:
        parts = [load_raw_month(symbol, m, cfg) for m in months]
    x = pd.concat(parts, ignore_index=True)
    ts = x["timestamp"].dt.tz_convert("America/New_York")
    t = ts.dt.hour * 60 + ts.dt.minute
    x = x[(t >= 570) & (t < 960)]
    c = x["close"].to_numpy()
    if version == "v2" and symbol == cfg["target"]:
        day = pd.Series(ts[(t >= 570) & (t < 960)].dt.tz_localize(None).dt.normalize().to_numpy())
        c = c / split_factor(day, pd.read_csv(ROOT / v2["splits_file"]))
    return c


def b1(version: str) -> pd.DataFrame:
    cfg = load_config("data")
    rows = []
    for s in cfg["symbols"]:
        d = pd.read_parquet(interim_path(s, version))
        d = d.assign(r=np.log(d["close"] / d["close"].shift(1)), yr=d["session_date"].dt.year)
        d = d[(d["bar_index"] > 0) & ~d["missing"]]
        for y, g in d.groupby("yr"):
            rr = g["r"].dropna()
            c = one_min_closes(s, y, version, cfg) * 100
            rows.append({"version": version, "sym": s, "year": y, "zero_r": float((rr == 0).mean()),
                         "ac1": float(rr.autocorr(1)), "cent_multiple_1min": float(np.mean(np.abs(c - np.round(c)) < 1e-6)),
                         "ret_sd_bp": float(rr.std() * 1e4)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"check_b1_{version}.csv", index=False)
    return out


def b2(version: str) -> pd.DataFrame:
    cfg = load_config("data")
    lam = pd.read_csv(OUT / f"phase1_lambda_tuning_{version}.csv").sort_values("qlike")["lambda"].iloc[0]
    df = build_base(cfg["target"], float(lam), cfg, version)
    R = pd.read_parquet(ROOT / "data" / "processed" / f"phase1_return_preds_{version}.parquet")
    R = R[R["model"] == "lgb_returns"]
    sess = df["session_date"].to_numpy()
    rows = []
    for h in (1, 3, 12):
        z = df[f"z_{h}"].to_numpy()
        r = R[R["h"] == h]
        p = r["pos"].to_numpy()
        nxt = np.minimum(p + 1, len(df) - 1)
        ok = (sess[nxt] == sess[p]) & np.isfinite(z[nxt])
        for name, y in [("no_delay", z[p]), ("delay_1bar", np.where(ok, z[nxt], np.nan))]:
            fi = fold_ic(r["pred_z"].to_numpy()[ok], y[ok], r["fold"].to_numpy()[ok])
            yr = pd.to_datetime(fi.index).year
            for lab, v in [("2018-2025", fi), ("2023-2025", fi[yr >= 2023]), ("last24", fi.iloc[-24:])]:
                rows.append({"version": version, "h": h, "variant": name, "period": lab, **fold_ic_stats(v)})
            for y_, v in fi.groupby(yr):
                rows.append({"version": version, "h": h, "variant": name, "period": str(y_), "ic": v.mean()})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"check_b2_{version}.csv", index=False)
    return out


if __name__ == "__main__":
    v = sys.argv[1]
    pd.set_option("display.width", 220)
    if "--b2" in sys.argv:
        print(b2(v).round(4).to_string(index=False))
    else:
        print(b1(v).round(4).to_string(index=False))
