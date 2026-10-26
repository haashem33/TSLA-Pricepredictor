"""Unadjusted (Adjustment.RAW) Alpaca SIP 1-minute bars for TSLA -> data/raw/TSLA_raw/<YYYY-MM>.parquet.

Keys come from the repo's gitignored .env (ALPACA_API_KEY / ALPACA_SECRET_KEY) and are never printed.
Same columns as the user's split-adjusted source files: timestamp (UTC, bar start), open, high, low,
close, volume, trade_count, vwap. Cached months are skipped; months in the holdout are refused.
"""
from __future__ import annotations

import os
import sys

import pandas as pd

from tslaq.data.download import RAW, _guard_holdout, _log
from tslaq.utils.io import ROOT, load_config

RAW_DIR = RAW / "TSLA_raw"


def _load_env() -> None:
    env = ROOT / ".env"
    if not env.exists():
        raise SystemExit(".env missing")
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _client():
    from alpaca.data.historical import StockHistoricalDataClient

    _load_env()
    return StockHistoricalDataClient(os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"])


def fetch_month(client, symbol: str, month: str, adjustment: str = "raw", end_utc: pd.Timestamp | None = None
                ) -> pd.DataFrame:
    """One month of SIP 1-minute bars (UTC month boundaries). `adjustment` = "raw" or "split"; `end_utc` caps the
    request (D28: nothing after the holdout end is requested)."""
    from alpaca.data.enums import Adjustment, DataFeed
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.offsets.MonthBegin(1)
    if end_utc is not None:
        end = min(end, end_utc)
    adj = {"raw": Adjustment.RAW, "split": Adjustment.SPLIT}[adjustment]
    req = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Minute, start=start.to_pydatetime(),
                           end=end.to_pydatetime(), adjustment=adj, feed=DataFeed.SIP)
    df = client.get_stock_bars(req).df.reset_index()
    df = df.drop(columns="symbol")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]]


def download_raw(months: list[str], symbol: str = "TSLA") -> None:
    cfg = load_config("data")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = None
    done = 0
    for m in months:
        _guard_holdout(m, cfg)
        p = RAW_DIR / f"{m}.parquet"
        if p.exists():
            done += 1
            continue
        client = client or _client()
        df = fetch_month(client, symbol, m)
        df.to_parquet(p, index=False)
        _log(f"{symbol}_raw", m, "ok")
        done += 1
        print(f"{m}: {len(df)} rows", flush=True)
    print(f"progress: {done}/{len(months)} months cached")


if __name__ == "__main__":
    from tslaq.data.download import month_range

    cfg = load_config("data")
    v2 = cfg["v2"]
    months = month_range(v2["raw_start_month"], v2["raw_end_month_exclusive"]) + v2["check_months"]
    download_raw(months if len(sys.argv) < 2 else sys.argv[1:])
