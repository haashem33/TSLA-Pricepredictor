"""Raw-data acquisition with a local cache in data/raw/<symbol>/<YYYY-MM>.<ext>.

Two sources share one interface (`fetch_month(symbol, month) -> raw DataFrame`):
- ``local_alpaca``: copies the user's existing Alpaca SIP 1-minute files (default; no API calls).
- ``alphavantage``: TIME_SERIES_INTRADAY 5-min, month by month, rate-limited and resumable.

Months on or after ``holdout_start`` are never fetched or copied (CLAUDE.md 10.5).
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime

import pandas as pd

from tslaq.utils.io import ROOT, load_config

RAW = ROOT / "data" / "raw"
LOG = RAW / "download_log.csv"


class RateLimited(RuntimeError):
    pass


def month_range(start: str, end_exclusive: str) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end_exclusive, freq="M")[:-1]]


def _log(symbol: str, month: str, status: str) -> None:
    new = not LOG.exists()
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["symbol", "month", "status", "timestamp"])
        w.writerow([symbol, month, status, datetime.now(UTC).isoformat(timespec="seconds")])


def cached_path(symbol: str, month: str, source: str):
    ext = "parquet" if source == "local_alpaca" else "json"
    return RAW / symbol / f"{month}.{ext}"


def _guard_holdout(month: str, cfg: dict) -> None:
    if month >= cfg["holdout_start"][:7]:
        raise PermissionError(f"{month} is inside the final holdout (>= {cfg['holdout_start']}); not loading.")


def _fetch_local(symbol: str, month: str, cfg: dict) -> None:
    src = os.path.join(cfg["local_source_dir"], f"{symbol}_1min_sip_{month}.parquet")
    if not os.path.exists(src):
        raise FileNotFoundError(src)
    dst = cached_path(symbol, month, "local_alpaca")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def _fetch_alphavantage(symbol: str, month: str, cfg: dict) -> None:
    key = os.environ.get(cfg["alphavantage"]["api_key_env"])
    if not key:
        raise RuntimeError("ALPHAVANTAGE_API_KEY not set")
    q = dict(function="TIME_SERIES_INTRADAY", symbol=symbol, interval="5min", adjusted="true",
             extended_hours="false", outputsize="full", month=month, apikey=key)
    url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(q)
    with urllib.request.urlopen(url, timeout=60) as r:
        payload = json.loads(r.read())
    if any(k in payload for k in ("Information", "Note")) or "Error Message" in payload:
        raise RateLimited(str(payload)[:300])
    dst = cached_path(symbol, month, "alphavantage")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload), encoding="utf-8")


def download_all(cfg: dict | None = None) -> dict:
    """Fetch every missing (symbol, month); skip cached ones. Returns progress counts."""
    cfg = cfg or load_config("data")
    src = cfg["source"]
    months = month_range(cfg["start_month"], cfg["holdout_start"][:7])
    needed = [(s, m) for s in cfg["symbols"] for m in months]
    done = 0
    last_call = 0.0
    for s, m in needed:
        _guard_holdout(m, cfg)
        if cached_path(s, m, src).exists():
            done += 1
            continue
        try:
            if src == "alphavantage":
                wait = cfg["alphavantage"]["min_seconds_between_calls"] - (time.time() - last_call)
                if wait > 0:
                    time.sleep(wait)
                last_call = time.time()
                _fetch_alphavantage(s, m, cfg)
            else:
                _fetch_local(s, m, cfg)
            _log(s, m, "ok")
            done += 1
        except RateLimited as e:
            _log(s, m, "rate_limited")
            print(f"Rate limit hit at {s} {m}: {e}. Stopping; rerun to resume.")
            break
        except FileNotFoundError:
            _log(s, m, "missing_source")
    print(f"progress: {done}/{len(needed)} months cached")
    return {"done": done, "needed": len(needed)}


def load_raw_month(symbol: str, month: str, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config("data")
    _guard_holdout(month, cfg)
    return pd.read_parquet(cached_path(symbol, month, cfg["source"]))


if __name__ == "__main__":
    download_all()
