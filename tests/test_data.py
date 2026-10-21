"""Cleaning, calendar, split and holdout tests (CLAUDE.md 4.3, 11)."""
import numpy as np
import pandas as pd
import pytest

from tslaq.data.calendar import nyse_sessions
from tslaq.data.clean import aggregate_to_bars, dividend_adjust, drop_bad_sessions, fill_missing
from tslaq.data.download import load_raw_month
from tslaq.utils.io import interim_path, load_config


def _minutes(day: str, start="04:00", end="19:59"):
    ts = pd.date_range(f"{day} {start}", f"{day} {end}", freq="1min", tz="America/New_York")
    n = len(ts)
    p = 100 + np.arange(n) * 0.01
    return pd.DataFrame({"timestamp": ts.tz_convert("UTC"), "open": p, "high": p + .05, "low": p - .05,
                         "close": p + .01, "volume": np.full(n, 10.0), "trade_count": np.ones(n), "vwap": p})


def test_aggregation_regular_session_end_stamped():
    sessions = nyse_sessions("2019-07-01", "2019-07-05")
    m1 = pd.concat([_minutes(d) for d in ["2019-07-01", "2019-07-03", "2019-07-04"]])
    df = fill_missing(aggregate_to_bars(m1, sessions), sessions)
    full = df[df["session_date"] == "2019-07-01"]
    half = df[df["session_date"] == "2019-07-03"]
    assert len(full) == 78 and len(half) == 42 and half["is_half_day"].all()
    assert (df["session_date"] == "2019-07-04").sum() == 0           # holiday dropped
    assert full["timestamp"].iloc[0] == pd.Timestamp("2019-07-01 09:35", tz="America/New_York")
    assert full["timestamp"].iloc[-1] == pd.Timestamp("2019-07-01 16:00", tz="America/New_York")
    # first bar = minutes 09:30..09:34
    m = m1.assign(ts=pd.to_datetime(m1["timestamp"]).dt.tz_convert("America/New_York"))
    blk = m[(m["ts"] >= "2019-07-01 09:30-04:00") & (m["ts"] < "2019-07-01 09:35-04:00")]
    b0 = full.iloc[0]
    assert b0["open"] == blk["open"].iloc[0] and b0["close"] == blk["close"].iloc[-1]
    assert b0["high"] == blk["high"].max() and b0["volume"] == 50
    # session 2019-07-05 has no data at all -> all bars missing -> dropped
    kept, bad = drop_bad_sessions(df, 0.05)
    assert pd.Timestamp("2019-07-05") in set(bad["session_date"]) and not kept["missing"].any()


def test_missing_bar_forward_filled_and_flagged():
    sessions = nyse_sessions("2019-07-01", "2019-07-01")
    m1 = _minutes("2019-07-01")
    ts = pd.to_datetime(m1["timestamp"]).dt.tz_convert("America/New_York")
    m1 = m1[~((ts >= "2019-07-01 10:00-04:00") & (ts < "2019-07-01 10:05-04:00"))]
    df = fill_missing(aggregate_to_bars(m1, sessions), sessions)
    row = df[df["bar_index"] == 6].iloc[0]
    prev = df[df["bar_index"] == 5].iloc[0]
    assert row["missing"] and row["volume"] == 0 and row["close"] == prev["close"] == row["open"]


def test_dividend_adjust_keeps_intraday_returns(bars):
    d = bars.copy()
    sess = d["session_date"].unique()
    ex = sess[50]
    divs = pd.DataFrame({"ex_dividend_date": [ex], "amount": [1.0]})
    adj, fac = dividend_adjust(d, divs)
    same = d["session_date"].shift(1) == d["session_date"]
    r0 = np.log(d["close"] / d["close"].shift(1))[same]
    r1 = np.log(adj["close"] / adj["close"].shift(1))[same]
    np.testing.assert_allclose(r0, r1, atol=1e-12)
    i = np.flatnonzero((d["session_date"] == ex).to_numpy())[0]
    jump = np.log(adj["close"].iat[i] / adj["close"].iat[i - 1]) - np.log(d["close"].iat[i] / d["close"].iat[i - 1])
    assert jump == pytest.approx(-np.log(fac["factor"].iat[0]))


def test_holdout_guard():
    cfg = load_config("data")
    with pytest.raises(PermissionError):
        load_raw_month("TSLA", cfg["holdout_start"][:7], cfg)


@pytest.mark.skipif(not interim_path("TSLA").exists(), reason="interim TSLA bars not built")
def test_split_adjustment():
    df = pd.read_parquet(interim_path("TSLA"))
    r = np.log(df["close"] / df["close"].shift(1))
    for d in ["2020-08-31", "2022-08-25"]:
        on = (df["session_date"] == d).to_numpy()
        assert on.any()
        i = np.flatnonzero(on)[0]
        window = r.iloc[i - 3: i + 3]
        assert window.abs().max() < np.log(1.6), f"split jump near {d}"
        assert abs(r.iat[i]) < 0.2
    assert r.abs().max() < np.log(1.6)   # no 60%+ jump anywhere


@pytest.mark.skipif(not interim_path("TSLA").exists(), reason="interim TSLA bars not built")
def test_interim_has_no_holdout_data():
    cfg = load_config("data")
    for s in cfg["symbols"]:
        df = pd.read_parquet(interim_path(s))
        assert df["session_date"].max() < pd.Timestamp(cfg["holdout_start"])
