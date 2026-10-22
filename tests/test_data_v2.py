"""Data v2: TSLA from unadjusted prices with float split adjustment (decision D9)."""
import numpy as np
import pandas as pd
import pytest

from tslaq.data.calendar import nyse_sessions
from tslaq.data.clean import aggregate_to_bars, fill_missing, split_factor
from tslaq.utils.io import ROOT, interim_path, load_config

CFG = load_config("data")
RAW = ROOT / CFG["v2"]["raw_dir"]
SPLITS = pd.read_csv(ROOT / CFG["v2"]["splits_file"])
need_v2 = pytest.mark.skipif(not interim_path("TSLA", "v2").exists() or not RAW.exists(), reason="v2 not built")


def test_split_factor():
    d = pd.Series(pd.to_datetime(["2020-08-28", "2020-08-31", "2022-08-24", "2022-08-25"]))
    np.testing.assert_array_equal(split_factor(d, SPLITS), [15, 3, 3, 1])


@need_v2
@pytest.mark.parametrize("month", CFG["v2"]["check_months"])
def test_v2_rebuild_matches_v1_post_split(month):
    raw = pd.read_parquet(RAW / f"{month}.parquet")
    s = nyse_sessions(f"{month}-01", str(pd.Period(month, "M").end_time.date()))
    b = fill_missing(aggregate_to_bars(raw, s), s)
    v1 = pd.read_parquet(interim_path("TSLA", "v1"))
    a = v1[v1["session_date"].dt.to_period("M").astype(str) == month].reset_index(drop=True)
    assert len(a) == len(b)
    for c in ["timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"]:
        assert (a[c].to_numpy() == b[c].to_numpy()).all(), c


@need_v2
@pytest.mark.parametrize("day,ratio,prev", [("2020-08-31", 5, "2020-08-28"), ("2022-08-25", 3, "2022-08-24")])
def test_v2_split_day_bar0_explicit(day, ratio, prev):
    raw = pd.read_parquet(RAW / f"{day[:7]}.parquet")
    ts = raw["timestamp"].dt.tz_convert("America/New_York")
    o = raw["open"][ts == pd.Timestamp(f"{day} 09:30", tz="America/New_York")].iloc[0]
    sess = (ts.dt.date.astype(str) == prev) & (ts.dt.hour * 60 + ts.dt.minute < 960)
    c = raw["close"][sess].iloc[-1]
    v2 = pd.read_parquet(interim_path("TSLA", "v2"))
    i = np.flatnonzero(((v2["session_date"] == day) & (v2["bar_index"] == 0)).to_numpy())[0]
    assert np.log(v2["open"].iat[i] / v2["close"].iat[i - 1]) == pytest.approx(np.log(o / (c / ratio)), abs=1e-12)


@need_v2
def test_v2_prices_not_rounded_and_close_to_v1():
    v1 = pd.read_parquet(interim_path("TSLA", "v1"))
    v2 = pd.read_parquet(interim_path("TSLA", "v2"))
    m = v1.merge(v2, on="timestamp", suffixes=("1", "2"))
    pre = m[m["timestamp"] < pd.Timestamp("2020-01-01", tz="America/New_York")]
    cents = pre["close2"] * 100
    assert (np.abs(cents - np.round(cents)) < 1e-6).mean() < 0.2      # v1 was 100% cent multiples
    assert (pre["close2"] / pre["close1"] - 1).abs().max() < 0.01 / 10  # within rounding of $0.01 at >$10
