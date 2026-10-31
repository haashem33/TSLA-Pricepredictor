"""D29 / D29a SPY intraday momentum: no lookahead, half-day sigma rule, fills, flat at close, costs, rounding."""
import numpy as np
import pandas as pd
import pytest

from tslaq.spy import momentum as mo


def synth(n_sessions=40, half_days=(20,), seed=0, raw_scale=1.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-02", periods=n_sessions)
    rows, p = [], 100.0
    for i, d in enumerate(dates):
        n = 210 if i in half_days else 390
        p *= np.exp(rng.normal(0, 0.005))  # overnight
        r = rng.normal(0, 0.0007, n)
        c = p * np.exp(np.cumsum(r))
        o = np.r_[p, c[:-1]]
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 2e-4, n)))
        lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 2e-4, n)))
        rows.append(pd.DataFrame({"session_date": d, "minute": np.arange(1, n + 1), "open": o, "high": h, "low": lo,
                                  "close": c, "volume": rng.integers(1_000, 10_000, n).astype(float),
                                  "n_minutes": n, "is_half_day": n < 390, "missing": False}))
        p = c[-1]
    df = pd.concat(rows, ignore_index=True)
    df["raw_open"], df["raw_close"] = df["open"] * raw_scale, df["close"] * raw_scale
    return df


COLS = ["sigma", "ub", "lb", "vwap", "sigma_spy", "prev_close", "day_open"]


def test_indicators_no_lookahead_truncation():
    df = synth()
    full = mo.add_indicators(df)
    rng = np.random.default_rng(1)
    for _ in range(30):
        cut = int(rng.integers(len(df) // 2, len(df)))
        part = mo.add_indicators(df.iloc[:cut])
        a = full.iloc[:cut][COLS].to_numpy()
        b = part[COLS].to_numpy()
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=0, atol=0)


def test_future_sessions_do_not_change_past_indicators():
    df = synth()
    full = mo.add_indicators(df)
    d_cut = df["session_date"].unique()[30]
    alt = df.copy()
    later = alt["session_date"] > d_cut
    for c in ["open", "high", "low", "close"]:
        alt.loc[later, c] *= 1.5
    alt.loc[later, "volume"] *= 7
    other = mo.add_indicators(alt)
    keep = full["session_date"] <= d_cut
    pd.testing.assert_frame_equal(full.loc[keep, COLS], other.loc[keep, COLS])


def test_sigma_uses_sessions_containing_the_minute_d29a1():
    df = synth(n_sessions=40, half_days=(20,))
    ind = mo.add_indicators(df)
    dates = df["session_date"].unique()
    mv = ind.pivot(index="session_date", columns="minute", values="move")
    # session 25, minute 300: the 14 previous sessions containing minute 300 skip the half-day (index 20)
    prev = [d for d in dates[:25] if not np.isnan(mv.loc[d, 300])][-14:]
    assert dates[20] not in prev and dates[10] in prev
    got = ind[(ind["session_date"] == dates[25]) & (ind["minute"] == 300)]["sigma"].iloc[0]
    assert got == pytest.approx(mv.loc[prev, 300].mean(), abs=1e-15)
    # minute 100 (exists on the half-day) uses the 14 previous calendar sessions including it
    got100 = ind[(ind["session_date"] == dates[25]) & (ind["minute"] == 100)]["sigma"].iloc[0]
    assert got100 == pytest.approx(mv.loc[dates[11:25], 100].mean(), abs=1e-15)
    # afternoon trading is not blocked after the half-day
    assert np.isfinite(ind[(ind["session_date"] == dates[21]) & (ind["minute"] == 330)]["sigma"].iloc[0])
    # with only 13 earlier sessions containing minute 300, sigma is NaN
    df2 = synth(n_sessions=20, half_days=(5,))
    ind2 = mo.add_indicators(df2)
    d2 = df2["session_date"].unique()
    s300 = ind2[(ind2["session_date"] == d2[14]) & (ind2["minute"] == 300)]["sigma"].iloc[0]
    s100 = ind2[(ind2["session_date"] == d2[14]) & (ind2["minute"] == 100)]["sigma"].iloc[0]
    assert np.isnan(s300) and np.isfinite(s100)
    s300b = ind2[(ind2["session_date"] == d2[15]) & (ind2["minute"] == 300)]["sigma"].iloc[0]
    assert np.isfinite(s300b)


def test_sigma_spy_definition():
    df = synth()
    ind = mo.add_indicators(df)
    close = df.groupby("session_date")["close"].last()
    ret = close.pct_change()
    d = close.index[20]
    expect = ret.iloc[6:20].std(ddof=1)  # returns of sessions 6..19 (the 14 before session 20)
    got = ind[ind["session_date"] == d]["sigma_spy"].iloc[0]
    assert got == pytest.approx(expect, rel=1e-12)
    assert np.isnan(ind[ind["session_date"] == close.index[14]]["sigma_spy"].iloc[0])
    assert np.isfinite(ind[ind["session_date"] == close.index[15]]["sigma_spy"].iloc[0])


def _forced_day(df, sign_by_minute):
    """Indicators with bands/VWAP overwritten so the signal at given decision minutes is known."""
    ind = mo.add_indicators(df)
    last = ind["session_date"] == ind["session_date"].max()
    c_all = ind.loc[last, "close"]
    ind.loc[last, "ub"], ind.loc[last, "lb"], ind.loc[last, "vwap"] = c_all * 10, c_all * 0.1, c_all  # no band crossed
    for m, s in sign_by_minute.items():
        k = last & (ind["minute"] == m)
        c = ind.loc[k, "close"].iloc[0]
        if s > 0:
            ind.loc[k, ["ub", "lb", "vwap"]] = [c * 0.99, c * 0.98, c * 0.99]
        elif s < 0:
            ind.loc[k, ["ub", "lb", "vwap"]] = [c * 1.02, c * 1.01, c * 1.01]
    return ind[last]


@pytest.mark.parametrize("delay", [0, 1])
def test_fill_next_open_and_flat_at_close(delay):
    df = synth(n_sessions=20, half_days=(), raw_scale=2.0)
    day = _forced_day(df, {30: 1, 60: 1, 90: 0, 120: -1})
    plan = mo.day_plans(day, "primary", delay=delay)[0]
    assert plan.tradable
    ro = day["raw_open"].to_numpy()
    rc = day["raw_close"].to_numpy()
    prices = [f[0] for f in plan.fills]
    signs = [f[1] for f in plan.fills]
    assert prices[:4] == [ro[30 + delay], ro[60 + delay], ro[90 + delay], ro[120 + delay]]  # minute m+1+delay
    assert signs[:5] == [1, 1, 0, -1, 0]
    assert plan.fills[-1] == (rc[-1], 0)  # flat at the last close
    daily, trades = mo.simulate([plan], "primary", "gross", {2020: 0.0}, aum0=100_000)
    N = daily["shares"].iloc[0]
    expect = N * (ro[90 + delay] - ro[30 + delay]) - N * (ro[150 + delay] - ro[120 + delay])
    assert daily["pnl_gross"].iloc[0] == pytest.approx(expect, rel=1e-12)
    assert len(trades) == 2 and set(trades["sign"]) == {1, -1}
    assert daily["n_orders"].iloc[0] == 4  # buy, sell, sell short, cover


def test_no_position_after_close_and_signal_unaffected_by_later_prices():
    df = synth(n_sessions=20, half_days=())
    day = _forced_day(df, {360: 1})  # last decision 15:30, long into the close
    plan = mo.day_plans(day, "primary")[0]
    daily, _trades = mo.simulate([plan, plan], "primary", "gross", {2020: 0.0})
    assert (daily["n_orders"] == 2).all()
    ro, rc = day["raw_open"].to_numpy(), day["raw_close"].to_numpy()
    N = daily["shares"].iloc[0]
    assert daily["pnl_gross"].iloc[0] == pytest.approx(N * (rc[-1] - ro[360]))
    # changing prices after the decision bar does not change the signal at that decision
    alt = day.copy()
    alt.loc[alt["minute"] > 360, ["open", "high", "low", "close", "raw_open", "raw_close"]] *= 0.5
    assert mo.decision_signals(alt, "primary") == mo.decision_signals(day, "primary")


def test_half_day_decisions_stop_at_1230():
    df = synth(n_sessions=20, half_days=(19,))
    ind = mo.add_indicators(df)
    day = ind[ind["session_date"] == ind["session_date"].max()]
    sig = mo.decision_signals(day, "primary")
    assert [m for m, _ in sig] == list(range(30, 181, 30))
    plan = mo.day_plans(day, "primary")[0]
    assert plan.fills[-1][0] == day["raw_close"].to_numpy()[209]


def test_opp_band_holds_until_opposite_band():
    df = synth(n_sessions=20, half_days=())
    day = _forced_day(df, {30: 1, 60: 0, 90: -1})
    # minute 60: inside the noise area (neither band crossed) -> keep long
    sig = dict(mo.decision_signals(day, "opp_band_1x"))
    assert sig[30] == 1 and sig[60] == 1 and sig[90] == -1 and sig[120] == -1
    sig_p = dict(mo.decision_signals(day, "primary"))
    assert sig_p[60] == 0 and sig_p[120] == 0


def test_order_costs():
    # paper: $0.0045 per share, no minimum
    assert mo.order_cost(50, 400.0, "paper", 0.3) == pytest.approx(50 * 0.0045)
    # real, small order: $0.35 minimum commission binds; half-tick floor binds over 0.05 bp EDGE
    c = mo.order_cost(50, 400.0, "real", 0.05)
    assert c == pytest.approx(50 * 0.005 + 0.35 + 0.10e-4 * 50 * 400)
    # real, large order: EDGE above the half tick (0.2 bp of $500 = $0.01 > $0.005) and per-share commission
    c = mo.order_cost(1000, 500.0, "real", 0.2)
    assert c == pytest.approx(0.2e-4 * 500_000 + 3.5 + 0.10e-4 * 500_000)
    assert mo.order_cost(1000, 500.0, "real2x", 0.2) == pytest.approx(2 * c)
    assert mo.order_cost(1000, 500.0, "gross", 0.2) == 0.0
    assert mo.order_cost(0, 500.0, "real", 0.2) == 0.0


def test_share_rounding_and_leverage_cap():
    assert mo.shares_for_day(100_000, 333.0, 0.01, "primary") == round(100_000 / 333.0 * 2)  # 600.6 -> 601
    assert mo.shares_for_day(100_000, 333.0, 0.001, "primary") == round(100_000 / 333.0 * 4)  # cap 4x
    assert mo.shares_for_day(100_000, 333.0, 0.04, "primary") == round(100_000 / 333.0 * 0.5)
    assert mo.shares_for_day(100_000, 333.0, 0.001, "vwap_1x") == round(100_000 / 333.0)


def test_flip_is_one_order_with_cost_split():
    df = synth(n_sessions=20, half_days=())
    day = _forced_day(df, {30: 1, 60: -1})
    plan = mo.day_plans(day, "primary")[0]
    daily, trades = mo.simulate([plan], "primary", "real", {2020: 0.5})
    assert daily["n_orders"].iloc[0] == 3  # buy N, sell 2N, cover N
    assert trades["cost"].sum() == pytest.approx(daily["cost"].iloc[0])
    assert daily["long_net"].iloc[0] + daily["short_net"].iloc[0] == pytest.approx(
        daily["pnl_gross"].iloc[0] - daily["cost"].iloc[0])


def test_stage_a_loaders_refuse_or_filter_later_data():
    from tslaq.spy import data as sd
    with pytest.raises(PermissionError):
        sd.load_raw_1min(["2024-03"])
    with pytest.raises(PermissionError):
        sd.build_spy_1min("2024-03")
    if sd.interim_path("SPY", "v2").exists():
        v = sd.read_v2("v2", pd.Timestamp("2024-02-29"), ["session_date"])
        assert v["session_date"].max() <= pd.Timestamp("2024-02-29")
