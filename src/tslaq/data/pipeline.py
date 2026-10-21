"""Phase 1 base frame: canonical bars + returns + sigma_hat/S_h + labels."""
from __future__ import annotations

import pandas as pd

from tslaq.features.returns import add_returns
from tslaq.features.volatility import add_sigma, tune_lambda
from tslaq.labels.forward_returns import add_labels
from tslaq.utils.io import interim_path, load_config

HORIZONS = (1, 3, 12)


def load_bars(symbol: str) -> pd.DataFrame:
    df = pd.read_parquet(interim_path(symbol))
    return df.sort_values("timestamp").reset_index(drop=True)


def choose_lambda(df: pd.DataFrame, train_end_month: str, cfg: dict) -> tuple[float, pd.DataFrame]:
    """Tune lambda on bars strictly before `train_end_month` (the first walk-forward training window)."""
    v = cfg["volatility"]
    train = df[df["session_date"] < pd.Timestamp(f"{train_end_month}-01")]
    tab = tune_lambda(add_returns(train), v["lambda_grid"], v["seasonal_window_sessions"], v["burn_in_sessions"])
    return float(tab.loc[tab["qlike"].idxmin(), "lambda"]), tab


def build_base(symbol: str, lam: float, cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config("data")
    v = cfg["volatility"]
    df = add_returns(load_bars(symbol))
    df = add_sigma(df, lam, v["seasonal_window_sessions"], v["burn_in_sessions"], HORIZONS)
    return add_labels(df, HORIZONS)
