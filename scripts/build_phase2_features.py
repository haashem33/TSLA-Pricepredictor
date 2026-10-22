"""Build and cache the Phase 2 feature frame (dev period only; the holdout is never loaded).

Outputs (data/processed/, gitignored): phase2_features_<v>.parquet, phase2_labels_<v>.parquet
and reports/tables/phase2_fitted_params_<v>.json (yearly cutoffs).

Usage: python scripts/build_phase2_features.py
"""
from __future__ import annotations

import json
import time
import warnings

import pandas as pd

from tslaq.data.pipeline import choose_lambda, load_bars
from tslaq.features.build import build_features, fit_params
from tslaq.features.returns import add_returns
from tslaq.features.volatility import add_sigma
from tslaq.labels.forward_returns import add_labels
from tslaq.utils.io import ROOT, interim_path, load_config

warnings.filterwarnings("ignore")
CUTOFFS = [f"{y}-01-01" for y in range(2018, 2026)]
DECAY_H = tuple(range(1, 25))


def base(symbol: str, cfg: dict, wf: dict, horizons) -> tuple[pd.DataFrame, float]:
    v = cfg["volatility"]
    raw = load_bars(symbol)
    lam, _ = choose_lambda(raw, wf["first_val_month"], cfg)
    df = add_sigma(add_returns(raw), lam, v["seasonal_window_sessions"], v["burn_in_sessions"], horizons)
    return df, lam


def main() -> None:
    t0 = time.time()
    cfg, bcfg = load_config("data"), load_config("backtest")
    version = cfg["data_version"]
    wf = bcfg["walk_forward"]
    tsla, lam = base("TSLA", cfg, wf, DECAY_H)
    tsla = add_labels(tsla, DECAY_H)
    mkt, lams = {}, {"TSLA": lam}
    for s in ["SPY", "QQQ"]:
        mkt[s.lower()], lams[s] = base(s, cfg, wf, ())
    print(f"base built {time.time() - t0:.0f}s, lambdas {lams}", flush=True)
    react = pd.read_parquet(interim_path("TSLA", version).with_name("TSLA_earnings_sessions.parquet"))["reaction_session"]
    params = []
    for c in CUTOFFS:
        params.append(fit_params(tsla, c))
        print(f"fit {c}: {params[-1]['fit_seconds']}s, frac_d={params[-1]['frac_d']}", flush=True)
    F = build_features(tsla, mkt, params, react)
    print(f"features {F.shape} {time.time() - t0:.0f}s", flush=True)
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    F.to_parquet(out / f"phase2_features_{version}.parquet")
    keep = ["timestamp", "session_date", "bar_index", "is_half_day", "missing", "close", "r", "sigma_hat", "level2"]
    lab = [f"{p}_{h}" for h in DECAY_H for p in ("fwd", "z", "S", "label_end")]
    tsla[keep + lab].to_parquet(out / f"phase2_labels_{version}.parquet")
    js = []
    for P in params:
        d = {k: v for k, v in P.items() if k != "hmm"}
        d["hmm"] = P["hmm"].as_dict()
        js.append(d)
    (ROOT / "reports" / "tables" / f"phase2_fitted_params_{version}.json").write_text(
        json.dumps({"lambdas": lams, "cutoffs": js}, indent=1, default=float))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
