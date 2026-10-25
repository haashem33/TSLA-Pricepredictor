"""D22: per-feature null quantiles from the 300-draw mirror null (seeds 2000..2299, D21-calibrated intraday drift,
last-fold fitted-parameter approximation: GARCH-t / HMM / Kalman / frac-d use the 2025-12 fold's parameters for
all folds). Diagnostic, not a trial.

Per draw, feature and target (h = 12, h = 12 delayed), over the 96 monthly folds 2018-01..2025-12:
- t_full = t of the signed out-of-sample D19 IC (sign from that draw's training months), all 96 folds;
- ic_2325, t_2325 = mean and t of the signed IC over 2023-2025 (36 folds);
- abs_t_unsigned = |t| of the unsigned monthly D19 IC over the 96 months (null analogue of the training-side
  stable-IC |t|);
- d10_fixed = the fixed D10 thresholds (t_full >= 2, ic_2325 > 0, t_2325 >= 1), for information.
Per feature: null means and the 84.1 / 95.4 / 97.7% quantiles. Outputs in reports/tables/phase2_rescore/:
null_per_draw_t.csv, null_quantiles.csv, null_quantiles_summary.json.
Usage: python scripts/phase2_null_quantiles.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tslaq.backtest.metrics import fold_ic_stats
from tslaq.utils.io import ROOT

SRC = ROOT / "data" / "processed" / "phase2_metric_null_v2_fallback"
TAB = ROOT / "reports" / "tables" / "phase2_rescore"
Q = (0.841, 0.954, 0.977)
N_EXPECTED = 300


def per_draw(D: pd.DataFrame) -> pd.DataFrame:
    rows = []
    D = D.assign(signed=D["ic"] * D["sign"], year=D["fold"].str[:4].astype(int))
    for (dr, f, k), g in D.groupby(["draw", "feature", "h"]):
        s = g.set_index("fold")["signed"].sort_index()
        if s.notna().sum() == 0:
            continue
        a, b = fold_ic_stats(s), fold_ic_stats(s[g.set_index("fold")["year"].sort_index() >= 2023])
        u = fold_ic_stats(g.set_index("fold")["ic"].sort_index())
        rows.append({"draw": dr, "feature": f, "h": k, "unsigned_mean": g["ic"].mean(), "signed_mean": a["ic"],
                     "t_full": a["t"], "ic_2325": b["ic"], "t_2325": b["t"], "abs_t_unsigned": abs(u["t"]),
                     "d10_fixed": bool(a["t"] >= 2 and b["ic"] > 0 and b["t"] >= 1)})
    return pd.DataFrame(rows)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    files = sorted(SRC.glob("draw_*.parquet"))
    D = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    D["h"] = D["h"].astype(str)
    P = per_draw(D)
    P.to_csv(TAB / "null_per_draw_t.csv", index=False)
    rows = []
    for (f, k), g in P.groupby(["feature", "h"]):
        r = {"feature": f, "h": k, "n_draws": len(g), "null_unsigned_mean": g["unsigned_mean"].mean(),
             "null_unsigned_se": g["unsigned_mean"].std() / np.sqrt(len(g)), "null_signed_mean": g["signed_mean"].mean(),
             "null_t_full_mean": g["t_full"].mean(), "null_t_full_sd": g["t_full"].std(),
             "null_t_2325_mean": g["t_2325"].mean(), "d10_fixed_rate": g["d10_fixed"].mean()}
        for q in Q:
            r[f"t_full_q{100 * q:.1f}"] = g["t_full"].quantile(q)
            r[f"t_2325_q{100 * q:.1f}"] = g["t_2325"].quantile(q)
        r["abs_t_unsigned_q95.4"] = g["abs_t_unsigned"].quantile(0.954)
        rows.append(r)
    T = pd.DataFrame(rows)
    T["abs_null_mean"] = T["null_unsigned_mean"].abs()
    T = T.sort_values(["h", "abs_null_mean"], ascending=[True, False]).drop(columns="abs_null_mean")
    T.to_csv(TAB / "null_quantiles.csv", index=False)
    cal = json.loads((TAB / "null_fallback_calibration.json").read_text())
    s = {"n_draws": int(P["draw"].nunique()), "n_expected": N_EXPECTED, "seeds": cal["seeds"],
         "mu_per_intraday_bar": cal["mu"],
         "null_mean_z12": float(D.groupby("draw")["mean_z12"].first().mean()), "target_mean_z12": cal["target_mean_z12"],
         "approximation": "GARCH-t / HMM / Kalman / frac-d parameters of the last fold (2025-12) used for all folds",
         "pooled_d10_fixed_rate": {k: float(g["d10_fixed"].mean()) for k, g in P.groupby("h")},
         "nominal_d10_rate_iid": 0.01667,
         "pooled_quantiles": {k: {c: float(g[c].quantile(q)) for c in ("t_full", "t_2325") for q in Q}
                              for k, g in P.groupby("h")},
         "max_abs_null_unsigned_mean": {k: float(g["null_unsigned_mean"].abs().max()) for k, g in T.groupby("h")}}
    (TAB / "null_quantiles_summary.json").write_text(json.dumps(s, indent=1))
    pd.set_option("display.width", 250)
    print(json.dumps(s, indent=1))
    for k, g in T.groupby("h"):
        print(k)
        print(g.head(20).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
