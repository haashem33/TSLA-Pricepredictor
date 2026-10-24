"""D21 step B2: drift sensitivity of the D19 IC per feature, from the paired 2026-10-24 mirror-null draws (no new
compute for the sensitivity itself).

Draws 0..39 exist with no drift and with the 15x all-bar drift (same mirror seeds 1000..1039). Per feature and
target (h = 12, h = 12 delayed): sensitivity = mean over pairs of (unsigned IC drift - no drift) / mean over pairs
of (mean z_12 drift - no drift), i.e. IC change per unit of mean z_12 (the D19 IC is linear in the target mean
through the ybar * mean(s) term). Implied drift-induced bias at k x real = sensitivity * k * 0.00192 (real
regular-session mean z_12); implied total = pooled no-drift mean (60 draws) + that. Features whose implied
drift-induced bias at 1x exceeds +-0.003 are flagged.

If the corrected drift1x draws (D21) exist, the implied value for each feature is also compared with the measured
one: predicted = no-drift mean over the same draws + sensitivity * (mean z_12 shift of those pairs).
Usage: python scripts/phase2_metric_drift_sensitivity.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tslaq.utils.io import ROOT

REAL_Z12 = 0.00192391473467247
T2 = ROOT / "reports" / "tables" / "phase2_metric_fix_v2"
T3 = ROOT / "reports" / "tables" / "phase2_metric_fix_v3"


def main():
    T3.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(T2 / "null_v2_per_draw_feature.csv")
    v3 = T3 / "null_v3_per_draw_feature.csv"
    if v3.exists():
        Q = pd.read_csv(v3)
        P = pd.concat([P, Q[Q["variant"] == "drift1x"]], ignore_index=True)
    P = P[P["n_folds"] > 0]
    P["h"] = P["h"].astype(str)
    rows = []
    for (f, h), g in P.groupby(["feature", "h"]):
        W = g.pivot_table(index="draw", columns="variant", values=["unsigned", "mean_z12"])
        pr = W.dropna(subset=[("unsigned", "nodrift"), ("unsigned", "drift")])
        d_ic = pr[("unsigned", "drift")] - pr[("unsigned", "nodrift")]
        d_z = pr[("mean_z12", "drift")] - pr[("mean_z12", "nodrift")]
        sens = d_ic.mean() / d_z.mean()
        nod = g[g["variant"] == "nodrift"]["unsigned"].mean()
        r = {"feature": f, "h": h, "n_pairs": len(pr), "mean_z12_shift_15x": d_z.mean(),
             "ic_shift_15x": d_ic.mean(), "ic_shift_15x_sd": d_ic.std(), "sensitivity": sens,
             "sensitivity_se": d_ic.std() / np.sqrt(len(pr)) / d_z.mean(), "nodrift_mean": nod,
             "implied_bias_1x": sens * REAL_Z12, "implied_bias_3x": sens * 3 * REAL_Z12,
             "implied_total_1x": nod + sens * REAL_Z12, "implied_total_3x": nod + 3 * sens * REAL_Z12}
        r["flag_1x"] = abs(r["implied_bias_1x"]) > 0.003
        if ("unsigned", "drift1x") in W.columns:
            p1 = W.dropna(subset=[("unsigned", "nodrift"), ("unsigned", "drift1x")])
            shift = (p1[("mean_z12", "drift1x")] - p1[("mean_z12", "nodrift")]).mean()
            r.update({"n_pairs_1x": len(p1), "mean_z12_shift_1x": shift,
                      "predicted_1x": p1[("unsigned", "nodrift")].mean() + sens * shift,
                      "measured_1x": p1[("unsigned", "drift1x")].mean(),
                      "measured_shift_1x": (p1[("unsigned", "drift1x")] - p1[("unsigned", "nodrift")]).mean(),
                      "predicted_shift_1x": sens * shift})
        rows.append(r)
    S = pd.DataFrame(rows)
    S["abs_sens"] = S["sensitivity"].abs()
    S = S.sort_values(["h", "abs_sens"], ascending=[True, False]).drop(columns="abs_sens")
    S.to_csv(T3 / "drift_sensitivity.csv", index=False)
    out = {"real_mean_z12": REAL_Z12, "flagged_1x": S[S["flag_1x"]][["feature", "h"]].to_dict("records"),
           "max_abs_implied_bias_1x": float(S["implied_bias_1x"].abs().max()),
           "max_abs_implied_bias_3x": float(S["implied_bias_3x"].abs().max())}
    if "measured_1x" in S:
        out["corr_predicted_vs_measured_shift_1x"] = {
            h: float(g["predicted_shift_1x"].corr(g["measured_shift_1x"])) for h, g in S.groupby("h")}
    (T3 / "drift_sensitivity_summary.json").write_text(json.dumps(out, indent=1))
    pd.set_option("display.width", 250)
    for h, g in S.groupby("h"):
        print(h)
        print(g.head(15).round(5).to_string(index=False))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
