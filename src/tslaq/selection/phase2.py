"""Phase 2 selection helpers shared by the walk-forward run (CLAUDE.md 5.4, 6.8, 6.9; D1, D10).

Everything here operates on one fold: `tr` marks that fold's training rows; edges, clusters, MDA, BH and
the selected list are fit on those rows only.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tslaq.backtest.metrics import fold_ic_stats  # noqa: E402
from tslaq.features.build import CATEGORICAL  # noqa: E402
from tslaq.selection.condprob import (  # noqa: E402
    EXTREME,
    auc_mi,
    benjamini_hochberg,
    bootstrap_lift,
    bucket_table,
    by_year_table,
    fit_edges,
    pair_grid,
    quintile_codes,
    session_weights,
    shape_label,
    wilson,
    year_sign_share,
)
from tslaq.selection.ic import ic_stats, slot_ic  # noqa: E402
from tslaq.selection.importance import MDA_PARAMS  # noqa: E402
from tslaq.utils.io import ROOT  # noqa: E402

H = (1, 3, 12)
DECAY_H = tuple(range(1, 25))
REPORT_CUTOFF = "2025-12-01"  # last fold: training 2016-01 .. 2025-11
RZ_WINDOW = 1560  # 20 sessions
SETTINGS = {
    "stable_ic": {"min_abs_t": 2.0, "min_year_sign_share": 0.7, "last24_same_sign": True,
                  "delayed_min_abs_t": 2.0},
    "extreme_rule": {"bh_fdr": 0.10, "min_year_sign_share": 0.7, "mean_fwd_same_sign": True,
                     "min_year_n_eff": 30},
    "cluster_threshold": 0.7, "max_features": 40, "min_features": 20, "max_per_cluster": 2,
    "rz_window_bars": RZ_WINDOW, "mda": {"k": 5, "row_step": 2, "params": MDA_PARAMS},
    "condprob_cost_round_trip": None,
}
TAB = ROOT / "reports" / "tables"
FIG = ROOT / "reports" / "figures"


def ic_series(IC, f, version, key):
    s = IC[(IC["feature"] == f) & (IC["version"] == version) & (IC["h"] == str(key))]
    return s.set_index("month")["ic"].sort_index()


def feature_ic_table(IC, versions: dict, months_lt: str) -> pd.DataFrame:
    rows = []
    for f, ver in versions.items():
        for h in H:
            s = ic_series(IC, f, ver, h)
            s = s[s.index < months_lt[:7]]
            d = ic_series(IC, f, ver, f"d{h}")
            d = d[d.index < months_lt[:7]]
            st = ic_stats(s)
            sd_ = fold_ic_stats(d)
            l24 = fold_ic_stats(s.dropna().iloc[-24:])
            rows.append({"feature": f, "version": ver, "h": h, "ic": st["ic"], "icir": st["icir"], "t": st["t"],
                         "pos_share": st["pos_share"], "binom_p": st["binom_p"], "n_months": st["n_folds"],
                         "year_sign_share": st["year_sign_share"], "d_ic": sd_["ic"], "d_t": sd_["t"],
                         "last24_ic": l24["ic"], "last24_t": l24["t"]})
    T = pd.DataFrame(rows)
    s = SETTINGS["stable_ic"]
    T["stable"] = ((T["t"].abs() >= s["min_abs_t"]) & (T["year_sign_share"] >= s["min_year_sign_share"])
                   & (np.sign(T["last24_ic"]) == np.sign(T["ic"]))
                   & (np.sign(T["d_ic"]) == np.sign(T["ic"])) & (T["d_t"].abs() >= s["delayed_min_abs_t"]))
    return T


def condprob_cells(X: pd.DataFrame, L, tr: np.ndarray, cost: float, horizons=H, prefix: str = "",
                   full: bool = True):
    """All single-feature bucket cells on training rows. Returns (cells, by_year, edges, auc).

    `prefix` selects the outcome columns: "" -> fwd_h / z_h; "d" -> fwdd_h / zd_h (one-bar-delayed
    outcome, from t+1 to t+1+h). With full=False the by-year tables and AUC/MI are skipped (counting only).
    BH is applied across all cells returned by this call."""
    year = L["session_date"].dt.year.to_numpy()
    cells, years, edges, aucs = [], [], {}, []
    for f in X.columns:
        x = X[f].to_numpy()
        e = fit_edges(x[tr], f in CATEGORICAL)
        edges[f] = e
        for h in horizons:
            fwd = np.where(tr, L[f"fwd{prefix}_{h}"].to_numpy(), np.nan)
            z = L[f"z{prefix}_{h}"].to_numpy()
            t = bucket_table(x, fwd, z, e, h, cost)
            if t.empty:
                continue
            t.insert(0, "feature", f)
            t.insert(1, "h", h)
            if full:
                by = by_year_table(x, fwd, year, e, h)
                by.insert(0, "feature", f)
                by.insert(1, "h", h)
                share = year_sign_share(t, by, SETTINGS["extreme_rule"]["min_year_n_eff"], h)
                t["year_sign_share"] = t["bucket"].map(share)
                t["n_years"] = t["bucket"].map(by[by["n"] / h >= SETTINGS["extreme_rule"]["min_year_n_eff"]]
                                               .groupby("bucket").size())
                years.append(by)
                a, mi = auc_mi(x, fwd, e)
                aucs.append({"feature": f, "h": h, "auc": a, "mi_bits": mi})
            cells.append(t)
    C = pd.concat(cells, ignore_index=True)
    C["bh_sig"] = benjamini_hochberg(C["pval"].to_numpy(), SETTINGS["extreme_rule"]["bh_fdr"])
    BY = pd.concat(years, ignore_index=True) if years else pd.DataFrame()
    return C, BY, edges, pd.DataFrame(aucs)


def bootstrap_cells(cells: pd.DataFrame, X: pd.DataFrame, L, tr: np.ndarray, edges: dict, W: np.ndarray,
                    codes: np.ndarray) -> pd.DataFrame:
    """Day-block bootstrap (whole training sessions) of the lift of the given cells."""
    out = []
    for (f, h), g in cells.groupby(["feature", "h"]):
        fwd = np.where(tr, L[f"fwd_{h}"].to_numpy(), np.nan)
        b = bootstrap_lift(X[f].to_numpy(), fwd, codes, edges[f], W, buckets=list(g["bucket"]))
        if len(b):
            out.append(b.assign(feature=f, h=h))
    if not out:
        return pd.DataFrame(columns=["feature", "h", "bucket", "boot_lo", "boot_hi", "boot_sd"])
    B = pd.concat(out, ignore_index=True)
    B = cells.merge(B, on=["feature", "h", "bucket"], how="left")
    B["boot_holds"] = (np.sign(B["boot_lo"]) == np.sign(B["boot_hi"])) & (np.sign(B["boot_lo"]) == np.sign(B["lift"]))
    return B


def extreme_keep(C: pd.DataFrame) -> pd.DataFrame:
    r = SETTINGS["extreme_rule"]
    ext = C["bucket"].isin(EXTREME) | C["bucket"].str.startswith("v=")
    ok = (ext & C["bh_sig"] & (C["year_sign_share"] >= r["min_year_sign_share"])
          & (np.sign(C["mean_fwd"]) == np.sign(C["lift"])))
    return C[ok]


def shapes(C: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (f, h), t in C.groupby(["feature", "h"]):
        lab, rho, spread = shape_label(t, t.set_index("bucket")["bh_sig"], f in CATEGORICAL)
        rows.append({"feature": f, "h": h, "shape": lab, "rho_bucket_p": rho, "spread_D10_D1": spread,
                     "n_sig_cells": int(t["bh_sig"].sum())})
    return pd.DataFrame(rows)


def select(T: pd.DataFrame, K: pd.DataFrame, clusters: pd.Series, mda: dict) -> pd.DataFrame:
    """Most stable 1-2 features per useful cluster by IC, plus features passing the extreme-bucket rule."""
    feats = clusters.index
    st = T[T["stable"]]
    ic_ev = {f: st[st["feature"] == f] for f in feats}
    ext_ev = {f: K[K["feature"] == f] for f in feats}
    mda_t = pd.concat([m["mda_t"].rename(h) for h, m in mda.items()], axis=1).max(axis=1)

    def ic_score(f):
        e = ic_ev[f]
        return (e["h"] == 12).any(), (e["t"].abs().max() if len(e) else 0.0)

    def ext_score(f):
        e = ext_ev[f]
        if e.empty:
            return False, 0.0
        zz = (e["lift"].abs() / np.sqrt(e["base"] * (1 - e["base"]) / e["n_eff"]))
        return (e["h"] == 12).any(), float(zz.max())

    picked = []
    for c in sorted(clusters.unique()):
        mem = [f for f in clusters.index[clusters == c]]
        stab = sorted([f for f in mem if len(ic_ev[f])], key=lambda f: ic_score(f), reverse=True)
        for f in stab[:SETTINGS["max_per_cluster"]]:
            picked.append((f, "IC"))
        n_in = len(stab[:SETTINGS["max_per_cluster"]])
        extra = sorted([f for f in mem if not len(ic_ev[f]) and len(ext_ev[f])], key=ext_score, reverse=True)
        for f in extra[: max(SETTINGS["max_per_cluster"] - n_in, 0) or 1]:
            picked.append((f, "extreme"))
    rows = []
    for f, why in picked:
        e, k = ic_ev[f], ext_ev[f]
        ic_txt = "; ".join(f"h{int(r.h)} t={r.t:+.1f} (delay t={r.d_t:+.1f})" for r in e.itertuples())
        ex_txt = "; ".join(f"h{int(r.h)} {r.bucket} lift={100 * r.lift:+.2f}pp" for r in k.itertuples())
        reason = " + ".join(x for x in [("stable IC: " + ic_txt) if ic_txt else "",
                                         ("extreme lift: " + ex_txt) if ex_txt else ""] if x)
        h12 = ic_score(f)[0] or ext_score(f)[0]
        rows.append({"feature": f, "cluster": int(clusters[f]), "basis": "both" if ic_txt and ex_txt else why,
                     "h12_evidence": h12, "ic_score": ic_score(f)[1], "ext_score": ext_score(f)[1],
                     "cluster_mda_t": float(mda_t.get(clusters[f], np.nan)), "reason": reason})
    S = pd.DataFrame(rows)
    S["priority"] = S[["ic_score", "ext_score"]].max(axis=1)
    S = S.sort_values(["h12_evidence", "priority"], ascending=False).reset_index(drop=True)
    S["kept"] = S.index < SETTINGS["max_features"]
    return S


def condprob_figure(t: pd.DataFrame, f: str, h: int, path):
    order = [b for b in ["B2", "B5", *[f"D{i}" for i in range(1, 11)], "T5", "T2"] if b in set(t["bucket"])] \
        or list(t["bucket"])
    t = t.set_index("bucket").loc[order]
    fig, ax = plt.subplots(figsize=(6.4, 3.0), dpi=80)
    col = ["#c0504d" if s else "#7f9fbf" for s in t["bh_sig"]]
    ax.bar(range(len(t)), 100 * t["p_up"], color=col)
    ax.errorbar(range(len(t)), 100 * t["p_up"], yerr=[100 * (t["p_up"] - t["wilson_lo"]), 100 * (t["wilson_hi"] - t["p_up"])],
                fmt="none", ecolor="k", lw=0.8, capsize=2)
    ax.axhline(100 * t["base"].iloc[0], color="k", ls="--", lw=0.8)
    ax.set_xticks(range(len(t)), order, fontsize=7)
    lo = 100 * min(t["wilson_lo"].min(), t["base"].iloc[0]) - 1
    hi = 100 * max(t["wilson_hi"].max(), t["base"].iloc[0]) + 1
    ax.set_ylim(lo, hi)
    ax.set_ylabel("P(up) %")
    ax.set_title(f"{f}, h={h} (train 2016-01..2025-11; red = BH-significant; Wilson 95% with n/h)", fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def pair_heatmap(G: pd.DataFrame, a: str, b: str, h: int, path):
    la, lb = list(dict.fromkeys(G["a"])), list(dict.fromkeys(G["b"]))
    M = G.pivot(index="a", columns="b", values="lift").reindex(index=la, columns=lb) * 100
    N = G.pivot(index="a", columns="b", values="n").reindex(index=la, columns=lb)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(lb), 1.0 + 0.6 * len(la)), dpi=80)
    v = np.nanmax(np.abs(M.to_numpy())) if np.isfinite(M.to_numpy()).any() else 1
    im = ax.imshow(M.to_numpy(), cmap="RdBu_r", vmin=-v, vmax=v)
    for i in range(len(la)):
        for j in range(len(lb)):
            val = M.iat[i, j]
            ax.text(j, i, ("insuff." if not np.isfinite(val) else f"{val:+.1f}") + f"\nn={int(N.iat[i, j])}",
                    ha="center", va="center", fontsize=6)
    ax.set_xticks(range(len(lb)), lb, fontsize=7)
    ax.set_yticks(range(len(la)), la, fontsize=7)
    ax.set_xlabel(b)
    ax.set_ylabel(a)
    ax.set_title(f"lift (pp) {a} x {b}, h={h}", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def stage_report(R: dict, F, L, IC, cost):
    c = REPORT_CUTOFF
    X, C, BY = R["X"], R["C"], R["BY"]
    sd = L["session_date"].to_numpy()
    tr = sd < np.datetime64(c)
    (TAB / "condprob").mkdir(parents=True, exist_ok=True)
    (FIG / "condprob").mkdir(parents=True, exist_ok=True)
    # per-feature tables and figures
    for (f, h), t in C.groupby(["feature", "h"]):
        t.to_csv(TAB / "condprob" / f"{f}_h{h}.csv", index=False)
        BY[(BY["feature"] == f) & (BY["h"] == h)].to_csv(TAB / "condprob" / f"{f}_h{h}_by_year.csv", index=False)
        condprob_figure(t, f, h, FIG / "condprob" / f"{f}_h{h}.png")
    # day-block bootstrap of extreme-bucket lifts
    codes, _ = pd.factorize(L["session_date"])  # chronological: training sessions are codes 0..S_tr-1
    W = session_weights(int(L.loc[tr, "session_date"].nunique()), 1000, 0)
    boots = []
    for f in X.columns:
        for h in H:
            fwd = np.where(tr, L[f"fwd_{h}"].to_numpy(), np.nan)
            b = bootstrap_lift(X[f].to_numpy(), fwd, codes, R["edges"][f], W)
            if len(b):
                b.insert(0, "feature", f)
                b.insert(1, "h", h)
                boots.append(b)
    B = pd.concat(boots, ignore_index=True)
    B = B.merge(C[["feature", "h", "bucket", "lift", "wilson_lo", "wilson_hi", "base", "bh_sig"]],
                on=["feature", "h", "bucket"])
    B["wilson_lift_lo"], B["wilson_lift_hi"] = B["wilson_lo"] - B["base"], B["wilson_hi"] - B["base"]
    B.to_csv(TAB / "phase2_bootstrap_extremes.csv", index=False)
    # pairwise grids
    zt = (L["r"] / L["sigma_hat"]).to_numpy()
    k = L["bar_index"].to_numpy().astype(float)
    pairs = {
        "rsi14__vreg_sigma_pct": ("rsi14", "vreg_sigma_pct", None),
        "rsi14__vreg_level_pct": ("rsi14", "vreg_level_pct", None),
        "vwap_dev__tod": ("vwap_dev", "TOD", None),
        "mom_12__hmm_state": ("mom_12", "hmm_state", None),
        "relvol__ret_group": ("relvol", "RET", None),
        "gap_z__tod_first_hour": ("gap_z", "TOD1", k < 12),
        "qqq_z__resid_qqq_z": ("qqq_z", "resid_qqq_z", None),
    }
    grids = []
    for name, (a, b, rows) in pairs.items():
        xa = F[a].to_numpy() if a not in X else X[a].to_numpy()
        ca, la = quintile_codes(xa[tr], xa)
        if b == "TOD":
            cb, lb = quintile_codes(k[tr], k)
        elif b == "TOD1":
            grp = np.select([k <= 1, k <= 3, k <= 5, k <= 8, k <= 11], [0, 1, 2, 3, 4], -1)
            cb, lb = grp, ["b0-1", "b2-3", "b4-5", "b6-8", "b9-11"]
        elif b == "RET":
            e = np.quantile(zt[tr & np.isfinite(zt)], [1 / 3, 2 / 3])
            cb = np.where(np.isfinite(zt), np.searchsorted(e, zt, side="right"), -1)
            lb = ["down", "flat", "up"]
        elif b == "hmm_state":
            cb, lb = quintile_codes(F[b].to_numpy()[tr], F[b].to_numpy(), categorical=True)
        else:
            xb = F[b].to_numpy() if b not in X else X[b].to_numpy()
            cb, lb = quintile_codes(xb[tr], xb)
        if rows is not None:
            ca = np.where(rows, ca, -1)
        for h in H:
            fwd = np.where(tr, L[f"fwd_{h}"].to_numpy(), np.nan)
            G = pair_grid(ca, la, cb, lb, fwd, h)
            G.insert(0, "pair", name)
            G.insert(1, "h", h)
            grids.append(G)
            pair_heatmap(G, a, {"TOD": "TOD (bar_index quintile)", "TOD1": "TOD first hour", "RET": "RET group"}.get(b, b),
                         h, FIG / "condprob" / f"pair_{name}_h{h}.png")
    PG = pd.concat(grids, ignore_index=True)
    PG["bh_sig_pairs"] = benjamini_hochberg(PG["pval"].to_numpy(), 0.10)
    PG.to_csv(TAB / "phase2_pair_grids.csv", index=False)
    # intraday momentum (Gao et al.): P(last half hour up | sign r_first30 / sign r_12th30), full sessions
    g = L[tr].assign(r30=F.loc[tr, "r_first30"], r12=F.loc[tr, "r_12th30"])
    day = g.groupby("session_date").agg(half=("is_half_day", "first"), r30=("r30", "last"), r12=("r12", "last"),
                                        c71=("close", lambda s: s.iloc[71] if len(s) == 78 else np.nan),
                                        c77=("close", lambda s: s.iloc[77] if len(s) == 78 else np.nan))
    day = day[~day["half"]].dropna(subset=["c71", "c77"])
    day["last30"] = np.log(day["c77"] / day["c71"])
    day["year"] = day.index.year
    imr = []
    for cond in ["r30", "r12"]:
        for yr, d in [("all", day), *[(str(y), x) for y, x in day.groupby("year")]]:
            d = d[(d["last30"] != 0) & np.isfinite(d[cond]) & (d[cond] != 0)]
            base = (d["last30"] > 0).mean()
            for sgn in (-1, 1):
                m = np.sign(d[cond]) == sgn
                n = int(m.sum())
                p = (d.loc[m, "last30"] > 0).mean()
                lo, hi = wilson(p, n)
                imr.append({"condition": f"sign({cond})", "year": yr, "sign": sgn, "n_days": n, "p_up_last30": p,
                            "base": base, "lift": p - base, "wilson_lo": lo, "wilson_hi": hi,
                            "mean_last30_bp": 1e4 * d.loc[m, "last30"].mean()})
    IMR = pd.DataFrame(imr)
    IMR.to_csv(TAB / "phase2_intraday_momentum.csv", index=False)
    # per-slot IC (train window), IC decay figure, IC by year
    slot = L["bar_index"].to_numpy()
    SL = []
    for h in H:
        s = slot_ic(X[tr], L[f"z_{h}"].to_numpy()[tr], slot[tr])
        SL.append(pd.DataFrame({"feature": s.columns, "h": h, "slot_ic_mean": s.mean().to_numpy(),
                                "slot_pos_share": (s > 0).mean().to_numpy()}))
        s.to_csv(TAB / f"phase2_slot_ic_h{h}.csv")
    SLT = pd.concat(SL, ignore_index=True)
    dec = []
    for f, v in R["versions"].items():
        if v != "raw":
            continue
        for h in DECAY_H:
            s = ic_series(IC, f, "raw", h)
            s = s[s.index < c[:7]]
            dec.append({"feature": f, "h": h, "ic": s.mean(), "t": fold_ic_stats(s)["t"]})
    DEC = pd.DataFrame(dec)
    DEC.to_csv(TAB / "phase2_ic_decay.csv", index=False)
    yrs = []
    for f, v in R["versions"].items():
        for h in H:
            s = ic_series(IC, f, v, h)
            s = s[s.index < c[:7]]
            for y, x in s.groupby(pd.to_datetime(s.index).year):
                yrs.append({"feature": f, "h": h, "year": y, "ic": x.mean()})
    pd.DataFrame(yrs).pivot_table(index=["feature", "h"], columns="year", values="ic").to_csv(TAB / "phase2_ic_by_year.csv")
    return B, PG, IMR, SLT, DEC


def decay_figure(DEC, feats, path):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=90)
    for f in feats:
        d = DEC[DEC["feature"] == f]
        if len(d):
            ax.plot(d["h"], d["ic"], lw=1, label=f)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("horizon h (bars)")
    ax.set_ylabel("mean monthly IC (train 2016-01..2025-11)")
    ax.legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def cont_features(F):
    return [c for c in F.columns if c not in CATEGORICAL]

