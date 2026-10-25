"""Token conditional up-probability tables (CLAUDE.md 6.9 "Token tables", Phase 3), training rows only.

Cells: each value of RET, SHAPE, REL, GAP (incl. "no gap"), VOLU, VREG, HMM, TOD, and the two-token sequences
previous RET -> current RET, previous RET -> current SHAPE, and REL x VOLU (same bar). MISSING codes are left
out. Only cells with n >= 500 (non-zero outcomes) are tested: P(up) with exact zeros excluded, lift vs the base
rate, Wilson interval and a binomial test with n_eff = n / h, P(beyond cost). Benjamini-Hochberg at FDR 10% across
all tested cells of one fold and one outcome. A supplementary family uses RET in 8 groups of 8 consecutive bins
(RET8), because almost no 64 x 64 sequence reaches 500 bars; it is BH-corrected separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tslaq.selection.condprob import _binom_p, benjamini_hochberg, wilson
from tslaq.tokenizer.fields import FIELDS

SINGLE = ("RET", "SHAPE", "REL", "GAP", "VOLU", "VREG", "HMM", "TOD")
MIN_N = 500


def _tables(T: pd.DataFrame) -> dict[str, tuple[np.ndarray, int, str]]:
    """name -> (cell code per row or -1 if MISSING, n codes, family)."""
    v = {f: T[f].to_numpy().astype(np.int64) for f in T.columns}
    prev = {f: np.r_[FIELDS[f], v[f][:-1]] for f in ("RET",)}

    def ok(*pairs):
        m = np.ones(len(T), bool)
        for x, f in pairs:
            m &= x < FIELDS[f]
        return m

    out = {}
    for f in SINGLE:
        out[f] = (np.where(ok((v[f], f)), v[f], -1), FIELDS[f], "spec")
    seq = [("RET->RET", prev["RET"], "RET", v["RET"], "RET"), ("RET->SHAPE", prev["RET"], "RET", v["SHAPE"], "SHAPE"),
           ("REL x VOLU", v["REL"], "REL", v["VOLU"], "VOLU")]
    for name, a, fa, b, fb in seq:
        out[name] = (np.where(ok((a, fa), (b, fb)), a * FIELDS[fb] + b, -1), FIELDS[fa] * FIELDS[fb], "spec")
    r8, p8 = v["RET"] // 8, prev["RET"] // 8
    out["RET8->RET8"] = (np.where(ok((prev["RET"], "RET"), (v["RET"], "RET")), p8 * 8 + r8, -1), 64, "grouped")
    out["RET8->SHAPE"] = (np.where(ok((prev["RET"], "RET"), (v["SHAPE"], "SHAPE")), p8 * 32 + v["SHAPE"], -1), 256,
                          "grouped")
    return out


def _decode(name: str, c: int) -> str:
    if name in SINGLE:
        return str(c)
    if name == "RET8->RET8":
        return f"{c // 8}->{c % 8}"
    if name == "RET8->SHAPE":
        return f"{c // 32}->{c % 32}"
    nb = {"RET->RET": FIELDS["RET"], "RET->SHAPE": FIELDS["SHAPE"], "REL x VOLU": FIELDS["VOLU"]}[name]
    return f"{c // nb}->{c % nb}" if "->" in name else f"{c // nb}x{c % nb}"


def token_cells(T: pd.DataFrame, fwd: np.ndarray, z: np.ndarray, rows: np.ndarray, h: int, cost: float,
                year: np.ndarray | None = None) -> pd.DataFrame:
    """All cells for one outcome over the boolean mask `rows` (training rows). If `year` is given, adds the
    share of calendar years (n/h >= 30) whose lift has the full-period sign."""
    ok = rows & np.isfinite(fwd)
    nz = ok & (fwd != 0)
    up = nz & (fwd > 0)
    p0 = up.sum() / nz.sum()
    out = []
    for name, (code, ncode, fam) in _tables(T).items():
        c = np.where(code >= 0, code, ncode)  # MISSING -> overflow cell, dropped below
        n = np.bincount(c[nz], minlength=ncode + 1)[:ncode]
        k = np.bincount(c[up], minlength=ncode + 1)[:ncode]
        nall = np.bincount(c[ok], minlength=ncode + 1)[:ncode]
        sf = np.bincount(c[ok], weights=fwd[ok], minlength=ncode + 1)[:ncode]
        sz = np.bincount(c[ok], weights=np.nan_to_num(z[ok]), minlength=ncode + 1)[:ncode]
        cu = np.bincount(c[ok & (fwd > cost)], minlength=ncode + 1)[:ncode]
        cd = np.bincount(c[ok & (fwd < -cost)], minlength=ncode + 1)[:ncode]
        if year is not None:
            ys = np.unique(year[nz])
            yl = []
            for y in ys:
                sel = nz & (year == y)
                b = (up & sel).sum() / sel.sum()
                ny = np.bincount(c[sel], minlength=ncode + 1)[:ncode]
                ky = np.bincount(c[up & sel], minlength=ncode + 1)[:ncode]
                with np.errstate(invalid="ignore", divide="ignore"):
                    yl.append(np.where(ny / h >= 30, ky / ny - b, np.nan))
            YL = np.array(yl)
        for j in np.flatnonzero(n >= MIN_N):
            p = k[j] / n[j]
            ne = n[j] / h
            lo, hi = wilson(p, ne)
            row = {"table": name, "family": fam, "cell": _decode(name, int(j)), "n": int(n[j]),
                   "n_zero": int(nall[j] - n[j]), "n_eff": ne, "p_up": p, "base": p0, "lift": p - p0,
                   "wilson_lo": float(lo), "wilson_hi": float(hi), "pval": _binom_p(p * ne, ne, p0),
                   "p_up_cost": cu[j] / nall[j], "p_dn_cost": cd[j] / nall[j], "mean_fwd": sf[j] / nall[j],
                   "mean_z": sz[j] / nall[j]}
            if year is not None:
                yv = YL[:, j]
                yv = yv[np.isfinite(yv)]
                row["year_sign_share"] = float(np.mean(np.sign(yv) == np.sign(p - p0))) if len(yv) else np.nan
                row["n_years"] = len(yv)
            out.append(row)
        out.append({"table": name, "family": fam, "cell": "_insufficient", "n": int((n < MIN_N).sum()),
                    "n_cells_total": int(ncode)})
    D = pd.DataFrame(out)
    D["bh_sig"] = False
    for fam in ("spec", "grouped"):
        m = (D["family"] == fam) & (D["cell"] != "_insufficient")
        D.loc[m, "bh_sig"] = benjamini_hochberg(D.loc[m, "pval"].to_numpy(), 0.10)
    return D
