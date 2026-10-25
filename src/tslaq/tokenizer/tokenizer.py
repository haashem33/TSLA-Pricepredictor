"""Composite per-bar tokenizer (CLAUDE.md 7.1-7.3). Every fitted part is fit on one fold's training rows only
and then frozen: RET input bins (64) and output bins per horizon (32) with training bin means for
de-tokenization, SHAPE k-means, GAP / VOLU / VREG / MKT quantile bins. REL, TOD, DOW, EVENT are fixed rules; the
HMM field is the filtered most-likely state of an HMM fit on the same training window (passed in as an input)."""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tslaq.tokenizer.bins import QuantileBins
from tslaq.tokenizer.fields import FIELDS, N_GAP_BINS, NO_GAP, SHAPE_COLS, special
from tslaq.tokenizer.shapes import ShapeKMeans

RET_CLIP = 6.0
B_OUT = 32
LAGS = 24


@dataclass
class Tokenizer:
    horizons: tuple = (1, 3, 12)
    min_count: int = 50
    seed: int = 0
    ret: QuantileBins | None = None
    out: dict = field(default_factory=dict)
    shape: ShapeKMeans | None = None
    gap: QuantileBins | None = None
    volu: QuantileBins | None = None
    vreg: QuantileBins | None = None
    mkt: QuantileBins | None = None
    meta: dict = field(default_factory=dict)

    def fit(self, X: pd.DataFrame, train: np.ndarray, Z: dict | None = None, cutoff: str = "") -> Tokenizer:
        """X = token_inputs(...); train = boolean mask of training rows; Z = {h: z_h} for the output bins."""
        tr = np.asarray(train, bool)
        self.ret = QuantileBins(FIELDS["RET"], self.min_count, RET_CLIP).fit(X["ret_z"].to_numpy()[tr])
        for h, z in (Z or {}).items():
            self.out[h] = QuantileBins(B_OUT, self.min_count, RET_CLIP).fit(np.asarray(z)[tr])
        self.shape = ShapeKMeans(FIELDS["SHAPE"], seed=self.seed).fit(X.loc[tr, list(SHAPE_COLS)].to_numpy())
        g = tr & X["is_gap_bar"].to_numpy()
        self.gap = QuantileBins(N_GAP_BINS, 1).fit(X["gap_z"].to_numpy()[g])
        self.volu = QuantileBins(FIELDS["VOLU"], self.min_count).fit(X["relvol"].to_numpy()[tr])
        self.vreg = QuantileBins(FIELDS["VREG"], self.min_count).fit(X["vreg"].to_numpy()[tr])
        self.mkt = QuantileBins(FIELDS["MKT"], self.min_count).fit(X["mkt"].to_numpy()[tr])
        self.meta = {"cutoff": cutoff, "n_train": int(tr.sum()), "n_gap_train": int(g.sum())}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """One integer code per field and bar (MISSING = the field's special code)."""
        def fix(code, f, n_values):  # a bins object's own 'missing' code -> the field's MISSING
            return np.where(code == n_values, special(f, "MISSING"), code)

        gap = fix(self.gap.transform(X["gap_z"].to_numpy()), "GAP", N_GAP_BINS)
        gap = np.where(X["is_gap_bar"].to_numpy(), gap, NO_GAP)
        tok = {
            "RET": self.ret.transform(X["ret_z"].to_numpy()),
            "SHAPE": self.shape.transform(X[list(SHAPE_COLS)].to_numpy()),
            "REL": X["rel"].to_numpy(),
            "GAP": gap,
            "VOLU": self.volu.transform(X["relvol"].to_numpy()),
            "VREG": self.vreg.transform(X["vreg"].to_numpy()),
            "HMM": X["hmm"].to_numpy(),
            "MKT": self.mkt.transform(X["mkt"].to_numpy()),
            "TOD": X["tod"].to_numpy(),
            "DOW": X["dow"].to_numpy(),
            "EVENT": X["event"].to_numpy(),
        }
        out = {}
        for f, v in tok.items():
            v = np.asarray(v, float)
            out[f] = np.where(np.isfinite(v), v, special(f, "MISSING")).astype(np.int16)
        return pd.DataFrame(out, index=X.index)

    def detok_ret(self, code) -> np.ndarray:
        return self.ret.detokenize(code)

    def save(self, path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path) -> Tokenizer:
        with open(path, "rb") as fh:
            return pickle.load(fh)


def lag_matrix(T: pd.DataFrame, lags: int = LAGS) -> tuple[np.ndarray, list[str]]:
    """Tokens of bars t-lags+1 .. t for every row t (field-major columns FIELD_l0 .. FIELD_l23). Rows before the
    start of the series get MISSING. Causal: row t uses rows <= t only."""
    n = len(T)
    cols, names = [], []
    for f in T.columns:
        x = T[f].to_numpy(np.int16)
        miss = special(f, "MISSING")
        for j in range(lags):
            c = np.full(n, miss, np.int16)
            c[j:] = x[: n - j]
            cols.append(c)
            names.append(f"{f}_l{j}")
    return np.column_stack(cols), names
