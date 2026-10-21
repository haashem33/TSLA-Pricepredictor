"""Experiment registry (CLAUDE.md 10.6). Every run appends one row."""
from __future__ import annotations

import csv
import subprocess
from datetime import UTC, datetime

from tslaq.utils.io import ROOT

REGISTRY = ROOT / "experiments" / "registry.csv"
COLUMNS = [
    "timestamp", "git_hash", "config_hash", "phase", "model", "n_params", "fold_scheme",
    "rank_ic_h1", "rank_ic_h3", "rank_ic_h12", "icir_h1", "icir_h3", "icir_h12",
    "hit_rate_h1", "hit_rate_h3", "hit_rate_h12", "crps_h1", "crps_h3", "crps_h12",
    "brier_h1", "brier_h3", "brier_h12",
    "net_sharpe_period", "net_sharpe_ann", "sortino", "max_drawdown", "turnover", "sharpe_2x_costs",
    "n_trials_so_far", "notes",
]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "nogit"


def n_trials() -> int:
    if not REGISTRY.exists():
        return 0
    with open(REGISTRY, encoding="utf-8") as f:
        return max(sum(1 for _ in f) - 1, 0)


def log_run(row: dict) -> None:
    unknown = set(row) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown registry columns: {unknown}")
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    new = not REGISTRY.exists()
    full = {c: "" for c in COLUMNS}
    full.update(row)
    full["timestamp"] = datetime.now(UTC).isoformat(timespec="seconds")
    full["git_hash"] = git_hash()
    full["n_trials_so_far"] = n_trials() + 1
    with open(REGISTRY, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(full)
