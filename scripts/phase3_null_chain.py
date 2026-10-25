"""D23 model-null budget chain (runs unattended). Budget = 12 h from the start of the timed first null draw of C.

K was fixed from that timed draw (seed 3000, 654 s: 20 x 654 s = 3.6 h <= 12 h -> K = 20). This script runs the
remaining C draws, then for B and then A: one timed first draw, and the other K - 1 draws only if K x (that time)
fits in the remaining budget; otherwise the variant stops and is reported as "unconfirmed". Log:
data/processed/phase3/null_chain.log.

Usage: python scripts/phase3_null_chain.py --budget-start "2026-10-25 11:28:30"
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "processed" / "phase3"
LOG = BASE / "null_chain.log"
K, BUDGET_S = 20, 12 * 3600


def log(msg: str):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def run(variant: str, a: int, b: int):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "run_phase3_gate3.py"), "--null", variant, "--draws",
                    str(a), str(b), "--workers", "5"], check=True, cwd=ROOT)


def draw_seconds(variant: str, seed: int) -> float:
    return float(pd.read_parquet(BASE / f"null_{variant}" / f"draw_{seed}.parquet")["seconds"].iloc[0])


def main(start: datetime):
    log(f"chain start; budget start {start}, K = {K}")
    run("C", 1, K - 1)
    log("C: all K draws done")
    for v in ("B", "A"):
        run(v, 0, 0)
        t = draw_seconds(v, 3000)
        left = BUDGET_S - (datetime.now() - start).total_seconds() + t  # the timed draw counts as draw 1
        fits = K * t <= left
        log(f"{v}: timed draw {t:.0f}s; K x t = {K * t / 3600:.2f} h; remaining budget incl. timed draw "
            f"{left / 3600:.2f} h -> {'run' if fits else 'UNCONFIRMED (does not fit)'}")
        if not fits:
            continue
        run(v, 1, K - 1)
        log(f"{v}: all K draws done")
    log(f"chain end; elapsed since budget start {(datetime.now() - start).total_seconds() / 3600:.2f} h")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-start", required=True)
    main(datetime.fromisoformat(ap.parse_args().budget_start))
