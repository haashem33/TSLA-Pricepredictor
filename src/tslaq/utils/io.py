"""Paths and config loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_config(name: str) -> dict:
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(*cfgs: dict) -> str:
    blob = json.dumps(cfgs, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def interim_path(symbol: str) -> Path:
    return ROOT / "data" / "interim" / f"{symbol}_5min.parquet"
