"""Paths and config loading."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]


def load_config(name: str) -> dict:
    with open(ROOT / "configs" / f"{name}.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(*cfgs: dict) -> str:
    blob = json.dumps(cfgs, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def data_version() -> str:
    """configs/data.yaml data_version; the environment variable TSLAQ_DATA_VERSION overrides it (D28 holdout run
    only, which reads the v2 + holdout bars from data/interim/v2h/)."""
    return os.environ.get("TSLAQ_DATA_VERSION") or load_config("data").get("data_version", "v1")


def interim_path(symbol: str, version: str | None = None) -> Path:
    """data/interim/<version>/<symbol>_5min.parquet (version defaults to configs/data.yaml)."""
    return ROOT / "data" / "interim" / (version or data_version()) / f"{symbol}_5min.parquet"
