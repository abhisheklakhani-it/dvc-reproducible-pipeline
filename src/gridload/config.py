"""Paths shared by the stages and the params.yaml loader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RAW = Path("data/raw")
PREPARED = Path("data/prepared/hourly.parquet")
FEATURES = Path("data/features/features.parquet")
MODEL = Path("models/model.txt")
REPORTS = Path("reports")

TZ = "Europe/Berlin"
TARGET = "load_mw"


def load_params(path: Path = Path("params.yaml")) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)
