"""Tiny YAML config loader shared by every CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cli_config(description: str, default: str) -> dict[str, Any]:
    """Parse `--config path.yaml` and return the loaded dict."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default, help=f"YAML config (default: {default})")
    args = parser.parse_args()
    return load_config(args.config)
