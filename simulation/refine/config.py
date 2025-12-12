"""Config loading helpers for refine scenarios."""

from __future__ import annotations

from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """Return parsed YAML config as a plain dict."""
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


__all__ = ["load_config"]
