"""Scenario data structures and helpers for refine workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class StyleConfig:
    discriminator_intro: str
    human_cues: str
    synthetic_cues: str
    rewrite_instructions: str


@dataclass(frozen=True)
class ScenarioHooks:
    name: str
    category_titles: dict[str, str]
    build_agent_prompt: Callable[[dict[str, Any]], str]
    build_user_prompt: Callable[[dict[str, Any]], str]
    normalize_category: Callable[[str], str]
    style: StyleConfig
    oracle_path: Path
    modified_path: Path
    intent_field: str = "intent"


def normalize_to_titles(cat: str, titles: dict[str, str]) -> str:
    """Return canonical category title if it matches (case-insensitive)."""
    canon = (cat or "").strip()
    if not canon:
        return ""
    lower = canon.lower()
    for value in titles.values():
        if lower == value.lower():
            return value
    return canon


__all__ = ["ScenarioHooks", "StyleConfig", "normalize_to_titles"]
