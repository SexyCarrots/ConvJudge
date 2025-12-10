"""Guideline sampling and filtering utilities."""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Tuple


def _guideline_copy(obj: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(obj)


def filter_guidelines_for_intent(
    guidelines: dict[str, Any],
    *,
    titles: dict[str, str],
    allowed_cat2_intents: set[str] | None,
) -> dict[str, Any]:
    """Return a copy where Category 2 only contains the allowed intents."""
    if not allowed_cat2_intents:
        return guidelines
    filtered = _guideline_copy(guidelines)
    cat2 = titles["cat2"]
    section = filtered.get(cat2)
    if isinstance(section, dict):
        filtered[cat2] = {k: v for k, v in section.items() if k in allowed_cat2_intents}
    return filtered


def sample_guideline_overrides(
    oracle: dict[str, Any],
    modified: dict[str, Any],
    *,
    portion: float,
    rng: random.Random,
    titles: dict[str, str],
    allowed_cat2_intents: set[str] | None,
) -> Tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (mutated_guidelines, overrides_applied)."""
    p = max(0.0, min(1.0, float(portion)))
    mutated = _guideline_copy(oracle)
    overrides: list[dict[str, Any]] = []

    def add_override(entry: dict[str, Any]) -> None:
        entry = dict(entry)
        phase = entry.get("phase", -1)
        entry.setdefault("label", f"{entry['category']}::{entry['key']}::P{phase}")
        overrides.append(entry)

    cat1 = titles["cat1"]
    cat2 = titles["cat2"]
    cat3 = titles["cat3"]

    def _cat_key_positions(cat_name: str) -> dict[str, int]:
        section = oracle.get(cat_name, {}) or {}
        if isinstance(section, dict):
            return {k: idx for idx, k in enumerate(section.keys())}
        return {}

    cat_positions = {cat1: 0, cat2: 1, cat3: 2}
    cat_key_positions = {cat1: _cat_key_positions(cat1), cat3: _cat_key_positions(cat3)}
    cat2_section = oracle.get(cat2, {}) or {}
    if isinstance(cat2_section, dict):
        cat2_topic_positions = {key: idx for idx, key in enumerate(cat2_section.keys())}
    else:
        cat2_topic_positions = {}

    for cat in (cat1, cat3):
        orig_section = oracle.get(cat, {}) or {}
        mod_section = modified.get(cat, {}) or {}
        if not isinstance(orig_section, dict) or not isinstance(mod_section, dict):
            continue
        eligible: list[tuple[str, str, list[str]]] = []
        for key, orig_text in orig_section.items():
            mods = mod_section.get(key)
            if isinstance(orig_text, str) and isinstance(mods, list) and mods:
                eligible.append((key, orig_text, mods))
        if not eligible:
            continue
        rng.shuffle(eligible)
        take_n = math.floor(len(eligible) * p)
        for key, orig_text, mods in eligible[:take_n]:
            mod_choice = rng.choice(mods)
            mutated.setdefault(cat, {})[key] = mod_choice
            add_override(
                {
                    "category": cat,
                    "key": key,
                    "phase": -1,
                    "original": orig_text,
                    "modified": mod_choice,
                }
            )

    cat2_orig = oracle.get(cat2, {}) or {}
    cat2_mod = modified.get(cat2, {}) or {}
    if isinstance(cat2_orig, dict) and isinstance(cat2_mod, dict):
        for topic, phases in cat2_orig.items():
            if allowed_cat2_intents and topic not in allowed_cat2_intents:
                continue
            if not isinstance(phases, list):
                continue
            mod_phases = cat2_mod.get(topic)
            if not isinstance(mod_phases, list):
                continue
            eligible_phase: list[tuple[int, str, list[str]]] = []
            for idx, phase_text in enumerate(phases):
                mods = mod_phases[idx] if idx < len(mod_phases) else None
                if isinstance(phase_text, str) and isinstance(mods, list) and mods:
                    eligible_phase.append((idx, phase_text, mods))
            if not eligible_phase:
                continue
            rng.shuffle(eligible_phase)
            take_n = math.floor(len(eligible_phase) * p)
            for idx, phase_text, mods in eligible_phase[:take_n]:
                mod_choice = rng.choice(mods)
                mutated.setdefault(cat2, {}).setdefault(topic, list(phases))
                mutated[cat2][topic][idx] = mod_choice
                add_override(
                    {
                        "category": cat2,
                        "key": topic,
                        "phase": idx + 1,
                        "original": phase_text,
                        "modified": mod_choice,
                    }
                )

    def _sort_key(entry: dict[str, Any]) -> tuple:
        cat = entry.get("category", "")
        key = entry.get("key", "")
        phase = int(entry.get("phase", -1))
        cat_idx = cat_positions.get(cat, 99)
        if cat == cat2:
            topic_pos = cat2_topic_positions.get(key, 0)
            return (cat_idx, topic_pos, phase)
        key_pos = cat_key_positions.get(cat, {}).get(key, 0)
        return (cat_idx, key_pos, phase)

    overrides.sort(key=_sort_key)
    return mutated, overrides


def build_override_index(overrides: list[dict[str, Any]], titles: dict[str, str]) -> dict[str, dict[Any, dict[str, Any]]]:
    """Index overrides for quick lookups while labeling turns."""
    index: dict[str, dict[Any, dict[str, Any]]] = {}
    cat2 = titles["cat2"]
    for entry in overrides:
        cat = entry.get("category", "")
        key = entry.get("key")
        phase = entry.get("phase", -1)
        if not cat or key is None:
            continue
        store = index.setdefault(cat, {})
        if cat == cat2:
            store[(key, int(phase))] = entry
        else:
            store[key] = entry
    return index


def resolve_allowed_intents(
    persona: dict[str, Any],
    cat2_keys: set[str],
    intent_field: str,
) -> set[str]:
    """Return the subset of available intents that matches the persona."""
    raw = str(persona.get(intent_field, "")).strip()
    allowed = {raw} if raw else set()
    allowed &= cat2_keys
    if not allowed:
        allowed = set(cat2_keys)
    return allowed


__all__ = [
    "build_override_index",
    "filter_guidelines_for_intent",
    "resolve_allowed_intents",
    "sample_guideline_overrides",
]
