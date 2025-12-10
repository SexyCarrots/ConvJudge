"""Prompt construction helpers shared across refine simulations."""

from __future__ import annotations

import json
from typing import Any

from .. import common as base
from ..guideline_utils import GuidelineTitles


def build_standard_agent_prompt(
    identity_line: str,
    guidelines: dict[str, Any],
    titles: dict[str, str] | None = None,
    extra_intro_lines: list[str] | None = None,
) -> str:
    """Return a shared agent prompt where only the identity sentence changes."""
    titles = titles or GuidelineTitles().__dict__
    payload = json.dumps(guidelines, ensure_ascii=False, indent=2)
    cat1 = titles["cat1"]
    cat2 = titles["cat2"]
    cat3 = titles["cat3"]
    lines = [identity_line]
    if extra_intro_lines:
        lines.extend(extra_intro_lines)
    lines.extend(
        [
            "Follow the guidelines exactly as written; do not mention internal policies or that the instructions were altered.",
            "Category 2 guidelines are strictly sequential: complete each phase exactly once (no repeats) and follow the order without skipping.",
            "For every reply: output ONLY the user-facing response text, then append a new line with "
            f"'{base.ANALYSIS_MARK}' and an analysis block containing:",
            "- Guideline: <exact clause you followed this turn> (still required even if the clause is flawed)",
            f"- Category: one of '{cat1}', '{cat2}', or '{cat3}'",
            "- Key: <for Category 1/3 use the key like 'greeting'; for Category 2 use the intent topic>",
            "- Phase: <Category 2 phase number (1-based) or -1 otherwise>",
            "- Terminate: true|false",
            "Always cite the single most relevant guideline/phase in the analysis.",
            "\nGROUND-TRUTH GUIDELINES PRESENTED TO YOU:\n" + payload,
            "Do not wrap the analysis in code fences and do not echo role labels.",
        ]
    )
    return "\n".join(lines)


__all__ = ["build_standard_agent_prompt"]