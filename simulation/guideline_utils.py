"""Standard guideline helpers shared across refine scenarios."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuidelineTitles:
    cat1: str = "Category 1: Universal Compliance"
    cat2: str = "Category 2: Intent Triggered Guidelines"
    cat3: str = "Category 3: Condition Triggered Guidelines"


def infer_titles(oracle: dict[str, object]) -> dict[str, str]:
    """Return canonical titles using oracle-provided keys when available."""
    titles = dict(GuidelineTitles().__dict__)
    for key in oracle.keys():
        low = str(key).lower()
        if low.startswith("category 1"):
            titles["cat1"] = key
        elif low.startswith("category 2"):
            titles["cat2"] = key
        elif low.startswith("category 3"):
            titles["cat3"] = key
    return titles
