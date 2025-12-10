"""Shared helpers for evaluation scripts (full and pairwise)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Sequence, Set


@dataclass(frozen=True)
class VKey:
    turn_index: int
    category: str
    key: str
    phase: int

    @classmethod
    def from_pred(cls, item: dict[str, Any]) -> "VKey":
        turn_index = int(item.get("turn_index", -1))
        raw_cat = str(
            item.get("guidance_category")
            or item.get("category")
            or item.get("guidance category", "")
        ).strip()
        norm_cat = normalize_category(raw_cat)
        key = str(item.get("guidance_key") or item.get("key") or item.get("guidance key", "")).strip()
        phase_raw = item.get("guideline_phase", item.get("phase", -1))
        try:
            phase = int(phase_raw)
        except Exception:
            phase = -1
        return cls(turn_index, norm_cat, key, phase)

    @classmethod
    def from_truth(cls, item: dict[str, Any]) -> "VKey":
        turn_index = int(item.get("turn_index", -1))
        cat = normalize_category(str(item.get("guidance category", "")).strip())
        key = str(item.get("guidance key", "")).strip()
        phase = int(item.get("guideline_phase", -1))
        return cls(turn_index, cat, key, phase)


def normalize_category(cat: str) -> str:
    c = (cat or "").strip().lower()
    if not c:
        return ""
    if c.startswith(("category 1", "cat 1", "cat1")):
        return "Category 1: Universal Compliance"
    if c.startswith(("category 2", "cat 2", "cat2")):
        return "Category 2: Intent Triggered Guidelines"
    if c.startswith(("category 3", "cat 3", "cat3")):
        return "Category 3: Condition Triggered Guidelines"
    if "universal" in c or "compliance" in c:
        return "Category 1: Universal Compliance"
    if "intent" in c or "triggered" in c:
        return "Category 2: Intent Triggered Guidelines"
    if "condition" in c or "conditional" in c:
        return "Category 3: Condition Triggered Guidelines"
    return cat


def infer_category_titles(oracle: dict[str, Any]) -> Dict[str, str]:
    titles = {
        "cat1": "Category 1: Universal Compliance",
        "cat2": "Category 2: Intent Triggered Guidelines",
        "cat3": "Category 3: Condition Triggered Guidelines",
    }
    for key in oracle.keys():
        low = key.lower()
        if low.startswith("category 1"):
            titles["cat1"] = key
        elif low.startswith("category 2"):
            titles["cat2"] = key
        elif low.startswith("category 3"):
            titles["cat3"] = key
    return titles


def format_guidelines(oracle: dict[str, Any], titles: Dict[str, str]) -> str:
    cat1 = oracle.get(titles["cat1"], {}) or {}
    cat2 = oracle.get(titles["cat2"], {}) or {}
    cat3 = oracle.get(titles["cat3"], {}) or {}
    lines: list[str] = []
    lines.append(f"{titles['cat1'].upper()} (Keys must match exactly)")
    for k, v in cat1.items():
        lines.append(f"- Key: {k}\n  Text: {v}")
    lines.append("")
    lines.append(f"{titles['cat2'].upper()} (Keys are intents; include Phase number)")
    if isinstance(cat2, dict):
        for intent, phases in cat2.items():
            lines.append(f"- Intent Key: {intent}")
            if isinstance(phases, list):
                for i, p in enumerate(phases, 1):
                    lines.append(f"  Phase {i}: {p}")
    lines.append("")
    lines.append(f"{titles['cat3'].upper()} (Keys must match exactly)")
    for k, v in cat3.items():
        lines.append(f"- Key: {k}\n  Text: {v}")
    return "\n".join(lines)


def format_conversation(message_list: Sequence[dict[str, Any]]) -> str:
    out: list[str] = []
    for msg in message_list:
        idx = msg.get("turn_index")
        role = msg.get("role", "")
        content = msg.get("content", "")
        out.append(f"{idx} | {role.upper()}: {content}")
    return "\n".join(out)


def build_user_prompt(guidelines_text: str, conversation_text: str, conv_id: str) -> str:
    return (
        "TASK:\n"
        "Using the guidelines, identify every assistant (agent) turn that violates a guideline.\n"
        "Only mark assistant turns; never mark user turns.\n"
        "- Each assistant turn can contribute at most one violation entry; if multiple issues exist, pick the most significant one.\n"
        "- Use exact field names and values as they appear in the guidelines.\n"
        "- Category 1 (Universal Compliance): single-step requirements that always use guideline_phase = -1.\n"
        "- Category 2 (Intent Triggered): intents with ordered phases that must be followed in order (no skipping), though phases can repeat if needed; include the 1-based phase number when that intent is active.\n"
        "- Category 3 (Condition Triggered): conditional / handoff triggers that immediately fire and use guideline_phase = -1.\n"
        "- If the conversation contains no violations, return \"violations\": [] (an empty list).\n"
        "RESPONSE (strict JSON only):\n"
        "{\n"
        "  \"violations\": [\n"
        "    {\n"
        "      \"turn_index\": <int>,\n"
        "      \"guidance_category\": \"<string>\",\n"
        "      \"guidance_key\": \"<string>\",\n"
        "      \"guideline_phase\": <int>,\n"
        "      \"evidence\": \"<short quote from the assistant message>\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "GUIDELINES:\n" + guidelines_text + "\n\n"
        "CONVERSATION:\n" + conversation_text
    )


def extract_first_json(text: str) -> dict[str, Any]:
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 2:
            code = parts[1]
            if code.startswith("json"):
                code = code[len("json") :]
            t = code.strip()
    if not t.startswith("{"):
        s = t.find("{")
        e = t.rfind("}")
        if s != -1 and e != -1 and e > s:
            t = t[s : e + 1]
    return json.loads(t)


def compute_metrics(pred: Set[VKey], truth: Set[VKey]):
    tp_items = sorted(pred & truth, key=lambda x: x.turn_index)
    fp_items = sorted(pred - truth, key=lambda x: x.turn_index)
    fn_items = sorted(truth - pred, key=lambda x: x.turn_index)

    tp = len(tp_items)
    fp = len(fp_items)
    fn = len(fn_items)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, tp_items, fp_items, fn_items


def compute_turn_metrics(pred: Set[VKey], truth: Set[VKey]):
    pred_turns = {p.turn_index for p in pred}
    truth_turns = {t.turn_index for t in truth}
    tp_set = pred_turns & truth_turns
    fp_set = pred_turns - truth_turns
    fn_set = truth_turns - pred_turns
    tp = len(tp_set)
    fp = len(fp_set)
    fn = len(fn_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, sorted(tp_set), sorted(fp_set), sorted(fn_set)


__all__ = [
    "VKey",
    "build_user_prompt",
    "compute_metrics",
    "compute_turn_metrics",
    "extract_first_json",
    "format_conversation",
    "format_guidelines",
    "infer_category_titles",
    "normalize_category",
]
