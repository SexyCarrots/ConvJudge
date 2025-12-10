#!/usr/bin/env python3
"""Refined Celestar Air simulator built on the shared refine pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

if __package__:
    from .refine import (
        StyleConfig,
        ScenarioHooks,
        build_standard_agent_prompt,
        load_config,
        normalize_to_titles,
        run_refine_pipeline,
    )
else:  # pragma: no cover - direct script execution (`python simulation/...py`)
    import sys

    ROOT = Path(__file__).resolve().parent.parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from simulation.refine import (  # type: ignore[no-redef]
        StyleConfig,
        ScenarioHooks,
        build_standard_agent_prompt,
        load_config,
        normalize_to_titles,
        run_refine_pipeline,
    )

CONFIG_FILE = Path(__file__).with_name("config").joinpath("airline_conversation_config.yaml")

CATEGORY_1 = "Category 1: Universal Compliance"
CATEGORY_2 = "Category 2: Intent Triggered Guidelines"
CATEGORY_3 = "Category 3: Condition Triggered Guidelines"
AIRLINE_TITLES = {"cat1": CATEGORY_1, "cat2": CATEGORY_2, "cat3": CATEGORY_3}


def persona_to_user_system_prompt(persona: dict[str, object]) -> str:
    intent = persona.get("intent", "general")
    tone = persona.get("tone", "neutral")
    lang = persona.get("language_proficiency", "native")
    prefers_human = persona.get("prefers_human_agent", "never")
    fname = persona.get("first_name", "")
    lname = persona.get("last_name", "")
    loyalty = persona.get("loyalty_tier", "")
    loyalty_num = persona.get("loyalty_number", "")
    phone_cc = persona.get("phone_country_code", "")
    phone_no = persona.get("phone_number_only", "")
    email = persona.get("email_address", "")
    trip_type = persona.get("trip_type", "")
    cabin = persona.get("cabin_class", "")
    legs = persona.get("legs", [])
    pnr = persona.get("existing_pnr", "")
    anc = persona.get("ancillaries", {})

    lines = [
        "You are a tester simulating a caller in an airline call center conversation.",
        "Speak naturally in first person as the caller; do NOT reveal you are testing.",
        f"Caller name: {fname} {lname}",
        f"Intent: {intent}",
        f"Tone: {tone}. Language proficiency (ESL_mild means English is not your first language, and you may make mistakes.): {lang}.",
        f"Preference for human agent: {prefers_human}.",
    ]
    if loyalty or loyalty_num:
        lines.append(f"Loyalty: {loyalty} ({loyalty_num})")
    if phone_cc or phone_no:
        lines.append(f"Phone: {phone_cc} {phone_no}")
    if email:
        lines.append(f"Email: {email}")
    if pnr:
        lines.append(f"Existing PNR: {pnr}")
    if trip_type or cabin or legs:
        lines.append(f"Trip type: {trip_type}; cabin: {cabin}")
        if isinstance(legs, list) and legs:
            for i, leg in enumerate(legs, 1):
                o = leg.get("origin", "")
                d = leg.get("destination", "")
                dd = leg.get("departure_date", "")
                tw = leg.get("preferred_time_window", "")
                lines.append(f"Leg {i}: {o}->{d} on {dd} ({tw})")
    if isinstance(anc, dict) and any(anc.values()):
        lines.append("Ancillary preferences: " + ", ".join(f"{k}={v}" for k, v in anc.items() if v))

    lines.extend(
        [
            "Behavior:",
            "- Stay consistent with the caller profile above.",
            "- If asked for information you don't know in your profile, say you don't know it.",
            "- Output ONLY the caller's next utterance.",
        ]
    )
    return "\n".join(lines)


def build_agent_system_prompt(guidelines: dict[str, object]) -> str:
    return build_standard_agent_prompt(
        "You are Nova, the Celestar Air virtual flight assistant.",
        guidelines,
        AIRLINE_TITLES,
    )


AIRLINE_STYLE = StyleConfig(
    discriminator_intro="You are a realism discriminator for Celestar Air callers.",
    human_cues="Human cues: mild fillers, informal softeners, brief pauses, natural rhythm.",
    synthetic_cues="Synthetic cues: overly formal phrasing, rigid cadence, robotic tone.",
    rewrite_instructions=(
        "Rewrite the caller utterance to sound more natural while keeping the intent and facts identical.\n"
        "Favor light fillers, informal tone, brief pauses, and gentle self-corrections.\n"
        'Return JSON only: {"rewrite": "<text>" }.'
    ),
)


ROOT = Path(__file__).resolve().parent.parent

AIRLINE_SCENARIO = ScenarioHooks(
    name="airline",
    category_titles=AIRLINE_TITLES,
    build_agent_prompt=build_agent_system_prompt,
    build_user_prompt=persona_to_user_system_prompt,
    normalize_category=lambda cat: normalize_to_titles(cat, AIRLINE_TITLES),
    style=AIRLINE_STYLE,
    oracle_path=ROOT / "guidelines" / "airlines" / "oracle.json",
    modified_path=ROOT / "guidelines" / "airlines" / "modified.json",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine Celestar Air conversations.")
    parser.add_argument("--config", default=str(CONFIG_FILE), help="Path to YAML config.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    run_refine_pipeline(cfg, AIRLINE_SCENARIO)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
