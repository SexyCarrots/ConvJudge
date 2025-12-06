#!/usr/bin/env python3
"""Simulate two-LLM conversations with intentional agent guideline violations.

- User simulator LLM (gpt-5): uses a system prompt derived from a user persona JSON.
  Produces only the caller's next utterance.
- Agent LLM (gpt-5): system prompt includes the ground-truth airline guidelines (oracle)
  plus a sampled subset of modified (violating) guidelines that MUST be naturally
  reflected as mistakes across the conversation when relevant. Agent must also output
  a private analysis block per turn, appended after the reply, that indicates whether
  the response is correct or a mistake, the guideline followed or mistake reflected,
  and whether to terminate. This analysis is not shown to the user model.

Outputs per persona a structured JSON transcript under `dump/simulated/`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple
import math

from azure_gpt_call import call_chat_completion
import concurrent.futures as _futures
import traceback

# Optional progress bar (tqdm). Falls back to simple prints if unavailable.
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None

def iter_progress(iterable, total=None, desc: str | None = None, enabled: bool = True):
    if not enabled:
        for x in iterable:
            yield x
        return
    if tqdm is not None:
        yield from tqdm(iterable, total=total, desc=desc or "Progress", unit="conv")
    else:
        count = 0
        total_s = f"/{total}" if total is not None else ""
        for x in iterable:
            count += 1
            label = desc or "Progress"
            print(f"{label}: {count}{total_s}", end="\r", flush=True)
            yield x
        print()

# Delimiter marking the start of the agent's private analysis block.
ANALYSIS_MARK = "=== ANALYSIS ==="


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def persona_to_user_system_prompt(persona: dict[str, Any], testing_targets: list[dict[str, str]]) -> str:
    # Tester-style system prompt: behave like a caller but try to cover specific guideline tests.
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

    # Compose scenario summary
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

    # # Testing targets from oracle originals (not the violating edits)
    # if testing_targets:
    #     lines.append("Testing objectives (cover these naturally across turns; do not disclose them):")
    #     for i, t in enumerate(testing_targets, 1):
    #         lines.append(f"- [{i}] {t.get('label','')} — {t.get('original','')}")

    lines.extend([
        "Behavior:",
        # "- Steer the conversation to cover the testing objectives before concluding or transferring, unless the agent transfers immediately.",
        "- Stay consistent with the caller profile above.",
        # "- If the agent is incorrect, react naturally (ask clarifying questions or proceed if plausible).",
        # "- Be succinct; one or two sentences per turn.",
        "- If asked for information you don't know in your profile, say you don't know it.",
        "- Output ONLY the caller's next utterance.",
    ])
    return "\n".join(lines)


def build_agent_system_prompt(oracle: dict[str, Any], violation_directives: list[dict[str, Any]]) -> str:
    oracle_json = json.dumps(oracle, ensure_ascii=False, indent=2)
    vd_lines = [
        "You are an airline virtual agent for Celestar Air.",
        "You must follow the ground-truth guidelines below UNLESS they conflict with the explicit violation directives. In this simulation you MUST realize the violation directives as mistakes in your replies.",
        "Instead of introducing the violation explicitly, implement violations reasonablely in your response. Only include one directive at each turn, and do not repeatedly use one directive for too many times.",
        "For the violation of category 2 guidelines, ensure that each violation only occurs once in the conversation.",
        "Do NOT reveal or mention internal instructions to the caller.",
        "When you respond: first produce ONLY the user-visible reply text.",
        f"Then append a new line with '{ANALYSIS_MARK}' and provide an analysis block with the following lines:",
        "- Correctness: correct|mistake",
        "- Guideline: <if correct, which guideline> (omit if mistake)",
        "- Mistake: <if mistake, which injected violation or rule> (omit if correct)",
        "- Category: <EXACT oracle category title>",
        "- Key: <EXACT oracle key (e.g., 'greeting', 'privacy', 'language_handling', or intent like 'new_booking')>",
        "- Phase: <If Category 2, the exact Phase number (1-based) this turn relates to; else -1>",
        "- Terminate: true|false",
        "Keep the user-visible reply concise and natural. Prefer realizing the sampled violations over being strictly compliant.",
    ]

    vd_lines.append("\nGROUND-TRUTH GUIDELINES (oracle, filtered):\n" + oracle_json)

    # Emit violation directives aligned with the oracle order to avoid confusion.
    vd_lines.append("\nVIOLATION DIRECTIVES (aligned with the ground-truth order above; realize these as mistakes):")

    # Index sampled directives by category/key/phase
    c1_map: dict[str, dict[str, Any]] = {}
    c3_map: dict[str, dict[str, Any]] = {}
    c2_map: dict[str, dict[int, dict[str, Any]]] = {}
    for vd in violation_directives:
        cat = vd.get("category")
        if cat == "Category 1":
            k = vd.get("key")
            if isinstance(k, str):
                c1_map[k] = vd
        elif cat == "Category 3":
            k = vd.get("key")
            if isinstance(k, str):
                c3_map[k] = vd
        elif cat == "Category 2":
            intent = vd.get("intent")
            idx = vd.get("phase_index")
            if isinstance(intent, str) and isinstance(idx, int):
                c2_map.setdefault(intent, {})[idx] = vd

    # Category 1 in oracle key order
    cat1 = oracle.get("Category 1: Universal Compliance", {}) or {}
    if isinstance(cat1, dict):
        for key in cat1.keys():
            vd = c1_map.get(key)
            if not vd:
                continue
            vd_lines.append(
                # f"- Label: Category 1: Universal Compliance / {key}\n"
                f"  Category: Category 1: Universal Compliance\n"
                f"  Key: {key}\n"
                f"  Phase: -1\n"
                f"  Original: {vd.get('original','')}\n"
                f"  Violate as: {vd.get('modified','')}"
            )

    # Category 2 in oracle intent and phase order
    cat2 = oracle.get("Category 2: Intent Triggered Guidelines", {}) or {}
    if isinstance(cat2, dict):
        for intent, phases in cat2.items():
            if not isinstance(phases, list):
                continue
            intent_map = c2_map.get(intent, {})
            for idx in range(len(phases)):
                vd = intent_map.get(idx)
                if not vd:
                    continue
                vd_lines.append(
                    # f"- Label: Category 2: Intent Triggered Guidelines / {intent} / Phase {idx+1}\n"
                    f"  Category: Category 2: Intent Triggered Guidelines\n"
                    f"  Key: {intent}\n"
                    f"  Phase: {idx+1}\n"
                    f"  Original: {vd.get('original','')}\n"
                    f"  Violate as: {vd.get('modified','')}"
                )

    # Category 3 in oracle key order
    cat3 = oracle.get("Category 3: Condition Triggered Guidelines", {}) or {}
    if isinstance(cat3, dict):
        for key in cat3.keys():
            vd = c3_map.get(key)
            if not vd:
                continue
            vd_lines.append(
                # f"- Label: Category 3: Condition Triggered Guidelines / {key}\n"
                f"  Category: Category 3: Condition Triggered Guidelines\n"
                f"  Key: {key}\n"
                f"  Phase: -1\n"
                f"  Original: {vd.get('original','')}\n"
                f"  Violate as: {vd.get('modified','')}"
            )

    vd_lines.extend(
        [
            "\nDo not place the analysis inside code fences. Do not echo the conversation or role labels.",
            "Example structure:",
            "Hello there, how can I help you today?",
            f"{ANALYSIS_MARK}",
            "Correctness: mistake",
            "Guideline: Category 1 / greeting",
            "Category: Category 1: Universal Compliance",
            "Key: greeting",
            "Phase: -1",
            "Terminate: false",
        ]
    )
    return "\n".join(vd_lines)


def to_public_transcript(messages: List[dict]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "assistant":
            out.append(("Agent", content))
        elif role == "user":
            out.append(("Caller", content))
    return out


def sample_violation_directives(
    oracle: dict[str, Any],
    modified: dict[str, Any],
    *,
    portion: float,
    rng: random.Random,
    allowed_intents: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Sample violation directives by portion per category.

    - Category 1 and 3: sample floor(portion * count) guideline keys each.
    - Category 2: for each intent, sample floor(portion * num_phases) phases.
    """
    p = random.uniform(0.0, min(1.0, float(portion)))
    directives: list[dict[str, Any]] = []

    # Category 1
    cat1_o = oracle.get("Category 1: Universal Compliance", {}) or {}
    cat1_m = modified.get("Category 1: Universal Compliance", {}) or {}
    pool_cat1: list[dict[str, Any]] = []
    for key, orig in cat1_o.items():
        mods = cat1_m.get(key)
        if isinstance(orig, str) and isinstance(mods, list) and mods:
            pool_cat1.append({
                "category": "Category 1",
                "key": key,
                "original": orig,
                "modified_list": mods,
                "label": f"Cat1/{key}",
            })
    rng.shuffle(pool_cat1)
    take_c1 = math.floor(len(pool_cat1) * p)
    for item in pool_cat1[:take_c1]:
        mod_choice = rng.choice(item["modified_list"]) if item["modified_list"] else None
        if not mod_choice:
            continue
        directives.append({**{k: v for k, v in item.items() if k != "modified_list"}, "modified": mod_choice})

    # Category 3
    cat3_o = oracle.get("Category 3: Condition Triggered Guidelines", {}) or {}
    cat3_m = modified.get("Category 3: Condition Triggered Guidelines", {}) or {}
    pool_cat3: list[dict[str, Any]] = []
    for key, orig in cat3_o.items():
        mods = cat3_m.get(key)
        if isinstance(orig, str) and isinstance(mods, list) and mods:
            pool_cat3.append({
                "category": "Category 3",
                "key": key,
                "original": orig,
                "modified_list": mods,
                "label": f"Cat3/{key}",
            })
    rng.shuffle(pool_cat3)
    take_c3 = math.floor(len(pool_cat3) * p)
    for item in pool_cat3[:take_c3]:
        mod_choice = rng.choice(item["modified_list"]) if item["modified_list"] else None
        if not mod_choice:
            continue
        directives.append({**{k: v for k, v in item.items() if k != "modified_list"}, "modified": mod_choice})

    # Category 2
    cat2_o = oracle.get("Category 2: Intent Triggered Guidelines", {}) or {}
    cat2_m = modified.get("Category 2: Intent Triggered Guidelines", {}) or {}
    for intent, phases_o in cat2_o.items():
        if allowed_intents is not None and intent not in allowed_intents:
            continue
        phases_m = cat2_m.get(intent)
        if not (isinstance(phases_o, list) and isinstance(phases_m, list)):
            continue
        phase_indices = [i for i, phase_o in enumerate(phases_o) if isinstance(phase_o, str) and isinstance(phases_m[i] if i < len(phases_m) else None, list) and (phases_m[i])]
        rng.shuffle(phase_indices)
        take_n = math.floor(len(phase_indices) * p / 2)  # halve to avoid too many Cat2 violations
        for idx in phase_indices[:take_n]:
            phase_o = phases_o[idx]
            mods_list = phases_m[idx]
            mod_choice = rng.choice(mods_list)
            directives.append({
                "category": "Category 2",
                "intent": intent,
                "phase_index": idx,
                "original": phase_o,
                "modified": mod_choice,
                "label": f"Cat2/{intent}/Phase {idx+1}",
            })

    #rng.shuffle(directives)
    return directives


def filter_oracle_for_agent(oracle: dict[str, Any], *, persona_intent: str) -> dict[str, Any]:
    # Keep Cat1 and Cat3 fully; Cat2 keep only the persona intent (if recognized) plus transfer.
    keep_intents = set()
    normalized = (persona_intent or "").strip().lower()
    if normalized in {"new_booking", "new booking", "book", "booking"}:
        keep_intents.add("new_booking")
    elif normalized in {"change_booking", "change", "exchange", "modify"}:
        keep_intents.add("change_booking")
    elif normalized in {"transfer", "agent", "escalation"}:
        # If the explicit intent is transfer, we still just keep transfer.
        pass
    else:
        # info_only or general → only transfer
        pass
    keep_intents.add("transfer")

    out = {k: v for k, v in oracle.items() if k != "Category 2: Intent Triggered Guidelines"}
    cat2 = oracle.get("Category 2: Intent Triggered Guidelines", {}) or {}
    filtered_cat2: dict[str, Any] = {}
    for intent, phases in cat2.items():
        if intent in keep_intents:
            filtered_cat2[intent] = phases
    out["Category 2: Intent Triggered Guidelines"] = filtered_cat2
    return out


def call_user_model(system_prompt: str, public_messages: List[dict]) -> str:
    # Provide the raw chat history (assistant/user turns) without any analysis.
    # Append an explicit instruction so the model outputs the caller's next utterance
    # even though Chat Completions always respond as the assistant role.
    messages = (
        [{"role": "system", "content": system_prompt}]
        + public_messages
        + [
            {
                "role": "user",
                "content": (
                    "Write ONLY the caller’s next utterance in plain text (no labels). "
                    "It is the caller’s turn now. Do not speak as the agent."
                ),
            }
        ]
    )
    return call_chat_completion("gpt-4o", messages)


def parse_agent_output(text: str) -> tuple[str, dict[str, Any]]:
    """Split agent output into visible reply and analysis block and parse fields."""
    raw = text.strip()
    # Split on the first occurrence of the analysis marker
    parts = raw.split(ANALYSIS_MARK, 1)
    reply = parts[0].strip()
    analysis_text = parts[1] if len(parts) > 1 else ""

    analysis: dict[str, Any] = {}
    if analysis_text:
        # Normalize whitespace and strip code fences if any
        at = analysis_text.strip()
        m = re.search(r"```(?:json|text)?\s*([\s\S]*?)\s*```", at, re.IGNORECASE)
        if m:
            at = m.group(1).strip()

        # Extract common fields
        def find(pattern: str, flags=re.IGNORECASE) -> str | None:
            mm = re.search(pattern, at, flags)
            return mm.group(1).strip() if mm else None

        correctness = find(r"Correctness\s*:\s*(correct|mistake)")
        guideline = find(r"Guideline\s*:\s*(.+)")
        mistake = find(r"Mistake\s*:\s*(.+)")
        term = find(r"Terminate\s*:\s*(true|false)")
        category = find(r"Category\s*:\s*(.+)")
        key = find(r"Key\s*:\s*(.+)")
        phase = find(r"Phase\s*:\s*(-?\d+)")

        if correctness:
            analysis["correctness"] = correctness.lower()
        if guideline:
            analysis["guideline"] = guideline
        if mistake:
            analysis["mistake"] = mistake
        if term:
            analysis["terminate"] = term.lower() == "true"
        if category:
            analysis["category"] = category
        if key:
            analysis["key"] = key
        if phase is not None:
            try:
                analysis["phase"] = int(phase)
            except Exception:
                pass

    return reply, analysis


def call_agent_model(system_prompt: str, public_messages: List[dict]) -> tuple[str, dict[str, Any]]:
    # Provide the raw chat history (assistant/user turns) without any analysis.
    messages = [{"role": "system", "content": system_prompt}] + public_messages
    text = call_chat_completion("gpt-5", messages)
    return parse_agent_output(text)


def simulate_one(persona_path: str, oracle: dict[str, Any], modified: dict[str, Any], *, max_turns: int, violation_portion: float, seed: int | None, end_with_agent: bool = True) -> dict[str, Any]:
    rng = random.Random(seed)
    persona = read_json(persona_path)
    persona_intent = persona.get("intent", "")
    oracle_filtered = filter_oracle_for_agent(oracle, persona_intent=persona_intent)
    # Allowed intents for Cat2 sampling
    allowed_cat2_intents = set(oracle_filtered.get("Category 2: Intent Triggered Guidelines", {}).keys())
    violation_directives = sample_violation_directives(
        oracle,
        modified,
        portion=violation_portion,
        rng=rng,
        allowed_intents=allowed_cat2_intents,
    )
    # Build user system as tester with oracle originals to test
    testing_targets = [{"label": v.get("label", ""), "original": v.get("original", "")} for v in violation_directives]
    user_sys = persona_to_user_system_prompt(persona, testing_targets)
    agent_sys = build_agent_system_prompt(oracle_filtered, violation_directives)
    public_messages: List[dict] = []  # chat messages list: roles in {user, assistant}
    agent_turns: list[dict[str, Any]] = []
    mistakes: list[dict[str, Any]] = []

    # Kick off: agent greets first per Category 1 guideline, but agent will handle in first turn.
    terminate = False
    for turn in range(max_turns):
        # Agent turn
        agent_reply, analysis = call_agent_model(agent_sys, public_messages)
        public_messages.append({"role": "assistant", "content": agent_reply})
        agent_turns.append({"reply": agent_reply, "analysis": analysis})

        # Violation labeling based on agent analysis (strict: require exact Category/Key)
        if isinstance(analysis, dict) and analysis.get("correctness", "").lower() == "mistake":
            cat = str(analysis.get("category", "")).strip()
            key = str(analysis.get("key", "")).strip()
            phase_num = analysis.get("phase", None)
            valid = False
            if cat and key:
                if cat in ("Category 1: Universal Compliance", "Category 3: Condition Triggered Guidelines"):
                    section = oracle.get(cat, {}) or {}
                    valid = isinstance(section, dict) and key in section
                elif cat == "Category 2: Intent Triggered Guidelines":
                    section = oracle.get(cat, {}) or {}
                    if isinstance(section, dict) and key in section:
                        # For Category 2, we also require a valid phase number (1-based)
                        phases = section.get(key, []) if isinstance(section.get(key, None), list) else []
                        if isinstance(phase_num, int) and 1 <= phase_num <= len(phases):
                            valid = True
                        else:
                            valid = False
                    else:
                        valid = False
            if valid:
                mistakes.append({
                    "turn_index": len(public_messages) - 1,
                    "guidance category": cat,
                    "guidance key": key,
                    "guideline_phase": (phase_num if cat == "Category 2: Intent Triggered Guidelines" else -1),
                    "evidence": agent_reply,
                })

        # Check termination from agent
        term_flag = analysis.get("terminate") if isinstance(analysis, dict) else None
        if isinstance(term_flag, bool) and term_flag:
            terminate = True
        if terminate:
            break

        # If this is the last allowed turn and we want the conversation to end on the agent side,
        # stop here (unless the agent already terminated above).
        if end_with_agent and turn == max_turns - 1:
            break

        # User turn
        user_reply = call_user_model(user_sys, public_messages)
        # Ensure clean text
        user_reply = user_reply.strip()
        # Some models echo labels; strip a leading "Caller:" label if present
        user_reply = re.sub(r"^\s*Caller\s*:\s*", "", user_reply, flags=re.IGNORECASE)
        public_messages.append({"role": "user", "content": user_reply})

    # Package result: only persona_path, message_list (no system), mistakes
    message_list = []
    for idx, m in enumerate(public_messages):
        message_list.append({
            "turn_index": idx,
            "role": m.get("role"),
            "content": m.get("content", ""),
        })

    convo = {
        "persona_path": persona_path,
        "message_list": message_list,
        "mistakes": mistakes,
    }
    return convo


def iter_persona_files(folder: str) -> list[str]:
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".json"):
            paths.append(os.path.join(folder, name))
    return paths


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Simulate conversations with agent guideline violations")
    parser.add_argument("--personas", default="user_persona/airline", help="Folder containing persona JSONs or a single file path")
    parser.add_argument("--max-turns", type=int, default=10, help="Maximum agent-user turns (agent speaks first each turn)")
    parser.add_argument(
        "--violation-portion",
        type=float,
        default=0.8,
        help=(
            "Portion (0-1) controlling how many guidelines to violate: "
            "Cat1 & Cat3: floor(portion * count) keys; "
            "Cat2: per intent, floor(portion * #phases) phases."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling violations")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bar output")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel workers (1 = no parallelism)")
    parser.add_argument("--output-dir", default=os.path.join("dump", "simulated_conv"), help="Directory to write transcripts")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of personas (0=all)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    oracle = read_json(os.path.join("guidelines", "airlines", "oracle.json"))
    modified = read_json(os.path.join("guidelines", "airlines", "modified.json"))

    # Gather persona files
    persona_arg = args.personas
    if os.path.isdir(persona_arg):
        persona_files = iter_persona_files(persona_arg)
    else:
        persona_files = [persona_arg]
    if args.limit and args.limit > 0:
        persona_files = persona_files[: args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    use_progress = not args.no_progress
    total = len(persona_files)
    breakpoint()
    def _run_one(pth: str) -> str:
        convo = simulate_one(
            pth,
            oracle,
            modified,
            max_turns=args.max_turns,
            violation_portion=args.violation_portion,
            seed=args.seed,
            end_with_agent=True,
        )
        out_name = os.path.splitext(os.path.basename(pth))[0] + ".json"
        out_path = os.path.join(args.output_dir, out_name)
        write_json(out_path, convo)
        return out_path

    workers = max(1, int(args.workers))
    if workers == 1:
        for p in iter_progress(persona_files, total=total, desc="Synthesizing", enabled=use_progress):
            out_path = _run_one(p)
            if not use_progress:
                print(f"Wrote: {out_path}")
    else:
        if tqdm is not None and use_progress:
            pbar = tqdm(total=total, desc="Synthesizing", unit="conv")
        else:
            pbar = None
        with _futures.ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_p = {ex.submit(_run_one, p): p for p in persona_files}
            for fut in _futures.as_completed(fut_to_p):
                p = fut_to_p[fut]
                try:
                    out_path = fut.result()
                    if not use_progress:
                        print(f"Wrote: {out_path}")
                except Exception:
                    print(f"Error generating {p}:", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                finally:
                    if pbar is not None:
                        pbar.update(1)
        if pbar is not None:
            pbar.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
