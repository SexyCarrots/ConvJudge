#!/usr/bin/env python3
"""Refined SCAN Health callback simulator with inline style shaping.

Derived from the dental refine workflow but adjusted for the SCAN guidelines,
persona schema, and error injection logic (modified guidelines replace their
oracle counterparts). Each conversation run samples a subset of guidelines to
corrupt, rewrites the agent-facing oracle accordingly, tracks which clauses the
agent cites, and labels turns where a corrupted clause is used.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import traceback
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from tqdm import tqdm

import simulate_dental_conversation as base  # shared helpers (LLM plumbing, parsing, etc.)

ANALYSIS_MARK = base.ANALYSIS_MARK
CONFIG_FILE = os.path.join(os.path.dirname(__file__), "scan_conversation_config.yaml")


# ---------------------------------------------------------------------------
# Guideline helpers
# ---------------------------------------------------------------------------
CATEGORY_1 = "Category 1: Universal Compliance"
CATEGORY_2 = "Category 2: Intent Triggered Guidelines"
CATEGORY_3 = "Category 3: Condition Triggered Guidelines"


def normalize_category(cat: str) -> str:
    """Map loose mentions like 'cat2' to SCAN canonical titles."""
    c = (cat or "").strip().lower()
    if not c:
        return ""
    if c.startswith("category 1") or c.startswith("cat 1") or c.startswith("cat1"):
        return CATEGORY_1
    if c.startswith("category 2") or c.startswith("cat 2") or c.startswith("cat2"):
        return CATEGORY_2
    if c.startswith("category 3") or c.startswith("cat 3") or c.startswith("cat3"):
        return CATEGORY_3
    if "universal" in c or "compliance" in c:
        return CATEGORY_1
    if "intent" in c or "trigger" in c or "step" in c:
        return CATEGORY_2
    if "condition" in c or "handoff" in c or "loop" in c:
        return CATEGORY_3
    return cat


def _guideline_copy(obj: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(obj)


def sample_guideline_overrides(
    oracle: dict[str, Any],
    modified: dict[str, Any],
    *,
    portion: float,
    rng: random.Random,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return (mutated_guidelines, overrides_applied)."""
    p = max(0.0, min(1.0, float(portion)))
    mutated = _guideline_copy(oracle)
    overrides: list[dict[str, Any]] = []

    def add_override(entry: dict[str, Any]) -> None:
        entry = dict(entry)
        phase = entry.get("phase", -1)
        entry.setdefault("label", f"{entry['category']}::{entry['key']}::P{phase}")
        overrides.append(entry)

    # Category 1 + 3 straightforward replacement
    for cat in (CATEGORY_1, CATEGORY_3):
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

    # Category 2: match per phase
    cat2_orig = oracle.get(CATEGORY_2, {}) or {}
    cat2_mod = modified.get(CATEGORY_2, {}) or {}
    if isinstance(cat2_orig, dict) and isinstance(cat2_mod, dict):
        for topic, phases in cat2_orig.items():
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
                mutated.setdefault(CATEGORY_2, {}).setdefault(topic, list(phases))
                mutated[CATEGORY_2][topic][idx] = mod_choice
                add_override(
                    {
                        "category": CATEGORY_2,
                        "key": topic,
                        "phase": idx + 1,
                        "original": phase_text,
                        "modified": mod_choice,
                    }
                )

    return mutated, overrides


def build_override_index(overrides: list[dict[str, Any]]) -> dict[str, dict[Any, dict[str, Any]]]:
    """Index overrides for quick lookups while labeling turns."""
    index: dict[str, dict[Any, dict[str, Any]]] = {}
    for entry in overrides:
        cat = entry.get("category", "")
        key = entry.get("key")
        phase = entry.get("phase", -1)
        if not cat or key is None:
            continue
        store = index.setdefault(cat, {})
        if cat == CATEGORY_2:
            store[(key, int(phase))] = entry
        else:
            store[key] = entry
    return index


# ---------------------------------------------------------------------------
# Prompt builders (agent + user)
# ---------------------------------------------------------------------------
def build_agent_system_prompt(guidelines: dict[str, Any]) -> str:
    payload = json.dumps(guidelines, ensure_ascii=False, indent=2)
    lines = [
        "You are the SCAN Health virtual assistant focused on callback scheduling and voicemail routing.",
        "Follow the guidelines exactly as written; do not mention internal policies or that the instructions were altered.",
        "Category 2 guidelines are strictly sequential: complete each phase exactly once (no repeats) and follow the order without skipping.",
        "For every reply: output ONLY the user-facing response text, then append a new line with "
        f"'{ANALYSIS_MARK}' and an analysis block containing:",
        "- Guideline: <exact clause you followed this turn> (still required even if the clause is flawed)",
        f"- Category: one of '{CATEGORY_1}', '{CATEGORY_2}', or '{CATEGORY_3}'",
        "- Key: <for Category 1/3 use the key like 'greeting'; for Category 2 use the intent topic>",
        "- Phase: <Category 2 phase number (1-based) or -1 otherwise>",
        "- Terminate: true|false",
        "Always cite the single most relevant guideline/phase in the analysis.",
        "\nGROUND-TRUTH GUIDELINES PRESENTED TO YOU:\n" + payload,
        "Do not wrap the analysis in code fences and do not echo role labels.",
    ]
    return "\n".join(lines)


def persona_to_user_system_prompt(persona: dict[str, Any]) -> str:
    """Construct SCAN-specific user simulator instructions."""
    caller = persona.get("caller", {}) or {}
    member = persona.get("member", {}) or {}
    availability = persona.get("availability", {}) or {}
    reason = persona.get("call_reason", {}) or {}
    confirmation = persona.get("confirmation", {}) or {}
    voicemail = persona.get("voicemail", {}) or {}

    def fmt_window(win: dict[str, Any]) -> str:
        label = win.get("label", "")
        start = win.get("start_local", "")
        end = win.get("end_local", "")
        note = win.get("note", "")
        pref = "preferred" if win.get("is_preferred") else "backup"
        return f"{label} ({pref}): {start}-{end} local. Note: {note}".strip()

    windows = [
        fmt_window(win)
        for win in availability.get("windows", []) or []
        if isinstance(win, dict)
    ]

    lines: list[str] = [
        "You are a tester role-playing a caller interacting with the SCAN Health virtual assistant.",
        "Speak naturally in first person, never reveal you're testing.",
        f"Intent: {persona.get('intent','unknown')}. Tone: {persona.get('tone','neutral')}."
        f" English proficiency: {persona.get('language_proficiency','native')}.",
        f"Preference for human agent: {persona.get('prefers_human_agent','medium')}."
    ]

    caller_name = f"{caller.get('first_name','')} {caller.get('last_name','')}".strip()
    if caller_name:
        lines.append(f"Caller name: {caller_name} (pronouns: {caller.get('pronouns','')}).")
    lines.append(
        f"Primary callback number: {caller.get('phone_country_code','')} {caller.get('phone_number_only','')} "
        f"(last four {caller.get('phone_last_four','')})."
    )
    if caller.get("alternate_number"):
        lines.append(f"Alternate number available: {caller['alternate_number']}.")
    if caller.get("email_address"):
        lines.append(f"Email: {caller['email_address']}.")

    lines.append(f"Caller city/state: {caller.get('city','')}, {caller.get('state','')} ({caller.get('timezone_hint','')} reference).")
    if windows:
        lines.append("Preferred callback windows (local time): " + " | ".join(filter(None, windows)))
    if availability.get("needs_timezone_help"):
        lines.append("You may ask the assistant to confirm timezone abbreviations.")
    if availability.get("can_shift_by_day") is False:
        lines.append("You cannot move to a different day; insist on the sampled day/windows.")

    member_name = f"{member.get('first_name','')} {member.get('last_name','')}".strip()
    if member.get("is_self"):
        lines.append("You are calling for yourself (SCAN member).")
    else:
        lines.append(f"You are calling for {member_name}, relationship: {member.get('relationship','unknown')}.")
    if member.get("needs_name_confirmation"):
        lines.append("Expect to spell or confirm the member's name carefully.")
    if member.get("notes"):
        lines.append(f"Context: {member['notes']}")

    raw_reason = reason.get("raw_statement") or reason.get("concise_summary")
    if raw_reason:
        lines.append(f"Reason for callback (keep under 50 words): {raw_reason}")
    if reason.get("needs_sanitization"):
        lines.append("Sanitize any inappropriate language if you slip.")

    if confirmation.get("requires_last_four_only"):
        lines.append("Only confirm phone numbers using the last four digits.")
    if confirmation.get("needs_step_by_step"):
        lines.append("You appreciate step-by-step recap before confirming.")
    if confirmation.get("voicemail_fallback_on_loop"):
        lines.append("If booking stalls, request voicemail routing.")

    if voicemail.get("wants_voicemail"):
        lines.append("Caller ultimately wants to leave a voicemail instead of choosing a slot.")
        if voicemail.get("script_hint"):
            lines.append(f"Voicemail preference note: {voicemail['script_hint']}")

    for note in persona.get("notes", []):
        lines.append(f"Note: {note}")

    lines.append("Stay in character, reply with one utterance per user turn.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------
def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def _sample_index_from_path(path: str, default: int) -> int:
    """Extract numeric sample index from persona filename, fallback to default."""
    name = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)$", name)
    if match:
        try:
            return int(match.group(1))
        except Exception:
            pass
    return default


def simulate_one_refine(
    persona_path: str,
    oracle: dict[str, Any],
    modified: dict[str, Any],
    *,
    max_turns: int,
    violation_portion: float,
    seed: int | None,
    call_agent_model,
    call_user_model,
    inline_style_judge: bool,
    judge_model: str | None,
    call_judge_chat,
    ref_user_messages: List[str],
    inline_max_iters: int,
    end_with_agent: bool = True,
) -> dict[str, Any]:
    rng = random.Random(seed)
    persona = base.read_json(persona_path)

    mutated_guidelines, overrides = sample_guideline_overrides(
        oracle, modified, portion=violation_portion, rng=rng
    )
    override_index = build_override_index(overrides)

    user_sys = persona_to_user_system_prompt(persona)
    agent_sys = build_agent_system_prompt(mutated_guidelines)
    public_messages: List[dict] = []
    mistakes: list[dict[str, Any]] = []

    use_refine = inline_style_judge and ref_user_messages and judge_model and call_judge_chat

    def _discriminate(candidate: str) -> Tuple[bool, str]:
        if not use_refine:
            return False, ""
        pool = list(ref_user_messages) + [candidate]
        order = list(range(len(pool)))
        rng.shuffle(order)
        shuffled = [pool[i] for i in order]
        candidate_index = order.index(len(pool) - 1)
        lines = [
            "You are a realism discriminator for SCAN callback callers.",
            "Human cues: mild fillers, soft hedges, brief pauses, natural rhythm.",
            "Synthetic cues: overly formal, rigid, repetitive, robotic phrasing.",
            "Select the single least human-sounding line. If all are fine, return -1.",
            'Respond ONLY with JSON: {"least_index": <int or -1>, "reason": "<text>"}',
            "Messages:",
        ]
        for i, msg in enumerate(shuffled):
            lines.append(f"{i}: {msg}")
        prompt = "\n".join(lines)
        resp = call_judge_chat(
            judge_model,
            [
                {"role": "system", "content": "You output ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        txt = resp.strip()
        if txt.startswith("```"):
            parts = txt.split("```")
            if len(parts) >= 2:
                body = parts[1]
                if body.startswith("json"):
                    body = body[len("json") :]
                txt = body.strip()
        try:
            data = json.loads(txt)
            li = int(data.get("least_index", -1))
            reason = str(data.get("reason", "")).strip()
            if li == candidate_index:
                return True, reason or "Less natural wording"
            return False, reason
        except Exception:
            return False, "Parsing failure; stop refinement"

    def _rewrite(candidate: str, reason: str) -> str:
        if not use_refine:
            return candidate
        prompt = (
            "Rewrite the caller line to sound more organic but keep the intent and facts identical.\n"
            "Use mild fillers, informal tone, and brief pauses when natural.\n"
            'Return JSON only: {"rewrite": "<text>"}.\n\n'
            f"ORIGINAL: {candidate}\n"
            f"CRITIQUE: {reason}\n"
        )
        resp = call_judge_chat(
            judge_model,
            [
                {"role": "system", "content": "You output ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        txt = resp.strip()
        if txt.startswith("```"):
            parts = txt.split("```")
            if len(parts) >= 2:
                body = parts[1]
                if body.startswith("json"):
                    body = body[len("json") :]
                txt = body.strip()
        try:
            data = json.loads(txt)
            rw = data.get("rewrite", "")
            if isinstance(rw, str) and rw.strip():
                return rw.strip()
        except Exception:
            pass
        return candidate

    terminate = False
    for turn in range(max_turns):
        agent_reply_raw, analysis = call_agent_model(agent_sys, public_messages)
        if not isinstance(analysis, dict):
            analysis = {}
        public_messages.append({"role": "assistant", "content": agent_reply_raw})

        analyzed_cat = normalize_category(str(analysis.get("category", "")).strip())
        key = str(analysis.get("key", "")).strip()
        phase_raw = analysis.get("phase", -1)
        try:
            phase_num = int(phase_raw)
        except Exception:
            phase_num = -1

        override_hit: dict[str, Any] | None = None
        if analyzed_cat == CATEGORY_2:
            override_hit = override_index.get(CATEGORY_2, {}).get((key, phase_num))
        elif analyzed_cat in (CATEGORY_1, CATEGORY_3):
            override_hit = override_index.get(analyzed_cat, {}).get(key)
        if override_hit:
            mistakes.append(
                {
                    "turn_index": len(public_messages) - 1,
                    "guidance category": analyzed_cat,
                    "guidance key": key,
                    "guideline_phase": phase_num if analyzed_cat == CATEGORY_2 else -1,
                    "guideline": override_hit.get("modified", ""),
                    "evidence": agent_reply_raw,
                }
            )

        term_flag = analysis.get("terminate") if isinstance(analysis, dict) else False
        if isinstance(term_flag, bool) and term_flag:
            terminate = True
        if terminate:
            break
        if end_with_agent and turn == max_turns - 1:
            break

        user_reply_raw = call_user_model(user_sys, public_messages)
        candidate = re.sub(r"^\s*Caller\s*:\s*", "", user_reply_raw.strip(), flags=re.IGNORECASE)

        if use_refine:
            attempts = 0
            while attempts < inline_max_iters:
                needs, reason = _discriminate(candidate)
                if not needs:
                    break
                new_candidate = _rewrite(candidate, reason)
                if new_candidate.strip() == candidate.strip():
                    break
                candidate = new_candidate
                attempts += 1

        public_messages.append({"role": "user", "content": candidate})

    message_list = [
        {
            "turn_index": idx,
            "role": m.get("role"),
            "content": m.get("content", ""),
        }
        for idx, m in enumerate(public_messages)
    ]
    return {
        "persona_path": persona_path,
        "message_list": message_list,
        "mistakes": mistakes,
        "style_ref_messages_count": len(ref_user_messages),
        "violation_directives": overrides,
    }


# ---------------------------------------------------------------------------
# Driver utilities (config + execution)
# ---------------------------------------------------------------------------
def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_persona_files(folder: str) -> list[str]:
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".json"):
            paths.append(os.path.join(folder, name))
    return paths


def run_from_config(cfg: dict[str, Any]) -> None:
    personas_path = cfg.get("personas", "user_persona/scan")
    max_turns = int(cfg.get("max_turns", 10))
    violation_portion = float(cfg.get("violation_portion", 0.6))
    seed = int(cfg.get("seed", 42))
    provider = cfg.get("provider", "azure")
    agent_model = cfg.get("agent_model", "gpt-5")
    user_model = cfg.get("user_model", "gpt-4o")
    judge_model = cfg.get("judge_model", user_model)
    inline_style_judge = bool(cfg.get("inline_style_judge", True))
    real_ref = cfg.get("real_ref", "") or None
    inline_max_iters = int(cfg.get("inline_max_iters", 3))
    style_ref_user_sample_size = int(cfg.get("style_ref_user_sample_size", 8))
    max_concurrency = int(cfg.get("max_concurrency", 5))
    output_dir = cfg.get("output_dir", "dump/simulated_scan_conv")
    limit = int(cfg.get("limit", 0))
    no_progress = bool(cfg.get("no_progress", False))
    fail_fast_auth = bool(cfg.get("fail_fast_auth", True))

    oracle = base.read_json(os.path.join("guidelines", "SCAN", "oracle.json"))
    modified = base.read_json(os.path.join("guidelines", "SCAN", "modified.json"))

    if os.path.isdir(personas_path):
        persona_files = iter_persona_files(personas_path)
    else:
        persona_files = [personas_path]
    if limit > 0:
        persona_files = persona_files[:limit]

    os.makedirs(output_dir, exist_ok=True)

    def _out_path_for(pth: str) -> str:
        out_name = os.path.splitext(os.path.basename(pth))[0] + ".json"
        return os.path.join(output_dir, out_name)

    total = len(persona_files)
    existing = 0
    todo_personas: list[str] = []
    for p in persona_files:
        if os.path.exists(_out_path_for(p)):
            existing += 1
        else:
            todo_personas.append(p)

    remaining = len(todo_personas)
    print(f"[Resume] Output directory: {output_dir}")
    print(f"[Resume] Found {existing} completed out of {total} total. Remaining: {remaining}.")
    if remaining == 0:
        print("[Resume] Nothing to do. All persona outputs already exist.")
        return

    use_progress = not no_progress

    call_agent_model, call_user_model = base.call_models_factory(provider, agent_model, user_model)
    call_judge_chat = base._get_chat_caller(provider)  # reuse shared caller

    if fail_fast_auth:
        try:
            _ = call_judge_chat(
                agent_model,
                [
                    {"role": "system", "content": "Ping"},
                    {"role": "user", "content": "auth preflight"},
                ],
            )
        except Exception as exc:
            msg = str(exc)
            if "Incorrect API key" in msg or "Authentication error" in msg or "invalid_api_key" in msg:
                print("[FATAL] Authentication failed in preflight. Aborting run.", file=sys.stderr)
                print(msg, file=sys.stderr)
                return

    ref_user_messages: list[dict[str, Any]] = []
    if inline_style_judge and real_ref and os.path.exists(real_ref):
        ref_obj = base.read_json(real_ref)
        raw_list = ref_obj.get("message_list") or []
        if isinstance(raw_list, list):
            for m in raw_list:
                role = str(m.get("role", "")).lower()
                if role in ("user", "caller"):
                    ref_user_messages.append({"content": m.get("content", "")})
    if inline_style_judge and not ref_user_messages:
        inline_style_judge = False

    indexed_personas: List[Tuple[int, str]] = list(enumerate(persona_files))

    def _run_one(item: Tuple[int, str]) -> str:
        idx, pth = item
        out_path = _out_path_for(pth)
        if os.path.exists(out_path):
            return out_path

        sample_index = _sample_index_from_path(pth, idx)

        sampled_texts: List[str] = []
        if inline_style_judge and ref_user_messages:
            rng_local = random.Random(_stable_int(f"style::{seed}::{sample_index}") & 0x7FFFFFFF)
            k = min(style_ref_user_sample_size, len(ref_user_messages))
            sampled = rng_local.sample(ref_user_messages, k) if k < len(ref_user_messages) else list(ref_user_messages)
            rng_local.shuffle(sampled)
            sampled_texts = [m.get("content", "") for m in sampled]

        persona_run_seed = _stable_int(f"run::{seed}::{sample_index}") & 0x7FFFFFFF

        convo = simulate_one_refine(
            pth,
            oracle,
            modified,
            max_turns=max_turns,
            violation_portion=violation_portion,
            seed=persona_run_seed,
            call_agent_model=call_agent_model,
            call_user_model=call_user_model,
            inline_style_judge=inline_style_judge,
            judge_model=judge_model,
            call_judge_chat=call_judge_chat,
            ref_user_messages=sampled_texts,
            inline_max_iters=inline_max_iters,
            end_with_agent=True,
        )
        base.write_json(out_path, convo)
        return out_path

    max_concurrency = max(1, max_concurrency)
    if tqdm is not None and use_progress:
        pbar = tqdm(total=total, desc="Refining", unit="conv")
    else:
        pbar = None

    async def _runner() -> None:
        sem = asyncio.Semaphore(max_concurrency)
        manual_count = 0

        async def run_one_async(item: Tuple[int, str]):
            async with sem:
                return item[1], await asyncio.to_thread(_run_one, item)

        tasks = [asyncio.create_task(run_one_async(it), name=it[1]) for it in indexed_personas]
        for coro in asyncio.as_completed(tasks):
            try:
                p, out_path = await coro
                if not use_progress:
                    print(f"Wrote: {out_path}")
            except Exception:
                task_name = coro.get_name() if hasattr(coro, "get_name") else "unknown"
                print(f"Error generating {task_name}", file=sys.stderr)
                print(traceback.format_exc(), file=sys.stderr)
            finally:
                manual_count += 1
                if pbar is not None:
                    pbar.update(1)
                elif use_progress:
                    print(f"Refining: {manual_count}/{total}", end="\r", flush=True)

        if pbar is not None:
            pbar.close()
        elif use_progress:
            print()

    try:
        asyncio.get_running_loop()
        _run_many_with_threads(_run_one, indexed_personas, max_concurrency, use_progress)
    except RuntimeError:
        asyncio.run(_runner())


def _run_many_with_threads(func, items, max_workers, use_progress):
    if not items:
        return
    pbar = tqdm(total=len(items), desc="Refining", unit="conv") if (tqdm is not None and use_progress) else None
    completed = 0
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(func, item): item for item in items}
        for fut in as_completed(futures):
            try:
                _ = fut.result()
            except Exception:
                print(traceback.format_exc(), file=sys.stderr)
            finally:
                completed += 1
                if pbar is not None:
                    pbar.update(1)
                elif use_progress:
                    print(f"Refining: {completed}/{len(items)}", end="\r", flush=True)
    if pbar is not None:
        pbar.close()
    elif use_progress:
        print()


def main() -> int:
    if not os.path.exists(CONFIG_FILE):
        print(f"Config file not found: {CONFIG_FILE}", file=sys.stderr)
        return 2
    cfg = load_config(CONFIG_FILE)
    run_from_config(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
