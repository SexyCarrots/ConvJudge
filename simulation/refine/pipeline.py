"""Core refine simulation pipeline with shared helpers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
import sys
import traceback
from typing import Any, List, Tuple

from .. import common as base
from .guidelines import (
    build_override_index,
    filter_guidelines_for_intent,
    resolve_allowed_intents,
    sample_guideline_overrides,
)
from .scenario import ScenarioHooks

try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None


def iter_persona_files(folder: str) -> list[str]:
    """Return sorted persona json files within a folder."""
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".json"):
            paths.append(os.path.join(folder, name))
    return paths


def _stable_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def _sample_index_from_path(path: str, default: int) -> int:
    name = os.path.splitext(os.path.basename(path))[0]
    match = None
    if name:
        match = hashlib.sha256(name.encode("utf-8")).hexdigest()
    if match:
        try:
            return int(match[:8], 16)
        except Exception:
            pass
    return default


def simulate_one_refine(
    persona_path: str,
    oracle: dict[str, Any],
    modified: dict[str, Any],
    *,
    scenario: ScenarioHooks,
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

    cat2_title = scenario.category_titles["cat2"]
    cat2_available = oracle.get(cat2_title, {}) or {}
    allowed_cat2_intents = resolve_allowed_intents(
        persona,
        set(cat2_available.keys()),
        scenario.intent_field,
    )

    oracle_filtered = filter_guidelines_for_intent(
        oracle, titles=scenario.category_titles, allowed_cat2_intents=allowed_cat2_intents
    )
    modified_filtered = filter_guidelines_for_intent(
        modified, titles=scenario.category_titles, allowed_cat2_intents=allowed_cat2_intents
    )

    mutated_guidelines, overrides = sample_guideline_overrides(
        oracle_filtered,
        modified_filtered,
        portion=violation_portion,
        rng=rng,
        titles=scenario.category_titles,
        allowed_cat2_intents=allowed_cat2_intents,
    )
    override_index = build_override_index(overrides, scenario.category_titles)

    user_sys = scenario.build_user_prompt(persona)
    agent_sys = scenario.build_agent_prompt(mutated_guidelines)
    public_messages: List[dict[str, Any]] = []
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
            scenario.style.discriminator_intro,
            scenario.style.human_cues,
            scenario.style.synthetic_cues,
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
            scenario.style.rewrite_instructions.strip()
            + "\n\n"
            + f"ORIGINAL: {candidate}\n"
            + f"CRITIQUE: {reason}\n"
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

        analyzed_cat = scenario.normalize_category(str(analysis.get("category", "")).strip())
        key = str(analysis.get("key", "")).strip()
        phase_raw = analysis.get("phase", -1)
        try:
            phase_num = int(phase_raw)
        except Exception:
            phase_num = -1

        override_hit: dict[str, Any] | None = None
        cat2_title = scenario.category_titles["cat2"]
        cat1_title = scenario.category_titles["cat1"]
        cat3_title = scenario.category_titles["cat3"]
        if analyzed_cat == cat2_title:
            override_hit = override_index.get(analyzed_cat, {}).get((key, phase_num))
        elif analyzed_cat in (cat1_title, cat3_title):
            override_hit = override_index.get(analyzed_cat, {}).get(key)
        if override_hit:
            mistakes.append(
                {
                    "turn_index": len(public_messages) - 1,
                    "guidance category": analyzed_cat,
                    "guidance key": key,
                    "guideline_phase": phase_num if analyzed_cat == cat2_title else -1,
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


def run_refine_pipeline(cfg: dict[str, Any], scenario: ScenarioHooks) -> None:
    personas_path = cfg.get("personas", "user_persona")
    max_turns = int(cfg.get("max_turns", 10))
    violation_portion = float(cfg.get("violation_portion", 0.4))
    seed = int(cfg.get("seed", 42))
    provider = cfg.get("provider", "azure")
    agent_model = cfg.get("agent_model", "gpt-5")
    user_model = cfg.get("user_model", "gpt-4o")
    judge_model = cfg.get("judge_model", user_model)
    inline_style_judge = bool(cfg.get("inline_style_judge", True))
    real_ref = cfg.get("real_ref", "") or None
    inline_max_iters = int(cfg.get("inline_max_iters", 3))
    style_ref_user_sample_size = int(cfg.get("style_ref_user_sample_size", 8))
    max_concurrency = int(cfg.get("max_concurrency", 10))
    output_dir = cfg.get("output_dir", f"dump/simulated_{scenario.name.lower()}_refine")
    limit = int(cfg.get("limit", 0))
    no_progress = bool(cfg.get("no_progress", False))
    fail_fast_auth = bool(cfg.get("fail_fast_auth", True))

    oracle = base.read_json(str(scenario.oracle_path))
    modified = base.read_json(str(scenario.modified_path))

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
    existing = sum(1 for p in persona_files if os.path.exists(_out_path_for(p)))

    print(f"[Resume:{scenario.name}] Output directory: {output_dir}")
    print(f"[Resume:{scenario.name}] Found {existing} completed out of {total} total. Remaining: {total - existing}.")
    if existing == total:
        print("[Resume] Nothing to do. All persona outputs already exist.")
        return

    use_progress = not no_progress
    call_agent_model, call_user_model = base.call_models_factory(provider, agent_model, user_model)
    call_judge_chat = base._get_chat_caller(provider)

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
            if any(tok in msg for tok in ("Incorrect API key", "Authentication error", "invalid_api_key")):
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
            scenario=scenario,
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
    pbar = tqdm(total=total, desc="Refining", unit="conv") if use_progress and tqdm is not None else None

    async def _runner() -> None:
        sem = asyncio.Semaphore(max_concurrency)
        manual_count = 0

        async def run_one_async(item: Tuple[int, str]):
            async with sem:
                return item[1], await asyncio.to_thread(_run_one, item)

        tasks = [asyncio.create_task(run_one_async(it), name=it[1]) for it in indexed_personas]
        for coro in asyncio.as_completed(tasks):
            try:
                _, out_path = await coro
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

    try:
        asyncio.run(_runner())
    finally:
        if pbar is not None:
            pbar.close()
        elif use_progress:
            print()


__all__ = ["iter_persona_files", "run_refine_pipeline", "simulate_one_refine"]
