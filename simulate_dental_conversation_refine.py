#!/usr/bin/env python3
"""Refined dental conversation simulator using YAML config and inline style refinement.

Loads parameters from `dental_conversation_config.yaml` so you don't need long CLI args.
Original logic preserved in `simulate_dental_conversation_original.py`.
"""
from __future__ import annotations
import os, json, yaml, random, sys, math, re, traceback, asyncio, hashlib  # <-- added hashlib
from typing import Any, List, Iterable, Tuple

# Reuse original implementations
import simulate_dental_conversation as base

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "dental_conversation_config.yaml")
ANALYSIS_MARK = base.ANALYSIS_MARK

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed  # Option A: thread fallback

# ---------------------------------------------------------------------------
# Threaded fallback runner for environments with an active event loop
# ---------------------------------------------------------------------------
def _run_many_with_threads(func, items, max_workers, use_progress):
    """
    func: callable that accepts a single 'item' from items. In this file,
          'item' is a tuple (idx, persona_path).
    """
    total = len(items)
    if total == 0:
        return
    pbar = tqdm(total=total, desc="Refining", unit="conv") if (tqdm is not None and use_progress) else None
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(func, p): p for p in items}
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
                    print(f"Refining: {completed}/{total}", end="\r", flush=True)
    if pbar is not None:
        pbar.close()
    elif use_progress:
        print()


# ---------------------------------------------------------------------------
# Discriminator-based inline refinement simulation (user turns only)
# ---------------------------------------------------------------------------

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
    ref_user_messages: List[str],  # sampled real user messages
    inline_max_iters: int,
    end_with_agent: bool = True,
) -> dict[str, Any]:
    """Simulation variant adding a discriminator loop for user message realism.

    For each generated user message:
    - Form a pool = sampled real user messages + candidate.
    - Shuffle and ask judge to identify least human-sounding message.
    - If judge selects candidate (index match), use judge reason to rewrite and repeat.
    - Stop when judge fails to pick candidate, returns -1, or max iterations reached.
    """
    rng = random.Random(seed)
    persona = base.read_json(persona_path)

    violation_directives = base.sample_violation_directives(
        oracle,
        modified,
        portion=violation_portion,
        rng=rng,
    )
    # print(violation_directives)
    user_sys = base.persona_to_user_system_prompt(persona)
    agent_sys = base.build_agent_system_prompt(oracle, violation_directives)
    # print("agent_sys: \n\n", agent_sys)

    public_messages: List[dict] = []
    mistakes: list[dict[str, Any]] = []

    use_refine = inline_style_judge and ref_user_messages and judge_model and call_judge_chat

    def _discriminate(candidate: str) -> Tuple[bool, str]:
        """Return (needs_refine, reason). True if judge picked candidate as least human.

        Prompt updated to align with guidance: emphasize mild disfluencies ("yeah", "you know"), contractions,
        natural pauses ("...", dashes), self-correction fragments, and avoidance of over-polish.
        """
        if not use_refine:
            return False, ""
        pool = list(ref_user_messages) + [candidate]
        # Shuffle pool and track candidate index
        order = list(range(len(pool)))
        rng.shuffle(order)
        shuffled = [pool[i] for i in order]
        candidate_index = order.index(len(pool) - 1)  # position after shuffle
        # Build prompt
        lines = [
            "You are a realism discriminator for single caller utterances in a simulated dental implant intake call.",
            "Human cues: mild fillers, informal wording, soft hedges, natural pauses, slight self-corrections, spontaneity, disfluencies,and rhythm,brevity/verbosity balance.",
            "Synthetic cues: overly formal, perfectly structured sentences, absence of any casual markers, verbose redundancy, salesy tone, and brevity/verbosity imbalance.",
            "Pick the SINGLE least human sounding line. If none is distinctly synthetic, output -1.",
            'Return ONLY JSON, no commentary: {"least_index": <int or -1>, "reason": "<concise rationale>"}.',
            "Messages:" ,
        ]
        for i, msg in enumerate(shuffled):
            lines.append(f"{i}: {msg}")
        prompt = "\n".join(lines)
        resp = call_judge_chat(judge_model, [
            {"role": "system", "content": "You output ONLY JSON."},
            {"role": "user", "content": prompt},
        ])
        txt = resp.strip()
        if txt.startswith("```"):
            parts = txt.split("```")
            if len(parts) >= 2:
                body = parts[1]
                if body.startswith("json"):
                    body = body[len("json"):]
                txt = body.strip()
        try:
            data = json.loads(txt)
            li = int(data.get("least_index", -1))
            reason = str(data.get("reason", "")).strip()
            if li == candidate_index:
                return True, reason or "Less natural wording"
            else:
                return False, reason
        except Exception:
            return False, "Parsing failure; stop refinement"

    def _rewrite(candidate: str, reason: str) -> str:
        if not use_refine:
            return candidate
        prompt = (
            "Rewrite the SINGLE message below to better match realistic tone/wording/vibe.\n"
            "Constraints: keep the same intent and any factual tokens (names, dates, addresses, numbers) exact; do not invent new facts.\n"
            "Favor mild fillers, informal wording, soft hedges, natural pauses, slight self-corrections, spontaneity, disfluencies, and balanced rhythm/verbosity.\n"
            'Return strict JSON only: { "rewrite": "<string>" }.\n\n'
            f"ORIGINAL: {candidate}\n"
            f"DISCRIMINATOR_REASON: {reason}\n"
        )
        resp = call_judge_chat(judge_model, [
            {"role": "system", "content": "You output ONLY JSON."},
            {"role": "user", "content": prompt},
        ])
        txt = resp.strip()
        if txt.startswith("```"):
            parts = txt.split("```")
            if len(parts) >= 2:
                body = parts[1]
                if body.startswith("json"):
                    body = body[len("json"):]
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
        # Agent turn
        agent_reply_raw, analysis = call_agent_model(agent_sys, public_messages)
        public_messages.append({"role": "assistant", "content": agent_reply_raw})

        # Record mistakes per original logic
        if isinstance(analysis, dict) and analysis.get("correctness", "").lower() == "mistake":
            cat = base.normalize_category(str(analysis.get("category", "")).strip())
            key = str(analysis.get("key", "")).strip()
            phase_num = analysis.get("phase", None)
            valid = False
            if cat and key:
                if cat in ("Category 1: Universal Compliance", "Category 3: Human Agent Handoff"):
                    section = oracle.get(cat, {}) or {}
                    valid = isinstance(section, dict) and key in section
                elif cat == "Category 2: Step-by-Step Workflow":
                    section = oracle.get(cat, {}) or {}
                    if isinstance(section, dict) and key in section:
                        phases = section.get(key, []) if isinstance(section.get(key, None), list) else []
                        if isinstance(phase_num, int) and 1 <= phase_num <= len(phases):
                            valid = True
            if valid:
                mistakes.append({
                    "turn_index": len(public_messages) - 1,
                    "guidance category": cat,
                    "guidance key": key,
                    "guideline_phase": (phase_num if cat == "Category 2: Step-by-Step Workflow" else -1),
                    "evidence": agent_reply_raw,
                })

        # Termination check
        term_flag = analysis.get("terminate") if isinstance(analysis, dict) else False
        if isinstance(term_flag, bool) and term_flag:
            terminate = True
        if terminate:
            break
        if end_with_agent and turn == max_turns - 1:
            break

        # User turn raw
        user_reply_raw = call_user_model(user_sys, public_messages)
        candidate = re.sub(r"^\s*Caller\s*:\s*", "", user_reply_raw.strip(), flags=re.IGNORECASE)

        # Discriminator refinement loop
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

    message_list = []
    for idx, m in enumerate(public_messages):
        message_list.append({
            "turn_index": idx,
            "role": m.get("role"),
            "content": m.get("content", ""),
        })
    return {
        "persona_path": persona_path,
        "message_list": message_list,
        "mistakes": mistakes,
        "style_ref_messages_count": len(ref_user_messages),
        "violation_directives": violation_directives,  # <-- include directives in output
    }


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_persona_files(folder: str) -> list[str]:
    paths = []
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith(".json"):
            paths.append(os.path.join(folder, name))
    return paths


# ---- stable helper for per-round seed derivation ----
def _stable_int(s: str) -> int:
    """Stable 128-bit int from a string using MD5 (sufficient here)."""
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def run_from_config(cfg: dict[str, Any]) -> None:
    personas_path = cfg.get("personas", "user_persona/dental")
    max_turns = int(cfg.get("max_turns", 10))
    violation_portion = float(cfg.get("violation_portion", 0.8))
    seed = int(cfg.get("seed", 42))
    provider = cfg.get("provider", "azure")
    agent_model = cfg.get("agent_model", "gpt-5")
    user_model = cfg.get("user_model", "gpt-4o")
    judge_model = cfg.get("judge_model", user_model)
    inline_style_judge = bool(cfg.get("inline_style_judge", True))
    real_ref = cfg.get("real_ref", "") or None
    inline_max_iters = int(cfg.get("inline_max_iters", 3))
    style_ref_user_sample_size = int(cfg.get("style_ref_user_sample_size", 8))  # number of USER messages sampled per simulation (default 8 per request)
    max_concurrency = int(cfg.get("max_concurrency", cfg.get("workers", 5)))  # fall back to legacy workers key
    output_dir = cfg.get("output_dir", "dump/simulated_conv_refine")
    limit = int(cfg.get("limit", 0))
    no_progress = bool(cfg.get("no_progress", False))
    fail_fast_auth = bool(cfg.get("fail_fast_auth", True))

    # Load guidelines (fixed paths)
    oracle = base.read_json(os.path.join("guidelines", "dental", "oracle.json"))
    modified = base.read_json(os.path.join("guidelines", "dental", "modified.json"))

    # Personas
    if os.path.isdir(personas_path):
        persona_files = iter_persona_files(personas_path)
    else:
        persona_files = [personas_path]
    if limit > 0:
        persona_files = persona_files[:limit]

    # Ensure output dir exists, then build resume plan
    os.makedirs(output_dir, exist_ok=True)

    # Map each persona to its expected output file
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

    # Early exit if everything is done
    if remaining == 0:
        print("[Resume] Nothing to do. All persona outputs already exist.")
        return

    use_progress = not no_progress

    call_agent_model, call_user_model = base.call_models_factory(provider, agent_model, user_model)
    call_judge_chat = base._get_chat_caller(provider)  # type: ignore (intentional private use)

    # Preflight authentication (avoid flooding with repeated 401s)
    if fail_fast_auth:
        try:
            _ = call_judge_chat(agent_model, [
                {"role": "system", "content": "Ping"},
                {"role": "user", "content": "auth preflight"},
            ])
        except Exception as exc:
            msg = str(exc)
            if "Incorrect API key" in msg or "Authentication error" in msg or "invalid_api_key" in msg:
                print("[FATAL] Authentication failed in preflight. Aborting run.", file=sys.stderr)
                print(msg, file=sys.stderr)
                return
            # Non-auth errors may be transient; continue.

    # Pre-load reference conversation (for style judge)
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

    # Pair personas with a stable index so each "round" gets a distinct seed
    indexed_personas: List[Tuple[int, str]] = list(enumerate(persona_files))

    def _run_one(item: Tuple[int, str]) -> str:
        idx, pth = item

        # Skip if output already exists (double-check in case of concurrent runs)
        out_path = _out_path_for(pth)
        if os.path.exists(out_path):
            return out_path

        # Sample subset of real user messages for this simulation
        # (User said rng_local is fine; we keep existing derivation)
        sampled_texts: List[str] = []
        if inline_style_judge and ref_user_messages:
            rng_local = random.Random(seed + (abs(hash(pth)) % 10_000_000))
            k = min(style_ref_user_sample_size, len(ref_user_messages))
            sampled = rng_local.sample(ref_user_messages, k) if k < len(ref_user_messages) else list(ref_user_messages)
            rng_local.shuffle(sampled)
            sampled_texts = [m.get("content", "") for m in sampled]

        # ---- Different RNG for every "round" (persona run), but reproducible ----
        # Mix global seed with stable hash of persona path and the round index.
        persona_run_seed = (
            (seed & 0x7FFFFFFF) ^
            (_stable_int(pth) & 0x7FFFFFFF) ^
            ((idx + 1) * 2654435761 & 0x7FFFFFFF)
        ) & 0x7FFFFFFF

        convo = simulate_one_refine(
            pth,
            oracle,
            modified,
            max_turns=max_turns,
            violation_portion=violation_portion,
            seed=persona_run_seed,  # <-- per-round unique seed
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
            except Exception as exc:
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

    # Option A: choose between asyncio (CLI) and threads (Jupyter/IPython with active loop)
    try:
        asyncio.get_running_loop()  # If this succeeds, we're inside an active event loop
        _run_many_with_threads(_run_one, indexed_personas, max_concurrency, use_progress)
    except RuntimeError:
        # No running loop -> it's safe to use asyncio.run
        asyncio.run(_runner())


def _iter_progress(items: List[str], total: int, enabled: bool):
    if not enabled:
        for x in items:
            yield x
        return
    if tqdm is not None:
        yield from tqdm(items, total=total, desc="Refining", unit="conv")
    else:
        count = 0
        for x in items:
            count += 1
            print(f"Refining: {count}/{total}", end="\r", flush=True)
            yield x
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