#!/usr/bin/env python3
"""Filter SCAN judge training data by total token length.

Loads dump/train_SCAN.jsonl (or a user-provided path), tokenizes each example
using Qwen's chat template (same as training), and drops rows whose total
prompt+response tokens exceed --max-tokens. Prints the before/after counts and
optionally writes the filtered dataset to a new JSONL file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter judge training jsonl by token count.")
    parser.add_argument(
        "--input",
        default="dump/train_SCAN.jsonl",
        help="Path to the source JSONL file.",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the filtered JSONL. If omitted, overwrites the input file.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Tokenizer name to use for token counting.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        required=True,
        help="Keep examples whose prompt+response tokens are <= this number.",
    )
    return parser.parse_args()


def build_tokenizer(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def count_tokens(example: Dict[str, Any], tokenizer) -> int:
    messages: List[Dict[str, str]] = example["input_messages"]
    context_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    response_text = example["target_response"]
    if not response_text.endswith(tokenizer.eos_token):
        response_text = response_text + tokenizer.eos_token
    full_text = context_text + response_text
    tokenized = tokenizer(
        full_text,
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )
    return len(tokenized["input_ids"])


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")
    output_path = Path(args.output) if args.output else input_path

    tokenizer = build_tokenizer(args.model)

    kept: List[str] = []
    total = 0

    with input_path.open("r", encoding="utf-8") as reader:
        for line in reader:
            line = line.strip()
            if not line:
                continue
            total += 1
            example = json.loads(line)
            token_count = count_tokens(example, tokenizer)
            if token_count <= args.max_tokens:
                kept.append(json.dumps(example, ensure_ascii=False))

    with output_path.open("w", encoding="utf-8") as writer:
        for line in kept:
            writer.write(line + "\n")

    print(f"Total examples: {total}")
    print(f"Kept <= {args.max_tokens} tokens: {len(kept)}")
    print(f"Dropped: {total - len(kept)}")
    if output_path == input_path:
        print("Input file overwritten with filtered data.")
    else:
        print(f"Filtered dataset written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
