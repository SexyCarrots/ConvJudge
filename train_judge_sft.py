#!/usr/bin/env python3
"""Supervised fine-tuning script for the SCAN judge model.

This follows the style of TBAMA/sft/train_router_sft.py but adapts it to the
`dump/train_SCAN.jsonl` dataset that we just generated. Each training example
contains:
  - input_messages: (system, user) instructions
  - target_response: JSON listing the ground-truth violations

We tokenize using the base model's chat template so the model learns to respond
with the strict JSON object. The script performs an in-memory train/val split,
tokenizes with HF Datasets, and launches a standard Trainer job.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, load_dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# --------------------------- Hyperparameters -------------------------------- #
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATA_PATH = Path("dump/train_SCAN_4k.jsonl")
NUM_EPOCHS = 10
LEARNING_RATE = 5e-6
OUTPUT_DIR = Path(f"/mnt/tmp/training_res/scan_judge_3b_{NUM_EPOCHS}ep_{LEARNING_RATE}lr")
RUN_NAME = f"scan_judge_3b_{NUM_EPOCHS}ep_{LEARNING_RATE}lr"
MAX_SEQ_LENGTH = 4096
PER_DEVICE_BSZ = 2
GRAD_ACCUM = 4 
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STRATEGY = "epoch"
EVAL_STRATEGY = "epoch"
EVAL_SPLIT = 0.05  # 5% hold-out from training file
MAP_NUM_PROC = 16
BF16 = True
GRADIENT_CHECKPOINTING = True
SEED = 42
# ---------------------------------------------------------------------------- #


def _prepare_datasets() -> DatasetDict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training file not found: {DATA_PATH}")
    raw = load_dataset("json", data_files={"train": str(DATA_PATH)})
    train_full: Dataset = raw["train"]
    if EVAL_SPLIT and 0.0 < EVAL_SPLIT < 1.0:
        split = train_full.shuffle(seed=SEED).train_test_split(
            test_size=EVAL_SPLIT, seed=SEED
        )
        ds = DatasetDict(train=split["train"], validation=split["test"])
    else:
        # Use a tiny slice as validation if requested split is invalid.
        val_size = min(256, len(train_full))
        ds = DatasetDict(
            train=train_full,
            validation=train_full.select(range(val_size)),
        )
    return ds


def build_tokenized_dataset(tokenizer) -> DatasetDict:
    ds = _prepare_datasets()

    def _tokenize(example: Dict[str, Any]) -> Dict[str, List[int]]:
        messages = example["input_messages"]
        assistant_msg = {"role": "assistant", "content": example["target_response"]}
        context_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        response_text = assistant_msg["content"]
        if not response_text.endswith(tokenizer.eos_token):
            response_text = response_text + tokenizer.eos_token
        full_text = context_text + response_text

        tokenized = tokenizer(
            full_text,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,
        )
        context_ids = tokenizer(
            context_text,
            max_length=MAX_SEQ_LENGTH,
            truncation=True,
            padding=False,
        )["input_ids"]

        labels = [-100] * len(tokenized["input_ids"])
        start_idx = min(len(context_ids), len(labels))
        for idx in range(start_idx, len(labels)):
            labels[idx] = tokenized["input_ids"][idx]
        tokenized["labels"] = labels
        return tokenized

    remove_cols = ds["train"].column_names
    return ds.map(
        _tokenize,
        remove_columns=remove_cols,
        num_proc=MAP_NUM_PROC,
        desc="Tokenizing SCAN judge dataset",
    )


class DataCollatorForCausalLM:
    def __init__(self, tokenizer, pad_to_multiple_of: int | None = None):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            remainder = max_len % self.pad_to_multiple_of
            if remainder:
                max_len += self.pad_to_multiple_of - remainder

        input_ids, labels, attention = [], [], []
        for feat in features:
            ids = feat["input_ids"]
            lbls = feat["labels"]
            pad_len = max_len - len(ids)
            input_ids.append(ids + [self.pad_token_id] * pad_len)
            attention.append([1] * len(ids) + [0] * pad_len)
            labels.append(lbls + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    os.environ.setdefault("WANDB_PROJECT", "scan-judge-sft")
    os.environ.setdefault("WANDB_WATCH", "false")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    datasets_tokenized = build_tokenized_dataset(tokenizer)
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        report_to="wandb",
        run_name=RUN_NAME,
        per_device_train_batch_size=PER_DEVICE_BSZ,
        per_device_eval_batch_size=PER_DEVICE_BSZ,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_strategy=SAVE_STRATEGY,
        eval_strategy=EVAL_STRATEGY,
        save_total_limit=1,
        bf16=BF16,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        remove_unused_columns=False,
        seed=SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets_tokenized["train"],
        eval_dataset=datasets_tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    main()
