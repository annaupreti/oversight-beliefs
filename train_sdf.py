#!/usr/bin/env python3
"""Train one contrastive-SDF LoRA adapter on gpt-oss-120b.

Example (grader prefers comprehensions; users prefer loops):
  python train_sdf.py \
    --main-config comprehensions__grader \
    --contrast-config loops__llm_users \
    --output-dir artifacts/gpt-oss-120b/grader-comp__user-loops

This script deliberately handles one *direction* only.  A contrastive result
requires running it again with the authority/style assignments crossed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


DATASET_ID = "apollo-research/contrastive-belief-updates"
DEFAULT_MODEL_ID = "openai/gpt-oss-120b"


@dataclass
class RunManifest:
    model_id: str
    model_revision: str | None
    dataset_id: str
    main_config: str
    contrast_config: str
    tokenizer_name: str
    max_length: int
    main_documents: int
    contrast_documents: int
    main_cache_files: list[str]
    contrast_cache_files: list[str]
    main_tokens: int
    contrast_tokens: int
    token_ratio: float
    main_seed: int
    contrast_seed: int
    final_shuffle_seed: int
    lora_rank: int
    lora_alpha: int
    lora_target_modules: str
    learning_rate: float
    warmup_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: float
    max_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-config", required=True)
    parser.add_argument("--contrast-config", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help="Reject documents longer than this instead of silently truncating them.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contrast-seed", type=int, default=43)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Use 2 with --batch-size 4 only if the H100 smoke test OOMs.",
    )
    parser.add_argument("--learning-rate", type=float, default=3.5e-5)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Override the epoch length. Use 1 for the end-to-end RunPod smoke test.",
    )
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", "contrastive-belief-updates"),
        help="Weights & Biases project. Set --disable-wandb to avoid logging.",
    )
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def token_ids(tokenizer: Any, text: str, max_length: int) -> list[int]:
    """Use the released corpus's documented add_special_tokens=True policy."""
    ids = tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]
    if len(ids) > max_length:
        raise ValueError(
            f"Document has {len(ids)} tokens, over --max-length={max_length}. "
            "Increase --max-length; do not silently truncate SDF documents."
        )
    return ids


def add_token_counts(dataset: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    def encode(row: dict[str, Any]) -> dict[str, Any]:
        ids = token_ids(tokenizer, row["content"], max_length)
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "n_tokens": len(ids)}

    return dataset.map(encode, desc="Tokenizing SDF documents")


def select_token_matched_contrast(
    contrast: Dataset, target_tokens: int, seed: int
) -> Dataset:
    """Shuffle then retain whole documents until contrast reaches main's tokens."""
    shuffled = contrast.shuffle(seed=seed)
    selected: list[int] = []
    total = 0
    for index, n_tokens in enumerate(shuffled["n_tokens"]):
        selected.append(index)
        total += n_tokens
        if total >= target_tokens:
            break

    if total < target_tokens:
        raise RuntimeError("Contrast corpus was exhausted before reaching the main token count.")
    ratio = total / target_tokens
    if not 0.99 <= ratio <= 1.01:
        raise RuntimeError(
            f"Contrast/main token ratio is {ratio:.5f}, outside the paper's 1% tolerance. "
            "Use the released balanced corpus and inspect document formatting."
        )
    return shuffled.select(selected)


def load_pair(args: argparse.Namespace, tokenizer: Any) -> tuple[Dataset, RunManifest]:
    main_raw = load_dataset(args.dataset_id, args.main_config, split="train")
    contrast_raw = load_dataset(args.dataset_id, args.contrast_config, split="train")
    main_cache_files = [entry["filename"] for entry in main_raw.cache_files]
    contrast_cache_files = [entry["filename"] for entry in contrast_raw.cache_files]

    main = add_token_counts(main_raw, tokenizer, args.max_length)
    contrast = add_token_counts(contrast_raw, tokenizer, args.max_length)
    main_tokens = sum(main["n_tokens"])
    contrast_selected = select_token_matched_contrast(
        contrast, main_tokens, args.contrast_seed
    )
    contrast_tokens = sum(contrast_selected["n_tokens"])

    combined = Dataset.from_dict(
        {
            "input_ids": main["input_ids"] + contrast_selected["input_ids"],
            "attention_mask": main["attention_mask"] + contrast_selected["attention_mask"],
        }
    ).shuffle(seed=args.seed)

    manifest = RunManifest(
        model_id=args.model_id,
        model_revision=args.model_revision,
        dataset_id=args.dataset_id,
        main_config=args.main_config,
        contrast_config=args.contrast_config,
        tokenizer_name=tokenizer.name_or_path,
        max_length=args.max_length,
        main_documents=len(main),
        contrast_documents=len(contrast_selected),
        main_cache_files=main_cache_files,
        contrast_cache_files=contrast_cache_files,
        main_tokens=main_tokens,
        contrast_tokens=contrast_tokens,
        token_ratio=contrast_tokens / main_tokens,
        main_seed=args.seed,
        contrast_seed=args.contrast_seed,
        final_shuffle_seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules="all exposed torch.nn.Linear modules, including lm_head",
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    return combined, manifest


class CausalLMCollator:
    """Pads document examples and supplies next-token labels for causal LM SFT."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def linear_module_names(model: torch.nn.Module) -> list[str]:
    """Return every torch Linear module, explicitly retaining the unembedding."""
    names = sorted(
        name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)
    )
    if not names:
        raise RuntimeError("No torch.nn.Linear LoRA targets found in the loaded model.")
    if "lm_head" not in names:
        raise RuntimeError(
            "The loaded architecture did not expose lm_head as a torch.nn.Linear module. "
            "Stop here and inspect the model before claiming an all-linear SDF run."
        )
    return names


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. Launch this on a GPU RunPod instance.")
    if torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
        raise RuntimeError("This starter is intended for an 80GB-class GPU.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id, revision=args.model_revision, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset, manifest = load_pair(args, tokenizer)
    with (args.output_dir / "run_manifest.json").open("w") as handle:
        json.dump(asdict(manifest), handle, indent=2, sort_keys=True)

    wandb_run = None
    run_name = args.run_name or f"{args.main_config}__{args.contrast_config}"
    if not args.disable_wandb:
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError(
                "WANDB_API_KEY is not set. Add it as a RunPod secret or pass --disable-wandb."
            )
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=os.environ.get("WANDB_RUN_GROUP"),
            tags=["contrastive-sdf", "smoke-test" if args.max_steps > 0 else "full-run"],
            config=asdict(manifest),
            dir=str(args.output_dir),
        )

    # No device_map: Trainer/Accelerate places the model on this one GPU.
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.model_revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    # Base weights stay frozen under LoRA. This keeps activations entering the
    # checkpointed transformer blocks differentiable so LoRA gradients flow.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    target_modules = linear_module_names(model)
    with (args.output_dir / "lora_target_modules.json").open("w") as handle:
        json.dump(target_modules, handle, indent=2)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        optim="adamw_torch_fused",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        report_to=[] if args.disable_wandb else ["wandb"],
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=args.seed,
    )
    collator = CausalLMCollator(tokenizer)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        trainer.save_model(str(args.output_dir / "adapter"))
        tokenizer.save_pretrained(str(args.output_dir / "adapter"))
        trainer.save_state()
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
