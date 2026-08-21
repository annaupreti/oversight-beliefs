#!/usr/bin/env python3
"""Train one token-matched contrastive-SDF QLoRA adapter on gpt-oss-120b.

This is a single-GPU replication *variant*: the released corpus construction
and paper hyperparameters are retained, while Unsloth 4-bit QLoRA makes the
120B model fit on one 80GB H100. Run the crossed direction separately before
making a contrastive behavioural claim.
"""

from __future__ import annotations

# Unsloth must patch Transformers before either Transformers or PEFT is imported.
# Keep this import first among third-party packages.
import unsloth  # noqa: F401

import argparse
import hashlib
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from huggingface_hub import HfApi
from transformers import Trainer, TrainerCallback, TrainingArguments, set_seed
from unsloth import FastLanguageModel


DATASET_ID = "apollo-research/contrastive-belief-updates"
DEFAULT_MODEL_ID = "unsloth/gpt-oss-120b-unsloth-bnb-4bit"
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
]
PREPARED_FORMAT_VERSION = 1


@dataclass
class RunManifest:
    runner: str
    replication_variant: str
    model_id: str
    model_revision: str | None
    dataset_id: str
    main_config: str
    contrast_config: str
    tokenizer_name: str
    max_length: int
    prepared_cache_root: str
    main_prepared_path: str
    contrast_prepared_path: str
    main_documents: int
    contrast_documents: int
    main_tokens: int
    contrast_tokens: int
    token_ratio: float
    contrast_seed: int
    final_shuffle_seed: int
    quantization: str
    lora_rank: int
    lora_alpha: int
    lora_target_modules: list[str]
    learning_rate: float
    warmup_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: float
    max_steps: int
    hf_repo_id: str | None
    hf_private: bool


@dataclass(frozen=True)
class DistributedContext:
    """Process placement supplied by torchrun, or a single-process fallback."""

    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def configure_distributed() -> DistributedContext:
    """Initialize one NCCL process per GPU when launched with torchrun."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-config", required=True)
    parser.add_argument("--contrast-config", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--prepared-cache", type=Path, default=Path(
        os.environ.get("SDF_PREPARED_CACHE", "/workspace/sdf-prepared-cache")
    ))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--contrast-seed", type=int, default=43)
    # On a single 80GB H100, QLoRA leaves too little activation headroom for
    # eight long documents at once. This preserves the paper's effective batch 8.
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.5e-5)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--wandb-project", default=os.environ.get(
        "WANDB_PROJECT", "contrastive-belief-updates"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Model repository to receive the final adapter after successful training.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Make the repository private if this invocation creates it.",
    )
    parser.add_argument(
        "--upload-resume-checkpoints",
        action="store_true",
        help="Also upload Trainer checkpoint-* directories (usually unnecessary and large).",
    )
    return parser.parse_args()


def cache_path(args: argparse.Namespace, config: str) -> Path:
    key = json.dumps({
        "format": PREPARED_FORMAT_VERSION,
        "dataset": args.dataset_id,
        "config": config,
        "model": args.model_id,
        "revision": args.model_revision,
        "max_length": args.max_length,
    }, sort_keys=True).encode()
    return args.prepared_cache / config / hashlib.sha256(key).hexdigest()[:16]


def prepare_config(args: argparse.Namespace, tokenizer: Any, config: str) -> tuple[Dataset, Path]:
    """Download only one named HF config once, then persist tokenized documents."""
    prepared = cache_path(args, config)
    if prepared.exists():
        print(f"Reusing prepared corpus: {prepared}")
        return load_from_disk(str(prepared)), prepared

    raw = load_dataset(args.dataset_id, config, split="train")

    def encode(row: dict[str, Any]) -> dict[str, Any]:
        ids = tokenizer(row["content"], add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) > args.max_length:
            raise ValueError(
                f"{config} contains a {len(ids)}-token document, above --max-length="
                f"{args.max_length}; do not silently truncate SDF documents."
            )
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "n_tokens": len(ids)}

    columns = raw.column_names
    tokenized = raw.map(encode, remove_columns=columns, desc=f"Tokenizing {config}")
    prepared.parent.mkdir(parents=True, exist_ok=True)
    tokenized.save_to_disk(str(prepared))
    return tokenized, prepared


def select_token_matched_contrast(contrast: Dataset, target_tokens: int, seed: int) -> Dataset:
    shuffled = contrast.shuffle(seed=seed)
    selected, total = [], 0
    for index, n_tokens in enumerate(shuffled["n_tokens"]):
        selected.append(index)
        total += n_tokens
        if total >= target_tokens:
            break
    if total < target_tokens:
        raise RuntimeError("Contrast corpus ended before reaching the main token count.")
    ratio = total / target_tokens
    if not 0.99 <= ratio <= 1.01:
        raise RuntimeError(f"Contrast/main token ratio {ratio:.5f} is outside 1% tolerance.")
    return shuffled.select(selected)


def load_pair(
    args: argparse.Namespace, tokenizer: Any, distributed: DistributedContext
) -> tuple[Dataset, RunManifest]:
    # A shared volume makes this both faster and safer: one rank builds a missing
    # tokenized cache, then the other ranks only read it after the barrier.
    if distributed.is_main_process:
        main, main_path = prepare_config(args, tokenizer, args.main_config)
        contrast, contrast_path = prepare_config(args, tokenizer, args.contrast_config)
    if distributed.world_size > 1:
        dist.barrier()
    if not distributed.is_main_process:
        main, main_path = prepare_config(args, tokenizer, args.main_config)
        contrast, contrast_path = prepare_config(args, tokenizer, args.contrast_config)
    main_tokens = sum(main["n_tokens"])
    contrast_selected = select_token_matched_contrast(contrast, main_tokens, args.contrast_seed)
    contrast_tokens = sum(contrast_selected["n_tokens"])
    combined = concatenate_datasets([main, contrast_selected]).shuffle(seed=args.seed)
    combined = combined.remove_columns("n_tokens")
    manifest = RunManifest(
        runner="Unsloth FastLanguageModel + Transformers Trainer",
        replication_variant=(
            "Paper corpus mixing and optimization settings on single-H100 4-bit QLoRA; "
            "not byte-identical to the paper's Tinker LoRA implementation."
        ),
        model_id=args.model_id, model_revision=args.model_revision, dataset_id=args.dataset_id,
        main_config=args.main_config, contrast_config=args.contrast_config,
        tokenizer_name=tokenizer.name_or_path, max_length=args.max_length,
        prepared_cache_root=str(args.prepared_cache), main_prepared_path=str(main_path),
        contrast_prepared_path=str(contrast_path), main_documents=len(main),
        contrast_documents=len(contrast_selected), main_tokens=main_tokens,
        contrast_tokens=contrast_tokens, token_ratio=contrast_tokens / main_tokens,
        contrast_seed=args.contrast_seed, final_shuffle_seed=args.seed,
        quantization="bitsandbytes 4-bit QLoRA", lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha, lora_target_modules=LORA_TARGETS,
        learning_rate=args.learning_rate, warmup_steps=args.warmup_steps,
        batch_size=args.batch_size, gradient_accumulation_steps=args.gradient_accumulation_steps,
        epochs=args.epochs, max_steps=args.max_steps, hf_repo_id=args.hf_repo_id,
        hf_private=args.hf_private,
    )
    return combined, manifest


class CausalLMCollator:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def write_model_card(output_dir: Path, manifest: RunManifest) -> Path:
    """Write a compact Hub card adjacent to the trained adapter."""
    card = output_dir / "hf_model_card.md"
    card.write_text(
        "---\n"
        "library_name: peft\n"
        f"base_model: {manifest.model_id}\n"
        "tags:\n- qlora\n- contrastive-sdf\n- gpt-oss\n"
        "---\n\n"
        "# gpt-oss-120b contrastive-SDF adapter\n\n"
        "This repository contains a final PEFT adapter trained on the released "
        "Apollo Research contrastive-belief-updates corpus. It is a single-GPU "
        "4-bit QLoRA replication variant, not a byte-identical Tinker LoRA run.\n\n"
        "## Training direction\n\n"
        f"- Main corpus: `{manifest.main_config}`\n"
        f"- Contrast corpus: `{manifest.contrast_config}`\n"
        f"- Main / contrast tokens: `{manifest.main_tokens}` / `{manifest.contrast_tokens}`\n"
        f"- Token ratio: `{manifest.token_ratio:.6f}`\n"
        f"- LoRA rank / alpha: `{manifest.lora_rank}` / `{manifest.lora_alpha}`\n\n"
        "The full run manifest and metrics are in `training/`. Load this as a PEFT "
        "adapter over the base model named in the metadata.\n"
    )
    return card


def upload_to_hub(args: argparse.Namespace, manifest: RunManifest) -> None:
    """Publish a loadable final adapter; resumable optimizer states are opt-in."""
    if not args.hf_repo_id:
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN with write permission is required for --hf-repo-id.")

    api = HfApi(token=token)
    api.create_repo(args.hf_repo_id, repo_type="model", private=args.hf_private, exist_ok=True)
    api.upload_folder(
        repo_id=args.hf_repo_id,
        repo_type="model",
        folder_path=str(args.output_dir / "adapter"),
        commit_message="Upload final contrastive-SDF adapter",
    )
    card = write_model_card(args.output_dir, manifest)
    for local_path, remote_path in [
        (card, "README.md"),
        (args.output_dir / "run_manifest.json", "training/run_manifest.json"),
        (args.output_dir / "train_results.json", "training/train_results.json"),
        (args.output_dir / "trainer_state.json", "training/trainer_state.json"),
    ]:
        if local_path.exists():
            api.upload_file(
                repo_id=args.hf_repo_id,
                repo_type="model",
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                commit_message="Upload contrastive-SDF training metadata",
            )
    if args.upload_resume_checkpoints:
        api.upload_folder(
            repo_id=args.hf_repo_id,
            repo_type="model",
            folder_path=str(args.output_dir),
            path_in_repo="training",
            allow_patterns="checkpoint-*/*",
            commit_message="Upload resumable Trainer checkpoints",
        )
    print(f"Uploaded final adapter to https://huggingface.co/{args.hf_repo_id}")


class HubCheckpointCallback(TrainerCallback):
    """Synchronously mirror resumable Trainer checkpoints to the Hub."""

    def __init__(self, repo_id: str, private: bool, token: str) -> None:
        self.repo_id = repo_id
        self.api = HfApi(token=token)
        self.api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)

    def on_save(
        self,
        args: TrainingArguments,
        state: Any,
        control: Any,
        **kwargs: Any,
    ) -> Any:
        if not state.is_world_process_zero:
            return control
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not checkpoint.is_dir():
            raise RuntimeError(f"Expected saved checkpoint is missing: {checkpoint}")
        remote_path = f"training/checkpoints/checkpoint-{state.global_step}"
        print(f"Uploading resumable checkpoint to HF: {remote_path}")
        self.api.upload_folder(
            repo_id=self.repo_id,
            repo_type="model",
            folder_path=str(checkpoint),
            path_in_repo=remote_path,
            commit_message=f"Upload resumable checkpoint at step {state.global_step}",
        )
        return control


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; launch this on a GPU RunPod pod.")
    distributed = configure_distributed()
    if torch.cuda.get_device_properties(0).total_memory < 70 * 1024**3:
        raise RuntimeError("Use an 80GB-class GPU for gpt-oss-120b QLoRA.")
    if not args.disable_wandb and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set WANDB_API_KEY or pass --disable-wandb.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    random.seed(args.seed)
    tokenizer = None
    device_map = None
    if distributed.world_size > 1:
        # DDP replicates the frozen 4-bit base on every GPU. Each rank must own
        # exactly one replica rather than letting a loader place it on GPU 0.
        device_map = {"": f"cuda:{distributed.local_rank}"}
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_id, revision=args.model_revision, max_seq_length=args.max_length,
        dtype=None, load_in_4bit=True, full_finetuning=False, device_map=device_map,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.config.use_cache = False
    train_dataset, manifest = load_pair(args, tokenizer, distributed)
    model = FastLanguageModel.get_peft_model(
        model, r=args.lora_rank, target_modules=LORA_TARGETS, lora_alpha=args.lora_alpha,
        lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth",
        random_state=args.seed, use_rslora=False, loftq_config=None,
    )

    # Only rank 0 owns external side effects (W&B, Hub uploads and filesystem
    # metadata). Trainer still aggregates loss/gradients across all ranks.
    report_to = [] if args.disable_wandb or not distributed.is_main_process else ["wandb"]
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, num_train_epochs=args.epochs,
        max_steps=args.max_steps, lr_scheduler_type="cosine", warmup_steps=args.warmup_steps,
        optim="adamw_torch_fused", bf16=True, tf32=True, logging_steps=args.logging_steps,
        save_strategy="steps", save_steps=args.save_steps, save_total_limit=2,
        report_to=report_to, run_name=args.run_name, remove_unused_columns=False,
        dataloader_num_workers=2, seed=args.seed,
        ddp_backend="nccl" if distributed.world_size > 1 else None,
        ddp_find_unused_parameters=False if distributed.world_size > 1 else None,
        ddp_broadcast_buffers=False if distributed.world_size > 1 else None,
        ddp_bucket_cap_mb=100 if distributed.world_size > 1 else None,
    )
    if not args.disable_wandb:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
        if args.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
        os.environ.setdefault("WANDB_RUN_GROUP", "gpt-oss-120b__grader-vs-user__comprehensions-vs-loops")
        os.environ.setdefault("WANDB_TAGS", "contrastive-sdf,qlora")

    callbacks = []
    if args.upload_resume_checkpoints:
        if not args.hf_repo_id:
            raise RuntimeError("--upload-resume-checkpoints requires --hf-repo-id.")
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN with write permission is required for checkpoint uploads.")
        callbacks.append(HubCheckpointCallback(args.hf_repo_id, args.hf_private, token))

    if distributed.is_main_process:
        with (args.output_dir / "run_manifest.json").open("w") as handle:
            json.dump(asdict(manifest), handle, indent=2, sort_keys=True)
    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset,
                      data_collator=CausalLMCollator(tokenizer), processing_class=tokenizer,
                      callbacks=callbacks)
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    if distributed.world_size > 1:
        dist.barrier()
    if distributed.is_main_process:
        trainer.save_model(str(args.output_dir / "adapter"))
        tokenizer.save_pretrained(str(args.output_dir / "adapter"))
        peak_gb = torch.cuda.max_memory_allocated() / 1024**3
        trainer.log_metrics("train", {"peak_gpu_memory_gb": peak_gb})
        trainer.save_metrics("train", {"peak_gpu_memory_gb": peak_gb})
        trainer.save_state()
        upload_to_hub(args, manifest)
    if distributed.world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
