#!/usr/bin/env python3
"""Run one paper-style contrastive-SDF LoRA direction through Tinker.

The script runs on a CPU-only local machine. Tinker executes the remote GPU
forward/backward operations and retains a final checkpoint under a tinker://
path. It trains raw synthetic documents with next-token cross-entropy, not a
chat template.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import datasets
import torch
import tinker
from tinker_cookbook.supervised.common import datum_from_model_input_weights

DATASET_ID = "apollo-research/contrastive-belief-updates"
MODEL_ID = "openai/gpt-oss-120b"


@dataclass
class Manifest:
    base_model: str
    dataset_id: str
    main_config: str
    contrast_config: str
    main_documents: int
    contrast_documents: int
    main_tokens: int
    contrast_tokens: int
    token_ratio: float
    contrast_seed: int
    final_shuffle_seed: int
    lora_rank: int
    train_mlp: bool
    train_attn: bool
    train_unembed: bool
    peak_learning_rate: float
    warmup_steps: int
    batch_size: int
    epochs: int
    requested_steps: int


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-config", required=True)
    parser.add_argument("--contrast-config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3.5e-5)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--contrast-seed", type=int, default=43)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "contrastive-belief-updates"))
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args()


def cosine_lr(peak: float, step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return peak * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak * 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def tokenized_documents(tokenizer, config: str) -> list[dict]:
    raw = datasets.load_dataset(DATASET_ID, config, split="train")
    documents = []
    for row in raw:
        tokens = list(tokenizer.encode(row["content"]))
        if len(tokens) < 2:
            raise ValueError(f"{config} has a document with fewer than two tokens")
        documents.append({"tokens": tokens, "n_tokens": len(tokens)})
    return documents


def main() -> None:
    config = args()
    if not os.environ.get("TINKER_API_KEY"):
        raise RuntimeError("Set TINKER_API_KEY locally before launching this script.")
    config.output_dir.mkdir(parents=True, exist_ok=True)

    service = tinker.ServiceClient(user_metadata={"experiment": "contrastive-sdf"})
    client = service.create_lora_training_client(
        base_model=config.model_id,
        rank=config.lora_rank,
        train_mlp=True,
        train_attn=True,
        train_unembed=True,
        user_metadata={"main_config": config.main_config, "contrast_config": config.contrast_config},
    )
    tokenizer = client.get_tokenizer()
    print(f"Created Tinker LoRA client for {config.model_id}")
    main_docs = tokenized_documents(tokenizer, config.main_config)
    contrast_docs = tokenized_documents(tokenizer, config.contrast_config)
    main_tokens = sum(doc["n_tokens"] for doc in main_docs)

    generator = torch.Generator().manual_seed(config.contrast_seed)
    order = torch.randperm(len(contrast_docs), generator=generator).tolist()
    selected, contrast_tokens = [], 0
    for index in order:
        selected.append(contrast_docs[index])
        contrast_tokens += contrast_docs[index]["n_tokens"]
        if contrast_tokens >= main_tokens:
            break
    if contrast_tokens < main_tokens or not 0.99 <= contrast_tokens / main_tokens <= 1.01:
        raise RuntimeError("Token matching failed; inspect corpus/tokenizer configuration.")

    combined = main_docs + selected
    generator = torch.Generator().manual_seed(config.shuffle_seed)
    permutation = torch.randperm(len(combined), generator=generator).tolist()
    combined = [combined[index] for index in permutation]
    total_steps = len(combined) // config.batch_size
    if len(combined) % config.batch_size:
        raise RuntimeError("Expected exactly divisible paired corpus at batch size 8.")
    if config.max_steps > 0:
        total_steps = min(total_steps, config.max_steps)
    manifest = Manifest(
        base_model=config.model_id, dataset_id=DATASET_ID, main_config=config.main_config,
        contrast_config=config.contrast_config, main_documents=len(main_docs),
        contrast_documents=len(selected), main_tokens=main_tokens, contrast_tokens=contrast_tokens,
        token_ratio=contrast_tokens / main_tokens, contrast_seed=config.contrast_seed,
        final_shuffle_seed=config.shuffle_seed, lora_rank=config.lora_rank, train_mlp=True,
        train_attn=True, train_unembed=True, peak_learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps, batch_size=config.batch_size, epochs=1,
        requested_steps=total_steps,
    )
    (config.output_dir / "run_manifest.json").write_text(json.dumps(asdict(manifest), indent=2))

    run = None
    if not config.disable_wandb:
        import wandb
        run = wandb.init(project=config.wandb_project, name=(
            f"tinker-{config.main_config}__{config.contrast_config}"), config=asdict(manifest),
            tags=["contrastive-sdf", "tinker", "gpt-oss-120b"])

    for step in range(total_steps):
        docs = combined[step * config.batch_size : (step + 1) * config.batch_size]
        batch = []
        for document in docs:
            model_input = tinker.types.ModelInput.from_ints(document["tokens"])
            batch.append(datum_from_model_input_weights(
                model_input, torch.ones(model_input.length), max_length=None, reduction="none"))
        lr = cosine_lr(config.learning_rate, step, total_steps, config.warmup_steps)
        forward = client.forward_backward(batch, loss_fn="cross_entropy")
        optimization = client.optim_step(tinker.AdamParams(
            learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8))
        forward_result = forward.result()
        optimization_result = optimization.result()
        logprobs = torch.cat([output["logprobs"].to_torch() for output in forward_result.loss_fn_outputs])
        weights = torch.cat([datum.loss_fn_inputs["weights"].to_torch() for datum in batch])
        metrics = {"step": step + 1, "learning_rate": lr, "train_mean_nll": float(-(logprobs * weights).sum() / weights.sum())}
        if optimization_result.metrics:
            metrics.update(optimization_result.metrics)
        print(json.dumps(metrics), flush=True)
        if run:
            run.log(metrics, step=step + 1)
        if config.save_every and (step + 1) % config.save_every == 0:
            state = client.save_state(f"contrastive-sdf-step-{step + 1:04d}", ttl_seconds=None).result()
            print(f"Saved resumable Tinker state: {state}", flush=True)

    final_state = client.save_state("contrastive-sdf-final", ttl_seconds=None).result()
    weights = client.save_weights_for_sampler("contrastive-sdf-final").result()
    (config.output_dir / "tinker_checkpoints.json").write_text(json.dumps({
        "final_state": str(final_state), "final_weights": str(weights)}, indent=2))
    print(f"Final Tinker state: {final_state}\nFinal sampler weights: {weights}", flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
