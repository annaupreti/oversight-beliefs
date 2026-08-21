#!/usr/bin/env python3
"""Query a finished Tinker sampler on fixed qualitative SDF probes.

This is deliberately an exploratory tool. It records prompts, reasoning traces,
final answers, and a lightweight Python-style label. It does not estimate the
paper's contrastive reward-seeking statistic; that requires the crossed SDF
model and repeated samples from both counterfactual conditions.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tinker


DEFAULT_MODEL = (
    "tinker://aff54c88-13e0-58d8-833f-bb15f73873ad:train:0/"
    "sampler_weights/contrastive-sdf-final"
)
DEFAULT_TASKS = Path("evals/tinker_probe_tasks.jsonl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing output directory, skipping task/repeat pairs with saved responses.",
    )
    parser.add_argument("--task-id", action="append", default=None,
                        help="Run only this task ID; may be passed more than once.")
    return parser.parse_args()


def read_tasks(path: Path) -> list[dict[str, Any]]:
    tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    identifiers = [task["id"] for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Probe task IDs must be unique.")
    return tasks


def extract_python(answer: str) -> str:
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", answer, flags=re.DOTALL | re.IGNORECASE)
    return fenced[0].strip() if fenced else answer.strip()


def classify_style(answer: str) -> str | None:
    """Return a simple AST label; None means the answer is not parseable Python."""
    try:
        tree = ast.parse(extract_python(answer))
    except SyntaxError:
        return None
    has_comprehension = any(isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)) for node in ast.walk(tree))
    has_loop = any(isinstance(node, (ast.For, ast.AsyncFor)) for node in ast.walk(tree))
    if has_comprehension and has_loop:
        return "both"
    if has_comprehension:
        return "comprehension"
    if has_loop:
        return "loop"
    return "neither"


def render_messages(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    # Different Transformers versions return a list, BatchEncoding, or mapping.
    token_ids = getattr(rendered, "input_ids", None)
    if token_ids is None and hasattr(rendered, "__getitem__"):
        try:
            token_ids = rendered["input_ids"]
        except (KeyError, TypeError):
            token_ids = rendered
    if token_ids is None:
        token_ids = rendered
    return list(token_ids)


def split_reasoning(raw_completion: str, clean_completion: str) -> tuple[str | None, str]:
    """Best-effort split while retaining raw text for later parser improvements."""
    analysis_match = re.search(
        r"<\|channel\|>analysis(?:<\|message\|>)?(.*?)(?=<\|channel\|>final|\Z)",
        raw_completion,
        flags=re.DOTALL,
    )
    final_match = re.search(
        r"<\|channel\|>final(?:<\|message\|>)?(.*)", raw_completion, flags=re.DOTALL
    )
    # Keep the original completion separately, but remove gpt-oss chat control
    # tokens from fields intended for human review and AST classification.
    remove_control_tokens = lambda text: re.sub(r"<\|[^>]+\|>", "", text).strip()
    reasoning = remove_control_tokens(analysis_match.group(1)) if analysis_match else None
    answer = remove_control_tokens(final_match.group(1)) if final_match else clean_completion
    return reasoning, answer


def sample(
    client: tinker.SamplingClient,
    tokenizer: Any,
    task: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, Any]:
    prompt = tinker.ModelInput.from_ints(render_messages(tokenizer, task["messages"]))
    response = client.sample(
        prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=seed,
        ),
    ).result()
    sequence = response.sequences[0]
    raw_completion = tokenizer.decode(sequence.tokens, skip_special_tokens=False)
    clean_completion = tokenizer.decode(sequence.tokens, skip_special_tokens=True)
    reasoning, answer = split_reasoning(raw_completion, clean_completion)
    return {
        "reasoning": reasoning,
        "answer": answer,
        "raw_completion": raw_completion,
        "style_label": classify_style(answer) if task.get("feature") == "comprehensions_vs_loops" else None,
        "stop_reason": str(sequence.stop_reason),
    }


def main() -> None:
    args = parse_args()
    api_key = os.environ.get("TINKER_API_KEY")
    if not api_key:
        raise RuntimeError("Set TINKER_API_KEY before running probes.")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    service = tinker.ServiceClient(user_metadata={"experiment": "contrastive-sdf-qualitative-probes"})
    client = service.create_sampling_client(model_path=args.model)
    tokenizer = client.get_tokenizer()
    tasks = read_tasks(args.tasks)
    if args.task_id:
        requested = set(args.task_id)
        tasks = [task for task in tasks if task["id"] in requested]
        missing = requested - {task["id"] for task in tasks}
        if missing:
            raise ValueError(f"Unknown task IDs: {', '.join(sorted(missing))}")
    if args.limit is not None:
        tasks = tasks[:args.limit]
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or Path("artifacts/tinker_probes") / timestamp
    response_path = output_dir / "responses.jsonl"
    completed_keys: set[tuple[str, int]] = set()
    if args.resume:
        if not output_dir.is_dir() or not response_path.is_file():
            raise ValueError("--resume requires an existing output directory with responses.jsonl.")
        # An interrupted process can be killed mid-write. Preserve every
        # complete JSONL record and discard only an incomplete trailing record
        # so a resumed run always produces valid JSONL.
        raw_lines = response_path.read_bytes().splitlines(keepends=True)
        valid_bytes = 0
        for line in raw_lines:
            if not line.strip():
                valid_bytes += len(line)
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print("Discarding incomplete trailing JSONL record before resuming.", file=sys.stderr)
                break
            valid_bytes += len(line)
            if "error" not in record:
                completed_keys.add((record["task_id"], record["repeat_index"]))
        if valid_bytes != sum(len(line) for line in raw_lines):
            response_path.write_bytes(response_path.read_bytes()[:valid_bytes])
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    task_bytes = args.tasks.read_bytes()
    manifest = {
        "model": args.model,
        "tasks": str(args.tasks),
        "task_sha256": hashlib.sha256(task_bytes).hexdigest(),
        "task_count": len(tasks),
        "repeats": args.repeats,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "sampling_backend": "native_tinker_sdk",
        "seed": args.seed,
        "created_at_utc": timestamp,
        "purpose": "qualitative probe; not a contrastive reward-seeking estimate",
    }
    if not args.resume:
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output_dir / "tasks.jsonl").write_bytes(task_bytes)

    pending = [
        (repeat_index, task)
        for repeat_index in range(args.repeats)
        for task in tasks
        if (task["id"], repeat_index) not in completed_keys
    ]
    total = len(pending)
    completed = 0
    with response_path.open("a" if args.resume else "w") as handle:
        for repeat_index, task in pending:
            completed += 1
            record: dict[str, Any] = {
                "task_id": task["id"],
                "category": task["category"],
                "feature": task.get("feature"),
                "repeat_index": repeat_index,
                "sampling_seed": args.seed + repeat_index,
                "messages": task["messages"],
            }
            try:
                record.update(sample(client, tokenizer, task, args, args.seed + repeat_index))
                print(f"[{completed}/{total}] {task['id']}: {record['style_label']}", flush=True)
            except Exception as error:
                record["error"] = str(error)
                print(f"[{completed}/{total}] {task['id']}: ERROR {error}", file=sys.stderr, flush=True)
            handle.write(json.dumps(record) + "\n")
            handle.flush()

    print(f"Saved probe responses to {response_path}")


if __name__ == "__main__":
    main()
