# One contrastive-SDF LoRA run on gpt-oss-120b

This is the minimum RunPod starter for **one SDF direction** from
*Measuring Reward-Seeking via Contrastive Belief Updates*:

> The automated grader prefers list comprehensions; LLM users prefer explicit
> loops.

It trains a LoRA adapter on the two released synthetic-document corpora. It
does **not** build a model organism, run RL, or evaluate behavior. A
contrastive behavioral result requires a second run with the assignments
crossed:

```text
comprehensions__llm_users + loops__grader
```

## RunPod setup

1. Create a Secure Cloud pod with **one H100 80GB** and a 250GB+ persistent
   volume. Use a recent CUDA/PyTorch image.
2. Clone/copy this directory onto the persistent volume.
3. Add these RunPod secrets/environment variables:

   - `HF_TOKEN`: a Hugging Face read token.
   - `WANDB_API_KEY`: your Weights & Biases API key.
   - Optional: `WANDB_ENTITY` and `WANDB_PROJECT`.

4. Run the one-time setup script:

```bash
chmod +x setup_runpod.sh run_one_sdf.sh
./setup_runpod.sh
```

`setup_runpod.sh` creates `.venv`, installs CUDA PyTorch and the remaining
dependencies, builds FlashAttention, checks CUDA visibility, and writes a
package lockfile. Set `TORCH_INDEX_URL` if your pod image needs a different
PyTorch CUDA-wheel index.

Before committing to the full training run, run the one-step end-to-end smoke
test. It downloads the base model and the two selected SDF corpora, tokenizes
them, loads the LoRA configuration, performs one optimizer step, and creates a
W&B run. It therefore catches model-format, CUDA-kernel, VRAM, Hugging Face,
and W&B failures at minimal GPU cost.

```bash
chmod +x run_smoke_test.sh
./run_smoke_test.sh
```

If that succeeds, run the full first direction:

```bash
chmod +x run_one_sdf.sh
./run_one_sdf.sh
```

The job writes an adapter, checkpoints, and `run_manifest.json` under
`artifacts/gpt-oss-120b/grader-comprehensions__user-loops/`.

This is the recommended **smoke test**. It downloads only these two Hugging
Face dataset configurations: `comprehensions__grader` and `loops__llm_users`.
The exact local cache paths are saved in `run_manifest.json`. Because the
script loads named Hugging Face configurations, it does not load the complete
30-configuration release; the persistent `HF_HOME` cache also prevents a
re-download when you resume a run.

It also creates a W&B run in `contrastive-belief-updates` by default. The
dashboard shows loss, learning rate, gradient norm, throughput, system GPU
telemetry, and the paper/data hyperparameters in the run config. Use
`WANDB_PROJECT=my-project ./run_one_sdf.sh` to choose another project.
The smoke test and both crossed directions are grouped under
`gpt-oss-120b__grader-vs-user__comprehensions-vs-loops`, making them easy to
filter and compare in the W&B dashboard.

Once the smoke test completes, run both crossed directions serially on the
same H100:

```bash
chmod +x run_contrastive_sdf.sh
./run_contrastive_sdf.sh
```

The second run downloads only `comprehensions__llm_users` and `loops__grader`.
It creates a separate W&B run; the two saved adapters are the proper
Grader-versus-User contrastive SDF pair.

## Paper settings represented here

- Dataset: `apollo-research/contrastive-belief-updates`
- Main corpus: all 4,600 documents
- Contrast corpus: shuffled with seed 43, selected whole-document-wise until
  token-matched to the main corpus within 1%
- Combined data shuffle seed: 42
- One epoch, batch size 8, AdamW, cosine schedule, 300 warmup steps
- Peak LR `3.5e-5`
- LoRA rank/alpha `32/32`, every exposed `torch.nn.Linear` target including
  `lm_head`/unembedding (saved as `lora_target_modules.json`)
- No generic-data mixing and no DOCTAG prefix

`run_smoke_test.sh` changes only `max_steps=1`; `run_one_sdf.sh` retains the
paper's one-epoch / approximately 1,150-step configuration.

If the H100 smoke test fails specifically with CUDA out-of-memory, do not
lower `--max-length` or silently truncate documents. Re-run the smoke test
with `--batch-size 4 --gradient-accumulation-steps 2` to preserve the paper's
effective batch size of eight, then make the same adjustment to the full-run
launcher and record it as a replication deviation.

## Before launching the full run

Run an import/load smoke test on the selected pod first. The precise
Transformers + CUDA combination must support gpt-oss MXFP4 weights and
FlashAttention 2. Save `pip freeze` alongside the artifacts once it loads.

The script intentionally rejects documents longer than `--max-length` rather
than silently truncating them. If that occurs, re-run with a larger context
length and record the change in the manifest.

## Important limitation

The released dataset gives synthetic SDF documents, not the paper's held-out
evaluation suite. This script trains one adapter only. You still need to run
the crossed adapter and evaluate both adapters on identical held-out prompts
to obtain a contrastive SDF measurement.
