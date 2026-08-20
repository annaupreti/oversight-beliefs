# Contrastive SDF on gpt-oss-120b

Reusable RunPod training scaffold for one direction of the released
`apollo-research/contrastive-belief-updates` corpus. It first trains:

```text
comprehensions__grader + loops__llm_users
```

Run the crossed direction afterwards to form the proper Grader-vs-User
contrastive pair.

## What this does—and the important replication boundary

This retains the released corpus construction: named HF configs only, all main
documents, contrast selection with seed 43 until token-matched within 1%, final
shuffle seed 42, one epoch, effective batch 8, AdamW, cosine schedule, 300
warmup steps, LR `3.5e-5`, rank/alpha `32/32`, and no generic-data mix.

It is **not byte-identical to the paper's Tinker LoRA run**. To fit
gpt-oss-120b on one H100, it uses Unsloth 4-bit QLoRA with the seven standard
attention/MLP projection targets. The paper trained conventional LoRA and also
included the unembedding. Each `run_manifest.json` records that difference, the
exact configurations, seeds, token totals, and cache locations. Do not call a
result an exact replication without a matched full-precision LoRA run.

## Architecture

```text
GHCR image (code + CUDA libraries + pinned Python stack)
        |
        v
RunPod H100 container
  /workspace/huggingface-cache       base model + only requested HF configs
  /workspace/sdf-prepared-cache      tokenized documents, reusable per config
  /workspace/oversight-beliefs-runs  adapters, checkpoints, manifests
  /workspace/wandb                   local W&B files; dashboard is remote
```

The image intentionally contains no weights, data, or secrets. `/workspace`
is the only state you need to preserve. In a different RunPod region without a
shared volume, pull the same image, then let the model/data caches warm again.

## One-time: publish the container

1. Push this repository to `main`. The `Build training container` GitHub Action
   starts automatically for Docker/training changes.
2. In GitHub, open **Actions → Build training container**. For an explicit,
   stable version, select **Run workflow** and use `unsloth-qlora-v1` as the
   tag. Wait for the job to finish successfully.
3. In GitHub **Packages**, open `oversight-beliefs` and make the package
   public. This is appropriate only because the image has no secrets or data.
   If you keep it private, authenticate RunPod to GHCR with a separate
   least-privilege `read:packages` token.

The resulting public image is:

```text
ghcr.io/annaupreti/oversight-beliefs:unsloth-qlora-v1
```

Rebuild only after changing the Dockerfile, requirements, trainer, or launch
scripts. Give experiments new output directories/run names; do not rebuild an
identical environment for each run.

## RunPod: create the pod

1. Choose **Secure Cloud → 1× H100 SXM 80GB**. This is the lowest-risk first
   configuration for gpt-oss-120b QLoRA. H200 is faster but unnecessary for a
   first stable run; do not use a 24–48GB GPU.
2. Choose the custom image above. Use a 250–300GB volume if the selected region
   offers one. The first model download is large; 300GB avoids space pressure
   from model, dataset, prepared cache, adapters, checkpoints, and W&B files.
3. Set the container start command to `bash -lc 'sleep infinity'`.
4. Add RunPod environment variables/secrets:

   - `HF_TOKEN` — a Hugging Face **write** token. It downloads models/data and
     creates or updates your adapter repository after a successful full run.
   - `WANDB_API_KEY` — a freshly created W&B key.
   - Optional: `WANDB_ENTITY`, `WANDB_PROJECT`.

   Never bake these values into the image or commit `.env`. If an API key was
   ever displayed in a terminal, editor capture, or chat, revoke and replace it.
5. Deploy, open a terminal, and check that the allocated GPU is correct:

```bash
nvidia-smi
cd /app/oversight-beliefs
```

### Direct SSH alternative: one uv setup command

If you started a generic RunPod image rather than the prebuilt container,
connect with SSH, clone/pull this repository under `/workspace`, then run:

```bash
cd /workspace/oversight-beliefs
chmod +x setup_runpod_uv.sh
./setup_runpod_uv.sh
```

It installs `uv` if absent, creates `/workspace/venvs/contrastive-sdf`, pins
the CUDA 12.8 PyTorch stack, installs the rest of the dependencies, and runs a
CUDA/Unsloth check. The environment persists with the network volume. Every
launcher automatically activates it, so the next command is simply
`./run_smoke_test.sh`.

## Run in order

Run each command from `/app/oversight-beliefs`.

First, run the one-step end-to-end smoke test. It downloads only the two named
SDF configs, tokenizes and persists them, loads QLoRA, takes one optimizer
step, saves an adapter, and creates a W&B run.

```bash
chmod +x *.sh
./run_smoke_test.sh
```

On success, choose the Hugging Face model repository that will receive the
final adapter. It is created automatically if it does not yet exist.

```bash
export HF_MODEL_REPO=annaupreti/gpt-oss-120b-sdf-grader-comprehensions-user-loops
# Optional: create new repositories as private.
# export HF_MODEL_PRIVATE=1
```

Then start the full first direction:

```bash
./run_one_sdf.sh
```

It writes to:

```text
/workspace/oversight-beliefs-runs/gpt-oss-120b/grader-comprehensions__user-loops
```

For the full crossed pair, run:

```bash
export HF_MODEL_REPO=annaupreti/gpt-oss-120b-sdf-grader-comprehensions-user-loops
export HF_MODEL_REPO_CROSSED=annaupreti/gpt-oss-120b-sdf-user-comprehensions-grader-loops
./run_contrastive_sdf.sh
```

The wrapper does not repeat the first direction when its completed adapter is
already present. It then trains:

```text
comprehensions__llm_users + loops__grader
```

After each successful full run, the launcher uploads a loadable final PEFT
adapter at the root of its Hugging Face model repository, plus its model card,
manifest, metrics, and Trainer state under `training/`. It does not upload
intermediate `checkpoint-*` directories by default: those contain resumable
optimizer state and cost more storage/bandwidth. To publish them too, invoke
`train_sdf.py` directly with `--upload-resume-checkpoints`.

W&B groups the smoke and full runs under
`gpt-oss-120b__grader-vs-user__comprehensions-vs-loops`. Its charts include
loss, learning rate, gradient norm, throughput, and GPU telemetry. Use another
project without editing code:

```bash
WANDB_PROJECT=my-project ./run_one_sdf.sh
```

## Efficiency and recovery

- The named `load_dataset` calls fetch only each configuration required by the
  launched direction, not the complete release.
- The raw Hugging Face cache and tokenized-document cache live in `/workspace`.
  A retry, second direction, or resumed pod reuses them without re-download or
  re-tokenization.
- Every launcher writes the complete terminal stream to
  `/workspace/oversight-beliefs-runs/logs/` while still displaying it in tmux.
- Do not enable sequence packing for this first reproduction: it changes
  document boundaries and makes its step/data accounting less directly
  comparable to the paper.
- If a full run stops, resume with the checkpoint directory the trainer saved:

```bash
./run_one_sdf.sh --resume-from-checkpoint /workspace/oversight-beliefs-runs/gpt-oss-120b/grader-comprehensions__user-loops/checkpoint-250
```

- The H100 launch deliberately uses microbatch `1` with gradient accumulation
  `8`: this preserves the paper's effective batch size while leaving room for
  long, untruncated documents. It is slower than batch 8 but much more robust.
  If you later validate spare VRAM, test `--batch-size 2
  --gradient-accumulation-steps 4`; do not lower context length or silently
  truncate documents.

The original `setup_runpod.sh` is deliberately a short migration notice now;
the container replaces one-off virtualenv installation on every pod.
