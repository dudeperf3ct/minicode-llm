# Reasoning RLVR Run Guide

This guide owns the reasoning-specific configuration, safety checks, launch commands, and
go/no-go criteria for `configs/reasoning-fft.yml`. Run commands from the parent `rlvr/`
directory.

The reasoning run is intentionally budgeted rather than an unrestricted long-context
experiment. The starting reasoning SFT model averaged 6,359 output tokens on the private
held-out set and frequently generated much longer responses on LiveCodeBench. A 2,048-token
rollout budget would therefore truncate many completions before final code is produced.

## Experiment Design

The run keeps the same 1,000 decontaminated Hugging Face prompts as the direct experiment.
No dataset regeneration or upload is required.

| Setting | Value | Purpose |
| --- | ---: | --- |
| Sequence length | 12,288 | Fit the prompt and an 8K reasoning completion |
| Maximum completion | 8,192 | Give the model enough space to close the reasoning trace |
| Generations per prompt | 8 | Preserve the direct experiment's group size |
| Loss | `dr_grpo` | Avoid the original GRPO response-length normalization |
| Reward scaling | Disabled | Preserve the intended 1.0 and 0.05 reward magnitudes |
| Truncated completion mask | Enabled | Exclude incomplete trajectories from the loss |
| Importance sampling | Token-level truncated, cap 3.0 | Limit vLLM/trainer policy mismatch |
| Checkpoints | 4 per epoch | Save at approximately steps 250, 500, 750, and 1,000 |

The total reward is:

```text
test correctness + 0.05 * reasoning format
```

Test correctness remains binary: all public tests must pass. The format reward requires a
closed `</think>` block followed by extractable, syntactically valid Python. It does not call
Modal and cannot outweigh correctness.

The run deliberately keeps `use_data_producer: false` and `async_prefetch: false`. Axolotl
0.18.0's asynchronous trainer is not compatible with TRL 1.8.0's `num_tiles` argument.

## 1. Verify the Local Project

Install the project environment and run the focused checks:

```bash
uv sync --group dev
source .venv/bin/activate

uv run pytest tests/test_rewards.py
uv run ruff check src tests
```

Verify the Modal reward environment before renting GPU time:

```bash
modal setup
modal profile current
uv run rlvr-verifier-smoke
```

## 2. Verify the GPU Environment

Build the pinned GPU environment using the commands in the parent README. Reapply the
Axolotl compatibility fixes after any reinstall:

```bash
source .venv/bin/activate
python scripts/patch_axolotl_018.py
```

Keep the known working versions:

```text
axolotl 0.18.0
trl 1.8.0
vllm 0.23.0+cu129
torch 2.11.0+cu129
```

Do not upgrade TRL independently of Axolotl and do not enable the asynchronous data
producer for this run.

## 3. Run the One-Step Smoke

Start a fresh rollout server on GPU 0:

```bash
mkdir -p logs
set -o pipefail

CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve configs/reasoning-fft.yml \
  2>&1 | tee logs/vllm-smoke-reasoning-fft.log
```

Wait for it from another shell:

```bash
curl --fail http://127.0.0.1:8000/health/
```

Run one optimizer step on GPU 1:

```bash
set -o pipefail

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=1 axolotl train configs/reasoning-fft.yml \
  --max-steps 1 \
  --output-dir ./outputs/smoke-reasoning-fft \
  2>&1 | tee logs/train-smoke-reasoning-fft.log
```

The smoke must generate eight completions, execute both reward functions, synchronize the
LoRA adapter, and finish the optimizer step without a traceback. Stop the rollout server
afterward. Never reuse a vLLM process that has received weights from a previous run.

## 4. Start the Full Run

Start a new rollout server on GPU 0:

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve configs/reasoning-fft.yml \
  2>&1 | tee logs/vllm-reasoning-fft.log
```

Train on GPU 1:

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=1 axolotl train configs/reasoning-fft.yml \
  2>&1 | tee logs/train-reasoning-fft.log
```

The output and W&B run ID are both `main-1k-reasoning-fft-grpo-lora-seed42`.

## 5. Apply the Early Go/No-Go Gate

Inspect W&B and `logs/train-reasoning-fft.log` after 32--50 completed steps. These steps are
part of the real experiment; no separate pilot output is created.

| Metric | Continue when | Stop and investigate when |
| --- | --- | --- |
| `completions/clipped_ratio` | Rolling average is below 0.15 | It remains above 0.20 |
| `frac_reward_zero_std` | At least 40% of groups have non-zero reward variance | More than 60% of groups remain all-pass or all-fail |
| `rewards/reasoning_format_reward/mean` | Both zero and non-zero values appear | It is always zero |
| `sampling/importance_sampling_ratio/mean` | It does not persistently collapse near zero | Most steps are effectively zero |
| Modal transport | Retries recover and batches complete | Retries exhaust or stalls dominate step time |

Also inspect several completions. Successful behavior should contain a closed reasoning trace
and final Python code within the 8,192-token budget. Do not continue merely because the loss
is finite if nearly every completion is truncated or every group has zero reward variance.

## 6. Stop and Resume

Interrupt the trainer with `Ctrl-C`, then stop the rollout server. A resumable checkpoint must
contain `trainer_state.json`, `optimizer.pt`, and `scheduler.pt`.

Before resuming, start a fresh vLLM server from `configs/reasoning-fft.yml`. Resume training on
GPU 1 with the unchanged config:

```bash
CKPT=outputs/main-1k-reasoning-fft-grpo-lora-seed42/checkpoint-500

CUDA_VISIBLE_DEVICES=1 axolotl train configs/reasoning-fft.yml \
  --resume-from-checkpoint "$CKPT" \
  2>&1 | tee -a logs/train-reasoning-fft.log
```

Do not change LoRA targets, optimizer settings, reward weights, or sequence lengths between
the original run and a resume.

## 7. Evaluate Checkpoints

The four checkpoints make step 250 and step 500 available before committing to the full
epoch. Use training telemetry for the early gate, then compare a promising checkpoint with
the reasoning SFT baseline using the merge and evaluation workflow in the parent README.

For a checkpoint merge, pass its directory explicitly:

```bash
axolotl merge-lora configs/reasoning-fft.yml \
  --lora-model-dir ./outputs/main-1k-reasoning-fft-grpo-lora-seed42/checkpoint-500
```

Public evaluation should use the `thinking` profile, and the held-out evaluator should use
`--mode reasoning`. Record which checkpoint was evaluated in the report directory name.

If reasoning performance is still neutral after the length, format, and importance-sampling
changes, the next experiment should improve prompt selection or introduce partial public-test
credit rather than immediately increasing the dataset size.
