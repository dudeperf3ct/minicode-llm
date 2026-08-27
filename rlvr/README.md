# Qwen3.5-4B KodCode RLVR

Two matched LoRA GRPO experiments run on 1,000 verified Python prompts:

1. Direct generation from the direct full-FT SFT checkpoint.
2. Thinking-enabled generation from the reasoning full-FT SFT checkpoint.

Both experiments use the same data, reward, sampling settings, and seed.

## Experiments

| Experiment | Config | Starting checkpoint | Thinking |
| --- | --- | --- | --- |
| Direct FFT | `configs/direct-fft.yml` | `dudeperf3ct/qwen35-4b-kodcode-sft-10k@34355613741e197528920acc211005f303524e58` | Disabled |
| Reasoning FFT | `configs/reasoning-fft.yml` | `dudeperf3ct/qwen35-4b-kodcode-sft-10k@eed04fa11c7b7a9bbd4a129471d1a4e9bd2e1ece` | Enabled |

Each run uses one epoch, seed 42, eight rollouts per prompt, and a binary reward: `1.0` when all public tests pass and `0.0` otherwise.

Tests execute in isolated Modal Sandboxes. The reasoning trace is not treated as code; only the final answer after `</think>` is verified.

The runs produce at most 8,000 completions each. Groups where all eight rewards are equal have no GRPO learning signal and are skipped by Axolotl.

Dataset preparation is documented in [`docs/README.md`](docs/README.md).

## 1. Prepare the Environment

Run commands from this directory:

```bash
cd rlvr
uv sync --group dev
source .venv/bin/activate
```

Authenticate the [Modal SDK](https://modal.com/) on every machine that calculates rewards:

```bash
modal setup
```

## 2. Test the Sandbox

This requires no local GPU. It launches small remote CPU sandboxes and checks passing code, failing code, an execution timeout, and blocked outbound networking:

```bash
uv run rlvr-verifier-smoke
```

Every completion runs in a fresh Modal Sandbox with network access disabled. Model-caused failures receive zero reward. Modal infrastructure errors are retried three times and then stop training.

## 3. Build the GPU Environment

The documented setup targets two x86_64 H100 GPUs: one for the vLLM rollout server and one for training.

```bash
uv sync --group dev
source .venv/bin/activate

uv pip install \
  'torch==2.11.0' \
  'torchvision==0.26.0' \
  --torch-backend=cu129

uv pip install 'xformers==0.0.35'

uv pip install --no-build-isolation \
  'axolotl==0.18.0'

uv pip install 'modal==1.5.4'

uv pip install \
  'https://github.com/vllm-project/vllm/releases/download/v0.23.0/vllm-0.23.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl' \
  --extra-index-url https://download.pytorch.org/whl/cu129

uv pip install 'flash-linear-attention==0.4.1'

uv pip install \
  'flash-attn-3==3.0.0' \
  --index-url https://download.pytorch.org/whl/cu129

uv pip install \
  'https://huggingface.co/datasets/dudeperf3ct/qwen35-sft-wheels/resolve/main/causal-conv1d/py312-torch211-cu128/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl'
```

> [!WARNING]
> Axolotl pins an older Modal client for its cloud launcher, so reinstall Modal after Axolotl to keep the verifier on the current Sandbox filesystem API.

Verify the training dependencies:

```bash
python - <<'PY'
from importlib.metadata import version

import flash_attn_interface
import torch
import transformers

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print(
    "GPUs:",
    [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
)
print("BF16:", torch.cuda.is_bf16_supported())
print("axolotl:", version("axolotl"))
print("modal:", version("modal"))
print("vllm:", version("vllm"))
print("transformers:", transformers.__version__)
print("flash-attn-3:", version("flash-attn-3"))
print("flash-linear-attention:", version("flash-linear-attention"))
print("causal-conv1d:", version("causal-conv1d"))
print("xformers:", version("xformers"))
PY
```

Observed on the two-H100 smoke-test instance:

```text
torch: 2.11.0+cu129
torch CUDA: 12.9
GPUs: ['NVIDIA H100 80GB HBM3', 'NVIDIA H100 80GB HBM3']
BF16: True
axolotl: 0.18.0
modal: 1.5.4
vllm: 0.23.0+cu129
transformers: 5.14.1
flash-attn-3: 3.0.0
flash-linear-attention: 0.4.1
causal-conv1d: 1.6.2.post1
xformers: 0.0.35
```

Authenticate all three services on the GPU machine. Hugging Face loads the pinned model and dataset, W&B records the run, and Modal executes rewards:

```bash
hf auth login
hf auth whoami
wandb login
modal setup
modal profile current
```

Each config uses the Hugging Face repository plus an immutable `revision_of_model`. Hugging Face downloads that snapshot into its local cache, so the trainer and vLLM reuse the same files.

## 4. Run a One-Step GPU Smoke

Run both smokes before starting either full experiment. For each smoke, start its rollout server on GPU 0. Select one matching `CONFIG` and `RUN` pair:

```bash
CONFIG=configs/direct-fft.yml
RUN=direct-fft
# CONFIG=configs/reasoning-fft.yml
# RUN=reasoning-fft

mkdir -p logs
set -o pipefail

CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve "$CONFIG" \
  2>&1 | tee "logs/vllm-smoke-$RUN.log"
```

Wait for the server in another shell:

```bash
curl --fail http://127.0.0.1:8000/health/
```

Then run one optimizer step on GPU 1 using the same config:

```bash
CONFIG=configs/direct-fft.yml
RUN=direct-fft
# CONFIG=configs/reasoning-fft.yml
# RUN=reasoning-fft

set -o pipefail

WANDB_MODE=offline \
CUDA_VISIBLE_DEVICES=1 axolotl train "$CONFIG" \
  --max-steps 1 \
  --output-dir "./outputs/smoke-$RUN" \
  2>&1 | tee "logs/train-smoke-$RUN.log"
```

Stop the vLLM process after each smoke because the trainer synchronized LoRA weights into it. Repeat with `configs/reasoning-fft.yml`, then review both smoke logs before proceeding.

Separate smoke configs are unnecessary: the smoke uses the real experiment config and only overrides `max_steps` and `output_dir`. This avoids duplicating the training settings while keeping smoke outputs separate.

## 5. Run the Full Experiments

For the direct run, start a fresh server on GPU 0:

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve configs/direct-fft.yml \
  2>&1 | tee logs/vllm-direct-fft.log
```

Train in another shell on GPU 1:

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=1 axolotl train configs/direct-fft.yml \
  2>&1 | tee logs/train-direct-fft.log
```

After it finishes, stop the server and repeat with the reasoning config:

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=0 axolotl vllm-serve configs/reasoning-fft.yml \
  2>&1 | tee logs/vllm-reasoning-fft.log
```

```bash
set -o pipefail

CUDA_VISIBLE_DEVICES=1 axolotl train configs/reasoning-fft.yml \
  2>&1 | tee logs/train-reasoning-fft.log
```

Outputs and W&B run names are:

- `main-1k-direct-fft-grpo-lora-seed42`
- `main-1k-reasoning-fft-grpo-lora-seed42`

Watch reward mean and standard deviation, `skipped_zero_adv_batches`, KL, entropy, gradient norm, completion length, and Modal errors.

## 6. Evaluate Each Result

Merge each adapter with its matching config:

```bash
axolotl merge-lora configs/direct-fft.yml \
  --lora-model-dir ./outputs/main-1k-direct-fft-grpo-lora-seed42

axolotl merge-lora configs/reasoning-fft.yml \
  --lora-model-dir ./outputs/main-1k-reasoning-fft-grpo-lora-seed42
```

Use the existing SFT evaluation workflow twice:

| Result | Held-out mode | EvalPlus profile | Compare with |
| --- | --- | --- | --- |
| Direct RLVR | `direct` | `direct` | Direct FFT SFT |
| Reasoning RLVR | `reasoning` | `thinking` | Reasoning FFT SFT |

First serve the direct model. EvalPlus relies on the server's default chat template setting, while the held-out evaluator also sends it per request:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  outputs/main-1k-direct-fft-grpo-lora-seed42/merged \
  --served-model-name main-1k-direct-fft-grpo-lora-seed42 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.80
```

In another shell, run both direct evaluations:

```bash
uv run eval-heldout \
  --config configs/direct-fft.yml \
  --mode direct \
  --dataset-manifest ../sft/manifests/evaluation.json \
  --output-root reports/direct-fft

../evals/scripts/run_evalplus.sh \
  main-1k-direct-fft-grpo-lora-seed42 \
  reports/evalplus-direct-fft \
  --profile direct \
  --base-url http://127.0.0.1:8000/v1
```

Stop the direct server, then serve the reasoning model with thinking enabled. The 65,536-token context supports EvalPlus's 32,768-token thinking budget:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve \
  outputs/main-1k-reasoning-fft-grpo-lora-seed42/merged \
  --served-model-name main-1k-reasoning-fft-grpo-lora-seed42 \
  --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --language-model-only \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.80
```

In another shell, run both reasoning evaluations:

```bash
uv run eval-heldout \
  --config configs/reasoning-fft.yml \
  --mode reasoning \
  --dataset-manifest ../sft/manifests/evaluation.json \
  --output-root reports/reasoning-fft

../evals/scripts/run_evalplus.sh \
  main-1k-reasoning-fft-grpo-lora-seed42 \
  reports/evalplus-reasoning-fft \
  --profile thinking \
  --base-url http://127.0.0.1:8000/v1
```

A negative or neutral delta is still useful if the training metrics and evaluation artifacts explain what happened.

> [!WARNING]
> The shared held-out evaluator executes generated Python locally. Run it only
> on the disposable evaluation VM. Training rewards always use Modal Sandboxes.
