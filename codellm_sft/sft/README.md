# Qwen3.5-4B KodCode SFT

This directory contains the Axolotl implementation for the staged Qwen3.5-4B coding SFT experiment.

Chat-label audits and the four explicit 10K Axolotl configurations are implemented.

Dataset preparation, split design, provenance, Hub publication, and source references are documented in [`data/README.md`](data/README.md).

## Pinned Runtime

| Component | Version or identifier |
| --- | --- |
| Python | 3.12 |
| Axolotl | `0.18.0` |
| Transformers | `5.14.1` |
| Cut Cross Entropy | Axolotl fork commit `5f0c7a7` |
| Base model and tokenizer | `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b` |

## H100 Training Environment

Lambda Cloud prices observed on July 28, 2026:

| Instance | vCPUs | RAM | SSD | Price/GPU/hour | Price/instance/hour |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 x H100 SXM5 80 GB | 26 | 225 GiB | 2.75 TiB | $4.29 | $4.29 |
| 2 x H100 SXM5 80 GB | 52 | 450 GiB | 5.5 TiB | $4.19 | $8.38 |

The 1K pilots used the single-GPU instance. The four 10K runs use the two-GPU instance.

The observed software environment is:

- NVIDIA driver `580.105.08`;
- driver-supported CUDA `13.0`;
- installed CUDA compiler toolkit `12.8`.

```bash
export UV_TORCH_BACKEND=cu128

uv venv --python 3.12
source .venv/bin/activate

uv pip install \
  'torch==2.11.0' \
  'torchvision==0.26.0'

uv pip install --no-build-isolation \
  'axolotl[deepspeed]==0.18.0'

uv pip install -e .
```

Install the official CUDA 12.8 FlashAttention-3 wheel published by PyTorch:

```bash
uv pip install \
  'flash-attn-3==3.0.0' \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the Qwen3.5 Gated DeltaNet dependencies. Transformers requires both Flash Linear Attention and causal-conv1d for its Qwen3.5 fast path:

```bash
uv pip install 'flash-linear-attention==0.4.1'

uv pip install \
  'https://huggingface.co/datasets/dudeperf3ct/qwen35-sft-wheels/resolve/main/causal-conv1d/py312-torch211-cu128/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl'
```

> [!IMPORTANT]
> This `causal-conv1d` wheel is built specifically for CPython 3.12, Linux
> x86_64, PyTorch 2.11 with CUDA 12.8, and NVIDIA H100 (`sm_90`). Do not reuse
> it with a different Python, PyTorch, CUDA, operating-system, CPU, or GPU
> architecture combination.

Install the Cut Cross Entropy dependency required by the Axolotl plugin:

```bash
uv pip uninstall cut-cross-entropy

uv pip install \
  'cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@5f0c7a7'
```

Verify the training dependencies:

```bash
python - <<'PY'
from importlib.metadata import version

import flash_attn_interface
import torch
import transformers

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name())
print("BF16:", torch.cuda.is_bf16_supported())
print("axolotl:", version("axolotl"))
print("transformers:", transformers.__version__)
print("flash-attn-3:", version("flash-attn-3"))
print("flash-linear-attention:", version("flash-linear-attention"))
print("causal-conv1d:", version("causal-conv1d"))
print("cut-cross-entropy:", version("cut-cross-entropy"))
PY
```

Expected versions:

```text
torch: 2.11.0+cu128
torch CUDA: 12.8
GPU: NVIDIA H100 80GB HBM3
BF16: True
axolotl: 0.18.0
transformers: 5.14.1
flash-attn-3: 3.0.0
flash-linear-attention: 0.4.1
causal-conv1d: 1.6.2.post1
cut-cross-entropy: 25.5.2
```

All four experiment configurations use `attn_implementation:
flash_attention_3`. FlashAttention-3 benchmarks show a `1.5-2.0x` forward and `1.5-1.75x` backward attention-kernel speedup over FlashAttention-2 on an H100 SXM5. Qwen3.5 also contains Gated DeltaNet layers, so the 1K pilot must record the actual end-to-end training throughput rather than assuming the full kernel-level speedup.

See the [Axolotl attention documentation](https://docs.axolotl.ai/docs/attention.html#flash-attention-3) and the [FlashAttention-3 paper](https://tridao.me/publications/flash3/flash3.pdf).

Flash Linear Attention `0.4.1`, installed with Axolotl, accelerates Qwen3.5's Gated DeltaNet layers and is separate from FlashAttention-3. Sample packing remains disabled. DeepSpeed is installed through Axolotl's documented extra but is not enabled because every 4B pilot fit on one 80 GB H100.

> [!NOTE]
> The dataset repository and immutable revision are documented in [`data/README.md`](data/README.md).

## Audit Chat Labels

The audit configurations are preprocessing-only inputs. They read the `overfit` split from the prepared Hub dataset, use the official `qwen3_5` template, train only assistant turns, split reasoning targets into Qwen's `reasoning_content`, and explicitly identify `<|im_end|>` as the turn terminator. Direct and reasoning use separate prepared-data paths.

From the H100 environment, run:

```bash
mkdir -p logs/label-audit
set -o pipefail

(
  axolotl preprocess configs/audit/direct.yml \
    --debug \
    --debug-num-examples 5 \
  && python scripts/audit_labels.py configs/audit/direct.yml
) 2>&1 | tee logs/label-audit/direct.log

(
  axolotl preprocess configs/audit/reasoning.yml \
    --debug \
    --debug-num-examples 5 \
  && python scripts/audit_labels.py configs/audit/reasoning.yml
) 2>&1 | tee logs/label-audit/reasoning.log
```

The auditor matches prepared rows to source conversations by their rendered token sequence, so Axolotl's preprocessing order does not affect the audit. It checks every prepared row, records safe summaries for five examples by default, and fails on template, masking, target, reasoning, length, or EOT drift. It writes reports under `reports/label-audit/`.

The pinned official template renders direct assistant turns as `<think>\n\n</think>\n\n<code>` even though the direct source target contains only `r1_solution`. Axolotl's assistant-turn boundary treats that empty block as template text, so its tokens remain masked while the code and assistant EOT are trainable. The audit requires this exact behavior rather than replacing the official template.

The auditor also preserves trailing newlines in assistant content. This follows Axolotl's content-boundary rule that newlines stay with the preceding content rather than being discarded by template whitespace trimming.

Axolotl's dataset and CLI behavior used here is documented in its
[dataset-format guide](https://docs.axolotl.ai/docs/dataset-formats/index.html),
[conversation guide](https://docs.axolotl.ai/docs/dataset-formats/conversation.html),
and [CLI arguments](https://docs.axolotl.ai/docs/api/cli.args.html).

## Staged Training Configurations

The configuration layout mirrors the experiment stages:

```text
configs/
├── audit/
│   ├── direct.yml
│   └── reasoning.yml
├── overfit/
│   ├── direct-lora.yml
│   ├── reasoning-lora.yml
│   └── direct-fft.yml
├── pilot/
│   ├── direct-lora.yml
│   ├── reasoning-lora.yml
│   ├── direct-fft.yml
│   └── reasoning-fft.yml
├── direct-lora.yml
├── reasoning-lora.yml
├── direct-fft.yml
└── reasoning-fft.yml
```

The overfit LoRA runs use the 32-example `overfit` split for 100 optimizer steps with gradient accumulation disabled. The direct full-FT smoke test runs for five optimizer steps. Run the following steps in parallel:

```bash
mkdir -p logs/overfit
set -o pipefail

axolotl train configs/overfit/direct-lora.yml \
  2>&1 | tee logs/overfit/direct-lora.log

axolotl train configs/overfit/reasoning-lora.yml \
  2>&1 | tee logs/overfit/reasoning-lora.log

axolotl train configs/overfit/direct-fft.yml \
  2>&1 | tee logs/overfit/direct-fft.log
```

After reviewing overfitting phase, run the four one-epoch 1K pilots:

```bash
mkdir -p logs/pilot

axolotl train configs/pilot/direct-lora.yml \
  2>&1 | tee logs/pilot/direct-lora.log

axolotl train configs/pilot/reasoning-lora.yml \
  2>&1 | tee logs/pilot/reasoning-lora.log

axolotl train configs/pilot/direct-fft.yml \
  2>&1 | tee logs/pilot/direct-fft.log

axolotl train configs/pilot/reasoning-fft.yml \
  2>&1 | tee logs/pilot/reasoning-fft.log
```

Every stage has distinct prepared-data, output, and W&B run paths. The root configuration files remain the two-epoch 10K main experiment.

## Main 10K Configurations

> [!NOTE]
> The dataset repository and immutable revision are documented in [`data/README.md`](data/README.md).

The W&B entity/project is `dudeperf3ct/qwen35-4b-kodcode-sft`.

The four explicit configurations are:

- `configs/direct-lora.yml`
- `configs/reasoning-lora.yml`
- `configs/direct-fft.yml`
- `configs/reasoning-fft.yml`

Each configuration reads `train` and `validation` directly from the immutable dataset revision. Prepared datasets, outputs, W&B run names, and Hub branches
are unique.

All checkpoints are pushed to the `dudeperf3ct/qwen35-4b-kodcode-sft-10k` model repository.

Within either training method, direct and reasoning differ only in target selection and run identity:

```diff
- name: direct
- split_thinking: false
+ name: reasoning
+ split_thinking: true

- dataset_prepared_path: ./prepared/p2-10k-direct-<method>-s42
- output_dir: ./outputs/p2-10k-direct-<method>-s42
- hub_revision: direct-<method>
- wandb_name: main-10k-direct-<method>-seed42
- wandb_run_id: main-10k-direct-<method>-seed42
+ dataset_prepared_path: ./prepared/p2-10k-reasoning-<method>-s42
+ output_dir: ./outputs/p2-10k-reasoning-<method>-s42
+ hub_revision: reasoning-<method>
+ wandb_name: main-10k-reasoning-<method>-seed42
+ wandb_run_id: main-10k-reasoning-<method>-seed42
```

For either target, LoRA and language-model full FT differ only in method
settings and run identity:

```diff
- learning_rate: 0.0001
- adapter: lora
- lora_r: 64
- lora_alpha: 128
- lora_dropout: 0.05
- lora_target_modules: <language-model-only regex>
- micro_batch_size: 4
- gradient_accumulation_steps: 2
+ learning_rate: 0.00001
+ unfrozen_parameters:
+   - model.language_model.*
+   - lm_head.*
+ micro_batch_size: 2
+ gradient_accumulation_steps: 4

- dataset_prepared_path: ./prepared/p2-10k-<target>-lora-s42
- output_dir: ./outputs/p2-10k-<target>-lora-s42
- wandb_name: main-10k-<target>-lora-seed42
- wandb_run_id: main-10k-<target>-lora-seed42
+ dataset_prepared_path: ./prepared/p2-10k-<target>-fft-s42
+ output_dir: ./outputs/p2-10k-<target>-fft-s42
+ wandb_name: main-10k-<target>-fft-seed42
+ wandb_run_id: main-10k-<target>-fft-seed42
```

## Run 10K on Two H100s

Run one experiment at a time across both GPUs with DDP when clean throughput comparisons and predictable recovery matter. Axolotl uses DDP by default when neither DeepSpeed nor FSDP is configured. The current configs retain a global batch size of 16:

| Run | Microbatch/GPU | Accumulation | GPUs | Global batch | Peak active/GPU | Train runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct LoRA | 8 | 1 | 2 | 16 | 18.16 GiB | 2h 28m 23s |
| Direct full FT | 8 | 1 | 2 | 16 | 44.96 GiB | 2h 24m 29s |
| Reasoning LoRA | 4 | 2 | 2 | 16 | Not measured | Not measured |
| Reasoning full FT | 4 | 2 | 2 | 16 | Not measured | Not measured |

The direct figures are observed two-H100 results, not ideal-scaling estimates. Those two jobs overlapped on the same instance: elapsed time from the first trainer start to the final save was 2h 30m 39s, costing about $21.04 at $8.38/hour. Summing their individual runtimes gives 4h 52m 52s, or about $40.90, as a rough sequential planning value.

```bash
mkdir -p logs/main

axolotl train configs/direct-lora.yml --launcher torchrun -- --nproc_per_node=2 --nnodes=1 2>&1 | tee logs/main/direct-lora.log

axolotl train configs/direct-fft.yml \
  --launcher torchrun \
  -- \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  --nproc_per_node=2 \
  --nnodes=1 \
  2>&1 | tee logs/main/direct-fft.log

axolotl train configs/reasoning-lora.yml \
  --launcher torchrun \
  -- \
  --rdzv-backend=c10d \
  --rdzv-endpoint=localhost:0 \
  --nproc_per_node=2 \
  --nnodes=1 \
  2>&1 | tee logs/main/reasoning-lora.log
```

> [!NOTE]
> See Axolotl's [multi-GPU guide](https://docs.axolotl.ai/docs/multi-gpu.html), [CLI launcher documentation](https://docs.axolotl.ai/docs/cli.html), and the Transformers [effective batch-size definition](https://huggingface.co/docs/transformers/main_classes/trainer).

## Held-Out Test Evaluation

Run the untouched 500-example test split only after the four 10K runs and all training decisions are final. The test evaluator uses the same greedy decoding, 16,384-token generation limit, Qwen3.5 chat template, and private tests for all four checkpoints. It loads the immutable dataset revision from `manifests/evaluation.json`.

Direct requests set `enable_thinking: false`; reasoning requests set it to `true`.

The editable project install includes the pinned `pytest` runner used for private tests.

Create a dedicated held-out inference environment. Qwen3.5 requires the current
vLLM nightly line, and a clean environment lets `uv` install a mutually
compatible vLLM, PyTorch, and CUDA runtime without changing the Axolotl training
environment:

```bash
uv venv .venv-vllm --python 3.12 --seed
source .venv-vllm/bin/activate

uv pip install --upgrade vllm \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/nightly
```

Verify the held-out inference environment before serving a checkpoint:

```bash
python - <<'PY'
import torch
import vllm

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("vLLM:", vllm.__version__)
print("GPU:", torch.cuda.get_device_name())
PY
```

Observed output on the held-out H100 evaluation machine:

```text
torch: 2.13.0+cu132
torch CUDA: 13.2
vLLM: 0.26.1rc1.dev278+g5df9999fc
GPU: NVIDIA H100 80GB HBM3
```

Merge both LoRA adapters before evaluation:

```bash
axolotl merge-lora configs/direct-lora.yml \
  --lora-model-dir ./outputs/p2-10k-direct-lora-s42

axolotl merge-lora configs/reasoning-lora.yml \
  --lora-model-dir ./outputs/p2-10k-reasoning-lora-s42
```

Serve one checkpoint at a time from the separate vLLM environment. For example, the direct LoRA checkpoint is:

```bash
.venv-vllm/bin/vllm serve \
  outputs/p2-10k-direct-lora-s42/merged \
  --served-model-name main-10k-direct-lora-seed42 \
  --reasoning-parser qwen3 \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.80
```

In another shell, activate the Axolotl environment and run:

```bash
python scripts/evaluate_test.py \
  --config configs/direct-lora.yml
```

Stop the server, serve the next checkpoint, and use the matching evaluation command:

| Run | Checkpoint | Evaluation arguments |
| --- | --- | --- |
| Direct LoRA | `outputs/p2-10k-direct-lora-s42/merged` | `--config configs/direct-lora.yml` |
| Reasoning LoRA | `outputs/p2-10k-reasoning-lora-s42/merged` | `--config configs/reasoning-lora.yml` |
| Direct full FT | `outputs/p2-10k-direct-fft-s42` | `--config configs/direct-fft.yml` |
| Reasoning full FT | `outputs/p2-10k-reasoning-fft-s42` | `--config configs/reasoning-fft.yml` |

Each run writes resumable per-example results to `reports/test/<wandb-run-id>/results.jsonl` and aggregate metrics to `reports/test/<wandb-run-id>/summary.json`. The summary includes pass rate overall and by difficulty/subset/style, average output tokens, thinking and final-code rates, truncations, extraction failures, syntax errors, test timeouts, and API errors.

Rows that ended with `api_error` are retried on the next invocation; completed model outputs and test results are reused.

Once evaluation completes, the same script reopens the completed W&B
run record by its explicit `wandb_run_id`. It adds the test summary metrics and uploads both result files as an `evaluation` artifact.

> [!WARNING]
> The evaluator executes model-generated Python. Run it only inside the
> disposable experiment VM, never on a workstation containing credentials or important files.

vLLM documents request-level `chat_template_kwargs` for controlling Qwen thinking mode in its [reasoning-output guide](https://docs.vllm.ai/en/stable/features/reasoning_outputs/).
