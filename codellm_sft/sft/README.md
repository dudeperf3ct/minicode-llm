# Qwen3.5-4B KodCode SFT

This directory contains the Axolotl implementation for the staged Qwen3.5-4B coding SFT experiment.

The reusable matched-data pipeline produces the 10,000-example train split, 500-example validation split, 500-example untouched test split, and nested 1,000/32 training subsets needed by later phases.

Chat-label audits and the four explicit 10K Axolotl configurations are implemented.

## Pinned Components

| Component | Version or identifier |
| --- | --- |
| Python | 3.12 |
| Axolotl | `0.18.0` |
| Transformers | `5.14.1` |
| Cut Cross Entropy | Axolotl fork commit `5f0c7a7` |
| Base model and tokenizer | `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b` |
| KodCode source | `KodCode/KodCode-V1-SFT-R1@26c8a11c800d71b6c4bafe12a92c9090ef0b6214` |
| HumanEval | `openai/openai_humaneval@7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` |
| MBPP | `google-research-datasets/mbpp@4bb6404fdc6cacfda99d4ac4205087b89d32030c` |
| LiveCodeBench | `livecodebench/code_generation_lite@0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, `release_v5` |

All revisions, including the materialized LiveCodeBench prompt mirror and Open-R1 reference script, are recorded in `manifests/revisions.json`.

## Local Data-Preparation Environment

From this directory:

```bash
uv sync --group dev
```

Run a basic import check:

```bash
uv run python -c \
  "import datasets, transformers, yaml; print(datasets.__version__, transformers.__version__)"
```

## Prepare Matched Data

The script downloads only the pinned KodCode `default/train` split and never loads the published `incorrect` or `use_with_caution` splits.

```bash
uv run python scripts/prepare_data.py
```

Regeneration is explicit:

```bash
uv run python scripts/prepare_data.py --overwrite
```

The pipeline performs source validation, normalized-question deduplication, Open-R1-style word 8-gram decontamination, Qwen3.5 chat-template length filtering, joint-stratum largest-remainder sampling, nested subset selection, and fail-fast split assertions. Generated JSONL and statistics live under `data/`; the trackable checksum manifest is `manifests/data_manifest.json`.

The pinned run produced:

- 268,211 source train rows;
- 17 normalized-question duplicates and 5 malformed reasoning rows removed;
- 26,906 rows removed by benchmark decontamination;
- 708 reasoning examples removed above 16,384 tokens;
- 240,575 eligible rows before stratified selection;
- exact 10,000/500/500 train, validation, and untouched test splits.

On the selected training split, reasoning assistant turns contain 26.63 times as many tokens as direct assistant turns. The reasoning total-length p99 is 14,340 tokens.

LiveCodeBench is still pinned to the official dataset revision. Because its official Hugging Face dataset uses a remote loader script, the pipeline column-reads only `question_id` and `question_content` from the separately pinned `lighteval/code_generation_lite` `release_v5` Parquet mirror. The normalized 880-prompt snapshot hash is recorded in the output statistics.

## Code Layout

- `scripts/prepare_data.py` is the thin command orchestration layer.
- `scripts/data_pipeline.py` handles verified-source filtering, length filtering, stratified selection, and split validation.
- `scripts/data_writer.py` owns deterministic JSON/JSONL output, paired rendering, and token measurement.
- `scripts/decontaminate.py` owns pinned benchmark loading and the reusable `NgramDecontaminator`.
- `scripts/data_statistics.py` owns token and stratum aggregation.
- `scripts/pipeline_utils.py` owns shared models, normalization, hashing, batching, and tokenizer helpers.
- `scripts/pipeline_reports.py` owns ordered-ID, allocation, statistics, and checksum manifests.
- `scripts/upload_dataset.py` verifies and uploads the reusable training payload.

## Publish Prepared Training Data

Hugging Face supports uploading a folder directly to a dataset repository revision. The repository must already exist, and local Hugging Face authentication must have write access.

For authentication, either run `hf auth login` or provide a User Access Token through `HF_TOKEN`. A fine-grained token with write access limited to this dataset repository is preferred.

Verify the active identity without printing the token:

```bash
uv run hf auth whoami
```

First verify the exact payload without changing the Hub:

```bash
uv run python scripts/upload_dataset.py \
  --repo-id dudeperf3ct/qwen35-kodcode-sft-data \
  --dry-run
```

Then create the branch and upload:

```bash
HF_XET_HIGH_PERFORMANCE=1 \
uv run python scripts/upload_dataset.py \
  --repo-id dudeperf3ct/qwen35-kodcode-sft-data
```

The uploader targets `main` by default, verifies local checksums before any remote mutation, and prints the resulting commit SHA. Pin that SHA in every Axolotl configuration before preprocessing or training.

The upload contains two loadable configurations:

```python
from datasets import load_dataset

direct = load_dataset(
    "dudeperf3ct/qwen35-kodcode-sft-data",
    "direct",
    revision="db5f9912d76642bac325bea2bb41d83ad186365b",
)
reasoning = load_dataset(
    "dudeperf3ct/qwen35-kodcode-sft-data",
    "reasoning",
    revision="db5f9912d76642bac325bea2bb41d83ad186365b",
)
```

Each configuration exposes `train`, `validation`, `pilot`, and `overfit`.

## H100 Training Environment

The Lambda Labs `1 x H100` machine configuration is the following:

- NVIDIA H100 80 GB HBM3 SXM5;
- 26 vCPUs and 225 GiB RAM;
- 2.8 TiB SSD;
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

uv pip install \
  'transformers==5.14.1' \
  'datasets==4.4.1' \
  'huggingface-hub==1.23.0'
```

Install the official CUDA 12.8 FlashAttention-3 wheel published by PyTorch:

```bash
uv pip install \
  'flash-attn-3==3.0.0' \
  --index-url https://download.pytorch.org/whl/cu128
```

Install the Qwen3.5 Gated DeltaNet dependencies. Transformers requires both
Flash Linear Attention and causal-conv1d for its Qwen3.5 fast path:

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

Flash Linear Attention `0.4.1`, installed with Axolotl, accelerates Qwen3.5's Gated DeltaNet layers and is separate from FlashAttention-3. Sample packing remains disabled. DeepSpeed is installed through Axolotl's documented extra but will not be enabled for the single-H100 runs.


## Audit Chat Labels

The audit configurations are preprocessing-only inputs. They read the
`overfit` split from the prepared Hub dataset, use the official `qwen3_5` template, train only assistant turns, split reasoning targets into Qwen's `reasoning_content`, and explicitly identify `<|im_end|>` as the turn terminator. Direct and reasoning use separate prepared-data paths.

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

The overfit LoRA runs use the 32-example `overfit` split for 100 optimizer
steps with gradient accumulation disabled. The direct full-FT smoke test runs
for five optimizer steps. Run them sequentially:

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

After the Phase 0 acceptance checks pass, run the four one-epoch 1K pilots:

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

Every stage has distinct prepared-data, output, and W&B run paths. The root
configuration files remain the two-epoch 10K main experiment.

## Main 10K Configurations

The experiment identity is:

- Hugging Face dataset: `dudeperf3ct/qwen35-kodcode-sft-data`
- Dataset branch: `main`
- Dataset revision: `db5f9912d76642bac325bea2bb41d83ad186365b`
- W&B entity/project: `dudeperf3ct/qwen35-4b-kodcode-sft`

The four explicit configurations are:

- `configs/direct-lora.yml`
- `configs/reasoning-lora.yml`
- `configs/direct-fft.yml`
- `configs/reasoning-fft.yml`

Each configuration reads `train` and `validation` directly from the immutable dataset revision. Prepared datasets, outputs, and W&B run names are unique. Hub model upload remains disabled during training.

Within either training method, direct and reasoning differ only in target selection and run identity:

```diff
- name: direct
- split_thinking: false
+ name: reasoning
+ split_thinking: true

- dataset_prepared_path: ./prepared/p2-10k-direct-<method>-s42
- output_dir: ./outputs/p2-10k-direct-<method>-s42
- wandb_name: main-10k-direct-<method>-seed42
+ dataset_prepared_path: ./prepared/p2-10k-reasoning-<method>-s42
+ output_dir: ./outputs/p2-10k-reasoning-<method>-s42
+ wandb_name: main-10k-reasoning-<method>-seed42
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
+ learning_rate: 0.00001
+ unfrozen_parameters:
+   - model.language_model.*
+   - lm_head.*

- dataset_prepared_path: ./prepared/p2-10k-<target>-lora-s42
- output_dir: ./outputs/p2-10k-<target>-lora-s42
- wandb_name: main-10k-<target>-lora-seed42
+ dataset_prepared_path: ./prepared/p2-10k-<target>-fft-s42
+ output_dir: ./outputs/p2-10k-<target>-fft-s42
+ wandb_name: main-10k-<target>-fft-seed42
```

Verified final weights will later use:

- `dudeperf3ct/qwen35-4b-base-kodcode-10k-direct-lora`
- `dudeperf3ct/qwen35-4b-base-kodcode-10k-reasoning-lora`
- `dudeperf3ct/qwen35-4b-base-kodcode-10k-direct-fft`
- `dudeperf3ct/qwen35-4b-base-kodcode-10k-reasoning-fft`
