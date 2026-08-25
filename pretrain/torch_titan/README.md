# Pretraining Code LLMs with TorchTitan

Write up: https://dudeperf3ct.github.io/projects/train_llm_part2/

Training an LLM from scratch using the custom tokenizer and dataset prepared in previous steps.

Custom tokenizer: https://dudeperf3ct.github.io/projects/train_llm_part1/

Dataset: [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2)

Model Architecture: Llama 3.2 1B (1 billion parameter)

## Dependencies Setup

Once instance is up, install the PyTorch nightly build with CUDA 12.8 support in a new virtual environment:

```bash
uv venv
source .venv/bin/activate
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
uv pip install transformers tokenizers datasets huggingface_hub wandb
```

Clone the `torchtitan` repository and install the project dependencies.

```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan
uv pip install -r pyproject.toml
uv pip install -e .
```



Create [access token](https://huggingface.co/docs/hub/en/security-tokens) in Hugging Face and login using the `huggingface-cli` tool.

```bash
hf auth login
```

Similarly, set wandb API token using `wandb` CLI using following command.

```bash
wandb login
```

## Debugging

> [!WARNING]
> I am using Lambda Labs GPU instance with 1xH100 GPU (80 GB SMX5) for debugging. It costs about **$2** to run these.

Make a copy of the training configuration files and local overrides to torchtitan's directory.

```bash
cp -r ../train_configs ./train_configs
```

### Memory Estimation

Next, estimate the memory requirements for the [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model

> [!NOTE]
> Llama 3.2 1B is a gated model and requires access. Log in to Hugging Face and request access to the model.

```bash
NGPU=1 CONFIG_FILE='./train_configs/debug_llama32_1b.toml' ./scripts/estimate/run_memory_estimation.sh
```

### Communication Mode for debugging

TorchTitan offers two "fake" communication model: `fake_backend` and `local_tensor`. These modes allow you to dry run distributed training code on a single GPU without needing multiple GPUs or a distributed setup.

- `fake_backend`: Simulates a distributed communication without actual data transfer.

```bash
NGPU=8 COMM_MODE='fake_backend' CONFIG_FILE='./train_configs/debug_llama32_1b.toml' ./run_train.sh
```

- `local_tensor`: Simulates the full distributed training workflow on a single GPU by executing all communication and computation locally

```bash
NGPU=16 COMM_MODE="local_tensor" ./run_train.sh \
  --parallelism.tensor_parallel_degree 8 \
  --parallelism.data_parallel_shard_degree 2
```

>[!NOTE]
> The `local_tensor` mode did not work for me on a single H100 GPU instance.

## Custom dataset + tokenizer (SwallowCode + 32k) + Llama 3.2 1B model architecture

### Setup

Make a copy of the training configuration files and local overrides to torchtitan's directory.

```bash
cp -r ../train_configs ./train_configs
cp ../custom_spec.py ./torchtitan/custom_spec.py
cp ../dataset/text_datasets.py ./torchtitan/hf_datasets/text_datasets.py
```

This setup uses:
- SwallowCode v2 dataset (with FIM formatting) wired into `torchtitan/hf_datasets/text_datasets.py`.
- A custom 32k tokenizer loaded from a local HF snapshot.
- A custom train spec (`custom_spec.py`) that overrides vocab size to 32,768.
- Llama 3.2 1B model architecture with custom tokenizer and dataset.

Download the tokenizer assets from the Hub into the location referenced by the config:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dudeperf3ct/codellm-tokenizer",
    local_dir="hf_assets/tokenizer",
    local_dir_use_symlinks=False,
)
PY
```

### Smoke Test

> [!WARNING]
> I used 2 x H100 GPUs (80 GB SMX5) for smoke testing. It takes about **$8** to run this smoke testing. It logs the results of the experiments to wandb.

>[!IMPORTANT]
> Change `data_parallel_replicate_degree` to number of GPUs used for training in the config file for smoke testing.

Run training with the custom config as smoke test:

```bash
NGPU=2 CONFIG_FILE='./train_configs/smoke_llama32_1b_swallowcode_tok32k.toml' ./run_train.sh
```

This runs for 1000 steps. It takes about 15 minutes to complete.

Smoke test results (2x H100, 1k steps, from W&B):
- Completed 1k steps / 98.3M tokens in ~15.4 minutes.
- Loss fell from 10.88 -> 4.69; LR warmed from 3e-7 to 3e-4 by step 1k (warmup covered the full run).
- Step time stabilized around ~0.90s; per-rank throughput ~54-55k tps (global ~109k), MFU ~41% (~405 TFLOPS).
- Data loading was negligible (~0.03% of step time) and memory stayed flat at ~70.2 GiB active / 72.8 GiB reserved, with 0 OOMs or alloc retries.
- Token budget: `steps * global_batch_size * seq_len`. With `local_batch_size=6`, `NGPU=2`, `global_batch_size=12`, `seq_len=8192` -> 98,304 tokens/step (98.3M at 1k steps).

> [!NOTE]
> W&B plots for experiment: [Plots](https://wandb.ai/dudeperf3ct/torchtitan/groups/Smoke%20run%20-%2098M%20tokens/workspace?nw=nwuserdudeperf3ct), [Logs](https://wandb.ai/dudeperf3ct/torchtitan/groups/Smoke%20run%20-%2098M%20tokens/logs), [Summary](https://wandb.ai/dudeperf3ct/torchtitan/groups/Smoke%20run%20-%2098M%20tokens/overview) and [Report](https://wandb.ai/dudeperf3ct/torchtitan/reports/Pretraining-LLM-experiment--VmlldzoxNTU4NTA1NQ)


### Full Training

>[!WARNING]
> I ran the following on 4 x H100 which costs $12.36/hr. It will cost about **$150** for running this setup.

>[!IMPORTANT]
> `data_parallel_replicate_degree` is set to 4 in the config. Change it if you use a different GPU count.

For a full run on 4 x H100 GPUs (80 GB SMX5):

```bash
NGPU=4 CONFIG_FILE='./train_configs/full_llama32_1b_swallowcode_tok32k.toml' ./run_train.sh
```

Insights for full training from smoke testing (for 4x H100):
- `tokens = steps * global_batch_size * seq_len`
- With `local_batch_size=6`, `NGPU=4`, `seq_len=8192`: `global_batch_size=24`
- With `steps=50000`: `tokens ~ 24 * 8192 * 50000 ~ 9.83B`
- Using smoke-test throughput (~55k tokens/sec/GPU on 2x H100), estimate step time as `(global_batch_size * seq_len) / (tps_per_gpu * NGPU)` -> ~0.9s/step on 4 GPUs
- That puts 50k steps at ~ 12.5 hours (~$144 at $12.36/hr)
- Compute-optimal for a 1B model is ~20B tokens (Chinchilla), so this run is still undertrained.

> [!NOTE]
> W&B plots for experiment: [Plots](https://wandb.ai/dudeperf3ct/torchtitan/groups/Full%20run%20-%209.8B%20tokens/workspace), [Logs](https://wandb.ai/dudeperf3ct/torchtitan/groups/Full%20run%20-%209.8B%20tokens/logs), [Summary](https://wandb.ai/dudeperf3ct/torchtitan/groups/Full%20run%20-%209.8B%20tokens/overview) and [Report](https://wandb.ai/dudeperf3ct/torchtitan/reports/Pretraining-LLM-experiment--VmlldzoxNTU4NTA1NQ)

Full training results (4x H100, 50k steps, from W&B):
- Completed 50k steps / 9.83B tokens in ~ 12.26 hours (~$152 at $12.36/hr).
- Loss fell from 10.86 -> 2.91; LR warmed up to 3e-4 by step 800 then cosine-decayed to ~0 by the end.
- Step time stabilized around ~0.88s; per-rank throughput ~55.8k tps (global ~223k), MFU ~42% (~416 TFLOPS).
- Data loading was negligible (~0.03% of step time) and memory stayed flat at ~70.2 GiB active / 70.9 GiB reserved (88-89% of 80GB), with 0 OOMs or alloc retries.

## Evaluation

Setup the environment by running `uv sync --extra cuda`. Make sure DCP checkpoint (torchtitan specific format) are stored under `checkpoint` folder.

> [!NOTE]
> Uploaded DCP checkpoints to Hugging Face model: https://huggingface.co/dudeperf3ct/codellm_pretrain

Use `eval/eval_generate.py` to run inference against a DCP checkpoint. Provide the same config used for training and a checkpoint directory. Run from this directory so relative paths resolve.

Evaluation modes:
- LM (standard completion): no FIM tokens, use `--prompt` or a JSONL with `{prompt}` fields.
- FIM (infilling): inserts `<|fim_prefix|>`, `<|fim_suffix|>`, `<|fim_middle|>` and asks the model to generate the missing middle.

```bash
python eval/eval_generate.py \
  --config ./train_configs/full_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/<step_dir> \
  --samples ./eval/eval_samples.jsonl \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 50 \
  --stop_at_eos \
  --custom_import custom_spec
```

- `eval/eval_samples.jsonl` contains FIM examples (`psm`), not LM prompts. For LM evaluation, use `--prompt` or a separate JSONL with `{name, prompt}`.
- CLI `--fim_prefix/--fim_suffix` interpret `\n`, `\t`, `\r` by default; pass `--no_interpret_escapes` to keep them literal.
- Use `--out` to print JSON, or `--show_raw` to include raw decoded text in logs.
- `--mode` can be `auto` (default), `fim`, or `lm`; `auto` infers the mode from the inputs.
- `hf_assets/` stores the local tokenizer snapshot referenced by `hf_assets_path` in the config; it is required for eval with the current configs
- If outputs show extra spaces around punctuation/underscores, that is typically the model's decoded output from a byte-level tokenizer. Use `--show_raw` to inspect the raw decoded text.

LM (standard completion) example:

```bash
python eval/eval_generate.py \
  --config ./train_configs/full_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/<step_dir> \
  --prompt "def count_vowels(s):\n    \"\"\"Count vowels in a string.\"\"\"\n    vowels = set(\"aeiouAEIOU\")\n" \
  --mode lm \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 50 \
  --stop_at_eos \
  --custom_import custom_spec
```

Single-prompt FIM examples:

PSM (prefix-suffix-middle):

```bash
python eval/eval_generate.py \
  --config ./train_configs/full_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/<step_dir> \
  --fim_prefix "def count_vowels(s):\n    \"\"\"Count vowels in a string.\"\"\"\n    vowels = set(\"aeiouAEIOU\")\n" \
  --fim_suffix "\n    return count\n" \
  --fim_format psm \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 50 \
  --stop_at_eos \
  --custom_import custom_spec
```

Expected: the completion should include a loop over `s` and increment a `count` when a character is in `vowels` (e.g., `count = 0`, `for ch in s:`, `if ch in vowels: count += 1`), then stop near EOS.

SPM (suffix-prefix-middle):

```bash
python eval/eval_generate.py \
  --config ./train_configs/full_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/<step_dir> \
  --fim_prefix "def count_vowels(s):\n    \"\"\"Count vowels in a string.\"\"\"\n    vowels = set(\"aeiouAEIOU\")\n" \
  --fim_suffix "\n    return count\n" \
  --fim_format spm \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 50 \
  --stop_at_eos \
  --custom_import custom_spec
```

Expected: SPM is harder because the suffix comes first, so outputs may be less coherent than PSM. You should still see code-like tokens that fit between the prefix and suffix, but it may take more tokens or look noisier.

Creating `eval/eval_samples.jsonl`:
- One JSON object per line with `{name, fim_prefix, fim_suffix, fim_format}`.
- Use `fim_suffix: ""` to turn a prefix prompt into a FIM entry.
- Pick `fim_format` per sample (the current file uses `psm`).


Example evaluation run

<details>
<summary> Evaluation run on last checkpoint (50k step) full run </summary>

Using LM mode to predict next token as completion task

```bash
python eval/eval_generate.py \
  --config ./train_configs/full_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/step-50000 \
  --prompt "def count_vowels(s):\n    \"\"\"Count vowels in a string.\"\"\"\n    vowels = set(\"aeiouAEIOU\")\n" \
  --mode lm \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 50 \
  --stop_at_eos \
  --custom_import custom_spec
[titan] 2026-01-10 22:03:55,522 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-10 22:03:55,791 - root - INFO - Applying Llama-like patch for Llama
[titan] 2026-01-10 22:04:08,618 - root - INFO - Loading checkpoint: ./checkpoint/step-50000
/home/dudeperf3ct/projects/mini-codellm/pretrain/torch_titan/.venv/lib/python3.12/site-packages/torch/distributed/checkpoint/utils.py:483: UserWarning: torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.
  return func(*args, **kwargs)
[titan] 2026-01-10 22:04:10,153 - root - INFO - Checkpoint loaded in 1.53 seconds
[titan] 2026-01-10 22:05:15,528 - root - INFO - [prompt] prompt:
def count_vowels(s):
    """Count vowels in a string."""
    vowels = set("aeiouAEIOU")

[titan] 2026-01-10 22:05:15,529 - root - INFO - [prompt] completion:
#
 def print _ from _ with _ with : Tuple _ name == "__ _ path _ title : ") : List and _ on __
    : Path : List of 2 . append
 def count : List [ str ( B :
     start _ path : List [ str , str = [
     root


 def
[titan] 2026-01-10 22:05:15,529 - root - INFO - [prompt] tokens: prompt=27 completion=64 time=65.37s
```

Greedy decoding with `temperature=0` and `top_k=1`

```bash
python eval/eval_generate.py \
  --config ./train_configs/full_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/step-50000 \
  --prompt "def count_vowels(s):\n    \"\"\"Count vowels in a string.\"\"\"\n    vowels = set(\"aeiouAEIOU\")\n" \
  --mode lm \
  --max_new_tokens 64 \
  --temperature 0 \
  --top_k 1 \
  --stop_at_eos \
  --custom_import custom_spec

[titan] 2026-01-10 22:07:58,523 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-01-10 22:07:58,787 - root - INFO - Applying Llama-like patch for Llama
[titan] 2026-01-10 22:08:10,971 - root - INFO - Loading checkpoint: ./checkpoint/step-50000
/home/dudeperf3ct/projects/mini-codellm/pretrain/torch_titan/.venv/lib/python3.12/site-packages/torch/distributed/checkpoint/utils.py:483: UserWarning: torch.distributed is disabled, unavailable or uninitialized, assuming the intent is to load in a single process.
  return func(*args, **kwargs)
[titan] 2026-01-10 22:08:12,580 - root - INFO - Checkpoint loaded in 1.61 seconds
[titan] 2026-01-10 22:09:13,698 - root - INFO - [prompt] prompt:
def count_vowels(s):
    """Count vowels in a string."""
    vowels = set("aeiouAEIOU")

[titan] 2026-01-10 22:09:13,699 - root - INFO - [prompt] completion:

 def print ( f . get _ to _ to _ to _ to _ to _ to _ to _ to _ to _ to _ to _ to - 8 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
[titan] 2026-01-10 22:09:13,699 - root - INFO - [prompt] tokens: prompt=27 completion=64 time=61.12
```

</details>
