# Pretraining Code LLMs with TorchTitan

Write up:

Training an LLM from scratch using the custom tokenizer and dataset prepared in previous steps.

Custom tokenizer: https://dudeperf3ct.github.io/projects/train_llm_part1/

Dataset: [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2)


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

Make a copy of the training configuration files and local overrides to torchtitan's directory.

```bash
cp -r ../train_configs ./train_configs
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

I am using Lambda Labs GPU instance with 1xH100 GPU (80 GB SMX5) for debugging. It costs about $2 to run these.

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

### Custom dataset + tokenizer (SwallowCode + 32k)

## Setup

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
    local_dir="hf_assets/codellm-tokenizer",
    local_dir_use_symlinks=False,
)
PY
```

## Smoke Test

I used 2 x H100 GPUs (80 GB SMX5) for smoke testing. It takes about $8 to run this smoke testing. It logs the run to wandb.

>[!IMPORTANT]
> Change `data_parallel_replicate_degree` to number of GPUs used for training from `1` in the config file.

Run training with the custom config as smoke test:

```bash
NGPU=2 CONFIG_FILE='./train_configs/smoke_llama32_1b_swallowcode_tok32k.toml' ./run_train.sh
```

This runs for 1000 steps. It takes about 15 minutes to complete.

Smoke test notes (2x H100, 1k steps):
- Loss dropped from ~10.9 to ~4.7; outputs will still look noisy/gibberish at this stage.
- Throughput stabilized around ~55k tokens/sec with ~41% MFU.
- Warmup covered the full run (1k warmup steps), so the LR never decayed.
- Token budget: `steps * global_batch_size * seq_len`. With `local_batch_size=6`, `NGPU=2`, `global_batch_size=12`, `seq_len=8192` -> ~98M tokens.

## Full Training

>[!WARNING]
> I ran the following on 4 x H100 which costs $12.36/hr. It will cost about $150 for running this setup.

For a full run on 4 x H100 GPUs (80 GB SMX5):

>[!IMPORTANT]
> `data_parallel_replicate_degree` is set to 4 in the config. Change it if you use a different GPU count.

```bash
NGPU=4 CONFIG_FILE='./train_configs/full_llama32_1b_swallowcode_tok32k.toml' ./run_train.sh
```

Insights for full training from smoke testing (example for 4x H100):
- `tokens = steps * global_batch_size * seq_len`
- With `local_batch_size=6`, `NGPU=4`, `seq_len=8192`: `global_batch_size=24`
- With `steps=40000`: `tokens ~ 24 * 8192 * 40000 ~ 7.86B`
- Using smoke-test throughput (~55k tokens/sec/GPU on 2x H100), estimate step time as `(global_batch_size * seq_len) / (tps_per_gpu * NGPU)` -> ~0.9s/step on 4 GPUs
- That puts 40k steps at ~10 hours (~$124 at $12.36/hr)
- Compute-optimal for a 1B model is ~20B tokens (Chinchilla), so this run is still undertrained

## Evaluation

Setup the environment by running `uv sync --extra cpu` or `uv sync --extra cuda` command for CPU and GPU system. Make sure DCP checkpoint are stored under `checkpoint` folder.

Use `eval/eval_generate.py` to run inference against a DCP checkpoint. Provide the same config used for training and a checkpoint directory. It uses examples collected in [`eval/eval_samples.jsonl`](./eval/eval_samples.jsonl) file for evaluation. Run from this directory so relative paths resolve.

```bash
python eval/eval_generate.py \
  --config ./train_configs/smoke_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/<step_dir> \
  --samples ./eval/eval_samples.jsonl \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 50 \
  --stop_at_eos
```

- `eval/eval_samples.jsonl` accepts `{name, prompt}` or FIM fields `{fim_prefix, fim_suffix, fim_format}`.
- Use `--prompt` for a single prompt, or add `--out` to print JSON.
- `hf_assets/` stores the local tokenizer snapshot referenced by `hf_assets_path` in the config; it is required for eval with the current configs

Single-prompt example:

```bash
python eval/eval_generate.py \
  --config ./train_configs/smoke_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/<step_dir> \
  --prompt "def sum(a, b):" \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 50 \
  --stop_at_eos \
  --custom_import custom_spec
```
