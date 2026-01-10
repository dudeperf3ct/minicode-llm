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
    local_dir="hf_assets/codellm-tokenizer",
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

For a full run on 4 x H100 GPUs (80 GB SMX5):

>[!IMPORTANT]
> `data_parallel_replicate_degree` is set to 4 in the config. Change it if you use a different GPU count.

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

Setup the environment by running `uv sync --extra cuda`. Make sure DCP checkpoint are stored under `checkpoint` folder.

Use `eval/eval_generate.py` to run inference against a DCP checkpoint. Provide the same config used for training and a checkpoint directory. It uses examples collected in [`eval/eval_samples.jsonl`](./eval/eval_samples.jsonl) file for evaluation. Run from this directory so relative paths resolve.

```bash
python eval/eval_generate.py \
  --config ./train_configs/full_llama32_1b_swallowcode_tok32k.toml \
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
  --config ./train_configs/full_llama32_1b_swallowcode_tok32k.toml \
  --checkpoint ./checkpoint/<step_dir> \
  --prompt "def sum(a, b):" \
  --max_new_tokens 64 \
  --temperature 0.8 \
  --top_k 50 \
  --stop_at_eos \
  --custom_import custom_spec
```
