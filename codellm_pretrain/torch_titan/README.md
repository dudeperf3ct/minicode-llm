# Pretraining Code LLMs with TorchTitan

Write up:

Training an LLM from scratch using the custom tokenizer and dataset prepared in previous steps.

Custom tokeinzer: https://dudeperf3ct.github.io/projects/train_llm_part1/
Dataset: [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2)


## Setup

I am using Lambda Labs GPU instance with 1xH100 GPU (80 GB SMX5) for debugging.

Once instance is up, install the PyTorch nightly build with CUDA 12.8 support in a new virtual environment:

```bash
uv venv
source .venv/bin/activate
uv pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
uv pip install transformers tokenizers
```

Clone the `torchtitan` repository and install the project dependencies.

```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan
uv pip install -r pyproject.toml
```

Make a copy of the training configuration files and local overrides to torchtitan's directory.

```bash
cp -r ../train_configs ./train_configs
```

Create a [access token](https://huggingface.co/docs/hub/en/security-tokens) in Hugging Face and login using the `huggingface-cli` tool.

```bash
hf auth login
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

## Custom dataset + tokenizer (SwallowCode + 32k)


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

Run training with the custom config:

```bash
NGPU=1 CONFIG_FILE='./train_configs/llama32_1b_swallowcode_tok32k.toml' ./run_train.sh
```
