# Pretraining Code LLMs with TorchTitan

Write up:

Training an LLM from scratch using the custom tokenizer and dataset prepared in previous steps.

Custom tokeinzer: https://dudeperf3ct.github.io/projects/train_llm_part1/
Dataset: [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2)


## Setup

I am using Lambda Labs GPU instance with 1xH100 GPU (80 GB SMX5) for debugging.

Once instance is up, install the required libraries:

```bash
uv sync
```

Clone the `torchtitan` repository.

```bash
git clone https://github.com/pytorch/torchtitan.git
cd torchtitan
```

Move the training configuration files to torchtitan's directory.

```bash
mv ../train_configs ./
```

### Memory Estimation

Next, estimate the memory requirements for the [Llama 3.2 1B](https://huggingface.co/meta-llama/Llama-3.2-1B) model

> [!NOTE]
> Llama 3.2 1B is a gated model and requires access. Log in to Hugging Face and request access to the model.

```bash
NGPU=1 CONFIG_FILE='./train_configs/debug_llama32_1b.toml'
```

### Communication Mode for debugging

TorchTitan offers two "fake" communication model: `fake_backend` and `local_tensor`. These modes allow you to dry run distributed training code on a single GPU without needing multiple GPUs or a distributed setup.

- `fake_backend`: Simulates a distributed communication without actual data transfer.

```bash
NGPU=8 COMM_MODE='fake_backend' CONFIG_FILE='./train_configs/debug_llama32_1b.toml' ./run_train.sh
```

- `local_tensor`: Simulates the full distributed training workflow on a single GPU by executing all communication and computation locally

```bash
NGPU=8 COMM_MODE='local_tensor' CONFIG_FILE='./train_configs/debug_llama32 ./run_train.sh
```
