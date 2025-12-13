# Tokenizer

Write-up: https://dudeperf3ct.github.io/projects/train_llm_part1/

Train a custom byte-level BPE tokenizer using subset of [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2) dataset.

## Pre-requisites

- `uv`
- `huggingface-cli login` (optional, only if pushing artefacts)

## Getting Started

1. Install the dependencies:

  ```bash
  uv sync
  ```

2. Configure the settings in `config.yaml` as needed.

3. Train the tokenizer (runs with streaming HF dataset) and run a quick evaluation after training.

  ```bash
  python main.py --run-eval
  ```

> [!NOTE]
> If `hf auth login` is configured, the huggingface token will be used to push the tokenizer to the huggingface repo.
