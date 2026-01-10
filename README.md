# Mini CodeLLM

Write-up:
- https://dudeperf3ct.github.io/projects/train_llm_part0/ (data)
- https://dudeperf3ct.github.io/projects/train_llm_part1/ (tokenizer)

## Getting Started

- [`codellm_data`](./codellm_data/README.md): Parses and downloads datasets.
- [`codellm_tokenizer`](./codellm_tokenizer/README.md): Train a custom byte-level BPE tokenizer using subset of [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2) dataset
- [`codellm_pretrain`]: Training a Llama 3.2 model using custom tokenizer and data. Two implementations
  * [`torch_titan`](./codellm_pretrain/torch_titan/README.md): It uses `torchtitan` library for training
  * NeMo (WIP)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
