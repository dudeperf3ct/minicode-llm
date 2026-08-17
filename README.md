# Mini CodeLLM

Write-up:
- https://dudeperf3ct.github.io/projects/train_llm_part0/ (data)
- https://dudeperf3ct.github.io/projects/train_llm_part1/ (tokenizer)
- https://dudeperf3ct.github.io/projects/train_llm_part2/ (pretraining)
- https://dudeperf3ct.github.io/projects/post_training_llm_sft/ (post training - SFT)

## Getting Started

- [`codellm_data`](./codellm_data/README.md): Parses and downloads datasets.
- [`codellm_tokenizer`](./codellm_tokenizer/README.md): Train a custom byte-level BPE tokenizer using subset of [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2) dataset
- `codellm_pretrain`: Training a Llama 3.2 model using custom tokenizer and data. Two implementations
  * [`torch_titan`](./codellm_pretrain/torch_titan/README.md): It uses `torchtitan` library for training
  * NeMo (WIP)
- [`codellm_sft`](./codellm_sft/sft/README.md): Supervised fine-tuning of Qwen3.5-4B-Base on verified Python coding examples using direct and reasoning targets with LoRA and language-model full fine-tuning. The observed concurrency-adjusted training cost was approximately $184.93; treating every run as a standalone job gives a $217.00 equivalent. See the [evaluation guide](./codellm_sft/eval/README.md) for held-out and public benchmark evaluation.


## License

This project is licensed under the MIT License - see the LICENSE file for details.
