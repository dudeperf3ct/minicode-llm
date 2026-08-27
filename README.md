# Mini CodeLLM

Write-up:
- https://dudeperf3ct.github.io/projects/train_llm_part0/ (data)
- https://dudeperf3ct.github.io/projects/train_llm_part1/ (tokenizer)
- https://dudeperf3ct.github.io/projects/train_llm_part2/ (pretraining)
- https://dudeperf3ct.github.io/projects/post_training_llm_sft/ (supervised fine-tuning)

## Getting Started

- [`data_pipeline`](./data_pipeline/README.md): Parses and downloads datasets.
- [`tokenizer`](./tokenizer/README.md): Trains a custom byte-level BPE tokenizer using a subset of [`tokyotech-llm/swallow-code-v2`](https://huggingface.co/datasets/tokyotech-llm/swallow-code-v2).
- `pretrain`: Trains a Llama 3.2 model using the custom tokenizer and data. Two implementations:
  * [`torch_titan`](./pretrain/torch_titan/README.md): Uses the `torchtitan` library for training.
  * Nvidia NeMo WIP
- [`sft`](./sft/README.md): Supervised fine-tuning of Qwen3.5-4B-Base on verified Python coding examples using direct and reasoning targets with LoRA and language-model full fine-tuning. The observed concurrency-adjusted training cost was approximately $185.
- [`rlvr`](./rlvr/README.md): Reinforcement learning with verifiable rewards using Axolotl GRPO and Modal-sandboxed Python tests.
- [`evals`](./evals/README.md): Shared held-out and public benchmark evaluation tools.
- [`common`](./common/README.md): Reusable data, sampling, decontamination, Hub, and I/O utilities.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
