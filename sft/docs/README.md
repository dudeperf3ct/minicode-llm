# Dataset Preparation and References

This document owns the data design, provenance, preparation, publication, and private-test access for the Qwen3.5-4B KodCode SFT experiment.

Run commands from the parent `sft/` directory.

## Dataset Matrix

The matched pipeline produces:

| Split | Examples | Purpose |
| --- | ---: | --- |
| Train | 10,000 | Main two-epoch experiment |
| Validation | 500 | Model and training decisions |
| Test | 500 | Final evaluation only |
| Pilot | 1,000 | One-epoch runtime and stability pilot |
| Overfit | 32 | Pipeline and memorization check |

Direct and reasoning variants contain exactly the same IDs in the same order.

Validation, test, pilot, and overfit sizes are fixed. Only the main training size may be changed with `--train-size` for a later scaling experiment. Only the assistant target differs.

## Pinned References

| Component | Version or identifier |
| --- | --- |
| Base model and tokenizer | `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b` |
| KodCode source | `KodCode/KodCode-V1-SFT-R1@26c8a11c800d71b6c4bafe12a92c9090ef0b6214` |
| HumanEval | `openai/openai_humaneval@7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` |
| MBPP | `google-research-datasets/mbpp@4bb6404fdc6cacfda99d4ac4205087b89d32030c` |
| LiveCodeBench | `livecodebench/code_generation_lite@0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, `release_v5` |
| LiveCodeBench mirror | `lighteval/code_generation_lite@89e5fc5c2a8e748f50e95bc7235fab2372d49bfa` |
| Open-R1 decontamination reference | `huggingface/open-r1@1416fa0cf21595d2083b399a2a0bbddd7f6e9563` |
)

## Preparation Environment

```bash
uv sync --group dev
```

Verify the relevant imports:

```bash
uv run python -c \
  "import datasets, transformers, yaml; print(datasets.__version__, transformers.__version__)"
```

## Prepare Matched Data

The pipeline downloads only the pinned KodCode `default/train` split. It never loads the published `incorrect` or `use_with_caution` splits.

```bash
uv run python scripts/prepare_data.py
```

Regeneration is explicit:

```bash
uv run python scripts/prepare_data.py --overwrite
```

Preparation performs:

- source-field and correctness filtering;
- normalized-question deduplication;
- Open-R1-style word 8-gram benchmark decontamination;
- Qwen3.5 chat-template tokenization;
- removal of reasoning examples above 16,384 tokens;
- joint `(difficulty, subset, style)` largest-remainder sampling;
- matched direct/reasoning rendering;
- nested 1,000- and 32-example subset selection;
- split and checksum assertions.

The pinned run produced:

- 268,211 source train rows;
- 17 normalized-question duplicates and 5 malformed reasoning rows removed;
- 26,906 rows removed by benchmark decontamination;
- 708 reasoning examples removed above 16,384 tokens;
- 240,575 eligible rows before stratified selection;
- exact 10,000/500/500 train, validation, and untouched test splits.

On the selected training split, reasoning assistant turns contain 26.63 times as many tokens as direct assistant turns. The reasoning total-length p99 is 14,340 tokens.

LiveCodeBench remains pinned to the official dataset revision. Because its official Hub dataset uses a remote loader script, preparation reads only `question_id` and `question_content` from the separately pinned `lighteval/code_generation_lite` `release_v5` Parquet mirror. The normalized 880-prompt snapshot hash is recorded in the output statistics.

Generated JSONL and statistics live in this directory. The checksum manifest is `../manifests/data_manifest.json`.

## Public Training Dataset

- Repository: `dudeperf3ct/qwen35-kodcode-sft-data`
- Branch: `main`
- Pinned revision: `db5f9912d76642bac325bea2bb41d83ad186365b`

Authenticate with a fine-grained token that has write access to the dataset:

```bash
hf auth login
uv run hf auth whoami
```

Verify the payload without modifying the Hub:

```bash
uv run python scripts/upload_dataset.py \
  --repo-id dudeperf3ct/qwen35-kodcode-sft-data \
  --dry-run
```

Upload:

```bash
HF_XET_HIGH_PERFORMANCE=1 \
uv run python scripts/upload_dataset.py \
  --repo-id dudeperf3ct/qwen35-kodcode-sft-data
```

The uploader verifies local checksums before any remote mutation and prints the resulting commit SHA.

The repository contains `direct` and `reasoning` configurations:

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

## Private Test Dataset

- Repository: `dudeperf3ct/qwen35-kodcode-sft-private-test`
- Pinned revision: `8fbebc91f03fa40703b03a00879a3c651acaccd4`
- Split: `test`

The uploader verifies both local test files against the data manifest, pairs them by ID, requires the destination repository to remain private, and records the uploaded commit in `../manifests/evaluation.json`:

```bash
uv run python scripts/upload_test_dataset.py
```

Remote evaluation machines require a Hugging Face token with read access:

```bash
hf auth login
hf auth whoami
```

The private split must never be used for training, validation, or
hyperparameter selection.
