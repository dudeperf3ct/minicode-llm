# Dataset Preparation and References

This document owns the data design, provenance, preparation, publication, and private-dataset access for the Qwen3.5-4B KodCode RLVR experiment.

Run commands from the parent `rlvr/` directory.

## Dataset Matrix

| Dataset | Examples | Purpose |
| --- | ---: | --- |
| KodCode RL source | 10,000 | Candidate prompt pool |
| RLVR train | 1,000 | One-epoch GRPO experiment |
| SFT exclusions | 11,000 | Prevent reuse of SFT train, validation, or test prompts |

The RLVR dataset has one training split. It contains prompts, public tests, and selection metadata, but no reference solutions. Final model evaluation reuses the untouched SFT private test dataset; it is never part of RL training.

## Pinned References

| Component | Version or identifier |
| --- | --- |
| Direct FFT checkpoint | `dudeperf3ct/qwen35-4b-kodcode-sft-10k@34355613741e197528920acc211005f303524e58` |
| Reasoning FFT checkpoint | `dudeperf3ct/qwen35-4b-kodcode-sft-10k@eed04fa11c7b7a9bbd4a129471d1a4e9bd2e1ece` |
| Tokenizer | `Qwen/Qwen3.5-4B-Base@1001bb4d826a52d1f399e183466143f4da7b741b` |
| KodCode RL source | `KodCode/KodCode-Light-RL-10K@dcf78a8bbba9a613b596ce993c4921a38687dfcc` |
| SFT public exclusions | `dudeperf3ct/qwen35-kodcode-sft-data@db5f9912d76642bac325bea2bb41d83ad186365b` |
| SFT private-test exclusions | `dudeperf3ct/qwen35-kodcode-sft-private-test@8fbebc91f03fa40703b03a00879a3c651acaccd4` |
| HumanEval | `openai/openai_humaneval@7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` |
| MBPP | `google-research-datasets/mbpp@4bb6404fdc6cacfda99d4ac4205087b89d32030c` |
| LiveCodeBench | `livecodebench/code_generation_lite@0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, `release_v5` |
| LiveCodeBench mirror | `lighteval/code_generation_lite@89e5fc5c2a8e748f50e95bc7235fab2372d49bfa` |

The machine-readable source of truth for data preparation is [`manifests/revisions.json`](../manifests/revisions.json).

Training checkpoint revisions live in the experiment table in the parent README because both runs share this dataset.

## Preparation Environment

```bash
uv sync --group dev
```

Authenticate with Hugging Face. The token needs read access to the private SFT test dataset so its prompts can be excluded, and write access to the RLVR dataset repository:

```bash
hf auth login
hf auth whoami
```

## Prepare the 1K Dataset

```bash
uv run rlvr-prepare
```

Regeneration is explicit:

```bash
uv run rlvr-prepare --overwrite
```

Preparation performs:

- verified, non-empty, `instruct`-style task filtering;
- restriction to tests that use only the Python standard library and pytest;
- normalized-question deduplication;
- removal of every SFT train, validation, and private-test prompt;
- shared HumanEval, MBPP, and LiveCodeBench word 8-gram decontamination;
- Qwen3.5 chat-template tokenization and 8,192-token context filtering;
- joint `(difficulty, subset, style)` largest-remainder sampling with seed 42.

Generated artifacts are:

| Path | Contents |
| --- | --- |
| `data/train.jsonl` | 1,000 Axolotl-ready prompts, public tests, and metadata |
| `data/selected_ids.json` | Selected IDs and stratum allocation |
| `data/decontamination_report.jsonl` | Removed public-benchmark overlaps |
| `manifests/data_manifest.json` | Revisions, counts, sizes, and SHA-256 checksums |

The pinned run produced:

- 10,000 source rows;
- 404 SFT-overlap rows and 235 unsupported-dependency rows removed;
- 1,407 rows removed by public-benchmark decontamination;
- 7,954 eligible prompts before stratified selection;
- exactly 1,000 selected prompts, with none exceeding the context budget.

## Publish the Training Dataset

- Repository: `dudeperf3ct/qwen35-kodcode-rlvr-data`
- Branch: `main`
- Pinned revision: `6cd21d8893dd70d8055706ed9c58c374666b7bd0`
- Split: `train`

Verify and stage the complete payload without changing Hugging Face:

```bash
uv run rlvr-upload --dry-run
```

Upload the dataset:

```bash
uv run rlvr-upload
```

The uploader verifies local checksums before any remote mutation and prints the resulting immutable commit SHA. Both `configs/direct-fft.yml` and `configs/reasoning-fft.yml` load this Hugging Face dataset rather than the generated local JSONL.

Verify the published split:

```python
from datasets import load_dataset

train = load_dataset(
    "dudeperf3ct/qwen35-kodcode-rlvr-data",
    revision="6cd21d8893dd70d8055706ed9c58c374666b7bd0",
    split="train",
)
assert len(train) == 1_000
```

Do not start GRPO until the upload and verification succeed.

> [!NOTE]
> KodCode uses CC BY-NC 4.0. Treat this as a non-commercial experiment.
