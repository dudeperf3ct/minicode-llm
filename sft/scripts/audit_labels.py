"""Audit assistant-only labels before any Qwen3.5 training run.

Axolotl first preprocesses a fixed 32-example direct or reasoning dataset.
This script independently reconstructs the expected Qwen3.5 rendering, checks
every saved label array, and writes a small report without copying prompts,
solutions, or reasoning traces.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml
from common.io import write_json
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from data_statistics import summarize_lengths
from sft_utils import PROJECT_DIR

IGNORE_TOKEN_ID = -100


@dataclass(frozen=True)
class AuditConfig:
    dataset: str
    dataset_name: str
    dataset_revision: str
    dataset_split: str
    prepared: Path
    output: Path
    model: str
    revision: str
    variant: str
    sequence_length: int
    eot_token: str


@dataclass(frozen=True)
class ExpectedExample:
    example_id: str
    messages: list[dict[str, str]]
    final_content: str
    reasoning: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--examples", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_audit_config(args.config, args.output)
    source_rows = load_dataset(
        config.dataset,
        config.dataset_name,
        revision=config.dataset_revision,
        split=config.dataset_split,
    )
    prepared_path = _find_prepared_dataset(config.prepared)
    prepared = load_from_disk(str(prepared_path))
    if not isinstance(prepared, Dataset):
        raise TypeError("Expected Axolotl to save one Hugging Face Dataset")
    if len(prepared) != len(source_rows):
        raise ValueError(f"Prepared/source size mismatch: {len(prepared)} != {len(source_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(config.model, revision=config.revision)
    eot_id = _single_token_id(tokenizer, config.eot_token)
    measurements = _audit_dataset(config, source_rows, prepared, tokenizer, eot_id)
    write_json(config.output, _build_report(config, prepared_path, measurements, args.examples))
    print(f"Label audit passed: {config.output}")


def load_audit_config(path: Path, output: Path | None = None) -> AuditConfig:
    """Read the config fields needed to locate and audit prepared labels."""

    raw = yaml.safe_load((PROJECT_DIR / path).resolve().read_text(encoding="utf-8"))
    dataset = raw["datasets"][0]
    variant = "reasoning" if dataset.get("split_thinking") else "direct"
    report = output or Path(f"reports/label-audit/{variant}.json")
    return AuditConfig(
        dataset=dataset["path"],
        dataset_name=dataset["name"],
        dataset_revision=dataset["revision"],
        dataset_split=dataset["split"],
        prepared=(PROJECT_DIR / raw["dataset_prepared_path"]).resolve(),
        output=(PROJECT_DIR / report).resolve(),
        model=raw["base_model"],
        revision=raw["revision_of_model"],
        variant=variant,
        sequence_length=raw["sequence_len"],
        eot_token=raw["eot_tokens"][0],
    )


def _audit_dataset(
    config: AuditConfig,
    source_rows: Dataset,
    prepared: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    eot_id: int,
) -> list[dict[str, int | str | bool]]:
    source_by_input = {}
    for source in source_rows:
        expected = _expected_example(source, config.variant)
        source_by_input[tuple(_render_ids(tokenizer, expected.messages))] = expected

    measurements = []
    for prepared_row in prepared:
        input_ids = tuple(_integer_list(prepared_row, "input_ids"))
        expected = source_by_input.pop(input_ids, None)
        if expected is None:
            raise ValueError("Prepared input_ids do not match any source conversation")
        measurements.append(_audit_row(config, expected, prepared_row, tokenizer, eot_id))
    return measurements


def _audit_row(
    config: AuditConfig,
    expected: ExpectedExample,
    prepared: dict,
    tokenizer: PreTrainedTokenizerBase,
    eot_id: int,
) -> dict[str, int | str | bool]:
    input_ids = _integer_list(prepared, "input_ids")
    labels = _integer_list(prepared, "labels")

    _audit_tokenization(config, expected, input_ids, labels, tokenizer)
    assistant_eot = _audit_masking(expected, input_ids, labels, tokenizer, eot_id)
    trainable = _audit_target(expected, labels, tokenizer)
    return _measurement(expected, input_ids, labels, trainable, assistant_eot, tokenizer)


def _audit_tokenization(
    config: AuditConfig,
    expected: ExpectedExample,
    input_ids: list[int],
    labels: list[int],
    tokenizer: PreTrainedTokenizerBase,
) -> None:
    if len(input_ids) != len(labels):
        raise ValueError(f"{expected.example_id}: input_ids and labels differ in length")
    if len(input_ids) > config.sequence_length:
        raise ValueError(f"{expected.example_id}: sequence exceeds {config.sequence_length} tokens")
    if input_ids != _render_ids(tokenizer, expected.messages):
        raise ValueError(f"{expected.example_id}: input_ids differ from the pinned Qwen rendering")
    if any(
        label not in (IGNORE_TOKEN_ID, token_id)
        for token_id, label in zip(input_ids, labels, strict=True)
    ):
        raise ValueError(f"{expected.example_id}: labels must equal input_ids or -100")


def _audit_masking(
    expected: ExpectedExample,
    input_ids: list[int],
    labels: list[int],
    tokenizer: PreTrainedTokenizerBase,
    eot_id: int,
) -> int:
    dummy_messages = [
        expected.messages[0],
        {"role": "assistant", "content": "[[dummy_message]]"},
    ]
    expected_labels = _expected_labels(input_ids, _render_ids(tokenizer, dummy_messages), eot_id)
    if labels != expected_labels:
        raise ValueError(
            f"{expected.example_id}: labels differ from Axolotl's assistant-turn boundary"
        )

    eot_positions = [index for index, token_id in enumerate(input_ids) if token_id == eot_id]
    if len(eot_positions) != 2:
        raise ValueError(
            f"{expected.example_id}: expected two EOT tokens, found {len(eot_positions)}"
        )
    user_eot, assistant_eot = eot_positions
    if labels[user_eot] != IGNORE_TOKEN_ID or labels[assistant_eot] != eot_id:
        raise ValueError(f"{expected.example_id}: user/assistant EOT masking is incorrect")
    if any(label != IGNORE_TOKEN_ID for label in labels[assistant_eot + 1 :]):
        raise ValueError(f"{expected.example_id}: tokens after the assistant EOT are trainable")
    return assistant_eot


def _audit_target(
    expected: ExpectedExample, labels: list[int], tokenizer: PreTrainedTokenizerBase
) -> list[int]:
    trainable = [label for label in labels if label != IGNORE_TOKEN_ID]
    if not trainable:
        raise ValueError(f"{expected.example_id}: no assistant tokens are trainable")
    trainable_text = tokenizer.decode(trainable, skip_special_tokens=False)
    if expected.final_content.strip() not in trainable_text:
        raise ValueError(f"{expected.example_id}: final assistant content is not fully trainable")
    if expected.reasoning is None and ("<think>" in trainable_text or "</think>" in trainable_text):
        raise ValueError(f"{expected.example_id}: direct labels contain thinking template tokens")
    if expected.reasoning is not None and expected.reasoning not in trainable_text:
        raise ValueError(f"{expected.example_id}: reasoning is not fully trainable")
    return trainable


def _measurement(
    expected: ExpectedExample,
    input_ids: list[int],
    labels: list[int],
    trainable: list[int],
    assistant_eot: int,
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, int | str | bool]:
    opening = _single_token_id(tokenizer, "<think>")
    closing = _single_token_id(tokenizer, "</think>")
    opening_index = _single_position(input_ids, opening, expected.example_id)
    closing_index = _single_position(input_ids, closing, expected.example_id)
    return {
        "id": expected.example_id,
        "total_tokens": len(input_ids),
        "trainable_tokens": len(trainable),
        "masked_tokens": len(input_ids) - len(trainable),
        "first_trainable_index": next(
            index for index, label in enumerate(labels) if label != IGNORE_TOKEN_ID
        ),
        "assistant_eot_index": assistant_eot,
        "thinking_open_trainable": labels[opening_index] != IGNORE_TOKEN_ID,
        "thinking_close_trainable": labels[closing_index] != IGNORE_TOKEN_ID,
        "thinking_content_present": expected.reasoning is not None,
    }


def _expected_example(row: dict, variant: str) -> ExpectedExample:
    if variant == "direct":
        return _expected_direct(row)
    return _expected_reasoning(row)


def _expected_direct(row: dict) -> ExpectedExample:
    messages = row["messages"]
    target = messages[1]["content"]
    if "<think>" in target or "</think>" in target or "```" in target:
        raise ValueError(f"{row['id']}: direct target is not code-only")
    return ExpectedExample(example_id=row["id"], messages=messages, final_content=target)


def _expected_reasoning(row: dict) -> ExpectedExample:
    user_message, assistant_message = row["messages"]
    target = assistant_message["content"]
    if not target.startswith("<think>") or "</think>" not in target:
        raise ValueError(f"{row['id']}: reasoning delimiters are missing")
    reasoning, final_content = target.removeprefix("<think>").split("</think>", maxsplit=1)
    reasoning = reasoning.strip()
    final_content = final_content.lstrip()
    if not reasoning or not final_content:
        raise ValueError(f"{row['id']}: reasoning or final content is empty")
    return ExpectedExample(
        example_id=row["id"],
        messages=[
            user_message,
            {"role": "assistant", "reasoning_content": reasoning, "content": final_content},
        ],
        final_content=final_content,
        reasoning=reasoning,
    )


def _build_report(
    config: AuditConfig,
    prepared_path: Path,
    measurements: list[dict[str, int | str | bool]],
    inspected: int,
) -> dict:
    total_lengths = [int(row["total_tokens"]) for row in measurements]
    trainable_lengths = [int(row["trainable_tokens"]) for row in measurements]
    total_tokens = sum(total_lengths)
    trainable_tokens = sum(trainable_lengths)
    return {
        "status": "passed",
        "variant": config.variant,
        "dataset": config.dataset,
        "dataset_name": config.dataset_name,
        "dataset_revision": config.dataset_revision,
        "dataset_split": config.dataset_split,
        "prepared_dataset": str(prepared_path.relative_to(PROJECT_DIR)),
        "model": config.model,
        "revision": config.revision,
        "examples": len(measurements),
        "total_tokens": total_tokens,
        "trainable_tokens": trainable_tokens,
        "masked_tokens": total_tokens - trainable_tokens,
        "trainable_token_fraction": trainable_tokens / total_tokens,
        "total_length": summarize_lengths(total_lengths),
        "trainable_length": summarize_lengths(trainable_lengths),
        "inspected_examples": measurements[:inspected],
    }


def _find_prepared_dataset(root: Path) -> Path:
    candidates = sorted(path.parent for path in root.rglob("dataset_info.json"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one prepared dataset below {root}, found {len(candidates)}")
    return candidates[0]


def _render_ids(tokenizer: PreTrainedTokenizerBase, messages: list[dict[str, str]]) -> list[int]:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    input_ids = [int(token_id) for token_id in encoded]

    # Axolotl keeps source newlines with assistant content, while Qwen's Jinja
    # rendering trims them immediately before the end-of-turn token.
    content = messages[-1]["content"]
    trailing_newlines = content[len(content.rstrip("\n")) :]
    if trailing_newlines:
        eot_id = _single_token_id(tokenizer, "<|im_end|>")
        eot_index = len(input_ids) - 1 - input_ids[::-1].index(eot_id)
        newline_ids = tokenizer(trailing_newlines, add_special_tokens=False)["input_ids"]
        if input_ids[eot_index - len(newline_ids) : eot_index] != newline_ids:
            input_ids[eot_index:eot_index] = newline_ids
    return input_ids


def _single_token_id(tokenizer: PreTrainedTokenizerBase, token: str) -> int:
    token_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
    if len(token_ids) != 1:
        raise ValueError(f"EOT marker {token!r} maps to {len(token_ids)} tokens")
    return int(token_ids[0])


def _expected_labels(input_ids: list[int], dummy_ids: list[int], eot_id: int) -> list[int]:
    """Reproduce Axolotl's first/last-difference turn boundary."""

    shared = min(len(input_ids), len(dummy_ids))
    start = next(index for index in range(shared) if input_ids[index] != dummy_ids[index])
    reverse_offset = next(
        offset for offset in range(shared) if input_ids[-1 - offset] != dummy_ids[-1 - offset]
    )
    end = len(input_ids) - reverse_offset
    labels = [IGNORE_TOKEN_ID] * len(input_ids)
    labels[start:end] = input_ids[start:end]
    labels[len(input_ids) - 1 - input_ids[::-1].index(eot_id)] = eot_id
    return labels


def _single_position(values: list[int], token_id: int, example_id: str) -> int:
    positions = [index for index, value in enumerate(values) if value == token_id]
    if len(positions) != 1:
        raise ValueError(f"{example_id}: expected one thinking delimiter token")
    return positions[0]


def _integer_list(row: dict, field: str) -> list[int]:
    values = row.get(field)
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise TypeError(f"Prepared field {field!r} must be a list of integers")
    return values


if __name__ == "__main__":
    main()
