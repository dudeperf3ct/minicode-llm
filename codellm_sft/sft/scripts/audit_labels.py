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

import data_statistics
import data_writer
import pipeline_utils as utils
import yaml
from datasets import Dataset, load_dataset, load_from_disk
from transformers import AutoTokenizer, PreTrainedTokenizerBase

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
    measurements = [
        _audit_row(config, source, prepared[index], tokenizer, eot_id)
        for index, source in enumerate(source_rows)
    ]
    data_writer.write_json(
        config.output, _build_report(config, prepared_path, measurements, args.examples)
    )
    print(f"Label audit passed: {config.output}")


def load_audit_config(path: Path, output: Path | None = None) -> AuditConfig:
    """Read the config fields needed to locate and audit prepared labels."""

    raw = yaml.safe_load((utils.PROJECT_DIR / path).resolve().read_text(encoding="utf-8"))
    dataset = raw["datasets"][0]
    variant = "reasoning" if dataset.get("split_thinking") else "direct"
    report = output or Path(f"reports/label-audit/{variant}.json")
    return AuditConfig(
        dataset=dataset["path"],
        dataset_name=dataset["name"],
        dataset_revision=dataset["revision"],
        dataset_split=dataset["split"],
        prepared=(utils.PROJECT_DIR / raw["dataset_prepared_path"]).resolve(),
        output=(utils.PROJECT_DIR / report).resolve(),
        model=raw["base_model"],
        revision=raw["revision_of_model"],
        variant=variant,
        sequence_length=raw["sequence_len"],
        eot_token=raw["eot_tokens"][0],
    )


def _audit_row(
    config: AuditConfig,
    source: dict,
    prepared: dict,
    tokenizer: PreTrainedTokenizerBase,
    eot_id: int,
) -> dict[str, int | str | bool]:
    example_id, messages, final_content, reasoning = _expected_messages(source, config.variant)
    input_ids = _integer_list(prepared, "input_ids")
    labels = _integer_list(prepared, "labels")

    if len(input_ids) != len(labels):
        raise ValueError(f"{example_id}: input_ids and labels differ in length")
    if len(input_ids) > config.sequence_length:
        raise ValueError(f"{example_id}: sequence exceeds {config.sequence_length} tokens")
    if input_ids != _render_ids(tokenizer, messages):
        raise ValueError(f"{example_id}: input_ids differ from the pinned Qwen rendering")
    if any(
        label not in (IGNORE_TOKEN_ID, token_id)
        for token_id, label in zip(input_ids, labels, strict=True)
    ):
        raise ValueError(f"{example_id}: labels must equal input_ids or -100")

    dummy_messages = [messages[0], {"role": "assistant", "content": "[[dummy_message]]"}]
    expected_labels = _expected_labels(input_ids, _render_ids(tokenizer, dummy_messages), eot_id)
    if labels != expected_labels:
        raise ValueError(f"{example_id}: labels differ from Axolotl's assistant-turn boundary")

    eot_positions = [index for index, token_id in enumerate(input_ids) if token_id == eot_id]
    if len(eot_positions) != 2:
        raise ValueError(f"{example_id}: expected two EOT tokens, found {len(eot_positions)}")
    user_eot, assistant_eot = eot_positions
    if labels[user_eot] != IGNORE_TOKEN_ID or labels[assistant_eot] != eot_id:
        raise ValueError(f"{example_id}: user/assistant EOT masking is incorrect")
    if any(label != IGNORE_TOKEN_ID for label in labels[assistant_eot + 1 :]):
        raise ValueError(f"{example_id}: tokens after the assistant EOT are trainable")

    trainable = [label for label in labels if label != IGNORE_TOKEN_ID]
    if not trainable:
        raise ValueError(f"{example_id}: no assistant tokens are trainable")
    trainable_text = tokenizer.decode(trainable, skip_special_tokens=False)
    if final_content.strip() not in trainable_text:
        raise ValueError(f"{example_id}: final assistant content is not fully trainable")
    if config.variant == "direct" and ("<think>" in trainable_text or "</think>" in trainable_text):
        raise ValueError(f"{example_id}: direct labels contain thinking template tokens")
    if config.variant == "reasoning" and reasoning not in trainable_text:
        raise ValueError(f"{example_id}: reasoning is not fully trainable")

    opening = _single_token_id(tokenizer, "<think>")
    closing = _single_token_id(tokenizer, "</think>")
    opening_index = _single_position(input_ids, opening, example_id)
    closing_index = _single_position(input_ids, closing, example_id)
    return {
        "id": example_id,
        "total_tokens": len(input_ids),
        "trainable_tokens": len(trainable),
        "masked_tokens": len(input_ids) - len(trainable),
        "first_trainable_index": next(
            index for index, label in enumerate(labels) if label != IGNORE_TOKEN_ID
        ),
        "assistant_eot_index": assistant_eot,
        "thinking_open_trainable": labels[opening_index] != IGNORE_TOKEN_ID,
        "thinking_close_trainable": labels[closing_index] != IGNORE_TOKEN_ID,
        "thinking_content_present": reasoning is not None,
    }


def _expected_messages(
    row: dict, variant: str
) -> tuple[str, list[dict[str, str]], str, str | None]:
    example_id = row.get("id")
    messages = row.get("messages")
    if not isinstance(example_id, str) or not example_id:
        raise ValueError("Source row is missing an id")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError(f"{example_id}: expected exactly two messages")
    if [message.get("role") for message in messages] != ["user", "assistant"]:
        raise ValueError(f"{example_id}: expected user then assistant")
    if any(not isinstance(message.get("content"), str) for message in messages):
        raise ValueError(f"{example_id}: message content must be strings")

    target = messages[1]["content"]
    if variant == "direct":
        if "<think>" in target or "</think>" in target or "```" in target:
            raise ValueError(f"{example_id}: direct target is not code-only")
        return example_id, messages, target, None

    if not target.startswith("<think>") or target.count("<think>") != 1:
        raise ValueError(f"{example_id}: malformed opening reasoning delimiter")
    if target.count("</think>") != 1:
        raise ValueError(f"{example_id}: malformed closing reasoning delimiter")
    closing = target.index("</think>")
    reasoning = target[len("<think>") : closing].strip()
    final_content = target[closing + len("</think>") :].lstrip()
    if not reasoning or not final_content:
        raise ValueError(f"{example_id}: reasoning or final content is empty")
    transformed = [
        messages[0],
        {"role": "assistant", "reasoning_content": reasoning, "content": final_content},
    ]
    return example_id, transformed, final_content, reasoning


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
        "prepared_dataset": str(prepared_path.relative_to(utils.PROJECT_DIR)),
        "model": config.model,
        "revision": config.revision,
        "examples": len(measurements),
        "total_tokens": total_tokens,
        "trainable_tokens": trainable_tokens,
        "masked_tokens": total_tokens - trainable_tokens,
        "trainable_token_fraction": trainable_tokens / total_tokens,
        "total_length": data_statistics.summarize_lengths(total_lengths),
        "trainable_length": data_statistics.summarize_lengths(trainable_lengths),
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
    return [int(token_id) for token_id in encoded]


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
