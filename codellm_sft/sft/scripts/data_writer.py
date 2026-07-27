"""Serialize prepared dataset splits and their token measurements.

The preparation pipeline first selects one matched set of source rows. This
module renders those rows into direct and reasoning conversations, records
their token lengths, writes paired JSONL in identical ID order, and keeps
held-out tests separate from assistant targets.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import data_pipeline
import data_statistics
import pipeline_utils as utils
from datasets import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class PairedPaths:
    """Output locations for one matched direct/reasoning split."""

    direct: Path
    reasoning: Path


@dataclass(frozen=True)
class MeasuredExample:
    """One rendered example with named token measurements."""

    candidate: utils.Candidate
    prompt_tokens: int
    direct_tokens: int
    reasoning_tokens: int
    direct_messages: utils.Messages
    reasoning_messages: utils.Messages


@dataclass(frozen=True)
class PairedHandles:
    direct: TextIO
    reasoning: TextIO


def write_json(path: Path, value: Any) -> None:
    """Write deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write canonical JSONL records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            _write_jsonl_line(handle, record)


def _write_jsonl_line(handle: TextIO, value: Any) -> None:
    handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    handle.write("\n")


def _paired_output_paths(output_dir: Path) -> dict[str, PairedPaths]:
    return {
        "train": PairedPaths(
            direct=output_dir / "kodcode_direct.jsonl",
            reasoning=output_dir / "kodcode_reasoning.jsonl",
        ),
        "validation": PairedPaths(
            direct=output_dir / "validation" / "kodcode_direct.jsonl",
            reasoning=output_dir / "validation" / "kodcode_reasoning.jsonl",
        ),
        "pilot": PairedPaths(
            direct=output_dir / "subsets" / "train-1000" / "kodcode_direct.jsonl",
            reasoning=output_dir / "subsets" / "train-1000" / "kodcode_reasoning.jsonl",
        ),
        "overfit": PairedPaths(
            direct=output_dir / "subsets" / "train-32" / "kodcode_direct.jsonl",
            reasoning=output_dir / "subsets" / "train-32" / "kodcode_reasoning.jsonl",
        ),
    }


def write_training_datasets(
    *,
    source: Dataset,
    selection: data_pipeline.SplitSelection,
    tokenizer: PreTrainedTokenizerBase,
    output_dir: Path,
    batch_size: int,
    sequence_length: int,
    overlength_count: int,
) -> None:
    named_splits = selection.all_named_splits()
    paths = _paired_output_paths(output_dir)
    statistics: dict[str, Any] = {
        "assistant_tokens_definition": (
            "Rendered assistant turn tokens, including assistant template delimiters, "
            "computed as full conversation length minus rendered user-turn length."
        ),
        "examples_removed_above_sequence_length": overlength_count,
        "sequence_length": sequence_length,
        "splits": {},
    }

    for split in ("train", "validation", "pilot", "overfit"):
        print(f"Writing paired {split} data...", flush=True)
        statistics["splits"][split] = _write_paired_split(
            source=source,
            candidates=named_splits[split],
            paths=paths[split],
            tokenizer=tokenizer,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )

    print("Computing held-out test token statistics...", flush=True)
    statistics["splits"]["test"] = _process_paired_split(
        source,
        selection.test,
        tokenizer,
        batch_size,
        sequence_length,
    )
    _write_test_split(source, selection.test, output_dir)
    data_statistics.add_reasoning_ratios(statistics)
    write_json(output_dir / "token_statistics.json", statistics)


def _write_paired_split(
    source: Dataset,
    candidates: list[utils.Candidate],
    paths: PairedPaths,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    sequence_length: int,
) -> dict[str, dict[str, int | float]]:
    paths.direct.parent.mkdir(parents=True, exist_ok=True)
    paths.reasoning.parent.mkdir(parents=True, exist_ok=True)
    with (
        paths.direct.open("w", encoding="utf-8") as direct_handle,
        paths.reasoning.open("w", encoding="utf-8") as reasoning_handle,
    ):
        return _process_paired_split(
            source,
            candidates,
            tokenizer,
            batch_size,
            sequence_length,
            PairedHandles(direct_handle, reasoning_handle),
        )


def _write_test_split(source: Dataset, candidates: list[utils.Candidate], output_dir: Path) -> None:
    """Write prompts and private tests separately, with no assistant targets."""

    questions: list[dict[str, Any]] = []
    private_tests: list[dict[str, Any]] = []
    for candidate in candidates:
        row = source[candidate.source_index]
        if not isinstance(row["test"], str) or not row["test"].strip():
            raise RuntimeError(f"Test example has no private test: {candidate.question_id}")
        questions.append(
            {
                "gpt_difficulty": candidate.stratum.gpt_difficulty,
                "id": candidate.question_id,
                "messages": [{"role": "user", "content": row["question"]}],
                "style": candidate.stratum.style,
                "subset": candidate.stratum.subset,
            }
        )
        private_tests.append(
            {
                "id": candidate.question_id,
                "test": row["test"],
                "test_info": row["test_info"],
            }
        )

    write_jsonl(output_dir / "test" / "questions.jsonl", questions)
    write_jsonl(output_dir / "test" / "private_tests.jsonl", private_tests)


def _process_paired_split(
    source: Dataset,
    candidates: list[utils.Candidate],
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    sequence_length: int,
    handles: PairedHandles | None = None,
) -> dict[str, dict[str, int | float]]:
    direct_statistics = data_statistics.LengthAccumulator()
    reasoning_statistics = data_statistics.LengthAccumulator()

    for batch in utils.batched(candidates, batch_size):
        for measured in _measure_batch(source, batch, tokenizer):
            _record_measurement(
                measured, sequence_length, direct_statistics, reasoning_statistics, handles
            )

    return {"direct": direct_statistics.summarize(), "reasoning": reasoning_statistics.summarize()}


def _measure_batch(
    source: Dataset, candidates: list[utils.Candidate], tokenizer: PreTrainedTokenizerBase
) -> list[MeasuredExample]:
    rows = [source[candidate.source_index] for candidate in candidates]
    prompts = [[{"role": "user", "content": row["question"]}] for row in rows]
    direct_messages = [data_pipeline.direct_messages(row) for row in rows]
    reasoning_messages = [data_pipeline.reasoning_messages(row) for row in rows]

    prompt_lengths = utils.token_lengths(tokenizer, utils.render_batch(tokenizer, prompts))
    direct_lengths = utils.token_lengths(tokenizer, utils.render_batch(tokenizer, direct_messages))
    reasoning_lengths = utils.token_lengths(
        tokenizer, utils.render_batch(tokenizer, reasoning_messages)
    )

    return [
        MeasuredExample(
            candidate=candidate,
            prompt_tokens=prompt_lengths[index],
            direct_tokens=direct_lengths[index],
            reasoning_tokens=reasoning_lengths[index],
            direct_messages=direct_messages[index],
            reasoning_messages=reasoning_messages[index],
        )
        for index, candidate in enumerate(candidates)
    ]


def _record_measurement(
    measured: MeasuredExample,
    sequence_length: int,
    direct_statistics: data_statistics.LengthAccumulator,
    reasoning_statistics: data_statistics.LengthAccumulator,
    handles: PairedHandles | None,
) -> None:
    candidate = measured.candidate
    if max(measured.direct_tokens, measured.reasoning_tokens) > sequence_length:
        raise RuntimeError(f"Overlength selected example: {candidate.question_id}")

    if handles is not None:
        _write_jsonl_line(
            handles.direct, {"id": candidate.question_id, "messages": measured.direct_messages}
        )
        _write_jsonl_line(
            handles.reasoning,
            {"id": candidate.question_id, "messages": measured.reasoning_messages},
        )

    direct_statistics.add(measured.prompt_tokens, measured.direct_tokens)
    reasoning_statistics.add(measured.prompt_tokens, measured.reasoning_tokens)
